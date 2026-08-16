#!/usr/bin/env python3
"""
Robuster Wikipedia-Revisionsdownload fuer grosse Artikel.

Schreibt Revisionen laufend in eine XML-Partial-Datei und speichert den
Fortschritt, damit ein abgebrochener Download wiederaufgenommen werden kann.

example:
python src/wiki_api.py --article "Climate change" --start 2020-01-01 --end 2026-07-04
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import requests

from config import BASE_DIR, article_slug


API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WikiTimeCurve/1.0 (research project; contact: your@email.com)"
BATCH_SIZE = 50
SLEEP_SEC = 1.0


def xml_text(value):
    return escape(str(value or ""), {'"': "&quot;"})


def revision_xml(rev):
    content = rev.get("slots", {}).get("main", {}).get("content", "") or ""
    user = rev.get("user", "anonymous")
    revid = rev.get("revid", "0")
    timestamp = rev.get("timestamp", "")
    sha1 = rev.get("sha1", "")
    comment = rev.get("comment", "")
    tags = rev.get("tags", []) or []

    return (
        "    <revision>\n"
        f"      <id>{xml_text(revid)}</id>\n"
        f"      <timestamp>{xml_text(timestamp)}</timestamp>\n"
        f"      <sha1>{xml_text(sha1)}</sha1>\n"
        f"      <comment>{xml_text(comment)}</comment>\n"
        f"      <tags>{xml_text('|'.join(tags))}</tags>\n"
        "      <contributor>\n"
        f"        <username>{xml_text(user)}</username>\n"
        "      </contributor>\n"
        f"      <text>{xml_text(content)}</text>\n"
        "    </revision>\n"
    )


def initial_params(article, start, end):
    return {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": article,
        "rvprop": "ids|timestamp|user|size|sha1|comment|tags|content",
        "rvlimit": BATCH_SIZE,
        "rvdir": "newer",
        "rvstart": f"{start}T00:00:00Z",
        "rvend": f"{end}T23:59:59Z",
        "rvslots": "main",
        "formatversion": "2",
    }


def write_header(handle, article):
    handle.write('<?xml version="1.0" encoding="utf-8"?>\n')
    handle.write('<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">\n')
    handle.write("  <siteinfo>\n")
    handle.write("    <sitename>Wikipedia</sitename>\n")
    handle.write(f"    <base>https://en.wikipedia.org/wiki/{xml_text(article)}</base>\n")
    handle.write("  </siteinfo>\n")
    handle.write("  <page>\n")
    handle.write(f"    <title>{xml_text(article)}</title>\n")


def write_footer(handle):
    handle.write("  </page>\n")
    handle.write("</mediawiki>\n")


def load_state(state_path):
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state_path, state):
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, state_path)


def state_payload(args, params, kept, skipped, page_count, partial_offset,
                  complete=False):
    """Build one durable resume checkpoint.

    ``partial_offset`` marks the last known-good byte/character position. On
    resume, any uncheckpointed tail is truncated before the API page is fetched
    again, preventing duplicated revisions after an interruption.
    """
    return {
        "article": args.article,
        "start": args.start,
        "end": args.end,
        "min_revision_chars": args.min_revision_chars,
        "params": params,
        "kept": kept,
        "skipped": skipped,
        "page_count": page_count,
        "partial_offset": partial_offset,
        "complete": complete,
    }


def validate_resume_state(state, args, partial_path):
    """Return a user-facing error when a resume checkpoint is unsafe."""
    if not isinstance(state, dict):
        return "Der Resume-State ist kein gültiges JSON-Objekt."
    if not os.path.exists(partial_path):
        return (
            "Resume-State vorhanden, aber die zugehörige .partial-Datei fehlt. "
            "Mit --restart einen kontrollierten Neudownload starten."
        )
    if "partial_offset" not in state:
        return (
            "Der Resume-State stammt aus einer älteren, nicht sicher prüfbaren "
            "Version. Mit --restart einen kontrollierten Neudownload starten."
        )
    if not isinstance(state.get("params"), dict):
        return "Der Resume-State enthält keine gültigen API-Parameter."
    expected = {
        "article": args.article,
        "start": args.start,
        "end": args.end,
        "min_revision_chars": args.min_revision_chars,
    }
    mismatches = [
        key for key, value in expected.items()
        if state.get(key) != value
    ]
    if mismatches:
        return (
            "Die aktuellen Optionen passen nicht zum Resume-State "
            f"({', '.join(mismatches)}). Entweder dieselben Optionen verwenden "
            "oder mit --restart neu beginnen."
        )
    try:
        offset = int(state["partial_offset"])
    except (TypeError, ValueError):
        return "Der Resume-State enthält eine ungültige Dateiposition."
    try:
        partial_size = os.path.getsize(partial_path)
    except OSError as exc:
        return f"Die .partial-Datei kann nicht geprüft werden: {exc}"
    if offset < 0 or partial_size < offset:
        return "Die .partial-Datei ist kürzer als der gespeicherte Resume-Stand."
    return None


def api_failure(data):
    """Return an API error description, or ``None`` for a usable response."""
    error = data.get("error") if isinstance(data, dict) else None
    if error:
        if isinstance(error, dict):
            code = error.get("code", "unknown")
            info = error.get("info", "keine Details")
            return f"Wikipedia-API-Fehler {code}: {info}"
        return f"Wikipedia-API-Fehler: {error}"
    if not isinstance(data, dict) or "query" not in data:
        return "Wikipedia-API-Antwort enthält kein query-Objekt."
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia revision history as resumable XML."
    )
    parser.add_argument(
        "--project-dir",
        default=BASE_DIR,
        help=f"Project root containing data/ (default: {BASE_DIR})",
    )
    parser.add_argument("--article", required=True)
    parser.add_argument("--start", default="2001-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--min-revision-chars", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    slug = article_slug(args.article)
    data_dir = os.path.join(args.project_dir, "data", slug)
    os.makedirs(data_dir, exist_ok=True)

    xml_path = os.path.join(data_dir, f"{slug}.xml")
    partial_path = xml_path + ".partial"
    state_path = xml_path + ".state.json"

    if args.restart:
        for path in (partial_path, state_path):
            if os.path.exists(path):
                os.remove(path)

    try:
        state = load_state(state_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[FEHLER] Resume-State kann nicht gelesen werden: {exc}")
        print("[HINWEIS] Mit --restart einen kontrollierten Neudownload starten.")
        return 2

    if state:
        resume_error = validate_resume_state(state, args, partial_path)
        if resume_error:
            print(f"[FEHLER] {resume_error}")
            return 2
        params = state["params"]
        kept = int(state.get("kept", 0))
        skipped = int(state.get("skipped", 0))
        page_count = int(state.get("page_count", 0))
        partial_offset = int(state["partial_offset"])
        complete = bool(state.get("complete", False))
        mode = "r+"
        if complete:
            print(f"[RESUME] Download vollständig, finalisiere {kept} Revisionen")
        else:
            print(f"[RESUME] Weiter bei Seite {page_count + 1}, behalten: {kept}")
    else:
        params = initial_params(args.article, args.start, args.end)
        kept = 0
        skipped = 0
        page_count = 0
        partial_offset = 0
        complete = False
        mode = "w"
        print("[START] Neuer Download")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    print(f"[API] Artikel: {args.article}")
    print(f"[API] Zeitraum: {args.start} bis {args.end}")
    print(f"[XML] Ziel: {xml_path}")

    with open(partial_path, mode, encoding="utf-8") as handle:
        if mode == "w":
            write_header(handle, args.article)
            handle.flush()
            os.fsync(handle.fileno())
            partial_offset = handle.tell()
        else:
            # Remove a response body that may have been written after the last
            # durable checkpoint but before a process interruption.
            handle.seek(partial_offset)
            handle.truncate()

        if complete:
            print("[RESUME] Alle API-Seiten sind vorhanden; XML wird finalisiert.")
        else:
            while True:
                # Store the exact position *before* the request. If the process
                # dies while writing this page, resume truncates and refetches it.
                partial_offset = handle.tell()
                save_state(
                    state_path,
                    state_payload(
                        args, params, kept, skipped, page_count,
                        partial_offset, complete=False,
                    ),
                )
                page_count += 1
                try:
                    response = session.get(API_URL, params=params, timeout=60)
                    response.raise_for_status()
                    data = response.json()
                except requests.RequestException as exc:
                    print(f"[WARN] Netzwerkfehler auf Seite {page_count}: {exc}")
                    print("[WARN] Fortschritt bleibt in .partial und .state.json erhalten.")
                    print("[WARN] Starte denselben Befehl erneut, um fortzusetzen.")
                    return 2
                except ValueError as exc:
                    print(f"[FEHLER] Ungültige JSON-Antwort auf Seite {page_count}: {exc}")
                    print("[HINWEIS] Bestehende XML-Dateien bleiben unverändert.")
                    return 2

                failure = api_failure(data)
                if failure:
                    print(f"[FEHLER] {failure}")
                    print("[HINWEIS] Bestehende XML-Dateien bleiben unverändert.")
                    return 2

                pages = data.get("query", {}).get("pages", [])
                if not pages:
                    print("[FEHLER] Keine Seiten in der API-Antwort.")
                    print("[HINWEIS] Bestehende XML-Dateien bleiben unverändert.")
                    return 2

                page = pages[0] if isinstance(pages, list) else list(pages.values())[0]
                if page.get("missing") is not None or page.get("invalid") is not None:
                    print(f"[FEHLER] Wikipedia-Artikel nicht gefunden: {args.article}")
                    print("[HINWEIS] Bestehende XML-Dateien bleiben unverändert.")
                    return 2
                revs = page.get("revisions", [])

                batch_kept = 0
                for rev in revs:
                    content = rev.get("slots", {}).get("main", {}).get("content", "") or ""
                    if len(content) < args.min_revision_chars:
                        skipped += 1
                        continue
                    handle.write(revision_xml(rev))
                    kept += 1
                    batch_kept += 1

                handle.flush()
                os.fsync(handle.fileno())

                cont = data.get("continue")
                next_params = {**params, **cont} if cont else params
                partial_offset = handle.tell()
                save_state(
                    state_path,
                    state_payload(
                        args, next_params, kept, skipped, page_count,
                        partial_offset, complete=(not bool(cont) and kept > 0),
                    ),
                )

                print(
                    f"  Seite {page_count:4d}: {len(revs):2d} abgerufen, "
                    f"{batch_kept:2d} geschrieben | gesamt: {kept}"
                )

                if not cont:
                    if kept == 0:
                        print("[FEHLER] Im gewählten Zeitraum wurden keine Revisionen gespeichert.")
                        print("[HINWEIS] Bestehende XML-Dateien bleiben unverändert.")
                        return 2
                    break

                params = next_params
                time.sleep(SLEEP_SEC)

        write_footer(handle)

    os.replace(partial_path, xml_path)
    if os.path.exists(state_path):
        os.remove(state_path)

    size_mb = os.path.getsize(xml_path) / (1024 * 1024)
    print(f"[FERTIG] XML gespeichert: {xml_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
