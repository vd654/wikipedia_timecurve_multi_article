#!/usr/bin/env python3
"""Download public metadata for the exact revisions used by a projection."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from revision_keywords import load_visualization_revision_ids, resolve_projection_path


API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WikiTimeCurve/1.0 (research project; contact: your@email.com)"
BATCH_SIZE = 50


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def metadata_output_path(projection_path: Path) -> Path:
    visualization_name = projection_path.stem.removesuffix("_projection")
    return projection_path.with_name(f"{visualization_name}_revision_metadata.json")


def parse_revision_rows(data: dict) -> dict[str, dict]:
    error = data.get("error") if isinstance(data, dict) else None
    if error:
        raise RuntimeError(f"Wikipedia-API-Fehler: {error}")
    pages = data.get("query", {}).get("pages", [])
    if not isinstance(pages, list):
        raise RuntimeError("Wikipedia-API-Antwort enthält keine gültige Seitenliste.")

    rows: dict[str, dict] = {}
    for page in pages:
        for revision in page.get("revisions", []):
            revision_id = str(revision.get("revid", ""))
            if not revision_id:
                continue
            rows[revision_id] = {
                "revision_id": revision_id,
                "author": str(revision.get("user", "")),
                "timestamp": str(revision.get("timestamp", "")),
                "comment": str(revision.get("comment", "")),
                "user_hidden": bool(revision.get("userhidden", False)),
                "comment_hidden": bool(revision.get("commenthidden", False)),
            }
    return rows


def download_metadata(
    revision_ids: list[str],
    output_path: Path,
    *,
    session: requests.Session | None = None,
) -> dict:
    """Fetch metadata in resumable batches and return it in projection order."""
    http = session or requests.Session()
    http.headers.update({"User-Agent": USER_AGENT})

    cached: dict[str, dict] = {}
    if output_path.exists():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        cached = {
            str(row.get("revision_id", "")): row
            for row in payload.get("revisions", [])
            if row.get("revision_id")
        }

    total_batches = (len(revision_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_number, start in enumerate(range(0, len(revision_ids), BATCH_SIZE), 1):
        batch = revision_ids[start:start + BATCH_SIZE]
        missing = [revision_id for revision_id in batch if revision_id not in cached]
        if not missing:
            print(f"[{batch_number}/{total_batches}] Metadaten aus Cache", flush=True)
            continue

        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "revisions",
            "revids": "|".join(missing),
            "rvprop": "ids|timestamp|user|comment",
        }
        last_error: Exception | None = None
        for retry in range(6):
            try:
                response = http.get(API_URL, params=params, timeout=60)
                response.raise_for_status()
                rows = parse_revision_rows(response.json())
                absent = [revision_id for revision_id in missing if revision_id not in rows]
                if absent:
                    raise RuntimeError(
                        "Wikipedia lieferte keine Metadaten für Revisionen: "
                        + ", ".join(absent[:5])
                    )
                cached.update(rows)
                last_error = None
                break
            except (requests.RequestException, ValueError, RuntimeError) as error:
                last_error = error
                if retry == 5:
                    raise
                delay = min(30, 2 ** retry)
                print(f"  Wiederholung in {delay}s: {error}", flush=True)
                time.sleep(delay)
        if last_error is not None:
            raise last_error

        partial_payload = {
            "source": "Wikipedia Action API",
            "api_url": API_URL,
            "revision_count": len(revision_ids),
            "complete": len(cached) >= len(revision_ids),
            "revisions": [cached[item] for item in revision_ids if item in cached],
        }
        write_json(output_path, partial_payload)
        print(
            f"[{batch_number}/{total_batches}] {len(missing)} Metadaten geladen",
            flush=True,
        )

    missing = [revision_id for revision_id in revision_ids if revision_id not in cached]
    if missing:
        raise RuntimeError(
            "Metadatendatei ist unvollständig; fehlend: " + ", ".join(missing[:5])
        )
    result = {
        "source": "Wikipedia Action API",
        "api_url": API_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "revision_count": len(revision_ids),
        "complete": True,
        "revisions": [cached[revision_id] for revision_id in revision_ids],
    }
    write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download author, timestamp, and edit comment for projection revisions."
    )
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projection_path = resolve_projection_path(args.projection)
    revision_ids = load_visualization_revision_ids(projection_path)
    output_path = args.output or metadata_output_path(projection_path)
    print(f"Projektion: {projection_path}", flush=True)
    print(f"Revisionen: {len(revision_ids)}", flush=True)
    download_metadata(revision_ids, output_path)
    print(f"Metadaten gespeichert: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
