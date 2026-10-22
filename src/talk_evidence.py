"""Extract contemporaneous evidence from a Wikipedia article's talk history."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from rapidfuzz.distance import Levenshtein

from text_cleaning import clean_wikitext


TALK_WINDOW_BEFORE_DAYS = 0
TALK_WINDOW_AFTER_DAYS = 0
MAX_TALK_ITEMS_PER_SESSION = 12
MAX_TALK_EXCERPT_CHARS = 1_200
TALK_CONTEXT_WORDS = 18
CONFLICT_TERMS = (
    "revert", "rv", "restore", "undo", "consensus", "dispute",
    "edit war", "3rr", "three-revert", "controvers", "object",
)
TOKEN_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)
STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "article", "been", "before",
    "being", "between", "but", "for", "from", "have", "into", "its",
    "make", "not", "only", "point", "section", "some", "talk", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "was", "were", "which", "with", "would",
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element | None, name: str) -> str:
    if element is None:
        return ""
    for child in element:
        if local_name(child.tag) == name:
            return child.text or ""
    return ""


def parse_timestamp(value: str) -> datetime:
    value = str(value or "").strip()
    if not value:
        raise ValueError("Leerer Wikipedia-Zeitstempel.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalise_space(value: object) -> str:
    return " ".join(str(value or "").split())


def _trim_excerpt(parts: list[str]) -> str:
    excerpt = " … ".join(part for part in parts if part)
    if len(excerpt) <= MAX_TALK_EXCERPT_CHARS:
        return excerpt
    return excerpt[:MAX_TALK_EXCERPT_CHARS].rsplit(" ", 1)[0].rstrip() + " …"


def changed_talk_excerpt(previous_text: str, current_text: str) -> str:
    """Return compact current-side passages changed by one talk revision."""
    previous_words = clean_wikitext(previous_text).split()
    current_words = clean_wikitext(current_text).split()
    if not current_words:
        return ""
    parts: list[str] = []
    for opcode in Levenshtein.opcodes(previous_words, current_words):
        if opcode.tag == "equal" or opcode.dest_start == opcode.dest_end:
            continue
        start = max(0, opcode.dest_start - TALK_CONTEXT_WORDS)
        end = min(len(current_words), opcode.dest_end + TALK_CONTEXT_WORDS)
        part = _normalise_space(" ".join(current_words[start:end]))
        if part and part not in parts:
            parts.append(part)
        if sum(len(item) for item in parts) >= MAX_TALK_EXCERPT_CHARS:
            break
    return _trim_excerpt(parts)


def session_time_windows(
    sessions: list[dict],
    *,
    before_days: int = TALK_WINDOW_BEFORE_DAYS,
    after_days: int = TALK_WINDOW_AFTER_DAYS,
) -> list[tuple[datetime, datetime]]:
    """Return merged UTC windows surrounding algorithmic candidates."""
    windows = sorted(
        (
            parse_timestamp(session["start_timestamp"]) - timedelta(days=before_days),
            parse_timestamp(session["end_timestamp"]) + timedelta(days=after_days),
        )
        for session in sessions
    )
    merged: list[tuple[datetime, datetime]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def load_talk_revision_changes(
    xml_path: Path,
    *,
    windows: list[tuple[datetime, datetime]] | None = None,
) -> list[dict]:
    """Read talk changes, optionally cleaning only revisions in time windows."""
    xml_path = xml_path.expanduser().resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"Diskussionsseiten-XML nicht gefunden: {xml_path}")
    revisions: list[dict] = []
    previous_text = ""
    for _, element in ET.iterparse(xml_path, events=("end",)):
        if local_name(element.tag) != "revision":
            continue
        contributor = next(
            (child for child in element if local_name(child.tag) == "contributor"),
            None,
        )
        current_text = child_text(element, "text")
        timestamp_text = child_text(element, "timestamp")
        timestamp = parse_timestamp(timestamp_text) if timestamp_text else None
        in_window = windows is None or (
            timestamp is not None
            and any(start <= timestamp <= end for start, end in windows)
        )
        excerpt = (
            changed_talk_excerpt(previous_text, current_text)
            if in_window
            else ""
        )
        revision = {
            "revision_id": child_text(element, "id") or "0",
            "timestamp": timestamp_text,
            "author": (
                child_text(contributor, "username")
                or child_text(contributor, "ip")
                or "anonymous"
            ),
            "comment": _normalise_space(child_text(element, "comment")),
            "changed_excerpt": excerpt,
        }
        if in_window and revision["timestamp"] and (
            excerpt or revision["comment"]
        ):
            revisions.append(revision)
        previous_text = current_text
        element.clear()
    return revisions


def _topic_tokens(session: dict) -> set[str]:
    values: list[str] = []
    values.extend(session.get("conflict_passages", []))
    values.extend(session.get("conflict_sentences", []))
    for analysis in session.get("full_context_edge_analyses", []):
        values.append(str(analysis.get("edge_label", "")))
        values.append(str(analysis.get("detailed_description", "")))
        values.extend(analysis.get("evidence", []))
        values.extend(analysis.get("added_keywords", []))
        values.extend(analysis.get("removed_keywords", []))
    return {
        token
        for value in values
        for token in TOKEN_RE.findall(str(value).casefold())
        if len(token) >= 4 and not token.isdigit() and token not in STOPWORDS
    }


def link_talk_evidence(
    sessions: list[dict],
    talk_revisions: list[dict],
    *,
    before_days: int = TALK_WINDOW_BEFORE_DAYS,
    after_days: int = TALK_WINDOW_AFTER_DAYS,
    max_items: int = MAX_TALK_ITEMS_PER_SESSION,
) -> None:
    """Attach ranked, contemporaneous talk revisions to each conflict candidate."""
    if before_days < 0 or after_days < 0 or max_items < 1:
        raise ValueError("Ungültiges Fenster für Diskussionsseiten-Evidenz.")
    prepared = [
        (parse_timestamp(revision["timestamp"]), revision)
        for revision in talk_revisions
    ]
    for session in sessions:
        start = parse_timestamp(session["start_timestamp"])
        end = parse_timestamp(session["end_timestamp"])
        window_start = start - timedelta(days=before_days)
        window_end = end + timedelta(days=after_days)
        participants = {
            str(participant).casefold()
            for participant in session.get("participants", session.get("pair", []))
        }
        topic_tokens = _topic_tokens(session)
        ranked: list[tuple[tuple, dict]] = []
        for timestamp, revision in prepared:
            if timestamp < window_start or timestamp > window_end:
                continue
            combined = " ".join((
                revision.get("comment", ""),
                revision.get("changed_excerpt", ""),
            )).casefold()
            tokens = set(TOKEN_RE.findall(combined))
            overlap = sorted(topic_tokens & tokens)
            conflict_terms = [term for term in CONFLICT_TERMS if term in combined]
            participant_match = revision.get("author", "").casefold() in participants
            if len(overlap) < 2 and not (
                participant_match and conflict_terms
            ):
                continue
            distance_seconds = min(
                abs((timestamp - start).total_seconds()),
                abs((timestamp - end).total_seconds()),
            )
            evidence = {
                **revision,
                "participant_match": participant_match,
                "topic_token_overlap": overlap[:20],
                "conflict_terms": conflict_terms,
            }
            score = (
                int(participant_match),
                min(len(overlap), 20),
                int(bool(conflict_terms)),
                -distance_seconds,
                timestamp.timestamp(),
            )
            ranked.append((score, evidence))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [evidence for _, evidence in ranked[:max_items]]
        selected.sort(key=lambda item: (item["timestamp"], item["revision_id"]))
        session["talk_page_evidence"] = selected
        session["talk_page_evidence_count"] = len(selected)
        session["talk_page_window"] = {
            "start": window_start.isoformat().replace("+00:00", "Z"),
            "end": window_end.isoformat().replace("+00:00", "Z"),
            "days_before": before_days,
            "days_after": after_days,
            "maximum_items": max_items,
        }
