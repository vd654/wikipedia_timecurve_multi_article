"""Detect exact reverts and mutual-revert conflict in Wikipedia histories."""

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from rapidfuzz.distance import Levenshtein

EDIT_WAR_MAX_HOURS = 48.0
EDIT_WAR_MAX_REVISION_GAP = 15
EDIT_WAR_MIN_EVENTS = 3
PARTIAL_REVERT_MIN_SCORE = 0.72
PARTIAL_REVERT_MIN_MATCHED_TOKENS = 4
PARTIAL_REVERT_MAX_EVIDENCE_CHARS = 360
EDIT_WAR_DEFINITION = (
    "Ein Edit War liegt vor, wenn zwei oder mehr Nutzer abwechselnd die "
    "Änderungen der jeweils anderen Seite wieder rückgängig machen "
    "(\"revertieren\"), statt eine inhaltliche Diskussion zu führen."
)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element | None, name: str) -> str:
    if element is None:
        return ""
    for child in element:
        if local_name(child.tag) == name:
            return child.text or ""
    return ""


def probable_bot(user: str, tags: str = "") -> bool:
    """Return a conservative offline bot heuristic for historical revisions."""
    folded_user = user.casefold()
    folded_tags = {tag.strip().casefold() for tag in tags.split("|") if tag.strip()}
    return "bot" in folded_tags or "bot" in folded_user


CONTENT_TOKEN_RE = re.compile(r"[\w'-]+", flags=re.UNICODE)
CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "which", "with",
}


def _content_tokens(words: list[str]) -> list[str]:
    """Normalize changed words while removing low-information glue words."""
    tokens = []
    for word in words:
        for token in CONTENT_TOKEN_RE.findall(word.casefold()):
            if token not in CONTENT_STOPWORDS and (len(token) > 2 or token.isdigit()):
                tokens.append(token)
    return tokens


def _snippet(words: list[str]) -> str:
    text = " ".join(words)
    text = re.sub(r"<!--.*?-->|<ref\b.*?</ref\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"[{}\[\]|'=]+", " ", text)
    text = re.sub(r"<[^>]+>|\s+", " ", text).strip()
    return text[:PARTIAL_REVERT_MAX_EVIDENCE_CHARS].strip()


def build_change_signature(previous_text: str, current_text: str) -> dict:
    """Return compact changed passages for passage-level reversal matching."""
    previous_words = previous_text.split()
    current_words = current_text.split()
    units = []
    for opcode in Levenshtein.opcodes(previous_words, current_words):
        if opcode.tag == "equal":
            continue
        removed_words = previous_words[opcode.src_start:opcode.src_end]
        added_words = current_words[opcode.dest_start:opcode.dest_end]
        removed_tokens = _content_tokens(removed_words)
        added_tokens = _content_tokens(added_words)
        if not removed_tokens and not added_tokens:
            continue
        units.append({
            "added_tokens": added_tokens,
            "removed_tokens": removed_tokens,
            "added_text": _snippet(added_words),
            "removed_text": _snippet(removed_words),
        })
    return {"units": units}


def load_revert_history(
    xml_path: Path, *, include_text_changes: bool = False
) -> list[dict]:
    """Read revision metadata and optionally compact consecutive text diffs."""
    history = []
    previous_text = ""
    for _, element in ET.iterparse(xml_path, events=("end",)):
        if local_name(element.tag) != "revision":
            continue
        contributor = next(
            (child for child in element if local_name(child.tag) == "contributor"),
            None,
        )
        user = (
            child_text(contributor, "username")
            or child_text(contributor, "ip")
            or "anonymous"
        )
        tags = child_text(element, "tags")
        revision = {
            "revision_id": child_text(element, "id") or "0",
            "timestamp": child_text(element, "timestamp"),
            "sha1": child_text(element, "sha1"),
            "user": user,
            "tags": tags,
            "probable_bot": probable_bot(user, tags),
            "history_index": len(history),
        }
        if include_text_changes:
            current_text = child_text(element, "text")
            revision["change_signature"] = (
                build_change_signature(previous_text, current_text)
                if history else {"units": []}
            )
            previous_text = current_text
        history.append(revision)
        element.clear()
    return history


def history_window_for_points(
    history: list[dict], points: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Slice full history to the plotted period and align sampled points."""
    by_revision_id = {
        str(revision["revision_id"]): index
        for index, revision in enumerate(history)
    }
    point_indices = []
    for point in points:
        revision_id = str(point["revision_id"])
        if revision_id not in by_revision_id:
            raise ValueError(
                f"Revisions-ID {revision_id} aus der Projektion fehlt im XML."
            )
        point_indices.append(by_revision_id[revision_id])
    if point_indices != sorted(point_indices):
        raise ValueError("Die Projektionspunkte sind nicht chronologisch geordnet.")
    start, end = point_indices[0], point_indices[-1]
    window = [revision.copy() for revision in history[start:end + 1]]
    for index, revision in enumerate(window):
        revision["history_index"] = index
    sampled = [
        {**point, "history_index": source_index - start}
        for point, source_index in zip(points, point_indices)
    ]
    return window, sampled


def editor_pair(left: str, right: str) -> tuple[str, str]:
    """Return a stable key for an unordered editor pair."""
    return tuple(sorted((left, right)))


def detect_sha1_reverts(revisions: list[dict]) -> list[dict]:
    """Find revisions that restore an earlier SHA1-identical article state."""
    latest_state = {}
    events = []

    for index, revision in enumerate(revisions):
        sha1 = revision.get("sha1", "")
        target_index = latest_state.get(sha1) if sha1 else None

        # A direct duplicate contains no intervening edit and is not a revert.
        if target_index is not None and target_index < index - 1:
            previous = revisions[index - 1]
            reverter = revision["user"]
            reverted_editor = previous["user"]
            if reverter != reverted_editor:
                event = {
                    "revision_index": index,
                    "revision_id": revision["revision_id"],
                    "timestamp": revision["timestamp"],
                    "reverter": reverter,
                    "reverted_editor": reverted_editor,
                    "reverted_revision_id": previous["revision_id"],
                    "restored_revision_id": revisions[target_index]["revision_id"],
                    "restored_revision_index": target_index,
                    "intervening_revision_count": index - target_index - 1,
                    "pair": list(editor_pair(reverter, reverted_editor)),
                    "revert_kind": "exact_sha1",
                    "reversal_score": 1.0,
                    "involves_probable_bot": bool(
                        revision.get("probable_bot")
                        or previous.get("probable_bot")
                    ),
                }
                signature = revision.get("change_signature", {})
                evidence = list(dict.fromkeys(
                    text
                    for unit in signature.get("units", [])
                    for text in (unit.get("removed_text", ""), unit.get("added_text", ""))
                    if text
                ))
                if evidence:
                    event["conflict_passages"] = evidence
                events.append(event)

        if sha1:
            latest_state[sha1] = index

    directions = defaultdict(set)
    for event in events:
        if event["involves_probable_bot"]:
            continue
        directions[tuple(event["pair"])].add(
            (event["reverter"], event["reverted_editor"])
        )
    mutual_pairs = {
        pair for pair, pair_directions in directions.items()
        if len(pair_directions) == 2
    }
    for event in events:
        event["mutual_in_full_history"] = (
            not event["involves_probable_bot"]
            and tuple(event["pair"]) in mutual_pairs
        )
    return events


def _counter_overlap(left: list[str], right: list[str]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def _reversal_unit_score(earlier: dict, later: dict) -> tuple[float, int]:
    """Measure how completely a later passage reverses an earlier passage."""
    matched = (
        _counter_overlap(earlier["added_tokens"], later["removed_tokens"])
        + _counter_overlap(earlier["removed_tokens"], later["added_tokens"])
    )
    earlier_size = len(earlier["added_tokens"]) + len(earlier["removed_tokens"])
    later_size = len(later["added_tokens"]) + len(later["removed_tokens"])
    if not matched or not earlier_size or not later_size:
        return 0.0, 0
    precision = matched / later_size
    recall = matched / earlier_size
    score = 2 * precision * recall / (precision + recall)
    return float(score), matched


def detect_partial_text_reverts(
    revisions: list[dict],
    *,
    max_hours: float = EDIT_WAR_MAX_HOURS,
    max_revision_gap: int = EDIT_WAR_MAX_REVISION_GAP,
    min_score: float = PARTIAL_REVERT_MIN_SCORE,
    min_matched_tokens: int = PARTIAL_REVERT_MIN_MATCHED_TOKENS,
) -> list[dict]:
    """Detect near-reverts by matching inverse changed passages in a short window."""
    events = []
    for later_index in range(1, len(revisions)):
        later = revisions[later_index]
        later_units = later.get("change_signature", {}).get("units", [])
        if not later_units:
            continue
        best = None
        for earlier_index in range(later_index - 1, 0, -1):
            earlier = revisions[earlier_index]
            revision_gap = later_index - earlier_index
            hour_gap = (
                _timestamp(later["timestamp"])
                - _timestamp(earlier["timestamp"])
            ).total_seconds() / 3600.0
            if hour_gap > max_hours and revision_gap > max_revision_gap:
                break
            if later["user"] == earlier["user"]:
                continue
            for earlier_unit in earlier.get("change_signature", {}).get("units", []):
                for later_unit in later_units:
                    score, matched = _reversal_unit_score(earlier_unit, later_unit)
                    candidate = (score, matched, -revision_gap, earlier_index, earlier_unit, later_unit)
                    if best is None or candidate[:4] > best[:4]:
                        best = candidate
        if best is None:
            continue
        score, matched, _, earlier_index, earlier_unit, later_unit = best
        if score < min_score or matched < min_matched_tokens:
            continue
        earlier = revisions[earlier_index]
        evidence = list(dict.fromkeys(
            text
            for text in (
                earlier_unit.get("removed_text", ""),
                earlier_unit.get("added_text", ""),
                later_unit.get("removed_text", ""),
                later_unit.get("added_text", ""),
            )
            if text
        ))
        events.append({
            "revision_index": later_index,
            "revision_id": later["revision_id"],
            "timestamp": later["timestamp"],
            "reverter": later["user"],
            "reverted_editor": earlier["user"],
            "reverted_revision_id": earlier["revision_id"],
            "restored_revision_id": None,
            "restored_revision_index": None,
            "intervening_revision_count": later_index - earlier_index - 1,
            "pair": list(editor_pair(later["user"], earlier["user"])),
            "revert_kind": "partial_text",
            "reversal_score": score,
            "matched_content_token_count": matched,
            "conflict_passages": evidence,
            "involves_probable_bot": bool(
                later.get("probable_bot") or earlier.get("probable_bot")
            ),
        })
    return events


def merge_revert_events(exact_events: list[dict], partial_events: list[dict]) -> list[dict]:
    """Prefer exact evidence when both detectors identify the same revision."""
    exact_revision_ids = {str(event["revision_id"]) for event in exact_events}
    merged = exact_events + [
        event for event in partial_events
        if str(event["revision_id"]) not in exact_revision_ids
    ]
    merged.sort(key=lambda event: (event["revision_index"], event["revision_id"]))
    directions = defaultdict(set)
    for event in merged:
        if not event.get("involves_probable_bot", False):
            directions[tuple(event["pair"])].add(
                (event["reverter"], event["reverted_editor"])
            )
    for event in merged:
        event["mutual_in_full_history"] = (
            not event.get("involves_probable_bot", False)
            and len(directions[tuple(event["pair"])]) == 2
        )
    return merged


def yasseri_m_measure(revisions: list[dict], events: list[dict]) -> dict:
    """Calculate M from mutual reverts, excluding the strongest editor pair."""
    edit_counts = Counter(
        revision["user"]
        for revision in revisions
        if not revision.get("probable_bot", False)
    )
    directions = defaultdict(set)
    for event in events:
        if event.get("involves_probable_bot", False):
            continue
        pair = tuple(event["pair"])
        directions[pair].add((event["reverter"], event["reverted_editor"]))

    mutual_pairs = {
        pair for pair, pair_directions in directions.items()
        if len(pair_directions) == 2
    }
    contributions = {
        pair: float(min(edit_counts[pair[0]], edit_counts[pair[1]]))
        for pair in mutual_pairs
    }
    mutual_events = [
        event for event in events
        if not event.get("involves_probable_bot", False)
        and tuple(event["pair"]) in mutual_pairs
    ]

    involved_editors = {editor for pair in mutual_pairs for editor in pair}
    raw_sum = float(sum(contributions.values()))
    if contributions:
        top_pair, top_value = max(
            contributions.items(), key=lambda item: (item[1], item[0])
        )
    else:
        top_pair, top_value = None, 0.0

    censored_sum = raw_sum - float(top_value)
    score = float(len(involved_editors) * censored_sum)
    return {
        "value": score,
        "raw_mutual_revert_sum": raw_sum,
        "censored_mutual_revert_sum": censored_sum,
        "mutually_reverting_editor_count": len(involved_editors),
        "mutual_pair_count": len(mutual_pairs),
        "mutual_revert_count": len(mutual_events),
        "excluded_top_pair": list(top_pair) if top_pair else None,
        "excluded_top_pair_contribution": float(top_value),
        "mutual_pairs": [list(pair) for pair in sorted(mutual_pairs)],
    }


def gandica_c_measure(events: list[dict]) -> dict:
    """Calculate Gandica et al.'s C from human mutual-revert pairs."""
    directions = defaultdict(set)
    for event in events:
        if event.get("involves_probable_bot", False):
            continue
        pair = tuple(event["pair"])
        directions[pair].add((event["reverter"], event["reverted_editor"]))
    mutual_pairs = {
        pair for pair, pair_directions in directions.items()
        if len(pair_directions) == 2
    }
    counts = Counter(
        tuple(event["pair"])
        for event in events
        if not event.get("involves_probable_bot", False)
        and tuple(event["pair"]) in mutual_pairs
    )
    involved_editors = {editor for pair in mutual_pairs for editor in pair}
    reversal_sum = int(sum(counts.values()))
    return {
        "value": float(len(involved_editors) * reversal_sum),
        "mutually_reverting_editor_count": len(involved_editors),
        "mutual_pair_count": len(mutual_pairs),
        "mutual_revert_count": reversal_sum,
        "mutual_pairs": [list(pair) for pair in sorted(mutual_pairs)],
        "pair_revert_counts": [
            {"pair": list(pair), "revert_count": counts[pair]}
            for pair in sorted(mutual_pairs)
        ],
    }


def _timestamp(timestamp: str) -> datetime:
    """Parse the UTC timestamps used by MediaWiki history dumps."""
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def detect_edit_war_sessions(
    events: list[dict],
    *,
    max_hours: float = EDIT_WAR_MAX_HOURS,
    max_revision_gap: int = EDIT_WAR_MAX_REVISION_GAP,
    min_events: int = EDIT_WAR_MIN_EVENTS,
) -> list[dict]:
    """Group human mutual reverts into conservative local conflict sessions.

    Consecutive reverts by the same unordered editor pair stay in one session
    when they are close either in real time or in revision order. A confirmed
    edit-war session must contain both revert directions and at least three
    events. No projected/geometric evidence is used at this stage.
    """
    human_events = [
        event for event in events
        if not event.get("involves_probable_bot", False)
    ]
    by_pair = defaultdict(list)
    for event in human_events:
        by_pair[tuple(event["pair"])].append(event)

    candidates = []
    for pair, pair_events in sorted(by_pair.items()):
        ordered = sorted(
            pair_events,
            key=lambda event: (event["revision_index"], event["revision_id"]),
        )
        sessions = []
        current = []
        for event in ordered:
            if current:
                previous = current[-1]
                hour_gap = (
                    _timestamp(event["timestamp"])
                    - _timestamp(previous["timestamp"])
                ).total_seconds() / 3600.0
                revision_gap = (
                    int(event["revision_index"])
                    - int(previous["revision_index"])
                )
                # "48 hours OR 15 revisions": only split when both forms of
                # proximity have been exceeded.
                if hour_gap > max_hours and revision_gap > max_revision_gap:
                    sessions.append(current)
                    current = []
            current.append(event)
        if current:
            sessions.append(current)

        for session_events in sessions:
            directions = {
                (event["reverter"], event["reverted_editor"])
                for event in session_events
            }
            if len(session_events) < min_events or len(directions) < 2:
                continue
            conflict_passages = list(dict.fromkeys(
                passage
                for event in session_events
                for passage in event.get("conflict_passages", [])
                if passage
            ))
            candidates.append({
                "pair": list(pair),
                "events": session_events,
                "event_count": len(session_events),
                "start_revision_index": session_events[0]["revision_index"],
                "end_revision_index": session_events[-1]["revision_index"],
                "start_timestamp": session_events[0]["timestamp"],
                "end_timestamp": session_events[-1]["timestamp"],
                "local_gandica_c": float(2 * len(session_events)),
                "revert_kinds": sorted({
                    event.get("revert_kind", "exact_sha1")
                    for event in session_events
                }),
                "mean_reversal_score": float(sum(
                    event.get("reversal_score", 1.0) for event in session_events
                ) / len(session_events)),
                "conflict_passages": conflict_passages,
            })

    candidates.sort(key=lambda session: (
        session["start_revision_index"],
        session["end_revision_index"],
        session["pair"],
    ))
    sessions = []
    for number, candidate in enumerate(candidates, start=1):
        session_id = f"edit-war-{number}"
        event_ids = []
        for event in candidate.pop("events"):
            event["edit_war_session_id"] = session_id
            event_ids.append(str(event["revision_id"]))
        sessions.append({
            "session_id": session_id,
            **candidate,
            "event_revision_ids": event_ids,
        })
    return sessions


def analyze_reverts(history: list[dict], sampled: list[dict]) -> dict:
    """Return full-history revert evidence aligned to sampled curve points."""
    exact_events = detect_sha1_reverts(history)
    partial_events = (
        detect_partial_text_reverts(history)
        if any(revision.get("change_signature") for revision in history)
        else []
    )
    events = merge_revert_events(exact_events, partial_events)
    edit_war_sessions = detect_edit_war_sessions(events)
    sessions_by_id = {
        session["session_id"]: session for session in edit_war_sessions
    }
    point_statistics = []
    previous_cutoff = -1
    previous_c = 0.0

    for revision in sampled:
        cutoff = revision["history_index"]
        prefix_events = [
            event for event in events if event["revision_index"] <= cutoff
        ]
        interval_events = [
            event for event in prefix_events
            if previous_cutoff < event["revision_index"] <= cutoff
        ]
        interval_exact_events = [
            event for event in interval_events
            if event.get("revert_kind") == "exact_sha1"
        ]
        interval_partial_events = [
            event for event in interval_events
            if event.get("revert_kind") == "partial_text"
        ]
        measure = yasseri_m_measure(history[:cutoff + 1], prefix_events)
        conflict = gandica_c_measure(prefix_events)
        current_mutual_pairs = {
            tuple(pair) for pair in conflict["mutual_pairs"]
        }
        interval_mutual_events = [
            event for event in interval_events
            if tuple(event["pair"]) in current_mutual_pairs
            and not event.get("involves_probable_bot", False)
        ]
        interval_pairs = sorted({
            tuple(event["pair"]) for event in interval_mutual_events
        })
        interval_edit_war_events = [
            event for event in interval_events
            if event.get("edit_war_session_id") in sessions_by_id
        ]
        interval_session_ids = sorted({
            event["edit_war_session_id"]
            for event in interval_edit_war_events
        })
        interval_sessions = [
            {
                "session_id": session_id,
                "pair": sessions_by_id[session_id]["pair"],
                "event_count": sessions_by_id[session_id]["event_count"],
                "local_gandica_c": sessions_by_id[session_id]["local_gandica_c"],
                "revert_kinds": sessions_by_id[session_id]["revert_kinds"],
                "mean_reversal_score": sessions_by_id[session_id][
                    "mean_reversal_score"
                ],
                "conflict_passages": sessions_by_id[session_id][
                    "conflict_passages"
                ],
            }
            for session_id in interval_session_ids
        ]
        point_statistics.append({
            "interval_revert_count": len(interval_events),
            "interval_exact_sha1_revert_count": len(interval_exact_events),
            "interval_partial_text_revert_count": len(interval_partial_events),
            "interval_mutual_revert_count": len(interval_mutual_events),
            "interval_edit_war_pairs": [list(pair) for pair in interval_pairs],
            "interval_confirmed_edit_war_event_count": len(
                interval_edit_war_events
            ),
            "interval_confirmed_edit_war_sessions": interval_sessions,
            "cumulative_revert_count": len(prefix_events),
            "cumulative_yasseri_m": measure["value"],
            "cumulative_gandica_c": conflict["value"],
            "gandica_c_delta": conflict["value"] - previous_c,
        })
        previous_cutoff = cutoff
        previous_c = conflict["value"]

    final_measure = yasseri_m_measure(history, events)
    final_conflict = gandica_c_measure(events)
    return {
        "method": "exact SHA1 and inverse passage-diff mutual-revert conflict",
        "definition": (
            "A revision is a revert when its SHA1 restores an earlier "
            "non-adjacent article state."
        ),
        "edit_war_definition": EDIT_WAR_DEFINITION,
        "total_revert_count": len(events),
        "exact_sha1_revert_count": sum(
            event.get("revert_kind") == "exact_sha1" for event in events
        ),
        "partial_text_revert_count": sum(
            event.get("revert_kind") == "partial_text" for event in events
        ),
        "total_mutual_revert_count": final_measure["mutual_revert_count"],
        "confirmed_edit_war_session_count": len(edit_war_sessions),
        "confirmed_edit_war_event_count": sum(
            session["event_count"] for session in edit_war_sessions
        ),
        "edit_war_session_parameters": {
            "max_hours": EDIT_WAR_MAX_HOURS,
            "max_revision_gap": EDIT_WAR_MAX_REVISION_GAP,
            "minimum_events": EDIT_WAR_MIN_EVENTS,
            "requires_both_directions": True,
            "proximity_rule": "time_or_revision_gap",
            "partial_revert_min_score": PARTIAL_REVERT_MIN_SCORE,
            "partial_revert_min_matched_tokens": PARTIAL_REVERT_MIN_MATCHED_TOKENS,
            "partial_revert_scope": "inverse consecutive changed passages",
        },
        "excluded_probable_bot_revert_count": sum(
            event.get("involves_probable_bot", False) for event in events
        ),
        "yasseri_m": final_measure,
        "gandica_c": final_conflict,
        "events": events,
        "edit_war_sessions": edit_war_sessions,
        "point_statistics": point_statistics,
    }
