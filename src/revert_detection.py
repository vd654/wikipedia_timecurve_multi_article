"""Detect exact reverts and mutual-revert conflict in Wikipedia histories."""

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET


EDIT_WAR_MAX_HOURS = 48.0
EDIT_WAR_MAX_REVISION_GAP = 15
EDIT_WAR_MIN_EVENTS = 3


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


def load_revert_history(xml_path: Path) -> list[dict]:
    """Read only metadata needed for exact SHA1 revert detection."""
    history = []
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
        history.append({
            "revision_id": child_text(element, "id") or "0",
            "timestamp": child_text(element, "timestamp"),
            "sha1": child_text(element, "sha1"),
            "user": user,
            "tags": tags,
            "probable_bot": probable_bot(user, tags),
            "history_index": len(history),
        })
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
                events.append({
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
                    "involves_probable_bot": bool(
                        revision.get("probable_bot")
                        or previous.get("probable_bot")
                    ),
                })

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
            candidates.append({
                "pair": list(pair),
                "events": session_events,
                "event_count": len(session_events),
                "start_revision_index": session_events[0]["revision_index"],
                "end_revision_index": session_events[-1]["revision_index"],
                "start_timestamp": session_events[0]["timestamp"],
                "end_timestamp": session_events[-1]["timestamp"],
                "local_gandica_c": float(2 * len(session_events)),
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
    events = detect_sha1_reverts(history)
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
            }
            for session_id in interval_session_ids
        ]
        point_statistics.append({
            "interval_revert_count": len(interval_events),
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
        "method": "SHA1 identity reverts and human mutual-revert conflict",
        "definition": (
            "A revision is a revert when its SHA1 restores an earlier "
            "non-adjacent article state."
        ),
        "total_revert_count": len(events),
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
