"""Detect exact SHA1 reverts in Wikipedia revision histories."""

from pathlib import Path
import xml.etree.ElementTree as ET

REVERT_ANALYSIS_VERSION = "2026-10-22-exact-sha1-reverts-v15"


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
    """Return an audit-only bot hint for historical revisions."""
    folded_user = user.casefold()
    folded_tags = {tag.strip().casefold() for tag in tags.split("|") if tag.strip()}
    return "bot" in folded_tags or "bot" in folded_user


def load_revert_history(xml_path: Path) -> list[dict]:
    """Read the revision metadata needed for exact SHA1 revert detection."""
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
        revision = {
            "revision_id": child_text(element, "id") or "0",
            "timestamp": child_text(element, "timestamp"),
            "sha1": child_text(element, "sha1"),
            "user": user,
            "comment": child_text(element, "comment"),
            "tags": tags,
            "probable_bot": probable_bot(user, tags),
            "history_index": len(history),
        }
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
            event = {
                "revision_index": index,
                "revision_id": revision["revision_id"],
                "timestamp": revision["timestamp"],
                "reverter": reverter,
                "reverted_editor": reverted_editor,
                "reverted_revision_id": previous["revision_id"],
                "reverted_revision_index": index - 1,
                "directly_reverts_previous_revision": True,
                "restored_revision_id": revisions[target_index]["revision_id"],
                "restored_revision_index": target_index,
                "intervening_revision_count": index - target_index - 1,
                "pair": list(editor_pair(reverter, reverted_editor)),
                "revert_kind": "exact_sha1",
                "reversal_score": 1.0,
                "is_self_revert": reverter == reverted_editor,
                "involves_probable_bot": bool(
                    revision.get("probable_bot")
                    or previous.get("probable_bot")
                ),
            }
            events.append(event)

        if sha1:
            latest_state[sha1] = index

    return events


def analyze_reverts(
    history: list[dict],
    sampled: list[dict],
) -> dict:
    """Return full-history revert evidence aligned to sampled curve points."""
    events = detect_sha1_reverts(history)
    point_statistics = []
    previous_cutoff = -1

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
        point_statistics.append({
            "interval_revert_count": len(interval_events),
            "interval_exact_sha1_revert_count": len(interval_exact_events),
            "cumulative_revert_count": len(prefix_events),
        })
        previous_cutoff = cutoff

    return {
        "analysis_version": REVERT_ANALYSIS_VERSION,
        "method": "exact SHA1 state restoration",
        "definition": (
            "A revision is a revert when it restores an earlier SHA1-identical "
            "article state."
        ),
        "total_revert_count": len(events),
        "exact_sha1_revert_count": sum(
            event.get("revert_kind") == "exact_sha1" for event in events
        ),
        "probable_bot_revert_count": sum(
            event.get("involves_probable_bot", False) for event in events
        ),
        "excluded_probable_bot_revert_count": 0,
        "events": events,
        "point_statistics": point_statistics,
    }
