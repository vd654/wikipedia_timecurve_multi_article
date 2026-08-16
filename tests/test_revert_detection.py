from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revert_detection import (  # noqa: E402
    analyze_reverts,
    detect_edit_war_sessions,
    detect_sha1_reverts,
    gandica_c_measure,
    yasseri_m_measure,
)


def revision(index, sha1, user, probable_bot=False):
    return {
        "revision_id": str(100 + index),
        "timestamp": f"2020-01-01T00:0{index}:00Z",
        "sha1": sha1,
        "user": user,
        "probable_bot": probable_bot,
        "history_index": index,
    }


def test_mutual_sha1_reverts_drive_gandica_c_and_not_single_reverts():
    history = [
        revision(0, "state-a", "Alice"),
        revision(1, "state-b", "Bob"),
        revision(2, "state-a", "Alice"),
        revision(3, "state-b", "Bob"),
    ]
    events = detect_sha1_reverts(history)
    assert [(event["reverter"], event["reverted_editor"]) for event in events] == [
        ("Alice", "Bob"),
        ("Bob", "Alice"),
    ]
    assert all(event["mutual_in_full_history"] for event in events)
    assert gandica_c_measure(events)["value"] == 4.0

    # Yasseri M counts the editor pair once; its strongest-pair censor then
    # removes this sole pair from the final value.
    measure = yasseri_m_measure(history, events)
    assert measure["raw_mutual_revert_sum"] == 2.0
    assert measure["value"] == 0.0


def test_conflict_is_marked_when_the_revert_pair_becomes_mutual():
    history = [
        revision(0, "state-a", "Alice"),
        revision(1, "state-b", "Bob"),
        revision(2, "state-a", "Alice"),
        revision(3, "state-b", "Bob"),
    ]
    analysis = analyze_reverts(history, history)
    statistics = analysis["point_statistics"]
    assert statistics[2]["interval_mutual_revert_count"] == 0
    assert statistics[3]["interval_mutual_revert_count"] == 1
    assert statistics[3]["interval_edit_war_pairs"] == [["Alice", "Bob"]]
    assert statistics[3]["cumulative_gandica_c"] == 4.0


def test_probable_bot_reverts_do_not_create_human_edit_wars():
    history = [
        revision(0, "state-a", "Alice"),
        revision(1, "state-b", "CleanupBot", probable_bot=True),
        revision(2, "state-a", "Alice"),
        revision(3, "state-b", "CleanupBot", probable_bot=True),
    ]
    events = detect_sha1_reverts(history)
    assert all(event["involves_probable_bot"] for event in events)
    assert gandica_c_measure(events)["value"] == 0.0


def test_edit_war_session_requires_three_events_and_both_directions():
    history = [
        revision(0, "state-a", "Alice"),
        revision(1, "state-b", "Bob"),
        revision(2, "state-a", "Alice"),
        revision(3, "state-b", "Bob"),
        revision(4, "state-a", "Alice"),
    ]
    events = detect_sha1_reverts(history)
    sessions = detect_edit_war_sessions(events)
    assert len(sessions) == 1
    assert sessions[0]["pair"] == ["Alice", "Bob"]
    assert sessions[0]["event_count"] == 3
    assert sessions[0]["local_gandica_c"] == 6.0
    assert all(event.get("edit_war_session_id") for event in events)

    analysis = analyze_reverts(history, history)
    assert analysis["confirmed_edit_war_session_count"] == 1
    assert analysis["confirmed_edit_war_event_count"] == 3
    assert sum(
        point["interval_confirmed_edit_war_event_count"]
        for point in analysis["point_statistics"]
    ) == 3


def test_edit_war_session_uses_time_or_revision_proximity():
    events = [
        {
            "revision_index": index,
            "revision_id": str(index),
            "timestamp": timestamp,
            "reverter": reverter,
            "reverted_editor": reverted,
            "pair": ["Alice", "Bob"],
            "involves_probable_bot": False,
        }
        for index, timestamp, reverter, reverted in [
            (10, "2020-01-01T00:00:00Z", "Alice", "Bob"),
            # More than 48 h later, but within 15 revisions: same session.
            (20, "2020-01-04T00:00:00Z", "Bob", "Alice"),
            (21, "2020-01-04T00:01:00Z", "Alice", "Bob"),
            # Both limits exceeded: new, unconfirmed one-event session.
            (50, "2020-01-10T00:00:00Z", "Bob", "Alice"),
        ]
    ]
    sessions = detect_edit_war_sessions(events)
    assert len(sessions) == 1
    assert sessions[0]["start_revision_index"] == 10
    assert sessions[0]["end_revision_index"] == 21
    assert events[-1].get("edit_war_session_id") is None
