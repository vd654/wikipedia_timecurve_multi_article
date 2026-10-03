from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revert_detection import (  # noqa: E402
    PARTIAL_REVERT_DEFINITION,
    analyze_reverts,
    build_change_signature,
    detect_edit_war_sessions,
    detect_partial_trajectory_reverts,
    detect_semantic_counter_reverts,
    detect_sha1_reverts,
    gandica_c_measure,
    merge_revert_events,
    yasseri_m_measure,
)


def revision(index, sha1, user, probable_bot=False, comment=""):
    row = {
        "revision_id": str(100 + index),
        "timestamp": f"2020-01-01T00:0{index}:00Z",
        "sha1": sha1,
        "user": user,
        "probable_bot": probable_bot,
        "history_index": index,
    }
    if comment:
        row["comment"] = comment
    return row


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


def test_exact_sha1_self_revert_is_kept_but_is_not_an_edit_war():
    history = [
        revision(0, "state-a", "Alice"),
        revision(1, "state-b", "Alice"),
        revision(2, "state-a", "Alice"),
    ]

    events = detect_sha1_reverts(history)
    sessions = detect_edit_war_sessions(events)

    assert len(events) == 1
    assert events[0]["revert_kind"] == "exact_sha1"
    assert events[0]["is_self_revert"] is True
    assert events[0]["restored_revision_id"] == "100"
    assert events[0]["pair"] == ["Alice", "Alice"]
    assert sessions == []


def test_partial_revert_takes_precedence_over_exact_sha1_for_same_revision():
    shared = {
        "revision_index": 2,
        "revision_id": "102",
        "reverter": "Alice",
        "reverted_editor": "Bob",
        "pair": ["Alice", "Bob"],
    }
    exact = {**shared, "revert_kind": "exact_sha1"}
    partial = {**shared, "revert_kind": "partial_trajectory"}

    merged = merge_revert_events([exact], [partial])

    assert len(merged) == 1
    assert merged[0]["revert_kind"] == "partial_trajectory"


def test_semantic_partial_revert_takes_precedence_over_exact_sha1():
    shared = {
        "revision_index": 2,
        "revision_id": "102",
        "reverter": "Alice",
        "reverted_editor": "Bob",
        "pair": ["Alice", "Bob"],
    }
    exact = {**shared, "revert_kind": "exact_sha1"}
    semantic = {**shared, "revert_kind": "semantic_text"}

    merged = merge_revert_events([exact], [], [semantic])

    assert len(merged) == 1
    assert merged[0]["revert_kind"] == "semantic_text"


def test_conflict_is_marked_when_the_revert_pair_becomes_mutual():
    history = [
        revision(0, "state-a", "Alice"),
        revision(1, "state-b", "Bob"),
        revision(2, "state-a", "Alice"),
        revision(3, "state-b", "Bob"),
    ]
    analysis = analyze_reverts(history, history)
    assert analysis["partial_revert_definition"] == PARTIAL_REVERT_DEFINITION
    parameters = analysis["edit_war_session_parameters"]
    assert parameters["partial_revert_allows_piecewise_return"] is True
    assert parameters["partial_revert_detection_basis"] == "visible_2d_trajectory"
    assert parameters["partial_revert_requires_consecutive_revisions"] is True
    assert parameters["partial_revert_requires_monotonic_return"] is True
    assert parameters["partial_revert_precedence_over_exact_sha1"] is True
    assert parameters["partial_revert_min_matched_tokens"] is None
    statistics = analysis["point_statistics"]
    assert statistics[2]["interval_mutual_revert_count"] == 0
    assert statistics[2]["interval_exact_sha1_revert_count"] == 1
    assert statistics[2]["interval_partial_text_revert_count"] == 0
    assert statistics[3]["interval_mutual_revert_count"] == 1
    assert statistics[3]["interval_exact_sha1_revert_count"] == 1
    assert statistics[3]["interval_edit_war_pairs"] == [["Alice", "Bob"]]
    assert statistics[3]["cumulative_gandica_c"] == 4.0


def test_probable_bot_reverts_contribute_to_edit_wars():
    history = [
        revision(0, "state-a", "Alice"),
        revision(1, "state-b", "CleanupBot", probable_bot=True),
        revision(2, "state-a", "Alice"),
        revision(3, "state-b", "CleanupBot", probable_bot=True),
        revision(4, "state-a", "Alice"),
    ]
    events = detect_sha1_reverts(history)
    assert all(event["involves_probable_bot"] for event in events)
    assert all(event["mutual_in_full_history"] for event in events)
    assert gandica_c_measure(events)["value"] == 6.0

    analysis = analyze_reverts(history, history)
    assert analysis["confirmed_edit_war_session_count"] == 1
    assert analysis["confirmed_edit_war_event_count"] == 3
    assert analysis["probable_bot_revert_count"] == 3
    assert analysis["excluded_probable_bot_revert_count"] == 0
    assert analysis["edit_war_session_parameters"]["exclude_probable_bots"] is False


def test_edit_war_session_accepts_an_uninterrupted_back_and_forth():
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


def test_consecutive_2d_counter_steps_detect_partial_reverts_and_form_a_session():
    users = ["Initial", "Alice", "Bob", "Alice", "Bob"]
    coordinates = [[0.0, 0.0], [1.0, 0.0], [0.2, 0.0], [1.0, 0.0], [0.2, 0.0]]
    history = [
        {**revision(index, f"unique-{index}", user), "mds": coordinate}
        for index, (user, coordinate) in enumerate(zip(users, coordinates))
    ]
    vectors = {
        row["revision_id"]: row["mds"] for row in history
    }

    events = detect_partial_trajectory_reverts(history, vectors)
    analysis = analyze_reverts(history, history)

    assert len(events) == 3
    assert all(event["revert_kind"] == "partial_trajectory" for event in events)
    assert all(event["trajectory_direction_cosine"] == -1.0 for event in events)
    assert analysis["partial_text_revert_count"] == 3
    assert analysis["confirmed_edit_war_session_count"] == 1
    session = analysis["edit_war_sessions"][0]
    assert session["revert_kinds"] == ["partial_trajectory"]


def test_two_consecutive_partial_counter_reverts_form_an_edit_war():
    events = [
        {
            "revision_index": 8,
            "revision_id": "108",
            "timestamp": "2020-01-01T00:08:00Z",
            "reverter": "Bob",
            "reverted_editor": "Alice",
            "pair": ["Alice", "Bob"],
            "revert_kind": "partial_trajectory",
            "reversal_score": 0.78,
            "directly_reverts_previous_revision": True,
        },
        {
            "revision_index": 9,
            "revision_id": "109",
            "timestamp": "2020-01-01T00:09:00Z",
            "reverter": "Carol",
            "reverted_editor": "Bob",
            "pair": ["Bob", "Carol"],
            "revert_kind": "partial_trajectory",
            "reversal_score": 0.81,
            "directly_reverts_previous_revision": True,
        },
    ]

    sessions = detect_edit_war_sessions(events)

    assert len(sessions) == 1
    assert sessions[0]["participants"] == ["Alice", "Bob", "Carol"]
    assert sessions[0]["revert_kinds"] == ["partial_trajectory"]
    assert sessions[0]["event_count"] == 2


def test_one_direct_partial_counter_edit_is_not_an_edit_war():
    events = [{
        "revision_index": 8,
        "revision_id": "108",
        "timestamp": "2020-01-01T00:08:00Z",
        "reverter": "Bob",
        "reverted_editor": "Alice",
        "pair": ["Alice", "Bob"],
        "revert_kind": "partial_trajectory",
        "reversal_score": 0.78,
        "directly_reverts_previous_revision": True,
    }]

    sessions = detect_edit_war_sessions(events)

    assert sessions == []
    assert "edit_war_session_id" not in events[0]


def test_partial_revert_can_return_across_multiple_consecutive_2d_steps():
    history = [
        revision(0, "state-0", "Initial"),
        revision(1, "state-1", "Alice"),
        revision(2, "state-2", "Bob"),
        revision(3, "state-3", "Bob"),
    ]
    vectors = {
        "100": [0.0, 0.0],
        "101": [1.0, 0.0],
        "102": [0.7, 0.0],
        "103": [0.3, 0.0],
    }

    events = detect_partial_trajectory_reverts(history, vectors)

    assert len(events) == 1
    assert events[0]["revision_id"] == "103"
    assert events[0]["directly_reverts_previous_revision"] is True
    assert events[0]["trajectory_path_revision_ids"] == ["101", "102", "103"]
    assert events[0]["trajectory_return_revision_count"] == 2
    assert events[0]["trajectory_return_ratio"] == 0.3


def test_partial_revert_does_not_skip_an_intervening_non_return_step():
    history = [
        revision(0, "state-0", "Initial"),
        revision(1, "state-1", "Alice"),
        revision(2, "state-2", "Bob"),
        revision(3, "state-3", "Carol"),
    ]
    vectors = {
        "100": [0.0, 0.0],
        "101": [1.0, 0.0],
        "102": [1.0, 0.2],
        "103": [0.2, 0.0],
    }

    assert detect_partial_trajectory_reverts(history, vectors) == []


def test_partial_revert_can_follow_a_three_point_outgoing_phase():
    history = [
        revision(0, "state-0", "Initial"),
        revision(1, "state-1", "Alice"),
        revision(2, "state-2", "Alice"),
        revision(3, "state-3", "Bob"),
        revision(4, "state-4", "Bob"),
    ]
    vectors = {
        "100": [0.0, 0.0],
        "101": [0.5, 0.0],
        "102": [1.0, 0.0],
        "103": [0.7, 0.0],
        "104": [0.3, 0.0],
    }

    events = detect_partial_trajectory_reverts(history, vectors)

    assert len(events) == 1
    assert events[0]["trajectory_outgoing_path_revision_ids"] == [
        "100", "101", "102"
    ]
    assert events[0]["trajectory_path_revision_ids"] == [
        "102", "103", "104"
    ]
    assert events[0]["trajectory_full_path_revision_ids"] == [
        "100", "101", "102", "103", "104"
    ]
    assert events[0]["trajectory_outgoing_point_count"] == 3


def test_semantic_counter_revert_requires_direction_and_close_return_only():
    history = [
        revision(0, "state-0", "Initial"),
        revision(1, "state-1", "Alice"),
        revision(2, "state-2", "Bob"),
    ]
    history[1]["change_signature"] = {"units": [{
        "added_tokens": ["palestine", "observer", "status", "framework"],
        "removed_tokens": ["earlier", "wording"],
        "added_text": "Palestine observer status framework",
        "removed_text": "earlier wording",
    }]}
    history[2]["change_signature"] = {"units": [{
        "added_tokens": ["completely", "different", "phrasing"],
        "removed_tokens": ["new", "terms"],
        "added_text": "Completely different phrasing",
        "removed_text": "new terms",
    }]}
    vectors = {
        "100": [1.0, 0.0],
        "101": [0.8, 0.2],
        "102": [0.98, 0.02],
    }

    events = detect_semantic_counter_reverts(history, vectors)

    assert len(events) == 1
    assert events[0]["revision_id"] == "102"
    assert events[0]["revert_kind"] == "semantic_text"
    assert events[0]["directly_reverts_previous_revision"] is True
    assert events[0]["semantic_direction_cosine"] < -0.99
    assert events[0]["semantic_return_ratio"] <= 0.3
    assert events[0]["shared_topic_token_count"] == 0


def test_semantic_counter_revert_rejects_more_than_30_percent_rest_distance():
    history = [
        revision(0, "state-0", "Initial"),
        revision(1, "state-1", "Alice"),
        revision(2, "state-2", "Bob"),
    ]
    vectors = {
        "100": [0.0, 0.0],
        "101": [1.0, 0.0],
        "102": [0.6, 0.0],
    }

    assert detect_semantic_counter_reverts(history, vectors) == []


def test_semantic_counter_revert_rejects_cosine_above_minus_098():
    history = [
        revision(0, "state-0", "Initial"),
        revision(1, "state-1", "Alice"),
        revision(2, "state-2", "Bob"),
    ]
    vectors = {
        "100": [0.0, 0.0],
        "101": [1.0, 0.0],
        # Counter-direction cosine is -0.96 and the return ratio is about 0.283.
        "102": [0.04, 0.28],
    }

    assert detect_semantic_counter_reverts(history, vectors) == []


def test_change_signature_keeps_both_affected_sentences_for_an_insertion():
    signature = build_change_signature(
        "Before. Palestine had status. After.",
        "Before. Palestine had observer status. After.",
    )

    assert len(signature["units"]) == 1
    unit = signature["units"][0]
    assert unit["previous_sentence"] == "Palestine had status."
    assert unit["current_sentence"] == "Palestine had observer status."


def test_separated_single_counter_edit_does_not_form_a_session():
    events = [
        {
            "revision_index": index,
            "revision_id": str(index),
            "timestamp": timestamp,
            "reverter": reverter,
            "reverted_editor": reverted,
            "pair": ["Alice", "Bob"],
            "directly_reverts_previous_revision": direct,
            "involves_probable_bot": False,
        }
        for index, timestamp, reverter, reverted, direct in [
            (10, "2020-01-01T00:00:00Z", "Alice", "Bob", True),
            # The elapsed time is irrelevant when the revision indexes are
            # uninterrupted and every event directly reverts its predecessor.
            (11, "2020-02-01T00:00:00Z", "Bob", "Alice", True),
            (12, "2020-03-01T00:00:00Z", "Alice", "Bob", True),
            # A gap in the article history leaves a standalone revert.
            (14, "2020-03-01T00:01:00Z", "Bob", "Alice", True),
        ]
    ]
    sessions = detect_edit_war_sessions(events)
    assert len(sessions) == 1
    assert sessions[0]["start_revision_index"] == 10
    assert sessions[0]["end_revision_index"] == 12
    assert events[-1].get("edit_war_session_id") is None


def test_edit_war_ignores_non_direct_events_without_joining_adjacent_runs():
    events = [
        {
            "revision_index": index,
            "revision_id": str(index),
            "timestamp": f"2020-01-01T00:0{index}:00Z",
            "reverter": reverter,
            "reverted_editor": reverted,
            "pair": ["Alice", "Bob"],
            "directly_reverts_previous_revision": direct,
        }
        for index, reverter, reverted, direct in [
            (2, "Alice", "Bob", True),
            (3, "Bob", "Alice", False),
            (4, "Alice", "Bob", True),
        ]
    ]

    sessions = detect_edit_war_sessions(events)
    assert sessions == []
    assert events[1].get("edit_war_session_id") is None


def test_edit_war_rejects_events_without_direct_revert_evidence():
    events = [
        {
            "revision_index": index,
            "revision_id": str(index),
            "timestamp": f"2020-01-01T00:0{index}:00Z",
            "reverter": reverter,
            "reverted_editor": reverted,
            "pair": ["Alice", "Bob"],
        }
        for index, reverter, reverted in [
            (2, "Alice", "Bob"),
            (3, "Bob", "Alice"),
            (4, "Alice", "Bob"),
        ]
    ]

    assert detect_edit_war_sessions(events) == []


def test_edit_war_session_can_include_more_than_one_editor_pair():
    events = [
        {
            "revision_index": index,
            "revision_id": str(index),
            "timestamp": f"2020-01-01T00:0{index}:00Z",
            "reverter": reverter,
            "reverted_editor": reverted,
            "pair": sorted([reverter, reverted]),
            "directly_reverts_previous_revision": True,
        }
        for index, reverter, reverted in [
            (2, "Alice", "Bob"),
            (3, "Bob", "Alice"),
            (4, "Carol", "Bob"),
        ]
    ]

    sessions = detect_edit_war_sessions(events)

    assert len(sessions) == 1
    assert sessions[0]["participants"] == ["Alice", "Bob", "Carol"]
    assert sessions[0]["participant_count"] == 3
    assert sessions[0]["editor_pairs"] == [
        ["Alice", "Bob"],
        ["Bob", "Carol"],
    ]
    assert all(event.get("edit_war_session_id") for event in events)


def test_edit_war_links_consecutive_compound_trajectory_phases():
    events = [
        {
            "revision_index": 4,
            "revision_id": "104",
            "timestamp": "2020-01-01T00:04:00Z",
            "reverter": "Bob",
            "reverted_editor": "Alice",
            "pair": ["Alice", "Bob"],
            "directly_reverts_previous_revision": True,
            "trajectory_start_revision_index": 0,
            "trajectory_first_return_revision_index": 3,
            "trajectory_final_revision_index": 4,
        },
        {
            "revision_index": 7,
            "revision_id": "107",
            "timestamp": "2020-01-01T00:07:00Z",
            "reverter": "Carol",
            "reverted_editor": "Bob",
            "pair": ["Bob", "Carol"],
            "directly_reverts_previous_revision": True,
            "trajectory_start_revision_index": 4,
            "trajectory_first_return_revision_index": 5,
            "trajectory_final_revision_index": 7,
        },
    ]

    sessions = detect_edit_war_sessions(events)

    assert len(sessions) == 1
    assert sessions[0]["participants"] == ["Alice", "Bob", "Carol"]
    assert sessions[0]["start_revision_index"] == 0
    assert sessions[0]["end_revision_index"] == 7
