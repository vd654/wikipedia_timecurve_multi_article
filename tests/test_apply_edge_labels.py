import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apply_edge_labels import (  # noqa: E402
    apply_edge_descriptions,
    apply_edge_labels,
    make_change_specific_edge_labels,
    normalise_unchanged_edges,
)


def test_neutral_edge_description_is_made_deterministically_consistent():
    hierarchy = {"edges": [{
        "node_id": "edge-1",
        "previous_revision_id": "1",
        "current_revision_id": "2",
        "label": "No content change",
        "detailed_description": "Editor removed a substantive sentence.",
    }]}
    metadata = {"revisions": [{
        "revision_id": "2",
        "author": "ExampleEditor",
        "timestamp": "2020-01-02T00:00:00Z",
    }]}

    assert normalise_unchanged_edges(hierarchy, metadata) == ["edge-1"]
    edge = hierarchy["edges"][0]
    assert edge["label"] == "No Content Change"
    assert edge["cleaned_text_unchanged"] is True
    assert edge["detailed_description"] == (
        "On 2020-01-02T00:00:00Z, ExampleEditor made a revision, and the "
        "supplied cleaned article content did not change."
    )


def test_short_edge_label_adds_change_direction_from_description():
    hierarchy = {"edges": [{
        "node_id": "edge-1",
        "label": "UN Observer Status",
        "detailed_description": "ExampleEditor removed the vote tally.",
    }]}

    assert make_change_specific_edge_labels(hierarchy) == ["edge-1"]
    assert hierarchy["edges"][0]["label"] == "UN Observer Status Removed"
    assert hierarchy["edges"][0]["topic_label"] == "UN Observer Status"


def test_change_specific_label_generation_is_idempotent():
    hierarchy = {"edges": [{
        "node_id": "edge-1",
        "label": "Vote Implications Added",
        "detailed_description": "ExampleEditor added treaty implications.",
    }]}

    assert make_change_specific_edge_labels(hierarchy) == []
    assert hierarchy["edges"][0]["label"] == "Vote Implications Added"


def test_full_description_overlay_preserves_existing_short_labels():
    hierarchy = {
        "edges": [
            {
                "node_id": "edge-1",
                "previous_revision_id": "1",
                "current_revision_id": "2",
                "label": "Existing short label",
                "short_label": "Existing short label",
            },
            {
                "node_id": "edge-2",
                "previous_revision_id": "2",
                "current_revision_id": "3",
                "label": "Another short label",
                "short_label": "Another short label",
            },
        ]
    }
    descriptions = {
        "revision_pairs": [
            {
                "previous_revision_id": "1",
                "current_revision_id": "2",
                "edge_label": "Ignored replacement label",
                "detailed_description": "Editor A changed the first topic in detail.",
            },
            {
                "previous_revision_id": "2",
                "current_revision_id": "3",
                "edge_label": "Another ignored replacement",
                "detailed_description": "Editor B changed the second topic in detail.",
            },
        ]
    }

    assert apply_edge_descriptions(hierarchy, descriptions) == ["edge-1", "edge-2"]
    assert hierarchy["edges"][0]["label"] == "Existing short label"
    assert hierarchy["edges"][0]["short_label"] == "Existing short label"
    assert hierarchy["edges"][0]["detailed_description"].startswith("Editor A")
    assert hierarchy["edges"][1]["label"] == "Another short label"


def test_partial_label_overlay_wins_after_full_description_overlay():
    hierarchy = {
        "edges": [
            {
                "node_id": "edge-1",
                "previous_revision_id": "1",
                "current_revision_id": "2",
                "label": "Existing first",
            },
            {
                "node_id": "edge-2",
                "previous_revision_id": "2",
                "current_revision_id": "3",
                "label": "Existing second",
            },
        ]
    }
    descriptions = {
        "revision_pairs": [
            {
                "previous_revision_id": "1",
                "current_revision_id": "2",
                "detailed_description": "Full description one.",
            },
            {
                "previous_revision_id": "2",
                "current_revision_id": "3",
                "detailed_description": "Full description two.",
            },
        ]
    }
    preview = {
        "revision_pairs": [{
            "previous_revision_id": "1",
            "current_revision_id": "2",
            "edge_label": "Chosen preview label",
            "detailed_description": "Chosen preview description.",
        }]
    }

    apply_edge_descriptions(hierarchy, descriptions)
    apply_edge_labels(hierarchy, preview)

    assert hierarchy["edges"][0]["label"] == "Chosen preview label"
    assert hierarchy["edges"][0]["detailed_description"] == (
        "Chosen preview description."
    )
    assert hierarchy["edges"][1]["label"] == "Existing second"
    assert hierarchy["edges"][1]["detailed_description"] == (
        "Full description two."
    )


def test_partial_overlay_updates_only_matching_edges():
    hierarchy = {
        "edges": [
            {
                "node_id": "edge-1",
                "previous_revision_id": "1",
                "current_revision_id": "2",
                "label": "Old first",
            },
            {
                "node_id": "edge-2",
                "previous_revision_id": "2",
                "current_revision_id": "3",
                "label": "Old second",
            },
        ]
    }
    preview = {
        "revision_pairs": [{
            "previous_revision_id": "1",
            "current_revision_id": "2",
            "edge_label": "New first",
            "detailed_description": "Editor A changed the first topic.",
        }]
    }

    assert apply_edge_labels(hierarchy, preview) == ["edge-1"]
    assert hierarchy["edges"][0]["label"] == "New first"
    assert hierarchy["edges"][0]["detailed_description"].startswith("Editor A")
    assert hierarchy["edges"][1]["label"] == "Old second"
