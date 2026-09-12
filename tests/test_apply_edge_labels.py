import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apply_edge_labels import apply_edge_labels  # noqa: E402


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
