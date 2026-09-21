import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refine_edge_labels_from_descriptions import (  # noqa: E402
    canonical_label_case,
    valid_specific_label,
)


def test_specific_edge_labels_require_object_and_action():
    assert valid_specific_label("Capital Designation Removed")
    assert valid_specific_label("Vote Tally Corrected")
    assert valid_specific_label("Recognition Wording Lowercased")


def test_broad_topic_labels_are_rejected():
    assert not valid_specific_label("State of Palestine")
    assert not valid_specific_label("Palestine Recognition")
    assert not valid_specific_label("Palestinian Statehood")
    assert not valid_specific_label("UN Membership Criteria")
    assert not valid_specific_label("Palestine Geography")


def test_label_case_is_consistent_without_damaging_acronyms():
    assert canonical_label_case("Recognition qualifier removed") == (
        "Recognition Qualifier Removed"
    )
    assert canonical_label_case("PLO succession restored") == (
        "PLO Succession Restored"
    )
    assert canonical_label_case("Fatah–Hamas split added") == (
        "Fatah–Hamas Split Added"
    )
