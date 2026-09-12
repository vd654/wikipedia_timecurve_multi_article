import sys
import unittest.mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import revision_metadata  # noqa: E402


def test_parse_revision_rows_keeps_requested_label_context():
    payload = {
        "query": {
            "pages": [{
                "revisions": [{
                    "revid": 123,
                    "user": "ExampleEditor",
                    "timestamp": "2012-12-02T10:00:00Z",
                    "comment": "Clarify observer status",
                }]
            }]
        }
    }

    assert revision_metadata.parse_revision_rows(payload)["123"] == {
        "revision_id": "123",
        "author": "ExampleEditor",
        "timestamp": "2012-12-02T10:00:00Z",
        "comment": "Clarify observer status",
        "user_hidden": False,
        "comment_hidden": False,
    }


def test_download_metadata_preserves_projection_order(tmp_path):
    response = unittest.mock.Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "query": {"pages": [{"revisions": [
            {"revid": 2, "user": "B", "timestamp": "t2", "comment": "c2"},
            {"revid": 1, "user": "A", "timestamp": "t1", "comment": "c1"},
        ]}]}
    }
    session = unittest.mock.Mock()
    session.headers = {}
    session.get.return_value = response

    result = revision_metadata.download_metadata(
        ["1", "2"], tmp_path / "metadata.json", session=session
    )

    assert [row["revision_id"] for row in result["revisions"]] == ["1", "2"]
    assert result["complete"] is True
