import json
import sys
import unittest.mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import edge_labels  # noqa: E402


def parsed(label: str):
    return unittest.mock.Mock(
        output_parsed=edge_labels.EdgeLabel(edge_label=label), output_text=""
    )


def test_request_uses_both_complete_revisions_and_returns_one_label():
    client = unittest.mock.Mock()
    client.responses.parse.return_value = parsed("Observer status wording")

    label = edge_labels.request_edge_label(
        client,
        "test-model",
        "The complete previous article.",
        "The complete current article.",
    )

    assert label == "Observer status wording"
    call = client.responses.parse.call_args.kwargs
    assert call["store"] is False
    assert call["text_format"] is edge_labels.EdgeLabel
    prompt = call["input"][1]["content"]
    assert "The complete previous article." in prompt
    assert "The complete current article." in prompt
    assert "word diff" not in prompt.casefold()


def test_request_retries_a_label_longer_than_four_words():
    client = unittest.mock.Mock()
    client.responses.parse.side_effect = [
        parsed("Palestinian observer status wording revision"),
        parsed("Observer status wording"),
    ]

    label = edge_labels.request_edge_label(client, "test-model", "before", "after")

    assert label == "Observer status wording"
    assert client.responses.parse.call_count == 2
    correction = client.responses.parse.call_args.kwargs["input"][1]["content"]
    assert "one to 4 words" in correction


def test_identical_cleaned_revisions_need_no_api_call():
    client = unittest.mock.Mock()

    label = edge_labels.request_edge_label(client, "test-model", "same", "same")

    assert label == "No content change"
    client.responses.parse.assert_not_called()


def test_cache_and_result_store_only_edge_mapping(tmp_path):
    client = unittest.mock.Mock()
    client.responses.parse.return_value = parsed("Spelling standardization")
    revisions = [
        {"revision_id": "1", "sha1": "a", "text": "favourably"},
        {"revision_id": "2", "sha1": "b", "text": "favorably"},
    ]
    cache_path = tmp_path / "cache.json"

    result = edge_labels.generate_edge_labels(
        revisions, client, "test-model", cache_path
    )

    assert result == [{
        "previous_revision_id": "1",
        "current_revision_id": "2",
        "edge_label": "Spelling standardization",
    }]
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    entry = next(iter(cache["entries"].values()))
    assert set(entry) == {
        "previous_revision_id", "current_revision_id", "edge_label"
    }
