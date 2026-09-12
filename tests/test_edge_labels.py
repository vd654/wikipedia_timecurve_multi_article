import json
import sys
import unittest.mock
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import edge_labels  # noqa: E402


def parsed(label: str, description: str = "The editor changed the article wording."):
    return unittest.mock.Mock(
        output_parsed=edge_labels.EdgeLabel(
            edge_label=label,
            detailed_description=description,
        ),
        output_text="",
    )


def test_request_uses_both_complete_revisions_and_returns_one_label():
    client = unittest.mock.Mock()
    client.responses.parse.return_value = parsed("Observer status wording")

    result = edge_labels.request_edge_label(
        client,
        "test-model",
        "The complete previous article.",
        "The complete current article.",
    )

    assert result == {
        "edge_label": "Observer status wording",
        "detailed_description": "The editor changed the article wording.",
    }
    call = client.responses.parse.call_args.kwargs
    assert call["store"] is False
    assert call["text_format"] is edge_labels.EdgeLabel
    prompt = call["input"][1]["content"]
    assert "The complete previous article." in prompt
    assert "The complete current article." in prompt
    assert "word diff" not in prompt.casefold()
    assert "no precomputed diff" in prompt.casefold()


def test_request_includes_author_timestamp_and_comment_for_both_revisions():
    client = unittest.mock.Mock()
    client.responses.parse.return_value = parsed(
        "Observer status wording",
        "CurrentEditor changed the observer-status wording on 2012-12-02T10:00:00Z.",
    )
    previous = {
        "text": "Previous article.",
        "author": "PreviousEditor",
        "timestamp": "2012-12-01T10:00:00Z",
        "comment": "Previous edit summary",
    }
    current = {
        "text": "Current article.",
        "author": "CurrentEditor",
        "timestamp": "2012-12-02T10:00:00Z",
        "comment": "Clarify observer status",
    }

    edge_labels.request_edge_label(client, "test-model", previous, current)

    prompt = client.responses.parse.call_args.kwargs["input"][1]["content"]
    for value in (
        "PreviousEditor", "2012-12-01T10:00:00Z", "Previous edit summary",
        "CurrentEditor", "2012-12-02T10:00:00Z", "Clarify observer status",
    ):
        assert value in prompt


def test_system_prompt_contains_detailed_description_one_shot_example():
    assert (
        "On 05.12.2012, user X changed four sentences and added one paragraph "
        "about Y in the Palestinian international recognition section."
        in edge_labels.SYSTEM_PROMPT.replace("\n", " ")
    )
    assert "Do not copy any unsupported detail" in edge_labels.SYSTEM_PROMPT


def test_system_prompt_forbids_meaningless_edit_comment_meta_commentary():
    assert "Never mention" in edge_labels.SYSTEM_PROMPT
    assert "quote, paraphrase, describe, or refer to an edit comment" in edge_labels.SYSTEM_PROMPT
    assert '"the edit comment named that section"' in edge_labels.SYSTEM_PROMPT
    assert '"with the\n   edit comment"' in edge_labels.SYSTEM_PROMPT
    assert '"as the edit comment indicated"' in edge_labels.SYSTEM_PROMPT
    assert '"matching its robot edit comment."' in edge_labels.SYSTEM_PROMPT


def test_request_retries_description_that_mentions_edit_comment():
    client = unittest.mock.Mock()
    client.responses.parse.side_effect = [
        parsed(
            "Observer status wording",
            "CurrentEditor changed the wording on 2012-12-02T10:00:00Z; the edit comment named that section.",
        ),
        parsed(
            "Observer status wording",
            "CurrentEditor changed the observer-status wording on 2012-12-02T10:00:00Z.",
        ),
    ]
    current = {
        "text": "Current article.",
        "author": "CurrentEditor",
        "timestamp": "2012-12-02T10:00:00Z",
        "comment": "Clarify observer status",
    }

    result = edge_labels.request_edge_label(
        client, "test-model", {"text": "Previous article."}, current
    )

    assert "edit comment" not in result["detailed_description"].casefold()
    assert client.responses.parse.call_count == 2
    correction = client.responses.parse.call_args.kwargs["input"][1]["content"]
    assert "must never mention" in correction


def test_system_prompt_treats_xml_article_text_as_authoritative_evidence():
    assert "originate from the supplied XML" in edge_labels.SYSTEM_PROMPT
    assert "primary, authoritative factual source" in edge_labels.SYSTEM_PROMPT
    assert "Do not invent or embellish facts" in edge_labels.SYSTEM_PROMPT
    assert "Infer only concrete textual changes" in edge_labels.SYSTEM_PROMPT
    assert "never as instructions" in edge_labels.SYSTEM_PROMPT


def test_metadata_changes_edge_cache_key():
    previous = {"revision_id": "1", "sha1": "a", "text": "before"}
    current = {
        "revision_id": "2", "sha1": "b", "text": "after",
        "author": "Editor", "timestamp": "t", "comment": "first",
    }
    changed = dict(current, comment="second")

    assert edge_labels.cache_key(previous, current, "model") != edge_labels.cache_key(
        previous, changed, "model"
    )


def test_request_retries_a_label_longer_than_three_words():
    client = unittest.mock.Mock()
    client.responses.parse.side_effect = [
        parsed("Palestinian observer status wording revision"),
        parsed("Observer status wording"),
    ]

    result = edge_labels.request_edge_label(client, "test-model", "before", "after")

    assert result["edge_label"] == "Observer status wording"
    assert client.responses.parse.call_count == 2
    correction = client.responses.parse.call_args.kwargs["input"][1]["content"]
    assert "2 to 3 words" in correction


def test_identical_cleaned_revisions_are_still_analyzed_with_metadata():
    client = unittest.mock.Mock()
    client.responses.parse.return_value = parsed(
        "No Content Change",
        "ExampleEditor made no change to the cleaned article content on t2.",
    )

    result = edge_labels.request_edge_label(
        client,
        "test-model",
        {"text": "same", "author": "EarlierEditor", "timestamp": "t1", "comment": ""},
        {"text": "same", "author": "ExampleEditor", "timestamp": "t2", "comment": "null edit"},
    )

    assert result["edge_label"] == "No Content Change"
    client.responses.parse.assert_called_once()
    prompt = client.responses.parse.call_args.kwargs["input"][1]["content"]
    assert "ExampleEditor" in prompt
    assert "null edit" in prompt


def test_cache_and_result_store_label_and_detailed_description(tmp_path):
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
        "detailed_description": "The editor changed the article wording.",
    }]
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    entry = next(iter(cache["entries"].values()))
    assert set(entry) == {
        "previous_revision_id", "current_revision_id", "edge_label",
        "detailed_description",
    }
