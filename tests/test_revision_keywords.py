import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import revision_keywords


def sample_difference_lod(edge_label="UN observer status"):
    labels = [
        "International affairs",
        "Diplomatic status",
        "UN observer status",
        "Observer wording change",
        edge_label,
    ]
    return [
        revision_keywords.DifferenceLODLabel(
            level=level,
            label=label,
            description=f"Level {level} description for {label}.",
        )
        for level, label in enumerate(labels, start=1)
    ]


class BuildDiffTests(unittest.TestCase):
    def test_complete_diff_has_no_truncation_marker(self):
        result = revision_keywords.build_diff(
            "alpha beta", "alpha gamma", max_chars=500
        )

        self.assertFalse(result.diff_truncated)
        self.assertNotIn("DIFF TRUNCATED", result.text)
        self.assertEqual(result.total_change_units, 1)
        self.assertEqual(result.included_change_units, 1)
        self.assertEqual(result.omitted_change_units, 0)
        self.assertEqual(result.shortened_change_units, 0)
        self.assertEqual(result.diff_original_chars, len(result.text))
        self.assertEqual(result.diff_included_chars, len(result.text))
        self.assertEqual(result.total_diff_opcodes, 1)
        self.assertEqual(result.included_diff_opcodes, 1)
        self.assertEqual(result.omitted_diff_opcodes, 0)
        self.assertEqual(result.previous_word_count, 2)
        self.assertEqual(result.current_word_count, 2)

    def test_truncation_stops_at_first_unit_that_does_not_fit(self):
        previous_words = [f"w{index}" for index in range(100)]
        current_words = previous_words.copy()
        current_words[10] = "FIRST"
        current_words[40] = "X" * 500
        current_words[70] = "LATE"

        result = revision_keywords.build_diff(
            " ".join(previous_words),
            " ".join(current_words),
            max_chars=600,
        )

        self.assertTrue(result.diff_truncated)
        self.assertIn("REMOVED 1:", result.text)
        self.assertIn("ADDED 1:", result.text)
        self.assertNotIn("REMOVED 2:", result.text)
        self.assertNotIn("ADDED 2:", result.text)
        self.assertNotIn("REMOVED 3:", result.text)
        self.assertNotIn("ADDED 3:", result.text)
        self.assertNotIn("LATE", result.text)
        self.assertIn("2 of 3 change units omitted", result.text)
        self.assertEqual(result.included_change_units, 1)
        self.assertEqual(result.omitted_change_units, 2)
        self.assertEqual(result.included_diff_opcodes, 1)
        self.assertEqual(result.omitted_diff_opcodes, 2)
        self.assertLess(result.last_included_current_position_fraction, 1.0)
        self.assertLessEqual(len(result.text), 600)

    def test_oversized_first_unit_is_shortened_at_word_boundaries(self):
        unchanged_prefix = [f"left{index}" for index in range(20)]
        unchanged_suffix = [f"right{index}" for index in range(20)]
        added_words = [f"change{index}" for index in range(100)]
        previous = " ".join(unchanged_prefix + unchanged_suffix)
        current = " ".join(unchanged_prefix + added_words + unchanged_suffix)

        result = revision_keywords.build_diff(
            previous, current, max_chars=350
        )

        self.assertTrue(result.diff_truncated)
        self.assertEqual(result.included_change_units, 1)
        self.assertEqual(result.omitted_change_units, 0)
        self.assertEqual(result.shortened_change_units, 1)
        self.assertIn("ADDED 1:", result.text)
        self.assertIn("change0", result.text)
        self.assertIn("change99", result.text)
        self.assertIn("changed words omitted", result.text)
        self.assertIn("1 change unit shortened", result.text)
        self.assertLessEqual(len(result.text), 350)

    def test_marker_only_is_used_when_no_grounded_passage_can_fit(self):
        result = revision_keywords.build_diff(
            "alpha", "a-very-long-replacement-word", max_chars=40
        )

        self.assertTrue(result.diff_truncated)
        self.assertEqual(result.text, "[DIFF TRUNCATED]")
        self.assertEqual(result.included_change_units, 0)
        self.assertEqual(result.omitted_change_units, 1)
        self.assertLessEqual(len(result.text), 40)

    def test_limit_must_leave_room_for_a_marker(self):
        with self.assertRaisesRegex(ValueError, "Kürzungsmarker"):
            revision_keywords.build_diff("alpha", "beta", max_chars=1)


class AffectedSentenceContextTests(unittest.TestCase):
    def test_replacement_includes_complete_previous_and_current_sentences(self):
        previous = (
            "Unrelated introduction. The General Assembly formally said that "
            "Palestine was not recognized as a member state by this organization "
            "in 2011. Unrelated ending."
        )
        current = (
            "Unrelated introduction. The General Assembly formally said that "
            "Palestine was recognized as a non-member observer state by this "
            "organization in 2012. Unrelated ending."
        )

        result = revision_keywords.build_diff(
            previous, current, max_chars=10_000, context=3
        )

        self.assertIn(
            "PREVIOUS AFFECTED SENTENCE 1: The General Assembly formally said",
            result.text,
        )
        self.assertIn(
            "CURRENT AFFECTED SENTENCE 1: The General Assembly formally said",
            result.text,
        )
        self.assertNotIn(
            "PREVIOUS AFFECTED SENTENCE 1: Unrelated introduction", result.text
        )
        self.assertIn("[[ not ]]", result.text)
        self.assertIn("[[ non-member observer ]]", result.text)

    def test_pure_insertion_includes_previous_counterpart_sentence(self):
        result = revision_keywords.build_diff(
            "Before. Palestine had status. After.",
            "Before. Palestine had observer status. After.",
            max_chars=10_000,
            context=2,
        )

        self.assertIn(
            "PREVIOUS AFFECTED SENTENCE 1: Palestine had status.", result.text
        )
        self.assertIn("ADDED 1: Palestine had [[ observer ]] status.", result.text)

    def test_pure_deletion_includes_current_counterpart_sentence(self):
        result = revision_keywords.build_diff(
            "Before. Palestine had temporary observer status. After.",
            "Before. Palestine had observer status. After.",
            max_chars=10_000,
            context=2,
        )

        self.assertIn("REMOVED 1: Palestine had [[ temporary ]]", result.text)
        self.assertIn(
            "CURRENT AFFECTED SENTENCE 1: Palestine had observer status.",
            result.text,
        )

    def test_initialism_does_not_split_affected_sentence(self):
        words = "The U.N. granted observer status. Another sentence.".split()

        start, end = revision_keywords.affected_sentence_window(words, 3, 4)

        self.assertEqual(" ".join(words[start:end]), "The U.N. granted observer status.")


class AnalyzeTests(unittest.TestCase):
    def test_analyze_persists_diff_metadata(self):
        revisions = iter([
            {"revision_id": "1", "timestamp": "t1", "sha1": "sha-a", "text": "alpha beta"},
            {"revision_id": "2", "timestamp": "t2", "sha1": "sha-b", "text": "alpha gamma"},
        ])
        keywords = {
            "edge_label": "gamma replacement",
            "change_type": "replacement",
            "evidence": ["gamma"],
            "confidence": 0.9,
            "added_keywords": ["gamma"],
            "removed_keywords": ["beta"],
            "focus_keywords": ["one", "two", "three", "four", "five"],
        }

        with (
            patch.object(revision_keywords, "iter_revisions", return_value=revisions),
            patch.object(
                revision_keywords, "request_keywords", return_value=keywords
            ),
        ):
            results = revision_keywords.analyze(
                Path("unused.xml"),
                ["1", "2"],
                client=unittest.mock.Mock(),
                model="test-model",
                limit=0,
                max_diff_chars=500,
            )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIn("diff_truncated", result)
        self.assertIn("diff_original_chars", result)
        self.assertIn("diff_included_chars", result)
        self.assertIn("total_diff_opcodes", result)
        self.assertIn("included_diff_opcodes", result)
        self.assertIn("omitted_diff_opcodes", result)
        self.assertIn("last_included_current_position_fraction", result)
        self.assertIn("total_change_units", result)
        self.assertIn("included_change_units", result)
        self.assertIn("omitted_change_units", result)
        self.assertIn("shortened_change_units", result)
        self.assertEqual(result["edge_label"], "gamma replacement")
        self.assertFalse(result["is_edit_war"])
        self.assertEqual(result["previous_sha1"], "sha-a")
        self.assertEqual(result["current_sha1"], "sha-b")
        self.assertRegex(result["diff_label_cache_key"], r"^[0-9a-f]{64}$")

    def test_matching_sha1_pair_and_prompt_configuration_reuses_cache(self):
        revision_rows = [
            {"revision_id": "1", "timestamp": "t1", "sha1": "sha-a", "text": "alpha beta"},
            {"revision_id": "2", "timestamp": "t2", "sha1": "sha-b", "text": "alpha gamma"},
        ]
        analysis = {
            "edge_label": "gamma replacement",
            "change_type": "replacement",
            "evidence": ["gamma"],
            "confidence": 0.9,
            "added_keywords": ["gamma"],
            "removed_keywords": ["beta"],
            "focus_keywords": ["gamma", "beta", "replacement", "wording", "term"],
        }
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "diff-label-cache.json"
            with (
                patch.object(revision_keywords, "iter_revisions", return_value=iter(revision_rows)),
                patch.object(revision_keywords, "request_keywords", return_value=analysis) as request,
            ):
                first = revision_keywords.analyze(
                    Path("unused.xml"), ["1", "2"], unittest.mock.Mock(),
                    "test-model", 0, 500, cache_path=cache_path,
                )
            with (
                patch.object(revision_keywords, "iter_revisions", return_value=iter(revision_rows)),
                patch.object(revision_keywords, "request_keywords") as second_request,
            ):
                second = revision_keywords.analyze(
                    Path("unused.xml"), ["1", "2"], None,
                    "test-model", 0, 500, cache_path=cache_path,
                )

        request.assert_called_once()
        second_request.assert_not_called()
        self.assertEqual(first, second)

    def test_confirmed_session_keeps_shared_label_but_analyzes_full_pair_context(self):
        revision_rows = [
            {"revision_id": "1", "timestamp": "t1", "sha1": "sha-a", "text": "alpha beta"},
            {"revision_id": "2", "timestamp": "t2", "sha1": "sha-b", "text": "alpha gamma"},
        ]
        description = {
            "conflict_label": "Observer-state wording",
            "conflict_description": "The wording of observer status was restored.",
            "evidence": ["observer status"],
            "confidence": 0.91,
        }
        contexts = [{
            "is_edit_war": True,
            "confirmed_event_count": 1,
            "editor_pairs": [["Alice", "Bob"]],
            "session_ids": ["edit-war-1"],
        }]
        pair_analysis = {
            "edge_label": "Temporary pair label",
            "change_type": "replacement",
            "evidence": ["gamma"],
            "confidence": 0.8,
            "added_keywords": ["gamma"],
            "removed_keywords": ["beta"],
            "focus_keywords": ["gamma", "beta", "status", "wording", "claim"],
        }
        with (
            patch.object(revision_keywords, "iter_revisions", return_value=iter(revision_rows)),
            patch.object(
                revision_keywords,
                "request_keywords",
                return_value=pair_analysis,
            ) as edge_request,
        ):
            results = revision_keywords.analyze(
                Path("unused.xml"), ["1", "2"], unittest.mock.Mock(),
                "test-model", 0, 500,
                edit_war_contexts=contexts,
                edit_war_descriptions={"edit-war-1": description},
            )

        edge_request.assert_called_once()
        self.assertEqual(
            edge_request.call_args.kwargs["previous_text"], "alpha beta"
        )
        self.assertEqual(
            edge_request.call_args.kwargs["current_text"], "alpha gamma"
        )
        self.assertEqual(results[0]["edge_label"], "Observer-state wording")
        self.assertEqual(results[0]["added_keywords"], ["gamma"])
        self.assertEqual(
            results[0]["conflict_description"],
            "The wording of observer status was restored.",
        )

    def test_session_aggregation_uses_outputs_of_full_revision_comparisons(self):
        revert_analysis = {
            "edit_war_sessions": [{"session_id": "edit-war-1"}]
        }
        results = [{
            "previous_revision_id": "1",
            "current_revision_id": "2",
            "edge_label": "Observer status",
            "change_type": "replacement",
            "evidence": ["observer status"],
            "added_keywords": ["non-member state"],
            "removed_keywords": ["observer entity"],
            "focus_keywords": ["observer status"],
            "confidence": 0.8,
            "edit_war_context": {"session_ids": ["edit-war-1"]},
        }]

        revision_keywords.attach_full_context_edge_analyses(
            revert_analysis, results
        )
        analyses = revert_analysis["edit_war_sessions"][0][
            "full_context_edge_analyses"
        ]
        self.assertEqual(analyses[0]["edge_label"], "Observer status")
        self.assertNotIn("diff", analyses[0])
        revision_keywords.apply_edit_war_descriptions(results, {
            "edit-war-1": {
                "conflict_label": "Observer-state wording",
                "conflict_description": "Observer wording was repeatedly restored.",
                "evidence": ["observer status"],
                "confidence": 0.91,
            }
        })
        self.assertEqual(results[0]["edge_label"], "Observer-state wording")


class PromptTests(unittest.TestCase):
    def test_prompt_contains_no_separate_diff_or_sentence_window(self):
        prompt = revision_keywords.build_user_prompt(
            "Previous complete article.", "Current complete article."
        )

        self.assertNotIn("[[", prompt)
        self.assertNotIn("AFFECTED SENTENCE", prompt)
        self.assertNotIn("GROUNDED REVISION DIFF", prompt)
        self.assertNotIn("DIFF TRUNCATED", prompt)

    def test_prompt_requires_one_atomic_edge_label_and_grounded_evidence(self):
        prompt = revision_keywords.build_user_prompt(
            "Previous article.", "Current article with observer state."
        )

        self.assertIn("exactly one edge_label", prompt)
        self.assertIn("1 to 4 words", prompt)
        self.assertIn("copied verbatim from either\ncomplete revision", prompt)
        self.assertIn("exactly 5 entries", prompt)
        self.assertIn("extremely abstract to extremely detailed", prompt)
        self.assertIn("Level 5 label must be exactly identical to edge_label", prompt)

    def test_prompt_contains_both_complete_cleaned_revisions(self):
        prompt = revision_keywords.build_user_prompt(
            "Complete previous article with old status.",
            "Complete current article with new status.",
        )

        self.assertIn("<previous_revision>", prompt)
        self.assertIn("Complete previous article with old status.", prompt)
        self.assertIn("<current_revision>", prompt)
        self.assertIn("Complete current article with new status.", prompt)
        self.assertIn("Compare the full texts yourself", prompt)

    def test_prompt_receives_definition_and_authoritative_edit_war_context(self):
        context = {
            "is_edit_war": True,
            "confirmed_event_count": 3,
            "editor_pairs": [["Alice", "Bob"]],
            "session_ids": ["edit-war-1"],
        }

        prompt = revision_keywords.build_user_prompt(
            "Previous complete article.",
            "Current complete article with observer state.",
            edit_war_context=context,
        )

        self.assertIn(revision_keywords.EDIT_WAR_DEFINITION, prompt)
        self.assertIn('"is_edit_war": true', prompt)
        self.assertIn("comes from the revision history", prompt)
        self.assertIn("only Edit-War-related task", prompt)
        self.assertIn("describe the concrete\nsubject being contested", prompt)
        self.assertIn("do not output an Edit-War\nclassification", prompt)
        self.assertNotIn("edit_comments", prompt)

    def test_edit_war_context_changes_pair_cache_key(self):
        previous = {"revision_id": "1", "sha1": "sha-a", "text": "a"}
        current = {"revision_id": "2", "sha1": "sha-b", "text": "b"}
        normal = revision_keywords.diff_label_cache_key(
            previous, current, "config", {"is_edit_war": False}
        )
        conflict = revision_keywords.diff_label_cache_key(
            previous, current, "config", {"is_edit_war": True}
        )

        self.assertNotEqual(normal, conflict)

    def test_session_prompt_uses_full_context_edge_analyses_without_diff(self):
        session = {
            "pair": ["Alice", "Bob"],
            "event_count": 3,
            "revert_kinds": ["exact_sha1", "partial_text"],
            "full_context_edge_analyses": [{
                "edge_label": "Observer status",
                "evidence": ["observer status", "non-member state"],
            }],
        }

        prompt = revision_keywords.build_edit_war_description_prompt(session)

        self.assertIn('"edge_label": "Observer status"', prompt)
        self.assertIn('"observer status"', prompt)
        self.assertIn("No separate word diff, sentence window", prompt)
        self.assertNotIn("[[", prompt)
        self.assertNotIn("AFFECTED SENTENCE", prompt)
        self.assertIn("must not decide whether it is an Edit War", prompt)
        self.assertIn("Use no outside knowledge", prompt)

    def test_session_description_is_one_structured_description_not_a_classifier(self):
        client = unittest.mock.Mock()
        client.responses.parse.return_value = unittest.mock.Mock(
            output_parsed=revision_keywords.EditWarSessionDescription(
                conflict_label="Observer-state wording",
                conflict_description=(
                    "The participants repeatedly changed the wording of observer status."
                ),
                evidence=["observer status", "non-member state"],
                confidence=0.92,
            ),
            output_text="",
        )
        session = {
            "pair": ["Alice", "Bob"],
            "event_count": 3,
            "revert_kinds": ["partial_text"],
            "full_context_edge_analyses": [{
                "edge_label": "Observer status",
                "evidence": ["observer status", "non-member state"],
            }],
        }

        result = revision_keywords.request_edit_war_description(
            client, "test-model", session
        )

        self.assertEqual(result["conflict_label"], "Observer-state wording")
        self.assertNotIn("is_edit_war", result)
        self.assertEqual(client.responses.parse.call_count, 1)

    def test_marker_only_diff_does_not_call_the_model(self):
        client = unittest.mock.Mock()

        result = revision_keywords.request_keywords(
            client, "test-model", "[DIFF TRUNCATED]"
        )

        self.assertEqual(result, {
            "edge_label": "No cleaned-text change",
            "change_type": "no_cleaned_text_change",
            "evidence": [],
            "confidence": 1.0,
            "added_keywords": [],
            "removed_keywords": [],
            "focus_keywords": [],
            "difference_lod": [],
        })
        client.responses.parse.assert_not_called()

    def test_structured_diff_label_is_returned_without_splitting_phrase(self):
        client = unittest.mock.Mock()
        client.responses.parse.return_value = unittest.mock.Mock(
            output_parsed=revision_keywords.RevisionKeywords(
                edge_label="UN observer status",
                change_type="addition",
                evidence=["observer status"],
                confidence=0.94,
                added_keywords=["observer status"],
                removed_keywords=[],
                focus_keywords=["observer status", "UN", "vote", "resolution", "statehood"],
                difference_lod=sample_difference_lod(),
            ),
            output_text="",
        )

        result = revision_keywords.request_keywords(
            client,
            "test-model",
            "ADDED 1: [[ observer status ]]",
            previous_text="Palestine held status.",
            current_text="Palestine received observer status.",
        )

        self.assertEqual(result["edge_label"], "UN observer status")
        self.assertEqual(result["evidence"], ["observer status"])
        self.assertEqual(client.responses.parse.call_count, 1)
        model_input = client.responses.parse.call_args.kwargs["input"]
        user_prompt = model_input[1]["content"]
        self.assertIn("Palestine held status.", user_prompt)
        self.assertIn("Palestine received observer status.", user_prompt)
        self.assertNotIn("ADDED 1", user_prompt)
        self.assertNotIn("[[", user_prompt)
        self.assertNotIn("AFFECTED SENTENCE", user_prompt)

    def test_ungrounded_evidence_uses_grounded_diff_fallback_without_retry(self):
        client = unittest.mock.Mock()
        invalid = revision_keywords.RevisionKeywords(
            edge_label="UN observer status",
            change_type="addition",
            evidence=["outside knowledge"],
            confidence=0.5,
            added_keywords=["observer status"],
            removed_keywords=[],
            focus_keywords=["observer status", "UN", "vote", "resolution", "statehood"],
            difference_lod=sample_difference_lod(),
        )
        client.responses.parse.return_value = unittest.mock.Mock(
            output_parsed=invalid, output_text=""
        )

        result = revision_keywords.request_keywords(
            client,
            "test-model",
            "ADDED 1: [[ observer status ]]",
            previous_text="Palestine held status.",
            current_text="Palestine received observer status.",
        )

        self.assertEqual(result["evidence"], ["observer status"])
        self.assertEqual(client.responses.parse.call_count, 1)

    def test_label_focus_changes_cache_configuration(self):
        first = revision_keywords.diff_label_configuration_sha256(
            "test-model", 500, "git", "histogram", 12, "UN status"
        )
        second = revision_keywords.diff_label_configuration_sha256(
            "test-model", 500, "git", "histogram", 12, "currency"
        )

        self.assertNotEqual(first, second)

    def test_provider_changes_cache_configuration(self):
        alternate = revision_keywords.diff_label_configuration_sha256(
            "same-model", 500, "git", "histogram", 12, "UN status", "alternate"
        )
        openai = revision_keywords.diff_label_configuration_sha256(
            "same-model", 500, "git", "histogram", 12, "UN status", "openai"
        )

        self.assertNotEqual(alternate, openai)


class OpcodeGroupingTests(unittest.TestCase):
    def test_overlapping_context_windows_are_grouped(self):
        opcodes = [
            ("replace", 2, 3, 2, 3),
            ("replace", 10, 11, 10, 11),
        ]

        groups = revision_keywords.group_nearby_opcodes(opcodes, context=5)

        self.assertEqual(groups, [opcodes])

    def test_non_overlapping_context_windows_stay_separate(self):
        opcodes = [
            ("replace", 2, 3, 2, 3),
            ("replace", 10, 11, 10, 11),
        ]

        groups = revision_keywords.group_nearby_opcodes(opcodes, context=3)

        self.assertEqual(groups, [[opcodes[0]], [opcodes[1]]])

    def test_validated_gap_lengths_must_match(self):
        opcodes = [
            ("replace", 2, 3, 2, 3),
            ("replace", 10, 11, 9, 10),
        ]

        with self.assertRaisesRegex(RuntimeError, "unterschiedliche"):
            revision_keywords.group_nearby_opcodes(opcodes, context=3)

    def test_grouped_rendering_preserves_unchanged_words_outside_brackets(self):
        previous = [f"w{index}" for index in range(20)]
        current = previous.copy()
        current[2] = "new2"
        current[10] = "new10"

        result = revision_keywords.build_diff(
            " ".join(previous),
            " ".join(current),
            max_chars=5_000,
            context=5,
        )

        self.assertEqual(result.total_diff_opcodes, 2)
        self.assertEqual(result.total_change_units, 1)
        self.assertEqual(result.text.count("REMOVED "), 1)
        self.assertEqual(result.text.count("ADDED "), 1)
        self.assertIn("[[ w2 ]] w3 w4 w5 w6 w7 w8 w9 [[ w10 ]]", result.text)
        self.assertNotIn("[[ w2 w3", result.text)

    def test_corrected_difflib_and_legacy_modes_are_explicit(self):
        original_matcher = revision_keywords.difflib.SequenceMatcher
        autojunk_values = []

        def recording_matcher(*args, **kwargs):
            autojunk_values.append(kwargs["autojunk"])
            return original_matcher(*args, **kwargs)

        with patch.object(
            revision_keywords.difflib,
            "SequenceMatcher",
            side_effect=recording_matcher,
        ):
            revision_keywords.diff_opcodes(["a"], ["b"], "difflib", "histogram")
            revision_keywords.diff_opcodes(
                ["a"], ["b"], "difflib-legacy", "histogram"
            )

        self.assertEqual(autojunk_values, [False, True])


class SnapshotTests(unittest.TestCase):
    def test_snapshot_id_and_source_hash_are_stable_shapes(self):
        self.assertEqual(revision_keywords.PIPELINE_SNAPSHOT_ID, "2026-08-21-v012")
        self.assertRegex(revision_keywords.source_tree_sha256(), r"^[0-9a-f]{64}$")

    def test_default_diff_limit_covers_measured_corpus_maximum(self):
        self.assertEqual(revision_keywords.DEFAULT_MAX_DIFF_CHARS, 100_000)


if __name__ == "__main__":
    unittest.main()
