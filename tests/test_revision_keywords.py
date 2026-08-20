import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import revision_keywords


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

    def test_confirmed_session_description_is_shared_without_per_edge_llm_call(self):
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
        with (
            patch.object(revision_keywords, "iter_revisions", return_value=iter(revision_rows)),
            patch.object(revision_keywords, "request_keywords") as edge_request,
        ):
            results = revision_keywords.analyze(
                Path("unused.xml"), ["1", "2"], unittest.mock.Mock(),
                "test-model", 0, 500,
                edit_war_contexts=contexts,
                edit_war_descriptions={"edit-war-1": description},
            )

        edge_request.assert_not_called()
        self.assertEqual(results[0]["edge_label"], "Observer-state wording")
        self.assertEqual(
            results[0]["conflict_description"],
            "The wording of observer status was restored.",
        )


class PromptTests(unittest.TestCase):
    def test_prompt_warns_that_visible_diff_may_be_incomplete(self):
        prompt = revision_keywords.build_user_prompt(
            "[DIFF TRUNCATED]", diff_truncated=True
        )

        self.assertIn("is truncated", prompt)
        self.assertIn("Use only the visible passages", prompt)
        self.assertIn("never\nassume that they represent every change", prompt)

    def test_complete_prompt_has_no_truncation_warning(self):
        prompt = revision_keywords.build_user_prompt("ADDED 1: [[ fact ]]")

        self.assertNotIn("truncated", prompt)
        self.assertNotIn("visible passages", prompt)

    def test_prompt_requires_one_atomic_edge_label_and_grounded_evidence(self):
        prompt = revision_keywords.build_user_prompt("ADDED 1: [[ observer state ]]")

        self.assertIn("exactly one edge_label", prompt)
        self.assertIn("1 to 4 words", prompt)
        self.assertIn("copied from inside [[ double brackets ]]", prompt)

    def test_prompt_receives_definition_and_authoritative_edit_war_context(self):
        context = {
            "is_edit_war": True,
            "confirmed_event_count": 3,
            "editor_pairs": [["Alice", "Bob"]],
            "session_ids": ["edit-war-1"],
        }

        prompt = revision_keywords.build_user_prompt(
            "ADDED 1: [[ observer state ]]", edit_war_context=context
        )

        self.assertIn(revision_keywords.EDIT_WAR_DEFINITION, prompt)
        self.assertIn('"is_edit_war": true', prompt)
        self.assertIn("not from the text diff", prompt)
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

    def test_session_prompt_bundles_all_conflict_passages_without_comments(self):
        session = {
            "pair": ["Alice", "Bob"],
            "event_count": 3,
            "revert_kinds": ["exact_sha1", "partial_text"],
            "conflict_passages": ["observer status", "non-member state"],
        }

        prompt = revision_keywords.build_edit_war_description_prompt(session)

        self.assertIn("PASSAGE 1: [[ observer status ]]", prompt)
        self.assertIn("PASSAGE 2: [[ non-member state ]]", prompt)
        self.assertIn("must not decide whether it is an Edit War", prompt)
        self.assertIn("Use no outside knowledge and no\nedit comments", prompt)

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
            "conflict_passages": ["observer status", "non-member state"],
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
            ),
            output_text="",
        )

        result = revision_keywords.request_keywords(
            client,
            "test-model",
            "ADDED 1: [[ observer status ]]",
        )

        self.assertEqual(result["edge_label"], "UN observer status")
        self.assertEqual(result["evidence"], ["observer status"])
        self.assertEqual(client.responses.parse.call_count, 1)

    def test_ungrounded_evidence_is_retried(self):
        client = unittest.mock.Mock()
        invalid = revision_keywords.RevisionKeywords(
            edge_label="UN observer status",
            change_type="addition",
            evidence=["outside knowledge"],
            confidence=0.5,
            added_keywords=["observer status"],
            removed_keywords=[],
            focus_keywords=["observer status", "UN", "vote", "resolution", "statehood"],
        )
        valid = invalid.model_copy(update={"evidence": ["observer status"]})
        client.responses.parse.side_effect = [
            unittest.mock.Mock(output_parsed=invalid, output_text=""),
            unittest.mock.Mock(output_parsed=valid, output_text=""),
        ]

        result = revision_keywords.request_keywords(
            client,
            "test-model",
            "ADDED 1: [[ observer status ]]",
        )

        self.assertEqual(result["evidence"], ["observer status"])
        self.assertEqual(client.responses.parse.call_count, 2)

    def test_label_focus_changes_cache_configuration(self):
        first = revision_keywords.diff_label_configuration_sha256(
            "test-model", 500, "git", "histogram", 12, "UN status"
        )
        second = revision_keywords.diff_label_configuration_sha256(
            "test-model", 500, "git", "histogram", 12, "currency"
        )

        self.assertNotEqual(first, second)


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
        self.assertEqual(revision_keywords.PIPELINE_SNAPSHOT_ID, "2026-08-16-v008")
        self.assertRegex(revision_keywords.source_tree_sha256(), r"^[0-9a-f]{64}$")

    def test_default_diff_limit_covers_measured_corpus_maximum(self):
        self.assertEqual(revision_keywords.DEFAULT_MAX_DIFF_CHARS, 100_000)


if __name__ == "__main__":
    unittest.main()
