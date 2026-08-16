import sys
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
            {"revision_id": "1", "timestamp": "t1", "text": "alpha beta"},
            {"revision_id": "2", "timestamp": "t2", "text": "alpha gamma"},
        ])
        keywords = {
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
                client=None,
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

    def test_marker_only_diff_does_not_call_the_model(self):
        client = unittest.mock.Mock()

        result = revision_keywords.request_keywords(
            client, "test-model", "[DIFF TRUNCATED]"
        )

        self.assertEqual(result, {
            "added_keywords": [],
            "removed_keywords": [],
            "focus_keywords": [],
        })
        client.responses.parse.assert_not_called()


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
        self.assertEqual(revision_keywords.PIPELINE_SNAPSHOT_ID, "2026-08-11-v004")
        self.assertRegex(revision_keywords.source_tree_sha256(), r"^[0-9a-f]{64}$")

    def test_default_diff_limit_covers_measured_corpus_maximum(self):
        self.assertEqual(revision_keywords.DEFAULT_MAX_DIFF_CHARS, 100_000)


if __name__ == "__main__":
    unittest.main()
