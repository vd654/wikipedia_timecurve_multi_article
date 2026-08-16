import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from git_word_diff import (
    GIT_DIFF_ALGORITHMS,
    git_version,
    git_word_opcodes,
    parse_git_diff,
    validate_opcodes,
)


def rebuild(previous_words, current_words, opcodes):
    result = []
    previous_position = 0
    for tag, i1, i2, j1, j2 in opcodes:
        result.extend(previous_words[previous_position:i1])
        if tag in {"insert", "replace"}:
            result.extend(current_words[j1:j2])
        previous_position = i2
    result.extend(previous_words[previous_position:])
    return result


class GitHunkParserTests(unittest.TestCase):
    def test_missing_count_defaults_to_one_and_zero_count_keeps_start(self):
        previous = "a b c".split()
        current = "a b c x".split()
        output = "@@ -3,0 +4 @@ c\n+x\n"

        opcodes = parse_git_diff(output, previous, current)

        self.assertEqual(opcodes, [("insert", 3, 3, 3, 4)])

    def test_delete_zero_count_uses_position_before_removed_target_gap(self):
        previous = "a b c x y z d".split()
        current = "a b c d".split()
        output = "@@ -4,3 +3,0 @@ c\n-x\n-y\n-z\n"

        opcodes = parse_git_diff(output, previous, current)

        self.assertEqual(opcodes, [("delete", 3, 6, 3, 3)])

    def test_unparseable_changed_output_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "keinen parsebaren Hunk"):
            parse_git_diff("unexpected output\n", ["a"], ["b"])

    def test_binary_output_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "binären Diff"):
            parse_git_diff("Binary files old and new differ\n", ["a"], ["b"])

    def test_hunk_content_must_match_source_slices(self):
        output = "@@ -1 +1 @@\n-wrong\n+b\n"
        with self.assertRaisesRegex(RuntimeError, "alten Wortbereich"):
            parse_git_diff(output, ["a"], ["b"])


class GitOpcodeIntegrationTests(unittest.TestCase):
    def assert_rebuilds(self, previous, current, expected=None):
        opcodes = git_word_opcodes(previous, current)
        validate_opcodes(previous, current, opcodes)
        self.assertEqual(rebuild(previous, current, opcodes), current)
        if expected is not None:
            self.assertEqual(opcodes, expected)

    def test_pure_insert(self):
        self.assert_rebuilds(
            "a b c d".split(),
            "a b c x d".split(),
            [("insert", 3, 3, 3, 4)],
        )

    def test_pure_delete(self):
        self.assert_rebuilds(
            "a b c x y z d".split(),
            "a b c d".split(),
            [("delete", 3, 6, 3, 3)],
        )

    def test_replace(self):
        self.assert_rebuilds(
            "a b c d e".split(),
            "a b x y e".split(),
            [("replace", 2, 4, 2, 4)],
        )

    def test_all_exposed_algorithms_produce_valid_opcodes(self):
        previous = "a b c d e f".split()
        current = "a b x d y f".split()
        for algorithm in GIT_DIFF_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                opcodes = git_word_opcodes(previous, current, algorithm)
                self.assertEqual(rebuild(previous, current, opcodes), current)

    def test_git_version_is_recordable(self):
        self.assertRegex(git_version(), r"^git version \d+")

    def test_valid_diff_with_stderr_warning_is_logged_not_rejected(self):
        completed = SimpleNamespace(
            returncode=1,
            stdout="@@ -1 +1 @@\n-a\n+b\n",
            stderr="warning: harmless test warning\n",
        )
        with (
            patch("git_word_diff.shutil.which", return_value="/usr/bin/git"),
            patch("git_word_diff.subprocess.run", return_value=completed),
            self.assertLogs("git_word_diff", level="WARNING") as logs,
        ):
            opcodes = git_word_opcodes(["a"], ["b"])

        self.assertEqual(opcodes, [("replace", 0, 1, 0, 1)])
        self.assertIn("harmless test warning", logs.output[0])

    def test_empty_sequences(self):
        self.assert_rebuilds([], ["x"], [("insert", 0, 0, 0, 1)])
        self.assert_rebuilds(["x"], [], [("delete", 0, 1, 0, 0)])
        self.assert_rebuilds([], [], [])

    def test_large_random_edit_script_reconstructs_exactly(self):
        generator = random.Random(20260810)
        previous = [f"word{generator.randrange(80)}" for _ in range(1_000)]
        current = previous.copy()
        for _ in range(160):
            operation = generator.choice(("insert", "delete", "replace"))
            position = generator.randrange(len(current) + 1)
            if operation == "insert" or not current:
                current[position:position] = [f"new{generator.randrange(500)}"]
            elif operation == "delete" and position < len(current):
                del current[position]
            elif position < len(current):
                current[position] = f"replacement{generator.randrange(500)}"

        self.assert_rebuilds(previous, current)

    def test_validator_rejects_off_by_one_opcode(self):
        previous = "a b c d".split()
        current = "a b c x d".split()
        with self.assertRaises(RuntimeError):
            validate_opcodes(
                previous, current, [("insert", 2, 2, 3, 4)]
            )


if __name__ == "__main__":
    unittest.main()
