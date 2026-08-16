"""Validated word-level opcodes produced by ``git diff --no-index``."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import logging
from functools import lru_cache
from pathlib import Path


Opcode = tuple[str, int, int, int, int]

GIT_DIFF_ALGORITHMS = ("myers", "minimal", "patience", "histogram")
DEFAULT_GIT_DIFF_ALGORITHM = "histogram"

LOGGER = logging.getLogger(__name__)
_LOGGED_GIT_WARNINGS: set[str] = set()

HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?: .*)?$"
)


def _range_from_hunk(start_text: str, count_text: str | None) -> tuple[int, int]:
    """Convert one Git hunk range to a zero-based half-open range."""
    start = int(start_text)
    count = 1 if count_text is None else int(count_text)
    first = start if count == 0 else start - 1
    return first, first + count


def validate_opcodes(
    previous_words: list[str], current_words: list[str], opcodes: list[Opcode]
) -> None:
    """Require ordered opcodes to reconstruct the target sequence exactly."""
    rebuilt = []
    previous_position = 0
    current_position = 0

    for opcode_number, (tag, i1, i2, j1, j2) in enumerate(opcodes, start=1):
        if tag not in {"insert", "delete", "replace"}:
            raise RuntimeError(
                f"Ungültiger Diff-Tag in Opcode {opcode_number}: {tag}"
            )
        if not (
            previous_position <= i1 <= i2 <= len(previous_words)
            and current_position <= j1 <= j2 <= len(current_words)
        ):
            raise RuntimeError(
                f"Ungültige oder überlappende Bereiche in Opcode {opcode_number}: "
                f"{(tag, i1, i2, j1, j2)}"
            )

        previous_gap = previous_words[previous_position:i1]
        current_gap = current_words[current_position:j1]
        if previous_gap != current_gap:
            raise RuntimeError(
                f"Unveränderter Bereich vor Opcode {opcode_number} stimmt nicht überein."
            )
        rebuilt.extend(previous_gap)

        if tag == "insert":
            if i1 != i2 or j1 == j2:
                raise RuntimeError(f"Ungültige Einfügung in Opcode {opcode_number}.")
            rebuilt.extend(current_words[j1:j2])
        elif tag == "delete":
            if i1 == i2 or j1 != j2:
                raise RuntimeError(f"Ungültige Löschung in Opcode {opcode_number}.")
        else:
            if i1 == i2 or j1 == j2:
                raise RuntimeError(f"Ungültige Ersetzung in Opcode {opcode_number}.")
            rebuilt.extend(current_words[j1:j2])

        previous_position = i2
        current_position = j2

    previous_tail = previous_words[previous_position:]
    current_tail = current_words[current_position:]
    if previous_tail != current_tail:
        raise RuntimeError("Unveränderter Bereich nach dem letzten Opcode stimmt nicht überein.")
    rebuilt.extend(previous_tail)

    if rebuilt != current_words:
        raise RuntimeError("Diff-Opcodes rekonstruieren den Zieltext nicht exakt.")


def parse_git_diff(
    output: str, previous_words: list[str], current_words: list[str]
) -> list[Opcode]:
    """Parse a zero-context word-per-line Git diff into validated opcodes."""
    if "Binary files " in output or "GIT binary patch" in output:
        raise RuntimeError("Git lieferte unerwartet einen binären Diff.")

    lines = output.splitlines()
    opcodes: list[Opcode] = []
    index = 0

    while index < len(lines):
        header = HUNK_HEADER_RE.match(lines[index])
        if header is None:
            index += 1
            continue

        i1, i2 = _range_from_hunk(
            header.group("old_start"), header.group("old_count")
        )
        j1, j2 = _range_from_hunk(
            header.group("new_start"), header.group("new_count")
        )
        removed_words = []
        added_words = []
        index += 1

        while index < len(lines) and HUNK_HEADER_RE.match(lines[index]) is None:
            line = lines[index]
            if line.startswith("diff --git ") or line.startswith("index "):
                break
            if line.startswith("--- ") or line.startswith("+++ "):
                break
            if line.startswith("-"):
                removed_words.append(line[1:])
            elif line.startswith("+"):
                added_words.append(line[1:])
            elif line.startswith("\\ No newline at end of file"):
                raise RuntimeError("Temporärer Git-Diff enthält keine Abschluss-Newline.")
            elif line.startswith(" "):
                raise RuntimeError("Git lieferte trotz --unified=0 Kontextzeilen.")
            elif line:
                raise RuntimeError(f"Unerwartete Zeile im Git-Hunk: {line!r}")
            index += 1

        expected_removed = previous_words[i1:i2]
        expected_added = current_words[j1:j2]
        if removed_words != expected_removed:
            raise RuntimeError(
                "Entfernte Git-Zeilen stimmen nicht mit dem alten Wortbereich überein."
            )
        if added_words != expected_added:
            raise RuntimeError(
                "Hinzugefügte Git-Zeilen stimmen nicht mit dem neuen Wortbereich überein."
            )

        if i1 == i2 and j1 < j2:
            tag = "insert"
        elif i1 < i2 and j1 == j2:
            tag = "delete"
        elif i1 < i2 and j1 < j2:
            tag = "replace"
        else:
            raise RuntimeError(
                f"Git-Hunk enthält keine gültige Änderung: {(i1, i2, j1, j2)}"
            )
        opcodes.append((tag, i1, i2, j1, j2))

    if not opcodes:
        raise RuntimeError("Git meldete Änderungen, lieferte aber keinen parsebaren Hunk.")

    validate_opcodes(previous_words, current_words, opcodes)
    return opcodes


@lru_cache(maxsize=1)
def git_version() -> str:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("Git wurde nicht gefunden.")
    environment = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git-Version konnte nicht gelesen werden: {result.stderr.strip()}")
    return result.stdout.strip()


def git_word_opcodes(
    previous_words: list[str],
    current_words: list[str],
    algorithm: str = DEFAULT_GIT_DIFF_ALGORITHM,
) -> list[Opcode]:
    """Return exact word-level Git opcodes without requiring a repository."""
    if algorithm not in GIT_DIFF_ALGORITHMS:
        raise ValueError(
            "Ungültiger Git-Diff-Algorithmus: "
            f"{algorithm}; erwartet wird einer von {', '.join(GIT_DIFF_ALGORITHMS)}."
        )
    if previous_words == current_words:
        return []

    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("Git wurde nicht gefunden.")

    environment = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    with tempfile.TemporaryDirectory(prefix="wikipedia-word-diff-") as temp_dir:
        temp_path = Path(temp_dir)
        previous_path = temp_path / "previous.words"
        current_path = temp_path / "current.words"
        previous_file_text = "\n".join(previous_words)
        current_file_text = "\n".join(current_words)
        if previous_file_text:
            previous_file_text += "\n"
        if current_file_text:
            current_file_text += "\n"
        previous_path.write_text(
            previous_file_text, encoding="utf-8", errors="strict"
        )
        current_path.write_text(
            current_file_text, encoding="utf-8", errors="strict"
        )

        command = [
            executable,
            "diff",
            "--no-index",
            "--text",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--no-indent-heuristic",
            "--inter-hunk-context=0",
            "--unified=0",
            f"--diff-algorithm={algorithm}",
            "--",
            str(previous_path),
            str(current_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=environment,
            check=False,
        )

    if result.returncode == 0:
        raise RuntimeError("Git meldete trotz unterschiedlicher Texte keinen Diff.")
    if result.returncode != 1:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"git diff ist mit Exit-Code {result.returncode} fehlgeschlagen: {message}"
        )
    opcodes = parse_git_diff(result.stdout, previous_words, current_words)
    warning = result.stderr.strip()
    if warning and warning not in _LOGGED_GIT_WARNINGS:
        _LOGGED_GIT_WARNINGS.add(warning)
        LOGGER.warning("git diff warning: %s", warning)
    return opcodes
