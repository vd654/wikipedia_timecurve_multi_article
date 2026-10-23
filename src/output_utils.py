"""
Helpers for output file names.
"""

import json
import os
import re
from pathlib import Path


def next_numbered_path(directory: str, base_name: str,
                       ext: str = "html", digits: int = 3) -> str:
    """
    Return the next free path '<base_name>_NNN.<ext>' in directory.
    Existing output files are kept instead of overwritten.
    """
    os.makedirs(directory, exist_ok=True)
    ext = ext.lstrip(".")
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)\.{re.escape(ext)}$")

    used = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            used.append(int(match.group(1)))

    next_number = max(used) + 1 if used else 1
    return os.path.join(directory, f"{base_name}_{next_number:0{digits}d}.{ext}")


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text through a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: Path, payload: object) -> None:
    """Write consistently formatted JSON without exposing partial output."""
    write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
