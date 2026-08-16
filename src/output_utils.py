"""
Helpers for output file names.
"""

import os
import re


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


def partial_path(path: str) -> str:
    """Return a sibling path with ``.partial`` before the final extension.

    ``str.replace`` must not be used here because an extension-like substring
    can also occur in the article directory or the output basename.
    """
    stem, ext = os.path.splitext(path)
    return f"{stem}.partial{ext}"
