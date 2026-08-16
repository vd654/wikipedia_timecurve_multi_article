#!/usr/bin/env python3
"""Extract grounded keywords from consecutive Wikipedia revisions."""

import argparse
import difflib
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from config import BASE_DIR, article_paths, article_slug
from git_word_diff import (
    DEFAULT_GIT_DIFF_ALGORITHM,
    GIT_DIFF_ALGORITHMS,
    Opcode,
    git_version,
    git_word_opcodes,
    validate_opcodes,
)
from output_utils import next_numbered_path
from text_cleaning import clean_wikitext, extract_infobox_text


# Change only this line to use another OpenAI model.
MODEL = "gpt-5.6-terra"

SYSTEM_PROMPT = """Only use concepts explicitly supported by this revision.
Never use outside knowledge."""

DEFAULT_LIMIT = 0 # Sonderwert, keine Begrenzung
DEFAULT_MAX_DIFF_CHARS = 100_000
DEFAULT_DIFF_CONTEXT_WORDS = 12
DEFAULT_DIFF_ENGINE = "git"
DIFF_ENGINES = ("git", "difflib", "difflib-legacy")
MIN_MAX_DIFF_CHARS = len("[DIFF TRUNCATED]")
PIPELINE_SNAPSHOT_ID = "2026-08-11-v004"


class RevisionKeywords(BaseModel):
    added_keywords: list[str] = Field(max_length=8)
    removed_keywords: list[str] = Field(max_length=8)
    focus_keywords: list[str] = Field(min_length=5, max_length=5)


@dataclass(frozen=True)
class DiffPassage:
    """One rendered side of a grouped change unit."""

    change_type: str
    change_number: int
    words: list[str]
    window_start: int
    window_end: int
    changed_spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DiffResult:
    """Rendered diff plus an auditable account of any prompt truncation."""

    text: str
    diff_truncated: bool
    diff_original_chars: int
    diff_included_chars: int
    previous_word_count: int
    current_word_count: int
    total_diff_opcodes: int
    included_diff_opcodes: int
    omitted_diff_opcodes: int
    first_included_previous_word: int | None
    last_included_previous_word_exclusive: int | None
    first_included_current_word: int | None
    last_included_current_word_exclusive: int | None
    last_included_previous_position_fraction: float | None
    last_included_current_position_fraction: float | None
    total_change_units: int
    included_change_units: int
    omitted_change_units: int
    shortened_change_units: int

    def metadata(self) -> dict:
        data = asdict(self)
        data.pop("text")
        return data


def source_tree_sha256(base_dir: Path | None = None) -> str:
    """Hash executable sources, tests, and dependency declarations."""
    root = base_dir or Path(BASE_DIR)
    paths = sorted(
        [*root.glob("src/*.py"), *root.glob("tests/*.py"), root / "requirements.txt"],
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot-Quelldatei fehlt: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_api_key(env_path: Path) -> str:
    """Read OPENAI_API_KEY from the environment or the project's .env file."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "OPENAI_API_KEY":
                api_key = value.strip().strip('"\'')
                if api_key:
                    return api_key

    raise RuntimeError(
        "OPENAI_API_KEY fehlt. Bitte den Schlüssel in .env eintragen."
    )


def local_name(tag: str) -> str:
    """Return an XML tag without its optional namespace."""
    return tag.rsplit("}", 1)[-1]


def child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if local_name(child.tag) == name:
            return child.text or ""
    return ""


def iter_revisions(
    xml_path: Path,
    min_revision_chars: int = 0,
    revision_ids: list[str] | None = None,
):
    """Yield requested cleaned revisions without cleaning unused XML entries."""
    if not xml_path.exists():
        raise FileNotFoundError(f"XML nicht gefunden: {xml_path}")
    if xml_path.stat().st_size == 0:
        raise ValueError(f"XML-Datei ist leer: {xml_path}")

    wanted_ids = set(revision_ids) if revision_ids is not None else None
    for _, element in ET.iterparse(xml_path, events=("end",)):
        if local_name(element.tag) != "revision": # alles überspringen, was keine Revision ist
            continue

        revision_id = child_text(element, "id")
        if wanted_ids is not None and revision_id not in wanted_ids:
            element.clear()
            continue

        raw_text = child_text(element, "text")
        article_text = clean_wikitext(raw_text)
        infobox_text = extract_infobox_text(raw_text)
        revision = {
            "revision_id": revision_id,
            "timestamp": child_text(element, "timestamp"),
            "text": " ".join(
                part for part in (article_text, infobox_text) if part
            ),
        }
        element.clear()

        if len(revision["text"]) >= min_revision_chars:
            yield revision


def resolve_projection_path(visualization_path: Path) -> Path:
    """Accept either a visualization HTML or its projection JSON sidecar."""
    visualization_path = visualization_path.expanduser().resolve()
    if visualization_path.suffix.casefold() == ".html":
        visualization_path = visualization_path.with_name(
            f"{visualization_path.stem}_projection.json"
        )
    if not visualization_path.exists():
        raise FileNotFoundError(
            f"Projektionsdatei der Visualisierung nicht gefunden: {visualization_path}"
        )
    return visualization_path


def find_latest_visualization(article: str) -> Path:
    """Return the highest-numbered complete visualization for an article."""
    paths = article_paths(article)
    output_dir = Path(paths["output_dir"])
    prefix = paths["output_basename"] + "_"
    candidates = []

    for html_path in output_dir.glob(f"{prefix}*.html"):
        number = html_path.stem.removeprefix(prefix)
        projection = html_path.with_name(f"{html_path.stem}_projection.json")
        if number.isdigit() and projection.exists():
            candidates.append((int(number), html_path))

    if not candidates:
        raise FileNotFoundError(
            f"Keine vollständige Visualisierung gefunden: {output_dir}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def load_visualization_revision_ids(projection_path: Path) -> list[str]:
    """Read the exact ordered point IDs used by the selected visualization."""
    data = json.loads(projection_path.read_text(encoding="utf-8"))
    points = data.get("points")
    if not isinstance(points, list):
        raise ValueError("Die Projektionsdatei enthält keine gültige points-Liste.")

    if any(not isinstance(point, dict) for point in points):
        raise ValueError("Die points-Liste enthält einen ungültigen Eintrag.")
    revision_ids = [str(point.get("revision_id", "")) for point in points]
    if len(revision_ids) < 2 or any(not revision_id for revision_id in revision_ids):
        raise ValueError(
            "Die Visualisierung muss mindestens zwei Punkte mit revision_id enthalten."
        )
    if len(set(revision_ids)) != len(revision_ids):
        raise ValueError("Die Visualisierung enthält doppelte Revisions-IDs.")
    return revision_ids


def select_visualized_revisions(revisions, revision_ids: list[str]):
    """Yield XML revisions in exactly the order stored by the visualization."""
    wanted_index = 0
    for revision in revisions:
        if revision["revision_id"] != revision_ids[wanted_index]:
            continue
        yield revision
        wanted_index += 1
        if wanted_index == len(revision_ids):
            return

    missing = revision_ids[wanted_index:]
    raise ValueError(
        "Revisions-IDs der Visualisierung fehlen in der XML-Datei: "
        + ", ".join(missing[:5])
    )


def render_changed_span(
    words: list[str], start: int, end: int, max_changed_words: int | None
) -> list[str]:
    """Render one changed span, optionally retaining only its ends."""
    changed = words[start:end]
    if max_changed_words is None or len(changed) <= max_changed_words:
        return ["[["] + changed + ["]]"]
    if max_changed_words < 1:
        raise ValueError("max_changed_words muss positiv sein.")

    head_count = (max_changed_words + 1) // 2
    tail_count = max_changed_words // 2
    omitted = len(changed) - max_changed_words
    rendered = ["[["] + changed[:head_count] + ["]]"]
    rendered.append(f"[... {omitted} changed words omitted ...]")
    if tail_count:
        rendered.extend(["[["] + changed[-tail_count:] + ["]]"])
    return rendered


def marked_spans_window(
    words: list[str],
    window_start: int,
    window_end: int,
    changed_spans: tuple[tuple[int, int], ...],
    max_changed_words: int | None = None,
) -> str:
    """Render shared context once while preserving every changed boundary."""
    rendered = []
    position = window_start
    for start, end in changed_spans:
        if not (position <= start < end <= window_end):
            raise RuntimeError(
                f"Ungültiger markierter Bereich: {(start, end)} in "
                f"{(window_start, window_end)}"
            )
        rendered.extend(words[position:start])
        rendered.extend(
            render_changed_span(words, start, end, max_changed_words)
        )
        position = end
    rendered.extend(words[position:window_end])
    return " ".join(rendered)


def marked_window(
    words: list[str],
    start: int,
    end: int,
    context: int = DEFAULT_DIFF_CONTEXT_WORDS,
    max_changed_words: int | None = None,
) -> str:
    """Backward-compatible renderer for a single changed span."""
    return marked_spans_window(
        words,
        max(0, start - context),
        min(len(words), end + context),
        ((start, end),),
        max_changed_words,
    )


def render_passage(
    passage: DiffPassage, max_changed_words: int | None = None
) -> str:
    display = marked_spans_window(
        passage.words,
        passage.window_start,
        passage.window_end,
        passage.changed_spans,
        max_changed_words,
    )
    return f"{passage.change_type} {passage.change_number}: {display}"


def render_unit(
    unit: list[DiffPassage], max_changed_words: int | None = None
) -> list[str]:
    return [render_passage(passage, max_changed_words) for passage in unit]


def truncation_marker(
    total_units: int, omitted_units: int, shortened_units: int = 0
) -> str:
    details = [
        f"{omitted_units} of {total_units} change units omitted",
    ]
    if shortened_units:
        details.append(f"{shortened_units} change unit shortened")
    return "[DIFF TRUNCATED: " + "; ".join(details) + "]"


def render_truncated_diff(lines: list[str], marker: str) -> str:
    if not lines:
        return marker
    return "\n".join(lines + ["", marker])


def compact_first_unit(
    unit: list[DiffPassage], marker: str, max_chars: int
) -> list[str]:
    """Fit a word-safe view of an oversized first unit into the prompt."""
    maximum_changed_words = max(
        end - start
        for passage in unit
        for start, end in passage.changed_spans
    )
    low = 1
    high = maximum_changed_words
    best_lines = []

    while low <= high:
        candidate_limit = (low + high) // 2
        candidate_lines = render_unit(unit, candidate_limit)
        candidate_text = render_truncated_diff(candidate_lines, marker)
        if len(candidate_text) <= max_chars:
            best_lines = candidate_lines
            low = candidate_limit + 1
        else:
            high = candidate_limit - 1

    return best_lines


def diff_opcodes(
    previous_words: list[str],
    current_words: list[str],
    engine: str,
    git_diff_algorithm: str,
) -> list[Opcode]:
    """Return validated change opcodes from the selected single diff engine."""
    if engine == "git":
        return git_word_opcodes(
            previous_words, current_words, algorithm=git_diff_algorithm
        )
    if engine not in {"difflib", "difflib-legacy"}:
        raise ValueError(
            f"Ungültige Diff-Engine: {engine}; erwartet wird eine von "
            f"{', '.join(DIFF_ENGINES)}."
        )

    matcher = difflib.SequenceMatcher(
        None,
        previous_words,
        current_words,
        autojunk=(engine == "difflib-legacy"),
    )
    opcodes = [
        opcode for opcode in matcher.get_opcodes() if opcode[0] != "equal"
    ]
    validate_opcodes(previous_words, current_words, opcodes)
    return opcodes


def group_nearby_opcodes(
    opcodes: list[Opcode], context: int
) -> list[list[Opcode]]:
    """Group changes whose old or new context windows would overlap."""
    if context < 0:
        raise ValueError("context muss nichtnegativ sein.")
    groups: list[list[Opcode]] = []
    for opcode in opcodes:
        if not groups:
            groups.append([opcode])
            continue

        previous = groups[-1][-1]
        old_gap = opcode[1] - previous[2]
        new_gap = opcode[3] - previous[4]
        if old_gap != new_gap:
            raise RuntimeError(
                "Validierte Opcodes haben unterschiedliche alte und neue Abstände."
            )
        if old_gap < 2 * context:
            groups[-1].append(opcode)
        else:
            groups.append([opcode])
    return groups


def passages_for_group(
    group: list[Opcode],
    change_number: int,
    previous_words: list[str],
    current_words: list[str],
    context: int,
) -> list[DiffPassage]:
    """Build at most one removed and one added line with shared context."""
    unit = []
    old_spans = tuple(
        (i1, i2) for tag, i1, i2, _, _ in group if tag in {"delete", "replace"}
    )
    new_spans = tuple(
        (j1, j2) for tag, _, _, j1, j2 in group if tag in {"insert", "replace"}
    )

    if old_spans:
        group_start = min(opcode[1] for opcode in group)
        group_end = max(opcode[2] for opcode in group)
        unit.append(DiffPassage(
            "REMOVED",
            change_number,
            previous_words,
            max(0, group_start - context),
            min(len(previous_words), group_end + context),
            old_spans,
        ))
    if new_spans:
        group_start = min(opcode[3] for opcode in group)
        group_end = max(opcode[4] for opcode in group)
        unit.append(DiffPassage(
            "ADDED",
            change_number,
            current_words,
            max(0, group_start - context),
            min(len(current_words), group_end + context),
            new_spans,
        ))
    return unit


def opcode_coverage_metadata(
    opcode_groups: list[list[Opcode]],
    included_units: int,
    previous_word_count: int,
    current_word_count: int,
) -> dict:
    """Describe how far the represented prefix reaches in both revisions."""
    included_opcodes = [
        opcode
        for group in opcode_groups[:included_units]
        for opcode in group
    ]
    included_count = len(included_opcodes)
    total_count = sum(len(group) for group in opcode_groups)

    if included_opcodes:
        first_previous = min(opcode[1] for opcode in included_opcodes)
        last_previous = max(opcode[2] for opcode in included_opcodes)
        first_current = min(opcode[3] for opcode in included_opcodes)
        last_current = max(opcode[4] for opcode in included_opcodes)
    else:
        first_previous = last_previous = None
        first_current = last_current = None

    return {
        "previous_word_count": previous_word_count,
        "current_word_count": current_word_count,
        "included_diff_opcodes": included_count,
        "omitted_diff_opcodes": total_count - included_count,
        "first_included_previous_word": first_previous,
        "last_included_previous_word_exclusive": last_previous,
        "first_included_current_word": first_current,
        "last_included_current_word_exclusive": last_current,
        "last_included_previous_position_fraction": (
            last_previous / previous_word_count
            if last_previous is not None and previous_word_count
            else None
        ),
        "last_included_current_position_fraction": (
            last_current / current_word_count
            if last_current is not None and current_word_count
            else None
        ),
    }


def build_diff(
    previous_text: str,
    current_text: str,
    max_chars: int = DEFAULT_MAX_DIFF_CHARS,
    diff_engine: str = DEFAULT_DIFF_ENGINE,
    git_diff_algorithm: str = DEFAULT_GIT_DIFF_ALGORITHM,
    context: int = DEFAULT_DIFF_CONTEXT_WORDS,
) -> DiffResult:
    """Build a bounded, prefix-faithful diff with truncation metadata."""
    if max_chars < MIN_MAX_DIFF_CHARS:
        raise ValueError(
            f"max_chars muss mindestens {MIN_MAX_DIFF_CHARS} betragen, "
            "damit ein Kürzungsmarker Platz hat."
        )

    previous_words = previous_text.split()
    current_words = current_text.split()
    opcodes = diff_opcodes(
        previous_words, current_words, diff_engine, git_diff_algorithm
    )
    opcode_groups = group_nearby_opcodes(opcodes, context)
    units = [
        passages_for_group(
            group,
            change_number,
            previous_words,
            current_words,
            context,
        )
        for change_number, group in enumerate(opcode_groups, start=1)
    ]

    rendered_units = [render_unit(unit) for unit in units]
    full_text = "\n".join(
        line for rendered_unit in rendered_units for line in rendered_unit
    )
    total_units = len(units)
    original_chars = len(full_text)

    if original_chars <= max_chars:
        coverage = opcode_coverage_metadata(
            opcode_groups, total_units, len(previous_words), len(current_words)
        )
        return DiffResult(
            text=full_text,
            diff_truncated=False,
            diff_original_chars=original_chars,
            diff_included_chars=original_chars,
            **coverage,
            total_diff_opcodes=len(opcodes),
            total_change_units=total_units,
            included_change_units=total_units,
            omitted_change_units=0,
            shortened_change_units=0,
        )

    selected_lines = []
    included_units = 0
    for rendered_unit in rendered_units:
        candidate_included = included_units + 1
        candidate_omitted = total_units - candidate_included
        marker = truncation_marker(total_units, candidate_omitted)
        candidate_text = render_truncated_diff(
            selected_lines + rendered_unit, marker
        )
        if len(candidate_text) > max_chars:
            break
        selected_lines.extend(rendered_unit)
        included_units = candidate_included

    shortened_units = 0
    if included_units == 0 and units:
        omitted_units = total_units - 1
        marker = truncation_marker(total_units, omitted_units, shortened_units=1)
        selected_lines = compact_first_unit(units[0], marker, max_chars)
        if selected_lines:
            included_units = 1
            shortened_units = 1

    omitted_units = total_units - included_units
    marker = truncation_marker(total_units, omitted_units, shortened_units)
    if len(marker) > max_chars:
        marker = "[DIFF TRUNCATED]"
    text = render_truncated_diff(selected_lines, marker)
    coverage = opcode_coverage_metadata(
        opcode_groups, included_units, len(previous_words), len(current_words)
    )

    return DiffResult(
        text=text,
        diff_truncated=True,
        diff_original_chars=original_chars,
        diff_included_chars=len(text),
        **coverage,
        total_diff_opcodes=len(opcodes),
        total_change_units=total_units,
        included_change_units=included_units,
        omitted_change_units=omitted_units,
        shortened_change_units=shortened_units,
    )


def build_user_prompt(diff_text: str, diff_truncated: bool = False) -> str:
    truncation_instruction = ""
    if diff_truncated:
        truncation_instruction = (
            "The supplied diff is truncated. Use only the visible passages and never\n"
            "assume that they represent every change in the revision.\n"
        )
    return f"""Compare these consecutive Wikipedia revisions.
{truncation_instruction}If ADDED passages exist, return 1 to 8 added keywords; otherwise return an empty list.
If REMOVED passages exist, return 1 to 8 removed keywords; otherwise return an empty list.
Never fill one list only because the other list contains keywords.
Each list must contain either zero keywords or 1 to 8 keywords.
Each keyword must contain one, two, three, or four words.
For added and removed keywords, use only concepts inside [[ double brackets ]].

For ADDED keywords, select concepts representing the important information
introduced by the added passage.

For REMOVED keywords, select concepts representing the important information
contained in the removed passage.

Treat ADDED and REMOVED independently.

Return exactly 5 focus keywords describing the specific content changed in this revision.
Determine the focus from the words inside [[ double brackets ]] and their immediate context.
Do not summarize the whole article.
Do not choose a section heading or the article's general topic unless the changed text itself supports it.
Prefer specific people, places, events, policies, claims, or institutions affected by the change.
Only describe a change as controversial or polarizing when the changed text explicitly supports that.

REVISION DIFF
{diff_text}"""


def request_keywords(
    client: OpenAI,
    model: str,
    diff_text: str,
    diff_truncated: bool = False,
) -> dict:
    """Ask for added, removed, and change-focused keywords."""
    diff_lines = diff_text.splitlines()
    if not any(
        line.startswith(("ADDED ", "REMOVED ")) for line in diff_lines
    ):
        return {
            "added_keywords": [],
            "removed_keywords": [],
            "focus_keywords": [],
        }

    change_types = {
        "added_keywords": "ADDED ",
        "removed_keywords": "REMOVED ",
    }
    prompt = build_user_prompt(diff_text, diff_truncated)

    for attempt in range(3):
        response = client.responses.parse(
            model=model,
            store=False,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            text_format=RevisionKeywords,
        )
        if response.output_parsed is None:
            raise RuntimeError(
                f"Keine auswertbare Modellantwort: {response.output_text}"
            )

        keywords = response.output_parsed.model_dump()
        keywords["focus_keywords"] = [
            " ".join(keyword.split()[:4])
            for keyword in keywords["focus_keywords"]
        ]
        invalid_types = []
        for name, prefix in change_types.items():
            if not any(line.startswith(prefix) for line in diff_lines):
                keywords[name] = []
                continue

            keywords[name] = [
                " ".join(keyword.split()[:4]) for keyword in keywords[name]
            ]
            if len(keywords[name]) < 1:
                invalid_types.append(prefix.strip())

        if not invalid_types:
            return keywords
        prompt += (
            "\nCorrection required: Return 1 to 8 grounded keywords for "
            + " and ".join(invalid_types)
            + "."
        )

    raise RuntimeError(
        "Das Modell lieferte trotz Wiederholung kein Stichwort."
    )


def analyze(
    xml_path: Path,
    revision_ids: list[str],
    client: OpenAI,
    model: str,
    limit: int,
    max_diff_chars: int,
    diff_engine: str = DEFAULT_DIFF_ENGINE,
    git_diff_algorithm: str = DEFAULT_GIT_DIFF_ALGORITHM,
    diff_context_words: int = DEFAULT_DIFF_CONTEXT_WORDS,
) -> list[dict]:
    revisions = iter(select_visualized_revisions(
        iter_revisions(xml_path, revision_ids=revision_ids), revision_ids
    ))
    try:
        previous = next(revisions)
    except StopIteration as exc:
        raise ValueError("Die XML-Datei enthält keine geeignete Revision.") from exc

    results = []
    for pair_number, current in enumerate(revisions, start=1):
        if limit > 0 and pair_number > limit:
            break

        print(
            f"[{pair_number}] Revision {previous['revision_id']} -> "
            f"{current['revision_id']}"
        )
        diff_result = build_diff(
            previous["text"],
            current["text"],
            max_chars=max_diff_chars,
            diff_engine=diff_engine,
            git_diff_algorithm=git_diff_algorithm,
            context=diff_context_words,
        )
        keywords = request_keywords(
            client,
            model,
            diff_result.text,
            diff_truncated=diff_result.diff_truncated,
        )
        results.append({
            "previous_revision_id": previous["revision_id"],
            "previous_timestamp": previous["timestamp"],
            "current_revision_id": current["revision_id"],
            "current_timestamp": current["timestamp"],
            **diff_result.metadata(),
            **keywords,
        })
        previous = current

    return results


def print_results(results: list[dict]) -> None:
    print("\nERGEBNISSE")
    for result in results:
        print(
            f"\n{result['previous_revision_id']} -> "
            f"{result['current_revision_id']}"
        )
        if result["total_change_units"] == 0:
            print("  Keine Textänderung erkannt.")
        elif result["included_change_units"] == 0:
            print("  Keine Änderungspassage passte in das Diff-Limit.")
        if result["diff_truncated"]:
            print(
                "  DIFF GEKÜRZT: "
                f"{result['omitted_change_units']} Units ausgelassen, "
                f"{result['shortened_change_units']} gekürzt"
            )
        print("  ADDED:")
        for keyword in result["added_keywords"]:
            print(f"    {keyword}")
        print("  REMOVED:")
        for keyword in result["removed_keywords"]:
            print(f"    {keyword}")
        print("  MAIN FOCUS:")
        for keyword in result["focus_keywords"]:
            print(f"    {keyword}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grounded keywords for changes between consecutive revisions."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--article", help='Article name, e.g. "Palestine"')
    source.add_argument("--xml", type=Path, help="Direct path to a revision XML")
    parser.add_argument(
        "--visualization", type=Path,
        help="Optional visualization; otherwise use the latest article visualization",
    )
    parser.add_argument("--model", default=MODEL, help=f"OpenAI model (default: {MODEL})")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help="Optional number of visualization pairs; 0 means all (default: all)",
    )
    parser.add_argument("--max-diff-chars", type=int, default=DEFAULT_MAX_DIFF_CHARS)
    parser.add_argument(
        "--diff-engine",
        choices=DIFF_ENGINES,
        default=DEFAULT_DIFF_ENGINE,
        help="Diff engine (default: git; difflib disables autojunk)",
    )
    parser.add_argument(
        "--git-diff-algorithm",
        choices=GIT_DIFF_ALGORITHMS,
        default=DEFAULT_GIT_DIFF_ALGORITHM,
        help=f"Git diff algorithm (default: {DEFAULT_GIT_DIFF_ALGORITHM})",
    )
    parser.add_argument(
        "--diff-context-words",
        type=int,
        default=DEFAULT_DIFF_CONTEXT_WORDS,
        help=f"Shared context around change groups (default: {DEFAULT_DIFF_CONTEXT_WORDS})",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit darf nicht negativ sein.")
    if args.max_diff_chars < MIN_MAX_DIFF_CHARS:
        raise ValueError(
            f"--max-diff-chars muss mindestens {MIN_MAX_DIFF_CHARS} betragen."
        )
    if args.diff_context_words < 0:
        raise ValueError("--diff-context-words darf nicht negativ sein.")

    xml_path = (
        Path(article_paths(args.article)["xml_path"])
        if args.article else args.xml.expanduser().resolve()
    )
    if args.visualization:
        visualization_path = args.visualization
    elif args.article:
        visualization_path = find_latest_visualization(args.article)
    else:
        raise ValueError("Bei --xml muss --visualization angegeben werden.")

    projection_path = resolve_projection_path(visualization_path)
    revision_ids = load_visualization_revision_ids(projection_path)
    slug = article_slug(args.article) if args.article else xml_path.stem
    visualization_name = projection_path.stem.removesuffix("_projection")
    output_path = args.output or Path(next_numbered_path(
        str(projection_path.parent),
        f"{visualization_name}_revision_keywords_{article_slug(args.model)}",
        ext="json",
    ))

    print(f"XML: {xml_path}")
    print(f"Visualisierung: {projection_path}")
    print(f"Modell: {args.model}")
    print(f"Pipeline-Snapshot: {PIPELINE_SNAPSHOT_ID}")
    print(f"Diff-Engine: {args.diff_engine}")
    if args.diff_engine == "git":
        print(f"Git: {git_version()}")
        print(f"Git-Diff-Algorithmus: {args.git_diff_algorithm}")
    print(f"Diff-Kontext: {args.diff_context_words} Wörter")
    print(f"Visualisierungspunkte: {len(revision_ids)}")
    print(f"Paare: {'alle' if args.limit == 0 else args.limit}")

    api_key = load_api_key(Path(BASE_DIR) / ".env")
    results = analyze(
        xml_path=xml_path,
        revision_ids=revision_ids,
        client=OpenAI(api_key=api_key),
        model=args.model,
        limit=args.limit,
        max_diff_chars=args.max_diff_chars,
        diff_engine=args.diff_engine,
        git_diff_algorithm=args.git_diff_algorithm,
        diff_context_words=args.diff_context_words,
    )
    print_results(results)

    payload = {
        "article": args.article or slug,
        "xml": str(xml_path),
        "visualization": str(projection_path),
        "model": args.model,
        "pipeline_snapshot_id": PIPELINE_SNAPSHOT_ID,
        "source_tree_sha256": source_tree_sha256(),
        "sampling": "exact_revision_order_from_visualization",
        "diff_engine": args.diff_engine,
        "diff_algorithm": (
            args.git_diff_algorithm if args.diff_engine == "git"
            else "ratcliff-obershelp"
        ),
        "diff_autojunk": (
            args.diff_engine == "difflib-legacy"
            if args.diff_engine in {"difflib", "difflib-legacy"}
            else None
        ),
        "git_version": git_version() if args.diff_engine == "git" else None,
        "diff_context_words": args.diff_context_words,
        "diff_input_unit": "whitespace-separated word, one word per Git input line",
        "diff_hunk_context_lines": 0 if args.diff_engine == "git" else None,
        "diff_grouping": "group when validated unchanged gap is less than 2 * context",
        "max_diff_chars": args.max_diff_chars,
        "truncated_revision_pair_count": sum(
            result["diff_truncated"] for result in results
        ),
        "system_prompt": SYSTEM_PROMPT,
        "revision_pairs": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nJSON gespeichert: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        UnicodeError,
        ET.ParseError,
    ) as error:
        print(f"[FEHLER] {error}")
        raise SystemExit(1) from error
