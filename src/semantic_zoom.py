#!/usr/bin/env python3
"""Build an offline label hierarchy and a semantic-zoom Vega timecurve."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from config import BASE_DIR, article_paths
from llm_provider import (
    DEFAULT_OPENAI_MODEL,
    LLM_PROVIDER,
    create_llm_client,
    model_or_default,
)
from revert_detection import (
    REVERT_ANALYSIS_VERSION,
    analyze_reverts,
    history_window_for_points,
    load_revert_history,
)
from revision_keywords import (
    DEFAULT_GIT_DIFF_ALGORITHM,
    diff_opcodes,
    iter_revisions,
    load_api_key,
    select_visualized_revisions,
)


MODEL = DEFAULT_OPENAI_MODEL
SEMANTIC_ZOOM_SNAPSHOT_ID = "2026-10-22-v101"
MIN_VIEW_SCALE = 0.01
MAX_VIEW_SCALE = 200.0
MIN_PARENT_LABEL_WORDS = 2
MAX_PARENT_LABEL_WORDS = 3
BACKGROUND_COLOR = "#fbfaf7"
TRAJECTORY_COLOR = "#72806f"
NOISE_COLOR = "#a8a8a8"
MAX_MACRO_GROUPS = 6
# Revision-point colour is a deterministic code for the incoming cleaned-text
# transition. Macro topics remain visible through their direct labels; they do
# not receive their own colour encoding or interactive focus state.
WORD_CHANGE_COLOURS = {
    "start": "#343a40",
    "added": "#f2c14e",
    "removed": "#8f5bd7",
    "mixed": "#d9822b",
    "no_change": "#b8b8b8",
    "unavailable": NOISE_COLOR,
}
WORD_CHANGE_LABELS = {
    "start": "Start",
    "added": "Added only",
    "removed": "Removed only",
    "mixed": "Added and removed",
    "no_change": "No change",
    "unavailable": "Change type unavailable",
}
REVISION_POINT_COLOUR_ENCODING = {
    "source": "audited Git word diff between consecutive cleaned revisions",
    "assignment": "each point receives the type of its incoming transition",
    "first_point": "start",
    "rules": {
        "added": "added_words > 0 and removed_words == 0",
        "removed": "added_words == 0 and removed_words > 0",
        "mixed": "added_words > 0 and removed_words > 0",
        "no_change": "added_words == 0 and removed_words == 0",
    },
    "colours": WORD_CHANGE_COLOURS,
}
EXACT_REVERT_COLOR = "#d64545"
REVERT_ARC_STROKE_WIDTH = 1.6
REVERT_ARC_STROKE_OPACITY = 0.45
REVERT_ARROW_OPACITY = 0.45
# Exact returns can land a few projection units apart even when their arcs are
# visually the same connection.  Treat endpoints within five percent of the
# connection length as one visual pair, independently of direction, and retain
# only the chronologically last return below.
REVERT_CONNECTION_ENDPOINT_TOLERANCE_RATIO = 0.05
REVISION_POINT_SIZE = 80
CLUSTER_CORRIDOR_STROKE_WIDTH = 1.4
EDGE_SHORT_LABEL_OFFSET = 9
EDGE_DETAILED_LABEL_OFFSET = 28
EDGE_LABEL_FONT_SIZE = 10
EDGE_LABEL_LINE_HEIGHT = 12
# Detailed descriptions need enough visible trajectory context; otherwise a
# long free-floating paragraph can dominate an only barely separated edge.
MIN_DETAILED_EDGE_SCREEN_LENGTH = 140
LABEL_COLLISION_PADDING = 6
LABEL_SPATIAL_HASH_CELL_SIZE = 64
PARENT_LABEL_RADIAL_GAP = 10
PARENT_LABEL_DATA_PADDING = 12
EDGE_LABEL_REPEAT_DISTANCE = 120
LABEL_DETAIL_MIN_ZOOM_LEVEL = 4.25
LABEL_OBSTACLE_GRID_SIZE = 8
LABEL_EDGE_DENSITY_ENTER = 0.34
LABEL_EDGE_DENSITY_EXIT = 0.46
LABEL_DETAIL_DENSITY_ENTER = 0.16
LABEL_DETAIL_DENSITY_EXIT = 0.24
# Every 1.5 binary zoom levels moves the preferred semantic depth one step:
# macro -> cluster -> section -> short edge label -> detailed edge label.
# Other levels remain candidates and can still appear wherever screen space is
# genuinely free; this value controls priority, not hard visibility windows.
SEMANTIC_DEPTH_ZOOM_STEP = 1.5
# A slightly reduced forward penalty lets the next, finer semantic level enter
# the competition before it becomes the exact target depth. Coarser context
# remains somewhat more persistent while zooming in.
LABEL_FINER_DEPTH_PENALTY = 210
LABEL_COARSER_DEPTH_PENALTY = 120

# Line width is an event code, not a semantic-distance scale. A normal edge is
# clearly legible, and repeatedly traversed undirected MDS segments receive an
# increment for every extra traversal. Exact SHA1 reverts are encoded by a
# separate directed dotted arc instead of by line width or line colour.
TRAJECTORY_WIDTH_SIGNAL = (
    "min(8, 2.25 + 1.1*max(datum.overlap_count-1, 0))"
)
TRAJECTORY_WIDTH_ENCODING = {
    "normal_edge_px": 2.25,
    "exact_sha1_revert_increment_px": 0.0,
    "exact_sha1_revert_encoding": (
        "directed dotted arc from reverted revision to reverting revision"
    ),
    "additional_same_geometry_increment_px": 1.1,
    "maximum_px": 8.0,
    "overlap_definition": (
        "same undirected MDS endpoints rounded to 12 decimal places"
    ),
    "semantic_jump_controls_width": False,
}
TRAJECTORY_SPLINE_ENCODING = {
    "type": "centripetal Catmull-Rom",
    "alpha": 0.5,
    "interpolates_revision_points": True,
    "rendering": "one cubic Bezier path per consecutive revision pair",
    "endpoint_rule": "reflected neighbour",
    "coincident_point_rule": "zero-length segment with bounded tangents",
}
LABEL_SYSTEM_PROMPT = """You create concise English map labels for a semantic revision
trajectory. Use only the supplied full-revision comparison labels, detailed descriptions,
and grounded evidence. The descriptions may contain authors, timestamps, and edit comments;
use those details to understand patterns, chronology, and editorial context when helpful,
but label the subject of the article changes rather than the editing activity. An edit
comment can be inaccurate, so repeated text-grounded evidence has priority. Never add
outside knowledge.
The final visualization is a five-level pyramid: article, macro topic, cluster,
chronological section, and consecutive revision edge. You are labeling cluster
and section nodes; macro topics are generated in a separate parent pass and the
detailed revision edges already exist as grounded leaf labels.
Each label must explain what its cluster or chronological section is about and must not be
a generic label such as 'changes', 'revision', 'miscellaneous', or a numbered level. Use two to three
words. Return exactly one English label per node. Never join, enumerate, or
contrast multiple candidate labels. Preserve meaningful names, institutions,
places, events, and policies. Name the article subject or proposition being discussed,
never the editing operation. Do not label a parent area with mechanics such as formatting,
punctuation, spacing, capitalization, wording, terminology, citations, references,
templates, infobox fields, spelling, or interlanguage links. For a formatting-only child
edit, recover the underlying people, institutions, places, claims, or policies from its
detailed description. For example, an em-dash spacing edit in sentences about UN
representation and Jordanian claims should yield "PLO Representation Claims", not
"Em Dash Spacing"."""

MACRO_LABEL_SYSTEM_PROMPT = """You create concise English overview labels for a semantic
map. Use only the supplied child cluster labels, detailed transition descriptions, and
grounded full-revision comparison evidence. Author, timestamp, and edit-comment details may
help reveal editorial or chronological patterns, but the label must name the shared subject
of the article changes. Never add outside knowledge. Each node combines several spatially adjacent HDBSCAN clusters and
needs one broad umbrella concept that makes the map understandable before the user
zooms into the individual clusters. Use two to three words. Preserve meaningful
names, institutions, places, events, and policies. Never use technical terms such as
'cluster', 'group', 'area', 'level', 'miscellaneous', or a number. Describe the
article content, never editing mechanics such as formatting, wording, citations,
templates, or infobox fields. Return exactly one English umbrella label per node.
Never join, enumerate, or contrast multiple candidate labels."""


class NodeLabel(BaseModel):
    node_id: str
    label: str = Field(min_length=2, max_length=80)


class NodeLabelBatch(BaseModel):
    labels: list[NodeLabel]


def _normalise_term(value: str) -> str:
    return " ".join(value.strip().split())


ISO_TIMESTAMP_PATTERN = re.compile(
    r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b"
)


def compact_display_timestamps(value: str) -> str:
    """Shorten ISO timestamps for labels without altering stored evidence."""
    return ISO_TIMESTAMP_PATTERN.sub(
        lambda match: (
            f"{match.group('day')}-{match.group('month')}-"
            f"{match.group('year')}"
        ),
        value,
    )


def valid_parent_label(value: str) -> bool:
    """Require one intact two-to-three-word hierarchy label."""
    words = _normalise_term(value).split()
    return MIN_PARENT_LABEL_WORDS <= len(words) <= MAX_PARENT_LABEL_WORDS


EDITORIAL_MECHANICS_PATTERN = re.compile(
    r"\b(?:format(?:ting)?|punctuation|spacing|capitali[sz]ation|wording|"
    r"terminology|citations?|references?|templates?|infobox|spelling|"
    r"misspelling|grammar|typography|copyedit(?:ing)?|interlanguage|links?)\b",
    re.IGNORECASE,
)

# These nouns are not wrong in ordinary prose, but as two-to-three-word map
# labels they merely describe a container or an undifferentiated collection.
# The hierarchy must instead name the actual subject represented by the child
# evidence (for example, "Palestinian State Indicators" rather than
# "Palestine Infobox Details").
GENERIC_PARENT_LABEL_PATTERN = re.compile(
    r"\b(?:article|background|content|context|details?|profile|affairs|"
    r"revisions?)\b",
    re.IGNORECASE,
)


def content_focused_parent_label(value: str) -> bool:
    """Reject hierarchy labels that describe editing mechanics, not content."""
    label = _normalise_term(value)
    return not (
        EDITORIAL_MECHANICS_PATTERN.search(label)
        or GENERIC_PARENT_LABEL_PATTERN.search(label)
    )


SPECIFICITY_STOPWORDS = {
    "a", "an", "and", "for", "in", "of", "the", "to",
    "palestine", "palestinian", "state",
}


def _specificity_tokens(value: object) -> set[str]:
    return {
        token
        for token in _term_key(str(value or "")).split()
        if token not in SPECIFICITY_STOPWORDS
    }


def hierarchy_specificity_issues(hierarchy: dict) -> list[dict]:
    """Return parent-child labels that fail to add grounded detail."""
    levels = ("roots", "macros", "clusters", "sections", "edges")
    nodes = {
        str(node.get("node_id", "")): node
        for level in levels
        for node in hierarchy.get(level, [])
    }
    issues: list[dict] = []
    for child in nodes.values():
        parent = nodes.get(str(child.get("parent_id") or ""))
        if parent is None:
            continue
        parent_tokens = _specificity_tokens(parent.get("label"))
        child_tokens = _specificity_tokens(child.get("label"))
        if child_tokens - parent_tokens:
            continue
        issues.append({
            "kind": "child_adds_no_specific_term",
            "parent_id": parent.get("node_id"),
            "parent_label": parent.get("label"),
            "child_id": child.get("node_id"),
            "child_label": child.get("label"),
        })
    for level in ("macros", "clusters", "sections"):
        for node in hierarchy.get(level, []):
            if not content_focused_parent_label(str(node.get("label", ""))):
                issues.append({
                    "kind": "parent_describes_editing_mechanics",
                    "node_id": node.get("node_id"),
                    "label": node.get("label"),
                })
    for edge in hierarchy.get("edges", []):
        label = _normalise_term(str(edge.get("label", "")))
        description = _normalise_term(
            str(edge.get("detailed_description", ""))
        )
        label_key = _term_key(label)
        macro_label_keys = {
            _term_key(str(macro.get("label", "")))
            for macro in hierarchy.get("macros", [])
            if macro.get("label")
        }
        if (
            label_key.endswith(" status")
            or label_key == "un observer state"
            or label_key in macro_label_keys
        ):
            issues.append({
                "kind": "edge_repeats_broad_topic",
                "node_id": edge.get("node_id"),
                "label": label,
            })
        if not description or len(description.split()) <= len(label.split()):
            issues.append({
                "kind": "edge_description_not_more_detailed",
                "node_id": edge.get("node_id"),
                "label": label,
            })
    return issues


def _term_key(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()


def deduplicate_label_rows(label_rows: list[dict]) -> list[dict]:
    """Remove duplicate parents and duplicate labels on coincident edges.

    Priority is deliberately compared before importance: a broad parent label
    must beat another parent label with the same text. Coincident edges keep
    only the most explanatory label, ranked first by the number of added and
    removed words. Spatially distinct edges must retain their own labels even
    when their short text is identical; otherwise a label elsewhere on the map
    can incorrectly erase the description of an open, clearly visible edge.
    Semantic importance, revision coverage, and finally the stable node id
    provide deterministic fallbacks.
    """
    def text_key(value: str | list[str] | None) -> str:
        if isinstance(value, list):
            value = " ".join(str(line) for line in value)
        return _term_key(str(value or ""))

    def rendered_keys(row: dict) -> tuple[str, str]:
        parent_text = row.get("display_label") or row.get("label")
        if row.get("level") != "edge":
            key = text_key(parent_text)
            return key, key
        return (
            text_key(row.get("label") or parent_text),
            text_key(row.get("display_label") or row.get("label")),
        )

    def edge_geometry_key(row: dict) -> tuple | None:
        if row.get("level") != "edge":
            return None
        coordinate_fields = ("edge_x", "edge_y", "edge_x2", "edge_y2")
        if any(row.get(field) is None for field in coordinate_fields):
            return None
        endpoints = tuple(sorted((
            (
                round(float(row["edge_x"]), 12),
                round(float(row["edge_y"]), 12),
            ),
            (
                round(float(row["edge_x2"]), 12),
                round(float(row["edge_y2"]), 12),
            ),
        )))
        return endpoints

    def rank(row: dict) -> tuple[int, int, int, int, float, float, int]:
        description = _normalise_term(row.get("detailed_description", ""))
        return (
            int(row.get("label_priority", 0)),
            int(bool(description)),
            int(row.get("change_magnitude") or 0),
            len(_term_key(description).split()),
            float(row.get("label_confidence") or 0.0),
            float(row.get("importance", 0.0)),
            int(row.get("revision_count") or 0),
        )

    ordered = sorted(
        enumerate(label_rows),
        key=lambda item: (
            -rank(item[1])[0],
            -rank(item[1])[1],
            -rank(item[1])[2],
            -rank(item[1])[3],
            -rank(item[1])[4],
            -rank(item[1])[5],
            -rank(item[1])[6],
            str(item[1].get("node_id", "")),
        ),
    )
    accepted: set[int] = set()
    used_parent_short: set[str] = set()
    used_parent_detailed: set[str] = set()
    used_geometryless_edge_short: set[str] = set()
    used_geometryless_edge_detailed: set[str] = set()
    used_edge_geometries: set[tuple] = set()
    for index, row in ordered:
        short_key, detailed_key = rendered_keys(row)
        geometry_key = edge_geometry_key(row)
        if row.get("level") == "edge":
            if geometry_key is not None:
                if geometry_key in used_edge_geometries:
                    continue
            elif (
                (short_key and short_key in used_geometryless_edge_short)
                or (
                    detailed_key
                    and detailed_key in used_geometryless_edge_detailed
                )
            ):
                continue
        elif (
            (short_key and short_key in used_parent_short)
            or (detailed_key and detailed_key in used_parent_detailed)
        ):
            continue
        accepted.add(index)
        if row.get("level") == "edge":
            if geometry_key is not None:
                used_edge_geometries.add(geometry_key)
            else:
                if short_key:
                    used_geometryless_edge_short.add(short_key)
                if detailed_key:
                    used_geometryless_edge_detailed.add(detailed_key)
        else:
            if short_key:
                used_parent_short.add(short_key)
            if detailed_key:
                used_parent_detailed.add(detailed_key)
    return [row for index, row in enumerate(label_rows) if index in accepted]


def top_terms(keyword_groups: Iterable[Iterable[str]], limit: int = 12) -> list[str]:
    """Return frequent grounded terms while preserving their best spelling."""
    counts: Counter[str] = Counter()
    spelling: dict[str, str] = {}
    for group in keyword_groups:
        seen = set()
        for raw in group:
            term = _normalise_term(raw)
            key = _term_key(term)
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            spelling.setdefault(key, term)
    return [spelling[key] for key, _ in counts.most_common(limit)]


def fallback_label(terms: list[str], default: str) -> str:
    """Use the single most relevant grounded term without combining phrases."""
    return terms[0] if terms else default


def prioritised_terms(
    primary: str,
    keyword_groups: Iterable[Iterable[str]],
    limit: int,
) -> list[str]:
    """Keep the matching LOD label first, then add frequent grounded evidence."""
    ranked = top_terms(keyword_groups, limit)
    if not primary:
        return ranked
    primary = _normalise_term(primary)
    primary_key = _term_key(primary)
    return [primary] + [
        term for term in ranked if _term_key(term) != primary_key
    ][:max(0, limit - 1)]


def edge_phrase(terms: Iterable[str], max_words: int = 4) -> str:
    """Return one intact grounded phrase suitable for a trajectory edge.

    Edge labels are atomic: words from different keyword candidates are never
    joined, and a long candidate is never truncated because either operation
    could change its meaning. If no complete candidate fits, the edge remains
    unlabeled.
    """
    for term in terms:
        phrase = _normalise_term(term)
        if phrase and 1 <= len(phrase.split()) <= max_words:
            return phrase
    return ""


NON_TOPICAL_EDGE_LABEL_KEYS = {
    "identical article text",
    "no cleaned text change",
    "no content change",
    "no content difference",
    "no substantive difference",
}


def parent_topic_label(label: str) -> str:
    """Exclude neutral transitions from higher-level thematic aggregation."""
    return "" if _term_key(label) in NON_TOPICAL_EDGE_LABEL_KEYS else label


def difference_lod_label(pair: dict, level: int) -> str:
    """Return one validated per-transition LOD label when present."""
    for item in pair.get("difference_lod", []):
        if int(item.get("level", -1)) == level:
            return edge_phrase([item.get("label", "")])
    return ""


def validate_sources(projection: dict, keywords: dict) -> tuple[list[dict], list[dict]]:
    points = projection.get("points")
    pairs = keywords.get("revision_pairs")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("Projektionsdatei enthält weniger als zwei Punkte.")
    if not isinstance(pairs, list) or len(pairs) != len(points) - 1:
        raise ValueError("Diff-Label-Datei muss genau ein Paar pro Trajektorienkante enthalten.")
    for index, pair in enumerate(pairs):
        if (
            str(pair.get("previous_revision_id")) != str(points[index]["revision_id"])
            or str(pair.get("current_revision_id")) != str(points[index + 1]["revision_id"])
        ):
            raise ValueError(f"Diff-Label-Paar {index + 1} passt nicht zur Projektion.")
    return points, pairs


def chronological_runs(points: list[dict]) -> list[dict]:
    """Return maximal chronological runs of one HDBSCAN cluster."""
    runs = []
    start = 0
    while start < len(points):
        cluster_id = int(points[start]["cluster_id"])
        end = start + 1
        while end < len(points) and int(points[end]["cluster_id"]) == cluster_id:
            end += 1
        runs.append({"cluster_id": cluster_id, "start": start, "end": end})
        start = end
    return runs


def _mean_xy(points: list[dict], indices: Iterable[int]) -> tuple[float, float]:
    selected = [points[index]["mds"] for index in indices]
    return (
        sum(float(item[0]) for item in selected) / len(selected),
        sum(float(item[1]) for item in selected) / len(selected),
    )


def spatial_macro_groups(
    clusters: list[dict], target_count: int | None = None
) -> list[list[int]]:
    """Agglomerate adjacent cluster centroids into deterministic overview groups."""
    if not clusters:
        return []
    if target_count is None:
        target_count = min(
            MAX_MACRO_GROUPS,
            max(1, round(math.sqrt(len(clusters)))),
        )
    target_count = max(1, min(int(target_count), len(clusters)))
    groups = [
        {
            "cluster_ids": [int(cluster["cluster_id"])],
            "x": float(cluster["x"]),
            "y": float(cluster["y"]),
            "weight": max(1, int(cluster.get("revision_count", 1))),
        }
        for cluster in clusters
    ]
    while len(groups) > target_count:
        closest = min(
            (
                (
                    (groups[first]["x"] - groups[second]["x"]) ** 2
                    + (groups[first]["y"] - groups[second]["y"]) ** 2,
                    tuple(groups[first]["cluster_ids"]),
                    tuple(groups[second]["cluster_ids"]),
                    first,
                    second,
                )
                for first in range(len(groups))
                for second in range(first + 1, len(groups))
            ),
            key=lambda item: item[:3],
        )
        first_index, second_index = closest[-2:]
        first = groups[first_index]
        second = groups[second_index]
        weight = first["weight"] + second["weight"]
        merged = {
            "cluster_ids": sorted(first["cluster_ids"] + second["cluster_ids"]),
            "x": (first["x"] * first["weight"] + second["x"] * second["weight"]) / weight,
            "y": (first["y"] * first["weight"] + second["y"] * second["weight"]) / weight,
            "weight": weight,
        }
        groups.pop(second_index)
        groups.pop(first_index)
        groups.append(merged)
        groups.sort(key=lambda group: tuple(group["cluster_ids"]))
    return [group["cluster_ids"] for group in groups]


def _counts_from_xml(xml_path: Path, points: list[dict]) -> list[tuple[int, int]]:
    """Compute true added/removed word counts with the configured Git word diff."""
    revision_ids = [str(point["revision_id"]) for point in points]
    revisions = list(select_visualized_revisions(
        iter_revisions(xml_path, revision_ids=revision_ids), revision_ids
    ))
    counts = []
    for previous, current in zip(revisions, revisions[1:]):
        old_words = previous["text"].split()
        new_words = current["text"].split()
        opcodes = diff_opcodes(
            old_words, new_words, "git", DEFAULT_GIT_DIFF_ALGORITHM
        )
        removed = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag in {"delete", "replace"})
        added = sum(j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in {"insert", "replace"})
        counts.append((added, removed))
    return counts


def deterministic_word_change_type(
    added_words: int | None,
    removed_words: int | None,
) -> str:
    """Classify one incoming transition from audited word-diff counts.

    The four substantive outcomes are mutually exclusive and exhaustive when
    both counts are available. ``unavailable`` is retained for lightweight
    renders that deliberately skip word-diff computation.
    """
    if added_words is None or removed_words is None:
        return "unavailable"
    if added_words > 0 and removed_words == 0:
        return "added"
    if added_words == 0 and removed_words > 0:
        return "removed"
    if added_words > 0 and removed_words > 0:
        return "mixed"
    return "no_change"


def cached_word_counts(
    labels_path: Path,
    projection_path: Path,
    keywords_path: Path,
    pair_count: int,
) -> list[tuple[int, int]] | None:
    """Reuse audited Git counts from an earlier hierarchy build."""
    if not labels_path.exists():
        return None
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    if (
        Path(payload.get("projection", "")) != projection_path
        or Path(payload.get("revision_keywords", "")) != keywords_path
        or not payload.get("word_counts_are_git_word_diff")
    ):
        return None
    edges = payload.get("hierarchy", {}).get("edges", [])
    if len(edges) != pair_count:
        return None
    result = []
    pattern = re.compile(r"^\+(\d+)/−(\d+) (?:words|Wörter)$")
    for edge in edges:
        match = pattern.match(edge.get("count_text", ""))
        if not match:
            return None
        result.append((int(match.group(1)), int(match.group(2))))
    return result


def build_hierarchy(
    projection: dict,
    keywords: dict,
    word_counts: list[tuple[int, int]] | None = None,
    root_label: str = "Article",
) -> dict:
    """Create root, macro, cluster, chronological-section, and edge candidates."""
    points, pairs = validate_sources(projection, keywords)
    if word_counts is not None and len(word_counts) != len(pairs):
        raise ValueError("Die Zahl der Wortzählungen passt nicht zu den Paaren.")

    edges = []
    for index, pair in enumerate(pairs):
        previous = points[index]
        current = points[index + 1]
        x = (float(previous["mds"][0]) + float(current["mds"][0])) / 2
        y = (float(previous["mds"][1]) + float(current["mds"][1])) / 2
        direct_label = edge_phrase([pair.get("edge_label", "")])
        thematic_label = parent_topic_label(direct_label)
        lod_labels = {
            level: difference_lod_label(pair, level)
            for level in range(1, 6)
        }
        terms = top_terms([
            [thematic_label] if thematic_label else [],
            [lod_labels[5]] if lod_labels[5] else [],
            pair.get("evidence", []),
            pair.get("focus_keywords", []),
            pair.get("added_keywords", []),
            pair.get("removed_keywords", []),
        ], 6)
        added, removed = word_counts[index] if word_counts else (None, None)
        count_text = f"+{added}/−{removed} words" if added is not None else ""
        word_change_type = deterministic_word_change_type(added, removed)
        # A trajectory edge carries exactly one intact grounded phrase. A
        # phrase may contain up to four words (for example "UN General
        # Assembly"), but separate candidates are never combined.
        if added == 0 and removed == 0:
            label = "No cleaned-text change"
        else:
            label = direct_label or edge_phrase(terms)
        if not label:
            label = (
                "Unspecified text change"
            )
        edges.append({
            "node_id": f"edge-{index + 1}",
            "level": "edge",
            "x": x,
            "y": y,
            "label": label,
            "short_label": label,
            "detailed_description": _normalise_term(
                pair.get("detailed_description", "")
            ),
            "terms": terms,
            "difference_lod": pair.get("difference_lod", []),
            "section_terms": prioritised_terms(lod_labels[4], [
                [lod_labels[5]] if lod_labels[5] else [],
                [thematic_label] if thematic_label else [],
                pair.get("focus_keywords", []),
                pair.get("evidence", []),
            ], 6),
            "cluster_terms": prioritised_terms(lod_labels[3], [
                [lod_labels[4]] if lod_labels[4] else [],
                [thematic_label] if thematic_label else [],
                pair.get("focus_keywords", []),
            ], 6),
            "macro_terms": prioritised_terms(lod_labels[2], [
                [lod_labels[1]] if lod_labels[1] else [],
                [thematic_label] if thematic_label else [],
                pair.get("focus_keywords", []),
            ], 6),
            "evidence": pair.get("evidence", []),
            "change_type": pair.get("change_type", "legacy_keyword_fallback"),
            "word_change_type": word_change_type,
            "label_confidence": pair.get("confidence"),
            "label_source": (
                "full_revision_lod_llm" if direct_label
                else "legacy_keyword_fallback"
            ),
            "count_text": count_text,
            "added_words": added,
            "removed_words": removed,
            "pair_index": index,
            "previous_revision_id": str(pair["previous_revision_id"]),
            "current_revision_id": str(pair["current_revision_id"]),
            "importance": max(
                float(current.get("semantic_jump_from_previous", 0.0)), 1e-9
            ),
        })

    sections = []
    edge_to_section: dict[int, str] = {}
    for run_number, run in enumerate(chronological_runs(points), start=1):
        # HDBSCAN -1 denotes unassigned noise, not a shared semantic cluster.
        # Consecutive noise revisions therefore remain individual leaf edges
        # instead of being grouped into a potentially misleading section.
        if int(run["cluster_id"]) < 0:
            continue
        # A section needs at least two edges. Shorter runs remain leaf-only.
        edge_indices = list(range(run["start"], run["end"] - 1))
        if len(edge_indices) < 2:
            continue
        node_id = f"section-{run_number}"
        x, y = _mean_xy(points, range(run["start"], run["end"]))
        terms = top_terms(
            (edges[index]["section_terms"] for index in edge_indices), 12
        )
        if not terms:
            continue
        sections.append({
            "node_id": node_id,
            "level": "section",
            "cluster_id": run["cluster_id"],
            "x": x,
            "y": y,
            "label": fallback_label(terms, "Local development"),
            "terms": terms,
            "descriptions": [
                edges[index]["detailed_description"] for index in edge_indices
                if edges[index]["detailed_description"]
            ],
            "edge_indices": edge_indices,
            "revision_count": len(edge_indices) + 1,
            "importance": len(edge_indices) * (
                1 + sum(edges[index]["importance"] for index in edge_indices)
            ),
        })
        for index in edge_indices:
            edge_to_section[index] = node_id

    clusters = []
    clustered_indices: dict[int, list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        cluster_id = int(point["cluster_id"])
        if cluster_id >= 0:
            clustered_indices[cluster_id].append(index)
    for cluster_id, indices in sorted(clustered_indices.items()):
        node_id = f"cluster-{cluster_id}"
        related_edges = [
            index for index in range(len(edges))
            if int(points[index + 1]["cluster_id"]) == cluster_id
        ]
        x, y = _mean_xy(points, indices)
        terms = top_terms(
            (edges[index]["cluster_terms"] for index in related_edges), 16
        )
        # A dense cluster made entirely from byte-identical cleaned revisions
        # has no article-content topic to name. Keep it unlabeled instead of
        # turning the article title or "No Content Change" into a misleading
        # thematic heading.
        macro_terms = top_terms(
            (edges[index]["macro_terms"] for index in related_edges), 16
        )
        clusters.append({
            "node_id": node_id,
            "level": "cluster",
            "cluster_id": cluster_id,
            "hdbscan_label": f"Cluster {cluster_id + 1}",
            "x": x,
            "y": y,
            "label": fallback_label(terms, ""),
            "terms": terms,
            "descriptions": [
                edges[index]["detailed_description"] for index in related_edges
                if edges[index]["detailed_description"]
            ],
            "macro_terms": macro_terms or terms,
            "point_indices": indices,
            "revision_count": len(indices),
            "importance": len(indices) * (
                1 + sum(edges[index]["importance"] for index in related_edges)
            ),
        })

    for index, edge in enumerate(edges):
        edge["parent_id"] = edge_to_section.get(index)
        if edge["parent_id"] is None:
            cluster_id = int(points[index + 1]["cluster_id"])
            edge["parent_id"] = f"cluster-{cluster_id}" if cluster_id >= 0 else None
    for section in sections:
        cluster_id = int(section["cluster_id"])
        section["parent_id"] = f"cluster-{cluster_id}" if cluster_id >= 0 else "root"

    root_x, root_y = _mean_xy(points, range(len(points)))
    roots = [{
        "node_id": "root",
        "level": "root",
        "x": root_x,
        "y": root_y,
        "label": root_label,
        "terms": [root_label],
        "revision_count": len(points),
        "importance": float(len(points)),
        "parent_id": None,
    }]
    macros = []
    cluster_by_id = {int(cluster["cluster_id"]): cluster for cluster in clusters}
    for macro_number, cluster_ids in enumerate(spatial_macro_groups(clusters), start=1):
        children = [cluster_by_id[cluster_id] for cluster_id in cluster_ids]
        revision_count = sum(int(child["revision_count"]) for child in children)
        x = sum(
            float(child["x"]) * int(child["revision_count"]) for child in children
        ) / revision_count
        y = sum(
            float(child["y"]) * int(child["revision_count"]) for child in children
        ) / revision_count
        terms = top_terms((child["macro_terms"] for child in children), 20)
        macro_id = f"macro-{macro_number}"
        macros.append({
            "node_id": macro_id,
            "level": "macro",
            "cluster_ids": cluster_ids,
            "x": x,
            "y": y,
            "label": fallback_label(terms, root_label),
            "terms": terms,
            "descriptions": [
                description
                for child in children
                for description in child.get("descriptions", [])
            ],
            "child_labels": [],
            "revision_count": revision_count,
            "importance": sum(float(child["importance"]) for child in children),
            "parent_id": "root",
        })
        for child in children:
            child["parent_id"] = macro_id
    for edge in edges:
        if edge["parent_id"] is None:
            edge["parent_id"] = "root"

    return {
        "roots": roots,
        "macros": macros,
        "clusters": clusters,
        "sections": sections,
        "edges": edges,
    }


def build_unlabelled_hierarchy(
    projection: dict,
    root_label: str = "Article",
) -> dict:
    """Build the final map hierarchy without cached or generated text labels.

    Spatial macro groups, HDBSCAN clusters, and chronological sections remain
    available to the renderer. Only the article title and deterministic legend
    names are retained as text.
    """
    points = projection.get("points", [])
    revision_pairs = [
        {
            "previous_revision_id": str(previous["revision_id"]),
            "current_revision_id": str(current["revision_id"]),
            # A temporary structural term lets build_hierarchy retain every
            # otherwise valid chronological section. It is removed before
            # rendering and is never presented as a generated content label.
            "edge_label": "Revision transition",
            "detailed_description": "",
            "difference_lod": [],
            "focus_keywords": [],
            "added_keywords": [],
            "removed_keywords": [],
            "evidence": [],
            "change_type": "unlabelled_base_visualization",
            "confidence": None,
        }
        for previous, current in zip(points, points[1:])
    ]
    hierarchy = build_hierarchy(
        projection,
        {"revision_pairs": revision_pairs},
        root_label=root_label,
    )
    for index, macro in enumerate(hierarchy["macros"], start=1):
        macro["legend_label"] = f"Macro group {index}"
    for section in hierarchy["sections"]:
        section["show_highlight"] = True
    for level in ("macros", "clusters", "sections", "edges"):
        for node in hierarchy[level]:
            node["label"] = ""
            if level == "edges":
                node["short_label"] = ""
                node["detailed_description"] = ""
    return hierarchy


def format_label_pyramid(hierarchy: dict) -> str:
    """Format a compact, continuous Root-to-Edge hierarchy for the terminal."""
    levels = ("macros", "clusters", "sections", "edges")
    nodes = [node for level in levels for node in hierarchy.get(level, [])]
    children: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        parent_id = node.get("parent_id")
        if parent_id:
            children[str(parent_id)].append(node)
    for siblings in children.values():
        siblings.sort(
            key=lambda node: (-float(node.get("importance", 0.0)), node["node_id"])
        )

    def counts_below(node_id: str) -> dict[str, int]:
        counts = {"macro": 0, "cluster": 0, "section": 0, "edge": 0}
        pending = list(children.get(node_id, []))
        while pending:
            node = pending.pop()
            level = node.get("level")
            if level in counts:
                counts[level] += 1
            pending.extend(children.get(str(node["node_id"]), []))
        return counts

    def best_child(node_id: str, level: str) -> dict | None:
        return next(
            (
                node for node in children.get(node_id, [])
                if node.get("level") == level and node.get("label")
            ),
            None,
        )

    roots = hierarchy.get("roots", [])
    root = roots[0] if roots else {"node_id": "root", "label": "Article"}
    total = counts_below(str(root["node_id"]))
    lines = [
        "",
        "Label hierarchy",
        "Scale: article → macro topic → topic cluster → temporal section → revision edge",
        "Map analogy: country → state → region → place → street",
        (
            f"{root.get('label') or 'Article'}/  "
            f"[{total['macro']} Macros | {total['cluster']} Cluster | "
            f"{total['section']} Sections | {total['edge']} Edges]"
        ),
    ]

    root_children = children.get(str(root["node_id"]), [])
    macros = [
        node for node in root_children
        if node.get("level") == "macro"
    ]
    direct_root_children = [
        node for node in root_children
        if node.get("level") != "macro"
    ]
    for macro_index, macro in enumerate(macros):
        macro_last = macro_index == len(macros) - 1 and not direct_root_children
        macro_branch = "└── " if macro_last else "├── "
        macro_indent = "    " if macro_last else "│   "
        counts = counts_below(str(macro["node_id"]))
        lines.append(
            f"{macro_branch}{macro['label']}  "
            f"[{counts['cluster']} Cluster | {counts['section']} Sections | "
            f"{counts['edge']} Edges]"
        )

        cluster = best_child(str(macro["node_id"]), "cluster")
        if not cluster:
            continue
        lines.append(f"{macro_indent}└── {cluster['label']}")
        section = best_child(str(cluster["node_id"]), "section")
        if section:
            lines.append(f"{macro_indent}    └── {section['label']}")
            edge = best_child(str(section["node_id"]), "edge")
            if edge:
                lines.append(f"{macro_indent}        └── {edge['label']}")
        else:
            edge = best_child(str(cluster["node_id"]), "edge")
            if edge:
                lines.append(f"{macro_indent}    └── {edge['label']}")

    if direct_root_children:
        direct_sections = sum(
            1 for node in direct_root_children if node.get("level") == "section"
        )
        direct_edges = sum(
            1 for node in direct_root_children if node.get("level") == "edge"
        ) + sum(
            counts_below(str(node["node_id"]))["edge"]
            for node in direct_root_children if node.get("level") == "section"
        )
        lines.append(
            "└── Directly below root / not assigned to a macro topic  "
            f"[{direct_sections} Sections | {direct_edges} Edges]"
        )
    return "\n".join(lines)


def generate_labels(
    client: Any,
    model: str,
    nodes: list[dict],
    batch_size: int = 16,
    system_prompt: str = LABEL_SYSTEM_PROMPT,
) -> dict[str, str]:
    """Generate hierarchy labels in offline batches and validate node IDs."""
    generated: dict[str, str] = {}
    for start in range(0, len(nodes), batch_size):
        batch = nodes[start:start + batch_size]
        evidence = [
            {
                "node_id": node["node_id"],
                "kind": node["level"],
                "revision_count": node.get("revision_count"),
                "grounded_child_evidence": node["terms"],
                "grounded_detailed_descriptions": node.get("descriptions", []),
                "child_cluster_labels": node.get("child_labels", []),
            }
            for node in batch
        ]
        prompt = (
            "Create one umbrella label for every node below. More abstract cluster labels "
            "must summarize the supplied evidence; section labels should be narrower. "
            "Use the keywords and child labels from that node's own area only. Name the "
            "underlying article subject, claim, institution, place, event, or policy; "
            "editing mechanics are evidence and must never become the parent label. Return "
            "every node_id exactly once and one English label per node.\n\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2)
        )
        for attempt in range(3):
            response = client.responses.parse(
                model=model,
                store=False,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                text_format=NodeLabelBatch,
            )
            if response.output_parsed is None:
                raise RuntimeError(
                    f"Keine auswertbaren Hierarchielabels: {response.output_text}"
                )
            expected = {node["node_id"] for node in batch}
            actual = {item.node_id for item in response.output_parsed.labels}
            if actual != expected:
                raise RuntimeError(
                    "LLM-Label-IDs stimmen nicht mit dem Batch überein: "
                    f"fehlend={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )
            labels = {
                item.node_id: _normalise_term(item.label)
                for item in response.output_parsed.labels
            }
            invalid = {
                node_id: label
                for node_id, label in labels.items()
                if (
                    not valid_parent_label(label)
                    or not content_focused_parent_label(label)
                )
            }
            if not invalid:
                generated.update(labels)
                break
            prompt += (
                "\n\nCorrection required: every label must describe article content, "
                "must not describe editing mechanics, and must contain exactly "
                f"{MIN_PARENT_LABEL_WORDS} to {MAX_PARENT_LABEL_WORDS} words. "
                "Return the complete batch again with every node_id exactly once. "
                "Invalid labels: "
                + json.dumps(invalid, ensure_ascii=False, sort_keys=True)
            )
        else:
            raise RuntimeError(
                "Das Modell lieferte nach drei Versuchen keine ausschließlich "
                "2-bis-3-Wort-Hierarchielabels."
            )
    return generated


def cached_llm_labels(
    labels_path: Path,
    projection_path: Path,
    keywords_path: Path,
    model: str,
    provider: str = LLM_PROVIDER,
) -> dict[str, str]:
    """Reuse already generated offline labels for unchanged hierarchy nodes."""
    if not labels_path.exists():
        return {}
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    fixed_labels = fixed_parent_label_sources_match(
        payload, projection_path, keywords_path
    )
    if (
        Path(payload.get("projection", "")) != projection_path
        or Path(payload.get("revision_keywords", "")) != keywords_path
        or (
            not fixed_labels
            and (
                payload.get("llm_provider") != provider
                or payload.get("model") != model
                or payload.get("label_system_prompt") != LABEL_SYSTEM_PROMPT
            )
        )
    ):
        return {}
    result = {}
    hierarchy = payload.get("hierarchy", {})
    for level in ("clusters", "sections"):
        for node in hierarchy.get(level, []):
            if (
                node.get("node_id")
                and node.get("label")
                and valid_parent_label(str(node["label"]))
                and (
                    fixed_labels
                    or content_focused_parent_label(str(node["label"]))
                )
            ):
                result[node["node_id"]] = _normalise_term(str(node["label"]))
    return result


def cached_macro_labels(
    labels_path: Path,
    projection_path: Path,
    keywords_path: Path,
    model: str,
    macros: list[dict],
    provider: str = LLM_PROVIDER,
) -> dict[str, str]:
    """Reuse macro labels, preserving a reviewed fixed label pyramid by node id."""
    if not labels_path.exists():
        return {}
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    fixed_labels = fixed_parent_label_sources_match(
        payload, projection_path, keywords_path
    )
    if (
        Path(payload.get("projection", "")) != projection_path
        or Path(payload.get("revision_keywords", "")) != keywords_path
        or (
            not fixed_labels
            and (
                payload.get("llm_provider") != provider
                or payload.get("model") != model
                or payload.get("macro_label_system_prompt")
                != MACRO_LABEL_SYSTEM_PROMPT
            )
        )
    ):
        return {}
    expected = {
        node["node_id"]: {
            "cluster_ids": tuple(node["cluster_ids"]),
            "terms": tuple(node.get("terms", [])),
            "child_labels": tuple(node.get("child_labels", [])),
        }
        for node in macros
    }
    result = {}
    for node in payload.get("hierarchy", {}).get("macros", []):
        node_id = node.get("node_id")
        if (
            node_id in expected
            and (
                fixed_labels
                or (
                    tuple(node.get("cluster_ids", []))
                    == expected[node_id]["cluster_ids"]
                    and tuple(node.get("terms", [])) == expected[node_id]["terms"]
                    and tuple(node.get("child_labels", []))
                    == expected[node_id]["child_labels"]
                )
            )
            and node.get("label")
            and valid_parent_label(str(node["label"]))
            and (
                fixed_labels
                or content_focused_parent_label(str(node["label"]))
            )
        ):
            result[node_id] = _normalise_term(str(node["label"]))
    return result


def _finite_extent(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    padding = max((high - low) * 0.06, 1e-9)
    return [low - padding, high + padding]


def catmull_rom_bezier_controls(
    points: list[tuple[float, float]],
    *,
    alpha: float = 0.5,
) -> list[tuple[float, float, float, float]]:
    """Return cubic Bezier controls for an open Catmull-Rom point chain.

    Each result describes the segment from ``points[i]`` to ``points[i+1]``
    as ``(c1x, c1y, c2x, c2y)``.  The centripetal parameterisation
    (``alpha=0.5``) is deliberately used for the highly non-uniform distances
    in a 2D semantic projection: it is less prone to loops and overshoot than
    the uniform variant while still interpolating every revision point.

    Exact reverts can place consecutive revisions at identical coordinates.
    Those segments remain zero-length; neighbouring tangent handles are
    bounded to half the segment length so a duplicate cannot create a spike.
    """
    if len(points) < 2:
        return []
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Catmull-Rom alpha must be between 0 and 1.")

    def reflected(point: tuple[float, float], neighbour: tuple[float, float]):
        return (2.0 * point[0] - neighbour[0], 2.0 * point[1] - neighbour[1])

    def distance(first: tuple[float, float], second: tuple[float, float]):
        return math.hypot(second[0] - first[0], second[1] - first[1])

    def knot_delta(first: tuple[float, float], second: tuple[float, float]):
        # A tiny positive interval keeps the non-uniform tangent formula
        # defined when two exact-revert points coincide.
        return max(distance(first, second) ** alpha, 1e-12)

    def bounded(vector: tuple[float, float], maximum: float):
        length = math.hypot(*vector)
        if length <= maximum or length <= 1e-18:
            return vector
        scale = maximum / length
        return (vector[0] * scale, vector[1] * scale)

    controls = []
    for index in range(len(points) - 1):
        p1, p2 = points[index], points[index + 1]
        if distance(p1, p2) <= 1e-18:
            controls.append((p1[0], p1[1], p2[0], p2[1]))
            continue
        p0 = points[index - 1] if index > 0 else reflected(p1, p2)
        p3 = (
            points[index + 2]
            if index + 2 < len(points)
            else reflected(p2, p1)
        )
        dt01 = knot_delta(p0, p1)
        dt12 = knot_delta(p1, p2)
        dt23 = knot_delta(p2, p3)
        dt02 = dt01 + dt12
        dt13 = dt12 + dt23

        tangent1 = tuple(
            dt12 * (
                (p1[axis] - p0[axis]) / dt01
                - (p2[axis] - p0[axis]) / dt02
                + (p2[axis] - p1[axis]) / dt12
            )
            for axis in (0, 1)
        )
        tangent2 = tuple(
            dt12 * (
                (p2[axis] - p1[axis]) / dt12
                - (p3[axis] - p1[axis]) / dt13
                + (p3[axis] - p2[axis]) / dt23
            )
            for axis in (0, 1)
        )
        # The Bezier handle is one third of the Hermite tangent. Limiting the
        # handle itself to half the chord protects sparse/duplicate sequences
        # without moving either interpolated revision point.
        maximum_handle = distance(p1, p2) * 0.5
        handle1 = bounded(
            (tangent1[0] / 3.0, tangent1[1] / 3.0), maximum_handle
        )
        handle2 = bounded(
            (tangent2[0] / 3.0, tangent2[1] / 3.0), maximum_handle
        )
        controls.append((
            p1[0] + handle1[0],
            p1[1] + handle1[1],
            p2[0] - handle2[0],
            p2[1] - handle2[1],
        ))
    return controls


def convex_hull(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute a deterministic monotonic-chain hull without extra dependencies."""
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def cluster_region_rows(
    point_rows: list[dict],
    expansion: float = 1.08,
) -> tuple[list[dict], list[dict]]:
    """Create visible regions around every non-noise HDBSCAN cluster.

    A regular two-dimensional cluster receives an expanded convex polygon.
    Collinear or effectively coincident projected points cannot enclose a
    visible polygon, so their distinct projected positions are connected by a
    thin line instead. Rounding only decides whether a cluster is
    degenerate; it removes meaningless floating-point jitter between
    revisions that project to the same visible position.
    """
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for point in point_rows:
        if point["cluster_id"] >= 0:
            grouped[point["cluster_id"]].append((point["x"], point["y"]))

    polygons: list[dict] = []
    corridors: list[dict] = []
    for cluster_id, coordinates in sorted(grouped.items()):
        effective_coordinates = sorted({
            (round(float(x), 12), round(float(y), 12))
            for x, y in coordinates
        })
        effective_hull = convex_hull(effective_coordinates)
        if len(effective_hull) < 3:
            if len(effective_hull) == 2:
                start, end = effective_hull
                corridors.append({
                    "cluster_id": cluster_id,
                    "order": 0,
                    "x": start[0],
                    "y": start[1],
                    "x2": end[0],
                    "y2": end[1],
                })
            continue

        hull = convex_hull(coordinates)
        center_x = sum(x for x, _ in hull) / len(hull)
        center_y = sum(y for _, y in hull) / len(hull)
        expanded = [
            (
                center_x + (x - center_x) * expansion,
                center_y + (y - center_y) * expansion,
            )
            for x, y in hull
        ]
        for order, (x, y) in enumerate(expanded + expanded[:1]):
            polygons.append({
                "cluster_id": cluster_id,
                "order": order,
                "x": x,
                "y": y,
            })
    return polygons, corridors


def build_revert_arc_rows(
    point_rows: list[dict], revert_analysis: dict | None
) -> list[dict]:
    """Map revert events to styled screen-space arcs between plotted revisions.

    An exact SHA1 revert runs from the reverted revision to the reverting
    revision.  If the same visual return connection occurs repeatedly in
    either direction, near-identical endpoints are treated as one pair and
    only its chronologically last event is drawn.  Its retained orientation
    therefore always points towards that final revert data point.
    """
    if not revert_analysis:
        return []

    points_by_revision = {
        str(point["revision_id"]): point for point in point_rows
    }
    coordinate_counts = Counter(
        (
            round(float(point["x"]), 12),
            round(float(point["y"]), 12),
        )
        for point in point_rows
    )
    rows = []
    for event_order, event in enumerate(revert_analysis.get("events", [])):
        source_id = str(event.get("revision_id", ""))
        target_id = str(event.get("reverted_revision_id", ""))
        revert_point = points_by_revision.get(source_id)
        undone_point = points_by_revision.get(target_id)
        if (
            revert_point is None
            or undone_point is None
            or source_id == target_id
        ):
            continue
        reverter = str(event.get("reverter", ""))
        reverted_editor = str(event.get("reverted_editor", ""))
        revert_kind = str(event.get("revert_kind", "exact_sha1"))
        if revert_kind != "exact_sha1":
            raise ValueError(
                f"Unsupported revert category: {revert_kind!r}"
            )
        restored_id = str(event.get("restored_revision_id") or "")
        pair = [str(editor) for editor in event.get("pair", []) if editor]
        start_revision_id = target_id
        end_revision_id = source_id
        segment_start = points_by_revision[start_revision_id]
        segment_end = points_by_revision[end_revision_id]
        source_key = (
            round(float(segment_start["x"]), 12),
            round(float(segment_start["y"]), 12),
        )
        target_key = (
            round(float(segment_end["x"]), 12),
            round(float(segment_end["y"]), 12),
        )
        rows.append({
            "arc_id": f"revert-{event_order + 1}-{source_id}-{target_id}",
            "event_order": int(event.get("revision_index", event_order)),
            "segment_order": 0,
            "x": segment_start["x"],
            "y": segment_start["y"],
            "x2": segment_end["x"],
            "y2": segment_end["y"],
            "revision_id": source_id,
            "start_revision_id": start_revision_id,
            "end_revision_id": end_revision_id,
            "reverted_revision_id": target_id,
            "restored_revision_id": restored_id,
            "return_anchor_revision_id": end_revision_id,
            "source_overlap_count": coordinate_counts[source_key],
            "target_overlap_count": coordinate_counts[target_key],
            "reverter": reverter,
            "reverted_editor": reverted_editor,
            "pair_text": " ↔ ".join(pair),
            "revert_kind": revert_kind,
            "revert_kind_label": "Exact revert",
            "visual_category": revert_kind,
            "visual_category_label": "Exact revert",
            "reversal_score": float(event.get("reversal_score", 1.0)),
        })

    def endpoint_match_score(left: dict, right: dict) -> float | None:
        """Return the best undirected endpoint error for one visual arc pair."""
        left_length = math.hypot(
            float(left["x2"]) - float(left["x"]),
            float(left["y2"]) - float(left["y"]),
        )
        right_length = math.hypot(
            float(right["x2"]) - float(right["x"]),
            float(right["y2"]) - float(right["y"]),
        )
        direct_error = max(
            math.hypot(
                float(left["x"]) - float(right["x"]),
                float(left["y"]) - float(right["y"]),
            ),
            math.hypot(
                float(left["x2"]) - float(right["x2"]),
                float(left["y2"]) - float(right["y2"]),
            ),
        )
        reverse_error = max(
            math.hypot(
                float(left["x"]) - float(right["x2"]),
                float(left["y"]) - float(right["y2"]),
            ),
            math.hypot(
                float(left["x2"]) - float(right["x"]),
                float(left["y2"]) - float(right["y"]),
            ),
        )
        best_error = min(direct_error, reverse_error)
        tolerance = (
            min(left_length, right_length)
            * REVERT_CONNECTION_ENDPOINT_TOLERANCE_RATIO
        )
        if tolerance <= 1e-15:
            return best_error if best_error <= 1e-12 else None
        return best_error if best_error <= tolerance else None

    grouped: list[list[dict]] = []
    for row in sorted(rows, key=lambda item: (item["event_order"], item["arc_id"])):
        matches = [
            (score, group_index)
            for group_index, group in enumerate(grouped)
            if (score := endpoint_match_score(row, group[-1])) is not None
        ]
        if matches:
            _, group_index = min(matches)
            grouped[group_index].append(row)
        else:
            grouped.append([row])

    retained_rows = []
    for group in grouped:
        row = group[-1]
        row["arc_rank"] = 0
        row["arc_level"] = 0
        row["arc_count"] = 1
        row["revert_occurrence_count"] = len(group)
        row["suppressed_revert_arc_count"] = len(group) - 1
        row["arc_direction"] = 1
        row["arc_side"] = 1
        retained_rows.append(row)
    return sorted(
        retained_rows,
        key=lambda row: (row["event_order"], row["arc_id"]),
    )


def snap_exact_revert_positions(
    point_rows: list[dict], revert_analysis: dict | None
) -> None:
    """Place exact revert revisions on the restored revision coordinate.

    Identical SHA1 states are semantically identical even when an embedding or
    projection round-trip leaves their stored coordinates a few pixels apart.
    Snapping exact reverts makes the visual return literally coincide with the
    restored state.
    """
    if not revert_analysis:
        return
    points_by_revision = {
        str(point["revision_id"]): point for point in point_rows
    }
    for event in revert_analysis.get("events", []):
        if event.get("revert_kind") != "exact_sha1":
            continue
        revert_point = points_by_revision.get(str(event.get("revision_id", "")))
        restored_point = points_by_revision.get(
            str(event.get("restored_revision_id", ""))
        )
        if revert_point is None or restored_point is None:
            continue
        revert_point["projected_x_before_revert_snap"] = revert_point["x"]
        revert_point["projected_y_before_revert_snap"] = revert_point["y"]
        revert_point["x"] = restored_point["x"]
        revert_point["y"] = restored_point["y"]
        revert_point["exact_revert_restored_revision_id"] = str(
            event["restored_revision_id"]
        )


def build_vega_spec(
    article: str,
    projection: dict,
    hierarchy: dict,
    revert_analysis: dict | None = None,
) -> dict:
    revert_analysis = revert_analysis or projection.get("revert_analysis")
    points = [dict(point) for point in projection["points"]]
    point_statistics = (
        revert_analysis.get("point_statistics", [])
        if revert_analysis else []
    )
    if point_statistics and len(point_statistics) != len(points):
        raise ValueError(
            "Revert-Statistiken passen nicht zu den projizierten Punkten: "
            f"{len(point_statistics)} statt {len(points)}."
        )
    for point, statistics in zip(points, point_statistics):
        point.update(statistics)
    incoming_word_change_types = ["start", *(
        str(edge.get("word_change_type", "unavailable"))
        for edge in hierarchy["edges"]
    )]
    if len(incoming_word_change_types) != len(points):
        raise ValueError(
            "Änderungstypen passen nicht zu den projizierten Punkten: "
            f"{len(incoming_word_change_types)} statt {len(points)}."
        )
    point_rows = [
        {
            "x": float(point["mds"][0]),
            "y": float(point["mds"][1]),
            "order": int(point["revision_index"]),
            "revision_id": str(point["revision_id"]),
            "timestamp": point["timestamp"],
            "author": point["author"],
            "words": int(point["clean_word_count"]),
            "cluster_id": int(point["cluster_id"]),
            "cluster_label": (
                "Noise"
                if int(point["cluster_id"]) == -1
                else f"Cluster {int(point['cluster_id']) + 1}"
            ),
            "cluster_probability": float(point.get("cluster_probability", 0.0)),
            "word_change_type": incoming_word_change_types[point_index],
            "word_change_label": WORD_CHANGE_LABELS.get(
                incoming_word_change_types[point_index],
                WORD_CHANGE_LABELS["unavailable"],
            ),
            "semantic_jump": float(point["semantic_jump_from_previous"]),
            "interval_revert_count": int(point.get("interval_revert_count", 0)),
            "interval_exact_sha1_revert_count": int(
                point.get("interval_exact_sha1_revert_count", 0)
            ),
        }
        for point_index, point in enumerate(points)
    ]
    snap_exact_revert_positions(point_rows, revert_analysis)
    hull_rows, cluster_corridor_rows = cluster_region_rows(point_rows)
    revert_arc_rows = build_revert_arc_rows(point_rows, revert_analysis)

    spline_controls = catmull_rom_bezier_controls([
        (float(point["x"]), float(point["y"])) for point in point_rows
    ])
    edge_rows = []
    for index, edge in enumerate(hierarchy["edges"]):
        a, b = point_rows[index], point_rows[index + 1]
        c1x, c1y, c2x, c2y = spline_controls[index]
        semantic_jump = b["semantic_jump"]
        edge_rows.append({
            "edge": index,
            "x": a["x"],
            "y": a["y"],
            "x2": b["x"],
            "y2": b["y"],
            "spline_c1x": c1x,
            "spline_c1y": c1y,
            "spline_c2x": c2x,
            "spline_c2y": c2y,
            "label": edge["label"],
            "change_type": edge.get("change_type", "legacy_keyword_fallback"),
            "evidence_text": " · ".join(edge.get("evidence", [])),
            "label_confidence": edge.get("label_confidence"),
            "label_source": edge.get("label_source"),
            "previous_revision_id": edge["previous_revision_id"],
            "current_revision_id": edge["current_revision_id"],
            "semantic_jump": semantic_jump,
            "revert_count": b["interval_revert_count"],
            "exact_revert_count": b["interval_exact_sha1_revert_count"],
            "has_exact_revert": b["interval_exact_sha1_revert_count"] > 0,
            "cluster_id": b["cluster_id"],
            "in_section": str(edge.get("parent_id", "")).startswith("section-"),
            "same_cluster": (
                a["cluster_id"] >= 0 and a["cluster_id"] == b["cluster_id"]
            ),
        })

    def undirected_geometry_key(edge_row: dict) -> tuple:
        """Group visually identical A→B and B→A projected segments."""
        return tuple(sorted((
            (round(float(edge_row["x"]), 12), round(float(edge_row["y"]), 12)),
            (round(float(edge_row["x2"]), 12), round(float(edge_row["y2"]), 12)),
        )))

    geometry_counts = Counter(
        undirected_geometry_key(edge_row) for edge_row in edge_rows
    )
    for edge_row in edge_rows:
        edge_row["overlap_count"] = geometry_counts[
            undirected_geometry_key(edge_row)
        ]

    xs = [row["x"] for row in point_rows]
    ys = [row["y"] for row in point_rows]
    x_extent = _finite_extent(xs)
    y_extent = _finite_extent(ys)
    hierarchy_nodes = (
        hierarchy.get("macros", []) + hierarchy["clusters"]
        + hierarchy["sections"] + hierarchy["edges"]
    )
    node_by_id = {
        str(node["node_id"]): node for node in hierarchy_nodes
    }
    root_node = hierarchy["roots"][0]

    def fixed_parent_placement(node: dict) -> dict:
        """Anchor hierarchy areas at their data centre; offset endpoints."""
        level = str(node["level"])
        if level in {"root", "macro", "cluster", "section"}:
            return {
                "label_dx": 0,
                "label_dy": 0,
                "label_align": "center",
                "label_baseline": "middle",
            }
        parent = node_by_id.get(str(node.get("parent_id", "")), root_node)
        # Start and End remain point-feature labels and point towards the map
        # centre so they do not drift outside the plotted trajectory.
        dx = float(parent["x"]) - float(node["x"])
        dy = float(parent["y"]) - float(node["y"])
        offset = 14
        if abs(dx) < 1e-18 and abs(dy) < 1e-18:
            # A single-child parent can share the exact anchor and wording of
            # its child. Give the child enough separation for both labels to
            # pass the screen-space collision check when the area is open.
            offset += 10
            direction = sum(
                (index + 1) * ord(character)
                for index, character in enumerate(str(node["node_id"]))
            ) % 4
            dx, dy = ((1, 0), (0, 1), (-1, 0), (0, -1))[direction]
        if abs(dx) >= abs(dy):
            right = dx >= 0
            return {
                "label_dx": offset if right else -offset,
                "label_dy": 0,
                "label_align": "left" if right else "right",
                "label_baseline": "middle",
            }
        below = dy >= 0
        return {
            "label_dx": 0,
            "label_dy": offset if below else -offset,
            "label_align": "center",
            "label_baseline": "top" if below else "bottom",
        }

    def label_ancestor_ids(node: dict) -> list[str]:
        ancestors: list[str] = []
        parent_id = str(node.get("parent_id") or "")
        while parent_id:
            ancestors.append(parent_id)
            parent = node_by_id.get(parent_id)
            if parent is None:
                break
            parent_id = str(parent.get("parent_id") or "")
        return ancestors

    def is_uninformative_label(node: dict) -> bool:
        """Hide labels that explicitly report the absence of a content edit."""
        return _term_key(node.get("label", "")) in {
            "no content change",
            "unchanged content",
            "content unchanged",
        }

    def node_support_point_indices(node: dict) -> list[int]:
        """Return the revision points whose area a hierarchy label describes."""
        level = str(node.get("level", ""))
        if level == "macro":
            cluster_ids = {int(value) for value in node.get("cluster_ids", [])}
            return [
                index for index, point in enumerate(point_rows)
                if int(point["cluster_id"]) in cluster_ids
            ]
        if level == "cluster":
            return sorted({int(index) for index in node.get("point_indices", [])})
        if level == "section":
            point_indices: set[int] = set()
            for edge_index in node.get("edge_indices", []):
                point_indices.update((int(edge_index), int(edge_index) + 1))
            return sorted(
                index for index in point_indices if 0 <= index < len(point_rows)
            )
        return []

    def edge_display_text(node: dict) -> str | list[str]:
        """Render detailed edge descriptions as readable multiline labels."""
        description = _normalise_term(node.get("detailed_description", ""))
        if node.get("level") != "edge" or not description:
            return node["label"]
        description = compact_display_timestamps(description)
        return textwrap.wrap(
            description,
            width=46,
            break_long_words=False,
            break_on_hyphens=False,
        )

    detail_labels = [
        node for node in hierarchy_nodes
        if node["label"]
        and not node.get("suppress_map_label", False)
        and not is_uninformative_label(node)
    ]
    label_rows = [
        {
            "node_id": node["node_id"],
            "level": node["level"],
            "x": node["x"],
            "y": node["y"],
            "label": node["label"],
            "display_label": edge_display_text(node),
            "detailed_description": node.get("detailed_description", ""),
            "cluster_id": node.get("cluster_id"),
            "label_colour": "#25322a",
            # Every level competes for screen space at every zoom scale. Broad
            # labels win collisions first, and increasingly detailed labels
            # fill every remaining gap as zooming creates room. The reviewed
            # semantic centre remains the origin for a bounded set of
            # screen-space variable-anchor alternatives.
            "label_priority": (
                {
                    "macro": 4,
                    "cluster": 3,
                    "section": 2,
                    "edge": 1,
                }[node["level"]]
            ),
            "importance": node["importance"],
            "change_magnitude": (
                int(node.get("added_words") or 0)
                + int(node.get("removed_words") or 0)
                if node["level"] == "edge" else 0
            ),
            "added_words": node.get("added_words"),
            "removed_words": node.get("removed_words"),
            "label_confidence": node.get("label_confidence"),
            "revision_count": node.get("revision_count"),
            "parent_id": node.get("parent_id"),
            "ancestor_ids": label_ancestor_ids(node),
            "support_point_indices": node_support_point_indices(node),
            **fixed_parent_placement(node),
            "edge_x": (
                point_rows[node["pair_index"]]["x"]
                if node["level"] == "edge" else None
            ),
            "edge_y": (
                point_rows[node["pair_index"]]["y"]
                if node["level"] == "edge" else None
            ),
            "edge_x2": (
                point_rows[node["pair_index"] + 1]["x"]
                if node["level"] == "edge" else None
            ),
            "edge_y2": (
                point_rows[node["pair_index"] + 1]["y"]
                if node["level"] == "edge" else None
            ),
            "edge_c1x": (
                spline_controls[node["pair_index"]][0]
                if node["level"] == "edge" else None
            ),
            "edge_c1y": (
                spline_controls[node["pair_index"]][1]
                if node["level"] == "edge" else None
            ),
            "edge_c2x": (
                spline_controls[node["pair_index"]][2]
                if node["level"] == "edge" else None
            ),
            "edge_c2y": (
                spline_controls[node["pair_index"]][3]
                if node["level"] == "edge" else None
            ),
        }
        for node in detail_labels
    ]
    for row in label_rows:
        display_label = row["display_label"]
        row["display_line_count"] = (
            len(display_label) if isinstance(display_label, list) else 1
        )
    label_rows.extend([
        {
            "node_id": "root",
            "level": "root",
            "x": hierarchy["roots"][0]["x"],
            "y": hierarchy["roots"][0]["y"],
            "label": hierarchy["roots"][0]["label"],
            "display_label": hierarchy["roots"][0]["label"],
            "cluster_id": None,
            "label_colour": "#25322a",
            # The article is the final overview label. Once the geometry has
            # collapsed at extreme zoom-out, it must win over Start, End, and
            # every thematic detail occupying the same screen-space point.
            "label_priority": 7,
            "importance": hierarchy["roots"][0]["importance"],
            "revision_count": hierarchy["roots"][0]["revision_count"],
            "parent_id": None,
            "ancestor_ids": [],
            "support_point_indices": list(range(len(point_rows))),
            **fixed_parent_placement(hierarchy["roots"][0]),
        },
        {
            "node_id": "trajectory-start",
            "level": "endpoint",
            "x": point_rows[0]["x"],
            "y": point_rows[0]["y"],
            "label": "Start",
            "display_label": "Start",
            "cluster_id": point_rows[0]["cluster_id"],
            "label_colour": "#25322a",
            "label_priority": 6,
            "importance": 1e12,
            "revision_count": 1,
            "parent_id": "root",
            "ancestor_ids": ["root"],
            "support_point_indices": [0],
            **fixed_parent_placement({
                "node_id": "trajectory-start",
                "level": "endpoint",
                "x": point_rows[0]["x"],
                "y": point_rows[0]["y"],
                "parent_id": "root",
            }),
        },
        {
            "node_id": "trajectory-end",
            "level": "endpoint",
            "x": point_rows[-1]["x"],
            "y": point_rows[-1]["y"],
            "label": "End",
            "display_label": "End",
            "cluster_id": point_rows[-1]["cluster_id"],
            "label_colour": "#25322a",
            "label_priority": 6,
            "importance": 1e12,
            "revision_count": 1,
            "parent_id": "root",
            "ancestor_ids": ["root"],
            "support_point_indices": [len(point_rows) - 1],
            **fixed_parent_placement({
                "node_id": "trajectory-end",
                "level": "endpoint",
                "x": point_rows[-1]["x"],
                "y": point_rows[-1]["y"],
                "parent_id": "root",
            }),
        },
    ])
    label_rows = deduplicate_label_rows(label_rows)
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "description": f"Semantic zoom revision timecurve for {article}",
        "autosize": "none",
        "width": 1180,
        "height": 760,
        "padding": {"top": 18, "left": 28, "right": 28, "bottom": 28},
        "background": BACKGROUND_COLOR,
        "signals": [
            {"name": "xext", "value": x_extent},
            {"name": "yext", "value": y_extent},
            {"name": "xdom", "update": "xext", "on": [
                {"events": {"signal": "delta"}, "update": "[xcur[0] + span(xcur) * delta[0] / width, xcur[1] + span(xcur) * delta[0] / width]"},
                {"events": {"signal": "zoom"}, "update": "[anchor[0] + (xdom[0] - anchor[0]) * zoom, anchor[0] + (xdom[1] - anchor[0]) * zoom]"},
                {"events": "dblclick", "update": "xext"},
            ]},
            {"name": "ydom", "update": "yext", "on": [
                {"events": {"signal": "delta"}, "update": "[ycur[0] + span(ycur) * delta[1] / height, ycur[1] + span(ycur) * delta[1] / height]"},
                {"events": {"signal": "zoom"}, "update": "[anchor[1] + (ydom[0] - anchor[1]) * zoom, anchor[1] + (ydom[1] - anchor[1]) * zoom]"},
                {"events": "dblclick", "update": "yext"},
            ]},
            {"name": "down", "value": None, "on": [
                {"events": "pointerdown", "update": "xy()"},
                {"events": "window:pointerup", "update": "null"},
            ]},
            {"name": "xcur", "value": None, "on": [
                {"events": "pointerdown", "update": "slice(xdom)"},
            ]},
            {"name": "ycur", "value": None, "on": [
                {"events": "pointerdown", "update": "slice(ydom)"},
            ]},
            {"name": "delta", "value": [0, 0], "on": [
                {"events": [{"source": "window", "type": "pointermove", "consume": True, "between": [{"type": "pointerdown"}, {"source": "window", "type": "pointerup"}]}],
                 "update": "down ? [down[0]-x(), y()-down[1]] : [0,0]"},
            ]},
            {"name": "anchor", "value": [0, 0], "on": [
                {"events": "wheel", "update": "[invert('xscale', x()), invert('yscale', y())]"},
            ]},
            {"name": "minViewScale", "value": MIN_VIEW_SCALE},
            {"name": "maxViewScale", "value": MAX_VIEW_SCALE},
            {"name": "viewScale", "update": "span(xext)/span(xdom)"},
            {
                "name": "rootOnlyOverview",
                "update": "viewScale <= minViewScale * 1.001",
            },
            {"name": "zoom", "value": 1, "on": [
                {
                    "events": "wheel!",
                    "force": True,
                    "update": (
                        "min(max(pow(1.001, event.deltaY * "
                        "pow(16, event.deltaMode)), viewScale/maxViewScale), "
                        "viewScale/minViewScale)"
                    ),
                },
            ]},
            {"name": "zooming", "value": False, "on": [
                {"events": "wheel", "update": "true"},
                {"events": {"type": "wheel", "debounce": 140}, "update": "false"},
            ]},
            {"name": "zoomLevel", "update": "max(0, log((xext[1]-xext[0])/(xdom[1]-xdom[0]))/log(2))"},
            {
                "name": "semanticTargetDepth",
                "update": (
                    f"min(5, 1 + zoomLevel/{SEMANTIC_DEPTH_ZOOM_STEP})"
                ),
            },
            # Filled by a post-render screen-space bounds audit. Hierarchy
            # labels use MapLibre-style variable anchors around their semantic
            # centre; edge labels slide along the visible spline.
            {"name": "blockedParentLabelIds", "value": []},
            {"name": "blockedEdgeLabelIds", "value": []},
            {"name": "parentLabelPlacementIds", "value": []},
            {"name": "parentLabelPlacementDx", "value": []},
            {"name": "parentLabelPlacementDy", "value": []},
            # The browser-side free-space search stores the selected Bezier
            # sample and normal side for every accepted edge label here.
            {"name": "edgeLabelPlacementIds", "value": []},
            {"name": "edgeLabelPlacementTs", "value": []},
            {"name": "edgeLabelPlacementSides", "value": []},
            # Filled by the screen-space collision audit. Detailed edge text
            # is preferred only when its bounds and trajectory length fit;
            # otherwise the short label is tried.
            {"name": "detailedEdgeLabelIds", "value": []},
        ],
        "data": [
            {"name": "points", "values": point_rows},
            {"name": "normalRevisionPoints", "source": "points"},
            {"name": "edges", "values": edge_rows},
            {"name": "exactRevertEdges", "source": "edges", "transform": [
                {"type": "filter", "expr": "datum.has_exact_revert"},
            ]},
            {"name": "splineEdgeGeometry", "source": "edges", "transform": [
                {"type": "formula", "as": "screenX1", "expr": "scale('xscale', datum.x)"},
                {"type": "formula", "as": "screenY1", "expr": "scale('yscale', datum.y)"},
                {"type": "formula", "as": "screenC1X", "expr": "scale('xscale', datum.spline_c1x)"},
                {"type": "formula", "as": "screenC1Y", "expr": "scale('yscale', datum.spline_c1y)"},
                {"type": "formula", "as": "screenC2X", "expr": "scale('xscale', datum.spline_c2x)"},
                {"type": "formula", "as": "screenC2Y", "expr": "scale('yscale', datum.spline_c2y)"},
                {"type": "formula", "as": "screenX2", "expr": "scale('xscale', datum.x2)"},
                {"type": "formula", "as": "screenY2", "expr": "scale('yscale', datum.y2)"},
                {
                    "type": "formula",
                    "as": "splinePath",
                    "expr": (
                        "'M'+datum.screenX1+','+datum.screenY1+' C'+"
                        "datum.screenC1X+','+datum.screenC1Y+' '+"
                        "datum.screenC2X+','+datum.screenC2Y+' '+"
                        "datum.screenX2+','+datum.screenY2"
                    ),
                },
                {
                    "type": "formula",
                    "as": "splineLength",
                    "expr": (
                        "sqrt(pow(datum.screenC1X-datum.screenX1,2)+"
                        "pow(datum.screenC1Y-datum.screenY1,2))+"
                        "sqrt(pow(datum.screenC2X-datum.screenC1X,2)+"
                        "pow(datum.screenC2Y-datum.screenC1Y,2))+"
                        "sqrt(pow(datum.screenX2-datum.screenC2X,2)+"
                        "pow(datum.screenY2-datum.screenC2Y,2))"
                    ),
                },
                {
                    "type": "formula",
                    "as": "splineMidX",
                    "expr": "(datum.screenX1+3*datum.screenC1X+3*datum.screenC2X+datum.screenX2)/8",
                },
                {
                    "type": "formula",
                    "as": "splineMidY",
                    "expr": "(datum.screenY1+3*datum.screenC1Y+3*datum.screenC2Y+datum.screenY2)/8",
                },
                {
                    "type": "formula",
                    "as": "splineTangentX",
                    "expr": "0.75*(datum.screenC1X-datum.screenX1)+1.5*(datum.screenC2X-datum.screenC1X)+0.75*(datum.screenX2-datum.screenC2X)",
                },
                {
                    "type": "formula",
                    "as": "splineTangentY",
                    "expr": "0.75*(datum.screenC1Y-datum.screenY1)+1.5*(datum.screenC2Y-datum.screenC1Y)+0.75*(datum.screenY2-datum.screenC2Y)",
                },
            ]},
            {"name": "visibleDirectionArrows", "source": "splineEdgeGeometry", "transform": [
                {"type": "filter", "expr": "!datum.has_exact_revert"},
                {"type": "filter", "expr": "datum.splineLength>=34"},
                {"type": "filter", "expr": "datum.splineMidX>=0 && datum.splineMidX<=width && datum.splineMidY>=0 && datum.splineMidY<=height"},
                {
                    "type": "formula",
                    "as": "arrowScreenX",
                    "expr": "datum.splineMidX",
                },
                {"type": "formula", "as": "arrowScreenY", "expr": "datum.splineMidY"},
                {"type": "formula", "as": "arrowAngle", "expr": "atan2(datum.splineTangentY,datum.splineTangentX)*57.29577951308232"},
            ]},
            {"name": "revertArcs", "values": revert_arc_rows},
            {"name": "revertArcGeometry", "source": "revertArcs", "transform": [
                {"type": "formula", "as": "screenX1", "expr": "scale('xscale', datum.x)"},
                {"type": "formula", "as": "screenY1", "expr": "scale('yscale', datum.y)"},
                {"type": "formula", "as": "screenX2", "expr": "scale('xscale', datum.x2)"},
                {"type": "formula", "as": "screenY2", "expr": "scale('yscale', datum.y2)"},
                {"type": "formula", "as": "screenDx", "expr": "datum.screenX2-datum.screenX1"},
                {"type": "formula", "as": "screenDy", "expr": "datum.screenY2-datum.screenY1"},
                {
                    "type": "formula",
                    "as": "screenLength",
                    "expr": "sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2))",
                },
                {"type": "filter", "expr": "datum.screenLength>=2"},
                {
                    "type": "formula",
                    "as": "arcHeight",
                    "expr": (
                        "min(190,max(22,datum.screenLength*0.28+"
                        "8*datum.arc_level))"
                    ),
                },
                {
                    "type": "formula",
                    "as": "controlX",
                    "expr": (
                        "(datum.screenX1+datum.screenX2)/2 - "
                        "datum.screenDy/datum.screenLength*datum.arcHeight*"
                        "datum.arc_direction*datum.arc_side"
                    ),
                },
                {
                    "type": "formula",
                    "as": "controlY",
                    "expr": (
                        "(datum.screenY1+datum.screenY2)/2 + "
                        "datum.screenDx/datum.screenLength*datum.arcHeight*"
                        "datum.arc_direction*datum.arc_side"
                    ),
                },
                {
                    "type": "formula",
                    "as": "arcPath",
                    "expr": (
                        "'M'+datum.screenX1+','+datum.screenY1+' Q'+"
                        "datum.controlX+','+datum.controlY+' '+"
                        "datum.screenX2+','+datum.screenY2"
                    ),
                },
                {
                    "type": "formula",
                    "as": "arrowX",
                    "expr": (
                        "0.1024*datum.screenX1 + 0.4352*datum.controlX + "
                        "0.4624*datum.screenX2"
                    ),
                },
                {
                    "type": "formula",
                    "as": "arrowY",
                    "expr": (
                        "0.1024*datum.screenY1 + 0.4352*datum.controlY + "
                        "0.4624*datum.screenY2"
                    ),
                },
                {
                    "type": "formula",
                    "as": "arrowDx",
                    "expr": (
                        "0.64*(datum.controlX-datum.screenX1) + "
                        "1.36*(datum.screenX2-datum.controlX)"
                    ),
                },
                {
                    "type": "formula",
                    "as": "arrowDy",
                    "expr": (
                        "0.64*(datum.controlY-datum.screenY1) + "
                        "1.36*(datum.screenY2-datum.controlY)"
                    ),
                },
                {
                    "type": "formula",
                    "as": "arrowAngle",
                    "expr": "atan2(datum.arrowDy,datum.arrowDx)*57.29577951308232",
                },
            ]},
            {"name": "intraClusterEdges", "source": "splineEdgeGeometry", "transform": [
                {"type": "filter", "expr": "datum.same_cluster"},
            ]},
            {"name": "labels", "values": label_rows},
            {"name": "clusterHulls", "values": hull_rows},
            {"name": "clusterCorridors", "values": cluster_corridor_rows},
            {"name": "visibleParentLabels", "source": "labels", "transform": [
                {
                    "type": "filter",
                    "expr": (
                        "rootOnlyOverview ? datum.level==='root' : "
                        "datum.level!=='edge'"
                    ),
                },
                {"type": "filter", "expr": "datum.x >= min(xdom[0], xdom[1]) && datum.x <= max(xdom[0], xdom[1]) && datum.y >= min(ydom[0], ydom[1]) && datum.y <= max(ydom[0], ydom[1])"},
                {"type": "filter", "expr": "datum.level!=='root' || (scale('xscale', datum.x) >= 95 && scale('xscale', datum.x) <= width-95 && scale('yscale', datum.y) >= 24 && scale('yscale', datum.y) <= height-24)"},
                {"type": "formula", "as": "parentPlacementIndex", "expr": "indexof(parentLabelPlacementIds, datum.node_id)"},
                {"type": "formula", "as": "parentPlacementDx", "expr": "datum.parentPlacementIndex>=0 ? parentLabelPlacementDx[datum.parentPlacementIndex] : 0"},
                {"type": "formula", "as": "parentPlacementDy", "expr": "datum.parentPlacementIndex>=0 ? parentLabelPlacementDy[datum.parentPlacementIndex] : 0"},
                {"type": "formula", "as": "placementOrder", "expr": "-(datum.label_priority*1e15 + datum.importance)"},
                {"type": "collect", "sort": {"field": "placementOrder", "order": "ascending"}},
            ]},
            {"name": "visibleEdgeLabels", "source": "labels", "transform": [
                {"type": "filter", "expr": "datum.level==='edge'"},
                {
                    "type": "filter",
                    "expr": "!rootOnlyOverview",
                },
                {"type": "formula", "as": "screenX1", "expr": "scale('xscale', datum.edge_x)"},
                {"type": "formula", "as": "screenY1", "expr": "scale('yscale', datum.edge_y)"},
                {"type": "formula", "as": "screenC1X", "expr": "scale('xscale', datum.edge_c1x)"},
                {"type": "formula", "as": "screenC1Y", "expr": "scale('yscale', datum.edge_c1y)"},
                {"type": "formula", "as": "screenC2X", "expr": "scale('xscale', datum.edge_c2x)"},
                {"type": "formula", "as": "screenC2Y", "expr": "scale('yscale', datum.edge_c2y)"},
                {"type": "formula", "as": "screenX2", "expr": "scale('xscale', datum.edge_x2)"},
                {"type": "formula", "as": "screenY2", "expr": "scale('yscale', datum.edge_y2)"},
                {"type": "formula", "as": "screenDx", "expr": "datum.screenX2-datum.screenX1"},
                {"type": "formula", "as": "screenDy", "expr": "datum.screenY2-datum.screenY1"},
                {
                    "type": "formula",
                    "as": "screenLength",
                    "expr": "sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2))",
                },
                {
                    "type": "formula",
                    "as": "xClipMin",
                    "expr": (
                        "abs(datum.screenDx)<1e-9 ? "
                        "(datum.screenX1>=0 && datum.screenX1<=width ? -1e9 : 1) : "
                        "min((0-datum.screenX1)/datum.screenDx,"
                        "(width-datum.screenX1)/datum.screenDx)"
                    ),
                },
                {
                    "type": "formula",
                    "as": "xClipMax",
                    "expr": (
                        "abs(datum.screenDx)<1e-9 ? "
                        "(datum.screenX1>=0 && datum.screenX1<=width ? 1e9 : 0) : "
                        "max((0-datum.screenX1)/datum.screenDx,"
                        "(width-datum.screenX1)/datum.screenDx)"
                    ),
                },
                {
                    "type": "formula",
                    "as": "yClipMin",
                    "expr": (
                        "abs(datum.screenDy)<1e-9 ? "
                        "(datum.screenY1>=0 && datum.screenY1<=height ? -1e9 : 1) : "
                        "min((0-datum.screenY1)/datum.screenDy,"
                        "(height-datum.screenY1)/datum.screenDy)"
                    ),
                },
                {
                    "type": "formula",
                    "as": "yClipMax",
                    "expr": (
                        "abs(datum.screenDy)<1e-9 ? "
                        "(datum.screenY1>=0 && datum.screenY1<=height ? 1e9 : 0) : "
                        "max((0-datum.screenY1)/datum.screenDy,"
                        "(height-datum.screenY1)/datum.screenDy)"
                    ),
                },
                {"type": "formula", "as": "clipT0", "expr": "max(0,datum.xClipMin,datum.yClipMin)"},
                {"type": "formula", "as": "clipT1", "expr": "min(1,datum.xClipMax,datum.yClipMax)"},
                {"type": "filter", "expr": "datum.clipT0<=datum.clipT1"},
                {
                    "type": "formula",
                    "as": "labelPlacementIndex",
                    "expr": "indexof(edgeLabelPlacementIds, datum.node_id)",
                },
                {
                    "type": "formula",
                    "as": "labelT",
                    "expr": (
                        "datum.labelPlacementIndex>=0 ? "
                        "edgeLabelPlacementTs[datum.labelPlacementIndex] : "
                        "(datum.clipT0+datum.clipT1)/2"
                    ),
                },
                {
                    "type": "formula",
                    "as": "labelSide",
                    "expr": (
                        "datum.labelPlacementIndex>=0 ? "
                        "edgeLabelPlacementSides[datum.labelPlacementIndex] : 1"
                    ),
                },
                {
                    "type": "formula",
                    "as": "labelScreenX",
                    "expr": "pow(1-datum.labelT,3)*datum.screenX1+3*pow(1-datum.labelT,2)*datum.labelT*datum.screenC1X+3*(1-datum.labelT)*pow(datum.labelT,2)*datum.screenC2X+pow(datum.labelT,3)*datum.screenX2",
                },
                {
                    "type": "formula",
                    "as": "labelScreenY",
                    "expr": "pow(1-datum.labelT,3)*datum.screenY1+3*pow(1-datum.labelT,2)*datum.labelT*datum.screenC1Y+3*(1-datum.labelT)*pow(datum.labelT,2)*datum.screenC2Y+pow(datum.labelT,3)*datum.screenY2",
                },
                {
                    "type": "formula",
                    "as": "labelTangentX",
                    "expr": "3*pow(1-datum.labelT,2)*(datum.screenC1X-datum.screenX1)+6*(1-datum.labelT)*datum.labelT*(datum.screenC2X-datum.screenC1X)+3*pow(datum.labelT,2)*(datum.screenX2-datum.screenC2X)",
                },
                {
                    "type": "formula",
                    "as": "labelTangentY",
                    "expr": "3*pow(1-datum.labelT,2)*(datum.screenC1Y-datum.screenY1)+6*(1-datum.labelT)*datum.labelT*(datum.screenC2Y-datum.screenC1Y)+3*pow(datum.labelT,2)*(datum.screenY2-datum.screenC2Y)",
                },
                {
                    "type": "formula",
                    "as": "labelTangentLength",
                    "expr": "sqrt(pow(datum.labelTangentX,2)+pow(datum.labelTangentY,2))",
                },
                {
                    "type": "formula",
                    "as": "labelUnitX",
                    "expr": (
                        "datum.labelSide*(datum.labelTangentLength>=1 ? "
                        "-datum.labelTangentY/datum.labelTangentLength : "
                        "(abs(datum.label_dx)+abs(datum.label_dy)>0 ? "
                        "datum.label_dx/max(sqrt(pow(datum.label_dx,2)+"
                        "pow(datum.label_dy,2)),1) : 1))"
                    ),
                },
                {
                    "type": "formula",
                    "as": "labelUnitY",
                    "expr": (
                        "datum.labelSide*(datum.labelTangentLength>=1 ? "
                        "datum.labelTangentX/datum.labelTangentLength : "
                        "(abs(datum.label_dx)+abs(datum.label_dy)>0 ? "
                        "datum.label_dy/max(sqrt(pow(datum.label_dx,2)+"
                        "pow(datum.label_dy,2)),1) : 0))"
                    ),
                },
                {"type": "formula", "as": "placementOrder", "expr": "-(datum.label_priority*1e15 + datum.importance)"},
                {"type": "collect", "sort": {"field": "placementOrder", "order": "ascending"}},
            ]},
        ],
        "scales": [
            {"name": "xscale", "zero": False, "domain": {"signal": "xdom"}, "range": "width"},
            {"name": "yscale", "zero": False, "domain": {"signal": "ydom"}, "range": "height"},
            {
                "name": "wordChangeColor",
                "type": "ordinal",
                "domain": list(WORD_CHANGE_COLOURS),
                "range": list(WORD_CHANGE_COLOURS.values()),
            },
        ],
        "marks": [
            {
                "name": "clusterRegions", "type": "group",
                "from": {"facet": {"name": "clusterHull", "data": "clusterHulls", "groupby": "cluster_id"}},
                "marks": [{
                    "name": "clusterRegionShape",
                    "type": "line", "from": {"data": "clusterHull"},
                    "encode": {"update": {
                        "x": {"scale": "xscale", "field": "x"},
                        "y": {"scale": "yscale", "field": "y"},
                        "order": {"field": "order"},
                        "interpolate": {"value": "linear"},
                        "fill": {"value": TRAJECTORY_COLOR},
                        "fillOpacity": {"value": 0.075},
                        "stroke": {"value": TRAJECTORY_COLOR},
                        "strokeOpacity": {"value": 0.38},
                        "strokeWidth": {"value": 1.4},
                    }}
                }],
            },
            {
                "name": "clusterRegionCorridor",
                "type": "rule",
                "from": {"data": "clusterCorridors"},
                "encode": {"update": {
                    "x": {"scale": "xscale", "field": "x"},
                    "y": {"scale": "yscale", "field": "y"},
                    "x2": {"scale": "xscale", "field": "x2"},
                    "y2": {"scale": "yscale", "field": "y2"},
                    "stroke": {"value": TRAJECTORY_COLOR},
                    "strokeOpacity": {"value": 0.42},
                    "strokeWidth": {"value": CLUSTER_CORRIDOR_STROKE_WIDTH},
                }},
            },
            {
                "name": "trajectory",
                "type": "path",
                "from": {"data": "splineEdgeGeometry"},
                "encode": {"update": {
                    "path": {"field": "splinePath"},
                    "fillOpacity": {"value": 0},
                    "stroke": {"value": TRAJECTORY_COLOR},
                    "strokeOpacity": {"value": 0.42},
                    "strokeWidth": {"signal": TRAJECTORY_WIDTH_SIGNAL},
                    "strokeCap": {"value": "round"},
                    "strokeJoin": {"value": "round"},
                }}
            },
            {
                "name": "intraClusterTrajectory",
                "type": "path", "from": {"data": "intraClusterEdges"},
                "encode": {"update": {
                    "path": {"field": "splinePath"},
                    "fillOpacity": {"value": 0},
                    "stroke": {"value": TRAJECTORY_COLOR},
                    "strokeOpacity": {"value": 0.52},
                    "strokeWidth": {"signal": TRAJECTORY_WIDTH_SIGNAL},
                    "strokeCap": {"value": "round"},
                    "strokeJoin": {"value": "round"},
                }}
            },
            {
                "name": "revertArcPaths",
                "type": "path",
                "from": {"data": "revertArcGeometry"},
                "encode": {
                    "enter": {
                        "tooltip": {
                            "signal": (
                                "{'Type': datum.visual_category_label, "
                                "'Revert method': datum.revert_kind_label, "
                                "'Start revision': datum.start_revision_id, "
                                "'End revision': datum.end_revision_id, "
                                "'Revert revision': datum.revision_id, "
                                "'Undone revision': datum.reverted_revision_id, "
                                "'Restored revision': datum.restored_revision_id, "
                                "'Return anchor': datum.return_anchor_revision_id, "
                                "'Points at return anchor': datum.target_overlap_count, "
                                "'Editors': datum.pair_text}"
                            )
                        },
                    },
                    "update": {
                        "path": {"field": "arcPath"},
                        "fillOpacity": {"value": 0},
                        "stroke": {"value": EXACT_REVERT_COLOR},
                        "strokeOpacity": {"value": REVERT_ARC_STROKE_OPACITY},
                        "strokeWidth": {"value": REVERT_ARC_STROKE_WIDTH},
                        "strokeDash": {"value": [1, 5]},
                        "strokeCap": {"value": "round"},
                        "strokeJoin": {"value": "round"},
                    },
                },
            },
            {
                "name": "revertArcDirectionArrows",
                "type": "symbol",
                "interactive": False,
                "from": {"data": "revertArcGeometry"},
                "zindex": 1,
                "encode": {"update": {
                    "x": {"field": "arrowX"},
                    "y": {"field": "arrowY"},
                    "shape": {"value": "triangle-right"},
                    "angle": {"field": "arrowAngle"},
                    "size": {"value": 46},
                    "fill": {"value": EXACT_REVERT_COLOR},
                    "fillOpacity": {"value": REVERT_ARROW_OPACITY},
                    "stroke": {"value": BACKGROUND_COLOR},
                    "strokeOpacity": {"value": REVERT_ARROW_OPACITY},
                    "strokeWidth": {"value": 0.8},
                }},
            },
            {
                "name": "directionArrows",
                "type": "symbol",
                "interactive": False,
                "from": {"data": "visibleDirectionArrows"},
                # Above the trajectory geometry, but below both text layers
                # (semanticLabels zindex 3 and edgeLabels zindex 7).
                "zindex": 2,
                "encode": {"update": {
                    "x": {"field": "arrowScreenX"},
                    "y": {"field": "arrowScreenY"},
                    "shape": {"value": "triangle-right"},
                    "angle": {"field": "arrowAngle"},
                    "size": {"value": 58},
                    "fill": {"value": TRAJECTORY_COLOR},
                    "fillOpacity": {"value": 0.9},
                    "stroke": {"value": BACKGROUND_COLOR},
                    "strokeOpacity": {"value": 0.95},
                    "strokeWidth": {"value": 0.8},
                }},
            },
            {
                "name": "revisionPoints", "type": "symbol", "from": {"data": "normalRevisionPoints"},
                "encode": {"enter": {
                    "tooltip": {
                        "signal": (
                            "{'Revision ID': datum.revision_id, "
                            "'Timestamp': datum.timestamp, "
                            "'Author': datum.author, "
                            "'Incoming text change': datum.word_change_label}"
                        )
                    }
                }, "update": {
                    "x": {"scale": "xscale", "field": "x"}, "y": {"scale": "yscale", "field": "y"},
                    "shape": {"value": "circle"},
                    "size": {"value": REVISION_POINT_SIZE},
                    "fill": {
                        "scale": "wordChangeColor",
                        "field": "word_change_type",
                    },
                    "fillOpacity": {
                        "signal": "0.9 + 0.1*min(zoomLevel/4,1)"
                    },
                    "strokeOpacity": {"value": 0},
                    "strokeWidth": {"value": 0},
                }}
            },
            {
                "name": "semanticLabels", "type": "text",
                "clip": True,
                "key": "node_id",
                "zindex": 3,
                "from": {"data": "visibleParentLabels"},
                "encode": {"update": {
                    "text": {"field": "display_label"},
                    "x": {"signal": "scale('xscale', datum.x) + datum.label_dx + datum.parentPlacementDx"},
                    "y": {"signal": "scale('yscale', datum.y) + datum.label_dy + datum.parentPlacementDy"},
                    "align": {"field": "label_align"},
                    "baseline": {"field": "label_baseline"},
                    "opacity": {"signal": "indexof(blockedParentLabelIds, datum.node_id)>=0 ? 0 : 1"},
                    "fillOpacity": {"value": 1},
                    "fill": {"field": "label_colour"},
                    "stroke": {"value": "#ffffff"},
                    "strokeOpacity": {"value": 1},
                    "strokeWidth": {"value": 4},
                    "font": {"value": "Inter, Arial, sans-serif"},
                    "fontSize": {"signal": "datum.level==='root' ? 28 : datum.level==='macro' ? 24 : datum.level==='cluster' ? 18 : datum.level==='section' ? 13 : 14"},
                    "fontWeight": {"value": 700},
                }}
            },
            {
                "name": "edgeShortMeasureLabels", "type": "text",
                "interactive": False,
                "key": "node_id",
                "from": {"data": "visibleEdgeLabels"},
                "encode": {"update": {
                    "text": {"field": "label"},
                    "x": {"signal": "datum.labelScreenX + 9*datum.labelUnitX"},
                    "y": {"signal": "datum.labelScreenY + 9*datum.labelUnitY"},
                    "align": {"value": "center"},
                    "baseline": {"value": "middle"},
                    "font": {"value": "Inter, Arial, sans-serif"},
                    "fontSize": {"value": EDGE_LABEL_FONT_SIZE},
                    "lineHeight": {"value": EDGE_LABEL_LINE_HEIGHT},
                    "fontWeight": {"value": 700},
                    "opacity": {"value": 0},
                    "fillOpacity": {"value": 0},
                    "strokeOpacity": {"value": 0},
                    "limit": {"value": 340},
                }}
            },
            {
                "name": "edgeDetailedMeasureLabels", "type": "text",
                "interactive": False,
                "key": "node_id",
                "from": {"data": "visibleEdgeLabels"},
                "encode": {"update": {
                    "text": {"field": "display_label"},
                    "x": {"signal": f"datum.labelScreenX + {EDGE_DETAILED_LABEL_OFFSET}*datum.labelUnitX"},
                    "y": {"signal": f"datum.labelScreenY + {EDGE_DETAILED_LABEL_OFFSET}*datum.labelUnitY"},
                    "align": {"value": "center"},
                    "baseline": {"value": "middle"},
                    "font": {"value": "Inter, Arial, sans-serif"},
                    "fontSize": {"value": EDGE_LABEL_FONT_SIZE},
                    "lineHeight": {"value": EDGE_LABEL_LINE_HEIGHT},
                    "fontWeight": {"value": 700},
                    "opacity": {"value": 0},
                    "fillOpacity": {"value": 0},
                    "strokeOpacity": {"value": 0},
                    "limit": {"value": 340},
                }}
            },
            {
                "name": "edgeDetailLeaders", "type": "rule",
                "clip": True,
                "interactive": False,
                "from": {"data": "visibleEdgeLabels"},
                "zindex": 6,
                "encode": {"update": {
                    "x": {"field": "labelScreenX"},
                    "y": {"field": "labelScreenY"},
                    "x2": {"signal": f"datum.labelScreenX + {EDGE_DETAILED_LABEL_OFFSET - 8}*datum.labelUnitX"},
                    "y2": {"signal": f"datum.labelScreenY + {EDGE_DETAILED_LABEL_OFFSET - 8}*datum.labelUnitY"},
                    "stroke": {"field": "label_colour"},
                    "strokeWidth": {"value": 1.25},
                    "strokeOpacity": {
                        "signal": (
                            "indexof(detailedEdgeLabelIds, datum.node_id)>=0 && "
                            "indexof(blockedEdgeLabelIds, datum.node_id)<0 ? 0.78 : 0"
                        )
                    },
                }},
            },
            {
                "name": "edgeLabels", "type": "text",
                "clip": True,
                "key": "node_id",
                "zindex": 7,
                "from": {"data": "visibleEdgeLabels"},
                "encode": {"enter": {
                    "tooltip": {
                        "signal": (
                            "{'Change': datum.label, "
                            "'Details': datum.detailed_description}"
                        )
                    },
                }, "update": {
                    "text": {
                        "signal": (
                            "indexof(detailedEdgeLabelIds, datum.node_id)>=0 ? "
                            "datum.display_label : datum.label"
                        )
                    },
                    "x": {"signal": f"datum.labelScreenX + (indexof(detailedEdgeLabelIds, datum.node_id)>=0 ? {EDGE_DETAILED_LABEL_OFFSET} : {EDGE_SHORT_LABEL_OFFSET})*datum.labelUnitX"},
                    "y": {"signal": f"datum.labelScreenY + (indexof(detailedEdgeLabelIds, datum.node_id)>=0 ? {EDGE_DETAILED_LABEL_OFFSET} : {EDGE_SHORT_LABEL_OFFSET})*datum.labelUnitY"},
                    "align": {"value": "center"},
                    "baseline": {"value": "middle"},
                    "font": {"value": "Inter, Arial, sans-serif"},
                    "fontSize": {"value": EDGE_LABEL_FONT_SIZE},
                    "lineHeight": {"value": EDGE_LABEL_LINE_HEIGHT},
                    "fontWeight": {"value": 700},
                    "opacity": {"signal": "indexof(blockedEdgeLabelIds, datum.node_id)<0 ? 1 : 0"},
                    "fillOpacity": {"value": 1},
                    "limit": {"value": 340},
                    "fill": {"field": "label_colour"},
                    "stroke": {"value": "#ffffff"},
                    "strokeOpacity": {"value": 1},
                    "strokeWidth": {"value": 2},
                }}
            },
        ],
    }


def render_html(article: str, spec: dict) -> str:
    encoded = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    word_change_legend = "".join(
        (
            '<li><span class="legend-dot" style="--legend-color:'
            f'{html.escape(WORD_CHANGE_COLOURS[change_type])}'
            '"></span><span>'
            f'{html.escape(WORD_CHANGE_LABELS[change_type])}'
            "</span></li>"
        )
        for change_type in ("start", "added", "removed", "mixed", "no_change")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meaningful Projection Space</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5.33.1"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-tooltip@0.35.2/build/vega-tooltip.min.js"></script>
<style>
html,body{{height:100%;margin:0;background:{BACKGROUND_COLOR};color:#151915;font-family:Inter,Arial,sans-serif}}
#app{{width:min(100%,1200px);height:min(100%,720px);margin:0 auto;display:flex;flex-direction:column;overflow:hidden;box-sizing:border-box}}
#topbar{{padding:14px 20px 8px;display:flex;align-items:center;gap:22px;flex-wrap:wrap}}
#topbar h1{{font-size:20px;line-height:1.2;font-weight:700;margin:0}}
#content{{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 250px;border-top:1px solid #e2e5df}}
#vis{{flex:1;min-width:0;min-height:0;overflow:hidden}}
#vis svg{{cursor:grab;overflow:hidden}}
#vis svg:active{{cursor:grabbing}}
#vis svg .semanticLabels text,#vis svg .edgeLabels text{{paint-order:stroke fill;stroke-linejoin:round;transition:opacity 150ms ease}}
#legend{{min-width:0;overflow:auto;border-left:1px solid #d9ddd6;padding:16px 16px 22px;background:#f7f6f2;box-sizing:border-box}}
#legend h2{{font-size:17px;margin:0 0 15px}}
#legend h3{{font-size:12px;line-height:1.2;letter-spacing:.06em;text-transform:uppercase;color:#687068;margin:18px 0 9px}}
.legend-list{{list-style:none;margin:0;padding:0;display:grid;gap:8px}}
.legend-list li{{min-width:0;display:flex;align-items:center;gap:8px;font-size:12px;line-height:1.25}}
.legend-dot{{--legend-color:{NOISE_COLOR};width:14px;height:14px;box-sizing:border-box;border-radius:50%;background:var(--legend-color);border:0;box-shadow:none;flex:none}}
.legend-line{{width:30px;height:0;border-top:2.25px solid {TRAJECTORY_COLOR};flex:none}}
.legend-line.thick{{border-top-width:4.45px}}
.legend-arc{{width:30px;height:10px;box-sizing:border-box;border:{REVERT_ARC_STROKE_WIDTH}px dotted {EXACT_REVERT_COLOR};border-bottom:0;border-radius:50% 50% 0 0;flex:none;opacity:{REVERT_ARC_STROKE_OPACITY}}}
.legend-arrow{{width:30px;flex:none;text-align:center;color:{TRAJECTORY_COLOR};font-size:23px;font-weight:800;line-height:10px}}
@media(max-width:760px){{#content{{grid-template-columns:minmax(0,1fr) 205px}}#legend{{padding:12px 10px}}}}
</style></head><body><main id="app">
<header id="topbar"><h1>Meaningful Projection Space</h1></header>
<div id="content">
<div id="vis" role="img" aria-label="Interactive semantic revision timecurve"></div>
<aside id="legend" aria-label="Visualization legend">
<h2>Legend</h2>
<h3>Display</h3>
<ul class="legend-list">
<li><span class="legend-arrow">&#8594;</span><span>Revision direction</span></li>
<li><span class="legend-line thick"></span><span>Repeated projected path</span></li>
<li><span class="legend-arc"></span><span>Exact revert return</span></li>
</ul>
<h3>Revision changes</h3>
<ul class="legend-list">{word_change_legend}</ul>
</aside>
</div>
</main>
<script>
const spec={encoded};
const handler=new vegaTooltip.Handler({{theme:'light'}});
const vis=document.getElementById('vis');
const view=new vega.View(vega.parse(spec),{{renderer:'svg',hover:true}})
  .initialize(vis).tooltip(handler.call);
let resizeFrame=0;
let visibilityTimer=0;
let visibilitySyncing=false;
let visibilityPending=false;
let lastWidth=0;
let lastHeight=0;
let previousAcceptedParentIds=new Set();
let previousAcceptedEdgeIds=new Set();
const parentPlacementCache=new Map();
const edgePlacementCache=new Map();
function sameIds(first,second){{
  return first.length===second.length&&first.every((value,index)=>value===second[index]);
}}
function normalizedLabelKey(value){{
  return String(value||'').toLocaleLowerCase('en')
    .replace(/[^a-z0-9]+/g,' ').trim();
}}
function renderedLabelItems(mode,...markNames){{
  const names=new Set(markNames);
  const blockedParents=new Set(view.signal('blockedParentLabelIds')||[]);
  const blockedEdges=new Set(view.signal('blockedEdgeLabelIds')||[]);
  const items=new Map();
  function visit(item){{
    if(!item)return;
    if(item.mark&&names.has(item.mark.name)&&item.bounds){{
      const datum=item.datum&&item.datum.datum?item.datum.datum:item.datum;
      const bounds=item.bounds;
      if(datum&&datum.node_id&&
         Number.isFinite(bounds.x1)&&Number.isFinite(bounds.y1)&&
         Number.isFinite(bounds.x2)&&Number.isFinite(bounds.y2)){{
        const markName=item.mark.name;
        const kind=markName==='semanticLabels'?'parent':'edge';
        const variant=markName==='edgeDetailedMeasureLabels'
          ? 'detailed'
          : markName==='edgeShortMeasureLabels'
            ? 'short'
            : 'visible';
        const id=String(datum.node_id);
        const level=String(datum.level||'');
        const collisionBlocked=kind==='parent'
          ? blockedParents.has(id)
          : blockedEdges.has(id);
        // Collision-hidden marks stay in the scenegraph and remain measurable.
        // They can therefore return immediately when zooming opens enough room.
        const include=mode==='collision'||!collisionBlocked;
        if(!include){{
          if(Array.isArray(item.items))item.items.forEach(visit);
          return;
        }}
        items.set(`${{kind}}:${{variant}}:${{id}}`,{{
          id,
          kind,
          variant,
          level,
          hasDetailed:Boolean(String(datum.detailed_description||'').trim()),
          parentId:datum.parent_id==null?'':String(datum.parent_id),
          ancestorIds:Array.isArray(datum.ancestor_ids)
            ? datum.ancestor_ids.map(String)
            : [],
          supportPointIndices:Array.isArray(datum.support_point_indices)
            ? datum.support_point_indices.map(Number)
            : [],
          labelKey:normalizedLabelKey(datum.label),
          priority:Number(datum.label_priority)||0,
          importance:Number(datum.importance)||0,
          parentPlacementDx:Number(datum.parentPlacementDx)||0,
          parentPlacementDy:Number(datum.parentPlacementDy)||0,
          edgeScreenLength:Number(datum.screenLength)||0,
          screenX1:Number(datum.screenX1),
          screenY1:Number(datum.screenY1),
          screenC1X:Number(datum.screenC1X),
          screenC1Y:Number(datum.screenC1Y),
          screenC2X:Number(datum.screenC2X),
          screenC2Y:Number(datum.screenC2Y),
          screenX2:Number(datum.screenX2),
          screenY2:Number(datum.screenY2),
          clipT0:Number(datum.clipT0),
          clipT1:Number(datum.clipT1),
          x1:bounds.x1,
          y1:bounds.y1,
          x2:bounds.x2,
          y2:bounds.y2
        }});
      }}
    }}
    if(Array.isArray(item.items))item.items.forEach(visit);
  }}
  visit(view.scenegraph().root);
  return Array.from(items.values());
}}
function renderedLabelIds(...markNames){{
  return renderedLabelItems('visible',...markNames)
    .map(item=>item.id)
    .sort();
}}
const labelCollisionPadding={LABEL_COLLISION_PADDING};
const labelSpatialHashCellSize={LABEL_SPATIAL_HASH_CELL_SIZE};
const parentLabelRadialGap={PARENT_LABEL_RADIAL_GAP};
const parentLabelDataPadding={PARENT_LABEL_DATA_PADDING};
const edgeLabelRepeatDistance={EDGE_LABEL_REPEAT_DISTANCE};
const labelDetailMinZoomLevel={LABEL_DETAIL_MIN_ZOOM_LEVEL};
const labelObstacleGridSize={LABEL_OBSTACLE_GRID_SIZE};
const edgeDensityEnter={LABEL_EDGE_DENSITY_ENTER};
const edgeDensityExit={LABEL_EDGE_DENSITY_EXIT};
const detailDensityEnter={LABEL_DETAIL_DENSITY_ENTER};
const detailDensityExit={LABEL_DETAIL_DENSITY_EXIT};
function boxesOverlap(first,second,padding=labelCollisionPadding){{
  return first.x1-padding<=second.x2+padding&&
    first.x2+padding>=second.x1-padding&&
    first.y1-padding<=second.y2+padding&&
    first.y2+padding>=second.y1-padding;
}}
function outsideViewport(candidate,padding=4){{
  return candidate.x1-padding<0||candidate.y1-padding<0||
    candidate.x2+padding>view.width()||candidate.y2+padding>view.height();
}}
class LabelSpatialHash{{
  constructor(cellSize=labelSpatialHashCellSize){{
    this.cellSize=cellSize;
    this.cells=new Map();
  }}
  bounds(box){{
    return {{
      minX:Math.floor((box.x1-labelCollisionPadding)/this.cellSize),
      maxX:Math.floor((box.x2+labelCollisionPadding)/this.cellSize),
      minY:Math.floor((box.y1-labelCollisionPadding)/this.cellSize),
      maxY:Math.floor((box.y2+labelCollisionPadding)/this.cellSize)
    }};
  }}
  overlaps(box){{
    const b=this.bounds(box);
    for(let x=b.minX;x<=b.maxX;x+=1){{
      for(let y=b.minY;y<=b.maxY;y+=1){{
        const bucket=this.cells.get(`${{x}},${{y}}`);
        if(bucket&&bucket.some(other=>boxesOverlap(box,other)))return true;
      }}
    }}
    return false;
  }}
  insert(box){{
    const b=this.bounds(box);
    for(let x=b.minX;x<=b.maxX;x+=1){{
      for(let y=b.minY;y<=b.maxY;y+=1){{
        const key=`${{x}},${{y}}`;
        const bucket=this.cells.get(key);
        if(bucket)bucket.push(box);else this.cells.set(key,[box]);
      }}
    }}
  }}
}}
class ObstacleDensityGrid{{
  constructor(cellSize=labelObstacleGridSize){{
    this.cellSize=cellSize;
    this.cells=new Map();
  }}
  add(x,y,weight=1,radius=0){{
    if(!Number.isFinite(x)||!Number.isFinite(y))return;
    const minX=Math.floor((x-radius)/this.cellSize);
    const maxX=Math.floor((x+radius)/this.cellSize);
    const minY=Math.floor((y-radius)/this.cellSize);
    const maxY=Math.floor((y+radius)/this.cellSize);
    for(let cellX=minX;cellX<=maxX;cellX+=1){{
      for(let cellY=minY;cellY<=maxY;cellY+=1){{
        const centerX=(cellX+0.5)*this.cellSize;
        const centerY=(cellY+0.5)*this.cellSize;
        const reach=radius+this.cellSize*0.72;
        if(Math.hypot(centerX-x,centerY-y)>reach)continue;
        const key=`${{cellX}},${{cellY}}`;
        this.cells.set(key,(this.cells.get(key)||0)+weight);
      }}
    }}
  }}
  score(box){{
    const minX=Math.floor(box.x1/this.cellSize);
    const maxX=Math.floor(box.x2/this.cellSize);
    const minY=Math.floor(box.y1/this.cellSize);
    const maxY=Math.floor(box.y2/this.cellSize);
    let total=0;
    let occupied=0;
    let weighted=0;
    let maximum=0;
    for(let cellX=minX;cellX<=maxX;cellX+=1){{
      for(let cellY=minY;cellY<=maxY;cellY+=1){{
        total+=1;
        const value=this.cells.get(`${{cellX}},${{cellY}}`)||0;
        if(value>0)occupied+=1;
        weighted+=Math.min(value,4)/4;
        maximum=Math.max(maximum,value);
      }}
    }}
    if(!total)return 1;
    const occupiedShare=occupied/total;
    const weightedShare=weighted/total;
    const peak=Math.min(maximum,4)/4;
    return 0.55*occupiedShare+0.30*weightedShare+0.15*peak;
  }}
}}
function parentSupportScreenPoints(base){{
  const points=view.data('points')||[];
  const xScale=view.scale('xscale');
  const yScale=view.scale('yscale');
  return base.supportPointIndices.map(index=>points[index])
    .filter(Boolean)
    .map(point=>({{x:xScale(point.x),y:yScale(point.y)}}))
    .filter(point=>Number.isFinite(point.x)&&Number.isFinite(point.y));
}}
function parentPlacementOptions(base){{
  const width=Math.max(1,base.x2-base.x1);
  const height=Math.max(1,base.y2-base.y1);
  const horizontal=Math.min(160,Math.max(24,width*0.5+parentLabelRadialGap));
  const vertical=Math.min(56,Math.max(18,height+parentLabelRadialGap));
  const currentDx=Number(base.parentPlacementDx)||0;
  const currentDy=Number(base.parentPlacementDy)||0;
  const baseCenterX=(base.x1+base.x2)/2-currentDx;
  const baseCenterY=(base.y1+base.y2)/2-currentDy;
  const cached=parentPlacementCache.get(base.id);
  const options=[];
  if(cached)options.push({{dx:cached.dx,dy:cached.dy,cached:true}});
  let variable;
  if(base.level==='root'){{
    variable=[{{dx:0,dy:0}}];
  }}else if(['macro','cluster','section'].includes(base.level)){{
    const support=parentSupportScreenPoints(base);
    if(support.length){{
      const centroid={{
        x:support.reduce((sum,point)=>sum+point.x,0)/support.length,
        y:support.reduce((sum,point)=>sum+point.y,0)/support.length
      }};
      const centralSupport=support.slice().sort((first,second)=>
        Math.hypot(first.x-centroid.x,first.y-centroid.y)-
        Math.hypot(second.x-centroid.x,second.y-centroid.y)
      ).slice(0,8);
      variable=[centroid,...centralSupport].map(point=>({{
        dx:point.x-baseCenterX,
        dy:point.y-baseCenterY
      }}));
      variable.unshift({{dx:0,dy:0}});
    }}else{{
      variable=[{{dx:0,dy:0}}];
    }}
  }}else{{
    variable=[
        {{dx:0,dy:0}},{{dx:0,dy:-vertical}},{{dx:0,dy:vertical}},
        {{dx:horizontal,dy:0}},{{dx:-horizontal,dy:0}},
        {{dx:horizontal*0.72,dy:-vertical*0.78}},
        {{dx:-horizontal*0.72,dy:-vertical*0.78}},
        {{dx:horizontal*0.72,dy:vertical*0.78}},
        {{dx:-horizontal*0.72,dy:vertical*0.78}}
      ];
  }}
  for(const option of variable){{
    if(!options.some(item=>
      Math.abs(item.dx-option.dx)<0.1&&Math.abs(item.dy-option.dy)<0.1
    ))options.push({{...option,cached:false}});
  }}
  return options;
}}
function parentCandidateAt(base,option){{
  const currentDx=Number(base.parentPlacementDx)||0;
  const currentDy=Number(base.parentPlacementDy)||0;
  return {{
    ...base,
    placementDx:option.dx,
    placementDy:option.dy,
    x1:base.x1-currentDx+option.dx,
    y1:base.y1-currentDy+option.dy,
    x2:base.x2-currentDx+option.dx,
    y2:base.y2-currentDy+option.dy
  }};
}}
function parentSupportCoverage(base,candidate){{
  const support=parentSupportScreenPoints(base);
  let count=0;
  for(const point of support){{
    if(point.x>=candidate.x1-parentLabelDataPadding&&
       point.x<=candidate.x2+parentLabelDataPadding&&
       point.y>=candidate.y1-parentLabelDataPadding&&
       point.y<=candidate.y2+parentLabelDataPadding)count+=1;
  }}
  if(count||base.supportPointIndices.length)return count;
  // Compatibility fallback for externally supplied hierarchies without
  // explicit membership: the reviewed semantic anchor itself is the support.
  const anchorX=(base.x1+base.x2)/2-(Number(base.parentPlacementDx)||0);
  const anchorY=(base.y1+base.y2)/2-(Number(base.parentPlacementDy)||0);
  return anchorX>=candidate.x1-parentLabelDataPadding&&
    anchorX<=candidate.x2+parentLabelDataPadding&&
    anchorY>=candidate.y1-parentLabelDataPadding&&
    anchorY<=candidate.y2+parentLabelDataPadding ? 1 : 0;
}}
function bestParentPlacement(base,occupied,obstacles){{
  let best=null;
  let rank=0;
  for(const option of parentPlacementOptions(base)){{
    const candidate=parentCandidateAt(base,option);
    if(outsideViewport(candidate,8)||occupied.overlaps(candidate)){{
      rank+=1;
      continue;
    }}
    const supportCoverage=parentSupportCoverage(base,candidate);
    if(supportCoverage<1){{
      rank+=1;
      continue;
    }}
    const density=obstacles.score(candidate);
    const displacement=Math.hypot(option.dx,option.dy);
    const supportReward=Math.min(6,supportCoverage)*0.08;
    const score=density*0.08+Math.min(0.08,displacement/2400)+
      rank*0.002-supportReward;
    if(option.cached)return {{...candidate,density,supportCoverage,score}};
    if(!best||score<best.score){{
      best={{...candidate,density,supportCoverage,score}};
    }}
    rank+=1;
  }}
  return best;
}}
function cubicPoint(edge,t){{
  const u=1-t;
  return {{
    x:u*u*u*edge.screenX1+3*u*u*t*edge.screenC1X+
      3*u*t*t*edge.screenC2X+t*t*t*edge.screenX2,
    y:u*u*u*edge.screenY1+3*u*u*t*edge.screenC1Y+
      3*u*t*t*edge.screenC2Y+t*t*t*edge.screenY2
  }};
}}
function cubicTangent(edge,t){{
  const u=1-t;
  return {{
    x:3*u*u*(edge.screenC1X-edge.screenX1)+
      6*u*t*(edge.screenC2X-edge.screenC1X)+
      3*t*t*(edge.screenX2-edge.screenC2X),
    y:3*u*u*(edge.screenC1Y-edge.screenY1)+
      6*u*t*(edge.screenC2Y-edge.screenC1Y)+
      3*t*t*(edge.screenY2-edge.screenC2Y)
  }};
}}
function buildObstacleDensityGrid(){{
  const grid=new ObstacleDensityGrid();
  for(const edge of view.data('splineEdgeGeometry')||[]){{
    const weight=1+0.22*Math.min(4,Math.max(0,
      Number(edge.overlap_count||1)-1));
    for(let step=0;step<=18;step+=1){{
      const point=cubicPoint(edge,step/18);
      grid.add(point.x,point.y,weight,2.5);
    }}
  }}
  for(const arc of view.data('revertArcGeometry')||[]){{
    for(let step=0;step<=14;step+=1){{
      const t=step/14;
      const u=1-t;
      grid.add(
        u*u*arc.screenX1+2*u*t*arc.controlX+t*t*arc.screenX2,
        u*u*arc.screenY1+2*u*t*arc.controlY+t*t*arc.screenY2,
        0.8,2
      );
    }}
  }}
  const xScale=view.scale('xscale');
  const yScale=view.scale('yscale');
  for(const point of view.data('points')||[]){{
    grid.add(xScale(point.x),yScale(point.y),3.5,7);
  }}
  return grid;
}}
function candidateAt(base,t,side,offset){{
  const point=cubicPoint(base,t);
  const tangent=cubicTangent(base,t);
  const length=Math.hypot(tangent.x,tangent.y);
  const unitX=side*(length>=1 ? -tangent.y/length : 1);
  const unitY=side*(length>=1 ? tangent.x/length : 0);
  const width=Math.max(1,base.x2-base.x1);
  const height=Math.max(1,base.y2-base.y1);
  const centerX=point.x+offset*unitX;
  const centerY=point.y+offset*unitY;
  return {{
    ...base,t,side,centerX,centerY,
    x1:centerX-width/2,y1:centerY-height/2,
    x2:centerX+width/2,y2:centerY+height/2
  }};
}}
function placementOptions(base){{
  const start=Math.max(0,Math.min(1,base.clipT0));
  const end=Math.max(start,Math.min(1,base.clipT1));
  const span=end-start;
  const cached=edgePlacementCache.get(base.id);
  const fractions=[0.5,0.35,0.65,0.2,0.8];
  const positions=[];
  if(cached&&cached.t>=start&&cached.t<=end){{
    positions.push({{t:cached.t,side:cached.side,cached:true}});
  }}
  for(const fraction of fractions){{
    const t=start+span*fraction;
    for(const side of [1,-1]){{
      if(!positions.some(item=>Math.abs(item.t-t)<1e-6&&item.side===side)){{
        positions.push({{t,side,cached:false}});
      }}
    }}
  }}
  return positions;
}}
function bestEdgePlacement(base,offset,occupied,obstacles,wasVisible,
                           enterThreshold,exitThreshold,
                           placementAllowed=()=>true){{
  const threshold=wasVisible?exitThreshold:enterThreshold;
  let best=null;
  for(const option of placementOptions(base)){{
    const candidate=candidateAt(base,option.t,option.side,offset);
    if(outsideViewport(candidate,8)||occupied.overlaps(candidate)||
       !placementAllowed(candidate))continue;
    const density=obstacles.score(candidate);
    if(density>threshold)continue;
    const distancePenalty=Math.abs(option.t-0.5)*0.06;
    const sidePenalty=option.side<0?0.004:0;
    const score=density+distancePenalty+sidePenalty;
    if(option.cached)return {{...candidate,density}};
    if(!best||score<best.score)best={{...candidate,density,score}};
  }}
  return best;
}}
function edgeLabelRepeatsNearby(acceptedPositions,labelKey,candidate){{
  if(!labelKey)return false;
  const positions=acceptedPositions.get(labelKey)||[];
  return positions.some(position=>
    Math.hypot(position.x-candidate.centerX,position.y-candidate.centerY)<
      edgeLabelRepeatDistance
  );
}}
function registerEdgeLabelPosition(acceptedPositions,labelKey,candidate){{
  if(!labelKey)return;
  const positions=acceptedPositions.get(labelKey)||[];
  positions.push({{x:candidate.centerX,y:candidate.centerY}});
  acceptedPositions.set(labelKey,positions);
}}
function semanticDepth(candidate,variant='visible'){{
  if(candidate.kind==='edge')return variant==='detailed'?5:4;
  if(candidate.level==='root')return 0;
  if(candidate.level==='macro')return 1;
  if(candidate.level==='cluster')return 2;
  if(candidate.level==='section')return 3;
  return 0;
}}
function semanticPlacementScore(candidate,variant,targetDepth){{
  const declaredPriority=Math.min(140,Math.max(0,candidate.priority)*18);
  const importanceBonus=Math.min(
    180,Math.log1p(Math.max(0,candidate.importance))*24
  );
  const wasVisible=candidate.kind==='edge'
    ? previousAcceptedEdgeIds.has(candidate.id)
    : previousAcceptedParentIds.has(candidate.id);
  const stabilityBonus=wasVisible?(candidate.kind==='edge'?220:160):0;
  if(candidate.level==='root'){{
    return 6000+declaredPriority+importanceBonus+stabilityBonus;
  }}
  if(candidate.level==='endpoint'){{
    return 5200+declaredPriority+importanceBonus+stabilityBonus;
  }}
  const depth=semanticDepth(candidate,variant);
  const depthDelta=depth-targetDepth;
  // Finer-than-current labels are delayed more strongly than useful parent
  // context, but they remain candidates and appear whenever space is free.
  const distancePenalty=Math.abs(depthDelta)*(
    depthDelta>0?{LABEL_FINER_DEPTH_PENALTY}:{LABEL_COARSER_DEPTH_PENALTY}
  );
  // Unlike SHACLens, this timecurve treats revision edges as first-class
  // semantic content. Their priority rises continuously from section detail
  // toward short and finally full descriptions as the user zooms in.
  const edgeZoomBonus=candidate.kind==='edge'
    ? Math.max(0,targetDepth-3)*180
    : 0;
  return 3000-distancePenalty+declaredPriority+importanceBonus+
    stabilityBonus+edgeZoomBonus;
}}
function collisionBlocks(){{
  const compareCandidates=(first,second)=>{{
    const priorityDifference=second.priority-first.priority;
    if(priorityDifference)return priorityDifference;
    const importanceDifference=second.importance-first.importance;
    if(importanceDifference)return importanceDifference;
    return `${{first.kind}}:${{first.id}}`.localeCompare(`${{second.kind}}:${{second.id}}`);
  }};
  const parents=renderedLabelItems('collision','semanticLabels')
    .sort(compareCandidates);
  const shortEdges=renderedLabelItems(
    'collision','edgeShortMeasureLabels'
  ).sort(compareCandidates);
  const detailedEdges=new Map(
    renderedLabelItems('collision','edgeDetailedMeasureLabels')
      .map(candidate=>[candidate.id,candidate])
  );
  const zoomLevel=Math.max(0,Number(view.signal('zoomLevel'))||0);
  const targetDepth=Math.max(
    0,Math.min(5,Number(view.signal('semanticTargetDepth'))||1)
  );
  const occupied=new LabelSpatialHash();
  const obstacles=buildObstacleDensityGrid();
  const acceptedEdgeLabelPositions=new Map();
  const acceptedParentIds=new Set();
  const acceptedEdgeIds=new Set();
  const parentPlacements=new Map();
  const placements=new Map();
  const blockedParents=[];
  const blockedEdges=[];
  const detailedEdgeIds=[];
  const acceptCandidate=candidate=>{{
    occupied.insert(candidate);
  }};
  const edgeActions=shortEdges.map(shortCandidate=>{{
    const detailedCandidate=detailedEdges.get(shortCandidate.id);
    const detailedWidth=detailedCandidate
      ? Math.max(0,detailedCandidate.x2-detailedCandidate.x1)
      : 0;
    const requiredDetailLength=Math.max(
      {MIN_DETAILED_EDGE_SCREEN_LENGTH},Math.min(240,detailedWidth*0.55)
    );
    const detailedEligible=zoomLevel>=labelDetailMinZoomLevel&&
      detailedCandidate&&detailedCandidate.hasDetailed&&
      shortCandidate.edgeScreenLength>=requiredDetailLength;
    const priorityCandidate=detailedEligible
      ? detailedCandidate
      : shortCandidate;
    const variant=detailedEligible?'detailed':'short';
    return {{
      kind:'edge',candidate:shortCandidate,detailedCandidate,
      detailedEligible,score:semanticPlacementScore(
        priorityCandidate,variant,targetDepth
      )
    }};
  }});
  const actions=parents.map(candidate=>({{
    kind:'parent',candidate,
    score:semanticPlacementScore(candidate,'visible',targetDepth)
  }})).concat(edgeActions).sort((first,second)=>{{
    const scoreDifference=second.score-first.score;
    if(scoreDifference)return scoreDifference;
    return compareCandidates(first.candidate,second.candidate);
  }});
  // SHACLens-style greedy screen-space placement, adapted to make edge labels
  // the preferred detailed layer. No semantic level is hard-disabled: coarse
  // labels win while zoomed out, edge descriptions win close up, and every
  // lower-priority label is still accepted when its actual screen box fits.
  for(const action of actions){{
    if(action.kind==='parent'){{
      const candidate=action.candidate;
      const parentPlacement=bestParentPlacement(
        candidate,occupied,obstacles
      );
      if(!parentPlacement){{
        blockedParents.push(candidate.id);
      }}else{{
        acceptCandidate(parentPlacement);
        acceptedParentIds.add(candidate.id);
        parentPlacements.set(candidate.id,{{
          dx:parentPlacement.placementDx,
          dy:parentPlacement.placementDy
        }});
      }}
      continue;
    }}
    const shortCandidate=action.candidate;
    const detailedCandidate=action.detailedCandidate;
    const detailedEligible=action.detailedEligible;
    const wasVisible=previousAcceptedEdgeIds.has(shortCandidate.id);
    const hasRepeatSpace=placement=>!edgeLabelRepeatsNearby(
      acceptedEdgeLabelPositions,shortCandidate.labelKey,placement
    );
    const detailedPlacement=detailedEligible
      ? bestEdgePlacement(
          detailedCandidate,{EDGE_DETAILED_LABEL_OFFSET},occupied,obstacles,
          wasVisible,detailDensityEnter,detailDensityExit,hasRepeatSpace
        )
      : null;
    // Prefer the concrete full revision description whenever zooming creates
    // enough contextual screen space. Repeated short text is permitted again
    // after a map-like minimum screen distance instead of being hidden
    // globally across the complete trajectory.
    if(detailedPlacement){{
      acceptCandidate(detailedPlacement);
      detailedEdgeIds.push(shortCandidate.id);
      acceptedEdgeIds.add(shortCandidate.id);
      placements.set(shortCandidate.id,{{
        t:detailedPlacement.t,side:detailedPlacement.side
      }});
      registerEdgeLabelPosition(
        acceptedEdgeLabelPositions,shortCandidate.labelKey,detailedPlacement
      );
    }}else{{
      const shortPlacement=bestEdgePlacement(
        shortCandidate,{EDGE_SHORT_LABEL_OFFSET},occupied,obstacles,
        wasVisible,edgeDensityEnter,edgeDensityExit,hasRepeatSpace
      );
      if(shortPlacement){{
        acceptCandidate(shortPlacement);
        acceptedEdgeIds.add(shortCandidate.id);
        placements.set(shortCandidate.id,{{
          t:shortPlacement.t,side:shortPlacement.side
        }});
        registerEdgeLabelPosition(
          acceptedEdgeLabelPositions,shortCandidate.labelKey,shortPlacement
        );
      }}else{{
        blockedEdges.push(shortCandidate.id);
      }}
    }}
  }}
  const parentPlacementRows=Array.from(parentPlacements.entries())
    .sort((first,second)=>first[0].localeCompare(second[0]));
  const placementRows=Array.from(placements.entries())
    .sort((first,second)=>first[0].localeCompare(second[0]));
  return {{
    parentIds:Array.from(new Set(blockedParents)).sort(),
    edgeIds:Array.from(new Set(blockedEdges)).sort(),
    detailedEdgeIds:Array.from(new Set(detailedEdgeIds)).sort(),
    acceptedParentIds:Array.from(acceptedParentIds).sort(),
    acceptedEdgeIds:Array.from(acceptedEdgeIds).sort(),
    parentPlacementIds:parentPlacementRows.map(row=>row[0]),
    parentPlacementDx:parentPlacementRows.map(row=>row[1].dx),
    parentPlacementDy:parentPlacementRows.map(row=>row[1].dy),
    placementIds:placementRows.map(row=>row[0]),
    placementTs:placementRows.map(row=>row[1].t),
    placementSides:placementRows.map(row=>row[1].side)
  }};
}}
async function syncLabelVisibility(){{
  if(view.signal('down'))return;
  if(visibilitySyncing){{
    visibilityPending=true;
    return;
  }}
  visibilitySyncing=true;
  try{{
    const blocked=collisionBlocks();
    let currentBlockedParents=(view.signal('blockedParentLabelIds')||[]).slice().sort();
    let currentBlockedEdges=(view.signal('blockedEdgeLabelIds')||[]).slice().sort();
    let currentDetailedEdges=(view.signal('detailedEdgeLabelIds')||[]).slice().sort();
    let currentParentPlacementIds=(view.signal('parentLabelPlacementIds')||[]).slice();
    let currentParentPlacementDx=(view.signal('parentLabelPlacementDx')||[]).slice();
    let currentParentPlacementDy=(view.signal('parentLabelPlacementDy')||[]).slice();
    let currentPlacementIds=(view.signal('edgeLabelPlacementIds')||[]).slice();
    let currentPlacementTs=(view.signal('edgeLabelPlacementTs')||[]).slice();
    let currentPlacementSides=(view.signal('edgeLabelPlacementSides')||[]).slice();
    if(!sameIds(blocked.parentIds,currentBlockedParents)||
       !sameIds(blocked.edgeIds,currentBlockedEdges)||
       !sameIds(blocked.detailedEdgeIds,currentDetailedEdges)||
       !sameIds(blocked.parentPlacementIds,currentParentPlacementIds)||
       !sameIds(blocked.parentPlacementDx,currentParentPlacementDx)||
       !sameIds(blocked.parentPlacementDy,currentParentPlacementDy)||
       !sameIds(blocked.placementIds,currentPlacementIds)||
       !sameIds(blocked.placementTs,currentPlacementTs)||
       !sameIds(blocked.placementSides,currentPlacementSides)){{
      view.signal('blockedParentLabelIds',blocked.parentIds);
      view.signal('blockedEdgeLabelIds',blocked.edgeIds);
      view.signal('detailedEdgeLabelIds',blocked.detailedEdgeIds);
      view.signal('parentLabelPlacementIds',blocked.parentPlacementIds);
      view.signal('parentLabelPlacementDx',blocked.parentPlacementDx);
      view.signal('parentLabelPlacementDy',blocked.parentPlacementDy);
      view.signal('edgeLabelPlacementIds',blocked.placementIds);
      view.signal('edgeLabelPlacementTs',blocked.placementTs);
      view.signal('edgeLabelPlacementSides',blocked.placementSides);
      await view.runAsync();
    }}
    previousAcceptedParentIds=new Set(blocked.acceptedParentIds);
    previousAcceptedEdgeIds=new Set(blocked.acceptedEdgeIds);
    for(let index=0;index<blocked.parentPlacementIds.length;index+=1){{
      parentPlacementCache.set(blocked.parentPlacementIds[index],{{
        dx:blocked.parentPlacementDx[index],
        dy:blocked.parentPlacementDy[index]
      }});
    }}
    for(let index=0;index<blocked.placementIds.length;index+=1){{
      edgePlacementCache.set(blocked.placementIds[index],{{
        t:blocked.placementTs[index],
        side:blocked.placementSides[index]
      }});
    }}
    const parentIds=renderedLabelIds('semanticLabels');
    const edgeIds=renderedLabelIds('edgeLabels');
    vis.dataset.zoom=Number(view.signal('zoomLevel')).toFixed(3);
    vis.dataset.visibleLabels=String(parentIds.length+edgeIds.length);
    vis.dataset.visibleHierarchyLabels=String(parentIds.length);
    vis.dataset.visibleEdgeLabels=String(edgeIds.length);
    vis.dataset.blockedParentLabels=String(blocked.parentIds.length);
    vis.dataset.blockedEdgeLabels=String(blocked.edgeIds.length);
    vis.dataset.detailedEdgeLabels=String(blocked.detailedEdgeIds.length);
    vis.dataset.shortEdgeLabels=String(
      Math.max(0,edgeIds.length-blocked.detailedEdgeIds.length)
    );
  }}finally{{
    visibilitySyncing=false;
    if(visibilityPending){{
      visibilityPending=false;
      scheduleLabelVisibility();
    }}
  }}
}}
function scheduleLabelVisibility(){{
  clearTimeout(visibilityTimer);
  visibilityTimer=setTimeout(()=>{{void syncLabelVisibility();}},20);
}}
view.addSignalListener('zooming',(_name,value)=>{{if(!value)scheduleLabelVisibility();}});
view.addSignalListener('zoomLevel',()=>{{
  if(!view.signal('zooming'))scheduleLabelVisibility();
}});
view.addSignalListener('down',(_name,value)=>{{if(value===null)scheduleLabelVisibility();}});
function resizeView(){{
  cancelAnimationFrame(resizeFrame);
  resizeFrame=requestAnimationFrame(async()=>{{
    const width=Math.max(320,vis.clientWidth-56);
    const height=Math.max(420,vis.clientHeight-46);
    if(width===lastWidth&&height===lastHeight)return;
    lastWidth=width;lastHeight=height;
    await view.width(width).height(height).runAsync();
    scheduleLabelVisibility();
  }});
}}
new ResizeObserver(resizeView).observe(vis);
resizeView();
</script></body></html>"""


def source_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def fixed_parent_label_sources_match(
    payload: dict, projection_path: Path, keywords_path: Path
) -> bool:
    """Return whether a reviewed fixed label pyramid matches the layout.

    Reviewed parent labels belong to the hierarchy geometry.  Edge-label text is
    deliberately excluded: editorial corrections to individual trajectories
    must not invalidate or regenerate the fixed macro/cluster/section pyramid.
    ``keywords_path`` remains part of the signature for call-site compatibility.
    """
    marker = payload.get("fixed_parent_labels")
    if not isinstance(marker, dict) or marker.get("status") != "reviewed":
        return False
    if not projection_path.exists():
        return False
    return marker.get("projection_sha256") == source_sha256([projection_path])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semantic-zoom hierarchy and Vega timecurve.")
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument(
        "--diff-labels",
        "--keywords",
        dest="diff_labels",
        metavar="DIFF_LABELS",
        type=Path,
        required=True,
        help="Offline full-revision comparison label JSON (--keywords remains a compatibility alias)",
    )
    parser.add_argument("--article", required=True)
    parser.add_argument("--xml", type=Path)
    parser.add_argument(
        "--model",
        help=f"OpenAI model name (default: {DEFAULT_OPENAI_MODEL})",
    )
    parser.add_argument("--no-llm", action="store_true", help="Keep deterministic grounded fallback labels.")
    parser.add_argument("--skip-word-counts", action="store_true")
    parser.add_argument("--labels-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model = model_or_default(args.model)
    projection_path = args.projection.expanduser().resolve()
    keywords_path = args.diff_labels.expanduser().resolve()
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    keywords = json.loads(keywords_path.read_text(encoding="utf-8"))
    points, pairs = validate_sources(projection, keywords)
    embedding_path = projection_path.with_name(
        projection_path.name.replace(
            "_projection.json", "_revision_embeddings.npy"
        )
    )
    labels_output = args.labels_output or projection_path.with_name(
        projection_path.stem.removesuffix("_projection") + "_semantic_zoom_labels.json"
    )
    html_output = args.html_output or projection_path.with_name(
        projection_path.stem.removesuffix("_projection") + "_semantic_zoom.html"
    )
    fixed_parent_labels = None
    fixed_edge_label_overrides = None
    if labels_output.exists():
        existing_payload = json.loads(labels_output.read_text(encoding="utf-8"))
        fixed_edge_label_overrides = existing_payload.get(
            "fixed_edge_label_overrides"
        )
        if fixed_parent_label_sources_match(
            existing_payload, projection_path, keywords_path
        ):
            fixed_parent_labels = existing_payload.get("fixed_parent_labels")

    xml_path = (args.xml or Path(article_paths(args.article)["xml_path"])).expanduser().resolve()
    revert_analysis = projection.get("revert_analysis")
    if (
        not revert_analysis
        or revert_analysis.get("analysis_version") != REVERT_ANALYSIS_VERSION
        or not all(
            "interval_exact_sha1_revert_count" in point for point in points
        )
    ):
        full_history = load_revert_history(xml_path)
        conflict_history, sampled_for_conflict = history_window_for_points(
            full_history, points
        )
        revert_analysis = analyze_reverts(
            conflict_history,
            sampled_for_conflict,
        )
        for point, statistics in zip(
            points, revert_analysis["point_statistics"]
        ):
            point.update(statistics)
    word_counts = None
    if not args.skip_word_counts:
        word_counts = cached_word_counts(
            labels_output, projection_path, keywords_path, len(pairs)
        ) or _counts_from_xml(xml_path, points)
    hierarchy = build_hierarchy(
        projection, keywords, word_counts, root_label=args.article
    )

    llm_labels = {}
    macro_labels = {}
    if not args.no_llm:
        nodes = [
            node for node in hierarchy["clusters"] + hierarchy["sections"]
            if node["terms"]
        ]
        llm_labels = cached_llm_labels(
            labels_output,
            projection_path,
            keywords_path,
            args.model,
            LLM_PROVIDER,
        )
        missing = [node for node in nodes if node["node_id"] not in llm_labels]
        if missing:
            api_key = load_api_key(Path(BASE_DIR) / ".env")
            client = create_llm_client(api_key=api_key)
            llm_labels.update(generate_labels(client, args.model, missing))
        for level in ("clusters", "sections"):
            for node in hierarchy[level]:
                node["label"] = llm_labels[node["node_id"]] if node["terms"] else ""

        cluster_labels = {
            int(node["cluster_id"]): node["label"]
            for node in hierarchy["clusters"]
        }
        for macro in hierarchy["macros"]:
            macro["child_labels"] = [
                cluster_labels[cluster_id]
                for cluster_id in macro["cluster_ids"]
                if cluster_labels.get(cluster_id)
            ]
        macro_labels = cached_macro_labels(
            labels_output,
            projection_path,
            keywords_path,
            args.model,
            hierarchy["macros"],
            LLM_PROVIDER,
        )
        missing_macros = [
            node for node in hierarchy["macros"]
            if node["node_id"] not in macro_labels
        ]
        if missing_macros:
            if "client" not in locals():
                api_key = load_api_key(Path(BASE_DIR) / ".env")
                client = create_llm_client(api_key=api_key)
            macro_labels.update(generate_labels(
                client,
                args.model,
                missing_macros,
                system_prompt=MACRO_LABEL_SYSTEM_PROMPT,
            ))
        for node in hierarchy["macros"]:
            node["label"] = macro_labels[node["node_id"]]

    payload = {
        "article": args.article,
        "projection": str(projection_path),
        "revision_diff_labels": str(keywords_path),
        "revision_keywords": str(keywords_path),
        "llm_provider": None if args.no_llm else LLM_PROVIDER,
        "model": None if args.no_llm else args.model,
        "labels_generated_offline": True,
        "interactive_llm_calls": False,
        "edge_label_strategy": (
            "direct full-revision analysis, one 2-to-3-word label plus a detailed description"
        ),
        "trajectory_width_encoding": TRAJECTORY_WIDTH_ENCODING,
        "trajectory_spline_encoding": TRAJECTORY_SPLINE_ENCODING,
        "revision_point_colour_encoding": REVISION_POINT_COLOUR_ENCODING,
        "revert_analysis": revert_analysis,
        "pipeline_snapshot_id": SEMANTIC_ZOOM_SNAPSHOT_ID,
        "source_sha256": source_sha256(
            [Path(__file__), projection_path, keywords_path]
            + ([embedding_path] if embedding_path.exists() else [])
        ),
        "label_system_prompt": None if args.no_llm else LABEL_SYSTEM_PROMPT,
        "macro_label_system_prompt": None if args.no_llm else MACRO_LABEL_SYSTEM_PROMPT,
        "fixed_parent_labels": fixed_parent_labels,
        "fixed_edge_label_overrides": fixed_edge_label_overrides,
        "word_counts_are_git_word_diff": word_counts is not None,
        "hierarchy": hierarchy,
    }
    labels_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_output.write_text(
        render_html(
            args.article,
            build_vega_spec(
                args.article, projection, hierarchy, revert_analysis
            ),
        ),
        encoding="utf-8",
    )
    print(f"Label hierarchy: {labels_output}")
    print(f"Semantic zoom timecurve: {html_output}")
    print(format_label_pyramid(hierarchy))
    print(
        "Macro topics/clusters/sections/edges: "
        f"{len(hierarchy['macros'])}/{len(hierarchy['clusters'])}/"
        f"{len(hierarchy['sections'])}/{len(hierarchy['edges'])}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, UnicodeError) as error:
        print(f"[FEHLER] {error}")
        raise SystemExit(1) from error
