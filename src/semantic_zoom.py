#!/usr/bin/env python3
"""Build an offline label hierarchy and a semantic-zoom Vega timecurve."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import html
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
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
SEMANTIC_ZOOM_SNAPSHOT_ID = "2026-09-18-v087"
MIN_VIEW_SCALE = 0.01
MAX_VIEW_SCALE = 50.0
MIN_PARENT_LABEL_WORDS = 2
MAX_PARENT_LABEL_WORDS = 3
BACKGROUND_COLOR = "#fbfaf7"
TRAJECTORY_COLOR = "#72806f"
NOISE_COLOR = "#8a8a8a"
EXACT_REVERT_COLOR = "#d64545"
PARTIAL_REVERT_COLOR = "#00875a"
EDIT_WAR_COLOR = "#2457d6"
EDGE_SHORT_LABEL_OFFSET = 9
EDGE_DETAILED_LABEL_OFFSET = 28
# A full revision description is only useful once its own edge is visibly
# distinguishable.  This screen-space rule naturally opens descriptions while
# zooming without tying the behaviour to an arbitrary global zoom level.
MIN_DETAILED_EDGE_SCREEN_LENGTH = 72

# Line width is an event code, not a semantic-distance scale. A normal edge is
# clearly legible, and repeatedly traversed undirected MDS segments receive an
# increment for every extra traversal. Exact SHA1 reverts are encoded by an
# undo glyph at their target revision instead of by line width or line colour.
TRAJECTORY_WIDTH_SIGNAL = (
    "min(8, 2.25 + 1.1*max(datum.overlap_count-1, 0))"
)
TRAJECTORY_WIDTH_ENCODING = {
    "normal_edge_px": 2.25,
    "exact_sha1_revert_increment_px": 0.0,
    "exact_sha1_revert_encoding": "undo glyph at target revision",
    "additional_same_geometry_increment_px": 1.1,
    "maximum_px": 8.0,
    "overlap_definition": (
        "same undirected MDS endpoints rounded to 12 decimal places"
    ),
    "semantic_jump_controls_width": False,
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
        target_count = max(1, round(math.sqrt(len(clusters))))
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
            "label_confidence": pair.get("confidence"),
            "conflict_description": pair.get("conflict_description", ""),
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


def cluster_colour(cluster_id: int) -> str:
    """Return the stable HDBSCAN colour used by the original timecurve."""
    if cluster_id == -1:
        return NOISE_COLOR
    hue = (cluster_id * 0.61803398875) % 1.0
    lightness = 0.48 if cluster_id % 2 == 0 else 0.60
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, 0.72)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def cluster_label_colour(cluster_id: int) -> str:
    """Return a darker, readable variant of the stable cluster hue."""
    if cluster_id == -1:
        return "#555b5e"
    hue = (cluster_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hls_to_rgb(hue, 0.32, 0.68)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


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


def cluster_hull_rows(point_rows: list[dict], expansion: float = 1.08) -> list[dict]:
    """Create lightly expanded closed polygons for non-noise HDBSCAN clusters."""
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for point in point_rows:
        if point["cluster_id"] >= 0:
            grouped[point["cluster_id"]].append((point["x"], point["y"]))

    rows = []
    for cluster_id, coordinates in sorted(grouped.items()):
        hull = convex_hull(coordinates)
        if len(hull) < 3:
            continue
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
            rows.append({
                "cluster_id": cluster_id,
                "order": order,
                "x": x,
                "y": y,
            })
    return rows


def macro_hull_rows(point_rows: list[dict], expansion: float = 1.04) -> list[dict]:
    """Create closed overview polygons around the clusters of each macro topic."""
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for point in point_rows:
        if point.get("macro_id"):
            grouped[point["macro_id"]].append((point["x"], point["y"]))

    rows = []
    for macro_id, coordinates in sorted(grouped.items()):
        hull = convex_hull(coordinates)
        if len(hull) < 3:
            continue
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
            rows.append({"macro_id": macro_id, "order": order, "x": x, "y": y})
    return rows


def section_region_rows(
    point_rows: list[dict],
    sections: list[dict],
    expansion: float = 1.12,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Create section hulls, with fixed-width path capsules as a fallback."""
    polygons: list[dict] = []
    corridors: list[dict] = []
    corridor_points: list[dict] = []
    for section in sections:
        edge_indices = [int(index) for index in section["edge_indices"]]
        point_indices = sorted({
            point_index
            for edge_index in edge_indices
            for point_index in (edge_index, edge_index + 1)
        })
        coordinates = [
            (point_rows[index]["x"], point_rows[index]["y"])
            for index in point_indices
        ]
        hull = convex_hull(coordinates)
        twice_area = abs(sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(hull, hull[1:] + hull[:1])
        )) if len(hull) >= 3 else 0.0
        section_id = section["node_id"]
        cluster_id = int(section["cluster_id"])
        # Near-identical embeddings can technically form a three-point hull
        # whose area is too small to be visible. Those sections use a
        # screen-space capsule around their actual trajectory instead.
        if len(hull) >= 3 and twice_area > 1e-24:
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
                    "section_id": section_id,
                    "cluster_id": cluster_id,
                    "order": order,
                    "x": x,
                    "y": y,
                })
            continue

        for edge_index in edge_indices:
            start = point_rows[edge_index]
            end = point_rows[edge_index + 1]
            corridors.append({
                "section_id": section_id,
                "cluster_id": cluster_id,
                "edge": edge_index,
                "x": start["x"],
                "y": start["y"],
                "x2": end["x"],
                "y2": end["y"],
            })
        for point_index in point_indices:
            point = point_rows[point_index]
            corridor_points.append({
                "section_id": section_id,
                "cluster_id": cluster_id,
                "point": point_index,
                "x": point["x"],
                "y": point["y"],
            })
    return polygons, corridors, corridor_points


def build_revert_arc_rows(
    point_rows: list[dict], revert_analysis: dict | None
) -> list[dict]:
    """Map revert events to dashed screen-space arcs between plotted revisions.

    Each exact arc starts at the revision whose change was undone and ends at
    the plotted anchor of ``restored_revision_id``. A direct partial counter-
    edit is drawn as an Edit-War arc only when the offline analysis assigned it
    to a repeated counter-revert session.
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
    session_sizes = {
        str(session.get("session_id", "")): int(session.get("event_count", 0))
        for session in revert_analysis.get("edit_war_sessions", [])
        if session.get("session_id")
    }
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
        session_id = str(event.get("edit_war_session_id", ""))
        revert_kind = str(event.get("revert_kind", "exact_sha1"))
        is_edit_war = bool(session_id)
        restored_id = str(event.get("restored_revision_id") or "")
        restored_point = points_by_revision.get(restored_id)
        return_anchor = (
            restored_point
            if revert_kind == "exact_sha1" and restored_point is not None
            else revert_point
        )
        return_anchor_id = (
            restored_id
            if revert_kind == "exact_sha1" and restored_point is not None
            else source_id
        )
        source_key = (
            round(float(undone_point["x"]), 12),
            round(float(undone_point["y"]), 12),
        )
        target_key = (
            round(float(return_anchor["x"]), 12),
            round(float(return_anchor["y"]), 12),
        )
        pair = [str(editor) for editor in event.get("pair", []) if editor]
        rows.append({
            "arc_id": f"revert-{event_order + 1}-{source_id}-{target_id}",
            "event_order": int(event.get("revision_index", event_order)),
            # Start at the undone change and point back to the resulting
            # revert revision, matching the chronological return direction.
            "x": undone_point["x"],
            "y": undone_point["y"],
            "x2": return_anchor["x"],
            "y2": return_anchor["y"],
            "revision_id": source_id,
            "reverted_revision_id": target_id,
            "restored_revision_id": restored_id,
            "return_anchor_revision_id": return_anchor_id,
            "source_overlap_count": coordinate_counts[source_key],
            "target_overlap_count": coordinate_counts[target_key],
            "reverter": reverter,
            "reverted_editor": reverted_editor,
            "pair_text": " ↔ ".join(pair),
            "revert_kind": revert_kind,
            "revert_kind_label": (
                "Exact revert"
                if event.get("revert_kind") == "exact_sha1"
                else "Partial revert"
            ),
            "reversal_score": float(event.get("reversal_score", 1.0)),
            "session_id": session_id,
            "is_edit_war": is_edit_war,
            "edit_war_label": "Edit War" if is_edit_war else "",
            "session_event_count": session_sizes.get(
                session_id, 0
            ),
        })

    # Edit War is the highest-priority encoding only when the blue arc marks
    # the same projected connection. A shared endpoint is insufficient: many
    # unrelated revisions can collapse onto one projection coordinate, and
    # suppressing by endpoint would hide valid revert arcs on other edges.
    conflict_geometry_keys = {
        tuple(sorted((
            (round(float(row["x"]), 12), round(float(row["y"]), 12)),
            (round(float(row["x2"]), 12), round(float(row["y2"]), 12)),
        )))
        for row in rows
        if row["is_edit_war"]
    }
    rows = [
        row for row in rows
        if row["is_edit_war"]
        or tuple(sorted((
                (
                    round(float(row["x"]), 12),
                    round(float(row["y"]), 12),
                ),
                (
                    round(float(row["x2"]), 12),
                    round(float(row["y2"]), 12),
                ),
            ))) not in conflict_geometry_keys
    ]

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        endpoints = tuple(sorted((
            (round(float(row["x"]), 12), round(float(row["y"]), 12)),
            (round(float(row["x2"]), 12), round(float(row["y2"]), 12)),
        )))
        grouped[(row["session_id"], endpoints)].append(row)

    for group in grouped.values():
        group.sort(key=lambda row: (row["event_order"], row["arc_id"]))
        for rank, row in enumerate(group):
            row["arc_rank"] = rank
            row["arc_level"] = rank // 2
            row["arc_count"] = len(group)
    # Vega paints later paths on top. Draw every standalone revert first and
    # every Edit War last, while retaining chronology within each category.
    return sorted(
        rows,
        key=lambda row: (
            bool(row["is_edit_war"]),
            row["event_order"],
            row["arc_id"],
        ),
    )


def snap_exact_revert_positions(
    point_rows: list[dict], revert_analysis: dict | None
) -> None:
    """Place exact revert revisions on the restored revision coordinate.

    Identical SHA1 states are semantically identical even when an embedding or
    projection round-trip leaves their stored coordinates a few pixels apart.
    Snapping only exact reverts makes the visual return literally coincide
    with the restored state while partial reverts retain their own positions.
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
    revert_analysis = revert_analysis or projection.get("edit_wars")
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
    labelled_section_point_indices = {
        point_index
        for section in hierarchy["sections"]
        if section.get("label")
        for edge_index in section["edge_indices"]
        for point_index in (edge_index, edge_index + 1)
    }
    cluster_to_macro = {
        int(cluster_id): macro["node_id"]
        for macro in hierarchy.get("macros", [])
        for cluster_id in macro["cluster_ids"]
    }
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
            "macro_id": cluster_to_macro.get(int(point["cluster_id"])),
            "cluster_label": (
                "Noise"
                if int(point["cluster_id"]) == -1
                else f"Cluster {int(point['cluster_id']) + 1}"
            ),
            "cluster_probability": float(point.get("cluster_probability", 0.0)),
            "semantic_jump": float(point["semantic_jump_from_previous"]),
            "interval_revert_count": int(point.get("interval_revert_count", 0)),
            "interval_exact_sha1_revert_count": int(
                point.get("interval_exact_sha1_revert_count", 0)
            ),
            "interval_partial_text_revert_count": int(
                point.get("interval_partial_text_revert_count", 0)
            ),
            "interval_mutual_revert_count": int(
                point.get("interval_mutual_revert_count", 0)
            ),
            "gandica_c": float(point.get("cumulative_gandica_c", 0.0)),
            "confirmed_edit_war_event_count": int(
                point.get("interval_confirmed_edit_war_event_count", 0)
            ),
            "confirmed_edit_war_sessions": point.get(
                "interval_confirmed_edit_war_sessions", []
            ),
            "labelled_section_point": int(point["revision_index"])
            in labelled_section_point_indices,
        }
        for point in points
    ]
    snap_exact_revert_positions(point_rows, revert_analysis)
    revert_arc_rows = build_revert_arc_rows(point_rows, revert_analysis)

    edge_rows = []
    for index, edge in enumerate(hierarchy["edges"]):
        a, b = point_rows[index], point_rows[index + 1]
        semantic_jump = b["semantic_jump"]
        edge_rows.append({
            "edge": index,
            "x": a["x"],
            "y": a["y"],
            "x2": b["x"],
            "y2": b["y"],
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
            "partial_revert_count": b["interval_partial_text_revert_count"],
            "has_exact_revert": b["interval_exact_sha1_revert_count"] > 0,
            "mutual_revert_count": b["interval_mutual_revert_count"],
            "gandica_c": b["gandica_c"],
            "is_edit_war": b["confirmed_edit_war_event_count"] > 0,
            "edit_war_event_count": b["confirmed_edit_war_event_count"],
            "edit_war_pairs_text": " · ".join(
                " ↔ ".join(session.get("pair", []))
                for session in b["confirmed_edit_war_sessions"]
                if session.get("pair")
            ),
            "edit_war_description": edge.get("conflict_description", ""),
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
    cluster_ids = sorted({row["cluster_id"] for row in point_rows})
    hull_rows = cluster_hull_rows(point_rows)
    hierarchy_nodes = (
        hierarchy.get("macros", []) + hierarchy["clusters"]
        + hierarchy["sections"] + hierarchy["edges"]
    )
    node_by_id = {
        str(node["node_id"]): node for node in hierarchy_nodes
    }
    root_node = hierarchy["roots"][0]

    def fixed_parent_placement(node: dict) -> dict:
        """Choose one deterministic placement towards the parent centre.

        The rendered label may disappear when this position is occupied, but
        it must never search another side of its anchor while zooming or
        panning.
        """
        level = str(node["level"])
        if level == "root":
            return {
                "label_dx": 0,
                "label_dy": 0,
                "label_align": "center",
                "label_baseline": "middle",
            }
        parent = node_by_id.get(str(node.get("parent_id", "")), root_node)
        # Point towards the parent centre: peripheral labels then extend into
        # the available map interior rather than beyond the viewport edge.
        dx = float(parent["x"]) - float(node["x"])
        dy = float(parent["y"]) - float(node["y"])
        offset = {
            "macro": 16,
            "cluster": 14,
            "section": 12,
            "endpoint": 14,
        }.get(level, 12)
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

    def edge_display_text(node: dict) -> str | list[str]:
        """Render detailed edge descriptions as readable multiline labels."""
        description = _normalise_term(node.get("detailed_description", ""))
        if node.get("level") != "edge" or not description:
            return node["label"]
        return textwrap.wrap(
            description,
            width=46,
            break_long_words=False,
            break_on_hyphens=False,
        )

    detail_labels = [
        node for node in hierarchy_nodes
        if node["label"]
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
            "label_colour": (
                cluster_colour(int(node["cluster_id"]))
                if node.get("cluster_id") is not None
                else "#25322a"
            ),
            # Every level competes for screen space at every zoom scale. Broad
            # labels win collisions first, and increasingly detailed labels
            # fill every remaining gap as zooming creates room around their
            # fixed anchors.
            "label_priority": {
                "macro": 4,
                "cluster": 3,
                "section": 2,
                "edge": 1,
            }[node["level"]],
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
        }
        for node in detail_labels
    ]
    label_rows.extend([
        {
            "node_id": "root",
            "level": "root",
            "x": hierarchy["roots"][0]["x"],
            "y": hierarchy["roots"][0]["y"],
            "label": hierarchy["roots"][0]["label"],
            "display_label": hierarchy["roots"][0]["label"],
            "cluster_id": None,
            "label_colour": "#151915",
            # The article is the final overview label. Once the geometry has
            # collapsed at extreme zoom-out, it must win over Start, End, and
            # every thematic detail occupying the same screen-space point.
            "label_priority": 7,
            "importance": hierarchy["roots"][0]["importance"],
            "revision_count": hierarchy["roots"][0]["revision_count"],
            "parent_id": None,
            "ancestor_ids": [],
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
            "label_colour": "#263029",
            "label_priority": 6,
            "importance": 1e12,
            "revision_count": 1,
            "parent_id": "root",
            "ancestor_ids": ["root"],
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
            "label_colour": "#263029",
            "label_priority": 6,
            "importance": 1e12,
            "revision_count": 1,
            "parent_id": "root",
            "ancestor_ids": ["root"],
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
            # Filled from the actually rendered label marks by render_html().
            # A large sort bonus keeps previously visible labels stable across
            # small zoom and pan changes without introducing zoom stages.
            {"name": "stableParentLabelIds", "value": []},
            {"name": "stableEdgeLabelIds", "value": []},
            # Filled by a post-render screen-space bounds audit. Labels have one
            # fixed anchor position; these signals hide a label when that
            # position collides instead of searching another side.
            {"name": "blockedParentLabelIds", "value": []},
            {"name": "blockedEdgeLabelIds", "value": []},
            # Filled by the screen-space collision audit. Detailed edge text
            # is preferred whenever its measured bounds fit; otherwise the
            # short label is used without relying on a zoom threshold.
            {"name": "detailedEdgeLabelIds", "value": []},
        ],
        "data": [
            {"name": "points", "values": point_rows},
            {"name": "normalRevisionPoints", "source": "points"},
            {"name": "labelledSectionPoints", "source": "points", "transform": [
                {
                    "type": "filter",
                    "expr": "datum.labelled_section_point",
                },
            ]},
            {"name": "edges", "values": edge_rows},
            {"name": "exactRevertEdges", "source": "edges", "transform": [
                {"type": "filter", "expr": "datum.has_exact_revert"},
            ]},
            {"name": "visibleDirectionArrows", "source": "edges", "transform": [
                {"type": "filter", "expr": "!datum.has_exact_revert"},
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
                    "as": "visibleScreenLength",
                    "expr": "(datum.clipT1-datum.clipT0)*datum.screenLength",
                },
                {"type": "filter", "expr": "datum.visibleScreenLength>=34"},
                {
                    "type": "formula",
                    "as": "arrowT",
                    "expr": (
                        "datum.clipT0 + "
                        "min(30,datum.visibleScreenLength*0.5)/datum.screenLength"
                    ),
                },
                {"type": "formula", "as": "arrowScreenX", "expr": "datum.screenX1+datum.screenDx*datum.arrowT"},
                {"type": "formula", "as": "arrowScreenY", "expr": "datum.screenY1+datum.screenDy*datum.arrowT"},
                {"type": "formula", "as": "arrowAngle", "expr": "atan2(datum.screenDy,datum.screenDx)*57.29577951308232"},
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
                        "datum.is_edit_war ? "
                        "min(190,max(22,datum.screenLength*"
                        "(0.25+0.075*datum.arc_level)+8*datum.arc_level)) : "
                        "datum.revert_kind==='exact_sha1' ? "
                        "min(190,max(22,datum.screenLength*0.28+"
                        "8*datum.arc_level)) : "
                        "min(95,max(10,datum.screenLength*0.12+"
                        "4*datum.arc_level))"
                    ),
                },
                {
                    "type": "formula",
                    "as": "controlX",
                    "expr": (
                        "(datum.screenX1+datum.screenX2)/2 - "
                        "datum.screenDy/datum.screenLength*datum.arcHeight"
                    ),
                },
                {
                    "type": "formula",
                    "as": "controlY",
                    "expr": (
                        "(datum.screenY1+datum.screenY2)/2 + "
                        "datum.screenDx/datum.screenLength*datum.arcHeight"
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
            {"name": "intraClusterEdges", "source": "edges", "transform": [
                {"type": "filter", "expr": "datum.same_cluster"},
            ]},
            {"name": "labels", "values": label_rows},
            {"name": "clusterHulls", "values": hull_rows},
            {"name": "visibleParentLabels", "source": "labels", "transform": [
                {"type": "filter", "expr": "rootOnlyOverview ? datum.level==='root' : (datum.level==='root' || datum.level==='macro' || datum.level==='endpoint' || datum.level==='cluster' || datum.level==='section')"},
                {"type": "filter", "expr": "datum.x >= min(xdom[0], xdom[1]) && datum.x <= max(xdom[0], xdom[1]) && datum.y >= min(ydom[0], ydom[1]) && datum.y <= max(ydom[0], ydom[1])"},
                {"type": "filter", "expr": "datum.level!=='root' || (scale('xscale', datum.x) >= 95 && scale('xscale', datum.x) <= width-95 && scale('yscale', datum.y) >= 24 && scale('yscale', datum.y) <= height-24)"},
                {"type": "formula", "as": "placementOrder", "expr": "-(datum.label_priority*1e15 + (indexof(stableParentLabelIds, datum.node_id)>=0 ? 1e14 : 0) + datum.importance)"},
                {"type": "collect", "sort": {"field": "placementOrder", "order": "ascending"}},
            ]},
            {"name": "visibleEdgeLabels", "source": "labels", "transform": [
                {"type": "filter", "expr": "datum.level==='edge'"},
                {"type": "filter", "expr": "!rootOnlyOverview"},
                {"type": "formula", "as": "screenX1", "expr": "scale('xscale', datum.edge_x)"},
                {"type": "formula", "as": "screenY1", "expr": "scale('yscale', datum.edge_y)"},
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
                {"type": "formula", "as": "labelT", "expr": "(datum.clipT0+datum.clipT1)/2"},
                {"type": "formula", "as": "labelScreenX", "expr": "datum.screenX1+datum.screenDx*datum.labelT"},
                {"type": "formula", "as": "labelScreenY", "expr": "datum.screenY1+datum.screenDy*datum.labelT"},
                {"type": "formula", "as": "placementOrder", "expr": "-(datum.label_priority*1e15 + (indexof(stableEdgeLabelIds, datum.node_id)>=0 ? 1e14 : 0) + datum.importance)"},
                {"type": "collect", "sort": {"field": "placementOrder", "order": "ascending"}},
            ]},
        ],
        "scales": [
            {"name": "xscale", "zero": False, "domain": {"signal": "xdom"}, "range": "width"},
            {"name": "yscale", "zero": False, "domain": {"signal": "ydom"}, "range": "height"},
            {"name": "clusterColor", "type": "ordinal", "domain": cluster_ids, "range": [cluster_colour(cluster_id) for cluster_id in cluster_ids]},
        ],
        "marks": [
            {
                "name": "clusterRegions", "type": "group",
                "from": {"facet": {"name": "clusterHull", "data": "clusterHulls", "groupby": "cluster_id"}},
                "marks": [{
                    "type": "line", "from": {"data": "clusterHull"},
                    "encode": {"update": {
                        "x": {"scale": "xscale", "field": "x"},
                        "y": {"scale": "yscale", "field": "y"},
                        "order": {"field": "order"},
                        "interpolate": {"value": "linear"},
                        "fill": {"scale": "clusterColor", "field": "cluster_id"},
                        "fillOpacity": {"value": 0.075},
                        "stroke": {"scale": "clusterColor", "field": "cluster_id"},
                        "strokeOpacity": {"value": 0.38},
                        "strokeWidth": {"value": 1.4},
                    }}
                }],
            },
            {
                "name": "trajectory",
                "type": "rule",
                "from": {"data": "edges"},
                "encode": {"update": {
                    "x": {"scale": "xscale", "field": "x"},
                    "y": {"scale": "yscale", "field": "y"},
                    "x2": {"scale": "xscale", "field": "x2"},
                    "y2": {"scale": "yscale", "field": "y2"},
                    "stroke": {"value": TRAJECTORY_COLOR},
                    "strokeOpacity": {"value": 0.42},
                    "strokeWidth": {"signal": TRAJECTORY_WIDTH_SIGNAL},
                }}
            },
            {
                "name": "intraClusterTrajectory",
                "type": "rule", "from": {"data": "intraClusterEdges"},
                "encode": {"update": {
                    "x": {"scale": "xscale", "field": "x"},
                    "y": {"scale": "yscale", "field": "y"},
                    "x2": {"scale": "xscale", "field": "x2"},
                    "y2": {"scale": "yscale", "field": "y2"},
                    "stroke": {"scale": "clusterColor", "field": "cluster_id"},
                    "strokeOpacity": {"value": 0.52},
                    "strokeWidth": {"signal": TRAJECTORY_WIDTH_SIGNAL},
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
                                "{'Type': datum.revert_kind_label, "
                                "'Revert revision': datum.revision_id, "
                                "'Undone revision': datum.reverted_revision_id, "
                                "'Restored revision': datum.restored_revision_id, "
                                "'Return anchor': datum.return_anchor_revision_id, "
                                "'Points at return anchor': datum.target_overlap_count, "
                                "'Editors': datum.pair_text, "
                                "'Edit War': datum.edit_war_label}"
                            )
                        },
                    },
                    "update": {
                        "path": {"field": "arcPath"},
                        "fillOpacity": {"value": 0},
                        "stroke": {
                            "signal": (
                                f"datum.is_edit_war ? '{EDIT_WAR_COLOR}' : "
                                "(datum.revert_kind==='exact_sha1' ? "
                                f"'{EXACT_REVERT_COLOR}' : '{PARTIAL_REVERT_COLOR}')"
                            )
                        },
                        "strokeOpacity": {
                            "signal": "datum.is_edit_war ? 0.9 : 0.78"
                        },
                        "strokeWidth": {
                            "signal": "datum.is_edit_war ? 2.5 : 2"
                        },
                        "strokeDash": {"value": [8, 5]},
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
                    "size": {"signal": "datum.is_edit_war ? 54 : 46"},
                    "fill": {
                        "signal": (
                            f"datum.is_edit_war ? '{EDIT_WAR_COLOR}' : "
                            "(datum.revert_kind==='exact_sha1' ? "
                            f"'{EXACT_REVERT_COLOR}' : '{PARTIAL_REVERT_COLOR}')"
                        )
                    },
                    "fillOpacity": {"value": 0.96},
                    "stroke": {"value": BACKGROUND_COLOR},
                    "strokeOpacity": {"value": 0.96},
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
                    "fill": {
                        "signal": (
                            "datum.same_cluster ? "
                            f"scale('clusterColor', datum.cluster_id) : '{TRAJECTORY_COLOR}'"
                        )
                    },
                    "fillOpacity": {"value": 0.9},
                    "stroke": {"value": BACKGROUND_COLOR},
                    "strokeOpacity": {"value": 0.95},
                    "strokeWidth": {"value": 0.8},
                }},
            },
            {
                "name": "revisionPoints", "type": "symbol", "from": {"data": "normalRevisionPoints"},
                "encode": {"enter": {
                    "tooltip": {"signal": "{'Revision ID': datum.revision_id, 'Timestamp': datum.timestamp, 'Author': datum.author, 'Cluster': datum.cluster_label}"}
                }, "update": {
                    "x": {"scale": "xscale", "field": "x"}, "y": {"scale": "yscale", "field": "y"},
                    "size": {"value": 120},
                    "fill": {"scale": "clusterColor", "field": "cluster_id"},
                    "fillOpacity": {"signal": "0.62 + 0.24*min(zoomLevel/4,1)"},
                    "stroke": {"value": BACKGROUND_COLOR}, "strokeWidth": {"value": 1.2},
                }}
            },
            {
                "name": "sectionPointHighlights", "type": "symbol",
                "interactive": False,
                "from": {"data": "labelledSectionPoints"},
                "encode": {"update": {
                    "x": {"scale": "xscale", "field": "x"},
                    "y": {"scale": "yscale", "field": "y"},
                    "size": {"value": 270},
                    "fillOpacity": {"value": 0},
                    "stroke": {"scale": "clusterColor", "field": "cluster_id"},
                    "strokeOpacity": {"value": 0.92},
                    "strokeWidth": {"value": 2.6},
                }}
            },
            {
                "name": "semanticLabels", "type": "text",
                "key": "node_id",
                "zindex": 3,
                "from": {"data": "visibleParentLabels"},
                "encode": {"update": {
                    "text": {"field": "display_label"},
                    "x": {"signal": "scale('xscale', datum.x) + datum.label_dx"},
                    "y": {"signal": "scale('yscale', datum.y) + datum.label_dy"},
                    "align": {"field": "label_align"},
                    "baseline": {"field": "label_baseline"},
                    "opacity": {"signal": "indexof(blockedParentLabelIds, datum.node_id)>=0 ? 0 : 1"},
                    "fillOpacity": {"value": 1},
                    "fill": {"field": "label_colour"},
                    "stroke": {"value": "#ffffff"},
                    "strokeWidth": {"value": 4},
                    "strokeOpacity": {"value": 1},
                    "font": {"value": "Inter, Arial, sans-serif"},
                    "fontSize": {"signal": "datum.level==='root' ? 28 : datum.level==='macro' ? 19 : datum.level==='cluster' ? 16 : datum.level==='section' ? 12 : 14"},
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
                    "x": {"signal": "datum.labelScreenX - 9*datum.screenDy/max(sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2)),1)"},
                    "y": {"signal": "datum.labelScreenY + 9*datum.screenDx/max(sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2)),1)"},
                    "align": {"value": "center"},
                    "baseline": {"value": "middle"},
                    "font": {"value": "Inter, Arial, sans-serif"},
                    "fontSize": {"value": 8},
                    "lineHeight": {"value": 10},
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
                    "x": {"signal": f"datum.labelScreenX - {EDGE_DETAILED_LABEL_OFFSET}*datum.screenDy/max(sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2)),1)"},
                    "y": {"signal": f"datum.labelScreenY + {EDGE_DETAILED_LABEL_OFFSET}*datum.screenDx/max(sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2)),1)"},
                    "align": {"value": "center"},
                    "baseline": {"value": "middle"},
                    "font": {"value": "Inter, Arial, sans-serif"},
                    "fontSize": {"value": 8},
                    "lineHeight": {"value": 10},
                    "fontWeight": {"value": 700},
                    "opacity": {"value": 0},
                    "fillOpacity": {"value": 0},
                    "strokeOpacity": {"value": 0},
                    "limit": {"value": 340},
                }}
            },
            {
                "name": "edgeDetailLeaders", "type": "rule",
                "interactive": False,
                "from": {"data": "visibleEdgeLabels"},
                "zindex": 6,
                "encode": {"update": {
                    "x": {"field": "labelScreenX"},
                    "y": {"field": "labelScreenY"},
                    "x2": {"signal": f"datum.labelScreenX - {EDGE_DETAILED_LABEL_OFFSET - 8}*datum.screenDy/max(datum.screenLength,1)"},
                    "y2": {"signal": f"datum.labelScreenY + {EDGE_DETAILED_LABEL_OFFSET - 8}*datum.screenDx/max(datum.screenLength,1)"},
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
                "key": "node_id",
                "zindex": 7,
                "from": {"data": "visibleEdgeLabels"},
                "encode": {"update": {
                    "text": {
                        "signal": (
                            "indexof(detailedEdgeLabelIds, datum.node_id)>=0 ? "
                            "datum.display_label : datum.label"
                        )
                    },
                    "x": {"signal": f"datum.labelScreenX - (indexof(detailedEdgeLabelIds, datum.node_id)>=0 ? {EDGE_DETAILED_LABEL_OFFSET} : {EDGE_SHORT_LABEL_OFFSET})*datum.screenDy/max(sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2)),1)"},
                    "y": {"signal": f"datum.labelScreenY + (indexof(detailedEdgeLabelIds, datum.node_id)>=0 ? {EDGE_DETAILED_LABEL_OFFSET} : {EDGE_SHORT_LABEL_OFFSET})*datum.screenDx/max(sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2)),1)"},
                    "align": {"value": "center"},
                    "baseline": {"value": "middle"},
                    "font": {"value": "Inter, Arial, sans-serif"},
                    "fontSize": {"value": 8},
                    "lineHeight": {"value": 10},
                    "fontWeight": {"value": 700},
                    "opacity": {"signal": "indexof(blockedEdgeLabelIds, datum.node_id)>=0 ? 0 : 1"},
                    "fillOpacity": {"value": 1},
                    "limit": {"value": 340},
                    "fill": {"value": "#4f5750"},
                    "stroke": {"value": "#ffffff"},
                    "strokeWidth": {"value": 2},
                    "strokeOpacity": {"value": 1},
                }}
            },
        ],
    }


def render_html(article: str, spec: dict) -> str:
    encoded = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    cluster_scale = next(
        scale for scale in spec.get("scales", [])
        if scale.get("name") == "clusterColor"
    )
    cluster_legend = "".join(
        (
            '<li><span class="legend-dot" style="--legend-color:'
            f'{html.escape(str(colour))}"></span><span>'
            f'{"Noise" if int(cluster_id) == -1 else f"Cluster {int(cluster_id) + 1}"}'
            "</span></li>"
        )
        for cluster_id, colour in zip(
            cluster_scale.get("domain", []), cluster_scale.get("range", [])
        )
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Zoom Timecurve – {html.escape(article)}</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5.33.1"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-tooltip@0.35.2/build/vega-tooltip.min.js"></script>
<style>
html,body{{height:100%;margin:0;background:{BACKGROUND_COLOR};color:#151915;font-family:Inter,Arial,sans-serif}}
#app{{width:min(100%,1200px);height:min(100%,720px);margin:0 auto;display:flex;flex-direction:column;overflow:hidden;box-sizing:border-box}}
#topbar{{padding:14px 20px 8px;display:flex;align-items:center;gap:22px;flex-wrap:wrap}}
#topbar h1{{font-size:20px;line-height:1.2;font-weight:700;margin:0}}
#content{{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 250px;border-top:1px solid #e2e5df}}
#vis{{flex:1;min-width:0;min-height:0}}
#vis svg{{cursor:grab}}
#vis svg:active{{cursor:grabbing}}
#vis svg .semanticLabels text,#vis svg .edgeLabels text{{paint-order:stroke fill;stroke-linejoin:round}}
#legend{{min-width:0;overflow:auto;border-left:1px solid #d9ddd6;padding:16px 16px 22px;background:#f7f6f2;box-sizing:border-box}}
#legend h2{{font-size:17px;margin:0 0 15px}}
#legend h3{{font-size:12px;line-height:1.2;letter-spacing:.06em;text-transform:uppercase;color:#687068;margin:18px 0 9px}}
.legend-list{{list-style:none;margin:0;padding:0;display:grid;gap:8px}}
.cluster-list{{grid-template-columns:1fr 1fr;column-gap:8px}}
.legend-list li{{min-width:0;display:flex;align-items:center;gap:8px;font-size:12px;line-height:1.25}}
.legend-dot{{--legend-color:{NOISE_COLOR};width:11px;height:11px;box-sizing:border-box;border-radius:50%;background:var(--legend-color);border:1px solid {BACKGROUND_COLOR};box-shadow:0 0 0 1px color-mix(in srgb,var(--legend-color) 75%,#777);flex:none}}
.legend-ring{{width:14px;height:14px;box-sizing:border-box;border:2px solid {NOISE_COLOR};border-radius:50%;background:transparent;flex:none}}
.legend-line{{width:30px;height:0;border-top:2.25px solid {TRAJECTORY_COLOR};flex:none}}
.legend-line.thick{{border-top-width:4.45px}}
.legend-arc{{width:30px;height:10px;box-sizing:border-box;border:2px dashed {EXACT_REVERT_COLOR};border-bottom:0;border-radius:50% 50% 0 0;flex:none}}
.legend-partial{{width:30px;height:10px;box-sizing:border-box;border:2px dashed {PARTIAL_REVERT_COLOR};border-bottom:0;border-radius:50% 50% 0 0;flex:none}}
.legend-spiral{{width:30px;height:10px;box-sizing:border-box;border:2px dashed {EDIT_WAR_COLOR};border-bottom:0;border-radius:50% 50% 0 0;flex:none}}
.legend-arrow{{width:30px;flex:none;text-align:center;color:{TRAJECTORY_COLOR};font-size:23px;font-weight:800;line-height:10px}}
@media(max-width:760px){{#content{{grid-template-columns:minmax(0,1fr) 205px}}#legend{{padding:12px 10px}}.cluster-list{{grid-template-columns:1fr}}}}
</style></head><body><main id="app">
<header id="topbar"><h1>Meaningful Projection Space</h1></header>
<div id="content">
<div id="vis" role="img" aria-label="Interactive semantic revision timecurve"></div>
<aside id="legend" aria-label="Visualization legend">
<h2>Legend</h2>
<h3>Display</h3>
<ul class="legend-list">
<li><span class="legend-dot"></span><span>Revision point (cluster colour)</span></li>
<li><span class="legend-ring"></span><span>Temporal-section point (cluster colour)</span></li>
<li><span class="legend-line"></span><span>Standard trajectory edge</span></li>
<li><span class="legend-arrow">&#8594;</span><span>Revision direction</span></li>
<li><span class="legend-line thick"></span><span>Repeated projected path</span></li>
<li><span class="legend-arc"></span><span>Exact revert return</span></li>
<li><span class="legend-partial"></span><span>Partial revert (selected text restored; unaffected text remains)</span></li>
<li><span class="legend-spiral"></span><span>Edit War</span></li>
</ul>
<h3>Cluster</h3>
<ul class="legend-list cluster-list">{cluster_legend}</ul>
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
let stabilityTimer=0;
let stabilitySyncing=false;
let stabilityPending=false;
let lastWidth=0;
let lastHeight=0;
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
          labelKey:normalizedLabelKey(datum.label),
          priority:Number(datum.label_priority)||0,
          importance:Number(datum.importance)||0,
          edgeScreenLength:Number(datum.screenLength)||0,
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
const labelCollisionPadding=4;
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
  constructor(cellSize=48){{
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
function collisionBlocks(){{
  const stableParents=new Set(view.signal('stableParentLabelIds')||[]);
  const stableEdges=new Set(view.signal('stableEdgeLabelIds')||[]);
  const compareCandidates=(first,second)=>{{
    const priorityDifference=second.priority-first.priority;
    if(priorityDifference)return priorityDifference;
    const firstStable=first.kind==='parent'?stableParents.has(first.id):stableEdges.has(first.id);
    const secondStable=second.kind==='parent'?stableParents.has(second.id):stableEdges.has(second.id);
    if(firstStable!==secondStable)return secondStable-firstStable;
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
  const occupied=new LabelSpatialHash();
  const acceptedParentLabelKeys=new Set();
  const acceptedShortEdgeLabelKeys=new Set();
  const blockedParents=[];
  const blockedEdges=[];
  const detailedEdgeIds=[];
  for(const candidate of parents){{
    if(outsideViewport(candidate)||occupied.overlaps(candidate)){{
      blockedParents.push(candidate.id);
    }}else{{
      occupied.insert(candidate);
      if(candidate.labelKey)acceptedParentLabelKeys.add(candidate.labelKey);
    }}
  }}
  for(const shortCandidate of shortEdges){{
    const detailedCandidate=detailedEdges.get(shortCandidate.id);
    const detailedFits=detailedCandidate&&detailedCandidate.hasDetailed&&
      shortCandidate.edgeScreenLength>={MIN_DETAILED_EDGE_SCREEN_LENGTH}&&
      !outsideViewport(detailedCandidate)&&
      !occupied.overlaps(detailedCandidate);
    const shortTextIsRedundant=Boolean(shortCandidate.labelKey)&&(
      acceptedParentLabelKeys.has(shortCandidate.labelKey)||
      acceptedShortEdgeLabelKeys.has(shortCandidate.labelKey)
    );
    // A broad visible region label already explains repeated short edge text.
    // Suppress that repetition, but let the edge return as its concrete full
    // revision description as soon as zooming creates enough screen space.
    if(detailedFits){{
      occupied.insert(detailedCandidate);
      detailedEdgeIds.push(shortCandidate.id);
    }}else if(shortTextIsRedundant){{
      blockedEdges.push(shortCandidate.id);
    }}else if(!outsideViewport(shortCandidate)&&
              !occupied.overlaps(shortCandidate)){{
      occupied.insert(shortCandidate);
      if(shortCandidate.labelKey){{
        acceptedShortEdgeLabelKeys.add(shortCandidate.labelKey);
      }}
    }}else{{
      blockedEdges.push(shortCandidate.id);
    }}
  }}
  return {{
    parentIds:Array.from(new Set(blockedParents)).sort(),
    edgeIds:Array.from(new Set(blockedEdges)).sort(),
    detailedEdgeIds:Array.from(new Set(detailedEdgeIds)).sort()
  }};
}}
async function syncLabelStability(){{
  if(view.signal('down'))return;
  if(stabilitySyncing){{
    stabilityPending=true;
    return;
  }}
  stabilitySyncing=true;
  try{{
    let blocked=collisionBlocks();
    let currentBlockedParents=(view.signal('blockedParentLabelIds')||[]).slice().sort();
    let currentBlockedEdges=(view.signal('blockedEdgeLabelIds')||[]).slice().sort();
    let currentDetailedEdges=(view.signal('detailedEdgeLabelIds')||[]).slice().sort();
    if(!sameIds(blocked.parentIds,currentBlockedParents)||
       !sameIds(blocked.edgeIds,currentBlockedEdges)||
       !sameIds(blocked.detailedEdgeIds,currentDetailedEdges)){{
      view.signal('blockedParentLabelIds',blocked.parentIds);
      view.signal('blockedEdgeLabelIds',blocked.edgeIds);
      view.signal('detailedEdgeLabelIds',blocked.detailedEdgeIds);
      await view.runAsync();
    }}
    let parentIds=renderedLabelIds('semanticLabels');
    let edgeIds=renderedLabelIds('edgeLabels');
    const oldParentIds=(view.signal('stableParentLabelIds')||[]).slice().sort();
    const oldEdgeIds=(view.signal('stableEdgeLabelIds')||[]).slice().sort();
    if(!sameIds(parentIds,oldParentIds)||!sameIds(edgeIds,oldEdgeIds)){{
      view.signal('stableParentLabelIds',parentIds);
      view.signal('stableEdgeLabelIds',edgeIds);
      await view.runAsync();
    }}
    // Audit once more after the stability order update. Collision-hidden
    // candidates remain measurable, so a former loser can return as soon as
    // it has sufficient room at its fixed anchor.
    const residual=collisionBlocks();
    if(!sameIds(residual.parentIds,blocked.parentIds)||
       !sameIds(residual.edgeIds,blocked.edgeIds)||
       !sameIds(residual.detailedEdgeIds,blocked.detailedEdgeIds)){{
      blocked=residual;
      view.signal('blockedParentLabelIds',blocked.parentIds);
      view.signal('blockedEdgeLabelIds',blocked.edgeIds);
      view.signal('detailedEdgeLabelIds',blocked.detailedEdgeIds);
      await view.runAsync();
      parentIds=renderedLabelIds('semanticLabels');
      edgeIds=renderedLabelIds('edgeLabels');
      view.signal('stableParentLabelIds',parentIds);
      view.signal('stableEdgeLabelIds',edgeIds);
      await view.runAsync();
    }}
    vis.dataset.zoom=Number(view.signal('zoomLevel')).toFixed(3);
    vis.dataset.visibleLabels=String(parentIds.length+edgeIds.length);
    vis.dataset.stableParentLabels=String(parentIds.length);
    vis.dataset.stableEdgeLabels=String(edgeIds.length);
    vis.dataset.blockedParentLabels=String(blocked.parentIds.length);
    vis.dataset.blockedEdgeLabels=String(blocked.edgeIds.length);
    vis.dataset.detailedEdgeLabels=String(blocked.detailedEdgeIds.length);
    vis.dataset.shortEdgeLabels=String(
      Math.max(0,edgeIds.length-blocked.detailedEdgeIds.length)
    );
  }}finally{{
    stabilitySyncing=false;
    if(stabilityPending){{
      stabilityPending=false;
      scheduleLabelStability();
    }}
  }}
}}
function scheduleLabelStability(){{
  clearTimeout(stabilityTimer);
  stabilityTimer=setTimeout(()=>{{void syncLabelStability();}},0);
}}
view.addSignalListener('zooming',(_name,value)=>{{if(!value)scheduleLabelStability();}});
view.addSignalListener('zoomLevel',()=>{{scheduleLabelStability();}});
view.addSignalListener('down',(_name,value)=>{{if(value===null)scheduleLabelStability();}});
function resizeView(){{
  cancelAnimationFrame(resizeFrame);
  resizeFrame=requestAnimationFrame(async()=>{{
    const width=Math.max(320,vis.clientWidth-56);
    const height=Math.max(420,vis.clientHeight-46);
    if(width===lastWidth&&height===lastHeight)return;
    lastWidth=width;lastHeight=height;
    await view.width(width).height(height).runAsync();
    scheduleLabelStability();
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
    semantic_vectors = None
    if embedding_path.exists():
        embeddings = np.load(embedding_path)
        if len(embeddings) != len(points):
            raise ValueError(
                "Revisions-Embeddings passen nicht zu den Projektionspunkten: "
                f"{len(embeddings)} statt {len(points)}."
            )
        semantic_vectors = {
            str(point["revision_id"]): vector
            for point, vector in zip(points, embeddings)
        }

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
    revert_analysis = projection.get("edit_wars")
    if (
        not revert_analysis
        or revert_analysis.get("analysis_version") != REVERT_ANALYSIS_VERSION
        or not all(
            "interval_mutual_revert_count" in point for point in points
        )
    ):
        full_history = load_revert_history(xml_path, include_text_changes=True)
        conflict_history, sampled_for_conflict = history_window_for_points(
            full_history, points
        )
        revert_analysis = analyze_reverts(
            conflict_history,
            sampled_for_conflict,
            semantic_vectors=semantic_vectors,
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
        "edit_wars": revert_analysis,
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
