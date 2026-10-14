#!/usr/bin/env python3
"""Overlay edge descriptions and partial labels onto an existing semantic map."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from semantic_zoom import (
    build_vega_spec,
    hierarchy_specificity_issues,
    render_html,
)


NO_CONTENT_LABELS = {
    "no content change",
    "no cleaned text change",
    "no content difference",
    "unchanged content",
    "content unchanged",
}
CHANGE_ACTIONS = (
    (r"\badded\b|\binserted\b", "Added"),
    (r"\bremoved\b|\bdeleted\b", "Removed"),
    (r"\brestored\b", "Restored"),
    (r"\breworded\b|\brewrote\b", "Reworded"),
    (r"\breplaced\b", "Replaced"),
    (r"\bcorrected\b", "Corrected"),
    (r"\bchanged\b", "Changed"),
    (r"\brelocated\b|\bmoved\b", "Moved"),
    (r"\breordered\b", "Reordered"),
    (r"\bretitled\b", "Retitled"),
    (r"\brenamed\b", "Renamed"),
    (r"\bexpanded\b", "Expanded"),
    (r"\bshortened\b", "Shortened"),
    (r"\bcapitalized\b", "Capitalized"),
    (r"\blowercased\b", "Lowercased"),
    (r"\breformatted\b", "Reformatted"),
    (r"\bstandardized\b", "Standardized"),
    (r"\breduced\b", "Reduced"),
    (r"\breworked\b", "Reworked"),
    (r"\bupdated\b|\bmodified\b", "Updated"),
)
CHANGE_ACTION_WORDS = {action.casefold() for _, action in CHANGE_ACTIONS}


def _normalised_key(value: object) -> str:
    return re.sub(r"[^\w]+", " ", str(value or "").casefold()).strip()


def _description_reports_no_change(description: object) -> bool:
    text = _normalised_key(description)
    return any(
        phrase in text
        for phrase in (
            "did not change",
            "no change to the cleaned article content",
            "supplied cleaned article content did not change",
            "cleaned article content is identical",
            "cleaned article content was unchanged",
        )
    )


def normalise_unchanged_edges(
    hierarchy: dict, metadata_payload: dict
) -> list[str]:
    """Make verified neutral labels and descriptions internally consistent."""
    metadata_by_id = {
        str(row.get("revision_id", "")): row
        for row in metadata_payload.get("revisions", [])
    }
    updated: list[str] = []
    for edge in hierarchy.get("edges", []):
        label_is_neutral = _normalised_key(edge.get("label")) in NO_CONTENT_LABELS
        description_is_neutral = _description_reports_no_change(
            edge.get("detailed_description")
        )
        if not label_is_neutral and not description_is_neutral:
            continue
        current_id = str(edge.get("current_revision_id", ""))
        metadata = metadata_by_id.get(current_id)
        if metadata is None:
            raise ValueError(
                f"Metadaten der unveränderten Revision {current_id} fehlen."
            )
        author = " ".join(str(metadata.get("author") or "Unknown editor").split())
        timestamp = " ".join(
            str(metadata.get("timestamp") or "unknown time").split()
        )
        edge["topic_label"] = edge.get("topic_label") or edge.get("label")
        edge["label"] = "No Content Change"
        edge["short_label"] = "No Content Change"
        edge["detailed_description"] = (
            f"On {timestamp}, {author} made a revision, and the supplied cleaned "
            "article content did not change."
        )
        edge["label_source"] = "verified_identical_cleaned_revision"
        edge["detailed_description_source"] = (
            "verified_identical_cleaned_revision"
        )
        edge["cleaned_text_unchanged"] = True
        updated.append(str(edge.get("node_id", "")))
    return updated


def projection_metadata_payload(
    projection: dict, override_payload: dict | None = None
) -> dict:
    """Build revision metadata from projection points with an optional overlay."""
    rows = [
        {
            "revision_id": str(point.get("revision_id", "")),
            "author": str(point.get("author", "")),
            "timestamp": str(point.get("timestamp", "")),
            "comment": "",
        }
        for point in projection.get("points", [])
        if point.get("revision_id")
    ]
    if override_payload is None:
        return {"source": "projection", "revisions": rows}
    overrides = {
        str(row.get("revision_id", "")): row
        for row in override_payload.get("revisions", [])
        if row.get("revision_id")
    }
    for row in rows:
        override = overrides.get(row["revision_id"])
        if override is not None:
            row.update(override)
    return {"source": "projection with optional metadata override", "revisions": rows}


def make_change_specific_edge_labels(hierarchy: dict) -> list[str]:
    """Add a concrete edit direction to every non-neutral short edge label."""
    updated: list[str] = []
    for edge in hierarchy.get("edges", []):
        if _normalised_key(edge.get("label")) in NO_CONTENT_LABELS:
            continue
        base_label = " ".join(
            str(edge.get("topic_label") or edge.get("label") or "").split()
        )
        if not base_label:
            continue
        base_words = set(_normalised_key(base_label).split())
        if base_words & CHANGE_ACTION_WORDS:
            continue
        description = str(edge.get("detailed_description") or "")
        action = next(
            (
                action
                for pattern, action in CHANGE_ACTIONS
                if re.search(pattern, description, re.IGNORECASE)
            ),
            "Changed",
        )
        edge["topic_label"] = base_label
        edge["label"] = f"{base_label} {action}"
        edge["short_label"] = edge["label"]
        edge["label_source"] = "change_specific_from_detailed_description"
        updated.append(str(edge.get("node_id", "")))
    return updated


def _edge_by_revision_pair(hierarchy: dict) -> dict[tuple[str, str], dict]:
    return {
        (
            str(edge.get("previous_revision_id", "")),
            str(edge.get("current_revision_id", "")),
        ): edge
        for edge in hierarchy.get("edges", [])
    }


def apply_edge_descriptions(
    hierarchy: dict, description_payload: dict
) -> list[str]:
    """Add detailed descriptions without changing the hierarchy's short labels."""
    edge_by_pair = _edge_by_revision_pair(hierarchy)
    updated: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in description_payload.get("revision_pairs", []):
        key = (
            str(pair.get("previous_revision_id", "")),
            str(pair.get("current_revision_id", "")),
        )
        if not all(key):
            raise ValueError(
                "Eine ausführliche Kantenbeschreibung enthält keine vollständige "
                "Revisions-ID."
            )
        if key in seen_pairs:
            raise ValueError(
                f"Doppelte ausführliche Kantenbeschreibung für {key[0]} -> {key[1]}."
            )
        seen_pairs.add(key)
        edge = edge_by_pair.get(key)
        if edge is None:
            raise ValueError(
                f"Beschriebene Kante {key[0]} -> {key[1]} fehlt in der Hierarchie."
            )
        description = " ".join(
            str(pair.get("detailed_description", "")).split()
        )
        if not description:
            raise ValueError(
                f"Kante {key[0]} -> {key[1]} hat keine ausführliche Beschreibung."
            )
        edge["detailed_description"] = description
        edge["detailed_description_source"] = "full_revision_metadata_llm"
        updated.append(str(edge.get("node_id", "")))
    if not updated:
        raise ValueError("Die Datei enthält keine ausführlichen Kantenbeschreibungen.")
    return updated


def apply_edge_labels(hierarchy: dict, edge_label_payload: dict) -> list[str]:
    edge_by_pair = _edge_by_revision_pair(hierarchy)
    updated: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in edge_label_payload.get("revision_pairs", []):
        key = (
            str(pair.get("previous_revision_id", "")),
            str(pair.get("current_revision_id", "")),
        )
        if not all(key):
            raise ValueError("Ein Preview-Label enthält keine vollständige Revisions-ID.")
        if key in seen_pairs:
            raise ValueError(f"Doppeltes Preview-Label für {key[0]} -> {key[1]}.")
        seen_pairs.add(key)
        edge = edge_by_pair.get(key)
        if edge is None:
            raise ValueError(
                f"Preview-Kante {key[0]} -> {key[1]} fehlt in der Hierarchie."
            )
        label = " ".join(str(pair.get("edge_label", "")).split())
        description = " ".join(
            str(pair.get("detailed_description", "")).split()
        )
        if not label or not description:
            raise ValueError(
                f"Preview-Kante {key[0]} -> {key[1]} hat kein vollständiges Label."
            )
        edge["label"] = label
        edge["short_label"] = label
        edge["detailed_description"] = description
        edge["detailed_description_source"] = "full_revision_metadata_llm_preview"
        edge["label_source"] = "full_revision_metadata_llm_preview"
        updated.append(str(edge.get("node_id", "")))
    if not updated:
        raise ValueError("Die Preview-Datei enthält keine Kantenlabels.")
    return updated


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply partial edge labels without regenerating parent labels."
    )
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--detailed-descriptions", type=Path)
    parser.add_argument(
        "--revision-metadata",
        type=Path,
        help="Optional metadata JSON override; projection metadata is used by default.",
    )
    parser.add_argument(
        "--edge-labels", type=Path, action="append", required=True,
        help="Repeat for ordered correction overlays; later files win.",
    )
    parser.add_argument("--html-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projection_path = args.projection.expanduser().resolve()
    hierarchy_path = args.hierarchy.expanduser().resolve()
    description_path = (
        args.detailed_descriptions.expanduser().resolve()
        if args.detailed_descriptions
        else None
    )
    metadata_path = (
        args.revision_metadata.expanduser().resolve()
        if args.revision_metadata
        else None
    )
    edge_label_paths = [path.expanduser().resolve() for path in args.edge_labels]
    html_output = args.html_output.expanduser().resolve()

    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    metadata_override = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path is not None
        else None
    )
    metadata_payload = projection_metadata_payload(projection, metadata_override)
    hierarchy = payload.get("hierarchy")
    if not isinstance(hierarchy, dict):
        raise ValueError("Die Zieldatei enthält keine gültige Labelhierarchie.")

    description_updated: list[str] = []
    if description_path is not None:
        description_payload = json.loads(
            description_path.read_text(encoding="utf-8")
        )
        description_updated = apply_edge_descriptions(
            hierarchy, description_payload
        )
        payload["full_edge_description_overlay"] = {
            "source": str(description_path),
            "updated_node_ids": description_updated,
            "updated_edge_count": len(description_updated),
            "short_labels_changed": False,
            "parent_labels_regenerated": False,
        }

    unchanged_updated = normalise_unchanged_edges(hierarchy, metadata_payload)
    change_specific_updated = make_change_specific_edge_labels(hierarchy)

    overlay_updates: list[dict] = []
    article_from_overlay = ""
    for edge_labels_path in edge_label_paths:
        edge_label_payload = json.loads(
            edge_labels_path.read_text(encoding="utf-8")
        )
        updated = apply_edge_labels(hierarchy, edge_label_payload)
        article_from_overlay = str(edge_label_payload.get("article") or "")
        overlay_updates.append({
            "source": str(edge_labels_path),
            "updated_node_ids": updated,
            "updated_edge_count": len(updated),
        })
    payload["partial_edge_label_overlay"] = {
        "overlays": overlay_updates,
        "updated_edge_count": sum(
            item["updated_edge_count"] for item in overlay_updates
        ),
        "parent_labels_regenerated": False,
    }
    payload["cleaned_text_consistency"] = {
        "revision_metadata": str(metadata_path) if metadata_path else None,
        "metadata_source": metadata_payload["source"],
        "normalised_unchanged_node_ids": unchanged_updated,
        "normalised_unchanged_count": len(unchanged_updated),
        "change_specific_node_ids": change_specific_updated,
        "change_specific_count": len(change_specific_updated),
    }
    specificity_issues = hierarchy_specificity_issues(hierarchy)
    if specificity_issues:
        raise ValueError(
            "Die Labelhierarchie wird nicht durchgehend spezifischer: "
            + json.dumps(specificity_issues[:20], ensure_ascii=False)
        )
    article = str(payload.get("article") or article_from_overlay or "Article")
    write_text_atomic(
        hierarchy_path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    write_text_atomic(
        html_output,
        render_html(
            article,
            build_vega_spec(
                article,
                projection,
                hierarchy,
                payload.get("edit_wars"),
            ),
        ),
    )
    print(f"Ergänzte Kantenbeschreibungen: {len(description_updated)}")
    print(f"Normalisierte unveränderte Kanten: {len(unchanged_updated)}")
    print(f"Änderungsspezifische Kurzlabels: {len(change_specific_updated)}")
    print(
        "Aktualisierte Kanten aus Overlays: "
        f"{sum(item['updated_edge_count'] for item in overlay_updates)}"
    )
    print(f"Hierarchie: {hierarchy_path}")
    print(f"Visualisierung: {html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
