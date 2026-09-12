#!/usr/bin/env python3
"""Overlay a partial edge-label result onto an existing semantic map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_zoom import build_vega_spec, render_html


def apply_edge_labels(hierarchy: dict, edge_label_payload: dict) -> list[str]:
    edges = hierarchy.get("edges", [])
    edge_by_pair = {
        (
            str(edge.get("previous_revision_id", "")),
            str(edge.get("current_revision_id", "")),
        ): edge
        for edge in edges
    }
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
    parser.add_argument("--edge-labels", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projection_path = args.projection.expanduser().resolve()
    hierarchy_path = args.hierarchy.expanduser().resolve()
    edge_labels_path = args.edge_labels.expanduser().resolve()
    html_output = args.html_output.expanduser().resolve()

    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    edge_label_payload = json.loads(edge_labels_path.read_text(encoding="utf-8"))
    hierarchy = payload.get("hierarchy")
    if not isinstance(hierarchy, dict):
        raise ValueError("Die Zieldatei enthält keine gültige Labelhierarchie.")

    updated = apply_edge_labels(hierarchy, edge_label_payload)
    payload["partial_edge_label_overlay"] = {
        "source": str(edge_labels_path),
        "updated_node_ids": updated,
        "updated_edge_count": len(updated),
        "parent_labels_regenerated": False,
    }
    article = str(payload.get("article") or edge_label_payload.get("article") or "Article")
    write_text_atomic(
        hierarchy_path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    write_text_atomic(
        html_output,
        render_html(article, build_vega_spec(article, projection, hierarchy)),
    )
    print(f"Aktualisierte Kanten: {len(updated)}")
    print(f"Hierarchie: {hierarchy_path}")
    print(f"Visualisierung: {html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
