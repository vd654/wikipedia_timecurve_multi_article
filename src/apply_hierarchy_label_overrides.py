#!/usr/bin/env python3
"""Apply reviewed parent-label corrections to a cached hierarchy artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semantic_zoom import (
    LABEL_SYSTEM_PROMPT,
    MACRO_LABEL_SYSTEM_PROMPT,
    build_vega_spec,
    render_html,
    source_sha256,
)


def apply_overrides(payload: dict, overrides: dict[str, str]) -> list[str]:
    hierarchy = payload.get("hierarchy")
    if not isinstance(hierarchy, dict):
        raise ValueError("The target JSON has no hierarchy object.")

    nodes = {
        str(node.get("node_id")): node
        for level in ("macros", "clusters", "sections")
        for node in hierarchy.get(level, [])
    }
    unknown = sorted(set(overrides) - set(nodes))
    if unknown:
        raise ValueError(f"Unknown hierarchy node IDs: {unknown}")

    for node_id, label in overrides.items():
        normalized = " ".join(str(label).split())
        if normalized and not 2 <= len(normalized.split()) <= 3:
            raise ValueError(
                f"Override {node_id} must contain two or three words: {normalized!r}"
            )
        nodes[node_id]["label"] = normalized

    cluster_labels = {
        int(node["cluster_id"]): str(node.get("label") or "")
        for node in hierarchy.get("clusters", [])
    }
    for macro in hierarchy.get("macros", []):
        macro["child_labels"] = [
            cluster_labels[int(cluster_id)]
            for cluster_id in macro.get("cluster_ids", [])
            if cluster_labels.get(int(cluster_id))
        ]
    return sorted(overrides)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--html-output", type=Path)
    args = parser.parse_args()

    hierarchy_path = args.hierarchy.expanduser().resolve()
    override_path = args.overrides.expanduser().resolve()
    payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    override_payload = json.loads(override_path.read_text(encoding="utf-8"))
    updated = apply_overrides(payload, override_payload["labels"])
    # The corrections were reviewed under the current content-label rules, so
    # they are a valid cache for the current prompts as well.
    payload["label_system_prompt"] = LABEL_SYSTEM_PROMPT
    payload["macro_label_system_prompt"] = MACRO_LABEL_SYSTEM_PROMPT
    projection_path = (
        args.projection.expanduser().resolve()
        if args.projection
        else Path(payload["projection"]).expanduser().resolve()
    )
    keywords_path = Path(
        payload.get("revision_keywords") or payload["revision_diff_labels"]
    ).expanduser().resolve()
    payload["fixed_parent_labels"] = {
        "status": "reviewed",
        "projection_sha256": source_sha256([projection_path]),
        "revision_keywords_sha256": source_sha256([keywords_path]),
        "label_count": sum(
            bool(node.get("label"))
            for level in ("macros", "clusters", "sections")
            for node in payload["hierarchy"].get(level, [])
        ),
        "source": str(override_path),
    }
    temporary = hierarchy_path.with_suffix(hierarchy_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(hierarchy_path)
    print(f"Updated {len(updated)} hierarchy labels in {hierarchy_path}")
    if args.projection or args.html_output:
        if not args.projection or not args.html_output:
            raise ValueError("--projection and --html-output must be supplied together.")
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        article = str(payload.get("article") or "Article")
        html_output = args.html_output.expanduser().resolve()
        html_output.write_text(
            render_html(
                article,
                build_vega_spec(
                    article,
                    projection,
                    payload["hierarchy"],
                    payload.get("edit_wars"),
                ),
            ),
            encoding="utf-8",
        )
        print(f"Rendered {html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
