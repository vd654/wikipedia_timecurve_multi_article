#!/usr/bin/env python3
"""Persist reviewed edge-label corrections in hierarchy, result, and cache JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def update_rows(rows: list[dict], labels: dict[str, str]) -> set[str]:
    updated: set[str] = set()
    for row in rows:
        revision_id = str(row.get("current_revision_id") or "")
        if revision_id not in labels:
            continue
        row["edge_label"] = labels[revision_id]
        if "label" in row:
            row["label"] = labels[revision_id]
        if "short_label" in row:
            row["short_label"] = labels[revision_id]
        updated.add(revision_id)
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--diff-labels", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        name: path.expanduser().resolve()
        for name, path in {
            "hierarchy": args.hierarchy,
            "diff_labels": args.diff_labels,
            "cache": args.cache,
            "overrides": args.overrides,
        }.items()
    }
    override_payload = json.loads(paths["overrides"].read_text(encoding="utf-8"))
    labels = {
        str(row["current_revision_id"]): " ".join(row["edge_label"].split())
        for row in override_payload["revision_pairs"]
    }
    if len(labels) != len(override_payload["revision_pairs"]):
        raise ValueError("Duplicate current_revision_id in edge-label overrides.")
    if any(not 2 <= len(label.split()) <= 3 for label in labels.values()):
        raise ValueError("Every fixed edge label must contain two or three words.")

    hierarchy_payload = json.loads(paths["hierarchy"].read_text(encoding="utf-8"))
    diff_payload = json.loads(paths["diff_labels"].read_text(encoding="utf-8"))
    cache_payload = json.loads(paths["cache"].read_text(encoding="utf-8"))

    updated_hierarchy = update_rows(
        hierarchy_payload["hierarchy"]["edges"], labels
    )
    updated_diff = update_rows(diff_payload["revision_pairs"], labels)
    updated_cache = update_rows(list(cache_payload["entries"].values()), labels)
    expected = set(labels)
    for source, updated in (
        ("hierarchy", updated_hierarchy),
        ("diff labels", updated_diff),
        ("cache", updated_cache),
    ):
        missing = sorted(expected - updated)
        if missing:
            raise ValueError(f"Missing revisions in {source}: {missing}")

    previous_marker = hierarchy_payload.get("fixed_edge_label_overrides")
    previous_marker = previous_marker if isinstance(previous_marker, dict) else {}
    sources = list(previous_marker.get("sources") or [])
    if not sources and previous_marker.get("source"):
        sources.append(str(previous_marker["source"]))
    current_source = str(paths["overrides"])
    if current_source not in sources:
        sources.append(current_source)
    all_revision_ids = {
        str(revision_id)
        for revision_id in previous_marker.get("updated_revision_ids", [])
    } | expected
    hierarchy_payload["fixed_edge_label_overrides"] = {
        "sources": sources,
        "latest_source": current_source,
        "updated_revision_ids": sorted(all_revision_ids),
        "updated_edge_count": len(all_revision_ids),
    }
    write_json(paths["hierarchy"], hierarchy_payload)
    write_json(paths["diff_labels"], diff_payload)
    write_json(paths["cache"], cache_payload)
    print(f"Updated {len(expected)} fixed edge labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
