#!/usr/bin/env python3
"""Estimate token volume for the full metadata-aware edge-label run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edge_labels import (
    EdgeLabel,
    SYSTEM_PROMPT,
    apply_metadata_overrides,
    build_prompt,
)
from revision_io import (
    iter_revisions,
    load_visualization_revision_ids,
    select_visualized_revisions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional metadata JSON override; XML metadata is used by default.",
    )
    parser.add_argument("--preview", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError(
            "tiktoken fehlt. Bitte `pip install -r requirements.txt` ausführen."
        ) from error
    revision_ids = load_visualization_revision_ids(args.projection)
    revisions = list(select_visualized_revisions(
        iter_revisions(args.xml, revision_ids=revision_ids), revision_ids
    ))
    apply_metadata_overrides(revisions, args.metadata)

    encoding = tiktoken.get_encoding("o200k_base")
    schema = json.dumps(EdgeLabel.model_json_schema(), ensure_ascii=False)
    fixed_tokens = len(encoding.encode(SYSTEM_PROMPT + schema)) + 12
    prompt_tokens: list[int] = []
    unchanged = 0
    for previous, current in zip(revisions, revisions[1:]):
        if previous["text"] == current["text"]:
            unchanged += 1
        prompt_tokens.append(
            fixed_tokens + len(encoding.encode(build_prompt(previous, current)))
        )

    preview = json.loads(args.preview.read_text(encoding="utf-8"))
    preview_output_tokens = [
        len(encoding.encode(json.dumps({
            "edge_label": pair["edge_label"],
            "detailed_description": pair["detailed_description"],
        }, ensure_ascii=False)))
        for pair in preview["revision_pairs"]
        if pair["edge_label"] != "No content change"
    ]
    average_visible_output = (
        sum(preview_output_tokens) / len(preview_output_tokens)
        if preview_output_tokens else 0
    )
    result = {
        "tokenizer": "o200k_base (local estimate)",
        "trajectory_edges": len(revisions) - 1,
        "estimated_edge_api_calls": len(prompt_tokens),
        "identical_cleaned_text_edges_still_sent_to_api": unchanged,
        "edge_input_tokens": sum(prompt_tokens),
        "minimum_edge_input_tokens": min(prompt_tokens),
        "average_edge_input_tokens": round(sum(prompt_tokens) / len(prompt_tokens)),
        "maximum_edge_input_tokens": max(prompt_tokens),
        "preview_average_visible_output_tokens": round(average_visible_output, 1),
        "estimated_all_edge_visible_output_tokens": round(
            average_visible_output * len(prompt_tokens)
        ),
        "reasoning_output_tokens_included": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
