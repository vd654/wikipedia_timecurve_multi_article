#!/usr/bin/env python3
"""Apply only validated Edit-War labels to a reviewed hierarchy and rebuild HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import BASE_DIR
from semantic_zoom import build_vega_spec, render_html


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def project_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path(BASE_DIR).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def apply_validated_edit_wars(hierarchy: dict, assessment_payload: dict) -> list[str]:
    """Overlay accepted conflict labels without changing unrelated edge labels."""
    edges = {
        (
            str(edge.get("previous_revision_id", "")),
            str(edge.get("current_revision_id", "")),
        ): edge
        for edge in hierarchy.get("edges", [])
    }
    updated: list[str] = []
    for pair in assessment_payload.get("revision_pairs", []):
        if not pair.get("edit_war_candidate_session_ids"):
            continue
        key = (
            str(pair.get("previous_revision_id", "")),
            str(pair.get("current_revision_id", "")),
        )
        edge = edges.get(key)
        if edge is None or pair.get("is_edit_war") is True:
            continue
        normal_label = " ".join(str(pair.get("edge_label", "")).split())
        normal_description = " ".join(
            str(pair.get("detailed_description", "")).split()
        )
        edge.update({
            "label": normal_label,
            "short_label": normal_label,
            "detailed_description": normal_description,
            "label_source": "full_revision_lod_llm",
            "detailed_description_source": "full_revision_lod_llm",
        })
        for stale_key in (
            "conflict_description",
            "is_edit_war",
            "edit_war_session_ids",
            "edit_war_evidence",
            "edit_war_conflict_group_id",
            "edit_war_label_session_id",
            "edit_war_label_anchor",
            "suppress_map_label",
        ):
            edge.pop(stale_key, None)
    for pair in assessment_payload.get("revision_pairs", []):
        if pair.get("is_edit_war") is not True:
            continue
        key = (
            str(pair.get("previous_revision_id", "")),
            str(pair.get("current_revision_id", "")),
        )
        edge = edges.get(key)
        if edge is None:
            raise ValueError(
                f"Validierte Edit-War-Kante {key[0]} -> {key[1]} fehlt."
            )
        session_ids = pair.get("edit_war_context", {}).get("session_ids", [])
        label_session_id = str(pair.get("edit_war_label_session_id", ""))
        edge.pop("edit_war_conflict_group_id", None)
        if pair.get("edit_war_label_anchor") is not True:
            normal_label = " ".join(str(pair.get("edge_label", "")).split())
            normal_description = " ".join(
                str(pair.get("detailed_description", "")).split()
            )
            edge.update({
                "label": normal_label,
                "short_label": normal_label,
                "detailed_description": normal_description,
                "conflict_description": "",
                "label_source": "full_revision_lod_llm",
                "detailed_description_source": "full_revision_lod_llm",
                "is_edit_war": True,
                "edit_war_session_ids": list(session_ids),
                "edit_war_label_session_id": label_session_id,
                "edit_war_label_anchor": False,
                "suppress_map_label": True,
                "edit_war_evidence": pair.get("evidence", []),
            })
            continue
        label = " ".join(str(pair.get("edge_label", "")).split())
        description = " ".join(
            str(pair.get("conflict_description", "")).split()
        )
        if not label or not description:
            raise ValueError(
                f"Edit-War-Kante {key[0]} -> {key[1]} hat kein Label oder keine Beschreibung."
            )
        edge.update({
            "label": label,
            "short_label": label,
            "detailed_description": description,
            "conflict_description": description,
            "label_source": "llm_validated_edit_war_session",
            "detailed_description_source": "llm_validated_edit_war_session",
            "is_edit_war": True,
            "edit_war_session_ids": list(session_ids),
            "edit_war_label_session_id": label_session_id,
            "edit_war_label_anchor": True,
            "suppress_map_label": False,
            "edit_war_evidence": pair.get("evidence", []),
        })
        updated.append(str(edge.get("node_id", "")))
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay validated Edit-War labels and rebuild the visualization."
    )
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--assessments", type=Path, required=True)
    parser.add_argument("--html-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projection_path = args.projection.expanduser().resolve()
    hierarchy_path = args.hierarchy.expanduser().resolve()
    assessments_path = args.assessments.expanduser().resolve()
    html_path = args.html_output.expanduser().resolve()

    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    assessments = json.loads(assessments_path.read_text(encoding="utf-8"))
    hierarchy = payload.get("hierarchy")
    edit_wars = assessments.get("edit_war_classification")
    if not isinstance(hierarchy, dict):
        raise ValueError("Die Zieldatei enthält keine gültige Hierarchie.")
    if not isinstance(edit_wars, dict):
        raise ValueError("Die Assessment-Datei enthält keine Edit-War-Analyse.")

    updated = apply_validated_edit_wars(hierarchy, assessments)
    payload["edit_wars"] = edit_wars
    payload["edit_war_semantic_overlay"] = {
        "source": project_relative(assessments_path),
        "updated_node_ids": updated,
        "updated_edge_count": len(updated),
        "labelled_edit_war_session_count": len(updated),
        "validated_edit_war_edge_count": sum(
            pair.get("is_edit_war") is True
            for pair in assessments.get("revision_pairs", [])
        ),
        "unrelated_edge_labels_changed": False,
        "talk_page_xml": assessments.get("talk_page_xml"),
        "assessment_prompt_version": assessments.get(
            "edit_war_assessment_prompt_version"
        ),
        "acceptance_rule": assessments.get("edit_war_acceptance_rule"),
    }
    article = str(payload.get("article") or assessments.get("article") or "Article")
    write_text_atomic(
        hierarchy_path,
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    write_text_atomic(
        html_path,
        render_html(
            article,
            build_vega_spec(article, projection, hierarchy, edit_wars),
        ),
    )
    print(f"Edit-War-Sitzungslabels aktualisiert: {len(updated)}")
    print(f"Hierarchie: {hierarchy_path}")
    print(f"Visualisierung: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
