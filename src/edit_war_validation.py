#!/usr/bin/env python3
"""Validate only algorithmic Edit-War candidates with a separate LLM pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import BASE_DIR, article_paths, article_slug
from llm_provider import LLM_PROVIDER, create_llm_client, model_or_default
from revert_detection import analyze_reverts, history_window_for_points, load_revert_history
from revision_keywords import (
    EDIT_WAR_ASSESSMENT_PROMPT_VERSION,
    EDIT_WAR_MIN_CONFIDENCE,
    apply_edit_war_assessments,
    apply_edit_war_descriptions,
    assess_edit_war_sessions,
    attach_edit_war_revision_context,
    attach_full_context_edge_analyses,
    load_api_key,
)
from talk_evidence import (
    link_talk_evidence,
    load_talk_revision_changes,
    session_time_windows,
)


def build_candidate_contexts(revert_analysis: dict) -> list[dict]:
    """Return one algorithmic-candidate context per plotted revision edge."""
    contexts = []
    for statistics in revert_analysis.get("point_statistics", [])[1:]:
        sessions = statistics.get("interval_confirmed_edit_war_sessions", [])
        contexts.append({
            "is_edit_war": bool(sessions),
            "confirmed_event_count": int(
                statistics.get("interval_confirmed_edit_war_event_count", 0)
            ),
            "editor_pairs": [session.get("pair", []) for session in sessions],
            "session_ids": [str(session.get("session_id", "")) for session in sessions],
        })
    return contexts


def attach_contexts_to_existing_pairs(
    revision_pairs: list[dict], contexts: list[dict]
) -> None:
    """Attach algorithm output to existing labels without any edge-label API call."""
    if len(revision_pairs) != len(contexts):
        raise ValueError(
            "Vorhandene Kantenlabels und algorithmische Intervalle haben "
            f"unterschiedliche Längen ({len(revision_pairs)} != {len(contexts)})."
        )
    for pair, context in zip(revision_pairs, contexts, strict=True):
        pair["edit_war_context"] = context
        pair["is_edit_war"] = bool(context["is_edit_war"])


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def project_relative(path: Path) -> str:
    """Store portable paths for files inside the project."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(Path(BASE_DIR).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run separate LLM validation only for algorithmic Edit-War candidates; "
            "ordinary revision edges are loaded from an existing label JSON."
        )
    )
    parser.add_argument("--article", required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--edge-labels", type=Path, required=True)
    parser.add_argument("--talk-xml", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = model_or_default(args.model)
    article_xml = Path(article_paths(args.article)["xml_path"]).resolve()
    projection_path = args.projection.expanduser().resolve()
    edge_labels_path = args.edge_labels.expanduser().resolve()
    talk_xml_path = args.talk_xml.expanduser().resolve()
    cache_path = args.cache.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    source_payload = json.loads(edge_labels_path.read_text(encoding="utf-8"))
    revision_pairs = [dict(row) for row in source_payload.get("revision_pairs", [])]
    if not revision_pairs:
        raise ValueError("Die vorhandene Kantenlabel-Datei enthält keine Paare.")

    full_history = load_revert_history(article_xml, include_text_changes=True)
    conflict_history, sampled = history_window_for_points(
        full_history, projection["points"]
    )
    revert_analysis = analyze_reverts(conflict_history, sampled)
    contexts = build_candidate_contexts(revert_analysis)
    attach_contexts_to_existing_pairs(revision_pairs, contexts)
    attach_full_context_edge_analyses(revert_analysis, revision_pairs)
    attach_edit_war_revision_context(
        revert_analysis, conflict_history, revision_pairs
    )

    talk_revisions = load_talk_revision_changes(
        talk_xml_path,
        windows=session_time_windows(revert_analysis["edit_war_sessions"]),
    )
    link_talk_evidence(revert_analysis["edit_war_sessions"], talk_revisions)

    candidates = revert_analysis["edit_war_sessions"]
    candidate_revision_ids = sorted({
        str(item.get("revision_id", ""))
        for session in candidates
        for item in session.get("revision_context", [])
        if item.get("revision_id")
    })
    candidate_edge_count = sum(context["is_edit_war"] for context in contexts)
    print(f"Algorithmische Kandidatensitzungen: {len(candidates)}")
    print(f"Betroffene vorhandene Kanten: {candidate_edge_count}")
    print(f"An das Assessment übergebbare Artikel-IDs: {len(candidate_revision_ids)}")
    print("Normale Kantenlabel-Aufrufe: 0")
    print(f"Separate Sitzungsaufrufe (maximal): {len(candidates)}")

    api_key = load_api_key(Path(BASE_DIR) / ".env")
    client = create_llm_client(api_key=api_key).with_options(
        timeout=180.0,
        max_retries=2,
    )
    assessments = assess_edit_war_sessions(
        client=client,
        model=model,
        revert_analysis=revert_analysis,
        cache_path=cache_path,
        refresh_cache=args.refresh_cache,
        provider=LLM_PROVIDER,
    )
    apply_edit_war_assessments(
        revert_analysis,
        revision_pairs,
        assessments,
        talk_xml_path=Path(project_relative(talk_xml_path)),
        model=model,
        provider=LLM_PROVIDER,
    )
    apply_edit_war_descriptions(revision_pairs, assessments)

    payload = {
        **source_payload,
        "article": args.article,
        "visualization": project_relative(projection_path),
        "llm_provider": LLM_PROVIDER,
        "model": model,
        "edit_war_validation_mode": "algorithmic_candidates_only",
        "ordinary_edge_label_api_calls": 0,
        "algorithmic_candidate_session_count": len(candidates),
        "algorithmic_candidate_edge_count": candidate_edge_count,
        "candidate_article_revision_ids": candidate_revision_ids,
        "candidate_article_revision_id_count": len(candidate_revision_ids),
        "talk_page_xml": project_relative(talk_xml_path),
        "edit_war_assessment_prompt_version": (
            EDIT_WAR_ASSESSMENT_PROMPT_VERSION
        ),
        "edit_war_minimum_confidence": EDIT_WAR_MIN_CONFIDENCE,
        "edit_war_acceptance_rule": (
            "same subject AND opposing changes AND substantive content AND "
            "article-text evidence AND >=2 supporting candidate article IDs "
            "AND confidence >= 0.75; comments and Talk are support only"
        ),
        "edit_war_classification": revert_analysis,
        "edit_war_session_assessments": assessments,
        "revision_pairs": revision_pairs,
        "source_edge_labels": project_relative(edge_labels_path),
    }
    write_json_atomic(output_path, payload)
    print(
        "LLM-validierte Edit-War-Sitzungen: "
        f"{revert_analysis['confirmed_edit_war_session_count']}"
    )
    print(f"Assessment-Datei: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
