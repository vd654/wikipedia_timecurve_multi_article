#!/usr/bin/env python3
"""Rebuild the final HTML from reviewed, repository-tracked artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import BASE_DIR
from semantic_zoom import build_vega_spec, render_html


DEFAULT_CONFIG = Path(BASE_DIR) / "final_build.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the final reviewed visualization without XML, API calls, "
            "or obsolete HTML files."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional HTML output override (default: value from the config).",
    )
    return parser.parse_args()


def resolve_config_path(config_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Final-Build-Konfiguration enthält keinen Pfad für {field}.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def load_final_inputs(config_path: Path) -> tuple[dict, dict, dict]:
    config_path = config_path.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Final-Build-Konfiguration fehlt: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Nicht unterstützte Final-Build-Konfigurationsversion.")

    projection_path = resolve_config_path(
        config_path, config.get("projection"), "projection"
    )
    hierarchy_path = resolve_config_path(
        config_path, config.get("reviewed_hierarchy"), "reviewed_hierarchy"
    )
    for path in (projection_path, hierarchy_path):
        if not path.exists():
            raise FileNotFoundError(f"Final-Build-Eingabe fehlt: {path}")

    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    reviewed = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    article = str(config.get("article") or reviewed.get("article") or "").strip()
    if not article:
        raise ValueError("Final-Build-Konfiguration enthält keinen Artikelnamen.")
    if reviewed.get("article") not in (None, article):
        raise ValueError("Artikelname und geprüfte Hierarchie passen nicht zusammen.")
    if not isinstance(projection.get("points"), list) or not projection["points"]:
        raise ValueError("Projektionsdatei enthält keine Revisionspunkte.")
    hierarchy = reviewed.get("hierarchy")
    if not isinstance(hierarchy, dict) or not hierarchy.get("edges"):
        raise ValueError("Geprüfte Hierarchiedatei enthält keine Kanten.")
    edit_wars = reviewed.get("edit_wars")
    if not isinstance(edit_wars, dict):
        raise ValueError("Geprüfte Hierarchiedatei enthält keine Revert-Analyse.")

    return config, projection, {
        "article": article,
        "hierarchy": hierarchy,
        "edit_wars": edit_wars,
    }


def build_final_html(config_path: Path, output_override: Path | None = None) -> Path:
    config_path = config_path.expanduser().resolve()
    config, projection, reviewed = load_final_inputs(config_path)
    output_path = (
        output_override.expanduser().resolve()
        if output_override is not None
        else resolve_config_path(config_path, config.get("html_output"), "html_output")
    )
    html = render_html(
        reviewed["article"],
        build_vega_spec(
            reviewed["article"],
            projection,
            reviewed["hierarchy"],
            reviewed["edit_wars"],
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(html, encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def main() -> int:
    args = parse_args()
    output_path = build_final_html(args.config, args.output)
    print(f"Finale Visualisierung: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"[FEHLER] {error}")
        raise SystemExit(1) from error
