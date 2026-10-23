"""Shared loading and selection of cleaned Wikipedia revisions."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Iterator

from config import article_paths
from text_cleaning import clean_wikitext, extract_infobox_text


def local_name(tag: str) -> str:
    """Return an XML tag without its optional namespace."""
    return tag.rsplit("}", 1)[-1]


def direct_child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    for child in element:
        if local_name(child.tag) == name:
            return child
    return None


def child_text(element: ET.Element | None, name: str) -> str:
    child = direct_child(element, name)
    return (child.text or "") if child is not None else ""


def element_is_hidden(element: ET.Element | None) -> bool:
    """Return whether a MediaWiki XML element is marked as deleted/hidden."""
    if element is None:
        return False
    return any(local_name(attribute) == "deleted" for attribute in element.attrib)


def iter_revisions(
    xml_path: Path,
    min_revision_chars: int = 0,
    revision_ids: list[str] | None = None,
) -> Iterator[dict]:
    """Yield requested cleaned revisions without cleaning unused XML entries."""
    if not xml_path.exists():
        raise FileNotFoundError(f"XML nicht gefunden: {xml_path}")
    if xml_path.stat().st_size == 0:
        raise ValueError(f"XML-Datei ist leer: {xml_path}")

    wanted_ids = set(revision_ids) if revision_ids is not None else None
    for _, element in ET.iterparse(xml_path, events=("end",)):
        if local_name(element.tag) != "revision":
            continue

        revision_id = child_text(element, "id")
        if wanted_ids is not None and revision_id not in wanted_ids:
            element.clear()
            continue

        raw_text = child_text(element, "text")
        article_text = clean_wikitext(raw_text)
        infobox_text = extract_infobox_text(raw_text)
        contributor = direct_child(element, "contributor")
        comment = direct_child(element, "comment")
        revision = {
            "revision_id": revision_id,
            "timestamp": child_text(element, "timestamp"),
            "sha1": child_text(element, "sha1"),
            "author": (
                child_text(contributor, "username")
                or child_text(contributor, "ip")
            ),
            "comment": (comment.text or "") if comment is not None else "",
            "user_hidden": element_is_hidden(contributor),
            "comment_hidden": element_is_hidden(comment),
            "text": " ".join(
                part for part in (article_text, infobox_text) if part
            ),
        }
        element.clear()

        if len(revision["text"]) >= min_revision_chars:
            yield revision


def resolve_projection_path(visualization_path: Path) -> Path:
    """Accept either a visualization HTML or its projection JSON sidecar."""
    visualization_path = visualization_path.expanduser().resolve()
    if visualization_path.suffix.casefold() == ".html":
        visualization_path = visualization_path.with_name(
            f"{visualization_path.stem}_projection.json"
        )
    if not visualization_path.exists():
        raise FileNotFoundError(
            f"Projektionsdatei der Visualisierung nicht gefunden: {visualization_path}"
        )
    return visualization_path


def find_latest_visualization(article: str) -> Path:
    """Return the highest-numbered complete visualization for an article."""
    paths = article_paths(article)
    output_dir = Path(paths["output_dir"])
    prefix = paths["output_basename"] + "_"
    candidates = []

    for html_path in output_dir.glob(f"{prefix}*.html"):
        number = html_path.stem.removeprefix(prefix)
        projection = html_path.with_name(f"{html_path.stem}_projection.json")
        if number.isdigit() and projection.exists():
            candidates.append((int(number), html_path))

    if not candidates:
        raise FileNotFoundError(
            f"Keine vollständige Visualisierung gefunden: {output_dir}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def load_visualization_revision_ids(projection_path: Path) -> list[str]:
    """Read the exact ordered point IDs used by the selected visualization."""
    data = json.loads(projection_path.read_text(encoding="utf-8"))
    points = data.get("points")
    if not isinstance(points, list):
        raise ValueError("Die Projektionsdatei enthält keine gültige points-Liste.")
    if any(not isinstance(point, dict) for point in points):
        raise ValueError("Die points-Liste enthält einen ungültigen Eintrag.")
    revision_ids = [str(point.get("revision_id", "")) for point in points]
    if len(revision_ids) < 2 or any(not revision_id for revision_id in revision_ids):
        raise ValueError(
            "Die Visualisierung muss mindestens zwei Punkte mit revision_id enthalten."
        )
    if len(set(revision_ids)) != len(revision_ids):
        raise ValueError("Die Visualisierung enthält doppelte Revisions-IDs.")
    return revision_ids


def select_visualized_revisions(
    revisions: Iterable[dict], revision_ids: list[str]
) -> Iterator[dict]:
    """Yield XML revisions in exactly the order stored by the visualization."""
    wanted_index = 0
    for revision in revisions:
        if revision["revision_id"] != revision_ids[wanted_index]:
            continue
        yield revision
        wanted_index += 1
        if wanted_index == len(revision_ids):
            return

    missing = revision_ids[wanted_index:]
    raise ValueError(
        "Revisions-IDs der Visualisierung fehlen in der XML-Datei: "
        + ", ".join(missing[:5])
    )


def revision_state_sha1(revision: dict) -> str:
    """Use MediaWiki SHA1 when available, otherwise hash the cleaned state."""
    sha1 = str(revision.get("sha1", "")).strip()
    if sha1:
        return sha1
    return hashlib.sha1(str(revision.get("text", "")).encode("utf-8")).hexdigest()
