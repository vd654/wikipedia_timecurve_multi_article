#!/usr/bin/env python3
"""Main pipeline for the semantic Wikipedia revision timecurve."""

import argparse
import colorsys
import html
import json
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, add_pipeline_args, article_paths, article_slug
from output_utils import next_numbered_path
from revert_detection import analyze_reverts, probable_bot
from text_cleaning import extract_plain_sections


BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
MAX_MODEL_TOKENS = 256
HDBSCAN_MIN_CLUSTER_SIZE = 5
HDBSCAN_MIN_SAMPLES = 5
PLOT_WIDTH = 880
PLOT_HEIGHT = 600

NON_PROSE_SECTIONS = {
    "bibliography", "citations", "external links", "further reading",
    "notes", "references", "see also", "sources",
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def direct_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if local_name(child.tag) == name:
            return child
    return None


def child_text(element: ET.Element | None, name: str) -> str:
    if element is None:
        return ""
    child = direct_child(element, name)
    return (child.text or "") if child is not None else ""


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_history(xml_path: str) -> list[dict]:
    """Load the complete revision history from the article XML."""
    if not os.path.exists(xml_path):
        raise FileNotFoundError(
            f"XML nicht gefunden: {xml_path}\n"
            "Bitte zuerst `python src/wiki_api.py --article \"ARTICLE\"` ausführen."
        )
    if os.path.getsize(xml_path) == 0:
        raise ValueError(f"XML-Datei ist leer: {xml_path}")

    print(f"[1] Lade XML: {xml_path}")
    revisions = []
    for _, element in ET.iterparse(xml_path, events=("end",)):
        if local_name(element.tag) != "revision":
            continue

        content = child_text(element, "text")
        timestamp = child_text(element, "timestamp")
        if timestamp:
            contributor = direct_child(element, "contributor")
            user = (
                child_text(contributor, "username")
                or child_text(contributor, "ip")
                or "anonymous"
            )
            revisions.append({
                "revision_id": child_text(element, "id") or "0",
                "timestamp": timestamp,
                "datetime": parse_timestamp(timestamp),
                "user": user,
                "sha1": child_text(element, "sha1"),
                "tags": child_text(element, "tags"),
                "probable_bot": probable_bot(user, child_text(element, "tags")),
                "size": len(content.encode("utf-8")),
                "content": content,
            })
        element.clear()

    if not revisions:
        raise ValueError("Die XML-Datei enthält keine Revisionen.")
    revisions.sort(key=lambda item: item["datetime"])
    print(f"    {len(revisions)} Revisionen geladen")
    return revisions


def filter_revision_dates(
    revisions: list[dict], start: str | None, end: str | None
) -> list[dict]:
    """Apply optional inclusive YYYY-MM-DD limits to the full history."""
    try:
        start_date = datetime.strptime(start, "%Y-%m-%d").date() if start else None
        end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else None
    except ValueError as exc:
        raise ValueError("Datumswerte müssen das Format YYYY-MM-DD haben.") from exc
    if start_date and end_date and start_date > end_date:
        raise ValueError("--start darf nicht nach --end liegen.")

    filtered = [
        revision for revision in revisions
        if (not start_date or revision["datetime"].date() >= start_date)
        and (not end_date or revision["datetime"].date() <= end_date)
    ]
    for index, revision in enumerate(filtered):
        revision["history_index"] = index
    if not filtered:
        raise ValueError("Im gewählten Zeitraum liegen keine Revisionen.")
    return filtered


def sort_and_sample(revisions: list[dict], strategy: str) -> list[dict]:
    """Select complete revisions; all revisions are used by default."""
    revisions = sorted(revisions, key=lambda item: item["datetime"])
    if strategy == "all":
        sampled = revisions
    elif strategy == "weekly":
        sampled = []
        last_date = None
        for revision in revisions:
            if last_date is None or (revision["datetime"] - last_date).days >= 7:
                sampled.append(revision)
                last_date = revision["datetime"]
    elif strategy == "monthly":
        sampled = []
        seen_months = set()
        for revision in revisions:
            month = (revision["datetime"].year, revision["datetime"].month)
            if month not in seen_months:
                sampled.append(revision)
                seen_months.add(month)
    elif strategy == "largest":
        previous_size = revisions[0]["size"]
        ranked = []
        for revision in revisions:
            ranked.append((abs(revision["size"] - previous_size), revision))
            previous_size = revision["size"]
        sampled = [
            item[1]
            for item in sorted(ranked, key=lambda item: item[0], reverse=True)[:300]
        ]
        sampled.sort(key=lambda item: item["datetime"])
    else:
        raise ValueError(f"Unbekannte Sampling-Strategie: {strategy}")

    if len(sampled) < 10:
        raise ValueError("Für die Timecurve werden mindestens zehn Revisionen benötigt.")
    if len({item["revision_id"] for item in sampled}) != len(sampled):
        raise ValueError("Die ausgewählten Revisionen enthalten doppelte IDs.")
    print(f"[2] Sampling ({strategy}): {len(sampled)} Revisionspunkte")
    return sampled


def clean_revision_text(content: str) -> tuple[str, list[str]]:
    """Return prose as one continuous text; sections are not comparison units."""
    parts = []
    dropped = []
    for section in extract_plain_sections(content):
        heading = section["section_heading"].strip() or "Introduction"
        if heading.casefold() in NON_PROSE_SECTIONS:
            dropped.append(heading)
        else:
            parts.append(section["text"])
    return " ".join(parts), dropped


def token_chunks(text: str, tokenizer, max_tokens: int) -> list[dict]:
    """Split only for model limits; chunks are never matched across revisions."""
    token_ids = tokenizer.encode(
        text, add_special_tokens=False, truncation=False
    )
    content_limit = max(1, max_tokens - 2)
    chunks = []
    for start in range(0, len(token_ids), content_limit):
        window = token_ids[start:start + content_limit]
        chunk_text = tokenizer.decode(
            window,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "token_count": len(window),
                "word_count": len(chunk_text.split()),
            })
    return chunks


def verify_normalized(embeddings: np.ndarray, label: str) -> dict:
    """Fail unless all vectors are finite L2 unit vectors."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"{label} sind leer oder enthalten ungültige Werte.")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(f"{label} sind nicht vollständig L2-normalisiert.")
    return {
        "verified": True,
        "minimum_norm": float(norms.min()),
        "maximum_norm": float(norms.max()),
        "tolerance": 1e-5,
    }


def compute_revision_embeddings(revisions: list[dict]):
    """Encode temporary chunks, average them, and return one vector per revision."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers fehlt.") from exc

    print(f"[3] Lade BERT-Modell: {BERT_MODEL}")
    model = SentenceTransformer(BERT_MODEL)
    max_tokens = min(int(model.max_seq_length), MAX_MODEL_TOKENS)
    all_chunks = []
    ranges = []

    # We only ask the tokenizer for token IDs here. The resulting windows passed
    # to the model below still obey max_tokens.
    original_tokenizer_limit = model.tokenizer.model_max_length
    model.tokenizer.model_max_length = 10**9
    try:
        for revision in revisions:
            clean_text, dropped = clean_revision_text(revision.pop("content"))
            chunks = token_chunks(clean_text, model.tokenizer, max_tokens)
            if not chunks:
                raise ValueError(
                    f"Revision {revision['revision_id']} enthält keinen Fließtext."
                )
            start = len(all_chunks)
            all_chunks.extend(chunks)
            ranges.append((start, len(all_chunks)))
            revision["clean_word_count"] = len(clean_text.split())
            revision["encoding_chunk_count"] = len(chunks)
            revision["dropped_non_prose_sections"] = dropped
    finally:
        model.tokenizer.model_max_length = original_tokenizer_limit

    # Consecutive revisions contain many byte-identical chunks. Encode each
    # unique text once, then reuse its vector at every original occurrence.
    unique_texts = list(dict.fromkeys(chunk["text"] for chunk in all_chunks))
    text_index = {text: index for index, text in enumerate(unique_texts)}
    chunk_indices = np.asarray(
        [text_index[chunk["text"]] for chunk in all_chunks], dtype=np.int64
    )
    print(
        f"[4] Kodiere {len(unique_texts)} eindeutige von "
        f"{len(all_chunks)} temporären Chunks"
    )
    unique_embeddings = np.asarray(model.encode(
        unique_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ), dtype=np.float64)
    chunk_normalization = verify_normalized(
        unique_embeddings, "Chunk-Embeddings"
    )

    revision_vectors = []
    for start, end in ranges:
        vector = unique_embeddings[chunk_indices[start:end]].mean(axis=0)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Ein gemittelter Revisionsvektor ist ungültig.")
        revision_vectors.append(vector / norm)
    revision_vectors = np.asarray(revision_vectors, dtype=np.float64)
    revision_normalization = verify_normalized(
        revision_vectors, "Revisionsvektoren"
    )
    print(
        "    Revisionsvektor-Normen: "
        f"{revision_normalization['minimum_norm']:.6f} bis "
        f"{revision_normalization['maximum_norm']:.6f}"
    )
    return revision_vectors, {
        "model": BERT_MODEL,
        "maximum_model_tokens": max_tokens,
        "temporary_chunk_count": len(all_chunks),
        "unique_encoded_chunk_count": len(unique_texts),
        "identical_chunk_reuse": True,
        "chunk_overlap": 0,
        "chunk_matching": False,
        "aggregation": "arithmetic mean of chunk embeddings, then L2 normalization",
        "chunk_normalization": chunk_normalization,
        "revision_vector_normalization": revision_normalization,
    }


def validate_distance_matrix(matrix: np.ndarray, label: str) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{label} ist nicht quadratisch.")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-12):
        raise ValueError(f"{label} enthält ungültige Werte.")
    if not np.allclose(values, values.T, atol=1e-10):
        raise ValueError(f"{label} ist nicht symmetrisch.")
    if not np.allclose(np.diag(values), 0.0, atol=1e-10):
        raise ValueError(f"{label} hat keine Nulldiagonale.")


def cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """Return one direct cosine distance for every revision pair."""
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
    distances = np.clip(1.0 - similarities, 0.0, 2.0)
    distances = (distances + distances.T) / 2.0
    np.fill_diagonal(distances, 0.0)
    validate_distance_matrix(distances, "Kosinus-Distanzmatrix")
    return distances


def euclidean_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=2))


def classical_mds(distance_matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    """Project global distances with deterministic classical metric MDS."""
    validate_distance_matrix(distance_matrix, "MDS-Eingabematrix")
    count = len(distance_matrix)
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distance_matrix ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    selected = np.maximum(eigenvalues[:2], 0.0)
    coordinates = eigenvectors[:, :2] * np.sqrt(selected)
    for axis in range(coordinates.shape[1]):
        anchor = int(np.argmax(np.abs(coordinates[:, axis])))
        if coordinates[anchor, axis] < 0:
            coordinates[:, axis] *= -1

    projected = euclidean_distance_matrix(coordinates)
    upper = np.triu_indices(count, k=1)
    original_values = distance_matrix[upper]
    projected_values = projected[upper]
    residual = original_values - projected_values
    raw_stress = float(np.sum(residual ** 2))
    denominator = float(np.sum(original_values ** 2))
    normalized_stress = float(np.sqrt(raw_stress / denominator)) if denominator else 0.0
    if np.std(original_values) > 0 and np.std(projected_values) > 0:
        shepard = float(np.corrcoef(original_values, projected_values)[0, 1])
    else:
        shepard = 1.0

    positive = eigenvalues[eigenvalues > 1e-12]
    explained = (
        float(selected.sum() / positive.sum()) if len(positive) else 0.0
    )
    negative_mass = float(np.abs(eigenvalues[eigenvalues < 0]).sum())
    total_mass = float(np.abs(eigenvalues).sum())
    return coordinates, {
        "algorithm": "classical metric MDS",
        "input": "precomputed cosine distance matrix",
        "dimensions": 2,
        "randomness": False,
        "quality": {
            "raw_stress": raw_stress,
            "normalized_stress_1": normalized_stress,
            "shepard_correlation": shepard,
            "positive_eigenvalue_variance_in_2d": explained,
            "negative_eigenvalue_fraction": (
                negative_mass / total_mass if total_mass else 0.0
            ),
        },
    }


def cluster_mds_coordinates(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run HDBSCAN directly on the two visible MDS dimensions."""
    try:
        import hdbscan
    except ImportError as exc:
        raise RuntimeError("hdbscan fehlt.") from exc

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("HDBSCAN erwartet genau zweidimensionale MDS-Koordinaten.")

    model = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = model.fit_predict(coordinates).astype(int)
    probabilities = np.asarray(model.probabilities_, dtype=np.float64)
    cluster_ids = sorted(set(labels) - {-1})
    return labels, probabilities, {
        "algorithm": "HDBSCAN",
        "input": "2D MDS coordinates",
        "input_dimensions": 2,
        "metric": "euclidean",
        "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
        "min_samples": HDBSCAN_MIN_SAMPLES,
        "cluster_selection_method": "eom",
        "cluster_count": len(cluster_ids),
        "noise_point_count": int(np.sum(labels == -1)),
    }


def build_points(
    revisions: list[dict],
    coordinates: np.ndarray,
    semantic: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_probabilities: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    projected = euclidean_distance_matrix(coordinates)
    points = []
    drift = []
    for index, revision in enumerate(revisions):
        semantic_jump = float(semantic[index, index - 1]) if index else 0.0
        projected_jump = float(projected[index, index - 1]) if index else 0.0
        points.append({
            "revision_index": index,
            "revision_id": revision["revision_id"],
            "timestamp": revision["timestamp"],
            "author": revision["user"],
            "clean_word_count": revision["clean_word_count"],
            "encoding_chunk_count": revision["encoding_chunk_count"],
            "mds": coordinates[index].tolist(),
            "semantic_jump_from_previous": semantic_jump,
            "projected_jump_from_previous": projected_jump,
            "cluster_id": int(cluster_labels[index]),
            "cluster_probability": float(cluster_probabilities[index]),
        })
        if index:
            drift.append({
                "revision_index_a": index - 1,
                "revision_index_b": index,
                "revision_id_a": revisions[index - 1]["revision_id"],
                "revision_id_b": revision["revision_id"],
                "timestamp_a": revisions[index - 1]["timestamp"],
                "timestamp_b": revision["timestamp"],
                "cosine_distance": semantic_jump,
                "mds_distance": projected_jump,
                "distance_error": projected_jump - semantic_jump,
            })
    return points, drift


def build_visualization(article: str, points: list[dict], diagnostics: dict) -> str:
    """Build the chronological MDS curve colored by 2D HDBSCAN clusters."""
    import altair as alt
    import pandas as pd

    rows = []
    for point in points:
        cluster_id = point["cluster_id"]
        rows.append({
            "x": point["mds"][0],
            "y": point["mds"][1],
            "order": point["revision_index"],
            "date_time": point["timestamp"].replace("T", " ").replace("Z", " UTC"),
            "revision_id": point["revision_id"],
            "author": point["author"],
            "clean_words": point["clean_word_count"],
            "chunks": point["encoding_chunk_count"],
            "semantic_jump": point["semantic_jump_from_previous"],
            "cluster_id": cluster_id,
            "cluster": "Noise" if cluster_id == -1 else f"Cluster {cluster_id + 1}",
            "cluster_probability": point["cluster_probability"],
            "endpoint": (
                "Start" if point["revision_index"] == 0
                else "Ende" if point["revision_index"] == len(points) - 1
                else None
            ),
        })
    frame = pd.DataFrame(rows)
    x = alt.X("x:Q", axis=None, scale=alt.Scale(zero=False, padding=40))
    y = alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False, padding=40))
    zoom = alt.selection_interval(name="zoom", bind="scales", encodings=["x", "y"])

    cluster_ids = sorted(set(frame["cluster_id"]) - {-1})
    has_noise = -1 in set(frame["cluster_id"])
    cluster_names = ["Noise"] if has_noise else []
    cluster_colors = ["#8a8a8a"] if has_noise else []
    for cluster_id in cluster_ids:
        # Golden-angle hues avoid repeating the same ten-color palette.
        hue = (cluster_id * 0.61803398875) % 1.0
        lightness = 0.48 if cluster_id % 2 == 0 else 0.60
        red, green, blue = colorsys.hls_to_rgb(hue, lightness, 0.72)
        cluster_names.append(f"Cluster {cluster_id + 1}")
        cluster_colors.append(
            f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"
        )

    path = alt.Chart(frame).mark_line(
        color="#6f7782", strokeWidth=1.8, opacity=0.68
    ).encode(x=x, y=y, order="order:O")
    base_points = alt.Chart(frame).mark_point(
        filled=True, stroke="#ffffff", strokeWidth=1.0, opacity=0.9
    ).encode(
        x=x,
        y=y,
        color=alt.Color(
            "cluster:N",
            scale=alt.Scale(domain=cluster_names, range=cluster_colors),
            legend=alt.Legend(
                title="Cluster",
                columns=3,
                symbolLimit=0,
                labelLimit=120,
            ),
        ),
        size=alt.Size(
            "clean_words:Q",
            scale=alt.Scale(range=[18, 135]),
            legend=alt.Legend(title="Artikelgröße"),
        ),
        tooltip=[
            alt.Tooltip("date_time:N", title="Datum und Uhrzeit"),
            alt.Tooltip("revision_id:N", title="Revision ID"),
            alt.Tooltip("author:N", title="Autor"),
            alt.Tooltip("clean_words:Q", title="Wörter"),
            alt.Tooltip("chunks:Q", title="Encoding-Chunks"),
            alt.Tooltip("semantic_jump:Q", title="Kosinus-Sprung", format=".4f"),
            alt.Tooltip("cluster:N", title="HDBSCAN"),
            alt.Tooltip(
                "cluster_probability:Q",
                title="Cluster-Wahrscheinlichkeit",
                format=".3f",
            ),
        ],
    )
    endpoints = alt.Chart(frame).transform_filter(
        "isValid(datum.endpoint)"
    ).mark_text(dy=-16, fontSize=14, fontWeight=500).encode(
        x=x, y=y, text="endpoint:N"
    )

    quality = diagnostics["projection"]["quality"]
    clustering = diagnostics["clustering"]
    chart = (path + base_points + endpoints).add_params(zoom).properties(
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        title=f"Semantische Revisions-Timecurve – Wikipedia: {article}",
    ).configure_view(strokeWidth=0)

    spec = chart.to_json(indent=None)
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>Semantic Revision Timecurve – {html.escape(article)}</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f5f5f5; padding: 24px; }}
    h1, p {{ text-align: center; }}
    .card {{ background: white; padding: 18px; border-radius: 10px; }}
    .quality {{ max-width: 880px; margin: 14px auto; line-height: 1.5; }}
  </style>
</head>
<body>
  <h1>Semantische Revisions-Timecurve: Wikipedia „{html.escape(article)}“</h1>
  <div class="card"><div id="vis"></div></div>
  <div class="card quality">
    <b>Projektionsqualität:</b>
    Stress-1 {quality['normalized_stress_1']:.4f};
    Shepard-Korrelation {quality['shepard_correlation']:.4f}.<br>
    <b>HDBSCAN auf MDS-2D:</b>
    {clustering['cluster_count']} Cluster,
    {clustering['noise_point_count']} Noise-Punkte.
  </div>
  <script>vegaEmbed('#vis', {spec}, {{actions: false, renderer: 'canvas'}});</script>
</body>
</html>"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic MDS timecurve with HDBSCAN on the 2D projection."
    )
    add_pipeline_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = article_paths(args.article)
    if args.output_name:
        output_slug = article_slug(args.output_name)
        paths["output_dir"] = os.path.join(OUTPUT_DIR, output_slug)
        paths["output_basename"] = f"{output_slug}_bert_timecurve"
    history = load_history(paths["xml_path"])
    history = filter_revision_dates(history, args.start, args.end)

    eligible = [
        revision.copy() for revision in history
        if len(revision["content"]) >= args.min_revision_chars
    ]
    print(
        f"    {len(eligible)} Revisionen erfüllen die Mindestlänge "
        f"von {args.min_revision_chars} Zeichen"
    )
    revisions = sort_and_sample(eligible, args.sample_strategy)

    revision_vectors, embedding_diagnostics = compute_revision_embeddings(revisions)

    print("[5] Direkte Kosinus-Distanzmatrix der Revisionsvektoren")
    semantic = cosine_distance_matrix(revision_vectors)
    print("[6] Klassische MDS und Projektionsqualität")
    coordinates, projection = classical_mds(semantic)
    print("[7] HDBSCAN direkt auf den 2D-MDS-Koordinaten")
    cluster_labels, cluster_probabilities, clustering = (
        cluster_mds_coordinates(coordinates)
    )
    points, drift = build_points(
        revisions,
        coordinates,
        semantic,
        cluster_labels,
        cluster_probabilities,
    )
    revert_analysis = analyze_reverts(history, revisions)
    for point, statistics in zip(points, revert_analysis["point_statistics"]):
        point.update(statistics)

    diagnostics = {
        "pipeline": [
            "full XML history",
            f"{args.sample_strategy} complete revisions",
            "temporary token chunks",
            "one L2-normalized mean vector per revision",
            "cosine distance matrix",
            "classical metric MDS",
            "HDBSCAN directly on 2D MDS coordinates",
            "chronological curve colored by cluster",
        ],
        "sampling": {
            "strategy": args.sample_strategy,
            "start": args.start,
            "end": args.end,
            "min_revision_chars": args.min_revision_chars,
            "history_revision_count": len(history),
            "sampled_revision_count": len(revisions),
        },
        "analytical_unit": {
            "unit": "complete sampled revision",
            "points_per_revision": 1,
            "section_matching": False,
            "sentence_tracking": False,
            "chunk_matching": False,
        },
        "embedding": embedding_diagnostics,
        "semantic_distance": {
            "definition": "cosine distance between complete revision vectors",
            "formula": "1 - dot(L2_normalized_revision_a, L2_normalized_revision_b)",
            "normalization_or_clipping": False,
        },
        "projection": projection,
        "clustering": clustering,
        "edit_wars": revert_analysis,
        "display_interaction": {
            "zoom": "wheel or trackpad",
            "pan": "drag",
        },
        "points": points,
        "drift_records": drift,
    }

    html_text = build_visualization(args.article, points, diagnostics)
    output = next_numbered_path(paths["output_dir"], paths["output_basename"])
    stem = os.path.splitext(output)[0]
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(html_text)
    with open(stem + "_projection.json", "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2)
    np.save(stem + "_revision_embeddings.npy", revision_vectors)
    np.save(stem + "_cosine_distance_matrix.npy", semantic)

    print(f"\nHTML: {output}")
    print(f"Projektion: {stem}_projection.json")
    print(f"Revisionsvektoren: {stem}_revision_embeddings.npy")
    print(f"Distanzmatrix: {stem}_cosine_distance_matrix.npy")
    print(f"Revisionen / Punkte: {len(revisions)}")
    print(f"Stress-1: {projection['quality']['normalized_stress_1']:.6f}")
    print(f"Shepard r: {projection['quality']['shepard_correlation']:.6f}")
    print(f"HDBSCAN-Cluster: {clustering['cluster_count']}")
    print(f"Noise-Punkte: {clustering['noise_point_count']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, ET.ParseError) as error:
        print(f"[FEHLER] {error}")
        raise SystemExit(1) from error
