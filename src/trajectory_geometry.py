"""Pure geometry helpers for trajectories, cluster regions, and revert arcs."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable


REVERT_CONNECTION_ENDPOINT_TOLERANCE_RATIO = 0.05


def catmull_rom_bezier_controls(
    points: list[tuple[float, float]],
    *,
    alpha: float = 0.5,
) -> list[tuple[float, float, float, float]]:
    """Return cubic Bezier controls for an open Catmull-Rom point chain.

    Each result describes the segment from ``points[i]`` to ``points[i+1]``
    as ``(c1x, c1y, c2x, c2y)``.  The centripetal parameterisation
    (``alpha=0.5``) is deliberately used for the highly non-uniform distances
    in a 2D semantic projection: it is less prone to loops and overshoot than
    the uniform variant while still interpolating every revision point.

    Exact reverts can place consecutive revisions at identical coordinates.
    Those segments remain zero-length; neighbouring tangent handles are
    bounded to half the segment length so a duplicate cannot create a spike.
    """
    if len(points) < 2:
        return []
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Catmull-Rom alpha must be between 0 and 1.")

    def reflected(point: tuple[float, float], neighbour: tuple[float, float]):
        return (2.0 * point[0] - neighbour[0], 2.0 * point[1] - neighbour[1])

    def distance(first: tuple[float, float], second: tuple[float, float]):
        return math.hypot(second[0] - first[0], second[1] - first[1])

    def knot_delta(first: tuple[float, float], second: tuple[float, float]):
        # A tiny positive interval keeps the non-uniform tangent formula
        # defined when two exact-revert points coincide.
        return max(distance(first, second) ** alpha, 1e-12)

    def bounded(vector: tuple[float, float], maximum: float):
        length = math.hypot(*vector)
        if length <= maximum or length <= 1e-18:
            return vector
        scale = maximum / length
        return (vector[0] * scale, vector[1] * scale)

    controls = []
    for index in range(len(points) - 1):
        p1, p2 = points[index], points[index + 1]
        if distance(p1, p2) <= 1e-18:
            controls.append((p1[0], p1[1], p2[0], p2[1]))
            continue
        p0 = points[index - 1] if index > 0 else reflected(p1, p2)
        p3 = (
            points[index + 2]
            if index + 2 < len(points)
            else reflected(p2, p1)
        )
        dt01 = knot_delta(p0, p1)
        dt12 = knot_delta(p1, p2)
        dt23 = knot_delta(p2, p3)
        dt02 = dt01 + dt12
        dt13 = dt12 + dt23

        tangent1 = tuple(
            dt12 * (
                (p1[axis] - p0[axis]) / dt01
                - (p2[axis] - p0[axis]) / dt02
                + (p2[axis] - p1[axis]) / dt12
            )
            for axis in (0, 1)
        )
        tangent2 = tuple(
            dt12 * (
                (p2[axis] - p1[axis]) / dt12
                - (p3[axis] - p1[axis]) / dt13
                + (p3[axis] - p2[axis]) / dt23
            )
            for axis in (0, 1)
        )
        # The Bezier handle is one third of the Hermite tangent. Limiting the
        # handle itself to half the chord protects sparse/duplicate sequences
        # without moving either interpolated revision point.
        maximum_handle = distance(p1, p2) * 0.5
        handle1 = bounded(
            (tangent1[0] / 3.0, tangent1[1] / 3.0), maximum_handle
        )
        handle2 = bounded(
            (tangent2[0] / 3.0, tangent2[1] / 3.0), maximum_handle
        )
        controls.append((
            p1[0] + handle1[0],
            p1[1] + handle1[1],
            p2[0] - handle2[0],
            p2[1] - handle2[1],
        ))
    return controls


def convex_hull(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute a deterministic monotonic-chain hull without extra dependencies."""
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def cluster_region_rows(
    point_rows: list[dict],
    expansion: float = 1.08,
) -> tuple[list[dict], list[dict]]:
    """Create visible regions around every non-noise HDBSCAN cluster.

    A regular two-dimensional cluster receives an expanded convex polygon.
    Collinear or effectively coincident projected points cannot enclose a
    visible polygon, so their distinct projected positions are connected by a
    thin line instead. Rounding only decides whether a cluster is
    degenerate; it removes meaningless floating-point jitter between
    revisions that project to the same visible position.
    """
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for point in point_rows:
        if point["cluster_id"] >= 0:
            grouped[point["cluster_id"]].append((point["x"], point["y"]))

    polygons: list[dict] = []
    corridors: list[dict] = []
    for cluster_id, coordinates in sorted(grouped.items()):
        effective_coordinates = sorted({
            (round(float(x), 12), round(float(y), 12))
            for x, y in coordinates
        })
        effective_hull = convex_hull(effective_coordinates)
        if len(effective_hull) < 3:
            if len(effective_hull) == 2:
                start, end = effective_hull
                corridors.append({
                    "cluster_id": cluster_id,
                    "order": 0,
                    "x": start[0],
                    "y": start[1],
                    "x2": end[0],
                    "y2": end[1],
                })
            continue

        hull = convex_hull(coordinates)
        center_x = sum(x for x, _ in hull) / len(hull)
        center_y = sum(y for _, y in hull) / len(hull)
        expanded = [
            (
                center_x + (x - center_x) * expansion,
                center_y + (y - center_y) * expansion,
            )
            for x, y in hull
        ]
        for order, (x, y) in enumerate(expanded + expanded[:1]):
            polygons.append({
                "cluster_id": cluster_id,
                "order": order,
                "x": x,
                "y": y,
            })
    return polygons, corridors


def build_revert_arc_rows(
    point_rows: list[dict], revert_analysis: dict | None
) -> list[dict]:
    """Map revert events to styled screen-space arcs between plotted revisions.

    An exact SHA1 revert runs from the reverted revision to the reverting
    revision.  If the same visual return connection occurs repeatedly in
    either direction, near-identical endpoints are treated as one pair and
    only its chronologically last event is drawn.  Its retained orientation
    therefore always points towards that final revert data point.
    """
    if not revert_analysis:
        return []

    points_by_revision = {
        str(point["revision_id"]): point for point in point_rows
    }
    coordinate_counts = Counter(
        (
            round(float(point["x"]), 12),
            round(float(point["y"]), 12),
        )
        for point in point_rows
    )
    rows = []
    for event_order, event in enumerate(revert_analysis.get("events", [])):
        source_id = str(event.get("revision_id", ""))
        target_id = str(event.get("reverted_revision_id", ""))
        revert_point = points_by_revision.get(source_id)
        undone_point = points_by_revision.get(target_id)
        if (
            revert_point is None
            or undone_point is None
            or source_id == target_id
        ):
            continue
        reverter = str(event.get("reverter", ""))
        reverted_editor = str(event.get("reverted_editor", ""))
        revert_kind = str(event.get("revert_kind", "exact_sha1"))
        if revert_kind != "exact_sha1":
            raise ValueError(
                f"Unsupported revert category: {revert_kind!r}"
            )
        restored_id = str(event.get("restored_revision_id") or "")
        pair = [str(editor) for editor in event.get("pair", []) if editor]
        start_revision_id = target_id
        end_revision_id = source_id
        segment_start = points_by_revision[start_revision_id]
        segment_end = points_by_revision[end_revision_id]
        source_key = (
            round(float(segment_start["x"]), 12),
            round(float(segment_start["y"]), 12),
        )
        target_key = (
            round(float(segment_end["x"]), 12),
            round(float(segment_end["y"]), 12),
        )
        rows.append({
            "arc_id": f"revert-{event_order + 1}-{source_id}-{target_id}",
            "event_order": int(event.get("revision_index", event_order)),
            "segment_order": 0,
            "x": segment_start["x"],
            "y": segment_start["y"],
            "x2": segment_end["x"],
            "y2": segment_end["y"],
            "revision_id": source_id,
            "start_revision_id": start_revision_id,
            "end_revision_id": end_revision_id,
            "reverted_revision_id": target_id,
            "restored_revision_id": restored_id,
            "return_anchor_revision_id": end_revision_id,
            "source_overlap_count": coordinate_counts[source_key],
            "target_overlap_count": coordinate_counts[target_key],
            "reverter": reverter,
            "reverted_editor": reverted_editor,
            "pair_text": " ↔ ".join(pair),
            "revert_kind": revert_kind,
            "revert_kind_label": "Exact revert",
            "visual_category": revert_kind,
            "visual_category_label": "Exact revert",
            "reversal_score": float(event.get("reversal_score", 1.0)),
        })

    def endpoint_match_score(left: dict, right: dict) -> float | None:
        """Return the best undirected endpoint error for one visual arc pair."""
        left_length = math.hypot(
            float(left["x2"]) - float(left["x"]),
            float(left["y2"]) - float(left["y"]),
        )
        right_length = math.hypot(
            float(right["x2"]) - float(right["x"]),
            float(right["y2"]) - float(right["y"]),
        )
        direct_error = max(
            math.hypot(
                float(left["x"]) - float(right["x"]),
                float(left["y"]) - float(right["y"]),
            ),
            math.hypot(
                float(left["x2"]) - float(right["x2"]),
                float(left["y2"]) - float(right["y2"]),
            ),
        )
        reverse_error = max(
            math.hypot(
                float(left["x"]) - float(right["x2"]),
                float(left["y"]) - float(right["y2"]),
            ),
            math.hypot(
                float(left["x2"]) - float(right["x"]),
                float(left["y2"]) - float(right["y"]),
            ),
        )
        best_error = min(direct_error, reverse_error)
        tolerance = (
            min(left_length, right_length)
            * REVERT_CONNECTION_ENDPOINT_TOLERANCE_RATIO
        )
        if tolerance <= 1e-15:
            return best_error if best_error <= 1e-12 else None
        return best_error if best_error <= tolerance else None

    grouped: list[list[dict]] = []
    for row in sorted(rows, key=lambda item: (item["event_order"], item["arc_id"])):
        matches = [
            (score, group_index)
            for group_index, group in enumerate(grouped)
            if (score := endpoint_match_score(row, group[-1])) is not None
        ]
        if matches:
            _, group_index = min(matches)
            grouped[group_index].append(row)
        else:
            grouped.append([row])

    retained_rows = []
    for group in grouped:
        row = group[-1]
        row["arc_rank"] = 0
        row["arc_level"] = 0
        row["arc_count"] = 1
        row["revert_occurrence_count"] = len(group)
        row["suppressed_revert_arc_count"] = len(group) - 1
        row["arc_direction"] = 1
        row["arc_side"] = 1
        retained_rows.append(row)
    return sorted(
        retained_rows,
        key=lambda row: (row["event_order"], row["arc_id"]),
    )


def snap_exact_revert_positions(
    point_rows: list[dict], revert_analysis: dict | None
) -> None:
    """Place exact revert revisions on the restored revision coordinate.

    Identical SHA1 states are semantically identical even when an embedding or
    projection round-trip leaves their stored coordinates a few pixels apart.
    Snapping exact reverts makes the visual return literally coincide with the
    restored state.
    """
    if not revert_analysis:
        return
    points_by_revision = {
        str(point["revision_id"]): point for point in point_rows
    }
    for event in revert_analysis.get("events", []):
        if event.get("revert_kind") != "exact_sha1":
            continue
        revert_point = points_by_revision.get(str(event.get("revision_id", "")))
        restored_point = points_by_revision.get(
            str(event.get("restored_revision_id", ""))
        )
        if revert_point is None or restored_point is None:
            continue
        revert_point["projected_x_before_revert_snap"] = revert_point["x"]
        revert_point["projected_y_before_revert_snap"] = revert_point["y"]
        revert_point["x"] = restored_point["x"]
        revert_point["y"] = restored_point["y"]
        revert_point["exact_revert_restored_revision_id"] = str(
            event["restored_revision_id"]
        )


