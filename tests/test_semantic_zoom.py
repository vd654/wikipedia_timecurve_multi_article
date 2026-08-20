import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_zoom import (  # noqa: E402
    build_hierarchy,
    build_vega_spec,
    cached_word_counts,
    chronological_runs,
    cluster_colour,
    cluster_hull_rows,
    cluster_label_colour,
    convex_hull,
    edge_phrase,
    format_label_pyramid,
    macro_hull_rows,
    render_html,
    spatial_macro_groups,
    validate_sources,
)


def sample_projection():
    clusters = [0, 0, 0, 1, 1, -1]
    return {
        "points": [
            {
                "revision_index": index,
                "revision_id": str(100 + index),
                "timestamp": f"2020-01-0{index + 1}T00:00:00Z",
                "author": "Editor",
                "clean_word_count": 100 + index,
                "mds": [index / 10, (index % 2) / 10],
                "cluster_id": cluster,
                "semantic_jump_from_previous": 0 if index == 0 else index / 100,
            }
            for index, cluster in enumerate(clusters)
        ]
    }


def sample_keywords():
    return {
        "revision_pairs": [
            {
                "previous_revision_id": str(100 + index),
                "current_revision_id": str(101 + index),
                "edge_label": f"Topic {index}",
                "change_type": "replacement",
                "evidence": [f"Topic {index}"],
                "confidence": 0.9,
                "conflict_description": (
                    "Shared session description" if index == 1 else ""
                ),
                "focus_keywords": [f"Topic {index}", "Security Council", "policy", "history", "vote"],
                "added_keywords": ["new claim"],
                "removed_keywords": ["old claim"],
            }
            for index in range(5)
        ]
    }


def test_validate_sources_detects_alignment_error():
    projection = sample_projection()
    keywords = sample_keywords()
    keywords["revision_pairs"][2]["current_revision_id"] = "wrong"
    try:
        validate_sources(projection, keywords)
    except ValueError as error:
        assert "passt nicht" in str(error)
    else:
        raise AssertionError("Fehlende Paarvalidierung")


def test_runs_and_five_level_hierarchy():
    projection = sample_projection()
    hierarchy = build_hierarchy(
        projection,
        sample_keywords(),
        word_counts=[(3, 1), (2, 4), (7, 0), (1, 2), (5, 5)],
        root_label="Example",
    )
    assert chronological_runs(projection["points"]) == [
        {"cluster_id": 0, "start": 0, "end": 3},
        {"cluster_id": 1, "start": 3, "end": 5},
        {"cluster_id": -1, "start": 5, "end": 6},
    ]
    assert len(hierarchy["clusters"]) == 2
    assert len(hierarchy["macros"]) == 1
    assert hierarchy["macros"][0]["cluster_ids"] == [0, 1]
    assert hierarchy["clusters"][0]["hdbscan_label"] == "Cluster 1"
    assert len(hierarchy["sections"]) == 1
    assert len(hierarchy["edges"]) == 5
    assert [edge["importance"] for edge in hierarchy["edges"]] == [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ]
    assert hierarchy["roots"][0]["label"] == "Example"
    assert hierarchy["clusters"][0]["parent_id"] == "macro-1"
    assert hierarchy["macros"][0]["parent_id"] == "root"
    assert hierarchy["edges"][0]["count_text"] == "+3/−1 Wörter"
    assert "Wörter" not in hierarchy["edges"][0]["label"]
    assert hierarchy["edges"][0]["label"] == "Topic 0"
    assert hierarchy["edges"][0]["label_source"] == "direct_diff_llm"
    assert hierarchy["edges"][0]["change_type"] == "replacement"
    assert " · " not in hierarchy["edges"][0]["label"]
    assert all(
        not edge["label"] or 1 <= len(edge["label"].split()) <= 4
        for edge in hierarchy["edges"]
    )
    assert hierarchy["edges"][0]["parent_id"].startswith("section-")
    assert hierarchy["edges"][-1]["parent_id"] == "root"


def test_edge_phrase_keeps_one_complete_phrase_of_at_most_four_words():
    assert edge_phrase(["UN General Assembly"]) == "UN General Assembly"
    assert edge_phrase(["one two three four five", "observer state status"]) == (
        "observer state status"
    )
    assert edge_phrase(["one two three four five"]) == ""
    assert edge_phrase(["  Currency   codes  "]) == "Currency codes"


def test_noise_revisions_remain_leaf_edges_without_sections():
    projection = sample_projection()
    for index in range(3, 6):
        projection["points"][index]["cluster_id"] = -1
    hierarchy = build_hierarchy(projection, sample_keywords())

    assert all(section["cluster_id"] >= 0 for section in hierarchy["sections"])
    assert len(hierarchy["edges"]) == 5
    assert hierarchy["edges"][3]["parent_id"] == "root"
    assert hierarchy["edges"][4]["parent_id"] == "root"


def test_edges_without_word_diff_receive_an_explicit_short_label():
    projection = sample_projection()
    keywords = sample_keywords()
    keywords["revision_pairs"][0].update({
        "edge_label": "",
        "evidence": [],
        "focus_keywords": [],
        "added_keywords": [],
        "removed_keywords": [],
    })
    hierarchy = build_hierarchy(
        projection,
        keywords,
        word_counts=[(0, 0), (2, 4), (7, 0), (1, 2), (5, 5)],
    )
    assert hierarchy["edges"][0]["label"] == "No cleaned-text change"


def test_legacy_keyword_file_remains_a_supported_fallback():
    keywords = sample_keywords()
    pair = keywords["revision_pairs"][0]
    pair.pop("edge_label")
    pair.pop("evidence")
    pair.pop("change_type")
    pair.pop("confidence")

    hierarchy = build_hierarchy(sample_projection(), keywords)

    assert hierarchy["edges"][0]["label"] == "Topic 0"
    assert hierarchy["edges"][0]["label_source"] == "legacy_keyword_fallback"


def test_invalid_long_direct_label_is_not_silently_truncated():
    keywords = sample_keywords()
    keywords["revision_pairs"][0]["edge_label"] = "one two three four five"

    hierarchy = build_hierarchy(sample_projection(), keywords)

    assert hierarchy["edges"][0]["label"] == "Topic 0"
    assert hierarchy["edges"][0]["label"] != "one two three four"


def test_terminal_label_pyramid_is_continuous_and_compact():
    hierarchy = build_hierarchy(sample_projection(), sample_keywords())
    output = format_label_pyramid(hierarchy)
    assert "Artikel → Makrothema → Themencluster" in output
    assert "Land → Bundesland → Region → Ort → Straße" in output
    assert "Article/  [1 Macros | 2 Cluster | 1 Sections | 5 Übergänge]" in output
    assert "└──" in output
    assert "Topic" in output


def test_exact_overlapping_edges_keep_only_the_earliest_timestamp_label():
    projection = sample_projection()
    projection["points"][0]["mds"] = [0.0, 0.0]
    projection["points"][1]["mds"] = [1.0, 0.0]
    projection["points"][2]["mds"] = [0.0, 5e-14]
    hierarchy = build_hierarchy(projection, sample_keywords())
    spec = build_vega_spec("Example", projection, hierarchy)

    label_rows = next(data for data in spec["data"] if data["name"] == "labels")["values"]
    edge_label_ids = {
        row["node_id"] for row in label_rows if row["level"] == "edge"
    }
    assert "edge-1" in edge_label_ids
    assert "edge-2" not in edge_label_ids
    rendered_edges = next(
        data for data in spec["data"] if data["name"] == "edges"
    )["values"]
    assert len(rendered_edges) == 5
    assert rendered_edges[0]["overlap_count"] == 2
    assert rendered_edges[1]["overlap_count"] == 2


def test_vega_spec_uses_semantic_zoom_and_cartographic_label_transform():
    projection = sample_projection()
    projection["points"][2].update({
        "interval_revert_count": 2,
        "interval_exact_sha1_revert_count": 1,
        "interval_partial_text_revert_count": 1,
        "interval_mutual_revert_count": 1,
        "interval_edit_war_pairs": [["Alice", "Bob"]],
        "cumulative_gandica_c": 4.0,
        "interval_confirmed_edit_war_event_count": 2,
        "interval_confirmed_edit_war_sessions": [{
            "session_id": "edit-war-1",
            "pair": ["Alice", "Bob"],
            "event_count": 3,
            "local_gandica_c": 6.0,
        }],
    })
    projection["points"][3].update({
        "interval_confirmed_edit_war_event_count": 1,
        "interval_confirmed_edit_war_sessions": [{
            "session_id": "edit-war-1",
            "pair": ["Alice", "Bob"],
            "event_count": 3,
            "local_gandica_c": 6.0,
        }],
    })
    hierarchy = build_hierarchy(projection, sample_keywords())
    spec = build_vega_spec("Example", projection, hierarchy)
    assert any(signal["name"] == "zoomLevel" for signal in spec["signals"])
    layer_signal = next(signal for signal in spec["signals"] if signal["name"] == "semanticLayer")
    assert "zoomLevel < 2.15 ? 'macro'" in layer_signal["update"]
    assert "zoomLevel < 3.45 ? 'cluster'" in layer_signal["update"]
    assert "zoomLevel < 4.4 ? 'section'" in layer_signal["update"]
    assert any(signal["name"] == "rootOpacity" for signal in spec["signals"])
    assert any(signal["name"] == "zooming" for signal in spec["signals"])
    assert spec["autosize"] == "none"
    trajectory = next(mark for mark in spec["marks"] if mark.get("name") == "trajectory")
    assert trajectory["type"] == "rule"
    assert "semanticLayer==='macro' ? 0.08" in trajectory["encode"]["update"]["strokeOpacity"]["signal"]
    assert "semanticLayer==='section' ? 0.34" in (
        trajectory["encode"]["update"]["strokeOpacity"]["signal"]
    )
    assert "datum.in_section ? 0.34 : 0" not in (
        trajectory["encode"]["update"]["strokeOpacity"]["signal"]
    )
    width_signal = trajectory["encode"]["update"]["strokeWidth"]["signal"]
    assert "datum.has_exact_revert" in width_signal
    assert "datum.overlap_count" in width_signal
    assert not any(scale["name"] == "jumpWidth" for scale in spec["scales"])
    rendered_edges = next(
        data for data in spec["data"] if data["name"] == "edges"
    )["values"]
    assert any(edge["in_section"] for edge in rendered_edges)
    assert any(not edge["in_section"] for edge in rendered_edges)
    assert rendered_edges[0]["change_type"] == "replacement"
    assert rendered_edges[0]["evidence_text"] == "Topic 0"
    assert rendered_edges[0]["label_confidence"] == 0.9
    assert rendered_edges[1]["edit_war_description"] == "Shared session description"
    assert rendered_edges[1]["has_exact_revert"] is True
    assert rendered_edges[1]["exact_revert_count"] == 1
    assert rendered_edges[1]["partial_revert_count"] == 1
    intra_edges = next(data for data in spec["data"] if data["name"] == "intraClusterEdges")
    assert intra_edges["transform"][0]["expr"] == "datum.same_cluster"
    intra_mark = next(mark for mark in spec["marks"] if mark.get("name") == "intraClusterTrajectory")
    assert intra_mark["encode"]["update"]["stroke"] == {
        "scale": "clusterColor", "field": "cluster_id"
    }
    assert intra_mark["encode"]["update"]["strokeWidth"] == {
        "signal": width_signal
    }
    exact_revert_data = next(
        data for data in spec["data"] if data["name"] == "exactRevertEdges"
    )
    assert exact_revert_data["transform"][0]["expr"] == "datum.has_exact_revert"
    exact_revert_mark = next(
        mark for mark in spec["marks"]
        if mark.get("name") == "exactRevertTrajectory"
    )
    assert exact_revert_mark["from"] == {"data": "exactRevertEdges"}
    assert exact_revert_mark["encode"]["update"]["stroke"] == {
        "value": "#d64545"
    }
    assert exact_revert_mark["encode"]["update"]["strokeOpacity"] == {
        "signal": "semanticLayer==='root' ? 0 : 0.92"
    }
    assert exact_revert_mark["encode"]["update"]["strokeWidth"] == {
        "signal": width_signal
    }
    assert "tooltip" not in exact_revert_mark["encode"]["update"]
    edit_war_data = next(
        data for data in spec["data"] if data["name"] == "editWarEdges"
    )
    assert edit_war_data["transform"][0]["expr"] == "datum.is_edit_war"
    assert all(data["name"] != "editWarArrowEdges" for data in spec["data"])
    assert sum(edge["is_edit_war"] for edge in rendered_edges) == 2
    edit_war_mark = next(
        mark for mark in spec["marks"] if mark.get("name") == "editWarHalo"
    )
    assert edit_war_mark["from"] == {"data": "editWarEdges"}
    assert edit_war_mark["encode"]["update"]["stroke"] == {"value": "#1f9d55"}
    assert "strokeDash" not in edit_war_mark["encode"]["update"]
    assert edit_war_mark["encode"]["update"]["strokeOpacity"] == {
        "signal": "semanticLayer==='root' ? 0 : 0.24"
    }
    assert "7.5" in edit_war_mark["encode"]["update"]["strokeWidth"]["signal"]
    mark_names = [mark.get("name") for mark in spec["marks"]]
    assert mark_names.index("editWarHalo") < mark_names.index("trajectory")
    assert mark_names.index("trajectory") < mark_names.index("exactRevertTrajectory")
    assert "tooltip" not in edit_war_mark["encode"]["update"]
    assert all(
        mark.get("name") != "editWarDirectionArrows" for mark in spec["marks"]
    )
    assert len(next(data for data in spec["data"] if data["name"] == "edges")["values"]) == 5
    colour_scale = next(scale for scale in spec["scales"] if scale["name"] == "clusterColor")
    assert colour_scale["domain"] == [-1, 0, 1]
    assert colour_scale["range"] == ["#8a8a8a", cluster_colour(0), cluster_colour(1)]
    point_mark = next(mark for mark in spec["marks"] if mark.get("name") == "revisionPoints")
    assert point_mark["encode"]["update"]["fill"] == {
        "scale": "clusterColor", "field": "cluster_id"
    }
    point_tooltip = point_mark["encode"]["enter"]["tooltip"]["signal"]
    assert point_tooltip == (
        "{'Revision-ID': datum.revision_id, 'Zeit': datum.timestamp, "
        "'Autor': datum.author, 'Cluster': datum.cluster_label}"
    )
    trajectory_mark = next(mark for mark in spec["marks"] if mark.get("name") == "trajectory")
    assert "tooltip" not in trajectory_mark["encode"]["update"]
    section_point_data = next(
        data for data in spec["data"] if data["name"] == "labelledSectionPoints"
    )
    assert section_point_data["transform"][0]["expr"] == "datum.labelled_section_point"
    section_point_mark = next(
        mark for mark in spec["marks"] if mark.get("name") == "sectionPointHighlights"
    )
    assert section_point_mark["interactive"] is False
    assert section_point_mark["encode"]["update"]["size"] == {"value": 170}
    assert section_point_mark["encode"]["update"]["fillOpacity"] == {"value": 0}
    assert section_point_mark["encode"]["update"]["stroke"] == {
        "scale": "clusterColor", "field": "cluster_id"
    }
    assert section_point_mark["encode"]["update"]["strokeOpacity"]["signal"] == "0.92 * sectionOpacity"
    assert section_point_mark["encode"]["update"]["strokeWidth"] == {"value": 2}
    root_data = next(data for data in spec["data"] if data["name"] == "rootNode")
    assert root_data["values"][0]["label"] == "Article"
    visible_macro = next(
        data for data in spec["data"] if data["name"] == "visibleMacroLabels"
    )
    assert visible_macro["transform"][0]["expr"] == (
        "!zooming && !down && datum.level==='macro' && macroOpacity>0.03"
    )
    assert "stableParentLabelIds" in visible_macro["transform"][-2]["expr"]
    visible = next(data for data in spec["data"] if data["name"] == "visibleParentLabels")
    assert visible["transform"][0]["expr"] == "!zooming && !down"
    assert "datum.level==='macro'" not in visible["transform"][1]["expr"]
    assert "datum.level==='endpoint'" in visible["transform"][1]["expr"]
    assert "stableParentLabelIds" in visible["transform"][-2]["expr"]
    assert not any(transform.get("type") == "window" for transform in visible["transform"])
    assert not any("levelRank" in transform.get("expr", "") for transform in visible["transform"])
    opacity_updates = {
        signal["name"]: signal["update"]
        for signal in spec["signals"]
        if signal["name"] in {"rootOpacity", "macroOpacity", "clusterOpacity", "sectionOpacity", "edgeOpacity"}
    }
    assert opacity_updates == {
        "rootOpacity": "semanticLayer==='root' ? 1 : 0",
        "macroOpacity": "semanticLayer==='macro' ? 1 : 0",
        "clusterOpacity": "semanticLayer==='cluster' ? 1 : 0",
        "sectionOpacity": "semanticLayer==='section' ? 1 : 0",
        "edgeOpacity": "semanticLayer==='edge' ? 1 : 0",
    }
    macro_text_mark = next(
        mark for mark in spec["marks"] if mark.get("name") == "macroLabels"
    )
    macro_halo_mark = next(
        mark for mark in spec["marks"] if mark.get("name") == "macroLabelHalos"
    )
    text_mark = next(mark for mark in spec["marks"] if mark.get("name") == "semanticLabels")
    halo_mark = next(
        mark for mark in spec["marks"] if mark.get("name") == "semanticLabelHalos"
    )
    labels_data = next(data for data in spec["data"] if data["name"] == "labels")
    endpoint_rows = [row for row in labels_data["values"] if row["level"] == "endpoint"]
    assert [row["label"] for row in endpoint_rows] == ["Start", "Ende"]
    assert not any(
        row.get("edge_label_dummy") or row.get("semantic_label_dummy")
        for row in labels_data["values"]
    )
    parent_candidates = next(data for data in spec["data"] if data["name"] == "visibleParentLabels")
    assert "datum.level==='macro'" not in parent_candidates["transform"][1]["expr"]
    cluster_row = next(row for row in labels_data["values"] if row["level"] == "cluster")
    assert cluster_row["display_label"] == cluster_row["label"]
    assert "Cluster 1" not in cluster_row["display_label"]
    assert text_mark["encode"]["update"]["text"]["field"] == "datum.display_label"
    assert text_mark["encode"]["update"]["fill"] == {"field": "datum.label_colour"}
    assert "opacity" not in text_mark["encode"]["update"]
    assert "stroke" not in text_mark["encode"]["update"]
    assert halo_mark["encode"]["update"]["stroke"] == {"value": "#fbfaf7"}
    assert halo_mark["encode"]["update"]["strokeWidth"] == {"value": 4}
    assert text_mark["encode"]["update"]["fontWeight"] == {"value": 700}
    hull_data = next(data for data in spec["data"] if data["name"] == "clusterHulls")
    assert {row["cluster_id"] for row in hull_data["values"]} == {0}
    hull_mark = next(mark for mark in spec["marks"] if mark.get("name") == "clusterRegions")
    assert hull_mark["from"]["facet"]["data"] == "clusterHulls"
    data_names = {data["name"] for data in spec["data"]}
    mark_names = {mark.get("name") for mark in spec["marks"]}
    assert "sectionHulls" not in data_names
    assert "sectionCorridors" not in data_names
    assert "sectionRegionPoints" not in data_names
    assert "sectionRegions" not in mark_names
    assert "sectionRegionCorridors" not in mark_names
    assert "sectionRegionPointCaps" not in mark_names
    label_transform = next(
        item for item in text_mark["transform"] if item["type"] == "label"
    )
    assert label_transform["avoidBaseMark"] is False
    assert label_transform["avoidMarks"] == ["revisionPoints", "macroLabels"]
    assert label_transform["anchor"] == [
        "top",
        "bottom",
        "right",
        "left",
        "top-right",
        "top-left",
        "bottom-right",
        "bottom-left",
    ]
    assert label_transform["offset"] == [9, 9, 10, 10, 12, 12, 12, 12]
    assert text_mark["from"]["data"] == "semanticLabelAnchors"
    assert halo_mark["from"]["data"] == "semanticLabels"
    assert text_mark["key"] == "datum.node_id"
    assert halo_mark["key"] == "datum.datum.node_id"
    assert text_mark["zindex"] > halo_mark["zindex"]
    macro_label_transform = next(
        item for item in macro_text_mark["transform"] if item["type"] == "label"
    )
    assert macro_text_mark["from"]["data"] == "macroLabelAnchors"
    assert macro_label_transform["anchor"] == ["middle"]
    assert macro_label_transform["offset"] == [0]
    assert "avoidMarks" not in macro_label_transform
    assert macro_text_mark["encode"]["update"]["align"] == {"value": "center"}
    assert macro_text_mark["encode"]["update"]["baseline"] == {"value": "middle"}
    assert macro_text_mark["encode"]["update"]["fontSize"] == {"value": 19}
    assert macro_halo_mark["from"]["data"] == "macroLabels"
    assert macro_text_mark["zindex"] > macro_halo_mark["zindex"]
    mark_names = [mark.get("name") for mark in spec["marks"]]
    assert mark_names.index("macroLabels") < mark_names.index("semanticLabels")
    assert mark_names.index("semanticLabels") < mark_names.index("semanticLabelHalos")
    edge_text = next(mark for mark in spec["marks"] if mark.get("name") == "edgeLabels")
    edge_halo = next(
        mark for mark in spec["marks"] if mark.get("name") == "edgeLabelHalos"
    )
    edge_anchors = next(
        mark for mark in spec["marks"] if mark.get("name") == "edgeLabelAnchors"
    )
    assert "datum.edge_x2" in edge_anchors["encode"]["update"]["x"]["signal"]
    assert "datum.edge_y2" in edge_anchors["encode"]["update"]["y"]["signal"]
    assert " - 9 * " in edge_anchors["encode"]["update"]["x"]["signal"]
    assert " + 9 * " in edge_anchors["encode"]["update"]["y"]["signal"]
    assert edge_text["from"]["data"] == "edgeLabelAnchors"
    assert edge_text["key"] == "datum.node_id"
    assert edge_halo["from"]["data"] == "edgeLabels"
    assert edge_halo["key"] == "datum.datum.node_id"
    assert edge_text["zindex"] > edge_halo["zindex"]
    assert edge_halo["encode"]["update"]["fillOpacity"] == {"value": 0}
    assert edge_halo["encode"]["update"]["stroke"] == {"value": "#fbfaf7"}
    assert edge_halo["encode"]["update"]["strokeWidth"] == {"value": 4}
    assert edge_text["transform"][0]["anchor"] == ["middle"]
    assert edge_text["transform"][0]["offset"] == [0]
    assert edge_text["transform"][0]["padding"] == 5
    assert edge_text["transform"][0]["avoidMarks"] == [
        "revisionPoints", "macroLabels", "semanticLabels"
    ]
    edge_candidates = next(data for data in spec["data"] if data["name"] == "visibleEdgeLabels")
    assert edge_candidates["transform"][0]["expr"] == "!zooming && !down && datum.level==='edge' && edgeOpacity>0.03"
    assert edge_candidates["transform"][2] == {
        "type": "formula",
        "as": "placementOrder",
        "expr": "-(datum.importance + (indexof(stableEdgeLabelIds, datum.node_id)>=0 ? 1e15 : 0))",
    }
    assert edge_candidates["transform"][3] == {
        "type": "collect",
        "sort": {"field": "placementOrder", "order": "ascending"},
    }
    assert not any(
        "span(xdom)" in transform.get("expr", "")
        or "edgeRank" in transform.get("expr", "")
        or transform.get("as") == ["edgeRank"]
        for transform in edge_candidates["transform"]
    )
    assert all(
        "datum.level==='endpoint'" not in transform.get("expr", "")
        for transform in edge_candidates["transform"]
    )
    signal_values = {signal["name"]: signal.get("value") for signal in spec["signals"]}
    assert signal_values["stableParentLabelIds"] == []
    assert signal_values["stableEdgeLabelIds"] == []
    assert edge_text["encode"]["update"]["text"] == {"field": "datum.label"}
    assert edge_text["encode"]["update"]["align"]["value"] == "center"
    assert edge_text["encode"]["update"]["baseline"]["value"] == "middle"
    assert edge_text["encode"]["update"]["fontWeight"] == {"value": 700}
    assert "opacity" not in edge_text["encode"]["update"]
    root_text = next(mark for mark in spec["marks"] if mark.get("name") == "rootLabel")
    assert root_text["encode"]["enter"]["fontWeight"] == {"value": 700}
    assert not any(scale["name"] == "pointSize" for scale in spec["scales"])
    revision_points = next(
        mark for mark in spec["marks"] if mark.get("name") == "revisionPoints"
    )
    section_highlights = next(
        mark for mark in spec["marks"]
        if mark.get("name") == "sectionPointHighlights"
    )
    assert revision_points["encode"]["update"]["size"] == {"value": 64}
    assert section_highlights["encode"]["update"]["size"] == {"value": 170}
    assert not any(mark.get("name") in {"rootCenterMark", "clusterCenterMarks"} for mark in spec["marks"])
    assert "+3/−1 Wörter" not in json.dumps(spec, ensure_ascii=False)
    assert all(
        " · " not in row["label"]
        for row in next(data for data in spec["data"] if data["name"] == "edges")["values"]
    )
    assert "Level 1" not in json.dumps(spec)
    assert "dblclick" in json.dumps(spec)


def test_html_has_no_interactive_model_call():
    projection = sample_projection()
    hierarchy = build_hierarchy(projection, sample_keywords())
    page = render_html("Example", build_vega_spec("Example", projection, hierarchy))
    assert "fetch(" not in page
    assert "OpenAI" not in page
    assert '"name":"editWarHalo"' in page
    assert '"name":"editWarDirectionArrows"' not in page
    assert '"name":"editWarTrajectory"' not in page
    assert '"strokeDash":{"value":[8,5]}' not in page
    assert "vega.parse(spec)" in page
    assert "vega@5.33.1" in page
    assert "<h1>Meaningful Projection Space</h1>" in page
    assert "<h1>Semantische Revisions-Timecurve" not in page
    assert "ResizeObserver" in page
    assert "width:min(100%,1200px)" in page
    assert "height:min(100%,720px)" in page
    assert 'id="content"' in page
    assert 'id="legend"' in page
    assert "grid-template-columns:minmax(0,1fr) 250px" in page
    assert "Normale Trajektorienkante" in page
    assert '<span>Mehrfach belegter Weg</span>' in page
    assert '<span>Exakter Revert</span>' in page
    assert "Exakter Revert oder mehrfach belegter Weg" not in page
    assert '<span>Edit War</span>' in page
    assert "Bestätigter Edit War" not in page
    assert "Punkt eines zeitlichen Abschnitts" in page
    assert "Semantisches Label der aktiven Zoomstufe" not in page
    assert "Erste und letzte Revision" not in page
    assert "Die Linienbreite steigt" not in page
    assert "Cluster 1" in page
    assert "Cluster 2" in page
    assert "Noise" in page
    assert "renderedLabelIds('macroLabels')" in page
    assert "renderedLabelIds('semanticLabels')" in page
    assert "renderedLabelIds('edgeLabels')" in page
    assert "stableParentLabelIds" in page
    assert "stableEdgeLabelIds" in page
    assert "vis.dataset.semanticLayer" in page
    assert "vis.dataset.stableParentLabels" in page
    assert "vis.dataset.stableEdgeLabels" in page
    assert "view.addSignalListener('zooming'" in page
    assert "view.addSignalListener('down'" in page
    assert "Zoom 1.0×" not in page
    assert "Scrollen zum Zoomen" not in page


def test_cluster_colours_match_original_timecurve_palette():
    assert cluster_colour(-1) == "#8a8a8a"
    assert cluster_colour(0) == "#d32222"
    assert cluster_colour(1) == "#507ae2"
    assert cluster_colour(0) != cluster_colour(1)
    assert cluster_label_colour(0) != cluster_label_colour(1)


def test_cluster_hulls_are_closed_and_exclude_noise():
    assert convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]) == [
        (0, 0), (1, 0), (1, 1), (0, 1)
    ]
    rows = cluster_hull_rows([
        {"cluster_id": 0, "x": 0.0, "y": 0.0},
        {"cluster_id": 0, "x": 1.0, "y": 0.0},
        {"cluster_id": 0, "x": 0.0, "y": 1.0},
        {"cluster_id": -1, "x": 5.0, "y": 5.0},
    ], expansion=1.0)
    assert len(rows) == 4
    assert rows[0]["x"] == rows[-1]["x"]
    assert rows[0]["y"] == rows[-1]["y"]
    assert {row["cluster_id"] for row in rows} == {0}


def test_macro_groups_and_hulls_are_deterministic():
    clusters = [
        {"cluster_id": 0, "x": 0.0, "y": 0.0, "revision_count": 3},
        {"cluster_id": 1, "x": 0.1, "y": 0.0, "revision_count": 2},
        {"cluster_id": 2, "x": 5.0, "y": 5.0, "revision_count": 4},
    ]
    assert spatial_macro_groups(clusters, target_count=2) == [[0, 1], [2]]
    rows = macro_hull_rows([
        {"macro_id": "macro-1", "x": 0.0, "y": 0.0},
        {"macro_id": "macro-1", "x": 1.0, "y": 0.0},
        {"macro_id": "macro-1", "x": 0.0, "y": 1.0},
        {"macro_id": None, "x": 5.0, "y": 5.0},
    ], expansion=1.0)
    assert len(rows) == 4
    assert {row["macro_id"] for row in rows} == {"macro-1"}


def test_cached_git_counts_are_reused(tmp_path):
    projection_path = (tmp_path / "projection.json").resolve()
    keywords_path = (tmp_path / "keywords.json").resolve()
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({
        "projection": str(projection_path),
        "revision_keywords": str(keywords_path),
        "word_counts_are_git_word_diff": True,
        "hierarchy": {"edges": [
            {"count_text": "+3/−1 Wörter"},
            {"count_text": "+0/−8 Wörter"},
        ]},
    }), encoding="utf-8")
    assert cached_word_counts(
        labels_path, projection_path, keywords_path, 2
    ) == [(3, 1), (0, 8)]
