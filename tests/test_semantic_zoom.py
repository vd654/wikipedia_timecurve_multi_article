import json
import sys
import unittest.mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_zoom import (  # noqa: E402
    LABEL_SYSTEM_PROMPT,
    MACRO_LABEL_SYSTEM_PROMPT,
    NodeLabel,
    NodeLabelBatch,
    build_hierarchy,
    build_vega_spec,
    cached_word_counts,
    chronological_runs,
    cluster_colour,
    cluster_hull_rows,
    cluster_label_colour,
    convex_hull,
    deduplicate_label_rows,
    edge_phrase,
    fallback_label,
    format_label_pyramid,
    generate_labels,
    macro_hull_rows,
    parent_topic_label,
    render_html,
    spatial_macro_groups,
    valid_parent_label,
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
                "difference_lod": [
                    {"level": 1, "label": "International affairs", "description": "Broad domain."},
                    {"level": 2, "label": f"Macro theme {index}", "description": "Macro theme."},
                    {"level": 3, "label": f"Cluster topic {index}", "description": "Focused topic."},
                    {"level": 4, "label": f"Section change {index}", "description": "Specific theme."},
                    {"level": 5, "label": f"Topic {index}", "description": "Exact transition."},
                ],
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
    assert hierarchy["edges"][0]["count_text"] == "+3/−1 words"
    assert "words" not in hierarchy["edges"][0]["label"]
    assert hierarchy["edges"][0]["label"] == "Topic 0"
    assert hierarchy["edges"][0]["label_source"] == "full_revision_lod_llm"
    assert hierarchy["edges"][0]["section_terms"][0] == "Section change 0"
    assert hierarchy["edges"][0]["cluster_terms"][0] == "Cluster topic 0"
    assert hierarchy["edges"][0]["macro_terms"][0] == "Macro theme 0"
    assert hierarchy["edges"][0]["change_type"] == "replacement"
    assert " · " not in hierarchy["edges"][0]["label"]
    assert all(
        not edge["label"] or 1 <= len(edge["label"].split()) <= 4
        for edge in hierarchy["edges"]
    )
    assert hierarchy["edges"][0]["parent_id"].startswith("section-")
    assert hierarchy["edges"][-1]["parent_id"] == "root"


def test_fallback_label_uses_only_the_most_relevant_complete_term():
    assert fallback_label(
        ["Statehood recognition", "Interlanguage links"],
        "Fallback",
    ) == "Statehood recognition"
    assert fallback_label([], "Fallback") == "Fallback"


def test_minimal_full_revision_edge_labels_feed_parent_terms():
    projection = sample_projection()
    keywords = {
        "revision_pairs": [
            {
                "previous_revision_id": str(100 + index),
                "current_revision_id": str(101 + index),
                "edge_label": f"Topic {index}",
            }
            for index in range(5)
        ]
    }

    hierarchy = build_hierarchy(projection, keywords, root_label="Example")

    assert hierarchy["edges"][0]["label"] == "Topic 0"
    assert hierarchy["edges"][0]["section_terms"] == ["Topic 0"]
    assert hierarchy["edges"][0]["cluster_terms"] == ["Topic 0"]
    assert hierarchy["edges"][0]["macro_terms"] == ["Topic 0"]
    assert hierarchy["sections"][0]["terms"]
    assert hierarchy["clusters"][0]["terms"]


def test_edge_phrase_keeps_one_complete_phrase_of_at_most_four_words():
    assert edge_phrase(["UN General Assembly"]) == "UN General Assembly"
    assert edge_phrase(["one two three four five", "observer state status"]) == (
        "observer state status"
    )
    assert edge_phrase(["one two three four five"]) == ""
    assert edge_phrase(["  Currency   codes  "]) == "Currency codes"


def test_neutral_edge_labels_only_ground_otherwise_empty_clusters():
    assert parent_topic_label("No content change") == ""
    assert parent_topic_label("Identical article text") == ""
    assert parent_topic_label("East Jerusalem capital") == "East Jerusalem capital"

    projection = sample_projection()
    keywords = {
        "revision_pairs": [
            {
                "previous_revision_id": str(100 + index),
                "current_revision_id": str(101 + index),
                "edge_label": (
                    "No content change" if index < 4 else "East Jerusalem capital"
                ),
            }
            for index in range(5)
        ]
    }

    hierarchy = build_hierarchy(projection, keywords, root_label="Example")

    assert hierarchy["edges"][0]["label"] == "No content change"
    assert hierarchy["edges"][0]["section_terms"] == []
    assert hierarchy["sections"] == []
    assert all(
        node["terms"] == ["No content change"]
        for node in hierarchy["clusters"]
    )


def test_parent_prompts_require_one_english_label():
    for prompt in (LABEL_SYSTEM_PROMPT, MACRO_LABEL_SYSTEM_PROMPT):
        normalized = " ".join(prompt.split())
        assert "one English" in normalized
        assert "two to three words" in normalized
        assert "Never join, enumerate, or contrast" in normalized


def test_parent_label_validation_requires_two_to_three_words():
    assert valid_parent_label("Observer status")
    assert valid_parent_label("UN observer status")
    assert not valid_parent_label("Statehood")
    assert not valid_parent_label("Palestinian United Nations observer status")


def test_parent_label_generation_retries_invalid_word_count():
    client = unittest.mock.Mock()
    client.responses.parse.side_effect = [
        unittest.mock.Mock(
            output_parsed=NodeLabelBatch(labels=[
                NodeLabel(node_id="cluster-1", label="Palestinian observer state status")
            ]),
            output_text="",
        ),
        unittest.mock.Mock(
            output_parsed=NodeLabelBatch(labels=[
                NodeLabel(node_id="cluster-1", label="Observer state status")
            ]),
            output_text="",
        ),
    ]
    nodes = [{
        "node_id": "cluster-1",
        "level": "cluster",
        "revision_count": 2,
        "terms": ["observer state status"],
        "descriptions": [],
    }]

    assert generate_labels(client, "test-model", nodes) == {
        "cluster-1": "Observer state status"
    }
    assert client.responses.parse.call_count == 2
    correction = client.responses.parse.call_args.kwargs["input"][1]["content"]
    assert "2 to 3 words" in correction


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
    assert "article → macro topic → topic cluster" in output
    assert "country → state → region → place → street" in output
    assert "Article/  [1 Macros | 2 Cluster | 1 Sections | 5 Edges]" in output
    assert "└──" in output
    assert "Topic" in output


def test_exact_overlapping_edges_all_remain_label_candidates():
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
    assert "edge-2" in edge_label_ids
    rendered_edges = next(
        data for data in spec["data"] if data["name"] == "edges"
    )["values"]
    assert len(rendered_edges) == 5
    assert rendered_edges[0]["overlap_count"] == 2
    assert rendered_edges[1]["overlap_count"] == 2


def test_detailed_edge_description_is_rendered_as_multiline_label():
    projection = sample_projection()
    hierarchy = build_hierarchy(projection, sample_keywords())
    description = (
        "On 2 December 2012, ExampleEditor revised the observer-status section "
        "and explained the international recognition wording."
    )
    hierarchy["edges"][0]["detailed_description"] = description

    spec = build_vega_spec("Example", projection, hierarchy)
    label_rows = next(
        data for data in spec["data"] if data["name"] == "labels"
    )["values"]
    edge_row = next(row for row in label_rows if row["node_id"] == "edge-1")

    assert isinstance(edge_row["display_label"], list)
    assert len(edge_row["display_label"]) > 1
    assert " ".join(edge_row["display_label"]) == description


def test_child_label_identical_to_ancestor_remains_a_map_candidate():
    projection = sample_projection()
    hierarchy = build_hierarchy(projection, sample_keywords())
    hierarchy["macros"][0]["label"] = "Shared parent label"
    hierarchy["clusters"][0]["label"] = "Shared parent label"
    hierarchy["clusters"][0]["x"] = hierarchy["macros"][0]["x"]
    hierarchy["clusters"][0]["y"] = hierarchy["macros"][0]["y"]
    hierarchy["clusters"][1]["label"] = "Shared parent label detail"

    spec = build_vega_spec("Example", projection, hierarchy)
    rows = next(data for data in spec["data"] if data["name"] == "labels")["values"]
    node_ids = {row["node_id"] for row in rows}

    assert hierarchy["macros"][0]["node_id"] in node_ids
    assert hierarchy["clusters"][0]["node_id"] in node_ids
    assert hierarchy["clusters"][1]["node_id"] in node_ids
    coincident_child = next(
        row for row in rows
        if row["node_id"] == hierarchy["clusters"][0]["node_id"]
    )
    assert max(
        abs(coincident_child["label_dx"]),
        abs(coincident_child["label_dy"]),
    ) == 24


def test_duplicate_labels_keep_highest_priority_before_importance():
    rows = deduplicate_label_rows([
        {
            "node_id": "cluster-9",
            "label": "Palestinian Statehood Recognition",
            "display_label": "Palestinian Statehood Recognition",
            "label_priority": 3,
            "importance": 999.0,
            "revision_count": 999,
        },
        {
            "node_id": "macro-2",
            "label": "palestinian statehood-recognition",
            "display_label": "palestinian statehood-recognition",
            "label_priority": 4,
            "importance": 1.0,
            "revision_count": 1,
        },
    ])

    assert [row["node_id"] for row in rows] == ["macro-2"]


def test_equal_priority_duplicate_labels_remain_map_candidates():
    projection = sample_projection()
    hierarchy = build_hierarchy(projection, sample_keywords())
    first_macro = hierarchy["macros"][0]
    first_macro["label"] = "Palestinian Statehood Recognition"
    first_macro["importance"] = 24.0
    second_macro = {
        **first_macro,
        "node_id": "macro-2",
        "cluster_ids": [],
        "x": first_macro["x"] + 0.2,
        "label": "palestinian statehood recognition",
        "importance": 82.0,
        "revision_count": 82,
    }
    hierarchy["macros"].append(second_macro)

    spec = build_vega_spec("Example", projection, hierarchy)
    rows = next(data for data in spec["data"] if data["name"] == "labels")["values"]
    matching = [
        row for row in rows
        if row["display_label"].casefold() == "palestinian statehood recognition"
    ]

    assert [row["node_id"] for row in matching] == ["macro-1", "macro-2"]


def test_equal_edge_labels_remain_distinct_candidates():
    rows = deduplicate_label_rows([
        {
            "node_id": "edge-1",
            "level": "edge",
            "label": "No content change",
            "display_label": "No content change",
            "label_priority": 1,
            "importance": 1.0,
        },
        {
            "node_id": "edge-2",
            "level": "edge",
            "label": "No content change",
            "display_label": "No content change",
            "label_priority": 1,
            "importance": 2.0,
        },
    ])

    assert [row["node_id"] for row in rows] == ["edge-1", "edge-2"]


def test_no_content_change_labels_are_not_rendered():
    projection = sample_projection()
    hierarchy = build_hierarchy(projection, sample_keywords())
    hierarchy["edges"][0]["label"] = "No content change"
    hierarchy["sections"][0]["label"] = "Unchanged content"

    spec = build_vega_spec("Example", projection, hierarchy)
    labels = next(data for data in spec["data"] if data["name"] == "labels")["values"]
    rendered_labels = {row["label"].casefold() for row in labels}

    assert "no content change" not in rendered_labels
    assert "unchanged content" not in rendered_labels
    assert len(next(data for data in spec["data"] if data["name"] == "edges")["values"]) == 5


def test_exact_revert_symbol_replaces_all_points_at_the_same_projection_position():
    projection = sample_projection()
    projection["points"][2]["interval_exact_sha1_revert_count"] = 1
    projection["points"][0]["mds"] = list(projection["points"][2]["mds"])
    hierarchy = build_hierarchy(projection, sample_keywords())

    spec = build_vega_spec("Example", projection, hierarchy)
    point_rows = next(data for data in spec["data"] if data["name"] == "points")["values"]
    glyph_rows = next(
        data for data in spec["data"] if data["name"] == "exactRevertPoints"
    )["values"]

    shared_position = tuple(projection["points"][2]["mds"])
    coincident_rows = [
        row for row in point_rows if (row["x"], row["y"]) == shared_position
    ]
    assert len(coincident_rows) == 2
    assert all(row["is_exact_revert_position"] for row in coincident_rows)
    assert len(glyph_rows) == 1
    assert (glyph_rows[0]["x"], glyph_rows[0]["y"]) == shared_position


def test_vega_spec_uses_continuous_cartographic_label_decluttering():
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
    hierarchy["sections"][0]["label"] = "Distinct section label"
    spec = build_vega_spec("Example", projection, hierarchy)
    assert any(signal["name"] == "zoomLevel" for signal in spec["signals"])
    signals = {signal["name"]: signal for signal in spec["signals"]}
    assert signals["minViewScale"]["value"] == 0.01
    assert signals["maxViewScale"]["value"] == 50.0
    assert signals["viewScale"]["update"] == "span(xext)/span(xdom)"
    assert "edgeDetailBlend" not in signals
    zoom_update = signals["zoom"]["on"][0]["update"]
    assert "viewScale/maxViewScale" in zoom_update
    assert "viewScale/minViewScale" in zoom_update
    assert "max(1, span(xext)/span(xdom))" not in zoom_update
    signal_names = {signal["name"] for signal in spec["signals"]}
    assert "semanticLayer" not in signal_names
    assert not signal_names & {
        "macroBlend", "clusterBlend", "sectionBlend", "edgeBlend",
        "macroOpacity", "clusterOpacity", "sectionOpacity", "edgeOpacity",
    }
    assert any(signal["name"] == "zooming" for signal in spec["signals"])
    assert spec["autosize"] == "none"
    trajectory = next(mark for mark in spec["marks"] if mark.get("name") == "trajectory")
    assert trajectory["type"] == "rule"
    assert trajectory["encode"]["update"]["strokeOpacity"] == {"value": 0.42}
    width_signal = trajectory["encode"]["update"]["strokeWidth"]["signal"]
    assert "datum.has_exact_revert" not in width_signal
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
    assert all("edge_relevance" not in edge for edge in rendered_edges)
    assert all("semantic_jump_percentile" not in edge for edge in rendered_edges)
    intra_edges = next(data for data in spec["data"] if data["name"] == "intraClusterEdges")
    assert intra_edges["transform"][0]["expr"] == "datum.same_cluster"
    intra_mark = next(mark for mark in spec["marks"] if mark.get("name") == "intraClusterTrajectory")
    assert intra_mark["encode"]["update"]["stroke"] == {
        "scale": "clusterColor", "field": "cluster_id"
    }
    assert intra_mark["encode"]["update"]["strokeWidth"] == {
        "signal": width_signal
    }
    assert intra_mark["encode"]["update"]["strokeOpacity"] == {"value": 0.52}
    exact_revert_data = next(
        data for data in spec["data"] if data["name"] == "exactRevertEdges"
    )
    assert exact_revert_data["transform"][0]["expr"] == "datum.has_exact_revert"
    normal_point_data = next(
        data for data in spec["data"] if data["name"] == "normalRevisionPoints"
    )
    exact_point_data = next(
        data for data in spec["data"] if data["name"] == "exactRevertPoints"
    )
    assert normal_point_data["transform"][0]["expr"] == (
        "!datum.is_exact_revert_position"
    )
    assert "source" not in exact_point_data
    assert len(exact_point_data["values"]) == 1
    assert exact_point_data["values"][0]["exact_revert_position_count"] == 1
    exact_revert_mask = next(
        mark for mark in spec["marks"] if mark.get("name") == "exactRevertGlyphMasks"
    )
    exact_revert_arc = next(
        mark for mark in spec["marks"] if mark.get("name") == "exactRevertGlyphArcs"
    )
    exact_revert_head = next(
        mark for mark in spec["marks"] if mark.get("name") == "exactRevertGlyphHeads"
    )
    assert exact_revert_mask["type"] == "symbol"
    assert exact_revert_mask["from"] == {"data": "exactRevertPoints"}
    assert exact_revert_mask["encode"]["update"]["shape"] == {"value": "square"}
    assert exact_revert_mask["encode"]["update"]["size"] == {"value": 196}
    assert exact_revert_mask["encode"]["update"]["fill"] == {"value": "#fbfaf7"}
    assert exact_revert_mask["zindex"] < exact_revert_arc["zindex"]
    assert exact_revert_arc["type"] == "arc"
    assert exact_revert_arc["from"] == {"data": "exactRevertPoints"}
    assert exact_revert_arc["encode"]["update"]["innerRadius"] == {"value": 6.2}
    assert exact_revert_arc["encode"]["update"]["outerRadius"] == {"value": 9.4}
    assert exact_revert_arc["encode"]["update"]["fill"] == {
        "value": "#d64545"
    }
    assert "Exact revert" in exact_revert_arc["encode"]["enter"]["tooltip"]["signal"]
    assert exact_revert_head["type"] == "symbol"
    assert exact_revert_head["from"] == {"data": "exactRevertPoints"}
    assert exact_revert_head["encode"]["update"]["shape"] == {
        "value": "triangle-right"
    }
    assert exact_revert_head["encode"]["update"]["size"] == {"value": 42}
    assert exact_revert_head["encode"]["update"]["fill"] == {
        "value": "#d64545"
    }
    assert exact_revert_arc["zindex"] < 3
    assert exact_revert_head["zindex"] < 3
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
    assert edit_war_mark["encode"]["update"]["strokeOpacity"] == {"value": 0.24}
    assert "10" in edit_war_mark["encode"]["update"]["strokeWidth"]["signal"]
    mark_names = [mark.get("name") for mark in spec["marks"]]
    assert mark_names.index("editWarHalo") < mark_names.index("trajectory")
    assert "exactRevertTrajectory" not in mark_names
    assert "tooltip" not in edit_war_mark["encode"]["update"]
    assert all(
        mark.get("name") != "editWarDirectionArrows" for mark in spec["marks"]
    )
    arrow_data = next(
        data for data in spec["data"] if data["name"] == "visibleDirectionArrows"
    )
    assert arrow_data["source"] == "edges"
    assert arrow_data["transform"][0] == {
        "type": "filter", "expr": "!datum.has_exact_revert"
    }
    assert any(
        transform.get("type") == "filter"
        and ">=34" in transform.get("expr", "")
        for transform in arrow_data["transform"]
    )
    arrow_transforms = {
        transform["as"]: transform["expr"]
        for transform in arrow_data["transform"]
        if "as" in transform
    }
    assert arrow_transforms["screenLength"] == (
        "sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2))"
    )
    assert arrow_transforms["visibleScreenLength"] == (
        "(datum.clipT1-datum.clipT0)*datum.screenLength"
    )
    assert arrow_transforms["arrowT"] == (
        "datum.clipT0 + min(30,datum.visibleScreenLength*0.5)/datum.screenLength"
    )
    arrow_mark = next(
        mark for mark in spec["marks"] if mark.get("name") == "directionArrows"
    )
    assert arrow_mark["from"] == {"data": "visibleDirectionArrows"}
    assert arrow_mark["interactive"] is False
    assert arrow_mark["zindex"] < next(
        mark["zindex"] for mark in spec["marks"]
        if mark.get("name") == "semanticLabels"
    )
    assert arrow_mark["zindex"] < next(
        mark["zindex"] for mark in spec["marks"]
        if mark.get("name") == "edgeLabels"
    )
    assert arrow_mark["encode"]["update"]["shape"] == {"value": "triangle-right"}
    assert arrow_mark["encode"]["update"]["angle"] == {"field": "arrowAngle"}
    assert arrow_mark["encode"]["update"]["size"] == {"value": 58}
    assert arrow_mark["encode"]["update"]["fillOpacity"] == {"value": 0.9}
    assert arrow_mark["encode"]["update"]["fill"] == {
        "signal": (
            "datum.same_cluster ? "
            "scale('clusterColor', datum.cluster_id) : '#72806f'"
        )
    }
    assert len(next(data for data in spec["data"] if data["name"] == "edges")["values"]) == 5
    colour_scale = next(scale for scale in spec["scales"] if scale["name"] == "clusterColor")
    assert colour_scale["domain"] == [-1, 0, 1]
    assert colour_scale["range"] == ["#8a8a8a", cluster_colour(0), cluster_colour(1)]
    point_mark = next(mark for mark in spec["marks"] if mark.get("name") == "revisionPoints")
    assert point_mark["from"] == {"data": "normalRevisionPoints"}
    assert point_mark["encode"]["update"]["fill"] == {
        "scale": "clusterColor", "field": "cluster_id"
    }
    point_tooltip = point_mark["encode"]["enter"]["tooltip"]["signal"]
    assert point_tooltip == (
        "{'Revision ID': datum.revision_id, 'Timestamp': datum.timestamp, "
        "'Author': datum.author, 'Cluster': datum.cluster_label}"
    )
    trajectory_mark = next(mark for mark in spec["marks"] if mark.get("name") == "trajectory")
    assert "tooltip" not in trajectory_mark["encode"]["update"]
    section_point_data = next(
        data for data in spec["data"] if data["name"] == "labelledSectionPoints"
    )
    assert section_point_data["transform"][0]["expr"] == (
        "datum.labelled_section_point && !datum.is_exact_revert_position"
    )
    section_point_mark = next(
        mark for mark in spec["marks"] if mark.get("name") == "sectionPointHighlights"
    )
    assert section_point_mark["interactive"] is False
    assert point_mark["encode"]["update"]["size"] == {"value": 120}
    assert point_mark["encode"]["update"]["strokeWidth"] == {"value": 1.2}
    assert section_point_mark["encode"]["update"]["size"] == {"value": 270}
    assert section_point_mark["encode"]["update"]["fillOpacity"] == {"value": 0}
    assert section_point_mark["encode"]["update"]["stroke"] == {
        "scale": "clusterColor", "field": "cluster_id"
    }
    assert section_point_mark["encode"]["update"]["strokeOpacity"] == {"value": 0.92}
    assert section_point_mark["encode"]["update"]["strokeWidth"] == {"value": 2.6}
    visible = next(data for data in spec["data"] if data["name"] == "visibleParentLabels")
    assert "datum.level==='root'" in visible["transform"][0]["expr"]
    assert "datum.level==='macro'" in visible["transform"][0]["expr"]
    assert "datum.level==='endpoint'" in visible["transform"][0]["expr"]
    assert "datum.level==='cluster'" in visible["transform"][0]["expr"]
    assert "datum.level==='section'" in visible["transform"][0]["expr"]
    assert "!zooming" not in visible["transform"][0]["expr"]
    assert "zoomLevel" not in json.dumps(visible["transform"])
    assert "width-95" in visible["transform"][2]["expr"]
    assert all(
        "blockedParentLabelIds" not in transform.get("expr", "")
        for transform in visible["transform"]
    )
    assert "label_priority*1e15" in visible["transform"][-2]["expr"]
    assert "stableParentLabelIds" in visible["transform"][-2]["expr"]
    assert not any(transform.get("type") == "window" for transform in visible["transform"])
    assert not any("levelRank" in transform.get("expr", "") for transform in visible["transform"])
    text_mark = next(mark for mark in spec["marks"] if mark.get("name") == "semanticLabels")
    labels_data = next(data for data in spec["data"] if data["name"] == "labels")
    root_row = next(row for row in labels_data["values"] if row["level"] == "root")
    assert root_row["label"] == "Article"
    assert root_row["label_priority"] == 7
    endpoint_rows = [row for row in labels_data["values"] if row["level"] == "endpoint"]
    assert [row["label"] for row in endpoint_rows] == ["Start", "End"]
    assert not any(
        row.get("edge_label_dummy") or row.get("semantic_label_dummy")
        for row in labels_data["values"]
    )
    parent_candidates = next(data for data in spec["data"] if data["name"] == "visibleParentLabels")
    assert "datum.level==='macro'" in parent_candidates["transform"][0]["expr"]
    cluster_row = next(row for row in labels_data["values"] if row["level"] == "cluster")
    assert cluster_row["display_label"] == cluster_row["label"]
    assert cluster_row["label_priority"] == 3
    section_row = next(row for row in labels_data["values"] if row["level"] == "section")
    edge_row = next(row for row in labels_data["values"] if row["level"] == "edge")
    assert section_row["label_priority"] == 2
    assert edge_row["label_priority"] == 1
    assert edge_row["parent_id"] == hierarchy["edges"][0]["parent_id"]
    assert edge_row["parent_id"] in edge_row["ancestor_ids"]
    assert "root" in edge_row["ancestor_ids"]
    assert root_row["label_priority"] > max(
        row["label_priority"] for row in endpoint_rows
    )
    assert root_row["label_priority"] > cluster_row["label_priority"]
    assert cluster_row["label_priority"] > section_row["label_priority"]
    assert section_row["label_priority"] > edge_row["label_priority"]
    assert all(row["label_priority"] == 6 for row in endpoint_rows)
    assert "Cluster 1" not in cluster_row["display_label"]
    assert text_mark["encode"]["update"]["text"]["field"] == "display_label"
    assert text_mark["encode"]["update"]["fill"] == {"field": "label_colour"}
    assert "datum.label_dx" in text_mark["encode"]["update"]["x"]["signal"]
    assert "datum.label_dy" in text_mark["encode"]["update"]["y"]["signal"]
    assert text_mark["encode"]["update"]["align"] == {"field": "label_align"}
    assert text_mark["encode"]["update"]["baseline"] == {
        "field": "label_baseline"
    }
    assert "blockedParentLabelIds" in text_mark["encode"]["update"]["opacity"]["signal"]
    assert text_mark["encode"]["update"]["fillOpacity"] == {"value": 1}
    assert text_mark["encode"]["update"]["stroke"] == {"value": "#ffffff"}
    assert text_mark["encode"]["update"]["strokeWidth"] == {"value": 4}
    assert text_mark["encode"]["update"]["strokeOpacity"] == {"value": 1}
    assert text_mark["encode"]["update"]["fontWeight"] == {"value": 700}
    assert "datum.level==='root' ? 28" in (
        text_mark["encode"]["update"]["fontSize"]["signal"]
    )
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
    assert "transform" not in text_mark
    assert text_mark["from"]["data"] == "visibleParentLabels"
    assert text_mark["key"] == "node_id"
    mark_names = [mark.get("name") for mark in spec["marks"]]
    assert "macroLabels" not in mark_names
    assert "rootLabel" not in mark_names
    assert "semanticLabelHalos" not in mark_names
    edge_text = next(mark for mark in spec["marks"] if mark.get("name") == "edgeLabels")
    assert edge_text["encode"]["update"]["text"] == {
        "signal": "zoomLevel>=2 ? datum.display_label : datum.label"
    }
    assert edge_text["encode"]["update"]["lineHeight"] == {"value": 10}
    assert "tooltip" not in edge_text["encode"]["update"]
    assert "datum.labelScreenX" in edge_text["encode"]["update"]["x"]["signal"]
    assert "datum.labelScreenY" in edge_text["encode"]["update"]["y"]["signal"]
    assert "datum.screenDy" in edge_text["encode"]["update"]["x"]["signal"]
    assert "datum.screenDx" in edge_text["encode"]["update"]["y"]["signal"]
    assert edge_text["from"]["data"] == "visibleEdgeLabels"
    assert edge_text["key"] == "node_id"
    assert "edgeLabelHalos" not in mark_names
    assert edge_text["encode"]["update"]["stroke"] == {"value": "#ffffff"}
    assert edge_text["encode"]["update"]["strokeWidth"] == {"value": 2}
    assert edge_text["encode"]["update"]["strokeOpacity"] == {"value": 1}
    assert "transform" not in edge_text
    edge_candidates = next(data for data in spec["data"] if data["name"] == "visibleEdgeLabels")
    assert edge_candidates["transform"][0]["expr"] == "datum.level==='edge'"
    assert not any(
        "zoomLevel" in transform.get("expr", "")
        for transform in edge_candidates["transform"]
    )
    assert not any("length(datum.display_label)" in transform.get("expr", "") for transform in edge_candidates["transform"])
    transforms_by_field = {
        transform["as"]: transform
        for transform in edge_candidates["transform"]
        if isinstance(transform.get("as"), str)
    }
    assert transforms_by_field["screenX1"]["expr"] == "scale('xscale', datum.edge_x)"
    assert transforms_by_field["screenY1"]["expr"] == "scale('yscale', datum.edge_y)"
    assert transforms_by_field["screenX2"]["expr"] == "scale('xscale', datum.edge_x2)"
    assert transforms_by_field["screenY2"]["expr"] == "scale('yscale', datum.edge_y2)"
    assert transforms_by_field["clipT0"]["expr"] == "max(0,datum.xClipMin,datum.yClipMin)"
    assert transforms_by_field["clipT1"]["expr"] == "min(1,datum.xClipMax,datum.yClipMax)"
    assert any(
        transform.get("type") == "filter"
        and transform.get("expr") == "datum.clipT0<=datum.clipT1"
        for transform in edge_candidates["transform"]
    )
    assert transforms_by_field["placementOrder"] == {
        "type": "formula",
        "as": "placementOrder",
        "expr": "-(datum.label_priority*1e15 + (indexof(stableEdgeLabelIds, datum.node_id)>=0 ? 1e14 : 0) + datum.importance)",
    }
    assert edge_candidates["transform"][-1] == {
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
    assert "stableParentPlacements" not in signal_values
    assert signal_values["blockedParentLabelIds"] == []
    assert signal_values["blockedEdgeLabelIds"] == []
    assert "coveredParentLabelIds" not in signal_values
    assert edge_text["encode"]["update"]["text"] == {
        "signal": "zoomLevel>=2 ? datum.display_label : datum.label"
    }
    assert edge_text["encode"]["update"]["align"]["value"] == "center"
    assert edge_text["encode"]["update"]["baseline"]["value"] == "middle"
    assert "angle" not in edge_text["encode"]["update"]
    assert edge_text["encode"]["update"]["fontWeight"] == {"value": 700}
    assert edge_text["encode"]["update"]["fontSize"] == {"value": 8}
    assert "blockedEdgeLabelIds" in edge_text["encode"]["update"]["opacity"]["signal"]
    assert edge_text["encode"]["update"]["fillOpacity"] == {"value": 1}
    assert "unstableParentLabels" not in data_names
    assert "stableParentLabels" not in data_names
    assert "edgeLabelAnchors" not in mark_names
    assert "semanticLabelAnchors" not in mark_names
    assert "stableSemanticLabels" not in mark_names
    assert "stableSemanticLabelHalos" not in mark_names
    assert root_row["label_dx"] == 0
    assert root_row["label_dy"] == 0
    assert root_row["label_align"] == "center"
    assert root_row["label_baseline"] == "middle"
    assert all(
        {"label_dx", "label_dy", "label_align", "label_baseline"} <= row.keys()
        for row in labels_data["values"]
        if row["level"] != "edge"
    )
    assert not any(scale["name"] == "pointSize" for scale in spec["scales"])
    revision_points = next(
        mark for mark in spec["marks"] if mark.get("name") == "revisionPoints"
    )
    section_highlights = next(
        mark for mark in spec["marks"]
        if mark.get("name") == "sectionPointHighlights"
    )
    assert revision_points["encode"]["update"]["size"] == {"value": 120}
    assert section_highlights["encode"]["update"]["size"] == {"value": 270}
    assert not any(mark.get("name") in {"rootCenterMark", "clusterCenterMarks"} for mark in spec["marks"])
    assert "+3/−1 words" not in json.dumps(spec, ensure_ascii=False)
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
    assert '"name":"directionArrows"' in page
    assert '"name":"visibleDirectionArrows"' in page
    assert '"name":"exactRevertGlyphMasks"' in page
    assert '"name":"exactRevertGlyphArcs"' in page
    assert '"name":"exactRevertGlyphHeads"' in page
    assert '"name":"exactRevertTrajectory"' not in page
    assert "!datum.has_exact_revert" in page
    assert '"name":"editWarTrajectory"' not in page
    assert '"strokeDash":{"value":[8,5]}' not in page
    assert "vega.parse(spec)" in page
    assert "renderer:'svg'" in page
    assert "transition:opacity" not in page
    assert "paint-order:stroke fill" in page
    assert "renderer:'canvas'" not in page
    assert "vega@5.33.1" in page
    assert "<h1>Meaningful Projection Space</h1>" in page
    assert '<html lang="en">' in page
    assert "<title>Semantic Zoom Timecurve" in page
    assert "ResizeObserver" in page
    assert "width:min(100%,1200px)" in page
    assert "height:min(100%,720px)" in page
    assert 'id="content"' in page
    assert 'id="legend"' in page
    assert "grid-template-columns:minmax(0,1fr) 250px" in page
    assert "Standard trajectory edge" in page
    assert '<span>Repeated path</span>' in page
    assert '<span>Exact revert target</span>' in page
    assert '<span>Edit War</span>' in page
    assert "Temporal-section point" in page
    assert "<h2>Legend</h2>" in page
    assert "Cluster 1" in page
    assert "Cluster 2" in page
    assert "Noise" in page
    assert "renderedLabelIds('macroLabels')" not in page
    assert "renderedLabelIds('semanticLabels')" in page
    assert "renderedLabelIds('edgeLabels')" in page
    assert "stableParentLabelIds" in page
    assert "stableEdgeLabelIds" in page
    assert "blockedParentLabelIds" in page
    assert "blockedEdgeLabelIds" in page
    assert "coveredParentLabelIds" not in page
    assert "renderedLabelItems" in page
    assert "boxesOverlap" in page
    assert "const labelCollisionPadding=4" in page
    assert "mode==='collision'" in page
    assert "outsideViewport" in page
    assert "class LabelSpatialHash" in page
    assert "occupied.overlaps(candidate)" in page
    assert "collisionBlocks" in page
    assert "hierarchyParentBlocks" not in page
    assert "detailChildThreshold" not in page
    assert "capturedParentPlacements" not in page
    assert "stableParentPlacements" not in page
    assert "renderedObstacleItems" not in page
    assert "vis.dataset.semanticLayer" not in page
    assert "vis.dataset.zoom" in page
    assert "vis.dataset.visibleLabels" in page
    assert "vis.dataset.stableParentLabels" in page
    assert "vis.dataset.stableEdgeLabels" in page
    assert "vis.dataset.blockedParentLabels" in page
    assert "vis.dataset.blockedEdgeLabels" in page
    assert "view.addSignalListener('zooming'" in page
    assert "view.addSignalListener('zoomLevel'" in page
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
