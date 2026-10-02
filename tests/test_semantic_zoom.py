import json
import sys
import unittest.mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_zoom import (  # noqa: E402
    EDIT_WAR_COLOR,
    EDIT_WAR_ARC_STROKE_WIDTH,
    EDIT_WAR_ARC_STROKE_OPACITY,
    EXACT_REVERT_COLOR,
    NOISE_COLOR,
    PARTIAL_REVERT_COLOR,
    REVERT_ARC_STROKE_WIDTH,
    REVERT_ARC_STROKE_OPACITY,
    REVERT_ARROW_OPACITY,
    REVISION_POINT_SIZE,
    SECTION_POINT_HIGHLIGHT_SIZE,
    TRAJECTORY_COLOR,
    LABEL_SYSTEM_PROMPT,
    MACRO_LABEL_SYSTEM_PROMPT,
    NodeLabel,
    NodeLabelBatch,
    build_hierarchy,
    build_revert_arc_rows,
    build_vega_spec,
    cached_llm_labels,
    cached_macro_labels,
    cached_word_counts,
    chronological_runs,
    cluster_colour,
    cluster_hull_rows,
    cluster_label_colour,
    content_focused_parent_label,
    convex_hull,
    deduplicate_label_rows,
    edge_phrase,
    fallback_label,
    format_label_pyramid,
    generate_labels,
    hierarchy_specificity_issues,
    macro_hull_rows,
    parent_topic_label,
    render_html,
    source_sha256,
    snap_exact_revert_positions,
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


def test_neutral_edge_labels_do_not_create_false_cluster_topics():
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
    assert all(node["terms"] == [] for node in hierarchy["clusters"])
    assert all(node["label"] == "" for node in hierarchy["clusters"])


def test_parent_prompts_require_one_english_label():
    for prompt in (LABEL_SYSTEM_PROMPT, MACRO_LABEL_SYSTEM_PROMPT):
        normalized = " ".join(prompt.split())
        assert "one English" in normalized
        assert "two to three words" in normalized
        assert "Never join, enumerate, or contrast" in normalized
        assert "editing" in normalized


def test_parent_label_validation_requires_two_to_three_words():
    assert valid_parent_label("Observer status")
    assert valid_parent_label("UN observer status")
    assert not valid_parent_label("Statehood")
    assert not valid_parent_label("Palestinian United Nations observer status")


def test_parent_content_validation_rejects_editing_mechanics():
    assert content_focused_parent_label("PLO Representation Claims")
    assert not content_focused_parent_label("Em dash spacing")
    assert not content_focused_parent_label("Tourism Capitalization")
    assert not content_focused_parent_label("Palestine infobox details")
    assert not content_focused_parent_label("Governance citations")
    assert not content_focused_parent_label("Article Typography")
    assert not content_focused_parent_label("Palestinian State Context")
    assert not content_focused_parent_label("Palestinian Public Profile")


def test_reviewed_fixed_labels_ignore_prompt_and_edge_label_changes(tmp_path):
    projection_path = (tmp_path / "projection.json").resolve()
    keywords_path = (tmp_path / "labels.json").resolve()
    hierarchy_path = (tmp_path / "hierarchy.json").resolve()
    projection_path.write_text('{"points": []}', encoding="utf-8")
    keywords_path.write_text('{"revision_pairs": []}', encoding="utf-8")
    hierarchy_path.write_text(json.dumps({
        "projection": str(projection_path),
        "revision_keywords": str(keywords_path),
        "llm_provider": "different-provider",
        "model": "different-model",
        "label_system_prompt": "older prompt",
        "fixed_parent_labels": {
            "status": "reviewed",
            "projection_sha256": source_sha256([projection_path]),
            "revision_keywords_sha256": source_sha256([keywords_path]),
        },
        "hierarchy": {
            "clusters": [{"node_id": "cluster-1", "label": "Reviewed Topic"}],
            "sections": [],
        },
    }), encoding="utf-8")

    assert cached_llm_labels(
        hierarchy_path, projection_path, keywords_path, "new-model"
    ) == {"cluster-1": "Reviewed Topic"}

    keywords_path.write_text('{"revision_pairs": [1]}', encoding="utf-8")
    assert cached_llm_labels(
        hierarchy_path, projection_path, keywords_path, "new-model"
    ) == {"cluster-1": "Reviewed Topic"}

    projection_path.write_text('{"points": [1]}', encoding="utf-8")
    assert cached_llm_labels(
        hierarchy_path, projection_path, keywords_path, "new-model"
    ) == {}


def test_reviewed_fixed_macro_label_ignores_changed_edge_evidence(tmp_path):
    projection_path = (tmp_path / "projection.json").resolve()
    keywords_path = (tmp_path / "labels.json").resolve()
    hierarchy_path = (tmp_path / "hierarchy.json").resolve()
    projection_path.write_text('{"points": []}', encoding="utf-8")
    keywords_path.write_text('{"revision_pairs": []}', encoding="utf-8")
    hierarchy_path.write_text(json.dumps({
        "projection": str(projection_path),
        "revision_keywords": str(keywords_path),
        "fixed_parent_labels": {
            "status": "reviewed",
            "projection_sha256": source_sha256([projection_path]),
            "revision_keywords_sha256": source_sha256([keywords_path]),
        },
        "hierarchy": {
            "macros": [{
                "node_id": "macro-1",
                "cluster_ids": [1],
                "terms": ["old evidence"],
                "child_labels": ["Old Topic"],
                "label": "Reviewed Macro",
            }],
        },
    }), encoding="utf-8")

    macros = [{
        "node_id": "macro-1",
        "cluster_ids": [1],
        "terms": ["new evidence"],
        "child_labels": ["New Topic"],
    }]
    assert cached_macro_labels(
        hierarchy_path, projection_path, keywords_path, "new-model", macros
    ) == {"macro-1": "Reviewed Macro"}


def test_neutral_only_cluster_has_no_thematic_label():
    points = [
        {"revision_id": "1", "mds": [0.0, 0.0], "cluster_id": 0,
         "semantic_jump_from_previous": 0.0},
        {"revision_id": "2", "mds": [0.0, 0.0], "cluster_id": 0,
         "semantic_jump_from_previous": 0.0},
    ]
    projection = {"points": points}
    labels = {"revision_pairs": [{
        "previous_revision_id": "1",
        "current_revision_id": "2",
        "edge_label": "No Content Change",
        "detailed_description": "The cleaned article content did not change.",
    }]}

    hierarchy = build_hierarchy(projection, labels, root_label="Palestine")

    assert hierarchy["clusters"][0]["terms"] == []
    assert hierarchy["clusters"][0]["label"] == ""


def test_hierarchy_specificity_requires_each_child_to_add_information():
    hierarchy = {
        "roots": [{"node_id": "root", "label": "Palestine"}],
        "macros": [{
            "node_id": "macro-1", "parent_id": "root",
            "label": "Statehood Recognition",
        }],
        "clusters": [{
            "node_id": "cluster-1", "parent_id": "macro-1",
            "label": "Statehood Recognition",
        }],
        "sections": [],
        "edges": [{
            "node_id": "edge-1", "parent_id": "cluster-1",
            "label": "Vote Implications Added",
            "detailed_description": (
                "ExampleEditor added legal implications of the observer-state vote."
            ),
        }],
    }

    issues = hierarchy_specificity_issues(hierarchy)

    assert issues == [{
        "kind": "child_adds_no_specific_term",
        "parent_id": "macro-1",
        "parent_label": "Statehood Recognition",
        "child_id": "cluster-1",
        "child_label": "Statehood Recognition",
    }]


def test_hierarchy_specificity_rejects_broad_status_edge_labels():
    hierarchy = {
        "roots": [{"node_id": "root", "label": "Palestine"}],
        "macros": [{
            "node_id": "macro-1", "parent_id": "root",
            "label": "UN Observer Status",
        }],
        "clusters": [],
        "sections": [],
        "edges": [{
            "node_id": "edge-1", "parent_id": "root",
            "label": "PLO Observer Status",
            "detailed_description": (
                "ExampleEditor added the year of the PLO observer designation."
            ),
        }],
    }

    assert hierarchy_specificity_issues(hierarchy) == [{
        "kind": "edge_repeats_broad_topic",
        "node_id": "edge-1",
        "label": "PLO Observer Status",
    }]


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


def test_parent_label_generation_retries_editing_mechanics():
    client = unittest.mock.Mock()
    client.responses.parse.side_effect = [
        unittest.mock.Mock(
            output_parsed=NodeLabelBatch(labels=[
                NodeLabel(node_id="section-1", label="Em dash spacing")
            ]),
            output_text="",
        ),
        unittest.mock.Mock(
            output_parsed=NodeLabelBatch(labels=[
                NodeLabel(node_id="section-1", label="PLO Representation Claims")
            ]),
            output_text="",
        ),
    ]
    nodes = [{
        "node_id": "section-1",
        "level": "section",
        "revision_count": 2,
        "terms": ["Em dash spacing"],
        "descriptions": [
            "The sentence discusses UN representation and Jordanian claims."
        ],
    }]

    assert generate_labels(client, "test-model", nodes) == {
        "section-1": "PLO Representation Claims"
    }
    assert client.responses.parse.call_count == 2
    correction = client.responses.parse.call_args.kwargs["input"][1]["content"]
    assert "must not describe editing mechanics" in correction


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


def test_overlapping_edges_keep_only_the_larger_duplicate_label():
    projection = sample_projection()
    projection["points"][0]["mds"] = [0.0, 0.0]
    projection["points"][1]["mds"] = [1.0, 0.0]
    projection["points"][2]["mds"] = [0.0, 5e-14]
    hierarchy = build_hierarchy(projection, sample_keywords())
    hierarchy["edges"][0]["label"] = "Smaller overlap change"
    hierarchy["edges"][1]["label"] = "Larger overlap change"
    spec = build_vega_spec("Example", projection, hierarchy)

    label_rows = next(data for data in spec["data"] if data["name"] == "labels")["values"]
    edge_label_ids = {
        row["node_id"] for row in label_rows if row["level"] == "edge"
    }
    assert "edge-1" not in edge_label_ids
    assert "edge-2" in edge_label_ids
    rendered_edges = next(
        data for data in spec["data"] if data["name"] == "edges"
    )["values"]
    assert len(rendered_edges) == 5
    assert rendered_edges[0]["overlap_count"] == 2
    assert rendered_edges[1]["overlap_count"] == 2


def test_overlapping_edges_prefer_an_explicit_detailed_explanation():
    rows = deduplicate_label_rows([
        {
            "node_id": "edge-1",
            "level": "edge",
            "label": "Explained change",
            "display_label": ["A precise explanation of the revision"],
            "detailed_description": "A precise explanation of the revision",
            "label_priority": 1,
            "change_magnitude": 5,
            "importance": 0.5,
            "edge_x": 0.0,
            "edge_y": 0.0,
            "edge_x2": 1.0,
            "edge_y2": 1.0,
        },
        {
            "node_id": "edge-2",
            "level": "edge",
            "label": "Larger unexplained change",
            "display_label": "Larger unexplained change",
            "detailed_description": "",
            "label_priority": 1,
            "change_magnitude": 12,
            "importance": 0.9,
            "edge_x": 1.0,
            "edge_y": 1.0,
            "edge_x2": 0.0,
            "edge_y2": 0.0,
        },
    ])

    assert [row["node_id"] for row in rows] == ["edge-1"]


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
    point_rows = next(
        data for data in spec["data"] if data["name"] == "points"
    )["values"]

    assert isinstance(edge_row["display_label"], list)
    assert len(edge_row["display_label"]) > 1
    assert " ".join(edge_row["display_label"]) == description
    assert all("detailed_description" not in point for point in point_rows)
    assert all(
        row["level"] == "edge"
        for row in label_rows
        if row.get("detailed_description")
    )


def test_child_label_identical_to_ancestor_is_removed():
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
    assert hierarchy["clusters"][0]["node_id"] not in node_ids
    assert hierarchy["clusters"][1]["node_id"] in node_ids


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


def test_equal_priority_duplicate_parent_labels_keep_more_important_candidate():
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

    assert [row["node_id"] for row in matching] == ["macro-2"]


def test_equal_edge_labels_keep_the_larger_textual_change():
    rows = deduplicate_label_rows([
        {
            "node_id": "edge-1",
            "level": "edge",
            "label": "No content change",
            "display_label": "No content change",
            "label_priority": 1,
            "importance": 1.0,
            "change_magnitude": 5,
        },
        {
            "node_id": "edge-2",
            "level": "edge",
            "label": "No content change",
            "display_label": "No content change",
            "label_priority": 1,
            "importance": 0.5,
            "change_magnitude": 12,
        },
    ])

    assert [row["node_id"] for row in rows] == ["edge-2"]


def test_duplicate_detailed_edge_text_keeps_the_larger_textual_change():
    rows = deduplicate_label_rows([
        {
            "node_id": "edge-1",
            "level": "edge",
            "label": "First short label",
            "display_label": ["The same detailed explanation"],
            "label_priority": 1,
            "importance": 2.0,
            "change_magnitude": 5,
        },
        {
            "node_id": "edge-2",
            "level": "edge",
            "label": "Second short label",
            "display_label": ["The same detailed explanation"],
            "label_priority": 1,
            "importance": 1.0,
            "change_magnitude": 12,
        },
    ])

    assert [row["node_id"] for row in rows] == ["edge-2"]


def test_equal_edge_labels_on_distinct_geometry_are_both_kept():
    rows = deduplicate_label_rows([
        {
            "node_id": "edge-1",
            "level": "edge",
            "label": "Palestinian Demographics",
            "display_label": ["First demographic change"],
            "detailed_description": "First demographic change",
            "label_priority": 1,
            "edge_x": 0.0,
            "edge_y": 0.0,
            "edge_x2": 1.0,
            "edge_y2": 1.0,
        },
        {
            "node_id": "edge-2",
            "level": "edge",
            "label": "Palestinian Demographics",
            "display_label": ["Second demographic change"],
            "detailed_description": "Second demographic change",
            "label_priority": 1,
            "edge_x": 2.0,
            "edge_y": 2.0,
            "edge_x2": 3.0,
            "edge_y2": 3.0,
        },
    ])

    assert [row["node_id"] for row in rows] == ["edge-1", "edge-2"]


def test_no_content_change_labels_are_not_rendered():
    projection = sample_projection()
    hierarchy = build_hierarchy(projection, sample_keywords())
    hierarchy["edges"][0]["label"] = "No content change"
    hierarchy["sections"][0]["label"] = "Unchanged content"
    hierarchy["clusters"][0]["label"] = "Content unchanged"

    spec = build_vega_spec("Example", projection, hierarchy)
    labels = next(data for data in spec["data"] if data["name"] == "labels")["values"]
    rendered_labels = {row["label"].casefold() for row in labels}
    page = render_html("Example", spec)

    assert "no content change" not in rendered_labels
    assert "unchanged content" not in rendered_labels
    assert "content unchanged" not in rendered_labels
    assert "<span>Content unchanged</span>" in page
    assert len(next(data for data in spec["data"] if data["name"] == "edges")["values"]) == 5


def test_exact_revert_keeps_normal_points_at_the_same_projection_position():
    projection = sample_projection()
    projection["points"][2]["interval_exact_sha1_revert_count"] = 1
    projection["points"][0]["mds"] = list(projection["points"][2]["mds"])
    hierarchy = build_hierarchy(projection, sample_keywords())

    spec = build_vega_spec("Example", projection, hierarchy)
    point_rows = next(data for data in spec["data"] if data["name"] == "points")["values"]
    normal_points = next(
        data for data in spec["data"] if data["name"] == "normalRevisionPoints"
    )

    shared_position = tuple(projection["points"][2]["mds"])
    coincident_rows = [
        row for row in point_rows if (row["x"], row["y"]) == shared_position
    ]
    assert len(coincident_rows) == 2
    assert normal_points == {"name": "normalRevisionPoints", "source": "points"}
    assert not any(
        data["name"] == "exactRevertPoints" for data in spec["data"]
    )
    assert not any(
        mark.get("name", "").startswith("exactRevertGlyph")
        for mark in spec["marks"]
    )


def test_revert_arc_rows_form_nested_edit_war_spiral():
    point_rows = [
        {"revision_id": "1", "x": 0.0, "y": 0.0},
        {"revision_id": "2", "x": 1.0, "y": 0.0},
        {"revision_id": "3", "x": 0.0, "y": 0.0},
        {"revision_id": "4", "x": 1.0, "y": 0.0},
    ]
    analysis = {
        "events": [
            {
                "revision_index": 1,
                "revision_id": "2",
                "reverted_revision_id": "1",
                "restored_revision_id": "0",
                "reverter": "Alice",
                "reverted_editor": "Bob",
                "pair": ["Alice", "Bob"],
                "revert_kind": "exact_sha1",
                "edit_war_session_id": "edit-war-1",
                "involves_probable_bot": True,
            },
            {
                "revision_index": 2,
                "revision_id": "3",
                "reverted_revision_id": "2",
                "reverter": "Bob",
                "reverted_editor": "Alice",
                "pair": ["Alice", "Bob"],
                "revert_kind": "partial_text",
                "edit_war_session_id": "edit-war-1",
            },
            {
                "revision_index": 3,
                "revision_id": "4",
                "reverted_revision_id": "3",
                "reverter": "Alice",
                "reverted_editor": "Bob",
                "pair": ["Alice", "Bob"],
                "revert_kind": "exact_sha1",
                "edit_war_session_id": "edit-war-1",
            },
        ],
        "edit_war_sessions": [
            {"session_id": "edit-war-1", "event_count": 3}
        ],
    }

    rows = build_revert_arc_rows(point_rows, analysis)

    assert len(rows) == 3
    assert [row["arc_rank"] for row in rows] == [0, 1, 2]
    assert [row["arc_level"] for row in rows] == [0, 0, 1]
    assert all(row["arc_count"] == 3 for row in rows)
    assert all(row["is_edit_war"] for row in rows)
    assert all(row["edit_war_label"] == "Edit War" for row in rows)
    assert all("is_bot_edit_war" not in row for row in rows)
    assert rows[0]["reverted_revision_id"] == "1"
    assert rows[0]["restored_revision_id"] == "0"
    assert (rows[0]["x"], rows[0]["y"]) == (0.0, 0.0)
    assert (rows[0]["x2"], rows[0]["y2"]) == (1.0, 0.0)
    assert (rows[1]["x"], rows[1]["y"]) == (1.0, 0.0)
    assert (rows[1]["x2"], rows[1]["y2"]) == (0.0, 0.0)


def test_exact_revert_position_snaps_to_restored_revision_only():
    rows = [
        {"revision_id": "1", "x": 0.0, "y": 0.0},
        {"revision_id": "2", "x": 1.0, "y": 0.0},
        {"revision_id": "3", "x": 0.03, "y": -0.02},
        {"revision_id": "4", "x": 0.45, "y": 0.15},
    ]
    analysis = {
        "events": [
            {
                "revision_id": "3",
                "restored_revision_id": "1",
                "revert_kind": "exact_sha1",
            },
            {
                "revision_id": "4",
                "restored_revision_id": None,
                "revert_kind": "partial_text",
            },
        ]
    }

    snap_exact_revert_positions(rows, analysis)

    assert (rows[2]["x"], rows[2]["y"]) == (0.0, 0.0)
    assert rows[2]["exact_revert_restored_revision_id"] == "1"
    assert rows[2]["projected_x_before_revert_snap"] == 0.03
    assert rows[2]["projected_y_before_revert_snap"] == -0.02
    assert (rows[3]["x"], rows[3]["y"]) == (0.45, 0.15)


def test_standalone_partial_counter_edit_has_dashed_partial_arc():
    point_rows = [
        {"revision_id": "1", "x": 0.0, "y": 0.0},
        {"revision_id": "2", "x": 1.0, "y": 0.0},
        {"revision_id": "3", "x": 0.1, "y": 0.0},
    ]
    analysis = {
        "events": [
            {
                "revision_index": 2,
                "revision_id": "3",
                "reverted_revision_id": "2",
                "restored_revision_id": None,
                "revert_kind": "partial_text",
                "reverter": "Bob",
                "reverted_editor": "Alice",
                "pair": ["Alice", "Bob"],
                "directly_reverts_previous_revision": True,
            }
        ]
    }

    rows = build_revert_arc_rows(point_rows, analysis)
    assert len(rows) == 1
    assert rows[0]["revert_kind"] == "partial_text"
    assert rows[0]["revert_kind_label"] == "Partial revert"
    assert rows[0]["is_edit_war"] is False
    assert rows[0]["return_anchor_revision_id"] == "3"
    assert (rows[0]["x"], rows[0]["x2"]) == (1.0, 0.1)


def test_piecewise_partial_revert_draws_every_consecutive_return_segment():
    point_rows = [
        {"revision_id": "1", "x": 0.0, "y": 0.0},
        {"revision_id": "2", "x": 1.0, "y": 0.0},
        {"revision_id": "3", "x": 0.7, "y": 0.0},
        {"revision_id": "4", "x": 0.3, "y": 0.0},
    ]
    analysis = {
        "events": [{
            "revision_index": 2,
            "revision_id": "3",
            "reverted_revision_id": "2",
            "restored_revision_id": "1",
            "revert_kind": "partial_trajectory",
            "trajectory_path_revision_ids": ["2", "3", "4"],
        }]
    }

    rows = build_revert_arc_rows(point_rows, analysis)

    assert len(rows) == 2
    assert [(row["x"], row["x2"]) for row in rows] == [
        (1.0, 0.7),
        (0.7, 0.3),
    ]
    assert [row["return_anchor_revision_id"] for row in rows] == ["3", "4"]


def test_edit_war_draws_outgoing_and_return_segments_of_compound_phase():
    point_rows = [
        {"revision_id": "1", "x": 0.0, "y": 0.0},
        {"revision_id": "2", "x": 0.5, "y": 0.0},
        {"revision_id": "3", "x": 1.0, "y": 0.0},
        {"revision_id": "4", "x": 0.7, "y": 0.0},
        {"revision_id": "5", "x": 0.3, "y": 0.0},
    ]
    analysis = {
        "events": [{
            "revision_index": 4,
            "revision_id": "5",
            "reverted_revision_id": "3",
            "revert_kind": "partial_trajectory",
            "trajectory_path_revision_ids": ["3", "4", "5"],
            "trajectory_full_path_revision_ids": ["1", "2", "3", "4", "5"],
            "edit_war_session_id": "edit-war-1",
        }],
        "edit_war_sessions": [{"session_id": "edit-war-1", "event_count": 2}],
    }

    rows = build_revert_arc_rows(point_rows, analysis)

    assert len(rows) == 4
    assert all(row["is_edit_war"] for row in rows)
    assert [(row["x"], row["x2"]) for row in rows] == [
        (0.0, 0.5),
        (0.5, 1.0),
        (1.0, 0.7),
        (0.7, 0.3),
    ]


def test_standalone_bot_partial_revert_uses_same_partial_arc_category():
    point_rows = [
        {"revision_id": "1", "x": 0.0, "y": 0.0},
        {"revision_id": "2", "x": 1.0, "y": 0.0},
        {"revision_id": "3", "x": 0.1, "y": 0.0},
    ]
    analysis = {
        "events": [
            {
                "revision_index": 2,
                "revision_id": "3",
                "reverted_revision_id": "2",
                "restored_revision_id": None,
                "revert_kind": "partial_text",
                "involves_probable_bot": True,
            }
        ]
    }

    rows = build_revert_arc_rows(point_rows, analysis)
    assert len(rows) == 1
    assert rows[0]["revert_kind_label"] == "Partial revert"
    assert rows[0]["is_edit_war"] is False
    assert "is_bot_edit_war" not in rows[0]


def test_edit_war_exact_revert_uses_only_the_dashed_arc():
    projection = sample_projection()
    hierarchy = build_hierarchy(projection, sample_keywords())
    analysis = {
        "events": [
            {
                "revision_index": 2,
                "revision_id": "102",
                "reverted_revision_id": "101",
                "restored_revision_id": "100",
                "reverter": "Alice",
                "reverted_editor": "Bob",
                "pair": ["Alice", "Bob"],
                "revert_kind": "exact_sha1",
                "edit_war_session_id": "edit-war-1",
            }
        ],
        "edit_war_sessions": [
            {"session_id": "edit-war-1", "event_count": 3}
        ],
    }

    spec = build_vega_spec("Example", projection, hierarchy, analysis)
    arc_rows = next(
        data for data in spec["data"] if data["name"] == "revertArcs"
    )["values"]

    assert len(arc_rows) == 1
    assert arc_rows[0]["is_edit_war"] is True
    assert not any(
        mark.get("name", "").startswith("exactRevertGlyph")
        for mark in spec["marks"]
    )


def test_edit_war_suppresses_only_revert_arcs_on_the_same_geometry():
    projection = sample_projection()
    hierarchy = build_hierarchy(projection, sample_keywords())
    analysis = {
        "events": [
            {
                "revision_index": 2,
                "revision_id": "102",
                "reverted_revision_id": "101",
                "restored_revision_id": "100",
                "revert_kind": "exact_sha1",
            },
            {
                "revision_index": 4,
                "revision_id": "104",
                "reverted_revision_id": "103",
                "restored_revision_id": "100",
                "revert_kind": "exact_sha1",
                "edit_war_session_id": "edit-war-1",
            },
            {
                "revision_index": 3,
                "revision_id": "100",
                "reverted_revision_id": "103",
                "restored_revision_id": None,
                "revert_kind": "partial_text",
            },
        ],
        "edit_war_sessions": [
            {"session_id": "edit-war-1", "event_count": 3}
        ],
    }

    spec = build_vega_spec("Example", projection, hierarchy, analysis)
    arc_rows = next(
        data for data in spec["data"] if data["name"] == "revertArcs"
    )["values"]

    assert len(arc_rows) == 2
    assert [row["revision_id"] for row in arc_rows] == ["102", "104"]
    assert arc_rows[0]["is_edit_war"] is False
    assert arc_rows[1]["is_edit_war"] is True


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
    revert_analysis = {
        "events": [
            {
                "revision_index": 2,
                "revision_id": "102",
                "reverted_revision_id": "101",
                "restored_revision_id": "100",
                "reverter": "Alice",
                "reverted_editor": "Bob",
                "pair": ["Alice", "Bob"],
                "revert_kind": "exact_sha1",
            },
            {
                "revision_index": 3,
                "revision_id": "103",
                "reverted_revision_id": "102",
                "restored_revision_id": None,
                "reverter": "Bob",
                "reverted_editor": "Alice",
                "pair": ["Alice", "Bob"],
                "revert_kind": "partial_text",
                "edit_war_session_id": "edit-war-1",
                "involves_probable_bot": True,
            },
        ],
        "edit_war_sessions": [
            {"session_id": "edit-war-1", "event_count": 2}
        ],
    }
    spec = build_vega_spec(
        "Example", projection, hierarchy, revert_analysis
    )
    assert any(signal["name"] == "zoomLevel" for signal in spec["signals"])
    signals = {signal["name"]: signal for signal in spec["signals"]}
    assert signals["minViewScale"]["value"] == 0.01
    assert signals["maxViewScale"]["value"] == 50.0
    assert signals["viewScale"]["update"] == "span(xext)/span(xdom)"
    assert signals["rootOnlyOverview"]["update"] == (
        "viewScale <= minViewScale * 1.001"
    )
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
    assert normal_point_data == {
        "name": "normalRevisionPoints", "source": "points"
    }
    assert not any(
        data["name"] == "exactRevertPoints" for data in spec["data"]
    )
    assert not any(
        mark.get("name", "").startswith("exactRevertGlyph")
        for mark in spec["marks"]
    )
    revert_arc_data = next(
        data for data in spec["data"] if data["name"] == "revertArcs"
    )
    assert len(revert_arc_data["values"]) == 2
    assert revert_arc_data["values"][0]["is_edit_war"] is False
    assert revert_arc_data["values"][1]["is_edit_war"] is True
    assert revert_arc_data["values"][1]["edit_war_label"] == "Edit War"
    assert all(
        "is_bot_edit_war" not in row for row in revert_arc_data["values"]
    )
    assert revert_arc_data["values"][0]["revision_id"] == "102"
    assert revert_arc_data["values"][0]["reverted_revision_id"] == "101"
    assert revert_arc_data["values"][0]["return_anchor_revision_id"] == "100"
    assert revert_arc_data["values"][0]["target_overlap_count"] == 2
    revert_geometry = next(
        data for data in spec["data"] if data["name"] == "revertArcGeometry"
    )
    geometry_fields = {
        transform.get("as"): transform.get("expr")
        for transform in revert_geometry["transform"]
        if transform.get("as")
    }
    assert geometry_fields["screenLength"] == (
        "sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2))"
    )
    assert "datum.arc_level" in geometry_fields["arcHeight"]
    assert "datum.is_edit_war" in geometry_fields["arcHeight"]
    assert "datum.revert_kind==='exact_sha1'" in geometry_fields["arcHeight"]
    assert "datum.screenLength*0.12" in geometry_fields["arcHeight"]
    assert "min(95,max(10" in geometry_fields["arcHeight"]
    assert " Q" in geometry_fields["arcPath"]
    assert sum(edge["is_edit_war"] for edge in rendered_edges) == 2
    revert_arc_mark = next(
        mark for mark in spec["marks"] if mark.get("name") == "revertArcPaths"
    )
    assert revert_arc_mark["type"] == "path"
    assert revert_arc_mark["from"] == {"data": "revertArcGeometry"}
    assert revert_arc_mark["encode"]["update"]["path"] == {
        "field": "arcPath"
    }
    assert revert_arc_mark["encode"]["update"]["strokeDash"] == {
        "value": [8, 5]
    }
    assert EDIT_WAR_COLOR in revert_arc_mark["encode"]["update"]["stroke"]["signal"]
    assert EXACT_REVERT_COLOR in revert_arc_mark["encode"]["update"]["stroke"]["signal"]
    assert revert_arc_mark["encode"]["update"]["strokeWidth"] == {
        "signal": (
            f"datum.is_edit_war ? {EDIT_WAR_ARC_STROKE_WIDTH} : "
            f"{REVERT_ARC_STROKE_WIDTH}"
        )
    }
    assert revert_arc_mark["encode"]["update"]["strokeOpacity"] == {
        "signal": (
            f"datum.is_edit_war ? {EDIT_WAR_ARC_STROKE_OPACITY} : "
            f"{REVERT_ARC_STROKE_OPACITY}"
        )
    }
    revert_arrow_mark = next(
        mark for mark in spec["marks"]
        if mark.get("name") == "revertArcDirectionArrows"
    )
    assert revert_arrow_mark["from"] == {"data": "revertArcGeometry"}
    assert revert_arrow_mark["encode"]["update"]["shape"] == {
        "value": "triangle-right"
    }
    assert revert_arrow_mark["encode"]["update"]["fillOpacity"] == {
        "value": REVERT_ARROW_OPACITY
    }
    assert revert_arrow_mark["encode"]["update"]["strokeOpacity"] == {
        "value": REVERT_ARROW_OPACITY
    }
    assert revert_arrow_mark["encode"]["update"]["angle"] == {
        "field": "arrowAngle"
    }
    mark_names = [mark.get("name") for mark in spec["marks"]]
    assert mark_names.index("trajectory") < mark_names.index("revertArcPaths")
    assert "editWarHalo" not in mark_names
    assert "exactRevertTrajectory" not in mark_names
    assert "tooltip" in revert_arc_mark["encode"]["enter"]
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
            f"scale('clusterColor', datum.cluster_id) : '{TRAJECTORY_COLOR}'"
        )
    }
    assert len(next(data for data in spec["data"] if data["name"] == "edges")["values"]) == 5
    colour_scale = next(scale for scale in spec["scales"] if scale["name"] == "clusterColor")
    assert colour_scale["domain"] == [-1, 0, 1]
    assert colour_scale["range"] == [NOISE_COLOR, cluster_colour(0), cluster_colour(1)]
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
        "datum.labelled_section_point"
    )
    section_point_mark = next(
        mark for mark in spec["marks"] if mark.get("name") == "sectionPointHighlights"
    )
    assert section_point_mark["interactive"] is False
    assert point_mark["encode"]["update"]["size"] == {
        "value": REVISION_POINT_SIZE
    }
    assert point_mark["encode"]["update"]["strokeWidth"] == {"value": 1.2}
    assert section_point_mark["encode"]["update"]["size"] == {
        "value": SECTION_POINT_HIGHLIGHT_SIZE
    }
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
    assert "rootOnlyOverview ? datum.level==='root'" in (
        visible["transform"][0]["expr"]
    )
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
    parent_font_size = text_mark["encode"]["update"]["fontSize"]["signal"]
    assert "datum.level==='root' ? 28" in parent_font_size
    assert "datum.level==='cluster' ? 16" in parent_font_size
    assert "datum.level==='section' ? 12" in parent_font_size
    hull_data = next(data for data in spec["data"] if data["name"] == "clusterHulls")
    assert {row["cluster_id"] for row in hull_data["values"]} <= {0}
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
    assert "macroHulls" not in data_names
    assert "macroRegions" not in mark_names
    assert "transform" not in text_mark
    assert text_mark["from"]["data"] == "visibleParentLabels"
    assert text_mark["key"] == "node_id"
    mark_names = [mark.get("name") for mark in spec["marks"]]
    assert "macroLabels" not in mark_names
    assert "rootLabel" not in mark_names
    assert "semanticLabelHalos" not in mark_names
    edge_text = next(mark for mark in spec["marks"] if mark.get("name") == "edgeLabels")
    assert edge_text["encode"]["update"]["text"] == {
        "signal": (
            "indexof(detailedEdgeLabelIds, datum.node_id)>=0 ? "
            "datum.display_label : datum.label"
        )
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
    assert edge_candidates["transform"][1]["expr"] == "!rootOnlyOverview"
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
    assert transforms_by_field["screenLength"]["expr"] == (
        "sqrt(pow(datum.screenDx,2)+pow(datum.screenDy,2))"
    )
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
    assert signal_values["detailedEdgeLabelIds"] == []
    assert "coveredParentLabelIds" not in signal_values
    assert edge_text["encode"]["update"]["text"] == {
        "signal": (
            "indexof(detailedEdgeLabelIds, datum.node_id)>=0 ? "
            "datum.display_label : datum.label"
        )
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
    short_measure = next(
        mark for mark in spec["marks"]
        if mark.get("name") == "edgeShortMeasureLabels"
    )
    detailed_measure = next(
        mark for mark in spec["marks"]
        if mark.get("name") == "edgeDetailedMeasureLabels"
    )
    assert short_measure["encode"]["update"]["text"] == {"field": "label"}
    assert detailed_measure["encode"]["update"]["text"] == {
        "field": "display_label"
    }
    assert short_measure["encode"]["update"]["opacity"] == {"value": 0}
    assert detailed_measure["encode"]["update"]["opacity"] == {"value": 0}
    assert short_measure["from"]["data"] == "visibleEdgeLabels"
    assert detailed_measure["from"]["data"] == "visibleEdgeLabels"
    assert "- 28*datum.screenDy" in (
        detailed_measure["encode"]["update"]["x"]["signal"]
    )
    assert "? 28 : 9" in edge_text["encode"]["update"]["x"]["signal"]
    detail_leader = next(
        mark for mark in spec["marks"]
        if mark.get("name") == "edgeDetailLeaders"
    )
    assert detail_leader["from"]["data"] == "visibleEdgeLabels"
    assert detail_leader["encode"]["update"]["x"] == {
        "field": "labelScreenX"
    }
    assert "datum.screenLength" in (
        detail_leader["encode"]["update"]["x2"]["signal"]
    )
    assert "detailedEdgeLabelIds" in (
        detail_leader["encode"]["update"]["strokeOpacity"]["signal"]
    )
    assert "blockedEdgeLabelIds" in (
        detail_leader["encode"]["update"]["strokeOpacity"]["signal"]
    )
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
    assert revision_points["encode"]["update"]["size"] == {
        "value": REVISION_POINT_SIZE
    }
    assert section_highlights["encode"]["update"]["size"] == {
        "value": SECTION_POINT_HIGHLIGHT_SIZE
    }
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
    assert '"name":"editWarHalo"' not in page
    assert '"name":"editWarDirectionArrows"' not in page
    assert '"name":"revertArcs"' in page
    assert '"name":"revertArcGeometry"' in page
    assert '"name":"revertArcPaths"' in page
    assert '"name":"revertArcDirectionArrows"' in page
    assert '"name":"directionArrows"' in page
    assert '"name":"visibleDirectionArrows"' in page
    assert '"name":"exactRevertPoints"' not in page
    assert '"name":"exactRevertGlyphMasks"' not in page
    assert '"name":"exactRevertGlyphArcs"' not in page
    assert '"name":"exactRevertGlyphHeads"' not in page
    assert '"name":"exactRevertTrajectory"' not in page
    assert "!datum.has_exact_revert" in page
    assert '"name":"editWarTrajectory"' not in page
    assert '"strokeDash":{"value":[8,5]}' in page
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
    assert '<span>Repeated projected path</span>' in page
    assert '<span>Exact revert revision</span>' not in page
    assert '<span>Exact revert return</span>' in page
    assert (
        '<span>Partial revert (selected text restored; unaffected text remains)</span>'
        in page
    )
    assert '<span>Edit War</span>' in page
    assert '<span>Edit War Bots</span>' not in page
    assert ".legend-spiral.bot" not in page
    assert "Temporal-section point" in page
    assert "<h2>Legend</h2>" in page
    cluster_labels = [
        node["label"] for node in hierarchy["clusters"] if node.get("label")
    ]
    assert cluster_labels
    assert all(f"<span>{label}</span>" in page for label in cluster_labels)
    assert "<span>Cluster 1</span>" not in page
    assert "Noise" in page
    assert "renderedLabelIds('macroLabels')" not in page
    assert "renderedLabelIds('semanticLabels')" in page
    assert "renderedLabelIds('edgeLabels')" in page
    assert "stableParentLabelIds" in page
    assert "stableEdgeLabelIds" in page
    assert "blockedParentLabelIds" in page
    assert "blockedEdgeLabelIds" in page
    assert "detailedEdgeLabelIds" in page
    assert "coveredParentLabelIds" not in page
    assert "renderedLabelItems" in page
    assert "normalizedLabelKey" in page
    assert "boxesOverlap" in page
    assert "const labelCollisionPadding=4" in page
    assert "mode==='collision'" in page
    assert "outsideViewport" in page
    assert "class LabelSpatialHash" in page
    assert "occupied.overlaps(candidate)" in page
    assert "edgeShortMeasureLabels" in page
    assert "edgeDetailedMeasureLabels" in page
    assert "detailedCandidate.hasDetailed" in page
    assert "shortCandidate.edgeScreenLength>=72" in page
    assert "edgeScreenLength:Number(datum.screenLength)||0" in page
    assert "edgeDetailLeaders" in page
    assert "acceptedParentLabelKeys" in page
    assert "acceptedShortEdgeLabelKeys" in page
    assert "shortTextIsRedundant" in page
    assert "if(detailedFits)" in page
    assert "occupied.overlaps(detailedCandidate)" in page
    assert "occupied.overlaps(shortCandidate)" in page
    assert "zoomLevel>=2 ? datum.display_label" not in page
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
    assert "vis.dataset.detailedEdgeLabels" in page
    assert "vis.dataset.shortEdgeLabels" in page
    assert "view.addSignalListener('zooming'" in page
    assert "view.addSignalListener('zoomLevel'" in page
    assert "view.addSignalListener('down'" in page
    assert "Zoom 1.0×" not in page
    assert "Scrollen zum Zoomen" not in page


def test_legend_colours_match_the_visual_encodings():
    projection = sample_projection()
    hierarchy = build_hierarchy(projection, sample_keywords())
    spec = build_vega_spec("Example", projection, hierarchy)
    page = render_html("Example", spec)

    trajectory = next(
        mark for mark in spec["marks"] if mark.get("name") == "trajectory"
    )
    revert_paths = next(
        mark for mark in spec["marks"] if mark.get("name") == "revertArcPaths"
    )
    revert_signal = revert_paths["encode"]["update"]["stroke"]["signal"]

    assert trajectory["encode"]["update"]["stroke"] == {
        "value": TRAJECTORY_COLOR
    }
    assert all(
        colour in revert_signal
        for colour in (
            EXACT_REVERT_COLOR,
            PARTIAL_REVERT_COLOR,
            EDIT_WAR_COLOR,
        )
    )
    assert f"border-top:2.25px solid {TRAJECTORY_COLOR}" in page
    assert f"border:{REVERT_ARC_STROKE_WIDTH}px dashed {EXACT_REVERT_COLOR}" in page
    assert f"border:{REVERT_ARC_STROKE_WIDTH}px dashed {PARTIAL_REVERT_COLOR}" in page
    assert f"border:{EDIT_WAR_ARC_STROKE_WIDTH}px dashed {EDIT_WAR_COLOR}" in page
    assert f"opacity:{REVERT_ARC_STROKE_OPACITY}" in page
    assert f"opacity:{EDIT_WAR_ARC_STROKE_OPACITY}" in page
    assert (
        ".legend-spiral{width:30px;height:10px;box-sizing:border-box;"
        f"border:{EDIT_WAR_ARC_STROKE_WIDTH}px dashed {EDIT_WAR_COLOR};border-bottom:0;"
        "border-radius:50% 50% 0 0;flex:none;"
        f"opacity:{EDIT_WAR_ARC_STROKE_OPACITY}}}"
    ) in page
    assert f"--legend-color:{NOISE_COLOR}" in page
    assert "Revision point (cluster colour)" in page
    assert "Temporal-section point (cluster colour)" in page
    for cluster_id in {-1, 0, 1}:
        assert f'--legend-color:{cluster_colour(cluster_id)}' in page


def test_cluster_legend_uses_semantic_labels_and_falls_back_for_unlabelled_cluster():
    projection = sample_projection()
    hierarchy = build_hierarchy(projection, sample_keywords())
    spec = build_vega_spec("Example", projection, hierarchy)
    labels = next(data for data in spec["data"] if data["name"] == "labels")
    labels["values"] = [
        row for row in labels["values"]
        if not (row.get("level") == "cluster" and row.get("cluster_id") == 1)
    ]
    legend = next(data for data in spec["data"] if data["name"] == "clusterLegend")
    legend["values"] = [
        row for row in legend["values"] if row.get("cluster_id") != 1
    ]

    page = render_html("Example", spec)
    first_cluster = next(
        node for node in hierarchy["clusters"] if node["cluster_id"] == 0
    )

    assert f"<span>{first_cluster['label']}</span>" in page
    assert "<span>Cluster 2</span>" in page
    assert "<span>Noise</span>" in page


def test_cluster_colours_match_original_timecurve_palette():
    assert cluster_colour(-1) == NOISE_COLOR
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
