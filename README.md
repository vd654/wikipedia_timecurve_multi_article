# Wikipedia Timecurve

Wikipedia Timecurve turns the revision history of an English Wikipedia article
into an interactive two-dimensional semantic map. Each point is one complete
article revision, consecutive revisions form a chronological trajectory, and
labels describe changes at several semantic levels. Zooming reveals more detail
only where enough screen space is available.

The project separates two use cases:

1. Reproduce the reviewed `Palestine_final` visualization without an API key.
2. Run the complete pipeline for Palestine or another Wikipedia article.

## Quick overview

| Component | Implementation |
| --- | --- |
| Revision source | English Wikipedia MediaWiki API, stored locally as XML |
| Semantic representation | `sentence-transformers/all-MiniLM-L6-v2` |
| Projection | Classical metric MDS over cosine distances |
| Clustering | HDBSCAN on the two-dimensional projection |
| Change labels | Complete before/after revision comparison with OpenAI |
| Change-type colors | Deterministic Git word diff, without an LLM |
| Reverts | Exact SHA1 returns only |
| Renderer | Vega plus the custom semantic-zoom JavaScript runtime |

```text
Wikipedia XML
→ cleaned complete revisions
→ revision embeddings
→ cosine-distance matrix
→ two-dimensional MDS projection
→ HDBSCAN clusters
→ chronological Catmull–Rom trajectory
→ edge labels and semantic hierarchy
→ interactive Vega HTML
```

## Reproduce `Palestine_final`

The reviewed final visualization and every input required to rebuild it are
stored in the repository. No Wikipedia XML file, embedding model download,
OpenAI API key, or previous generated HTML file is required.

The complete current snapshot is on the
[`current-visualization-2026-08-20`](https://github.com/vd654/wikipedia_timecurve_multi_article/tree/current-visualization-2026-08-20)
branch:

```bash
git clone --branch current-visualization-2026-08-20 --single-branch \
  https://github.com/vd654/wikipedia_timecurve_multi_article.git
cd wikipedia_timecurve_multi_article

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python src/build_final.py
```

The result is written to:

```text
output/Palestine_final.html
```

The expected SHA-256 checksum is:

```text
cbe62b112554bde8e0ad4703c97e044aa03a567417ce2718c81ef12a02e4fffb
```

`build_final.py` reads four reviewed inputs from `final_build.json`:

1. Revision projection
2. Semantic label hierarchy
3. Complete edge-label dataset
4. Exact-revert analysis

This compact final-build path is specifically configured for the reviewed
Palestine visualization. A new article is produced with the complete pipeline
described below.

## Final Palestine dataset

- Period: December 2, 2012 to July 23, 2014
- 620 complete revisions and 619 trajectory edges
- 619 non-empty labels, one for every edge
- 44 HDBSCAN clusters and 140 noise points
- 71 exact SHA1 reverts
- Edge-label model: `gpt-5.6-terra`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Change types: 87 added-only, 63 removed-only, 288 mixed, 181
  no-cleaned-text-change revisions, plus the initial start point

The ready-to-open file is
[`output/Palestine_final.html`](output/Palestine_final.html).

## Requirements

- Python 3.10 or newer
- Git
- Internet access for downloading Wikipedia revisions
- Internet access on the first projection run to download the embedding model
- An OpenAI API key only when generating new edge or hierarchy labels

Install all project and local-test dependencies from the single requirements
file:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`pytest` is included in this file only for the local test suite; the production
pipeline does not import it. The project does not use Altair, pandas, or
tiktoken directly.

For LLM-backed labeling, store the key in the ignored `.env` file:

```text
OPENAI_API_KEY=sk-...
```

Never commit `.env` or API credentials.

## Run the complete pipeline

The following example recreates the Palestine analysis period. Replace the
article name and dates to process another English Wikipedia article. Article
names containing spaces should be quoted.

### 1. Download the revision history

```bash
python src/wiki_api.py \
  --article Palestine \
  --start 2012-12-02 \
  --end 2014-07-23
```

The resumable downloader writes:

```text
data/Palestine/Palestine.xml
```

If a request is interrupted, run the same command again. Progress is recovered
from the temporary `.partial` and `.state.json` files. Use `--restart` only to
discard that incomplete download and start it again. Raw XML and resume files
remain local and are not committed.

### 2. Generate the semantic projection

```bash
python src/main_pipeline.py \
  --article Palestine \
  --start 2012-12-02 \
  --end 2014-07-23 \
  --sample-strategy all \
  --output-name Palestine_small
```

The command prints the exact generated paths. Output names are numbered to
avoid overwriting an earlier run. The important files are:

```text
*_projection.json
*_revision_embeddings.npy
*_cosine_distance_matrix.npy
*.html
```

The HTML produced at this stage is a label-free map using the same renderer as
the final visualization. This step requires no OpenAI API key. On the first
run, Sentence Transformers downloads the embedding model.

### 3. Generate one label per revision edge

Use the projection path printed by step 2:

```bash
python src/edge_labels.py \
  --article Palestine \
  --visualization \
  output/Palestine_small/Palestine_small_bert_timecurve_001_projection.json
```

The example assumes a fresh run numbered `001`. If the command in step 2
prints another number, use that exact projection path here and in step 4.

`edge_labels.py` compares the complete cleaned text before and after each
trajectory edge. It also supplies the author, timestamp, and available edit
comments. The model returns a short two-to-three-word label and a detailed
description. No precomputed text diff is sent to the model.

This step requires `OPENAI_API_KEY`. Every successful response is written
immediately to a resumable `*_edge_label_cache_*.json` file. Run the same
command again after an interruption; only missing edges are requested. Use
`--refresh-cache` to deliberately regenerate cached responses. Generated cache
files are excluded from Git.

### 4. Build the hierarchy and interactive visualization

Pass the projection from step 2 and the edge-label JSON from step 3:

```bash
python src/semantic_zoom.py \
  --projection \
  output/Palestine_small/Palestine_small_bert_timecurve_001_projection.json \
  --diff-labels \
  output/Palestine_small/Palestine_small_bert_timecurve_001_full_revision_edge_labels_gpt-5.6-terra.json \
  --article Palestine \
  --xml data/Palestine/Palestine.xml
```

This generates a hierarchy JSON and the semantic-zoom HTML next to the
projection. By default, OpenAI supplies the macro-topic, topic-cluster, and
temporal-section labels. Existing compatible hierarchy labels are reused from
the hierarchy output file.

To use deterministic fallback hierarchy names without making hierarchy-label
API calls, add:

```text
--no-llm
```

`--no-llm` does not generate the edge-label JSON; that file must already exist.
The initial projection HTML from step 2 remains the completely label-free,
no-OpenAI alternative.

### API and network requirements by command

| Command | Wikipedia/network | Embedding download | OpenAI API |
| --- | ---: | ---: | ---: |
| `wiki_api.py` | Yes | No | No |
| `main_pipeline.py` | No after XML exists | First run only | No |
| `edge_labels.py` | No after XML exists | No | Yes |
| `semantic_zoom.py` | No after inputs exist | No | Yes by default; no with `--no-llm` |
| `build_final.py` | No | No | No |

## How the visualization works

### Semantic geometry

Each point represents one complete sampled revision. Long revisions are split
into temporary chunks of at most 256 tokens. Their chunk embeddings are
averaged and L2-normalized into one vector per revision. Semantic distance is:

```text
1 - dot(L2-normalized revision vector A,
        L2-normalized revision vector B)
```

Classical metric MDS projects the distance matrix into two dimensions. HDBSCAN
clusters the visible MDS coordinates. The chronological path is rendered as an
open centripetal Catmull–Rom spline; geometry never changes the chronological
revision order.

### Label hierarchy and semantic zoom

```text
Article
└── Macro topic
    └── Topic cluster
        └── Temporal section
            └── Revision edge
```

Labels are evaluated in screen coordinates rather than switched through rigid
zoom tiers. Important broad labels are preferred in the overview, while short
edge labels and detailed descriptions gain priority as the viewer zooms in.
Lower levels may still appear earlier when free space is available.

Every candidate receives a measured screen-space bounding box. It must remain
inside the plot, avoid already placed labels, and overlap data belonging to the
topic or edge it describes. Edge candidates sample several positions along
both sides of their Bézier segment and prefer the position with the lowest edge
and point density. Hysteresis, cached positions, and short opacity transitions
reduce flickering during pan and zoom.

Font sizes communicate hierarchy:

- Article: 28 px
- Macro topic: 24 px
- Topic cluster: 18 px
- Temporal section: 13 px
- Edge label: 10 px
- Start and End: 14 px

Detailed multiline descriptions become eligible at zoom level 4.25 when their
visible spline segment and surrounding free space are large enough. The full
description remains available in the revision tooltip when it cannot be placed
on the map.

### Visual encodings

Revision-point fill describes the cleaned-text change relative to the direct
predecessor. This classification is produced deterministically by the Git word
diff and does not call an LLM:

- Gold `#F2C14E`: added only
- Violet `#8F5BD7`: removed only
- Orange `#D9822B`: added and removed
- Gray `#B8B8B8`: no cleaned-text change
- Dark gray `#343A40`: first revision (`Start`)

Neutral green-gray lines show the standard trajectory and its direction.
Macro topics and clusters have no separate color encoding. Temporal-section
point rings are not rendered.

### Exact reverts

A revert is shown only when a revision restores the exact SHA1 hash of an
earlier article state. A dotted signal-red arc starts at the revision whose
change is undone and points to the later reverting revision. Detection is
independent of the two-dimensional projection and includes self-reverts.

When repeated returns create the same visual connection, only the
chronologically latest arc is drawn. Partial reverts and edit-war inference are
not part of the current project.

## Interaction

- Mouse wheel or trackpad: zoom around the pointer
- Drag: pan the visualization
- Double-click: restore the full view
- Hover a revision point: show revision ID, timestamp, author, and change type

The generated HTML embeds its data and Vega specification, but currently loads
Vega and Vega Tooltip from a CDN. Opening the interactive HTML therefore
requires internet access. The rebuild itself does not.

## Files and reproducibility

```text
src/
├── wiki_api.py              resumable Wikipedia revision download
├── main_pipeline.py        embeddings, MDS, HDBSCAN, and base map
├── edge_labels.py          complete-revision edge labeling
├── semantic_zoom.py         hierarchy and Vega specification
├── semantic_zoom_runtime.py browser interaction and label placement
├── trajectory_geometry.py   spline, hull, and revert geometry
├── revert_detection.py      exact SHA1 revert detection
├── git_word_diff.py         deterministic word-change analysis
├── revision_io.py           shared XML revision loading
├── text_cleaning.py         wikitext cleaning
├── llm_provider.py          OpenAI client and model configuration
├── build_final.py           API-free reviewed Palestine rebuild
├── config.py                paths and shared arguments
└── output_utils.py          output naming and atomic writes

final_build.json                           reviewed Palestine input manifest
output/Palestine_final_data/               four reviewed build inputs
output/Palestine_final.html                presentation-ready visualization
requirements.txt                           runtime and local-test dependencies
```

Only the four JSON inputs referenced by `final_build.json` are required to
reconstruct `Palestine_final`. The XML dump, embeddings, distance matrix,
resumable API caches, and earlier HTML outputs are not required for that build.

## Local tests

The local test directory is deliberately excluded from the public repository.
When it is available, run:

```bash
PYTHONPATH=src python -m pytest -q
```

The suite covers revision ordering, complete-text selection, word-diff
reconstruction, exact reverts, final-build reproducibility, geometry, label
priorities, and collision handling.
