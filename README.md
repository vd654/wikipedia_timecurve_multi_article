# Wikipedia Timecurve

Wikipedia Timecurve visualizes the evolution of a Wikipedia article as a
two-dimensional semantic landscape. Each point represents a complete article
revision, and each line connects two chronologically consecutive revisions.
Labels are placed with continuous cartographic decluttering rather than fixed
zoom tiers: as zooming or panning creates enough screen space, the next
eligible label appears according to its semantic priority.

## Final Palestine visualization

The presentation-ready visualization is available at
[`output/Palestine_final.html`](output/Palestine_final.html).

The final dataset contains:

- Article: `Palestine`
- Time range: December 2, 2012 to July 23, 2014
- 620 complete revisions and therefore 619 trajectory edges
- 619 labels for 619 edges, generated from complete revision comparisons
- Label model: `gpt-5.6-terra`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Projection: classical metric multidimensional scaling (MDS)
- Clustering: HDBSCAN on the visible two-dimensional projection
- 44 clusters and 140 noise points
- 71 exact SHA1 reverts
- Deterministic point classification: 87 added-only, 63 removed-only,
  288 mixed, and 181 revisions with no cleaned-text change; the first point is
  classified as the start

## Method

```text
Wikipedia XML
→ complete cleaned revisions
→ encoding chunks of at most 256 tokens
→ one averaged and L2-normalized vector per revision
→ cosine-distance matrix
→ classical MDS in two dimensions
→ HDBSCAN on the MDS coordinates
→ chronological revision trajectory
→ Terra full-text comparison for every edge
→ continuous cartographic labeling based on a five-level hierarchy
```

The distance between two revisions is defined as:

```text
1 - dot(L2-normalized revision vector A,
        L2-normalized revision vector B)
```

## Edge labels

For each of the 619 edges, `gpt-5.6-terra` receives the complete cleaned
revision text before and after the change, together with the author, timestamp,
and—when available in the XML—the edit comment for both revisions. No
precomputed diff is supplied; the model compares the complete texts directly.
It produces one indivisible two-to-three-word label that already names the
subject and action or outcome of the specific change, plus a detailed
description of the detected change. As the viewer zooms in, an edge therefore
becomes more specific instead of merely repeating its parent topic. Metadata
provides supporting context only; the actual text difference remains the
decisive evidence. Word diffs and sentence windows are neither generated nor
sent to the model.

Model responses are cached after every edge. An interrupted run can therefore
resume without repeating completed API calls. Zooming the finished
visualization never triggers an API call.

Byte-identical cleaned revision texts are deterministically labeled
`No Content Change` and are not sent to the model. Conversely, a neutral model
answer is rejected whenever the two cleaned full texts differ. This prevents a
short label and its detailed description from contradictorily describing an
actual content change as unchanged.

Before rendering, the higher hierarchy levels are generated through the same
OpenAI API from the stored edge labels and keywords within their respective
areas. Terra creates exactly one umbrella term for each temporal section,
topic cluster, and macro topic. These responses are cached in the hierarchy
JSON.

## Cartographic hierarchy

```text
Article
└── Macro topic
    └── Topic cluster
        └── Temporal section
            └── Revision edge
```

The hierarchy provides label candidates and cartographic priorities. Article,
macro-topic, cluster, section, and short edge labels are evaluated together in
screen coordinates at every normal scale. Their order is determined by the
semantic depth preferred at the current zoom level, their content importance,
and a stability bonus for labels that are already visible. There is no global
label-count or screen-area budget: lower-priority candidates are still tested
after every higher-priority candidate. A label is displayed when its measured
screen-space bounding box lies fully inside the viewport, does not collide
with an already placed label, and covers at least one revision point belonging
to the data it describes.

Macro topics have no dedicated areas, outlines, or color encoding. Only their
direct labels remain visible. Associated cluster hulls are drawn in the
neutral trajectory color.

The reviewed position of a parent label remains its semantic center. Macro,
cluster, and section labels are initially centered there and may then move to
as many as eight centrally located revision points within their own data area;
the article title remains fixed at the center. For every edge label, up to ten
candidate positions are evaluated at five locations on both sides of the
currently visible Bézier segment. The best valid position is cached and
preferred during small viewport changes. An alternative hierarchy-label
position is valid only when its text box still overlaps revision points from
the macro topic, cluster, or temporal section it describes. Empty space
outside that data area is not a valid fallback location.

All labels retain a constant screen-space font size. Size communicates the
hierarchy level:

- Article: 28 px
- Macro topic: 24 px
- Topic cluster: 18 px
- Temporal section: 13 px
- Edge label: 10 px
- Start and End: 14 px

Zoom does not impose a hard visibility tier. Instead, it defines a preferred
semantic depth. In the overview, the article, macro topics, and topic clusters
are evaluated first. As the viewer zooms in, priority moves continuously
toward topic clusters, temporal sections, short edge labels, and finally
detailed edge descriptions. Lower-level labels remain candidates at every
normal zoom level and may appear whenever their actual screen-space box fits
into a free area. The next finer level receives only a moderate forward
penalty, allowing it to use free space shortly before reaching its ideal
semantic depth. Start and End remain high-priority special labels.

Every visible child level should contribute at least one new content term or a
specific direction of change. Short edge labels therefore name the subject
plus the action or result; their detailed variants add the author, time, and
full scope. An audit function can flag semantically unspecific child labels for
editorial review, but such findings do not currently abort the standard map
build.

When two text boxes touch, the lower-priority label is hidden. In addition, an
8 px density grid indexes visible Catmull–Rom edges and revision points. For
hierarchy labels, this density is used only to choose among positions that
still cover the label's own data area. Edge labels choose the candidate with
the lowest edge and node density. Separate enter and exit thresholds provide
hysteresis: an existing label may remain visible after a small deterioration,
while a new label must satisfy a stricter free-space threshold. Placement is
recomputed only after a zoom or pan gesture ends and is blended with a 150 ms
opacity transition. An 8 px inner margin and SVG clipping keep text inside the
plot area and prevent overlap with the title bar or legend.

Identical article, macro, cluster, and section labels are deduplicated before
rendering. Spatially separate revision edges retain their own labels even when
their text is identical. Similar to symbol spacing on a road map, identical
edge labels are suppressed only when their selected screen positions are less
than 120 px apart. Completely overlapping edges receive one label for the same
geometry. The preferred label is selected by the availability of a detailed
description, followed by the size of the word change, information content,
label confidence, and semantic jump. All other edges remain present as
geometry and revision events.

Every visible label uses a fully opaque, uniformly dark text fill. Hierarchy
labels use a 4 px white halo, while edge labels use a 2 px white halo. Hidden
collision candidates are removed with a short opacity transition.

Every distinct revision edge remains a label candidate. There is no global
count or area cap. A detailed multiline description is attempted only when the
visible spline segment is at least 140 screen pixels long and its measured
text area is both label-collision-free and largely free of edges and nodes.
Otherwise, the short map label is tried at the best remaining Bézier position.
The edge label is hidden only when neither variant has enough free space. The
complete description remains available through the hover tooltip.

Hierarchy and edge labels are placed together using a zoom-dependent priority.
Coarse topic levels reserve space first while zoomed out; important edge labels
gradually gain priority as the viewer zooms in. Previously visible labels
receive a stability bonus but must still pass the collision and free-space
checks. Detailed descriptions become eligible at zoom level 4.25. Eligibility
does not depend on whether a parent label or the short label was visible in the
previous frame. Detailed descriptions are embedded independently from short
map labels, so reviewed short labels remain unchanged. Where enough space is
available, the full description is displayed; otherwise, the short label is
used or the annotation remains hidden until a closer zoom. Effectively
point-like edges use the short label only. Their detailed descriptions remain
embedded in the data and available in the hover tooltip.

## Visual encodings

The fill color of every revision point deterministically describes the change
relative to its direct predecessor:

- `added` — gold `#F2C14E`: words were only added
- `removed` — violet `#8F5BD7`: words were only removed
- `mixed` — orange `#D9822B`: words were both added and removed
- `no_change` — gray `#B8B8B8`: no change in the cleaned text
- `start` — dark gray `#343A40`: the first point has no predecessor

The classification comes exclusively from the Git word diff and does not use
an LLM call. Standard revision points are drawn without an additional outline
or ring and use a high fill opacity.

Cluster hulls, intra-group trajectories, and standard direction arrows are
drawn in the neutral trajectory color. Temporal-section point rings are not
rendered. Macro topics remain available as semantic labels but have no color
encoding or legend entry. Signal red identifies exact SHA1 reverts. Noise
revisions remain circular like all other revisions and receive the same
change-type color encoding. Direct semantic cluster labels communicate cluster
identity; the color legend contains no macro topics.

Each cluster hull contains all projected revision points assigned to its
HDBSCAN cluster. A slightly expanded convex hull is drawn for regular
two-dimensional point sets. If the projected points lie on a line or at
effectively the same position, a thin line connects their distinguishable
positions instead. No additional round end caps are drawn.

The chronological main trajectory is rendered as an open centripetal
Catmull–Rom spline (`alpha = 0.5`). Every connection between consecutive
revisions is represented by its own cubic Bézier segment, keeping edge
metadata and width individually addressable. The curve interpolates every
revision point and therefore changes neither point positions nor chronology.
Reflected neighboring points define the endpoint tangents. Coincident revision
points produce a zero-length segment, while bounded tangents prevent visual
spikes. Revert arcs remain an independent overlay.

Points and trajectory lines are deliberately prominent in the detailed view.
Every standard spline segment that is at least 34 screen pixels long receives
an arrow at the center of its visible curve to show chronological revision
direction. The arrow angle follows the local spline tangent and retains its
screen-space size while zooming. Trajectory and exact-revert colors are derived
from the same constants as their legend symbols. Hulls and intra-group lines
remain neutral; point fills represent the four change types plus Start.

## Exact reverts

Reverts are detected exclusively as exact SHA1 returns: a revision is a revert
when it restores an earlier article state with an identical SHA1 hash. The
two-dimensional projection does not influence this classification.

Exact-revert trajectory segments do not receive standard direction arrows,
preventing outward and return movements from becoming unreadable through
overlapping arrowheads. A dotted red return arc begins at the revision whose
change is being undone and ends at the new reverting revision. In a sequence
`Revision 1 → Revision 2 → return to Revision 1`, the arrowhead therefore
points from position 2 back to position 1. A separate revert glyph with a white
background mask is no longer drawn; the dotted arc provides the complete
encoding while the standard revision point remains visible.

SHA1 equality determines the revert independently of whether the involved
revisions were created by different editors or by the same editor. Self-reverts
are therefore also shown in red. If the projection or stored rounding places an
exact SHA1 revert slightly beside the restored revision, only that exact revert
point is visually snapped to the coordinates of its `restored_revision_id`.

If the same visual return connection occurs repeatedly in either direction,
nearby endpoints within five percent of the connection length are treated as
the same pair. Only the chronologically latest exact revert is drawn. Its arc
starts at the revision undone by that final event, and its arrowhead points to
the latest reverting revision. Earlier overlapping forward or return arcs for
the same connection are suppressed. The arcs are computed in screen
coordinates and do not modify the semantic MDS positions.

Pure status labels without a content change (`No content change`,
`Unchanged content`, or `Content unchanged`) are not rendered on the map. The
corresponding revisions and trajectory edges remain present.

## Installation

Python 3 and Git are required. Generating new labels additionally requires an
OpenAI API key.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Store the API key only in the ignored `.env` file:

```text
OPENAI_API_KEY=sk-...
```

## Running the pipeline

### 1. Download the revision history

```bash
python src/wiki_api.py --article Palestine
```

The XML file is stored at `data/Palestine/Palestine.xml`.

### 2. Generate the projection

```bash
python src/main_pipeline.py \
  --article Palestine \
  --start 2012-12-02 \
  --end 2014-07-23 \
  --sample-strategy all \
  --output-name Palestine_small
```

The pipeline produces a label-free map using the same interactive foundation
as the final semantic-zoom visualization. It also stores projection metadata,
revision vectors, and the cosine-distance matrix. Macro groups are derived
deterministically from the projection; this step requires neither LLM calls nor
label caches. Output files receive sequential version numbers.

### 3. Build the final map

The final map can be reproduced from the reviewed projection, hierarchy, edge
labels, and revert analysis without making another API call:

```bash
python src/build_final.py
```

The compact final build reads four relative input paths from
`final_build.json`:

1. The revision projection
2. The reviewed semantic hierarchy
3. The complete edge-label dataset
4. The exact-revert analysis

It does not require previous HTML visualizations, the Wikipedia XML dump,
embeddings, obsolete intermediate caches, or API access. This allows the
checked-in `Palestine_final.html` to be reproduced byte for byte on another
machine.

Complete ISO timestamps stored in edge descriptions are shortened locally to
`DD-MM-YYYY` in visible detailed labels. Source data retains the complete
timestamp, and no additional API call is required.

A reviewed label pyramid can be marked as fixed in the hierarchy JSON. Its
macro, cluster, and section labels are then preserved across later prompt or
logic changes. Fixing the hierarchy also survives editorial corrections to
individual edge labels. It is invalidated only when the projection—and
therefore the geometric basis of the hierarchy—changes materially.

## Interaction

- Mouse wheel or trackpad: continuously zoom around the pointer from 1% to
  200×. When zooming out, geometry contracts while lines, points, and text
  retain their screen-pixel sizes. At the outermost scale, the map collapses
  into a point with the prioritized article label from the active dataset.
- Drag: pan the map.
- Double-click: restore the full view.
- Revision point: display revision ID, timestamp, author, and incoming change
  type.

The HTML file embeds the data and Vega specification but loads Vega and
Vega-Tooltip from a CDN. It is therefore not yet a fully self-contained offline
document. For a presentation without internet access, additionally provide a
static SVG or PDF export, or bundle both JavaScript libraries locally.

## Project structure

```text
src/
├── build_final.py        reproducible final build without API or XML access
├── wiki_api.py           download Wikipedia revisions
├── text_cleaning.py      convert wikitext into comparison text
├── main_pipeline.py      embeddings, MDS, HDBSCAN, and base map
├── revert_detection.py   detect exact SHA1 reverts
├── edge_labels.py        one Terra label per complete revision pair
├── apply_edge_labels.py  apply reviewed edge labels to an existing dataset
├── estimate_label_cost.py
│                          estimate the scope and cost of a label run
├── semantic_zoom.py      hierarchy and interactive final map
├── revision_keywords.py  optional detailed diff-audit pipeline
├── git_word_diff.py      validated word diff for audit data
├── llm_provider.py       OpenAI client and model configuration
├── config.py             shared paths and CLI arguments
└── output_utils.py       sequentially numbered output files

tests/                    local tests excluded from Git
data/Palestine/           local XML revision history excluded from Git
output/Palestine_final_data/
                          four reviewed source files for the final build
output/Palestine_final.html
final_build.json          relative inputs for the reproducible final build
```

## Local tests

The `tests/` directory is intentionally excluded through `.gitignore` and is
not part of the public repository. A fresh clone therefore contains no tests.
When the local test suite is available, run it with:

```bash
PYTHONPATH=src python -m pytest -q
```

The local tests cover revision order, full-text context, diff reconstruction,
exact-revert detection, fixed label positions, priorities, and collision
handling. Independently of the local test suite, the checked-in final `v8`
label dataset contains 619 non-empty labels for all 619 edges.
