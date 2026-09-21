# E6 prospective public RGB task queries

**Preliminary annotation/data preparation. No experiment has run.** This package
contains 24 query annotations (four per fixed scene) and 72 prospective
scene/query/condition rows. It contains no construction validity labels,
construction measurements, evaluated graphs, or policy results.

## Files and use

- `sources.json`: 30 visually inspected public RGB files and 12 public camera
  files, each with SHA-256, byte count, and exact path relative to the declared
  public root. RGB records also identify their public COLMAP camera record.
- `queries.json`: the canonical 24 instructions and frame-local role annotations.
- `condition_expansion.json`: the Cartesian product with clean, mild, severe;
  every row refers to the unchanged canonical query hash.
- `audit_roster_overlay.json`: the scene/task roster proposed for integration.
  **This is not a construction scene `public_manifest`** and cannot replace one.
- `review.md`: compact query list with one reference-image link per scene.
- `validate.py`: standalone standard-library integrity validator and five
  rejection tests. It does not import or modify an evaluator.
- `validation.json`: captured output of the integrity check.

Run `python configs/experiments/icra2027/public_task_queries/validate.py`.
This reads only the package and the declared public RGB/camera source files.

## Prospective selection and source boundary

The fixed roster is `behavior_task0020`, `behavior_task0011`,
`behavior_task0023`, `behavior_task0027`, `behavior_task0045`,
`behavior_task0002`. The initial delegated roster briefly included task0016;
its allowed capture paths were listed, yielded no files, and no contents were
read. The roster was corrected before image selection or annotation.

Before visual inspection, five source frames per scene were selected by sorting
public `*.jpg` basenames lexicographically and choosing Python
`round((N-1)*q/4)` for `q=0..4` (ties to even). All 30 selected images were viewed.
No further images were viewed. Queries and each scene's single reference frame
were then chosen using visible manipulands and target-region observability
within that subset. The reference-frame choice was an annotation judgment;
only the five-frame subset rule was fixed before visual inspection.
It did not use reconstruction quality, policy outcomes, or validity labels.

Read scope was restricted to the E6 README, `task_graph.py`, `grounding.py`,
the initial active checkout's audit config, the corrected integration audit
config, and the six allowed public capture RGB/camera directories. No `gt/`,
oracle configuration, hidden object mapping, vault, reconstruction output,
evaluation output, or other agent transcript was opened. Camera-file header
comments name their capture generator; no generator implementation or private
data referenced by those comments was read. Camera metadata is source provenance,
not GT object identity or measured construction correctness.

## Annotation contract

Each query has `scene_id`, a scene-local `task_id`, `task_family`, `instruction`,
`source_state_anchor`, semantic `roles`, explicit `required_unresolved_roles`,
`missing_annotation`, and `query_sha256`. Identity is `(scene_id, task_id)`.
The query hash covers every query field except `query_sha256`, using UTF-8 JSON
with sorted keys, no ASCII escaping, and compact separators `(',', ':')`.

Roles use approximate manual visible-extent boxes and, for targets, visible
surface/interior polygons in the exact 1280 x 720 source image. Pixel origin is
top left; x increases right and y down. Coordinates are also normalized by image
width/height. The polygons specify task regions; they are not segmentation,
metric geometry, collision volumes, or accuracy tolerances. A receptacle polygon
marks a visible interior patch. A support-surface polygon marks the desired
placement patch. No hidden object IDs or `rubric.role_refs` are supplied.

Each scene uses a single reference image for all four queries. Other inspected
frames provide review context only; they are not combined into a static object
state. The captures visibly contain changing object, door, and robot states.
Temporal consistency, cross-frame 3D identity, and consistency with the eventual
construction state remain unverified.

The four queries are templates, not a claim of four distinct task families.
Some reuse the same visible object with different meaningful placement regions;
others cross two visible manipulands with two receptacles/regions. These are
prospective counterfactual task requests, not reconstructions of the original
demonstration's intended task. No query was selected to reproduce an observed
success. Small colored indicators in captures were not interpreted as outcomes.

All 24 slots have source-visible instructions and role regions, so there are
zero missing *2D annotations*. This does not mean all queries are supported by
construction. Future missing annotations must use `annotation_status:
missing_annotation` with an exact reason and retain their scene/task/condition
slots; they must not be silently dropped.

## Explicit limitations and downstream requirements

- All metric object/target grounding is unresolved. Book-like, yellow curved,
  and orange ridged are visible appearance descriptions, not recovered category
  or instance identities. RGB boxes are approximate and require review before
  an immutable experiment freeze.
- The visible robot box is a reference only. Metric robot alignment, reach,
  manipulation feasibility, swept obstacles, and policy-camera identity or
  visibility require separate public construction evidence.
- A source-support candidate is a visual hypothesis, not a contact label.
  Destination capacity, object fit, collision clearance, and support relations
  are unknown. The microwave turntable is partly occupied. The burner query
  requests placement only; appliance/temperature state is unspecified.
- Bathroom mirror reflections are excluded from entity counts. The pink object
  is identified on the physical ledge, not from a reflected duplicate.
- Public COLMAP files are emulated capture extrinsics; the Nerfstudio JSONs
  contain intrinsics only. Neither establishes a validated scene/robot frame or
  the eventual policy-camera configuration.
- The existing task-grounding interface resolves scene-object labels and IDs;
  it does not consume these pixel regions directly. A public construction
  adapter must associate these regions with construction-derived entities and
  preserve unresolved roles before the E6 feature stage. No such association
  or adapter is claimed by this package.
- The condition expansion creates 72 prospective units only. Real features,
  separate label join, LOSO fold health, and E6 claim gates remain outstanding.

Integration may copy the declared task roster into the audit config, but must
not substitute this annotation file for construction outputs or treat its
`annotation_status` as a validity label. Keep the query content unchanged across
conditions; any deliberate annotation revision requires new hashes and renewed
prospective freeze before feature/label evaluation.
