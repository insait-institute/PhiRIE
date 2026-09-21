# E6 construction input readiness

The 24 public query annotations expand to 72 prospective units. **They cannot
currently be converted into real task-local features.** The inspected integration
audit configuration declares no public construction roots or manifest, and no
separately audited public construction manifest was supplied. This is a scoped
input-readiness conclusion, not a claim that no artifacts exist elsewhere.

`missing_construction_inputs.json` preserves all 72 query/condition identities
and their original query hashes. It records available public RGB/camera inputs,
the exact feature-stage bundle contract, unresolved requirements, and hashes of
the inspected config/code. All terminal build statuses are null. **Missing or
unattempted inputs are not recorded as failed builds or invalid labels.**

## Reproduce the audit

Use the repository environment, which provides PyYAML (the system Python does
not). From the checkout containing this package:

```bash
/group/worldcept/code/SimAny/.venv/bin/python \
  configs/experiments/icra2027/public_task_queries/audit_construction_inputs.py \
  --integration-root /group/worldcept/code/SimAny-wt/icra-integration \
  --out /tmp/e6-public-construction-input-audit.json
```

Choose a new output filename; existing files are never overwritten. Expected
exit code is **3**, with `BLOCKED_MISSING_OR_UNREVIEWED_CONSTRUCTION_INPUTS`,
72 retained prospective rows, zero feature rows written, and three passing
rejection tests. Exit 2 means malformed input. Source checks also verify all
42 public source hashes and the existing annotation integrity contract.

The command does not execute/import the central feature driver, scan outputs,
follow construction manifest references, launch GPU work, or open evaluation
inputs. If a future config declares a manifest, this audit reports it as
uninspected rather than silently accepting it. Admissibility must first be
established for that particular source.

## Exact required public construction inputs

The current driver is `robo.eval.build_task_support_dataset`, stage `features`.
It needs one exact 72-row population manifest, referencing one hashed bundle per
row. The six scenes times three conditions require 18 construction states;
scene evidence may be shared across a scene-condition's four query bundles,
while role/region associations remain query-specific.

| Layer | Required content |
|---|---|
| Resolved config | Concrete common `freeze_id`; full/real population; 24 task IDs; clean/mild/severe; explicitly audited `public_roots`; hashed `public_manifest`; opaque pinned label-protocol hash |
| Population manifest | `freeze_id`, `rows`; each row has freeze/scene/task/condition identity, `task_family`, and `build_manifest: {path, sha256}` |
| Every bundle | Matching identity; `evidence_source: construction_observation`; `source_kind: real`; an observed terminal `build_status` from complete/failed/rejected/abstained |
| Complete bundle | `scene` and `task`; optional `audit` observations; original `query_sha256` and source-frame association provenance retained by the adapter |
| Metric scene | Construction-local entities with `id`, semantic `label`, and metric `aabb`; metric `robot.base_pos`; declared policy cameras with `id`, `pos`, `look_at`; a common validated frame |
| Task | Original `task_id` and `language.instruction`, requested roles, and public construction-derived association evidence; no hidden mappings or `role_refs` |
| Target region | Distinct public construction-grounded placement patch or receptacle interior corresponding to the annotated polygon; do not merge different patches into one generic countertop target |
| Optional observations | Per-entity visual removal/ghosting, registration/disagreement/scale/surface-distance, and support/drop/settle checks; exact consumed field paths are in the JSON audit |

Missing optional measurements should remain missing with indicators. The
consumer can process a genuinely failed/rejected/abstained build without scene
geometry and preserves its row; this does not authorize inventing a terminal
outcome for unsealed, unattempted, or absent construction data.

The feature driver directly calls `build_graph(scene, task)`, bypassing the
file-loading helpers that fill defaults. Public bundle producers therefore need
to supply the necessary scene structure explicitly. A construction-local entity
ID is permitted only when derived from public construction provenance; the RGB
annotation package supplies no such IDs.

## Why an image-box converter is not an admissible adapter

The graph builder does not consume image boxes or polygons. Its support,
visibility, and reach calculations require metric AABBs, robot pose, and camera
geometry. Public capture intrinsics/extrinsics do not supply object depth,
reconstruction scale, policy-camera identity, or robot-to-scene alignment.
Dynamic capture states also require association to the query's single reference
frame. No geometry or successful recovery was inferred from the 2D annotations.

The existing graph's swept-workspace proxy anchors the robot-to-manipulated
object when that role resolves. It does not describe the full subsequent
object-to-destination transport segment. This limitation matters for templates
that share a manipulated object but request different destination regions.
It is documented here without changing the graph builder or evaluator.

Once an audited public construction manifest and real metric association
evidence are available, a bounded adapter can preserve each canonical task hash,
produce complete or observed-terminal-failure bundles, and verify exact 72-row
coverage before invoking the existing feature stage. The present audit is not
a feature bundle, validity table, or substitute for that construction work.
