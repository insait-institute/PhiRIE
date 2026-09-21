# Prospective public geometry and role grounding

This stage extends the existing derived-render/TSDF/SAM3 producers. It produces
construction evidence and query associations, not validity labels, successful
physics, calibrated policy cameras, or task-local prediction results.

## Fixed producer and data boundaries

The input must be an independently validated unit in the exact full RGB
constructor at `b2c827aa5c675ae8578a6bd1aec5c4a8f0402d99`, whose leaf measurement
source is `7c76b2c39ceb515237d8eac490fbf76813092569`. The original validator runs
from its recorded clean checkout. A reused original unit retains its original
path and source. A failed RGB constructor never invokes grounding models; its
four original queries remain explicit failed-construction rows.

The existing `agents.discover.derive_mesh_from_splat` accepts explicit
`--scene-dir` for predicted camera metadata and `--splat-ply` for the sealed
Gaussian. The new `--frame-stride 1` uses every selected public reconstruction
view, including sparse three-frame conditions. Old defaults remain unchanged.
TSDF voxel size, truncation, depth cutoff, alpha filter and all SAM3 detection /
merge thresholds remain unchanged. Discovery uses the existing full vocabulary
and `--frame-stride 1 --mesh-path <derived_mesh.ply>`.

The new bridge `run.icra2027.e6_public_grounding` calls these producers. Render
uses the existing Gaussian environment, TSDF uses primary Python, and discovery
uses the existing SAM3 environment/source/checkpoint, authenticated through the
existing runtime identity producer and E0 content hash. Worker reads of legacy
public capture directories, ScanNet++ reference directories, and vault/oracle
paths fail closed. `s3_lift` is never invoked because its current CLI reads GT.
No source env or model installation is required.

## Predeclared query association

`robo.certification.public_grounding.PROTOCOL` fixes the full protocol before
running this stage. An automatic mesh instance is projected into the actual
query source view and must agree with predicted rendered depth within 0.05m,
alpha >=0.6 and at least10 visible vertices. Public annotation boxes remain
approximate boxes; they are not treated as segmentation masks. Association
retains all lexically supported candidates with box IoU >=0.15; a top-two IoU
gap below0.05 is ambiguous. No outcome-dependent object replacement is allowed.

A query polygon is sampled on the predicted visible depth surface, requiring
at least30 samples at fixed stride4. It represents an observed metric surface
patch, explicitly **not** a verified solid or receptacle volume. Distinct query
IDs receive distinct region IDs. Missing source views stay unresolved; clean
views/depth are not inserted into degraded inputs. The original24 query hashes
expand to all72 scene/query/condition identities, including failures.

## Virtual benchmark frame declaration

Clean public discovered geometry alone may propose a virtual robot placement.
The existing tabletop/base planner is used with an explicitly geometry-only
predicate (fixed public label/observed size constraints). No mass, drift or
physical-stability value is fabricated. Its fallback is rejected if the base
fails the same observed-footprint clearance. The declaration is explicitly not
real robot calibration or verified mounting/support.

A later cross-condition frame binding uses common **predicted public** cameras:
at least3 cameras, centered baseline rank >=2 at1e-4m tolerance, rigid SE3 fit,
translation RMS <=0.05m and orientation RMS <=10 degrees. Scale is never fitted
or applied; the baseline scale ratio is diagnostic and a scale discrepancy
remains in the residual. Rank-deficient/missing/high-residual alignment returns
no transform. `transport_virtual_frame` carries the unchanged clean declaration
with the inverse SE3, preserving the complete4x4 orientation and applying scale1.

The current producer records the clean virtual declaration and keeps actual
`robot_frame=null`. Degraded units never independently select another base.
Binding a clean declaration, transported frame, required policy-camera evidence
and physical body audits into the canonical task-graph/feature producer remains
a subsequent authenticated step; these intermediate outputs cannot be promoted
to complete robot/task graphs or labeled E6 features.

## Execution and validation

After an exact clean source and reviewed generated config/E0:

```bash
python -m run.icra2027.e6_public_grounding --config "$STAGE/execution.json" --stage-root "$STAGE" --validate
python -m run.icra2027.e6_public_grounding --config "$STAGE/execution.json" --stage-root "$STAGE" --scene behavior_task0020 --condition clean --prepare
# Separate ordinary jobs: GPU render, CPU fuse, GPU discover.
python -m run.icra2027.e6_public_grounding --config "$STAGE/execution.json" --stage-root "$STAGE" --scene behavior_task0020 --condition clean --phase render
python -m run.icra2027.e6_public_grounding --config "$STAGE/execution.json" --stage-root "$STAGE" --scene behavior_task0020 --condition clean --phase fuse
python -m run.icra2027.e6_public_grounding --config "$STAGE/execution.json" --stage-root "$STAGE" --scene behavior_task0020 --condition clean --phase discover
# CPU association/publication only after the required phase receipts exist.
python -m run.icra2027.e6_public_grounding --config "$STAGE/execution.json" --stage-root "$STAGE" --scene behavior_task0020 --condition clean --finalize
python -m run.icra2027.e6_public_grounding --config "$STAGE/execution.json" --stage-root "$STAGE" --scene behavior_task0020 --condition clean --validate-output
```

Every condition owns a new output directory. No overwrite or fallback is
allowed. A nonzero producer stops the chain and retains all four queries;
validation exceptions preserve partial evidence without a PASS seal. Terminal
validation rehashes all outputs, authenticates the original construction,
checks every exact worker/leaf command and independently replays associations.

CPU/schema smoke: `pytest -q tests/test_e6_public_grounding.py
tests/test_training_views.py tests/test_e3_auto_discovery_pilot.py
tests/test_e6_missing_graph_state.py` — 66 passed. Tests include sparse all-view
rendering with a forbidden GT-loader sentinel, denied legacy/vault reads,
source/config/checkpoint/protocol drift, no-overwrite, failed-build denominators,
ambiguous matches, missing views, real surface-depth association, rank failure,
scale-error preservation, and rejection of the planner's unclear fallback.
No grounding GPU job or label join has run at this commit.

## Separate resource claims and immutable phase receipts

Each unit first seals its source/config/query plan with `--prepare`. GPU render,
CPU TSDF fusion, and GPU SAM3 discovery then execute as distinct ordinary Slurm
jobs. Every phase has an exclusive claim, predecessor manifest hashes, actual
worker/leaf command, job ID, resource class, elapsed time, return code and output
hashes. A changed predecessor or an existing claim/output is rejected. CPU fusion
and final association reject GPU allocations. Workers require ordinary jobs on
Hala/gcp*/sof1*; no arrays. Failed phases prohibit downstream inference and the
CPU finalizer still writes all four query failures. An upstream rejected RGB
unit requires no model phase at all. `--finalize` can close that unit directly
after preparation. Successful construction no longer accepts combined execution.

Resource-stage tests add dependency ordering, predecessor/claim tamper rejection,
GPU-reservation rejection for CPU stages, no-overwrite, and complete four-query
failure closure without launching later models. Total focused smoke:71 passed.
