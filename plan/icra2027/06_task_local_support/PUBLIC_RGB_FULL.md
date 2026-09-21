# Full public RGB reconstruction cohort

This is the reconstruction prerequisite, not task-local prediction evidence.
The cohort retains six declared scenes × clean/mild/severe = 18 conditions and
72 query rows. It does not unlock labels, load legacy depth/poses/splats, infer
robot calibration, or claim that a room is a complete simulatable twin.

## Separate source identities

`run.icra2027.e6_public_reconstruction_full` owns admission and publication.
The measurement producer remains the exact clean checkout
`/group/worldcept/code/SimAny-wt/e6-public-input-audit` at
`7c76b2c39ceb515237d8eac490fbf76813092569`. Fresh subprocesses run its unchanged
Omega, DA3 metricization, scene packaging and 15,000-step Gaussian modules,
arguments and environment. Both source identities are recorded. The historical
smoke/pilot validator is replayed from that original checkout; its tier guard is
never changed or bypassed for historical artifacts.

The new wrapper admits all 18 units. Four references are fixed prospectively:
scene0023 clean from `20260906-4e440ad-v1`, and scene0020 clean/mild/severe from
`20260906-0af302d-v1`. Original paths, seals, stage execution records and measured
runtimes remain authoritative. Reuse publishes a reference receipt; it does not
rewrite original gates or pretend to rerun generation. The remaining 14 invoke
the same pinned leaf measurement modules. No legacy reconstruction is eligible.

## Measurement identity and exclusions

The recipe checks the exact key set of both old configs, fixed48 selection,
seed0, 15k iterations, no fallback, source commit, full runtime/weight identity,
and normalized producer commands. The original smoke/pilot recipe digest is
`e0629b45b1c5b48d801245821c8acb55b08067b059565d73e783c7f8988aa7bb`.
The complete 18-condition roster contents must match across configs. Per-unit
scene/condition, all source RGB hashes, selected names, query IDs/hashes and
source-view availability remain bound. Object identities and controller are
explicitly not applicable at this RGB-only stage. Camera values are outputs of
the identical pinned RGB inference, not supplied reference cameras. No held-out
fidelity evaluator or treatment threshold changes.

Only publication freeze/output paths and engineering tier/scope/selection labels
are excluded from the common recipe, with reasons encoded in the recipe JSON.
Scene and roster fields are compared separately rather than ignored. Unknown
config fields fail closed. Original source/config/E0/runtime and terminal bundle
validation is required before reuse. The full stage gets a new canonical freeze,
its own exact-wrapper E0 and content-bound configs/roster/runtime/checkpoints.

## Commands and failure semantics

After a clean committed source, reviewed generated `execution.json`, and E0:

```bash
python -m run.icra2027.e6_public_reconstruction_full --config "$STAGE/execution.json" --stage-root "$STAGE" --validate
python -m run.icra2027.e6_public_reconstruction_full --config "$STAGE/execution.json" --stage-root "$STAGE" --scene behavior_task0011 --condition mild
python -m run.icra2027.e6_public_reconstruction_full --config "$STAGE/execution.json" --stage-root "$STAGE" --scene behavior_task0011 --condition mild --validate-output
python -m run.icra2027.e6_public_reconstruction_full --config "$STAGE/execution.json" --stage-root "$STAGE" --cohort-output "$STAGE/audit/cohort_snapshot_v1"
```

A producer nonzero exit seals a typed failure and stops that unit. Unexpected
validation failures preserve partial directories/logs without issuing a PASS.
No output or status snapshot can be overwritten. Snapshots retain PASS, FAIL,
INCOMPLETE and NOT_RUN units in all18/all72 denominators. Construction runtime
and internal training diagnostics do not become held-out or downstream metrics.

Launch only ordinary individual jobs on verified allowed nodes, with explicit
GPU type and no requeue, 30-minute initial cap and unchanged fixed48 inputs.
GPU launches await root review of the concrete full config and exact E0.

## Validation

39 focused tests cover exact recipe partition, unknown parameters/source relabel,
original seal/config/runtime/E0 drift, changed input manifests, no-overwrite,
failed original validation, true producer failure without fallback, pinned CWD,
and all18/72 status accounting. Real old configs have equal roster contents and
identical measurement recipe hashes. No new GPU jobs were launched for code QA.

Remaining prerequisites: public metric role grounding, declared virtual robot
frame across conditions, construction-only evidence, sealed feature rows and
then a separate evaluation-only label join. Nine condition-specific query source
views remain absent; no clean view is inserted into a degraded condition.
