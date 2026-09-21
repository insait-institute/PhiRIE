# F2 — Native scorer sensitivity, without changing official outcomes

Owner: evaluation agent. Implemented module `robo.roundtrip.scorer_sensitivity`.

The native sink goal uses body-origin and gripper distance. An object's generated local origin need not correspond to its native origin. This diagnostic asks whether reference-point changes affect interpretation; it does NOT retrospectively define a better official score.

## Procedure

Use original L0 planned units and the authenticated terminal ledger. Preserve unexecuted units. For every complete executed trace, restore the exact imported XML/assets in a fresh evaluator, set the stored final qpos/qvel/time, and call `sim.forward`. No policy calls, no physics steps, no geometry repair. Verify tracked poses and logged native success before reporting a supplemental diagnostic.

Extract the actual world-space visual triangle mesh from compiled MuJoCo geometry. Compute its area-weighted surface centroid, body-origin offset, gripper-to-origin and gripper-to-centroid distances. Report whether the original 25 cm retreat comparison disagrees between those points. This is a reference-point sensitivity test, not a claim that surface centroid is the unique physical ground truth.

For sink/cabinet tasks with declared native interior sites, also report visual vertex membership in the union of the corresponding parallelepipeds. Normalize axes explicitly. This does not duplicate upstream's unnormalized margin, certify every triangle of a nonconvex union, or certify physical cavity/contact validity. `SinkToCounter` may have no supported fixture region and must remain unavailable for this check rather than substituting a guessed box.

## Run and inspect

Use parent README command in the existing allocated native renderer environment. The implementation is serial by default and can be expensive because each unit restores an environment. Start on the frozen DEV roster; do not conclude correctness from pure triangle tests. On a runtime/schema mismatch preserve `DIAGNOSTIC_UNAVAILABLE`, inspect the first exact exception and fix independently diagnosed adapter bugs in a new code/output version.

Outputs: `analysis_manifest.json`, one `units/<unit_id>.json`, `sensitivity_records.jsonl`, `sensitivity_summary.json`. Each measured row includes native components, source hashes and explicit `policy_calls=0`, `physics_steps=0`. The original ledger is read-only. Changes in aggregate interpretation must be described as post hoc sensitivity, not replacement of official native success.

## Acceptance

- Known local-frame reexpression preserves world visual geometry and centroid tests.
- Official and restored native outcomes match before supplementary checks.
- No real trajectory is synthesized and no construction artifact is edited.
- All absent/invalid/unexecuted observations have typed status and remain counted.
- An external anatomy/containment claim requires stronger independent reference data than this bounding-region probe.
