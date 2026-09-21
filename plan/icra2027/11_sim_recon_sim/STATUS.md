# Active native scale-up — 2026-09-07 02:20 UTC

The original one-object milestone below is archived, complete DEV evidence. Its old budget and “no running jobs” statement describe that historical milestone only. Current execution is governed by [09_scale_up/EXECUTION_STATUS.md](09_scale_up/EXECUTION_STATUS.md).

- Original five-reset continuation is complete, including the failed reference seed. New eight-instance DEV has200/200 terminal units across five arms,105 actual rollouts; REF34/40,B03/40,B37/40,B44/40,BM2/40 successes/planned. Complete generated DEV tables and paper are pushed; no preservation claim is inferred.
- TEST L0:48 canonical captures,48 ordinary policy-engine jobs and2400 planned units. Canonical snapshot007 has1305 terminal,265 executed and1095 unmeasured; outcome totals remain preliminary. `20260907-73ff4ab-v1/sim_recon_sim/scale_up/test_execution/release_snapshot_007/merged` is authoritative for that snapshot. Independent construction:32/48 B0 builds,16 automatic TRAIN-discovery failures, shared candidate pools reused.
- All24 declared destination builds have ordinary jobs. L1 canonical policy worker and full-substep contact logging passed a real DEV repeat. L1 B4/BM repair pilots both exhausted two actions and abstained after retained-context collisions; failures are preserved. L2 obstacle inventory remains unresolved. Scope TEST policy outcomes and confirmatory equal-horizon TEST feedback are NOT_RUN; the latter lacks a reset roster frozen before outcomes.
- Source-GS fitting and two original heldout views are measured DEV evidence; clean background and GS policy results are not yet available. Main code487cbe9 and paper053eb2e were independently confirmed on origin/main. Full raw captures/models/videos remain in shared immutable stages; generated JSON/CSV exports and their hash index are committed.
- Existing queue controller PID2941001 and final collector PID3269657 continue the primary jobs; warm evaluator watcher PID3620196 and destination watcher PID3991333 resume only missing units. Job IDs and exact commands are in the scale-up board and individual task STATUS files. No old ScanNet++ result or frozen experiment was overwritten.

---

# First native comparison completed — 2026-09-06 18:26 UTC

The requested first milestone is complete. The full sim-recon-sim benchmark is NOT_RUN. Existing ScanNet++/E4/paper cohorts are unchanged.

- Owner: root; policy/replay/paired e4_endpoints; capture e4_planning; constructor/import/report sr2_import.
- Branch: main; experiment source85051d4; report producer6c50f76. Final documentation commit is the Git commit containing this file.
- Package: [/group/worldcept/code/SimAny/outputs/icra2027/20260906-6c50f76-v1/sim_recon_sim/first_milestone/README.md](/group/worldcept/code/SimAny/outputs/icra2027/20260906-6c50f76-v1/sim_recon_sim/first_milestone/README.md).
- Machine sources: `milestone_report.json`, `episodes.csv`, `coverage.csv`, `diagnosis.json`, `source_manifest.json`; videos are exact original episode bytes.
- Hardware: one allocated H200 onsof1-h200-5; ordinary Slurm service job837413 and independent overlapping steps, no arrays. Service stopped deliberately after completion; scheduler state CANCELLED by10064/0:0, elapsed55m21s. No running jobs for this track.

## Measured results (DEV, not paper-scale)

Native PickPlaceCounterToSink, layout11/style11, PandaOmron, native left/right/wrist cameras, frozen pi05_pretrain_human300/75000, horizon600, replan5. Native libraries unchanged: RoboCasa4f8a298 /robosuite5ce6643 /MuJoCo3.3.1. Policy/openpi source5a6beda and checkpoint revisionc484448 are bound by exact receipts.

- Native pilot:5 planned/5 executed/5 completed,4 successes and1 horizon failure. Seeds0/1/2/3/4 took392/564/306/600/289 ticks. The failed seed3 is retained.
- U0/U1: both10-step controls pass exact initial observations, qpos and native predicates. Full native import replay:392/392 actions, qpos, qvel, observation hashes and predicates byte/value identical.
- Capture:6 TRAIN/2 held-out TEST views at1280x720,21.1031s; restored exact pre-policy seed0 canonical state; all bodies static. Only TRAIN enters construction.
- Build:1 planned/1 built target. Automatic SAM3 confirmation in5/6TRAIN views;6724points. Fixed TRELLIS proposal, observation registration, existing CoACD producer5parts. Segment16.3410s/generate29.8771s/align17.2264s. Uniform-color native rendering, metric RGB-D, original room/destination retained as oracle context.
- Same-policy pair: REF succeeds392ticks; reconstructed target succeeds429ticks. One evaluated pair, no population claim.
- Same-action reconstructed replay:392/392 identical actions executed; native failure. Existing relative-marker displacement RMSE6.5612097613cm, finalrelative discrepancy11.7070333372cm. Absolute common-frame pose metrics remain NULL.
- Exact terminal diagnosis: target BODY ORIGIN inside sink in all3arms. REF gripper distance0.2556951299m and reconstructed closed-loop0.2507560647m exceed the native0.25m retreat threshold; reconstructed replay0.2402232701m does not. This is terminal retreat failure, not no transport. Whole-mesh containment is not certified by the native predicate. Detailed component causes among geometry/pose/physical priors remain unisolated.
- Isolated construction probe remains negative (`settle_stable=false`, `settle_sunk=true`); it was not used to reject this fixed B0 diagnostic or relabeled native stability.

## Reproducible commands and source lineage

All paths below are under `/group/worldcept/code/SimAny/outputs/icra2027/`.

| Stage | Source | Command/script | Result |
|---|---|---|---|
| Native pilot20260906-4ee6462-v2 |4ee6462|`sim_recon_sim/dispatch/native_pilot.sh`|PASS interface;4/5native success|
| Identity20260906-67ee364-v1 |67ee364|`sim_recon_sim/dispatch/native_identity.sh`|PASS U0/U1|
| Capture20260906-67ee364-v2 |67ee364|`sim_recon_sim/dispatch/native_capture.sh`|PASS static6/2|
| Build20260906-85051d4-v1 |85051d4|`sim_recon_sim/dispatch/build.sh`|BUILT1/1; real allowlist/network isolation|
| Pair/replay20260906-85051d4-v2 |85051d4|`sim_recon_sim/dispatch/paired.sh`, `replay.sh`|Both execute;policy success/replay task failure|
| Full import control20260906-85051d4-v3 |85051d4|`sim_recon_sim/dispatch/reference_replay.sh`|PASS392-step exact identity|
| Final-state diagnosis20260906-85051d4-v4 |85051d4|`sim_recon_sim/dispatch/diagnose_final.sh`|PASS logged/recomputed predicate agreement|
| Package20260906-6c50f76-v1 |6c50f76|`sim_recon_sim/dispatch/package.sh`|PASS source/action/config/hash validation|

Stage-specific source commits are explicit. This is a development lineage, not a single-source full experiment freeze. The immutable policy server predates client stage versions; its source/checkpoint and RNG contract remained fixed.

Smoke commands: explicit native interpreter `/group/worldcept/code/SimAny-wt/sr0-native/.venv-native/bin/python -m pytest -q ...`. Integrated57tests passed for paired/replay/state/build/isolation; report8tests passed including negative controls. Earlier92integrated adapter/capture/policy/spec tests passed; real namespace/GPU runtime and real native identity are separately recorded. No new global all-repository test pass is claimed.

Configuration and checkpoint hashes are authoritative in each stage manifest, build_config.json, constructor_launch.json, policy_metadata.json and package source_manifest.json. No manually entered paper values.

## Preserved failures and limitations

Environment setup failures:837411 missing official generative texture;837416 CUDA device parsing;837427 EGL initialization;837436 OSMesa SIGSEGV. Their logs remain under20260906-ea02497-v1/v3/v4/v5. Later ea02497-v7/v8 preserve UUID renderer selection / controller metadata API failures. These are environment/code failures, not policy outcomes.

Identity failure stages775f1fb-v1,745114e-v1,bb9e2ca-v1,4ee6462-v1 remain. Repairs preserve native metadata/XML, integration state, semantic controller state, sensor sampling phase and warmstart; no numerical tolerance was relaxed. Upstream fixture RNG loss means seed alone is not a valid scene identifier; canonical metadata/state binding follows official replay.

PASS: native learned-policy competence; U0/U1; complete392-step import identity; public-only reconstruction; requested first one-object pair/videos/diagnosis.
NOT_RUN: broad manipulation retention, agentic B3/B4 improvement, room/GS/Harmonizer intervention, BEHAVIOR, full T1–T4 and paper promotion. This does not close prior E4/E9 applicability/submission gates.

Next: preregister a compact multi-object/native-task DEV cohort using immutable canonical instances; add support/receptacle replacement and B3/B4 only after their controls. Keep policy failures and unavailable reconstructions in planned counts; then freeze a separate TEST matrix. No outcome-driven favorable reset replacement.
