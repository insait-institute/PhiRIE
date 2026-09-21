# N0 — Versioned protocol, canonical identity and native scorer

**Priority P0. Owner: native-contract agent.** Read `../EXPERIMENT_MATRIX.md` and the existing first-milestone STATUS. Goal: make independent instances and true paired resets possible without losing the exact identity control already obtained. Do not rebuild the manifest stack.

## Read first

`robo/roundtrip/spec.py`, `paired.py`, `native_policy.py`, `native_policy_server.py`, `adapters/robocasa.py`, `importers/robocasa.py`, `replay.py`, `robo/eval/harness_runner.py`, existing `robo/manifest/` and native tests. The current schema explicitly permits only development/target-only/ideal-RGB-D. Add a versioned extension rather than deleting those guards or relabeling old results.

## TODO

- [ ] Resolve actual pinned RoboCasa/robosuite/MuJoCo/OpenPI and checkpoint content receipts from the successful milestone. Keep the baseline policy/camera/action contract. Do not update dependencies to latest during scale-up.
- [ ] Add schema v2 with separate `cohort_id`, `canonical_instance_id`, `reset_id`, `policy_rng_seed`, `scope`, `sensor_regime`, `controller_method` and `execution_protocol`. Keep schema v1 functional.
- [ ] Bind each canonical instance to XML and referenced asset closure, native task metadata, integration state, semantic controller state, sensor sampling/timing, solver warmstart where required, robot state, fixture/object identity and policy identity. Add direct XML/state hashes to new reference `result.json`; retain the older sibling-path validation for old receipts only.
- [ ] Treat repeated seeds with differing native fixture/object identities as different instances. Expose an assertion rather than attempting to repair identity by reseeding.
- [ ] Store reset perturbations separately from native scene generation. Apply the same declared world-frame Delta to reference placement and the method's estimated placement. Preserve initial reconstruction error. Never rerun native scene randomization during a paired arm.
- [ ] Separate native task success from additional geometric diagnostics. Inspect body-origin/site use in each chosen native predicate and freeze scorer bindings after import.
- [ ] Add a physically identical object expressed in two different local frames as a scorer metamorphic test. Adjust mesh/body/inertia/site frames consistently while preserving world geometry; native success should not change solely due to representation. If it does, diagnose and bind the intended task reference point without importing GT shape into construction or changing official thresholds. Unresolved cases cannot support a strong task-retention claim.
- [ ] Hand N5 a versioned full-horizon diagnostic contract distinct from primary native termination. Pin each task's actual native horizon; the first example's 600 steps is not universal.
- [ ] Verify per-client episode/chunk RNG under concurrent, reordered and resumed requests to the warm policy service. Bind stream state to instance/reset identity, not connection ordering. Reconnects must not silently change randomness.
- [ ] Write narrow interface tests plus existing native-identity regression. Run one real no-change native replay for every newly supported task/scorer family on the pinned runtime, not a ceremonial repeat per every identical reset.

## Shared schema contract

`canonical_instance.json`: schema/version, dataset split, task/layout/style, object/fixture IDs (privileged evaluator copy), XML/state/asset hashes, robot/controller/camera/rubric identities, capture association and authorized public task description.

`reset_bank.json`: canonical instance ID, ordered reset IDs, perturbation transforms, unchanged robot/fixture state binding, policy RNG seed, validity diagnostics and reason codes. No generated object is moved to a hidden reference absolute pose.

`comparison_contract.json`: allowed changed asset fields and unchanged-field fingerprints. B4 native vs B4 GS permits RGB implementation changes only. Native-vs-reconstruction is explicitly an asset-bundle contrast. Privileged state/role files remain outside the constructor allowlist.

## Required evidence

Under `outputs/icra2027/<new-stage>/sim_recon_sim/scale_up/contract/`: resolved schema/configs, `identity_controls.json`, `scorer_frame_checks.json`, `policy_interleaving_test.json`, reset contract and source hashes. Each result states real vs synthetic coverage. No same-engine numerical tolerance is relaxed after inspecting a failure.

## Smoke / negative tests

Synthetic fixtures must reject duplicate instances/resets, unresolved TEST roster, mismatched XML assets, controller phase drift, substituted checkpoint, GT pose reset and unrecorded success-threshold changes. Verify old schema still loads; mocked interleaved policy clients produce the same chunk seeds; native scorer-frame fixture diagnoses an origin-dependent predicate.

## Admission and handoff

N1 may inventory and implement dispatch immediately. Real REF/B0 DEV jobs are released after their relevant identity/scorer/schema controls pass; they do not wait for unrelated GS/Harmonizer tests. Write `STATUS.md` with exact source, commands, actual outputs, known scorer limitations and `PAIR_CONTRACT_READY=true|false`. Coordinate N4's importer changes and N5's horizon handling; only this owner edits shared `spec.py` and shared identity contracts.
