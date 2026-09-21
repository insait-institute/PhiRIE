# N5 — Paired native policies and matched-horizon action replay

**Priority P0. Owner: rollout/physics-evaluation agent.** Reuse `robo/eval/harness_runner.py::run_native_episode`, `robo/roundtrip/{paired,replay,native_policy,native_policy_server}.py` and existing trace/video writers. Own execution semantics, not a second independent ledger. N1 owns scheduling/merging; N0 owns shared identity validation.

## Closed-loop TODO

- [ ] Complete genuine DEV pairings before adding methods. The first milestone has five reference episodes but one reconstruction pair, not a 4/5-vs-1/1 population comparison.
- [ ] Generalize `paired.py` to resolved canonical instances/resets/treatments with direct XML/state/asset hashes. Keep old frozen result readers backward compatible.
- [ ] Freeze native robot/controller, proprioception packing, three policy cameras, preprocessing, action convention, control rate, policy/checkpoint, task instruction, native rubric, per-task horizon and RNG. Method differences are confined to declared asset bundle, repair or RGB implementation.
- [ ] Generate one terminal record per planned unit. An arm that abstains or cannot build has no execution and no end-to-end service success, but is not mislabeled a policy failure. Unscheduled/external-blocked arms remain unmeasured.
- [ ] Only mandatory load/schema/nonfinite/numerical safety gates halt otherwise scheduled baseline execution. Finite drift, poor grasp or predicted low confidence are possible observed task failures. Do not impose B4's conservative acceptance on B0/B3/REF.
- [ ] Execute 40 DEV resets/arm first for REF/B0, then method pilot B3/B4/BM. TEST core is 48 instances/480 resets/arm, five arms = 2,400 planned episodes. Do not start the entire TEST until relevant contracts and DEV bugs are frozen; do not wait for favorable method gains to release a valid test.
- [ ] Preserve native success definitions; report meaningful reference performance, planned/attempted/executed counts, native successes and task-specific failures. Store per-tick predicate components to diagnose borderline body-origin/retreat cases.
- [ ] Reuse valid source reference episodes across comparisons only with exact instance/reset/policy/termination fingerprints. Existing first-milestone DEV never becomes TEST.

## Fixed-action TODO

The existing replay uses the original 392-action success-terminated prefix. It is valid as a fixed-prefix diagnostic but cannot isolate feedback benefit against a closed-loop arm that gets 600 steps.

- [ ] Add a NEW `full_horizon_feedback_diagnostic` protocol. Generate real reference-policy actions for the entire task horizon H; continue the frozen policy after first success without changing its commands or success thresholds. Run the corresponding reconstructed closed-loop diagnostic for the same H. Primary benchmark termination remains a separate protocol.
- [ ] Replay all H recorded robot actions unchanged in each arm. No zero padding, hand-written retreat, object-pose injection, controller switches or post-hoc horizon extension.
- [ ] Log native predicate per tick, first success, success-at-H and exact trace length. Do not relabel success-ever as success-at-H or mix primary/diagnostic episode counts.
- [ ] Require action/reference/import identity before computing physical differences. Use identical time bases; no dynamic time warping that hides timing error.
- [ ] Keep existing relative-marker metrics labeled relative and initial-error-removing. Establish a single fixed evaluator-side body-frame correspondence for absolute pose metrics where possible. Never fit a transform per frame or use it to reset the reconstructed simulation. Without reliable correspondence, leave absolute metrics null and show separate initial registration/shape error.
- [ ] Keep relative and absolute units in separate fields. Crashed traces retain their available duration and coverage; missing tails are not zero-padded. No successful-reference-only primary sample filtering.

## Budgets for diagnostic replay

Before TEST outcomes, select TWO reset IDs per each of the 48 instances for the full-horizon diagnostic, independent of success: 96 traces/arm. Additional full-horizon reference-policy generation is explicitly budgeted (up to 96 inference episodes if not reusable). Fixed-action physics replays do not need model inference. Compare REF import identity and B0/B3/B4/BM, a maximum 480 replay traces, keeping them separate from the 2,400 primary-policy episodes. Any smaller resource-driven budget is frozen before measuring outcomes, not chosen to show a feedback benefit.

## Outputs

Canonical per-episode manifest, initial/final state, actions, per-tick traces, native predicate components, timings, termination, failures and selected/debug continuous videos. N1 merges compatible terminal shards. N7 consumes `episode_ledger.jsonl`, `replay_metrics.csv`, `paired_contingency.csv`, instance/scene/family support and `comparison_statistics.json`.

## Tests / acceptance

Tests cover canonical instance mismatch, late changed checkpoint, replay-action mutation, H-length invariants, early native success without diagnostic termination, missing tail, reset/renderer RNG isolation, duplicate resumed episode and body-origin metric mismatch. Run a real DEV same-horizon trace and identity control; synthetic passing is insufficient.

Acceptance means all scheduled units have valid measurements or explicit terminal failures, with interpretable native success and coverage. A method losing to REF is a legitimate result. Broader feedback compensation needs multiple paired examples under this equal-budget diagnostic; the existing one unequal-prefix example is only motivation.

Handoff `STATUS.md`: units planned/executed/completed by instance/method/scope/protocol, all failure counts, scorer/identity health, measured timing, output paths and whether any retention claim has statistical support. Do not silently tune the policy on reconstructed TEST observations.
