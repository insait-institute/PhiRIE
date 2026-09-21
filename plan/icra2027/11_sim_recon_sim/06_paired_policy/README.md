# SR6 — Paired closed-loop robot evaluation

**Owner:** canonical-harness agent. **Priority:** P0. Dependencies: SR0 reference and SR2 import. SR3/SR4 enable their arms; do not block native arms on Harmonizer. Read ../PROTOCOL.md and ../TABLES_AND_METRICS.md.

## Integration requirement

Use the existing `robo.eval.harness_runner`, ledger, freeze and validation infrastructure through native environment adapters. The current typed scene/collision/observation enums and DROID-specific state helpers cannot represent RoboCasa/OmniGibson directly. Implement backward-compatible adapter dispatch and versioned schema extensions. Native task/scorer code remains authoritative. Do not instantiate `pi05_tasks` and call the result an official RoboCasa task.

A convenience `robo.roundtrip.policy` CLI may prepare the native workload, but it must write the same canonical ledger through the shared writer. No independent success formula in the convenience CLI.

## Core matrix

On RoboCasa TEST: 8 layouts x 2 task instances/layout x 20 reset interventions = 320 planned episodes per arm, one frozen compatible checkpoint.

1. REF_NATIVE.
2. FIXED_NATIVE (B0).
3. AGENT_NATIVE (B3).
4. ROOM_REPAIR_NATIVE (B4).
5. ROOM_REPAIR_GS (same B4 physics).

Total 1,600 episodes. Reference/native construction arms can run while SR4 finishes. REF_GS_DIAGNOSTIC and ROOM_REPAIR_HC are optional, 320 each. A second policy repeats the declared complete matrix as its own block. No pooling policies with different action/control contracts.

Training is not a prerequisite when a compatible released checkpoint exists. Do not fine-tune on reconstructed TEST outcomes. If training becomes necessary, limit it to native TRAIN tasks/scenes, fix the budget and checkpoint before reconstruction TEST, and retain policy-training scope in the report.

## Preflight and task admission

- Validate all task/scene/reset IDs, native task goals, exact robot and cameras.
- U0 and U1 must pass their engineering tolerances.
- Fixed reference task admission precedes construction and uses native capability, not future method success.
- A failed reconstruction/role match/load or rejected task remains in planned coverage. No alternate easier reset or smaller object replaces it.
- Conservative system-verification rejection is an abstention with explicit lost coverage. It is not an executed policy failure.
- The evaluator may run a clearly labeled offline validation probe on rejected reconstructed scenes if safe, to study false rejection. Such diagnostic runs cannot become accepted operational-policy results or feed TEST construction.

Do not repeat the previous failure mode of requiring robot/camera to be inferred from the room scan. These are known benchmark inputs. Conversely do not bypass real support/penetration failures solely to achieve nonzero rollout count.

## Reset, stepping and randomness

One immutable reset bank supplies paired world-frame perturbations. Preserve each method's estimated placement; see PROTOCOL.md. Persist commanded and realized post-settle states. Do not force identical reconstructed object qpos using oracle GT. Match known robot state and task semantics, not state outcomes produced by different geometry.

Same checkpoint/stats, proprioception, RGB order, controller, gains, action convention, clipping, timestep, horizon and success predicate. Native/RGB comparison changes only observation adapter with identical compiled physics. Warmup is isolated; per-episode policy history and sampling noise reset deterministically. Freeze stochastic policy RNG keys by scene/task/reset/action-chunk. Order treatment execution in balanced randomized blocks to control service/hardware drift.

Log policy input and action timestamps so inference lag is distinguishable from simulator stepping. Synchronous slower-than-real-time rollout is allowed and reported. A video at 30fps does not prove real-time control.

## Per-episode record

Extend existing schema with `benchmark`, native release/task/instance IDs, reconstruction method, replacement scope, capture/build hashes, execution_kind=`closed_loop_visual_policy`, native policy config/stats hash, reset intervention, oracle-assistance flags and native predicate outputs. Include unique episode ID, terminal outcome, success nullable where unmeasured, stages with native definitions, frame/trace paths, wall time, observation/policy latency and source hashes.

No-result states and measured failures are different. Every planned unit receives a terminal record or remains visibly incomplete. Do not equate 2,690 repeated prerequisite cells with 2,690 policy attempts. Shard writers never concurrently append to a shared network JSONL without transactional guarantees; use disjoint shards and a validating merger.

## Metrics and analysis

Primary: native task success, executed/planned coverage, success count/planned requests, and success conditional on executed episodes. For the complete admitted benchmark use task-macro averages with exact denominators, plus episode-micro supplements. Gaps to reference are percentage points, not relative percentage gain. Report all planned outcomes and common-evaluable paired analysis separately. No success-gap claim from an all-zero native reference.

Paired bootstrap: resample layouts, then task instances, then matched reset IDs, keeping both arms together. Report 95% intervals and McNemar discordant counts as a diagnostic. With few layouts, label intervals descriptive and avoid universal claims. Do not tune sample count after significance inspection. No Pearson/policy-ranking headline with one policy.

## Proposed commands

```bash
python -m robo.roundtrip.policy --config configs/experiments/sim_recon_sim/policy.yaml --phase validate --out "$OUT/rollouts/preflight"
python -m robo.roundtrip.policy --config configs/experiments/sim_recon_sim/policy.yaml --phase run --out "$OUT/rollouts"
```

Implement first; canonical runner and current CLIs stay backward compatible. Use ordinary independent Slurm jobs by task/method shard, not arrays. Export the exact planned matrix before job submission.

## Pilot and acceptance

Pilot: two DEV rooms, two tasks each, five resets, reference plus FIXED_NATIVE first. Add B3/B4 and GS incrementally; preserve same pilot identity. Tests cover native scorer binding, replay mislabeled as policy rejection, checkpoint drift, estimate-preserving reset, unique IDs, crash recovery and no silent enhancer fallback.

Deliver complete ledger, per-task/per-method counts, coverage, success and paired intervals, failure taxonomy, continuous videos and a small declared demo-candidate list. A truthful negative result completes the experiment; claiming improved manipulation requires measured gains with stated support and uncertainty.
