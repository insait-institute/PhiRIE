# SR3 — Task-Conditioned System Verification and agentic repair

**Owner:** verification/agentic agent. **Priority:** P0 after one import. Read ../PROTOCOL.md. Reuse `agents/orchestrator/{controller,evidence,policies,runtime,job_graph}.py`, the task graph and collision exporter. User-facing terminology is System Verification; do not spend time renaming all legacy audit filenames.

## Why this experiment is needed

The existing E3 isolated probe repositions a hull over a standard plane. It is not a measurement of contact at the inferred room placement. This new module evaluates the actual imported room state and makes targeted, observation-supported corrections. Do not promote the old acceptance-probe pass to room validity.

## Task graph

Use public task instruction, reconstructed object roles and known robot/cameras. Nodes: manipulated object, destination, supports, relevant obstacles, known robot and cameras. Cover both approach AND object-to-destination transport corridors. A conservative inflated geometric corridor is a proxy, not a motion-plan guarantee; label it. Keep other room geometry in physics even if it is not used for feature aggregation.

Verification stages:

1. Schema/frame/unit and finite pose/scale.
2. Required role present and uniquely grounded from public observations.
3. Actual object-support gap, overlap and initial penetration at inferred placement.
4. Actual room settle: displacement, rotation, contacts and finite state, without hidden teleportation to the floor/table.
5. Open-container physical accessibility using reconstructed collision; no cavity hallucinated from category labels.
6. Known robot/camera reach/visibility and collision checks. Real IK checks are distinguished from distance proxies.
7. Typed decision: accept, targeted retry/repair, reject candidate, abstain task.

Use named raw quantities and standard errors, not one new headline score. Input evidence cannot contain hidden GT geometry, reference outcome or evaluator error. The same full-room simulator may be used as a construction tool; a successful test in that simulator is not proof of reference fidelity. SR5/SR6 provide independent consequences.

## Bounded repair policy

Start from existing B3 (evidence selection + signed-axis registration retry). The new B4 permits at most TWO additional tool actions per failed object/task dependency in the development-configured action bank:

- rerun registration under an unused pose/scale hypothesis supported by observed geometry;
- adjust support-consistent pose using the estimated room support, bounded displacement and observed-surface residual constraints;
- rebuild collision with a stricter multi-part decomposition or declared minimum thickness;
- reselect an already generated candidate after room-context evidence.

Each action produces a new artifact or an executed validation with new evidence. Repeatedly loading the same mesh is not a repair. Store before/after transform, source observation hashes, reason, action parameters, CPU/GPU time, and invalidated descendants. Updating pose/mesh must refresh collision, live GS transform, removal/completion when affected, and native task handles. Never change verification thresholds, freeze an object to stop drift, remove a relevant obstacle, replace a missing asset with GT, or exceed bounds to force a pass.

Parameters and thresholds are set on DEV based on task scale and reconstruction uncertainty, then frozen. Separate mandatory safety/load failures from conservative confidence rejection. Preserve all TEST task outcomes. Do not discard every unstable object and claim a better simulator by conditional quality alone.

## Controlled ablation

Use the same initial captures and candidate pool:

- B0: fixed single TRELLIS.
- B1: two initial candidates, fixed priority.
- B2: existing evidence selection.
- B3: existing bounded registration retry.
- V1: B3 + room verification/abstention, NO repair.
- B4: B3 + task-conditioned room verification + bounded repair.
- BM: fixed-order execution of the same extra action bank, same maximum action/candidate budget as B4, no failure-dependent action choice.

For BM, prerecord the action ordering and same final selection rule; charge all executed actions. Report realized wall/GPU/CPU cost. A maximum-budget match is not exact FLOP equality and must not be claimed as such. B4 vs BM is the important check against 'just spending more compute'. Run full generation/controller diagnostics first on all task-build objects; only B0/B3/B4 enter the initial expensive closed-loop matrix.

Optional two small ablations on frozen development-independent TEST tasks: scene-global versus task-local verification with identical features/thresholds, and w/o room-contact features. Do not train a high-capacity verifier just to add an AUROC table.

## Diagnosis before repair

On a development failure, render synchronized collision/visual views, report actual support contacts and closest points, inspect native mesh pose/COM and scale. Run one-variable diagnostic interventions. Oracle-support/pose/physics interventions are permitted ONLY in the privileged diagnostic process and clearly labeled; their gains are not B4 performance.

## Proposed outputs and commands

Implement `robo/roundtrip/system_verification.py` and configs for thresholds/actions. Extend the existing job ledger.

```bash
python -m robo.roundtrip.system_verification --build "$BUILD_MANIFEST" --task "$PUBLIC_TASK" --config configs/experiments/sim_recon_sim/verification.yaml --out "$OUT/verification"
```

Produce `verification_before.json`, `repair_ledger.jsonl`, `verification_after.json`, new build manifest, task-coverage summary, action costs, and authentic before/after videos. Controller changes remain separate from table formatting.

## Smoke and acceptance

Fixtures: floating object, overlapping support, incorrect unit scale, closed convex container, genuinely unsupported target, and an already-correct scene. Tests verify bounded retries, no GT access, no threshold mutation, no fake welded support, dependency refresh, idempotent logging and original failed evidence preservation.

Pass the implementation when diagnosis is correct and actions are bounded/reproducible, even if a scientific gain is not yet known. Claim improvement only from the matched SR5/SR6 outcomes at stated coverage/cost. If B4 fails to help, report it and keep B3 as the primary method; never force a positive result.
