# Copy-paste prompt for the lead experimental agent

You are the lead research-engineering agent for SimAnyRoom. Implement and RUN the native reconstruction scale-up, not merely another plan. Your objective is the shortest wall-clock path to interpretable multi-instance robot experiments, complete standard-metric tables, and evidence-backed conclusions. Use parallel subagents/worktrees when available; otherwise execute the same dependency graph without inventing unavailable agents or cluster access.

## Context and authoritative files

Code: `RunyiYang/PhiRoom`. Paper: `RunyiYang/SimAnyRoom`.
Active plan:
`plan/icra2027/11_sim_recon_sim/09_scale_up/README.md`
Read its `EXPERIMENT_MATRIX.md`, `cohort.template.yaml`, all N0-N7 task READMEs, and the parent `STATUS.md`. This scale-up supersedes the older unexecuted 16-instance/1,600-episode RoboCasa budget, not historical measurements or data-isolation requirements. Always inspect current HEAD and STATUS before assuming work remains.

Verified starting point at e40dbdbf: one DEV target-only RGB-D/TRELLIS reconstruction, native oracle room/destination, uniform-color native rendering. Five native episodes ran, four succeeded; only ONE reconstructed pair ran and succeeded, reference/reconstruction 392/429 ticks. A 392-action reconstructed replay failed a terminal retreat condition. The native import replay was identical across all 392 steps. No broad retention, B3/B4 benefit, Gaussian observation or whole-room result follows from that one pair. Preserve every original artifact and the failed native seed.

## Start now: reuse and measurable progress

1. Inspect code, private data/checkpoint availability, existing native environment, canonical bundles, receipts, GPU allocation/quotas and queue. Do not download all datasets, reinstall working environments, cancel unrelated jobs or rerun old completed cohorts.
2. Create/update this directory's `EXECUTION_STATUS.md` with owners, actual commits, job IDs, next artifacts and blockers. Read existing tests/CLIs; proposed matrix commands in the plan are interfaces to implement, not proof code already exists.
3. Use one worktree/branch per task. N0 owns shared schema/identity, N1 scheduling/cohort, N2 candidates, N3 verification, N4 importers, N5 rollout semantics, N6 GS, N7 tables/demo. One owner per shared file. Merge small tested changes without force-pushing or overwriting other work.
4. Deliver the next REAL result first: finish missing first-pilot reconstructed pairings when canonical asset/capture identities permit, including failed reference seeds. Capture a new instance if its objects/context differ. Then run the 8-independent-instance DEV expansion: 2 layouts x 2 tasks x 2 instances x 5 resets = 40 units/arm, REF_NATIVE and B0_FIXED_NATIVE first.
5. Do not wait for BEHAVIOR, Cosmos/Harmonizer, a learned risk classifier, complete room-GS or every optional control before running those native DEV comparisons.

## Repair only validity-critical contracts before scale

Reuse `robo.roundtrip` and `robo.eval.harness_runner.run_native_episode`, the existing policy server, orchestrator, manifest stack, metrics and paper pipeline. Do not create another incompatible ledger or rewrite the project.

N0 must add a versioned schema preserving the current DEV-only protocol. Distinguish canonical instance, reset perturbation and policy RNG. Bind XML/asset closure, task metadata, robot/controller/integration state and sensor timing. Seed equality alone does not define the same scene. Preserve estimated initial object placement: apply the same world-frame perturbation to reference and estimated reconstruction, never reset reconstruction to GT pose.

Validate native success semantics after import. The current sink task uses body-origin containment and a 25cm gripper-distance condition; a generated origin must not silently make the task easier. Test physically identical geometry re-expressed in another local frame, preserve scorer/reference-point meaning, and retain independent contact/containment diagnostics. Do not change native thresholds or infer whole-mesh containment from a body-origin predicate.

The current relative-marker RMSE is not absolute pose error. Establish a fixed evaluator-only correspondence before absolute metrics; never align each frame or correct rollout state using GT. Otherwise keep absolute metrics null and label relative diagnostics explicitly.

Test concurrent policy requests, restart/resume and per-client chunk RNG. Keep native camera/proprioception/action preprocessing and checkpoint fixed. Native task horizons come from the pinned version, not a universal 600-step assumption.

## Execute the method and scale

N1 captures static public TRAIN RGB-D/poses with hidden native geometry/state excluded from construction. Freeze independent DEV/TEST layout/object identities and capture rosters. Record overlap with policy training separately; reconstruction TEST is not automatically policy-unseen.

N2 generates one shared TRELLIS/RVG candidate pool per canonical instance. B0 uses fixed TRELLIS; B3 uses existing evidence selection plus bounded registration retry. Preserve cheap B1/B2 geometry diagnostics. Reuse candidate/collision assets across methods and resets when hashes match, and report logical per-method compute attribution honestly. Add the same-input observed-surface/TSDF baseline initially for geometry/replay.

N4 expands import from target-only L0 to target+destination L1 and actual task-workspace L2. Preserve native robot/controller/evaluator and correct role/contact/bounds/state mappings. Reconstruct container openings and relevant supports; do not copy GT collision into a supposedly reconstructed workspace. Every result lists retained oracle context. Large numbers of L0 episodes do not establish room reconstruction.

N3 implements deployment-matched Task-Conditioned System Verification on actual native CoACD, estimated placement and support/contacts, not the old isolated convex-hull probe. B4 may take at most TWO extra actions per dependency from the frozen observation-supported registration, bounded support placement, collision regeneration or candidate-reselection bank. Update descendants and freeze repaired canonical assets before TEST policy outcomes. No welding, obstacle removal, GT asset substitution, test-threshold change or outcome-driven retries.

BM uses the SAME extra action bank, maximum calls and final selection/acceptance as B4 but a predeclared fixed order without failure-specific scheduling. Report actual wall/CPU/GPU cost; maximum-budget matching is not exact FLOP equality. V1 verification/abstention-only is an optional separate arm and is required for a direct reject-vs-repair policy claim.

After real DEV and relevant identity/import tests pass, freeze the TEST protocol:
- 8 layouts x 3 native task families x 2 independent instances = 48 builds.
- 10 paired resets/instance = 480 units/arm.
- REF_NATIVE, B0_FIXED_NATIVE, B3_AGENT_NATIVE, B4_ROOM_REPAIR_NATIVE, BM_BUDGET_MATCHED_NATIVE = 2,400 primary episodes, one policy, L0 oracle context.
- Predeclared 4-layout subset: 240 units. B3/B4 at L1 and L2 add 960 episodes. Reuse exact matching L0/REF records.
- Optional B4_GS at a DEV-selected scope of that subset adds 240 episodes. Native and GS differ ONLY in observation; validate depth-aware robot/world occlusion and shared live state.
These are prospective budgets, not targets to hit by deleting failures. Resolve exact compatible task IDs before TEST; no task replacement after reconstruction outcomes.

N5 additionally fixes the feedback diagnostic: generate full-H reference actions with a predeclared continue-after-success rule, then replay those SAME H actions without padding, replanning or object-pose playback. Compare reconstructed feedback with the same H in a separately labeled protocol. Select 2 reset IDs per 48 TEST instances before outcomes: 96 diagnostic units, up to 96 new reference action-generation episodes and 480 physics replay traces. Keep this separate from primary native termination and the old 392-action prefix. Record per-tick native predicates, first success and success-at-H distinctly.

## Avoid another all-invalid, never-executed benchmark

Mandatory load/schema/nonfinite/numerical-safety failures halt the affected unit and remain in coverage. A finite drifting/falling object, low confidence or poor expected grasp is a possible scientific negative, not a blanket ban on baseline execution. Do not impose B4's conservative verifier on REF/B0/B3. Explicit method abstentions remain nonexecuted outcomes with coverage loss. Externally blocked/unscheduled jobs remain unmeasured, not invented zero-success episodes.

Engineering admission means the measurement is valid, not the method succeeds. If B4 loses, finish the valid experiment and report it. Fix independently diagnosed software defects in a NEW version; never silently change TEST cohorts, thresholds, action budgets or success rules to obtain a positive conclusion.

## Maximize throughput without sacrificing comparability

Use ordinary independent Slurm jobs ONLY, no arrays. Bound concurrency by actual memory/quota measurements. Keep large models warm and test request interleaving. Build once per canonical instance, replay many paired resets. Launch valid CPU metrics/import/assembly concurrently with GPU generation/policy. Workers write unique episode shards atomically; a single validated merger assembles the canonical ledger. Resume only missing units. Do not repeatedly regenerate hashes/reports instead of executing the next admissible experiment. Never claim an active job is complete without terminal receipts and actual outputs.

Freeze small source changes before full runs. Different stages may use different source commits with explicit compatible provenance; final release binds them. Do not rerun valid stages just to make all Git hashes look identical. Stay within existing permissions, quotas and known cost limits. Ask the user only for genuine missing access, new hardware/physical operation approval or expenditure outside authorization; continue other independent work meanwhile.

## Deliver results and conclude precisely

N7 extends the canonical paper pipeline for four table groups: fidelity/coverage/cost; native paired manipulation; verification/repair/compute control; replacement-scope effects. Use standard PSNR/SSIM/LPIPS/CD/F1, native success and coverage, justified pose errors and time. No composite simulator score. Cluster paired uncertainty by layout, then instance and reset; repeated resets are not independent reconstructions. Report native reference difficulty, success/planned, executed/planned, success/executed and signed differences. A nonsignificant difference is not preservation; require a predeclared margin for that claim.

Maintain separate conclusions for feasibility, multi-instance usability, B3/B4 gains, benefit beyond compute/rejection, equal-horizon feedback effects, GS observations and room scope. Do not infer one from another. Do not silently rename the user-approved paper title. Preserve old ScanNet++ failures while adding new native evidence under its actual scope.

Produce an early honest ~30s progress video from real multi-instance episodes; then final 80-95s hero/30s teaser/8s loop if required source footage exists. Show real selection/repair, continuous episodes, actual native predicates and exact replacement scope. Use same-state traces for synchronized native/GS rendering; never splice divergent rollouts into a fake success. Result cards read the exact release JSON. No restricted assets/checkpoints/font files in distributable bundles.

## Reporting and completion

Every task STATUS records implementation, smoke, real pilot, full results, precise commands, code/config/checkpoint hashes, job IDs, outputs, failures and next action. Update the shared board at meaningful milestones. Do not mark a task DONE merely because documentation or synthetic tests pass.

Each progress response states: completed actual artifacts; running jobs and expected outputs; measured preliminary/frozen results with sample counts; blockers; scientific claims supported/unmeasured; next critical-path actions. Final handoff includes source-bound tables, all planned terminal units or explicit incomplete blocks, confidence intervals, claim ledger, videos, code/paper commits and reproducible launchers.

Start with current-source verification, N0/N1/N7 in parallel, and release the first legitimate native DEV pair as soon as its local integrity checks pass. The immediate objective is measured multi-instance robot behavior, not another large status document.
