# Experimental contract, budgets, metrics and conclusions

Prospective scale-up protocol. The parent protocol continues to govern data isolation. This document replaces the earlier unexecuted RoboCasa 16-instance/1,600-episode budget with the following versioned design. Nothing here changes recorded DEV or ScanNet++ outcomes. Defaults must be resolved on DEV and sealed before TEST.

## 1. Experimental unit and splits

A `canonical_instance_id` identifies actual native task/layout/style, object instances, canonical XML, asset closure, integration state, controller state, observable timing and task metadata. An integer seed is insufficient. A `reset_id` is a declared perturbation of that SAME instance; changed native asset identities require a new instance/build. A `policy_rng_seed` is a separate paired random stream. Reconstruction seed 42 is initially fixed; this is not multi-seed generation evidence.

DEV: 2 layouts x 2 tasks x 2 independent object/context instances = 8 builds, 5 resets each = 40 units/arm. Core TEST: 8 layouts x 3 tasks x 2 instances = 48 independent builds, 10 resets each = 480 units/arm. Actual dataset IDs are resolved from the pinned checkout, not assumed from current online examples. Candidate rigid tasks include `PickPlaceCounterToSink`, `PickPlaceSinkToCounter`, and `PickPlaceCounterToCabinet`; confirm native goals, open/closed fixture state and frozen-policy compatibility. If the third requires unsupported articulation, choose a compatible native rigid task BEFORE TEST and record the scope change. Do not replace it after seeing poor reconstruction success.

Separate controller DEV/TEST generalization from policy pretrain/target dataset splits. Record layout and asset overlap with policy training when known; otherwise mark unknown. Never equate a held-out reconstruction layout with a policy-unseen layout. Freeze native-valid task eligibility independent of reconstruction performance or TEST policy success. Keep original reference failures in results; do not sample only successful reference episodes.

The baseline five-reference/one-reconstruction DEV milestone is not 5 paired reconstructions. Additional pairings require compatible canonical capture/asset identity; capture a new instance when needed. Preserve the original failed seed and every prior artifact.

## 2. Treatment registry

| ID | Meaning | State/appearance contract |
|---|---|---|
| REF_NATIVE | Native reference | Native assets, physics, renderer and evaluator |
| B0_FIXED_NATIVE | Single fixed TRELLIS proposal | Reconstructed asset bundle, native renderer |
| B3_AGENT_NATIVE | Shared TRELLIS/RVG pool + evidence selection + bounded registration retry | Same initial captures/pool as relevant controls |
| B4_ROOM_REPAIR_NATIVE | B3 + task-conditioned native-context verification and at most 2 extra actions per dependency | Measured native collision, frozen repair bank |
| BM_BUDGET_MATCHED_NATIVE | B3 + fixed ordering of same extra action bank/max budget | Same final selection/acceptance rule as B4, no failure-dependent scheduling |
| V1_VERIFY_ONLY_NATIVE | B3 + same verification and abstention, no repairs | Optional 480-episode arm, needed for a direct repair-vs-rejection claim |
| B4_GS | Identical B4 physics with GS observations | Optional observation-only arm |
| B4_HC | Identical B4 physics with GS + real Harmonizer Option C | Optional after real-model preservation tests |
| OBSERVED_SURFACE_NATIVE | RGB-D TSDF/observed surface without generative completion | Cheap geometry/replay control, not initially a full policy arm |

Resolve material policy on DEV: either uniform-color meshes for ALL reconstructed native arms or equivalent frozen texture export for all. Native-reference differences are a complete asset-bundle comparison, not a pure geometry/physics intervention. Never give B4 better materials while calling B4-B3 a verification-only contrast.

B0/B1/B2/B3 geometry diagnostics reuse the same candidate pool; B1 fixed priority and B2 selection need not get new policy arms. Only B0/B3/B4/BM get the initial costly native-policy matrix. Preserve failed candidates and charge generation/retry costs logically per method even when physical computation is cached.

## 3. Replacement scope and budget

L0 target-only: native room and destination retained, explicitly `oracle_context=true`.
L1 target + destination: reconstruct the container or destination support surface. Do not delete the original enclosing fixture unless the declared scope includes it. Every retained component is listed.
L2 task workspace: target, destination, supports and relevant obstacles are reconstructed. Define the swept approach AND transport region; outside native context is disclosed and collision interactions outside the region recorded.
L3 room: declaration includes a finite room extent, all required static collision/appearance and discovered movable inventory. A cropped workspace or omitted failed object is not a complete room.

Core: 5 methods x 480 L0 units = 2,400 episodes, one policy. This estimates asset-level transfer under oracle context, NOT room reconstruction.
Scope subset: choose 4 TEST layouts by deterministic pre-metric rule, 3 tasks x 2 instances x 10 resets = 240 units. B3/B4 x L1/L2 = 960 additional episodes. Reuse L0 B3/B4 and REF observations ONLY when exact instance/reset/policy/contract hashes match; the scope report then has L0/L1/L2 with no redundant L0 run. Fixed budgets are not power guarantees.
GS block: B4_GS on one predeclared scope of the 240-unit subset adds 240 episodes. Select the intended scope on DEV (prefer L2); if unavailable, mark that block unavailable rather than silently substituting L0. V1 or B4_HC adds its explicitly budgeted arm. L3, second policy, RGB-only and BEHAVIOR have separate prospective budgets, never an automatic full factorial.

## 4. Sensor and privileged-data boundary

Primary input is static ideal RGB-D plus metric camera poses. Reconstruct from TRAIN only. Native object identity, original mesh/collision/inertia/mass, and evaluation outcomes live in an access-restricted evaluator process. Public instructions and the known robot/camera model are authorized inputs. Native role mapping occurs after construction is frozen and may not choose the most favorable generated shape using GT.

Use deterministic fixed capture trajectories selected on DEV for each scope, not the current single 6-view target scan for every workspace. Store per-view intrinsics/extrinsics, RGB/depth conventions and visibility. TEST cameras and surfaces cannot guide generation, registration, repair or exposure fitting. Primary quality uses declared common views with coverage; retain each method's available-set summary separately.

RGB-only is a separate condition: no depth-initialized point cloud, TSDF or GT-depth crop selection. Robot masking may use known embodiment rendered at the capture state, but may not erase task objects or declare a phantom support. L0 verification sees a native context, so call it oracle-context diagnostics even if asset construction itself is public-only. L2 repair must derive support corrections from reconstructed/observed surfaces, not hidden native geometry.

## 5. Paired reset and success semantics

Apply a common world-frame reset perturbation to reference and estimated reconstructed placement: `T_ref,k = Delta_k @ T_ref,0`, `T_recon,k = Delta_k @ T_est,0`. Do not set `T_recon,k = T_ref,k`, align to GT before rollout, or silently snap to a support. Robot/controller initialization and nonintervened components are identical. A declared support repair is a logged treatment, not a reset convenience. The reset protocol is public experiment scaffolding; no native asset shape enters construction.

Native task predicates and thresholds stay unchanged. Verify scorer object origins/sites, contact lists and bounds after import; a different generated body origin can alter inside/distance predicates despite identical physical placement. Require a representation-invariance test for physically identical geometry under a local-frame change. Either supply a justified scorer frame binding with unchanged physical geometry or scope native-success conclusions as binding-specific. Any additional containment/penetration diagnostic is separately named and cannot replace official success silently. Never loosen the 25cm retreat condition after the current DEV example.

Known native robot/cameras are provided by the benchmark. Do not make every task invalid because construction did not infer them. Mandatory load/schema/nonfinite-state failures terminate; finite drifting/falling objects may be valid negative measurement outcomes. Conservative verification failure does NOT block baseline execution globally. An explicitly abstaining arm retains its lost coverage.

## 6. Fixed-action vs closed-loop protocol

The current 392-action replay and 429-tick closed-loop success have unequal available horizons; they do not prove feedback is responsible. Create a NEW diagnostic protocol with the same full task horizon H. The reference policy continues to generate real commands to H after its first success; replay those H commands unchanged in each reconstructed arm. Freeze post-success/termination handling before runs. No zero padding, handcrafted retreat or replay replanning. Keep native binary predicates at every tick, first-success time and success-at-H separate. Primary native benchmark termination conventions remain unchanged and separately reported.

All replay object states arise from physics; no per-tick pose injection or trajectory alignment. Time steps, action interpolation, robot controller and sampling noise are pinned. Report full trace and prefix coverage. Absolute pose errors require a physically justified object-frame correspondence fixed once in a privileged evaluator. Never align per frame. Without correspondence leave absolute errors null and retain the existing explicitly relative marker diagnostic plus separately measured initial placement error. Symmetric-object orientation ambiguity is documented.

## 7. Metrics and denominators

T1 geometry: mean unsquared symmetric Euclidean CD in cm with fixed area samples, F1 at 20mm, accepted/planned and matched counts. Registration's clipped residual is NOT evaluation CD. No GT alignment of the primary geometry result.
T1 appearance: PSNR, standard SSIM, pinned LPIPS on common held-out frames at fixed resolution/color space. Report availability/planned and scene-balanced means. Native RGB is the target, not an infinite-PSNR baseline. Masked PSNR/SSIM are supplementary and use independently fixed masks/windows. No novel combined simulator score.
T2 replay: position RMSE/final position error in cm and SO(3) rotation error only with valid correspondence, plus native task completion and trace coverage. Keep relative and absolute columns distinct.
T3 policy: native successes/planned (end-to-end service rate on a completed roster), executed/planned, successes/executed, per-task family success, signed reconstructed-minus-reference difference and absolute gap in percentage points. A rejected build is no service success but not an executed policy failure. External pending/blocked/unscheduled units remain unmeasured, not inferred zero-success outcomes. Report reference difficulty; zero gap between all-failure arms is not retention.
T4 verification: independent native outcomes, room stability, accepted/planned tasks, extra tool actions and actual stage runtime. Its own check-pass count is not independent accuracy.

Primary aggregation is instance-macro; equal reset budgets also permit episode-micro supplementary rates. Bootstrap 2,000 draws/seed42 over layouts, instances within layouts, then matched resets jointly across arms. Do not treat 480 resets as 480 independent reconstructions or 3-camera frames as 3 independent episodes. Preserve declared contrasts and descriptive limitations with only 8 layouts. Report paired discordant success counts. No optional stopping for significance.

A nonsignificant difference is not preservation/equivalence. For a preservation claim, freeze a practically justified noninferiority margin on DEV and require an appropriate interval on TEST, acceptable coverage, and meaningful reference performance. Otherwise report the observed difference without declaring preservation.

## 8. Integrity vs outcome and publication

Checkpoints/configs, controller state, native instance and unchanged-asset closure are hashed. Reuse is based on relevant input/code/config identity, not whether two stage commits happen to match. New source versions generate new immutable stages; final release binds the stage DAG. One terminal episode record per `(cohort, instance, reset, policy, method, scope, sensor, renderer, protocol)`; workers write separate shards, one merger validates uniqueness/completeness.

Do not rewrite old negatives or promote this DEV pair to TEST. Synthetic tests validate software only. Code changes after TEST inspection require a declared new experiment version; preserve and report the previous run. Scientific failure can complete a protocol. Paper claims, however, require actual support and explicitly state sensor access, retained native context, replacement scope and policy identity.
