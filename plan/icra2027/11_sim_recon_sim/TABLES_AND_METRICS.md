# Tables, standard metrics and statistical contract

All cells below are deliberately empty. These are planned layouts, not results. Store unavailable values as null with status/reason; distinguish not-applicable (e.g. reference-to-itself reconstruction error) from not measured. Generate final LaTeX through the existing paper pipeline. Reference simulator is ground truth/environment, not a competing reconstruction method.

## T1 — Reconstruction fidelity and usable-build coverage

Question: how faithfully do observed inputs become independently usable assets? One table block per platform, sensor regime and replacement scope. Do not pool RGB and RGB-D, RoboCasa and BEHAVIOR, or target-only and full-room builds.

### T1a: room appearance

| Method | Rooms available/planned | TEST views available/planned | PSNR dB up | SSIM up | LPIPS down |
|---|---|---|---|---|---|
| Source Gaussian, before factorization | | | | | |
| Fixed B0 composite | | | | | |
| Agent B3 composite | | | | | |
| Room-verified/repaired B4 composite | | | | | |
| B4 + Harmonizer C (optional) | | | | | |

Native RGB is the target, not a row with infinite PSNR. Pair the camera roster across methods for conditional comparisons and report all missing views in coverage. Keep a method's full available-set summary separately to avoid confusing all-method intersection with population coverage.

### T1b: object geometry

| Method | Accepted/planned objects | Matched geometry n | CD cm down | F1@20mm up | Construction time/build down |
|---|---|---|---|---|---|
| B0 fixed TRELLIS | | | | | |
| B1 two proposals, fixed priority | | | | | |
| B2 evidence selection | | | | | |
| B3 registration retry | | | | | |
| B4 + room-context verification/repair | | | | | |
| BM fixed-order matched-budget actions | | | | | |

Use all planned objects for coverage; independent reference geometry only for matched objects. Report unmatched/duplicate/merged detection counts separately. No cherry-picked correspondence per method. Conditional geometry cannot imply global improvement at a different coverage.

**CD definition:** CD_cm = 100 * 0.5 * [mean_x min_y ||x-y||_2 + mean_y min_x ||y-x||_2] with x,y in meters, deterministic area-uniform samples, e.g. 20,000 points/object. This is the mean unsquared symmetric Euclidean convention, not squared CD or registration's clipped objective. Pin the definition and thresholds across every method. F1 uses precision/recall at 0.02m. Align only with the method's estimated world transform. A GT-aligned canonical-shape diagnostic is separate.

**Appearance:** standard PSNR, SSIM and pinned LPIPS/AlexNet on identical color/resolution inputs. Validate against a trusted pinned implementation. No silent resizing of mismatched outputs. Keep full-frame metrics primary. Optional object-masked PSNR/SSIM use the SAME independent mask and explicitly documented valid SSIM windows; ordinary LPIPS over a bounding-box crop is labeled crop LPIPS, not masked LPIPS. No per-image exposure fitting on TEST. Degenerate identical images have infinite PSNR and must be handled transparently, not clipped to a favorable finite score.

## T2 — Action-conditioned physical fidelity

Question: with perception/policy removed from the action decision, does reconstructed geometry preserve physical response?

| Environment | Completed/planned traces | Position RMSE cm down | Rotation error deg down | Final position error cm down | Native task completion % up |
|---|---|---|---|---|---|
| Native reference | | | | | |
| Native import identity control U1 | | | | | |
| B0 reconstructed | | | | | |
| B3 reconstructed | | | | | |
| B4 room repair | | | | | |

Errors are relative to native reference traces; reference self-error cells are not applicable. Position RMSE uses a common fixed object frame and matched control timestamps, not mismatched COM origins. Rotation error is geodesic SO(3) distance in degrees. Report absolute pose error and optionally displacement-from-start error separately. Crashed/incomplete traces have conditional-prefix errors with retained duration and coverage, never zero-padded tails. Native task completion is evaluated by unchanged native semantics on each environment.

This table is fixed-action replay, not learned visuomotor-policy success. Source reference action traces may themselves fail and remain in full counts. A reference-successful replay subset can be a labeled diagnostic, never the only denominator.

## T3 — Closed-loop native manipulation

Question: how much native task performance survives reconstruction?

| Environment/observation | Executed/planned | Surface-placement success % up | Container-placement success % up | Native success/planned % up | Success among executed % up | Absolute gap to REF, pp down |
|---|---|---|---|---|---|---|
| REF_NATIVE | | | | | | |
| FIXED_NATIVE | | | | | | |
| AGENT_NATIVE | | | | | | |
| ROOM_REPAIR_NATIVE | | | | | | |
| ROOM_REPAIR_GS | | | | | | |
| REF_GS_DIAGNOSTIC, oracle physics (optional) | | | | | | |
| ROOM_REPAIR_HC (optional) | | | | | | |

Use a separate block per policy/platform/scope. Task-family labels must correspond to real native IDs and their full native goals. Grasp/lift/place or native partial-goal completion goes into a small supplementary table only where the native evaluator defines it; do not invent weighted stages.

Unavailable constructed tasks contribute no successes to the end-to-end service rate, but remain explicitly nonexecuted in coverage. External blocked or unscheduled experiments are incomplete, not inferred zero-success outcomes. A complete table has a record for every planned unit. Publish native success and reference difficulty alongside gaps: two all-failure environments can have zero gap without useful fidelity.

Reference-vs-full gap is an end-to-end contrast across the complete changed asset bundle, not an isolated physical-parameter effect. B4_NATIVE vs B4_GS vs B4_HC is the isolated RGB comparison. Native visual geometry changes in B0/B3/B4, so their closed-loop difference alone does not isolate physics; T2 helps attribute it.

## T4 — What does system verification add?

| Construction controller | Objects accepted/planned | Room-stable/tested objects | Native successes/planned tasks | Extra tool calls/object | Algorithm-stage time/build |
|---|---|---|---|---|---|
| B3, logging only | | | | | |
| V1, verify and abstain, no repair | | | | | |
| B4, task-conditioned verify + repair | | | | | |
| BM, fixed-order same maximum action budget | | | | | |

B4's own verification pass rate is NOT independent evidence that it is physically accurate. Use native replay/task outcomes to test consequences. Report conservative rejection coverage and conditional rates together. Extra tool calls and runtime test the compute explanation. BM shares the action bank, maximum calls and final selection rule; report actual cost rather than claiming exact FLOP equality.

This is the only new core controller ablation beyond the existing B0-B3 study. A learned risk predictor is optional and is NOT required for this experimental track. Do not make the project wait for a new AUROC table.

## Supplementary controls

S1: U0/U1 native wrapper/import identity. S2: rigid replacement ladder (target, target+container, workspace, full room). S3: oracle pose / native physical parameters / native background, explicitly privileged diagnostics. S4: posed RGB-D vs posed RGB at the same roster. S5: per-task/per-scene outcome counts, failed builds, missed detections and collision/role-binding failures. S6: runtime and renderer/transport/policy p50/p95. S7: robot-core exactness, target IoU and motion-compensated temporal diagnostics for optional Harmonizer.

Avoid adding FID/KID or custom aggregate simulator scores. A small paired-view set is not a strong FID experiment. Robot-core byte equality is an engineering invariant, not a new headline learned metric.

## Aggregation and uncertainty

Primary unit for policy metrics: frozen native task instance; report task-macro success, with scene and family counts. Archive episode-micro results. Reconstruction appearance: scene-balanced means plus view-pooled supplement. Geometry: object means with scene-cluster uncertainty and matched counts. Never mix conditional subset means into an unconditional improvement claim.

Paired bootstrap uses 2,000 draws by default, seed 42, resampling layout/scene clusters then task instances and matched reset IDs together across arms. Intervals with too few independent scenes are labeled descriptive. Report percentage-point differences and discordant-success counts. Keep all predeclared contrasts; no significance-based subset selection. A nonsignificant difference is not proof of equivalence. Noninferiority/preservation requires a DEV-predeclared margin and suitable interval, otherwise say 'no clear difference detected'.

No numerical value is a target to hit. Negative results complete a valid experiment. Freeze budgets before TEST; power analysis can revise future protocols, not repeatedly extend the current test until significance.

## Machine-readable files

Keep distinct `build_records.jsonl`, `frame_metrics.csv`, `object_metrics.csv`, canonical `episode_ledger.jsonl`, `replay_metrics.csv`, `comparison_statistics.json`, `coverage.csv`, `failure_inventory.csv`, `runtime.jsonl`, and `table_provenance.json`. Each numeric output records metric version, source hashes, sample unit and numerator/denominator. Raw images/traces are not table cells. The final paper producer consumes these, and retains planned rows with null/unmeasured status instead of dropping methods.
