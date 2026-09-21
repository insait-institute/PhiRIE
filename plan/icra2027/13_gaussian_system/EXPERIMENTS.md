# Finite experimental matrix and standard-metric tables

## Scientific sequence

First close the 40 missing F1 units through a fresh valid paired-block protocol and preserve original outcomes. New systems experiments use new source-bound cohorts; they do not retroactively convert the previous ideal-RGB-D/native-render results into video/GS evidence. Do not tune on the observed 48-instance TEST cohort and then call it unseen.

The campaign has three independent factors: construction, background completion, and online observation. Avoid a generator x inpainter x semantic encoder x renderer x policy x dataset Cartesian product. Compare each factor on shared intermediate inputs, select the system's default recipe using DEV only, then freeze the end-to-end benchmark.

## A. Official system baselines (main)

Methods: original native reference; official SimFoundry; official PolaRiS; complete frozen SimAnyRoom. Run both a declared common-input/budget track and each method's native recommended acquisition track when inputs differ. Human composition and additional scans must be reported. Unsupported zero-manual PolaRiS configurations are not fake zero-success official results.

Two evaluation levels:
1. Asset-controlled common importer: preserve method geometry and estimated placement, use declared common decomposition/physics assumptions, same native robot/controller/rubric. Suffix method labels `-adapted/common-importer`. Our asset bridge currently uses the existing box-inertia importer; no claim that this preserves each baseline's complete original dynamics.
2. Native end-to-end observation: retain each method's native GS/mesh observation and backend. Report within-backend matched-reference differences. Cross-engine raw success differences are not a clean construction effect. Select one compatible frozen policy per backend; do not compare different checkpoint training recipes as scene-construction novelty.

Initial smoke: 2 DEV RoboCasa instances + 1 BEHAVIOR native scene. Primary planned scale: 72 RoboCasa canonical builds x10 resets x4 methods =2880 logical episode units. BEHAVIOR:24x10x4=960 only after full native policy/rubric/import admission. Reference success, reconstruction failures, abstentions and missing external integrations remain distinct.

| Method / adaptation | Sensor + extra scans | Human min | Independent builds | Exec./planned | Success/planned | Success/executed | Paired gap vs native [95% CI] | Build min |
|---|---|---|---|---|---|---|---|---|
| Native reference | configuration | | | | | | | |
| Official SimFoundry | | | | | | | | |
| Official PolaRiS | | | | | | | | |
| SimAnyRoom | | | | | | | | |

Report per-task success and layout/instance-cluster paired intervals. A reconstructed scene that deletes obstacles may yield higher success but lower fidelity. Therefore inspect collision/context and report reference-task relation, not merely which success number is highest. No guarantee that our method wins.

## B. Generator comparison

Frozen backends: TRELLIS v1; TRELLIS.2-4B; official ReconViaGen v0.2. Optional RVG-v0.5 has a distinct TRELLIS.2 multiview codepath and model identity. Never rename v0.2 as v0.5. Use the same observed object inventory and upstream segmentation. Two conditions: one matched anchor per method where supported; each method's declared native view budget (RVG multi-view, single-view generators one anchor). Report view counts rather than hiding the information advantage.

DEV:48 objects x3 methods x3 predeclared seeds=432 proposals. Full component TEST:240 source-stratified objects x3 methods x1 frozen seed=720 proposals. Reuse each proposal for shared registration/CoACD/import and all reset conditions. Keep original backend-native texture/GS output AND common geometry-only import separately. No best-of-N choice using evaluator CD or policy success.

| Generator / inputs | Generated/planned | Registered/planned | F1@20mm | Symmetric CD cm | Masked PSNR | Crop SSIM | Crop LPIPS | Native success/planned | GPU min / peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| TRELLIS | | | | | | | | | |
| TRELLIS.2 | | | | | | | | | |
| ReconViaGen v0.2 | | | | | | | | | |
| Observed-surface baseline | | | | | | | | | |

Use fixed TRAIN-only registration and independent evaluation surfaces. Do not re-ICP to GT at evaluation. Declare non-squared/squared CD, sample count/seed and units exactly as existing fidelity evaluator. Mesh/PBR attractiveness is not physical validity. Container opening and support failures receive separate case analysis.

## C. Open vocabulary / Gaussian identity

Same TRAIN Gaussians, generators and budgets. Rows: current SAM3 lifting; frozen Chorus + 3D grouping; Chorus + same 2D boundary refinement. Source IDs persist through filtering, segmentation, movement and inpainting. Record the exact matching text encoder, prompts and pretraining overlap.

Use standard mIoU, instance AP at declared IoU thresholds, query localization accuracy, object discovery recall, build coverage and task success. Novel prompts/categories are held out from controller tuning, not automatically absent from foundation-model pretraining. Policy competence on new categories is evaluated, not inferred from a semantic heatmap.

## D. Background inpainting (construction time, NOT online Harmonizer)

DEV24 cases, TEST96 cases from source-selected room families. Models: Telea; SDXL inpaint; Qwen-Image-Edit-2511; Gemini2.5FlashImage (Nano Banana); Gemini3.1FlashImage (Nano Banana2). Optional LaMa or another official editor is separately named/admitted. Commercial APIs require approved image rights and spend limits. Unknown model aliases must be resolved before measurement.

Each case fixes original image, mask, prompt meaning, output canvas and number of calls. Save provider_raw and outside-mask-preserving composite. Qwen/Gemini receive an explicit mask-reference instruction, not falsely described as native binary-mask APIs. Enforce the same compositing rule for all.

| Editor | Cases returned/planned | Hole PSNR | Hole SSIM / declared crop SSIM | Hole/crop LPIPS | Raw outside-mask error | Post-GS heldout PSNR/SSIM/LPIPS | Wall s / API usage |
|---|---|---|---|---|---|---|---|
| Telea | | | | | | | |
| SDXL inpaint | | | | | | | |
| Qwen Edit2511 | | | | | | | |
| Nano Banana pinned model | | | | | | | |
| Nano Banana2 pinned model | | | | | | | |

Whole-image preserved-pixel error after compositing is trivially zero; report it as an invariant, not superiority. Also score RAW provider drift to expose altered outside pixels. Refit the SAME masked Gaussian background stage with each completion to measure multiview consistency. A good single image is insufficient. All targets must be clean renders at exactly matching state/camera or valid observed-surface holdouts. Real unobserved backgrounds have no fabricated PSNR GT.

## E. Online harmonization (same physics / state)

Rows: raw GS; TRAIN-fit affine color baseline; official temporal Harmonizer; official + existing robot restoration; 3D-state bounded correction; optional learned StateResidual16; optional semantic conditioning/3D-color-cache extension. The last extension remains a research task until implemented, trained and independently tested, not a completed feature.

Fixed buffers: positive camera-Z depth, visible persistent ID, robot mask, contact mask, confidence, normals, camera/rigid-object poses. Use actual deployed raw renderer, not hidden GT geometry. Multiview IDs and backward warp reject occlusions/disocclusions. Geometric correctness is not proved by copying a robot mask.

Train pairs use identical physical geometry/state/camera, varying appearance defects only. No divergent reference and reconstructed rollout matched by frame number. Train/dev/test split by room family. Evaluate with standard PSNR/SSIM/LPIPS on room and target, mask IoU, protected-pixel exactness, valid correspondence temporal error, p95 end-to-end latency, native success and missing-frame coverage.

Visual TEST:36 independent scene/task sequences, two camera types, predeclared views/states. Closed-loop budget:36x10 resets per observation arm; six arms=2160 logical units plus any fresh-reference controls required by process identity. This is not multiplied by every inpainting/generation combination. In synchronous evaluation, pause simulated time while waiting for observation and report wall latency. Never silently supply stale frames or an unlabelled raw fallback.

## F. Scope expansion

Target-only, target+destination, task-workspace, and full declared room are distinct rows. Include the actual recovered/retained geometry list. Use existing L0/L1 baselines as historical negative evidence, not claims of full room recovery. A drawable large room with one replaced mug is not a full-room manipulation experiment.

## Stop / selection rules

Run complete frozen quotas whether differences are positive, null or negative. Before TEST, DEV selects one default generator, inpainter, semantic front-end and visual recipe, with explicit compute budget. Do not stop merely when significance becomes positive. New method choices on old TEST make a follow-up exploratory; require independent new scenes for confirmatory claims. Unscheduled/gated/unsupported conditions stay missing or not applicable, not zero-success fabricated trials.
