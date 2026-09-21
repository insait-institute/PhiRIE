# SimAny — ICRA paper notes (2026-07-14)

## Working title
**SimAny: Automated Image-to-Simulation for Real Indoor Scenes via
Gaussian-Native Scene Editing** (system: `automated_image2sim`)

## One-sentence pitch
From posed RGB images of a real room (plus its reconstruction: mesh +
Gaussian splat), fully automatically produce an *editable, photorealistic,
simulation-ready* digital twin - every manipulable object replaced by a
generated, physics-annotated asset, and the original object erased from the
background splat - exported to PyBullet / MuJoCo / Isaac.

## What is reproduction vs what is OURS

**Reproduced from SimFoundry (arXiv:2606.28276), re-implemented:** modular
slot design; TRELLIS in the V_mesh slot; CoACD collision; VLM physics
annotation (Qwen2.5-7B for Gemini); PyBullet velocity-zeroed settling; SAM3
segmentation slot.

**Novel contributions (ours):**
1. **GT-free 3D instance discovery on real scans** (`auto_segment`):
   multi-frame SAM3 + occlusion-aware raycast lifting onto the scene mesh +
   voxel union-find merging with cross-view confirmation. Replaces
   SimFoundry's single-frame Gemini detect/inpaint loop; needs no
   annotations, no proprietary VLM. Pilot: 25 instances discovered vs 18
   GT-whitelist; yield 72% vs 89% (the measurable "cost of autonomy").
2. **Gaussian-native object removal + background completion** (`inpaint_*`):
   (a) transparency-aware gaussian removal by MULTI-VIEW MASK VOTING
   (diffuse gaussian clouds of glass bottles are invisible to geometric
   proximity); (b) support-plane fill with normal-oriented disks; (c)
   diffusion-guided refinement (Qwen-Image-Edit-2509 / LaMa) with
   PRIMARY-VIEW supervision (cross-frame diffusion outputs are mutually
   inconsistent - single-frame supervision removes the blotch artifact,
   keyboard-desk hole PSNR 23.6 -> 55.0 dB). Produces a drop-in
   `clean_background.ply` in ~10 min/scene, vs SimFoundry's inpaint-video +
   retrain-splat track (~90 min inpainting alone + splat training).
3. **Unattended quality control at dataset scale**: F1-tier gates
   (vs GT or vs own extraction), size-sanity rejection, automated drop
   tests - replaces SimFoundry's human-in-the-loop GUI. Enables the first
   (to verify) sim-ready object benchmark on a public scan dataset:
   **ScanNet++ val, 50 scenes, 789 instances**, F1@20mm 0.636 (A+B),
   yield 76% (excl. library outlier), 80% drop-stable, twin rendering
   -0.7 dB vs the SceneSplat reconstruction ceiling, 42 s/object A6000.
4. **Multi-simulator export + interactive editor**: URDF (PyBullet),
   MJCF (MuJoCo, convex-exact via CoACD parts, verified headless), Isaac
   Lab manifest; viser-based photoreal editor (drag objects in the splat,
   physics playback, original/clean background toggle).

## Evidence in hand (tables ready)
- 50-scene val: yield/F1/drop/render/efficiency (outputs/val_summary.json)
- registration robustness: synthetic suite (one-way chamfer fails 15-146%
  scale error -> symmetric-clipped 15/18 pass, <4% scale err)
- Qwen vs LaMa inpainting backends (hole PSNR, per-surface-type)
- GT vs automated enumeration (pilot; EXTEND to all 50 scenes - cheap)
- MuJoCo settle stats (pilot auto: 8/18 <3cm, no NaN)

## To run for the paper
1. auto mode on all 50 val scenes (~12 A6000h) -> GT-vs-auto table at scale
2. ablations: w/o SAM3 mask refine; w/o mask-vote removal; w/o best-view
   (rep-frame only = SimFoundry-style); Hunyuan3D in V_mesh slot
3. baselines: SAM3D (theirs: F1 0.66-0.71), TSDF-crop, GT-submesh oracle
4. background CoACD for MuJoCo (local machine) + a small pick-place policy
   demo in MuJoCo for the "usable for robot learning" claim (optional but
   strong for ICRA)
5. duplicate-instance NMS improvement in auto_segment (bottles)

## Honest limitations (write them)
- factory mode needs mesh+splat reconstruction (from the dataset); not
  monocular; zero-shot single-frame mode exists but degrades
- non-planar support (floor boxes) fill is weak; deformables as rigid
- no real-robot validation; sim-eval-correlation cited from SimFoundry
- ScanNet++ license: assets not redistributable

---
# Novelty audit vs literature (2026-07-15, 4-axis web scout)

## DEAD as standalone claims (do not put in contribution bullets)
- Physics-driven gaussian rendering (our C6): SplatSim (ICRA'25), RoboGSim,
  Vid2Sim, and NVIDIA NuRec/3DGRUT productized in Isaac Sim 5/6. Keep as
  infrastructure only.
- segment->generate->register->URDF recipe: SimFoundry, SceneComplete
  (tabletop, 2024), GASE (arXiv 2606.17520 - SAM3+TRELLIS+LaMa+Isaac,
  NEAR-CLONE but semi-automatic, 9 scenes).
- auto_segment as a segmentation method: SAM3D'23, MaskClustering (CVPR24,
  same benchmark), Any3DIS (CVPR25), MV3DIS (CVPR26). Ours = "MaskClustering
  -style lifting with SAM3 + mesh raycast". Position as engineering + ablate
  vs MaskClustering.
- Primary-view supervision for 3DGS inpainting: InFusion'24 + >=5 papers.
- Multi-view mask voting for gaussian selection: FlashSplat (ECCV24) solves
  it OPTIMALLY - must cite + baseline.
- "First sim-ready ScanNet++": HoloScene (2510.05560; 3 scenes, 8.3 h/scene);
  MetaScenes (CVPR25; 706 ScanNet scenes but GT annotations + human ranking).

## SURVIVING claims (the paper)
1. TRANSPARENT-object gaussian removal via opacity-robust multi-view mask
   voting (only TRAN-D adjacent, different setting). Narrow, real, ablatable.
2. Support-plane normal-oriented disk fill tied to the metric mesh contact
   region (mechanism claim; geometry-aware init in general is known).
3. THE SYSTEM + SCALE claim: fully automatic (no GT masks, no user clicks -
   GASE needs prompts/clicks, MetaScenes needs GT+humans, HoloScene 8.3h/scene
   vs our 11 min = 45x), one training-free discovery stage feeding BOTH mesh
   assets AND gaussian-native editing, evaluated jointly (segmentation F1 vs
   GT + geometry F1 + drop tests + PSNR/SSIM/LPIPS) on all 50 ScanNet++ val
   scenes / 789 instances. Nobody has the no-GT + scale + unified-eval combo.
4. The evaluation protocol/benchmark itself as a released artifact.

## MUST READ FULLY before submission
SimRecon (arXiv 2603.02133 - possibly near-duplicate recipe!), HoloScene,
GASE, RoboPearls (ICCV25 - gaussian remove+LaMa-fill in robot sim), FlashSplat.

## Revised experiment list (priority)
P0 auto mode on all 50 val scenes (GT-free at-scale table; ~12 A6000h)
P0 discovery baseline: MaskClustering on our 789-instance protocol
P0 removal baseline/ablation: FlashSplat vs our voting; w/o transparency vote
P1 one policy result in MuJoCo (photoreal obs, pick/push, even scripted-BC)
   - every close competitor shows one; without it reviewers ding usability
P1 w/o SAM3-refine ablation; Hunyuan3D slot swap
P2 articulation = limitation (Articulate-Anything exists; don't attempt)

---
# FINAL numbers (2026-07-15, all P0 experiments landed)

## Auto (GT-free) vs GT-driven, all 50 ScanNet++ val scenes
| | GT mode | AUTO mode |
|---|---|---|
| instances | 789 | 1,082 (discovers more: dupes + extra classes) |
| yield A+B | 58% (76% excl library) | 62% |
| F1@20mm A+B | 0.636 (vs GT) | 0.560 (vs own extraction) |
| drop-stable | 80% (365/457) | 75% (506/671) |

## Instance discovery baseline (pilot scene, SAME SAM3 2D masks)
ours auto_segment vs MaskClustering(CVPR24): IoU@0.25 F1 0.605 vs 0.321;
IoU@0.5 0.419 vs 0.107 (MC over-segments: 38 pred vs our 25, GT 18).

## Gaussian removal baseline (pilot, 16 objects)
FlashSplat(ECCV24, exact gsplat reimpl; zero-evidence->background fix needed
for object-centric view sets): union IoU vs ours 0.567; per-object 0.23-0.25
(we additionally remove transparent-volume + asset-surface gaussians - the
delta IS the transparency contribution, visualize for the paper).

## V_mesh slot: ReconViaGen (multi-view, VGGT-cond) vs TRELLIS (single-view)
6 pilot objects, identical registration+eval: MEAN F1@20 0.713 vs 0.635
(box 0.799v0.564, mug 0.874v0.024, telephone 0.850v0.577, keyboard/mouse
tie), BUT transparent bottle 0.014 vs 0.817 - VGGT collapses on glass.
=> paper story: HYBRID slot - reconstruction-consistent generation (RVG)
for well-observed opaque objects, generative single-view prior (TRELLIS)
for transparent/low-confidence; select by alignment chamfer. ~40s/obj vs 9s.

## Removal verification suite (2026-07-15, supervision-independent)
verify_removal_{render,check}.py: (1) alpha coverage in hole (void check),
(2) rendered-depth vs support-plane (float/sink), (3) SAM3 residual
re-detection on clean-bg renders (before->after det score), (4) held-out
view renders (+ hook for VLM judge). Pilot: alpha all >0.95; depth-plane
12-26mm opaque vs 32-111mm bottles; RE-DETECTION 13/16 removed (all 3
failures = transparent bottles, det 0.16-0.32 residual). => quantitative
confirmation that transparency is THE remaining failure mode; enables
verification-in-the-loop (re-inpaint failures with more views/backend).
Paper: this battery replaces self-referential hole-PSNR as the primary
removal metric. VLM-judge (Claude/GPT API) slots in as check #5.
