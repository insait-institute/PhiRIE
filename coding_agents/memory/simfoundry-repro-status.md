---
name: simfoundry-repro-status
description: "PhiRoom/SimFoundry: pipeline+factory+inpainting locations, val-fleet results, run recipes, env gotchas"
metadata: 
  node_type: memory
  type: project
  originSessionId: f802ca8c-b83c-4332-9543-40492bf0b816
  modified: 2026-07-23T11:16:22.512Z
---

Project renamed **PhiRoom**, repo github.com/RunyiYang/PhiRoom (pushed from
/group/worldcept/code/SimFoundry, .gitignore excludes outputs/third_party/.venv).
Report artifact: https://claude.ai/code/artifact/eba315ad-b62c-4309-b2db-4130e482d150
Showcase webpage (2026-07-22, post-audit numbers only, embedded demo loops):
https://claude.ai/code/artifact/6980fa88-85b6-414c-922c-ef6e0a53e627
(source: scratchpad page_template.html + assemble.py; disk-verified: the 18 RVG
collapses are mixed labels NOT all transparent; verify 14/16 fails = plastic
bottles obj_14/obj_16 in verify/verify_report.json; clean_background.ply exists
for only 4 scenes)

Pipelines (simfoundry/): zero-shot `run_scene.sh` (s0-s8, paper-faithful);
GT-driven factory `run_factory.sh` (prepare/refine_masks/s4/align/s6/report/
eval_render); inpainting `run_inpaint.sh` (prepare/masks/qwen/fill);
`fleet_val.sbatch` (50-scene array); viewer.py = multi-scene viser editor
(:8090, drag gizmos + PyBullet playback); aggregate_results.py -> val_summary.

Val-fleet FINAL (2026-07-13, all 50/50 scenes): 789 objects, yield 58% A+B
(76% excluding the library outlier scene 3f15a9266d whose 202 shelf books are
correctly rejected), F1@20mm 0.636, drop-stable 80% (365/457), render twin
28.65 dB vs SceneSplat bg 29.34 (SSIM .906/.909, LPIPS .159/.155), 9.2 A6000h
= 11 min/scene = 42 s/object (align+CoACD dominate, not TRELLIS).
Extrapolation: 1018 scenes/15.2k objects ~= 215 A6000h. User decision: stay
on the val set, do NOT run the full 1018 yet. Online viewer = viser share URL
(relay flaky; VS Code forward :8090 preferred); viser is SH0-only + browser
renderer, so it looks worse than the gsplat CUDA renders used for metrics.

Inpainting (new contribution, piloted on c50d2d1d42): removal = GT-surface
radius + aligned-asset radius + MULTI-VIEW MASK VOTE (essential for
transparent bottles - their splat gaussians are diffuse clouds far from
geometry; vote: center projects into >=2 view masks within AABB+15cm);
fill = MAD-robust support-plane disks + 500-iter differentiable gsplat refine
against per-frame COMPOSITED inpaint targets (per-object mask paste - never
last-writer-wins, else neighbor objects get baked in). Backends: Qwen-Image-Edit-2509 beats LaMa on flat structured surfaces
(keyboard desk 23.6->33.0 dB hole PSNR; mean 21.6 vs 21.2 over 31 views;
hard cases limited by residual gaussians). Qwen ran as 4 sharded A6000 jobs
(SIMF_SHARD=i/4, ~8 min/view sequential-offload). LaMa run archived in
inpaint_lama_v3/, Qwen in inpaint/; both have clean_background.ply (drop-in
Inria ply). Floor boxes (non-planar support) remain weak.

P0 FINAL (2026-07-15): AUTO mode all 50 scenes = 1,082 inst / 62% yield /
75% drop-stable (GT mode: 789 / 58% / 80%). Discovery baseline: ours beats
MaskClustering 0.605-vs-0.321 F1@IoU0.25 (same SAM3 masks). Removal
baseline: FlashSplat exact-gsplat reimpl, union IoU 0.567 (delta = our
transparency removal); needed zero-evidence->background fix for
object-centric views. ReconViaGen (VGGT-cond TRELLIS, multi-view) in V_mesh
slot: mean F1@20 0.713 vs TRELLIS 0.635 - wins big on opaque structured
objects, catastrophic on transparent bottle (0.014) -> planned HYBRID slot
selected by alignment chamfer. RVG integration gotchas: sys.path needs repo
root AND wheels/vggt; BiRefNet (ZhengPeng7/BiRefNet) + kornia needed;
low_vram path broken (move all modules to cuda manually); run() returns a
3-tuple; ~40s/object. dreamsim weights land in CWD/weights - gitignored.

HYBRID V_mesh DONE (2026-07-16, simfoundry/factory_hybrid.py +
hybrid_pilot.sbatch, job 626488 14:49 on 1xA6000): per non-rejected object,
existing TRELLIS asset vs fresh RVG asset, both registered with
factory_align, winner = lower s5_align.sym_score (symmetric clipped chamfer)
vs gt_points.ply. c50d2d1d42_factory: 16 objects, 9 RVG wins, mean F1@20
0.764->0.861 (RVG-only would be 0.707); all 3 transparent-bottle RVG
collapses (F1 0.007-0.044) correctly caught -> TRELLIS kept. Originals
snapshotted to obj_XX/trellis/ (marker .snapshot_done), RVG in obj_XX/rvg/,
decision in obj_XX/hybrid.json; canonical trellis_gs.ply/mesh/mesh_sim/
aligned.json/collision/object.urdf materialized from winner (URDF via s6
coacd_parts+write_urdf, physics.json reused, no VLM). aligned_all.json
refreshed (pre-hybrid copy: aligned_all_prehybrid.json). Gotchas: RVG fork's
Gaussian.save_ply(path) takes NO transform kwarg (writes raw canonical =
mesh frame; the pilot's transform=None call was the save failure); never
import s4_trellis inside the hybrid (its sys.path insert shadows RVG's
trellis pkg). Chamfer-vs-F1 disagreed only on obj_03/obj_09 (<=0.05 F1
cost). obj_07 mug is tier-C rejected but RVG pilot got 0.874 there - a
rescue pass over rejected objects could raise yield.

Branch `automated_image2sim` (pushed): GT-FREE variant - inputs only
images+poses+mesh+gaussians. auto_segment.py (sam3 env): SAM3 on every 12th
frame x 16 prompts -> mask pixels raycast onto mesh -> voxel-IoU union-find
merge (>=2-frame confirmation) -> auto_instances.npz (same vert_idx contract
as GT); common.load_instances() dispatches on SIMF_AUTO=1. Pilot: 25
instances found (GT mode: 18), tiers A15/B3/C7 = 72% yield, TRELLIS is
actually ~9 s/object on A6000 (fleet-confirmed). export_mjcf.py: MuJoCo MJCF
(CoACD parts as exact convex geoms; background = visual mesh + per-object
static micro-slabs at asset bottoms since MuJoCo convexifies collision
meshes) + isaac_manifest.json (per-object URDF + world pose for Isaac Lab).
MuJoCo 2s settle on pilot: 8/18 stable, no NaN; residual drift = tipping of
irregular assets + interpenetrating duplicate bottle instances (auto-NMS too
weak) - local-machine TODOs. sam3 env now has scipy/open3d/diffusers/
transformers, numpy upgraded to 2.x (ndarray.ptp removed - use np.ptp in
anything running there). IsaacGym untestable on cluster; deferred to user's
local box.

Hard-won env facts (beyond [[cluster-gpu-and-envs]]):
- batch/H200 nodes (msp3-*) DO NOT mount /group/worldcept nor shared /home ->
  H200 unusable for this project; efficiency table uses assumed 2.5x factor.
- Qwen-Image-Edit-2509: cached in HF hub; needs torch>=2.5 (enable_gqa) ->
  run via sam3 env (torch 2.10, diffusers 0.39 installed there); on A6000
  needs enable_sequential_cpu_offload (>60GB otherwise). venv diffusers
  pinned 0.36.0 (0.39 breaks on torch 2.4.1). Use QwenImageEditPlusPipeline
  (2509 = Edit-Plus checkpoint; EditInpaintPipeline is wrong conditioning).
- LaMa fallback: big-lama.pt prefetched in ~/.cache/torch/hub/checkpoints;
  SimpleLama(device='cpu') to avoid fighting resident Qwen weights.
- debug QoS: max 4 concurrent GPU jobs per user (AssocGrpGRES).
- Library scene 3f15a9266d (202 books) needs ~3h, timed out at 1h in fleet.

## 2026-07-17 update: hybrid V_mesh + demo film
- Furniture DESCOPED per user: SIMF_FULL=1 opt-in only; main protocol = 54-cat
  small objects. Paper reframed (furniture = preliminary extension section).
- Hybrid V_mesh (factory_hybrid.py, fleet_hybrid.sbatch): dual-generate
  TRELLIS + RVG, register both with factory_align, winner = lower symmetric
  clipped chamfer (s5_align.sym_score vs gt_points.ply). Canonical files
  atomically replaced (snapshots in obj_XX/trellis/, RVG in obj_XX/rvg/,
  verdict in hybrid.json; per-scene objects/hybrid_all.json has rows[] with
  nested trellis/rvg dicts, f1_20 keys). FLEET RESULT (45 scenes/457 obj):
  F1@20 0.708->0.783, 53% RVG wins, 18/18 RVG collapses (<0.1) caught by
  chamfer gate, 54 s/obj, 6.8 A6000h. Paper abstract+SecV updated.
- Demo film (demo_movie.py: scan/discover/interact-sim/interact-render/
  assemble; storyboard docs/DEMO_STORYBOARD.md): v2 = 61s in artifact.
  Scenes: scan 578511c8a9/c50d2d1d42/f3d64c30f8 (d755b3d9d8 dropped: magenta
  floaters in-splat, color-gamut cull can't catch them); discover+interact
  c50; interact 578 (inpaint ran 2026-07-16, hole PSNR 27.1 dB; Qwen stage
  needs SIMF_QWEN=force on A6000 and ~5h > 4h debug wall -> resume works via
  per-view inpainted_k.png cache).
- Demo curation: SIMF_DEMO_F1MIN (default 0.70) + SIMF_DEMO_EXCLUDE=idx list;
  excluded objects get their ORIGINAL splat gaussians restored (covers both
  bad assets and removal residue). c50 corner = stacked boxes obj_05/06.
- Interact camera lessons: (1) lerping between two DSLR positions can pass
  behind furniture -> use ONE sight-checked position (mesh raycast in
  interact-sim, open3d only in venv) + 0.35m dolly + EMA lookat gaze;
  (2) fixed-Newton shoves launch light objects out of the scene (120N on a
  mouse ~ dv 1000 m/s) -> impulse scaled by mass (dv~2.7 m/s); gaze freezes
  if actor escapes >3m or falls below floor.

## 2026-07-21 AUDIT + FIXES: most pre-audit numbers above are SUPERSEDED
159-agent audit -> 70 confirmed issues (artifact bcf06951-9ab6-40e4-ac04-
b059d98f17ec); fixes on automated_image2sim (ca38ff2, 4e53221, 76b3c10,
1b07732). Corrections: F1 now POOLED per-object (macro dragged by 5 zero-
yield scenes at 0.0): GT 0.783 post-hybrid / 0.708 single-view; auto 0.582
vs matched GT (eval_vs_gt.py, 381/671 match; 0.630 was vs-own-extraction).
Timing sum-semantics: GT 9.7 A6000h/11.6min; AUTO 14.1h/17.0min (old
11min/45x was GT-only); speedups 43x/29x. Drop-stable post-hybrid link-
frame: 77.2%/75.4% (old 80% = pre-hybrid assets + COM bug; redrop.py).
Row C never set SIMF_AUTO=1 -> was GT-seg+derived (relabeled C_GT
678/49/56.0%/0.693); true SAM3+derived rerun = outputs/*_rowC2 (07-21).
MC "2x" was a frame-density confound: matched 28-frame MC F1@0.25=0.571
(ours 0.605), F1@0.5=0.286 (ours 0.419) -> honest 1.5x at IoU0.5 only.
Verify (held views region-localized): 14/16 pass, both fails transparent
bottles; boxes alpha 0.83/0.86 = box-scale-hole fill limit. FlashSplat vs
TRUE final removal set = 0.649 (old 0.567 = stale pre-vote ref).
Official-split renders (render_metrics_v2; old heldout = 7/8 TRAINING
frames): factory bg 28.89/twin 28.32, auto twin 27.52; clean-bg composite
pilot 28.39 vs 27.55 clean-only. B1K strict matching (data vendored in
third_party/bddl_data): 2.3% vocab-complete, 0.6%=6 groundable, coverage
24.9%/9.8%. SAM3 vocab = 16 prompts (+14 furniture), NOT 54/93 (54 = GT
whitelist aliases). Qwen-vs-LaMa: keyboard 34.3 vs 23.9dB, elsewhere 19.4
vs 20.5 (not "ties"); old 23.6->55.0 was actually backend cmp on DSC01714.

## 2026-07-19 finding: background PSNR != removal-hole-fill quality
Scene 27dd4da69e ranks #1/50 on background render PSNR (34.4dB) but its
per-object hole-fill PSNR (inpaint/fill_stats.json, joined via
inpaint/obj_XX/views.json frame lists) is terrible: 10/12 objects fail a
15dB min-across-views gate, because the scene is full of large objects
(backpack/box/bag/tray) whose removal holes exceed the support-plane
fill's design scale - matches the known furniture-scale weak point.
Background PSNR measures original-splat render fidelity; it says nothing
about post-removal hole quality - don't use it alone to pick a demo/eval
scene for the removal+inpaint pipeline. demo_movie.py's demo_ok() now
gates on BOTH asset registration F1 (SIMF_DEMO_F1MIN, default 0.70) AND
per-object hole-fill PSNR (SIMF_DEMO_HOLEMIN, default 15.0, via
_hole_psnr_by_object() - min not mean, since one bad supervision frame is
enough to show a blotch). Caveat: fill_stats.json is per-FRAME not
per-object (one composited hole-PSNR per supervision view, shared across
whatever objects that frame supervises), so the gate can over-exclude an
otherwise-fine object that merely shares a frame with a bad neighbor -
accepted as a conservative heuristic for demo curation, not a precise
per-object attribution.
Demo film v3 (53s, in artifact eba315ad-b62c-4309-b2db-4130e482d150):
2x2 grid opening (4 scenes: 45b0dac5e3/f3d64c30f8/27dd4da69e/825d228aec,
staggered scan reveal) + full 1752x1168 render res (SIMF_DEMO_SCALE env,
was 876x584) + close-up per-beat interact cameras (c50d2d1d42 + bathroom
scene 45b0dac5e3, PSNR 34.1dB/#4). Camera lessons added this round:
(1) per-beat-kind distance ranges (carry 1.6-3.2m, throw 1.2-2.6m, push
1.0-2.2m) beat the one-size-fits-all range; (2) center-ray sight check
alone lets a shelf/cabinet right above the camera fill most of the frame
even though the direct line to the target is clear - fixed with a
5-ray frustum sample (center + 4 offset by 0.5x direction in a
right/up basis), require ALL rays clear past 0.55x distance.

## 2026-07-22: single-image SHARP splat -> per-object hybrid asset pipeline (in progress)
User request: after [[sharp-single-image-feedforward]]'s DSC08561 desk demo,
run per-object generation + physics annotation on that single splat (no real
scan/mesh/multi-view). Investigated whether ReconViaGen's SimFoundry wrapper
(s4_reconviagen.collect_views) could run directly on it: NO -
collect_views() is 100% dependent on real COLMAP poses across the whole DSLR
trajectory + a real/derived scene mesh (occlusion masking); RVG's own model
(TrellisVGGTTo3DPipeline.run()) has no hard view-count/pose requirement, but
SimFoundry's crop-selection machinery around it does. s4_trellis.py (plain
TRELLIS) is the only V_mesh-slot module with zero mesh/pose/multi-view
dependency anywhere in its code - confirmed by full reads of
s1_segment.py/s3_lift.py/factory_prepare.py/auto_segment.py/s5_align.py/
factory_align.py/s6_physics.py/factory_hybrid.py/common.py.
User's decision (asked via AskUserQuestion, 3 options): TRELLIS-only,
TRELLIS+RVG-hybrid-with-synthetic-views, or RVG-only -> chose the HYBRID
option - feed each object BOTH plain TRELLIS and RVG (conditioned on N=12
synthetic near-field views rendered from the SHARP splat itself, staying
within SHARP's own validated small-disparity regime - NOT a wide orbit),
reuse factory_hybrid.py's existing chamfer-based winner selection verbatim.
Key enabling fact: SHARP's predict_image() unprojects with extrinsics=eye(4)
- the input photo's own camera pose in the splat's frame IS the identity
matrix, and the ply's own "intrinsic"/"image_size" elements store the exact
K/W/H used (read directly, don't recompute from EXIF). This makes
common.render_view(gs, w2c=eye(4), K, W, H, "RGB+ED") + common.unproject()
a drop-in substitute for s3_lift.py's mesh-calibrated-monodepth path, using
the mini-viewer env (gsplat) for rendering and the main SimFoundry .venv
(open3d/trimesh/coacd/xformers - confirmed all present) for everything else
except SAM3 segmentation (sam3 env, s1_segment.py reused verbatim, fully
generic already). New files: sharp_ply_meta.py, sharp_render_depth.py,
sharp_s2_lift.py, sharp_render_object_views.py, sharp_hybrid.py (all in
simfoundry/); s4_trellis.py/s5_align.py/s6_physics.py/factory_hybrid.py
helper functions (make_align_record, sample_mesh, registered_residual,
materialize_rvg/trellis, snapshot_trellis, etc.) reused UNMODIFIED. Known
caveat baked into the design: registration "F1@20mm/40mm" numbers in this
scenario are fit-to-own-single-view-partial-cloud, not true GT accuracy -
relabel as "fit@" not "f1@" in any report.

### RESULT (2026-07-22, DSC08561 desk scene, /group/worldcept/code/SimFoundry/outputs/DSC08561_sharp/)
First-pass run: SAM3 found 15 objects, 13 survived the lift gate, but only
5 survived s5_align.py's size-sanity gate (8/13 wrongly rejected at
2.6x-33.7x inflated scale). Root cause: align_object's `rz(yaw)` +
longest-axis scale-matching hard-assumes a z-up world (matches ScanNet++
mesh convention, per common.py's own "World frame everywhere: z-up"
comment) but SHARP's points.ply is in OpenCV camera frame (z=depth, not
up) - reusing z-up-authored alignment code on camera-frame data silently
mismatches which axis is "up" for anything whose true thin axis actually
is vertical (keyboard, pens, paper, glass). Fix (applied only in our new
bridge code, s5_align.py itself untouched): a FIXED rotation
`R_ZUP = [[1,0,0],[0,0,1],[0,-1,0]]` (x_w=x_c, y_w=z_c, z_w=-y_c; verified
orthogonal det=+1) applied to points.ply ONLY when writing it in
sharp_s2_lift.py (and un-applied again in sharp_render_object_views.py,
which also reads points.ply for its own projection math - not just
meta.json's centroid, a scope-widening the implementing agent caught
beyond the literal instruction). This assumes a roughly level/non-tilted
photo (true for DSC08561); a real fix for a tilted photo would need
vanishing-point-based up-estimation, not a fixed permutation.
POST-FIX: all 8 wrongly-rejected objects recovered, ratios now 0.83-1.20
(was 2.6-33.7x); bonus - a spurious window-edge "paper" detection that had
WRONGLY PASSED pre-fix now correctly fails (ratio 0.29), i.e. the fix
fixed a false positive too, not just false negatives. FINAL: 12 objects
reached full done status (URDF+physics+collision), 11 real (both monitors,
keyboard, mouse, bottle, glass, 2 pens, 2 papers) + 1 known-spurious
(obj_12, a glass mislabeled off a door edge, kept flagged not hidden).
RVG won 4/11 hybrid comparisons (mouse, bottle, both papers) - up from
1/4 pre-fix, now that the registration target itself is correct; mean
sym_chamfer trellis 19.8mm vs rvg 24.1mm. Total incremental rerun (fix
only, reusing cached SAM3/depth-render/TRELLIS-mesh/RVG-view-crop outputs
where valid) ~20.5 min.
Gotcha hit mid-rerun: factory_hybrid.snapshot_trellis()'s one-time
`.snapshot_done` marker had already fired during the FIRST (pre-fix,
wrong-rejection) hybrid pass for the 8 objects that got
snapshotted-then-immediately-rejected - so a naive rerun silently kept
serving the stale pre-fix aligned.json out of obj_XX/trellis/ even after
the points.ply fix, showing old wrong-frame rejection reasons verbatim.
Fix: delete the stale obj_XX/trellis/ snapshot dirs before rerunning
hybrid (safe - fully regenerated by the pipeline itself, confirmed
identical content/fresh timestamps after rerun; this is scoped to
internal cache dirs only, not source data).
Process note: the implementing background agent twice ended its turn
assuming an armed background wait would "notify" it later - it doesn't,
for a normal agent turn boundary. Fixed by switching to literal
foreground-blocking `while kill -0 $PID; do sleep N; done` tied to a real
PID for every subsequent wait. Worth remembering for any future
long-running delegated agent task on this cluster.

### Physics sim video (2026-07-23)
User asked for an actual video of the 11 real objects (skip obj_12
spurious) running in a physics sim, as a sanity check on the generated
URDF/mass/friction values. Used `export_mjcf.py` + the repo's existing
`mujoco_video.py` (already has camera auto-framing + offscreen render +
mp4 write + a built-in "lift 3 heaviest objects and drop" stress test) -
NOT `s7_sim.py` (PyBullet), because export_mjcf.py's scene-building has
zero ScanNet++/GT dependency (unlike s7_sim.py's carved real-scan
background), a direct fit for this no-real-mesh scenario. Two small
general fixes needed in export_mjcf.py (not scene-specific): floor plane
was hardcoded z=0 (only valid for real ScanNet++ scenes) -> changed to
min(objects' true bottoms)-5cm; one CoACD hull (obj_03/bottle) had only 3
vertices, MuJoCo's compiler rejects <4 -> added a filter dropping
degenerate hulls before MJCF emission.
RESULT: valid 1280x720/30fps/8.0s mp4, no NaN/explosion/tunneling, but
0/11 objects stable within the script's 3cm/2s threshold (drift
54-276mm, median 159mm) - real moderate repositioning, not a crash.
Root causes (sim-setup, not asset-quality): (1) each object's "support"
is its own tiny isolated floating-island slab sized to just that
object's footprint - no real continuous desk, so any object that drifts
a few cm slides off its own island and free-falls further, unlike a real
desk that would catch it; (2) R_ZUP is a fixed axis-permutation
approximation of "up," not calibrated gravity, so flat/thin objects
(papers, pens) tip/slide a few degrees before finding equilibrium.
Results viewer artifact (video + all 11 object cards with mass/friction/
restitution/drift/hybrid-source, self-contained base64):
https://claude.ai/code/artifact/56ef0496-24b3-47db-b2a3-f539d1c20af5
Video/scene files: outputs/DSC08561_sharp/sim/settle_video.mp4,
outputs/DSC08561_sharp/sim_export/{scene.xml,mujoco_settle.json}.
