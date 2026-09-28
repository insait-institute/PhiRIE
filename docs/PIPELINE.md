# The SimAny generation pipeline

How posed RGB images + a scan mesh + a 3D Gaussian splat become a
simulation-ready digital twin, stage by stage. Environment details (why three
python envs exist and what runs where) are in [ENVIRONMENTS.md](ENVIRONMENTS.md);
audited results live in [CONTRIBUTIONS.md](CONTRIBUTIONS.md) and
[BASELINES.md](BASELINES.md).

Every stage is a python module run from the repo root (`python -m
agents.assets.s5_align`), normally via the helpers defined in
[`run/env.sh`](../run/env.sh): `run` (main `.venv`), `run_sam3` (sam3 env),
`run_gs` (mini-viewer/gsplat env), `run_qwen` (`QWEN_PY`, defaults to the main
venv). Scene selection and paths come from `SIMANY_*` environment variables
(`SIMANY_SCENE`, `SIMANY_OUT`, `SIMANY_SCANNETPP_ROOT`, `SIMANY_SPLATS_ROOT`,
...); the legacy `SIMF_*` prefix is still read with a deprecation warning.
`RESUME=1` makes the guarded early stages skip themselves when their output
already exists (stages wrapped in `done_skip`; the generation stages always
run, though s4 keeps its own mtime cache).

## 1. The four modes

### [`run/run_factory.sh`](../run/run_factory.sh) — GT-driven benchmark

Uses ScanNet++ GT instance annotations to enumerate objects (perception of
each object is still image-based). This is the paper's "GT-driven" column /
ablation row A. Output: `outputs/<scene>_factory`.

| # | stage | module | env helper |
|---|---|---|---|
| 1 | GT enumerate + best-view crops | `agents.discover.factory_prepare` | `run` |
| 2 | SAM3 mask refinement | `agents.discover.factory_refine_masks` | `run_sam3` |
| 3 | TRELLIS image-to-3D | `models.s4_trellis` | `run` |
| 4 | register to GT submesh + tiers | `agents.assets.factory_align` | `run` |
| 5 | CoACD + physics + URDF | `agents.assets.s6_physics` | `run` |
| 6 | drop test + yield report | `agents.eval.factory_report` | `run` |
| 7 | PSNR/SSIM/LPIPS on held-out views | `agents.eval.factory_eval_render` | `run_gs` |

### [`run/run_auto.sh`](../run/run_auto.sh) — fully automatic (no GT)

Sets `SIMANY_AUTO=1`. Posed images + mesh + gaussians in, MJCF + Isaac
manifest out; no semantic annotations anywhere. The paper's headline
"Automatic (no GT)" column / row B. Output: `outputs/<scene>_auto`.

| # | stage | module | env helper |
|---|---|---|---|
| 1 | SAM3 multi-frame discovery | `agents.discover.auto_segment` | `run_sam3` |
| 2 | instances + best-view crops | `agents.discover.factory_prepare` | `run` |
| 3 | SAM3 mask refinement | `agents.discover.factory_refine_masks` | `run_sam3` |
| 4 | TRELLIS image-to-3D | `models.s4_trellis` | `run` |
| 5 | register to own extraction | `agents.assets.factory_align` | `run` |
| 6 | CoACD + physics + URDF | `agents.assets.s6_physics` | `run` |
| 7 | drop test + yield report | `agents.eval.factory_report` | `run` |
| 8 | MJCF + Isaac manifest + MuJoCo settle test | `robo.sim.export_mjcf --test` | `run` |

Because `auto_segment` emits the same per-instance contract that
`load_gt_instances()` provides, stages 2-7 run byte-identically in both modes
— that is what makes the GT-vs-automatic ablation a controlled comparison.

### [`run/run_simfoundry.sh`](../run/run_simfoundry.sh) — SimFoundry-reproduction baseline (single-frame zero-shot)

One representative frame, monocular metric depth, no GT: the paper-faithful
reproduction of the prior SimFoundry system (arXiv:2606.28276) that this
project began from, kept as a baseline. The paper's ablation row D. Output:
`outputs/<scene>`. (Formerly `run_scene.sh`.)

| # | stage | module | env helper |
|---|---|---|---|
| 1 | s0 select frame | `agents.discover.s0_select_frame` | `run` |
| 2 | s1 SAM3 segmentation | `models.s1_segment` | `run_sam3` |
| 3 | s2 DA3 metric depth | `models.s2_depth` | `run` |
| 4 | s3 lift objects | `agents.discover.s3_lift` | `run` |
| 5 | s4 TRELLIS image-to-3D | `models.s4_trellis` | `run` |
| 6 | s5 pose alignment + F1 eval | `agents.assets.s5_align` | `run` |
| 7 | s6 CoACD + physics + URDF | `agents.assets.s6_physics` | `run` |
| 8 | s7 PyBullet settle + dynamics | `robo.sim.s7_sim` | `run` |
| 9 | s8 gsplat renders | `agents.render.s8_render` | `run_gs` |

### [`run/run_inpaint.sh`](../run/run_inpaint.sh) — removal + background completion

Gaussian-native object removal and background fill for a factory scene
(defaults to `outputs/<scene>_factory`). Produces
`inpaint/clean_background.ply`, a drop-in Inria-format splat. Details in
section 4.

| # | stage | module | env helper |
|---|---|---|---|
| 1 | removal sets + support planes + view lists | `agents.edit.inpaint_prepare` | `run` |
| 2 | SAM3 union masks | `agents.edit.inpaint_masks` | `run_sam3` |
| 3 | erase objects in views | `agents.edit.inpaint_qwen` | `run_qwen` |
| 4 | carve + plane fill + gsplat refine | `agents.edit.inpaint_fill` | `run_gs` |

## 2. Stage catalog

### s0 — representative frame selection (`agents/discover/s0_select_frame.py`)

Zero-shot mode only. ScanNet++ DSLR captures walk the room, so instead of
"frame 0" it picks the frame that sees the most target objects, by projecting
GT centroids with a mesh occlusion check. GT is used only to choose the frame;
perception never sees it. Main venv. Writes `frame/rep_frame.json` + a copy of
the chosen image.

### s1 — SAM3 segmentation (`models/s1_segment.py`)

Text-prompted instance segmentation on the representative frame. Standalone on
purpose (no `common.py` import) because it must run under the sam3 env, whose
torch version differs from the main venv. Input: the image + class prompts;
output: `masks/masks.npz` (masks, labels, scores) + an overlay image.

### s2 — metric depth (`models/s2_depth.py`)

Monocular metric depth on the representative frame with DA3METRIC-LARGE
(Depth Anything 3). DA3 outputs canonical depth for a focal-300px camera; the
stage converts it to metric via the processed focal, then estimates a single
robust scale `s*` against the scan mesh's z-depth so objects land correctly
relative to the background collision mesh. Main venv, GPU. Writes
`depth/depth.npz`.

### s3 — lift (`agents/discover/s3_lift.py`)

Lifts each SAM3 instance to a world-frame point cloud using the corrected
depth, with size gates (0.025-0.9 m) and a minimum point count. Per object
writes `obj_XX/rgba.png` (the TRELLIS input crop), `points.ply` (the alignment
target for s5), and `meta.json`; each instance is also matched to a GT object
for evaluation and background carving only. Main venv.

### s4 — TRELLIS image-to-3D (`models/s4_trellis.py`)

Generates one asset per object from `obj_XX/rgba.png` — the paper's `V_mesh`
slot. Outputs, all in the same canonical z-up ~[-0.5,0.5]^3 frame:
`trellis_mesh.ply` (vertex-colored FlexiCubes mesh), `mesh_sim.ply/.obj`
(<=40k-triangle decimation for collision/sim), and `trellis_gs.ply`
(gaussians). Main venv, GPU.

### s4 alternative — ReconViaGen (`models/s4_reconviagen.py`)

Multi-view alternative for the `V_mesh` slot: per object, collect up to 12
occlusion-aware masked crops across the DSLR trajectory and run
VGGT-conditioned TRELLIS (ReconViaGen, arXiv 2510.23306), saving
`rvg/` mesh + gaussians. Registered and F1-scored identically to the TRELLIS
asset for a head-to-head comparison; the hybrid stage (section 3) arbitrates.
Main venv, GPU.

### s5 — Sim(3) registration (`agents/assets/s5_align.py`)

Registers each canonical asset into the metric scene: 3-DoF translation +
3-DoF rotation + isotropic scale, with an upright (z-up) prior. Yaw candidates
every 10 degrees are scored by clipped-mean one-way chamfer, refined by ICP,
upright re-snap, then alternating scale polish and translation refinement.
Also evaluates F1@2/4cm against the GT instance mesh. Main venv, CPU;
dataset-free stress tests live in
[`tests/test_align_synthetic.py`](../tests/test_align_synthetic.py).

### s6 — physics annotation (`agents/assets/s6_physics.py`)

Makes each asset sim-ready: CoACD convex decomposition of `mesh_sim.ply` into
`collision/part_*.obj` (<=16 parts, more for large objects), physical
parameters (mass/friction/restitution) from a local VLM (Qwen2.5-7B-Instruct
stands in for the paper's Gemini `V_scene` slot) with a density-table fallback,
and `object.urdf` in the canonical frame with the registration scale baked
into `<mesh scale>`. Main venv.

### s7 — PyBullet compose + settle (`robo/sim/s7_sim.py`)

Composes the twin in PyBullet: the background collision mesh is the scan mesh
with the extracted objects carved out, cropped and decimated; objects are
spawned at their registered poses and settled by stepping the sim while
force-zeroing velocities until poses converge (poses are cached). Then a
dynamics demo (drop a duplicated bottle, push a mug) records 30 fps
trajectories for s8. Main venv.

### s8 — photoreal rendering (`agents/render/s8_render.py`)

Renders the twin with gsplat: background scene splat + per-object TRELLIS
gaussians transformed by the settled or per-frame simulated poses, composited
into the real scene. Outputs `render/photo_vs_twin.png`, `physics.mp4`,
`asset_gallery.png`, per-object turntables, `stats.json`. Mini-viewer env,
GPU. The general-purpose version,
[`agents/render/gsplat_sim_render.py`](../agents/render/gsplat_sim_render.py),
consumes a pose log from any simulator (MuJoCo/Isaac/PyBullet) and renders
video with the clean background splat when it exists.

### auto_segment — GT-free discovery (`agents/discover/auto_segment.py`)

The automatic mode's stage 0: discover 3D instances from RGB + poses + mesh
alone. Samples frames uniformly over the trajectory, runs SAM3 with an open
vocabulary of manipulable classes on each (`SIMANY_FULL=1` adds a furniture
tier), lifts each 2D mask to 3D by ray-casting onto the mesh keeping the
dominant depth cluster, and merges proposals across frames by voxel-IoU
union-find; instances confirmed in >=2 frames survive. Emits
`auto_instances.npz` with the same contract as GT instances. Sam3 env, GPU.

### factory_prepare (`agents/discover/factory_prepare.py`)

Enumerates instances (GT annotations in factory mode, `auto_instances.npz`
under `SIMANY_AUTO=1`), applies a whitelist and size gate, and for each
instance scores every DSLR frame by visibility (mesh raycast occlusion),
projected pixel area, and crop sharpness to cut the best-view RGBA crop with
an occlusion-aware mask. Writes the same layout the zero-shot stages use
(`objects/objects.json` + `obj_XX/{rgba.png, gt_points.ply, meta.json}`), so
s4/s6 run unchanged. Main venv.

### factory_refine_masks (`agents/discover/factory_refine_masks.py`)

Transparent/reflective objects (bottles) scan incompletely, so the
mesh-projected mask misses most of the object. On each object's best frame,
run SAM3 with the class prompt and swap in the instance that best overlaps the
projected mask; falls back to the projected mask when nothing matches. Sam3
env, GPU.

### factory_align (`agents/assets/factory_align.py`)

Registers assets to their instance submesh with the same validated
`align_object` as s5 — complete-to-complete registration, much better
conditioned than the zero-shot partial-depth case. Assigns quality tiers on
the free F1: A (F1@20mm >= 0.40), B (F1@40mm >= 0.20), C (rejected), with F1
evidence outranking the size heuristic because scan-incomplete GT inflates
extents. Writes `aligned.json` per object. Main venv.

### factory_hybrid (`agents/assets/factory_hybrid.py`)

Dual-generation winner selection for the `V_mesh` slot; see section 3. Main
venv, GPU.

### derive_mesh_from_splat (`agents/discover/derive_mesh_from_splat.py`)

Ablation rung C (`SIMANY_MESH_SRC=derived`): build a scan-mesh substitute from
the trained splat only, via rendered-depth TSDF fusion (5 mm voxel / 2 cm
truncation, chosen so small objects are not smoothed into their support
surface). Two subcommands because gsplat and open3d do not coexist in one env
here: `render` (mini-viewer env, GPU) then `fuse` (main venv).

## 3. Hybrid V_mesh selection (`agents/assets/factory_hybrid.py`)

Single-view TRELLIS and multi-view ReconViaGen fail on different objects, so
the hybrid stage generates both and lets the registration residual arbitrate
per object, with no category rules:

1. snapshot the existing TRELLIS canonical files into `obj_XX/trellis/`
   (once; the snapshot is the authoritative TRELLIS candidate on reruns);
2. generate the ReconViaGen asset into `obj_XX/rvg/` (cached if present; the
   TRELLIS asset is never regenerated);
3. register the RVG mesh exactly like `factory_align` (same `align_object`,
   same tiers and size gate) -> `obj_XX/rvg/aligned.json`;
4. score both registered candidates with `s5_align.sym_score` — a symmetric
   clipped chamfer against `gt_points.ply`, whose mesh->target direction
   saturates at the clip distance so unobserved back sides cost a bounded
   constant while an inflated mesh is heavily penalized — and pick the lower
   residual (a tier-C/rejected RVG asset, or one without gaussians, can never
   win);
5. write the verdict to `obj_XX/hybrid.json` and materialize the winner into
   the canonical files downstream stages read (`trellis_gs.ply`,
   `trellis_mesh.ply`, `mesh_sim.*`, `aligned.json`, `collision/`,
   `object.urdf`) via atomic copies; collision and URDF are rebuilt with the
   s6 machinery, physics parameters reused from `physics.json` (no VLM call).

The stage is additive and idempotent: originals live on under `obj_XX/trellis/`
and `obj_XX/rvg/`, and a TRELLIS win restores the snapshot. A per-scene
summary goes to `objects/hybrid_all.json`.

Measured effect ([CONTRIBUTIONS.md](CONTRIBUTIONS.md)): the residual gate
lifts pooled F1@20 mm from 0.708 to 0.783 over 457 objects, picks multi-view
for 53% of them, and catches all 18 multi-view collapses (F1 < 0.1) — within
0.006 of the oracle that always picks the better asset.

## 4. Gaussian-native editing (`agents/edit/`)

Moving an object in the twin exposes its ghost baked into the background
splat, so the pipeline erases each object from the splat and completes the
background — in Gaussian space, no retraining.

**Removal** (`inpaint_prepare.py` + `inpaint_masks.py` + the vote in
`inpaint_fill.py`) unions three selectors:

1. scene-splat gaussians within 3 cm of the GT instance surface;
2. gaussians within 2.4 cm of the *registered asset* surface — the scan
   misses transparent parts (bottle bodies), the asset covers the full extent;
3. a multi-view mask vote: transparent objects are modeled by the splat as
   diffuse low-opacity gaussian clouds whose centers can sit far from any
   physical surface (behind glass, above the desk), so geometric proximity
   misses them. A gaussian near the object's AABB whose center projects inside
   the object's 2D mask in >=2 related views is removed; opacity-agnostic by
   construction.

The 2D masks per view are `union(projected mask, best-overlapping SAM3
instance)`, dilated (`inpaint_masks.py`, sam3 env) — a projected-only mask
leaves ghosts of parts the scan never captured.

**2D inpainting** (`inpaint_qwen.py`) erases each object from its top-K
related views. Preferred backend: Qwen-Image-Edit-2509
(`QwenImageEditPlusPipeline`; it has no mask input, so the paste step
composites only inside the mask and pixels outside stay bit-identical). It
needs torch >= 2.5, i.e. the sam3 env (`QWEN_PY=$SAM3PY`), and on 48 GB GPUs
like the A6000 additionally `SIMANY_QWEN=force` to enable sequential CPU
offload; `SIMANY_QWEN=0` disables it. Fallback: LaMa on CPU. The measured
supervisor comparison, LaMa vs Qwen ([BASELINES.md](BASELINES.md)):
keyboard-desk 34.3 vs 23.9 dB; elsewhere 19.4 vs 20.5.

**Fill + refinement** (`inpaint_fill.py`, mini-viewer env, GPU): carve the
removal set out of the splat, seed new thin normal-oriented gaussian disks on
the MAD-trimmed support plane (fitted by `inpaint_prepare.py` through the
scan-mesh ring around the footprint) on a 5 mm grid, colors initialized by
projecting each fill point into its object's best inpainted view (kNN
neighbor-gaussian colors only as a fallback), then optimize only the new
gaussians
(position/scale/opacity/color, orientation frozen to the plane) with an L1
loss inside the removal masks against the per-frame composited inpainted
images, via differentiable gsplat rendering. Output:
`inpaint/clean_background.ply` (carved + filled, Inria format) plus
before/hole/filled comparison renders.

## 5. Single-image variant (`agents/single_image/`)

Replaces the multi-view scan with a SHARP feed-forward splat predicted from
one photo; everything downstream is the standard machinery.

- The photo's own camera pose is the identity in the splat frame (SHARP
  unprojects with `extrinsics=eye(4)`), and SHARP stores the intrinsics and
  image size inside the .ply itself; `sharp_ply_meta.py` reads K/W/H back out
  rather than re-deriving them, so projections match how the gaussians were
  placed.
- `sharp_render_depth.py` (mini-viewer env) renders the splat from that
  identity pose to recover RGB + depth + alpha exactly matching the photo —
  the stand-in for s2.
- `models.s1_segment` runs on the photo as usual (sam3 env), then
  `sharp_s2_lift.py` (main venv, CPU) mirrors `s3_lift.py`'s gates and
  outputs, sourcing depth from the splat render, with all GT fields null.
- Frame caveat: SHARP's splat frame is OpenCV camera convention (y-down,
  z-forward), not z-up, which `s5_align` structurally assumes. `points.ply` is
  rotated by a fixed axis permutation `R_ZUP` (`sharp_ply_meta.py`) that
  assumes a roughly level, non-tilted photo — no per-frame gravity estimation.
  `meta.json`'s centroid/extent stay in the original camera frame on purpose.
- Then the same s4 -> s5 -> hybrid machinery: `sharp_render_object_views.py`
  (mini-viewer env) renders 12 small-disparity views of the splat per object
  for ReconViaGen conditioning, and `sharp_hybrid.py` runs the verbatim
  `factory_hybrid` sym_score arbitration with the object's own lifted
  `points.ply` as the target — no GT exists for this input, so its "F1"
  numbers are relabeled fit@20/40mm ("how well does the registered mesh
  explain the one observed view"), not reconstruction accuracy.

There is no `run/` launcher for this variant; the stages are invoked
individually.

## 6. Baselines (`agents/baselines/`)

- [`flashsplat.py`](../agents/baselines/flashsplat.py) — FlashSplat
  (ECCV 2024) optimal per-gaussian mask assignment, as a removal-set baseline.
  Their custom CUDA rasterizer fork is unbuildable here, so this is an
  equivalent reimplementation on gsplat: rendering is linear in per-gaussian
  colors, so one backward pass through a 2-channel dummy color recovers their
  inside/outside blending-weight counts exactly. Measured: union IoU 0.649 vs
  our final removal set (recall 0.87), single pilot scene.
- [`maskclustering.py`](../agents/baselines/maskclustering.py) —
  MaskClustering (CVPR 2024) discovery baseline on the same protocol as
  `auto_segment` (images + poses + mesh, no GT); subcommands `prepare` (build
  their dataset layout, mesh-raycast depth), `srun` (print the GPU commands),
  `convert` (map their output back to the `auto_instances.npz` contract for
  `eval_instances`). Measured at a matched 28-frame budget with the same SAM3
  masks: F1@IoU0.25 0.605 (ours) vs 0.571, F1@IoU0.5 0.419 vs 0.286, single
  scene.
- [`mc_masks.py`](../agents/baselines/mc_masks.py) — substitutes SAM3 for
  Cropformer (unbuildable CUDA op) as MaskClustering's 2D segmenter, which
  also isolates the 3D-aggregation comparison. Sam3 env.

## 7. Evaluation (`agents/eval/`)

- [`eval_instances.py`](../agents/eval/eval_instances.py) — discovery
  precision/recall/F1 at vertex-IoU 0.25/0.5 vs GT whitelist instances; works
  for `auto_segment` and converted MaskClustering output alike.
- [`eval_vs_gt.py`](../agents/eval/eval_vs_gt.py) — honest re-scoring for
  automatic mode: under `SIMANY_AUTO` the factory's own F1 is "vs own
  extraction", so this matches each discovered object to a true GT instance
  and recomputes F1@20/40mm against the real GT submesh.
- [`factory_eval_render.py`](../agents/eval/factory_eval_render.py) —
  PSNR/SSIM/LPIPS on up to 8 held-out frames from the official DSLR test
  split: background splat alone (the SceneSplat/GaussianWorld baseline) vs the
  composite twin, plus clean-background variants when
  `inpaint/clean_background.ply` exists. Mini-viewer env, GPU.
- [`factory_report.py`](../agents/eval/factory_report.py) — sim-readiness QA:
  drop each asset on a plane in PyBullet, settle, 2 s free dynamics; "stable"
  if it neither sinks nor walks (<3 cm drift). Emits `report.json` + a contact
  sheet.
- [`verify_removal_render.py`](../agents/eval/verify_removal_render.py) /
  [`verify_removal_check.py`](../agents/eval/verify_removal_check.py) — the
  supervision-independent verification battery for the edited scene: stage 1
  (mini-viewer env) renders the clean background with alpha/expected-depth and
  checks hole coverage, support-plane depth residual, and held-out-view
  photometric consistency; stage 2 (sam3 env) re-runs SAM3 with each removed
  object's own prompt on the clean renders — success is no residual
  detection. Failures become a re-processing worklist.
- [`aggregate_results.py`](../agents/eval/aggregate_results.py) — folds the
  50-scene validation fleet (`outputs/<scene>_factory/{report.json,
  render_metrics.json, timings.txt}`) into `outputs/val_summary.json` and a
  markdown table: render quality, tier yield, drop-test stability, per-stage
  time.
- [`behavior1k_coverage.py`](../agents/eval/behavior1k_coverage.py) —
  BEHAVIOR-1K coverage: for each of the 1018 BDDL activities, how much of its
  object/room requirement the SimAny vocabulary (or the objects actually
  discovered in the 50 processed scenes) can satisfy. Pure text parsing, no
  GPU.
- [`make_paper_tables.py`](../agents/eval/make_paper_tables.py) — regenerates
  every LaTeX data table for the paper from the on-disk results, so the paper
  never drifts from the measurements. The tables are written to
  `docs/paper/tables/` (the module's default `--out`).
