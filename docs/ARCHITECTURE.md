# PhiRIE code architecture

PhiRIE turns posed RGB images of a real indoor scene + its mesh + 3D Gaussian
splat (or a single phone video) into an editable, photorealistic,
simulation-ready digital twin. This page covers where the code lives, how it
executes, what each stage reads and writes, and every `SIMANY_*` environment
variable. The internal package and variable names (`simany`, `SIMANY_*`,
`phiroom`) predate the PhiRIE name and are kept for compatibility.

Related pages: [ENVIRONMENTS.md](ENVIRONMENTS.md) (python environments),
[PIPELINE.md](PIPELINE.md) (stage details), [MODULES.md](MODULES.md) (the
`phiroom` control CLI), [ROBOT.md](ROBOT.md) (robot layer),
[DATA_AND_WEIGHTS.md](DATA_AND_WEIGHTS.md) (paths and weights).

## 1. Code map

```
PhiRIE/
├── phiroom/                    control CLI: lists, plans and runs module actions (CPU, no GPU imports)
│   ├── cli.py                  `phiroom modules|describe|plan|run|pipeline`
│   ├── core/                   registry.py (contracts), runtime.py (interpreter aliases),
│   │                           execution.py (argv planning, fail-fast execution, receipts)
│   ├── modules/*.json          the 15 feature contracts
│   └── pipelines/<module>/README.md   one guide per module (actions, runtimes, implementations)
├── agents/                     construction pipeline: scan -> sim-ready twin
│   ├── core/
│   │   └── common.py           shared paths/env(), camera loaders, vocabulary,
│   │                           GT/auto instance loaders, gaussian IO + gsplat render helpers
│   ├── discover/               instance discovery (frame selection, lifting, GT enumeration)
│   │   ├── s0_select_frame.py       pick the representative frame (single-frame mode)
│   │   ├── s3_lift.py               lift SAM3 instances to world-frame clouds + RGBA crops
│   │   ├── auto_segment.py          GT-free 3D instance discovery: SAM3 on sampled frames,
│   │   │                            mesh-backprojected 3D merging (sam3 env)
│   │   ├── derive_mesh_from_splat.py  scan-mesh substitute via TSDF fusion of splat-rendered depth
│   │   ├── factory_prepare.py       enumerate instances (GT or auto), best-view crops
│   │   └── factory_refine_masks.py  refine projected masks with SAM3 image evidence (sam3 env)
│   ├── assets/                 per-object asset registration + physics annotation
│   │   ├── s5_align.py              Sim(3) registration: yaw grid + clipped one-way chamfer
│   │   ├── s6_physics.py            CoACD convex decomposition + VLM physics params -> URDF
│   │   ├── factory_align.py         register assets to the instance submesh; quality tiers A/B/C
│   │   └── factory_hybrid.py        TRELLIS-vs-ReconViaGen winner selection per object
│   ├── edit/                   gaussian-native object removal + background completion
│   │   ├── inpaint_prepare.py       removal sets, support planes, view lists (main env, CPU)
│   │   ├── inpaint_masks.py         SAM3 union masks for removal (sam3 env)
│   │   ├── inpaint_qwen.py          erase objects in views: Qwen-Image-Edit, LaMa fallback
│   │   └── inpaint_fill.py          carve + plane fill + photometric refine (gsplat env)
│   ├── render/                 photoreal rendering of simulator states
│   │   ├── s8_render.py             stage 8: photo-vs-twin, physics video, asset gallery
│   │   └── gsplat_sim_render.py     render any pose log (MuJoCo/Isaac/PyBullet) as splats
│   ├── eval/                   scoring
│   │   ├── eval_instances.py        discovery precision/recall/F1 vs GT (vertex IoU)
│   │   ├── eval_vs_gt.py            registered assets re-scored vs independent GT submesh
│   │   ├── factory_report.py        drop-test stability + yield report
│   │   ├── factory_eval_render.py   PSNR/SSIM/LPIPS on held-out views (gsplat env)
│   │   ├── verify_removal_render.py removal verification stage 1: clean renders (gsplat env)
│   │   ├── verify_removal_check.py  removal verification stage 2: SAM3 re-detection (sam3 env)
│   │   ├── build_audit.py           GT-free build audit: one JSON + report per build
│   │   └── droid_alignment_eval.py  held-out check of the DROID FK<->SfM alignment
│   ├── recon/                  video / DROID / BEHAVIOR -> posed metric scene dir + splat
│   │   ├── frames.py                uniform frame extraction from a video (ffmpeg)
│   │   ├── colmap_poses.py          COLMAP pose backend (pycolmap)
│   │   ├── metricize.py             metric scale + z-up + floor at z=0 for feed-forward poses
│   │   ├── make_scene_dir.py        emit the ScanNet++-style scene directory common.py reads
│   │   ├── gsplat_train.py          train an Inria-format 3DGS splat (gsplat env)
│   │   ├── droid_extract.py         DROID raw episode -> wrist frames + FK camera trajectory (h5 env)
│   │   ├── droid_static_select.py   viewpoint-diverse static-frame selection + robot masking
│   │   ├── align_to_traj.py         Umeyama-align a reconstruction to the FK trajectory
│   │   ├── robot_align.py           robot/metric-scale alignment (fiducial, surveyed, dimensions, FK)
│   │   ├── alignment_report.py      go/no-go certificate for an alignment
│   │   └── behavior_extract.py      BEHAVIOR episodes -> scene directory (h5 env)
│   ├── baselines/              comparison methods re-run on our inputs
│   │   ├── flashsplat.py            FlashSplat gaussian-selection removal baseline
│   │   ├── maskclustering.py        MaskClustering 3D instance discovery baseline
│   │   └── mc_masks.py              SAM3 stand-in for MaskClustering's 2D segmenter
│   ├── single_image/           single-photo variant: SHARP feed-forward splat replaces the scan
│   │   ├── sharp_s2_lift.py         lift from SHARP-splat depth (mirrors s3_lift)
│   │   ├── sharp_render_depth.py    render SHARP splat at the input photo's camera (gsplat env)
│   │   ├── sharp_render_object_views.py  synthetic near-field views for ReconViaGen (gsplat env)
│   │   ├── sharp_hybrid.py          TRELLIS-vs-RVG winner selection, SHARP variant
│   │   └── sharp_ply_meta.py        recover the SHARP camera from a SHARP-saved .ply
│   └── models/                 adapters to external neural models (runnable stages;
│       │                       vendored code in third_party/, weights in the HF cache)
│       ├── s1_segment.py            SAM3 text-prompted segmentation (sam3 env, standalone)
│       ├── s2_depth.py              metric monocular depth (DA3) + robust scale vs mesh depth
│       ├── s4_trellis.py            TRELLIS image-to-3D: mesh + gaussians per object
│       ├── s4_trellis2.py           TRELLIS.2 mesh/PBR producer (trellis2 env)
│       ├── s4_sam3d.py              SAM 3D Objects image+mask-to-3D (sam3d env)
│       ├── s4_reconviagen.py        ReconViaGen multi-view (VGGT-conditioned TRELLIS) alternative
│       ├── rvg_pinned.py            resolve ReconViaGen's pretrained resources to local snapshots
│       └── vggt_scene.py            feed-forward scene poses/points: VGGT or VGGT-Omega
├── robo/                       robot layer: policy evaluation + simulator export
│   ├── envs/pi05_env.py        DROID-convention MuJoCo env (15 Hz) over a PhiRIE scene
│   ├── rigs/pi05_rig.py        Franka Panda + Robotiq 2F-85 rig assembled via MjSpec
│   ├── tasks/pi05_tasks.py     pick-and-place suite generation + staged scoring
│   ├── rendering/
│   │   ├── pi05_render.py           photoreal composite observations per control tick
│   │   ├── mujoco_masks.py          exact robot masks from MuJoCo segmentation rendering
│   │   ├── harmonizer_client.py     client for the tools.harmonizer.server socket service
│   │   ├── robot_restore.py         robot-preserving full-frame enhancement
│   │   └── color_match.py           background-fitted RGB affine baseline
│   ├── policy/                 policy registry (configs/policies/*.yaml), control contract,
│   │                           runtime identity, bound server, clients/ (pi05, scripted)
│   ├── manifest/               rollout manifest schema, IO/diff, canonical hashing
│   ├── eval/
│   │   ├── pi05_eval.py             websocket policy client, closed-loop episode runner
│   │   ├── episode_log.py           outcome taxonomy + per-episode ledger
│   │   ├── fidelity_metrics.py      held-out appearance and metric-geometry evaluation
│   │   ├── harmony_visual_metrics.py  harmonized-observation quality/consistency/latency
│   │   └── metric_utils.py          shared metric helpers
│   ├── sim/                    simulator export and dynamics
│   │   ├── s7_sim.py                stage 7: compose twin in PyBullet, settle, dynamics demo
│   │   ├── export_mjcf.py           MuJoCo MJCF + Isaac-Lab manifest export + headless settle test
│   │   ├── room_collision.py        full-room static collision export (floor, boxes, CoACD)
│   │   ├── export_omnigibson.py     package sim-ready objects for BEHAVIOR-1K/OmniGibson
│   │   ├── omnigibson_bridge/       OmniGibson-side install, import/convert/demo scripts
│   │   ├── redrop.py                re-run drop tests with the COM->link frame correction
│   │   ├── mujoco_video.py          MuJoCo rollout video from the exported scene.xml
│   │   └── viewer_settle.py         physics backend for interface/viewer.py
│   └── simfactory/             YAML pipeline runner: runner.py, registry.py, blocks/
├── interface/                  human-facing tools
│   ├── viewer.py               viser multi-scene digital-twin editor (default port 8090)
│   ├── mujoco_live_viewer.py   live MuJoCo sim rendered as gaussians in the browser (port 8091)
│   └── demo_movie.py           demo-film segments
├── capture/                    phone-capture bundle: extract_metadata.py, validate_video.py
├── tools/
│   ├── phiview/                bundled PhiView viewer source (SOURCE_PROVENANCE.json)
│   ├── harmonizer/server.py    Unix-socket NVIDIA Harmonizer service
│   └── release/                verify.py (contracts, versions, bundle checksums), archive.py
├── run/                        launchers
│   ├── env.sh                  shared env: paths, interpreters, run helpers (source it)
│   ├── setup_env.sh            installs the main .venv
│   ├── run_factory.sh          GT-driven mode
│   ├── run_auto.sh             fully automatic mode (no GT annotations)
│   ├── run_inpaint.sh          object removal + background completion
│   ├── run_video2sim.sh        phone video -> scene dir + splat -> automatic mode
│   ├── run_droid_recon.sh      DROID episode -> twin;  run_behavior_recon.sh: BEHAVIOR clips -> twin
│   ├── run_simfoundry.sh       single-frame SimFoundry-reproduction baseline
│   ├── phiroom.sh, phiview.sh  control CLI and PhiView CLI wrappers (uv projects)
│   ├── pi05_serve.sh           serve the pi0.5 openpi policy (websocket, port 8000)
│   ├── pi05_serve_bound.sh     serve one registry policy with an identity receipt
│   ├── serve_harmonizer.sh     start tools.harmonizer.server
│   ├── smoke_imports.sh        import-smoke harness (one process per module)
│   └── fetch_droid_raw.py, gcs_fetch.py   public-bucket downloads (DROID raw, openpi assets)
├── configs/                    runtime.example.json, capture/, calibration/, policies/, evaluation/
├── envs/control/               locked uv control environment (pyproject + uv.lock)
├── media/                      README images and recordings
├── data/, videos/, third_party/   dataset mount point, phone-video drop folder, vendored checkouts
├── outputs/, weights/, checkpoints/   run products and downloaded weights (all gitignored)
└── docs/                       this page, PIPELINE, MODULES, PHIVIEW, ROBOT, DATA_AND_WEIGHTS, ENVIRONMENTS
```

## 2. How execution works

The GPU packages are **not pip-installed** into any environment
(`pyproject.toml` lists only the control CLI dependency — the pipeline spans
mutually incompatible torch environments). Stage modules run with the repo
root on `sys.path`:

```bash
cd ${SIMANY_ROOT}
python -m agents.assets.s5_align        # works: python -m puts CWD on sys.path
python agents/assets/s5_align.py        # fails: package imports unresolved
```

From anywhere else, `export PYTHONPATH=${SIMANY_ROOT}` first
(this is what `run/smoke_imports.sh` does).

### run/env.sh

Every launcher sources [`run/env.sh`](../run/env.sh), which resolves paths,
`cd`s to the repo root, and defines one run helper per python environment:

| Helper | Interpreter variable | Default interpreter | Environment |
|---|---|---|---|
| `run` | `SIMANY_PY` | `$ROOT/.venv/bin/python` | main pipeline: torch 2.4.1+cu124, open3d/trimesh/coacd/xformers |
| `run_sam3` | `SIMANY_SAM3_PY` | `$ROOT/.envs/sam3/bin/python` | SAM3 + Qwen-Image-Edit: torch >= 2.5 |
| `run_gs` | `SIMANY_GSPLAT_PY` | `$ROOT/.envs/mini-viewer/bin/python` | gsplat CUDA rendering (py3.10) |
| `run_qwen` | `QWEN_PY` (no `SIMANY_` prefix) | `$SIMANY_PY` | set `QWEN_PY=$SIMANY_SAM3_PY` on machines with torch >= 2.5 and enough VRAM |
| `run_sam3d` | `SIMANY_SAM3D_PY` | `$ROOT/.envs/sam3d-objects/bin/python` | SAM 3D Objects |

Usage: `run <package>.<module> [args...]`, e.g.

```bash
source run/env.sh
run agents.assets.s5_align
run_sam3 agents.models.s1_segment --image ... --out-dir ... --prompts bottle mug
run_gs agents.render.s8_render
run_qwen agents.edit.inpaint_qwen
```

Why the environments cannot be merged is documented in
[ENVIRONMENTS.md](ENVIRONMENTS.md). `env.sh` also sets `HF_HUB_OFFLINE=1`,
probes for `g++-12`/`g++-13` (gsplat JIT), and pins
`TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"` so a gsplat JIT cache built once is
reused.

Two more `env.sh` conveniences:

- `RESUME=1` — the `done_skip` helper lets a launcher skip a stage whose
  output already exists.
- `stage_timed` appends `<stage> <seconds>` to `$SIMANY_OUT/timings.txt`.

### The `phiroom` control CLI

`bash run/phiroom.sh <modules|describe|plan|run|pipeline>` runs the CPU
control package from the locked `envs/control` environment. It reads the
contracts in `phiroom/modules/*.json`, resolves the runtime alias of an action
to an interpreter (`phiroom/core/runtime.py`: the same `SIMANY_*_PY`
variables, or `configs/runtime.local.json`), and either prints the exact
command (`plan`) or executes it and writes a receipt under
`outputs/module-runs/` (`run`). Shell actions (`construction/construct-auto`,
`robotics/policy-server`, ...) wrap the launchers below. See
[MODULES.md](MODULES.md).

### Launchers

| Script | Mode |
|---|---|
| `run/run_factory.sh` | GT-driven asset factory |
| `run/run_auto.sh` | fully automatic (no GT annotations); exports MJCF + Isaac manifest |
| `run/run_inpaint.sh` | gaussian-native removal + background completion for a factory scene |
| `run/run_video2sim.sh` | phone video -> scene directory + splat -> automatic mode + task suite |
| `run/run_droid_recon.sh`, `run/run_behavior_recon.sh` | DROID episode / BEHAVIOR clips -> the same automatic tail |
| `run/run_simfoundry.sh` | single-frame SimFoundry-reproduction baseline: one representative frame, monocular metric depth, no GT |
| `run/pi05_serve.sh` | openpi pi0.5 policy server (run on a GPU machine) |
| `run/pi05_serve_bound.sh` | serve one policy from `configs/policies/` with a runtime identity receipt |
| `run/serve_harmonizer.sh` | start the harmonizer socket service (inside the Harmonizer environment) |
| `run/phiview.sh` | PhiView's `physicalview` CLI from the bundled uv project |
| `run/smoke_imports.sh` | imports every module in a fresh process; diff before/after refactors |

## 3. Data flow for one scene

All stages communicate through the filesystem under one output tree, selected
by `SIMANY_OUT` (default `outputs/$SIMANY_SCENE`). The launchers use the
conventions `outputs/<scene>` (single-frame baseline), `outputs/<scene>_factory`
(GT-driven), `outputs/<scene>_auto` (automatic) and `outputs/video_<name>`
(video); any other `SIMANY_OUT` works the same way. Scene *inputs* always
come from the ScanNet++-style tree (`SIMANY_SCANNETPP_ROOT`) and the splat
(`SIMANY_SPLATS_ROOT/<scene>.ply`), resolved in `agents/core/common.py`;
the video and robot-dataset launchers write that tree themselves under
`data/recon_scenes/`.

### Single-frame baseline (`run_simfoundry.sh`) — `outputs/<scene>/`

| Stage | Reads | Writes |
|---|---|---|
| `agents.discover.s0_select_frame` | GT centroids + mesh (frame choice only) | `frame/rep_frame.json` + a copy of the chosen image |
| `agents.models.s1_segment` (sam3) | `frame/<image>` | `masks/` (`masks.npz`, `detections.json`, `overlay.png`) |
| `agents.models.s2_depth` | `frame/`, scan mesh (scale bridge) | `depth/` (`depth.npz`, `depth_vis.png`, `stats.json`) |
| `agents.discover.s3_lift` | `masks/`, `depth/` | `objects/objects.json`, per object `objects/obj_XX/` (`rgba.png`, `points.ply`, `meta.json`) |
| `agents.models.s4_trellis` | `obj_XX/rgba.png` | `obj_XX/` (`trellis_mesh.ply`, `mesh_sim.ply`/`.obj`, `trellis_gs.ply`) |
| `agents.assets.s5_align` | `obj_XX/points.ply` + asset mesh | `obj_XX/aligned.json`, `objects/aligned_all.json` |
| `agents.assets.s6_physics` | `obj_XX/mesh_sim.*` | `obj_XX/collision/part_*.obj`, `physics.json`, `object.urdf` |
| `robo.sim.s7_sim` | URDFs + poses + scan mesh | `sim/` (`background.obj`, `settled.json`, `dynamics.json`, `debug.mp4`) |
| `agents.render.s8_render` (gsplat) | scene splat + `obj_XX/trellis_gs.ply` + `sim/` poses | `render/` (`photo_vs_twin.png`, `physics.mp4`, `asset_gallery.png`, `stats.json`) |

### Factory mode (`run_factory.sh`) — `outputs/<scene>_factory/`

Same `objects/` layout, GT-driven discovery, timing to `timings.txt`:

| Stage | Reads | Writes |
|---|---|---|
| `agents.discover.factory_prepare` | GT `segments_anno.json`, full DSLR trajectory | `objects/objects.json`, `obj_XX/rgba.png` best-view crops |
| `agents.discover.factory_refine_masks` (sam3) | `obj_XX` crops + DSLR images | refined `obj_XX/rgba.png`; marker `objects/.masks_refined` |
| `agents.models.s4_trellis` | `obj_XX/rgba.png` | `obj_XX/trellis_*.ply`, `mesh_sim.*` |
| `agents.assets.factory_align` | asset mesh + GT instance submesh | `obj_XX/aligned.json` (tiers A/B/C), `objects/aligned_all.json` |
| `agents.assets.s6_physics` | `obj_XX/mesh_sim.*` | `obj_XX/collision/`, `physics.json`, `object.urdf` |
| `agents.eval.factory_report` | all non-rejected assets | `report.json`, `crops_sheet.png` |
| `agents.eval.factory_eval_render` (gsplat) | held-out DSLR views + splat + assets | `render_metrics.json`, `eval_photo_bg_twin.jpg` |

Optional hybrid slot (run after the TRELLIS stage):
`agents.models.s4_reconviagen` writes `obj_XX/rvg/` (multi-view crops +
`rvg_mesh.ply`/`rvg_gs.ply` + `aligned.json`), and
`agents.assets.factory_hybrid` picks the winner per object
(`obj_XX/hybrid.json`, `objects/hybrid_all.json`; the pre-hybrid registration
is kept as `objects/aligned_all_prehybrid.json`, the TRELLIS candidate under
`obj_XX/trellis/`). `agents.models.s4_trellis2` and `agents.models.s4_sam3d`
write `obj_XX/trellis2/` and `obj_XX/sam3d/` candidates in the same way
(`SIMANY_HYBRID_CANDIDATES` lists the candidates to arbitrate).

### Automatic mode (`run_auto.sh`) — `outputs/<scene>_auto/`, `SIMANY_AUTO=1`

Discovery is replaced by `agents.discover.auto_segment` (sam3), which writes
`auto_instances.npz`; `factory_prepare` then consumes those instances instead
of GT, and the rest of the factory chain runs unchanged. At the end,
`robo.sim.export_mjcf --test` writes `sim_export/`
(`scene.xml`, `isaac_manifest.json`, `mujoco_settle.json`) and settle-tests
the MJCF headlessly. Evaluation extras (run separately):
`agents.eval.eval_instances` (discovery F1 vs GT) and
`agents.eval.eval_vs_gt` (`eval_vs_gt.json`).

### Video mode (`run_video2sim.sh`) — `outputs/video_<name>/`

`agents.recon.frames` -> `recon/frames/`; `agents.recon.colmap_poses` (or
`agents.models.vggt_scene`) -> `recon/recon.npz`; `agents.recon.metricize` ->
`recon/recon_metric.npz`; `agents.recon.make_scene_dir` ->
`data/recon_scenes/data/<name>/dslr/...`; `agents.recon.gsplat_train` (gsplat)
-> `data/recon_scenes/splats/<name>.ply`; `agents.discover.derive_mesh_from_splat
render|fuse` -> `derived_mesh.ply`; then the automatic chain above with
`SIMANY_MESH_SRC=derived`, ending in `sim_export/` and
`sim_export/pi05_tasks.json`.

### Inpainting (`run_inpaint.sh`) — `<out>/inpaint/`

| Stage | Writes |
|---|---|
| `agents.edit.inpaint_prepare` (main env, CPU) | `inpaint/obj_XX/` (`removal_idx.npy`, `plane.json`, `views.json`, `proj_masks.npz`) |
| `agents.edit.inpaint_masks` (sam3) | `inpaint/obj_XX/mask_<k>.png` union removal masks |
| `agents.edit.inpaint_qwen` (`run_qwen`) | `inpaint/obj_XX/inpainted_<k>.png`, `inpaint/inpaint_meta*.json` |
| `agents.edit.inpaint_fill` (gsplat) | `inpaint/clean_background.ply`, `fill_gaussians.npz`, `fill_stats.json`, `compare_<frame>.jpg` |

The removal verification battery reads this tree:
`agents.eval.verify_removal_render` (gsplat) writes `inpaint/verify/`
(before/after renders + `verify_render.json`), then
`agents.eval.verify_removal_check` (sam3) re-detects on the clean renders.
The FlashSplat baseline (`agents.baselines.flashsplat`) writes its removal
sets alongside (`inpaint/flashsplat_*.{npy,json}`).

### Robot layer

`robo.tasks.pi05_tasks generate` reads a scene's `objects/` tree +
`sim_export/mujoco_settle.json` and writes `sim_export/pi05_tasks.json`.
`run/pi05_serve.sh` serves the policy (checkpoint selected by
`SIMANY_PI05_CKPT`/`SIMANY_PI05_CONFIG`); `robo.eval.pi05_eval --tasks
<...>/sim_export/pi05_tasks.json --out <dir>` runs closed-loop episodes with
photoreal composite observations (`robo.rendering.pi05_render`, using
`inpaint/clean_background.ply` when present) and writes `results.json` +
per-episode logs and mp4s to its `--out` directory. Details in
[ROBOT.md](ROBOT.md).

## 4. Environment variables

All knobs use the `SIMANY_` prefix. `agents.core.common.env(NAME)` also reads
the legacy `SIMF_<NAME>` prefix with a deprecation warning on stderr
("honoured for one release" — do not rely on it); `run/env.sh` likewise falls
back to `SIMF_SCENE`. Sources: [`run/env.sh`](../run/env.sh),
[`agents/core/common.py`](../agents/core/common.py),
[`phiroom/core/runtime.py`](../phiroom/core/runtime.py), and the modules named
below.

| Variable | Default | Read by | Effect |
|---|---|---|---|
| `SIMANY_ROOT` | derived from `env.sh` / `common.py` location | `env.sh`, `common.py` | repo root; everything else is resolved relative to it |
| `SIMANY_SCENE` | `c50d2d1d42` | `env.sh`, `common.py` | scene id (inputs) |
| `SIMANY_OUT` | `$ROOT/outputs/$SIMANY_SCENE` | `common.py`, launchers | output tree; `run_factory.sh` defaults it to `outputs/<scene>_factory`, `run_auto.sh` to `outputs/<scene>_auto`, `run_video2sim.sh` to `outputs/video_<name>` |
| `SIMANY_SCANNETPP_ROOT` | `/data/ScanNetpp` | `env.sh`, `common.py` | ScanNet++-style dataset root (`data/<scene>/...`) |
| `SIMANY_SPLATS_ROOT` | `/data/ScanNetppv2_gsplat/splats` | `env.sh`, `common.py` | trained scene splats (`<scene>.ply`) |
| `SIMANY_PY` | `$ROOT/.venv/bin/python` | `env.sh`, `phiroom` | main-pipeline interpreter (`run`) |
| `SIMANY_SAM3_PY` | `$ROOT/.envs/sam3/bin/python` | `env.sh`, `phiroom` | SAM3/Qwen interpreter (`run_sam3`) |
| `SIMANY_GSPLAT_PY` | `$ROOT/.envs/mini-viewer/bin/python` (`phiroom`: `.envs/gsplat/bin/python`) | `env.sh`, `phiroom` | gsplat interpreter (`run_gs`) |
| `SIMANY_SAM3D_PY` | `$ROOT/.envs/sam3d-objects/bin/python` | `env.sh`, `phiroom` | SAM 3D Objects interpreter (`run_sam3d`) |
| `SIMANY_TRELLIS2_PY` | `$ROOT/.envs/trellis2/bin/python` | `phiroom` | TRELLIS.2 interpreter |
| `SIMANY_H5_PY` | `$ROOT/.envs/h5/bin/python` (`simfactory`: `.venv/bin/python`) | `phiroom`, `simfactory`, `run_droid_recon.sh`, `run_behavior_recon.sh` | h5py interpreter for DROID/BEHAVIOR extraction |
| `QWEN_PY` (no prefix) | `$SIMANY_PY` | `env.sh`, `phiroom` | interpreter for `run_qwen`; set to the sam3 python for the Qwen backend |
| `SIMANY_AUTO` | unset | `common.py` (`load_instances`), `factory_prepare` | `1` = GT-free mode: instances come from `auto_instances.npz` instead of GT annotations |
| `SIMANY_MESH_SRC` | unset (GT scan mesh) | `common.py` | `derived` = use `$OUT/derived_mesh.ply`; any other value = explicit mesh path. Affects only `PIPELINE_MESH_PLY` (raycasting/registration/collision); evaluation always scores against the true GT mesh |
| `SIMANY_NO_GT` | unset | `s0_select_frame`, `s5_align` | `1` = skip every GT-dependent step (same as `--no-gt`) |
| `SIMANY_FULL` | unset | `factory_prepare`, `auto_segment` | `1` = furniture tier: full vocabulary instead of the small-object whitelist / extra SAM3 prompts |
| `SIMANY_VIS_TOL_M` | `0.02` | `factory_prepare` | visibility raycast tolerance (m) for best-view scoring |
| `SIMANY_MIN_BBOX_PX`, `SIMANY_MIN_MASK_PX` | `48`, `400` | `factory_prepare` | pixel gates for best-view crops (the DROID/BEHAVIOR launchers lower them) |
| `SIMANY_QWEN` | unset | `inpaint_qwen` | `0` = never use Qwen (LaMa only); `force` = skip the VRAM gate (sequential CPU offload) |
| `SIMANY_REQUIRE_QWEN` | unset | `inpaint_qwen` | `1` = re-raise if the Qwen pipeline fails to load instead of falling back to LaMa |
| `SIMANY_SHARD` | unset | `inpaint_qwen` | `"i/N"` = process only objects where `pos % N == i` (parallel sharding) |
| `SIMANY_HYBRID_CANDIDATES` | `trellis,rvg` | `factory_hybrid`, `simfactory` | candidate generators arbitrated per object |
| `SIMANY_RVG_PINNED_CONFIG` | unset | `factory_hybrid` | JSON mapping ReconViaGen's pretrained resources to local snapshots (`agents/models/rvg_pinned.py`) |
| `SIMANY_TRELLIS_DIR` | `$ROOT/third_party/TRELLIS` | `common.py` | TRELLIS checkout |
| `SIMANY_TRELLIS_MODEL` | `microsoft/TRELLIS-image-large` | `s4_trellis` | TRELLIS weights (HF id or local path) |
| `SIMANY_DINOV2_REPO` | unset | `s4_trellis` | local DINOv2 source for TRELLIS's `torch.hub` load |
| `SIMANY_GENERATION_RECORDS` | unset | `s4_trellis`, `s4_trellis2` | strict mode: refuse to overwrite existing proposals; record directory for TRELLIS.2 |
| `SIMANY_TRELLIS2_DIR`, `SIMANY_TRELLIS2_MODEL`, `SIMANY_TRELLIS2_SOURCE_COMMIT`, `SIMANY_DINOV3_MODEL`, `SIMANY_SS_DECODER` | required | `s4_trellis2` | TRELLIS.2 source, model, pinned commit, DINOv3 and sparse-structure decoder paths |
| `SIMANY_TRELLIS2_SEED`, `SIMANY_TRELLIS2_PIPELINE_TYPE` | `42`, `512` | `s4_trellis2` | seed; `512`, `1024`, `1024_cascade` or `1536_cascade` |
| `SIMANY_SAM3D_CONFIG` | `third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml` | `s4_sam3d` | SAM 3D Objects pipeline config |
| `SIMANY_RVG_DIR` | `$ROOT/third_party/ReconViaGen` | `s4_reconviagen` | ReconViaGen checkout |
| `SIMANY_SAM3_CKPT` | unset | `common.resolve_sam3_ckpt` | absolute path to `sam3.pt`; otherwise the HF cache is searched (`HF_HOME`) |
| `SIMANY_MUJOCO_MENAGERIE_ROOT` | `$ROOT/third_party/mujoco_menagerie` | `robo/rigs/pi05_rig.py` | MuJoCo Menagerie checkout |
| `SIMANY_DROID_RLDS_DIR`, `SIMANY_DROID_RAW_ROOT` | `data/droid/droid_100/1.0.0`, `data/droid/raw` | `run/fetch_droid_raw.py`, `agents/recon/droid_extract.py` | DROID RLDS episodes and raw downloads |
| `SIMANY_BEHAVIOR_FLOWS_ROOT` | see module | `agents/recon/behavior_extract.py` | BEHAVIOR episode root |
| `SIMANY_DEMO_SCALE` | `0.5` | `interface/demo_movie.py` | render scale vs intrinsics (0.5 -> 876x584, 1.0 -> full res 1752x1168) |
| `SIMANY_DEMO_EXCLUDE` | empty | `interface/demo_movie.py` | `idx,idx` — object indices to keep as original scene gaussians in the demo |
| `SIMANY_DEMO_F1MIN`, `SIMANY_DEMO_HOLEMIN`, `SIMANY_DEMO_OBJ_SRC` | `0.70`, `15.0`, `asset` | `interface/demo_movie.py` | demo curation thresholds and object source |
| `SIMANY_PI05_CKPT` | `pi05_droid_jointpos` | `run/pi05_serve.sh` | checkpoint subpath under `openpi-assets-simeval` |
| `SIMANY_PI05_CONFIG` | `pi05_droid_jointpos` | `run/pi05_serve.sh` | openpi policy config name |
| `SIMANY_POLICY_ID`, `SIMANY_OPENPI_ROOT`, `SIMANY_OPENPI_COMMIT`, `SIMANY_OPENPI_PYTHON`, `SIMANY_POLICY_IDENTITY_FILE`, `SIMANY_CHECKPOINT_CACHE_ROOT` | required | `run/pi05_serve_bound.sh`, `robo/policy/bound_server.py` | policy id from `configs/policies/`, pinned openpi worktree/commit/interpreter, identity receipt path, checkpoint cache root |

Non-`SIMANY` knobs used by the launchers: `RESUME=1` (skip stages whose
output exists, via `done_skip` in `env.sh`), `LAMA_MODEL` (path to a LaMa
checkpoint; `run_inpaint.sh` auto-detects a cached one), `GS_ITERS`,
`MAX_FRAMES` and `POSE_BACKEND` (`run_video2sim.sh`), `OPENPI_ROOT` and
`OPENPI_DATA_HOME` (`run/pi05_serve.sh`), `HARMONIZER_SOURCE`,
`HARMONIZER_CHECKPOINT`, `HARMONIZER_SOCKET`, `HARMONIZER_RESOLUTION`,
`HARMONIZER_TIMESTEP` (`run/serve_harmonizer.sh`), `PHIROOM_ROOT` and
`PHIROOM_CONTROL_PY` (`phiroom`), and the build/runtime variables `env.sh`
sets itself (`HF_HUB_OFFLINE`, `CC`/`CXX`/`CUDAHOSTCXX`, `NVCC_APPEND_FLAGS`,
`TORCH_CUDA_ARCH_LIST`); `HF_HOME` and `TORCH_HOME` are honoured as-is.
