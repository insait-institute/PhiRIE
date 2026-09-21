# SimAny code architecture

Code map for the repository after the 2026-08-04 reorganization
(commit `b1fef9a`, "reorganize into the PhiRoom layout"). SimAny turns posed
RGB images of a real indoor scene + its mesh + 3D Gaussian splat into an
editable, photorealistic, simulation-ready digital twin. This page covers
where the code lives, how it executes, what each stage reads and writes, how
old module paths map to new ones, and every `SIMANY_*` environment variable.

Related pages: [ENVIRONMENTS.md](ENVIRONMENTS.md) (why there are three python
envs), [CONTRIBUTIONS.md](CONTRIBUTIONS.md) (what is measured),
[BASELINES.md](BASELINES.md) (comparison methods).

## 1. Code map

```
SimAny/
├── agents/                     generation pipeline: scan -> sim-ready twin
│   ├── core/
│   │   └── common.py           shared paths/env(), camera loaders, vocabulary,
│   │                           GT/auto instance loaders, gaussian IO + gsplat render helpers
│   ├── discover/               instance discovery (frame selection, lifting, GT enumeration)
│   │   ├── s0_select_frame.py       pick the representative frame (zero-shot mode)
│   │   ├── s3_lift.py               lift SAM3 instances to world-frame clouds + RGBA crops
│   │   ├── auto_segment.py          GT-free 3D instance discovery: SAM3 on sampled frames,
│   │   │                            mesh-backprojected 3D merging (sam3 env)
│   │   ├── derive_mesh_from_splat.py  ablation: scan-mesh substitute via TSDF fusion of
│   │   │                            splat-rendered depth (no dataset mesh)
│   │   ├── factory_prepare.py       benchmark mode: enumerate GT instances, best-view crops
│   │   └── factory_refine_masks.py  refine projected masks with SAM3 image evidence (sam3 env)
│   ├── assets/                 per-object asset registration + physics annotation
│   │   ├── s5_align.py              Sim(3) registration: yaw grid + clipped one-way chamfer
│   │   ├── s6_physics.py            CoACD convex decomposition + VLM physics params -> URDF
│   │   ├── factory_align.py         register assets to GT submesh; quality tiers A/B/C
│   │   └── factory_hybrid.py        TRELLIS-vs-ReconViaGen winner selection per object
│   ├── edit/                   gaussian-native object removal + background completion
│   │   ├── inpaint_prepare.py       removal sets, support planes, view lists (venv, CPU)
│   │   ├── inpaint_masks.py         SAM3 union masks for removal (sam3 env)
│   │   ├── inpaint_qwen.py          erase objects in views: Qwen-Image-Edit, LaMa fallback
│   │   └── inpaint_fill.py          carve + plane fill + photometric refine (mini-viewer env)
│   ├── render/                 photoreal rendering of simulator states
│   │   ├── s8_render.py             stage 8: photo-vs-twin, physics video, asset gallery
│   │   └── gsplat_sim_render.py     render any pose log (MuJoCo/Isaac/PyBullet) as splats
│   ├── eval/                   benchmark scoring and aggregation
│   │   ├── eval_instances.py        discovery precision/recall/F1 vs GT (vertex IoU)
│   │   ├── eval_vs_gt.py            registered assets re-scored vs independent GT submesh
│   │   ├── factory_report.py        drop-test stability + yield report
│   │   ├── factory_eval_render.py   PSNR/SSIM/LPIPS on held-out views (mini-viewer env)
│   │   ├── verify_removal_render.py removal verification stage 1: clean renders (mini-viewer)
│   │   ├── verify_removal_check.py  removal verification stage 2: SAM3 re-detection (sam3 env)
│   │   ├── behavior1k_coverage.py   BEHAVIOR-1K BDDL activity coverage analysis
│   │   ├── aggregate_results.py     fleet results -> paper-style summary tables
│   │   └── make_paper_tables.py     LaTeX tables in docs/paper/tables/ from outputs/
│   ├── baselines/              comparison methods re-run on our inputs
│   │   ├── flashsplat.py            FlashSplat gaussian-selection removal baseline
│   │   ├── maskclustering.py        MaskClustering 3D instance discovery baseline
│   │   └── mc_masks.py              SAM3 stand-in for MaskClustering's 2D segmenter
│   └── single_image/           single-photo variant: SHARP feed-forward splat replaces the scan
│       ├── sharp_s2_lift.py         lift from SHARP-splat depth (mirrors s3_lift)
│       ├── sharp_render_depth.py    render SHARP splat at the input photo's camera (mini-viewer)
│       ├── sharp_render_object_views.py  synthetic near-field views for ReconViaGen (mini-viewer)
│       ├── sharp_hybrid.py          TRELLIS-vs-RVG winner selection, SHARP variant
│       └── sharp_ply_meta.py        recover the SHARP camera from a SHARP-saved .ply
├── models/                     bridges to external neural models (runnable stages;
│   │                           vendored code in third_party/, weights in checkpoints/)
│   ├── s1_segment.py           SAM3 text-prompted segmentation (sam3 env, standalone)
│   ├── s2_depth.py             metric monocular depth (DA3) + robust scale vs mesh depth
│   ├── s4_trellis.py           TRELLIS image-to-3D: mesh + gaussians per object
│   └── s4_reconviagen.py       ReconViaGen multi-view (VGGT-conditioned TRELLIS) alternative
├── robo/                       robot layer: policy evaluation + simulator export
│   ├── envs/
│   │   └── pi05_env.py         DROID-convention MuJoCo env (15 Hz) over a SimAny scene
│   ├── rigs/
│   │   └── pi05_rig.py         Franka Panda + Robotiq 2F-85 rig assembled via MjSpec
│   ├── tasks/
│   │   └── pi05_tasks.py       pick-and-place suite generation + staged scoring
│   ├── rendering/
│   │   └── pi05_render.py      photoreal composite observations per control tick
│   ├── eval/
│   │   └── pi05_eval.py        websocket policy client, closed-loop episode runner
│   └── sim/                    simulator export and dynamics
│       ├── s7_sim.py           stage 7: compose twin in PyBullet, settle, dynamics demo
│       ├── export_mjcf.py      MuJoCo MJCF + Isaac-Lab manifest export + headless settle test
│       ├── export_omnigibson.py package sim-ready objects for BEHAVIOR-1K/OmniGibson
│       ├── omnigibson_bridge/  OmniGibson-side import/convert/demo scripts + sbatch files
│       ├── redrop.py           re-run drop tests with the COM->link frame correction
│       ├── mujoco_video.py     MuJoCo rollout video from the exported scene.xml
│       └── viewer_settle.py    physics backend for interface/viewer.py
├── interface/                  human-facing tools
│   ├── viewer.py               viser multi-scene digital-twin editor (default port 8090)
│   ├── mujoco_live_viewer.py   live MuJoCo sim rendered as gaussians in the browser (port 8091)
│   ├── demo_movie.py           demo-film segments (storyboard: docs/DEMO_STORYBOARD.md)
│   └── demo_session.py         scripted interactive-session demo (pybullet, CPU)
├── run/                        launchers
│   ├── env.sh                  shared env: paths, three interpreters, run helpers (source it)
│   ├── setup_env.sh            installs the main .venv
│   ├── run_simfoundry.sh       SimFoundry-reproduction baseline (single frame, monodepth, no GT)
│   ├── run_factory.sh          GT-driven benchmark mode
│   ├── run_auto.sh             fully automatic mode (no GT annotations)
│   ├── run_inpaint.sh          object removal + background completion
│   ├── pi05_serve.sh           serve the pi0.5 openpi policy (websocket, port 8000)
│   ├── smoke_imports.sh        import-smoke harness (one process per module)
│   └── slurm/                  sbatch launchers (fleet runs, ablations, demo, pi0.5)
├── tests/
│   └── test_align_synthetic.py dataset-free synthetic stress suite for registration
├── coding_agents/              AI coding-agent traces: memory/, history/, skills/
├── data/                       datasets (contents gitignored)
├── checkpoints/                downloaded weights (gitignored; dreamsim/ lives here;
│                               top-level weights -> checkpoints compat symlink, because
│                               dreamsim writes to CWD/weights)
├── third_party/                vendored checkouts (gitignored except bddl_data/, README):
│                               TRELLIS, ReconViaGen, MaskClustering, FlashSplat,
│                               BEHAVIOR-1K, bddl_data, behavior1k_datasets, mujoco_menagerie
├── outputs/                    per-scene result trees (see section 3)
└── docs/                       this page, PIPELINE, ROBOT, DATA_AND_WEIGHTS,
    │                           CONTRIBUTIONS, BASELINES, ENVIRONMENTS,
    │                           PAPER_NOTES, PAPER_REVISIONS, DEMO_STORYBOARD
    ├── related/                reference material on cited systems
    └── paper/                  LaTeX source + build.sh (do not rename anything in here:
                                "SimFoundry" in the paper cites the prior system arXiv:2606.28276)
```

## 2. How execution works

The packages are **not pip-installed** into any environment
(`pyproject.toml` dependencies are deliberately empty — the pipeline spans
three mutually incompatible torch environments). Stage modules run with the
repo root on `sys.path`:

```bash
cd /group/worldcept/code/SimAny
python -m agents.assets.s5_align        # works: python -m puts CWD on sys.path
python agents/assets/s5_align.py        # fails: package imports unresolved
```

From anywhere else, `export PYTHONPATH=/group/worldcept/code/SimAny` first
(this is what `run/smoke_imports.sh` does).

### run/env.sh

Every launcher sources [`run/env.sh`](../run/env.sh), which resolves paths,
`cd`s to the repo root, and defines one run helper per python environment:

| Helper | Interpreter variable | Default interpreter | Environment |
|---|---|---|---|
| `run` | `VENV` (override: `SIMANY_PY`) | `$ROOT/.venv/bin/python` | main pipeline: torch 2.4.1+cu124, open3d/trimesh/coacd/xformers |
| `run_sam3` | `SAM3PY` (override: `SIMANY_SAM3_PY`) | `/group/streetsplat/worldcept/.envs/sam3/bin/python` | SAM3 + Qwen-Image-Edit: torch 2.10 |
| `run_gs` | `MVPY` (override: `SIMANY_GSPLAT_PY`) | `/group/worldcept/code/affordancept/.envs/mini-viewer/bin/python` | gsplat CUDA rendering (cp310) |
| `run_qwen` | `QWEN_PY` (override: `QWEN_PY` — no `SIMANY_` prefix) | `$VENV` | set `QWEN_PY=$SAM3PY` on nodes with torch>=2.5 + enough VRAM |

Usage: `run <package>.<module> [args...]`, e.g.

```bash
source run/env.sh
run agents.assets.s5_align
run_sam3 models.s1_segment --image ... --out-dir ... --prompts bottle mug
run_gs agents.render.s8_render
run_qwen agents.edit.inpaint_qwen
```

Why three environments is unavoidable is documented in
[ENVIRONMENTS.md](ENVIRONMENTS.md). `env.sh` also sets `HF_HUB_OFFLINE=1`,
probes for `g++-12`/`g++-13` (gsplat JIT), and pins
`TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"` to match the prebuilt gsplat JIT cache.

Two more `env.sh` conveniences:

- `RESUME=1` — the `done_skip` helper lets a launcher skip a stage whose
  output already exists.
- `stage_timed` appends `<stage> <seconds>` to `$SIMANY_OUT/timings.txt`,
  which the efficiency table is computed from.

### Launchers

| Script | Mode |
|---|---|
| `run/run_simfoundry.sh` | SimFoundry-reproduction baseline: one representative frame, monocular metric depth, no GT (formerly `run_scene.sh`) |
| `run/run_factory.sh` | GT-driven asset factory (benchmark mode) |
| `run/run_auto.sh` | fully automatic (no GT annotations); exports MJCF + Isaac manifest |
| `run/run_inpaint.sh` | gaussian-native removal + background completion for a factory scene |
| `run/pi05_serve.sh` | openpi pi0.5 policy server (run on a GPU node) |
| `run/smoke_imports.sh` | imports every module in a fresh process; diff before/after refactors |
| `run/slurm/*.sbatch` | fleet/ablation/demo/pi0.5 batch jobs; all get `env.sh` either directly or via the `run_*.sh` launchers they invoke |

## 3. Data flow for one scene

All stages communicate through the filesystem under one output tree, selected
by `SIMANY_OUT` (default `outputs/$SIMANY_SCENE`). The launchers use the
conventions `outputs/<scene>` (zero-shot), `outputs/<scene>_factory`
(GT-driven), `outputs/<scene>_auto` (automatic); other suffixes seen in
`outputs/` (`_full`, `_rowC`, ...) are just different `SIMANY_OUT` values set
by the slurm launchers. Scene *inputs* always come from the ScanNet++ tree
(`SIMANY_SCANNETPP_ROOT`) and the splat (`SIMANY_SPLATS_ROOT/<scene>.ply`),
resolved in `agents/core/common.py`.

### SimFoundry-reproduction baseline (`run_simfoundry.sh`) — `outputs/<scene>/`

| Stage | Reads | Writes |
|---|---|---|
| `agents.discover.s0_select_frame` | GT centroids + mesh (frame choice only) | `frame/rep_frame.json` + a copy of the chosen image |
| `models.s1_segment` (sam3) | `frame/<image>` | `masks/` (`masks.npz`, `detections.json`, `overlay.png`) |
| `models.s2_depth` | `frame/`, scan mesh (scale bridge) | `depth/` (`depth.npz`, `depth_vis.png`, `stats.json`) |
| `agents.discover.s3_lift` | `masks/`, `depth/` | `objects/objects.json`, per object `objects/obj_XX/` (`rgba.png`, `points.ply`, `meta.json`) |
| `models.s4_trellis` | `obj_XX/rgba.png` | `obj_XX/` (`trellis_mesh.ply`, `mesh_sim.ply`/`.obj`, `trellis_gs.ply`) |
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
| `models.s4_trellis` | `obj_XX/rgba.png` | `obj_XX/trellis_*.ply`, `mesh_sim.*` |
| `agents.assets.factory_align` | asset mesh + GT instance submesh | `obj_XX/aligned.json` (tiers A/B/C), `objects/aligned_all.json` |
| `agents.assets.s6_physics` | `obj_XX/mesh_sim.*` | `obj_XX/collision/`, `physics.json`, `object.urdf` |
| `agents.eval.factory_report` | all non-rejected assets | `report.json`, `crops_sheet.png` |
| `agents.eval.factory_eval_render` (gsplat) | held-out DSLR views + splat + assets | `render_metrics.json`, `eval_photo_bg_twin.jpg` |

Optional hybrid slot (slurm `fleet_hybrid.sbatch`):
`models.s4_reconviagen` writes `obj_XX/rvg/` (multi-view crops +
`rvg_mesh.ply`/`rvg_gs.ply` + `aligned.json`), and
`agents.assets.factory_hybrid` picks the winner per object
(`obj_XX/hybrid.json`, `objects/hybrid_all.json`; the pre-hybrid registration
is kept as `objects/aligned_all_prehybrid.json`, the TRELLIS candidate under
`obj_XX/trellis/`).

### Automatic mode (`run_auto.sh`) — `outputs/<scene>_auto/`, `SIMANY_AUTO=1`

Discovery is replaced by `agents.discover.auto_segment` (sam3), which writes
`auto_instances.npz`; `factory_prepare` then consumes those instances instead
of GT, and the rest of the factory chain runs unchanged. At the end,
`robo.sim.export_mjcf --test` writes `sim_export/`
(`scene.xml`, `isaac_manifest.json`, `mujoco_settle.json`) and settle-tests
the MJCF headlessly. Evaluation extras (run separately):
`agents.eval.eval_instances` (discovery F1 vs GT) and
`agents.eval.eval_vs_gt` (`eval_vs_gt.json`).

### Inpainting (`run_inpaint.sh`) — `<out>/inpaint/`

| Stage | Writes |
|---|---|
| `agents.edit.inpaint_prepare` (venv, CPU) | `inpaint/obj_XX/` (`removal_idx.npy`, `plane.json`, `views.json`, `proj_masks.npz`) |
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
per-episode mp4s to its `--out` directory.

## 4. Migration table

Two renames happened: the flat `simfoundry/` scripts became the `simany`
package on 2026-07-27 (git commit `e2fbc1b` records this state), and the
`simany` package fanned out into `agents`/`models`/`robo`/`interface` on
2026-08-04 (commit `b1fef9a`). Both mappings below are taken from git rename
detection on those commits. "—" means the module first appears in git at the
interim layout (no flat-era path recorded).

| Flat (`simfoundry/`, until 2026-07-27) | Interim (`simany.*`, until 2026-08-04) | Current module |
|---|---|---|
| `simfoundry/common.py` | `simany.core.common` | `agents.core.common` |
| `simfoundry/s0_select_frame.py` | `simany.discover.s0_select_frame` | `agents.discover.s0_select_frame` |
| `simfoundry/s1_segment.py` | `simany.discover.s1_segment` | `models.s1_segment` |
| `simfoundry/s2_depth.py` | `simany.discover.s2_depth` | `models.s2_depth` |
| `simfoundry/s3_lift.py` | `simany.discover.s3_lift` | `agents.discover.s3_lift` |
| `simfoundry/auto_segment.py` | `simany.discover.auto_segment` | `agents.discover.auto_segment` |
| `simfoundry/derive_mesh_from_splat.py` | `simany.discover.derive_mesh_from_splat` | `agents.discover.derive_mesh_from_splat` |
| `simfoundry/factory_prepare.py` | `simany.discover.factory_prepare` | `agents.discover.factory_prepare` |
| `simfoundry/factory_refine_masks.py` | `simany.discover.factory_refine_masks` | `agents.discover.factory_refine_masks` |
| `simfoundry/s4_trellis.py` | `simany.assets.s4_trellis` | `models.s4_trellis` |
| `simfoundry/s4_reconviagen.py` | `simany.assets.s4_reconviagen` | `models.s4_reconviagen` |
| `simfoundry/s5_align.py` | `simany.assets.s5_align` | `agents.assets.s5_align` |
| `simfoundry/s6_physics.py` | `simany.assets.s6_physics` | `agents.assets.s6_physics` |
| `simfoundry/factory_align.py` | `simany.assets.factory_align` | `agents.assets.factory_align` |
| `simfoundry/factory_hybrid.py` | `simany.assets.factory_hybrid` | `agents.assets.factory_hybrid` |
| `simfoundry/inpaint_prepare.py` | `simany.edit.inpaint_prepare` | `agents.edit.inpaint_prepare` |
| `simfoundry/inpaint_masks.py` | `simany.edit.inpaint_masks` | `agents.edit.inpaint_masks` |
| `simfoundry/inpaint_qwen.py` | `simany.edit.inpaint_qwen` | `agents.edit.inpaint_qwen` |
| `simfoundry/inpaint_fill.py` | `simany.edit.inpaint_fill` | `agents.edit.inpaint_fill` |
| `simfoundry/s8_render.py` | `simany.render.s8_render` | `agents.render.s8_render` |
| `simfoundry/gsplat_sim_render.py` | `simany.render.gsplat_sim_render` | `agents.render.gsplat_sim_render` |
| `simfoundry/eval_instances.py` | `simany.eval.eval_instances` | `agents.eval.eval_instances` |
| `simfoundry/eval_vs_gt.py` | `simany.eval.eval_vs_gt` | `agents.eval.eval_vs_gt` |
| `simfoundry/factory_report.py` | `simany.eval.factory_report` | `agents.eval.factory_report` |
| `simfoundry/factory_eval_render.py` | `simany.eval.factory_eval_render` | `agents.eval.factory_eval_render` |
| `simfoundry/verify_removal_render.py` | `simany.eval.verify_removal_render` | `agents.eval.verify_removal_render` |
| `simfoundry/verify_removal_check.py` | `simany.eval.verify_removal_check` | `agents.eval.verify_removal_check` |
| `simfoundry/behavior1k_coverage.py` | `simany.eval.behavior1k_coverage` | `agents.eval.behavior1k_coverage` |
| `simfoundry/aggregate_results.py` | `simany.eval.aggregate_results` | `agents.eval.aggregate_results` |
| — | `simany.eval.make_paper_tables` | `agents.eval.make_paper_tables` |
| `simfoundry/baseline_flashsplat.py` | `simany.baselines.flashsplat` | `agents.baselines.flashsplat` |
| `simfoundry/baseline_maskclustering.py` | `simany.baselines.maskclustering` | `agents.baselines.maskclustering` |
| `simfoundry/baseline_mc_masks.py` | `simany.baselines.mc_masks` | `agents.baselines.mc_masks` |
| — | `simany.single_image.sharp_s2_lift` | `agents.single_image.sharp_s2_lift` |
| — | `simany.single_image.sharp_render_depth` | `agents.single_image.sharp_render_depth` |
| — | `simany.single_image.sharp_render_object_views` | `agents.single_image.sharp_render_object_views` |
| — | `simany.single_image.sharp_hybrid` | `agents.single_image.sharp_hybrid` |
| — | `simany.single_image.sharp_ply_meta` | `agents.single_image.sharp_ply_meta` |
| — | `simany.robot.pi05_env` | `robo.envs.pi05_env` |
| — | `simany.robot.pi05_rig` | `robo.rigs.pi05_rig` |
| — | `simany.robot.pi05_tasks` | `robo.tasks.pi05_tasks` |
| — | `simany.robot.pi05_render` | `robo.rendering.pi05_render` |
| — | `simany.robot.pi05_eval` | `robo.eval.pi05_eval` |
| `simfoundry/s7_sim.py` | `simany.sim.s7_sim` | `robo.sim.s7_sim` |
| `simfoundry/export_mjcf.py` * | `simany.sim.export_mjcf` | `robo.sim.export_mjcf` |
| — | `simany.sim.export_omnigibson` | `robo.sim.export_omnigibson` |
| — | `simany/sim/omnigibson_bridge/` | `robo/sim/omnigibson_bridge/` |
| `simfoundry/redrop.py` | `simany.sim.redrop` | `robo.sim.redrop` |
| `simfoundry/mujoco_video.py` | `simany.sim.mujoco_video` | `robo.sim.mujoco_video` |
| `simfoundry/viewer_settle.py` | `simany.sim.viewer_settle` | `robo.sim.viewer_settle` |
| `simfoundry/viewer.py` | `simany.viz.viewer` | `interface.viewer` |
| `simfoundry/mujoco_live_viewer.py` | `simany.sim.mujoco_live_viewer` | `interface.mujoco_live_viewer` |
| `simfoundry/demo_movie.py` | `simany.viz.demo_movie` | `interface.demo_movie` |
| `simfoundry/demo_session.py` | `simany.viz.demo_session` | `interface.demo_session` |
| `simfoundry/test_align_synthetic.py` | `simany.tests.test_align_synthetic` | `tests.test_align_synthetic` |

\* `export_mjcf.py` was rewritten at the package split, so git records
delete+add rather than a rename; the flat file did exist.

Non-module moves:

| Old | Interim | Current |
|---|---|---|
| `run_scene.sh`, `run_factory.sh`, `run_inpaint.sh` (repo root) | `scripts/` | `run/` (`run_scene.sh` later renamed `run_simfoundry.sh`) |
| `run_auto_image2sim.sh` | `scripts/run_auto.sh` | `run/run_auto.sh` |
| `*.sbatch` (repo root) | `scripts/slurm/` | `run/slurm/` |
| `setup_env.sh` | `scripts/setup_env.sh` | `run/setup_env.sh` |
| — | `scripts/env.sh`, `scripts/pi05_serve.sh` | `run/env.sh`, `run/pi05_serve.sh` |
| `paper/` (added during the flat era, 2026-07-16) | `paper/` | `docs/paper/` |
| `weights/` | `weights/` | `checkpoints/` (compat symlink `weights -> checkpoints` kept: dreamsim writes `CWD/weights`) |

Naming reminder: "SimFoundry" now refers only to the cited prior system
(arXiv:2606.28276); "PhiRoom" survives as the GitHub repo name
(github.com/RunyiYang/PhiRoom) and the package name in `pyproject.toml`. The
system and paper name is **SimAny**. Nothing inside `docs/paper/` is ever
renamed.

## 5. Environment variables

All knobs use the `SIMANY_` prefix. `agents.core.common.env(NAME)` also reads
the legacy `SIMF_<NAME>` prefix with a deprecation warning on stderr
("honoured for one release" — do not rely on it); `run/env.sh` likewise falls
back to `SIMF_SCENE`. Sources: [`run/env.sh`](../run/env.sh),
[`agents/core/common.py`](../agents/core/common.py), and the modules named
below.

| Variable | Default | Read by | Effect |
|---|---|---|---|
| `SIMANY_ROOT` | derived from `env.sh` / `common.py` location | `env.sh`, `common.py` | repo root; everything else is resolved relative to it |
| `SIMANY_SCENE` | `c50d2d1d42` | `env.sh`, `common.py` | ScanNet++ scene id (inputs) |
| `SIMANY_OUT` | `$ROOT/outputs/$SIMANY_SCENE` | `common.py`, launchers | output tree; `run_factory.sh` defaults it to `outputs/<scene>_factory`, `run_auto.sh` to `outputs/<scene>_auto` |
| `SIMANY_SCANNETPP_ROOT` | `/data/ScanNetpp` | `env.sh`, `common.py` | ScanNet++ dataset root |
| `SIMANY_SPLATS_ROOT` | `/data/ScanNetppv2_gsplat/splats` | `env.sh`, `common.py` | trained scene splats (`<scene>.ply`) |
| `SIMANY_PY` | `$ROOT/.venv/bin/python` | `env.sh` | main-pipeline interpreter (`run`) |
| `SIMANY_SAM3_PY` | sam3 env python | `env.sh` | SAM3/Qwen interpreter (`run_sam3`) |
| `SIMANY_GSPLAT_PY` | mini-viewer env python | `env.sh` | gsplat interpreter (`run_gs`) |
| `QWEN_PY` (no prefix) | `$VENV` | `env.sh` | interpreter for `run_qwen`; set to the sam3 python on torch>=2.5 nodes |
| `SIMANY_AUTO` | unset | `common.py` (`load_instances`), `factory_prepare` | `1` = GT-free mode: instances come from `auto_instances.npz` instead of GT annotations |
| `SIMANY_MESH_SRC` | unset (GT scan mesh) | `common.py` | `derived` = use `$OUT/derived_mesh.ply`; any other value = explicit mesh path. Affects only `PIPELINE_MESH_PLY` (raycasting/registration/collision); evaluation always scores against the true GT mesh |
| `SIMANY_FULL` | unset | `factory_prepare`, `auto_segment` | `1` = furniture tier: full vocabulary instead of the small-object whitelist / extra SAM3 prompts |
| `SIMANY_VIS_TOL_M` | `0.02` | `factory_prepare` | visibility raycast tolerance (m) for best-view scoring |
| `SIMANY_QWEN` | unset | `inpaint_qwen` | `0` = never use Qwen (LaMa only); `force` = skip the >90 GB VRAM gate (sequential offload) |
| `SIMANY_REQUIRE_QWEN` | unset | `inpaint_qwen` | `1` = re-raise if the Qwen pipeline fails to load instead of falling back to LaMa |
| `SIMANY_SHARD` | unset | `inpaint_qwen` | `"i/N"` = process only objects where `pos % N == i` (parallel sharding) |
| `SIMANY_DEMO_SCALE` | `0.5` | `interface/demo_movie.py` | render scale vs intrinsics (0.5 -> 876x584, 1.0 -> full res 1752x1168) |
| `SIMANY_DEMO_EXCLUDE` | empty | `interface/demo_movie.py` | `idx,idx` — object indices to keep as original scene gaussians in the demo |
| `SIMANY_DEMO_F1MIN` | `0.70` | `interface/demo_movie.py` | demo curation: minimum asset F1@20mm |
| `SIMANY_DEMO_HOLEMIN` | `15.0` | `interface/demo_movie.py` | demo curation: minimum hole-fill PSNR |
| `SIMANY_PI05_CKPT` | `pi05_droid_jointpos` | `run/pi05_serve.sh` | checkpoint subpath under `openpi-assets-simeval` |
| `SIMANY_PI05_CONFIG` | `pi05_droid_jointpos` | `run/pi05_serve.sh` | openpi policy config name |

Non-`SIMANY` knobs used by the launchers: `RESUME=1` (skip stages whose
output exists, via `done_skip` in `env.sh`), `LAMA_MODEL` (path to a LaMa
checkpoint; `run_inpaint.sh` auto-detects a cached one), and the build/runtime
variables `env.sh` sets itself (`HF_HUB_OFFLINE`, `CC`/`CXX`/`CUDAHOSTCXX`,
`NVCC_APPEND_FLAGS`, `TORCH_CUDA_ARCH_LIST`).
