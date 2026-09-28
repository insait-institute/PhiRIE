# Data, weights and outputs

Four repo-root directories hold everything that is not code: `data/`
(datasets), `checkpoints/` (downloaded weights), `third_party/` (vendored
checkouts) and `outputs/` (run products). Per [`.gitignore`](../.gitignore),
their contents are not tracked — the only exceptions are
`third_party/bddl_data/` and the READMEs. This page records what lives where,
how it got there, and which environment variables point at it.

## 1. Datasets

### ScanNet++ v2

Default root: `/data/ScanNetpp` (an autofs mount on the cluster). Override
with `SIMANY_SCANNETPP_ROOT`; [`run/env.sh`](../run/env.sh) exports it and
[`agents/core/common.py`](../agents/core/common.py) reads it (the legacy
`SIMF_` prefix still works, with a deprecation warning).

Per scene, the pipeline uses exactly these paths under
`$SIMANY_SCANNETPP_ROOT/data/<scene_id>/`:

| path | role |
|---|---|
| `dslr/resized_undistorted_images/` | posed RGB frames (1752×1168), the pipeline's image input and render ground truth |
| `dslr/colmap/images.txt` | camera poses — OpenCV w2c, used directly |
| `dslr/nerfstudio/transforms_undistorted.json` | PINHOLE intrinsics (note: its c2w poses are in a permuted OpenGL frame; the pipeline takes poses from `images.txt` instead) |
| `scans/mesh_aligned_0.05.ply` | GT scan mesh (metric, z-up — the world frame of everything downstream) |
| `scans/segments.json`, `scans/segments_anno.json` | GT instance annotations — GT-driven (`run_factory.sh`) mode only; `SIMANY_AUTO=1` never touches them |

The evaluation fleet runs over the 50 scenes of
`$SIMANY_SCANNETPP_ROOT/splits/nvs_sem_val.txt` (note: the
[`run/slurm/fleet_*.sbatch`](../run/slurm) scripts hardcode the cluster
path to that split file). This
camera/pose/intrinsics combination is the verified render recipe (PSNR 33+
against the GT frames; see the header of
[`agents/core/common.py`](../agents/core/common.py)).

### Prebuilt 3DGS splats

Default root: `/data/ScanNetppv2_gsplat/splats`, override with
`SIMANY_SPLATS_ROOT`. One Inria-format `<scene_id>.ply` per scene
(GaussianWorld MCMC, 1.5M gaussians, SH degree 3), stored **in the same
mesh/colmap world frame as the scan mesh** — no extra transform is needed to
composite splat renders with mesh-derived geometry. `meta.csv` one level up
records per-scene training PSNR. `common.py` resolves
`SPLAT_PLY = $SIMANY_SPLATS_ROOT/<scene_id>.ply`.

Cluster gotcha: `/data` is autofs. An `ls` on a deep path can return empty
before the mount is triggered; `ls /data/ScanNetpp` (one level up) first
mounts it reliably.

### The in-repo `data/` directory

`data/` at the repo root is an empty, gitignored mount point (`data/*` in
[`.gitignore`](../.gitignore)). On this cluster the datasets live at the
`/data/...` paths above and the defaults point there; on another machine,
put or symlink the datasets wherever you like and set
`SIMANY_SCANNETPP_ROOT`/`SIMANY_SPLATS_ROOT` — nothing in the pipeline
hardcodes a location.

## 2. Weights and caches

### `checkpoints/` and the `weights -> checkpoints` symlink

`checkpoints/` (gitignored) holds downloaded weights that do not live in a
standard cache. Currently that is `checkpoints/dreamsim/` (~1.9 GB): the
`dreamsim` perceptual metric — a dependency of the vendored ReconViaGen —
downloads its weights into `$CWD/weights`. Since every launcher `cd`s to the
repo root ([`run/env.sh`](../run/env.sh) does it for `python -m`), a
repo-root compatibility symlink `weights -> checkpoints` redirects that
download into `checkpoints/`. Both names are gitignored; keep the symlink.

### Hugging Face hub cache

All neural-model bridges load `from_pretrained` with `HF_HUB_OFFLINE=1` by
default (set in [`run/env.sh`](../run/env.sh) and again via
`os.environ.setdefault` in the stage modules), so the checkpoints must
already sit in the HF hub cache:

| HF model id | used by |
|---|---|
| `facebook/sam3` | [`models/s1_segment.py`](../models/s1_segment.py) (checkpoint path inside the cache is hardcoded there), `agents/discover/auto_segment`, `agents/edit/inpaint_masks` — all under the sam3 env |
| `Qwen/Qwen-Image-Edit-2509` | [`agents/edit/inpaint_qwen.py`](../agents/edit/inpaint_qwen.py) (needs torch ≥ 2.5 → sam3 env; see [ENVIRONMENTS.md](ENVIRONMENTS.md)) |
| `microsoft/TRELLIS-image-large` | [`models/s4_trellis.py`](../models/s4_trellis.py) |
| `depth-anything/DA3METRIC-LARGE` | [`models/s2_depth.py`](../models/s2_depth.py) |

Cluster gotcha: on hala, `~/.cache` is a symlink to **node-local**
`/scratch`, so these caches are invisible from other nodes. When running
HF-cached stages off-hala, set `HF_HOME` to a shared-storage path
holding the needed checkpoints.

### LaMa (image-inpainting fallback)

`inpaint_qwen.py` falls back to LaMa when Qwen is unavailable or fails.
`big-lama.pt` is prefetched in `~/.cache/torch/hub/checkpoints/`;
[`run/run_inpaint.sh`](../run/run_inpaint.sh) globs for it and exports
`LAMA_MODEL` so GPU nodes without egress still work.

### pi0.5 policy checkpoints (robot layer)

The closed-loop evaluation ([`robo/eval/pi05_eval.py`](../robo/eval/pi05_eval.py))
talks to an openpi websocket policy server started by
[`run/pi05_serve.sh`](../run/pi05_serve.sh). That script runs from the openpi
fork at `${OPENPI_ROOT}` (github.com/xuningy/openpi — it has
the `pi05_droid_jointpos` config that mainline openpi lacks) and sets
`OPENPI_DATA_HOME=${OPENPI_DATA_HOME}`. Checkpoints, predownloaded
under `${OPENPI_DATA_HOME}/openpi-assets-simeval/`:

- `pi05_droid_jointpos/` — the zero-shot DROID joint-position policy
  (12.4 GB), the default.
- `droid_pi05_jointpos_with_web_and_sim/80000` — the sim-co-trained variant;
  select it with `SIMANY_PI05_CKPT`, and set
  `SIMANY_PI05_CONFIG=pi05_droid_jointpos_sim` (fork-only config for its
  unpadded 8-dim action head).

`pi05_serve.sh` rsyncs the checkpoint to node-local scratch before serving:
orbax's scattered-read restore pattern has hung indefinitely against the
CephFS copy, while a plain copy of the same 12 GB takes seconds.

## 3. `third_party/` inventory

Everything here is gitignored except `bddl_data/` and
[`third_party/README.md`](../third_party/README.md).
[`run/setup_env.sh`](../run/setup_env.sh) re-clones `TRELLIS/` if missing;
the rest are manual clones/installs, documented below so they can be
restored by hand.

**TRELLIS/** — Microsoft's image-to-3D generative model, the default
per-object asset generator. [`models/s4_trellis.py`](../models/s4_trellis.py)
inserts the checkout on `sys.path` (`common.TRELLIS_DIR`) and loads the
`microsoft/TRELLIS-image-large` weights from the HF cache — the checkout is
code only. Cloned automatically by `run/setup_env.sh` (`git clone --depth 1
https://github.com/microsoft/TRELLIS.git`).

**ReconViaGen/** — multi-view image-to-3D (VGGT-conditioned TRELLIS,
arXiv 2510.23306). Used by
[`models/s4_reconviagen.py`](../models/s4_reconviagen.py) and by
[`agents/assets/factory_hybrid.py`](../agents/assets/factory_hybrid.py),
which generates both a TRELLIS and an RVG asset per object and keeps the one
with the lower registration residual. Manual clone; it ships its own
`wheels/` (vggt) that must also be on `sys.path`, needs BiRefNet + kornia,
and its `dreamsim` dependency is what populates `checkpoints/dreamsim/`.

**MaskClustering/** — CVPR 2024 class-agnostic 3D instance discovery, the
discovery baseline.
[`agents/baselines/maskclustering.py`](../agents/baselines/maskclustering.py)
stages ScanNet++ scenes into `third_party/MaskClustering/data/scannetpp/` in
their expected layout, runs their code, and converts the result back to our
`auto_instances.npz` format. Manual clone; runs in its own `.venv-mc` env
(incompatible pins — see [ENVIRONMENTS.md](ENVIRONMENTS.md)).

**FlashSplat/** — ECCV 2024 optimal 2D-mask-to-3DGS segmentation
(arXiv 2409.08270), the object-removal baseline. Manual clone, kept as the
reference implementation:
[`agents/baselines/flashsplat.py`](../agents/baselines/flashsplat.py)
reimplements their per-gaussian weight accumulation on gsplat (their custom
CUDA rasterizer has no prebuilt wheels and cannot be source-built on this
cluster; the reimplementation is verified against a float64 reference via
`--selftest`).

**BEHAVIOR-1K/** — the BEHAVIOR-1K monorepo: OmniGibson simulator + BDDL
toolchain. Installed into the `behavior1k` conda env by
[`robo/sim/omnigibson_bridge/install_omnigibson.sbatch`](../robo/sim/omnigibson_bridge/install_omnigibson.sbatch)
(runs their `setup.sh --omnigibson --bddl --dataset`). Used by
[`robo/sim/export_omnigibson.py`](../robo/sim/export_omnigibson.py) (synset
data from `BEHAVIOR-1K/bddl3/bddl/generated_data`) and by everything in
[`robo/sim/omnigibson_bridge/`](../robo/sim/omnigibson_bridge).

**bddl_data/** — BDDL activity definitions (`activity_definitions/`, 1000+
tasks) plus `generated_data/` (`synsets.csv` etc.). **Tracked in git** — it
backs
[`agents/eval/behavior1k_coverage.py`](../agents/eval/behavior1k_coverage.py)
(paper Sec V.H), which is pure text parsing and must work from a bare clone
with no simulator installed.

**behavior1k_datasets/** — OmniGibson asset datasets (~36 GB:
`behavior-1k-assets/`, `omnigibson-robot-assets/`, `omnigibson.key`,
`2026-challenge-task-instances/`), downloaded by the BEHAVIOR-1K `setup.sh`
run in the install sbatch. Also contains `phiroom_custom/` — our exported
twins converted to a custom OmniGibson USD dataset by
[`robo/sim/omnigibson_bridge/convert_assets.py`](../robo/sim/omnigibson_bridge/convert_assets.py)
and loaded by name in `import_and_run.py`.

**mujoco_menagerie/** — DeepMind's MuJoCo robot model zoo. Manual clone;
[`robo/rigs/pi05_rig.py`](../robo/rigs/pi05_rig.py) loads
`franka_emika_panda/panda_nohand.xml` and `robotiq_2f85/2f85.xml` from it to
build the DROID-style rig.

## 4. `outputs/` anatomy

Everything under `outputs/` is gitignored. One directory per (scene, mode)
run, named by scene id plus a mode suffix:

| directory | mode | launcher |
|---|---|---|
| `<scene>` | SimFoundry-reproduction baseline, single frame (paper row D) | [`run/run_simfoundry.sh`](../run/run_simfoundry.sh) |
| `<scene>_factory` | GT-driven asset factory | [`run/run_factory.sh`](../run/run_factory.sh) |
| `<scene>_auto` | fully automatic, no GT (`SIMANY_AUTO=1`) | [`run/run_auto.sh`](../run/run_auto.sh) |
| `<scene>_full` | furniture tier (`SIMANY_FULL=1`) | [`run/slurm/fleet_full.sbatch`](../run/slurm/fleet_full.sbatch) |
| `<scene>_rowC`, `<scene>_rowC2` | ablation: derived mesh + auto discovery | [`run/slurm/ablation_rowC_scene.sbatch`](../run/slurm/ablation_rowC_scene.sbatch) |

The launchers set `SIMANY_OUT` accordingly; every stage reads and writes
only under it. Inside a factory/auto run, the big products are:

- **`objects/obj_XX/`** — one directory per discovered object: `rgba.png`
  (best-view crop fed to the generator), `trellis/` and `rvg/` (the two
  candidate assets), `hybrid.json` (winner decision), the canonical asset
  `trellis_gs.ply` / `trellis_mesh.ply` / `mesh_sim.obj`, `aligned.json`
  (Sim(3) registration `T`), `gt_points.ply` (registration/eval target),
  `collision/` (CoACD parts), `physics.json` and `object.urdf`. Scene-level
  roll-ups sit beside them (`objects/aligned_all.json`,
  `objects/hybrid_all.json`).
- **`inpaint/`** — Gaussian-native removal + completion:
  `clean_background.ply` (the object-free background splat, drop-in
  Inria-format ply), `fill_gaussians.npz`, `fill_stats.json`,
  `compare_*.jpg` before/after sheets, plus the FlashSplat-baseline
  artifacts (`flashsplat_report.json`, `flashsplat_removal_union_idx.npy`).
- **`sim_export/`** — simulator packaging: `scene.xml` (MuJoCo MJCF),
  `isaac_manifest.json`, `mujoco_settle.json`, and — for the scenes with a
  task suite — `pi05_tasks.json`.
- **reports** — `report.json`, `render_metrics*.json`, `eval_vs_gt.json`
  (auto runs; written by a separate `agents.eval.eval_vs_gt` pass),
  `drop_v2.json` (frame-corrected drop-stability replay), `crops_sheet.png`,
  and `timings.txt` (appended by `stage_timed` in `run/env.sh`; the
  efficiency table is computed from it).

Cross-scene directories: `outputs/pi05_runs/<run>_<scene>_composite/`
(raster runs pool scenes into `<run>_raster/`) hold closed-loop pi0.5
evaluation runs written by
[`robo/eval/pi05_eval.py`](../robo/eval/pi05_eval.py) via
[`run/slurm/pi05_closedloop.sbatch`](../run/slurm/pi05_closedloop.sbatch);
`outputs/omnigibson_export/` (`objects/`, `manifest.json`, `results/`) is
the Tier-1 OmniGibson bridge output of `export_omnigibson.py`.

## 5. Getting the code running elsewhere

1. `git clone https://github.com/RunyiYang/PhiRoom` (the repo of the SimAny
   system).
2. Create the main env (`.venv`, conda python 3.11) and run
   [`run/setup_env.sh`](../run/setup_env.sh). `ROOT` is derived from the
   script's own location (override with `SIMANY_ROOT`). It installs torch
   2.4.1+cu124 + the pipeline deps, clones `third_party/TRELLIS`, and ends
   with an import smoke test.
3. The other two interpreters (sam3 env for SAM3/Qwen, cp310 env for gsplat)
   are not created by that script — see [ENVIRONMENTS.md](ENVIRONMENTS.md)
   for what they need, and point `SIMANY_SAM3_PY` / `SIMANY_GSPLAT_PY` at
   them.
4. Set the two dataset roots: `SIMANY_SCANNETPP_ROOT` (ScanNet++ v2, layout
   of §1) and `SIMANY_SPLATS_ROOT` (per-scene 3DGS plys).
5. Prefetch weights once with `HF_HUB_OFFLINE=0` (SAM3, Qwen-Image-Edit-2509,
   TRELLIS-image-large, DA3METRIC-LARGE) plus `big-lama.pt` into
   `~/.cache/torch/hub/checkpoints/`; then the offline default works.
   Clone the remaining `third_party/` checkouts you need (§3).
6. What fails without which piece: no sam3 env → `models.s1_segment`,
   `agents.discover.auto_segment` and the inpaint mask/Qwen stages;
   no gsplat env → every render stage (`agents.render.*`,
   `agents.edit.inpaint_fill`); missing HF cache with `HF_HUB_OFFLINE=1` →
   immediate `from_pretrained` file-not-found; no splat ply → rendering and
   inpainting (geometry-only stages still run); no GT `scans/` → use
   `SIMANY_AUTO=1` with `SIMANY_MESH_SRC=derived` (the rowC configuration);
   pi0.5 evaluation additionally needs the openpi fork + checkpoint of §2.
   `bash run/smoke_imports.sh` checks that every module still imports
   (known environment-dependent failures: viser, omnigibson).
