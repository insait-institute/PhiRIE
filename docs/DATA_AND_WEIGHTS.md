# Data, weights and outputs

Five repo-root directories hold everything that is not code: `data/`
(datasets), `weights/` and `checkpoints/` (downloaded weights), `third_party/`
(vendored checkouts) and `outputs/` (run products). Per
[`.gitignore`](../.gitignore), their contents are not tracked — the only
exceptions are the READMEs. This page records what lives where, how it gets
there, and which environment variables point at it.

## 1. Datasets

### ScanNet++ v2

Default root: `/data/ScanNetpp`. Override with `SIMANY_SCANNETPP_ROOT`;
[`run/env.sh`](../run/env.sh) exports it and
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

The held-out render evaluation uses the official DSLR test split
(`$SIMANY_SCANNETPP_ROOT/splits/`). This camera/pose/intrinsics combination is
the verified render recipe (PSNR 33+ against the GT frames; see the header of
[`agents/core/common.py`](../agents/core/common.py)).

### Prebuilt 3DGS splats

Default root: `/data/ScanNetppv2_gsplat/splats`, override with
`SIMANY_SPLATS_ROOT`. One Inria-format `<scene_id>.ply` per scene (SH degree 3
works; degree 0 is accepted too), stored **in the same mesh/colmap world frame
as the scan mesh** — no extra transform is needed to composite splat renders
with mesh-derived geometry. `common.py` resolves
`SPLAT_PLY = $SIMANY_SPLATS_ROOT/<scene_id>.ply`. `agents.recon.gsplat_train`
produces splats in this format for scenes reconstructed from video.

### The in-repo `data/` directory

`data/` at the repo root is an empty, gitignored mount point (`data/*` in
[`.gitignore`](../.gitignore)). Put or symlink the datasets wherever you like
and set `SIMANY_SCANNETPP_ROOT`/`SIMANY_SPLATS_ROOT` — nothing in the pipeline
hardcodes a location. Two subdirectories are created by launchers:
`data/recon_scenes/` (ScanNet++-style scene directories and `splats/` written
by `run/run_video2sim.sh`, `run/run_droid_recon.sh` and
`run/run_behavior_recon.sh`) and `data/droid/` (`run/fetch_droid_raw.py`;
`SIMANY_DROID_RLDS_DIR`, `SIMANY_DROID_RAW_ROOT`).

## 2. Weights and caches

### Hugging Face hub cache

All neural-model adapters load `from_pretrained` with `HF_HUB_OFFLINE=1` by
default (set in [`run/env.sh`](../run/env.sh) and again via
`os.environ.setdefault` in the stage modules), so the checkpoints must
already sit in the HF hub cache. Prefetch them once with `HF_HUB_OFFLINE=0`;
`HF_HOME` relocates the cache.

| HF model id | used by |
|---|---|
| `facebook/sam3` | [`agents/models/s1_segment.py`](../agents/models/s1_segment.py), `agents/discover/auto_segment`, `agents/discover/factory_refine_masks`, `agents/edit/inpaint_masks` — all under the sam3 env. The checkpoint is resolved by `common.resolve_sam3_ckpt()`: `SIMANY_SAM3_CKPT` (absolute path) if set, else `$HF_HOME/hub/models--facebook--sam3/snapshots/<rev>/sam3.pt`, else the same path under `~/.cache/huggingface` |
| `Qwen/Qwen-Image-Edit-2509` | [`agents/edit/inpaint_qwen.py`](../agents/edit/inpaint_qwen.py) (needs torch >= 2.5 → sam3 env; see [ENVIRONMENTS.md](ENVIRONMENTS.md)) |
| `Qwen/Qwen2.5-7B-Instruct` | [`agents/assets/s6_physics.py`](../agents/assets/s6_physics.py) (physical-parameter annotation) |
| `microsoft/TRELLIS-image-large` | [`agents/models/s4_trellis.py`](../agents/models/s4_trellis.py) (`SIMANY_TRELLIS_MODEL` overrides; `SIMANY_DINOV2_REPO` pins a local DINOv2 source) |
| `depth-anything/DA3METRIC-LARGE` | [`agents/models/s2_depth.py`](../agents/models/s2_depth.py) |
| `Stable-X/trellis-vggt-v0-2` | [`agents/models/s4_reconviagen.py`](../agents/models/s4_reconviagen.py) |
| `facebook/VGGT-1B` | [`agents/models/vggt_scene.py`](../agents/models/vggt_scene.py) (`--backend vggt`) |

### `weights/` and `checkpoints/`

Both are gitignored. `dreamsim` — a dependency of the vendored ReconViaGen —
downloads its weights (~1.9 GB) into `$CWD/weights`; since every launcher
`cd`s to the repo root, that is `weights/dreamsim/`. `checkpoints/` holds:

- `checkpoints/vggt-omega/` — VGGT-Omega weights for
  `agents.models.vggt_scene --backend omega`. The HF repo
  `facebook/VGGT-Omega` is gated; request access, then
  `snapshot_download('facebook/VGGT-Omega', local_dir='checkpoints/vggt-omega')`
  (the note at the top of `vggt_scene.py` has the full command).
- `checkpoints/openpi_cache/` — the default `OPENPI_DATA_HOME` (see below).

Other generator weights live inside their checkouts:
`third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml`
(`SIMANY_SAM3D_CONFIG`) for SAM 3D Objects, and the TRELLIS.2 model directory
given by `SIMANY_TRELLIS2_MODEL` (with `SIMANY_DINOV3_MODEL` and
`SIMANY_SS_DECODER`).

### LaMa (image-inpainting fallback)

`inpaint_qwen.py` falls back to LaMa when Qwen is unavailable or fails. Put
`big-lama.pt` in `~/.cache/torch/hub/checkpoints/`;
[`run/run_inpaint.sh`](../run/run_inpaint.sh) globs for it and exports
`LAMA_MODEL` so GPU machines without network access still work.

### pi0.5 policy checkpoints (robot layer)

The closed-loop evaluation ([`robo/eval/pi05_eval.py`](../robo/eval/pi05_eval.py))
talks to an openpi websocket policy server started by
[`run/pi05_serve.sh`](../run/pi05_serve.sh). That script runs from the openpi
fork at `${OPENPI_ROOT}` ([xuningy/openpi](https://github.com/xuningy/openpi) —
it has the `pi05_droid_jointpos` config that mainline openpi lacks) with the
fork's own uv environment, and sets `OPENPI_DATA_HOME` (default
`checkpoints/openpi_cache/`). Checkpoints are expected under
`${OPENPI_DATA_HOME}/openpi-assets-simeval/`:

- `pi05_droid_jointpos/` — the zero-shot DROID joint-position policy
  (about 12 GB), the default.
- `droid_pi05_jointpos_with_web_and_sim/80000` — the sim-co-trained variant;
  select it with `SIMANY_PI05_CKPT`, and set
  `SIMANY_PI05_CONFIG=pi05_droid_jointpos_sim` (fork-only config for its
  unpadded 8-dim action head).

`run/gcs_fetch.py` downloads from the public `openpi-assets` bucket over HTTPS
without `gsutil`. The server stages the checkpoint through host RAM while JAX
restores it, so give it generous memory.

## 3. `third_party/` inventory

Everything here is gitignored except
[`third_party/README.md`](../third_party/README.md).
[`run/setup_env.sh`](../run/setup_env.sh) re-clones `TRELLIS/` if missing;
the rest are manual clones/installs, documented below so they can be
restored by hand.

**TRELLIS/** — Microsoft's image-to-3D generative model, the default
per-object asset generator. [`agents/models/s4_trellis.py`](../agents/models/s4_trellis.py)
inserts the checkout on `sys.path` (`common.TRELLIS_DIR`, override
`SIMANY_TRELLIS_DIR`) and loads the `microsoft/TRELLIS-image-large` weights
from the HF cache — the checkout is code only. Cloned automatically by
`run/setup_env.sh` (`git clone --depth 1 https://github.com/microsoft/TRELLIS.git`).

**ReconViaGen/** — multi-view image-to-3D (VGGT-conditioned TRELLIS,
arXiv 2510.23306). Used by
[`agents/models/s4_reconviagen.py`](../agents/models/s4_reconviagen.py),
[`agents/models/vggt_scene.py`](../agents/models/vggt_scene.py) (its vendored
VGGT) and [`agents/assets/factory_hybrid.py`](../agents/assets/factory_hybrid.py),
which generates both a TRELLIS and an RVG asset per object and keeps the one
with the lower registration residual. Manual clone (`SIMANY_RVG_DIR`
overrides); it ships its own `wheels/vggt` that is also put on `sys.path`,
needs BiRefNet + kornia, and its `dreamsim` dependency is what populates
`weights/dreamsim/`. `agents/models/rvg_pinned.py` (`SIMANY_RVG_PINNED_CONFIG`)
resolves its pretrained resources to local snapshots.

**vggt-omega/** — VGGT-Omega source for the `omega` pose backend of
`agents.models.vggt_scene`; weights in `checkpoints/vggt-omega/`.

**sam-3d-objects/** — SAM 3D Objects, used by
[`agents/models/s4_sam3d.py`](../agents/models/s4_sam3d.py) in its own
environment (`SIMANY_SAM3D_PY`); the checkpoint configuration is read from
`checkpoints/hf/pipeline.yaml` inside the checkout.

**MaskClustering/** — CVPR 2024 class-agnostic 3D instance discovery, the
discovery baseline.
[`agents/baselines/maskclustering.py`](../agents/baselines/maskclustering.py)
stages ScanNet++ scenes into `third_party/MaskClustering/data/scannetpp/` in
their expected layout, prints the GPU commands to run there, and converts the
result back to our `auto_instances.npz` format. Manual clone; runs in its own
`.venv-mc` env (incompatible pins — see [ENVIRONMENTS.md](ENVIRONMENTS.md)).

**FlashSplat/** — ECCV 2024 optimal 2D-mask-to-3DGS segmentation
(arXiv 2409.08270), the object-removal baseline. Manual clone, kept as the
reference implementation:
[`agents/baselines/flashsplat.py`](../agents/baselines/flashsplat.py)
reimplements their per-gaussian weight accumulation on gsplat (their custom
CUDA rasterizer has no prebuilt wheels; the reimplementation is verified
against a float64 reference via `--selftest`) and does not import the checkout.

**BEHAVIOR-1K/** — the BEHAVIOR-1K monorepo: OmniGibson simulator + BDDL
toolchain. Installed into the `behavior1k` conda env by
[`robo/sim/omnigibson_bridge/install_omnigibson.sh`](../robo/sim/omnigibson_bridge/install_omnigibson.sh)
(runs their `setup.sh --omnigibson --bddl --dataset`). Used by
[`robo/sim/export_omnigibson.py`](../robo/sim/export_omnigibson.py) (synset
data from `BEHAVIOR-1K/bddl3/bddl/generated_data`) and by everything in
[`robo/sim/omnigibson_bridge/`](../robo/sim/omnigibson_bridge).

**behavior1k_datasets/** — OmniGibson asset datasets (~36 GB), downloaded by
the BEHAVIOR-1K `setup.sh` run of the install script. Also contains
`phiroom_custom/` — exported twins converted to a custom OmniGibson USD
dataset by
[`robo/sim/omnigibson_bridge/convert_assets.py`](../robo/sim/omnigibson_bridge/convert_assets.py)
and loaded by name in `import_and_run.py`.

**mujoco_menagerie/** — DeepMind's MuJoCo robot model zoo
(`git clone https://github.com/google-deepmind/mujoco_menagerie`;
`SIMANY_MUJOCO_MENAGERIE_ROOT` overrides the location).
[`robo/rigs/pi05_rig.py`](../robo/rigs/pi05_rig.py) loads
`franka_emika_panda/panda_nohand.xml` and `robotiq_2f85/2f85.xml` from it to
build the DROID-style rig.

## 4. `outputs/` anatomy

Everything under `outputs/` is gitignored. One directory per (scene, mode)
run, named by scene id plus a mode suffix:

| directory | mode | launcher |
|---|---|---|
| `<scene>` | single-frame SimFoundry-reproduction baseline | [`run/run_simfoundry.sh`](../run/run_simfoundry.sh) |
| `<scene>_factory` | GT-driven asset factory | [`run/run_factory.sh`](../run/run_factory.sh) |
| `<scene>_auto` | fully automatic, no GT (`SIMANY_AUTO=1`) | [`run/run_auto.sh`](../run/run_auto.sh) |
| `video_<name>` | phone video, derived mesh (`SIMANY_MESH_SRC=derived`) | [`run/run_video2sim.sh`](../run/run_video2sim.sh) |

The launchers set `SIMANY_OUT` accordingly (any other value works; every
stage reads and writes only under it). Inside a factory/auto run, the big
products are:

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
  and `timings.txt` (appended by `stage_timed` in `run/env.sh`).

Cross-scene directories: `outputs/pi05_runs/<run>/` holds closed-loop pi0.5
evaluation runs written by
[`robo/eval/pi05_eval.py`](../robo/eval/pi05_eval.py) (`--out`);
`outputs/module-runs/<attempt>/` holds the receipts of `run/phiroom.sh run`;
`outputs/omnigibson_export/` (`objects/`, `manifest.json`, `results/`) is the
OmniGibson bridge output of `export_omnigibson.py`.

## 5. Getting the code running elsewhere

1. `git clone https://github.com/insait-institute/PhiRIE.git`.
2. Create the main env (`.venv`, python 3.11) and run
   [`run/setup_env.sh`](../run/setup_env.sh). `ROOT` is derived from the
   script's own location (override with `SIMANY_ROOT`). It installs torch
   2.4.1+cu124 + the pipeline deps, clones `third_party/TRELLIS`, and ends
   with an import smoke test.
3. The other two interpreters (sam3 env for SAM3/Qwen, py3.10 env for gsplat)
   are not created by that script — see [ENVIRONMENTS.md](ENVIRONMENTS.md)
   for what they need, and point `SIMANY_SAM3_PY` / `SIMANY_GSPLAT_PY` at
   them.
4. Set the two dataset roots: `SIMANY_SCANNETPP_ROOT` (ScanNet++ v2, layout
   of §1) and `SIMANY_SPLATS_ROOT` (per-scene 3DGS plys) — or skip the
   dataset and start from a phone video ([videos/README.md](../videos/README.md)).
5. Prefetch weights once with `HF_HUB_OFFLINE=0` (SAM3, Qwen-Image-Edit-2509,
   Qwen2.5-7B-Instruct, TRELLIS-image-large, DA3METRIC-LARGE) plus
   `big-lama.pt` into `~/.cache/torch/hub/checkpoints/`; then the offline
   default works. Clone the remaining `third_party/` checkouts you need (§3).
6. What fails without which piece: no sam3 env → `agents.models.s1_segment`,
   `agents.discover.auto_segment` and the inpaint mask/Qwen stages;
   no gsplat env → every render stage (`agents.render.*`,
   `agents.edit.inpaint_fill`, `agents.recon.gsplat_train`); missing HF cache
   with `HF_HUB_OFFLINE=1` → immediate `from_pretrained` file-not-found; no
   splat ply → rendering and inpainting (geometry-only stages still run); no
   GT `scans/` → use `SIMANY_AUTO=1` with `SIMANY_MESH_SRC=derived`;
   pi0.5 evaluation additionally needs the openpi fork + checkpoint of §2.
   `bash run/smoke_imports.sh` checks that every module still imports
   (known environment-dependent failures: viser, omnigibson).
