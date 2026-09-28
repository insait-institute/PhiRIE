# Gap study: reconstructed twin vs native simulator

Protocol for quantifying what is lost between **(a)** real or rendered video
→ gaussian splatting → the SimAny sim-ready twin and **(b)** the same scene
living natively in a simulator. Companion pages:
[PIPELINE.md](PIPELINE.md) (the stages being measured),
[ROBOT.md](ROBOT.md) (the pi0.5 closed loop), and the experiment history in
the internal pi0.5 DROID sim-eval notes (2026-08-04, not part of this repository).

The study is deliberately axis-by-axis: a single "gap number" would average
photometric loss against policy variance and mean nothing. Every comparison
below is labeled **apples-to-apples** (same scene, same metric, both arms
measured) or **reference-point-only** (our absolute number placed next to a
published number measured under different conditions).

## 1. The four axes

### RECONSTRUCTION — did the splat and derived geometry capture the scene?

- **Splat photometrics**: holdout PSNR/SSIM on views excluded from training.
  For recon-track scenes this is `train_report.json` written by
  `agents/recon/gsplat_train.py` next to the trained ply (`--holdout-every
  10`); for ScanNet++ it is
  [`agents/eval/factory_eval_render.py`](../agents/eval/factory_eval_render.py)
  on the official DSLR test split (PSNR/SSIM/LPIPS, background splat alone vs
  composite twin).
- **Derived-mesh chamfer vs GT mesh, where GT geometry exists**: the recon
  tracks have no scan, so the pipeline runs on the TSDF mesh from
  [`agents/discover/derive_mesh_from_splat.py`](../agents/discover/derive_mesh_from_splat.py)
  (`SIMANY_MESH_SRC=derived`, ablation rung C). ScanNet++ has
  `scans/mesh_aligned_0.05.ply`; BEHAVIOR and LIBERO have exact native meshes.
  Symmetric chamfer between `derived_mesh.ply` and the GT surface quantifies
  the geometry floor everything downstream stands on. (Small eval script
  still to be written — see the run matrix.)

### GENERATION — are the per-object assets the right shape in the right place?

- **F1@20/40mm vs GT meshes** where GT object geometry exists, computed the
  way [`agents/eval/eval_vs_gt.py`](../agents/eval/eval_vs_gt.py) does it:
  match each discovered object to an independent GT instance (vertex IoU on a
  shared mesh, else mutual 2 cm nearest-neighbor overlap, threshold 0.25),
  then F1 of the registered asset surface against 20k GT-surface samples.
  This matters because under `SIMANY_AUTO` the factory's own F1 is "vs own
  extraction" — only the re-score against independent GT is honest.
  - **BEHAVIOR**: exact GT object meshes *and* poses exist → the strongest
    apples-to-apples cell in the study.
  - **ScanNet++**: GT scan submeshes exist (scan-incomplete for transparent
    objects; the metric inherits that).
  - **LIBERO**: native MJCF asset meshes at known poses.
  - **Video / DROID**: no GT geometry exists, so per
    [`agents/single_image/sharp_hybrid.py`](../agents/single_image/sharp_hybrid.py)'s
    precedent the same numbers must be **relabeled `fit@20/40mm`** — "how
    well does the registered mesh explain the observed views," not
    reconstruction accuracy. Never report them in an F1 column.

### ARTICULATION — an explicit limitation, not a metric we can win

Our twin is **rigid-only**: one rigid body per object, CoACD convex parts,
no joints (the BDDL validation likewise scripts substance/cloth literals as
out of rigid-body scope — see
`outputs/omnigibson_export/comparison_table.md`, generated at run time).
The honest number to report is the **fraction of task-relevant objects that
are articulated in the source simulator** (joint count from the native
BEHAVIOR scene assets), stated as coverage lost, with no attempt to fake a
comparison. LIBERO manipulanda are mostly rigid, so its articulation exposure
is low; ScanNet++/DROID have no articulation GT at all (fraction unknown, say
so).

### POLICY DEPLOYMENT — does the same policy succeed here and there?

pi0.5 success in the native environment vs in our twin of the same scene, on
the same tasks where a task mapping exists. Only LIBERO offers the full
round trip (section 2). Calibration from
the internal pi0.5 DROID sim-eval notes:
pi0.5 zero-shot in sim is weak everywhere — 28% on RoboLab-120 — and PolaRiS
attributes ~25% success cost to the visual real2sim gap with ~0 to dynamics,
so low absolute numbers are the normal regime and the *delta* between arms is
the quantity of interest.

## 2. The four tracks

### Track 1 — ScanNet++ zero-shot 100 (real scans; no native sim)

Scene list: [`data/zeroshot_100_scenes.txt`](../data/zeroshot_100_scenes.txt),
selection record [`data/zeroshot_100_scenes.json`](../data/zeroshot_100_scenes.json).
Criteria (from its `criteria`/`skipped` sections): a trained splat must exist
with **PSNR ≥ 26 dB**, GT annotations present, **≥ 3 graspable objects**, and
the scene must not be in the `nvs_sem_val` benchmark split; 389 scenes were
eligible and the top 100 were selected by a score over capped
manipulable/graspable counts, category diversity, workspace presence, and
splat PSNR (skips: 62 no splat, 50 no annotations, 50 benchmark split, 186
low PSNR, 281 too few graspables).

- **Axes covered**: reconstruction + generation (GT scans exist), plus
  pi05-in-twin **absolute** numbers.
- **Status**: the mature track — the existing 50-scene factory fleet, the
  6-scene composite pool, and all pi0.5 numbers in section 3 come from here.
- **Comparability**: recon/generation are apples-to-apples vs the GT scan;
  policy numbers are absolute only (real scans have no native simulator to
  compare against).

### Track 2 — BEHAVIOR (native OmniGibson scene as ground truth)

Posed RGB-D from the PointWorld release → `run/run_behavior_recon.sh` →
gsplat → the standard pipeline (`SIMANY_AUTO=1 SIMANY_MESH_SRC=derived`) →
compare against the **same scene natively in OmniGibson**.

- **Axes covered**: all four. Exact GT object meshes + poses make generation
  F1 apples-to-apples; native joint lists make the articulation fraction
  computable; BDDL-checker parity extends
  `outputs/omnigibson_export/comparison_table.md`
  (current result on curated-asset export: **7/7 BDDL tasks pass with our
  CoACD collision vs 6/7 with single-convex-hull baseline**, over 14
  geometric + 8 scripted literals).
- **Policy caveat**: pi0.5 does **not** natively run in OmniGibson, so
  "pi05 native vs pi05 twin" is impossible here. The honest comparison is
  task-success of the *same manipulation objectives* — pi05 in our MuJoCo
  twin vs scripted/native BEHAVIOR baselines — plus BDDL parity, clearly
  labeled as objective-level, not policy-level, comparison.
- **Status**: OmniGibson bridge and BDDL checker exist
  ([`robo/sim/export_omnigibson.py`](../robo/sim/export_omnigibson.py),
  `omnigibson_bridge/import_and_run.py`); the RGB-D→gsplat recon path is new
  in this change-set.

### Track 3 — LIBERO (the clean round trip)

Native MuJoCo benchmark; suites on disk at `${SIMANY_ROOT}/data/libero`
(`libero_spatial`, `libero_object`, `libero_goal`, `libero_10` — 4 suites /
40 tasks; demonstration HDF5s + BDDL task definitions). `pi05_libero`
checkpoint is downloading.

- **The point of this track**: native suite success vs success in our
  reconstruction-of-LIBERO-renders twin. Because the source is *already a
  simulator*, the sim → video → `run/run_video2sim.sh` → sim round trip
  **isolates OUR pipeline loss exactly** — no capture noise, no
  scan-incompleteness, GT meshes/poses/dynamics all known. This is the only
  track where the policy-deployment axis is fully apples-to-apples.
- **Axes covered**: all four (articulation exposure is low — LIBERO objects
  are mostly rigid — which also makes it the track where our rigid-only
  limitation costs least).
- **Status**: data present; recon-of-renders and native pi05_libero eval not
  yet run.

### Track 4 — DROID (real robot video; no GT of any kind)

`droid_100` RLDS at `${SIMANY_ROOT}/data/droid` — real robot episodes,
external camera. Video → `run/run_video2sim.sh` → twin →
`pi05_droid_jointpos` closed loop.

- **Axes covered**: reconstruction (holdout PSNR only — no GT geometry),
  generation as **fit@20/40mm only** (sharp_hybrid relabeling precedent),
  policy as absolute pi05-in-twin numbers.
- **Comparability**: **reference-point-only.** There is no native sim and no
  GT; our numbers sit next to the published RoboLab (pi0.5 = 28% on
  RoboLab-120) and PolaRiS (~25% visual-gap cost) figures from the memory
  note, measured on *their* scenes and protocols, not ours.
- **Status**: data present; nothing run.

### Comparability summary

| track | reconstruction | generation | articulation | policy |
|---|---|---|---|---|
| ScanNet++ 100 | apples (GT scan) | apples (GT scan submesh) | n/a (no joint GT) | absolute only |
| BEHAVIOR | apples (native meshes) | **apples (exact GT mesh+pose)** | apples (native joints) | objective-level only (pi05 ∉ OmniGibson) |
| LIBERO | apples (native meshes) | apples (native meshes) | low exposure, mostly rigid | **apples (same policy, same tasks)** |
| DROID | holdout PSNR only | fit@ only (no GT) | unknown | reference-point-only |

## 3. Honest numbers: what pi05-in-twin currently looks like

All numbers from
the internal pi0.5 DROID sim-eval notes.
**Set expectations accordingly before promising policy-gap deltas.**

- The score landscape is **0-heavy**: every pre-geometry-fix run was 0.0 (32
  episodes, zero successes). Post-fix, per-scene summary
  (episodes / nonzero / lifts / best ee-distance):

  | scene | mode | eps | nz | lifts | ee_best |
  |---|---|---|---|---|---|
  | c50d2d1d42 | composite | 10 | 4 | 2 | 0.027 |
  | d755b3d9d8 | raster | 10 | 5 | 0 | 0.010 |
  | f3d64c30f8 | raster | 15 | 3 | 1 | 0.014 |
  | c50d2d1d42 | raster | 15 | 2 | 2 | 0.016 |
  | 45b0dac5e3 | composite | 18 | 0 | 0 | 0.041 |
  | 825d228aec | composite | 18 | 0 | 0 | 0.078 |

- **c50d2d1d42 composite is a favourable outlier**, not the typical case:
  composite scored 0/36 outside it, and the pre-selection criteria that
  predicted 825d228aec would work (jaw fit, bearing, inpaint quality) were
  falsified. Pooled over all 47 post-fix c50 composite episodes: nonzero rate
  13/47 = 0.28, mean staged score 0.106, per-run means spanning 0.000–0.500.
- **Variance dominates any plausible gap signal.** From the 0.28 base rate,
  detecting +0.15 in nonzero rate needs ~158 episodes per arm (80% power,
  α 0.05); +0.25 needs 60; +0.40 needs 24. Policy-gap numbers from this study
  will be **low-N and camera-sensitive** — report them with episode counts
  and confidence, and prefer physics metrics (penetration, settle drift) when
  the question is "did a sim change help".
- **The exterior camera materially changes the score**: same task, same
  everything else — DSC01594 = 0.50, DSC01593 = 0.25, DSC01616 = 0.00.

### Protocol knobs that MUST be held fixed across both arms of any comparison

1. `--time-limit 32` **passed explicitly** — the suite default is 16 s and
   silently reverts otherwise.
2. **Camera choice reported together with the score** (scan-frame pose id for
   composite; fixed rig camera for native), identical across arms.
3. `--episodes >= 5` **with `--jitter`**, matched across arms; n=2 is pure
   noise and n=5 is barely a hint.
4. Observation mode (`raster` | `composite`) fixed and stated; composite
   requires `inpaint/clean_background.ply`
   ([`run/run_inpaint.sh`](../run/run_inpaint.sh), hours per scene).
5. Never use the demo-only flags (`--ext-cam-frame`, `--demo-declutter`,
   `--render-wh`) for benchmark numbers.

## 4. Run matrix

Recon-track scenes live in the **emulated ScanNet++ layout** so the whole
pipeline runs unmodified (see [`agents/core/common.py`](../agents/core/common.py):
`SCENE_DIR = $SIMANY_SCANNETPP_ROOT/data/<scene>`, splat =
`$SIMANY_SPLATS_ROOT/<scene>.ply`):

```sh
# every recon-track invocation
export SIMANY_SCANNETPP_ROOT=${SIMANY_ROOT}/data/recon_scenes
export SIMANY_SPLATS_ROOT=${SIMANY_ROOT}/data/recon_scenes/splats
export SIMANY_AUTO=1 SIMANY_MESH_SRC=derived   # no GT anno, no scan mesh
# scene dir: data/recon_scenes/data/<scene>/dslr/{resized_undistorted_images/,
#            nerfstudio/transforms_undistorted.json, colmap/images.txt}
```

The video→scene path (`run/run_video2sim.sh`) chains
`agents.recon.frames` → `models.vggt_scene` → `agents.recon.metricize` →
`agents.recon.make_scene_dir` → `agents.recon.gsplat_train` (mini-viewer
env), then runs the AUTO+derived-mesh stage list inline (same stage order as
[`run/slurm/ablation_rowC_scene.sbatch`](../run/slurm/ablation_rowC_scene.sbatch),
plus `export_mjcf --test` as in `run/run_auto.sh`, plus a `pi05_tasks` suite).

| track | axis | command / launcher | numbers land in |
|---|---|---|---|
| ScanNet++ | recon (PSNR/SSIM) | `run_gs python -m agents.eval.factory_eval_render` | `outputs/<scene>_factory/render_metrics.json` → `aggregate_results` → `outputs/val_summary.json` |
| ScanNet++ | recon (mesh chamfer) | `derive_mesh_from_splat render` (run_gs) + `fuse` (run) via [`ablation_rowC_scene.sbatch`](../run/slurm/ablation_rowC_scene.sbatch); chamfer vs `scans/mesh_aligned_0.05.ply` **(eval script TBD)** | `outputs/<scene>_auto/derived_mesh.ply` (+ TBD `mesh_chamfer.json`) |
| ScanNet++ | generation | [`run/run_auto.sh`](../run/run_auto.sh) then `run python -m agents.eval.eval_vs_gt outputs/<scene>_auto` | `outputs/<scene>_auto/eval_vs_gt.json` (F1@20/40 vs true GT) |
| ScanNet++ | policy (absolute) | `export_mjcf --test` + `pi05_tasks`, then `sbatch --export=SCENES=...,OBS=... `[`run/slurm/pi05_closedloop.sbatch`](../run/slurm/pi05_closedloop.sbatch) | `outputs/pi05_runs/<run>/` |
| BEHAVIOR | recon | `run/run_behavior_recon.sh` (posed RGB-D → init ply → `agents.recon.gsplat_train`); chamfer vs native scene meshes (TBD) | `train_report.json` next to `data/recon_scenes/splats/<scene>.ply` |
| BEHAVIOR | generation | pipeline as above, then eval_vs_gt **adapted to native GT meshes+poses** (its `load_scene_gt` assumes ScanNet++ `scans/`; the matcher/F1 core is reusable) | `outputs/<scene>_auto/eval_vs_gt.json` |
| BEHAVIOR | articulation | count joints of task-relevant objects in the native scene assets **(small script TBD)** | articulated-fraction table in this doc |
| BEHAVIOR | policy (objective-level) | twin: `pi05_closedloop.sbatch`; native: [`export_omnigibson`](../robo/sim/export_omnigibson.py) + `omnigibson_bridge/import_and_run.py` BDDL runs + scripted baselines | `outputs/pi05_runs/`, `outputs/omnigibson_export/` |
| LIBERO | recon | render native scene → `run/run_video2sim.sh`; chamfer vs MJCF asset meshes (TBD) | `train_report.json`, twin under `outputs/<scene>_auto/` |
| LIBERO | generation | eval_vs_gt adapted to LIBERO asset meshes at GT poses | `outputs/<scene>_auto/eval_vs_gt.json` |
| LIBERO | policy (**round trip**) | native: pi05_libero on the 4 suites (checkpoint downloading); twin: `pi05_closedloop.sbatch` on the reconstructed scene, mapped tasks | native suite success vs `outputs/pi05_runs/` |
| DROID | recon (PSNR only) | `run/run_video2sim.sh` on droid_100 ext-camera video | `train_report.json` |
| DROID | generation (fit@ only) | pipeline as above; report factory F1 columns **relabeled fit@20/40mm** | `outputs/<scene>_auto/objects/` per-object `aligned.json` |
| DROID | policy (reference-point) | `pi05_closedloop.sbatch` with `pi05_droid_jointpos`; cite RoboLab 28% / PolaRiS ~25% as context, never as a baseline row | `outputs/pi05_runs/` |

Notes on the matrix:

- `run/run_video2sim.sh` and `run/run_behavior_recon.sh` are being added in
  the same change-set as this document; the per-stage CLIs they chain
  (`agents/recon/*.py`, `models/vggt_scene.py`) are the contract if the
  launchers drift.
- Composite-observation policy runs on recon tracks additionally need
  `run/run_inpaint.sh` per scene; raster mode needs nothing extra. Whichever
  is used, use the same mode in both arms and state it.
- The two **TBD** items (mesh-chamfer eval, articulation counter) are small,
  CPU-only, and eval-side; nothing in the pipeline blocks on them.
- Where a fleet is involved, [`aggregate_results.py`](../agents/eval/aggregate_results.py)
  is the collection point; keep per-track results in separate output roots so
  its glob does not mix tracks.

## 5. What this study will and will not claim

- It **will** put one number per (track, axis) cell with its comparability
  label attached, and a pipeline-loss delta only where both arms were
  measured (LIBERO policy; BEHAVIOR/LIBERO/ScanNet++ recon+generation).
- It **will** state the articulation fraction as an unmitigated limitation.
- It will **not** average across axes, promote fit@ numbers to F1 columns,
  present DROID/RoboLab/PolaRiS juxtapositions as measured gaps, or report
  any policy delta without episode counts and the fixed-knob checklist of
  section 3.
