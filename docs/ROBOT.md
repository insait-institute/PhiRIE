# Robot layer (`robo/`)

Closed-loop evaluation of a real robot policy inside SimAny digital twins,
in the style of RoboLab (arXiv:2604.09860) and PolaRiS (arXiv:2512.16881):
pi0.5 (openpi `pi05_droid_jointpos`, a DROID joint-position policy) drives a
Franka Panda + Robotiq 2F-85 rig in our exported MuJoCo scenes on a
RoboLab-style pick-and-place task suite.

The differentiator is the **photoreal composite observation path**: instead of
showing the policy a gray MuJoCo raster, each control tick renders the clean
inpainted background splat, adds every object's generated asset gaussians posed
from the *live* simulation state, and masks the MuJoCo-rendered robot in on top
via segmentation. PolaRiS reports that the visual real2sim gap costs ~25%
success while sim dynamics cost ~0 — and photoreal renders are exactly what the
SimAny asset pipeline produces.

Related pages: [ENVIRONMENTS.md](ENVIRONMENTS.md) (the three python envs),
[CONTRIBUTIONS.md](CONTRIBUTIONS.md). Full experiment history:
[`coding_agents/memory/pi05-droid-sim-eval.md`](../coding_agents/memory/pi05-droid-sim-eval.md)
and
[`coding_agents/memory/pi05-zero-score-root-cause.md`](../coding_agents/memory/pi05-zero-score-root-cause.md).

## Components

### `robo/envs` — DROID-convention MuJoCo environment

[`pi05_env.py`](../robo/envs/pi05_env.py) (`DroidSimEnv`) wraps the rig-built
model with the exact openpi `droid_policy` observation dict:

| key | value |
|---|---|
| `observation/exterior_image_1_left` | uint8 (H, W, 3), left-ZED-equivalent |
| `observation/wrist_image_left` | uint8 (H, W, 3) |
| `observation/joint_position` | float (7,), absolute rad |
| `observation/gripper_position` | float (1,), 0 = open .. 1 = closed |

Control contract:

- 15 Hz control, physics `dt = 1/600` s, 40 substeps per control tick.
- Actions per tick: (8,) = **absolute** joint-position targets (7) + gripper
  [0,1], binarized at 0.5 into ctrl 0/255.
- Per-tick joint delta clamped to 0.2 rad (matching `droid/robot_ik_solver.py`).
- Grasp detection from the 2F-85 finger-pad geoms; `reset()` settles 1.5 s and
  supports per-target XY jitter.

### `robo/rigs` — Panda + Robotiq 2F-85 rig

[`pi05_rig.py`](../robo/rigs/pi05_rig.py) assembles, via MjSpec, the Menagerie
`panda_nohand` + `robotiq_2f85` (from `third_party/mujoco_menagerie`, SERL
−90° z mount quat) into an exported `scene.xml`, adds a ZED-like external
camera and a wrist camera, and a solid tabletop collision box — the exported
scene only carries per-object support shims, so the arm needs a real tabletop
to sweep against. DROID reset pose: `[0, -pi/5, 0, -4pi/5, 0, 3pi/5, 0]`.
Robot mesh paths are absolutized before attach so the scene compiler's
`meshdir` does not swallow them.

### `robo/tasks` — suite generation + staged scoring

[`pi05_tasks.py`](../robo/tasks/pi05_tasks.py) reads a scene's `objects/` tree
+ `sim_export/mujoco_settle.json`, picks the main tabletop cluster, places the
robot at the table edge, and emits `sim_export/pi05_tasks.json` with one task
per (graspable, receptacle) pair (or move-to-region when the scene has no
receptacle). Base placement enforces a radius **and** frontal-cone workspace
(0.25–0.80 m, ±60°, preferring the 0.35–0.65 m annulus) — the original
radius-only test admitted targets up to 143° behind the arm and was the root
cause of the early all-zero runs (see
[pi05-zero-score-root-cause](../coding_agents/memory/pi05-zero-score-root-cause.md)).

`TaskScorer` gives staged credit, 0.25 per stage
(grasp → lift → hover → place); success = place held for 1 s.

```bash
python -m robo.tasks.pi05_tasks --out-dir outputs/<scene>_factory [--max-tasks 10]
```

### `robo/rendering` — photoreal composite observations

[`pi05_render.py`](../robo/rendering/pi05_render.py) (`CompositeObs`) renders,
per camera and tick: `inpaint/clean_background.ply` (falls back to the raw
scene splat, with object ghosting) + each object's canonical `trellis_gs.ply`
transformed by its live MuJoCo body pose (scale from `aligned.json`), then the
MuJoCo robot masked in via segmentation rendering (SIMPLER-style
green-screening, but sim-over-splat). Needs GPU (gsplat/torch) +
`MUJOCO_GL=egl`, and `SIMANY_SCENE`/`SIMANY_OUT` set for the scene being
evaluated. Roughly 0.28 s per two-camera tick at 640×360.

### `robo/eval` — websocket policy client

[`pi05_eval.py`](../robo/eval/pi05_eval.py) runs episodes against an openpi
websocket policy server. The model predicts joint *deltas*; the server-side
`AbsoluteActions` transform returns a chunk of shape (15, 8) of **absolute**
joint positions + gripper [0,1]; the client executes the full 15-step chunk at
15 Hz, then requeries (RoboLab default). It warms up once after connect and
reconnect-retries around the JAX jit window (see gotchas). Key flags:
`--tasks`, `--out`, `--obs raster|composite`, `--episodes`, `--jitter`,
`--time-limit`, `--video`, `--task-filter`, `--port`, `--debug-obs`, and the
demo-only overrides `--ext-cam-frame`, `--demo-declutter`, `--render-wh`
(these change what the policy sees — never use them for benchmark numbers).
A `--policy scripted` sinusoidal policy exists as an env/scorer smoke test.

### `robo/sim` — export and settling

- [`export_mjcf.py`](../robo/sim/export_mjcf.py) — exports the reconstructed
  scene to MJCF + an Isaac-Lab-ready manifest and smoke-tests it headlessly
  (`--test`, which also writes `sim_export/mujoco_settle.json`). Object
  collision = the CoACD convex parts (exact under MuJoCo's convexification);
  the scan background is visual-only plus one *private* static support shim
  per object on its own collision channel, so shims never touch other objects
  or the robot.
- [`export_omnigibson.py`](../robo/sim/export_omnigibson.py) — packages
  s6-sim-ready objects (CoACD parts + URDF + physics.json) for
  BEHAVIOR-1K/OmniGibson and maps labels to BDDL WordNet synsets; the actual
  URDF→USD conversion runs in
  [`omnigibson_bridge/import_and_run.py`](../robo/sim/omnigibson_bridge/import_and_run.py)
  inside a live Isaac Sim process.
- [`s7_sim.py`](../robo/sim/s7_sim.py) — composes the twin in PyBullet,
  settles (velocity-zeroing until poses converge), runs a dynamics demo for
  s8 rendering.
- [`redrop.py`](../robo/sim/redrop.py) — replays the factory drop test with
  the COM→link-frame correction, writing `drop_v2.json` per scene.
- [`mujoco_video.py`](../robo/sim/mujoco_video.py) — settle/lift/drop rollout
  video from an exported `scene.xml` (headless EGL).
- [`viewer_settle.py`](../robo/sim/viewer_settle.py) — physics backend for
  [`interface/viewer.py`](../interface/viewer.py): 4 s gravity dynamics from
  user-edited poses, returned as a 30 fps link-frame trajectory.

## Running it

### 1. Serve the policy

```bash
srun -p debug --gres=gpu:a6000:1 --mem=100G --time=240 bash run/pi05_serve.sh
```

[`run/pi05_serve.sh`](../run/pi05_serve.sh) launches the openpi websocket
server (default port 8000) from the checkout at `/group/worldcept/code/openpi`
— a fork of `github.com/xuningy/openpi`, which has the `pi05_droid_jointpos`
config (mainline does not). The checkpoint is pre-downloaded to
`/group/worldcept/openpi_cache/openpi-assets-simeval/` (12.4 GB), so no GCS
access happens at serve time.

- `SIMANY_PI05_CKPT` selects the checkpoint subpath (default
  `pi05_droid_jointpos`; e.g. `droid_pi05_jointpos_with_web_and_sim/80000` for
  the sim-co-trained variant, which also needs
  `SIMANY_PI05_CONFIG=pi05_droid_jointpos_sim`).
- **The `/scratch` copy trick:** orbax/zarr restore does many small scattered
  reads, and that pattern stalled for 1 h+ (zero progress, no error) against
  the CephFS-hosted copy — while a plain rsync of the same 12 GB finished in
  seconds. The script therefore rsyncs the checkpoint to node-local
  `/scratch/runyi_yang/openpi_cache_local/` (gated on a `.copy_complete`
  marker) and serves from there; restore drops to 15.4 s. `/scratch` is
  node-local, so the copy only helps on the node that made it.

### 2. Closed-loop eval (server + client on one A6000)

```bash
sbatch [--export=SCENES="c50d2d1d42",OBS=composite,EXTRA="--episodes 5 --jitter 0.02 --time-limit 32"] \
    run/slurm/pi05_closedloop.sbatch
```

[`run/slurm/pi05_closedloop.sbatch`](../run/slurm/pi05_closedloop.sbatch)
knobs: `SCENES` (space-separated scene ids), `OBS` (`raster` | `composite`),
`EXTRA` (extra `pi05_eval.py` args), `RUN` (run-name prefix). Results land in
`outputs/pi05_runs/`. Details baked into the script, learned the hard way:

- unique `--port` per job (`8000 + SLURM_JOB_ID % 1000`) — concurrent jobs
  share the node's localhost; tyro also wants `--port` *before*
  `policy:checkpoint`;
- `#SBATCH --mem=100G` — the server holds the 12 GB checkpoint in host RAM
  while JAX stages it; the 2G cgroup default OOM-kills it mid-restore;
- `TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"` pinned unconditionally so the login
  profile cannot leak an H200 setting and force a gsplat JIT rebuild;
- composite mode loops one scene per process (see gotchas); raster mode passes
  all suites to a single client.

Direct invocation (e.g. for a variant output dir the sbatch cannot address):

```bash
source run/env.sh
MUJOCO_GL=egl SIMANY_SCENE=<scene> SIMANY_OUT=outputs/<scene>_factory \
  .venv/bin/python -m robo.eval.pi05_eval \
  --tasks outputs/<scene>_factory/sim_export/pi05_tasks.json \
  --out outputs/pi05_runs/<run> --obs composite --video --time-limit 32
```

### 3. Prerequisites for a new scene

1. `python -m robo.sim.export_mjcf --test` (with `SIMANY_SCENE`/`SIMANY_OUT`
   set) — creates `sim_export/` and `mujoco_settle.json`; without the latter,
   `exclude_objects` is silently always empty.
2. `python -m robo.tasks.pi05_tasks --out-dir outputs/<scene>_factory`.
3. Composite observations additionally require
   `outputs/<scene>_factory/inpaint/clean_background.ply`. Six scenes have it:
   **27dd4da69e, 45b0dac5e3, 578511c8a9, 7b6477cb95, 825d228aec, c50d2d1d42**.
   Widen the pool with [`run/run_inpaint.sh`](../run/run_inpaint.sh) /
   [`run/slurm/inpaint_scene.sbatch`](../run/slurm/inpaint_scene.sbatch)
   (hours per scene).

## Results (honest, as of 2026-08-04)

Numbers below are quoted from
[pi05-droid-sim-eval](../coding_agents/memory/pi05-droid-sim-eval.md) and
[pi05-zero-score-root-cause](../coding_agents/memory/pi05-zero-score-root-cause.md).

**Calibrate expectations first:** pi0.5 zero-shot in sim is weak everywhere —
it scores 28% on RoboLab-120 — so low absolute numbers are the normal regime,
not a SimAny defect.

- **Baseline (pre geometry fix): all zero.** Every early run scored 0.0 — 32
  episodes across raster and composite, zero successes. Root cause was a
  base-placement bug (isotropic reachability with no frontal cone, so targets
  were admitted up to 143° behind the arm), fixed 2026-08-01 in
  `pi05_tasks.py`.
- **After the fix:** first nonzero episode ever in raster (d755 cup, grasp,
  0.25), and the first lift ever in the photoreal composite (c50 mug,
  grasp+lift, 0.50, `ee_dist_min` 2.7 cm).
- **Current per-scene summary** (episodes / nonzero / lifts / best ee-distance):

  | scene | mode | eps | nz | lifts | ee_best |
  |---|---|---|---|---|---|
  | c50d2d1d42 | composite | 10 | **4** | **2** | 0.027 |
  | d755b3d9d8 | raster | 10 | 5 | 0 | 0.010 |
  | f3d64c30f8 | raster | 15 | 3 | 1 | 0.014 |
  | c50d2d1d42 | raster | 15 | 2 | 2 | 0.016 |
  | 45b0dac5e3 | composite | 18 | 0 | 0 | 0.041 |
  | 825d228aec | composite | 18 | 0 | 0 | 0.078 |

  c50d2d1d42 composite (4/10 nonzero, 2 lifts) is the only configuration that
  works, and it is a **favourable outlier**, not the typical case: composite
  scored 0/36 outside c50d2d1d42 (45b0dac5e3 and 825d228aec, 18 episodes
  each), even though 825d228aec — one of the two newly inpainted scenes — was
  best-in-class on jaw fit, bearing, and inpaint quality. Those pre-selection
  criteria were falsified; scene-hunting is a dead end. Pooled over all 47 c50
  composite episodes after the geometry fix: nonzero rate 13/47 = 0.28, mean
  staged score 0.106, per-run means spanning 0.000–0.500.
- **Variance dominates.** From the 0.28 base rate, detecting a +0.15 change in
  nonzero rate needs ~158 episodes per arm (80% power, α 0.05); +0.25 needs
  60; +0.40 needs 24. n=2 is pure noise and n=5 is barely a hint — match
  `--episodes`/`--jitter` and pin the camera before calling any A/B. Prefer
  physics metrics (penetration, settle drift) for verifying sim changes.
- **Demo clip (score 0.50, grasp+lift, `ee_dist_min` 4.0 cm):**
  `outputs/pi05_runs/demo_final/c50_mug_grasp_lift_720p.mp4`, rendered with
  `--task-filter obj_09 --ext-cam-frame DSC01594 --render-wh 1280 720
  --time-limit 32` (no declutter), ext pane cropped in post. **Caveat:** the
  exterior camera materially changes the score — same task, same everything
  else: DSC01594 = 0.50, DSC01593 = 0.25, DSC01616 = 0.00. Pick camera and
  report score together, and never present a demo-flag run as a benchmark
  number.

## Known gotchas

- **First inference = JAX jit (minutes), and the websockets 20 s keepalive
  kills the connection meanwhile.** The client reconnect-retries (RoboLab's
  client does the same); expect ~3 reconnects on the first infer, and the
  server-side handshake tracebacks during that window are noise. The retry cap
  is 5 — do not lower it.
- **Composite mode is one scene per process.** `agents/core/common.py` binds
  `SIMANY_SCENE`/`SIMANY_OUT` at import time; `pi05_eval` asserts a single
  suite in composite mode, and the sbatch loops scenes in separate processes.
- **Wrist near-field splat mush.** The scene splat was never trained at
  10–30 cm, so the wrist camera's near field renders as beige mush (the
  TRELLIS object splats themselves look fine up close). Known open issue;
  moving the wrist camera did not help, and depth-compositing the MuJoCo table
  into the wrist view is the untried lever.
- **The exterior camera must be a real scan-trajectory pose.** Synthetic
  DROID-offset viewpoints are outside the splat's training views and render as
  blurry gaussian mush; scan-frame poses are photoreal.
  `pi05_tasks.ext_cam_from_frame(frame)` pins any scan frame.
- **Pass `--time-limit 32` explicitly.** The suite default is 16 s, which is
  too short; the 32 s setting silently reverts unless passed on the command
  line.
