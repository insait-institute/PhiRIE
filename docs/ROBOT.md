# Robot layer (`robo/`)

Closed-loop evaluation of a real robot policy inside PhiRIE digital twins,
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
asset pipeline produces.

Related pages: [ENVIRONMENTS.md](ENVIRONMENTS.md) (python environments),
[DATA_AND_WEIGHTS.md](DATA_AND_WEIGHTS.md) (policy checkpoints),
[PIPELINE.md](PIPELINE.md) (how the scenes are built).

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

Control contract (frozen in
[`configs/policies/frozen_fields.yaml`](../configs/policies/frozen_fields.yaml)):

- 15 Hz control, physics `dt = 1/600` s, 40 substeps per control tick.
- Actions per tick: (8,) = **absolute** joint-position targets (7) + gripper
  [0,1], binarized at 0.5 into ctrl 0/255.
- Per-tick joint delta clamped to 0.2 rad (matching `droid/robot_ik_solver.py`).
- Grasp detection from the 2F-85 finger-pad geoms; `reset()` settles 1.5 s and
  supports per-target XY jitter.

### `robo/rigs` — Panda + Robotiq 2F-85 rig

[`pi05_rig.py`](../robo/rigs/pi05_rig.py) assembles, via MjSpec, the Menagerie
`panda_nohand` + `robotiq_2f85` (from `third_party/mujoco_menagerie`, override
`SIMANY_MUJOCO_MENAGERIE_ROOT`; SERL −90° z mount quat) into an exported
`scene.xml`, adds a ZED-like external camera and a wrist camera, and a solid
tabletop collision box — the exported scene only carries per-object support
shims, so the arm needs a real tabletop to sweep against. DROID reset pose:
`[0, -pi/5, 0, -4pi/5, 0, 3pi/5, 0]`. Robot mesh paths are absolutized before
attach so the scene compiler's `meshdir` does not swallow them.

### `robo/tasks` — suite generation + staged scoring

[`pi05_tasks.py`](../robo/tasks/pi05_tasks.py) reads a scene's `objects/` tree
+ `sim_export/mujoco_settle.json`, picks the main tabletop cluster, places the
robot at the table edge, and emits `sim_export/pi05_tasks.json` with one task
per (graspable, receptacle) pair (or move-to-region when the scene has no
receptacle). Base placement enforces a radius **and** frontal-cone workspace
(0.25–0.80 m, ±60°, preferring the 0.35–0.65 m annulus); a radius-only test
admits targets far behind the arm and produces all-zero runs.

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
MuJoCo robot masked in via segmentation rendering
([`mujoco_masks.py`](../robo/rendering/mujoco_masks.py); SIMPLER-style
green-screening, but sim-over-splat). Needs GPU (gsplat/torch) +
`MUJOCO_GL=egl`, and `SIMANY_SCENE`/`SIMANY_OUT` set for the scene being
evaluated. Roughly 0.28 s per two-camera tick at 640×360.

Optional observation treatments: [`harmonizer_client.py`](../robo/rendering/harmonizer_client.py)
talks to the `tools.harmonizer.server` socket service
([tools/README.md](../tools/README.md)); [`robot_restore.py`](../robo/rendering/robot_restore.py)
copies the eroded robot core back from the raw simulator image after a
full-frame enhancement; [`color_match.py`](../robo/rendering/color_match.py) is
a deterministic background-fitted RGB affine baseline.

### `robo/policy` and `robo/manifest` — policy registry and run manifests

[`robo/policy/registry.py`](../robo/policy/registry.py) loads
`configs/policies/*.yaml` (`pi05_droid_jointpos`, `pi05_droid_jointpos_sim`,
`scripted_sinusoid`) into typed entries with pinned checkpoint identities and
constructs the matching clients (`robo/policy/clients/`), all normalized to the
control contract in [`control_contract.py`](../robo/policy/control_contract.py).
[`robo/policy/bound_server.py`](../robo/policy/bound_server.py)
(`run/pi05_serve_bound.sh`) serves exactly one registry policy with a
fail-closed runtime identity receipt. [`robo/manifest`](../robo/manifest)
defines the rollout manifest schema and hashes
(`python -m robo.manifest.io validate configs/evaluation/example_manifest.yaml`,
`... diff a.yaml b.yaml`).

### `robo/eval` — policy client and metrics

[`pi05_eval.py`](../robo/eval/pi05_eval.py) runs episodes against an openpi
websocket policy server. The model predicts joint *deltas*; the server-side
`AbsoluteActions` transform returns a chunk of shape (15, 8) of **absolute**
joint positions + gripper [0,1]; the client executes the full 15-step chunk at
15 Hz, then requeries (RoboLab default). It warms up once after connect and
reconnect-retries around the JAX jit window (see gotchas). Flags:
`--tasks`, `--out`, `--policy server|scripted`, `--host`, `--port`,
`--checkpoint-path` (provenance only), `--open-loop-horizon`, `--episodes`,
`--variant vague|default|specific`, `--jitter`, `--seed`, `--video`,
`--task-filter`, `--time-limit`, `--debug-obs`, `--obs raster|composite`, and
the demo-only overrides `--ext-cam-frame`, `--demo-declutter`, `--render-wh`
(these change what the policy sees — never use them for benchmark numbers).
`--policy scripted` is a sinusoidal policy for env/scorer smoke tests. Each run
writes `results.json`, per-episode logs
([`episode_log.py`](../robo/eval/episode_log.py)) and, with `--video`, mp4s.

Offline metrics: [`fidelity_metrics.py`](../robo/eval/fidelity_metrics.py)
(`--manifest`, `--out`) scores held-out appearance and metric geometry from a
manifest such as
[`configs/evaluation/fidelity_manifest.template.json`](../configs/evaluation/fidelity_manifest.template.json);
[`harmony_visual_metrics.py`](../robo/eval/harmony_visual_metrics.py) scores
visual quality, task-mask preservation, temporal consistency and latency of
harmonized observations from
[`configs/evaluation/harmony_visual_manifest.template.json`](../configs/evaluation/harmony_visual_manifest.template.json).

### `robo/sim` — export and settling

- [`export_mjcf.py`](../robo/sim/export_mjcf.py) — exports the reconstructed
  scene to MJCF + an Isaac-Lab-ready manifest and smoke-tests it headlessly
  (`--test`, which also writes `sim_export/mujoco_settle.json`). Object
  collision = the CoACD convex parts (exact under MuJoCo's convexification);
  the scan background is visual-only plus one *private* static support shim
  per object on its own collision channel, so shims never touch other objects
  or the robot. [`room_collision.py`](../robo/sim/room_collision.py) is the
  full-room alternative: a floor plane, validated support/wall boxes and a
  CoACD decomposition of the remaining room mesh on one shared collision
  channel.
- [`export_omnigibson.py`](../robo/sim/export_omnigibson.py) — packages
  s6-sim-ready objects (CoACD parts + URDF + physics.json) for
  BEHAVIOR-1K/OmniGibson and maps labels to BDDL WordNet synsets; the actual
  URDF→USD conversion runs in
  [`omnigibson_bridge/import_and_run.py`](../robo/sim/omnigibson_bridge/import_and_run.py)
  inside a live Isaac Sim process
  (`omnigibson_bridge/install_omnigibson.sh` installs the environment).
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

### `robo/simfactory` — YAML pipeline runner

[`runner.py`](../robo/simfactory/runner.py) (`simfactory run <config.yaml>`
console script, or `bash run/phiroom.sh run construction factory -- run <config.yaml>`;
`simfactory list` prints every block and registered backend) parses a YAML
config, resolves one backend per block from
[`registry.py`](../robo/simfactory/registry.py) (`blocks/`: segmentation,
generation, reconstruction, inpainting, simulation, evaluation) and runs the
blocks in pipeline order with the `SIMANY_*` environment the underlying
modules expect.

## Running it

### 1. Serve the policy

On a GPU machine with the openpi fork checked out (`OPENPI_ROOT`) and the
checkpoint under `${OPENPI_DATA_HOME}/openpi-assets-simeval/`
([DATA_AND_WEIGHTS.md](DATA_AND_WEIGHTS.md)):

```bash
export OPENPI_ROOT=/path/to/openpi
bash run/pi05_serve.sh            # websocket server on port 8000
```

[`run/pi05_serve.sh`](../run/pi05_serve.sh) launches openpi's
`scripts/serve_policy.py` with `--policy.dir` pointing at the local checkpoint,
so no network access happens at serve time. `SIMANY_PI05_CKPT` selects the
checkpoint subpath (default `pi05_droid_jointpos`; e.g.
`droid_pi05_jointpos_with_web_and_sim/80000` for the sim-co-trained variant,
which also needs `SIMANY_PI05_CONFIG=pi05_droid_jointpos_sim`). Extra
arguments (for example `--port 8001`) are passed through to `serve_policy.py`
before the `policy:checkpoint` subcommand. JAX stages the ~12 GB checkpoint
through host RAM while restoring, so the machine needs generous memory. About
8 GB of GPU memory suffices for inference.

### 2. Closed-loop evaluation

With the server running (same machine or `--host`), in the main environment:

```bash
source run/env.sh
export SIMANY_SCENE=<scene> SIMANY_OUT=outputs/<scene>_factory
MUJOCO_GL=egl python -m robo.eval.pi05_eval \
  --tasks outputs/<scene>_factory/sim_export/pi05_tasks.json \
  --obs composite --out outputs/pi05_runs/<run> \
  --episodes 5 --jitter 0.02 --time-limit 32 --video
```

`--obs raster` runs the gray MuJoCo observations instead and accepts several
`--tasks` suites in one process; composite mode is one scene per process (see
gotchas). Use a distinct `--port` per server when several run on one host.
The same call is available as `bash run/phiroom.sh run robotics evaluate -- ...`.

### 3. Prerequisites for a new scene

1. `python -m robo.sim.export_mjcf --test` (with `SIMANY_SCENE`/`SIMANY_OUT`
   set) — creates `sim_export/` and `mujoco_settle.json`; without the latter,
   `exclude_objects` is silently always empty.
2. `python -m robo.tasks.pi05_tasks --out-dir outputs/<scene>_factory`.
3. Composite observations additionally require
   `outputs/<scene>_factory/inpaint/clean_background.ply`, produced by
   [`run/run_inpaint.sh`](../run/run_inpaint.sh) (hours per scene). Without
   it the raw scene splat is used and removed objects leave ghosts.

Absolute success rates of pi0.5 zero-shot in simulation are low everywhere
(RoboLab reports 28% on RoboLab-120), so expect low numbers, match
`--episodes`/`--jitter` and pin the exterior camera before comparing two
configurations, and prefer physics metrics (penetration, settle drift) for
verifying simulation changes. The exterior camera materially changes the
score of a given task, so report camera and score together.

## Known gotchas

- **First inference = JAX jit (minutes), and the websockets 20 s keepalive
  kills the connection meanwhile.** The client reconnect-retries (RoboLab's
  client does the same); expect a few reconnects on the first infer, and the
  server-side handshake tracebacks during that window are noise. The retry cap
  is 5 — do not lower it.
- **Composite mode is one scene per process.** `agents/core/common.py` binds
  `SIMANY_SCENE`/`SIMANY_OUT` at import time, and `pi05_eval` asserts a single
  suite in composite mode; loop scenes in separate processes.
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
- **Keep `TORCH_CUDA_ARCH_LIST` consistent** between the machine that built
  the gsplat JIT cache and the evaluation run, or gsplat rebuilds from scratch
  (`run/env.sh` pins `"8.6;9.0+PTX"`).
