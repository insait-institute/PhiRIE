# Modular PhiRoom

PhiRoom (the `phiroom` package) owns scene construction, workflows and backend
interfaces. PhiView, bundled under `tools/phiview/`, owns the server-rendered
visualizer and interactive demo. The browser receives images and sends input
events; rendering, picking, generation, editing and simulation execute on the
server.

The control package loads feature contracts without importing CUDA libraries.
Each contract (`phiroom/modules/<module>.json`) owns its actions, tool
dependencies, input/output descriptions and the runtime alias each action
needs. Existing algorithms keep their stable Python import paths. SimFactory's
adapter groups live in `robo/simfactory/blocks/`; its YAML configurations and
CLI calls remain supported through the `construction` module.

## Install and inspect

```bash
git clone https://github.com/insait-institute/PhiRIE.git
cd PhiRIE
uv sync --project envs/control --locked
bash run/phiroom.sh modules
bash run/phiroom.sh describe generation
bash run/phiroom.sh plan generation trellis -- --help
```

`envs/control/uv.lock` is the CPU control environment. GPU backends keep
their isolated environments documented in [ENVIRONMENTS.md](ENVIRONMENTS.md).
PhiView separately locks its CPU, studio, inference and generation
environments. The root wheel includes the control package and the Python
backend code; use the full source checkout for scripts, configurations, model
adapters and the bundled viewer.

## Feature ownership

| Feature | Module contract and guide | Backend ownership |
|---|---|---|
| Capture | [capture](../phiroom/pipelines/capture/) | `capture/` |
| Dataset preparation | [datasets](../phiroom/pipelines/datasets/) | `agents/recon/`, PhiView dataset adapters |
| Metric reconstruction and 3DGS | [reconstruction](../phiroom/pipelines/reconstruction/) | `agents/recon/`, `agents/models/vggt_scene.py` |
| Object discovery | [discovery](../phiroom/pipelines/discovery/) | `agents/discover/` |
| Asset generation | [generation](../phiroom/pipelines/generation/) | `agents/models/s4_*` |
| Registration and candidate selection | [registration](../phiroom/pipelines/registration/) | `agents/assets/` |
| Removal and inpainting | [inpainting](../phiroom/pipelines/inpainting/) | `agents/edit/`, PhiView interactive editing |
| Physical parameters | [physics](../phiroom/pipelines/physics/) | `agents/assets/s6_physics.py` |
| Simulator export and dynamics | [simulation](../phiroom/pipelines/simulation/) | `robo/sim/` |
| Robot tasks and policies | [robotics](../phiroom/pipelines/robotics/) | `robo/tasks/`, `robo/eval/`, `run/pi05_serve.sh` |
| Rendering and observations | [rendering](../phiroom/pipelines/rendering/) | `agents/render/`, `tools/harmonizer/` |
| Evaluation | [evaluation](../phiroom/pipelines/evaluation/) | `agents/eval/`, `robo/eval/` |
| Baselines and single-image methods | [baselines](../phiroom/pipelines/baselines/) | `agents/baselines/`, `agents/single_image/` |
| Scene construction | [construction](../phiroom/pipelines/construction/) | `run/run_auto.sh`, `run/run_factory.sh`, `robo/simfactory/` |
| Viewer and demo | [phiview](../phiroom/pipelines/phiview/) | `tools/phiview/` |

`bash run/phiroom.sh modules` prints the same 15 contracts as JSON;
`describe <module>` prints one.

## Runtime configuration

Every action declares a runtime alias. The CLI resolves each alias to an
interpreter from `configs/runtime.local.json`, then from the environment
variable, then from the repository-local default:

| alias | variable | default | used for |
|---|---|---|---|
| `control` | `PHIROOM_CONTROL_PY` | the CLI's own interpreter | CPU tools, shell launchers |
| `main` | `SIMANY_PY` | `.venv/bin/python` | most stages (torch 2.4.1+cu124) |
| `sam3` | `SIMANY_SAM3_PY` | `.envs/sam3/bin/python` | SAM3 discovery and masks |
| `gsplat` | `SIMANY_GSPLAT_PY` | `.envs/gsplat/bin/python` | gsplat renders and training |
| `qwen` | `QWEN_PY` | `.envs/qwen/bin/python` | Qwen-Image-Edit |
| `sam3d` | `SIMANY_SAM3D_PY` | `.envs/sam3d-objects/bin/python` | SAM 3D Objects |
| `trellis2` | `SIMANY_TRELLIS2_PY` | `.envs/trellis2/bin/python` | TRELLIS.2 |
| `h5` | `SIMANY_H5_PY` | `.envs/h5/bin/python` | DROID and BEHAVIOR extraction (h5py) |
| `phiview` | — | `run/phiview.sh` | PhiView actions |

Copy `configs/runtime.example.json` to `configs/runtime.local.json` and set
actual interpreter and dataset paths. Interpreter paths are absolute or
relative to the source root. Explicit runtime configuration wins over
environment overrides. Virtual-environment interpreter symlinks are
preserved. A missing interpreter fails execution instead of selecting a
different backend. The `env` object of the configuration may only carry
non-secret `SIMANY_*` context, `CUDA_VISIBLE_DEVICES` and `MUJOCO_GL`.

```bash
bash run/phiroom.sh plan discovery automatic --config configs/runtime.local.json -- \
  --scene-dir /datasets/ScanNetpp/data/room --out-dir /outputs/new-room
bash run/phiroom.sh run discovery automatic --config configs/runtime.local.json -- \
  --scene-dir /datasets/ScanNetpp/data/room --out-dir /outputs/new-room
```

Arguments after `--` are passed literally to the selected backend. Native
backend flags and semantic input validation remain owned by that backend.
`plan` checks the entry point and produces exact argv, cwd, runtime, GPU
requirement and environment overrides; it does not import models, download
weights or create output folders. GPU readiness and scientific validity are
separate steps.

PhiView has its own YAML runtime configuration. Supply it through the native
arguments, as documented in [PHIVIEW.md](PHIVIEW.md):

```bash
bash run/phiroom.sh plan phiview demo -- \
  --config configs/phiview.local.yaml --scene room --out outputs/room_factory
```

## Compose independent calls

A pipeline is an ordered JSON recipe with `schema_version: 1` and `steps`.
Each step selects `module` and `action`, literal `args`, and optional
`requires`/`produces` paths. Paths resolve from the source root; absolute
paths remain absolute. For example, `my_recipe.json`:

```json
{
  "schema_version": 1,
  "steps": [
    {"module": "capture", "action": "metadata",
     "args": ["videos/clip.mp4", "--out", "outputs/new-clip/metadata.json"],
     "requires": ["videos/clip.mp4"],
     "produces": ["outputs/new-clip/metadata.json"]},
    {"module": "capture", "action": "validate",
     "args": ["videos/clip.mp4", "--config", "configs/capture/phone_default.yaml",
              "--out", "outputs/new-clip/validation"],
     "requires": ["videos/clip.mp4"]}
  ]
}
```

```bash
bash run/phiroom.sh pipeline plan my_recipe.json
bash run/phiroom.sh pipeline run my_recipe.json
```

`plan <module> <action>` prints the same one-step structure, so a recipe can be
assembled from individual plans. Execution stops at the first failure, saves
stdout/stderr and an operational receipt under
`outputs/module-runs/<attempt>/receipt.json`, and records remaining steps as
not run. Input existence is checked immediately before each step. Explicit
expected outputs must be new and exist after execution; stale outputs cannot
satisfy a new successful run. These are existence checks, not semantic
artifact validation. A failed step never causes a fallback model or an
automatic retry. Receipt status does not imply a scientific claim; its
`scientific_validation` field is `NOT_RUN`.
