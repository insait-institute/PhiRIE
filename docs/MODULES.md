# Modular PhiRoom

PhiRoom owns scene construction, scientific workflows and backend interfaces.
PhiView is an independently versioned Git submodule that owns the server-rendered
visualizer and interactive demo. The browser receives images and sends input events;
rendering, picking, generation, editing and simulation execute on the server.

The control package loads feature contracts without importing CUDA libraries. Each
contract owns its actions, tool dependencies, input/output descriptions, runtime alias
and contribution branch. Existing algorithms keep their stable Python import paths.
SimFactory's six adapter groups now live in separate feature packages under
`simfactory/blocks/`; existing YAML configurations and CLI calls remain supported.

## Install and inspect

```bash
git clone --recurse-submodules git@github.com:RunyiYang/PhiRoom.git
cd PhiRoom
uv sync --project envs/control --locked
bash run/phiroom.sh modules
bash run/phiroom.sh describe generation
bash run/phiroom.sh plan generation trellis -- --help
```

`envs/control/uv.lock` is the CPU orchestration and test environment. GPU backends
retain isolated environments documented in [ENVIRONMENTS.md](ENVIRONMENTS.md).
PhiView separately locks its CPU, studio, inference and generation environments.
The root wheel includes the control package and Python backend code; use the full
source checkout for scripts, configurations, model bridges and the submodule.

## Feature ownership

| Feature | Module contract and calls | Backend ownership |
|---|---|---|
| Capture | [capture](../pipelines/capture/) | `capture/` |
| Dataset preparation | [datasets](../pipelines/datasets/) | `agents/recon/`, PhiView dataset adapters |
| Metric reconstruction and 3DGS | [reconstruction](../pipelines/reconstruction/) | `agents/recon/`, `models/` |
| Object discovery | [discovery](../pipelines/discovery/) | `agents/discover/` |
| Asset generation | [generation](../pipelines/generation/) | `models/s4_*` |
| Registration and candidate selection | [registration](../pipelines/registration/) | `agents/assets/` |
| Removal and inpainting | [inpainting](../pipelines/inpainting/) | `agents/edit/`, PhiView interactive editing |
| Physical parameters | [physics](../pipelines/physics/) | `agents/assets/s6_physics.py` |
| Simulator export and dynamics | [simulation](../pipelines/simulation/) | `robo/sim/` |
| Robot tasks and policies | [robotics](../pipelines/robotics/) | `robo/tasks/`, `robo/eval/` |
| Rendering and observations | [rendering](../pipelines/rendering/) | `agents/render/`, `integrations/harmonizer/` |
| Evaluation | [evaluation](../pipelines/evaluation/) | `agents/eval/`, `robo/eval/` |
| Baselines and single-image methods | [baselines](../pipelines/baselines/) | `agents/baselines/`, `agents/single_image/` |
| Scientific campaigns and construction | [experiments](../pipelines/experiments/) | `robo/campaign/`, `simfactory/` |
| Paper capture and packaging | [paper](../pipelines/paper/) | PhiView media tools, `robo/eval/` |
| Viewer and demo | [phiview](../pipelines/phiview/) | pinned `integrations/phiview/` |

## Runtime configuration

Copy `configs/runtime.example.json` to `configs/runtime.local.json` and set actual
interpreter and dataset paths. Paths for interpreters are absolute or relative to the
PhiRoom source root. Explicit runtime configuration wins over environment overrides;
otherwise the documented `SIMANY_*_PY` variables and repository-local defaults apply.
Virtual-environment interpreter symlinks are preserved. A missing interpreter fails
execution instead of selecting a different backend.

```bash
bash run/phiroom.sh plan discovery automatic --config configs/runtime.local.json -- \
  --scene-dir /datasets/ScanNetpp/data/room --out-dir /outputs/new-room
bash run/phiroom.sh run discovery automatic --config configs/runtime.local.json -- \
  --scene-dir /datasets/ScanNetpp/data/room --out-dir /outputs/new-room
```

Arguments after `--` are passed literally to the selected backend. Native backend
flags and semantic input validation remain owned by that backend. `plan` checks the
entry point and produces exact argv, cwd, runtime, GPU requirement and environment
overrides; it does not import models, download weights, create output folders or submit
jobs. GPU readiness and scientific validity are separate validation steps.

PhiView has its own YAML runtime configuration and compatibility backend. Supply it
through the native arguments, as documented in [PHIVIEW.md](PHIVIEW.md):

```bash
bash run/phiroom.sh plan phiview demo -- \
  --config configs/phiview.local.yaml --scene room --out outputs/room_factory
```

## Compose independent calls

A pipeline is an ordered JSON recipe with `schema_version: 1` and `steps`. Each step
selects `module` and `action`, literal `args`, and optional `requires`/`produces` paths.
Paths resolve from the source root; absolute paths remain absolute. For example:

```json
{
  "schema_version": 1,
  "steps": [
    {"module": "capture", "action": "metadata",
     "args": ["/data/clip.mp4", "--out", "outputs/new-clip/metadata.json"],
     "requires": ["/data/clip.mp4"],
     "produces": ["outputs/new-clip/metadata.json"]}
  ]
}
```

```bash
bash run/phiroom.sh pipeline plan pipelines/preflight.json
bash run/phiroom.sh pipeline run pipelines/preflight.json
```

Execution stops at the first failure, saves stdout/stderr and an operational receipt
under `outputs/module-runs/<attempt>/`, and records remaining steps as not run. Input
existence is checked immediately before each step. Explicit expected outputs must be
new and exist after execution; stale outputs cannot satisfy a new successful run.
These are existence checks, not semantic artifact validation. A failed step never
causes a fallback model or an automatic retry. Receipt status does not imply a scientific
claim; its `scientific_validation` field is `NOT_RUN`.

The example preflight recipe is CPU-only and uses the existing authoritative campaign
preflight. Scientific dispatch, admission, collection and freeze provenance continue to
use the campaign tooling in `robo/campaign/` and `run/campaign/`; do not create a second campaign ledger.

## Requested demo feature map

| Requested interaction | Owning PhiView module |
|---|---|
| Original full-resolution Gaussian scene (a) | `splats.py`, `phiview_scene.py`, `render.py` |
| Automatic highlighting and click selection (b, c) | `pipeline.py`, `phiview_scene.py`, `phiview.py` |
| Make objects simulatable; inspect physical parameters (d, e) | `pipeline.py`, `phiview_sim.py`, `scene_state.py` |
| Choose a generated asset (f) | `panels/generate_panel.py`, `pipeline.py`, `scene_state.py` |
| Remove one/all simulatable objects and prompt inpainting (g, repeated f, h) | `phiview_mask_edit.py`, `phiview_inpaint.py`, `phiview_inpaint_guard.py` |
| Fall, friction, throw and projectile interactions (i, j, k) | `phiview_sim.py` |
| Robot command and manipulation demonstration (l) | `robot.py`, `ik.py`, `phiview_sim.py` |
| WASDQE translation and camera rotation (m) | `phiview.py`, `web/phiview.html` |
| Ten-scene media capture and downloadable packs | `paper_campaign.py`, `paper_capture.py`, `paper_download_pack.py` |

This map identifies implementation ownership. It does not certify image quality,
all scenes, or general robot command success. Robot demonstrations retain **GT assistance
plus scripted IK, not a learned policy**. Interactive editing uses the separately pinned
compatibility API; the parent's public-contract inpainting path remains independently
validated. See the release validation receipt for the checks actually run.
