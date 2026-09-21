# PhiRIE

Research code for constructing editable scene assets and interactive environments
from posed RGB images and reconstructed scenes.

> **PhiRIE: From Photorealistic Reconstruction to Interactive Environments.**
> Previously developed as SimAny/PhiRoom. The `phiroom` Python package and CLI remain compatible.
> 中文说明: [README_Chinese.md](README_Chinese.md).

**Authors:** Runyi Yang<sup>1</sup>, Deheng Zhang<sup>1</sup>, Xiaoye Wang<sup>1</sup>, Kanzhi Wu<sup>2</sup>, Lei Sun<sup>1</sup>, Ajad Chhatkuli<sup>1</sup>, Kunyu Peng<sup>3,∗</sup>, Luc Van Gool<sup>1</sup>, Danda Pani Paude<sup>1</sup>

<sup>1</sup> INSAIT, Sofia University “St. Kliment Ohridski”.<br>
<sup>2</sup> vivo Mobile Communication Co., Ltd.<br>
<sup>3</sup> Karlsruhe Institute of Technology<br>
∗ Corresponding author: Kunyu Peng. See [author metadata](AUTHORS.md) and [citation](CITATION.cff).

**PhiRIE v2.0.1** includes the complete pinned [PhiView source](integrations/phiview).
Clone with `git clone https://github.com/insait-institute/PhiRIE.git`. No recursive
submodule checkout is required. See
[PhiView setup](docs/PHIVIEW.md), [modular feature interfaces](docs/MODULES.md),
[contribution workflow](CONTRIBUTING.md), and [release notes and readiness report](docs/releases/v2.0.1.md).

**Project page:** [PhiRIE](https://redesigned-tribble-o8eej69.pages.github.io/) — paper figures,
three interactive WebGPU demos, 45 visualization views, and six recorded videos.
See [website source, provenance, and local setup](website/README.md).

```bash
uv sync --project envs/control --locked
bash run/phiroom.sh modules
bash run/phiroom.sh plan generation trellis -- --help
bash run/phiroom.sh pipeline run pipelines/preflight.json
```

**ScanNet++ demo (`fb5a96b1a2`):** [download the red-mask ZIP](https://github.com/RunyiYang/PhiRIE/releases/download/v2.0.0/PhiRIE_ScanNetpp_fb5a96b1a2_RedMask_Demo.zip)
or [watch the 54-second video](https://github.com/RunyiYang/PhiRIE/releases/download/v2.0.0/PhiRIE_fb5a96b1a2_RedMask_Demo.mp4).
The separate [PhiView shooting video](https://github.com/RunyiYang/PhiRIE/releases/download/v2.0.0/PhiView_fb5a96b1a2_Shooting_Demo.mp4)
shows actual server rendering and projectile simulation.
Extract the ZIP and open `PhiRIE_ScanNetpp_Demo/index.html`.
See [contents, checksums and scope](docs/demos/scannetpp-fb5a96b1a2.md).
This demonstration uses **GT assistance plus scripted IK, not a learned policy**.

Given posed RGB images of an indoor scene plus its off-the-shelf
reconstruction (a mesh and a 3D Gaussian splat), SimAny discovers every
object, replaces it with a generated 3D asset registered back into the metric
scene, annotates it with collision geometry and physical parameters (URDF),
and erases it from the background splat — so the twin stays photorealistic
once objects move. No semantic annotations, no per-object prompts, no clicks;
the output is verified without any ground truth.

## What it can do

- Automatic object discovery (SAM3) → image-to-3D asset generation
  (TRELLIS / ReconViaGen hybrid) → Sim(3) registration → collision geometry
  (CoACD) + physical parameters → URDF, per object.
- Gaussian-native object removal and background completion, so edits stay
  photoreal — including transparent objects that geometric selectors miss.
- Exports to PyBullet, MuJoCo/MJCF, Isaac Lab, and OmniGibson.
- GT-free automatic mode (`run/run_auto.sh`) alongside the GT-driven
  benchmark mode; downstream stages run identically in both.
- Single-image zero-shot mode: one frame + monocular metric depth, no
  reconstruction.
- Closed-loop pi0.5 robot policy evaluation inside the exported scenes, with
  photoreal composite observations ([robo/](robo/)).
- Interactive viser editor: drag assets, re-run physics, watch playback
  ([interface/viewer.py](interface/viewer.py)).

## Repository layout

| Directory | Contents |
|---|---|
| [phiroom/](phiroom/) | portable module CLI, shared contracts and 16 feature interfaces |
| [pipelines/](pipelines/) | separate feature calls, tool ownership and composable recipes |
| [envs/control/](envs/control/) | locked uv environment for CPU orchestration and tests |
| [simfactory/blocks/](simfactory/blocks/) | separate reconstruction, segmentation, generation, editing, simulation and evaluation adapters |
| [integrations/phiview/](integrations/phiview/) | bundled PhiView source, server rendering and interactive demo |
| [agents/](agents/) | generation pipeline: `core` (paths/IO/cameras/vocabulary), `discover`, `assets`, `edit` (Gaussian-native inpainting), `render`, `eval`, `baselines`, `single_image` |
| [models/](models/) | runnable bridges to external neural models: `s1_segment` (SAM3), `s2_depth`, `s4_trellis`, `s4_reconviagen` |
| [robo/](robo/) | robot layer: pi0.5 MuJoCo env/rig/tasks/eval, photoreal rendering, simulator exports (`sim/`) |
| [interface/](interface/) | viser multi-scene editor, MuJoCo live viewer, demo movie/session |
| [run/](run/) | `env.sh` (paths + env knobs), setup, pipeline launchers, slurm jobs |
| [coding_agents/](coding_agents/) | AI coding-agent traces: memory, history, skills |
| [data/](data/) | datasets (contents gitignored) |
| [checkpoints/](checkpoints/) | downloaded model weights (gitignored) |
| [third_party/](third_party/) | vendored checkouts: TRELLIS, ReconViaGen, MaskClustering, FlashSplat, BEHAVIOR-1K, mujoco_menagerie, ... |
| [docs/](docs/) | documentation and the paper LaTeX |
| [tests/](tests/) | dataset-free registration stress suite |

## Quickstart

```bash
bash run/setup_env.sh        # installs the main .venv; two more envs are needed,
                             # see docs/ENVIRONMENTS.md

bash run/run_factory.sh      # GT-driven benchmark mode (one scene)
bash run/run_auto.sh         # fully automatic mode, no annotations
bash run/run_inpaint.sh      # Gaussian-native removal + background completion
bash run/run_simfoundry.sh   # SimFoundry-reproduction baseline (single frame)

# interactive editor (runs in the gsplat env, serves on :8090)
source run/env.sh
run_gs interface.viewer --outputs-root outputs
```

Scene selection and every path/flag are environment variables (`SIMANY_SCENE`,
`SIMANY_OUT`, `SIMANY_AUTO=1`, ...), documented in [run/env.sh](run/env.sh)
and the `run/run_*.sh` launchers. Stages are python modules run from the repo root
(`python -m agents.assets.s5_align`) or via the `run` / `run_sam3` / `run_gs`
/ `run_qwen` helpers that `run/env.sh` defines — three python environments
are unavoidable; [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) explains why.

## Documentation

| Page | What it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | annotated code map, execution model, env-var reference, old→new migration table |
| [docs/PIPELINE.md](docs/PIPELINE.md) | the generation pipeline stage by stage, all four run modes |
| [docs/ROBOT.md](docs/ROBOT.md) | the pi0.5 closed loop: components, recipes, honest results |
| [docs/DATA_AND_WEIGHTS.md](docs/DATA_AND_WEIGHTS.md) | datasets, weights, third_party inventory, outputs/ anatomy |
| [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md) | the claims, the audited numbers, and what is deliberately not claimed |
| [docs/BASELINES.md](docs/BASELINES.md) | positioning against concurrent systems; which comparisons are still owed |
| [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md) | why three python environments, and what lives in each |
| [docs/PAPER_NOTES.md](docs/PAPER_NOTES.md) | paper working notes |
| [docs/PAPER_REVISIONS.md](docs/PAPER_REVISIONS.md) | revisions owed after the July-2026 concurrent work |
| [docs/DEMO_STORYBOARD.md](docs/DEMO_STORYBOARD.md) | demo video storyboard |
| [docs/related/](docs/related/) | source material on cited prior work |
| [docs/paper/](docs/paper/) | paper LaTeX source and `build.sh` |
| [coding_agents/](coding_agents/) | how the AI coding agent's memory/history/skills are organized |

## Naming

The project is **PhiRIE**, previously developed under the names SimAny, PhiRoom,
and SimFoundry. The GitHub repository is `insait-institute/PhiRIE`; `phiroom`,
`simfactory`, `SIMANY_*`, and legacy `SIMF_*` entry points remain compatible.
References to the external SimFoundry baseline retain their original meaning.
Development is consolidated on `main`. Historical branch tips are recoverable
from the original RunyiYang/PhiRIE v2.0.1 history bundle. This organization repository
starts from the recorded release snapshot in [RELEASE_SOURCE.json](RELEASE_SOURCE.json).

## Historical results

These are historical measurements, not experiments rerun for v2.0.1. Current
scientific completion remains governed by [FINAL_EXPERIMENTS.md](FINAL_EXPERIMENTS.md).

Protocol: all 50 ScanNet++ v2 validation scenes, unless noted
(full tables and caveats in [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md)).

- Geometry F1@20 mm **0.783** GT-driven (hybrid generation; 0.708
  single-view), **0.582** fully automatic (scored vs matched GT).
- Automatic mode with no annotations: 1,082 instances, 62.0% tier-A+B yield,
  75.4% drop-test stable.
- Cost: 9.7 A6000-hours for 50 scenes GT-driven, 14.1 automatic
  (11.6 / 17.0 min per scene).
- Appearance on the official DSLR test split: composite twin 28.32 dB vs
  28.89 dB SceneSplat background — a 0.57 dB gap; automatic twin 27.52 dB.
