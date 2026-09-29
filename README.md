# ϕ-RIE

### From Photorealistic Reconstruction to Interactive Environments

[Project page](https://insait-institute.github.io/PhiRIE/) · [Paper](https://arxiv.org/abs/2609.26795) · [Citation](#citation) · [Demo video](https://youtu.be/3-YdcBh6Tbw) · [PhiView code](https://github.com/RunyiYang/PhysicalView)

**ϕ-RIE turns captured rooms into editable environments.** It connects object
discovery, 3D asset generation, metric registration, background completion, and
physics. PhiView lets you explore the result, change object parameters, shoot
projectiles, and run robot interactions.

![ϕ-RIE teaser: reconstruction, object assets, physical interaction, and harmonization](media/paper-teaser-arxiv.png)

[Runyi Yang](https://runyiyang.github.io/), [Deheng Zhang](https://dehezhang2.github.io/), [Xiaoye Wang](https://adamwang0224.github.io/), [Kanzhi Wu](https://www.kanzhi.tech/about), [Lei Sun](https://ahupujr.github.io/), [Ajad Chhatkuli](https://ajadchhatkuli.github.io/), [Kunyu Peng](https://kpeng9510.github.io/)*, [Luc Van Gool](https://insait.ai/prof-luc-van-gool/), [Danda Paudel](https://insait.ai/dr-danda-paudel/)

INSAIT, Sofia University “St. Kliment Ohridski” · vivo Mobile Communication Co., Ltd. · Karlsruhe Institute of Technology

Contact: [runyi.yang@insait.ai](mailto:runyi.yang@insait.ai)

*Corresponding author

## Image demos

| Shooting and physical parameters | Grasp and place |
|---|---|
| ![Four mass and friction settings in PhiView](media/spray-mass-friction-impact.webp) | ![Bottle grasp and placement on a mouse pad](media/demo-09.webp) |
| Three colorful shots per setting | Lift the bottle and move it to the target |

![Appearance harmonization in matched views](media/paper-teaser.webp)

Watch [the full demo](https://youtu.be/3-YdcBh6Tbw) or explore
[all nine recordings](https://insait-institute.github.io/PhiRIE/#motion) on the project page.

## Method overview

![ϕ-RIE main figure: scene observation, coupled construction, and interactive environments](media/paper-main.png)

The pipeline connects captured Gaussian scenes to complete object assets and
backgrounds, then keeps rendered appearance aligned with simulated body poses.

## 1. Installation

### Start with the control tools

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/insait-institute/PhiRIE.git
cd PhiRIE
uv sync --project envs/control --locked
bash run/phiroom.sh modules
```

The control environment lists components, prepares commands, and runs CPU tools.
The organization checkout includes PhiView source in `integrations/phiview/`.

### Set up reconstruction and simulation

GPU reconstruction and rendering require an NVIDIA GPU and the runtime for the
selected model. Prepare the environments, data, and weights with these guides:

| Setup | Guide |
|---|---|
| Python, CUDA, and model environments | [Environment setup](docs/ENVIRONMENTS.md) |
| Captured views, camera poses, reconstructions, and weights | [Data and weights](docs/DATA_AND_WEIGHTS.md) |
| PhiView rendering and interaction | [PhiView setup](docs/PHIVIEW.md) |
| Independent component commands | [Module interfaces](docs/MODULES.md) |

## 2. Basic demos

### Watch or download the recordings

The recordings are stored in [`media/`](media/); the [demo gallery](https://insait-institute.github.io/PhiRIE/#motion)
on the project page plays the same files in the browser.

| Demo | Interaction |
|---|---|
| [01 — Spray bottle](media/demo-01.mp4) | Six shots with colorful trails |
| [02 — Bottle to paper](media/demo-02.mp4) | Grasp, lift, and place |
| [03 — LIBERO bottle](media/demo-03.mp4) | Grasp and move |
| [04 — Kitchen](media/demo-04.mp4) / [05 — Kitchen, harmonized](media/demo-05.mp4) | Matched camera walk, harmonizer off / on |
| [06 — Office](media/demo-06.mp4) / [07 — Office, harmonized](media/demo-07.mp4) | Matched camera walk, harmonizer off / on |
| [08 — Office bottles](media/demo-08.mp4) | Eighteen shots across nine bottles, with harmonization |
| [09 — Bottle to mouse pad](media/demo-09.mp4) | Grasp, lift, and place |
| [Mass × friction](media/spray-mass-friction-2x2.mp4) | Four settings, three shots per setting |

### Try the browser playground

The [project page](https://insait-institute.github.io/PhiRIE/#playground) hosts
scene editing, a physics sandbox, and a recorded robot replay. It runs in a
WebGPU-capable browser; the recordings above remain available everywhere.

### Inspect a component before running it

```bash
bash run/phiroom.sh describe generation
bash run/phiroom.sh plan generation trellis -- --help
bash run/phiroom.sh describe phiview
```

`plan` shows the command and required runtime. Replace `plan` with `run` when
the model environment, input data, and weights are ready.

### Open a prepared scene in PhiView

After completing [PhiView setup](docs/PHIVIEW.md), initialize your configuration
and launch a prepared scene:

```bash
bash run/phiview.sh init --config configs/phiview.local.yaml \
  --simany-root "$PWD/integrations/phiview/backends/simany" \
  --outputs-root "$PWD/outputs" \
  --data-root /path/to/scannetpp --splats-root /path/to/splats

bash run/phiview.sh run viewer --action demo \
  --config configs/phiview.local.yaml --scene ROOM_factory
```

Select an object, make it simulatable, then use the viewer's shooting, friction,
and movement controls. The harmonizer can be enabled separately.

## 3. Components

### Scene reconstruction

Build the scene representation from captured views, camera geometry, meshes,
and 3D Gaussian splats. See [reconstruction](pipelines/reconstruction/).

![Reconstructed kitchen](media/Q01_27dd4da69e.webp)

### Object discovery and generation

Find objects in the captured scene and generate complete asset candidates with
the available model adapters. See [discovery](pipelines/discovery/) and
[generation](pipelines/generation/).

![Object selection and generated candidate](media/component-generation.webp)

### Metric registration

Align each candidate with its observed object so position, orientation, and
scale agree with the scene. See [registration](pipelines/registration/).

![Generated bottle registered to the scene](media/bottle-06.webp)

### Background completion

Remove the selected object's original appearance and fill its exposed background
before moving the replacement. See [inpainting](pipelines/inpainting/).

![Background removal and completion](media/component-background.webp)

### Physics and PhiView

Connect collision geometry, mass, friction, and Gaussian appearance. The example
below compares masses of 0.15 and 0.75 kg with friction coefficients of 0.05 and
0.80. See [physics](pipelines/physics/) and [PhiView](https://github.com/RunyiYang/PhysicalView).

[![Mass and friction comparison in PhiView](media/spray-mass-friction-impact.webp)](media/spray-mass-friction-2x2.mp4)

### Robot interaction

Render the robot alongside reconstructed objects and use the simulation interfaces
for interaction. The recordings above show scripted grasp-and-place sequences.
Policy integration and evaluation tools are documented in [robotics](docs/ROBOT.md).

[![Bottle manipulation in PhiView](media/demo-09.webp)](media/demo-09.mp4)

### Appearance harmonization

Apply optional harmonization to the rendered view while preserving the recorded
camera path and physical state. See the [harmonizer integration](integrations/harmonizer/)
and [matched-view comparison](https://insait-institute.github.io/PhiRIE/#motion).

![Matched views with and without harmonization](media/paper-teaser.webp)

## 4. Build a scene

Prepare posed images, the scene mesh, and a Gaussian reconstruction using
[the data guide](docs/DATA_AND_WEIGHTS.md). Configure the paths in your environment,
then run the automatic construction pipeline:

```bash
export SIMANY_SCENE=YOUR_SCENE_ID
export SIMANY_OUT="$PWD/outputs/${SIMANY_SCENE}_auto"
bash run/run_auto.sh
```

This runs object discovery, crop preparation, asset generation, registration,
collision construction, and simulator export. For background editing, run:

```bash
bash run/run_inpaint.sh
```

See [the pipeline guide](docs/PIPELINE.md) for inputs, stage outputs, and the
separate benchmark and single-image workflows.

## 5. Code map

| Directory | Purpose |
|---|---|
| `phiroom/`, `pipelines/` | Component interfaces and pipeline recipes |
| `agents/`, `models/` | Discovery, generation, registration, editing, and model adapters |
| `robo/`, `simfactory/` | Robot interaction and simulator integration |
| `integrations/phiview/` | PhiView source; upstream at [PhysicalView](https://github.com/RunyiYang/PhysicalView) |
| `integrations/harmonizer/` | Optional appearance harmonization |
| `run/`, `envs/` | Launchers and runtime environments |
| `media/` | Images and recordings used in this README |
| `configs/`, `tools/` | Runtime, capture, policy and frozen experiment configurations; release tools |
| `docs/`, `tests/` | Guides and automated checks |

The Python package and command remain named `phiroom` for compatibility with
earlier versions. Development uses `main`.

## 6. Versions

The project has developed from **0.0.0: reconstruction prototype**, through
**1.0.0: object construction and simulation**, to **2.0.0: modular pipeline and
PhiView**. The current maintenance version is **2.0.1**.

Next steps are guided installation, downloadable prepared scenes and repeatable
demos, followed by documented evaluation configurations and results. Release
notes live in [docs/releases/](docs/releases/).

## License

PhiRIE is released under the [MIT License](LICENSE). PhiView
(`integrations/phiview/`) is a separate MIT-licensed project bundled here with
its license and provenance, and the model checkouts under `third_party/` keep
their own upstream licenses.

## Citation

[Paper on arXiv](https://arxiv.org/abs/2609.26795)

```bibtex
@misc{yang2026phirie,
  title = {{$\phi$-RIE}: From Photorealistic Reconstruction to Interactive Environments},
  author = {Runyi Yang and Deheng Zhang and Xiaoye Wang and Kanzhi Wu and Lei Sun and Ajad Chhatkuli and Kunyu Peng and Luc Van Gool and Danda Pani Paudel},
  year = {2026},
  eprint = {2609.26795},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2609.26795}
}
```
