# PhiView integration

This release bundles the complete PhiView source at
`2b7a01505d84d09e925c0d8b77ef61bfd7151b8f` under `tools/phiview/`.
Its upstream repository is
[RunyiYang/PhysicalView](https://github.com/RunyiYang/PhysicalView); the
original MIT license and all tracked upstream files are preserved.
`tools/phiview/SOURCE_PROVENANCE.json` records the immutable revision and
per-file checksums, and `tools/release/verify.py` checks them. No submodule or
extra clone step is required.

## Checkout and CPU tools

```bash
git clone https://github.com/insait-institute/PhiRIE.git
cd PhiRIE
uv sync --project tools/phiview --locked
bash run/phiview.sh blocks
bash run/phiview.sh doctor --profile cpu
```

`run/phiview.sh` forwards arguments to PhiView's `physicalview` CLI using the
bundled source's uv project. `plan` prints commands; `run` executes them and
retains job receipts. The viewer serves rendered images and accepts input
events; construction and simulation execute on the server. The original viser
studio is also available in the bundled source. The same actions are exposed
through the `phiview` module of the `phiroom` CLI
(`bash run/phiroom.sh describe phiview`).

## GPU and backend setup

```bash
bash tools/phiview/tools/env/sync.sh studio
bash tools/phiview/tools/env/sync.sh inference
bash tools/phiview/tools/env/sync.sh generation
uv run --project tools/phiview python tools/phiview/tools/backends/bootstrap.py simany trellis menagerie
bash run/phiview.sh init --config configs/phiview.local.yaml \
  --simany-root "$PWD/tools/phiview/backends/simany" --outputs-root "$PWD/outputs" \
  --data-root /path/to/scannetpp --splats-root /path/to/splats
bash run/phiview.sh run viewer --config configs/phiview.local.yaml --scene ROOM_factory
```

The interactive edit/registration API of the viewer talks to a separate,
pinned construction backend. `tools/phiview/tools/backends/sources.json`
records the sources the bootstrap fetches beneath the bundled source's ignored
`backends/` directory: the compatibility backend (PhiRoom revision
`26f834ca3468e21c584abd5808082a0b2fdf4c66`, which requires access to the
`RunyiYang/PhiRoom` repository), TRELLIS, `mujoco_menagerie` and the openpi
fork, each at a fixed revision. Substituting the current PhiRIE tree for that
compatibility backend is not validated for interactive construction.

An existing `SIMANY_ROOT` environment variable overrides the config, so make
sure it points to the intended backend checkout. Model weights, datasets, CUDA
native extensions and the dedicated SAM3D/openpi/Isaac Sim environments retain
their own prerequisites; see the bundled
[environment guide](../tools/phiview/envs/README.md).

## Updating the bundled source

Take a new upstream revision and regenerate `SOURCE_PROVENANCE.json` together;
`python tools/release/verify.py` verifies every file checksum. Source archives
built with `tools/release/archive.py` and a regular Git clone both include the
complete viewer source. Viewer changes go upstream to PhysicalView first
([CONTRIBUTING.md](../CONTRIBUTING.md)).

## Viewer features at this revision

- Selection: bright red selected-object masks, left-drag box selection,
  click-to-simulate, **Deselect** and **Esc** to clear the selection while
  keeping physics and saved assets. Clicking a reconstructed object outside the
  prepared regions segments it on the server (SAM3) and highlights its visible
  Gaussians; **Make simulatable** then creates a collision body.
- Mouse toolbar: **View** (left-drag orbit, right-drag pan, wheel zoom),
  **Select object** (clicks and boxes with a fixed camera) and **Move object**
  (drags the selected body horizontally at its current height). Selection
  highlights disappear while a robot policy loads or executes.
- Robot: the hardware preset is fixed to the DROID setup (Franka Panda +
  Robotiq 2F-85) with physical exterior and wrist camera views; the pi0.5
  DROID policy receives 1280×720 renders padded to 224×224 and absolute joint
  targets at 15 Hz of simulation time. Both the base and the sim co-trained
  checkpoints can execute action chunks from the viewer. Live policy execution
  does not by itself establish grasp or placement success; the packaged
  v2.0.0 robot video remains **GT assistance plus scripted IK, not a learned
  policy**. See the bundled
  [controls and validation notes](../tools/phiview/docs/PHIVIEW_SELECTION_ROBOT.md).
- Scene `fb5a96b1a2`: the demo automatically segments the green spray bottle
  from a curated source-view box and prepares a collision proxy, paused and
  ready for interaction; **Prepare green bottle** restores the same target
  without duplicates and `--no-demo-prepare` skips startup preparation. Mass
  and hidden collision geometry are unmeasured approximations.
- Rendering: close robot views use physical clipping distances of 1 cm to
  100 m, so distant inactive projectiles no longer slice through nearby robot
  links.
- Capture tools: `tools/phiview/tools/demos/capture_green_bottle.py` produces a
  downloadable folder with original/moved renders, scripted robot contact and
  push images, a shooting video and receipts; the
  [shooting capture tools](../tools/phiview/tools/demos/README.md) separate
  MuJoCo recording from Gaussian replay rendering and export slow-motion and
  clean videos with curling trails and contact sparks drawn in world space.
  The effects do not modify physics or a live viewer session.
