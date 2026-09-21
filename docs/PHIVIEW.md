# PhiView integration

This release bundles the complete PhiView source at
`2b7a01505d84d09e925c0d8b77ef61bfd7151b8f` under `integrations/phiview/`.
Its upstream repository is `RunyiYang/PhysicalView`; the original MIT license and
all 207 tracked upstream files are preserved. `SOURCE_PROVENANCE.json` records
the immutable revision and file checksums. No submodule access is required.

## Checkout and CPU tools

```bash
git clone https://github.com/insait-institute/PhiRIE.git
cd PhiRIE
uv sync --project integrations/phiview --locked
bash run/phiview.sh blocks
bash run/phiview.sh doctor --profile cpu
```

`run/phiview.sh` forwards arguments to PhiView's `physicalview` CLI using the bundled source's
uv project. `plan` prints commands; `run` executes them and retains job receipts.
The viewer serves rendered images and accepts input events; construction and simulation
execute on the server. The original viser studio is also available in the bundled source.

## GPU and backend setup

```bash
bash integrations/phiview/tools/env/sync.sh studio
bash integrations/phiview/tools/env/sync.sh inference
bash integrations/phiview/tools/env/sync.sh generation
uv run --project integrations/phiview python integrations/phiview/tools/backends/bootstrap.py simany trellis menagerie
bash run/phiview.sh init --config configs/phiview.local.yaml --simany-root "$PWD/integrations/phiview/backends/simany" --outputs-root "$PWD/outputs" --data-root /path/to/scannetpp --splats-root /path/to/splats
bash run/phiview.sh run viewer --config configs/phiview.local.yaml --scene ROOM_factory
```

The interactive edit/registration API currently uses PhiRoom compatibility revision
`26f834ca3468e21c584abd5808082a0b2fdf4c66`, recorded in the bundled source's
`tools/backends/sources.json` and available on PhiRoom branch `compat/phiview-v0.2.0`.
The bootstrap fetches that separate checkout beneath
the bundled source's ignored `backends/` directory. It does not recursively clone a new
PhiView submodule at that historical backend revision. The parent PhiRoom main branch
has a different public-context/contract-manifest experiment API; substituting it for
this compatibility backend is not validated for interactive construction.

An existing `SIMANY_ROOT` environment variable overrides config, so ensure it points
to the intended compatible checkout. Model weights, datasets, CUDA native extensions
and dedicated SAM3D/OpenPI/Isaac Sim environments retain their own prerequisites. See
the bundled source's [environment guide](../integrations/phiview/envs/README.md).

## Updating the bundled source

Update the upstream source and regenerate `SOURCE_PROVENANCE.json` together.
`python tools/release/verify.py` verifies every upstream file checksum.
Source archives and a regular Git clone both include the complete viewer source.

Main now includes bright red selected-object masks, left-drag box selection,
click-to-simulate, a control-policy-only selector, and physical exterior and wrist
camera views. Robot hardware stays fixed to the DROID preset. The π0.5 DROID setup
uses 1280×720 camera renders padded to
224×224, with absolute joint targets at 15 Hz of simulation time. Both the base and
sim co-trained checkpoints have executed real action chunks on the A6000 demo.
See the bundled source's [controls and validation](../integrations/phiview/docs/PHIVIEW_SELECTION_ROBOT.md).

The packaged v2.0.0 robot video remains **GT assistance plus scripted IK, not a learned
policy**. New live policy execution does not establish grasp/placement success or
publication approval. Release tags and existing downloadable media are unchanged.

The demo now provides **Deselect** and **Esc** to clear selection while keeping
physics and saved assets. For `fb5a96b1a2`, it automatically segments the green spray
bottle from a curated source-view box and prepares a collision proxy, paused and
ready for interaction. **Prepare green bottle** restores the same target without
duplicates; `--no-demo-prepare` skips startup preparation. Earlier unprepared click
fragments covered by the bottle mask are consolidated with their original files
preserved. Mass and hidden collision geometry remain unmeasured approximations.

The mouse toolbar now separates **View**, **Select object**, and **Move object**.
View supports left-drag orbit, right-drag pan, and wheel zoom. Select object supports
clicks and boxes with a fixed camera. Move object drags the selected physical body
horizontally at its current height. Selection highlights disappear while the robot
policy loads or executes, then return when execution stops.

`integrations/phiview/tools/demos/capture_green_bottle.py` produces a downloadable
folder with original/moved GS images, scripted robot contact/push images, a shooting
video and reproducible receipts. The robot demo is scripted IK, not a learned-policy
success result; the partial GS reconstruction can leave exposed gaps after motion.

Close robot views use physical clipping distances of 1 cm to 100 m. Distant inactive
projectiles therefore no longer enlarge the near plane and slice through nearby
robot links. Actual EGL regression renders cover both small and large scene extents.

The [shooting capture tools](../integrations/phiview/tools/demos/README.md) separate
MuJoCo recording from GS replay rendering. The refined `fb5a96b1a2` demo requires
two distinct projectile contacts, renders gold/cyan curling trails and contact
sparks in world space, and exports a slow-motion video, a clean video, and PNGs.
The effects do not modify physics or a live viewer session. The scene-specific
GT-assisted geometry and estimated physical parameters remain explicit in the
capture notes. H200 validation reproduced both hits and the full contact trace.
