# Simulation

Physical annotation and collision-scene export.

Development branch: `block/simulation`. All blocks are integrated on `main`;
branch switching is for development, not a runtime requirement.

Tools: MuJoCo, CoACD, physical parameter estimation.

## Inputs and outputs

Accepted object assets and physical metadata.

MJCF and physical parameters. The viewer exposes fall, friction, throw and projectile interactions.

## Calls

```bash
uv run physicalview run simulation --action annotate --config configs/local.yaml --scene room --out outputs/scenes/room_factory
uv run physicalview run simulation --action export --collision room --config configs/local.yaml --scene room --out outputs/scenes/room_factory
```

Use `plan` instead of `run` to inspect the exact argv, environment, inputs and declared
artifacts. `run` executes sequentially, saves job receipts, and stops on a failed stage.

Implementation:

- [`physicalview/phiview_sim.py`](../../physicalview/phiview_sim.py)
- [`physicalview/paper_libero.py`](../../physicalview/paper_libero.py)
- [`physicalview/pipeline.py`](../../physicalview/pipeline.py)
