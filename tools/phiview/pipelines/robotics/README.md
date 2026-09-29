# Robotics

Robot task construction, policy serving and viewer interaction.

Development branch: `block/robotics`. All blocks are integrated on `main`;
branch switching is for development, not a runtime requirement.

Tools: Franka IK, OpenPI client/server, MuJoCo.

## Inputs and outputs

Exported simulation, task targets and optional policy checkpoint.

Tasks, action traces and explicit outcomes. Scripted reach/IK motion is not evidence of reliable manipulation.

## Calls

```bash
uv run physicalview run robotics --action tasks --config configs/local.yaml --scene room --out outputs/scenes/room_factory
uv run physicalview run robotics --action policy-server --model pi05_droid_jointpos --config configs/local.yaml --where slurm --port 8000
```

Use `plan` instead of `run` to inspect the exact argv, environment, inputs and declared
artifacts. `run` executes sequentially, saves job receipts, and stops on a failed stage.

Implementation:

- [`physicalview/robot.py`](../../physicalview/robot.py)
- [`physicalview/ik.py`](../../physicalview/ik.py)
- [`physicalview/phiview_sim.py`](../../physicalview/phiview_sim.py)
