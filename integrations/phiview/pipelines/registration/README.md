# Registration

Align generated objects to measured scene geometry.

Development branch: `block/registration`. All blocks are integrated on `main`;
branch switching is for development, not a runtime requirement.

Tools: Sim(3) yaw sweep ICP, signed source-up, alternative source-up.

## Inputs and outputs

Generated candidates and scene geometry.

Alignment transforms and acceptance/rejection evidence; rejected geometry is not silently promoted.

## Calls

```bash
uv run physicalview run registration --config configs/local.yaml --scene room --model yaw_sweep_icp --objects 0,1 --out outputs/scenes/room_factory
```

Use `plan` instead of `run` to inspect the exact argv, environment, inputs and declared
artifacts. `run` executes sequentially, saves job receipts, and stops on a failed stage.

Implementation:

- [`physicalview/pipeline.py`](../../physicalview/pipeline.py)
