# Discovery

Object proposals and mask refinement.

Development branch: `block/discovery`. All blocks are integrated on `main`;
branch switching is for development, not a runtime requirement.

Tools: SAM3, GT segment adapter.

## Inputs and outputs

Scene images, geometry and optional GT annotations.

Object records, RGBA crops and masks. GT-assisted and model-predicted proposals must remain distinguishable.

## Calls

```bash
uv run physicalview run discovery --config configs/local.yaml --scene room --model sam3_auto --out outputs/scenes/room_factory
uv run physicalview plan discovery --config configs/local.yaml --scene room --model gt_segments
```

Use `plan` instead of `run` to inspect the exact argv, environment, inputs and declared
artifacts. `run` executes sequentially, saves job receipts, and stops on a failed stage.

Implementation:

- [`physicalview/pipeline.py`](../../physicalview/pipeline.py)
- [`physicalview/paper_sam3.py`](../../physicalview/paper_sam3.py)
