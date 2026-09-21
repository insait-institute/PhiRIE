# Generation

Generate selectable 3D object candidates.

Development branch: `block/generation`. All blocks are integrated on `main`;
branch switching is for development, not a runtime requirement.

Tools: TRELLIS, ReconViaGen, SAM 3D Objects, hybrid.

## Inputs and outputs

Discovered object crops and model weights.

Canonical Gaussian and mesh candidates. Native extensions and model-specific environments remain prerequisites.

## Calls

```bash
uv run physicalview run generation --config configs/local.yaml --scene room --model trellis --objects 0,1 --out outputs/scenes/room_factory
uv run physicalview plan generation --config configs/local.yaml --scene room --model sam3d --objects 0
```

Use `plan` instead of `run` to inspect the exact argv, environment, inputs and declared
artifacts. `run` executes sequentially, saves job receipts, and stops on a failed stage.

Implementation:

- [`physicalview/pipeline.py`](../../physicalview/pipeline.py)
- [`physicalview/paper_foreign.py`](../../physicalview/paper_foreign.py)
