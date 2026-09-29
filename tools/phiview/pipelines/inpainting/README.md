# Inpainting

Prompted object removal and fitted Gaussian background completion.

Development branch: `block/inpainting`. All blocks are integrated on `main`;
branch switching is for development, not a runtime requirement.

Tools: Qwen Image Edit 2511, LaMa, Gaussian fitting.

## Inputs and outputs

Selected object IDs, masks, original scene and prompt.

Edited views, model receipts and clean background Gaussian PLY. Surface refinement modules are experiments; current removal artifacts are documented.

## Calls

```bash
uv run physicalview run inpainting --config configs/local.yaml --scene room --objects 0 --out outputs/scenes/room_factory --prompt "Remove the mug and restore the desk." --iters 1000
```

Use `plan` instead of `run` to inspect the exact argv, environment, inputs and declared
artifacts. `run` executes sequentially, saves job receipts, and stops on a failed stage.

Implementation:

- [`physicalview/phiview_inpaint.py`](../../physicalview/phiview_inpaint.py)
- [`physicalview/phiview_inpaint_guard.py`](../../physicalview/phiview_inpaint_guard.py)
- [`physicalview/inpaint_surface_fill.py`](../../physicalview/inpaint_surface_fill.py)
- [`physicalview/inpaint_surface_seeds.py`](../../physicalview/inpaint_surface_seeds.py)
