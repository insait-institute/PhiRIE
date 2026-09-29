# Viewer

Server-rendered Gaussian scene viewer with image-only browser controls.

Development branch: `block/viewer`. All blocks are integrated on `main`;
branch switching is for development, not a runtime requirement.

Tools: gsplat, MuJoCo, viser.

## Inputs and outputs

Registered scene assets and original Gaussian PLY.

Rendered frames, picking, navigation and scene interaction. Robot commands retain their measured limitations.

## Calls

```bash
uv run physicalview run viewer --config configs/local.yaml --scene room_factory --out outputs/viewer
uv run physicalview run viewer --action studio --config configs/local.yaml --scene room_factory
```

Use `plan` instead of `run` to inspect the exact argv, environment, inputs and declared
artifacts. `run` executes sequentially, saves job receipts, and stops on a failed stage.

Implementation:

- [`physicalview/phiview.py`](../../physicalview/phiview.py)
- [`physicalview/phiview_scene.py`](../../physicalview/phiview_scene.py)
- [`physicalview/web/`](../../physicalview/web/)
- [`physicalview/app.py`](../../physicalview/app.py)
