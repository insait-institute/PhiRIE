# Reconstruction

Train a Gaussian scene from posed RGB and an initial point cloud.

Development branch: `block/reconstruction`. All blocks are integrated on `main`;
branch switching is for development, not a runtime requirement.

Tools: SimAny agents.recon.gsplat_train, gsplat.

## Inputs and outputs

Calibrated scene contract and initial colored point cloud.

Inria-format Gaussian PLY and training report. Holdout RGB metrics are separate from capture quality.

## Calls

```bash
uv run physicalview run reconstruction --config configs/local.yaml --scene room --scene-dir data/scannetpp/data/room --out data/splats/room.ply -- --init-ply data/scannetpp/data/room/init_points.ply --iters 15000
```

Use `plan` instead of `run` to inspect the exact argv, environment, inputs and declared
artifacts. `run` executes sequentially, saves job receipts, and stops on a failed stage.

Implementation:

- [`physicalview/native_demo.py`](../../physicalview/native_demo.py)
- [`physicalview/pipeline.py`](../../physicalview/pipeline.py)
