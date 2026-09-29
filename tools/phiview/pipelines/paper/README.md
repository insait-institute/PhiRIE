# Paper

Capture, review and package reproducible scene-feature figures.

Development branch: `block/paper`. All blocks are integrated on `main`;
branch switching is for development, not a runtime requirement.

Tools: Lossless PNG, editable SVG, offline ZIP, execution receipts.

## Inputs and outputs

Constructed scenes; campaign root for multi-scene reports.

14 feature groups per scene, sidecar hashes, panels and downloadable review packs. Captured, complete and publication-approved are separate states.

## Calls

```bash
uv run physicalview run paper --action capture --config configs/local.yaml --scene room_factory --out outputs/paper/room
uv run physicalview run paper --action pack --root outputs/paper
uv run physicalview run paper --action zip --root outputs/paper --out outputs/downloads
```

Use `plan` instead of `run` to inspect the exact argv, environment, inputs and declared
artifacts. `run` executes sequentially, saves job receipts, and stops on a failed stage.

Implementation:

- [`physicalview/paper_capture.py`](../../physicalview/paper_capture.py)
- [`physicalview/paper_campaign.py`](../../physicalview/paper_campaign.py)
- [`physicalview/paper_pack.py`](../../physicalview/paper_pack.py)
- [`physicalview/paper_download_pack.py`](../../physicalview/paper_download_pack.py)
