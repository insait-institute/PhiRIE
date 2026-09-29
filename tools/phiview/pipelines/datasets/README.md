# Datasets

Native LIBERO and BEHAVIOR source adapters; ScanNet++ and DROID scene contracts.

Development branch: `block/datasets`. All blocks are integrated on `main`;
branch switching is for development, not a runtime requirement.

Tools: MuJoCo, PointWorld-BEHAVIOR WDS.

## Inputs and outputs

User-provided licensed simulator assets, native XML/state or WDS shards.

Posed RGB-D, calibration and provenance. LIBERO and BEHAVIOR state alignment are GT-assisted; WDS RGB may be 320x180.

## Calls

```bash
uv run physicalview run datasets --action libero -- --roster examples/libero-roster.json --index 0 --root data/libero
uv run physicalview run datasets --action behavior -- --task 0002 --scene-name behavior_task0002 --root data/behavior --tasks-config /path/to/behavior-tasks.yaml
```

Use `plan` instead of `run` to inspect the exact argv, environment, inputs and declared
artifacts. `run` executes sequentially, saves job receipts, and stops on a failed stage.

Implementation:

- [`physicalview/paper_libero.py`](../../physicalview/paper_libero.py)
- [`physicalview/paper_behavior.py`](../../physicalview/paper_behavior.py)
- [`physicalview/scene_state.py`](../../physicalview/scene_state.py)
