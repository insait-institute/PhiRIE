# datasets

Native dataset preparation.

- Inputs: DROID episode or BEHAVIOR task; dataset paths.
- Outputs: posed scene directories; dataset inventories.
- Tools: DROID, BEHAVIOR, LIBERO, ScanNet++.
- Contribution branch: `module/datasets-v2` (new work: `module/datasets/<change>`).
- Contract: [phiroom/modules/datasets.json](../../phiroom/modules/datasets.json).

| Action | Runtime | Implementation |
|---|---|---|
| `droid` | `h5` | `agents.recon.droid_extract` |
| `behavior` | `h5` | `agents.recon.behavior_extract` |
| `libero` | `phiview` | `physicalview.cli` |

```bash
bash run/phiroom.sh describe datasets
bash run/phiroom.sh plan datasets droid -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run datasets droid -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../docs/MODULES.md) for composition and validation.
