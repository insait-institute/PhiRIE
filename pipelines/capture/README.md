# capture

Video metadata and capture quality.

- Inputs: video file; capture configuration.
- Outputs: metadata JSON; validation report.
- Tools: ffprobe.
- Contribution branch: `module/capture-v2` (new work: `module/capture/<change>`).
- Contract: [phiroom/modules/capture.json](../../phiroom/modules/capture.json).

| Action | Runtime | Implementation |
|---|---|---|
| `metadata` | `control` | `capture.extract_metadata` |
| `validate` | `control` | `capture.validate_video` |

```bash
bash run/phiroom.sh describe capture
bash run/phiroom.sh plan capture metadata -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run capture metadata -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../docs/MODULES.md) for composition and validation.
