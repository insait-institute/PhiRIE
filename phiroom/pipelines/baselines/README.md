# baselines

Single-image and comparison methods.

- Inputs: single image or reconstructed scene as required by backend.
- Outputs: baseline assets; depth or reconstruction outputs.
- Tools: SHARP, MaskClustering, FlashSplat.
- Contract: [phiroom/modules/baselines.json](../../modules/baselines.json).

| Action | Runtime | Implementation |
|---|---|---|
| `single-image-depth` | `gsplat` | `agents.single_image.sharp_render_depth` |
| `maskclustering` | `main` | `agents.baselines.maskclustering` |
| `flashsplat` | `main` | `agents.baselines.flashsplat` |

```bash
bash run/phiroom.sh describe baselines
bash run/phiroom.sh plan baselines single-image-depth -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run baselines single-image-depth -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../../docs/MODULES.md) for composition and validation.
