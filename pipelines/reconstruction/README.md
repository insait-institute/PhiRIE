# reconstruction

Metric reconstruction and original Gaussian splats.

- Inputs: RGB frames; poses or video; initialization PLY.
- Outputs: metric scene directory; full Gaussian PLY.
- Tools: COLMAP, VGGT, gsplat.
- Contribution branch: `module/reconstruction-v2` (new work: `module/reconstruction/<change>`).
- Contract: [phiroom/modules/reconstruction.json](../../phiroom/modules/reconstruction.json).

| Action | Runtime | Implementation |
|---|---|---|
| `frames` | `main` | `agents.recon.frames` |
| `poses` | `main` | `agents.recon.colmap_poses` |
| `feedforward` | `main` | `models.vggt_scene` |
| `metricize` | `main` | `agents.recon.metricize` |
| `scene` | `main` | `agents.recon.make_scene_dir` |
| `train` | `gsplat` | `agents.recon.gsplat_train` |

```bash
bash run/phiroom.sh describe reconstruction
bash run/phiroom.sh plan reconstruction frames -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run reconstruction frames -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../docs/MODULES.md) for composition and validation.
