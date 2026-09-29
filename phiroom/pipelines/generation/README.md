# generation

Interchangeable object generation backends.

- Inputs: object crops and object metadata.
- Outputs: generated mesh and Gaussian asset candidates.
- Tools: TRELLIS, TRELLIS2, SAM3D, ReconViaGen.
- Contract: [phiroom/modules/generation.json](../../modules/generation.json).

| Action | Runtime | Implementation |
|---|---|---|
| `trellis` | `main` | `agents.models.s4_trellis` |
| `trellis2` | `trellis2` | `agents.models.s4_trellis2` |
| `sam3d` | `sam3d` | `agents.models.s4_sam3d` |
| `reconviagen` | `main` | `agents.models.s4_reconviagen` |

```bash
bash run/phiroom.sh describe generation
bash run/phiroom.sh plan generation trellis -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run generation trellis -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../../docs/MODULES.md) for composition and validation.
