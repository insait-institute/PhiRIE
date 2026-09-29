# construction

Scene construction launchers and the SimFactory YAML pipeline runner.

`construct-auto` and `construct-gt` wrap `run/run_auto.sh` (fully automatic, no GT)
and `run/run_factory.sh` (GT-driven) and need the GPU environments and the scene
variables (`SIMANY_SCENE`, `SIMANY_OUT`, dataset roots) in the process environment or
the runtime config. `factory` runs `robo.simfactory.runner` on a YAML recipe that
selects one backend per block (segmentation, generation, reconstruction, inpainting,
simulation, evaluation).

- Inputs: a prepared scene directory (posed images, mesh, Gaussian splat) or a SimFactory YAML recipe.
- Outputs: the per-scene construction tree under `outputs/` (objects, physics, simulator exports, reports).
- Tools: SimFactory.
- Contract: [phiroom/modules/construction.json](../../modules/construction.json).

| Action | Runtime | Implementation |
|---|---|---|
| `factory` | `control` | `robo.simfactory.runner` |
| `construct-auto` | `control` | `run/run_auto.sh` |
| `construct-gt` | `control` | `run/run_factory.sh` |

```bash
bash run/phiroom.sh describe construction
bash run/phiroom.sh plan construction construct-auto -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run construction factory -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../../docs/MODULES.md) for composition and validation.
