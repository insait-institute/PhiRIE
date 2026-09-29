# evaluation

Geometry and observation evaluation.

- Inputs: construction outputs; evaluation references where declared.
- Outputs: metrics; explicit audits and failure status.
- Tools: geometry metrics, fidelity metrics.
- Contract: [phiroom/modules/evaluation.json](../../modules/evaluation.json).

| Action | Runtime | Implementation |
|---|---|---|
| `report` | `main` | `agents.eval.factory_report` |
| `render` | `gsplat` | `agents.eval.factory_eval_render` |
| `fidelity` | `main` | `robo.eval.fidelity_metrics` |

```bash
bash run/phiroom.sh describe evaluation
bash run/phiroom.sh plan evaluation report -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run evaluation report -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../../docs/MODULES.md) for composition and validation.
