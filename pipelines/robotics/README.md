# robotics

Robot task construction and policy evaluation.

- Inputs: simulation scene; task targets; policy checkpoint when evaluating.
- Outputs: task definitions; recorded action traces and outcomes.
- Tools: MuJoCo, OpenPI, scripted IK.
- Contribution branch: `module/robotics-v2` (new work: `module/robotics/<change>`).
- Contract: [phiroom/modules/robotics.json](../../phiroom/modules/robotics.json).

| Action | Runtime | Implementation |
|---|---|---|
| `tasks` | `main` | `robo.tasks.pi05_tasks` |
| `evaluate` | `main` | `robo.eval.pi05_eval` |
| `policy-server` | `control` | `run/pi05_serve.sh` |

```bash
bash run/phiroom.sh describe robotics
bash run/phiroom.sh plan robotics tasks -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run robotics tasks -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../docs/MODULES.md) for composition and validation.

The viewer demonstration uses **GT assistance plus scripted IK, not a learned policy**.
A generated command or a successful API call is not a measured manipulation success.
