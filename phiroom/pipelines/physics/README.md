# physics

Physical annotation and collision geometry.

- Inputs: registered object assets.
- Outputs: physical parameters; collision geometry; URDF.
- Tools: CoACD, physical annotation.
- Contract: [phiroom/modules/physics.json](../../modules/physics.json).

| Action | Runtime | Implementation |
|---|---|---|
| `annotate` | `main` | `agents.assets.s6_physics` |

```bash
bash run/phiroom.sh describe physics
bash run/phiroom.sh plan physics annotate -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run physics annotate -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../../docs/MODULES.md) for composition and validation.
