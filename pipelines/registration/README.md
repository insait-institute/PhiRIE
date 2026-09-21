# registration

Metric asset alignment and candidate selection.

- Inputs: generated assets; observed object geometry.
- Outputs: registration transforms; candidate selection metadata.
- Tools: Sim(3), ICP.
- Contribution branch: `module/registration-v2` (new work: `module/registration/<change>`).
- Contract: [phiroom/modules/registration.json](../../phiroom/modules/registration.json).

| Action | Runtime | Implementation |
|---|---|---|
| `align` | `main` | `agents.assets.factory_align` |
| `select` | `main` | `agents.assets.factory_hybrid` |

```bash
bash run/phiroom.sh describe registration
bash run/phiroom.sh plan registration align -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run registration align -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../docs/MODULES.md) for composition and validation.
