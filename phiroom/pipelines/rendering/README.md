# rendering

Composite rendering and observation harmonization.

- Inputs: Gaussian scene; object transforms; camera poses.
- Outputs: rendered observations; image stream.
- Tools: gsplat, harmonizer.
- Contract: [phiroom/modules/rendering.json](../../modules/rendering.json).

| Action | Runtime | Implementation |
|---|---|---|
| `composite` | `gsplat` | `agents.render.gsplat_sim_render` |
| `harmonizer` | `gsplat` | `tools.harmonizer.server` (see `run/serve_harmonizer.sh`) |

```bash
bash run/phiroom.sh describe rendering
bash run/phiroom.sh plan rendering composite -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run rendering composite -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../../docs/MODULES.md) for composition and validation.
