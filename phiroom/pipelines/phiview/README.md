# phiview

Image-only browser and server-side interactive demo.

- Inputs: original full-resolution Gaussian scene; prepared assets; local viewer config.
- Outputs: server-rendered frames; selections; simulation interactions.
- Tools: gsplat, MuJoCo, PhiView.
- Contract: [phiroom/modules/phiview.json](../../modules/phiview.json).

| Action | Runtime | Implementation |
|---|---|---|
| `blocks` | `phiview` | `physicalview.cli` |
| `init` | `phiview` | `physicalview.cli` |
| `doctor` | `phiview` | `physicalview.cli` |
| `demo` | `phiview` | `physicalview.cli` |
| `studio` | `phiview` | `physicalview.cli` |
| `pipeline` | `phiview` | `physicalview.cli` |

```bash
bash run/phiroom.sh describe phiview
bash run/phiroom.sh plan phiview blocks -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run phiview blocks -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../../docs/MODULES.md) for composition and validation.
