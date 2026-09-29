# inpainting

Object removal and scene completion.

- Inputs: factory scene with registered assets; explicit removal selection.
- Outputs: removal masks; completed views; clean background Gaussian scene.
- Tools: SAM3, Qwen, gsplat.
- Contract: [phiroom/modules/inpainting.json](../../modules/inpainting.json).

| Action | Runtime | Implementation |
|---|---|---|
| `prepare` | `main` | `agents.edit.inpaint_prepare` |
| `masks` | `sam3` | `agents.edit.inpaint_masks` |
| `views` | `qwen` | `agents.edit.inpaint_qwen` |
| `fill` | `gsplat` | `agents.edit.inpaint_fill` |
| `interactive` | `phiview` | `physicalview.cli` |

```bash
bash run/phiroom.sh describe inpainting
bash run/phiroom.sh plan inpainting prepare -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run inpainting prepare -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../../docs/MODULES.md) for composition and validation.

The `prepare`, `masks`, `views` and `fill` actions are the stages of
`run/run_inpaint.sh` (see [PIPELINE.md](../../../docs/PIPELINE.md), section 4).
Interactive object selection and prompt editing use PhiView with its pinned
compatibility backend; see [PHIVIEW.md](../../../docs/PHIVIEW.md).
