# paper

Evidence-based paper capture, packaging and verification.

- Inputs: real pipeline artifacts; frozen evidence manifests.
- Outputs: paper media packages; verified source bundles.
- Tools: PhiView capture, paper bundle validator.
- Contribution branch: `module/paper-v2` (new work: `module/paper/<change>`).
- Contract: [phiroom/modules/paper.json](../../phiroom/modules/paper.json).

| Action | Runtime | Implementation |
|---|---|---|
| `capture` | `phiview` | `physicalview.cli` |
| `pack` | `phiview` | `physicalview.cli` |
| `zip` | `phiview` | `physicalview.cli` |
| `verify` | `control` | `robo.eval.paper_bundle` |

```bash
bash run/phiroom.sh describe paper
bash run/phiroom.sh plan paper capture -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run paper capture -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../docs/MODULES.md) for composition and validation.
