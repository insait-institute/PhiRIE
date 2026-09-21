# experiments

Existing final-experiment and construction entry points.

- Inputs: frozen cases; protocol; explicit runtime bindings.
- Outputs: existing campaign manifests; admission and validation receipts.
- Tools: finalize, SimFactory.
- Contribution branch: `module/experiments-v2` (new work: `module/experiments/<change>`).
- Contract: [phiroom/modules/experiments.json](../../phiroom/modules/experiments.json).

| Action | Runtime | Implementation |
|---|---|---|
| `preflight` | `control` | `run/campaign/final_preflight.sh` |
| `finalize` | `control` | `robo.campaign.finalize` |
| `factory` | `control` | `simfactory.runner` |
| `construct-auto` | `control` | `run/run_auto.sh` |
| `construct-gt` | `control` | `run/run_factory.sh` |

```bash
bash run/phiroom.sh describe experiments
bash run/phiroom.sh plan experiments preflight -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run experiments preflight -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../docs/MODULES.md) for composition and validation.

`finalize` remains the authoritative experimental workflow in
[FINAL_EXPERIMENTS.md](../../FINAL_EXPERIMENTS.md). Operational receipts from the
module runner are not scientific evidence or a replacement campaign ledger.
