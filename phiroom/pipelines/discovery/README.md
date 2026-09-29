# discovery

Automatic object discovery and mask refinement.

- Inputs: scene images and reconstructed geometry.
- Outputs: auto_instances.npz; objects/objects.json; refined masks.
- Tools: SAM3.
- Contract: [phiroom/modules/discovery.json](../../modules/discovery.json).

| Action | Runtime | Implementation |
|---|---|---|
| `automatic` | `sam3` | `agents.discover.auto_segment` |
| `prepare` | `main` | `agents.discover.factory_prepare` |
| `refine` | `sam3` | `agents.discover.factory_refine_masks` |

```bash
bash run/phiroom.sh describe discovery
bash run/phiroom.sh plan discovery automatic -- --help
# Execute an action with its native backend flags after --:
bash run/phiroom.sh run discovery automatic -- --help
```

`plan` is read-only. `run` starts the backend and records its exit status and log.
Set runtime paths and scene context using `--config configs/runtime.local.json`.
See [module interfaces](../../../docs/MODULES.md) for composition and validation.
