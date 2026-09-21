# Runtime infrastructure

Branch: `block/runtime`.

Owns portable config, the CLI, job receipts and independent uv profiles.
`configs/cluster/insait.yaml` preserves the historical site setup.

```bash
bash tools/env/sync.sh cpu
uv run physicalview init --simany-root /path/to/PhiRoom
uv run physicalview doctor --profile studio --config configs/local.yaml
```

See [envs](../../envs/README.md), [config](../../physicalview/config.py),
[CLI](../../physicalview/cli.py) and [jobs](../../physicalview/jobs.py).
