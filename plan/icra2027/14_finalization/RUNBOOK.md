# Compatibility pointer

Use [the current runbook](../14_final_experiments/RUNBOOK.md).

```bash
bash run/campaign/final_preflight.sh
bash run/campaign/finalize.sh --help
```

Do not execute command flags copied from the earlier ZIP. The compatibility
shell entry `run/finalize/run.sh` forwards the current CLI without translating
old semantics. Existing frozen experiments and outputs remain unchanged.
