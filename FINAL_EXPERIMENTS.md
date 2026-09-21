# SimAnyRoom final experiments: start here

The authoritative implementation and TODO are already in this repository. Use
[the final plan](plan/icra2027/14_final_experiments/README.md),
[agent instructions](plan/icra2027/14_final_experiments/AGENT_PROMPT.md), and
[the runbook](plan/icra2027/14_final_experiments/RUNBOOK.md).

## One implementation, one budget

- System comparison: 24 instances x 10 resets x 4 arms = **960 planned units**.
- Gaussian/Harmonizer observation: 12 instances x 10 resets x 6 arms = **720**.
- Target-to-destination scope: the same 12 instances x 10 resets x 3 arms = **360**.
- Total: **2,040 planned units in 48 paired blocks**, including fresh references.

This is the protocol already committed under `configs/experiments/final_submission/`.
Do not add the earlier ZIP's 1,800-unit recipe on top, silently change the
replacement scope, or restart campaign 13's expansion quotas. Preserve all
previous results. A frozen/running study must never be retroactively migrated.

## Code and runnable checks

```bash
# In a clean checkout, using an existing CPU test environment:
export SIMANY_CAMPAIGN_PY=/absolute/existing/driver/bin/python
bash run/campaign/final_preflight.sh
bash run/campaign/finalize.sh --help
```

The agent resolves the interpreter and input paths from existing receipts.
The preflight runs CPU tests and CLI checks only. It does not submit Slurm jobs,
call model services, install dependencies, accept licenses, or spend API credits.

The production implementation is:

- `robo/campaign/finalize.py`: selection, freeze, worker binding and bounded dispatch.
- `robo/eval/final_release.py`: full-denominator joins and paired result tables.
- `robo/rendering/final_observation.py`: synchronized observation and restoration.
- `run/campaign/finalize.sh`: the actual command-line entry point.
- `tests/test_final_experiments.py`: final protocol and execution-contract tests.

Follow `select -> freeze -> verify -> admit -> prepare -> dispatch -> link -> collect`
with the exact options documented by `--help` and the runbook. `dispatch` without
`--submit` is a dry run. The first actual DEV pair is mandatory; CPU test success
is not native-model admission or experimental evidence.

## Package-name reconciliation

An earlier chat attachment called this work `14_finalization` and proposed a
separate `robo.finalize` module. The newer repository implementation is
`14_final_experiments` / `robo.campaign.finalize`. Do not apply that older ZIP
on top of this checkout. Compatibility documentation points here and
`run/finalize/run.sh` forwards to the current CLI, without translating old flags.
There is no second experiment ledger or second set of quotas.

Read the six task READMEs under `14_final_experiments` for remaining native
integration and TODOs. No full scientific completion is implied by publication.
