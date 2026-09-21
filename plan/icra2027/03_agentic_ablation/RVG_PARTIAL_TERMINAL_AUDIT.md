# RVG partial terminal audit

This CPU-only sidecar extends `run.icra2027.e3_rvg_terminal_audit`; it does not
rerun proposals, change the source pool, compute metrics, or create acceptance
decisions. All planned jobs remain present and ordered.

A process that exits zero may use this terminal path only when at least one
proposal is independently classified `artifact_invalid`; the process failure
classification is then `artifact_validation_failure`. Completed pools without
invalid artifacts must retain the normal post-run audit. Scheduler completion
and exit-code authentication still apply.

The existing `terminal_pool_rows` adapter independently replays
the sidecar before publication. Its `available_verified` status requires matching
artifact hashes and sizes, nonempty PLY vertices, finite numeric PLY properties,
and a valid generated seed-42 runtime. Invalid available artifacts are quarantined.
The audit additionally authenticates every original per-object JSON runtime,
finite nonnegative attempted-generation wall time, and the original generator's
consumed-PNG/source-view bindings. Missing/extra runtime records or changed input
images stop publication. Source checkout, config, E0, views receipt, input manifest,
process claim, logs, pool, JSONL, and terminal Slurm accounting remain bound.

The extra `attempt_classification` distinguishes `completed_generation`,
`proven_failed_attempt`, `view_eligibility_unavailable`,
`unknown_unattempted_suffix`, and `preparation_unavailable`.
`generation_attempt_proven` is true only for generated or typed failed runtime
records. A missing record does not prove the object was never attempted. The
legacy adapter's `per_object_attempt_proven` still means record presence.
Unavailable reasons distinguish `producer_reported_generation_failure`,
`view_eligibility_unavailable`, and (only with an absent record)
`no_per_object_completion_record_process_exit_<code>`. A generated record with
incomplete artifacts yields `producer_reported_incomplete_generation`.
`producer_reason` and the hashed original runtime record retain exact details.

## Reproduce

From a clean checkout of the audit source:

```bash
/group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e3_rvg_partial_audit.py tests/test_e3_terminal_pools.py \
  tests/test_e3_fresh_rvg.py --basetemp outputs/rvg-partial-audit-smoke/pytest

env -u SIMANY_EVIDENCE_ROOT SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python \
  bash run/icra2027/preflight.sh --smoke

SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny \
/group/worldcept/code/SimAny/.venv/bin/python -m run.icra2027.e3_rvg_terminal_audit \
  --producer-config /group/worldcept/code/SimAny-wt/e3-rvg-cohort-launch/configs/experiments/icra2027/rvg_cohort/6115eddb86.yaml \
  --producer-freeze /group/worldcept/code/SimAny/outputs/icra2027/20260905-d1e8e21-v1 \
  --out /group/worldcept/code/SimAny/outputs/icra2027/20260905-d1e8e21-v1/terminal_audit/partial-<audit-source-sha>/6115eddb86.json
```

Use a new audit output path for every audit source. Existing outputs are never
overwritten. Original producer source remains `3ff95f4`; the sidecar separately
records its own source commit and script hash. This audit cannot promote a claim
or authorize a recovery generation run.
