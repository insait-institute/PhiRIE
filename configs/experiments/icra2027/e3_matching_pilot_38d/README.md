# Fixed 15-job evaluation-only pilot

Reserved evaluation freeze: `20260905-1577027-v1` (main was `1577027` at allocation).
Construction freeze: `20260905-33bd974-v1`; scene `38d58a7a31`; all 15 jobs/75 policy rows.

Before hashing any original GT file, config preparation authenticated the complete
controller seal, all job/policy rows, the TRAIN-only discovery binding, 433
proposal artifact hashes, and the unchanged policy digest. Construction seal:
`b888f50305886bc8cf5c3713991cb97c1b1bc17e00d937830083d45cd9ed13ef`.
Policy canonical digest:
`12960b451a9b52d8eac10a30ae98b956ed1a64dc9283b7c9a5c63cc5858cd54f`.

`construction_jobs.yaml` and `construction_policies.yaml` are byte-identical
copies of the executed pilot configs. The original three GT files' content
hashes appear in `matching.yaml`; no GT geometry metric or matching has run.
The config's absolute population-roster anchor lives in the retained
`${SIMANY_ROOT}/worktrees/e3-evaluation-matching` worktree and must remain
available and byte-identical. Do not retarget it after freezing.

Only after this source/config commit is merged, pushed and checked out cleanly,
run the central preflight and create E0 from that exact source. No GPU required.

```bash
export SIMANY_EVIDENCE_ROOT=${SIMANY_ROOT}
export SIMANY_PY=${SIMANY_ROOT}/.venv/bin/python
bash run/icra2027/preflight.sh --smoke
"$SIMANY_PY" -m robo.eval.freeze \
  --config configs/experiments/icra2027/e3_matching_pilot_38d/freeze.yaml \
  --out ${SIMANY_ROOT}/outputs/icra2027/20260905-1577027-v1/contract
```

Then execute only the declared matching pilot, recording its manifest SHA256:

```bash
"$SIMANY_PY" -m agents.eval.eval_vs_gt \
  --matching-config configs/experiments/icra2027/e3_matching_pilot_38d/matching.yaml \
  --contract-manifest ${SIMANY_ROOT}/outputs/icra2027/20260905-1577027-v1/contract/freeze_manifest.json \
  --out ${SIMANY_ROOT}/outputs/icra2027/20260905-1577027-v1/evaluation_matching
```

Use the evaluator and aggregation commands in
`run/icra2027/E3_AUTOMATIC_EVALUATION.md`, with these exact values:

- `EVAL_FREEZE=20260905-1577027-v1`
- `CONSTRUCTION_FREEZE=20260905-33bd974-v1`
- `SCENE=38d58a7a31`
- `JOBS=configs/experiments/icra2027/e3_matching_pilot_38d/construction_jobs.yaml`
- `POLICIES=configs/experiments/icra2027/e3_matching_pilot_38d/construction_policies.yaml`
- `EVAL_CONTRACT=${SIMANY_ROOT}/outputs/icra2027/20260905-1577027-v1/contract/freeze_manifest.json`

The full 50-scene evaluation still requires a separate frozen config and its own
pilot gate. Preserve unmatched jobs and negative geometry/stability results.
