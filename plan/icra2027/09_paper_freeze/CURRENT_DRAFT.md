# Current E9 draft: authenticated raw appearance and independent controller pilot

Owner: E9 formatter agent; branch `agent/icra-e9-current-draft`.
Reserved audit freeze: `20260905-ab79101-v1` (canonical shared allocator).
This is a preliminary formatter freeze, not a common scientific experiment freeze.

The configuration `configs/experiments/icra2027/paper_current_draft.yaml` binds
legacy E1 construction, the completed fresh E2 raw-room source, and the independent
E3 automatic-discovery pilot by original path and SHA256. E2 completion/coverage
and E3 independent-evaluation receipts are authenticated before formatting.
Original producer commits and checkpoint/evidence identities remain in provenance.
No model, metric, camera, evaluator, or frozen producer output is changed.

The E3 caption is selected only by explicit supported schema/scope and authenticated
completion audit. Geometry counts are distinct from accepted/planned counts;
stability is stable/tested from the acceptance probe, not independent manipulation.
Unmatched jobs remain in coverage, and A4 has a different accepted geometry subset.
Incomplete pilot runtime is omitted rather than recomputed from partial stages.
The E2 caption states that only raw input-Gaussian appearance is populated.
The unmeasured E2 object roster remains separate from the matched E3 case study.
Legacy source captions remain supported for reproducibility.

Every editorial decision includes an exact removed/enabled sentence in the YAML
and generated CSV claim ledger. These are proposed edits: generated provenance
sets `prose_decisions_applied=false`. The root agent owns paper transfer and prose.
The title remains narrowed; full agentic, manipulation, Harmonizer, task-local,
real-capture, and sim-real claims are not promoted by this draft.

## Reproduction and gates

Run from this worktree at its committed source (resolve with `git rev-parse HEAD`):

```bash
/group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_paper_pipeline_audit.py tests/test_freeze.py --basetemp=.t/e9-contract
SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python \
  bash run/icra2027/preflight.sh --smoke
/group/worldcept/code/SimAny/.venv/bin/python -m robo.eval.freeze \
  --config outputs/icra2027/20260905-ab79101-v1-contract-config.yaml \
  --out /group/worldcept/code/SimAny/outputs/icra2027/20260905-ab79101-v1/contract
/group/worldcept/code/SimAny/.venv/bin/python -m robo.eval.paper_pipeline \
  --config configs/experiments/icra2027/paper_current_draft.yaml \
  --out /group/worldcept/code/SimAny/outputs/icra2027/20260905-ab79101-v1/paper_tables
```

Focused smoke: 26 tests passed before commit, including changed source/evidence,
unsupported scope, independent-evaluation gate, geometry/physical denominator,
raw-room coverage, runtime suppression, missing methods, and no-overwrite failures.
The generated E0 config hashes this tracked config, source tables, and supporting
receipts; exact post-commit results and source SHA are retained in the freeze.
No Slurm jobs, GPU allocation, generation checkpoint loading, or expensive rerender.
Pilot: real-input formatter command above; full scientific/submission run NOT_RUN.
Outputs include generated LaTeX, JSON/CSV, provenance, and claim decisions;
submission audit remains FAIL and `paper_ready=false` by construction.
Every existing frozen experiment remains immutable. Reruns require a new freeze ID.
