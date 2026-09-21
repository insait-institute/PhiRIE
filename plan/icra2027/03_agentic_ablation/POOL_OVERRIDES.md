# Explicit RVG recovery pools for canonical cohort construction

The existing `e3_fresh_canonical_config` builder accepts
`--pool-override-manifest`. The checked-in
`configs/experiments/icra2027/canonical_rvg_pool_overrides.yaml` declares exactly
`40aec5fffa` and `3f15a9266d`, RVG only. Their selected pools must come from
`20260905-58cabb1-v1`, source `9e44da2cf769ffc57d3283b6ec09220072b9348a`.
The other 48 RVG pools use the explicit ordinary RVG root.

The override pins each replacement path, producer root and source commit, plus
its original failed pool hash and terminal audit hash. Missing replacement files
return WAITING and never select an old pool. Missing override entries, unknown
scenes, duplicate entries and mismatched paths/producers fail closed. The
replacement is authenticated through the same `_source_payload` producer as
ordinary scenes; no common-root symlinks or outcome-based pool discovery occurs.

When all required pools are ready, the original failed-pool closure is verified
against its terminal audit. The replacement's frozen generator config must name
that exact original as its parent. The builder places both pool path/hash anchors
in `jobs.runtime_accounting.process_histories`; the existing E0 config producer
binds the whole jobs YAML. The original process cost is retained exactly once,
without selecting original failed proposals. The manifest identity is retained
in generated jobs and readiness JSON.

The override path preserves the full 50-scene, 1,871-job, 9,355-policy-row roster.
The readiness command below does not reserve a freeze or write configs. A final
writer invocation requires all pools and audits, a new reserved freeze ID, and
an unused config destination, as before.

```bash
simany_out=/group/worldcept/code/SimAny/outputs/icra2027
SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/group/worldcept/code/SimAny/.venv/bin/python -m run.icra2027.e3_fresh_canonical_config \
  --cohort-roster configs/experiments/icra2027/construction_regimes.yaml \
  --discovery "$simany_out/20260905-76c15d5-v1/auto_discovery_pilot" \
  --trellis "$simany_out/20260905-02b54da-v2/trellis_initial" \
  --rvg "$simany_out/20260905-d1e8e21-v1/rvg_initial" \
  --trellis-freeze-root "$simany_out/20260905-02b54da-v2" \
  --rvg-freeze-root "$simany_out/20260905-d1e8e21-v1" \
  --pool-override-manifest configs/experiments/icra2027/canonical_rvg_pool_overrides.yaml \
  --trellis-terminal-audit-root "$simany_out/20260905-02b54da-v2/terminal_audit/20260905T161447Z" \
  --trellis-terminal-audit-root "$simany_out/20260905-02b54da-v2/terminal_audit/20260905T162025Z" \
  --trellis-terminal-audit-root "$simany_out/20260905-02b54da-v2/terminal_audit/20260905T164723Z" \
  --trellis-terminal-audit-root "$simany_out/20260905-02b54da-v2/terminal_audit/20260905T172952Z" \
  --rvg-terminal-audit-root "$simany_out/20260905-d1e8e21-v1/terminal_audit/partial-6a9c466" \
  --rvg-terminal-audit-root "$simany_out/20260905-d1e8e21-v1/terminal_audit/terminal-76683a6" \
  --rvg-terminal-audit-root "$simany_out/20260905-d1e8e21-v1/terminal_audit/terminal-45e46f7-20260905T194643Z"
```

Focused validation:

```bash
mkdir -p outputs
PYTHONPATH=. /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  --basetemp outputs/override-tests \
  tests/test_e3_canonical_pool_overrides.py tests/test_e3_canonical_cohort_config.py \
  tests/test_e3_fresh_canonical.py tests/test_agentic_runtime_accounting.py
```
