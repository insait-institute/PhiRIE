# Runtime accounting in newly frozen canonical aggregates

`robo.eval.agentic_ablation --aggregate` adds `runtime_accounting.json` to its
existing atomic output/seal. For automatic inventories, the main CSV/JSON runtime
is attributed raw generation process time plus the existing per-policy canonical
registration, physics and retry time. `canonical_runtime_minutes_per_scene`
retains the previous phase scope; `generation_process_minutes_per_scene` reports
the added component. `runtime_accounting_status` is PASS or NOT_RUN. This changes
no decisions, thresholds, proposal schema, or previously frozen outputs.

A0 is charged one TRELLIS process per scene. A1–A4 each receive the same TRELLIS
and RVG process costs. Complete process timers include loading, failures and
invalid/unavailable outputs. Object timers are diagnostics contained within the
process timer, never an additional charge. Process residual includes startup and
work without a completed object timer; it is not necessarily pure overhead.
These policy-attributed costs must not be summed to estimate physical fleet time.

## Recovery histories

The selected initial pool is authenticated by the sealed inventory audit. Before
creating the new canonical E0 contract, append complete recovery histories to its
jobs YAML. The whole jobs config must be an E0 experiment-config resource:

```yaml
runtime_accounting:
  schema_version: 1
  process_histories:
    - scene_id: 40aec5fffa
      tool: reconviagen
      pools:
        - path: /absolute/old-freeze/rvg_initial/40aec5fffa/proposal_pool.json
          sha256: <original-failed-pool-content-sha256>
        - path: /absolute/recovery-freeze/rvg_initial/40aec5fffa/proposal_pool.json
          sha256: <completed-recovery-pool-content-sha256>
```

Use real exact pool anchors, not the example paths above. For the current RVG
recovery, include original `20260905-d1e8e21-v1` and replacement
`20260905-58cabb1-v1` for each repaired scene. Ordinary unrecovered scenes need no
history entry. Histories must contain the selected pool once, every parent
declared in the frozen generator config, and no unrelated process. Pool source,
E0, config, input and complete job roster are verified. Original failed processes
are charged once even when no object completed. Duplicate, tampered, mismatched
or unrelated records raise errors. Missing parent anchors or timings produce
null totals and NOT_RUN; they cannot silently become zero.

The sidecar leaves actual new-wave elapsed time and cache materialization time
null until separately measured. Capture, discovery, Gaussian training,
observation export, evaluation and scheduler queue time are excluded. Never
label this sum capture-to-sim or end-to-end latency.

Validation:

```bash
mkdir -p outputs
PYTHONPATH=. /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  --basetemp outputs/runtime-tests \
  tests/test_agentic_runtime_accounting.py tests/test_agentic_ablation.py
```

The existing 15-job pilot can be inspected read-only with
`robo.eval.agentic_runtime_accounting.account_runtime` after loading its sealed
controller through `_load_control_scene`. Any diagnostic sidecar belongs in a
new validation directory, never in an existing aggregate or seal. Its archived
runtime remains the previously reported incomplete scope.
