# Common-view diagnostic paper transfer

Owner: root / E9. Branch: `agent/icra-e9-room-diagnostic`.

This optional `engineering_appendix.room_common_view` source extends the existing
`robo.eval.paper_pipeline`. It keeps the complete canonical eight-method JSON and
its original row indices. It transfers the original coverage JSON separately, so
numeric claims refer to real fields in the correct original source file.

The source validator invokes clean producer `11866350bf3a12160f922b89db2fda9bf3a06fe3`
to replay its E0, original render receipts, exact paired manifest, coverage, metric
receipt, and file seal. It does not run a renderer or recompute image metrics.
The two quality rows must each contain the same predeclared eight views from one
scene, while coverage retains sixteen planned views per method. Raw availability
is sixteen views and composite availability is eight. No single-scene confidence
interval or population improvement claim is enabled.

Smoke (positive, changed denominator, changed source, alias, missing metric,
nonfinite metric, wrong source, and numeric claim-field lineage checks):

```bash
${SIMANY_ROOT}/.venv/bin/python -m pytest -q \
  tests/test_paper_room_diagnostic.py tests/test_paper_pipeline_audit.py \
  tests/test_paper_engineering_appendix.py tests/test_paper_droid_stages.py \
  tests/test_paper_paired_uncertainty.py tests/test_paper_paired_figure.py
```

Actual source replay passed in 28.51 seconds, recorded in
`outputs/room-diagnostic-review/actual_validation.json`. Both source metrics and
their original failed bootstrap job remain immutable.

Publication requires clean committed source and exact E0 first:

```bash
SIMANY_EVIDENCE_ROOT=${SIMANY_ROOT} \
${SIMANY_ROOT}/.venv/bin/python -m robo.eval.paper_pipeline \
  --config configs/experiments/icra2027/paper_room_diagnostic_draft.yaml \
  --out ${SIMANY_ROOT}/outputs/icra2027/20260906-61abf4c-v2/paper_tables \
  --paper-root ${PAPER_REPO_ROOT}
```

The publication is a working draft; E3 full evaluation, E4, E6 and final scientific
submission gates remain independent. Every subsequent source/config rerun needs
a new allocator-reserved freeze.
