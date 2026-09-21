# Sealed TRELLIS.2 mesh geometry pilot

`e3_trellis2_mesh_probe` exposes an evaluation-only command for its existing
15-object engineering pilot. It authenticates the original clean TRELLIS.2
source/E0/config, complete generation audit, TRAIN observation closure and every
sealed registration/probe artifact before invoking the original matching-source
validator. The latter runs in its own unchanged checkout, validates its original
E0 and reference seal, and retains the original policy-independent matches.
Discovery identities must be identical across both studies. No generation,
registration, matching, selection or threshold is rerun or changed.

The evaluator reuses the canonical agentic mesh loader, 20,000-sample surface
sampler and proposal-ID evaluation seed, applies the recorded TRAIN transform,
and calls `robo.eval.fidelity_metrics.geometry_metrics` at exactly 20 mm.
All 15 jobs remain present; the 12 unmatched jobs have null geometry. Appearance
is null for every row. Geometry is conditional on the three matched jobs.
Construction settle evidence is retained as such, never held-out manipulation.
No native Gaussian, full twin, A0--A4 treatment or scientific improvement claim
is enabled. Original generation/registration costs are not relabeled as new
runtime. No legacy paper table is changed by this command.

After the tested source/config commit is merged and its exact-source E0 smoke
passes, create a new contract from `trellis2_geometry_pilot/freeze.yaml`, then run:

```bash
SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny \
  /group/worldcept/code/SimAny/.venv/bin/python -m run.icra2027.e3_trellis2_mesh_probe \
  --geometry-config configs/experiments/icra2027/trellis2_geometry_pilot/evaluation.json \
  --contract /group/worldcept/code/SimAny/outputs/icra2027/RESERVED_ID/contract/freeze_manifest.json \
  --out /group/worldcept/code/SimAny/outputs/icra2027/RESERVED_ID/trellis2_geometry
```

`RESERVED_ID` is the checked-in `evaluation.json` freeze_id. Use one ordinary
CPU job, no array or GPU, with BLAS/OpenMP threads capped. Both JSON and CSV are
published atomically without overwriting any prior output. Smoke tests:

```bash
python -m pytest -q tests/test_trellis2_geometry_evaluation.py tests/test_e3_trellis2_mesh_probe.py
```

The tests exercise real canonical surface metrics, all-job/null conditioning,
repeatability, atomic output/no-overwrite, source-before-GT ordering, and
source/registration/reference/population/discovery/audit tamper rejection.

## Shared-evidence publication recovery

The initial pilot job832575 completed validation/geometry computation but failed
before writing any metric output because the legacy fidelity bundle publisher
only accepts its checkout-local root. The unchanged canonical agentic publisher
(`_atomic_directory`, `_write_json_inside`, `_write_bytes_inside`) supports the
already-authenticated shared evidence repository and is now used here. A seal
binds the exact JSON/CSV bytes. Metric loading/sampling/estimation is unchanged.

Use the new `trellis2_geometry_publication/evaluation.json` and sibling
`freeze.yaml` for the recovery. Its separate reserved ID and source preserve
all original source/reference anchors and record the failed attempt's scheduler
receipt/log. No original freeze or source is altered. Shared-root publication,
escape/symlink rejection and seal tamper detection have dedicated regressions.
