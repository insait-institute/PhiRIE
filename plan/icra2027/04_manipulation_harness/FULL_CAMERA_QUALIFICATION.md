# Full-cohort camera and scorer qualification

The existing camera/scorer producer now accepts the explicit
`automatic_full_cohort_camera_diagnostic` scope. The compact scope continues to
require its original 40 cells. Full scope authenticates the same exact-source,
E0-bound full qualification stage, original protocol, all 50 terminal scene
handoffs, and every predeclared task/arm/reset identity. No task or scene is
selected by camera or policy outcomes in this adapter.

Completed CPU qualifiers provide canonical measurement rows and frozen paired
task bundles. The existing renderer/scorer performs the same exact reset replay
and unchanged diagnostics. A genuine source-replayed planning failure has no
executable task or reset definition. Its planned identities are instead retained
in `planned_unavailable_cells.jsonl`, with null task/reset/camera/policy telemetry.
These identities plus canonical CPU rows cover exactly 269 queries / 2,690
planned cells. Zero-query scenes remain in the full 50-scene source closure.
The 6,155 original semantic queries and 5,886 predeclared budget exclusions are
reported in the upstream summary before conditional diagnostics.

The output reports total planned cells, actual prerequisite records,
planning-unavailable cells, CPU-eligible cells and executed camera diagnostics
separately. The existing `build_failures` field denotes cells rejected by CPU
prerequisites; it must not be presented as a count of physical trials. No policy
runs here, and no manipulation success or policy-launch gate is produced.

Derive the camera config only after full qualification closes, using the same
immutable source and E0 stage. This is a generated input artifact, not a change to
the original tracked qualification configuration:

```bash
python -m robo.eval.e4_camera_scorer_gate --write-full-automatic-config \
  --qualification-config "$FROZEN_QUALIFICATION_CONFIG" \
  --expected-code-commit "$EXACT_SOURCE_SHA" --out "$NEW_CAMERA_CONFIG"
python -m robo.eval.e4_camera_scorer_gate --automatic-config "$NEW_CAMERA_CONFIG" \
  --expected-code-commit "$EXACT_SOURCE_SHA" --automatic-preflight-only
```

The actual existing camera command omits `--automatic-preflight-only`. It remains
gated by real-data pilot validation and scheduler review; implementation tests
are not a claim that the real pilot or full camera diagnostics ran. Use the same
CPU host and pinned OSMesa/OpenPI/Menagerie environment as the qualifier. CPU
allocation checks reject hidden Slurm GPUs and arrays. Output is atomically
published at the same stage's `harness/automatic_camera_scorer`; validation rejects
another destination, source/config drift, modified cells or unavailable records.

Verification uses `tests/test_e4_full_camera.py` and the existing
`tests/test_e4_automatic_camera.py`, including genuine OSMesa reset/render/scorer
execution, unchanged visibility thresholds, missing/duplicate/replaced task
cells, foreign manifest producers, paired camera drift, unavailable-telemetry
fabrication, and no-overwrite tests. Existing full-qualification regressions are
run alongside these tests. Later qualified-matrix selection must still use the
predeclared `e4_full_protocol.select_matrix` after these gates; there is no
compact-scene shortcut or implicit policy authorization.
