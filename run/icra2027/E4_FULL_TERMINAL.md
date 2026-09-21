# Full E4 terminal prerequisite coverage

The existing `e4_planning_terminal` producer supports a separate explicit
`full_discovery_qualification_terminal_coverage` scope. It preserves all 50
scenes, 1,871 construction inputs, 6,155 semantic queries, 5,886 preregistered
budget exclusions, 269 selected queries, and 2,690 planned prerequisite cells.
It does not produce a rollout ledger or assign policy outcomes.

The new producer authenticates the original qualification E0, exact clean Git
checkout, runtime, configuration, and fixed CPU compatibility envelope. Its
read-only worker executes in the original interpreter and checkout and calls
the original canonical source, materialization, export, planning, and qualifier
validators. It does not rerun construction or simulator steps. Each completed
canonical qualifier row is copied with its exact logical identity and evidence.
Absent tasks receive logical cells with null physical/reset/camera/policy data.

Only two additional observed failures are supported: a protected-carve collision
rejection and no tabletop cluster with graspable objects. Both require original
failed-job logs plus exact rejection replay. Collision rejection additionally
requires the canonical report to match the authenticated paired static package.
Missing reports, unrelated errors, and partial downstream measurements fail
closed. OOM/timeouts require missing-unit recovery before complete publication.
The original compact scope and commands are unchanged.

Prepare after every original scene has a terminal successful or supported-failure
attempt. `jobs.json` maps each original scene to its final ordinary job ID;
`prior_attempts.json` maps recovered scenes to the original failed job IDs.
Both failed-attempt logs and scheduler elapsed time remain attributed.

```bash
python -m run.icra2027.e4_planning_terminal prepare-full \
  --config ORIGINAL_EXECUTION_JSON --source-root ORIGINAL_FREEZE \
  --expected-source-commit ORIGINAL_SHA --job-ids jobs.json \
  --prior-attempts prior_attempts.json --compatibility ORIGINAL_ISA_AUTHORIZATION \
  --freeze-id NEW_RESERVED_ID --out NEW_CONFIG
```

Reserve IDs with the shared `reserve_freeze_id` allocator. Commit the configuration
or bind its immutable identity as E0 resource `e4_full_terminal_config`; publish
only after exact-source smoke/schema tests and the new stage E0. The output path
must be `NEW_FREEZE/full_qualification` and must not already exist.

```bash
python -m run.icra2027.e4_planning_terminal produce-full \
  --config NEW_CONFIG --freeze-root NEW_FREEZE \
  --expected-code-commit NEW_SHA --out NEW_FREEZE/full_qualification
python -m run.icra2027.e4_planning_terminal validate-full \
  --expected-code-commit NEW_SHA --out NEW_FREEZE/full_qualification
```

Outputs use the existing sealed bundle: `gate.json`, `scene_receipts.json`,
`planned_qualification_cells.jsonl`, `manifest.json`, and `seal.json`.
`status=PASS` means complete authenticated coverage, not scientific success.
`prerequisite_checked_cells`, `actual_900_step_cells`, and `qualified_cells` are
separate. The policy target is 160; this coverage stage instantiates and executes
zero policy episodes, and success is null. Camera execution is outside this
stage. A later camera/policy producer must independently authenticate eligible
cells; it cannot treat unavailable cells as measurements or change the roster.
Scheduler elapsed sums include declared failed attempts and are hardware-specific,
not fleet elapsed or capture-to-sim runtime.

Smoke (includes missing-report, tamper, false-pass, no-overwrite, original-source,
full-denominator, and null-telemetry negatives):

```bash
PYTHONPATH=. python -m pytest -q tests/test_e4_full_terminal.py \
  tests/test_e4_planning_terminal.py tests/test_e4_full_qualification.py \
  tests/test_e4_compact_harness.py
```
