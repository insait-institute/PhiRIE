# Automatic compact CPU/camera eligibility and policy non-invocation

The new `cpu_reset_eligibility.kind: automatic_compact` mode consumes the sealed
canonical two-scene qualification and camera/scorer outputs. It authenticates
all 280 qualification cells before selecting the **predeclared** 40 episode
cells and exactly 20 shared reset identities. Full planning task suites remain
bound; no failed task can be substituted after observing qualification.

CPU rejection and camera/scorer rejection are distinct typed prebuild outcomes.
Every failed selected cell is written to the canonical harness ledger before
connecting to a policy service. Its frozen contract records
`policy_execution: not_invoked_prebuild`, the expected checkpoint identity and
planned reset, with no invented server identity or measured simulator state.
The adapter replays the original source chain before certifying such a row.

An eligible episode must still authenticate the actual policy server and
checkpoint. Service failures remain service failures. Resume keeps already
written rejection records and retries only missing cells. Pair validation
compares expected policy/checkpoint, task/reset, camera, controller and other
frozen fields even when one arm was not invoked. Only the absent live service
observation and measured reset are handled by the independently authenticated
prebuild certificate. The executed arm's identity and state remain validated.

This is an engineering adapter, not a manipulation result or policy-launch
claim gate. Existing scripted winner diagnostics remain unchanged. Final
qualification must run from the final integrated clean source; source hashes
must never be relabeled. The canonical planner may consume a persisted reset
bank only after sealed task artifacts are validated; the bank must contain the
exact predeclared 20 definitions and all 40 treatment/reset cells remain planned.

Verification includes CPU/camera artifact tampering, missing/duplicate/replaced
reset cells, frozen policy/task/controller/reset drift, forged execution
telemetry, and an offline real-policy lifecycle test proving rejected cells are
preserved before service connection and are not duplicated on resume.

```bash
source /group/worldcept/code/SimAny/outputs/test-headless-setup/env.sh
mkdir -p outputs
PYTHONPATH=.:/group/worldcept/code/openpi-wt/e4-policy-server/packages/openpi-client/src \
/group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e4_automatic_camera.py tests/test_e4_camera_scorer_gate.py \
  tests/test_e4_winner_camera.py tests/test_e4_automatic_eligibility.py \
  tests/test_e4_reset_eligibility.py tests/test_harness_validation.py \
  tests/test_harness_prebuild_policy_identity.py \
  tests/test_harness_construction_variants.py \
  --basetemp outputs/automatic-eligibility-smoke
```
