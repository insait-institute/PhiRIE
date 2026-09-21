# Full qualification publication source

`robo.eval.paper_pipeline` accepts optional engineering source
`full_qualification`: the complete terminal `gate.json` and the externally pinned
`completion_audit.json` generated after original-source independent validation.
The shared producer remains `run.icra2027.e4_planning_terminal`; no geometry,
qualification, reset, rollout, or metric computation is added to the formatter.

The paper adapter validates the original E0/config/clean source, exact bundle
members, independent replay receipt, and original 269-query roster. It recounts
all 2,690 logical cells and preserves all 50 scenes/1,871 objects and 5,886 budget
exclusions. Unavailable cells cannot acquire reset/camera/physics/policy telemetry.
Missing construction and planning outcomes remain distinct from failed measured
prerequisites; checked cells are distinct from completed 900-step checks.

The generated `full_qualification_engineering.tex` traces every numeric cell to
an original gate field in `claim_ledger.csv`. It reports executed rollouts as a
count; unmeasured policy success stays null. The 160-episode policy target is not
an instantiated reset bank. Publication is preliminary and cannot enable a
manipulation or full scientific submission claim.

Reproduce smoke, including tamper, dropped-denominator, false measurement,
failed-attempt runtime, and no-overwrite negatives:

```bash
PYTHONPATH=. /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_paper_full_qualification.py tests/test_paper_pipeline_audit.py \
  tests/test_e4_full_terminal.py --basetemp=.t/e9-full
```

Result: 124 passed (3.57 seconds). Initial two development invocations failed
only the test helper import; exact logs remain `outputs/review/focused*.log`.
Actual full source publication waits the final E4 recovery and independent audit.
Reserve a new paper freeze/config; never append a table to an existing freeze.

The publication refinement displays selected/input semantic queries in the same
cell, making the predeclared selection budget visible in the paper as well as
the source JSON. It additionally authenticates the E0-bound independent driver
and raw validation-process receipt (successful exit, exact command, elapsed time
and returned gate), rather than trusting only the receipt summary. The expanded
negative suite passes126tests in3.48seconds. Real publication remains blocked
until the complete independent E4 audit is available.
