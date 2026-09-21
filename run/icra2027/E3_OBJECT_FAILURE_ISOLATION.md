# Per-object generator failure isolation

Owner: e3_rvg_cohort; branch `agent/icra-e3-object-failure-isolation`.
This is a code-only repair. Existing frozen generation worktrees and outputs
remain unchanged. Any recovery requires a new source/config/freeze and an
explicit missing-unit/reuse plan; this patch does not authorize in-place
resumption or relabel earlier aborted jobs as successful.

Strict TRELLIS now records an object's exception and continues to the next
planned object. It retains partial output bytes, the exception type/message,
seed and wall time. Existing producer records or proposal artifacts still
reject an attempted overwrite before an object call. The collector requires
an explicit `generated` record with seed42 plus every required artifact;
partial files cannot override a failure record. Non-strict legacy execution
continues to raise errors as before.

RVG had the same failure propagation: it wrote one failed record and then
aborted the rest of the scene. It now retains that typed failure and continues
the unchanged loop. Neither tool retries the failed object, changes opaque
RGBA handling/rembg behavior, changes model calls, or changes any threshold.
Completing a scene process means its jobs were visited; individual failed
objects remain failed in the original denominator.

Smoke: 44 passed in4.55s:

```bash
python -m pytest -q tests/test_e3_object_failure_isolation.py tests/test_e3_trellis_generation_pilot.py tests/test_e3_rvg_generation_pilot.py tests/test_e3_fresh_rvg.py tests/test_e3_rvg_cohort.py tests/test_e3_generation_cohort.py --basetemp=.t/object-isolation
```

Compatibility: 58 passed in4.58s (includes the eight isolation tests again):

```bash
python -m pytest -q tests/test_e3_fresh_generation.py tests/test_e3_fresh_generation_config.py tests/test_agentic_automatic_population.py tests/test_e3_object_failure_isolation.py --basetemp=.t/object-compat2
```

Tests inject empty sparse coordinates and the observed rembg attribute error,
verify later objects execute once with unchanged seed/formats, preserve
unprepared jobs, reject existing failed records, and reject apparent complete
files whose producer record is failed or lacks explicit success.

Pilot/full: NOT_RUN for this new source. No GPU jobs submitted. Claim gate:
NOT_RUN; the exception isolation does not establish scientific improvement.
