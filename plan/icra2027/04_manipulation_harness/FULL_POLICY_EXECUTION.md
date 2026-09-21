# Full-cohort qualification to canonical policy execution

The existing `run.icra2027.e4_compact_policy` producer supports explicit full
camera scope. It authenticates all 50 construction sources and 2690 planned
qualification cells before calling the original frozen protocol selector.
Planning-unavailable and failed prerequisite cells remain in the source coverage.
No policy outcome enters selection. Fewer than four qualified rooms is NOT_RUN;
no threshold relaxation or replacement population is supported.

A ready matrix contains the first four qualified scene IDs, first two task IDs
per family, and five common A0-layout reset definitions: 80 resets / 160 paired
episodes. Existing task bundles, cameras, reset seeds, checkpoint, deterministic
sampling and controller remain frozen. Preparation records definitions, not
measured poses. `automatic_full` is distinct from the legacy compact scope.

Use the existing producer's `prepare --mode scripted` and `execute --mode scripted`
in a separately reserved, exact-source/E0-bound stage. After its canonical runtime
and scorer receipt passes, prepare the real matrix with `--mode real` and its
scripted receipt. No policy service is launched by preparation.

Execute that real stage first with `execute --mode real-pilot`. This runs only the
predeclared first two selected rooms and first selected task per family (40 paired
episodes), retaining the complete 80-reset bank and original configuration. It
writes the canonical ledger plus `pilot_validation.json`, `harness_pilot_summary.json`
and authenticated `real_pilot_gate.json`; it does not generate a final table.
The canonical coverage validator receives an in-memory pilot projection only
AFTER the full reset bank and eligibility are authenticated. Saved contracts and
manifests retain their original identities. All failed manipulation outcomes are
preserved; scientific success is not required. Incomplete runtime evidence,
crashes, changed source, traces, resets or policy identities block release.

`execute --mode real` reauthenticates the original pilot receipt, resumes only
missing cells in the SAME ledger, and invokes the original final validator/table
producer at 160-row completion. Original pilot rows are never rerun. The full
selected-cohort denominator does not replace the 50-scene / 6155-input-query /
269-selected-query qualification coverage. This is a single-policy engineering
study, not completion of the paper's multi-policy matrix.

Validation: `pytest -q tests/test_e4_full_policy.py tests/test_e4_automatic_eligibility.py
 tests/test_e4_compact_policy.py tests/test_harness_construction_variants.py
 tests/test_policy_sampling.py` includes roster/source tampering, full-bank
projection, all-failed but runtime-complete pilots, untouched nonpilot rows,
canonical runner resume, missing gate and no premature final-table generation.
No real policy execution is claimed by these unit tests.
