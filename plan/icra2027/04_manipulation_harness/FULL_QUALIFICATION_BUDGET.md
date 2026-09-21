# Authenticated full-cohort qualification budget

`e4_candidate_screen.prepare_automatic_candidates` accepts an optional
`qualification_protocol` path (CLI: `--qualification-protocol`). Copy the frozen
protocol byte-for-byte inside the experiment evidence root before invoking it.
The protocol is validated against its original cohort, role implementation, all
50 discovery manifests/audits, and the complete E3 inventory before materializing
any treatment. The existing `e4_full_protocol.protocol_from_population` selector
is replayed exactly. No controller quality or policy outcome selects the budget.

The candidate population still includes all 1,871 objects and 6,155 semantic
queries. Its sealed `qualification_selection` records the 269 selected queries,
2,690 planned reset/arm qualification cells, and 5,886 budget exclusions. Task
placement uses the complete source geometry before the task definitions are
projected onto the authenticated selection. Both arms retain every selected
query, including unavailable endpoints. Existing qualification, reset, camera,
scorer and policy thresholds are unchanged. The task-freeze remains schema 2;
`max_tasks` remains forbidden for automatic populations.

This adapter does not launch experiments, certify the full construction gate, or
create a policy ledger. The stage launcher must first authenticate all 50 scene
controls and 9,355 policy/object terminal rows. Nine scenes have no selected
queries; they remain in population coverage and need no invented task/reset
geometry. A planner unable to construct a selected task remains explicitly
unqualified, without replacement. The existing compact camera chain cannot yet
be used as if it validated the full-cohort matrix.

Reproducible focused verification:

```bash
PYTHONPATH=. /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e4_qualification_budget.py \
  tests/test_e4_automatic_task_freeze.py \
  tests/test_e4_automatic_task_planning.py \
  tests/test_e4_automatic_candidates.py tests/test_e4_full_protocol.py
SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python bash run/icra2027/preflight.sh --smoke
```

Tests cover whole-roster/source authentication, missing/duplicate/swapped budget
members, source tampering, numeric YAML scene IDs, unchanged full-geometry
placement, selected-task loss, and preservation of legacy schema/denominators.
The actual discovery/inventory read-only smoke is recorded under
`outputs/e4-budget-input-check-20260906T031533Z/binding.json` in the primary
repository. It confirms the 6,155/269/5,886 populations; it is not qualification
or an experiment freeze.
