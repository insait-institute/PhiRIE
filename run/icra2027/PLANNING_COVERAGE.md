# Compact planning failure coverage

When the frozen automatic planner cannot produce a task definition, the
canonical episode writer cannot honestly encode a rollout: its build-failure
record requires a task hash and emits false success/zero score. Do not call it
with placeholder tasks. Extend the existing `e4_compact_harness` preparation
producer instead, with a verified planning-terminal receipt.

The sealed output retains 28 semantic pairs, 280 planned qualification cells,
20 canonical ID/seed definitions and all 40 planned A0/A4 episode identities.
Every policy execution is `NOT_RUN`, with null outcome, score, grasp/lift/place,
reset provenance, camera measurements and latency. ID/seed definitions are
prospective metadata, not a measured reset bank. No rollout ledger is produced.

`source_qualification_summaries` preserves authenticated completed CPU qualifier
evidence separately. Per-cell `qualification_state=NOT_RUN` means this
preparation producer has not imported those qualifier rows; it does not erase or
reinterpret their source results. A planner PASS for one scene does not imply
task applicability, camera/scorer validity or policy eligibility.

After merging and testing, use a new immutable output stage and the same clean
producer checkout for the receipt and coverage. The receipt producer validates
the untouched historical experiment source. Bind the receipt and protocol SHA256
in the stage's E0 inputs; do not edit tracked configs in the historical checkout.

```bash
python -m run.icra2027.e4_planning_terminal \
  --config "$ORIGINAL_QUALIFICATION_CONFIG" \
  --source-root "$ORIGINAL_QUALIFICATION_FREEZE" \
  --expected-source-commit "$ORIGINAL_SOURCE_COMMIT" \
  --out "$NEW_STAGE/planning_terminal.json"
python -m run.icra2027.e4_compact_harness \
  --protocol "$ORIGINAL_PROTOCOL" --protocol-sha256 "$PROTOCOL_SHA256" \
  --planning-terminal "$NEW_STAGE/planning_terminal.json" \
  --planning-terminal-sha256 "$TERMINAL_SHA256" \
  --expected-code-commit "$MERGED_PRODUCER_COMMIT" \
  --out "$NEW_STAGE/planned_coverage"
```

The output uses the existing atomic bundle publisher and rejects an existing
destination, path escape, symlink, changed receipt/source/protocol, changed
population, or promoted execution/claim field. Publication revalidates the
receipt. This establishes integrity only: applicability is `FAIL`, execution
and manipulation claim gate are `NOT_RUN`, and paper/policy launch remain false.

Focused validation (CPU only):

```bash
PYTHONPATH=. python -m pytest -q tests/test_e4_planning_coverage.py \
  tests/test_e4_compact_harness.py tests/test_e4_planning_terminal.py
```
