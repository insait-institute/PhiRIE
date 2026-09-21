# Final commands and artifact contracts

Use the checked-out code, not copied snippets with stale environments. Agent
resolves actual paths from existing source/run receipts. No new GPU work is
launched just by creating the plan.

## 1. Inspect, test, select

```bash
git fetch origin
git worktree add --detach ../PhiRoom-final-experiments origin/main
cd ../PhiRoom-final-experiments
export SIMANY_CAMPAIGN_PY=/absolute/existing/driver/bin/python
"$SIMANY_CAMPAIGN_PY" -m pytest -q tests/test_final_experiments.py tests/test_system_campaign.py
bash run/campaign/finalize.sh --help
```

The driver needs the already used numpy/PyYAML/Pillow/scipy and test dependencies.
Never upgrade the fixed native or pretrained-model environments to install these.
Use ordinary CPU jobs for tests and graphics/native allocations for actual scenes.

Make a new RUN directory. Inventory actual native source instances in `pool.jsonl`:
`case_id, canonical_instance_id, layout_id, task_id, dataset:"robocasa", split:"test",
prior_outcomes_seen:boolean, source_index:{path,sha256}`. Reference acquisition
can establish canonical identity; it is not a successful construction filter.
`source_index` must identify actual source files, not an invented receipt.

```bash
bash run/campaign/finalize.sh select --pool "$RUN/pool.jsonl" \
  --protocol configs/experiments/final_submission/protocol.yaml \
  --out "$RUN/selection"
```

Copy recipe.template.json to RUN and bind actual model-lock, DEV-selection and
policy-contract receipts. The generator/front-end/editor choices must be concrete,
not RESOLVE placeholders. Set frozen_before_outcomes true only after actual DEV
selection. No defaults are chosen using final policy outcomes.

```bash
bash run/campaign/finalize.sh freeze --cases "$RUN/selection/cases.jsonl" \
  --protocol configs/experiments/final_submission/protocol.yaml \
  --recipe "$RUN/recipe.resolved.json" --out "$RUN/final"
bash run/campaign/finalize.sh verify --root "$RUN/final"
```

Expected:48 blocks,2040 units. Freeze errors on insufficient/duplicate source
strata. A smaller prospective study needs a new protocol/study_id and an explicit
reason before outcomes; it must not be advertised as the original quota.

## 2. Admit real native integration, bind workers

After an actual DEV pair and identity/scorer checks (success need not be positive):

```bash
bash run/campaign/finalize.sh admit \
  --identity "$DEV/identity_gate.json" --rubric "$DEV/rubric_gate.json" \
  --reference "$DEV/reference/result.json" --comparison "$DEV/comparison/result.json" \
  --out "$RUN/admission"
```

Both episode files must explicitly declare DEV. This wrapper copies receipts,
not outcomes. It does not fix a failed import or approve unmeasured correctness.

Each binding row is `{block_id, task:{path,sha256}}`. The task is a REAL existing
campaign `command` task, with the same block ID, `purpose:native_policy`, source
and environment pins, `fresh_reference:true`, official method labels, and the
admission receipts. Add an input `final_contract` pointing to that exact sealed
`final/blocks/<block_id>.json`; the argv must consume `{final_contract}`.

The argv invokes the admitted native worker implementing SOURCE_INTEGRATION.md.
Do not invent a nonexistent `--final-contract` flag on the old worker. Implement
and test that narrow driver argument if missing. Its output manifest must list
`final_result_index` (see below). All paths in tasks use actual receipts.

```bash
bash run/campaign/finalize.sh prepare --root "$RUN/final" \
  --bindings "$RUN/worker_bindings.jsonl" --runtime "$RUN/runtime.resolved.yaml" \
  --out "$RUN/bound-001"
```

Runtime uses the existing campaign format with execution_ready, source_root and
per-environment python/sbatch_args. Preserve interpreter symlinks. Only admitted
workers are bound. Unbound blocks stay in `unbound.json` and the final denominator.
A partial ready wave is permitted, a false complete release is not.

## 3. Global ordinary-job scheduling

Write `registry.json` with `max_active_jobs:4`, absolute existing `bundles` (each
ends in the prepared `jobs` directory), and IDs of other active project jobs in
`additional_active_job_ids`. Include all final/follow-up bundles. Optional
account_active_job_ceiling is a job-count bound, not a GPU/QOS replacement.

```bash
bash run/campaign/finalize.sh dispatch --registry "$RUN/registry.json" --max-jobs 1
bash run/campaign/finalize.sh dispatch --registry "$RUN/registry.json" --max-jobs 1 --submit
# After real smoke, not only successful sbatch:
bash run/campaign/finalize.sh dispatch --registry "$RUN/registry.json" --max-jobs 4 --submit
```

No arrays. Lock/cap spans registered bundles. Respect account GPU/memory quotas
and unrelated jobs as well. Do not delete uncertain submission intents. Reconcile
with squeue/sacct, retain the old attempt and create a new bound attempt only if
needed. A new policy process repeats its entire paired block, not just a favored
arm. Never select the best of repeated attempts.

## 4. Export an existing native block, not a new outcome ledger

Native worker finishes by writing its existing canonical planned/ledger JSONL.
The crosswalk has **every** final_unit_id and its native_unit_id, even missing or
unexecuted units. Recorded units include an `execution_contract` receipt.
The latter binds native_unit_id,result_sha256,recipe_sha256 plus file receipts:
`policy_contract,camera_contract,task_contract,reset_contract,physics`.

These files are the actual canonical runtime definitions, not display names.
Physics is canonical physical geometry/parameters/reset state without renderer
paths; it must be byte-equivalent between observation arms at the same reset.
Same-process policy_identity is read from original result.json, not crosswalk.

GS/enhanced arms also bind `state_sync`: passed, measurement_kind=real_runtime,
frames_checked>0,max_state_lag_ticks=0,silent_fallback_frames=0. Export this from
actual observation events, not a manual pass flag. The observer checks sync but
raw geometry/occlusion correctness remains a separate DEV visual test.

Official method units bind `official_method`: method=SimFoundry/PolaRiS,
upstream_commit,adaptation=common_importer,acquisition,manual_minutes. Actual
assisted/extra capture information is retained. A build failure/abstention binds
`failure_evidence` with native_unit_id and its recorded classification.

```bash
bash run/campaign/finalize.sh link --contract "$BLOCK_CONTRACT" \
  --native-plan "$BLOCK_OUT/planned_units.jsonl" --ledger "$BLOCK_OUT/episode_ledger.jsonl" \
  --bindings "$BLOCK_OUT/unit_bindings.jsonl" --out "$BLOCK_OUT/final-index"
```

This checks a bijective full-roster join, actual result hashes, process identity,
method provenance and fixed fields. Its native_outputs.json is the command
adapter's output_manifest. It NEVER rewrites native outcomes or method labels.

## 5. Collect all results, including missing blocks

Provide each actual `final_result_index.json` once:

```bash
bash run/campaign/finalize.sh collect --root "$RUN/final" \
  --index "$FIRST_INDEX" --index "$SECOND_INDEX" --out "$RUN/release-001"
```

`--index` can be repeated for all48 blocks. With none, collection deliberately
returns all2040 units unmeasured, demonstrating that an empty task list is not
completion. Missing/invalid canonical data cannot turn into zero policy success.

Outputs: native_tables.csv,per_task.csv,comparisons.json,analysis_projection.jsonl,
summary.json,AGENT_REPORT.md,SHA256SUMS.txt. Projection is an analysis index; the
original native ledgers remain authoritative. Use the existing paper pipeline
for other standard geometry/appearance/semantic producers and final LaTeX.
Do not manually type scores into the manuscript.

Complete data is not a winning method or automatic paper readiness. Old F1,
real scans, the separate official PolaRiS run and all new tables keep separate
cohort/sensor/backend labels. Return a compact archive, not model weights or
credentials. See05_release for exact scientific acceptance and claim rules.
