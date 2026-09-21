# E4 — Paired Closed-Loop Manipulation Harness

**Priority:** P0  
**Paper output:** manipulation table  
**Depends on:** E0, E3 outputs, full-room collision, frozen task suites

## Research question

Do construction policy, collision representation, and photoreal observation choices change closed-loop manipulation when every procedural variable is held fixed?

This task owns all policy rollout ledgers. Other agents may add treatments or metrics, but they must use this harness and its reset bank.

## Read first

- `robo/eval/harness_runner.py`
- `robo/eval/harness_spec.py`
- `robo/eval/harness_validation.py`
- `robo/eval/episode_log.py`
- `robo/eval/main_table.py`
- `robo/eval/paired_runner.py` for tested helpers only
- `configs/experiments/harness_paper.final.template.yaml`
- `run/harness/run_paper.sh`
- `run/pi05_serve.sh`
- `robo/tasks/pi05_tasks.py`
- `robo/policy/control_contract.py`
- `docs/ROBOT.md`

Use `harness_runner.py`, not the legacy two-condition runner, for paper experiments.

## Fixed intervention blocks

### Block C — construction policy

- `fixed_single_path_raster`
- `agentic_simanyroom_raster`

Both use the same discovered object jobs and room capture. The first uses the A0 asset decisions from E3; the second uses A4.

### Block P — collision

- `private_shims_raster`
- `full_room_collision_raster`

The visual scene, object assets, policy, robot, task, and reset are identical. Each condition must point to a genuinely different collision XML/tasks file. The harness must refuse a relabeled duplicate.

### Block O — policy observation

- `mujoco_raster`
- `raw_factorized_composite`
- `harmonizer_robot_restore_c`

All three use the same full-room physics and live object state. Only RGB generation changes.

## Frozen fields

Within a comparison block, the following must be byte-identical or hash-identical:

```text
policy ID and checkpoint
policy preprocessing
robot asset and base pose
camera intrinsics/extrinsics and names
action convention and dimension
controller gains and control rate
physics timestep/substeps
language instruction
rubric and stage definitions
horizon
reset-state ID and reset seed
initial robot/object states
```

The only permitted differences are the fields declared by the block axis.

## Task suite design

Use two task families already supported by the paper:

1. object to receptacle;
2. object to region.

A task enters the frozen suite only when:

- target and destination are present in both compared constructions;
- the target is in the declared frontal reach region;
- the initial state is collision-valid;
- the task is not trivially successful at reset;
- a scripted or oracle controller can exercise the stage predicates;
- the camera sees the relevant workspace;
- the same reset can be instantiated in every treatment.

Do not select tasks based on pi0.5 success. Use geometric/task validity before policy rollouts.

## Minimum evaluation matrix

### Pilot gate

Before the full run:

- 2 rooms;
- 2 tasks per room, one per family where possible;
- 2 treatments per block;
- 5 matched reset states;
- scripted policy plus one real policy smoke.

### Paper target

- at least 4 rooms;
- at least 4 valid tasks per room;
- at least 2 independently frozen policy checkpoints if available;
- at least 60 matched episodes per compared arm, 100 preferred;
- no single room or task contributes more than 35% of all episodes;
- report exact rooms, tasks, policies, seeds, and planned episode count before success rates.

If only one policy is available, do not label the result multi-policy. Increase task/scene diversity instead.

## Required config

Copy:

```text
configs/experiments/harness_paper.final.template.yaml
    -> configs/experiments/icra2027/harness.yaml
```

Add the E3 construction-policy paths. Keep one explicit treatment record per combination. Do not create hidden environment-variable variants.

Each scene entry must declare:

```yaml
id:
factory_dir:
tasks_json:
construction_variants:
  fixed_single_path:
    tasks_json:
  agentic:
    tasks_json:
collision_variants:
  private_shims:
    tasks_json:
  full_room:
    tasks_json:
```

Extend `harness_runner.py` only if these variants cannot be represented by `options.tasks_json` and the existing treatment resolver. Preserve backwards compatibility and tests.

## Canonical commands

### Preflight

```bash
python -m robo.eval.harness_smoke \
  --config configs/experiments/icra2027/harness.yaml \
  --out outputs/icra2027/<freeze_id>/harness_smoke \
  --resets 3
```

Use a fresh smoke output directory. This exercises the canonical synthetic
ledger and its validation; it is not a physical rollout or scientific result.
The equivalent launcher is
`bash run/harness/run_paper.sh --smoke CONFIG NEW_OUTPUT_DIRECTORY`.
Set `SIMANY_PY` to the frozen interpreter when using the shell launcher.
`harness_validation` is a library invoked by the harness, not a `--dry-run` CLI.

### Policy server

```bash
bash run/pi05_serve.sh
```

### Rollout matrix

```bash
bash run/harness/run_paper.sh \
  configs/experiments/icra2027/harness.yaml \
  outputs/icra2027/<freeze_id>/harness
```

### Main table

```bash
python -m robo.eval.main_table \
  --config configs/experiments/icra2027/harness.yaml \
  --ledger outputs/icra2027/<freeze_id>/harness/harness_ledger.jsonl \
  --out outputs/icra2027/<freeze_id>/harness/table
```

If the exact CLI differs, update this README and the shell wrapper in the same commit.

## Required outputs

```text
outputs/icra2027/<freeze_id>/harness/
  resolved_harness_config.json
  reset_states.json
  harness_ledger.jsonl
  episodes/<episode_id>/
    manifest.json
    timeseries.json.gz
    video.mp4              # selected/debug episodes only
  validation.json
  table/
    main_table.json
    main_table.csv
    main_table.tex
  bootstrap/
    paired_differences.csv
    confidence_intervals.json
  failure_inventory.csv
```

Every planned `(treatment_id, reset_state_id)` must have exactly one terminal ledger row.

## Metrics

Use only:

- rollout coverage;
- object-to-receptacle success;
- object-to-region success;
- overall success;
- grasp, lift, and place stage rates;
- p95 observation/policy latency as a diagnostic;
- paired confidence intervals.

`hover` may remain in traces but is not required in the main table. Do not introduce a custom weighted success score as the headline.

## Statistics

- Primary comparison is paired by reset ID.
- Bootstrap hierarchically over room/task and then matched reset.
- Report absolute point differences and 95% intervals.
- Archive McNemar counts for binary success as a diagnostic.
- A sparse-success regime must still report stage differences and coverage.
- Do not pool treatments across different frozen contracts.

## Failure accounting

Keep separate terminal outcomes:

```text
success
task_failure
build_failure
policy_timeout
safety_termination
environment_crash
enhancer_failure
```

All except success remain in the denominator. A failed build cannot be replaced by a new easier reset.

## Tests

- same reset file reused after restart;
- one and only one row per planned episode;
- frozen-field drift detected with exact field path;
- duplicate treatment labels with identical assets rejected;
- build failure creates rows for every affected reset;
- resume skips only terminal episode IDs;
- crash injection does not resample;
- observation treatment cannot modify simulator state;
- Harmonizer failure never falls back to raw;
- table aggregation reproduces hand-calculated toy data.

## Demo video sampling

The demo agent may read this ledger but may not cherry-pick by visual preference. Export:

```text
outputs/icra2027/<freeze_id>/harness/demo_candidates.csv
```

Include one successful, one partial-progress, and one informative failure episode selected by fixed criteria: success/stage, then median latency, then lexicographic episode ID.

## Acceptance criteria

- [ ] All integrity tests pass before performance analysis.
- [ ] Every comparison changes one declared axis only.
- [ ] At least 60 matched episodes per primary arm, unless the paper explicitly narrows the claim.
- [ ] Both task families appear.
- [ ] Coverage is shown before success.
- [ ] Confidence intervals and per-task/per-policy supplements are generated.
- [ ] The construction, collision, and observation blocks are not conflated.

## Handoff

`STATUS.md` must list planned/completed episode counts, outcome counts, frozen-field validation, policy checkpoint hashes, rooms/tasks, table paths, and whether the manipulation claim is non-degenerate.
