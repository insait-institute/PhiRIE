# E8 — Matched Simulation and Real-Robot Trials

**Priority:** P2 unless hardware and policy access are confirmed; P0 once confirmed  
**Paper output:** real-world table, part (b)  
**Depends on:** E0, E4, E7, physical robot access

## Research question

For the same room, policy, cameras, task, and initial-state specification, how closely does success in a SimAnyRoom reconstruction match autonomous physical outcomes?

This task is the only source allowed to populate `real success` or `sim-real gap` cells. Demonstration replay, teleoperation, scripted trajectories, and human judgment do not count as autonomous physical policy trials.

## Go/no-go gate

Before implementation, record in `STATUS.md`:

```text
robot available:
policy checkpoint available:
policy can run autonomously on hardware:
camera models/calibration available:
room can be captured and reconstructed:
repeatable initial-state fixture/protocol available:
safety operator available:
```

If any required item is unavailable, mark `PHYSICAL_TRIALS_AVAILABLE=false`, leave paper cells empty, and provide a claim-removal note to E9. Do not substitute simulation proxies.

## Read first

- `robo/eval/harness_runner.py`
- `robo/eval/episode_log.py`
- `robo/eval/real_world_pairs.py`
- `robo/eval/real_world_metrics.py`
- `robo/eval/real_world_table.py`
- `robo/policy/control_contract.py`
- `configs/experiments/frozen_fields.yaml`
- `plan/18_REAL_ROBOT_PAIRED_EVALUATION.md`
- the robot platform's safety and reset documentation

## Experimental design

### Rooms and tasks

Target:

- at least 3 physically accessible rooms/workspaces;
- at least 4 task-policy pairs per room;
- both object-to-receptacle and object-to-region tasks where physically meaningful;
- at least 10 real trials per task-policy pair, preferably 20;
- the same number of matched simulation reset records per task.

A smaller pilot may use 2 rooms × 3 tasks × 10 trials, but it cannot support a broad “any room” sim-real claim.

### Policies

Use frozen autonomous policy checkpoints. The same checkpoint and preprocessing must run in simulation and on the robot. If the hardware stack applies an additional action filter, controller, or safety clamp, either reproduce it in simulation or declare the mismatch and exclude the pair from the matched analysis.

### Initial states

Exact physical equality is impossible, so define a measurable reset specification before trials:

```text
robot joint/home tolerance
object pose region and orientation tolerance
target/receptacle pose tolerance
camera pose/calibration tolerance
allowed clutter inventory
lighting/exposure protocol
```

Photograph or automatically record every real reset. Estimate the realized object/robot state and pair it with the corresponding simulation reset ID. Trials outside tolerance remain rows with `reset_invalid`, not silently retried until favorable.

### Cameras

Freeze:

- camera model;
- intrinsics;
- camera-to-robot/world transform;
- image resolution/crop;
- exposure/white-balance policy;
- latency and synchronization.

The simulation camera must reproduce the same declared model. Report residual calibration error.

## Trial ordering

Use randomized, blocked ordering across tasks and conditions to avoid time drift. Pre-generate the order from a seed and persist it before collecting outcomes.

A safe example:

```text
room block -> task-policy pair -> randomized reset IDs
```

Do not collect all easy tasks first and all hard tasks later.

## Success and stages

Use the exact same semantic rubric in simulation and reality:

- grasp;
- lift;
- place/final success;
- task-specific object-to-region or object-to-receptacle criterion;
- hold duration.

The real implementation may use perception to score stages, but ambiguous trials must be independently adjudicated by two blinded reviewers or a frozen video-scoring rule. Archive the raw video and scoring provenance.

## Required record schema

Create:

```text
outputs/icra2027/<freeze_id>/real_world/paired_trials.csv
```

Required columns:

```text
freeze_id
source
room_id
task_id
task_family
policy_id
policy_checkpoint_hash
reset_state_id
trial_index
domain                 # sim or real
outcome
success
grasp
lift
place
reset_valid
reset_error_translation_cm
reset_error_rotation_deg
camera_calibration_id
rubric_version
video_path
manifest_path
failure_reason
```

The join key is `(room_id, task_id, policy_id, reset_state_id)`. Simulation and real rows must join one-to-one for the primary paired analysis.

## Required commands

Implement or complete a real-trial ingest command:

```bash
python -m robo.eval.real_world_pairs \
  --sim-ledger outputs/icra2027/<freeze_id>/harness/harness_ledger.jsonl \
  --real-records outputs/icra2027/<freeze_id>/real_world/real_trial_records.jsonl \
  --config configs/experiments/icra2027/real_robot.yaml \
  --out outputs/icra2027/<freeze_id>/real_world/paired_trials.csv
```

Then generate the table:

```bash
python -m robo.eval.real_world_table \
  --construction outputs/icra2027/<freeze_id>/real_world/workspaces.csv \
  --trials outputs/icra2027/<freeze_id>/real_world/paired_trials.csv \
  --out outputs/icra2027/<freeze_id>/real_world/table
```

Update commands and schemas in this README if implementation requires a different interface.

## Reported results

For both the full constructed set and the agent-accepted subset, report:

- rooms;
- task-policy pairs;
- simulation episodes;
- real trials;
- simulation success;
- real success;
- absolute success-rate gap;
- retained coverage.

Also archive per-task paired contingency tables, stage gaps, and bootstrap confidence intervals. Do not headline Pearson correlation with too few task-policy points.

## Agent-accepted subset

The acceptance decision must be produced before viewing real outcomes and use only construction/task-support evidence from E6. Freeze its threshold on development data. The full set must always be reported beside the selected subset.

## Safety

- hardware emergency stop and workspace limits tested before collection;
- fragile/unsafe objects excluded and documented;
- human operator remains outside the robot's moving workspace;
- every safety stop becomes a typed trial outcome;
- no autonomous retry after a collision without a fresh trial ID.

## Tests

- one-to-one sim/real join;
- duplicate real trial IDs fail;
- policy/checkpoint/camera/rubric mismatch fails matched status;
- out-of-tolerance reset remains a row;
- agent-accepted flag cannot read real outcomes;
- table full-set denominator includes failed/invalid trials according to frozen protocol;
- synthetic paired data reproduces expected success gaps;
- video and manifest paths exist for every real trial.

## Pilot acceptance criteria

- [ ] One room, two tasks, and five trials per task complete end to end.
- [ ] Simulation and physical records join without manual spreadsheet edits.
- [ ] Camera/reset residuals are measured.
- [ ] Stage scorers agree on a manually checked subset.
- [ ] Safety and failure labels are complete.

## Paper acceptance criteria

- [ ] At least 3 rooms and 12 task-policy pairs, unless claims are explicitly narrowed.
- [ ] At least 10 real trials per pair.
- [ ] Same autonomous policy checkpoint and semantic rubric in both domains.
- [ ] Full and agent-accepted sets both reported with coverage.
- [ ] Acceptance threshold frozen before real outcomes.
- [ ] Raw videos and manifests archived.
- [ ] No demonstration or teleoperation mislabeled as autonomous success.

## Handoff

`STATUS.md` must include the go/no-go result, robot/platform details, safety approval, trial counts, reset/camera residuals, matched table paths, confidence intervals, and `SIM_REAL_CLAIM=SUPPORTED|NOT_SUPPORTED|NOT_RUN`.