# Task 09 — Paired Closed-Loop Rollout Runner

**Priority:** P0  
**Suggested owner:** robotics evaluation engineer  
**Depends on:** Tasks 01, 06–08  
**Blocks:** Tasks 10, 12–13, 18–20

## Objective
Execute official, SimAny, raw-reconstruction, and ablation environments on matched initial-state IDs and seeds, producing synchronized videos, robot/object states, contacts, rubric events, and manifests.

## Existing code
- `robo/eval/pi05_eval.py`
- `robo/tasks/pi05_tasks.py`
- `run/slurm/pi05_closedloop.sbatch`

## Outputs
- `robo/eval/paired_runner.py`
- `robo/eval/episode_log.py`
- `run/slurm/paired_matrix.sbatch`
- `tests/test_paired_runner.py`

## Implementation steps
1. Generate reset states once and serialize IDs; never independently resample by method.
2. Treat missing assets/uninstantiable states as coverage failures.
3. Synchronize policy query schedule, action chunks, physics dt, horizon, and termination.
4. Store full state/contact/rubric time series plus compressed policy-camera videos.
5. Resume at episode granularity without changing seeds.
6. Add early diagnostics for all-zero/all-one tasks while preserving completed data.
7. Separate environment crash, build failure, policy timeout, safety termination, and task failure.

## Tests
- Repeating one manifest yields identical deterministic scripted-policy logs.
- Paired reset poses match numerically across methods.
- Crash recovery does not duplicate or change episode IDs.
- Coverage accounting includes failed builds.

## Acceptance criteria
- [ ] One command launches a complete policy × task × method × seed matrix.
- [ ] Every episode has video, state, contact, rubric, and manifest records.
- [ ] Official and SimAny runs differ only in declared scene-construction fields.

## Paper artifact unlocked
All paired policy results and qualitative failure videos.
