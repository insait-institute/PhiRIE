# Task 18 — Matched Real-Robot and Simulated Evaluation

**Priority:** P0  
**Suggested owner:** robot experiment lead  
**Depends on:** Tasks 07–13, 17  
**Blocks:** predictive headline result

## Objective
Run matched initial-state distributions for frozen policies in phone-built simulation and physical rooms, yielding success, staged progress, failure categories, and calibrated sim-real error.

## Protocol
For each room/task/policy, sample reset state IDs in advance. Use measurement aids or fixtures to instantiate the same distribution in real and sim; do not require impossible millimeter-identical states. Freeze language, cameras, controller, horizon, safety constraints, and rubric. Randomize execution order and record all attempted trials.

## Outputs
- `configs/real_robot/*.yaml`
- `robo/real/reset_protocol.py`
- `robo/real/rubric_logger.py`
- `agents/eval/paired_real_sim.py`
- `docs/REAL_ROBOT_PROTOCOL.md`

## Implementation steps
1. Calibrate robot/cameras and record versions before each session.
2. Pilot rubric reliability with blinded dual annotation where automatic sensing is insufficient.
3. Execute power-analysis episode budget; log safety aborts separately.
4. Run corresponding sim seeds and compute all Task 10 metrics.
5. Evaluate raw, certified subset, and repaired environments without tuning on held-out real outcomes.
6. Conduct leave-one-room/task/policy analyses.

## Acceptance criteria
- [ ] At least two phone rooms have matched real trials; target three.
- [ ] Trial exclusions are predeclared and all attempts are logged.
- [ ] Certificate calibration uses disjoint rooms or cross-validation.
- [ ] Main claim is narrowed if correlation/ranking gates fail.

## Paper artifact unlocked
Primary predictive table, sim-real scatter, certificate risk-coverage, and best-paper-level central evidence.
