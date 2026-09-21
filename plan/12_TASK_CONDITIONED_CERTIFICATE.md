# Task 12 — Task-Conditioned Twin Certificate

**Priority:** P0/P1  
**Suggested owner:** method lead  
**Depends on:** Tasks 05, 09–11  
**Blocks:** Tasks 13, 18–20

## Objective
Predict absolute sim-real success error for each policy-task pair and support calibrated accept/abstain decisions on held-out scenes.

## Feature groups
1. Visual: task-local render feature residual, alpha/depth consistency, residual detection, ghosting, camera reprojection.
2. Pose/geometry: registration residual, candidate disagreement, visible fraction, scale uncertainty, visual-collision distance.
3. Support/contact: penetration, support area/COM margin, settle drift, bounded perturbation sensitivity.
4. Dynamics probes: standardized push, lift, release, grasp closure/retention response.
5. Robot/control: IK and collision margins, target visibility, open-loop replay error.

## Outputs
- `robo/certification/features/*.py`
- `robo/certification/model.py`
- `robo/certification/calibrate.py`
- `robo/certification/report.py`
- `tests/test_certificate.py`

## Implementation steps
1. Compute raw features with missingness indicators; never impute missing evidence as good.
2. Fit deliberately small baselines: monotonic score, regularized linear/logistic model, calibrated tree model.
3. Split/calibrate leave-one-scene-out and additionally report leave-one-task-family-out.
4. Predict error and bad-pair probability using thresholds fixed by Task 00.
5. Produce risk-coverage, AUROC, calibration, and accepted-subset ranking metrics.
6. Compare against global PSNR, geometry F1, and drop stability.
7. Log per-feature explanations for repair routing.

## Acceptance criteria
- [ ] Accepted-set error decreases monotonically as coverage is reduced, within CI.
- [ ] Held-out-scene performance beats every global single-metric baseline or the claim is narrowed.
- [ ] Results include 100/80/60% coverage and failure cases.
- [ ] No scene identity or real success value leaks into features.

## Paper artifact unlocked
The core method contribution and certificate ablation table.
