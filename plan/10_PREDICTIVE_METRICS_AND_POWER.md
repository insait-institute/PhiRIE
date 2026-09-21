# Task 10 — Predictive Metrics, Statistics, and Episode Power

**Priority:** P0  
**Suggested owner:** statistics / evaluation researcher  
**Depends on:** Tasks 00–01  
**Blocks:** experiment freeze and Task 20

## Objective
Implement the complete statistical protocol before reading final SimAny results, including coverage, calibration, ranking, behavior agreement, bootstrap intervals, and episode budgets.

## Outputs
- `agents/eval/predictive_metrics.py`
- `agents/eval/bootstrap.py`
- `agents/eval/power.py`
- `tests/test_predictive_metrics.py`
- `docs/METRICS.md`

## Required metrics
- success-rate MAE and Brier/calibration error;
- Pearson, Spearman, Kendall τ;
- MMRV with unit tests against the reference definition;
- pairwise policy-preference accuracy with tie tolerance;
- within-task policy ranking and within-policy task ranking;
- residual correlation after removing policy/task fixed effects;
- staged progress, trajectory/contact agreement, and failure-label agreement;
- build coverage and risk-coverage curves for certification.

## Implementation steps
1. Define aggregation unit: policy-task pair, not individual correlated frames.
2. Use hierarchical bootstrap over scenes/tasks, then paired seeds; publish intervals and samples.
3. Predefine tie margins and minimum valid pair counts.
4. Run simulation-based power analysis using plausible success rates and observed variance.
5. Produce leave-one-policy/task/scene-out diagnostics.
6. Add tests showing pooled Pearson can be high despite incorrect within-task rankings.

## Acceptance criteria
- [ ] Metric code passes analytic toy examples.
- [ ] Episode budget is frozen per effect size before main execution.
- [ ] Main reporting includes coverage and ranking, not Pearson alone.
- [ ] No post-hoc seed removal is supported by the API.

## Stop conditions
If fewer than the predefined number of non-degenerate pairs remain, report descriptive results and narrow the predictive claim rather than increasing correlation by filtering.

## Paper artifact unlocked
Primary table columns, confidence intervals, and claims about preserved policy conclusions.
