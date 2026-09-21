# Predictive metrics, bootstrap, and power (Task 10)

Implements `plan/10_PREDICTIVE_METRICS_AND_POWER.md` for the `oracle_causal`
protocol named in `docs/ICRA_RESEARCH_CONTRACT.md` sec 4 and
`configs/experiments/icra_contract_v1.yaml`. Code lives in:

- `agents/eval/predictive_metrics.py` - all pointwise/pairwise/ranking/
  coverage statistics.
- `agents/eval/bootstrap.py` - hierarchical (scene -> seed) bootstrap CIs.
- `agents/eval/power.py` - simulation-based episode-budget power analysis.
- `tests/test_predictive_metrics.py` - analytic toy-example tests, including
  the MMRV hand-computed reference case and the pooled-vs-within-task
  negative check.

This file is the reading guide: what each contract metric name maps to in
code, why the formulas are what they are, and what the code deliberately
refuses to do.

## Contract metric name -> code

`icra_contract_v1.yaml`'s `protocols.oracle_causal` names five metrics.
Every one has a same-named (or trivially-aliased) function so the mapping
from paper table column to code is a `grep`, not an inference:

| contract name              | function                                          | primitive it wraps |
|-----------------------------|---------------------------------------------------|---------------------|
| `staged_progress_mae`       | `predictive_metrics.staged_progress_mae`          | `success_rate_mae` |
| `spearman_task_difficulty`  | `predictive_metrics.spearman_task_difficulty`     | `spearman` (general) |
| `mmrv`                      | `predictive_metrics.mmrv`                         | -- |
| `calibration_brier`         | `predictive_metrics.calibration_brier`            | `brier_score` |
| `risk_coverage_auc`         | `predictive_metrics.risk_coverage_curve(...).auc` | -- |

`spearman_task_difficulty` is intentionally *just* `spearman` -- per Task
10's instruction, no task-difficulty-specific logic exists; the "task
difficulty" framing is purely in what arrays the caller passes in (predicted
vs. oracle task-difficulty rankings).

## Aggregation unit

Per plan step 1, the unit of aggregation is the **policy-task pair**, never
an individual frame or a within-episode timestep. Every correlation/ranking
function in `predictive_metrics.py` takes one scalar per policy-task pair
(a success rate, a staged-progress score, a certificate's predicted error);
frame- or timestep-level agreement (contacts, staged-progress transition
sequences) is handled separately by the set/sequence-agreement primitives
and is expected to already be reduced to one scalar or one label sequence
per episode before it reaches these functions.

## Non-degenerate pairs and the stop condition

The contract's stop condition (`min_nondegenerate_pairs: 12`, also stated in
`docs/ICRA_RESEARCH_CONTRACT.md` sec 4 and inherited from Task 00): *"If
fewer than 12 non-degenerate policy-task pairs exist ... report descriptive
results only and narrow the predictive claim ... do not drop pairs post-hoc
to raise correlation."*

This is operationalized as two distinct, loud failure modes, not a silent
`None`/`nan` return:

- `InsufficientPairsError` - too few pairs (`require_min_pairs(n, min_pairs=12)`,
  or a low-level correlation call with < 2 points).
- `DegenerateInputError` - enough pairs, but one array is constant (zero
  variance), so a correlation is mathematically undefined regardless of `n`
  (e.g. every policy scored exactly 0.0 on a task family).

**No function in this module accepts a "drop worst-N" / "exclude outliers"
/ "post-hoc seed removal" parameter.** This is a design property, checkable
by grep, not just a docstring promise: the only sanctioned way to reduce the
population that a metric sees is the pre-registered `included_bool` mask in
`risk_coverage_curve` (a certificate's own abstention decision, fixed before
results are read -- see below), and `two_way_demean`'s row/col grouping
(which reduces confounding, not sample size). If a reporting script needs to
enforce the 12-pair floor before printing a pooled correlation table column,
it should call `require_min_pairs` explicitly and catch
`InsufficientPairsError` to fall back to descriptive per-pair reporting --
never lower the threshold or drop pairs to get past it.

`within_group_correlation` (used for within-task/within-policy diagnostics
over many *small* groups, e.g. 2-3 policies per task) intentionally does
**not** enforce the 12-pair floor -- that floor is a property of the single
pooled table entry the contract governs, not of every diagnostic slice.
Groups too small to correlate get `nan`, not an exception, since the caller
is iterating over many groups and a single small group failing shouldn't
abort the whole diagnostic.

## MMRV (Mean Maximum Rank Violation)

From Li et al., *"Evaluating Real-World Robot Manipulation Policies in
Simulation"* (the SIMPLER benchmark), arXiv:2405.05941 -- the standard
robotics sim-to-real metric for whether a simulator's ranking of policies
hides real-world reversals, not just how correlated the two rankings are on
average.

For `N` items with predicted/sim score `x_i` and real/oracle score `y_i`:

```
MMRV = (1/N) * sum_i  max_{j : x_j < x_i}  max(y_j - y_i, 0)
```

(inner max over an empty set is 0). Reading it: for item `i`, look at every
`j` the predictor ranked strictly below `i`; if any such `j` actually scored
*higher* in reality, that's a rank violation, and its size is the real gap
it's hiding behind `i`'s predicted rank. MMRV is the mean, over all `i`, of
the single worst violation hidden behind it -- 0 iff the predicted ranking
never hides a real reversal.

Hand-computed reference case (`tests/test_predictive_metrics.py`,
`test_mmrv_hand_computed_reversed_ranking`): 3 policies, real success
`y = [A=0.9, B=0.5, C=0.1]`, sim score fully reversed
`x = [A=0.1, B=0.5, C=0.9]`:

- `i=A` (`x=0.1`): no `j` with `x_j < 0.1` -> term `0`.
- `i=B` (`x=0.5`): `j=A` (`x=0.1<0.5`), gap `y_A - y_B = 0.4` -> term `0.4`.
- `i=C` (`x=0.9`): `j` in `{A,B}`, `max(0.9-0.1, 0.5-0.1) = 0.8` -> term `0.8`.
- `MMRV = (0 + 0.4 + 0.8) / 3 = 0.4`.

A perfectly-agreeing ranking (`x == y`) gives `MMRV = 0.0` by construction
(every violation term is a `max(negative_or_zero, 0) = 0`) -- also tested. A
third partial-violation case (one adjacent swap out of 4 items) is tested at
`MMRV = 0.05`.

## Pairwise policy-preference accuracy with a tie margin

`pairwise_preference_accuracy(predicted, actual, tie_margin=0.0)`: over all
`(i, j)` pairs, excludes pairs whose *real* scores differ by `<= tie_margin`
(the ground truth doesn't clearly prefer one, so neither predictor can be
"wrong" about it), then reports the fraction of the remaining pairs where
`sign(predicted[i]-predicted[j]) == sign(actual[i]-actual[j])`. A predicted
*exact* tie on a non-excluded pair counts as incorrect -- failing to state a
preference reality does have is a miss.

## Within-task / within-policy ranking, and pooled-vs-within reporting

`within_task_policy_ranking` / `within_policy_task_ranking` return, per
task/policy, the other axis's items ordered best-to-worst.
`within_group_correlation` correlates two score arrays *inside* each group
(task or policy) rather than pooled across all of them.
`pooled_vs_within_group_correlation` returns both together -- this is the
literal implementation of the contract's "main reporting includes coverage
and ranking, not Pearson alone."

### Why this matters: the negative check

`tests/test_predictive_metrics.py::test_pooled_pearson_high_despite_reversed_within_task_ranking`
is the FAST_VALIDATION_MATRIX check for this task, and it is the reason
`within_group_correlation` exists at all rather than trusting a single
pooled number. Construction: 2 tasks x 3 policies; within each task,
predicted and actual scores are an exact affine reversal
(`pred + actual == const` for every policy in that task, so the within-task
Pearson correlation is exactly `-1`), but task 1 is uniformly easy
(scores around `0.85`) and task 2 is uniformly hard (scores around `0.15`).
The between-task alignment of means dominates pooled variance and produces
`pooled Pearson r ~= 0.973` -- a number that looks like an excellent
predictor -- while every single task's policy ranking is exactly backwards.
`within_group_correlation` (and the raw `within_task_policy_ranking` lists)
catch this immediately; naive pooled `pearson_r` does not.

## Residual correlation after removing policy/task fixed effects

`two_way_demean(records, value_key, row_key="policy", col_key="task")`
implements a simple two-way ANOVA-style additive demeaning:

```
residual = value - mean(value | row) - mean(value | col) + mean(value)
```

`residual_correlation` demeans `x_key` and `y_key` independently this way
and then correlates the residuals -- isolating policy-task interaction
signal from the (much larger, and much less interesting) fact that some
tasks are uniformly harder and some policies are uniformly better. Handles
repeated `(row, col)` combinations (multiple seeds per policy-task cell) by
computing each row/col mean over every matching record, not one-per-cell.

## Staged progress / trajectory-contact / failure-label agreement

Per the task file, these are "simple set/sequence agreement functions"
consuming already-reduced label lists, not raw sensor data:

- **staged progress** - numeric, so it's `staged_progress_mae` (MAE, 0.25
  per stage: grasp/lift/hover/place).
- **trajectory-contact agreement** - contacts at a timestep are naturally a
  *set* of object/link ids; `set_agreement` is a Jaccard index between two
  label sets, and `mean_set_agreement` averages it over a sequence of
  timesteps (predicted vs. actual contact sets per frame).
- **failure-label agreement** - one categorical label per episode;
  `label_agreement_rate` is elementwise exact-match rate.
- **general sequence agreement** - `sequence_agreement` (and
  `mean_sequence_agreement`) handle ordered, possibly different-length label
  sequences (e.g. a staged-progress transition sequence like
  `[reach, grasp, lift, place]`) via normalized Levenshtein similarity
  (`1 - edit_distance / max(len_a, len_b)`), so it degrades gracefully when
  rollouts run for different numbers of steps.

## Coverage and risk-coverage curves for certification

`risk_coverage_curve(predicted_error_or_score, actual_error, included_bool=None)`
implements the selective-prediction risk-coverage curve (Geifman & El-Yaniv,
*"Selective Classification for Deep Neural Networks,"* NeurIPS 2017): sweep
an acceptance threshold over the predicted error/score (most-confident items
accepted first), and at each threshold report:

- **coverage** - fraction of the *eligible* population accepted at that
  threshold,
- **mean_error** - mean *actual* error among the accepted subset.

`included_bool` marks eligibility (e.g. has ground truth / not excluded for
reasons unrelated to the predicted score) *before* the sweep; it is the one
sanctioned filtering knob in this module, and it is meant to be a decision
recorded ahead of time by the certificate itself, not chosen after seeing
results. `auc` is the Area Under the Risk-Coverage curve (AURC, trapezoidal
integration over coverage) -- the code behind the contract's
`risk_coverage_auc` secondary metric -- lower is better. `coverage_at_error_budget`
answers the certification-flavored question directly: "what fraction of
items can I accept while keeping mean error under budget X?"

Hand-computed reference case (`test_risk_coverage_curve_hand_computed`): 4
items, `predicted_error_or_score = [0.1, 0.4, 0.2, 0.3]`,
`actual_error = [0.0, 1.0, 0.0, 1.0]`. Sorted by predicted score ascending,
`coverage = [0.25, 0.5, 0.75, 1.0]`, `mean_error = [0.0, 0.0, 0.3333, 0.5]`,
`auc = 0.14583`.

## Hierarchical bootstrap (`bootstrap.py`)

Per plan step 2: bootstrap over **scenes**, then over **paired seeds within
each scene**. `hierarchical_bootstrap(records, statistic_fn, scene_key="scene",
seed_key="seed", ...)`:

1. Groups records into `scene -> seed-bucket -> [records]`. A seed-bucket
   holds every record sharing that `(scene, seed)` pair -- typically one
   record per policy being compared for that episode seed, so the pairing
   across policies is preserved *by construction*: resampling only chooses
   which scenes and which whole seed-buckets are drawn, never splits a
   bucket apart.
2. Each bootstrap replicate: resample scene ids with replacement, then for
   each drawn scene resample its seed-buckets with replacement, concatenate
   everything drawn, and evaluate `statistic_fn` on the result.
3. The **point estimate** is `statistic_fn(records)` on the original,
   unresampled data -- never the mean of the bootstrap replicates. The CI is
   the empirical percentile interval of the replicate distribution.

`test_hierarchical_bootstrap_preserves_seed_pairing` verifies the pairing
guarantee directly: a paired-difference statistic that depends on every
`(scene, seed)` having both a `policy=a` and `policy=b` record stays
computable (and correct) on every single bootstrap replicate, because
resampling never breaks up a `(scene, seed)` bucket.

`flat_bootstrap` is the non-hierarchical fallback for data with no
scene/seed structure (e.g. bootstrapping a single already-paired list of
per-item errors).

**Reproducibility**: pass `seed=<int>` for a deterministic replicate stream
(tests, frozen paper numbers); omit it to get fresh entropy from
`np.random.default_rng()` at call time. Neither function hardcodes a default
seed.

## Simulation-based power analysis (`power.py`)

Per plan step 4: episode budgets come from simulating the actual test, not
quoting a closed-form formula. `simulate_two_proportion_power(n_per_arm, p1,
p2, alpha, n_sims, seed, two_sided)` draws `n_sims` independent pairs of
binomial trials (`n_per_arm` episodes each, true rates `p1`/`p2`), runs a
pooled-variance two-proportion z-test (equivalent to a 2x2 chi-square test)
on each pair, and returns the empirical fraction that reject `H0: p1==p2` at
level `alpha` -- that fraction *is* the simulated power.
`required_episodes_two_proportion(base_rate, effect_size, power, alpha, ...)`
searches (exponential bracket + bisection) for the smallest `n_per_arm`
whose simulated power clears the target.

### Anchor validation

`docs/ROBOT.md`'s pi0.5/DROID-sim result gives three worked real numbers
directly usable as ground truth for this module: *"from a 0.28 base
nonzero rate, detecting +0.15 needs ~158 episodes/arm (80% power, alpha
0.05); +0.25 needs 60; +0.40 needs 24."*
`tests/test_predictive_metrics.py::test_power_analysis_matches_robot_md_anchors`
reproduces all three within a 15% relative-error tolerance; observed error
with `seed=12345, n_sims=6000` is well inside that band:

| base rate | effect | anchor (episodes/arm) | this module | rel. error |
|---|---|---|---|---|
| 0.28 | +0.15 | ~158 | 158 | -0.0% |
| 0.28 | +0.25 | 60   | 59  | -1.7% |
| 0.28 | +0.40 | 24   | 23  | -4.2% |

(For reference, these also match the closed-form Fleiss/Levin/Paik
two-proportion sample-size formula --
`n = [z_a/2 * sqrt(2*pbar*qbar) + z_b * sqrt(p1*q1+p2*q2)]^2 / delta^2` --
to within a fraction of an episode; the simulation approach was kept as the
primary/tested API per the task's explicit "simulation-based" framing, since
it directly exercises the actual test statistic rather than an asymptotic
approximation of it, and generalizes to design questions the closed form
doesn't cover for free -- e.g. one-sided tests, or swapping in a different
rejection rule later without re-deriving a formula.)

## What was implemented vs. deferred

**Implemented** (all required items from `plan/10_PREDICTIVE_METRICS_AND_POWER.md`):
success-rate MAE, staged-progress MAE, Brier score, calibration Brier,
binned expected-calibration-error (bonus diagnostic), Pearson/Spearman/
Kendall tau, MMRV with a hand-computed reference test, pairwise
policy-preference accuracy with a tie margin, within-task/within-policy
ranking and correlation helpers, pooled-vs-within reporting, two-way-demean
residual correlation, set/sequence/label agreement primitives, risk-coverage
curves + AURC + coverage-at-budget, hierarchical (scene->seed) bootstrap with
point estimate + CI + seed control, flat bootstrap fallback, and
simulation-based two-proportion power analysis validated against the
`docs/ROBOT.md` anchors.

**Deferred** (out of scope for this file set, owned elsewhere or blocked on
upstream data per `docs/ICRA_RESEARCH_CONTRACT.md`):
- Leave-one-policy/task/scene-out diagnostics (plan step 5) -- this is a
  *reporting-pipeline* concern (which script re-runs the metrics above with
  one group excluded and tabulates the delta), not a new statistic; every
  primitive it would call (`pearson_r`, `mmrv`, `risk_coverage_curve`, ...)
  already exists here. Building the actual leave-one-out driver belongs
  with whatever script consumes `configs/oracle/tasks.yaml` and produces the
  paper tables (Task 20), since it needs the real oracle-track manifest to
  iterate over, which does not exist yet (`oracle_causal` status:
  `in_progress`, 1/10 target scenes piloted).
- Populating the actual oracle-track numbers (go/no-go gate evaluation,
  `min_scenes: 6` / `min_task_families: 2` / `min_nondegenerate_pairs: 12`
  against real data) -- blocked on Task 10's own dependency chain (oracle
  scene/task build-out), not a metrics-code gap. All the gating machinery
  (`require_min_pairs`, `InsufficientPairsError`) is ready to be called by
  that later step.
