"""Tests for agents/eval/predictive_metrics.py, agents/eval/bootstrap.py, and
agents/eval/power.py (Task 10, plan/10_PREDICTIVE_METRICS_AND_POWER.md).

Run: .venv/bin/python -m pytest -q tests/test_predictive_metrics.py
"""
import math

import numpy as np
import pytest

from agents.eval import predictive_metrics as pm
from agents.eval.bootstrap import flat_bootstrap, hierarchical_bootstrap
from agents.eval.power import required_episodes_two_proportion, simulate_two_proportion_power


# ==========================================================================
# Success-rate MAE / staged-progress MAE / Brier / calibration.
# ==========================================================================

def test_success_rate_mae_toy():
    pred = [0.2, 0.5, 0.8]
    actual = [0.3, 0.5, 0.6]
    # |0.1| + |0.0| + |0.2| = 0.3 -> mean 0.1
    assert pm.success_rate_mae(pred, actual) == pytest.approx(0.1)


def test_staged_progress_mae_matches_success_rate_mae():
    pred = [0.0, 0.25, 0.5, 0.75, 1.0]
    actual = [0.25, 0.25, 0.25, 0.5, 1.0]
    assert pm.staged_progress_mae(pred, actual) == pytest.approx(
        pm.success_rate_mae(pred, actual)
    )


def test_staged_progress_mae_rejects_out_of_range():
    with pytest.raises(ValueError):
        pm.staged_progress_mae([0.0, 1.5], [0.0, 1.0])


def test_brier_score_toy():
    # perfect probabilistic prediction -> 0; maximally wrong -> 1
    assert pm.brier_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert pm.brier_score([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    # (0.7-1)^2 + (0.3-0)^2 = 0.09 + 0.09 = 0.18 -> mean 0.09
    assert pm.brier_score([0.7, 0.3], [1.0, 0.0]) == pytest.approx(0.09)


def test_calibration_brier_is_brier_score():
    pred = [0.9, 0.6, 0.2, 0.1]
    actual = [1.0, 1.0, 0.0, 0.0]
    assert pm.calibration_brier(pred, actual) == pm.brier_score(pred, actual)


def test_expected_calibration_error_toy():
    # a perfectly calibrated forecaster within each bin gets ECE 0
    pred = [0.1, 0.1, 0.9, 0.9]
    actual = [0.0, 0.2, 1.0, 0.8]  # bin means 0.1 vs 0.1, 0.9 vs 0.9
    assert pm.expected_calibration_error(pred, actual, n_bins=10) == pytest.approx(0.0, abs=1e-9)


# ==========================================================================
# Pearson / Spearman / Kendall analytic toy examples.
# ==========================================================================

def test_pearson_perfect_positive_and_negative():
    x = [1, 2, 3, 4, 5]
    assert pm.pearson_r(x, x) == pytest.approx(1.0)
    assert pm.pearson_r(x, list(reversed(x))) == pytest.approx(-1.0)


def test_spearman_and_kendall_reversed_ranking():
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    assert pm.spearman(x, y) == pytest.approx(-1.0)
    assert pm.kendall_tau(x, y) == pytest.approx(-1.0)


def test_spearman_task_difficulty_is_general_spearman():
    x = [3, 1, 2, 5, 4]
    y = [2, 1, 3, 4, 5]
    assert pm.spearman_task_difficulty(x, y) == pm.spearman(x, y)


def test_correlation_raises_on_degenerate_input():
    x = [1, 1, 1, 1]
    y = [1, 2, 3, 4]
    with pytest.raises(pm.DegenerateInputError):
        pm.pearson_r(x, y)
    with pytest.raises(pm.DegenerateInputError):
        pm.spearman(x, y)


def test_correlation_raises_on_too_few_points():
    with pytest.raises(pm.InsufficientPairsError):
        pm.pearson_r([1.0], [2.0])


def test_require_min_pairs_gate():
    pm.require_min_pairs(12, min_pairs=12)  # exactly at threshold: ok
    with pytest.raises(pm.InsufficientPairsError):
        pm.require_min_pairs(11, min_pairs=12)


# ==========================================================================
# MMRV: hand-computed reference cases.
#
# 3 policies, real success y = [A=0.9, B=0.5, C=0.1] (A best, C worst).
#
# Case 1 -- sim score x completely reverses the real ranking:
#   x = [A=0.1, B=0.5, C=0.9]  (sim thinks C is best, A is worst)
#   For i=A (x=0.1): no j has x_j < 0.1                       -> term 0
#   For i=B (x=0.5): j=A has x_A=0.1<0.5, gap = y_A-y_B = 0.4  -> term 0.4
#   For i=C (x=0.9): j in {A,B}; max(y_A-y_C, y_B-y_C)
#                    = max(0.9-0.1, 0.5-0.1) = max(0.8, 0.4)   -> term 0.8
#   MMRV = (0 + 0.4 + 0.8) / 3 = 0.4
#
# Case 2 -- sim score matches the real ranking exactly (x == y):
#   every violation term is 0 by construction -> MMRV = 0.0
# ==========================================================================

def test_mmrv_hand_computed_reversed_ranking():
    y = [0.9, 0.5, 0.1]  # A, B, C real success
    x = [0.1, 0.5, 0.9]  # A, B, C sim score, fully reversed
    assert pm.mmrv(x, y) == pytest.approx(0.4)


def test_mmrv_hand_computed_perfect_ranking_is_zero():
    y = [0.9, 0.5, 0.1]
    x = [0.9, 0.5, 0.1]  # sim score agrees with real ranking exactly
    assert pm.mmrv(x, y) == pytest.approx(0.0)


def test_mmrv_partial_violation_hand_computed():
    # A, B, C, D real success:
    y = [0.9, 0.7, 0.5, 0.1]
    # sim score swaps only B and C relative to y's order, keeps A best and D worst.
    x = [0.9, 0.5, 0.7, 0.1]
    # i=A (x=0.9): no j with x_j<0.9 among... wait B=0.5,C=0.7,D=0.1 all <0.9
    #   gaps: y_B-y_A=-0.2, y_C-y_A=-0.4, y_D-y_A=-0.8 -> all clipped to 0 -> term 0
    # i=B (x=0.5): j with x_j<0.5: D(x=0.1). gap y_D-y_B = 0.1-0.7=-0.6 -> 0 -> term 0
    # i=C (x=0.7): j with x_j<0.7: B(x=0.5), D(x=0.1).
    #   gaps: y_B-y_C=0.7-0.5=0.2, y_D-y_C=0.1-0.5=-0.4 -> max(0.2,0)=0.2 -> term 0.2
    # i=D (x=0.1): no j with x_j<0.1 -> term 0
    # MMRV = (0+0+0.2+0)/4 = 0.05
    assert pm.mmrv(x, y) == pytest.approx(0.05)


def test_mmrv_needs_at_least_two_items():
    with pytest.raises(pm.InsufficientPairsError):
        pm.mmrv([0.5], [0.5])


# ==========================================================================
# Pairwise policy-preference accuracy with tie margin.
# ==========================================================================

def test_pairwise_preference_accuracy_toy():
    # 3 items, real scores strictly ordered A>B>C; predicted score gets the
    # A-vs-B pair right but reverses B-vs-C and A-vs-C.
    pred = [0.9, 0.1, 0.5]     # A, B, C
    actual = [0.9, 0.5, 0.1]   # A, B, C
    # pairs: (A,B) pred 0.9>0.1 real 0.9>0.5 -> correct
    #        (A,C) pred 0.9>0.5 real 0.9>0.1 -> correct
    #        (B,C) pred 0.1<0.5 real 0.5>0.1 -> incorrect
    result = pm.pairwise_preference_accuracy(pred, actual)
    assert result.n_total_pairs == 3
    assert result.n_ties_excluded == 0
    assert result.accuracy == pytest.approx(2 / 3)


def test_pairwise_preference_accuracy_tie_margin_excludes_near_ties():
    pred = [0.9, 0.1]
    actual = [0.50, 0.49]  # real difference is 0.01, within tie_margin
    result = pm.pairwise_preference_accuracy(pred, actual, tie_margin=0.05)
    assert result.n_ties_excluded == 1
    assert result.n_considered == 0
    assert math.isnan(result.accuracy)


# ==========================================================================
# Within-task / within-policy ranking helpers.
# ==========================================================================

def test_within_task_policy_ranking_toy():
    records = [
        {"task": "t1", "policy": "A", "actual": 0.9},
        {"task": "t1", "policy": "B", "actual": 0.5},
        {"task": "t1", "policy": "C", "actual": 0.1},
        {"task": "t2", "policy": "A", "actual": 0.2},
        {"task": "t2", "policy": "B", "actual": 0.8},
        {"task": "t2", "policy": "C", "actual": 0.5},
    ]
    ranking = pm.within_task_policy_ranking(records)
    assert ranking["t1"] == ["A", "B", "C"]
    assert ranking["t2"] == ["B", "C", "A"]


def test_within_policy_task_ranking_toy():
    records = [
        {"task": "t1", "policy": "A", "actual": 0.9},
        {"task": "t2", "policy": "A", "actual": 0.2},
        {"task": "t3", "policy": "A", "actual": 0.5},
    ]
    ranking = pm.within_policy_task_ranking(records)
    assert ranking["A"] == ["t1", "t3", "t2"]


# ==========================================================================
# CRITICAL: pooled Pearson can be high despite reversed within-task ranking.
#
# 2 tasks x 3 policies. Within each task, pred and actual are an exact affine
# reversal (pred + actual == const for every policy in that task), so the
# within-task correlation is exactly -1. But task 1 is uniformly "easy"
# (actual ~0.85, pred ~0.85) and task 2 is uniformly "hard" (actual ~0.15,
# pred ~0.15): the *between-task* alignment of means dominates the pooled
# variance and produces a strong positive pooled Pearson r, hiding the
# within-task reversal completely.
# ==========================================================================

_CONFOUNDED_RECORDS = [
    {"task": "t1", "policy": "A", "pred": 0.80, "actual": 0.90},
    {"task": "t1", "policy": "B", "pred": 0.85, "actual": 0.85},
    {"task": "t1", "policy": "C", "pred": 0.90, "actual": 0.80},
    {"task": "t2", "policy": "A", "pred": 0.10, "actual": 0.20},
    {"task": "t2", "policy": "B", "pred": 0.15, "actual": 0.15},
    {"task": "t2", "policy": "C", "pred": 0.20, "actual": 0.10},
]


def test_pooled_pearson_high_despite_reversed_within_task_ranking():
    pooled = pm.pearson_r([r["pred"] for r in _CONFOUNDED_RECORDS],
                           [r["actual"] for r in _CONFOUNDED_RECORDS])
    # Naive pooled correlation looks excellent...
    assert pooled == pytest.approx(0.9732, abs=1e-3)
    assert pooled > 0.9

    # ...but within EVERY task the ranking is exactly, perfectly reversed.
    within = pm.within_group_correlation(_CONFOUNDED_RECORDS, group_key="task",
                                          x_key="pred", y_key="actual", method="pearson")
    assert set(within) == {"t1", "t2"}
    for task, corr in within.items():
        assert corr == pytest.approx(-1.0, abs=1e-9), f"task {task} within-corr should be -1"

    # The ranking helper shows the reversal explicitly: predicted best-to-worst
    # order is the exact reverse of the actual best-to-worst order, every task.
    actual_ranks = pm.within_task_policy_ranking(_CONFOUNDED_RECORDS, score_key="actual")
    pred_ranks = pm.within_task_policy_ranking(_CONFOUNDED_RECORDS, score_key="pred")
    for task in actual_ranks:
        assert pred_ranks[task] == list(reversed(actual_ranks[task]))

    # pooled_vs_within_group_correlation reports both side by side, which is
    # exactly the contract's "coverage and ranking, not Pearson alone" rule.
    pooled2, within2 = pm.pooled_vs_within_group_correlation(
        _CONFOUNDED_RECORDS, x_key="pred", y_key="actual", group_key="task", method="pearson"
    )
    assert pooled2 == pytest.approx(pooled)
    assert within2 == within


# ==========================================================================
# Residual correlation after removing policy/task fixed effects.
# ==========================================================================

def test_two_way_demean_removes_additive_effects():
    # value = row_effect + col_effect exactly, no interaction/noise
    # -> every residual should be ~0.
    records = []
    row_effect = {"A": 0.0, "B": 0.3}
    col_effect = {"t1": 0.0, "t2": 0.5, "t3": 1.0}
    for rk, re in row_effect.items():
        for ck, ce in col_effect.items():
            records.append({"policy": rk, "task": ck, "value": re + ce})
    resid = pm.two_way_demean(records, "value")
    assert np.allclose(resid, 0.0, atol=1e-9)


def test_residual_correlation_isolates_interaction_signal():
    # x and y both have a large, perfectly shared additive task effect
    # (which would make the pooled correlation huge and uninformative) plus
    # a small interaction term that is deliberately ANTI-correlated between
    # x and y. Residual correlation after demeaning should surface that
    # anti-correlation; pooled correlation should be strongly positive
    # (dominated by the shared task effect), the opposite sign.
    task_effect = {"t1": 0.0, "t2": 40.0}
    policies = ["A", "B"]
    # interaction: for x, A gets +1 in t1/-1 in t2 style pattern; for y, reversed.
    x_interaction = {("A", "t1"): 1.0, ("A", "t2"): -1.0, ("B", "t1"): -1.0, ("B", "t2"): 1.0}
    y_interaction = {("A", "t1"): -1.0, ("A", "t2"): 1.0, ("B", "t1"): 1.0, ("B", "t2"): -1.0}
    records = []
    for p in policies:
        for t in task_effect:
            records.append({
                "policy": p, "task": t,
                "x": task_effect[t] + x_interaction[(p, t)],
                "y": task_effect[t] + y_interaction[(p, t)],
            })
    pooled = pm.pearson_r([r["x"] for r in records], [r["y"] for r in records])
    resid_corr = pm.residual_correlation(records, "x", "y")
    assert pooled > 0.9         # dominated by the shared task effect
    assert resid_corr < -0.9    # true interaction is anti-correlated


# ==========================================================================
# Set / sequence / label agreement.
# ==========================================================================

def test_set_agreement_toy():
    assert pm.set_agreement({"a", "b"}, {"a", "b"}) == pytest.approx(1.0)
    assert pm.set_agreement({"a", "b"}, {"a", "c"}) == pytest.approx(1 / 3)
    assert pm.set_agreement(set(), set()) == pytest.approx(1.0)
    assert pm.set_agreement({"a"}, set()) == pytest.approx(0.0)


def test_mean_set_agreement_trajectory_contact_toy():
    pred_seq = [{"table"}, {"table", "cup"}, {"cup"}]
    actual_seq = [{"table"}, {"cup"}, {"cup"}]
    # per-step jaccard: 1.0, 1/2, 1.0 -> mean 0.8333
    assert pm.mean_set_agreement(pred_seq, actual_seq) == pytest.approx(2.5 / 3)


def test_label_agreement_rate_failure_labels_toy():
    pred = ["grasp_fail", "drop", "success", "drop"]
    actual = ["grasp_fail", "success", "success", "drop"]
    assert pm.label_agreement_rate(pred, actual) == pytest.approx(0.75)


def test_sequence_agreement_toy():
    assert pm.sequence_agreement(["reach", "grasp", "lift"], ["reach", "grasp", "lift"]) == pytest.approx(1.0)
    # one substitution out of 3 -> 1 - 1/3
    assert pm.sequence_agreement(["reach", "grasp", "lift"], ["reach", "grasp", "drop"]) == pytest.approx(2 / 3)
    # different lengths: edit distance 1 insertion, normalized by max(len)=4
    assert pm.sequence_agreement(["reach", "grasp", "lift"], ["reach", "grasp", "hover", "lift"]) == pytest.approx(3 / 4)


# ==========================================================================
# Risk-coverage curve for certification.
#
# 4 items, predicted_error_or_score = [0.1, 0.4, 0.2, 0.3], actual_error =
# [0.0, 1.0, 0.0, 1.0]. Sorted by predicted score ascending: idx 0(0.1),
# 2(0.2), 3(0.3), 1(0.4) -> sorted actual_error = [0.0, 0.0, 1.0, 1.0].
# coverage = [1/4, 2/4, 3/4, 4/4], mean_error = cumsum/rank =
#   [0/1, 0/2, 1/3, 2/4] = [0.0, 0.0, 0.33333, 0.5].
# AUC via trapz over (coverage, mean_error):
#   seg1 (0.25->0.5): width .25, heights 0,0 -> area 0
#   seg2 (0.5->0.75): width .25, heights 0,0.33333 -> area 0.0416667
#   seg3 (0.75->1.0): width .25, heights 0.33333,0.5 -> area 0.1041667
#   total = 0.1458333
# ==========================================================================

def test_risk_coverage_curve_hand_computed():
    pred = [0.1, 0.4, 0.2, 0.3]
    actual_error = [0.0, 1.0, 0.0, 1.0]
    curve = pm.risk_coverage_curve(pred, actual_error)
    assert curve.n_included == 4
    assert curve.n_total == 4
    np.testing.assert_allclose(curve.coverage, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(curve.mean_error, [0.0, 0.0, 1 / 3, 0.5])
    assert curve.auc == pytest.approx(0.1458333, abs=1e-6)


def test_risk_coverage_curve_respects_included_bool():
    pred = [0.1, 0.4, 0.2, 0.3]
    actual_error = [0.0, 1.0, 0.0, 1.0]
    included = [True, False, True, True]
    curve = pm.risk_coverage_curve(pred, actual_error, included_bool=included)
    assert curve.n_included == 3
    assert curve.n_total == 4
    # only items 0, 2, 3 remain: pred [0.1, 0.2, 0.3], actual [0.0, 0.0, 1.0]
    # sorted ascending by pred is already that order.
    np.testing.assert_allclose(curve.coverage, [1 / 3, 2 / 3, 1.0])
    np.testing.assert_allclose(curve.mean_error, [0.0, 0.0, 1 / 3])


def test_coverage_at_error_budget():
    pred = [0.1, 0.4, 0.2, 0.3]
    actual_error = [0.0, 1.0, 0.0, 1.0]
    curve = pm.risk_coverage_curve(pred, actual_error)
    # a budget of 0.4 allows the first 3 items (mean_error 0.3333) but not all 4 (0.5)
    assert pm.coverage_at_error_budget(curve, max_error=0.4) == pytest.approx(0.75)
    assert pm.coverage_at_error_budget(curve, max_error=0.0) == pytest.approx(0.5)
    assert pm.coverage_at_error_budget(curve, max_error=-1.0) == pytest.approx(0.0)


def test_risk_coverage_curve_needs_included_items():
    with pytest.raises(pm.InsufficientPairsError):
        pm.risk_coverage_curve([0.1, 0.2], [0.0, 1.0], included_bool=[False, False])


# ==========================================================================
# Bootstrap: point estimate, CI shape, hierarchical pairing, reproducibility.
# ==========================================================================

def _mean_of(key):
    def _fn(records):
        return float(np.mean([r[key] for r in records]))
    return _fn


def test_flat_bootstrap_point_estimate_and_ci_shape():
    values = [{"v": v} for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
    result = flat_bootstrap(values, _mean_of("v"), n_boot=500, seed=0)
    assert result.point_estimate == pytest.approx(3.0)
    assert result.ci_lo <= result.point_estimate <= result.ci_hi
    assert result.n_boot == 500
    assert result.samples.shape == (500,)


def test_flat_bootstrap_reproducible_with_fixed_seed():
    values = [{"v": v} for v in [1.0, 5.0, 2.0, 9.0, 3.0, 7.0]]
    r1 = flat_bootstrap(values, _mean_of("v"), n_boot=300, seed=42)
    r2 = flat_bootstrap(values, _mean_of("v"), n_boot=300, seed=42)
    np.testing.assert_array_equal(r1.samples, r2.samples)
    assert r1.ci_lo == r2.ci_lo and r1.ci_hi == r2.ci_hi


def test_hierarchical_bootstrap_point_estimate_matches_full_data():
    records = []
    for scene in ["s1", "s2", "s3"]:
        for seed_id in range(4):
            records.append({"scene": scene, "seed": seed_id, "v": hash((scene, seed_id)) % 100})
    result = hierarchical_bootstrap(records, _mean_of("v"), n_boot=200, seed=7)
    assert result.point_estimate == pytest.approx(np.mean([r["v"] for r in records]))
    assert result.ci_lo <= result.ci_hi
    assert result.n_boot <= 200


def test_hierarchical_bootstrap_preserves_seed_pairing():
    # Two "policies" recorded per (scene, seed); the statistic is the mean
    # paired difference. Because resampling always draws whole (scene, seed)
    # buckets together, every bootstrap replicate's records still contain
    # exactly matched policy_a/policy_b pairs for each drawn seed -- so the
    # paired-difference statistic must remain computable on every replicate.
    records = []
    for scene in ["s1", "s2"]:
        for seed_id in range(5):
            records.append({"scene": scene, "seed": seed_id, "policy": "a", "v": 1.0})
            records.append({"scene": scene, "seed": seed_id, "policy": "b", "v": 0.6})

    def paired_diff(recs):
        a = {(r["scene"], r["seed"]): r["v"] for r in recs if r["policy"] == "a"}
        b = {(r["scene"], r["seed"]): r["v"] for r in recs if r["policy"] == "b"}
        common = set(a) & set(b)
        assert common, "resampling broke seed pairing: no matched (scene, seed) pairs left"
        return float(np.mean([a[k] - b[k] for k in common]))

    result = hierarchical_bootstrap(records, paired_diff, n_boot=200, seed=3)
    assert result.point_estimate == pytest.approx(0.4)
    # every bootstrap replicate should also be exactly 0.4, since the pairing
    # is preserved and every (a, b) pair has the same fixed values.
    assert np.allclose(result.samples, 0.4)


def test_hierarchical_bootstrap_reproducible_with_fixed_seed():
    records = [{"scene": s, "seed": k, "v": (s_i * 10 + k)}
               for s_i, s in enumerate(["s1", "s2", "s3"]) for k in range(3)]
    r1 = hierarchical_bootstrap(records, _mean_of("v"), n_boot=150, seed=99)
    r2 = hierarchical_bootstrap(records, _mean_of("v"), n_boot=150, seed=99)
    np.testing.assert_array_equal(r1.samples, r2.samples)


# ==========================================================================
# Power analysis: anchor test against docs/ROBOT.md's worked example.
#
# "from a 0.28 base nonzero rate, detecting +0.15 needs ~158 episodes/arm
# (80% power, alpha 0.05); +0.25 needs 60; +0.40 needs 24."
# ==========================================================================

@pytest.mark.parametrize("effect_size,anchor_n", [(0.15, 158), (0.25, 60), (0.40, 24)])
def test_power_analysis_matches_robot_md_anchors(effect_size, anchor_n):
    result = required_episodes_two_proportion(
        base_rate=0.28, effect_size=effect_size, power=0.8, alpha=0.05,
        seed=12345, n_sims=6000,
    )
    rel_err = abs(result.n_per_arm - anchor_n) / anchor_n
    assert rel_err <= 0.15, (
        f"n_per_arm={result.n_per_arm} vs anchor={anchor_n}, rel_err={rel_err:.1%}"
    )
    assert result.achieved_power >= 0.8 - 0.03


def test_power_increases_with_n():
    p_small = simulate_two_proportion_power(20, 0.28, 0.43, seed=1, n_sims=4000)
    p_large = simulate_two_proportion_power(200, 0.28, 0.43, seed=1, n_sims=4000)
    assert p_large > p_small


def test_power_analysis_seed_reproducible():
    r1 = required_episodes_two_proportion(base_rate=0.28, effect_size=0.25, seed=5, n_sims=2000)
    r2 = required_episodes_two_proportion(base_rate=0.28, effect_size=0.25, seed=5, n_sims=2000)
    assert r1.n_per_arm == r2.n_per_arm
    assert r1.achieved_power == r2.achieved_power


def test_power_analysis_rejects_out_of_range_rates():
    with pytest.raises(ValueError):
        required_episodes_two_proportion(base_rate=0.9, effect_size=0.5)  # p2 > 1
