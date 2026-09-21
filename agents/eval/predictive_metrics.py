"""Predictive-validity metrics for the ICRA oracle_causal protocol (Task 10,
plan/10_PREDICTIVE_METRICS_AND_POWER.md; docs/ICRA_RESEARCH_CONTRACT.md sec 4;
configs/experiments/icra_contract_v1.yaml protocols.oracle_causal).

Every function here is a pure, generic statistic over already-computed scalar
scores (success rates, staged-progress fractions, certificate error
predictions, label sequences, ...). Nothing in this module touches raw
sensor/trajectory data -- that aggregation happens upstream in the eval
pipeline; this module only consumes the resulting numbers.

Naming convention: where icra_contract_v1.yaml names a metric explicitly
(`staged_progress_mae`, `spearman_task_difficulty`, `mmrv`,
`calibration_brier`, `risk_coverage_auc`), a function of that exact name
exists here, even when it is a thin wrapper over a more general primitive
(e.g. `spearman_task_difficulty` is exactly `spearman`, per the task's
instruction to implement it as a general Spearman correlation with no
task-difficulty-specific logic).

Non-degenerate pairs and the stop condition
--------------------------------------------
`icra_contract_v1.yaml`'s `oracle_causal.min_nondegenerate_pairs` is 12. This
module never silently drops pairs to raise a correlation: `require_min_pairs`
raises `InsufficientPairsError` (loud, catchable, not swallowed) instead of
returning a degraded-but-plausible number, and no function anywhere in this
file accepts a "drop worst-N" / "exclude outliers" parameter. The only
sanctioned filtering is the pre-registered `included_bool` mask passed into
`risk_coverage_curve` (a certificate's own abstention decision, recorded
before results are read) -- see docs/METRICS.md for the full policy.

References
----------
- MMRV: Li et al., "Evaluating Real-World Robot Manipulation Policies in
  Simulation" (SIMPLER benchmark), arXiv:2405.05941.
- Risk-coverage curves / AURC: Geifman & El-Yaniv, "Selective Classification
  for Deep Neural Networks", NeurIPS 2017.
- Brier score: Brier, "Verification of Forecasts Expressed in Terms of
  Probability", Monthly Weather Review, 1950.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
from scipy import stats

Record = Mapping[str, Any]


# --------------------------------------------------------------------------
# Errors: the only two ways a metric function is allowed to refuse an answer.
# --------------------------------------------------------------------------

class InsufficientPairsError(ValueError):
    """Too few (non-degenerate) pairs to report a pooled correlation.

    Per icra_contract_v1.yaml's stop condition, the sanctioned response is to
    report descriptive per-pair results and narrow the claim -- not to widen
    the pool by resampling favorable scenes, and not to lower this threshold
    post-hoc. Callers that want per-pair diagnostics below the threshold
    should compute them directly (e.g. via `within_group_correlation` on a
    single group) rather than calling `require_min_pairs`.
    """


class DegenerateInputError(ValueError):
    """A correlation is mathematically undefined: one input array is constant
    (zero variance), so there is no ranking/linear-relationship information
    to measure. This is distinct from "too few pairs" -- it can happen with
    arbitrarily many pairs if, e.g., every policy scored exactly 0.0.
    """


_DEGENERATE_EPS = 1e-12


def require_min_pairs(n_pairs: int, min_pairs: int = 12) -> None:
    """Gate used before reporting any pooled correlation table entry.

    `min_pairs` defaults to the contract's `min_nondegenerate_pairs: 12`
    (2 policies x >=6 task instances, oracle track). Raises
    `InsufficientPairsError` rather than returning a boolean, so a caller
    cannot accidentally continue past the check with an `if not ok: pass`.
    """
    if n_pairs < min_pairs:
        raise InsufficientPairsError(
            f"only {n_pairs} non-degenerate policy-task pairs available, need "
            f">= {min_pairs} (icra_contract_v1.yaml oracle_causal."
            "min_nondegenerate_pairs). Report descriptive per-pair results "
            "and narrow the predictive claim instead of raising this bound "
            "or dropping pairs to pass it."
        )


def is_degenerate(values: Sequence[float], eps: float = _DEGENERATE_EPS) -> bool:
    """True iff `values` has (numerically) zero variance."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return True
    return bool(np.std(arr) <= eps)


def _as_arrays(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape:
        raise ValueError(f"shape mismatch: x has {xa.shape}, y has {ya.shape}")
    return xa, ya


def _check_correlatable(x: np.ndarray, y: np.ndarray) -> None:
    if x.size < 2:
        raise InsufficientPairsError(f"need >= 2 points to define a correlation, got {x.size}")
    if is_degenerate(x) or is_degenerate(y):
        raise DegenerateInputError(
            "zero-variance input: correlation is undefined for a constant array "
            f"(std(x)={np.std(x):.3g}, std(y)={np.std(y):.3g})"
        )


# --------------------------------------------------------------------------
# Success-rate MAE, staged-progress MAE, Brier/calibration.
# --------------------------------------------------------------------------

def success_rate_mae(predicted_rate: Sequence[float], actual_rate: Sequence[float]) -> float:
    """Mean absolute error between predicted and actual per-item success rate."""
    p, a = _as_arrays(predicted_rate, actual_rate)
    if p.size == 0:
        raise InsufficientPairsError("no items to compute MAE over")
    return float(np.mean(np.abs(p - a)))


def staged_progress_mae(predicted_stage_progress: Sequence[float],
                         oracle_stage_progress: Sequence[float]) -> float:
    """Contract primary metric `staged_progress_mae` (oracle_causal).

    MAE between reconstructed and oracle rollout staged-progress scores,
    where progress is staged at 0.25 per stage (grasp/lift/hover/place), so
    values are expected in [0, 1]. Numerically identical to
    `success_rate_mae`; kept as a separate, contract-traceable name and with
    a range sanity check since staged-progress values are a specific rubric,
    not an arbitrary rate.
    """
    p = np.asarray(predicted_stage_progress, dtype=float)
    a = np.asarray(oracle_stage_progress, dtype=float)
    if p.size and (p.min() < -1e-6 or p.max() > 1 + 1e-6 or a.min() < -1e-6 or a.max() > 1 + 1e-6):
        raise ValueError(
            "staged-progress values are expected in [0, 1] (0.25 per stage); "
            f"got range x=[{p.min():.3f},{p.max():.3f}] y=[{a.min():.3f},{a.max():.3f}]"
        )
    return success_rate_mae(p, a)


def brier_score(predicted_prob: Sequence[float], outcome: Sequence[float]) -> float:
    """Classic Brier score: mean squared error between a predicted
    probability (or predicted success rate) and the realized outcome
    (typically 0/1, but a realized rate in [0, 1] is also valid)."""
    p, o = _as_arrays(predicted_prob, outcome)
    if p.size == 0:
        raise InsufficientPairsError("no items to compute Brier score over")
    return float(np.mean((p - o) ** 2))


def calibration_brier(predicted_error_or_success_prob: Sequence[float],
                       actual_outcome: Sequence[float]) -> float:
    """Contract secondary metric `calibration_brier` (oracle_causal):
    Brier score of the task certificate's predicted error/success
    probability against the realized outcome. Identical formula to
    `brier_score`; named separately for direct traceability to
    icra_contract_v1.yaml's metric list.
    """
    return brier_score(predicted_error_or_success_prob, actual_outcome)


def expected_calibration_error(predicted_prob: Sequence[float], outcome: Sequence[float],
                                n_bins: int = 10) -> float:
    """Binned Expected Calibration Error (Guo et al. 2017), as a secondary
    calibration diagnostic alongside the Brier score. Bins on predicted
    probability into `n_bins` equal-width bins in [0, 1]; empty bins
    contribute zero."""
    p, o = _as_arrays(predicted_prob, outcome)
    if p.size == 0:
        raise InsufficientPairsError("no items to compute ECE over")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = p.size
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        bin_conf = float(np.mean(p[mask]))
        bin_acc = float(np.mean(o[mask]))
        ece += (np.sum(mask) / n) * abs(bin_conf - bin_acc)
    return float(ece)


# --------------------------------------------------------------------------
# Pearson / Spearman / Kendall.
# --------------------------------------------------------------------------

def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = _as_arrays(x, y)
    _check_correlatable(xa, ya)
    return float(stats.pearsonr(xa, ya)[0])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """General Spearman rank correlation."""
    xa, ya = _as_arrays(x, y)
    _check_correlatable(xa, ya)
    return float(stats.spearmanr(xa, ya)[0])


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    xa, ya = _as_arrays(x, y)
    _check_correlatable(xa, ya)
    return float(stats.kendalltau(xa, ya)[0])


def spearman_task_difficulty(predicted_difficulty: Sequence[float],
                              actual_difficulty: Sequence[float]) -> float:
    """Contract primary metric `spearman_task_difficulty` (oracle_causal):
    rank agreement of reconstructed-vs-oracle task-difficulty ordering.

    This is exactly `spearman` -- per Task 10's instruction, no
    task-difficulty-specific logic is implemented, only a general Spearman
    correlation applied to whichever two difficulty-ranking arrays the
    caller supplies.
    """
    return spearman(predicted_difficulty, actual_difficulty)


_CORR_FNS: dict[str, Callable[[Sequence[float], Sequence[float]], float]] = {
    "pearson": pearson_r,
    "spearman": spearman,
    "kendall": kendall_tau,
}


# --------------------------------------------------------------------------
# MMRV (Mean Maximum Rank Violation).
# --------------------------------------------------------------------------

def mmrv(sim_score: Sequence[float], real_score: Sequence[float], eps: float = 0.0) -> float:
    """Mean Maximum Rank Violation (Li et al., "Evaluating Real-World Robot
    Manipulation Policies in Simulation" / SIMPLER benchmark, arXiv:2405.05941).

    Given N items (policy-task pairs) with a simulated/predicted score
    `sim_score[i]` and a real/oracle score `real_score[i]`:

        MMRV = (1/N) * sum_i  max_{j : sim_score[j] < sim_score[i] - eps}
                                   max(real_score[j] - real_score[i], 0)

    with the inner max over an empty set defined as 0. In words: for each
    item i, look at every other item j that the predictor ranked *below* i;
    if any such j actually scored higher in reality, that is a rank
    violation, and its size is the real-world gap it hides. MMRV is the mean
    over i of the worst violation hidden behind i's predicted rank -- 0 iff
    the predicted ranking never hides a real-world reversal, and grows with
    the size of the worst hidden reversal.

    `eps` is a float-tolerance slack for the strict "<" comparison; the
    formula itself has no tie handling, since a genuine tie in `sim_score`
    (sim_score[j] == sim_score[i]) is not a case where j was ranked strictly
    below i.
    """
    x, y = _as_arrays(sim_score, real_score)
    n = x.size
    if n < 2:
        raise InsufficientPairsError(f"MMRV needs >= 2 items, got {n}")
    violations = np.zeros(n, dtype=float)
    for i in range(n):
        below = x < (x[i] - eps)
        if not np.any(below):
            continue
        gaps = y[below] - y[i]
        violations[i] = max(0.0, float(np.max(gaps)))
    return float(np.mean(violations))


# --------------------------------------------------------------------------
# Pairwise policy-preference accuracy with a tie margin.
# --------------------------------------------------------------------------

@dataclass
class PairwisePreferenceResult:
    accuracy: float
    n_considered: int
    n_ties_excluded: int
    n_total_pairs: int


def pairwise_preference_accuracy(predicted_score: Sequence[float], actual_score: Sequence[float],
                                  tie_margin: float = 0.0) -> PairwisePreferenceResult:
    """Fraction of item pairs (i, j) where the predicted preference direction
    matches the actual one, i.e. sign(predicted[i]-predicted[j]) ==
    sign(actual[i]-actual[j]).

    Pairs whose *actual* scores differ by <= `tie_margin` are excluded from
    both the numerator and denominator (a real-world near-tie is not a
    preference either predictor can be "wrong" about). A predicted exact tie
    on a non-excluded pair counts as incorrect: failing to state a
    preference the real world does have is a miss, not a free pass.
    """
    x, y = _as_arrays(predicted_score, actual_score)
    n = x.size
    if n < 2:
        raise InsufficientPairsError(f"need >= 2 items to form a pair, got {n}")
    total = 0
    ties = 0
    correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            real_diff = y[i] - y[j]
            if abs(real_diff) <= tie_margin:
                ties += 1
                continue
            pred_diff = x[i] - x[j]
            if np.sign(pred_diff) == np.sign(real_diff):
                correct += 1
    considered = total - ties
    accuracy = correct / considered if considered > 0 else float("nan")
    return PairwisePreferenceResult(accuracy=accuracy, n_considered=considered,
                                     n_ties_excluded=ties, n_total_pairs=total)


# --------------------------------------------------------------------------
# Within-task policy ranking / within-policy task ranking / pooled-vs-within.
# --------------------------------------------------------------------------

def rank_within_group(records: Sequence[Record], group_key: str, item_key: str,
                       score_key: str, descending: bool = True) -> dict[Hashable, list[Any]]:
    """Group `records` by `group_key` and, within each group, sort the
    distinct `item_key` values by `score_key` (default: highest first).

    E.g. `rank_within_group(records, "task", "policy", "actual")` gives, for
    each task, the policies ordered best-to-worst by actual score.
    """
    groups: dict[Hashable, list[tuple[Any, float]]] = {}
    for r in records:
        groups.setdefault(r[group_key], []).append((r[item_key], float(r[score_key])))
    return {
        g: [item for item, _ in sorted(items, key=lambda p: p[1], reverse=descending)]
        for g, items in groups.items()
    }


def within_task_policy_ranking(records: Sequence[Record], task_key: str = "task",
                                policy_key: str = "policy", score_key: str = "actual",
                                descending: bool = True) -> dict[Hashable, list[Any]]:
    """For each task, policies ordered by `score_key` (best first)."""
    return rank_within_group(records, task_key, policy_key, score_key, descending)


def within_policy_task_ranking(records: Sequence[Record], policy_key: str = "policy",
                                task_key: str = "task", score_key: str = "actual",
                                descending: bool = True) -> dict[Hashable, list[Any]]:
    """For each policy, tasks ordered by `score_key` (best first)."""
    return rank_within_group(records, policy_key, task_key, score_key, descending)


def within_group_correlation(records: Sequence[Record], group_key: str, x_key: str, y_key: str,
                              method: str = "spearman", min_pairs: int = 2) -> dict[Hashable, float]:
    """Correlate `x_key` against `y_key` separately within each `group_key`
    value (e.g. within each task, correlate predicted vs actual policy
    scores). Groups with fewer than `min_pairs` members, or with degenerate
    (zero-variance) x or y, get `nan` rather than raising -- this is a
    diagnostic over many small groups, not a single pooled report subject to
    the contract's min-pairs gate (use `require_min_pairs` at the reporting
    layer for that).
    """
    if method not in _CORR_FNS:
        raise ValueError(f"unknown method {method!r}, expected one of {sorted(_CORR_FNS)}")
    corr_fn = _CORR_FNS[method]
    groups: dict[Hashable, list[tuple[float, float]]] = {}
    for r in records:
        groups.setdefault(r[group_key], []).append((float(r[x_key]), float(r[y_key])))
    out: dict[Hashable, float] = {}
    for g, pairs in groups.items():
        if len(pairs) < min_pairs:
            out[g] = math.nan
            continue
        xs, ys = zip(*pairs)
        try:
            out[g] = corr_fn(xs, ys)
        except (InsufficientPairsError, DegenerateInputError):
            out[g] = math.nan
    return out


def pooled_vs_within_group_correlation(
    records: Sequence[Record], x_key: str, y_key: str, group_key: str, method: str = "pearson",
) -> tuple[float, dict[Hashable, float]]:
    """Convenience pairing of the pooled correlation with the per-group
    (within-task or within-policy) correlation, side by side -- exactly the
    comparison the contract's acceptance criteria require reporting together
    ("main reporting includes coverage and ranking, not Pearson alone")."""
    if method not in _CORR_FNS:
        raise ValueError(f"unknown method {method!r}, expected one of {sorted(_CORR_FNS)}")
    corr_fn = _CORR_FNS[method]
    xs = [r[x_key] for r in records]
    ys = [r[y_key] for r in records]
    pooled = corr_fn(xs, ys)
    within = within_group_correlation(records, group_key, x_key, y_key, method=method)
    return pooled, within


# --------------------------------------------------------------------------
# Residual correlation after removing policy/task fixed effects.
# --------------------------------------------------------------------------

def two_way_demean(records: Sequence[Record], value_key: str, row_key: str = "policy",
                    col_key: str = "task") -> np.ndarray:
    """Simple two-way ANOVA-style additive demeaning: for each record,
    subtract its row-group mean and column-group mean and add back the grand
    mean, i.e. residual = value - mean(value | row) - mean(value | col) +
    mean(value). Returns an array aligned 1:1 with `records`'s order.

    Handles repeated (row, col) combinations (e.g. multiple seeds per
    policy-task cell) by computing row/col means over all matching records,
    not just one per cell.
    """
    values = [float(r[value_key]) for r in records]
    rows = [r[row_key] for r in records]
    cols = [r[col_key] for r in records]
    if not values:
        raise InsufficientPairsError("no records to demean")
    grand_mean = float(np.mean(values))
    row_sum: dict[Hashable, float] = {}
    row_n: dict[Hashable, int] = {}
    col_sum: dict[Hashable, float] = {}
    col_n: dict[Hashable, int] = {}
    for v, rk, ck in zip(values, rows, cols):
        row_sum[rk] = row_sum.get(rk, 0.0) + v
        row_n[rk] = row_n.get(rk, 0) + 1
        col_sum[ck] = col_sum.get(ck, 0.0) + v
        col_n[ck] = col_n.get(ck, 0) + 1
    row_mean = {k: row_sum[k] / row_n[k] for k in row_sum}
    col_mean = {k: col_sum[k] / col_n[k] for k in col_sum}
    return np.array([
        v - row_mean[rk] - col_mean[ck] + grand_mean
        for v, rk, ck in zip(values, rows, cols)
    ])


def residual_correlation(records: Sequence[Record], x_key: str, y_key: str,
                          row_key: str = "policy", col_key: str = "task",
                          method: str = "pearson") -> float:
    """Correlate `x_key` and `y_key` *after* removing each's own additive
    policy (row) and task (col) fixed effect via `two_way_demean`. This
    isolates policy-task interaction signal from the (much larger, and much
    less interesting) fact that some tasks are uniformly harder and some
    policies are uniformly better.
    """
    if method not in _CORR_FNS:
        raise ValueError(f"unknown method {method!r}, expected one of {sorted(_CORR_FNS)}")
    x_resid = two_way_demean(records, x_key, row_key, col_key)
    y_resid = two_way_demean(records, y_key, row_key, col_key)
    return _CORR_FNS[method](x_resid, y_resid)


# --------------------------------------------------------------------------
# Staged progress / trajectory-contact agreement / failure-label agreement.
# Generic set- and sequence-agreement primitives over label lists.
# --------------------------------------------------------------------------

def set_agreement(predicted_set: Sequence[Any], actual_set: Sequence[Any]) -> float:
    """Jaccard-index agreement between two label sets (e.g. the set of
    object/link ids in contact at a given timestep, or a set of failure
    tags). Two empty sets are defined as fully agreeing (1.0)."""
    p = set(predicted_set)
    a = set(actual_set)
    if not p and not a:
        return 1.0
    return len(p & a) / len(p | a)


def mean_set_agreement(predicted_sets: Sequence[Sequence[Any]],
                        actual_sets: Sequence[Sequence[Any]]) -> float:
    """Mean `set_agreement` over paired sequences of sets, e.g. one contact
    set per timestep across a rollout ("trajectory-contact agreement")."""
    if len(predicted_sets) != len(actual_sets):
        raise ValueError("predicted_sets and actual_sets must be the same length")
    if not predicted_sets:
        raise InsufficientPairsError("no timesteps to compare")
    return float(np.mean([set_agreement(p, a) for p, a in zip(predicted_sets, actual_sets)]))


def label_agreement_rate(predicted_labels: Sequence[Any], actual_labels: Sequence[Any]) -> float:
    """Elementwise exact-match rate between two equal-length label lists,
    e.g. per-episode failure-mode classification ("failure-label
    agreement")."""
    if len(predicted_labels) != len(actual_labels):
        raise ValueError("predicted_labels and actual_labels must be the same length")
    if not predicted_labels:
        raise InsufficientPairsError("no labels to compare")
    return float(np.mean([p == a for p, a in zip(predicted_labels, actual_labels)]))


def _levenshtein(a: list[Any], b: list[Any]) -> int:
    """Standard O(len(a)*len(b)) edit-distance DP over arbitrary hashable
    labels (not just characters)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def sequence_agreement(predicted_seq: Sequence[Any], actual_seq: Sequence[Any]) -> float:
    """Normalized similarity between two (possibly different-length) label
    sequences: 1 - edit_distance / max(len(predicted_seq), len(actual_seq)).
    Generic over any hashable label type; intended for staged-progress stage
    sequences (e.g. [reach, grasp, lift, place]) or frame-level
    trajectory/contact label sequences of differing rollout length. Two
    empty sequences are defined as fully agreeing (1.0)."""
    p = list(predicted_seq)
    a = list(actual_seq)
    if not p and not a:
        return 1.0
    dist = _levenshtein(p, a)
    return 1.0 - dist / max(len(p), len(a))


def mean_sequence_agreement(predicted_seqs: Sequence[Sequence[Any]],
                             actual_seqs: Sequence[Sequence[Any]]) -> float:
    """Mean `sequence_agreement` over paired sequences of sequences."""
    if len(predicted_seqs) != len(actual_seqs):
        raise ValueError("predicted_seqs and actual_seqs must be the same length")
    if not predicted_seqs:
        raise InsufficientPairsError("no sequences to compare")
    return float(np.mean([sequence_agreement(p, a) for p, a in zip(predicted_seqs, actual_seqs)]))


# --------------------------------------------------------------------------
# Coverage / risk-coverage curve for certification.
# --------------------------------------------------------------------------

@dataclass
class RiskCoverageCurve:
    thresholds: np.ndarray       # accepted-if-<=-this predicted score, ascending
    coverage: np.ndarray         # fraction of the *included* population accepted
    mean_error: np.ndarray       # mean actual_error among the accepted subset
    auc: float                   # area under the risk(coverage) curve (lower = better)
    n_included: int
    n_total: int


def risk_coverage_curve(predicted_error_or_score: Sequence[float], actual_error: Sequence[float],
                         included_bool: Sequence[bool] | None = None) -> RiskCoverageCurve:
    """Build the selective-prediction risk-coverage curve (Geifman &
    El-Yaniv 2017) for a certificate: sweep an acceptance threshold over
    `predicted_error_or_score` (lower predicted error = more confident, so
    it is accepted first) and, at each threshold, report the coverage
    (fraction of eligible items accepted) and the mean *actual* error among
    the accepted subset.

    `included_bool` marks which items are eligible for certification at all
    (e.g. have ground truth / are not excluded for reasons unrelated to the
    predicted score); items with `included_bool[i] is False` are dropped
    before the sweep and counted only in `n_total`. If `included_bool` is
    omitted, every item is eligible.

    `auc` is the area under the risk(coverage) curve via trapezoidal
    integration (the "Area Under the Risk-Coverage curve", AURC) over
    coverage in [1/n_included, 1] -- lower is better (a well-calibrated
    certificate should have low mean error even at high coverage).
    """
    pred = np.asarray(predicted_error_or_score, dtype=float)
    act = np.asarray(actual_error, dtype=float)
    n_total = pred.size
    if pred.shape != act.shape:
        raise ValueError(f"shape mismatch: predicted has {pred.shape}, actual has {act.shape}")
    if included_bool is None:
        mask = np.ones(n_total, dtype=bool)
    else:
        mask = np.asarray(included_bool, dtype=bool)
        if mask.shape != pred.shape:
            raise ValueError(f"shape mismatch: included_bool has {mask.shape}, expected {pred.shape}")

    pred_i = pred[mask]
    act_i = act[mask]
    n_included = pred_i.size
    if n_included == 0:
        raise InsufficientPairsError("no included items to build a risk-coverage curve from")

    order = np.argsort(pred_i, kind="stable")
    sorted_pred = pred_i[order]
    sorted_act = act_i[order]
    ranks = np.arange(1, n_included + 1)
    coverage = ranks / n_included
    mean_error = np.cumsum(sorted_act) / ranks
    auc = float(np.trapz(mean_error, coverage))
    return RiskCoverageCurve(thresholds=sorted_pred, coverage=coverage, mean_error=mean_error,
                              auc=auc, n_included=n_included, n_total=n_total)


def coverage_at_error_budget(curve: RiskCoverageCurve, max_error: float) -> float:
    """Largest coverage achievable while keeping the accepted subset's mean
    error <= `max_error` (0.0 if even the single most-confident item
    exceeds the budget)."""
    ok = curve.coverage[curve.mean_error <= max_error]
    return float(ok.max()) if ok.size else 0.0
