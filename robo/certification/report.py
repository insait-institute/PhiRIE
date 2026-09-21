"""Task 12 -- assembles the certificate's evaluation report JSON.

Consumes already-computed out-of-fold (OOF) predictions (from
`robo.certification.calibrate`) and already-implemented metric primitives
(`agents.eval.predictive_metrics`) -- this module does no metric math of its
own beyond trivial array indexing (e.g. "read the mean_error off an
already-built risk-coverage curve at coverage>=0.8"). Per the task brief:
call `risk_coverage_curve`/`calibration_brier`/`brier_score`/
`coverage_at_error_budget` from `agents.eval.predictive_metrics`, never
reimplement them.

Output shape (`build_certificate_report`'s return value) is documented
inline on each assembly function below; the top-level keys a future repair
module (Task 13, not yet built) should read are `top_contributing_features`
(global, dataset-level) and `per_pair_diagnostics[*].top_contributing_features`
(local, per policy-task pair) -- both lists of
`{feature, direction, magnitude}` dicts, per the plan's instruction 7.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from agents.eval.predictive_metrics import (
    InsufficientPairsError,
    RiskCoverageCurve,
    calibration_brier,
    risk_coverage_curve,
)

COVERAGE_LEVELS = (1.0, 0.8, 0.6)
MONOTONIC_TOL = 1e-9


def mean_error_at_coverage(curve: RiskCoverageCurve, target_coverage: float) -> float | None:
    """Mean actual error of the accepted subset at (at least) `target_coverage`.

    `curve.coverage` is ascending; the smallest achieved coverage >=
    `target_coverage` is used (falls back to the largest available coverage
    when `target_coverage` exceeds what the curve has, which only happens
    for degenerate 1-item curves)."""
    idx = np.searchsorted(curve.coverage, target_coverage)
    idx = min(idx, len(curve.coverage) - 1)
    return float(curve.mean_error[idx])


def is_accepted_error_monotonic(curve: RiskCoverageCurve, tol: float = MONOTONIC_TOL) -> bool:
    """Strict, point-by-point version of the acceptance criterion
    "Accepted-set error decreases monotonically as coverage is reduced" --
    `mean_error` must be non-decreasing at every single point as `coverage`
    increases one item at a time. This is a much stricter bar than the plan
    text's own "within CI" qualifier: with a few dozen records, adding one
    single item to a tiny accepted subset can easily produce a one- or
    two-item local wiggle from sampling noise alone, well within any
    reasonable confidence interval, even when the certificate is genuinely
    informative. See `is_accepted_error_monotonic_at_coverage_checkpoints`
    for the coarser, checkpoint-level version that is closer in spirit to
    what the plan actually asks to be reported (100/80/60% coverage) and
    what a bootstrap-CI version of this check would likely conclude without
    requiring this module to implement its own bootstrap."""
    if len(curve.mean_error) < 2:
        return True
    return bool(np.all(np.diff(curve.mean_error) >= -tol))


def is_accepted_error_monotonic_at_coverage_checkpoints(
        curve: RiskCoverageCurve, levels: Sequence[float] = COVERAGE_LEVELS) -> bool:
    """Coarser companion to `is_accepted_error_monotonic`: checks monotonicity
    only across the specific coverage checkpoints the plan's acceptance
    criteria ask to report (100/80/60% by default), which are far enough
    apart that a real signal survives the kind of single-item sampling noise
    that can trip the strict, every-point version above."""
    values = [mean_error_at_coverage(curve, c) for c in sorted(levels)]
    if len(values) < 2:
        return True
    return bool(np.all(np.diff(values) >= -MONOTONIC_TOL))


def curve_summary(curve: RiskCoverageCurve) -> dict[str, Any]:
    return {
        "auc": curve.auc,
        "n_included": curve.n_included,
        "n_total": curve.n_total,
        "accepted_error_monotonic_strict": is_accepted_error_monotonic(curve),
        "accepted_error_monotonic_at_coverage_checkpoints":
            is_accepted_error_monotonic_at_coverage_checkpoints(curve),
        "coverage_mean_error": {
            f"{int(c * 100)}pct": mean_error_at_coverage(curve, c) for c in COVERAGE_LEVELS
        },
    }


def safe_auroc(y_true: Sequence[float], y_score: Sequence[float]) -> float | None:
    y_true = np.asarray(y_true, dtype=float)
    if len(set(y_true.tolist())) < 2:
        return None  # AUROC undefined for a single-class held-out pool
    return float(roc_auc_score(y_true, y_score))


def safe_calibration_brier(y_prob: Sequence[float], y_outcome: Sequence[float]) -> float | None:
    try:
        return calibration_brier(y_prob, y_outcome)
    except InsufficientPairsError:
        return None


def failure_cases(pair_ids: Sequence[str], predicted: Sequence[float], actual: Sequence[float],
                   top_k: int = 5) -> list[dict]:
    """Largest |predicted - actual| residuals -- "failure cases" the plan's
    acceptance criteria explicitly ask to report alongside the coverage
    numbers, so a reader sees where the certificate is most wrong, not only
    its aggregate curve."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    residual = np.abs(predicted - actual)
    order = np.argsort(-residual)[:top_k]
    return [
        {"pair_id": pair_ids[i], "predicted": float(predicted[i]), "actual": float(actual[i]),
         "abs_residual": float(residual[i])}
        for i in order
    ]


def model_tier_report(pair_ids: Sequence[str], oof_error_pred: np.ndarray,
                       oof_bad_pair_prob: np.ndarray | None, actual_error: np.ndarray,
                       bad_pair_label: np.ndarray | None) -> dict[str, Any]:
    """One model tier's full evaluation: risk-coverage on the error
    prediction, plus (when a classifier prediction is given) AUROC and
    calibration Brier on the bad-pair probability."""
    included = ~np.isnan(oof_error_pred)
    out: dict[str, Any] = {"n_evaluated": int(included.sum())}
    if included.sum() >= 2:
        curve = risk_coverage_curve(oof_error_pred, actual_error, included_bool=included)
        out["risk_coverage"] = curve_summary(curve)
        out["failure_cases"] = failure_cases(
            [p for p, m in zip(pair_ids, included) if m],
            oof_error_pred[included], actual_error[included])
    else:
        out["risk_coverage"] = None
        out["failure_cases"] = []

    if oof_bad_pair_prob is not None and bad_pair_label is not None:
        cls_included = ~np.isnan(oof_bad_pair_prob)
        out["n_evaluated_classification"] = int(cls_included.sum())
        if cls_included.sum() >= 2:
            out["auroc"] = safe_auroc(bad_pair_label[cls_included], oof_bad_pair_prob[cls_included])
            out["calibration_brier"] = safe_calibration_brier(
                oof_bad_pair_prob[cls_included], bad_pair_label[cls_included])
        else:
            out["auroc"] = None
            out["calibration_brier"] = None
    return out


def baseline_report(pair_ids: Sequence[str], baseline_goodness: np.ndarray,
                     actual_error: np.ndarray) -> dict[str, Any]:
    """A global single-metric baseline (PSNR / geometry F1 / drop stability):
    `baseline_goodness` is oriented "higher is better" (e.g. raw PSNR in dB);
    negated here so it plugs into `risk_coverage_curve`'s "lower predicted
    score accepted first" convention on equal footing with the certificate's
    own predicted-error score."""
    included = ~np.isnan(baseline_goodness)
    if included.sum() < 2:
        return {"risk_coverage": None, "n_evaluated": int(included.sum())}
    predicted_risk = -baseline_goodness
    curve = risk_coverage_curve(predicted_risk, actual_error, included_bool=included)
    return {"n_evaluated": int(included.sum()), "risk_coverage": curve_summary(curve)}


def beats_all_baselines(cert_aucs: Sequence[float | None],
                         baseline_aucs: Sequence[float | None]) -> bool | None:
    """G4 / plan acceptance criterion: "Held-out-scene performance beats
    every global single-metric baseline." Lower risk-coverage AUC is better
    (see `RiskCoverageCurve.auc`'s docstring). Compares the *best* available
    certificate model tier against the *worst* (most favorable-to-baseline)
    global baseline, i.e. the hardest version of this claim: `True` only if
    every certificate tier beats every baseline is too strict for a
    3-baseline x 3-tier grid with limited data, so this checks "does the
    best certificate tier beat the best baseline" -- the literal minimum bar
    the G4 gate language asks for. Returns `None` (not `False`) when either
    side has no valid AUC at all, so an honest "cannot be assessed" is never
    conflated with an honest "no, it does not beat them"."""
    cert_valid = [a for a in cert_aucs if a is not None]
    base_valid = [a for a in baseline_aucs if a is not None]
    if not cert_valid or not base_valid:
        return None
    return min(cert_valid) < min(base_valid)


def global_feature_importance(pipeline, column_names: Sequence[str], X: np.ndarray,
                               top_k: int = 8) -> list[dict[str, Any]]:
    """Dataset-level ranking of features by `|standardized coefficient| *
    std(imputed feature value)` -- a global counterpart to
    `model.top_contributing_features`'s per-record local explanation, for a
    fleet-wide "what generally matters" view. Empty list for any tier that
    is not the linear pipeline (monotonic/tree tiers don't expose a single
    linear coefficient vector to summarize this way)."""
    try:
        scaler = pipeline.named_steps["scale"]
        model = pipeline.named_steps.get("ridge") or pipeline.named_steps.get("logreg")
        imputer = pipeline.named_steps["impute"]
    except (KeyError, AttributeError, TypeError):
        return []
    if model is None or not hasattr(model, "coef_"):
        return []
    coef = np.ravel(model.coef_)
    x_imputed = imputer.transform(X)
    x_std = scaler.transform(x_imputed)
    importance = np.abs(coef) * np.std(x_std, axis=0)
    order = np.argsort(-importance)[:top_k]
    return [
        {"feature": column_names[i],
         "direction": "increases_risk" if coef[i] > 0 else "decreases_risk",
         "magnitude": float(importance[i])}
        for i in order
    ]
