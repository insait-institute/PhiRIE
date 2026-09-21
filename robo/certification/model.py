"""Task 12 -- small model tiers for the task-conditioned twin certificate.

Per the plan's instruction 2 ("Fit deliberately small baselines") and the
ICRA contract's own bias toward interpretable models given how little
oracle_causal data exists (`docs/ICRA_RESEARCH_CONTRACT.md` sec 4's
`min_nondegenerate_pairs: 12`), this module offers exactly three model tiers,
each available for both regression (predict the error magnitude) and
classification (predict bad-pair probability):

  1. `monotonic` -- a single feature, chosen at fit time by rank-correlation
     with the target, used directly (sign-corrected) as the score. No
     interaction terms, no more than one coefficient in the loosest sense.
  2. `linear_l2` -- `sklearn.linear_model.Ridge` / `LogisticRegression`,
     L2-regularized, over an imputed+standardized feature matrix.
  3. `tree_calibrated` -- a small `GradientBoostingRegressor` /
     `GradientBoostingClassifier` (<=40 shallow trees), the classifier
     wrapped in `CalibratedClassifierCV` (isotonic/sigmoid) so its predicted
     bad-pair probability is an actual probability, not a raw score.

Missingness handling: `build_feature_matrix` turns a list of feature dicts
(as returned by `robo.certification.features.extract_all_features`) into a
numpy matrix with **both** the raw value column and its `_missing` indicator
column for every feature name -- the indicator is never dropped, so every
model here can (and the linear/tree tiers typically do) learn "this
evidence being absent is itself informative" directly from data, rather than
that information being silently thrown away at imputation time. The
`SimpleImputer(strategy="median")` used inside the linear/tree pipelines only
ever touches the raw-value columns (paired indicator columns hold no NaNs to
begin with) and imputes to the training fold's median -- a neutral value, not
the "pass" value -- exactly so imputation cannot pass as evidence of a good
build.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from robo.certification.features import ALL_FEATURE_NAMES

DEFAULT_CONTRACT_PATH = Path("configs/experiments/icra_contract_v1.yaml")
# Fallback if the contract file is unreachable from the caller's cwd -- kept
# in sync manually with icra_contract_v1.yaml's
# protocols.oracle_causal.go_no_go.staged_progress_mae_ci_upper_lt.
FALLBACK_BAD_PAIR_ERROR_THRESHOLD = 0.35


def load_bad_pair_error_threshold(contract_path: str | Path | None = DEFAULT_CONTRACT_PATH) -> float:
    """`oracle_causal.go_no_go.staged_progress_mae_ci_upper_lt` from
    `configs/experiments/icra_contract_v1.yaml` -- "roughly one stage of
    disagreement" per the contract's own comment. A pair whose actual
    staged-progress error exceeds this is a "bad pair" for the
    classification tiers below. Falls back to the frozen literal
    (`FALLBACK_BAD_PAIR_ERROR_THRESHOLD`) if the file cannot be read, so
    tests/CLI runs from an arbitrary cwd never silently use a *different*,
    undocumented threshold -- they use the one frozen value either way.
    """
    if contract_path is not None:
        try:
            cfg = yaml.safe_load(Path(contract_path).read_text())
            return float(cfg["protocols"]["oracle_causal"]["go_no_go"]
                         ["staged_progress_mae_ci_upper_lt"])
        except (OSError, KeyError, TypeError, ValueError):
            pass
    return FALLBACK_BAD_PAIR_ERROR_THRESHOLD


def bad_pair_label(actual_error: Sequence[float], threshold: float) -> np.ndarray:
    """1.0 ("bad pair") where `actual_error > threshold`, else 0.0."""
    return (np.asarray(actual_error, dtype=float) > threshold).astype(float)


# --------------------------------------------------------------------------
# Feature matrix assembly
# --------------------------------------------------------------------------

def feature_matrix_columns() -> list[str]:
    """Fixed, deterministic column order: for each name in
    `features.ALL_FEATURE_NAMES`, the value column then its `_missing`
    column. Every caller of `build_feature_matrix` gets this exact order,
    so a fitted model's coefficients/importances always line up with the
    same named column regardless of which records were passed in."""
    cols = []
    for name in ALL_FEATURE_NAMES:
        cols.append(name)
        cols.append(f"{name}_missing")
    return cols


def build_feature_matrix(feature_dicts: Sequence[Mapping[str, float | None]]
                          ) -> tuple[np.ndarray, list[str]]:
    """`feature_dicts`: one `extract_all_features(...)`-shaped dict per
    record. Returns `(X, column_names)`, `X` shape `(n_records, n_columns)`,
    `None` values mapped to `np.nan` (left for `SimpleImputer` to handle
    inside each model pipeline -- never imputed here)."""
    cols = feature_matrix_columns()
    X = np.full((len(feature_dicts), len(cols)), np.nan, dtype=float)
    for i, d in enumerate(feature_dicts):
        for j, c in enumerate(cols):
            v = d.get(c)
            if v is not None:
                X[i, j] = float(v)
    return X, cols


# --------------------------------------------------------------------------
# Tier 1: monotonic single-feature models
# --------------------------------------------------------------------------

class MonotonicSingleFeatureRegressor:
    """Baseline model tier: at fit time, pick the one feature column whose
    Spearman rank-correlation with the target has the largest magnitude
    (ties broken by column order), and predict `sign * (imputed) column
    value` directly -- no other coefficients, no interactions. `sign` is
    -1 when the correlation is negative, so "more of this feature" always
    maps to "predicted more error", matching every other model tier's
    prediction convention (higher predicted value = certificate believes
    the pair is worse)."""

    def __init__(self) -> None:
        self.selected_idx_: int | None = None
        self.sign_: float = 1.0
        self.impute_value_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MonotonicSingleFeatureRegressor":
        from scipy import stats
        y = np.asarray(y, dtype=float)
        best_j, best_abs_rho, best_sign = None, -1.0, 1.0
        for j in range(X.shape[1]):
            col = X[:, j]
            mask = ~np.isnan(col)
            if mask.sum() < 3 or np.std(col[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = stats.spearmanr(col[mask], y[mask])
            if rho != rho:
                continue
            if abs(rho) > best_abs_rho:
                best_abs_rho, best_j = abs(rho), j
                best_sign = 1.0 if rho >= 0 else -1.0
        if best_j is None:
            best_j = 0
        self.selected_idx_ = best_j
        self.sign_ = best_sign
        col = X[:, best_j]
        mask = ~np.isnan(col)
        self.impute_value_ = float(np.median(col[mask])) if mask.any() else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.selected_idx_].copy()
        col[np.isnan(col)] = self.impute_value_
        return self.sign_ * col


class MonotonicSingleFeatureClassifier:
    """Classification counterpart: select the single most-informative
    feature by point-biserial correlation with the binary label, then fit a
    1-variable L2 logistic regression on it (so `predict_proba` is a real,
    monotone-in-that-one-feature probability, not just a raw score)."""

    def __init__(self) -> None:
        self.selected_idx_: int | None = None
        self.impute_value_: float = 0.0
        # LogisticRegression's default penalty is already L2 (sklearn>=1.8
        # deprecates the explicit penalty="l2" spelling in favor of leaving
        # it at the default), so C alone controls the L2 regularization
        # strength here.
        self._clf = LogisticRegression(C=1.0, max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MonotonicSingleFeatureClassifier":
        from scipy import stats
        y = np.asarray(y, dtype=float)
        best_j, best_abs_r = None, -1.0
        for j in range(X.shape[1]):
            col = X[:, j]
            mask = ~np.isnan(col)
            if mask.sum() < 3 or np.std(col[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, _ = stats.pointbiserialr(y[mask], col[mask])
            if r != r:
                continue
            if abs(r) > best_abs_r:
                best_abs_r, best_j = abs(r), j
        if best_j is None:
            best_j = 0
        self.selected_idx_ = best_j
        col = X[:, best_j]
        mask = ~np.isnan(col)
        self.impute_value_ = float(np.median(col[mask])) if mask.any() else 0.0
        col_filled = col.copy()
        col_filled[np.isnan(col_filled)] = self.impute_value_
        if len(set(y.tolist())) < 2:
            self._clf = _ConstantClassifier(float(y[0]) if len(y) else 0.0)
        else:
            self._clf.fit(col_filled.reshape(-1, 1), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.selected_idx_].copy()
        col[np.isnan(col)] = self.impute_value_
        return self._clf.predict_proba(col.reshape(-1, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


class _ConstantClassifier:
    """Degenerate fallback when a training fold has only one class (small
    LOSO/LOTFO folds can do this) -- predicts that class's probability as a
    constant, rather than letting sklearn raise on a single-class `.fit()`."""

    def __init__(self, constant_label: float) -> None:
        self.constant_label = constant_label

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p1 = 1.0 if self.constant_label >= 0.5 else 0.0
        n = X.shape[0]
        return np.column_stack([np.full(n, 1 - p1), np.full(n, p1)])


# --------------------------------------------------------------------------
# Tier 2 & 3: regularized linear and calibrated-tree pipelines
# --------------------------------------------------------------------------

def _imputer() -> SimpleImputer:
    # keep_empty_features=True is load-bearing, not cosmetic: several
    # feature columns (e.g. the forward-declared dynamics-probe/IK-margin
    # features in dynamics_probes.py/robot_control.py) are honestly all-NaN
    # for every record until an upstream probe component exists. sklearn's
    # default (keep_empty_features=False) silently DROPS such a column from
    # the transformed output, which would shift every later pipeline step's
    # column indices out of sync with `feature_matrix_columns()` -- silently
    # breaking `report.global_feature_importance`/`model.top_contributing_
    # features`'s `column_names[i]` lookups (and doing so differently per
    # LOSO/LOTFO fold, since which columns are all-NaN can differ by
    # training subset). Keeping the column (imputed to 0 when literally
    # every training value is missing) trades a harmless constant column for
    # a guaranteed-stable column count.
    return SimpleImputer(strategy="median", keep_empty_features=True)


def make_linear_regressor(alpha: float = 1.0) -> Pipeline:
    return Pipeline([("impute", _imputer()), ("scale", StandardScaler()),
                      ("ridge", Ridge(alpha=alpha))])


def make_linear_classifier(C: float = 1.0) -> Pipeline:
    # LogisticRegression's default penalty is already L2 -- see the comment
    # in MonotonicSingleFeatureClassifier.fit above.
    return Pipeline([("impute", _imputer()), ("scale", StandardScaler()),
                      ("logreg", LogisticRegression(C=C, max_iter=2000))])


def make_tree_regressor(n_estimators: int = 30, max_depth: int = 2) -> Pipeline:
    return Pipeline([("impute", _imputer()),
                      ("gbrt", GradientBoostingRegressor(n_estimators=n_estimators,
                                                          max_depth=max_depth,
                                                          learning_rate=0.1,
                                                          random_state=0))])


def _safe_calibration_cv(y: np.ndarray, requested: int = 3) -> int:
    """CalibratedClassifierCV's internal `cv` folds each need >=1 example of
    every class. Shrinks `requested` to fit the smaller class's count on
    small LOSO/LOTFO training folds rather than letting sklearn raise."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 0  # signal: caller should skip calibration entirely
    return max(2, min(requested, int(counts.min())))


class CalibratedTreeClassifier:
    """`GradientBoostingClassifier` (small, shallow) wrapped in
    `CalibratedClassifierCV`. Falls back to the uncalibrated pipeline's own
    `predict_proba` (still a valid, if less-calibrated, probability) when a
    training fold has too few examples of one class to support internal
    calibration folds -- never raises out of a LOSO/LOTFO fold."""

    def __init__(self, n_estimators: int = 30, max_depth: int = 2, method: str = "sigmoid"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.method = method
        self._model: Any = None

    def _base_pipeline(self) -> Pipeline:
        return Pipeline([("impute", _imputer()),
                          ("gbc", GradientBoostingClassifier(n_estimators=self.n_estimators,
                                                              max_depth=self.max_depth,
                                                              learning_rate=0.1,
                                                              random_state=0))])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CalibratedTreeClassifier":
        y = np.asarray(y, dtype=float)
        if len(set(y.tolist())) < 2:
            self._model = _ConstantClassifier(float(y[0]) if len(y) else 0.0)
            return self
        cv = _safe_calibration_cv(y, requested=3)
        base = self._base_pipeline()
        if cv == 0:
            base.fit(X, y)
            self._model = base
        else:
            self._model = CalibratedClassifierCV(base, method=self.method, cv=cv)
            self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


class SingleClassSafeClassifier:
    """Wraps any sklearn-style classifier factory; falls back to
    `_ConstantClassifier` when a training fold contains only one class
    (small LOSO/LOTFO folds can produce this -- e.g. a scene whose every
    other-scene training fold happens to be all "good" pairs) instead of
    letting the wrapped classifier's `.fit()` raise. `CalibratedTreeClassifier`
    above has the same guard built in directly; this wrapper gives the plain
    `linear_l2` sklearn Pipeline the same safety without changing its
    coefficients/behavior on any fold that does have both classes."""

    def __init__(self, pipeline_factory: Callable[[], Any]) -> None:
        self._pipeline_factory = pipeline_factory
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SingleClassSafeClassifier":
        y = np.asarray(y, dtype=float)
        if len(set(y.tolist())) < 2:
            self._model = _ConstantClassifier(float(y[0]) if len(y) else 0.0)
        else:
            self._model = self._pipeline_factory()
            self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


# --------------------------------------------------------------------------
# Registries used by calibrate.py
# --------------------------------------------------------------------------

REGRESSION_MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "monotonic": MonotonicSingleFeatureRegressor,
    "linear_l2": make_linear_regressor,
    "tree_calibrated": make_tree_regressor,
}

CLASSIFICATION_MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "monotonic": MonotonicSingleFeatureClassifier,
    "linear_l2": lambda: SingleClassSafeClassifier(make_linear_classifier),
    "tree_calibrated": CalibratedTreeClassifier,
}


@dataclass
class FeatureContribution:
    feature: str
    direction: str  # "increases_risk" | "decreases_risk"
    magnitude: float


def top_contributing_features(pipeline: Pipeline, column_names: Sequence[str],
                               x_row: np.ndarray, top_k: int = 5
                               ) -> list[FeatureContribution]:
    """Lightweight per-record explanation for a fitted `linear_l2` pipeline
    (Ridge or LogisticRegression): rank columns by
    `|standardized_coefficient * standardized_feature_value|` and return the
    top `top_k`. This is the "simple 'top contributing features by
    |coefficient * feature value|'" version the plan explicitly allows as an
    acceptable lightweight substitute for full SHAP, logged here so Task 13
    (repair routing, not yet built) has a stable field to read
    (`report.py`'s `top_contributing_features`)."""
    try:
        scaler: StandardScaler = pipeline.named_steps["scale"]
        model = pipeline.named_steps.get("ridge") or pipeline.named_steps.get("logreg")
        imputer: SimpleImputer = pipeline.named_steps["impute"]
    except (KeyError, AttributeError):
        return []
    if model is None or not hasattr(model, "coef_"):
        return []
    coef = np.ravel(model.coef_)
    x_imputed = imputer.transform(x_row.reshape(1, -1))[0]
    x_std = scaler.transform(x_imputed.reshape(1, -1))[0]
    contrib = coef * x_std
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [
        FeatureContribution(
            feature=column_names[i],
            direction="increases_risk" if contrib[i] > 0 else "decreases_risk",
            magnitude=float(abs(contrib[i])),
        )
        for i in order
    ]
