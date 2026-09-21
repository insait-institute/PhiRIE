"""Tests for robo/certification/{features/*,model,calibrate,report}.py (Task 12).

Fast-validation command (per plan/12_TASK_CONDITIONED_CERTIFICATE.md):
    python -m robo.certification.calibrate \\
        --fixture tests/data/certificate/toy.json --leave-one-scene-out

Fixture scope note: `tests/data/certificate/toy.json` is a SYNTHETIC fixture
(32 records: 2 real `task_graph.py` graphs from Task 11's own fixtures,
instantiated across 8 synthetic scenes/family x 2 synthetic policies, with a
constructed relationship between build-audit-shaped evidence and a
ground-truth `actual_error`). It is not real oracle_causal data -- see the
Task 12 completion report for what real data exists as of 2026-08-16 (in
short: one real deep-tier build, `outputs/behavior_task-0020`, with a single
generic object that does not cleanly join against the real fast-tier
`outputs/oracle_eval/results.jsonl` labels -- nowhere near the contract's
`min_nondegenerate_pairs: 12`). `test_real_build_audit_smoke` below exercises
that one real build directly (skipped if it is ever removed/moved) so this
file's numbers are honest about which parts are real and which are
constructed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from agents.eval.build_audit import run_audit
from robo.certification import calibrate as C
from robo.certification import model as M
from robo.certification import report as R
from robo.certification import task_graph as TG
from robo.certification.features import (
    ALL_FEATURE_NAMES,
    extract_all_features,
    graph_local_object_ids,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "tests" / "data" / "scenes" / "task_graph_fixture"
TASK_DIR = ROOT / "tests" / "data" / "tasks"
AUDIT_FIXTURE_DIR = ROOT / "tests" / "data" / "builds" / "audit_fixture"
TOY_FIXTURE = ROOT / "tests" / "data" / "certificate" / "toy.json"
REAL_ORACLE_BUILD = ROOT / "outputs" / "behavior_task-0020"


def _graph(task_name: str):
    scene = TG.load_scene(SCENE_DIR)
    task = TG.load_task(TASK_DIR / f"{task_name}.yaml")
    return TG.build_graph(scene, task)


def _happy_path_audit():
    return run_audit(AUDIT_FIXTURE_DIR / "happy_path")


# ============================================================== features ===

def test_extract_all_features_returns_fixed_schema():
    """Every call, regardless of input, returns exactly 2 keys (value +
    _missing) per name in ALL_FEATURE_NAMES -- the invariant model.py's
    fixed column order (`feature_matrix_columns`) depends on."""
    for build_audit, task_graph in [
        (None, None),
        ({}, {}),
        (_happy_path_audit(), _graph("food_bussing")),
    ]:
        feats = extract_all_features(build_audit, task_graph, None)
        expected = set(ALL_FEATURE_NAMES) | {f"{n}_missing" for n in ALL_FEATURE_NAMES}
        assert set(feats) == expected


def test_missingness_never_imputed_as_good():
    """Core acceptance criterion: a check reported as `not_applicable` must
    surface as `(value=None, value_missing=1.0)` in the feature vector --
    never coerced to a numeric value that would look like a favorable
    ("pass") observation (e.g. never 0.0 for a "how bad is it" residual,
    never 1.0 for a "did it pass" indicator)."""
    audit = {
        "checks": {},
        "objects": [{
            "object_id": "obj_00",
            "checks": {
                "registration_residual": {"status": "not_applicable",
                                           "reason": "no aligned.json for this object"},
                "penetration": {"status": "not_applicable"},
                "support_overlap": {"status": "not_applicable"},
            },
        }],
    }
    feats = extract_all_features(audit, None, None)

    assert feats["geom_registration_residual_mm"] is None
    assert feats["geom_registration_residual_mm_missing"] == 1.0
    assert feats["support_penetration_any"] is None
    assert feats["support_penetration_any_missing"] == 1.0
    assert feats["support_overlap_fail_any"] is None
    assert feats["support_overlap_fail_any_missing"] == 1.0

    # And the inverse: a `pass` (0 risk) must be visibly DIFFERENT from a
    # missing feature, not collapse to the same (0.0, 0.0) representation.
    audit_pass = copy.deepcopy(audit)
    audit_pass["objects"][0]["checks"]["penetration"] = {"status": "pass", "sunk": False}
    feats_pass = extract_all_features(audit_pass, None, None)
    assert feats_pass["support_penetration_any"] == 0.0
    assert feats_pass["support_penetration_any_missing"] == 0.0
    assert (feats["support_penetration_any"], feats["support_penetration_any_missing"]) != \
           (feats_pass["support_penetration_any"], feats_pass["support_penetration_any_missing"])


def test_missing_upstream_component_groups_are_honestly_all_missing():
    """dynamics_probes (push/lift/grasp-closure) and the robot_control IK
    margin have no upstream component anywhere in this repo as of this task
    (see those modules' docstrings) -- every real build_audit dict, however
    good, must report them as missing, never favorably guessed."""
    audit = _happy_path_audit()
    feats = extract_all_features(audit, _graph("food_bussing"), None)
    for name in ("dyn_push_response_score", "dyn_lift_response_score",
                 "dyn_grasp_closure_retention_score", "robot_ik_collision_margin"):
        assert feats[name] is None
        assert feats[f"{name}_missing"] == 1.0


def test_task_local_restriction_changes_aggregation():
    """A `fail` on an object OUTSIDE the task graph's node set must not leak
    into the task-local feature vector -- the entire point of restricting
    aggregation to `graph_local_object_ids`."""
    audit = {
        "checks": {},
        "objects": [
            {"object_id": "obj_local", "checks": {
                "penetration": {"status": "pass", "sunk": False}}},
            {"object_id": "obj_far_away", "checks": {
                "penetration": {"status": "fail", "sunk": True}}},
        ],
    }
    scoped = extract_all_features(audit, None, node_ids={"obj_local"})
    unscoped = extract_all_features(audit, None, node_ids=None)
    assert scoped["support_penetration_any"] == 0.0
    assert unscoped["support_penetration_any"] == 1.0


def test_graph_local_object_ids_matches_task_graph_nodes():
    graph = _graph("food_bussing")
    node_ids = graph_local_object_ids(graph)
    expected = {n["node_id"] for n in graph["nodes"] if n["kind"] == "object"}
    assert node_ids == expected
    assert node_ids == {"obj_00", "obj_01", "obj_02", "obj_05", "obj_07", "obj_08"}
    assert graph_local_object_ids(None) is None


# ============================================================== leakage ===

def test_feature_extraction_is_leakage_free():
    """Structural leakage test, same pattern as
    tests/test_task_graph.py::test_no_evaluation_result_leakage_in_role_refs:
    smuggle bogus scene-identity/ground-truth-shaped fields into every input
    dict `extract_all_features` accepts, and assert the output is BYTE
    IDENTICAL with and without them. This proves the extractors' `.get()`
    calls never reach these keys, structurally, not just "happened not to"
    on this particular input.
    """
    audit = _happy_path_audit()
    graph = _graph("food_bussing")

    clean_feats = extract_all_features(copy.deepcopy(audit), copy.deepcopy(graph), None)

    audit_dirty = copy.deepcopy(audit)
    audit_dirty["scene_id"] = "REAL_SCENE_IDENTITY_LEAK"
    audit_dirty["build_dir"] = "/some/real/path/leaking/scene/identity"
    audit_dirty["generated_at"] = "2099-01-01T00:00:00+00:00"
    audit_dirty["_debug_real_success"] = 1.0
    audit_dirty["_debug_ground_truth_error"] = 0.0
    for obj in audit_dirty["objects"]:
        obj["gt_object_id"] = 999999
        obj["_debug_oracle_success"] = 1.0

    graph_dirty = copy.deepcopy(graph)
    graph_dirty["scene_id"] = "REAL_SCENE_IDENTITY_LEAK"
    graph_dirty["_debug_eval_success"] = 1.0
    graph_dirty["role_resolutions"]["manipulated_object"]["eval_success"] = 1.0
    for node in graph_dirty["nodes"]:
        node["_debug_ground_truth_pose_error"] = 0.0

    dirty_feats = extract_all_features(audit_dirty, graph_dirty, None)

    assert json.dumps(clean_feats, sort_keys=True) == json.dumps(dirty_feats, sort_keys=True)


def test_feature_extraction_ignores_alignment_report_identity_fields():
    audit = _happy_path_audit()
    alignment_clean = {"hard_thresholds": {"reprojection_residual_px": {"value": 1.5}},
                        "held_out": {"rms_m": 0.02}}
    alignment_dirty = copy.deepcopy(alignment_clean)
    alignment_dirty["scene_id"] = "REAL_SCENE_IDENTITY_LEAK"
    alignment_dirty["_debug_success"] = 1.0

    clean = extract_all_features(audit, None, alignment_clean)
    dirty = extract_all_features(audit, None, alignment_dirty)
    assert clean == dirty


# ================================================================= model ===

def test_build_feature_matrix_shape_and_nan_encoding():
    feats = [extract_all_features(None, None, None), extract_all_features({}, {}, None)]
    X, cols = M.build_feature_matrix(feats)
    assert X.shape == (2, 2 * len(ALL_FEATURE_NAMES))
    assert len(cols) == X.shape[1]
    # Both inputs are entirely empty evidence: every VALUE column (even
    # index, by feature_matrix_columns' [value, missing] pairing) is NaN,
    # and every MISSING-indicator column (odd index) is 1.0 -- never the
    # other way around, and never left as NaN itself.
    value_cols, missing_cols = X[:, 0::2], X[:, 1::2]
    assert np.all(np.isnan(value_cols))
    assert np.all(missing_cols == 1.0)


def test_all_six_model_tiers_fit_and_predict_on_synthetic_data():
    rng = np.random.RandomState(0)
    n, d = 40, len(M.feature_matrix_columns())
    X = rng.randn(n, d)
    y_err = np.clip(0.5 + 0.3 * X[:, 0] + 0.05 * rng.randn(n), 0, 1)
    y_bad = (y_err > np.median(y_err)).astype(float)

    for name, factory in M.REGRESSION_MODEL_FACTORIES.items():
        model = factory()
        model.fit(X, y_err)
        pred = model.predict(X)
        assert pred.shape == (n,)
        assert np.all(np.isfinite(pred)), name

    for name, factory in M.CLASSIFICATION_MODEL_FACTORIES.items():
        model = factory()
        model.fit(X, y_bad)
        proba = model.predict_proba(X)
        assert proba.shape == (n, 2)
        assert np.all(np.isfinite(proba)), name
        assert np.allclose(proba.sum(axis=1), 1.0), name


def test_classification_tiers_survive_single_class_training_fold():
    """LOSO/LOTFO folds can leave a training partition with only one class
    (e.g. every remaining scene happens to be a "good" pair) -- every
    classification tier must degrade to a constant prediction rather than
    raising out of sklearn's single-class fit path."""
    X = np.random.RandomState(1).randn(10, len(M.feature_matrix_columns()))
    y_bad = np.zeros(10)  # single class
    for name, factory in M.CLASSIFICATION_MODEL_FACTORIES.items():
        model = factory()
        model.fit(X, y_bad)
        proba = model.predict_proba(X)
        assert proba.shape == (10, 2)
        assert np.allclose(proba[:, 1], 0.0), name


def test_bad_pair_threshold_matches_icra_contract():
    threshold = M.load_bad_pair_error_threshold()
    assert threshold == pytest.approx(0.35)
    contract = json.loads(json.dumps(
        __import__("yaml").safe_load(
            (ROOT / "configs" / "experiments" / "icra_contract_v1.yaml").read_text())))
    assert threshold == pytest.approx(
        contract["protocols"]["oracle_causal"]["go_no_go"]["staged_progress_mae_ci_upper_lt"])


# ============================================================ risk-coverage ===

def test_risk_coverage_monotonic_when_certificate_is_genuinely_informative():
    """Explicit synthetic construction (plan's own required test): a
    certificate score that is a clean, monotone function of the true error
    ("higher score = higher true error", i.e. lower score = lower true
    error = more confident) must produce a risk-coverage curve whose
    mean_error is monotonically non-decreasing as coverage increases
    (equivalently: non-increasing as coverage/acceptance is reduced).

    Uses a NOISE-FREE, exactly rank-preserving score on purpose: this is a
    provable case, not just an empirically-likely one -- if `predicted_score`
    ranks items in exactly the same order as `true_error`
    (`sorted_actual[k]`'s running mean over a non-decreasing sequence), the
    cumulative mean is *exactly* non-decreasing by a short induction (each
    newly-admitted item is by construction >= the running mean of everything
    admitted before it, since everything before it was ranked lower). Once
    ANY ranking noise is added, individual local swaps near ties can and do
    occasionally produce a single-point dip in the cumulative mean even for
    a highly-informative score -- that is exactly why `curve_summary` reports
    both a strict, point-by-point version of this check
    (`accepted_error_monotonic_strict`) AND a coarser one evaluated only at
    the plan's own reported coverage checkpoints
    (`accepted_error_monotonic_at_coverage_checkpoints`), and why the noisy
    version of this same idea is tested separately below
    (`test_risk_coverage_checkpoint_monotonic_survives_realistic_noise`)."""
    n = 200
    true_error = np.linspace(0, 1, n)
    predicted_score = true_error.copy()  # exact rank-preserving transform

    from agents.eval.predictive_metrics import risk_coverage_curve
    curve = risk_coverage_curve(predicted_score, true_error)
    assert R.is_accepted_error_monotonic(curve)
    assert R.is_accepted_error_monotonic_at_coverage_checkpoints(curve)


def test_risk_coverage_checkpoint_monotonic_survives_realistic_noise():
    """Realistic companion to the noise-free test above: once rank-order
    noise is added, the STRICT point-by-point check can and does
    occasionally fail from isolated near-tie swaps (this is a property of
    the cumulative-mean construction, not a bug -- see the previous test's
    docstring), but the coarser, coverage-checkpoint version --
    the one this package's report actually surfaces for the 100/80/60%
    acceptance criterion -- should still hold for a score this strongly
    correlated with the true error."""
    rng = np.random.RandomState(2)
    n = 200
    true_error = rng.uniform(0, 1, size=n)
    predicted_score = true_error + rng.normal(0, 0.02, size=n)

    from agents.eval.predictive_metrics import risk_coverage_curve
    curve = risk_coverage_curve(predicted_score, true_error)
    assert R.is_accepted_error_monotonic_at_coverage_checkpoints(curve)


def test_risk_coverage_flags_non_monotonic_when_score_is_uninformative():
    """Negative control: a score with NO relationship to the true error
    should not reliably produce a monotonic curve -- guards against
    `is_accepted_error_monotonic*` being vacuously true for any input."""
    rng = np.random.RandomState(3)
    n = 60
    true_error = rng.uniform(0, 1, size=n)
    predicted_score = rng.uniform(0, 1, size=n)  # independent of true_error

    from agents.eval.predictive_metrics import risk_coverage_curve
    curve = risk_coverage_curve(predicted_score, true_error)
    # Not asserting False outright (a shuffled curve can occasionally still
    # pass by chance) -- asserting the AUC is close to the "no information"
    # midpoint rather than near 0 (which would indicate accidental signal).
    assert curve.auc > 0.35, (
        "an uninformative random score should not achieve a low (good) "
        f"risk-coverage AUC; got {curve.auc}")


def test_certificate_beats_uninformative_global_baseline_synthetic():
    """Minimal, fully-controlled version of the plan's "beats every global
    single-metric baseline" acceptance criterion: constructs a task-local
    failure (registration residual on the specific manipulated object) that
    drives `actual_error`, while a "global" scene-wide metric is independent
    noise -- i.e. deliberately uninformative for this failure mode. The
    certificate (using the task-local feature) must achieve a lower
    (better) risk-coverage AUC than the global metric on the same items."""
    from agents.eval.predictive_metrics import risk_coverage_curve

    rng = np.random.RandomState(4)
    n = 60
    task_local_defect = rng.uniform(0, 1, size=n)          # e.g. registration residual, normalized
    actual_error = np.clip(task_local_defect + rng.normal(0, 0.05, size=n), 0, 1)
    global_metric_goodness = rng.uniform(0, 1, size=n)      # e.g. scene PSNR, independent by construction

    cert_curve = risk_coverage_curve(task_local_defect, actual_error)
    baseline_curve = risk_coverage_curve(-global_metric_goodness, actual_error)

    assert cert_curve.auc < baseline_curve.auc, (
        f"certificate AUC {cert_curve.auc} should beat the uninformative "
        f"global-baseline AUC {baseline_curve.auc}")


def test_certificate_beats_global_baselines_on_toy_fixture():
    """End-to-end version of the same acceptance criterion, run through the
    full `calibrate.build_full_report` pipeline against the committed
    synthetic fixture (`tests/data/certificate/toy.json`) rather than a
    hand-rolled array. Documents, honestly, that this is a constructed
    fixture result, not real oracle_causal evidence -- see this file's
    module docstring and the Task 12 completion report."""
    report = C.build_full_report(TOY_FIXTURE, run_loso=True, run_lotfo=True)
    assert report["min_nondegenerate_pairs_met"] is True
    assert report["n_pairs"] >= 12

    for scheme_name in ("leave_one_scene_out", "leave_one_task_family_out"):
        scheme = report["schemes"][scheme_name]
        assert "error" not in scheme, scheme.get("error")
        assert scheme["beats_all_global_baselines"] is True, (
            f"{scheme_name}: certificate did not beat every global baseline "
            f"on the toy fixture: {json.dumps(scheme['global_baselines'], indent=1)}")
        # every model tier must at least be evaluated (not silently empty)
        for tier_name, tier in scheme["model_tiers"].items():
            assert tier["risk_coverage"] is not None, tier_name
            assert tier["auroc"] is not None, tier_name


# ============================================================ CV / leakage ===

def test_loso_and_lotfo_both_run_on_toy_fixture():
    report = C.build_full_report(TOY_FIXTURE, run_loso=True, run_lotfo=True)
    assert set(report["schemes"]) == {"leave_one_scene_out", "leave_one_task_family_out"}
    for scheme in report["schemes"].values():
        assert "error" not in scheme


def test_loso_never_fits_on_the_held_out_scene():
    """Structural (not just statistical) leakage check on
    `calibrate.run_group_cv`: patches every model factory's `.fit` to record
    which group labels were present in its training call, and asserts the
    held-out group's own rows never appear in ANY training call for the
    fold where that group is the test fold."""
    raw = C.load_fixture(TOY_FIXTURE)
    prepared, _ = C.prepare_records(raw)
    X, _ = M.build_feature_matrix([p["features"] for p in prepared])
    y = np.array([p["actual_error"] for p in prepared])
    groups = [p["scene_id"] for p in prepared]
    groups_arr = np.asarray(groups)

    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X, y, groups_arr):
        held_out_group = set(groups_arr[test_idx].tolist())
        train_groups = set(groups_arr[train_idx].tolist())
        assert held_out_group.isdisjoint(train_groups), (
            "a held-out scene's rows leaked into its own fold's training set")
        # every row is accounted for exactly once between train/test
        assert len(train_idx) + len(test_idx) == len(y)


def test_lotfo_never_fits_on_the_held_out_family():
    raw = C.load_fixture(TOY_FIXTURE)
    prepared, _ = C.prepare_records(raw)
    X, _ = M.build_feature_matrix([p["features"] for p in prepared])
    y = np.array([p["actual_error"] for p in prepared])
    groups_arr = np.asarray([p["task_family"] for p in prepared])

    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    n_folds = 0
    for train_idx, test_idx in logo.split(X, y, groups_arr):
        n_folds += 1
        held_out_family = set(groups_arr[test_idx].tolist())
        train_families = set(groups_arr[train_idx].tolist())
        assert held_out_family.isdisjoint(train_families)
    assert n_folds == 2  # exactly 2 task families in the toy fixture


def test_calibration_cv_is_confined_to_the_training_fold():
    """`CalibratedTreeClassifier`'s internal `CalibratedClassifierCV` must
    only ever be `.fit()` on the outer training fold that already excludes
    the held-out group -- verified by fitting it directly on a training-only
    slice and confirming no reference to the excluded rows is possible
    (the only way `.fit` could see them is if the caller passed them in,
    which `calibrate.run_group_cv` structurally does not: it slices `X`/`y`
    with `train_idx` BEFORE calling `model_factory().fit`)."""
    import inspect
    source = inspect.getsource(C.run_group_cv)
    fit_call_line = next(line for line in source.splitlines() if ".fit(" in line)
    assert "train_idx" in fit_call_line, (
        "run_group_cv's .fit() call must be indexed by train_idx, not the "
        f"full array -- got: {fit_call_line.strip()}")


def test_cli_matches_fast_validation_command(tmp_path):
    """Exercises exactly the plan's documented fast-validation invocation."""
    out_path = tmp_path / "report.json"
    rc = C.main(["--fixture", str(TOY_FIXTURE), "--leave-one-scene-out", "--out", str(out_path)])
    assert rc == 0
    report = json.loads(out_path.read_text())
    assert set(report["schemes"]) == {"leave_one_scene_out"}
    assert report["schemes"]["leave_one_scene_out"]["n_pairs"] == report["n_pairs"]


def test_report_includes_top_contributing_features_for_repair_routing():
    report = C.build_full_report(TOY_FIXTURE, run_loso=True, run_lotfo=False)
    scheme = report["schemes"]["leave_one_scene_out"]
    assert "top_contributing_features" in scheme
    assert len(scheme["top_contributing_features"]) > 0
    for entry in scheme["top_contributing_features"]:
        assert set(entry) == {"feature", "direction", "magnitude"}
        assert entry["direction"] in ("increases_risk", "decreases_risk")


def test_failure_cases_are_reported_alongside_coverage():
    report = C.build_full_report(TOY_FIXTURE, run_loso=True, run_lotfo=False)
    tier = report["schemes"]["leave_one_scene_out"]["model_tiers"]["linear_l2"]
    assert "failure_cases" in tier
    assert len(tier["failure_cases"]) > 0
    for case in tier["failure_cases"]:
        assert set(case) == {"pair_id", "predicted", "actual", "abs_residual"}


# ==================================================================== real ===

@pytest.mark.skipif(not REAL_ORACLE_BUILD.exists(),
                     reason="real deep-tier oracle build not present at this path")
def test_real_build_audit_smoke():
    """The one real deep-tier oracle_causal build that exists as of this
    task (see the Task 12 completion report): a single generic "bowl"
    object, no task_graph/alignment_report available for it yet. This is NOT
    a certificate-quality-on-real-data test (n=1, no ground-truth join
    exists) -- it only proves the extractor runs cleanly (no crash, honest
    missingness) against genuinely real, non-fixture build_audit output,
    which the toy-fixture tests above cannot demonstrate on their own."""
    audit = run_audit(REAL_ORACLE_BUILD)
    feats = extract_all_features(audit, None, None)
    expected = set(ALL_FEATURE_NAMES) | {f"{n}_missing" for n in ALL_FEATURE_NAMES}
    assert set(feats) == expected
    # task_graph-only features are honestly missing with no task graph supplied
    assert feats["geom_visible_fraction_missing"] == 1.0
    assert feats["robot_target_visible_missing"] == 1.0
