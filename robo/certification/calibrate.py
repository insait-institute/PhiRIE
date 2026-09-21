"""Task 12 -- leave-one-scene-out / leave-one-task-family-out calibration and
evaluation runner for the task-conditioned twin certificate.

Fast-validation command (per `plan/12_TASK_CONDITIONED_CERTIFICATE.md`):
    python -m robo.certification.calibrate \\
        --fixture tests/data/certificate/toy.json --leave-one-scene-out

Fixture schema (see `tests/data/certificate/toy.json`): a JSON object with a
top-level `"records"` list, one entry per policy-task pair --

    {
      "pair_id": "sceneA__food_bussing__pi05",
      "scene_id": "sceneA",
      "task_id": "food_bussing_demo",
      "task_family": "tabletop_pick_place",
      "policy_id": "pi05_droid_jointpos",
      "build_audit": {...},          # agents.eval.build_audit.run_audit() output
      "task_graph": {...},           # optional, robo.certification.task_graph.build_graph() output
      "alignment_report": {...},     # optional, agents.recon.alignment_report.build_report() output
      "actual_error": 0.12,          # staged-progress-style ground truth error in [0, 1]
      "global_psnr_db": 31.2,               # global baseline 1 (higher is better)
      "global_geometry_f1_at_20mm": 0.55,   # global baseline 2 (higher is better)
      "global_drop_stability_frac": 0.9     # global baseline 3 (higher is better)
    }

`scene_id` and `task_family` are used ONLY as cross-validation group keys by
this module -- they are never passed into
`robo.certification.features.extract_all_features` (which only ever sees
`build_audit`/`task_graph`/`alignment_report`), so they cannot leak into the
feature vector. See `tests/test_certificate.py::test_feature_extraction_is_leakage_free`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from robo.certification import model as M
from robo.certification import report as R
from robo.certification.features import extract_all_features


def load_fixture(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return data["records"]


def prepare_records(raw_records: list[dict], bad_pair_threshold: float | None = None
                     ) -> tuple[list[dict], float]:
    """Extracts features for every record (via
    `features.extract_all_features`, the only function here allowed to see
    `build_audit`/`task_graph`/`alignment_report`) and attaches the
    bookkeeping fields (`scene_id`, `task_family`, `actual_error`,
    global-baseline scores) that the rest of this module needs but the
    feature extractors themselves must never see."""
    threshold = (bad_pair_threshold if bad_pair_threshold is not None
                 else M.load_bad_pair_error_threshold())
    prepared = []
    for r in raw_records:
        feats = extract_all_features(r.get("build_audit"), r.get("task_graph"),
                                      r.get("alignment_report"))
        actual_error = float(r["actual_error"])
        prepared.append({
            "pair_id": r["pair_id"],
            "scene_id": r["scene_id"],
            "task_family": r["task_family"],
            "task_id": r.get("task_id"),
            "policy_id": r.get("policy_id"),
            "features": feats,
            "actual_error": actual_error,
            "bad_pair": float(M.bad_pair_label([actual_error], threshold)[0]),
            "global_psnr_db": r.get("global_psnr_db"),
            "global_geometry_f1_at_20mm": r.get("global_geometry_f1_at_20mm"),
            "global_drop_stability_frac": r.get("global_drop_stability_frac"),
        })
    return prepared, threshold


def run_group_cv(X: np.ndarray, y: np.ndarray, groups: list, model_factory: Callable[[], Any],
                  predict_fn: Callable[[Any, np.ndarray], np.ndarray]) -> np.ndarray:
    """Leave-one-group-out CV: fits a fresh model (via `model_factory()`) on
    every training fold that excludes one whole group, predicts on that
    group, and returns pooled out-of-fold (OOF) predictions aligned to the
    original row order. A model is never fit on data that includes the rows
    it is about to predict -- the structural guarantee behind
    `test_certificate.py::test_loso_and_lotfo_do_not_leak_held_out_scene`."""
    groups_arr = np.asarray(groups)
    unique_groups = sorted(set(groups_arr.tolist()))
    if len(unique_groups) < 2:
        raise ValueError(
            f"leave-one-group-out CV needs >=2 distinct groups, got "
            f"{len(unique_groups)}: {unique_groups}")
    oof = np.full(len(y), np.nan)
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X, y, groups_arr):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = predict_fn(model, X[test_idx])
    return oof


def run_scheme(prepared: list[dict], group_key: str) -> dict[str, Any]:
    """Runs every regression + classification model tier under
    leave-one-<group_key>-out CV. Also fits a `linear_l2` regressor on the
    FULL dataset (no split) purely to report global feature importances
    (`report.global_feature_importance`) -- that fit is never used for any
    OOF prediction/metric, only for the human-readable explanation field."""
    X, cols = M.build_feature_matrix([p["features"] for p in prepared])
    y_err = np.array([p["actual_error"] for p in prepared])
    y_bad = np.array([p["bad_pair"] for p in prepared])
    groups = [p[group_key] for p in prepared]

    tiers: dict[str, dict[str, np.ndarray]] = {}
    for tier_name, factory in M.REGRESSION_MODEL_FACTORIES.items():
        oof = run_group_cv(X, y_err, groups, factory, lambda m, Xte: m.predict(Xte))
        tiers.setdefault(tier_name, {})["error_pred"] = oof
    for tier_name, factory in M.CLASSIFICATION_MODEL_FACTORIES.items():
        oof = run_group_cv(X, y_bad, groups, factory, lambda m, Xte: m.predict_proba(Xte)[:, 1])
        tiers.setdefault(tier_name, {})["bad_pair_prob"] = oof

    full_linear = M.make_linear_regressor()
    full_linear.fit(X, y_err)

    return {"columns": cols, "X": X, "y_err": y_err, "y_bad": y_bad, "tiers": tiers,
            "full_linear_model": full_linear}


def build_scheme_report(prepared: list[dict], scheme_result: dict[str, Any],
                         threshold: float) -> dict[str, Any]:
    pair_ids = [p["pair_id"] for p in prepared]
    y_err, y_bad = scheme_result["y_err"], scheme_result["y_bad"]

    tiers_out: dict[str, Any] = {}
    cert_aucs: list[float] = []
    for tier_name, preds in scheme_result["tiers"].items():
        tier_report = R.model_tier_report(pair_ids, preds.get("error_pred"),
                                           preds.get("bad_pair_prob"), y_err, y_bad)
        tiers_out[tier_name] = tier_report
        rc = tier_report.get("risk_coverage")
        if rc is not None:
            cert_aucs.append(rc["auc"])

    baselines_out: dict[str, Any] = {}
    baseline_aucs: list[float] = []
    for key, label in (("global_psnr_db", "psnr"),
                        ("global_geometry_f1_at_20mm", "geometry_f1"),
                        ("global_drop_stability_frac", "drop_stability")):
        vals = np.array([p[key] if p[key] is not None else np.nan for p in prepared], dtype=float)
        b = R.baseline_report(pair_ids, vals, y_err)
        baselines_out[label] = b
        if b.get("risk_coverage") is not None:
            baseline_aucs.append(b["risk_coverage"]["auc"])

    global_importance = R.global_feature_importance(
        scheme_result["full_linear_model"], scheme_result["columns"], scheme_result["X"])

    return {
        "bad_pair_threshold": threshold,
        "n_pairs": len(prepared),
        "model_tiers": tiers_out,
        "global_baselines": baselines_out,
        "beats_all_global_baselines": R.beats_all_baselines(cert_aucs, baseline_aucs),
        "top_contributing_features": global_importance,
    }


def build_full_report(fixture_path: str | Path, run_loso: bool = True, run_lotfo: bool = True,
                       bad_pair_threshold: float | None = None) -> dict[str, Any]:
    raw = load_fixture(fixture_path)
    prepared, threshold = prepare_records(raw, bad_pair_threshold)

    n_scenes = len({p["scene_id"] for p in prepared})
    n_families = len({p["task_family"] for p in prepared})
    min_pairs_ok = len(prepared) >= 12  # icra_contract_v1.yaml oracle_causal.min_nondegenerate_pairs

    report: dict[str, Any] = {
        "fixture": str(fixture_path),
        "n_pairs": len(prepared),
        "n_scenes": n_scenes,
        "n_task_families": n_families,
        "bad_pair_threshold": threshold,
        "min_nondegenerate_pairs_met": min_pairs_ok,
        "schemes": {},
    }
    if not min_pairs_ok:
        report["note"] = (
            f"Only {len(prepared)} pairs < icra_contract_v1.yaml's "
            "oracle_causal.min_nondegenerate_pairs (12). Per the contract's "
            "stop condition (docs/ICRA_RESEARCH_CONTRACT.md sec 4 / Task "
            "10's stop condition), this report is descriptive only -- do "
            "not quote the numbers below as a pooled correlation claim."
        )

    if run_loso:
        if n_scenes < 2:
            report["schemes"]["leave_one_scene_out"] = {
                "error": f"need >=2 scenes for leave-one-scene-out CV, got {n_scenes}"}
        else:
            scheme = run_scheme(prepared, "scene_id")
            report["schemes"]["leave_one_scene_out"] = build_scheme_report(prepared, scheme, threshold)

    if run_lotfo:
        if n_families < 2:
            report["schemes"]["leave_one_task_family_out"] = {
                "error": f"need >=2 task families for leave-one-task-family-out CV, got {n_families}"}
        else:
            scheme = run_scheme(prepared, "task_family")
            report["schemes"]["leave_one_task_family_out"] = build_scheme_report(prepared, scheme, threshold)

    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", required=True,
                     help="certificate fixture JSON, see tests/data/certificate/toy.json")
    ap.add_argument("--leave-one-scene-out", action="store_true")
    ap.add_argument("--leave-one-task-family-out", action="store_true")
    ap.add_argument("--out", default=None, help="write the full report JSON here")
    ap.add_argument("--bad-pair-threshold", type=float, default=None,
                     help="override the go/no-go bad-pair error threshold "
                          "(default: read from icra_contract_v1.yaml)")
    args = ap.parse_args(argv)

    neither_flag_given = not (args.leave_one_scene_out or args.leave_one_task_family_out)
    run_loso = args.leave_one_scene_out or neither_flag_given
    run_lotfo = args.leave_one_task_family_out or neither_flag_given

    report = build_full_report(args.fixture, run_loso=run_loso, run_lotfo=run_lotfo,
                                bad_pair_threshold=args.bad_pair_threshold)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"[calibrate] wrote {args.out}")
    else:
        print(text)
    print(f"[calibrate] n_pairs={report['n_pairs']} n_scenes={report['n_scenes']} "
          f"n_task_families={report['n_task_families']} "
          f"min_pairs_met={report['min_nondegenerate_pairs_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
