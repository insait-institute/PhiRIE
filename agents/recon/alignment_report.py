"""Reporting & validation layer for agents.recon.robot_align.

Turns an AlignmentResult (fitted Sim(3)/SE(3); a calibration-set residual;
an optional held-out-set residual) into a build go/no-go certificate:

  - `build_report` evaluates a fixed set of HARD thresholds (reasonable
    pipeline-wide defaults in DEFAULT_HARD_THRESHOLDS, overridable per
    scene/mode) that gate whether a build is trusted at all, and a
    separate set of SOFT metrics that are only logged (for a possible
    future per-task certificate) - never used to reject a build on their
    own.
  - The held-out residual (result.held_out), NOT the fitted/calibration
    residual (result.fit), is what the hard thresholds gate on wherever
    both exist; when no held-out set was supplied the report says so
    explicitly (`held_out_available: false`) rather than silently grading
    the fit residual as if it were held-out.
  - `plot_overlay` draws a simple 3-view (top-down/front/side) scatter of
    fitted-vs-target points for calibration and held-out sets, for a fast
    visual sanity check alongside the numeric report.

CPU, numpy + matplotlib only (matplotlib import is lazy/optional - see
plot_overlay). Usage as a library only; see agents/recon/robot_align.py's
CLI for the end-to-end config -> AlignmentResult -> report pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------- thresholds ---

# HARD: below/above these, alignment_report marks overall_pass=False and
# (via robot_align.main()'s exit code) a build should be REJECTED. Pulled
# from the pipeline's own existing tolerances wherever one already exists
# (see THRESHOLD_RATIONALE), not invented fresh, so this layer agrees with
# the rest of the pipeline instead of introducing a second opinion.
DEFAULT_HARD_THRESHOLDS = {
    "min_correspondences": 3,
    "held_out_rms_m": 0.05,
    "held_out_median_m": 0.03,
    "reprojection_residual_px": 5.0,
    "floor_normal_error_deg": 10.0,
    "table_height_error_m": 0.03,
    "scale_ci95_halfwidth_pct": 3.0,
    "fk_center_rms_m": 0.10,
}

THRESHOLD_RATIONALE = {
    "min_correspondences": (
        "below this a Sim(3)/SE(3) fit has too few constraints to be "
        "trusted regardless of residual - it is a degeneracy, not a "
        "precision, problem."),
    "held_out_rms_m": (
        "5 cm: the phone-room metric-scale budget used elsewhere in the "
        "pipeline (matches metricize.py's floor-sanity tolerance order of "
        "magnitude); above this, object placements/reachability computed "
        "in the aligned frame are not trustworthy."),
    "held_out_median_m": (
        "half the RMS bound; catches a fit dominated by one or two bad "
        "markers even when RMS alone would still pass."),
    "reprojection_residual_px": (
        "matches agents/recon/align_to_traj.py's empirical good-fit band "
        "(see its FK-vs-COLMAP reprojection cross-check); a large "
        "residual usually means a wrong frame offset/convention, not just "
        "noise."),
    "floor_normal_error_deg": (
        "matches agents/recon/metricize.py's UP_CONE_DEG search prior; a "
        "larger deviation means the 'floor' plane found is probably a "
        "wall or table, not the floor."),
    "table_height_error_m": (
        "3 cm: reachability/grasp-height validity for tabletop "
        "manipulation is sensitive at roughly this scale."),
    "scale_ci95_halfwidth_pct": (
        "if the bootstrap CI on scale is wider than this, the reported "
        "metric sizes are not precise enough to certify even if the "
        "point estimate itself looks fine."),
    "fk_center_rms_m": (
        "mode 4 (droid_fk) only; mirrors align_to_traj.py's own "
        "--max-rms-m abort threshold so this layer agrees with it."),
}

# SOFT: logged into the report for the future per-task certificate, never
# gates a build. Listed here (rather than only implicitly, by what
# build_report happens to copy) so the schema is auditable in one place.
SOFT_METRIC_KEYS = [
    "condition_number_src", "condition_number_dst",
    "fit_rms_m", "fit_median_m",
    "n_ransac_inliers", "ransac_iters_used", "huber_final_weights_min",
    "rotation_residual_deg", "reproj_median_px", "cloud_z_mode_m",
    "frame_offset",
]


def _rotation_angle_from_identity_deg(R) -> float:
    c = (np.trace(np.asarray(R)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _check(hard: dict, name: str, value, cmp, limit) -> None:
    """Record one hard-threshold evaluation. value=None means "not
    computed for this mode/config" - recorded with pass=None (neither a
    pass nor a fail) rather than silently defaulting to pass or fail."""
    if value is None:
        hard[name] = {"value": None, "limit": limit, "pass": None,
                     "note": "not computed for this mode/config"}
        return
    hard[name] = {"value": value, "limit": limit, "pass": bool(cmp(value, limit))}


def build_report(result, *, thresholds: dict | None = None,
                  extra_checks: dict | None = None,
                  scene_id: str | None = None,
                  notes: list | None = None) -> dict:
    """result: an agents.recon.robot_align.AlignmentResult.

    extra_checks (optional): {"floor_normal_error_deg": float,
    "table_height_error_m": float} - these two come from metricize.py's
    floor-fit / a separate table-height measurement, not from robot_align
    itself (robot_align fits a transform; it does not re-derive a floor
    plane or a table height), so a caller that has them passes them in
    here to fold them into the same certificate.
    """
    thresholds = {**DEFAULT_HARD_THRESHOLDS, **(thresholds or {})}
    extra_checks = extra_checks or {}
    notes = list(notes or [])

    s, R, t = result.s, np.asarray(result.R), np.asarray(result.t)
    report: dict = {
        "mode": result.mode,
        "scene_id": scene_id,
        "scale": {"value": s,
                 "ci95": list(result.scale_ci95) if result.scale_ci95 else None},
        "rotation": {"matrix": R.tolist(),
                    "angle_from_identity_deg": _rotation_angle_from_identity_deg(R)},
        "translation_m": t.tolist(),
        "T": np.asarray(result.T).tolist(),
        "correspondences": {"n_total": result.n_total, "n_used": result.n_used},
        "fit": result.fit,
        "held_out": result.held_out,
        "degeneracy": result.degeneracy,
        "extra": result.extra,
        "notes": notes,
    }

    hard: dict = {}
    _check(hard, "min_correspondences", result.n_used,
          lambda v, l: v >= l, thresholds["min_correspondences"])

    held = result.held_out or {}
    held_available = bool(result.held_out)
    _check(hard, "held_out_rms_m", held.get("rms_m") if held_available else None,
          lambda v, l: v <= l, thresholds["held_out_rms_m"])
    _check(hard, "held_out_median_m", held.get("median_m") if held_available else None,
          lambda v, l: v <= l, thresholds["held_out_median_m"])

    reproj = (result.extra or {}).get("reproj_median_px")
    _check(hard, "reprojection_residual_px", reproj,
          lambda v, l: v <= l, thresholds["reprojection_residual_px"])

    _check(hard, "floor_normal_error_deg", extra_checks.get("floor_normal_error_deg"),
          lambda v, l: v <= l, thresholds["floor_normal_error_deg"])
    _check(hard, "table_height_error_m", extra_checks.get("table_height_error_m"),
          lambda v, l: v <= l, thresholds["table_height_error_m"])

    ci = result.scale_ci95
    ci_halfwidth_pct = None
    if ci:
        ci_halfwidth_pct = (ci[1] - ci[0]) / 2.0 / max(abs(s), 1e-9) * 100.0
    _check(hard, "scale_ci95_halfwidth_pct", ci_halfwidth_pct,
          lambda v, l: v <= l, thresholds["scale_ci95_halfwidth_pct"])

    if result.mode == "droid_fk":
        _check(hard, "fk_center_rms_m", (result.fit or {}).get("rms_m"),
              lambda v, l: v <= l, thresholds["fk_center_rms_m"])

    evaluated = [v for v in hard.values() if v["pass"] is not None]
    overall_pass = bool(evaluated) and all(v["pass"] for v in evaluated)

    soft: dict = {}
    deg = result.degeneracy or {}
    if isinstance(deg, dict):
        for side, d in deg.items():
            if isinstance(d, dict) and "condition_number" in d:
                soft[f"condition_number_{side}"] = d["condition_number"]
    if isinstance(result.fit, dict):
        for k in ("rms_m", "median_m"):
            if k in result.fit:
                soft[f"fit_{k}"] = result.fit[k]
    for k in ("n_ransac_inliers", "ransac_iters_used", "huber_final_weights_min",
             "rotation_residual_deg", "reproj_median_px", "cloud_z_mode_m",
             "frame_offset"):
        if k in (result.extra or {}):
            soft[k] = result.extra[k]

    report["hard_thresholds"] = hard
    report["soft_metrics"] = soft
    report["overall_pass"] = overall_pass
    report["held_out_available"] = held_available

    if overall_pass and held_available:
        report["recommendation"] = "ACCEPT"
    elif overall_pass and not held_available:
        report["recommendation"] = "ACCEPT_NO_HELDOUT_EVIDENCE"
    else:
        report["recommendation"] = "REJECT"

    if not held_available:
        report["notes"].append(
            "No held-out evaluation set was provided for this alignment - "
            "the 'fit' residual above was computed on the SAME data used "
            "to fit the transform and is optimistic. Do not certify a "
            "build on the fit residual alone; supply a held-out set.")
    if result.mode == "droid_fk":
        report["notes"].append(
            "mode=droid_fk: align_to_traj.py fits ALL FK-vs-SfM frame "
            "correspondences and does not hold out an evaluation subset "
            "itself (see extra.note); fk_center_rms_m below is a FIT "
            "residual gated at align_to_traj.py's own --max-rms-m level, "
            "not independent held-out evidence.")
    return report


def write_report(report: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(report, indent=1, default=default))
    return path


def plot_overlay(path, *, calib_dst=None, calib_pred=None,
                  held_dst=None, held_pred=None, title=None):
    """Simple top-down (XY) + front (XZ) + side (YZ) scatter of fitted vs.
    target points, for calibration and (if given) held-out sets. Returns
    the written path, or None (with a log line) if matplotlib is missing
    in the current env - this is the "nice to have" overlay the task
    allows stubbing, implemented in full since matplotlib is available in
    this repo's .venv."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[alignment_report] matplotlib unavailable - skipping "
             f"overlay plot ({path}). TODO: pip install matplotlib to "
             f"enable this.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [(0, 1, "top-down (XY)"), (0, 2, "front (XZ)"), (1, 2, "side (YZ)")]
    any_data = False
    for ax, (i, j, name) in zip(axes, views):
        if calib_dst is not None and len(calib_dst):
            any_data = True
            ax.scatter(np.asarray(calib_dst)[:, i], np.asarray(calib_dst)[:, j],
                      c="#2b6cb0", marker="o", s=34, label="calib target")
        if calib_pred is not None and len(calib_pred):
            ax.scatter(np.asarray(calib_pred)[:, i], np.asarray(calib_pred)[:, j],
                      c="#e53e3e", marker="x", s=34, label="calib fitted")
        if held_dst is not None and len(held_dst):
            any_data = True
            ax.scatter(np.asarray(held_dst)[:, i], np.asarray(held_dst)[:, j],
                      facecolors="none", edgecolors="#38a169", marker="o",
                      s=50, label="held-out target")
        if held_pred is not None and len(held_pred):
            ax.scatter(np.asarray(held_pred)[:, i], np.asarray(held_pred)[:, j],
                      c="#d69e2e", marker="x", s=50, label="held-out fitted")
        ax.set_title(name)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    if any_data:
        axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(title or "alignment overlay: fitted vs. target")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[alignment_report] overlay -> {path}")
    return path
