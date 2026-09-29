"""Unified robot/metric-scale alignment: T_base<-scan plus scale, with
uncertainty and a held-out residual, across four modes:

  1. fiducial   - ChArUco/fiducial correspondences (scan frame <-> known
                  robot/world positions), robust weighted Sim(3)/SE(3).
  2. surveyed   - phone depth/AR odometry + surveyed robot-base points;
                  same solver as (1), kept as a separate entry point so the
                  report/config carries the right semantic label.
  3. dimensions - known robot/table dimensions used as scale constraints
                  (optionally plus a handful of reference correspondences
                  for rotation/translation once scale is fixed).
  4. droid_fk   - DROID FK trajectory vs. reconstructed wrist-camera
                  centers. This mode does NOT reimplement the Umeyama fit:
                  agents/recon/align_to_traj.py already does exactly this
                  (frame-offset search + --max-rms-m abort) and is being
                  actively maintained by someone else right now, so it is
                  treated as READ-ONLY here. align_droid_fk() below invokes
                  it as a subprocess (its public CLI contract) and parses
                  its align_report.json defensively (.get() with fallbacks)
                  so a concurrent change to its internal field names
                  degrades gracefully instead of crashing this wrapper.

All four modes return the same AlignmentResult, so a caller (or the CLI at
the bottom of this file) can build one alignment_report.py certificate
regardless of mode. See agents/recon/alignment_report.py for the hard
build-rejection thresholds vs. soft certificate metrics.

Convention (matches agents/core/common.py's per-object registration
convention exactly): a fitted transform is a single 4x4 matrix T with the
isotropic scale folded into the upper-left 3x3, i.e.

    x_dst = T[:3, :3] @ x_src + T[:3, 3]      (T[:3, :3] == s * R)

`transform_points` applies this directly. Rigid POSES (camera c2w, robot
link frames, physics body frames - anything with an orientation that must
stay a pure rotation, never scaled) use `transform_rigid_pose`/
`transform_w2c` instead, which decompose T into (s, R, t) via
agents.core.common.decompose_similarity and apply the scale only to the
position, never to the orientation - see the module-level derivation in the
docstrings of those two functions if the algebra looks surprising.

CPU, numpy(+scipy-free)+PyYAML+matplotlib only. Usage:
    python -m agents.recon.robot_align --config configs/calibration/<f>.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents.core.common import decompose_similarity

_REPO_ROOT = Path(__file__).resolve().parents[2]


class AlignmentDegeneracyError(Exception):
    """Raised instead of returning a transform when the input geometry
    cannot possibly constrain a stable Sim(3)/SE(3) fit (collinear/
    coincident markers, an insufficient-baseline trajectory, too few
    points, all-zero weights, ...). Never silently swallowed by the
    solvers in this module - only bootstrap_scale_ci catches it, and only
    to skip a single degenerate resample."""


# --------------------------------------------------------------- geometry --

def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, degrees. Deliberately
    reimplemented (not imported from align_to_traj.py's rot_angle_deg) to
    keep this module decoupled from a file another agent is concurrently
    editing; the formula is standard and stable."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _make_T(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = s * np.asarray(R)
    T[:3, 3] = np.asarray(t)
    return T


def _triangle_degenerate(pts3: np.ndarray, min_area_m2: float = 1e-8) -> bool:
    a, b, c = pts3
    area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
    return area < min_area_m2


# ----------------------------------------------------------- degeneracy ---

def check_point_set_degeneracy(pts, *, min_points: int = 4,
                                max_condition_number: float = 1e4,
                                min_extent_m: float = 0.01,
                                what: str = "correspondence",
                                raise_on_fail: bool = True) -> dict:
    """Condition-number / rank check on a 3D point set.

    A stable Sim(3)/SE(3) fit needs the point set to span all 3 axes with
    real spread. Default min_points is 4, not 3: ANY 3 points are exactly
    coplanar (rank <= 2, smallest singular value == 0 up to float noise)
    no matter how they are chosen, so a "does this set span 3D" check can
    only ever pass at n>=4. RANSAC's own minimal-sample fit (3 points,
    exact Umeyama) does NOT go through this function - see
    `_triangle_degenerate` - precisely to sidestep this trap; this
    function guards the OVERALL calibration/evaluation set, where real
    markers/survey points are expected to have genuine 3D extent.
    Collinear markers (e.g. a ChArUco board seen edge-on, or survey points
    along one wall) leave the fit's out-of-plane / cross-axis rotation
    almost entirely unconstrained, so the residual can look fine while the
    transform is actually garbage in directions the data never tested.
    Two independent failure shapes are checked: an (near-)exact rank
    deficiency (smallest principal spread below `min_extent_m`) and a
    merely ill-conditioned spread (large aspect ratio,
    `max_condition_number`) that is technically full-rank but numerically
    unstable to fit against noise.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < min_points:
        msg = (f"{what} set has only {len(pts)} point(s) (need >= "
               f"{min_points}) - cannot fit a stable Sim(3)/SE(3) transform")
        if raise_on_fail:
            raise AlignmentDegeneracyError(msg)
        return {"degenerate": True, "reason": msg, "n_points": int(len(pts))}

    c = pts - pts.mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-12))
    diag = {"singular_values_m": sv.tolist(), "condition_number": cond,
            "n_points": int(len(pts))}

    if sv[-1] < min_extent_m:
        msg = (f"{what} set is degenerate: smallest principal spread "
               f"{sv[-1] * 1000:.2f} mm < {min_extent_m * 1000:.1f} mm - "
               f"points are collinear/coincident along at least one axis "
               f"(singular values {np.round(sv, 4).tolist()} m)")
        if raise_on_fail:
            raise AlignmentDegeneracyError(msg)
        diag.update(degenerate=True, reason=msg)
        return diag

    if cond > max_condition_number:
        msg = (f"{what} set is ill-conditioned: condition number "
               f"{cond:.1e} > {max_condition_number:.1e} (singular values "
               f"{np.round(sv, 4).tolist()} m) - fit would be unstable "
               f"under noise")
        if raise_on_fail:
            raise AlignmentDegeneracyError(msg)
        diag.update(degenerate=True, reason=msg)
        return diag

    diag["degenerate"] = False
    return diag


def check_trajectory_baseline(pts, *, min_points: int = 10,
                               min_extent_m: float = 0.15,
                               raise_on_fail: bool = True) -> dict:
    """Bounding-box-diagonal spread check for a trajectory-derived point
    set (e.g. DROID FK / SfM camera centers). A short trajectory (the arm
    barely moved, or only a handful of frames were kept) constrains a
    Sim(3) fit about as well as a single point: the residual can be tiny
    simply because there is nothing to disagree with."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < min_points:
        msg = f"trajectory has only {len(pts)} point(s) (need >= {min_points})"
        if raise_on_fail:
            raise AlignmentDegeneracyError(msg)
        return {"degenerate": True, "reason": msg, "n_points": int(len(pts))}

    extent = pts.max(0) - pts.min(0)
    span = float(np.linalg.norm(extent))
    diag = {"extent_m": extent.tolist(), "span_m": span, "n_points": int(len(pts))}
    if span < min_extent_m:
        msg = (f"trajectory baseline too short: {span * 100:.1f} cm span "
               f"(need >= {min_extent_m * 100:.0f} cm) - insufficient "
               f"camera/end-effector motion for a stable Sim(3) fit")
        if raise_on_fail:
            raise AlignmentDegeneracyError(msg)
        diag.update(degenerate=True, reason=msg)
        return diag
    diag["degenerate"] = False
    return diag


# -------------------------------------------------------- point transforms

def transform_points(T, pts):
    """pts (N,3) in the src/world/scan frame -> dst frame. T[:3,:3] is
    assumed to already be s*R (this module's folded-scale convention)."""
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]


def invert_transform(T):
    """Sim(3)/SE(3) inverse. Because T is a genuine 4x4 with scale folded
    into the upper-left 3x3 (det(T[:3,:3]) = s**3), a plain 4x4 matrix
    inverse IS the correct Sim(3) inverse - no special-casing needed, and
    this is exactly what makes the forward/inverse round trip below exact
    to floating-point precision rather than only to fit tolerance."""
    return np.linalg.inv(np.asarray(T, dtype=np.float64))


def transform_rigid_pose(T, M):
    """M is a RIGID 4x4 pose (camera c2w, a robot link/physics body frame,
    an object's position+orientation with NO scale in its rotation block).

    Similarity transforms must not scale a rotation matrix (it would stop
    being orthonormal), so unlike transform_points this decomposes T into
    (s, R, t) and applies the "point rule" (s*R@x+t) to the pose's
    POSITION only, while composing the pose's ORIENTATION with the pure
    rotation R alone:
        M_new[:3,:3] = R @ M[:3,:3]
        M_new[:3,3]  = s * R @ M[:3,3] + t
    (Verified against agents/recon/align_to_traj.py's own w2c-update
    algebra - `transform_w2c` below reduces to exactly its
    `w2c_new = w2c; w2c_new[:,:3,3] *= s; w2c_new = w2c_new @ inv(T_rig)`
    recipe when going through c2w and back.)
    """
    s, R, t = decompose_similarity(T)
    M = np.asarray(M, dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = R @ M[:3, :3]
    out[:3, 3] = s * (R @ M[:3, 3]) + t
    return out


def transform_w2c(T, w2c):
    """World->camera extrinsics: transform via c2w (a rigid pose) so the
    rotation block never gets scaled, then invert back."""
    c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    c2w_new = transform_rigid_pose(T, c2w)
    return np.linalg.inv(c2w_new)


def transform_folded_pose(T, P):
    """P is a scale-folded object-registration pose (TRELLIS canonical->
    world convention documented in agents/core/common.py: x_world =
    s*R@x_canonical+t, stored as one 4x4 with scale folded into the
    rotation block - exactly this module's own T convention). Composing
    two such matrices with a plain 4x4 product is valid here (unlike rigid
    poses) because the product of two "scalar * orthogonal" matrices is
    again "scalar * orthogonal": T_new = T_align @ P."""
    return np.asarray(T, dtype=np.float64) @ np.asarray(P, dtype=np.float64)


def transform_length(T, x):
    """Isotropic length quantities (gaussian per-axis scales, physics
    collision extents/radii, object bounding-box sizes) scale by s only."""
    s, _, _ = decompose_similarity(T)
    return np.asarray(x, dtype=np.float64) * s


def transform_recon_dict(rec: dict, T) -> dict:
    """Apply T to an agents/recon/*.py recon.npz-schema dict (w2c, points,
    depth - the exact keys agents/recon/align_to_traj.py and
    agents/recon/metricize.py both read/write), consistently: points as
    points, depth as a camera-local length (scale only, never rotated -
    see those two files' own `depth = depth * s`), w2c as rigid camera
    extrinsics via transform_w2c. Pure numpy; splat gaussians go through
    agents.core.common.transform_gaussians (torch-based) using the same s/R/t
    this function derives from T via decompose_similarity - not duplicated
    here to avoid a hard torch/cuda dependency in this module's tests."""
    s, _, _ = decompose_similarity(T)
    out = dict(rec)
    if "points" in out and out["points"] is not None and len(out["points"]):
        out["points"] = transform_points(T, out["points"]).astype(np.float32)
    if "w2c" in out and out["w2c"] is not None and len(out["w2c"]):
        out["w2c"] = np.stack([transform_w2c(T, m) for m in out["w2c"]])
    if "depth" in out and out["depth"] is not None:
        dtype = np.asarray(out["depth"]).dtype
        out["depth"] = (np.asarray(out["depth"], dtype=np.float64) * s).astype(dtype)
    return out


# ------------------------------------------------------------- residuals --

def evaluate_correspondence_residual(T, src, dst, weights=None) -> dict:
    """Apply T to src and compare against dst. Used for BOTH the fitted
    (calibration) residual and the held-out residual - callers must make
    sure `src`/`dst` here are whichever set they mean to report."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    pred = transform_points(T, src)
    res = np.linalg.norm(pred - dst, axis=1)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(res) == 0:
        return {"n": 0, "rms_m": None, "median_m": None, "max_m": None,
                "per_point_m": []}
    rms = float(np.sqrt(np.average(res ** 2, weights=w))) if w.sum() > 0 else \
        float(np.sqrt(np.mean(res ** 2)))
    return {"n": int(len(src)), "rms_m": rms, "median_m": float(np.median(res)),
            "max_m": float(res.max()), "per_point_m": res.tolist()}


# --------------------------------------------------------- core Sim(3) fit

def weighted_umeyama(src, dst, weights=None, *, allow_scale: bool = True):
    """Weighted generalization of agents/recon/align_to_traj.py's `umeyama`
    (reduces to it exactly at uniform weights). `allow_scale=False` gives a
    weighted Kabsch/SE(3) fit
    (s fixed to 1), used by mode 3 once scale has already been pinned down
    by a known dimension."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    wsum = w.sum()
    if wsum <= 0:
        raise AlignmentDegeneracyError(
            "all correspondence weights are zero/negative - nothing to fit")
    wn = w / wsum
    mu_s = (wn[:, None] * src).sum(0)
    mu_d = (wn[:, None] * dst).sum(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (xd * wn[:, None]).T @ xs
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    if allow_scale:
        var_s = max(float((wn[:, None] * xs ** 2).sum()), 1e-12)
        s = float(np.trace(np.diag(D) @ S) / var_s)
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def huber_irls_sim3(src, dst, weights=None, *, allow_scale: bool = True,
                     huber_delta_m: float = 0.02, max_iter: int = 25,
                     tol: float = 1e-9):
    """Iteratively-reweighted weighted_umeyama with a Huber influence
    function: correspondences with residual <= huber_delta_m keep full
    weight, larger ones are downweighted as huber_delta_m / residual. This
    is what makes the "robust" path tolerate MODERATE noise/outliers even
    without RANSAC; RANSAC (in fit_sim3_robust) additionally handles GROSS
    outliers where IRLS alone could still be dragged off by a bad start."""
    n = len(src)
    prior_w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    w = prior_w / max(prior_w.mean(), 1e-12)
    s, R, t = weighted_umeyama(src, dst, w, allow_scale=allow_scale)
    for _ in range(max_iter):
        pred = np.asarray(src, dtype=np.float64) @ (s * R).T + t
        res = np.linalg.norm(np.asarray(dst, dtype=np.float64) - pred, axis=1)
        res = np.maximum(res, 1e-9)
        huber_w = np.minimum(1.0, huber_delta_m / res)
        w_new = huber_w * prior_w
        if w_new.sum() < 1e-9:
            break
        w_new = w_new / w_new.mean()
        s2, R2, t2 = weighted_umeyama(src, dst, w_new, allow_scale=allow_scale)
        delta = abs(s2 - s) + float(np.linalg.norm(t2 - t)) + rotation_angle_deg(R, R2)
        s, R, t, w = s2, R2, t2, w_new
        if delta < tol:
            break
    return s, R, t, w


def fit_sim3_robust(src, dst, weights=None, *, allow_scale: bool = True,
                     robust: bool = True, ransac: bool = True,
                     huber_delta_m: float = 0.02, ransac_iters: int = 500,
                     ransac_inlier_thresh_m: float = 0.03,
                     min_inliers: int = 3, min_points: int = 4,
                     rng=None, check_degeneracy: bool = True,
                     max_condition_number: float = 1e4,
                     min_extent_m: float = 0.01) -> dict:
    """The mode-1/2 (and dimensions-mode rigid-refit) workhorse: degeneracy
    check, then optional RANSAC (gross-outlier rejection) into optional
    Huber IRLS (moderate-noise robustness). `robust=False` disables BOTH
    (a single plain weighted_umeyama on 100% of the data) - this is the
    "naive" baseline the robust path is compared against under outliers."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n != len(dst):
        raise ValueError(f"src/dst length mismatch ({n} vs {len(dst)})")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)

    degeneracy: dict[str, Any] = {}
    if check_degeneracy:
        degeneracy["src"] = check_point_set_degeneracy(
            src, min_points=min_points, max_condition_number=max_condition_number,
            min_extent_m=min_extent_m, what="correspondence(src)")
        degeneracy["dst"] = check_point_set_degeneracy(
            dst, min_points=min_points, max_condition_number=max_condition_number,
            min_extent_m=min_extent_m, what="correspondence(dst)")

    extra: dict[str, Any] = {"robust": bool(robust)}
    if not robust:
        s, R, t = weighted_umeyama(src, dst, w, allow_scale=allow_scale)
        inlier_mask = np.ones(n, dtype=bool)
        extra["ransac"] = False
    else:
        inlier_mask = np.ones(n, dtype=bool)
        extra["ransac"] = bool(ransac)
        if ransac and n > min_inliers:
            rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
            best = None
            for _ in range(ransac_iters):
                idx = rng.choice(n, size=3, replace=False)
                if _triangle_degenerate(src[idx]):
                    continue
                try:
                    s0, R0, t0 = weighted_umeyama(src[idx], dst[idx], None,
                                                  allow_scale=allow_scale)
                except AlignmentDegeneracyError:
                    continue
                pred = transform_points(_make_T(s0, R0, t0), src)
                res = np.linalg.norm(pred - dst, axis=1)
                mask = res < ransac_inlier_thresh_m
                score = int(mask.sum())
                cost = float(res[mask].sum()) if score else float("inf")
                if best is None or score > best[0] or \
                        (score == best[0] and cost < best[1]):
                    best = (score, cost, mask)
            if best is not None and best[0] >= min_inliers:
                inlier_mask = best[2]
                extra["ransac_inliers"] = int(best[0])
                extra["ransac_iters_used"] = ransac_iters
        if check_degeneracy and inlier_mask.sum() < n:
            # RANSAC narrowed the correspondence set; re-check the SURVIVING
            # inliers specifically, since a full set that passed the check
            # above can still narrow down to a degenerate (too-few/coplanar)
            # inlier subset - never hand that silently to the Huber refit.
            degeneracy["src_ransac_inliers"] = check_point_set_degeneracy(
                src[inlier_mask], min_points=min_points,
                max_condition_number=max_condition_number,
                min_extent_m=min_extent_m, what="correspondence(ransac inliers)")
        s, R, t, final_w = huber_irls_sim3(
            src[inlier_mask], dst[inlier_mask],
            w[inlier_mask] if weights is not None else None,
            allow_scale=allow_scale, huber_delta_m=huber_delta_m)
        extra["huber_delta_m"] = huber_delta_m
        extra["huber_final_weights_min"] = float(final_w.min()) if len(final_w) else None

    T = _make_T(s, R, t)
    fit = evaluate_correspondence_residual(T, src[inlier_mask], dst[inlier_mask],
                                           w[inlier_mask])
    return {"s": s, "R": R, "t": t, "T": T, "inlier_mask": inlier_mask,
            "n_used": int(inlier_mask.sum()), "n_total": n, "fit": fit,
            "degeneracy": degeneracy, "extra": extra}


def bootstrap_scale_ci(src, dst, weights=None, *, allow_scale: bool = True,
                        n_boot: int = 200, rng=None, ci: float = 0.95):
    """Percentile bootstrap CI on scale (no closed-form CI for the
    Huber/RANSAC pipeline, so resample-and-refit is the simplest honest
    option). Returns None if there are too few points or too few bootstrap
    draws converged (rather than a misleadingly tight/absent CI)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 4:
        return None
    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    scales = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            check_point_set_degeneracy(src[idx], min_points=3,
                                       max_condition_number=1e6,
                                       min_extent_m=1e-4, what="bootstrap",
                                       raise_on_fail=True)
        except AlignmentDegeneracyError:
            continue
        w = None if weights is None else np.asarray(weights, dtype=np.float64)[idx]
        try:
            s, _, _ = weighted_umeyama(src[idx], dst[idx], w, allow_scale=allow_scale)
        except AlignmentDegeneracyError:
            continue
        scales.append(s)
    if len(scales) < max(10, n_boot // 4):
        return None
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return (float(np.percentile(scales, lo_pct)), float(np.percentile(scales, hi_pct)))


# ------------------------------------------------------ dimension (mode 3)

def fit_scale_from_dimensions(pairs, *, robust: bool = True,
                               huber_delta_pct: float = 3.0, max_iter: int = 25):
    """pairs: iterable of (p1(3,), p2(3,), known_length_m). Each pair's
    known/observed ratio is a noisy observation of the SAME scalar scale;
    fit it as a robust 1-D location estimate (median start, then Huber
    IRLS) so a bent tape-measurement or a mislabeled edge doesn't wreck the
    whole scale estimate."""
    p1 = np.array([p[0] for p in pairs], dtype=np.float64)
    p2 = np.array([p[1] for p in pairs], dtype=np.float64)
    known = np.array([p[2] for p in pairs], dtype=np.float64)
    observed = np.linalg.norm(p2 - p1, axis=1)
    if np.any(observed < 1e-6):
        raise AlignmentDegeneracyError(
            "a dimension constraint has (near-)zero observed length in the "
            "scan frame - the two endpoints coincide, cannot form a scale ratio")
    ratio = known / observed
    s = float(np.median(ratio))
    w = np.ones(len(ratio))
    if robust and len(ratio) >= 3:
        for _ in range(max_iter):
            resid_pct = np.abs(ratio / s - 1.0) * 100.0
            w_new = np.minimum(1.0, huber_delta_pct / np.maximum(resid_pct, 1e-6))
            s_new = float(np.average(ratio, weights=w_new))
            if abs(s_new - s) < 1e-9:
                s, w = s_new, w_new
                break
            s, w = s_new, w_new
    inlier_mask = w > 0.5
    resid_pct_final = (np.abs(ratio / s - 1.0) * 100.0).tolist()
    return {"scale": s, "ratio": ratio.tolist(), "weights": w.tolist(),
            "inlier_mask": inlier_mask, "residual_pct": resid_pct_final,
            "n_used": int(inlier_mask.sum()), "n_total": int(len(ratio))}


# ------------------------------------------------------- correspondences --

def _corr_arrays(items):
    """List[{"src":[x,y,z], "dst":[x,y,z], "weight"?, "label"?}] -> arrays.
    Returns (None, None, None, None) for empty/None input (a valid, and
    common, "no held-out set provided" case)."""
    if not items:
        return None, None, None, None
    src = np.array([it["src"] for it in items], dtype=np.float64)
    dst = np.array([it["dst"] for it in items], dtype=np.float64)
    w = np.array([it.get("weight", 1.0) for it in items], dtype=np.float64)
    labels = [it.get("label", "") for it in items]
    return src, dst, w, labels


# ------------------------------------------------------------ the result --

@dataclass
class AlignmentResult:
    mode: str
    s: float
    R: np.ndarray
    t: np.ndarray
    T: np.ndarray                       # 4x4, x_dst = T @ x_src (scale folded in)
    n_total: int | None
    n_used: int | None
    inlier_mask: np.ndarray | None
    fit: dict                           # residual on the CALIBRATION set (optimistic)
    held_out: dict | None               # residual on data never used to fit; None if absent
    scale_ci95: tuple | None
    degeneracy: dict
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------- mode 1 & 2

def _align_correspondences(mode, calib, held_out=None, *, allow_scale: bool = True,
                            robust: bool = True, ransac: bool = True,
                            huber_delta_m: float = 0.02, ransac_iters: int = 500,
                            ransac_inlier_thresh_m: float = 0.03,
                            min_inliers: int = 3, min_points: int = 4,
                            rng=None, bootstrap_ci: bool = True, n_boot: int = 200,
                            max_condition_number: float = 1e4,
                            min_extent_m: float = 0.01) -> AlignmentResult:
    src, dst, w, _labels = _corr_arrays(calib)
    if src is None:
        raise AlignmentDegeneracyError(
            f"mode={mode!r}: no calibration correspondences given - nothing to fit")
    fit = fit_sim3_robust(src, dst, w, allow_scale=allow_scale, robust=robust,
                          ransac=ransac, huber_delta_m=huber_delta_m,
                          ransac_iters=ransac_iters,
                          ransac_inlier_thresh_m=ransac_inlier_thresh_m,
                          min_inliers=min_inliers, min_points=min_points, rng=rng,
                          max_condition_number=max_condition_number,
                          min_extent_m=min_extent_m)
    ci = None
    if bootstrap_ci:
        m = fit["inlier_mask"]
        ci = bootstrap_scale_ci(src[m], dst[m], w[m] if w is not None else None,
                                allow_scale=allow_scale, n_boot=n_boot, rng=rng)
    held = None
    if held_out:
        hsrc, hdst, hw, hlabels = _corr_arrays(held_out)
        held = evaluate_correspondence_residual(fit["T"], hsrc, hdst, hw)
        held["labels"] = hlabels
    return AlignmentResult(mode=mode, s=fit["s"], R=fit["R"], t=fit["t"], T=fit["T"],
                           n_total=fit["n_total"], n_used=fit["n_used"],
                           inlier_mask=fit["inlier_mask"], fit=fit["fit"],
                           held_out=held, scale_ci95=ci,
                           degeneracy=fit["degeneracy"], extra=fit["extra"])


def align_fiducial(calib, held_out=None, **kwargs) -> AlignmentResult:
    """Mode 1: ChArUco/fiducial correspondences, scan frame <-> known
    robot/world positions. `calib`/`held_out`: list of
    {"src":[x,y,z], "dst":[x,y,z], "weight"?, "label"?}."""
    return _align_correspondences("fiducial", calib, held_out, **kwargs)


def align_surveyed(calib, held_out=None, **kwargs) -> AlignmentResult:
    """Mode 2: phone depth/AR-odometry points <-> surveyed robot-base
    points. Same solver as align_fiducial; kept separate for the report's
    semantic label and so a caller can pick different solver defaults
    (e.g. allow_scale=False if the AR odometry is already trusted metric)."""
    return _align_correspondences("surveyed_points", calib, held_out, **kwargs)


# ------------------------------------------------------------- mode 3 ----

def align_dimensions(dimension_calib, dimension_held_out=None,
                      reference_calib=None, reference_held_out=None, *,
                      robust: bool = True, huber_delta_pct: float = 3.0,
                      min_constraints: int = 2) -> AlignmentResult:
    """Mode 3: known robot/table dimensions as scale constraints.

    `dimension_calib`/`dimension_held_out`: list of
        {"p1":[x,y,z], "p2":[x,y,z], "known_length_m": float, "label"?}
    giving the scan-frame endpoints of a real, independently-measured
    length (a table edge, a robot base plate width, ...).

    `reference_calib`/`reference_held_out` (optional): point
    correspondences (same schema as align_fiducial) used ONLY to fit
    rotation+translation once scale is already fixed by the dimension
    constraints (a rigid, scale=1, fit on scale-corrected src points). If
    omitted, R=I, t=0 and only scale is returned - callers combining this
    with another mode (e.g. droid_fk for R/t, dimensions for a scale
    sanity check) should treat that combination themselves.
    """
    pairs = [(np.asarray(d["p1"], dtype=np.float64),
              np.asarray(d["p2"], dtype=np.float64),
              float(d["known_length_m"])) for d in (dimension_calib or [])]
    if len(pairs) < min_constraints:
        raise AlignmentDegeneracyError(
            f"only {len(pairs)} dimension constraint(s) given, need >= "
            f"{min_constraints} for a robust scale estimate")
    scale_fit = fit_scale_from_dimensions(pairs, robust=robust,
                                          huber_delta_pct=huber_delta_pct)
    s = scale_fit["scale"]
    R, t = np.eye(3), np.zeros(3)
    extra: dict[str, Any] = {
        "dimension_fit": scale_fit,
        "rotation_translation": "identity (no reference_calib given)",
    }
    fit_residual: dict[str, Any] = {
        "n": scale_fit["n_total"], "rms_m": None, "median_m": None,
        "note": "scale-only fit; see extra.dimension_fit.residual_pct for "
                "per-constraint scale agreement instead",
    }
    if reference_calib:
        rsrc, rdst, rw, _ = _corr_arrays(reference_calib)
        check_point_set_degeneracy(rsrc, min_points=4, what="reference(src)")
        _, R, t = weighted_umeyama(rsrc * s, rdst, rw, allow_scale=False)
        extra["rotation_translation"] = ("fit via reference_calib (rigid "
                                         "Kabsch, scale fixed by dimensions)")
        T = _make_T(s, R, t)
        fit_residual = evaluate_correspondence_residual(T, rsrc, rdst, rw)
    T = _make_T(s, R, t)

    held: dict[str, Any] | None = None
    if dimension_held_out:
        hp1 = np.array([d["p1"] for d in dimension_held_out], dtype=np.float64)
        hp2 = np.array([d["p2"] for d in dimension_held_out], dtype=np.float64)
        hknown = np.array([d["known_length_m"] for d in dimension_held_out],
                          dtype=np.float64)
        hobs = np.linalg.norm(hp2 - hp1, axis=1)
        err_pct = np.abs(hobs * s / hknown - 1.0) * 100.0
        held = {"n": len(dimension_held_out),
               "length_error_pct_median": float(np.median(err_pct)),
               "length_error_pct_max": float(np.max(err_pct))}
    if reference_held_out:
        hsrc, hdst, hw, hlabels = _corr_arrays(reference_held_out)
        pt_res = evaluate_correspondence_residual(T, hsrc, hdst, hw)
        pt_res["labels"] = hlabels
        held = held or {}
        held["point_residual"] = pt_res
        # surface the point-residual RMS/median at the top level too so
        # alignment_report's generic held_out_rms_m/held_out_median_m
        # thresholds can gate on it exactly like modes 1/2.
        held.setdefault("rms_m", pt_res["rms_m"])
        held.setdefault("median_m", pt_res["median_m"])

    degeneracy = {"dimension_constraints": {"n_total": len(pairs),
                                            "n_used": scale_fit["n_used"]}}
    return AlignmentResult(mode="dimensions", s=s, R=R, t=t, T=T,
                           n_total=len(pairs), n_used=scale_fit["n_used"],
                           inlier_mask=np.asarray(scale_fit["inlier_mask"]),
                           fit=fit_residual, held_out=held, scale_ci95=None,
                           degeneracy=degeneracy, extra=extra)


# ------------------------------------------------------------- mode 4 ----

def align_droid_fk(recon_npz, traj_json, out_npz, *, max_offset: int = 2,
                    max_rms_m: float = 0.10, python_exe: str | None = None,
                    timeout_s: int = 1800,
                    min_trajectory_baseline_m: float = 0.15,
                    extra_args: list[str] | None = None) -> AlignmentResult:
    """Mode 4: dispatch to agents/recon/align_to_traj.py (READ-ONLY here -
    another agent is actively fixing it). Runs it as a subprocess against
    its documented CLI contract, then parses its align_report.json (and
    the T_align/metric_scale it writes into --out) DEFENSIVELY: every
    field is read with .get()/try-except so a concurrent change to its
    internal schema degrades this wrapper's output (missing fields, a
    warning note) instead of crashing it.

    Runs its own pre-flight trajectory-baseline degeneracy check on the FK
    centers (agents/recon/droid_extract.py's `full.c2w_base`) BEFORE
    invoking align_to_traj, since that file's own --max-rms-m abort
    catches a bad FIT but not a technically-tiny-residual, no-real-motion
    trajectory that would still make an unstable fit look clean.

    NOTE on held-out reporting: align_to_traj.py fits ALL available FK-vs-
    SfM frame correspondences and does not itself hold out an evaluation
    subset, so `result.held_out` is None here and `result.fit` is
    explicitly a FIT residual (see result.extra["note"]). Do not read
    result.fit as a held-out residual for this mode.
    """
    recon_npz, traj_json, out_npz = Path(recon_npz), Path(traj_json), Path(out_npz)
    traj = json.loads(traj_json.read_text())
    full = traj.get("full", {}) if isinstance(traj, dict) else {}
    c2w_base = full.get("c2w_base")
    traj_checked = False
    if c2w_base:
        fk_centers = np.asarray(c2w_base, dtype=np.float64)[:, :3, 3]
        check_trajectory_baseline(fk_centers, min_points=10,
                                  min_extent_m=min_trajectory_baseline_m)
        traj_checked = True
    else:
        print("[robot_align] WARNING: traj json has no full.c2w_base - "
             "skipping the pre-flight trajectory-baseline degeneracy check "
             "(align_to_traj.py's own schema may have changed)")

    python_exe = python_exe or sys.executable
    cmd = [python_exe, "-m", "agents.recon.align_to_traj",
          "--recon", str(recon_npz), "--traj", str(traj_json),
          "--out", str(out_npz), "--max-offset", str(max_offset),
          "--max-rms-m", str(max_rms_m)] + list(extra_args or [])
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True,
                          text=True, timeout=timeout_s)
    report_path = out_npz.with_name("align_report.json")
    if proc.returncode != 0 or not report_path.exists():
        raise RuntimeError(
            f"[robot_align] mode 4 (droid_fk): align_to_traj.py failed "
            f"(returncode={proc.returncode})\n--- stdout (tail) ---\n"
            f"{proc.stdout[-4000:]}\n--- stderr (tail) ---\n{proc.stderr[-4000:]}")
    report = json.loads(report_path.read_text())

    s = report.get("scale")
    R, t = np.eye(3), np.zeros(3)
    warn = None
    try:
        with np.load(out_npz, allow_pickle=False) as z:
            if "T_align" in z.files:
                T_rig = np.asarray(z["T_align"], dtype=np.float64)
                R, t = T_rig[:3, :3], T_rig[:3, 3]
            elif s is not None:
                warn = "T_align not found in output npz; R=I,t=0 (scale only recovered)"
            if s is None and "metric_scale" in z.files:
                s = float(z["metric_scale"])
    except (OSError, ValueError, KeyError) as e:
        warn = f"could not read {out_npz} for T_align/metric_scale: {e}"
    if s is None:
        warn = (warn + "; " if warn else "") + "no scale found in report or npz"
        s = float("nan")
    T = _make_T(s, R, t) if np.isfinite(s) else np.eye(4)

    fit = {"n": report.get("n_frames"), "rms_m": report.get("center_rms_m"),
          "median_m": report.get("center_median_m")}
    extra = {
        "frame_offset": report.get("frame_offset"),
        "rotation_residual_deg": report.get("rotation_residual_deg"),
        "reproj_median_px": report.get("reproj_median_px"),
        "cloud_z_mode_m": report.get("cloud_z_mode_m"),
        "convention": report.get("convention"),
        "align_report_path": str(report_path),
        "warning": warn,
        "note": ("align_to_traj.py fits ALL available FK-vs-SfM frame "
                "correspondences and does not hold out an evaluation "
                "subset itself; `fit` above is a FIT residual (center_rms_m), "
                "NOT held-out. held_out is intentionally None for this mode."),
    }
    return AlignmentResult(mode="droid_fk", s=s, R=R, t=t, T=T,
                           n_total=report.get("n_frames"),
                           n_used=report.get("n_frames"), inlier_mask=None,
                           fit=fit, held_out=None, scale_ci95=None,
                           degeneracy={"trajectory": {"checked": traj_checked}},
                           extra=extra)


# --------------------------------------------------------------- unified --

_MODE_FUNCS = {
    "fiducial": align_fiducial,
    "surveyed": align_surveyed,
    "surveyed_points": align_surveyed,
    "dimensions": align_dimensions,
    "droid_fk": align_droid_fk,
}


def align(mode: str, **kwargs) -> AlignmentResult:
    """One entry point for all four modes; `mode` in
    {"fiducial","surveyed","dimensions","droid_fk"}."""
    fn = _MODE_FUNCS.get(mode)
    if fn is None:
        raise ValueError(f"unknown alignment mode {mode!r}; choose one of "
                         f"{sorted(set(_MODE_FUNCS))}")
    return fn(**kwargs)


# -------------------------------------------------------------------- CLI

def _load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def run_from_config(config_path, out_dir=None) -> dict:
    from agents.recon import alignment_report as AR

    cfg = _load_yaml(config_path)
    mode = cfg["mode"]
    out_cfg = cfg.get("output", {})
    out_dir = Path(out_dir) if out_dir else Path(out_cfg.get("dir", "."))
    report_path = Path(out_cfg.get("report_json", out_dir / "align_report.json"))

    calib_pts = held_pts = None
    try:
        if mode in ("fiducial", "surveyed"):
            corr = cfg["correspondences"]
            calib, held = corr.get("calib", []), corr.get("held_out")
            result = align(mode, calib=calib, held_out=held, **cfg.get("solver", {}))
            calib_pts = _corr_arrays(calib)
            held_pts = _corr_arrays(held) if held else None
        elif mode == "dimensions":
            dc = cfg["dimension_constraints"]
            rc = cfg.get("reference_correspondences", {})
            result = align("dimensions",
                           dimension_calib=dc.get("calib", []),
                           dimension_held_out=dc.get("held_out"),
                           reference_calib=rc.get("calib"),
                           reference_held_out=rc.get("held_out"),
                           **cfg.get("solver", {}))
        elif mode == "droid_fk":
            result = align("droid_fk", recon_npz=cfg["recon_npz"],
                           traj_json=cfg["traj_json"], out_npz=cfg["out_npz"],
                           **cfg.get("solver", {}))
        else:
            raise ValueError(f"unknown mode {mode!r} in {config_path}")
    except AlignmentDegeneracyError as e:
        # "No rollout begins without an alignment report" applies even to a
        # calibration attempt that could not produce a transform at all -
        # write a minimal REJECT report instead of letting the exception
        # propagate as a bare traceback with no artifact on disk.
        report = {"mode": mode, "scene_id": cfg.get("scene_id"),
                  "overall_pass": False, "held_out_available": False,
                  "recommendation": "REJECT_DEGENERATE",
                  "error": {"type": "AlignmentDegeneracyError", "message": str(e)},
                  "notes": [f"alignment aborted before a transform could be "
                           f"fit: {e}"]}
        AR.write_report(report, report_path)
        print(f"[robot_align] mode={mode} DEGENERATE - {e} -> {report_path}")
        return report

    report = AR.build_report(result, thresholds=cfg.get("thresholds"),
                             extra_checks=cfg.get("extra_checks"),
                             scene_id=cfg.get("scene_id"))
    AR.write_report(report, report_path)
    print(f"[robot_align] mode={mode} scale={result.s:.5f} "
         f"overall_pass={report['overall_pass']} "
         f"held_out_available={report['held_out_available']} -> {report_path}")

    overlay = out_cfg.get("overlay_png")
    if overlay and calib_pts and calib_pts[0] is not None:
        csrc, cdst, _, _ = calib_pts
        cpred = transform_points(result.T, csrc)
        hdst = hpred = None
        if held_pts and held_pts[0] is not None:
            hsrc, hdst, _, _ = held_pts
            hpred = transform_points(result.T, hsrc)
        AR.plot_overlay(overlay, calib_dst=cdst, calib_pred=cpred,
                        held_dst=hdst, held_pred=hpred, title=f"{mode} alignment")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    report = run_from_config(args.config, args.out_dir)
    raise SystemExit(0 if report["overall_pass"] else 1)


if __name__ == "__main__":
    main()
