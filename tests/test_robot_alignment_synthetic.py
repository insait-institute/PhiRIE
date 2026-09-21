"""Synthetic recovery/robustness/round-trip/degeneracy suite for
agents/recon/robot_align.py (plan/04_METRIC_SCALE_ROBOT_ALIGNMENT.md).

Ground truth, marker layout, noise spec and round-trip probes all come
from tests/data/alignment/synthetic.yaml (see that file's header) so the
numeric fixture recipe is auditable in one place instead of scattered
through assertions.

Run: .venv/bin/python -m pytest -q tests/test_robot_alignment_synthetic.py
"""
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from agents.recon import robot_align as RA

FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "alignment" / "synthetic.yaml"


@pytest.fixture(scope="module")
def fixture():
    return yaml.safe_load(FIXTURE_PATH.read_text())


def _gt_transform(fx):
    gt = fx["ground_truth"]
    R_gt = Rotation.from_euler("xyz", gt["rotation_euler_xyz_deg"],
                               degrees=True).as_matrix()
    s_gt = float(gt["scale"])
    t_gt = np.asarray(gt["translation_m"], dtype=np.float64)
    T_gt = RA._make_T(s_gt, R_gt, t_gt)
    return s_gt, R_gt, t_gt, T_gt


def _corr_dicts(src, dst):
    return [{"src": list(map(float, s)), "dst": list(map(float, d))}
            for s, d in zip(src, dst)]


# --------------------------------------------------------------- recovery

def test_recovers_sim3_under_nominal_noise(fixture):
    """Synthetic random Sim(3): scale <1%, translation <5mm, rotation
    <0.5deg under nominal Gaussian noise (plan's exact numeric bar) -
    exercised through the real align_fiducial API (mode 1), including its
    held-out path (a few extra markers never used to fit)."""
    s_gt, R_gt, t_gt, T_gt = _gt_transform(fixture)
    rng = np.random.default_rng(fixture["seed"])
    sigma = fixture["noise"]["correspondence_sigma_m"]

    markers = np.asarray(fixture["markers_src"], dtype=np.float64)
    dst_clean = RA.transform_points(T_gt, markers)
    dst_noisy = dst_clean + rng.normal(scale=sigma, size=dst_clean.shape)

    # last 2 markers held out, never used to fit
    calib = _corr_dicts(markers[:-2], dst_noisy[:-2])
    held_out = _corr_dicts(markers[-2:], dst_noisy[-2:])

    result = RA.align_fiducial(calib, held_out, rng=fixture["seed"])

    tol = fixture["tolerances"]
    scale_err_pct = abs(result.s / s_gt - 1.0) * 100.0
    # translation error at the marker centroid (repo convention: see
    # tests/test_align_synthetic.py's own t_err definition), not the raw
    # T[:3,3] vector, since that alone is not a frame-independent quantity.
    ctr = markers.mean(axis=0, keepdims=True)
    trans_err_mm = float(np.linalg.norm(
        RA.transform_points(result.T, ctr) - RA.transform_points(T_gt, ctr))) * 1000.0
    rot_err_deg = RA.rotation_angle_deg(result.R, R_gt)

    print(f"\n[recovery] scale_err={scale_err_pct:.4f}% "
         f"trans_err={trans_err_mm:.4f}mm rot_err={rot_err_deg:.4f}deg "
         f"fit_rms={result.fit['rms_m'] * 1000:.3f}mm "
         f"held_out_rms={result.held_out['rms_m'] * 1000:.3f}mm "
         f"scale_ci95={result.scale_ci95}")

    assert scale_err_pct < tol["scale_pct"], \
        f"scale error {scale_err_pct:.3f}% >= {tol['scale_pct']}%"
    assert trans_err_mm < tol["translation_mm"], \
        f"translation error {trans_err_mm:.3f}mm >= {tol['translation_mm']}mm"
    assert rot_err_deg < tol["rotation_deg"], \
        f"rotation error {rot_err_deg:.3f}deg >= {tol['rotation_deg']}deg"

    # requirement #2: held-out residual must be reported and small, and
    # must be distinct from (not silently aliased to) the fit residual.
    assert result.held_out is not None
    assert result.held_out["n"] == 2
    assert result.held_out["rms_m"] < 0.02, \
        f"held-out RMS {result.held_out['rms_m']} m too large for {sigma} m noise"


# ---------------------------------------------------------- outlier reject

def test_outliers_rejected_by_robust_solver(fixture):
    """20% gross outliers injected into a larger correspondence set: the
    robust (RANSAC + Huber) path must recover near-ground-truth accuracy,
    while the SAME data fit without robustness (a single plain weighted
    Umeyama over 100% of the data) must be measurably worse - i.e. the
    robust path is actually doing something, not just not-hurting."""
    s_gt, R_gt, t_gt, T_gt = _gt_transform(fixture)
    rng = np.random.default_rng(fixture["seed"] + 1)
    sigma = fixture["noise"]["correspondence_sigma_m"]
    outlier_frac = fixture["noise"]["outlier_frac"]
    outlier_sigma = fixture["noise"]["outlier_offset_sigma_m"]

    n = 60
    src = rng.uniform(-1.0, 1.0, size=(n, 3))
    dst = RA.transform_points(T_gt, src) + rng.normal(scale=sigma, size=(n, 3))
    n_out = int(round(n * outlier_frac))
    out_idx = rng.choice(n, n_out, replace=False)
    dst[out_idx] += rng.normal(scale=outlier_sigma, size=(n_out, 3))

    calib = _corr_dicts(src, dst)

    robust = RA.align_fiducial(calib, robust=True, ransac=True,
                               rng=fixture["seed"], bootstrap_ci=False)
    naive = RA.align_fiducial(calib, robust=False, bootstrap_ci=False)

    def errs(res):
        s_err = abs(res.s / s_gt - 1.0) * 100.0
        t_err = float(np.linalg.norm(res.t - t_gt)) * 1000.0
        r_err = RA.rotation_angle_deg(res.R, R_gt)
        return s_err, t_err, r_err

    rs, rt, rr = errs(robust)
    ns, nt, nr = errs(naive)
    print(f"\n[outliers] robust: scale={rs:.3f}% trans={rt:.2f}mm rot={rr:.3f}deg "
         f"n_used={robust.n_used}/{robust.n_total}")
    print(f"[outliers] naive : scale={ns:.3f}% trans={nt:.2f}mm rot={nr:.3f}deg")

    tol = fixture["tolerances"]
    # the robust path should land within (a generous multiple of) the
    # nominal-noise tolerances despite 20% gross outliers...
    assert rs < tol["scale_pct"] * 3
    assert rt < tol["translation_mm"] * 3
    assert rr < tol["rotation_deg"] * 3
    # ...while the naive path, given the SAME data, is clearly worse -
    # this is the actual "robust helps" comparison, not just "robust is ok".
    assert rs < ns / 2, "robust scale error should be well below naive's"
    assert rt < nt / 2, "robust translation error should be well below naive's"
    # RANSAC should have identified roughly the true inlier fraction
    expected_inliers = n - n_out
    assert abs(robust.n_used - expected_inliers) <= max(3, int(0.1 * n))


# -------------------------------------------------------------- round trip

def test_forward_inverse_roundtrip_all_modalities(fixture):
    """Forward-then-inverse under ground_truth recovers points, a rigid
    pose (position+orientation), and a camera w2c extrinsic to floating-
    point tolerance - not just to fit tolerance, since no fitting is
    involved here at all (pure transform algebra)."""
    _, _, _, T_gt = _gt_transform(fixture)
    T_inv = RA.invert_transform(T_gt)
    probe = fixture["roundtrip_probe"]

    # 1) a raw point
    pt = np.asarray(probe["point_m"], dtype=np.float64)
    pt_rt = RA.transform_points(T_inv, RA.transform_points(T_gt, pt[None, :]))[0]
    assert np.allclose(pt_rt, pt, atol=1e-9), (pt_rt, pt)

    # 2) a rigid pose (position + orientation, e.g. an object/robot-link frame)
    R_pose = Rotation.from_quat(
        [probe["pose"]["quaternion_wxyz"][1], probe["pose"]["quaternion_wxyz"][2],
         probe["pose"]["quaternion_wxyz"][3], probe["pose"]["quaternion_wxyz"][0]]
    ).as_matrix()   # scipy quat order is (x,y,z,w); fixture stores (w,x,y,z)
    M = np.eye(4)
    M[:3, :3] = R_pose
    M[:3, 3] = probe["pose"]["position_m"]
    M_rt = RA.transform_rigid_pose(T_inv, RA.transform_rigid_pose(T_gt, M))
    assert np.allclose(M_rt, M, atol=1e-9), (M_rt, M)
    # orientation must stay a pure rotation throughout (never scaled)
    fwd = RA.transform_rigid_pose(T_gt, M)
    assert np.allclose(fwd[:3, :3].T @ fwd[:3, :3], np.eye(3), atol=1e-9)

    # 3) a camera w2c extrinsic
    R_cam = Rotation.from_euler(
        "xyz", probe["camera_w2c"]["rotation_euler_xyz_deg"], degrees=True).as_matrix()
    w2c = np.eye(4)
    w2c[:3, :3] = R_cam
    w2c[:3, 3] = probe["camera_w2c"]["translation_m"]
    w2c_rt = RA.transform_w2c(T_inv, RA.transform_w2c(T_gt, w2c))
    assert np.allclose(w2c_rt, w2c, atol=1e-9), (w2c_rt, w2c)
    fwd_w2c = RA.transform_w2c(T_gt, w2c)
    assert np.allclose(fwd_w2c[:3, :3].T @ fwd_w2c[:3, :3], np.eye(3), atol=1e-9)

    # bonus: transform_recon_dict round trip on a minimal fake recon dict
    rec = {"points": np.asarray(fixture["markers_src"], dtype=np.float64),
          "w2c": np.stack([w2c]), "depth": np.array([1.5, 2.0], dtype=np.float32)}
    rec_fwd = RA.transform_recon_dict(rec, T_gt)
    rec_rt = RA.transform_recon_dict(rec_fwd, T_inv)
    assert np.allclose(rec_rt["points"], rec["points"], atol=1e-4)
    assert np.allclose(rec_rt["w2c"], rec["w2c"], atol=1e-6)
    assert np.allclose(rec_rt["depth"], rec["depth"], atol=1e-4)


# --------------------------------------------------------------- degeneracy

def test_collinear_markers_raise_degeneracy_error(fixture):
    """A correspondence set whose src markers are exactly collinear must
    raise AlignmentDegeneracyError, never return a transform."""
    _, _, _, T_gt = _gt_transform(fixture)
    src = np.asarray(fixture["degenerate_collinear_src"], dtype=np.float64)
    dst = RA.transform_points(T_gt, src)  # otherwise-perfect data - only the
                                          # geometry itself is degenerate
    calib = _corr_dicts(src, dst)
    with pytest.raises(RA.AlignmentDegeneracyError):
        RA.align_fiducial(calib)

    # the lower-level check used directly (as align_droid_fk's pre-flight
    # check would use it on FK/SfM centers) must also raise
    with pytest.raises(RA.AlignmentDegeneracyError):
        RA.check_point_set_degeneracy(src, what="correspondence(src)")


def test_short_baseline_trajectory_raises_degeneracy_error(fixture):
    """A trajectory confined to a few mm (insufficient camera/end-effector
    motion) must raise AlignmentDegeneracyError, never return a transform."""
    traj = np.asarray(fixture["degenerate_short_baseline_traj"], dtype=np.float64)
    with pytest.raises(RA.AlignmentDegeneracyError):
        RA.check_trajectory_baseline(traj)

    # non-raising diagnostic mode must still flag it without raising, so
    # callers that want a report field instead of an exception can get one
    diag = RA.check_trajectory_baseline(traj, raise_on_fail=False)
    assert diag["degenerate"] is True
    assert diag["span_m"] < 0.15
