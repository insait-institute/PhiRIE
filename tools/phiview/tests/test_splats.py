"""CPU tests for physicalview.splats."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from physicalview import splats as SP


def _gs(n: int = 50, seed: int = 0, sh_degree: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    k = (sh_degree + 1) ** 2
    return {
        "means": torch.from_numpy(rng.normal(size=(n, 3)).astype(np.float32)),
        "quats": torch.from_numpy(q.astype(np.float32)),
        "scales": torch.from_numpy(np.exp(rng.normal(size=(n, 3)) - 3).astype(np.float32)),
        "opacities": torch.from_numpy(rng.uniform(0.05, 1.0, size=n).astype(np.float32)),
        "sh": torch.from_numpy(rng.normal(size=(n, k, 3)).astype(np.float32)),
        "sh_degree": sh_degree,
    }


def _rotz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def _quat_from_axis_angle(axis, deg) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.array([np.cos(h), *(np.sin(h) * axis)])


def test_quat_to_matrix_matches_closed_form_and_is_rotation():
    rng = np.random.default_rng(1)
    q = rng.normal(size=(20, 4))
    R = SP.quat_wxyz_to_matrix(q)
    assert R.shape == (20, 3, 3)
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-9)
    assert np.allclose(np.linalg.det(R), 1.0)
    # 90 deg about z, wxyz
    Rz = SP.quat_wxyz_to_matrix(np.array([[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]]))[0]
    assert np.allclose(Rz, _rotz(90), atol=1e-9)
    # against agents.core.common's scalar implementation
    common = pytest.importorskip("agents.core.common", reason="Backend quaternion cross-check requires SimAny")
    quat_to_rot_wxyz = common.quat_to_rot_wxyz
    for qi in q[:5]:
        assert np.allclose(SP.quat_wxyz_to_matrix(qi[None])[0], quat_to_rot_wxyz(qi), atol=1e-9)


def test_to_viser_arrays_shapes_colors_and_covariances():
    gs = _gs(40)
    arr = SP.to_viser_arrays(gs, max_splats=1000)
    assert len(arr) == 40
    assert arr.centers.shape == (40, 3) and arr.centers.dtype == np.float32
    assert arr.covariances.shape == (40, 3, 3) and arr.covariances.dtype == np.float32
    assert arr.rgbs.shape == (40, 3) and arr.rgbs.dtype == np.float32
    assert arr.opacities.shape == (40, 1) and arr.opacities.dtype == np.float32
    assert arr.rgbs.min() >= 0.0 and arr.rgbs.max() <= 1.0
    dc = gs["sh"][:, 0].numpy()
    assert np.allclose(arr.rgbs, np.clip(0.5 + 0.2820948 * dc, 0, 1), atol=1e-6)
    # cov = R diag(s^2) R^T, symmetric PSD with trace = sum(s^2)
    s2 = (gs["scales"].numpy().astype(np.float64) ** 2).sum(1)
    assert np.allclose(np.trace(arr.covariances, axis1=1, axis2=2), s2, rtol=1e-4)
    assert np.allclose(arr.covariances, arr.covariances.transpose(0, 2, 1), atol=1e-7)
    assert np.allclose(arr.centers, gs["means"].numpy())


def test_to_viser_arrays_importance_subsampling_matches_viewer_rule():
    gs = _gs(200, seed=3)
    arr = SP.to_viser_arrays(gs, max_splats=25)
    assert len(arr) == 25
    opac = gs["opacities"].numpy()
    scales = gs["scales"].numpy().astype(np.float64)
    w = opac * scales.mean(axis=1)
    idx = np.argsort(-w)[:25]
    assert np.allclose(arr.centers, gs["means"].numpy()[idx])
    assert np.allclose(arr.opacities[:, 0], opac[idx])
    # no subsampling when the budget is not exceeded / disabled
    assert len(SP.to_viser_arrays(gs, max_splats=200)) == 200
    assert len(SP.to_viser_arrays(gs, max_splats=0)) == 200


def test_to_viser_arrays_extra_scale():
    gs = _gs(10)
    a1 = SP.to_viser_arrays(gs, 100)
    a2 = SP.to_viser_arrays(gs, 100, extra_scale=2.0)
    assert np.allclose(a2.centers, 2 * a1.centers)
    assert np.allclose(a2.covariances, 4 * a1.covariances, rtol=1e-4)
    assert np.allclose(a2.rgbs, a1.rgbs) and np.allclose(a2.opacities, a1.opacities)


def test_pose_arrays_identity_and_similarity():
    arr = SP.to_viser_arrays(_gs(30), 100)
    same = SP.pose_arrays(arr, np.eye(4))
    assert np.allclose(same.centers, arr.centers) and np.allclose(same.covariances, arr.covariances)
    s, R, t = 0.5, _rotz(30), np.array([1.0, -2.0, 3.0])
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    posed = SP.pose_arrays(arr, T)
    assert np.allclose(posed.centers, (s * arr.centers @ R.T + t), atol=1e-5)
    exp_cov = s * s * (R[None] @ arr.covariances.astype(np.float64) @ R.T[None])
    assert np.allclose(posed.covariances, exp_cov, atol=1e-6)
    assert posed.rgbs is arr.rgbs and posed.opacities is arr.opacities


@pytest.mark.backend
def test_pose_arrays_match_backend_gaussian_transform():
    T = np.eye(4)
    T[:3, :3] = .5 * _rotz(30)
    T[:3, 3] = [1, -2, 3]
    from agents.core.common import transform_gaussians
    gs = _gs(30)
    tg = transform_gaussians(gs, T)
    assert np.allclose(SP.to_viser_arrays(tg, 100).centers, SP.pose_arrays(SP.to_viser_arrays(gs, 100), T).centers, atol=1e-5)
    assert np.allclose(SP.to_viser_arrays(tg, 100).covariances,
                       SP.pose_arrays(SP.to_viser_arrays(gs, 100), T).covariances, atol=1e-6)


def test_decompose_similarity():
    T = np.eye(4)
    T[:3, :3] = 0.3 * _rotz(45)
    T[:3, 3] = [1, 2, 3]
    s, R, t = SP.decompose_similarity(T)
    assert s == pytest.approx(0.3) and np.allclose(R, _rotz(45)) and np.allclose(t, [1, 2, 3])


def test_frame_transform_identity():
    aligned = np.eye(4)
    aligned[:3, :3] = 0.4 * _rotz(20)
    aligned[:3, 3] = [2.0, 1.0, 0.8]
    p0 = np.array([2.0, 1.0, 0.8])
    q0 = _quat_from_axis_angle([0, 0, 1], 20)
    T = SP.frame_transform(aligned, p0, q0, p0, q0)
    assert np.allclose(T, aligned, atol=1e-12)


def test_frame_transform_rotation_translation_case():
    # canonical -> world at rest: scale 0.5, yaw 10 deg, at (1, 2, 0.75)
    aligned = np.eye(4)
    aligned[:3, :3] = 0.5 * _rotz(10)
    aligned[:3, 3] = [1.0, 2.0, 0.75]
    # MuJoCo body at reset: some pose that is NOT the canonical frame (offset origin)
    p0 = np.array([1.1, 2.05, 0.7])
    q0 = _quat_from_axis_angle([0, 0, 1], 10)
    # body moves: rotate 90 deg about z and translate by (0.3, -0.2, 0.1)
    R0 = SP.quat_wxyz_to_matrix(q0[None])[0]
    Rd = SP.quat_wxyz_to_matrix(_quat_from_axis_angle([0.1, 0.2, 1.0], 90)[None])[0]
    R = Rd @ R0
    from scipy.spatial.transform import Rotation
    def rot_to_quat_wxyz(matrix):
        return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
    q = rot_to_quat_wxyz(R)
    p = Rd @ p0 + np.array([0.3, -0.2, 0.1])  # the body origin follows the rigid motion
    T = SP.frame_transform(aligned, p, q, p0, q0)
    # every canonical point must undergo the same rigid motion as the body
    x = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [0.5, 0.5, -0.5]])
    w_rest = x @ aligned[:3, :3].T + aligned[:3, 3]
    w_new = (w_rest - p0) @ Rd.T + p          # rigid: Rd (w - p0) + p
    assert np.allclose(x @ T[:3, :3].T + T[:3, 3], w_new, atol=1e-9)
    # scale preserved, rotation part = Rd @ R_aligned
    s, Rt, _ = SP.decompose_similarity(T)
    assert s == pytest.approx(0.5)
    assert np.allclose(Rt, Rd @ _rotz(10), atol=1e-9)


def test_frame_transform_equals_absolute_when_body_frame_is_canonical():
    """export_mjcf places body origins at the canonical origin: then the relative form
    and pi05_render.py's absolute [s R(q) | p] agree."""
    s, R0, t0 = 0.3, _rotz(35), np.array([0.5, -0.4, 0.9])
    aligned = np.eye(4)
    aligned[:3, :3] = s * R0
    aligned[:3, 3] = t0
    from scipy.spatial.transform import Rotation
    def rot_to_quat_wxyz(matrix):
        return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
    q0 = rot_to_quat_wxyz(R0)
    Rn = SP.quat_wxyz_to_matrix(_quat_from_axis_angle([1, 1, 0], 40)[None])[0] @ R0
    qn, pn = rot_to_quat_wxyz(Rn), np.array([0.7, 0.1, 1.2])
    rel = SP.frame_transform(aligned, pn, qn, t0, q0)
    absolute = SP.body_pose_transform(aligned, pn, qn)
    assert np.allclose(rel, absolute, atol=1e-9)


def test_splat_cache_lru_by_total_gaussians():
    cache = SP.SplatCache(max_total=100)
    calls = []

    def builder(n):
        def b():
            calls.append(n)
            return SP.to_viser_arrays(_gs(n), 10_000)
        return b

    a = cache.get("a", builder(40))
    assert cache.get("a", builder(40)) is a and calls == [40]
    cache.get("b", builder(40))
    assert cache.total == 80 and len(cache) == 2
    cache.get("a", builder(40))             # touch a -> b is LRU
    cache.get("c", builder(40))             # 120 > 100 -> evict b
    assert "b" not in cache and "a" in cache and "c" in cache and cache.total == 80
    big = cache.get("big", builder(150))    # larger than the budget: kept alone
    assert "big" in cache and len(cache) == 1 and cache.total == 150 and len(big) == 150
    cache.pop("big")
    assert cache.total == 0 and len(cache) == 0
