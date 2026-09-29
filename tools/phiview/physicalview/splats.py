"""Gaussian dict -> viser splat arrays, with importance subsampling and posing.

CONTRACT (scene/render agent). Pure numpy/torch-CPU; no viser import here.

    to_viser_arrays(gs, max_splats, extra_scale=1.0) -> SplatArrays
        centers float32 [N,3], covariances float32 [N,3,3], rgbs float32 [N,3] in [0,1]
        (SH DC only), opacities float32 [N,1]. Subsampling keeps the top-N by
        opacity * mean(scale) exactly as interface/viewer.py does. Uses
        utils3d.numpy.quaternion_to_matrix if available, else a local wxyz->R.

    pose_arrays(arrays, T) -> SplatArrays   apply 4x4 similarity T (scale allowed:
        cov' = s^2 R cov R^T, centers' = s R c + t) — used to place canonical object
        gaussians at aligned["T"] and to follow MuJoCo body poses each tick.

    matrix_to_quat_wxyz(R) -> wxyz   (Shepperd; inverse of quat_wxyz_to_matrix for viser
        node poses and camera conversions)

    frame_transform(aligned_T, body_pos, body_quat_wxyz, T_at_reset) -> 4x4
        For live physics: an object's canonical gaussians were registered with aligned T
        (canonical -> world at rest). When MuJoCo moves body b from its reset pose
        (p0,q0) to (p,q), the new world transform is  T = Delta @ aligned_T with
        Delta = [R(q) R(q0)^T | p - R(q) R(q0)^T p0]  (same math as
        agents/render/gsplat_sim_render.py); implement and unit-test it.

    SplatCache: LRU (by key) of SplatArrays bounded by total gaussians (default 6e6).
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

SH_C0 = 0.2820948  # same constant as interface/viewer.py (DC term -> linear rgb)


@dataclass
class SplatArrays:
    centers: np.ndarray
    covariances: np.ndarray
    rgbs: np.ndarray
    opacities: np.ndarray

    def __len__(self) -> int:
        return int(self.centers.shape[0])


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor (CPU or CUDA)
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Batched wxyz quaternion -> [N,3,3] rotation. Prefers utils3d (bit-identical to
    the existing viewers) and falls back to the closed form."""
    q = np.asarray(q, dtype=np.float64)
    try:
        import utils3d  # noqa: WPS433 - optional shim present in the pipeline envs
        return np.asarray(utils3d.numpy.quaternion_to_matrix(q), dtype=np.float64)
    except Exception:  # noqa: BLE001 - utils3d missing or API changed
        pass
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion wxyz (Shepperd's method; no utils3d needed)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / (np.linalg.norm(q) + 1e-12)


def to_viser_arrays(gs: dict, max_splats: int, extra_scale: float = 1.0) -> SplatArrays:
    means = _as_numpy(gs["means"]).astype(np.float64) * extra_scale
    quats = _as_numpy(gs["quats"]).astype(np.float64)  # wxyz
    scales = _as_numpy(gs["scales"]).astype(np.float64) * extra_scale
    opac = _as_numpy(gs["opacities"]).astype(np.float64).reshape(-1)
    sh = _as_numpy(gs["sh"])
    dc = sh[:, 0] if sh.ndim == 3 else sh
    rgb = np.clip(0.5 + SH_C0 * dc.astype(np.float64), 0.0, 1.0)
    if max_splats is not None and 0 < max_splats < len(means):
        # importance subsampling (opacity x size) instead of random: keeps the
        # load-bearing gaussians, visibly better at the same budget
        w = opac * scales.mean(axis=1)
        idx = np.argsort(-w)[:max_splats]
        means, quats, scales, opac, rgb = (a[idx] for a in (means, quats, scales, opac, rgb))
    R = quat_wxyz_to_matrix(quats)
    cov = (R * scales[:, None, :] ** 2) @ R.transpose(0, 2, 1)
    return SplatArrays(
        centers=np.ascontiguousarray(means, dtype=np.float32),
        covariances=np.ascontiguousarray(cov, dtype=np.float32),
        rgbs=np.ascontiguousarray(rgb, dtype=np.float32),
        opacities=np.ascontiguousarray(opac, dtype=np.float32).reshape(-1, 1))


def decompose_similarity(T: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """4x4 with isotropic scale folded into the 3x3 -> (s, R, t); local copy of
    agents.core.common.decompose_similarity so this module needs no SIMANY_* env."""
    T = np.asarray(T, dtype=np.float64)
    A = T[:3, :3]
    s = float(np.cbrt(max(np.linalg.det(A), 1e-12)))
    return s, A / s, T[:3, 3].copy()


def pose_arrays(arrays: SplatArrays, T: np.ndarray) -> SplatArrays:
    s, R, t = decompose_similarity(T)
    c = arrays.centers.astype(np.float64)
    centers = (s * (c @ R.T) + t).astype(np.float32)
    cov = arrays.covariances.astype(np.float64)
    cov = (s * s) * (R[None] @ cov @ R.T[None])
    return SplatArrays(centers=np.ascontiguousarray(centers),
                       covariances=np.ascontiguousarray(cov.astype(np.float32)),
                       rgbs=arrays.rgbs, opacities=arrays.opacities)


def frame_transform(aligned_T: np.ndarray, body_pos: np.ndarray, body_quat_wxyz: np.ndarray,
                    reset_pos: np.ndarray, reset_quat_wxyz: np.ndarray) -> np.ndarray:
    """World transform of an object's canonical gaussians once its MuJoCo body moved from
    the reset pose (reset_pos, reset_quat) to (body_pos, body_quat):
        T = Delta @ aligned_T,  Delta = [R R0^T | p - R R0^T p0].
    Identity when the body has not moved."""
    R = quat_wxyz_to_matrix(np.asarray(body_quat_wxyz, dtype=np.float64)[None])[0]
    R0 = quat_wxyz_to_matrix(np.asarray(reset_quat_wxyz, dtype=np.float64)[None])[0]
    p = np.asarray(body_pos, dtype=np.float64).reshape(3)
    p0 = np.asarray(reset_pos, dtype=np.float64).reshape(3)
    Rd = R @ R0.T
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = Rd
    delta[:3, 3] = p - Rd @ p0
    return delta @ np.asarray(aligned_T, dtype=np.float64)


def body_pose_transform(aligned_T: np.ndarray, body_pos: np.ndarray,
                        body_quat_wxyz: np.ndarray) -> np.ndarray:
    """Absolute variant used when the MuJoCo body frame IS the (scaled) canonical frame
    (export_mjcf places body origins at the canonical origin, as gsplat_sim_render.py and
    pi05_render.py assume): T = [s R(q) | p] with s from aligned_T."""
    s, _, _ = decompose_similarity(aligned_T)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = s * quat_wxyz_to_matrix(np.asarray(body_quat_wxyz, dtype=np.float64)[None])[0]
    T[:3, 3] = np.asarray(body_pos, dtype=np.float64).reshape(3)
    return T


class SplatCache:
    """LRU of SplatArrays keyed by string, bounded by the total number of gaussians."""

    def __init__(self, max_total: int = 6_000_000):
        self.max_total = int(max_total)
        self._items: "OrderedDict[str, SplatArrays]" = OrderedDict()
        self._total = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def total(self) -> int:
        return self._total

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._items

    def get(self, key: str, builder: Callable[[], SplatArrays]) -> SplatArrays:
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit
        arrays = builder()  # outside the lock: builders load plys and may take seconds
        with self._lock:
            if key in self._items:
                self._total -= len(self._items.pop(key))
            self._items[key] = arrays
            self._total += len(arrays)
            while self._total > self.max_total and len(self._items) > 1:
                _, evicted = self._items.popitem(last=False)
                self._total -= len(evicted)
        return arrays

    def pop(self, key: str) -> None:
        with self._lock:
            item = self._items.pop(key, None)
            if item is not None:
                self._total -= len(item)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._total = 0
