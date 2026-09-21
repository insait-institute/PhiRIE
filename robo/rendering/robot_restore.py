"""Robot-preserving full-frame enhancement (paper Option C).

The eroded robot core is copied byte-for-byte from the raw simulator image.
Only a narrow boundary band is feathered, while all non-robot pixels come from
the enhanced image. The implementation is NumPy-only so its invariant is tested
in lightweight CI without a renderer or GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RestorationStats:
    core_pixels: int
    boundary_pixels: int
    outside_pixels: int
    core_equal: bool
    max_core_error: int


def _window_reduce(mask: np.ndarray, radius: int, operation: str) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (2 * radius + 1, 2 * radius + 1))
    if operation == "dilate":
        return windows.any(axis=(-2, -1))
    if operation == "erode":
        return windows.all(axis=(-2, -1))
    raise ValueError(operation)


def normalize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.shape != shape:
        raise ValueError(f"mask shape {array.shape} does not match image {shape}")
    if array.dtype == np.bool_:
        return array
    if np.issubdtype(array.dtype, np.floating):
        return array >= 0.5
    return array >= 128


def robot_restore_alpha(robot_mask: np.ndarray, *, erode_px: int = 1,
                        dilate_px: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if erode_px < 0 or dilate_px < 0:
        raise ValueError("erosion and dilation radii must be non-negative")
    mask = np.asarray(robot_mask, dtype=bool)
    core = _window_reduce(mask, erode_px, "erode")
    dilated = _window_reduce(mask, dilate_px, "dilate")
    alpha = core.astype(np.float32)
    ring_total = max(erode_px + dilate_px, 1)
    current = core.copy()
    for ring in range(1, ring_total + 1):
        expanded = _window_reduce(current, 1, "dilate") & dilated
        new_ring = expanded & ~current
        if not new_ring.any():
            break
        alpha[new_ring] = 1.0 - ring / (ring_total + 1.0)
        current = expanded
    alpha[~dilated] = 0.0
    return alpha, core, dilated


def restore_robot_core(raw_rgb: np.ndarray, enhanced_rgb: np.ndarray,
                       robot_mask: np.ndarray, *, erode_px: int = 1,
                       dilate_px: int = 4) -> tuple[np.ndarray, RestorationStats]:
    raw = np.asarray(raw_rgb)
    enhanced = np.asarray(enhanced_rgb)
    if raw.shape != enhanced.shape or raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError(
            f"raw/enhanced images must be equal HxWx3 arrays, got "
            f"{raw.shape} and {enhanced.shape}")
    mask = normalize_mask(robot_mask, raw.shape[:2])
    alpha, core, dilated = robot_restore_alpha(
        mask, erode_px=erode_px, dilate_px=dilate_px)
    out = (alpha[..., None] * raw.astype(np.float32)
           + (1.0 - alpha[..., None]) * enhanced.astype(np.float32))
    if np.issubdtype(raw.dtype, np.integer):
        bounds = np.iinfo(raw.dtype)
        out = np.clip(np.rint(out), bounds.min, bounds.max).astype(raw.dtype)
    else:
        out = out.astype(raw.dtype)
    out[core] = raw[core]
    if core.any():
        error = np.abs(out[core].astype(np.int64) - raw[core].astype(np.int64))
        max_error = int(error.max())
    else:
        max_error = 0
    stats = RestorationStats(
        core_pixels=int(core.sum()),
        boundary_pixels=int((dilated & ~core).sum()),
        outside_pixels=int((~dilated).sum()),
        core_equal=bool(np.array_equal(out[core], raw[core])),
        max_core_error=max_error)
    if not stats.core_equal:
        raise AssertionError("robot-core restoration invariant was violated")
    return out, stats
