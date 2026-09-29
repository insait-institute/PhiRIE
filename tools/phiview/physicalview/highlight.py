"""High-contrast server-side overlays derived from actual visibility masks."""
from __future__ import annotations

import cv2
import numpy as np

RED = np.array([1.0, 0.0, 0.03], dtype=np.float32)


def red_mask(rgb, mask, *, opacity=.82, outline_px=2):
    """Overlay a boolean mask on float RGB without changing input pixels or labels."""
    rgb = np.asarray(rgb)
    mask = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError('RGB and mask dimensions must match')
    if not 0 <= opacity <= 1 or not 0 <= outline_px <= 12:
        raise ValueError('Invalid overlay opacity or outline width')
    out = rgb.astype(np.float32, copy=True)
    out[mask] = out[mask] * (1-opacity) + RED * opacity
    if outline_px and mask.any():
        kernel = np.ones((2*outline_px+1, 2*outline_px+1), np.uint8)
        # White outer contour stays visible on red objects and dark backgrounds.
        rim = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask
        out[rim] = 1.0
    return np.clip(out, 0, 1)


def highlight_objects(rgb, labels, selected_id=None, *, show_all=True):
    """Keep selection visible independently of the discovered-object overlay."""
    labels = np.asarray(labels)
    out = red_mask(rgb, labels > 0, opacity=.22, outline_px=1) if show_all else np.asarray(rgb).copy()
    if selected_id is not None:
        selected = labels == selected_id
        if selected.any():
            focused = red_mask(rgb, selected, opacity=.94, outline_px=3)
            area = cv2.dilate(selected.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            out[area] = focused[area]
    return out
