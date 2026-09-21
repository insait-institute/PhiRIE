"""Deterministic background-fitted RGB affine baseline; never fits at evaluation."""
from __future__ import annotations

import hashlib
import numpy as np
from robo.manifest.hash import canonical_hash

ALGORITHM = "construction_train_rgb_mean_std_affine_population_std_constant_source_gain_one_rint_uint8_v1"


def _rgb(value):
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("color-match input must be uint8 HxWx3 RGB")
    return value


def _pixels_hash(value):
    digest = hashlib.sha256(str((value.shape, str(value.dtype))).encode())
    digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def fit_background_affine(source_rgb, reference_rgb, valid_background_mask, *,
                          camera_id, calibration_view_id, construction_view_ids,
                          evaluation_view_ids):
    """Fit once on a declared construction view and archive input pixel hashes.

    The caller's E0 must authenticate view identities and the background mask.
    This function prevents an explicit split overlap; it cannot independently
    establish the origin of arbitrary arrays passed by an unauthenticated caller.
    """
    construction, evaluation = set(construction_view_ids), set(evaluation_view_ids)
    if not camera_id or not calibration_view_id or calibration_view_id not in construction:
        raise ValueError("calibration requires a named camera and construction view")
    if construction & evaluation:
        raise ValueError("color calibration construction/evaluation split overlaps")
    source, reference = _rgb(source_rgb), _rgb(reference_rgb)
    mask = np.asarray(valid_background_mask)
    if source.shape != reference.shape or mask.dtype != np.bool_ or mask.shape != source.shape[:2] or not mask.any():
        raise ValueError("calibration requires aligned RGB and a nonempty boolean background mask")
    x, y = source[mask].astype(np.float64), reference[mask].astype(np.float64)
    mx, my, sx, sy = x.mean(0), y.mean(0), x.std(0, ddof=0), y.std(0, ddof=0)
    gain = np.divide(sy, sx, out=np.ones(3), where=sx > 0)
    offset = my - gain * mx
    result = dict(schema_version=1, algorithm=ALGORITHM, camera_id=camera_id,
                  calibration_view_id=calibration_view_id,
                  construction_view_ids=sorted(construction), evaluation_view_ids=sorted(evaluation),
                  source_pixels_sha256=_pixels_hash(source), reference_pixels_sha256=_pixels_hash(reference),
                  background_mask_sha256=_pixels_hash(mask), background_pixels=int(mask.sum()),
                  gain=gain.tolist(), offset=offset.tolist(),
                  source_mean=mx.tolist(), reference_mean=my.tolist(),
                  source_std=sx.tolist(), reference_std=sy.tolist())
    result["calibration_sha256"] = canonical_hash(result)
    return result


def apply_background_affine(image, calibration, *, camera_id):
    """Apply frozen parameters; RGB clipping and ties-to-even rounding are fixed."""
    raw = _rgb(image)
    payload = {k: v for k, v in calibration.items() if k != "calibration_sha256"}
    if (calibration.get("schema_version") != 1 or calibration.get("algorithm") != ALGORITHM
            or calibration.get("camera_id") != camera_id
            or calibration.get("calibration_sha256") != canonical_hash(payload)):
        raise ValueError("color calibration identity/hash/camera differs")
    gain, offset = np.asarray(calibration["gain"], dtype=np.float64), np.asarray(calibration["offset"], dtype=np.float64)
    if gain.shape != (3,) or offset.shape != (3,) or not np.isfinite(gain).all() or not np.isfinite(offset).all():
        raise ValueError("color affine parameters must be three finite channels")
    return np.rint(np.clip(raw.astype(np.float64) * gain + offset, 0, 255)).astype(np.uint8)
