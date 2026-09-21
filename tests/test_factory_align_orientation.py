"""Focused regressions for factory signed-source-up registration."""

import numpy as np

from agents.assets.factory_align import (SIZE_RATIO_RANGE, TIER_A_F1_20,
                                         TIER_B_F1_40)
from agents.assets.s5_align import (SIGNED_SOURCE_UP_HYPOTHESES, align_object,
                                    align_object_with_signed_source_up, apply_T,
                                    f1_eval, make_T, rz)
from agents.core import common as C


def _asymmetric_long_x_cloud(seed=9, n=300):
    """An anisotropic solid with two sign-disambiguating protrusions."""
    rng = np.random.default_rng(seed)
    body = rng.uniform((-0.55, -0.09, -0.045),
                       (0.42, 0.09, 0.045), (n, 3))
    side_tab = rng.uniform((0.15, 0.09, -0.03),
                           (0.38, 0.22, 0.035), (n // 4, 3))
    top_bump = rng.uniform((-0.48, -0.06, 0.045),
                           (-0.30, 0.035, 0.14), (n // 5, 3))
    return np.vstack((body, side_tab, top_bump))


def test_signed_source_up_bases_are_ordered_proper_rotations():
    axes = {
        "+z": (0, 0, 1), "-z": (0, 0, -1),
        "+x": (1, 0, 0), "-x": (-1, 0, 0),
        "+y": (0, 1, 0), "-y": (0, -1, 0),
    }
    assert [name for name, _ in SIGNED_SOURCE_UP_HYPOTHESES] == [
        "+z", "-z", "+x", "-x", "+y", "-y"]
    for name, base_R in SIGNED_SOURCE_UP_HYPOTHESES:
        np.testing.assert_allclose(base_R.T @ base_R, np.eye(3), atol=1e-15)
        np.testing.assert_allclose(np.linalg.det(base_R), 1.0, atol=1e-15)
        np.testing.assert_allclose(base_R @ axes[name], (0, 0, 1), atol=1e-15)


def test_long_source_x_is_aligned_to_world_z_and_passes_strict_gates():
    source = _asymmetric_long_x_cloud()
    base_R = dict(SIGNED_SOURCE_UP_HYPOTHESES)["+x"]
    expected = make_T(
        0.24, rz(np.deg2rad(30.0)) @ base_R, np.array([0.3, -0.2, 0.7]))
    target = apply_T(expected, source)
    assert np.argmax(np.ptp(source, axis=0)) == 0
    assert np.argmax(np.ptp(target, axis=0)) == 2

    T, chamfer, _, tilt, source_up = align_object_with_signed_source_up(
        source, target)
    scale, R, _ = C.decompose_similarity(T)
    ev = f1_eval(apply_T(T, source), target)
    ratio = scale * np.ptp(source, axis=0).max() / np.ptp(target, axis=0).max()

    assert source_up == "+x"
    assert tilt < 1e-4
    np.testing.assert_allclose(R @ (1, 0, 0), (0, 0, 1), atol=1e-12)
    assert chamfer < 1e-12
    assert SIZE_RATIO_RANGE[0] <= ratio <= SIZE_RATIO_RANGE[1]
    assert ev["f1@20mm"]["f1"] >= TIER_A_F1_20
    assert ev["f1@40mm"]["f1"] >= TIER_B_F1_40


def test_z_up_multistart_preserves_legacy_alignment_and_four_value_api():
    # Move the long/asymmetric source axis from x to z while preserving shape.
    source = _asymmetric_long_x_cloud()[:, [1, 2, 0]]
    expected = make_T(
        0.31, rz(np.deg2rad(40.0)), np.array([-0.2, 0.1, 0.6]))
    target = apply_T(expected, source)

    legacy = align_object(source, target)
    expanded = align_object_with_signed_source_up(source, target)

    assert len(legacy) == 4
    assert len(expanded) == 5
    assert expanded[4] == "+z"
    np.testing.assert_allclose(legacy[0], expected, atol=1e-12)
    np.testing.assert_allclose(legacy[1:3], (0.0, 0.0), atol=1e-12)
    # arccos loses precision near 1: an O(eps) cosine error becomes an
    # O(sqrt(eps)) angle error. This is a degree-valued diagnostic, not metres.
    angle_roundoff_deg = np.rad2deg(np.sqrt(8 * np.finfo(float).eps))
    np.testing.assert_allclose(legacy[3], 0.0, atol=angle_roundoff_deg)
    np.testing.assert_allclose(expanded[0], legacy[0], atol=1e-12)
    np.testing.assert_allclose(expanded[1:3], legacy[1:3], atol=1e-12)
    np.testing.assert_allclose(expanded[3], legacy[3], atol=angle_roundoff_deg)
