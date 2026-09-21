import numpy as np

from robo.rendering.robot_restore import restore_robot_core, robot_restore_alpha


def test_robot_core_is_byte_exact_and_outside_is_enhanced():
    raw = np.zeros((25, 25, 3), dtype=np.uint8)
    raw[..., 0] = 10
    enhanced = np.full_like(raw, 200)
    mask = np.zeros((25, 25), dtype=np.uint8)
    mask[7:18, 8:17] = 255
    output, stats = restore_robot_core(raw, enhanced, mask, erode_px=1, dilate_px=3)
    alpha, core, dilated = robot_restore_alpha(mask >= 128, erode_px=1, dilate_px=3)
    assert stats.core_equal
    assert np.array_equal(output[core], raw[core])
    assert np.array_equal(output[~dilated], enhanced[~dilated])
    assert ((alpha > 0) & (alpha < 1)).any()
