import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from agents.core.common import quat_to_rot_wxyz, rot_to_quat_wxyz


@pytest.mark.parametrize("axis", np.eye(3))
def test_half_turn_has_stable_positive_axis(axis):
    matrix = 2 * np.outer(axis, axis) - np.eye(3)
    np.testing.assert_allclose(rot_to_quat_wxyz(matrix), [0, *axis], atol=1e-12)


def test_batched_rotation_roundtrip_and_component_convention():
    matrices = Rotation.random(12, random_state=42).as_matrix().reshape(2, 6, 3, 3)
    quaternions = rot_to_quat_wxyz(matrices)
    assert quaternions.shape == (2, 6, 4)
    for matrix, quaternion in zip(matrices.reshape(-1, 3, 3), quaternions.reshape(-1, 4)):
        np.testing.assert_allclose(quat_to_rot_wxyz(quaternion), matrix, atol=1e-10)
        assert quaternion[np.argmax(np.abs(quaternion))] >= 0


def test_empty_and_invalid_matrix_shapes():
    assert rot_to_quat_wxyz(np.empty((0, 3, 3))).shape == (0, 4)
    with pytest.raises(ValueError, match="shape"):
        rot_to_quat_wxyz(np.eye(4))
