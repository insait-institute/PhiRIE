import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from robo.roundtrip.gaussian_motion import asset_world_transform,body_matrix


def test_initial_static_exact_and_scale_once():
    pose=[1,2,3,1,0,0,0];T=np.eye(4);T[:3,:3]*=.3;T[:3,3]=[1.1,2,3]
    assert np.array_equal(asset_world_transform(T,pose,pose),T)
    moved=asset_world_transform(T,pose,[1.2,2,3,1,0,0,0])
    assert np.allclose(moved[:3,3],[1.3,2,3]) and np.allclose(moved[:3,:3],np.eye(3)*.3)


def test_rotation_uses_body_origin_and_keeps_asset_offset():
    T=np.eye(4);T[:3,3]=[1,0,0];q=Rotation.from_euler('z',90,degrees=True).as_quat()[[3,0,1,2]]
    moved=asset_world_transform(T,[0,0,0,1,0,0,0],[0,0,0,*q])
    assert np.allclose(moved[:3,3],[0,1,0],atol=1e-12)
    assert np.allclose(np.linalg.inv(body_matrix([0,0,0,*q]))@moved,T)


def test_invalid_pose_and_nonuniform_scale_rejected():
    with pytest.raises(ValueError,match='unit quaternion'):body_matrix([0,0,0,0,0,0,0])
    T=np.diag([1,2,1,1])
    with pytest.raises(ValueError,match='exactly once'):asset_world_transform(T,[0,0,0,1,0,0,0],[0,0,0,1,0,0,0])
