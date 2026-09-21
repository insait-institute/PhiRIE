import numpy as np
from robo.roundtrip.workspace_coverage import sample_depth


def test_depth_visibility_does_not_turn_occlusion_into_free_space():
    points=np.array([[0,0,.5],[0,0,1.],[0,0,1.5],[0,0,-1.],[10,0,.5]])
    free,surface,_=sample_depth(points,np.ones((3,3)),np.array([[1,0,1],[0,1,1],[0,0,1]]),np.eye(4),.02)
    assert free.tolist()==[True,False,False,False,False]
    assert surface.tolist()==[False,True,False,False,False]


def test_camera_world_transform_and_missing_depth():
    T=np.eye(4);T[:3,3]=[2,3,4]
    points=np.array([[2,3,5],[2,3,4.5]])
    free,surface,_=sample_depth(points,np.ones((1,1)),np.eye(3),T,.02)
    assert free.tolist()==[False,True] and surface.tolist()==[True,False]
    for invalid in (0.,np.nan,np.inf):
        free,surface,_=sample_depth(points,np.full((1,1),invalid),np.eye(3),T,.02)
        assert not free.any() and not surface.any()
