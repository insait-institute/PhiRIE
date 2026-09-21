import numpy as np
import pytest
from robo.rendering.pi05_render import depth_test_robot


def test_robot_front_behind_unknown_and_invalid_depth():
    rgb=np.zeros((1,5,3),dtype=np.uint8)
    robot=np.full_like(rgb,213)
    depth=np.array([[2.,2.,2.,np.nan,2.]])
    alpha=np.array([[1.,1.,.1,1.,1.]])
    rz=np.array([[1.,3.,1.,1.,np.nan]])
    out,e=depth_test_robot(rgb,depth,alpha,robot,rz,np.ones((1,5),bool))
    assert np.array_equal(out[0,0],robot[0,0])
    assert not out[0,1:].any()
    assert e['robot_visible'].tolist()==[[True,False,False,False,False]]
    assert e['robot_occluded'].tolist()==[[False,True,False,False,False]]
    assert e['unknown_coverage'].tolist()==[[False,False,True,True,True]]


def test_nonrobot_never_pasted_and_equal_surface_is_visible():
    rgb=np.zeros((1,2,3),dtype=np.uint8);robot=np.full_like(rgb,17)
    d=np.ones((1,2));mask=np.array([[True,False]])
    out,e=depth_test_robot(rgb,d,d,robot,d,mask)
    assert (out[0,0]==17).all() and not out[0,1].any()
    assert not e['unknown_coverage'].any()


def test_different_camera_grids_and_nonbyte_rgb_fail_closed():
    rgb=np.zeros((2,3,3),dtype=np.uint8);d=np.ones((2,3))
    with pytest.raises(ValueError,match='calibrated pixel grid'):
        depth_test_robot(rgb,d,d,rgb,d[:,1:],d)
    with pytest.raises(ValueError,match='byte RGB'):
        depth_test_robot(rgb.astype(float),d,d,rgb,d,d)
    with pytest.raises(ValueError,match='opacity threshold'):
        depth_test_robot(rgb,d,d,rgb,d,d,alpha_min=0)


def test_existing_composite_producer_uses_depth_and_restores_renderer_mode(monkeypatch):
    from robo.rendering import pi05_render
    from types import SimpleNamespace
    mujoco=pytest.importorskip('mujoco')
    class Renderer:
        mode='rgb'
        def update_scene(self,*args,**kwargs):pass
        def render(self):
            if self.mode=='depth':return np.array([[1.,3.]])
            if self.mode=='seg':return np.array([[[7,int(mujoco.mjtObj.mjOBJ_GEOM)],[7,int(mujoco.mjtObj.mjOBJ_GEOM)]]])
            return np.full((1,2,3),201,np.uint8)
        def enable_depth_rendering(self):self.mode='depth'
        def disable_depth_rendering(self):self.mode='rgb'
        def enable_segmentation_rendering(self):self.mode='seg'
        def disable_segmentation_rendering(self):self.mode='rgb'
    obj=pi05_render.CompositeObs.__new__(pi05_render.CompositeObs)
    obj.env=SimpleNamespace(data=None);obj.bg={};obj.canon={}
    obj.W=2;obj.H=1;obj.scale=1.;obj.robot_compositing='expected_depth_v1';obj.alpha_min=.95
    obj._cam_K_w2c=lambda _: (np.eye(3),np.eye(4));obj._robot_geoms={7};obj.mj_renderer=Renderer()
    monkeypatch.setattr(pi05_render.C,'cat_gaussians',lambda x:x)
    def render(*args,**kw):
        assert kw['render_mode']=='RGB+ED'
        return np.zeros((1,2,3)),np.full((1,2),2.),np.ones((1,2))
    monkeypatch.setattr(pi05_render.C,'render_view',render)
    out=obj.render('fixed_camera')
    assert (out[0,0]==201).all() and not out[0,1].any()
    assert obj.mj_renderer.mode=='rgb'
    assert obj.last_visibility['robot_occluded'].tolist()==[[False,True]]
