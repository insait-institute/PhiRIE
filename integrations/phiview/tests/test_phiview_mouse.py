"""Camera gestures, explicit placement and policy presentation contracts."""
from types import SimpleNamespace
import numpy as np
import pytest

from physicalview.phiview import Demo
from physicalview.phiview_scene import FlyCamera
from physicalview.phiview_navigation import navigate
from physicalview.phiview_drag import ObjectDrag, plane_point


def test_orbit_and_zoom_keep_the_target_and_pan_translates_it():
    camera = FlyCamera(np.array([0., -2., 1.]), .7, -.2, orbit_distance=3.)
    target = camera.position+camera.forward()*3
    navigate(camera, {'orbit': [.1, .05]})
    np.testing.assert_allclose(camera.position+camera.forward()*camera.orbit_distance, target)
    assert camera.orbit_distance == 3
    before = camera.position.copy()
    navigate(camera, {'zoom': -.3})
    assert .05 < camera.orbit_distance < 3 and not np.allclose(before, camera.position)
    np.testing.assert_allclose(camera.position+camera.forward()*camera.orbit_distance, target)
    forward = camera.forward().copy()
    navigate(camera, {'pan': [.1, -.1]})
    np.testing.assert_allclose(camera.forward(), forward)
    assert not np.allclose(camera.position+camera.forward()*camera.orbit_distance, target)


@pytest.mark.parametrize('message', [{'orbit': [float('nan'), 0]}, {'pan': [1, 2, 3]}, {'zoom': float('inf')}])
def test_bad_gesture_does_not_change_camera(message):
    camera = FlyCamera(np.zeros(3), 0, 0)
    with pytest.raises(ValueError, match='gesture'):
        navigate(camera, message)
    np.testing.assert_array_equal(camera.position, np.zeros(3))
    assert camera.orbit_distance is None


def test_navigation_uses_visible_depth_and_limits_zoom_and_pitch():
    camera = FlyCamera(np.zeros(3), 0, 0)
    navigate(camera, {}, np.full((20, 20), 4.))
    assert camera.orbit_distance == 4.
    for _ in range(30):navigate(camera, {'zoom': -1, 'orbit': [0, 1]})
    assert camera.orbit_distance == .05 and -np.pi/2 < camera.pitch < np.pi/2
    d = Demo.__new__(Demo);d.camera_view = 'wrist'
    assert d.execute({'op': 'navigate', 'zoom': 1})['fixed_camera']


@pytest.mark.parametrize('active,running,plan,expected', [
    (True, False, [], True), (False, True, ['action'], True),
    (False, False, ['action'], False), (False, True, [], False)])
def test_policy_highlight_suppression_keeps_selection_and_user_preference(active, running, plan, expected):
    d=Demo.__new__(Demo);d.policy=SimpleNamespace(active=active)
    d.physics=SimpleNamespace(running=running,plan=plan)
    d.selected='bottle';d.highlight=True
    assert d.highlights_suppressed() is expected
    assert d.selected=='bottle' and d.highlight


def test_manual_drag_changes_only_xy_and_ignores_late_packets(tmp_path):
    from physicalview.phiview_sim import DemoPhysics
    xml=tmp_path/'source.xml'
    xml.write_text('<mujoco><worldbody><body name="obj_00" pos="0 0 .3"><freejoint/>'
                   '<geom type="sphere" size=".1" mass=".2"/></body></worldbody></mujoco>')
    state=SimpleNamespace(scene_xml=xml,objects={'obj_00':SimpleNamespace(meta={},physics=None)})
    sim=DemoPhysics(state,tmp_path)
    camera=FlyCamera(np.array([0.,-2.,1.3]),np.pi/2,-np.arctan(.5))
    w2c,K=camera.matrices((100,100))
    d=SimpleNamespace(selected_required=lambda:'obj_00',physics=sim,scene=SimpleNamespace(ids={'obj_00':1}),
        frames={1:(np.ones((100,100),int),None,w2c,K)},policy=SimpleNamespace(stop=lambda:None),look=np.zeros(2))
    drag=ObjectDrag(d);token=drag.begin(1,[.5,.5])
    before=sim.data.qpos.copy();drag.move(token,[.6,.5])
    assert sim.data.qpos[0] != before[0] and sim.data.qpos[2]==pytest.approx(before[2])
    np.testing.assert_array_equal(sim.data.qpos[3:7],before[3:7])
    assert not sim.running and np.all(sim.data.qvel==0)
    drag.end(token);after=sim.data.qpos.copy();drag.move(token,[.4,.5])
    np.testing.assert_array_equal(sim.data.qpos,after)
    with pytest.raises(ValueError,match='outside'):
        plane_point(w2c,K,(100,100),[float('nan'),.5],.3)
