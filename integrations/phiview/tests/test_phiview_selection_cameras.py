"""Selection gestures and real rig/camera/action compatibility regressions."""
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from physicalview.highlight import highlight_objects
from physicalview.phiview import Demo
from physicalview.phiview_box import pixel_box, known_object_in_box
from physicalview.phiview_point_segment import choose_mask
from physicalview.phiview_policy import apply_action, validate_actions
from physicalview.phiview_rigs import build_robot, RobotCamera
from physicalview.phiview_selection import restore_objects
from physicalview.phiview_sim import preserve_supports, DemoPhysics


def test_selected_object_is_red_when_other_highlights_are_off():
    rgb = np.full((40, 40, 3), .3, dtype=np.float32)
    labels = np.zeros((40, 40), np.uint16)
    labels[5:15, 5:15] = 1
    labels[25:35, 25:35] = 2
    out = highlight_objects(rgb, labels, 2, show_all=False)
    assert out[30, 30, 0] > .9 and out[30, 30, 1] < .08
    np.testing.assert_array_equal(out[5:15, 5:15], rgb[5:15, 5:15])


def test_box_normalization_rejects_invalid_and_ambiguous_selection():
    assert pixel_box([.8, .9, .2, .1], (100, 200)) == [40, 10, 160, 90]
    for value in ([0, 0, .001, .001], [-.1, 0, 1, 1], [0, 0, np.nan, 1], [0, 1]):
        with pytest.raises(ValueError):
            pixel_box(value, (100, 200))
    labels = np.zeros((20, 20), np.uint16)
    labels[3:10, 3:10] = 1
    assert known_object_in_box(labels, (3, 3, 10, 10)) == 1
    assert known_object_in_box(labels, (0, 0, 20, 20)) is None
    labels[10:17, 3:10] = 2
    assert known_object_in_box(labels, (3, 3, 10, 17)) is None


def test_box_prompt_can_select_hollow_object_without_mask_at_center():
    mask = np.zeros((30, 30), bool)
    mask[5:15, 5:15] = True
    mask[8:12, 8:12] = False
    chosen, _ = choose_mask([mask], [.9], 10, 10, (4, 4, 16, 16))
    np.testing.assert_array_equal(chosen, mask)
    with pytest.raises(ValueError):
        choose_mask([mask], [.9], 10, 10)
    with pytest.raises(ValueError):
        choose_mask([mask], [.9], 10, 10, (-1, 4, 16, 16))


def test_box_mask_drops_disconnected_distractors():
    mask = np.zeros((60, 60), bool)
    mask[25:35, 25:35] = True
    mask[40:55, 20:50] = True
    mask[5:10, 5:10] = True
    selected, _ = choose_mask([mask], [.9], 30, 30, (15, 15, 55, 55))
    assert selected.sum() == 100 and selected[30, 30] and not selected[45, 30]


def test_box_routes_unknown_region_to_sam_and_rejects_stale_frame():
    calls = []
    d = Demo.__new__(Demo)
    d.frames = {7: (np.zeros((100, 100), int), None, None, None)}
    d.frame_id = 7
    d.click_selection = SimpleNamespace(start=lambda *args, **kwargs: calls.append((args, kwargs)))
    d.physics = SimpleNamespace(data=SimpleNamespace(time=0))
    d.selected = None
    d.audit = io.StringIO()
    d.execute({'op': 'box_select', 'frame': 7, 'box': [.2, .3, .6, .8]})
    assert calls == [((7, 40, 55), {'box': [20, 30, 60, 80]})]
    with pytest.raises(ValueError, match='expired'):
        d.execute({'op': 'box_select', 'frame': 6, 'box': [.2, .3, .6, .8]})


def test_legacy_proxy_does_not_attach_to_rediscovered_id(tmp_path):
    directory = tmp_path/'interactive_objects'/'obj_10'
    directory.mkdir(parents=True)
    (directory/'selection.json').write_text(json.dumps({'name': 'obj_10', 'scene_splat': '/scene.ply'}))
    (directory/'proxy.json').write_text(json.dumps({'physics': {'mass_kg': 99}}))
    rec = SimpleNamespace(meta={}, physics=None)
    state = SimpleNamespace(result_set=SimpleNamespace(splat_ply='/scene.ply', out_dir=tmp_path/'new-build'),
                            objects={'obj_10': rec})
    restore_objects(state, tmp_path)
    assert rec.physics is None and not rec.meta


@pytest.mark.parametrize('name', ['droid', 'panda'])
def test_rig_cameras_are_physical_and_gripper_semantics_match(name, tmp_path):
    mujoco = pytest.importorskip('mujoco')
    pytest.importorskip('robo.rigs.pi05_rig')
    source = tmp_path/'scene.xml'
    source.write_text('<mujoco><worldbody/></mujoco>')
    model, info = build_robot(name, source, [1, 2, .5], .4)
    data = mujoco.MjData(model)
    qadr = np.array([model.joint(j).qposadr[0] for j in info['arm_joints']])
    aids = np.array([model.actuator(a).id for a in info['arm_actuators']])
    data.qpos[qadr] = info['home']
    mujoco.mj_forward(model, data)
    physics = SimpleNamespace(model=model, data=data, robot={
        'info': info, 'model': name, 'qadr': qadr, 'aids': aids,
        'grip': model.actuator(info['gripper_actuator']).id})
    exterior, K = RobotCamera(physics, 'exterior').matrices((1280, 720))
    wrist, _ = RobotCamera(physics, 'wrist').matrices((1280, 720))
    assert K[0, 2] == 640 and K[1, 2] == 360
    assert K[1, 1] == pytest.approx(720/(2*np.tan(np.deg2rad(68)/2)))
    data.qpos[qadr[0]] += .3
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(RobotCamera(physics, 'exterior').matrices((1280, 720))[0], exterior)
    assert not np.allclose(RobotCamera(physics, 'wrist').matrices((1280, 720))[0], wrist)
    if name == 'droid':
        q = data.qpos[qadr].copy()
        apply_action(physics, [10]*7 + [1])
        assert np.max(np.abs(data.ctrl[aids]-q)) <= .200001
        assert data.ctrl[physics.robot['grip']] == 255
        apply_action(physics, [*q, 0])
        assert data.ctrl[physics.robot['grip']] == 0
    else:
        assert info['grip_open'] == 255 and info['grip_closed'] == 0
        with pytest.raises(ValueError, match='DROID'):
            apply_action(physics, [0]*8)


def test_policy_action_contract_rejects_bad_chunks():
    assert validate_actions(np.zeros((15, 8))).shape == (15, 8)
    for value in (np.zeros((0, 8)), np.zeros((15, 7)), np.full((15, 8), np.nan), np.ones((15, 8))*2):
        with pytest.raises(ValueError):
            validate_actions(value)


def test_rediscovery_preserves_supports_without_old_dynamic_object_ids(tmp_path):
    export = tmp_path/'old'/'sim_export'
    export.mkdir(parents=True)
    (export/'scene.xml').write_text('''<mujoco><worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <geom name="table" type="box" size=".5 .5 .05" pos="0 0 .7"/>
      <body name="obj_00"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')
    result = SimpleNamespace(out_dir=tmp_path/'old', splat_ply=tmp_path/'original.ply')
    preserve_supports(result, tmp_path)
    result.out_dir = tmp_path/'new'
    state = SimpleNamespace(result_set=result, scene_xml=None, objects={})
    sim = DemoPhysics(state, tmp_path)
    assert sim.model.geom('floor').id >= 0 and sim.model.geom('table').id >= 0
    assert not sim.available
    assert 'obj_00' not in [sim.model.body(i).name for i in range(sim.model.nbody)]


def test_stopped_inference_cannot_resume_physics():
    from concurrent.futures import Future
    from physicalview.phiview_policy import LearnedPolicy
    policy = LearnedPolicy.__new__(LearnedPolicy)
    policy.active = True
    policy.actions = [np.ones(8)]
    policy.epoch = 2
    policy.status = {'state': 'inferencing'}
    policy.demo = SimpleNamespace(physics=SimpleNamespace(running=True))
    policy.future = Future()
    policy.stop()
    policy.future.set_result((2, np.ones((15, 8))))
    assert not policy.advance()
    assert policy.epoch == 3 and not policy.actions and not policy.demo.physics.running
