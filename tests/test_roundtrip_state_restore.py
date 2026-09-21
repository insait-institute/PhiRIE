"""State-import contract regressions; no native assets or renderer required."""
import importlib.util
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest


@pytest.fixture
def adapter_module():
    override = os.environ.get('ROUNDTRIP_ADAPTER_REVIEW_PATH')
    if override:
        spec = importlib.util.spec_from_file_location('adapter_under_review', override)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from robo.roundtrip.adapters import robocasa
    return robocasa


@pytest.fixture
def setup_adapter(monkeypatch, adapter_module):
    events = []
    data = NS(qpos=np.array([2., 3.]), qvel=np.array([.2, .3]), act=np.array([.4]), time=1.25)
    data._data = data
    data.set_joint_qpos = lambda name, pose: (events.append(('pose', name)), setattr(data, 'replacement', pose.copy()))
    model = NS(_model=object(), dynamics=7.)
    sim = NS(model=model, data=data)
    sim.get_state = lambda: np.r_[data.time, data.qpos, data.qvel, data.act]
    sim.forward = lambda: events.append('forward')

    class Controller:
        def __init__(self):
            self.initial_joint = np.array([.1, .2])
            self.initial_ref_pos = np.array([.3, .4, .5])
            self.goal_pos = np.array([.6, .7, .8])
            self.origin_pos = np.array([.9, 1., 1.1])
            self.init_ori = np.eye(3)
            self._goal_update_mode = 'desired'
            self.summed_err = np.array([.12, .13])
            self.derr_buf = NS(buf=np.arange(10.).reshape(5, 2), ptr=2, _size=3)
            self.new_update = False
            self.joint_pos = np.array([-5., -5.])
            self.mass_matrix = np.eye(2) * 1000

        def update(self, force=False):
            if self.new_update or force:
                events.append(('controller_update', force))
                self.joint_pos = data.qpos.copy()
                self.mass_matrix = np.eye(2) * model.dynamics
                self.new_update = False

    controller = Controller()
    gripper = NS(current_action=np.array([.25]))
    robot = NS(composite_controller=NS(part_controllers={'right':controller}), gripper={'right':gripper})
    native = NS(sim=sim, robots=[robot], timestep=25, cur_time=1.25,
                model=NS(get_xml=lambda:'<mujoco full_precision="0.12345678912345678"/>'))
    native._observables = {}
    native.get_ep_meta = lambda: {'seed':0}
    native.set_ep_meta = lambda meta: events.append(('metadata', meta))
    native.reset = lambda: events.append('metadata_reset')

    def reset_xml(xml):
        events.append(('reset', xml))
        data.qpos[:] = -10
        data.qvel[:] = -20
        controller.initial_joint[:] = -99
        controller.goal_pos[:] = -99
        controller.origin_pos[:] = -99
        controller.summed_err[:] = -99
        controller.derr_buf.buf[:] = -99
        controller.derr_buf.ptr = 0
        controller.derr_buf._size = 0
        gripper.current_action[:] = -99
        controller.mass_matrix[:] = 9999
    native.reset_from_xml_string = reset_xml
    adapter = adapter_module.RoboCasaAdapter({})
    adapter.env = NS(unwrapped=NS(env=native))
    adapter.tracked_objects = lambda: {'obj':[0, 0, 0, 1, 0, 0, 0]}
    adapter.refresh_observation = lambda: events.append('observation')

    def get_state(model, data, out, mask):
        assert mask == 123
        out[:] = np.r_[data.time, data.qpos, data.qvel, data.act]

    def set_state(model, data, value, mask):
        assert mask == 123
        events.append('integration')
        data.time = value[0]
        data.qpos[:] = value[1:3]
        data.qvel[:] = value[3:5]
        data.act[:] = value[5:6]

    monkeypatch.setitem(sys.modules, 'mujoco', NS(mjtState=NS(mjSTATE_INTEGRATION=123),
        mj_stateSize=lambda model, mask:6, mj_getState=get_state, mj_setState=set_state))
    return adapter, controller, gripper, events


def test_original_xml_precision_not_serialized_compiled_model(setup_adapter):
    adapter, _, _, _ = setup_adapter
    assert '0.12345678912345678' in adapter.source_xml()


def test_semantic_controller_and_integration_state_survive_xml_import(setup_adapter):
    adapter, controller, gripper, events = setup_adapter
    state = adapter.get_state()
    adapter.import_xml('<mujoco/>', canonical_state=state)
    np.testing.assert_array_equal(adapter.native.sim.data.qpos, [2, 3])
    np.testing.assert_array_equal(adapter.native.sim.data.qvel, [.2, .3])
    np.testing.assert_array_equal(controller.initial_joint, [.1, .2])
    np.testing.assert_array_equal(controller.goal_pos, [.6, .7, .8])
    np.testing.assert_array_equal(controller.origin_pos, [.9, 1., 1.1])
    np.testing.assert_array_equal(controller.summed_err, [.12, .13])
    np.testing.assert_array_equal(controller.derr_buf.buf, np.arange(10.).reshape(5, 2))
    assert (controller.derr_buf.ptr, controller.derr_buf._size) == (2, 3)
    np.testing.assert_array_equal(gripper.current_action, [.25])
    assert controller._goal_update_mode == 'desired'
    assert adapter.native.timestep == 25 and adapter.native.cur_time == 1.25


def test_dynamics_cache_recomputed_from_imported_model_not_copied(setup_adapter):
    adapter, controller, _, events = setup_adapter
    state = adapter.get_state()
    assert 'mass_matrix' not in state['controller_state'][0]['parts']['right']
    adapter.native.sim.model.dynamics = 23.
    adapter.import_xml('<new-geometry/>', canonical_state=state)
    np.testing.assert_array_equal(controller.joint_pos, [2., 3.])
    np.testing.assert_array_equal(controller.mass_matrix, np.eye(2) * 23.)
    assert events.index('integration') < events.index(('controller_update', True)) < events.index('observation')


def test_reconstruction_estimated_pose_preserved_before_dynamics_refresh(setup_adapter):
    adapter, _, _, events = setup_adapter
    pose = [.11, .22, .33, 1, 0, 0, 0]
    adapter.import_xml('<new-geometry/>', canonical_state=adapter.get_state(),
                       replaced_joint='obj_joint', estimated_pose=pose)
    np.testing.assert_array_equal(adapter.native.sim.data.replacement, pose)
    assert events.index(('pose', 'obj_joint')) < events.index(('controller_update', True))


@pytest.mark.parametrize('missing', ['integration_state', 'controller_state'])
def test_incomplete_checkpoint_is_rejected(setup_adapter, missing):
    adapter, _, _, _ = setup_adapter
    state = adapter.get_state()
    del state[missing]
    with pytest.raises(ValueError, match='integration and controller state'):
        adapter.import_xml('<mujoco/>', canonical_state=state)


def test_missing_reconstruction_pose_is_rejected(setup_adapter):
    adapter, _, _, _ = setup_adapter
    with pytest.raises(ValueError, match='estimated pose'):
        adapter.import_xml('<mujoco/>', canonical_state=adapter.get_state(), replaced_joint='obj_joint')


def test_named_multi_pose_hook_preserves_state_and_rejects_topology_drift(setup_adapter):
    adapter, controller, _, events=setup_adapter
    model=adapter.native.sim.model
    model.njnt=2;model.nu=0;model.jnt_type=[0,0];model.jnt_qposadr=[0,7];model.jnt_dofadr=[0,6]
    names=['target_joint','destination_joint'];model.joint_id2name=lambda i:names[i]
    state=adapter.get_state()
    poses={name:[i*.1,0,0,1,0,0,0] for i,name in enumerate(names)}
    adapter.import_xml('<multiple/>',canonical_state=state,replaced_joints=poses)
    assert ('pose','target_joint') in events and ('pose','destination_joint') in events
    np.testing.assert_array_equal(controller.initial_joint,[.1,.2])
    old_reset=adapter.native.reset_from_xml_string
    def reorder(xml):
        old_reset(xml);names.reverse()
    adapter.native.reset_from_xml_string=reorder
    with pytest.raises(ValueError,match='dynamic topology'):
        adapter.import_xml('<reordered/>',canonical_state=state,replaced_joints=poses)


def test_invalid_multi_pose_rejects_before_native_reset(setup_adapter):
    adapter, _, _, events=setup_adapter
    with pytest.raises(ValueError,match='finite xyz/unit'):
        adapter.import_xml('<multiple/>',canonical_state=adapter.get_state(),
                           replaced_joints={'target':[0,0,0,0,0,0,0]})
    assert events==[]
