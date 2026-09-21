import numpy as np
import pytest
mujoco=pytest.importorskip('mujoco')
from robo.roundtrip.adapters.integration_state import remap_integration_state


def model(child=True,joint='obj_joint'):
    nested='<body name="fixed_child"><geom size=".01" pos=".1 0 0"/></body>' if child else ''
    return mujoco.MjModel.from_xml_string(f'<mujoco><worldbody><body name="obj"><freejoint name="{joint}"/><geom size=".03"/>{nested}</body><body name="robot"><joint name="robot_joint"/><geom size=".1"/></body></worldbody><actuator><motor joint="robot_joint"/></actuator></mujoco>')


def state(m):
    d=mujoco.MjData(m);d.time=1.2;d.qvel[:]=np.arange(m.nv)*.01
    d.qacc_warmstart[:]=np.arange(m.nv)*.02;d.ctrl[:]=.3;d.qfrc_applied[:]=.4
    d.xfrc_applied[-1]=np.arange(6)
    a=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m,d,a,mujoco.mjtState.mjSTATE_INTEGRATION)
    return a,d


def test_fixed_child_removal_preserves_named_robot_force_and_all_dynamic_fields():
    a,b=model(),model(False);s,d=state(a)
    result=remap_integration_state(a,b,s);out=mujoco.MjData(b)
    mujoco.mj_setState(b,out,result,mujoco.mjtState.mjSTATE_INTEGRATION)
    for field in ['qpos','qvel','qacc_warmstart','ctrl','qfrc_applied','act','eq_active']:
        np.testing.assert_array_equal(getattr(out,field),getattr(d,field))
    assert out.time==d.time
    np.testing.assert_array_equal(out.xfrc_applied[-1],d.xfrc_applied[-1])


def test_identical_model_integration_is_bit_exact():
    a=model();s,_=state(a)
    np.testing.assert_array_equal(remap_integration_state(a,a,s),s)


def test_nonzero_removed_body_force_rejected():
    a,b=model(),model(False);s,d=state(a);d.xfrc_applied[2,0]=1
    mujoco.mj_getState(a,d,s,mujoco.mjtState.mjSTATE_INTEGRATION)
    with pytest.raises(ValueError,match='nonzero external force'):remap_integration_state(a,b,s)


def test_joint_identity_drift_rejected():
    a,b=model(),model(False,'different');s,_=state(a)
    with pytest.raises(ValueError,match='dynamic topology'):remap_integration_state(a,b,s)
