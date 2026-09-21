import json
import numpy as np
import pytest
from robo.roundtrip.observations import named_joint_state,render


def test_named_robot_only_state_excludes_foreign_object():
    mujoco=pytest.importorskip('mujoco')
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="robot"><joint name="robot_joint"/><geom type="sphere" size=".1"/></body><body name="hidden_object"><freejoint name="hidden_joint"/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    q=np.zeros(model.nq);q[0]=.2
    assert named_joint_state(model,q,['robot_joint'])=={'robot_joint':[.2]}
    with pytest.raises(ValueError,match='absent'):named_joint_state(model,q,['missing'])
    with pytest.raises(ValueError,match='shape'):named_joint_state(model,q[:-1],['robot_joint'])
    q[0]=np.nan
    with pytest.raises(ValueError,match='values'):named_joint_state(model,q,['robot_joint'])


def test_render_rejects_changed_runtime_receipt_before_gpu(tmp_path):
    pytest.importorskip('mujoco')
    p=tmp_path/'states.json';p.write_text('{}')
    config={'states':str(p),'states_sha256':'wrong','capsule':str(tmp_path),'background':str(p),'object':str(tmp_path)}
    with pytest.raises(ValueError,match='states identity changed'):render(config,tmp_path/'out')
    assert not (tmp_path/'out').exists()
