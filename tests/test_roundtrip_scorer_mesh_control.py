import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from robo.roundtrip.scorer_mesh_control import compare_compiled,reexpress_xml


def test_real_noncentered_mesh_known_reexpression(tmp_path):
    mujoco=pytest.importorskip('mujoco');trimesh=pytest.importorskip('trimesh')
    mesh=trimesh.creation.box([.2,.3,.4]);mesh.apply_translation([.03,.05,.1]);mesh.export(tmp_path/'mesh.obj')
    xml=f'<mujoco><asset><mesh name="m" file="{tmp_path / "mesh.obj"}"/></asset><worldbody><body name="obj" pos="1 2 3"><freejoint/><inertial pos=".03 .05 .1" mass=".3" diaginertia=".01 .02 .03"/><geom type="mesh" mesh="m" group="1"/></body></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    B=np.eye(4);B[:3,3]=[.1,-.2,.03];B[:3,:3]=Rotation.from_euler('xyz',[.1,.3,.7]).as_matrix()
    report,changed,inverse=compare_compiled(xml,'obj',d,B)
    assert report['passed'] and not report['mesh_assets_modified']
    np.testing.assert_allclose(changed.xpos[1]+changed.xmat[1].reshape(3,3)@inverse[:3,3],d.xpos[1],atol=1e-14)


def test_reject_nonfree_or_implicit_orientation():
    with pytest.raises(ValueError,match='one free joint'):
        reexpress_xml('<mujoco><worldbody><body name="obj"><joint/></body></worldbody></mujoco>','obj',np.eye(4))
