import numpy as np
import pytest
from robo.roundtrip.robot_capsule import export_robot


def test_exports_only_known_robot_qpos_and_no_room_geometry(tmp_path):
    mujoco=pytest.importorskip('mujoco')
    robot='<mujoco><worldbody><body name="robot0_base"><joint name="robot0_j" type="slide" axis="1 0 0"/><geom name="robot0_geom" type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    native=robot.replace('</worldbody>','<geom name="private_table" type="box" size="1 1 .1" pos="0 0 -1"/></worldbody>')
    model=mujoco.MjModel.from_xml_string(native);data=mujoco.MjData(model);data.qpos[0]=.2;mujoco.mj_forward(model,data)
    r=export_robot(robot,model,data,tmp_path/'capsule',asset_roots=[])
    assert r['joint_qpos']=={'robot0_j':[.2]} and r['robot_geoms']==1
    assert 'private_table' not in (tmp_path/'capsule/robot.xml').read_text()
    assert r['max_static_geometry_pose_error']==0
    with pytest.raises(FileExistsError):export_robot(robot,model,data,tmp_path/'capsule',asset_roots=[])


def test_missing_joint_or_external_xml_is_rejected(tmp_path):
    mujoco=pytest.importorskip('mujoco')
    xml='<mujoco><worldbody><body name="robot0_base"><geom name="robot0_geom" type="sphere" size=".1"/></body></worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    with pytest.raises(ValueError,match='external XML'):
        export_robot(xml.replace('<worldbody>','<include file="private.xml"/><worldbody>'),model,data,tmp_path/'x',asset_roots=[])
    with pytest.raises(ValueError,match='joint identity'):
        export_robot(xml.replace('<geom','<joint name="other"/><geom'),model,data,tmp_path/'y',asset_roots=[])


def test_arbitrary_asset_paths_rejected_before_model_compile(tmp_path):
    mujoco=pytest.importorskip('mujoco')
    xml='<mujoco><worldbody><body name="robot0_base"><geom name="robot0_geom" type="sphere" size=".1"/></body></worldbody></mujoco>'
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    private=tmp_path/'room.obj';private.write_text('hidden room')
    changed=xml.replace('<worldbody>',f'<asset><mesh name="private" file="{private}"/></asset><worldbody>')
    with pytest.raises(ValueError,match='embodiment asset roots'):
        export_robot(changed,model,data,tmp_path/'out',asset_roots=[tmp_path/'robot_assets'])
    assert not (tmp_path/'out').exists()


def test_public_depth_visibility_preserves_occluded_and_unknown_pixels():
    from robo.roundtrip.robot_capsule import public_depth_visibility
    mask=np.ones((1,4),bool);robot=np.array([[1.,2.,1.,1.]])
    obs=np.array([[1.,1.,2.,0.]])
    visible,occluded,unknown=public_depth_visibility(mask,robot,obs)
    assert visible.tolist()==[[True,False,False,False]]
    assert occluded.tolist()==[[False,True,False,False]]
    assert unknown.tolist()==[[False,False,True,True]]
    with pytest.raises(ValueError,match='calibrated depth'):
        public_depth_visibility(mask,robot,obs[:,:2])


def test_existing_surface_removal_primitive_exact_union():
    from scipy.spatial import cKDTree
    from agents.edit.inpaint_prepare import surface_removal_indices
    xyz=np.array([[0.,0.,0.],[.02,0.,0.],[1.,0.,0.]])
    actual=surface_removal_indices(cKDTree(xyz),np.array([[0.,0.,0.],[.01,0.,0.]]))
    assert actual.tolist()==[0,1]
    assert surface_removal_indices(cKDTree(xyz),np.zeros((0,3))).tolist()==[]
