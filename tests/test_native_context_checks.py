import json
import numpy as np
import pytest
from robo.roundtrip.native_context_checks import context_checks, native_destination, surface_residual
from robo.roundtrip.system_verification import CONTEXT_CONFIG, validate_config


@pytest.fixture
def scene(tmp_path):
    import mujoco
    import trimesh
    from robo.roundtrip.importers.robocasa import import_reconstructed_object
    asset=tmp_path/'asset';asset.mkdir();(asset/'collision').mkdir()
    mesh=trimesh.creation.box([.02,.02,.02]);mesh.export(asset/'mesh_sim.obj');mesh.export(asset/'collision/part_00.obj')
    T=np.eye(4);T[2,3]=.02
    (asset/'aligned.json').write_text(json.dumps({'T':T.tolist(),'scale':1.}))
    (asset/'physics.json').write_text(json.dumps({'mass_kg':.1,'friction':.5}))
    xml='''<mujoco><option timestep=".002"/><worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <site name="gripper0_right_grip_site" pos="-.2 0 .2"/>
      <body name="obj_main"><freejoint name="obj_joint"/><geom type="box" size=".01 .01 .01"/></body>
      <body name="container_main" pos=".4 0 .05"><geom name="plate" type="box" size=".08 .08 .01"/></body>
      </worldbody></mujoco>'''
    xml,receipt=import_reconstructed_object(xml,body_name='obj_main',object_dir=asset,object_id='target')
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    return xml,model,data,receipt,asset


def test_actual_native_destination_contact_and_free_corridors(scene):
    xml,m,d,receipt,_=scene
    import mujoco
    kind=mujoco.mjtState.mjSTATE_INTEGRATION
    before=np.empty(mujoco.mj_stateSize(m,kind));mujoco.mj_getState(m,d,before,kind)
    report=context_checks(xml,m,d,receipt,'PickPlaceSinkToCounter',{},CONTEXT_CONFIG)
    assert report['passed'] and report['destination_access']['final_destination_contact']
    assert report['oracle_context'] and not report['native_task_success_accessed']
    after=np.empty_like(before);mujoco.mj_getState(m,d,after,kind)
    assert before.tobytes()==after.tobytes()
    assert report["destination_access"]["probe_geometry"]=="actual_reconstructed_target_CoACD"


def test_obstacle_blocks_approach_proxy(scene):
    import mujoco
    xml,_,_,receipt,_=scene
    xml=xml.replace('<worldbody>','<worldbody><geom name="obstacle" type="box" pos="-.1 0 .19" size=".025 .025 .025"/>')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    report=context_checks(xml,m,d,receipt,'PickPlaceSinkToCounter',{},CONTEXT_CONFIG)
    assert not report['corridors']['approach']['passed']
    assert any(r['geom']=='obstacle' for r in report['corridors']['approach']['collisions'])


def test_closed_sink_cavity_fails_access(scene):
    import mujoco
    xml,_,_,receipt,_=scene
    import xml.etree.ElementTree as ET
    root=ET.fromstring(xml);world=root.find('worldbody')
    world.remove(world.find("body[@name='container_main']"))
    sink=ET.SubElement(world,'body',name='sink_main',pos='.4 0 .05')
    ET.SubElement(sink,'geom',name='sink_reg_basin',type='box',size='.08 .08 .05',contype='0',conaffinity='0')
    ET.SubElement(sink,'geom',name='sink_sealed',type='box',size='.09 .09 .10')
    xml=ET.tostring(root,encoding='unicode')
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    report=context_checks(xml,m,d,receipt,'PickPlaceCounterToSink',{'sink':'sink'},CONTEXT_CONFIG)
    assert not report['destination_access']['passed']
    with pytest.raises(ValueError):native_destination(m,d,'UNKNOWN',{})


def test_public_residual_detects_pose_damage_without_native_data(scene,tmp_path):
    import shutil,trimesh
    _,_,_,_,asset=scene
    mesh=trimesh.load(asset/'mesh_sim.obj',process=False);points,_=trimesh.sample.sample_surface(mesh,20000,seed=42);points[:,2]+=.02
    base=surface_residual(asset,points)
    changed=tmp_path/'changed';shutil.copytree(asset,changed)
    aligned=json.loads((changed/'aligned.json').read_text());aligned['T'][0][3]+=.04
    (changed/'aligned.json').write_text(json.dumps(aligned))
    assert surface_residual(changed,points)-base>.005
    validate_config(CONTEXT_CONFIG)


def test_public_points_hash_binding_and_no_gt_metric_load(scene,tmp_path,monkeypatch):
    from robo.roundtrip.native_context_checks import observation_points
    from agents.orchestrator.artifact import sha256_file
    from agents.core import common
    monkeypatch.setattr(common,'load_gt_instances',lambda:(_ for _ in ()).throw(AssertionError('GT loader accessed')))
    *_,asset=scene
    surface_residual(asset,np.array([[0,0,.02],[.01,0,.02],[0,.01,.02]]))
    capture=tmp_path/'capture';capture.mkdir();(capture/'capture_manifest.json').write_text('{}')
    build=tmp_path/'build';(build/'discovery').mkdir(parents=True)
    points=build/'discovery/observation_points.npy';np.save(points,np.zeros((4,3)))
    manifest={'capture_manifest_sha256':sha256_file(capture/'capture_manifest.json'),
              'source_hashes':{'discovery/observation_points.npy':sha256_file(points)}}
    (build/'build_manifest.json').write_text(json.dumps(manifest))
    assert observation_points(build,capture)[0].shape==(4,3)
    np.save(points,np.ones((4,3)))
    with pytest.raises(ValueError,match='bytes changed'):observation_points(build,capture)
