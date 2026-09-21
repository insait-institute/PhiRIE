import json
import numpy as np
import pytest
from robo.roundtrip.system_verification import CONTEXT_CONFIG,V4_CONFIG,validate_config
from robo.roundtrip.native_context_checks import native_destination,_world_bounds
from robo.roundtrip.cabinet_context import cabinet_access


@pytest.fixture
def cabinet(tmp_path):
    import mujoco,trimesh
    from robo.roundtrip.importers.robocasa import import_reconstructed_object
    root=tmp_path/'asset';(root/'collision').mkdir(parents=True)
    mesh=trimesh.creation.box([.025,.025,.025]);mesh.export(root/'mesh_sim.obj');mesh.export(root/'collision/part_00.obj')
    T=np.eye(4);T[:3,3]=[0,-.4,.6]
    (root/'aligned.json').write_text(json.dumps(dict(T=T.tolist(),scale=1.)))
    (root/'physics.json').write_text(json.dumps(dict(mass_kg=.1,friction=.5)))
    xml='''<mujoco><option timestep=".002"/><worldbody><geom name="floor" type="plane" size="1 1 .1"/>
    <body name="obj_main"><freejoint name="obj_joint"/><geom type="box" size=".01 .01 .01"/></body>
    <body name="cab" pos="0 0 .7"><geom name="cab_reg_level0" type="box" size=".1 .12 .1" contype="0" conaffinity="0"/>
    <geom name="cab_bottom" type="box" pos="0 0 -.11" size=".12 .14 .01"/>
    <geom name="cab_top" type="box" pos="0 0 .11" size=".12 .14 .01"/>
    <geom name="cab_back" type="box" pos="0 .13 0" size=".12 .01 .1"/>
    <geom name="cab_left" type="box" pos="-.11 0 0" size=".01 .12 .1"/>
    <geom name="cab_right" type="box" pos=".11 0 0" size=".01 .12 .1"/></body></worldbody></mujoco>'''
    xml,receipt=import_reconstructed_object(xml,body_name='obj_main',object_dir=root,object_id='target')
    return xml,receipt


def probe(xml,receipt):
    import mujoco
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    bounds=_world_bounds(m,d,[m.geom(n).id for n in receipt['contact_geoms']]);dest=native_destination(m,d,'PickPlaceCounterToCabinet',{'cab':'cab'})
    return cabinet_access(m,d,receipt,bounds,dest,V4_CONFIG)


def test_open_cabinet_actual_collider_entry_release_preserves_state(cabinet):
    report=probe(*cabinet)
    assert report['passed'] and report['entry_passed'] and report['final_destination_contact']
    assert report['source_reset_unchanged'] and report['release_unassisted'] and not report['unassisted']


def test_closed_face_cannot_be_overridden_by_successful_interior_release(cabinet):
    xml,receipt=cabinet
    xml=xml.replace('<body name="cab"','<body name="cab"')
    xml=xml.replace('<geom name="cab_top"','<geom name="cab_door" type="box" pos="0 -.13 0" size=".12 .01 .1"/><geom name="cab_top"')
    report=probe(xml,receipt)
    assert not report['passed'] and not report['entry_passed']
    assert report['final_destination_contact'] and any('cab_door'==r['geom'] for r in report['entry_collisions'])


def test_rotated_open_face_uses_local_frame(cabinet):
    xml,receipt=cabinet
    xml=xml.replace('<body name="cab" pos="0 0 .7"','<body name="cab" quat=".7071067811865476 0 0 .7071067811865476" pos="0 0 .7"')
    report=probe(xml,receipt)
    assert report['passed']
    delta=np.asarray(report['release_body_position_world_m'])-report['start_body_position_world_m']
    assert delta[0]<-.1 and abs(delta[1])<1e-9


def test_test_admission_is_versioned_and_parameters_identical():
    validate_config(dict(V4_CONFIG,tier='TEST'));validate_config(V4_CONFIG)
    assert all(V4_CONFIG[k]==v for k,v in CONTEXT_CONFIG.items() if k!='schema_version')
    with pytest.raises(ValueError):validate_config(dict(CONTEXT_CONFIG,tier='TEST'))
    with pytest.raises(ValueError,match='frozen'):validate_config(dict(V4_CONFIG,tier='TEST',max_penetration_m=.01))

from tests.test_native_context_checks import scene as native_scene


def test_existing_sink_counter_probe_is_identical_between_v3_and_v4(native_scene):
    from robo.roundtrip.native_context_checks import context_checks
    xml,m,d,receipt,_=native_scene
    v3=context_checks(xml,m,d,receipt,'PickPlaceSinkToCounter',{},CONTEXT_CONFIG)
    v4=context_checks(xml,m,d,receipt,'PickPlaceSinkToCounter',{},V4_CONFIG)
    assert v3==v4


def test_existing_counter_sink_probe_is_identical_between_v3_and_v4(native_scene):
    import mujoco,xml.etree.ElementTree as ET
    from robo.roundtrip.native_context_checks import context_checks
    xml,_,_,receipt,_=native_scene
    root=ET.fromstring(xml);body=root.find(".//body[@name='container_main']")
    body.find('geom').set('name','sink_bottom')
    ET.SubElement(body,'geom',name='sink_reg_basin',type='box',size='.08 .08 .03',contype='0',conaffinity='0')
    xml=ET.tostring(root,encoding='unicode');m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    assert context_checks(xml,m,d,receipt,'PickPlaceCounterToSink',{'sink':'sink'},CONTEXT_CONFIG)==context_checks(xml,m,d,receipt,'PickPlaceCounterToSink',{'sink':'sink'},V4_CONFIG)
