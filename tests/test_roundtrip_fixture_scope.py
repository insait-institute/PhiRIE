import copy
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from robo.roundtrip.fixture_scope import component_roster, import_fixture_component, bind_fixture_component
from robo.roundtrip.build import task_role_prompt
from tests.test_roundtrip_scope import factory


XML = '''<mujoco><compiler angle="degree"/><worldbody>
<body name="sink_main" pos="1 2 3" euler="0 0 90">
<geom name="sink_basin" type="box" size=".3 .3 .02"/>
<geom name="sink_region" type="box" size=".2 .2 .1" contype="0" conaffinity="0" rgba="0 1 0 0"/>
<site name="sink_goal" pos="0 0 .1"/>
<body name="sink_faucet" pos=".3 0 .2"><joint name="sink_hinge"/>
<geom name="sink_spout" type="capsule" size=".02 .1"/></body>
</body></worldbody><actuator><motor joint="sink_hinge"/></actuator></mujoco>'''


def test_estimated_world_vertices_and_articulated_child_are_preserved(tmp_path):
    import mujoco
    asset = factory(tmp_path, translation=(.2, .1, .3))
    output, receipt = import_fixture_component(XML, body_name='sink_main', object_dir=asset, object_id='observed')
    before, after = ET.fromstring(XML), ET.fromstring(output)
    for path in ('./worldbody/body/body', './worldbody/body/site', './worldbody/body/geom[@name="sink_region"]', './actuator'):
        assert ET.tostring(before.find(path)) == ET.tostring(after.find(path))
    assert before.find('./worldbody/body').attrib == after.find('./worldbody/body').attrib
    assert after.find('.//geom[@name="sink_basin"]') is None
    model = mujoco.MjModel.from_xml_string(output); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    gid = model.geom(receipt['visual_geoms'][0]).id
    # Compiled MuJoCo mesh centering is accounted for by mesh vertices + geom frame.
    mid = model.geom_dataid[gid]; start = model.mesh_vertadr[mid]; count = model.mesh_vertnum[mid]
    world = model.mesh_vert[start:start+count] @ data.geom_xmat[gid].reshape(3,3).T + data.geom_xpos[gid]
    np.testing.assert_allclose((world.min(0)+world.max(0))/2, [.2,.1,.3], atol=1e-7)
    assert model.body('sink_faucet').parentid == model.body('sink_main').id
    assert receipt['scale_applications'] == 1 and not receipt['dynamic_mass_prior_applied']
    assert receipt['component_roster']['retained_child_bodies'] == ['sink_faucet']


def test_component_negative_cases(tmp_path):
    asset = factory(tmp_path)
    for xml in (XML.replace('name="sink_main"', 'name="sink_main" mocap="true"'),
                XML.replace('<site name="sink_goal"', '<joint name="bad"/><site name="sink_goal"'),
                XML.replace('</mujoco>', '<contact><pair geom1="sink_basin" geom2="sink_spout"/></contact></mujoco>')):
        with pytest.raises(ValueError):
            import_fixture_component(xml, body_name='sink_main', object_dir=asset, object_id='x')
    with pytest.raises(ValueError, match='only static sink_basin'):
        import_fixture_component(XML, body_name='sink_main', object_dir=asset, object_id='x', component_kind='cabinet')


def test_fixture_binding_preserves_native_regions_and_children(tmp_path):
    output, receipt = import_fixture_component(XML, body_name='sink_main', object_dir=factory(tmp_path), object_id='x')
    class Fixture:
        root_body='sink_main'; naming_prefix='sink_'
        _contact_geoms=['basin','spout']; _visual_geoms=['basin','spout']
        _regions={'basin': np.array([[.1,.2,.3],[.4,.5,.6]])}
        @property
        def contact_geoms(self):return [self.naming_prefix+n for n in self._contact_geoms]
        @property
        def visual_geoms(self):return [self.naming_prefix+n for n in self._visual_geoms]
    obj=Fixture(); regions=copy.deepcopy(obj._regions)
    names={g.get('name') for g in ET.fromstring(output).iter('geom')}
    env=SimpleNamespace(get_fixture=lambda _:obj,sim=SimpleNamespace(model=SimpleNamespace(geom_name2id=lambda n: list(names).index(n))))
    bind_fixture_component(env,fixture_name='sink',receipt=receipt)
    np.testing.assert_array_equal(obj._regions['basin'],regions['basin'])
    assert 'sink_spout' in obj.contact_geoms and 'sink_basin' not in obj.contact_geoms


def test_public_sink_component_requires_explicit_protocol():
    instruction='Pick the whisk from the counter and place it in the sink.'
    assert task_role_prompt(instruction,'receptacle','sink basin','sink_basin') == 'sink basin'
    for args in ((instruction,'receptacle',None,'sink_basin'),
                 (instruction.replace('sink','cabinet'),'receptacle','sink basin','sink_basin'),
                 (instruction,'receptacle','native_sink_12','sink_basin')):
        with pytest.raises(ValueError):task_role_prompt(*args)


def test_native_predicate_probe_has_real_positive_and_negative():
    pytest.importorskip("robocasa", reason="requires the separate native robot runtime")
    from robo.roundtrip.fixture_scope import check_native_sink_scorer_identity, import_fixture_identity
    xml=XML.replace('sink_region', 'sink_reg_basin')
    output, _ = import_fixture_identity(xml,body_name='sink_main')
    report=check_native_sink_scorer_identity(xml,output,body_name='sink_main')
    assert report['status']=='PASS' and report['contains_positive_and_negative']
    assert len(report['probes'])==8 and report['real_native_episodes']==0


def test_multi_scope_replaces_target_and_direct_basin_without_relabeling_children(tmp_path):
    from robo.roundtrip.scope import import_scope
    xml=XML.replace('<worldbody>', '<worldbody><body name="obj_main"><freejoint name="obj_joint"/><geom name="obj_geom" type="box" size=".02 .02 .02"/></body>')
    output, report=import_scope(xml,scope='L1_target_destination',entities=[
        dict(body_name='obj_main',object_id='target',object_dir=factory(tmp_path,'target'),role='target'),
        dict(body_name='sink_main',object_id='destination',object_dir=factory(tmp_path,'dest'),role='receptacle',component_kind='sink_basin')])
    assert set(report['replaced_joints'])=={'obj_joint'}
    assert next(r for r in report['inventory'] if r.get('native_body_name')=='sink_faucet')['classification']=='retained_reference'
    assert next(r for r in report['inventory'] if r.get('native_component_body')=='sink_main')['classification']=='reconstructed_direct_physical_geoms'
    assert ET.fromstring(output).find('.//geom[@name="sink_basin"]') is None


def test_static_parent_has_generated_compilation_inertia_with_joint_child(tmp_path):
    import mujoco
    # Mirrors the native sink's explicit zero-mass visual/collision convention.
    xml=XML.replace('name="sink_basin"', 'name="sink_basin" density="0" mass="0"')
    # Original positive mass is a compiled prerequisite, not a construction input.
    xml=xml.replace('<geom name="sink_basin"','<inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/><geom name="sink_basin"')
    output, receipt=import_fixture_component(xml,body_name='sink_main',object_dir=factory(tmp_path),object_id='x')
    model=mujoco.MjModel.from_xml_string(output);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    bid=model.body('sink_main').id
    assert model.body_mass[bid]==receipt['mass_kg']
    np.testing.assert_allclose(data.xipos[bid],[.2,.1,.3],atol=1e-12)
    assert receipt['static_compilation_inertial_prior'].startswith('generated s6')
