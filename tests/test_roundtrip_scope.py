import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robo.roundtrip.importers.robocasa import import_reconstructed_object
from robo.roundtrip.scope import import_scope, import_scope_identity
from robo.roundtrip.receptacle_probe import probe_entry


def factory(tmp_path, name='asset', parts=None, translation=(.2, .1, .3)):
    import trimesh
    path = tmp_path / name
    path.mkdir()
    (path / 'collision').mkdir()
    parts = parts or [trimesh.creation.box([.1, .2, .05])]
    trimesh.util.concatenate(parts).export(path / 'mesh_sim.obj')
    for i, mesh in enumerate(parts):
        mesh.export(path / 'collision' / f'part_{i:02}.obj')
    T = np.eye(4)
    T[:3, 3] = translation
    (path / 'aligned.json').write_text(json.dumps({'T': T.tolist(), 'scale': 1.0}))
    (path / 'physics.json').write_text(json.dumps({'mass_kg': .1, 'friction': .5}))
    return path


XML = '''<mujoco><worldbody><geom name="ground" type="plane" size="5 5 .1"/>
<body name="robot"><joint name="robot_joint" type="hinge"/><geom name="robot_geom" type="sphere" size=".1"/></body>
<body name="obj_main"><freejoint name="obj_joint"/><geom name="obj_old" type="box" size=".03 .03 .03"/></body>
<body name="container_main" pos="0 0 .3"><freejoint name="container_joint"/><geom name="container_old" type="box" size=".1 .1 .01"/></body>
<body name="cabinet" pos="1 2 0" quat=".7071067811865476 0 0 .7071067811865476">
<body name="counter" pos="0 0 1"><geom name="counter_old" type="box" size=".3 .4 .01"/></body></body>
</worldbody></mujoco>'''


def test_static_transformed_parent_is_world_aligned_and_robot_unchanged(tmp_path):
    mujoco = pytest.importorskip('mujoco')
    asset = factory(tmp_path)
    output, receipt = import_reconstructed_object(XML, body_name='counter', object_dir=asset,
                                                   object_id='observed_counter', role='support')
    model = mujoco.MjModel.from_xml_string(output)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[model.body('counter').id], [.2, .1, .3], atol=1e-14)
    np.testing.assert_allclose(data.xmat[model.body('counter').id], np.eye(3).ravel(), atol=1e-14)
    assert receipt['body_semantics'] == 'static' and receipt['free_joint_name'] is None
    assert ET.tostring(ET.fromstring(XML).find('./worldbody/body[@name="robot"]')) == ET.tostring(ET.fromstring(output).find('./worldbody/body[@name="robot"]'))
    assert 'counter_old' not in [g.get('name') for g in ET.fromstring(output).iter('geom')]


def test_multi_replacement_is_named_and_retained_inventory_explicit(tmp_path):
    a, b = factory(tmp_path, 'target'), factory(tmp_path, 'destination', translation=(.7, 0, .4))
    entities = [dict(body_name='container_main', object_id='observed_container', object_dir=b, role='receptacle'),
                dict(body_name='obj_main', object_id='observed_target', object_dir=a, role='target')]
    output, manifest = import_scope(XML, entities=entities, scope='L1_target_destination')
    assert set(manifest['replaced_joints']) == {'obj_joint', 'container_joint'}
    assert manifest['replaced_joints']['container_joint'][:3] == [.7, 0, .4]
    assert any(r.get('native_body_name') == 'counter' and r['classification'] == 'retained_reference' for r in manifest['inventory'])
    assert any(r.get('native_world_geom_name') == 'ground' for r in manifest['inventory'])
    assert manifest['strong_task_retention_claim'] == 'NOT_RUN'
    assert not any(g.get('name') in {'obj_old', 'container_old'} for g in ET.fromstring(output).iter('geom'))
    # Import order is independent of integration order; joint names survive.
    assert [j.get('name') for j in ET.fromstring(output).iter('freejoint')] == ['obj_joint', 'container_joint']


def test_scope_refuses_overlap_omission_and_unsupported_level(tmp_path):
    entity = dict(body_name='obj_main', object_id='obj', object_dir=factory(tmp_path), role='target')
    for entities, scope in (([entity, entity], 'L1_target_destination'), ([entity], 'L1_target_destination'), ([entity], 'L2_workspace')):
        with pytest.raises(ValueError):
            import_scope(XML, entities=entities, scope=scope)


def test_articulated_context_rejected(tmp_path):
    xml = XML.replace('<body name="counter" pos="0 0 1">', '<body name="counter" pos="0 0 1"><joint name="hinge" type="hinge"/>')
    with pytest.raises(ValueError, match='articulation'):
        import_reconstructed_object(xml, body_name='counter', object_dir=factory(tmp_path), object_id='c', role='support')


def test_destination_identity_has_exact_continuous_physics():
    mujoco = pytest.importorskip('mujoco')
    output, report = import_scope_identity(XML, entities=[{'body_name': 'container_main', 'role': 'receptacle'}, {'body_name': 'counter', 'role': 'support'}])
    models = [mujoco.MjModel.from_xml_string(x) for x in (XML, output)]
    datas = [mujoco.MjData(m) for m in models]
    for field in ('body_pos', 'body_mass', 'body_inertia', 'geom_friction', 'geom_contype', 'geom_conaffinity'):
        np.testing.assert_array_equal(getattr(models[0], field), getattr(models[1], field))
    for _ in range(200):
        for m, d in zip(models, datas):
            mujoco.mj_step(m, d)
        np.testing.assert_array_equal(datas[0].qpos, datas[1].qpos)
        np.testing.assert_array_equal(datas[0].qvel, datas[1].qvel)
    assert all(r['body_xml_equal'] for r in report['entities'])


def test_open_container_entry_passes_and_closed_hull_fails(tmp_path):
    mujoco = pytest.importorskip('mujoco')
    import trimesh
    parts = []
    for dimensions, center in [([.2, .2, .02], [0, 0, 0]),
                               ([.02, .2, .12], [-.09, 0, .06]),
                               ([.02, .2, .12], [.09, 0, .06]),
                               ([.16, .02, .12], [0, -.09, .06]),
                               ([.16, .02, .12], [0, .09, .06])]:
        mesh = trimesh.creation.box(dimensions)
        mesh.apply_translation(center)
        parts.append(mesh)
    closed = trimesh.util.concatenate(parts).convex_hull
    results = []
    for name, meshes in [('open', parts), ('closed', [closed])]:
        asset = factory(tmp_path, name, meshes, translation=(0, 0, 0))
        xml, receipt = import_reconstructed_object('<mujoco><worldbody><body name="container"><geom type="box" size=".1 .1 .1"/></body></worldbody></mujoco>',
            body_name='container', object_dir=asset, object_id=name, role='receptacle')
        results.append(probe_entry(xml, contact_geoms=receipt['contact_geoms'], start_world_m=[0, 0, .2], interior_world_z_m=[.01, .1]))
    assert results[0]['passed'] and results[0]['entered_steps'] > 0
    assert not results[1]['passed'] and results[1]['entered_steps'] == 0


def test_mocap_and_moving_parent_refuse_implicit_state_changes(tmp_path):
    asset = factory(tmp_path)
    for xml in (XML.replace('<body name="counter"', '<body mocap="true" name="counter"'),
                XML.replace('<body name="cabinet"', '<body name="cabinet"').replace('<body name="counter"', '<joint name="parent_hinge" type="hinge"/><body name="counter"')):
        with pytest.raises(ValueError):
            import_reconstructed_object(xml, body_name='counter', object_dir=asset,
                                        object_id='c', role='support')


def test_identity_state_size_is_checked():
    from robo.roundtrip.scope import check_scope_identity_physics
    with pytest.raises(ValueError, match='state shape'):
        check_scope_identity_physics(XML, entities=[{'body_name':'container_main', 'role':'receptacle'}],
                                     integration_state=[0.0], steps=1)


def test_zero_step_probe_rejected():
    with pytest.raises(ValueError, match='at least one'):
        probe_entry('<mujoco/>', contact_geoms=['x'], start_world_m=[0,0,1],
                    interior_world_z_m=[.1,.2], seconds=.00001)
