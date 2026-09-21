import numpy as np
import pytest
from robo.roundtrip.fixture_scope import component_roster,import_fixture_identity
from robo.roundtrip.scope import check_scope_identity_physics
from robo.roundtrip.workspace_import import import_observed_workspace


XML='''<mujoco><worldbody><body name="counter_main" pos="1 2 0" quat=".9238795 0 0 .3826834">
<geom name="counter_base" type="box" size=".5 .4 .3"/>
<geom name="counter_top_0" type="box" size=".25 .4 .02" pos="-.25 0 .32"/>
<geom name="counter_top_1" type="box" size=".25 .4 .02" pos=".25 0 .32"/>
<geom name="counter_top_visual" type="box" size=".5 .4 .02" pos="0 0 .32" contype="0" conaffinity="0" group="1"/>
<body name="door"><joint name="hinge"/><geom name="door_geom" type="box" size=".1 .1 .1"/></body>
</body></worldbody></mujoco>'''


def test_counter_chunks_exclude_base_and_articulation_and_identity():
    import mujoco
    r=component_roster(XML,'counter_main','counter_top')
    assert set(r['direct_physical_geoms'])=={'counter_top_0','counter_top_1','counter_top_visual'}
    assert r['retained_direct_physical_geoms']==['counter_base'] and r['retained_child_bodies']==['door']
    model=mujoco.MjModel.from_xml_string(XML);data=mujoco.MjData(model)
    state=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    _,report=check_scope_identity_physics(XML,entities=[dict(body_name='counter_main',role='support',component_kind='counter_top')],integration_state=state,steps=20)
    assert report['runtime_identity']=='PASS'


def test_unknown_counter_naming_rejected():
    with pytest.raises(ValueError,match='named top'):component_roster(XML.replace('counter_top_visual','untyped_visual'),'counter_main','counter_top')


def test_workspace_rejects_native_leakage_before_import():
    with pytest.raises(ValueError,match='TRAIN-only'):import_observed_workspace('invalid',base_entities=[],support_entities=[],coverage=dict(native_asset_access=True),bounds=[])


def test_factored_probe_has_identical_existing_outputs_and_rejects_hash_tamper(tmp_path):
    import hashlib
    import mujoco
    from robo.roundtrip.scope import import_scope
    from robo.roundtrip.scope_verification import measure_scope,measure_imported_scope,CONFIG
    import trimesh,json
    entities=[]
    for name,role in [('target','target'),('destination','receptacle')]:
        p=tmp_path/name;p.mkdir();(p/'collision').mkdir()
        mesh=trimesh.creation.box([.05,.05,.05]);mesh.export(p/'mesh_sim.obj');mesh.export(p/'collision/part_00.obj')
        T=np.eye(4);T[:3,3]=[0 if name=='target' else 1,0,.5]
        (p/'aligned.json').write_text(json.dumps(dict(T=T.tolist(),scale=1)))
        (p/'physics.json').write_text(json.dumps(dict(mass_kg=.3,friction=.5)))
        entities.append(dict(body_name=name,object_id=name,role=role,object_dir=str(p)))
    xml='<mujoco><worldbody>'+''.join(f'<body name="{n}" pos="{i} 0 .5"><freejoint name="{n}_joint"/><geom name="{n}_g" size=".025"/></body>' for i,n in enumerate(('target','destination')))+'</worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);v=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,v,mujoco.mjtState.mjSTATE_INTEGRATION);state=dict(integration_state=v.tolist())
    imported,manifest=import_scope(xml,entities=entities,scope='L1_target_destination')
    original,_=measure_scope(xml,state,entities,CONFIG);factored,_=measure_imported_scope(xml,state,imported,manifest,CONFIG)
    assert original==factored
    manifest['imported_xml_sha256']='wrong'
    with pytest.raises(ValueError,match='hash differs'):measure_imported_scope(xml,state,imported,manifest,CONFIG)
