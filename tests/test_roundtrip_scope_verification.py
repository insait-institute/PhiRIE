import copy
import json
import numpy as np
import pytest
from robo.roundtrip.scope_verification import CONFIG,validate_config,next_action,regenerate_collision,measure_scope


def test_frozen_dev_and_reason_scheduler():
    validate_config(CONFIG)
    bad=dict(CONFIG,tier='TEST')
    with pytest.raises(ValueError,match='TEST'):validate_config(bad)
    measured={'passed':False,'reason_codes':['invalid_collision']}
    assert next_action('B4',measured,[])=='regenerate_collision'
    assert next_action('BM',measured,[])=='reselect_candidate'
    assert next_action('B4',{'passed':True},[]) is None
    assert next_action('BM',{'passed':True},[])=='reselect_candidate'
    assert next_action('BM',measured,CONFIG['bm_order']) is None


def make_asset(path,position,extent=(.05,.05,.05)):
    import trimesh
    path.mkdir();(path/'collision').mkdir()
    mesh=trimesh.creation.box(extent);mesh.export(path/'mesh_sim.obj');mesh.export(path/'collision/part_00.obj')
    T=np.eye(4);T[:3,3]=position
    (path/'aligned.json').write_text(json.dumps({'T':T.tolist(),'scale':1.}))
    (path/'physics.json').write_text(json.dumps({'mass_kg':.3,'friction':.5,'source':'frozen_prior'}))
    return path


def test_collision_action_preserves_visual_pose_and_immutable_source(tmp_path):
    from robo.roundtrip.scope_bundle import artifact_hashes
    src=make_asset(tmp_path/'src',[0,0,.1]);before=artifact_hashes(src)
    receipt=regenerate_collision(src,tmp_path/'new')
    assert artifact_hashes(src)==before
    assert receipt['source_artifact_hashes']!=receipt['result_artifact_hashes']
    for key in ['aligned.json','physics.json','mesh_sim.obj']:
        assert receipt['source_artifact_hashes'][key]==receipt['result_artifact_hashes'][key]
    assert receipt['whole_object_hull'] is False
    with pytest.raises(FileExistsError):regenerate_collision(src,tmp_path/'new')


def test_scope_records_retained_distractor_and_restores_exact_state(tmp_path):
    import mujoco
    xml='''<mujoco><option timestep=".002"/><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
    <body name="obj" pos="1 0 .03"><freejoint name="obj_joint"/><geom name="old_target" type="box" size=".025 .025 .025" mass=".3"/></body>
    <body name="container" pos="0 0 .1"><geom name="old_container" type="box" size=".1 .1 .1"/></body>
    <body name="distractor" pos="0 0 .03"><freejoint name="distractor_joint"/><geom name="distractor_geom" type="box" size=".025 .025 .025" mass=".3"/></body>
    </worldbody></mujoco>'''
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);kind=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(model,kind));mujoco.mj_getState(model,data,state,kind)
    target=make_asset(tmp_path/'target',[1,0,.03]);destination=make_asset(tmp_path/'destination',[0,0,.04],(.1,.1,.08))
    entities=[dict(body_name='obj',object_id='target',object_dir=target,role='target'),
              dict(body_name='container',object_id='destination',object_dir=destination,role='receptacle')]
    result,manifest=measure_scope(xml,{'integration_state':state.tolist()},entities,CONFIG)
    assert result['state_restored_byte_exact']
    assert not result['passed']
    assert result['max_penetration_m']>.005
    assert any('distractor_geom' in r['geom_names'] for r in result['contacts'])
    assert next(r for r in result['bodies'] if r['body_name']=='distractor')['classification']=='retained_reference'
    assert result['native_policy_success'] is None


@pytest.fixture
def repair_receipt(tmp_path):
    from robo.roundtrip.scope_bundle import artifact_hashes
    from robo.roundtrip.scope_verification import _write,_source,EDGES
    from robo.manifest.hash import canonical_hash
    from agents.orchestrator.job_graph import descendants
    target=make_asset(tmp_path/'target',[1,0,.03]);original=make_asset(tmp_path/'original',[0,0,.04])
    regenerated=tmp_path/'regenerated';details=regenerate_collision(original,regenerated)
    pool=tmp_path/'pool.json';_write(pool,{'canonical_instance_id':'native-dev'})
    bundle=dict(schema_version=1,scope='L1_target_destination',canonical_instance_id='native-dev',
        canonical_manifest_sha256='manifest',capture_manifest_sha256='capture',cohort_id='dev',
        destination_selection={'candidate_pool_sha256':_source(pool)['sha256']},
        entities=[dict(object_id='target',native_role='obj',role='target',object_dir=str(target),artifact_hashes=artifact_hashes(target)),
                  dict(object_id='destination',native_role='sink',role='receptacle',object_dir=str(original),artifact_hashes=artifact_hashes(original))])
    source=tmp_path/'bundle.json';_write(source,bundle)
    before=tmp_path/'before.json';_write(before,{'passed':False,'reason_codes':['invalid_collision']})
    after=tmp_path/'after.json';_write(after,{'passed':True,'reason_codes':[],'state_restored_byte_exact':True})
    receipt=dict(source={'dirty':False},schema_version=1,tier='DEV',scope=CONFIG['scope'],method='B4_ROOM_REPAIR_NATIVE',method_key='B4',
        canonical_instance_id='native-dev',canonical_manifest_sha256='manifest',capture_manifest_sha256='capture',cohort_id='dev',
        input_scope_bundle=_source(source),original_B3_pool=_source(pool),
        frozen_target_artifact_hashes=artifact_hashes(target),
        action_bank={'config':copy.deepcopy(CONFIG),'config_sha256':canonical_hash(CONFIG),'max_calls_per_dependency':2},
        actual_calls=1,actions=[dict(action_id='repair0',kind='regenerate_collision',dependency_id='destination',
            source_object_dir=str(original),result_object_dir=str(regenerated),
            source_artifact_hashes=artifact_hashes(original),result_artifact_hashes=artifact_hashes(regenerated),
            invalidated_dependencies=sorted(descendants(EDGES,'destination')),verification_after=_source(after),details=details)],
        verification_before=_source(before),verification_after=_source(after),accepted=True,terminal_status='READY',
        destination_selected=dict(object_dir=str(regenerated),artifact_hashes=artifact_hashes(regenerated)))
    path=tmp_path/'repair.json';_write(path,receipt)
    config={'controller_method':'B4_ROOM_REPAIR_NATIVE','instance':{'split':'development'},'canonical_instance_id':'native-dev','scope':CONFIG['scope']}
    return path,bundle,config


def test_receipt_accepts_same_artifacts_new_baseline_path(repair_receipt):
    from robo.roundtrip.scope_verification import validate_scope_repair
    path,bundle,config=repair_receipt
    bundle['baseline_episode']='/new/unexecuted/engine/path'
    assert validate_scope_repair(path,original_scope_bundle=bundle,config=config)['accepted']


@pytest.mark.parametrize('mutation',['target','capture','budget','duplicate','dependency','failed_evidence','wrong_method','changed_part'])
def test_repair_admission_rejects_tampering(repair_receipt,mutation):
    from robo.roundtrip.scope_verification import validate_scope_repair
    path,bundle,config=repair_receipt;receipt=json.loads(path.read_text())
    if mutation=='target':receipt['frozen_target_artifact_hashes']['mesh_sim.obj']='changed'
    elif mutation=='capture':receipt['capture_manifest_sha256']='changed'
    elif mutation=='budget':receipt['actual_calls']=3
    elif mutation=='duplicate':receipt['actions']*=2;receipt['actual_calls']=2
    elif mutation=='dependency':receipt['actions'][0]['invalidated_dependencies']=[]
    elif mutation=='failed_evidence':receipt['verification_after']=receipt['verification_before']
    elif mutation=='wrong_method':config['controller_method']='BM_BUDGET_MATCHED_NATIVE'
    elif mutation=='changed_part':
        from pathlib import Path
        part=Path(receipt['destination_selected']['object_dir'])/'collision/part_00.obj';part.write_text(part.read_text()+'\n')
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):validate_scope_repair(path,original_scope_bundle=bundle,config=config)


def test_test_admission_preserves_all_dev_parameters_and_old_gate():
    from robo.roundtrip.scope_verification import TEST_CONFIG
    validate_config(TEST_CONFIG)
    assert {k:v for k,v in TEST_CONFIG.items() if k not in ['schema_version','tier']}=={
        k:v for k,v in CONFIG.items() if k not in ['schema_version','tier']}
    with pytest.raises(ValueError):validate_config(dict(CONFIG,tier='TEST'))
    with pytest.raises(ValueError):validate_config(dict(TEST_CONFIG,max_penetration_m=.006))
    with pytest.raises(ValueError):validate_config(dict(TEST_CONFIG,bm_order=list(reversed(CONFIG['bm_order']))))


def test_test_receipt_requires_original_b4_target_source(repair_receipt,tmp_path):
    from pathlib import Path
    from robo.roundtrip.scope_verification import validate_scope_repair,TEST_CONFIG,_source
    from robo.roundtrip.system_verification import asset_identity
    from robo.manifest.hash import canonical_hash
    path,bundle,config=repair_receipt;receipt=json.loads(path.read_text())
    target=Path(bundle['entities'][0]['object_dir']);config['instance']['split']='test'
    bundle['entities'][0]['constructor_method']='B4_ROOM_REPAIR_NATIVE'
    source=Path(receipt['input_scope_bundle']['path']);source.write_text(json.dumps(bundle));receipt['input_scope_bundle']=_source(source)
    receipt.update(tier='TEST',action_bank={'config':TEST_CONFIG,'config_sha256':canonical_hash(TEST_CONFIG),'max_calls_per_dependency':2})
    m=tmp_path/'target_build.json';m.write_text(json.dumps({'method':'B4','accepted':True,'capture_manifest_sha256':'capture',
        'selected_asset':str(target),'selected_asset_identity':asset_identity(target)}))
    binding=dict(canonical_instance_id='native-dev',controller_method='B4_ROOM_REPAIR_NATIVE',accepted=True,terminal_status='READY',
        object_dir=str(target),build_manifest=str(m),build_manifest_sha256=_source(m)['sha256'])
    b=tmp_path/'original_binding.jsonl';b.write_text(json.dumps(binding)+'\n')
    binding.update(binding_source=str(b),binding_source_sha256=_source(b)['sha256']);receipt['target_build_binding']=binding
    path.write_text(json.dumps(receipt));assert validate_scope_repair(path,original_scope_bundle=bundle,config=config)['accepted']
    receipt['target_build_binding']['controller_method']='B3_AGENT_NATIVE';path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='target method'):validate_scope_repair(path,original_scope_bundle=bundle,config=config)
