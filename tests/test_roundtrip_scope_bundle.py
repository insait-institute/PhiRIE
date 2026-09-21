import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from robo.manifest.hash import canonical_hash
from robo.roundtrip.identity import file_hash
from robo.roundtrip.scope_bundle import artifact_hashes,seal_bundle,validate_bundle,validate_scope_pair
from robo.roundtrip.spec import load_spec


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def sources(tmp_path):
    c=load_spec(Path(__file__).parents[1]/'configs/experiments/sim_recon_sim/reference.yaml')
    c.update(schema_version=2,cohort_id='dev',canonical_instance_id='native-a',reset_id='r0',policy_rng_seed=3,
        canonical_manifest_sha256='a'*64,reset_contract_sha256='b'*64,scope='L0_target_only',controller_method='B3_AGENT_NATIVE',execution_protocol='primary_native',horizon=900)
    c['instance']['task_id']='PickPlaceSinkToCounter';c['policy']['checkpoint_receipt_sha256']='c'*64
    config=tmp_path/'baseline.json';write(config,c)
    ep=tmp_path/'baseline/episode';result=dict(controller_method='B3_AGENT_NATIVE',scope='L0_target_only',native_schema_version=2,execution_kind='closed_loop_visual_policy',config_sha256=canonical_hash(c),executed=True,error=None,success=False)
    write(ep/'result.json',result)
    target=tmp_path/'target';destination=tmp_path/'component/construction/objects/obj_00'
    for directory in [target,destination]:
        for name in ['aligned.json','physics.json','mesh_sim.obj','collision/part_00.obj']:
            p=directory/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    write(ep.parent/'import_receipt.json',dict(source_hashes={str(target/k):v for k,v in artifact_hashes(target).items()}))
    manifest=tmp_path/'component/build_manifest.json'
    build=dict(status='BUILT',built_objects=1,component_only=True,object_role='receptacle',declared_scope='L1_target_destination',canonical_instance_id='native-a',cohort_id='dev',method='B0_fixed_trellis',object_dir=str(destination),capture_manifest_sha256='d'*64,
        source_hashes={str((destination/k).relative_to(manifest.parent)):v for k,v in artifact_hashes(destination).items()})
    write(manifest,build)
    return c,ep,config,target,manifest


def test_failing_l0_is_eligible_and_scope_bundle_seals_both_roles(sources):
    c,ep,config,target,manifest=sources
    bundle=seal_bundle(ep,config,target,manifest);l1=copy.deepcopy(c);l1.update(scope='L1_target_destination',replacement_scope='target_destination')
    assert validate_bundle(bundle,l1,ep,config)==bundle
    assert bundle['system_variant']=='hybrid_scope_diagnostic_target_B3_destination_B0'
    assert [e['native_role'] for e in bundle['entities']]==['obj','container']
    (target/'physics.json').write_text('changed friction')
    with pytest.raises(ValueError,match='identical'):validate_bundle(bundle,l1,ep,config)


def test_static_sink_requires_matching_component_and_keeps_privileged_goal_explicit(sources):
    c,ep,config,target,manifest=sources
    c['instance']['task_id']='PickPlaceCounterToSink';c['horizon']=600;write(config,c)
    result=json.loads((ep/'result.json').read_text());result['config_sha256']=canonical_hash(c);write(ep/'result.json',result)
    with pytest.raises(ValueError,match='does not match'):seal_bundle(ep,config,target,manifest)
    build=json.loads(manifest.read_text());build['component_kind']='sink_basin';write(manifest,build)
    bundle=seal_bundle(ep,config,target,manifest)
    assert bundle['entities'][1]['native_role']=='sink'
    assert bundle['entities'][1]['native_binding_kind']=='fixture_task_attribute'
    assert bundle['system_variant']=='hybrid_scope_diagnostic_target_B3_destination_B0'
    assert 'native goal regions' in bundle['retained_fixture_context']
    l1=copy.deepcopy(c);l1.update(scope='L1_target_destination',replacement_scope='target_destination')
    assert validate_bundle(bundle,l1,ep,config)==bundle


def test_cabinet_bottom_support_is_explicit_partial_fixture_scope(sources):
    c,ep,config,target,manifest=sources
    c['instance']['task_id']='PickPlaceCounterToCabinet';c['horizon']=750;write(config,c)
    result=json.loads((ep/'result.json').read_text());result['config_sha256']=canonical_hash(c);write(ep/'result.json',result)
    build=json.loads(manifest.read_text());build.update(component_kind='cabinet_bottom_shelf',object_role='support');write(manifest,build)
    bundle=seal_bundle(ep,config,target,manifest)
    assert bundle['entities'][1]['native_role']=='cab' and bundle['entities'][1]['role']=='support'
    assert bundle['native_success_may_use_retained_upper_shelves']
    assert bundle['static_destination_scope']=='bottom cabinet support surface only'
    build['object_role']='receptacle';write(manifest,build)
    with pytest.raises(ValueError,match='does not match'):seal_bundle(ep,config,target,manifest)


def test_scope_cannot_change_policy_reset_or_target_method(sources):
    c,ep,config,target,manifest=sources;l1=copy.deepcopy(c);l1.update(scope='L1_target_destination',replacement_scope='target_destination')
    l1['policy_rng_seed']+=1
    with pytest.raises(ValueError,match='another frozen'):validate_scope_pair(l1,c,json.loads((ep/'result.json').read_text()))
    b=json.loads(manifest.read_text());b['canonical_instance_id']='different';write(manifest,b)
    with pytest.raises(ValueError,match='matching sealed'):seal_bundle(ep,config,target,manifest)


def test_isolated_output_mount_resolves_only_within_sealed_build(sources):
    c,ep,config,target,manifest=sources
    b=json.loads(manifest.read_text());b['object_dir']='/output/construction/objects/obj_00';write(manifest,b)
    bundle=seal_bundle(ep,config,target,manifest)
    assert bundle['entities'][1]['object_dir']==str(manifest.parent/'construction/objects/obj_00')
    b['object_dir']='/output/../other';write(manifest,b)
    with pytest.raises(ValueError,match='escapes'):seal_bundle(ep,config,target,manifest)


@pytest.fixture
def role_pool(sources):
    from robo.roundtrip.shared_candidates import proposal_prefix
    c,ep,config,target,manifest=sources;build=json.loads(manifest.read_text())
    pool_dir=manifest.parent.parent/'role_pool';selected=pool_dir/'selection/selected_00'
    import shutil
    shutil.copytree(Path(build['object_dir']),selected)
    pid=proposal_prefix(build,{'capture_manifest_sha256':build['capture_manifest_sha256']})+':trellis:initial:42'
    row=dict(policy_id='A3',native_method='B3_AGENT_NATIVE',terminal_action='accept',selected_proposal_id=pid,
        object_dir='/output/selection/selected_00',artifact_hashes=artifact_hashes(selected),reason_codes=['evidence_selection'],retry_invoked=False)
    rr=pool_dir/'rvg/rvg_receipt.json';write(rr,dict(status='tool_failure',b0_manifest_sha256=file_hash(manifest)))
    pool=dict(schema_version=2,planned_objects=1,controller='agents.orchestrator.controller.run_policies',
        isolated_probe_is_native_context_evidence=False,object_role='receptacle',b0_manifest_sha256=file_hash(manifest),
        canonical_instance_id=c['canonical_instance_id'],cohort_id=c['cohort_id'],capture_manifest_sha256=build['capture_manifest_sha256'],
        initial_candidates=[dict(proposal_id=pid)],retry_candidate=None,outcomes=[row],rvg_receipt_sha256=file_hash(rr))
    path=pool_dir/'selection/candidate_pool.json';write(path,pool)
    (path.parent/'selection_ledger.jsonl').write_text(json.dumps(row)+'\n')
    return path,selected


def test_b3_destination_preserves_target_and_seals_selected_role(sources,role_pool):
    c,ep,config,target,manifest=sources;pool,selected=role_pool
    hybrid=seal_bundle(ep,config,target,manifest);full=seal_bundle(ep,config,target,manifest,pool)
    assert full['system_variant']=='B3_target_B3_destination_L1'
    assert full['entities'][0]==hybrid['entities'][0]
    assert full['entities'][1]['constructor_method']=='B3_AGENT_NATIVE'
    assert full['entities'][1]['object_dir']==str(selected)
    l1=copy.deepcopy(c);l1.update(scope='L1_target_destination',replacement_scope='target_destination')
    assert validate_bundle(full,l1,ep,config)==full
    (selected/'aligned.json').write_text('different pose')
    with pytest.raises(ValueError,match='closure changed'):validate_bundle(full,l1,ep,config)


@pytest.mark.parametrize('mutation',['canonical','role','abstain','proposal','ledger'])
def test_b3_destination_fails_closed_without_other_role_or_b0_fallback(sources,role_pool,mutation):
    c,ep,config,target,manifest=sources;path,selected=role_pool;pool=json.loads(path.read_text())
    if mutation=='canonical':pool['canonical_instance_id']='other'
    elif mutation=='role':pool['object_role']='target'
    elif mutation=='abstain':pool['outcomes'][0]['terminal_action']='abstain'
    elif mutation=='proposal':pool['initial_candidates'][0]['proposal_id']='target-capture:trellis:initial:42'
    elif mutation=='ledger':(path.parent/'selection_ledger.jsonl').write_text('{}\n')
    write(path,pool)
    with pytest.raises(ValueError):seal_bundle(ep,config,target,manifest,path)


def test_prepare_scope_uses_both_estimates_and_keeps_target_import(monkeypatch):
    import robo.roundtrip.paired as paired
    import robo.roundtrip.scope as scope
    events=[];target={'role':'target','source_hashes':{'mesh':'a'}}
    manifest={'entities':[target,{'role':'receptacle'}],'replaced_joints':{'obj_joint':[1,2,3,1,0,0,0],'container_joint':[4,5,6,1,0,0,0]}}
    monkeypatch.setattr(paired,'prepare_paired_adapter',lambda *a,**kw:events.append('canonical'))
    monkeypatch.setattr(scope,'import_scope',lambda *a,**kw:('generated xml',copy.deepcopy(manifest)))
    monkeypatch.setattr(scope,'bind_scope_objects',lambda *a,**kw: {'ok':True})
    import robo.roundtrip.scope_contacts as contacts
    monkeypatch.setattr(contacts,'bind_adapter_contacts',lambda *a: {})
    adapter=NS(native=NS(objects={'obj':NS(joints=['obj_joint'],root_body='obj_body'),'container':NS(joints=['container_joint'],root_body='container_body')}),
        import_xml=lambda xml,**kw:events.append(kw))
    bundle={'xml':'native xml','state':{'canonical':1},'provenance':{}}
    roles={'entities':[dict(native_role='obj',object_id='target',object_dir='target',role='target'),dict(native_role='container',object_id='destination',object_dir='plate',role='receptacle')]}
    paired.prepare_scope_adapter(adapter,bundle,roles,target)
    assert events[0]=='canonical' and events[1]['replaced_joints']==manifest['replaced_joints']
    assert events[1]['canonical_state']==bundle['state']
    with pytest.raises(ValueError,match='target artifact'):
        paired.prepare_scope_adapter(adapter,bundle,roles,{'source_hashes':{'mesh':'changed'}})


def test_l1_reaches_canonical_runner_with_same_target_reset_and_hybrid_metadata(sources,tmp_path,monkeypatch):
    from robo.roundtrip.identity import create_canonical_instance,create_reset_bank
    import robo.roundtrip.paired as paired
    import robo.eval.harness_runner as harness
    c,ep,config_path,target,build_path=sources
    directory=tmp_path/'canonical';directory.mkdir()
    write(directory/'canonical_state.json',dict(integration_state=[],controller_state=[],observable_timing={},native_metadata={},qpos=[],qvel=[],object_states={}))
    (directory/'scene.xml').write_text('<mujoco/>')
    manifest=create_canonical_instance(directory,c)
    bank=create_reset_bank(manifest,[dict(reset_id='r0',delta_world=np.eye(4).tolist(),policy_rng_seed=3)])
    c.update(canonical_instance_id=manifest['canonical_instance_id'],canonical_manifest_sha256=manifest['manifest_sha256'],reset_contract_sha256=bank['reset_contract_sha256'])
    write(config_path,c);manifest_path=tmp_path/'manifest.json';bank_path=tmp_path/'bank.json';write(manifest_path,manifest);write(bank_path,bank)
    binding=dict(canonical_instance_id=manifest['canonical_instance_id'],canonical_manifest_sha256=manifest['manifest_sha256'],scene_xml_sha256=manifest['identity']['scene_xml_sha256'],state_sha256=manifest['identity']['state_sha256'],asset_closure_sha256=manifest['identity']['asset_closure_sha256'])
    metadata={'checkpoint_receipt_sha256':c['policy']['checkpoint_receipt_sha256']}
    baseline=json.loads((ep/'result.json').read_text());baseline.update(config_sha256=canonical_hash(c),policy_identity=metadata,canonical_reference=binding);write(ep/'result.json',baseline)
    build=json.loads(build_path.read_text());build['canonical_instance_id']=c['canonical_instance_id'];write(build_path,build)
    scope=seal_bundle(ep,config_path,target,build_path);scope_path=tmp_path/'scope.json';write(scope_path,scope)
    l1=copy.deepcopy(c);l1.update(scope='L1_target_destination',replacement_scope='target_destination')
    calls=[];adapter=NS(apply_paired_reset=lambda reset,role: calls.append(('reset',reset,role)) or {})
    monkeypatch.setattr(paired,'prepare_scope_adapter',lambda *a: calls.append(('prepare_scope',a[2])) or {})
    def run(adapter,policy,**kw):
        calls.append(('run',kw));out=kw['out_dir'];out.mkdir()
        r=dict(executed=True,success=False,error=None);write(out/'result.json',r);return r
    monkeypatch.setattr(harness,'run_native_episode',run)
    result=paired.run_resolved_episode(adapter,NS(metadata=metadata),config=l1,canonical_reference=directory,
        canonical_manifest=manifest_path,reset_bank=bank_path,out_dir=tmp_path/'run',reference_episode=ep,
        baseline_config=config_path,scope_bundle=scope_path)
    assert result['success'] is False and calls[1][0]=='reset' and calls[1][2]=='obj'
    assert calls[2][1]['config']['policy_rng_seed']==c['policy_rng_seed']
    assert adapter.scope_binding['system_variant']=='hybrid_scope_diagnostic_target_B3_destination_B0'
    receipt=json.loads((tmp_path/'run/pair_receipt.json').read_text())
    assert receipt['state']=='RECORDED' and receipt['treatment_axis']=='replacement_scope'


def test_b4_scope_cannot_relabel_b0_destination_as_repaired(sources):
    c,ep,config_path,target,build_path=sources
    c['controller_method']='B4_ROOM_REPAIR_NATIVE';write(config_path,c)
    result=json.loads((ep/'result.json').read_text());result.update(controller_method='B4_ROOM_REPAIR_NATIVE',config_sha256=canonical_hash(c));write(ep/'result.json',result)
    with pytest.raises(ValueError,match='explicit destination repair'):
        seal_bundle(ep,config_path,target,build_path)


def test_unexecuted_construction_input_never_becomes_episode(sources,role_pool):
    from robo.roundtrip.scope_bundle import prepare_construction_bundle,validate_bundle
    c,ep,config,target,manifest=sources;pool,_=role_pool
    bundle=prepare_construction_bundle(config,target,manifest,pool)
    assert bundle['kind']=='unexecuted_scope_construction_input' and bundle['policy_invoked'] is False
    assert bundle['baseline_episode'] is None and bundle['baseline_result_sha256'] is None
    with pytest.raises(ValueError,match='not a sealed'):validate_bundle(bundle,c,ep,config)


def test_support_pool_requires_matching_source_role(sources,role_pool):
    from robo.roundtrip.scope_bundle import selected_destination
    from robo.roundtrip.shared_candidates import proposal_prefix
    c,ep,config,target,manifest=sources;path,selected=role_pool
    build=json.loads(manifest.read_text());build.update(object_role='support',component_kind='cabinet_bottom_shelf');write(manifest,build)
    pool=json.loads(path.read_text());pool.update(object_role='support',b0_manifest_sha256=file_hash(manifest))
    pid=proposal_prefix(build,{'capture_manifest_sha256':build['capture_manifest_sha256']})+':trellis:initial:42'
    pool['initial_candidates'][0]['proposal_id']=pid;pool['outcomes'][0]['selected_proposal_id']=pid
    rvg=path.parent.parent/'rvg/rvg_receipt.json';write(rvg,dict(status='tool_failure',b0_manifest_sha256=file_hash(manifest)))
    pool['rvg_receipt_sha256']=file_hash(rvg);write(path,pool);(path.parent/'selection_ledger.jsonl').write_text(json.dumps(pool['outcomes'][0])+'\n')
    assert selected_destination(path,manifest,build)[0]==selected.resolve()
    build['object_role']='receptacle'
    with pytest.raises(ValueError):selected_destination(path,manifest,build)


@pytest.mark.parametrize('mutation',[None,'method','source','asset','abstained'])
def test_b4_construction_preserves_typed_original_target(sources,role_pool,tmp_path,mutation):
    from robo.roundtrip.scope_bundle import prepare_construction_bundle
    from robo.roundtrip.system_verification import asset_identity
    c,ep,config,target,manifest=sources;pool,_=role_pool
    c['controller_method']='B4_ROOM_REPAIR_NATIVE';write(config,c)
    m=tmp_path/'target_build.json';write(m,dict(method='B4',accepted=True,selected_asset=str(target),selected_asset_identity=asset_identity(target),capture_manifest_sha256='d'*64))
    row=dict(canonical_instance_id=c['canonical_instance_id'],controller_method=c['controller_method'],accepted=True,terminal_status='READY',object_dir=str(target),build_manifest=str(m),build_manifest_sha256=file_hash(m))
    src=tmp_path/'target_bindings.jsonl';src.write_text(json.dumps(row)+'\n')
    row.update(binding_source=str(src),binding_source_sha256=file_hash(src))
    if mutation=='method':row['controller_method']='B3_AGENT_NATIVE'
    if mutation=='source':src.write_text('{}\n')
    if mutation=='asset':(target/'physics.json').write_text('different')
    if mutation=='abstained':row.update(accepted=False,terminal_status='ABSTAINED')
    if mutation:
        with pytest.raises(ValueError):prepare_construction_bundle(config,target,manifest,pool,target_build_binding=row)
    else:
        b=prepare_construction_bundle(config,target,manifest,pool,target_build_binding=row)
        assert b['target_method']=='B4_ROOM_REPAIR_NATIVE' and b['target_build_binding']==row
        assert b['entities'][1]['constructor_method']=='B3_AGENT_NATIVE'
        with pytest.raises(ValueError,match='typed target'):prepare_construction_bundle(config,target,manifest,pool)
