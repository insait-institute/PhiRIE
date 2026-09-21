"""Full-source closure and delegation; canonical producers own every metric."""
import copy
import json
from pathlib import Path
import pytest
from run.icra2027 import e4_full_qualification as m


def put(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload));return m.shared.identity(path)


@pytest.fixture
def source(tmp_path,monkeypatch):
    root=tmp_path/m.SOURCE_FREEZE/'agentic';root.mkdir(parents=True)
    monkeypatch.setattr(m.screen,'evidence_root',lambda:tmp_path)
    monkeypatch.setattr(m.materializer.e3,'checked_repo_path',lambda p,*a,**k:Path(p))
    monkeypatch.setattr(m.subprocess,'check_output',lambda args,**k:m.SOURCE_COMMIT if 'HEAD' in args else '')
    contract={'freeze_id':m.SOURCE_FREEZE,'code':{'repository':str(tmp_path),'commit':m.SOURCE_COMMIT,'dirty':False}}
    put(root.parent/'contract/freeze_manifest.json',contract)
    monkeypatch.setattr(m.shared,'contract_digest',lambda c:m.SOURCE_CONTRACT)
    scenes=[f'{i:010x}' for i in range(50)];sources=[];jobs=[];controls=[];observations=[]
    for i,sid in enumerate(scenes):
        n=37 if i<49 else 58;jobs.append({'scene_id':sid,'jobs':[{}]*n})
        sources.append({'scene_id':sid,'discovery_directory':'original','discovery_hashes':{}})
        ids={k:put(root/folder/sid/name,{'scene':sid,'kind':k}) for k,folder,name in (
            ('seal','control','seal.json'),('shard','control','controller_shard.json'),
            ('ledger','control','job_ledger.jsonl'),('observation','observations','seal.json'))}
        controls.append(dict(scene_id=sid,planned_jobs=n,policy_rows=5*n,seal_sha256=ids['seal']['sha256'],
            shard_sha256=ids['shard']['sha256'],ledger_sha256=ids['ledger']['sha256']))
        observations.append(dict(scene_id=sid,planned_jobs=n,seal_sha256=ids['observation']['sha256']))
    config_id=put(tmp_path/'jobs.json',{'automatic_sources':sources})
    cfg=m.shared.read(config_id['path'])
    inv={'freeze_id':m.SOURCE_FREEZE,'counts':{'scenes':50,'jobs':1871,'policy_object_rows':9355},
        'source_contract':{'code_commit':m.SOURCE_COMMIT,'contract_sha256':m.SOURCE_CONTRACT,'jobs_sha256':m.shared.canonical_hash(cfg)},'scenes':jobs}
    monkeypatch.setattr(m.materializer,'_verify_inventory',lambda p:(inv,None,{},{}))
    monkeypatch.setattr(m.materializer,'_automatic_scene_audit',lambda *a:None)
    monkeypatch.setattr(m.shared,'resource',lambda *a:config_id)
    monkeypatch.setattr(m.screen,'qualification_selection',lambda *a,**k:{})
    protocol=put(tmp_path/'protocol.json',{'source_populations':[{'scene_id':sid,'planned_objects':len(j['jobs']),'queries':[]} for sid,j in zip(scenes,jobs)]})
    driver=put(root.parent/'dispatch/audit_full50_controls.py',{})
    monkeypatch.setattr(m,'CONTROL_DRIVER',driver['sha256'])
    common=dict(status='PASS',source_commit=m.SOURCE_COMMIT,freeze_id=m.SOURCE_FREEZE,
        planned_scenes=50,planned_jobs=1871,evaluation_geometry_read=False,paper_ready=False)
    cp=put(root.parent/'dispatch/full50_control_integrity_gate.json',{**common,
        'scope':'complete_construction_integrity_only','audit_driver_sha256':driver['sha256'],
        'contract_sha256':m.SOURCE_CONTRACT,'planned_policy_rows':9355,'scenes':controls})
    op=put(root.parent/'dispatch/full50_observation_integrity_gate.json',{**common,
        'scope':'complete_training_observation_integrity_only','records':observations})
    return root,Path(protocol['path']),Path(cp['path']),Path(op['path'])


def test_complete_original_audit_closure_without_evaluation(source):
    root,protocol,_,_=source;r=m.inspect_source(protocol,root)
    assert r['status']=='READY' and len(r['source']['scenes'])==50
    assert all('evaluation' not in k for k in r['source'])


def test_missing_control_audit_waits_without_fabricated_failure(source):
    root,protocol,cp,_=source;cp.unlink()
    assert m.inspect_source(protocol,root)['status']=='WAITING'


@pytest.mark.parametrize('kind',['missing_scene','duplicate_scene','wrong_count','paper_ready','source','driver','control_tamper','observation_tamper'])
def test_full_audit_tamper(source,kind):
    root,protocol,cp,op=source;c=m.shared.read(cp)
    if kind=='missing_scene':c['scenes'].pop()
    elif kind=='duplicate_scene':c['scenes'][1]=c['scenes'][0]
    elif kind=='wrong_count':c['scenes'][0]['policy_rows']-=1
    elif kind=='paper_ready':c['paper_ready']=True
    elif kind=='source':c['source_commit']='wrong'
    elif kind=='driver':c['audit_driver_sha256']='wrong'
    elif kind=='control_tamper':(root/'control'/c['scenes'][0]['scene_id']/'controller_shard.json').write_text('changed')
    else:(root/'observations'/c['scenes'][0]['scene_id']/'seal.json').write_text('changed')
    cp.write_text(json.dumps(c))
    with pytest.raises(ValueError):m.inspect_source(protocol,root)


@pytest.mark.parametrize('name,value',[('SLURM_ARRAY_JOB_ID','1'),('SLURM_JOB_GPUS','0'),('SLURM_STEP_GPUS','1'),('SLURM_GPUS_ON_NODE','1'),('CUDA_VISIBLE_DEVICES','0')])
def test_gpu_or_array_allocation_rejected(monkeypatch,name,value):
    for key in ('SLURM_ARRAY_JOB_ID','SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE','CUDA_VISIBLE_DEVICES'):
        monkeypatch.delenv(key,raising=False)
    monkeypatch.setenv(name,value)
    with pytest.raises(ValueError):m.cpu_guard()


@pytest.fixture
def execution(tmp_path,monkeypatch):
    stage=tmp_path/'outputs/icra2027/new';stage.mkdir(parents=True)
    config_path=Path(put(tmp_path/'config.json',{})['path'])
    cfg={'freeze_id':'new','pilot_scene':'first','source':{'scenes':{s:{'descriptor':{'scene_id':s}} for s in ('first','second','empty')},
        'protocol':{'path':'protocol'},'e3_root':'original'},'menagerie_root':'menagerie'}
    protocol={'qualification_tasks':[{'scene_id':'first','task_id':'first__obj_to_region'},
                                     {'scene_id':'second','task_id':'second__obj_to_region'}]}
    monkeypatch.setattr(m,'validate_stage',lambda *a:(copy.deepcopy(cfg),copy.deepcopy(protocol)))
    monkeypatch.setattr(m,'cpu_guard',lambda:None)
    monkeypatch.setattr(m.screen,'evidence_root',lambda:tmp_path)
    monkeypatch.setattr(m.screen,'_code_snapshot',lambda c:{'commit':c})
    calls=[]
    def call(name):
        def inner(**kwargs):calls.append((name,kwargs))
        return inner
    for function in ('prepare_automatic_candidates','prepare_automatic_task_suites','qualify_scene'):
        monkeypatch.setattr(m.screen,function,call(function))
    qualifier=stage/'scene_qualifiers/first';put(qualifier/'manifest.json',{})
    task='first__obj_to_region'
    cells=[dict(task_id=task,policy_id=arm,episode=episode,
        cell_id=f'{arm.lower()}__{task}__seed0__ep{episode}') for arm in ('A0','A4') for episode in range(5)]
    result={'metric_rows':cells,'gate':{'cell_count':10,'exact_900_step_cells':0,'strict_pass_task_ids':[],
        'task_summaries':[{'task_id':task}]}}
    monkeypatch.setattr(m.screen,'_validate_qualifier_output',lambda **kw:result)
    def publish(out,**kw):
        out.mkdir(parents=True)
        for name,data in kw['payloads'].items():(out/name).write_bytes(data)
    monkeypatch.setattr(m.screen,'_publish_bundle',publish)
    return stage,config_path,calls,result


def test_existing_qualifier_replayed_and_negative_results_retained(execution):
    stage,cfg,calls,_=execution
    r=m.run(config_path=cfg,stage_root=stage,expected_commit='code',scene_id='first')
    assert r['status']=='QUALIFICATION_COMPLETE' and r['strict_pass_task_ids']==[]
    assert r['simulator_cells']==0 and r['prerequisite_cells']==10 and r['policy_success'] is None
    assert [c[0] for c in calls]==['prepare_automatic_candidates','prepare_automatic_task_suites']
    assert calls[0][1]['qualification_protocol']=='protocol' and calls[0][1]['export'] is True
    with pytest.raises(FileExistsError):m.run(config_path=cfg,stage_root=stage,expected_commit='code',scene_id='first')


def test_zero_query_scene_preserved_without_exports_or_fake_measurements(execution):
    stage,cfg,calls,_=execution;r=m.run(config_path=cfg,stage_root=stage,expected_commit='code',scene_id='empty')
    assert r['status']=='NOT_RUN' and r['planned_qualification_cells']==0 and r['qualifier'] is None and calls==[]


def test_missing_pilot_blocks_remaining_without_exports(execution):
    stage,cfg,calls,_=execution
    with pytest.raises(Exception):m.run(config_path=cfg,stage_root=stage,expected_commit='code',scene_id='second')
    assert calls==[]


def test_planner_failure_keeps_selected_denominator_and_null_telemetry(execution,monkeypatch):
    stage,cfg,_,_=execution
    def fail(**kw):raise m.screen.CandidateScreenError(m.PLANNING_UNQUALIFIED)
    monkeypatch.setattr(m.screen,'prepare_automatic_task_suites',fail)
    r=m.run(config_path=cfg,stage_root=stage,expected_commit='code',scene_id='first')
    assert r['status']=='PLANNING_UNQUALIFIED' and r['planned_qualification_cells']==10
    assert r['reset_definitions'] is None and r['camera'] is None and r['policy_success'] is None


def test_unexpected_planner_bug_not_relabelled_as_scientific_failure(execution,monkeypatch):
    stage,cfg,_,_=execution
    def fail(**kw):raise m.screen.CandidateScreenError('other bug')
    monkeypatch.setattr(m.screen,'prepare_automatic_task_suites',fail)
    with pytest.raises(m.screen.CandidateScreenError,match='other bug'):
        m.run(config_path=cfg,stage_root=stage,expected_commit='code',scene_id='first')
    assert not (stage/'qualification_handoff/first').exists()


def test_missing_canonical_cells_rejected(execution):
    stage,cfg,_,result=execution;result['metric_rows'].pop()
    with pytest.raises(ValueError,match='omitted'):
        m.run(config_path=cfg,stage_root=stage,expected_commit='code',scene_id='first')


@pytest.fixture
def bound_stage(tmp_path,monkeypatch):
    stage=tmp_path/'outputs/icra2027/new';stage.mkdir(parents=True)
    runtime=m.shared.identity(Path(m.sys.executable).resolve())
    protocol=put(tmp_path/'protocol.json',{'source_populations':[{'scene_id':'first'}],
        'qualification_tasks':[{'scene_id':'first','task_id':'first_task'}]})
    source={'protocol':protocol,'e3_root':'original'}
    config=dict(schema_version=1,scope=m.SCOPE,paper_ready=False,freeze_id='new',source=source,
        python=runtime['path'],menagerie_root='menagerie',menagerie={'identity':'fixed'},openpi_resize_identity={'resize':'fixed'},
        pilot_scene='first',planned_scenes=50,planned_objects=1871,planned_semantic_queries=269,
        planned_qualification_cells=2690,input_semantic_queries=6155,budget_exclusions=5886)
    path=Path(put(tmp_path/'config.json',config)['path'])
    bound_config=m.shared.identity(path)
    contract={'freeze_id':'new','code':{'commit':'code','dirty':False}}
    put(stage/'contract/freeze_manifest.json',contract)
    monkeypatch.setattr(m.screen,'evidence_root',lambda:tmp_path)
    monkeypatch.setattr(m.screen,'_code_snapshot',lambda expected:{'commit':'code'})
    monkeypatch.setattr(m.shared,'contract_digest',lambda c:'digest')
    monkeypatch.setattr(m.shared,'resource',lambda c,n:bound_config if n=='e4_full_qualification_config' else runtime)
    monkeypatch.setattr(m,'inspect_source',lambda *a:dict(status='READY',source=copy.deepcopy(source)))
    monkeypatch.setattr(m.camera,'_menagerie_snapshot',lambda *a:{'identity':'fixed','files':[]})
    monkeypatch.setattr(m.camera,'_openpi_snapshot',lambda:{'resize':'fixed'})
    return stage,path,config,source


def test_stage_binds_source_config_runtime_and_denominator(bound_stage):
    stage,path,cfg,_=bound_stage
    assert m.validate_stage(path,stage,'code')[0]==cfg


@pytest.mark.parametrize('kind',['config','source','commit','freeze','dirty','pilot','denominator','runtime','model'])
def test_stage_binding_tamper_rejected(bound_stage,monkeypatch,kind):
    stage,path,cfg,source=bound_stage
    if kind in ('commit','freeze','dirty'):
        p=stage/'contract/freeze_manifest.json';v=m.shared.read(p)
        if kind=='commit':v['code']['commit']='wrong'
        elif kind=='freeze':v['freeze_id']='wrong'
        else:v['code']['dirty']=True
        p.write_text(json.dumps(v))
    elif kind=='source':source['e3_root']='changed'
    else:
        if kind=='config':cfg['paper_ready']=True
        elif kind=='pilot':cfg['pilot_scene']='other'
        elif kind=='denominator':cfg['planned_semantic_queries']=268
        elif kind=='runtime':cfg['python']='wrong'
        elif kind=='model':cfg['menagerie']={'identity':'other'}
        path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):m.validate_stage(path,stage,'code')


@pytest.mark.parametrize('field,value',[('task_id','foreign_task'),('policy_id','A2'),('episode',7),('cell_id','foreign')])
def test_same_count_replaced_cell_identity_rejected(execution,field,value):
    stage,cfg,_,result=execution;result['metric_rows'][0][field]=value
    with pytest.raises(ValueError,match='replaced'):
        m.run(config_path=cfg,stage_root=stage,expected_commit='code',scene_id='first')


def test_same_count_replaced_task_summary_rejected(execution):
    stage,cfg,_,result=execution;result['gate']['task_summaries'][0]['task_id']='foreign'
    with pytest.raises(ValueError,match='replaced'):
        m.run(config_path=cfg,stage_root=stage,expected_commit='code',scene_id='first')
