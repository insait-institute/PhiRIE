"""Exact-source reuse and full query coverage, without rendering or GT fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from run.icra2027 import e6_grounding_cohort as d


def test_measurement_partition_excludes_only_publication_freeze():
    cfg=dict(schema_version=1,scope='e6_public_grounding_v1',freeze_id='old',source_commit=d.PRODUCER_SHA,
             construction={'sha256':'input'},discovery_runtime={'sha256':'sam'},rgb_runtime={'sha256':'rgb'},protocol={'p':1})
    assert d.recipe(cfg)==d.recipe({**cfg,'freeze_id':'new'})
    for key in ('construction','discovery_runtime','rgb_runtime','protocol','source_commit'):
        changed={**cfg,key:{'changed':True}}
        if key=='source_commit':
            with pytest.raises(ValueError):d.recipe(changed)
        else:assert d.recipe(changed)!=d.recipe(cfg)
    with pytest.raises(ValueError):d.recipe({**cfg,'ignored_metric_change':True})


def context(monkeypatch,tmp_path):
    root=tmp_path/'evidence';root.mkdir();stage=root/'outputs/icra2027/20260906-abcdef0-v1';stage.mkdir(parents=True)
    monkeypatch.setattr(d.shared,'ROOT',root)
    monkeypatch.setattr(d.shared,'git_snapshot',lambda p:dict(commit='wrapper',dirty=False))
    monkeypatch.setattr(d,'source',lambda ref:tmp_path/'source')
    monkeypatch.setattr(d.socket,'gethostname',lambda:'hala')
    rows=[dict(scene_id=s,condition_id=c,query_hashes={f'q{i}':str(i) for i in range(4)}) for s in d.shared.SCENES for c in d.shared.CONDITIONS]
    roster=root/'roster.json';d.shared.write_new_json(roster,dict(rows=rows))
    construction=root/'construction.json';d.shared.write_new_json(construction,dict(roster=d.shared.identity(roster)))
    refs=[]
    for i in range(3):
        directory=root/f'original{i}';directory.mkdir();path=directory/'execution.json'
        old=dict(schema_version=1,scope='e6_public_grounding_v1',freeze_id=str(i),source_commit=d.PRODUCER_SHA,
                 construction=d.shared.identity(construction),discovery_runtime={'fixed':'sam'},rgb_runtime={'fixed':'rgb'},protocol={'fixed':True})
        d.shared.write_new_json(path,old);refs.append(d.shared.identity(path))
    reuse=[]
    for scene,condition in d.REUSED:
        ref=refs[1 if scene=='behavior_task0023' else 2]
        unit=Path(ref['path']).parent/'audit/public_grounding'/(scene+'_'+condition)/'terminal';unit.mkdir(parents=True)
        for filename in ('manifest.json','seal.json'):d.shared.write_new_json(unit/filename,{'fixture':scene+condition})
        reuse.append(dict(scene_id=scene,condition_id=condition,config=ref,
                          terminal_manifest=d.shared.identity(unit/'manifest.json'),terminal_seal=d.shared.identity(unit/'seal.json')))
    audits={}
    for c in d.shared.CONDITIONS:
        p=root/(c+'_audit.json');d.shared.write_new_json(p,{'fixture':c});audits[c]=d.shared.identity(p)
    cfg=dict(schema_version=1,scope='e6_public_grounding_full18',freeze_id=stage.name,source_commit='wrapper',
             native_replay_audits=audits,validation_host='hala',
             measurement_producer={'path':'p','commit':d.PRODUCER_SHA},fresh_config=refs[0],reference_configs=refs[1:],
             roster=d.shared.identity(roster),recipe_sha256=d.shared.canonical_hash(d.recipe(old)),reuse=reuse)
    path=stage/'execution.json';d.shared.write_new_json(path,cfg)
    monkeypatch.setattr(d,'original_roster',lambda construction,ref:d.shared.read(ref['path']))
    calls=[]
    def replay(cfg,ref,scene=None,condition=None):
        calls.append((scene,condition));return (None if scene is None else {'stage_status':'PASS'}),{}
    monkeypatch.setattr(d,'replay',replay)
    return stage,path,cfg,rows,calls


def test_full_admission_replays_all_four_original_gates(monkeypatch,tmp_path):
    stage,path,cfg,rows,calls=context(monkeypatch,tmp_path)
    d.validate(path,stage,require_e0=False)
    assert calls[-4:]==d.REUSED and len(rows)==18


@pytest.mark.parametrize('field',['discovery_runtime','rgb_runtime','protocol','construction'])
def test_changed_measurement_dependency_rejected(monkeypatch,tmp_path,field):
    stage,path,cfg,rows,calls=context(monkeypatch,tmp_path)
    ref=cfg['reference_configs'][0];p=Path(ref['path']);old=d.shared.read(p);old[field]={'drift':True};p.write_text(json.dumps(old))
    cfg['reference_configs'][0]=d.shared.identity(p)
    for r in cfg['reuse']:
        if r['scene_id']=='behavior_task0023':r['config']=cfg['reference_configs'][0]
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='recipe'):d.validate(path,stage,require_e0=False)


def test_pilot_failure_cannot_open_full_gate(monkeypatch,tmp_path):
    stage,path,cfg,rows,calls=context(monkeypatch,tmp_path)
    monkeypatch.setattr(d,'replay',lambda cfg,ref,scene=None,condition=None:({'stage_status':'FAIL'} if scene else None,{}))
    with pytest.raises(ValueError,match='integrity gate failed'):d.validate(path,stage,require_e0=False)


def test_reuse_subset_and_changed_seal_rejected(monkeypatch,tmp_path):
    stage,path,cfg,rows,calls=context(monkeypatch,tmp_path)
    cfg['reuse']=cfg['reuse'][:-1];path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='four reuse'):d.validate(path,stage,require_e0=False)


def publish_fixture(monkeypatch,tmp_path,status='PASS',reused=True):
    root=tmp_path/'evidence';root.mkdir();stage=root/'full';stage.mkdir();original=root/'original';original.mkdir()
    monkeypatch.setattr(d.shared,'ROOT',root)
    scene,condition=d.REUSED[0] if reused else ('behavior_task0011','severe')
    ref={'path':str(original/'execution.json'),'sha256':'config','size_bytes':1}
    query_hashes={f'q{i}':str(i) for i in range(4)}
    queries=[dict(task_id=k,query_sha256=v,roles={'manipulated_object':{'status':'unresolved'}}) for k,v in query_hashes.items()]
    unit=original/'audit/public_grounding'/(scene+'_'+condition);(unit/'terminal').mkdir(parents=True)
    gate=dict(scene_id=scene,condition_id=condition,planned_queries=4,planned_cohort_queries=72,feature_rows_written=0,
              paper_ready=False,stage_status=status,queries=queries,robot_frame=None,physics_verified=False)
    for name in ('gate.json','manifest.json','seal.json'):d.shared.write_new_json(unit/'terminal'/name,gate if name=='gate.json' else {'fixture':True})
    cfg=dict(scope='e6_public_grounding_full18',source_commit='wrapper',freeze_id='full',measurement_producer={'commit':d.PRODUCER_SHA},
             recipe_sha256='recipe',reuse=[dict(scene_id=scene,condition_id=condition,config=ref)] if reused else [],fresh_config=ref)
    roster=dict(rows=[dict(scene_id=scene,condition_id=condition,query_hashes=query_hashes)])
    monkeypatch.setattr(d,'validate',lambda *a,**k:(cfg,roster))
    monkeypatch.setattr(d,'replay',lambda *a:(d.shared.read(unit/'terminal/gate.json'),{'returncode':0 if status=='PASS' else 3}))
    return stage,cfg,roster,scene,condition,unit


def test_reuse_preserves_original_missing_roles_and_link(monkeypatch,tmp_path):
    stage,cfg,roster,scene,c,unit=publish_fixture(monkeypatch,tmp_path)
    report=d.publish('cfg',stage,scene,c)
    assert report['mode']=='reused_original' and report['original_gate']['robot_frame'] is None
    assert all(q['roles']['manipulated_object']['status']=='unresolved' for q in report['original_gate']['queries'])
    assert d.validate_output('cfg',stage,scene,c)==report
    with pytest.raises(FileExistsError):d.publish('cfg',stage,scene,c)
    (unit/'terminal/seal.json').write_text('{}')
    with pytest.raises(ValueError,match='attribution'):d.validate_output('cfg',stage,scene,c)


def test_failed_constructor_four_queries_remain_failed_not_measured_success(monkeypatch,tmp_path):
    stage,cfg,roster,scene,c,unit=publish_fixture(monkeypatch,tmp_path,status='FAIL',reused=False)
    result=d.publish('cfg',stage,scene,c)
    assert result['stage_status']=='FAIL' and len(result['original_gate']['queries'])==4
    assert result['original_gate']['physics_verified'] is False and result['feature_rows_written']==0


def test_changed_query_ids_rejected(monkeypatch,tmp_path):
    stage,cfg,roster,scene,c,unit=publish_fixture(monkeypatch,tmp_path)
    gate=d.shared.read(unit/'terminal/gate.json');gate['queries'].pop();(unit/'terminal/gate.json').write_text(json.dumps(gate))
    with pytest.raises(ValueError,match='query/denominator'):d.publish('cfg',stage,scene,c)


def test_empty_snapshot_retains_all18_conditions72_queries(monkeypatch,tmp_path):
    stage,path,cfg,rows,calls=context(monkeypatch,tmp_path)
    monkeypatch.setattr(d,'validate',lambda *a,**k:(cfg,dict(rows=rows)))
    result=d.snapshot(path,stage,stage/'snapshot')
    assert result['counts']=={'PASS':0,'FAIL':0,'INCOMPLETE':0,'NOT_RUN':18}
    assert sum(r['planned_queries'] for r in result['rows'])==72 and not result['complete'] and not result['paper_ready']


def test_missing_original_source_fails_before_replay(tmp_path):
    with pytest.raises(FileNotFoundError):d.source({'path':str(tmp_path/'absent'),'commit':d.PRODUCER_SHA})


@pytest.mark.parametrize('dirty,commit,top_matches',[(True,d.PRODUCER_SHA,True),(False,'changed',True),(False,d.PRODUCER_SHA,False)])
def test_dirty_changed_or_nonroot_producer_rejected(monkeypatch,tmp_path,dirty,commit,top_matches):
    p=tmp_path/'producer';p.mkdir();monkeypatch.setattr(d.shared,'ROOT',tmp_path/'evidence')
    monkeypatch.setattr(d.shared,'git_snapshot',lambda path:dict(commit=commit,dirty=dirty))
    monkeypatch.setattr(d.subprocess,'check_output',lambda *a,**k:str(p if top_matches else tmp_path)+'\n')
    with pytest.raises(ValueError,match='source changed'):d.source({'path':str(p),'commit':d.PRODUCER_SHA})


def test_changed_link_target_cannot_masquerade_as_original(monkeypatch,tmp_path):
    stage,cfg,roster,scene,c,unit=publish_fixture(monkeypatch,tmp_path)
    d.publish('cfg',stage,scene,c)
    link=stage/'audit/public_grounding_cohort'/(scene+'_'+c)/'artifacts';link.unlink();link.symlink_to(tmp_path)
    with pytest.raises(ValueError,match='linked artifacts'):d.validate_output('cfg',stage,scene,c)


def native_fixture(monkeypatch,tmp_path):
    from types import SimpleNamespace
    root=tmp_path/'evidence';root.mkdir();monkeypatch.setattr(d.shared,'ROOT',root)
    stage=root/'original';stage.mkdir();(stage/'launchers').mkdir();code=tmp_path/'code';code.mkdir()
    scene='behavior_task0020';condition='clean';unit=stage/'audit/public_grounding'/f'{scene}_{condition}';unit.mkdir(parents=True)
    config=stage/'execution.json';d.shared.write_new_json(config,{'fixture':True});ref=d.shared.identity(config)
    (unit/'derived_mesh.ply').write_bytes(b'actual mesh bytes')
    gate={'original_construction':{'fixed':'input'},'stage_status':'PASS','queries':[{'value':1.258466940000758}]}
    d.shared.sealed._publish_bundle(unit/'terminal',manifest_kind='e6_public_grounding',
        payloads={'gate.json':(json.dumps(gate)+'\n').encode()},manifest_fields={'output_members':{'derived_mesh.ply':d.shared.file_identity(unit/'derived_mesh.ply')}})
    script=code/'auditor.py';script.write_bytes(b'approved fixture auditor');scriptref=d.shared.identity(script)
    monkeypatch.setattr(d,'NATIVE_AUDIT_SCRIPT_SHA',scriptref['sha256'])
    stdout=stage/'launchers/stdout.json';stdout.write_bytes((unit/'terminal/gate.json').read_bytes());stderr=stage/'launchers/stderr.log';stderr.write_text('')
    allocation=stage/'launchers/allocation_833155.json';d.shared.write_new_json(allocation,dict(condition='clean',job_id='833155',node='sof1-h200-3',phase='finalize',resource_class='cpu'))
    rgb={'python':'python'}
    command=['python','-m','run.icra2027.e6_public_grounding','--config',str(config),'--stage-root',str(stage),'--scene',scene,'--condition',condition,'--validate-output']
    audit=dict(schema_version=1,scope='exact_original_host_e7_validation',status='PASS',returncode=0,condition='clean',
       source={'commit':d.PRODUCER_SHA,'dirty':False},code_root=str(code),config=ref,command=command,metric_tolerance_changed=False,model_rerun=False,paper_ready=False,
       audit_script=scriptref,stdout=d.shared.identity(stdout),stderr=d.shared.identity(stderr),original_allocation=d.shared.identity(allocation),original_finalizer_job_id='833155',hostname='sof1-h200-3',job_id='833176',
       terminal={f:d.shared.identity(unit/'terminal'/f) for f in ('gate.json','manifest.json','seal.json')})
    audit_path=stage/'launchers/audit.json';d.shared.write_new_json(audit_path,audit)
    cfg={'native_replay_audits':{'clean':d.shared.identity(audit_path)}}
    monkeypatch.setattr(d.shared,'environment',lambda *args:{})
    calls=[]
    def run(command,**kwargs):
        calls.append(command);return SimpleNamespace(returncode=0,stdout=json.dumps(gate['original_construction']))
    monkeypatch.setattr(d.subprocess,'run',run)
    return cfg,ref,scene,condition,code,rgb,unit,audit_path,audit,calls


def test_native_audit_retains_exact_float_without_local_association(monkeypatch,tmp_path):
    cfg,ref,scene,c,code,rgb,unit,path,audit,calls=native_fixture(monkeypatch,tmp_path)
    gate,receipt=d.native_replay(cfg,ref,scene,c,code,rgb)
    assert gate['queries'][0]['value']==1.258466940000758
    assert receipt['original_host']=='sof1-h200-3'
    assert len(calls)==1 and 'd.original_unit' in calls[0][-1] and 'associate' not in calls[0][-1]


@pytest.mark.parametrize('field,value',[('hostname','hala'),('metric_tolerance_changed',True),('returncode',3),('code_root','wrong')])
def test_native_audit_wrong_host_tolerance_failure_source_rejected(monkeypatch,tmp_path,field,value):
    cfg,ref,scene,c,code,rgb,unit,path,audit,calls=native_fixture(monkeypatch,tmp_path)
    audit[field]=value;path.write_text(json.dumps(audit));cfg['native_replay_audits']['clean']=d.shared.identity(path)
    with pytest.raises(ValueError):d.native_replay(cfg,ref,scene,c,code,rgb)


def test_native_audit_changed_mesh_rejected(monkeypatch,tmp_path):
    cfg,ref,scene,c,code,rgb,unit,path,audit,calls=native_fixture(monkeypatch,tmp_path)
    (unit/'derived_mesh.ply').write_bytes(b'changed geometry')
    with pytest.raises(ValueError,match='output bytes changed'):d.native_replay(cfg,ref,scene,c,code,rgb)
    assert not calls


def test_publication_host_is_explicit_not_a_tolerance_fallback(monkeypatch,tmp_path):
    stage,path,cfg,rows,calls=context(monkeypatch,tmp_path)
    monkeypatch.setattr(d.socket,'gethostname',lambda:'different-host')
    with pytest.raises(ValueError,match='Hala host'):d.validate(path,stage,require_e0=False)


def test_native_audit_forged_script_or_changed_input_rejected(monkeypatch,tmp_path):
    from types import SimpleNamespace
    cfg,ref,scene,c,code,rgb,unit,path,audit,calls=native_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(d.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='{"changed":"input"}'))
    with pytest.raises(ValueError,match='construction inputs changed'):d.native_replay(cfg,ref,scene,c,code,rgb)
    script=Path(audit['audit_script']['path']);script.write_text('forged script')
    audit['audit_script']=d.shared.identity(script);path.write_text(json.dumps(audit));cfg['native_replay_audits']['clean']=d.shared.identity(path)
    with pytest.raises(ValueError,match='source/command/verdict'):d.native_replay(cfg,ref,scene,c,code,rgb)


def test_original_roster_uses_pinned_source_loader(monkeypatch,tmp_path):
    from run.icra2027 import e6_public_reconstruction_full as original
    from types import SimpleNamespace
    p=tmp_path/'roster.json';p.write_text('{"rows": []}')
    ref=d.shared.identity(p);producer_ref={'path':'original7c','commit':original.PRODUCER_SHA}
    seen=[]
    def load(r):
        seen.append(r)
        return SimpleNamespace(validate_roster=lambda x:{**x,'original_source_authenticated':True})
    monkeypatch.setattr(original,'producer',load)
    monkeypatch.setattr(d.shared,'validate_roster',lambda _:pytest.fail('wrapper-relative roster validator forbidden'))
    result=d.original_roster({'roster':ref,'measurement_producer':producer_ref},ref)
    assert seen==[producer_ref] and result['original_source_authenticated']
    p.write_text('{"rows": ["changed"]}')
    with pytest.raises(ValueError,match='identity changed'):
        d.original_roster({'roster':ref,'measurement_producer':producer_ref},ref)


def test_original_roster_rejects_unpinned_query_source(tmp_path):
    from run.icra2027 import e6_public_reconstruction_full as original
    p=tmp_path/'roster.json';p.write_text('{"rows": []}');ref=d.shared.identity(p)
    with pytest.raises(ValueError,match='pinned 7c'):
        d.original_roster({'roster':ref,'measurement_producer':{'path':str(tmp_path),'commit':'changed'}},ref)
