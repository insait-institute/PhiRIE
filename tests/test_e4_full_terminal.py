"""Full terminal coverage retains absent prerequisites without fabricating rollouts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from run.icra2027 import e4_planning_terminal as terminal
from run.icra2027 import e4_full_terminal_worker as worker


@pytest.fixture
def full_observed(monkeypatch):
    scenes={};tasks=[];cells=[]
    for i in range(50):
        sid=f'{i:010d}';n=(7 if i<23 else 6) if i<41 else 0
        selected=[f'{sid}__obj_{j}_to_region' for j in range(n)]
        status=('EXPORT_REJECTED' if i==0 else 'PLANNING_UNAVAILABLE' if i==1
                else 'QUALIFICATION_COMPLETE' if n else 'SOURCE_NO_QUERIES')
        scenes[sid]=dict(status=status,selected_task_ids=selected,planned_cells=n*10,
            actual_900_step_cells=0,qualified_cells=0,strict_pass_task_ids=[],
            policy_executed=0,policy_success=None)
        for task in selected:
            tasks.append(dict(task_id=task,scene_id=sid))
            for arm in ('A0','A4'):
                for ep in range(5):
                    cid=f'{arm.lower()}__{task}__seed0__ep{ep}'
                    measured=status=='QUALIFICATION_COMPLETE'
                    evidence=dict(cell_id=cid,passed=False) if measured else None
                    cells.append(dict(scene_id=sid,task_id=task,policy_id=arm,episode=ep,cell_id=cid,
                        qualification_state='FAIL' if measured else 'NOT_RUN',
                        source_kind='canonical_qualifier' if measured else status.lower(),
                        qualification_evidence=evidence,policy_executed=False,policy_success=None,
                        reset_definition=None,camera=None,physics=None))
    source=dict(source=dict(scenes={s:{} for s in scenes}),planned_scenes=50,planned_objects=1871,
        input_semantic_queries=6155,budget_exclusions=5886,planned_semantic_queries=269,
        planned_qualification_cells=2690)
    observed=dict(source_config=source,protocol=dict(qualification_tasks=tasks),scenes=scenes,cells=cells)
    monkeypatch.setattr(terminal,'_full_replay',lambda c:copy.deepcopy(observed))
    return observed,dict(jobs={s:{} for s in scenes},source={},freeze_id='fresh')


def test_full_all_cells_and_negative_measurements_preserved(full_observed):
    _,config=full_observed;gate,scenes,cells=terminal._full_payload(config,{})
    assert gate['planned_qualification_cells']==len(cells)==2690
    assert gate['input_semantic_queries']==6155 and gate['budget_exclusions']==5886
    assert gate['scene_status_counts']==dict(EXPORT_REJECTED=1,PLANNING_UNAVAILABLE=1,
        QUALIFICATION_COMPLETE=39,SOURCE_NO_QUERIES=9)
    assert gate['prerequisite_checked_cells']==2550 and gate['actual_900_step_cells']==0
    assert gate['policy_target_episodes']==160 and gate['instantiated_policy_episodes']==0
    assert gate['policy_success'] is None and gate['claim_gate']=='NOT_RUN'
    assert not gate['paper_ready'] and gate['rollout_ledger'] is None
    assert sum(r['qualification_state']=='NOT_RUN' for r in cells)==140


@pytest.mark.parametrize('change',['drop_scene','drop_cell','duplicate_cell','replace_task',
    'fake_policy','fake_reset','fake_pass','fake_simulator','fake_summary','fake_strict',
    'fake_source_only','drop_input','drop_budget','wrong_cell_id','wrong_scene_status'])
def test_full_fail_closed_population_and_telemetry(full_observed,change):
    observed,config=full_observed;r=observed['cells'][0];s=observed['scenes']['0000000000']
    if change=='drop_scene':observed['scenes'].pop('0000000049')
    elif change=='drop_cell':observed['cells'].pop()
    elif change=='duplicate_cell':observed['cells'][-1]=copy.deepcopy(r)
    elif change=='replace_task':r['task_id']='replacement_after_results'
    elif change=='fake_policy':r['policy_executed']=True
    elif change=='fake_reset':r['reset_definition']={}
    elif change=='fake_pass':r['qualification_state']='PASS'
    elif change=='fake_simulator':s['actual_900_step_cells']=1
    elif change=='fake_summary':s['qualified_cells']=1
    elif change=='fake_strict':s['strict_pass_task_ids']=[s['selected_task_ids'][0]]
    elif change=='fake_source_only':s['status']='SOURCE_NO_QUERIES'
    elif change=='drop_input':observed['source_config']['input_semantic_queries']=269
    elif change=='drop_budget':observed['source_config']['budget_exclusions']=0
    elif change=='wrong_cell_id':r['cell_id']='replacement'
    else:s['status']='PASS'
    with pytest.raises(ValueError):terminal._full_payload(config,{})


def test_full_output_no_overwrite(tmp_path,monkeypatch):
    out=tmp_path/'full_qualification';out.mkdir();(out/'original').write_text('preserve')
    monkeypatch.setattr(terminal.screen,'_code_snapshot',lambda c:{})
    monkeypatch.setattr(terminal,'_validate_full_stage',lambda *a:({},tmp_path))
    monkeypatch.setattr(terminal,'_full_payload',lambda *a:pytest.fail('must reject before replay'))
    with pytest.raises(FileExistsError):terminal.produce_full(config_path='config',stage_root=tmp_path,
        expected_code_commit='x',out=out)
    assert (out/'original').read_text()=='preserve'


@pytest.fixture
def failed_worker(tmp_path,monkeypatch):
    from run.icra2027 import e4_full_qualification as full
    from run.icra2027 import e4_compact_materialization as api
    screen=terminal.screen;sid='scene';stage=tmp_path/'freeze';stage.mkdir()
    task=f'{sid}__obj_1000_to_region';d=stage/'automatic_candidates'/sid
    factories={p:d/'materialized'/p for p in ('A0','A4')}
    report={'collision_exclusion':{'primitive_intrusions':[{'geom':'room_support_1'}],
                                   'unresolved_intrusion_count':1}}
    paths=[stage/'logs'/f'{sid}-123.err',stage/'logs'/f'{sid}-123.out']
    paths += [factories['A0']/'sim_export'/n for n in ('scene.xml','room_collision_report.json','mujoco_settle.json','isaac_manifest.json')]
    paths += [d/'materialized/shared_room_static'/n for n in ('manifest.json','seal.json','payload.json')]
    paths += [factories[p]/'materialization_manifest.json' for p in factories]
    for path in paths:path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
    paths[0].write_text('robo.eval.e4_candidate_screen.CandidateScreenError: '+worker.COLLISION+'\n')
    (factories['A0']/'sim_export/room_collision_report.json').write_text(json.dumps(report))
    request=dict(source=dict(e0={'path':str(stage/'contract/freeze_manifest.json')},config={'path':'config'},code_commit='original',code_root='old'),jobs={sid:dict(job_id='123',state='FAILED',exit_code='1:0')})
    monkeypatch.setattr(full,'validate_stage',lambda *a:({'source':{'scenes':{sid:{}}}},
        {'qualification_tasks':[dict(task_id=task,scene_id=sid)]}))
    monkeypatch.setattr(full,'cpu_guard',lambda:None)
    monkeypatch.setattr(screen,'evidence_root',lambda:tmp_path)
    monkeypatch.setattr(screen,'_automatic_export_context',lambda *a,**kw:dict(
        rosters={p:dict(accepted_slots=['obj_1000']) for p in factories},object_slots=['obj_1000']))
    monkeypatch.setattr(screen,'_load_automatic_static_package',lambda *a,**kw:{'room_report':report})
    def reject(*a,**kw):raise screen.CandidateScreenError(worker.COLLISION)
    monkeypatch.setattr(screen,'validate_full_room_export',reject)
    return request,stage,factories,report


def test_real_validator_rejection_retains_null_cells(failed_worker):
    request,stage,_,_=failed_worker;r=worker.audit(request)
    assert r['scenes']['scene']['status']=='EXPORT_REJECTED'
    assert len(r['cells'])==10
    assert all(c['qualification_state']=='NOT_RUN' and c['physics'] is None for c in r['cells'])
    assert str(stage/'logs/scene-123.err') in r['scenes']['scene']['evidence']


@pytest.mark.parametrize('change',['missing_report','changed_report','no_intrusion','different_error',
    'validator_passes','fake_log','missing_log','job_oom','existing_measurements'])
def test_collision_rejection_requires_report_log_and_canonical_replay(failed_worker,monkeypatch,change):
    request,stage,factories,report=failed_worker
    path=factories['A0']/'sim_export/room_collision_report.json'
    if change=='missing_report':path.unlink()
    elif change=='changed_report':path.write_text('{}')
    elif change=='no_intrusion':
        report['collision_exclusion']['primitive_intrusions']=[];path.write_text(json.dumps(report))
    elif change=='different_error':
        def reject(*a,**kw):raise terminal.screen.CandidateScreenError('unrelated source failure')
        monkeypatch.setattr(terminal.screen,'validate_full_room_export',reject)
    elif change=='validator_passes':monkeypatch.setattr(terminal.screen,'validate_full_room_export',lambda *a,**kw:{})
    elif change=='fake_log':(stage/'logs/scene-123.err').write_text('fabricated '+worker.COLLISION)
    elif change=='missing_log':(stage/'logs/scene-123.err').unlink()
    elif change=='job_oom':request['jobs']['scene']['state']='OUT_OF_MEMORY'
    else:(stage/'scene_qualifiers/scene').mkdir(parents=True)
    with pytest.raises((ValueError,FileNotFoundError,terminal.screen.CandidateScreenError)):worker.audit(request)


def test_no_tabletop_requires_actual_planner_rejection(failed_worker,monkeypatch):
    request,stage,_,_=failed_worker
    (stage/'logs/scene-123.err').write_text('robo.eval.e4_candidate_screen.CandidateScreenError: '+worker.NO_TABLE+'\n')
    for name in ('gate.json','manifest.json','seal.json'):
        p=stage/'automatic_candidates/scene/population'/name;p.parent.mkdir(exist_ok=True);p.write_text('{}')
    def reject(**kw):raise terminal.screen.CandidateScreenError(worker.NO_TABLE)
    monkeypatch.setattr(terminal.screen,'_automatic_prepare_inputs',reject)
    assert worker.audit(request)['scenes']['scene']['status']=='PLANNING_UNAVAILABLE'
    monkeypatch.setattr(terminal.screen,'_automatic_prepare_inputs',lambda **kw:{})
    with pytest.raises(ValueError,match='does not replay'):worker.audit(request)


@pytest.fixture
def stage_inputs(tmp_path,monkeypatch):
    stage=tmp_path/'fresh';(stage/'contract').mkdir(parents=True)
    config_path=tmp_path/'config.json';config={'scope':terminal.FULL_CONFIG_SCOPE,
        'schema_version':1,'paper_ready':False,'freeze_id':'fresh'}
    config_path.write_text(json.dumps(config))
    code={'code_root':str(tmp_path),'commit':'a'*40,'dirty':False}
    e0={'freeze_id':'fresh','code':{'commit':code['commit'],'repository':str(tmp_path),'dirty':False}}
    (stage/'contract/freeze_manifest.json').write_text(json.dumps(e0))
    monkeypatch.setattr(terminal.api,'contract_digest',lambda c:'digest')
    monkeypatch.setattr(terminal.api,'resource',lambda c,k:terminal.api.identity(config_path))
    monkeypatch.setattr(terminal,'_exact_code',lambda *a:code)
    return stage,config_path,config,e0,code


def test_full_stage_authenticates_config_and_e0(stage_inputs):
    stage,path,config,e0,code=stage_inputs
    observed,_=terminal._validate_full_stage(path,stage,code['commit'],stage/'full_qualification',code)
    assert observed==config


@pytest.mark.parametrize('field,value',[('freeze_id','old'),('schema_version',2),('paper_ready',True),('scope','compact')])
def test_full_stage_config_scope_not_relabelable(stage_inputs,field,value):
    stage,path,config,e0,code=stage_inputs;config[field]=value;path.write_text(json.dumps(config))
    with pytest.raises(ValueError):terminal._validate_full_stage(path,stage,code['commit'],stage/'full_qualification',code)


@pytest.mark.parametrize('change',['dirty','wrong_source','wrong_freeze','wrong_resource','wrong_output'])
def test_full_stage_negative_contract(stage_inputs,monkeypatch,change):
    stage,path,config,e0,code=stage_inputs;out=stage/'full_qualification'
    if change=='dirty':e0['code']['dirty']=True
    elif change=='wrong_source':e0['code']['commit']='b'*40
    elif change=='wrong_freeze':e0['freeze_id']='old'
    elif change=='wrong_resource':monkeypatch.setattr(terminal.api,'resource',lambda *a:{})
    else:out=stage/'elsewhere'
    (stage/'contract/freeze_manifest.json').write_text(json.dumps(e0))
    with pytest.raises(ValueError):terminal._validate_full_stage(path,stage,code['commit'],out,code)


def test_original_full_replay_uses_original_cwd_and_fixed_environment(tmp_path,monkeypatch):
    source={'config':{'path':'config'},'e0':{'path':str(tmp_path/'old/contract/freeze_manifest.json')},
        'code_commit':'oldcommit','code_root':str(tmp_path/'original'),
        'environment':{'path':'environment.sh'},'runtime':{'path':'original-python'}}
    compatibility=tmp_path/'compat.json';compatibility.write_text('{}')
    config={'source':source,'compatibility':terminal.api.identity(compatibility),
        'cpu_feature_mode':terminal.ISA_MODE}
    monkeypatch.setattr(terminal,'full_source_contract',lambda *a:source)
    monkeypatch.setattr(terminal.screen,'evidence_root',lambda:tmp_path)
    monkeypatch.setenv('PYTHONPATH','contaminated');monkeypatch.setenv('SIMANY_SCENE','wrong')
    def run(command,**kw):
        assert kw['cwd']==source['code_root']
        assert 'PYTHONPATH' not in kw['env'] and 'SIMANY_SCENE' not in kw['env']
        assert 'source environment.sh' in command[-1]
        assert 'NPY_DISABLE_CPU_FEATURES='+terminal.ISA_MODE in command[-1]
        assert 'exec original-python -c' in command[-1]
        return SimpleNamespace(stdout='{"validated": true}')
    monkeypatch.setattr(terminal.subprocess,'run',run)
    assert terminal._full_replay(config)=={'validated':True}
    compatibility.write_text('tamper')
    with pytest.raises(ValueError,match='compatibility'):terminal._full_replay(config)


def test_readonly_scene_probe_cannot_introduce_scene(failed_worker):
    request,*_=failed_worker
    with pytest.raises(ValueError,match='subset'):worker.audit(request,scene_ids=['replacement'])
    with pytest.raises(ValueError,match='subset'):worker.audit(request,scene_ids=['scene','scene'])
    assert worker.audit(request,scene_ids=['scene'])['scenes']['scene']['status']=='EXPORT_REJECTED'
