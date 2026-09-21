"""Explicit full scope retains every preregistered query without compact shortcuts."""
import copy
import json
from pathlib import Path
import pytest
from robo.eval import e4_camera_scorer_gate as camera
from robo.eval import e4_candidate_screen as screen
from run.icra2027 import e4_full_qualification as full
from robo.eval.episode_log import derive_reset_seed
from tests.test_e4_automatic_camera import suites,real_reset


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return full.shared.identity(path)


@pytest.fixture
def full_chain(tmp_path,monkeypatch):
    stage=tmp_path/'outputs/icra2027/full-camera';stage.mkdir(parents=True)
    scenes=[f'{n:010x}' for n in range(50)]
    queries=[dict(scene_id=s,task_id=f'{s}__obj_{1000+k}_to_region',target=f'obj_{1000+k}',receptacle=None,
        task_family='object_to_region') for i,s in enumerate(scenes) for k in range(7 if i<23 else 6 if i<41 else 0)]
    protocol={'source_populations':[{'scene_id':s} for s in scenes],'qualification_tasks':queries}
    protocol_id=put(tmp_path/'protocol.json',protocol)
    model={'root':str(tmp_path),'hash':'fixed','files':[]};model_summary={k:v for k,v in model.items() if k!='files'}
    qualification={'freeze_id':'full-camera','source':{'protocol':protocol_id},'menagerie_root':str(tmp_path),
        'menagerie':model_summary,'openpi_resize_identity':{'resize':'fixed'}}
    qi=put(tmp_path/'qualification.json',qualification)
    cfg={'schema_version':1,'study_scope':camera.FULL_AUTOMATIC_SCOPE,'paper_ready':False,
        'freeze_id':'full-camera','qualifier_screen_id':'full-camera','producer_commit':'a'*40,
        'qualification_config':qi,'protocol':{k:protocol_id[k] for k in ('path','sha256')},
        'menagerie_root':str(tmp_path),'openpi_resize_identity':{'resize':'fixed'},'render_backend':'osmesa'}
    code={'commit':'a'*40,'dirty':False}
    monkeypatch.setattr(camera,'_git_snapshot',lambda _:code)
    monkeypatch.setattr(screen,'_code_snapshot',lambda _:code)
    monkeypatch.setattr(camera,'_evidence_root',lambda:tmp_path)
    monkeypatch.setattr(camera,'_menagerie_snapshot',lambda *a:model)
    monkeypatch.setattr(camera,'_openpi_snapshot',lambda:qualification['openpi_resize_identity'])
    monkeypatch.setattr(full,'validate_stage',lambda *a:(copy.deepcopy(qualification),copy.deepcopy(protocol)))
    qualifiers={};prepared={};results={}
    for i,s in enumerate(scenes):
        tasks=[q for q in queries if q['scene_id']==s];ids=sorted(q['task_id'] for q in tasks)
        result=dict(schema_version=1,scope=full.SCOPE,scene_id=s,source_commit='a'*40,freeze_id='full-camera',
            config=qi,selected_task_ids=ids,planned_qualification_cells=10*len(ids),policy_executed=0,
            policy_success=None,camera=None,rollout_ledger=None,paper_ready=False)
        if not tasks:result.update(status='NOT_RUN',reason='no_queries_in_original_protocol',qualifier=None)
        elif i==0:result.update(status='PLANNING_UNQUALIFIED',reason=full.PLANNING_UNQUALIFIED,qualifier=None,reset_definitions=None)
        else:
            rows=[dict(cell_id=f'{arm.lower()}__{q["task_id"]}__seed0__ep{ep}',scene_id=s,task_id=q['task_id'],
                policy_id=arm,episode=ep,reset_seed=derive_reset_seed(0,q['task_id'],ep),target=q['target'],
                passed=False,checks={'construction_endpoints_accepted':False})
                for q in tasks for arm in ('A0','A4') for ep in range(5)]
            gate=dict(cell_count=len(rows),exact_900_step_cells=0,strict_pass_task_ids=[],
                task_summaries=[{'task_id':t} for t in ids],menagerie=model_summary)
            qualifiers[s]={'metric_rows':rows,'gate':gate}
            directory=stage/'scene_qualifiers'/s
            for name in ('gate.json','manifest.json','seal.json','metrics.jsonl'):put(directory/name,{})
            variants={}
            for arm,value in suites().items():
                value['tasks']=[{'task_id':t} for t in ids]
                variants[arm]=put(tmp_path/s/f'{arm}_tasks.json',value)['path']
            prepared[s]={'gate':{'automatic_population':{}},'factories':{'A0':tmp_path,'A4':tmp_path},
                'task_bundle':{'variant_tasks':variants,'logical_task_ids':ids,'bundle_manifest_sha256':s}}
            result.update(status='QUALIFICATION_COMPLETE',qualifier=full.shared.identity(directory/'manifest.json'),
                prerequisite_cells=len(rows),simulator_cells=0,strict_pass_task_ids=[])
        directory=stage/'qualification_handoff'/s
        put(directory/'manifest.json',{'code':code,'freeze_id':'full-camera','scene_id':s})
        put(directory/'seal.json',{})
        put(directory/'result.json',result);results[s]=directory/'result.json'
    monkeypatch.setattr(screen,'_validate_bundle',lambda path,**k:{'directory':Path(path),'manifest':json.loads((Path(path)/'manifest.json').read_text())})
    monkeypatch.setattr(screen,'_read_json_member',lambda directory,name:json.loads((directory/name).read_text()))
    monkeypatch.setattr(screen,'_validate_qualifier_output',lambda **k:qualifiers[k['scene_id']])
    monkeypatch.setattr(screen,'_load_prepare',lambda **k:prepared[k['scene_id']])
    def unqualified(**k):raise screen.CandidateScreenError(full.PLANNING_UNQUALIFIED)
    monkeypatch.setattr(screen,'_automatic_prepare_inputs',unqualified)
    return cfg,stage,protocol,qualifiers,prepared,results


def test_full_scope_authenticates_all50_and2690_with_null_planning_cells(full_chain,tmp_path):
    config,_,_,_,_,_=full_chain;r=camera.validate_automatic_chain(config,root=tmp_path)
    assert len(r['summary']['qualifier_sources'])==50 and r['summary']['planned']==2690
    assert len(r['rows'])==2620 and len(r['planned_unavailable'])==70 and len(r['scenes'])==40
    assert all(x['camera_metrics'] is None and x['reset_provenance'] is None for x in r['planned_unavailable'])
    assert r['summary']['eligible']==0


@pytest.mark.parametrize('mutation',['missing_handoff','changed_task','changed_cell','duplicate_cell','planning_reason','planning_replays','fake_planning_telemetry','camera_drift','paired_task_drift','foreign_qualifier'])
def test_full_chain_missing_tampered_or_substituted_cells_rejected(full_chain,tmp_path,monkeypatch,mutation):
    config,stage,protocol,qualifiers,prepared,results=full_chain
    scene=list(results)[1];first=list(results)[0]
    if mutation=='missing_handoff':results[scene].unlink()
    elif mutation=='changed_task':protocol['qualification_tasks'][7]['task_id']='foreign'
    elif mutation=='changed_cell':qualifiers[scene]['metric_rows'][0]['cell_id']='foreign'
    elif mutation=='duplicate_cell':qualifiers[scene]['metric_rows'][-1]=qualifiers[scene]['metric_rows'][0]
    elif mutation=='planning_replays':monkeypatch.setattr(screen,'_automatic_prepare_inputs',lambda **k:{})
    elif mutation in ('planning_reason','fake_planning_telemetry','foreign_qualifier'):
        p=results[first if mutation!='foreign_qualifier' else scene];v=json.loads(p.read_text())
        if mutation=='planning_reason':v['reason']='other failure'
        elif mutation=='fake_planning_telemetry':v['reset_definitions']={'invented':True}
        else:v['qualifier']['sha256']='foreign'
        p.write_text(json.dumps(v))
    else:
        p=Path(prepared[scene]['task_bundle']['variant_tasks']['A4']);v=json.loads(p.read_text())
        v['ext_cam' if mutation=='camera_drift' else 'tasks']='changed';p.write_text(json.dumps(v))
    with pytest.raises((ValueError,camera.CameraScorerGateError,FileNotFoundError,KeyError)):
        camera.validate_automatic_chain(config,root=tmp_path)


@pytest.mark.parametrize('key,value',[('study_scope','automatic_compact_camera_diagnostic'),('freeze_id','different'),('render_backend','egl'),('paper_ready',True)])
def test_full_scope_cannot_use_compact_or_other_stage(full_chain,tmp_path,key,value):
    cfg,*_=full_chain;cfg[key]=value
    with pytest.raises((ValueError,camera.CameraScorerGateError,KeyError)):
        camera.validate_full_automatic_chain(cfg,root=tmp_path)


@pytest.fixture
def published_full(full_chain,tmp_path,monkeypatch):
    cfg,stage,*_=full_chain
    before=camera.validate_automatic_chain(cfg,root=tmp_path)
    config=Path(put(tmp_path/'camera.json',cfg)['path'])
    monkeypatch.setattr(camera,'validate_automatic_chain',lambda *a,**k:copy.deepcopy(before))
    monkeypatch.setattr(full,'cpu_guard',lambda:None)
    monkeypatch.setenv('MUJOCO_GL','osmesa');monkeypatch.setenv('PYOPENGL_PLATFORM','osmesa')
    from robo.envs import pi05_env
    def forbidden(*a,**k):raise AssertionError('unqualified cells must not create a renderer')
    monkeypatch.setattr(pi05_env,'DroidSimEnv',forbidden)
    result=camera.run_automatic_gate(config_path=config,expected_code_commit='a'*40)
    return config,stage/'harness/automatic_camera_scorer',result,before


def test_full_output_keeps_cpu_and_planning_unavailable_denominators_separate(published_full):
    config,out,result,_=published_full
    assert result['planned_cells']==2690 and result['prerequisite_cells']==2620
    assert result['planning_unqualified_cells']==70 and result['executed_cells']==0
    assert not result['policy_launch_allowed']
    checked=camera.validate_automatic_camera_output(config_path=config,output=out,expected_code_commit='a'*40)
    assert len(checked['cells'])==2620
    assert len((out/'planned_unavailable_cells.jsonl').read_text().splitlines())==70


@pytest.mark.parametrize('name',['planned_unavailable_cells.jsonl','cells.jsonl','gate.json','manifest.json','seal.json'])
def test_full_output_tamper_rejected(published_full,name):
    config,out,_,_=published_full;p=out/name;p.write_text(p.read_text()+' ')
    with pytest.raises((ValueError,camera.CameraScorerGateError)):
        camera.validate_automatic_camera_output(config_path=config,output=out,expected_code_commit='a'*40)


def test_full_writer_is_derived_and_no_overwrite(full_chain,tmp_path):
    cfg,*_=full_chain;out=tmp_path/'new_camera.json'
    result=camera.write_full_automatic_camera_config(qualification_config=cfg['qualification_config']['path'],expected_code_commit='a'*40,out=out)
    assert result==cfg
    with pytest.raises(FileExistsError):
        camera.write_full_automatic_camera_config(qualification_config=cfg['qualification_config']['path'],expected_code_commit='a'*40,out=out)


def test_full_handoff_manifest_cannot_relabel_producer(full_chain,tmp_path):
    cfg,_,_,_,_,results=full_chain
    path=next(iter(results.values())).parent/'manifest.json'
    value=json.loads(path.read_text());value['code']['commit']='foreign';path.write_text(json.dumps(value))
    with pytest.raises(camera.CameraScorerGateError,match='manifest producer'):
        camera.validate_automatic_chain(cfg,root=tmp_path)


def test_full_summary_reuses_real_osmesa_reset_render_and_scorer(real_reset,tmp_path):
    from robo.envs.pi05_env import DroidSimEnv
    kwargs,task,suite,task_rows,row=real_reset
    env=DroidSimEnv(**kwargs,render_wh=(640,360))
    try:cell=camera._automatic_render_cell(env,task,suite,task_rows,row,tmp_path/'full-real-render')
    finally:env.renderer.close()
    rows=[row];cells=[cell]
    for i in range(2689):
        other=dict(cell_id=f'planned-{i}',scene_id='synthetic',task_id=f'planned-task-{i}',
            policy_id='A0',episode=0,reset_seed=i,target='obj_1001',passed=False,checks={'construction':False})
        rejected={k:other[k] for k in ('cell_id','scene_id','task_id','policy_id','episode','reset_seed','target')}
        rejected.update(reset_state_id=f'{other["task_id"]}__seed0__ep0',executed=False,passed=False,
            outcome='build_failure',cpu_failed_checks=['construction'],camera_metrics=None,
            workspace_metrics=None,disambiguation_metrics=None,scorer_metrics=None,reset_provenance=None)
        rows.append(other);cells.append(rejected)
    summary=camera._automatic_camera_summary(cells,rows,planned_unavailable=[])
    assert summary['planned_cells']==2690 and summary['executed_cells']==1
    assert summary['diagnostic_passed_cells']==int(cell['passed'])
    assert cell['checks']['exact_cpu_reset_replay'] and cell['scorer_metrics']['passed']
    assert not summary['policy_launch_allowed']
    cells[0]['disambiguation_metrics']['observed_margin_m']=.123
    with pytest.raises(camera.CameraScorerGateError,match='telemetry'):
        camera._automatic_camera_summary(cells,rows,planned_unavailable=[])
