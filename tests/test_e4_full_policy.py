"""Full selection and canonical staging retain frozen pre-policy identities."""
import copy
import json
from pathlib import Path
import pytest
import yaml
from robo.eval import e4_reset_eligibility as adapter
from robo.eval import e4_camera_scorer_gate as camera
from robo.eval import e4_candidate_screen as screen
from robo.eval import episode_log
from run.icra2027 import e4_compact_policy as preparer
from tests.test_e4_automatic_eligibility import adapter_spec


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)


@pytest.fixture
def full_fixture(tmp_path,monkeypatch):
    scene_ids=[f'{i:010x}' for i in range(50)]
    tasks=[dict(scene_id=s,task_id=f'{s}__obj_{1000+k}_to_'+('region' if k%2==0 else 'obj_1999'),
        target=f'obj_{1000+k}',receptacle=None if k%2==0 else 'obj_1999',
        task_family='object_to_region' if k%2==0 else 'object_to_receptacle')
        for i,s in enumerate(scene_ids) for k in range(7 if i<23 else 6 if i<41 else 0)]
    protocol={'source_populations':[{'scene_id':s} for s in scene_ids],'qualification_tasks':tasks}
    protocol_path=put(tmp_path/'protocol.json',protocol)
    cam={'study_scope':camera.FULL_AUTOMATIC_SCOPE,'freeze_id':'full-source',
        'protocol':{'path':protocol_path,'sha256':screen._sha256(Path(protocol_path))},'menagerie_root':str(tmp_path/'menagerie')}
    config_path=put(tmp_path/'camera.json',cam)
    output=tmp_path/'outputs/icra2027/full-source/harness/automatic_camera_scorer';output.mkdir(parents=True)
    summary={'planned':2690,'source':'authenticated full source'}
    gate={'code':{'commit':'a'*40,'dirty':False},'upstream':summary};put(output/'gate.json',gate)
    scenes={};rows=[];cells=[]
    for i,s in enumerate(scene_ids[:41]):
        selected=[q for q in tasks if q['scene_id']==s]
        suite={'scene':s,'robot':{},'ext_cam':{},'table':{},'tasks':[{**q,'instructions':{'default':'Move object.'}} for q in selected]}
        d=tmp_path/s;manifest=Path(put(d/'manifest.json',{}));planning=put(d/'planning.json',suite)
        variants={a:put(d/f'{a}.json',suite) for a in ('A0','A4')}
        bundle={'bundle_manifest_sha256':screen._sha256(manifest),'planning_tasks':planning,
            'variant_tasks':variants,'factories':{a:str(d/a) for a in ('A0','A4')},
            'scene_xml':{a:str(d/f'{a}.xml') for a in ('A0','A4')}}
        scenes[s]={'suites':{'A0':suite,'A4':copy.deepcopy(suite)},'task_bundle':bundle}
        for q in selected:
            for arm in ('A0','A4'):
                for ep in range(5):
                    tid=q['task_id'];passed=i<5
                    row=dict(cell_id=f'{arm.lower()}__{tid}__seed0__ep{ep}',task_id=tid,scene_id=s,
                        target=q['target'],policy_id=arm,episode=ep,reset_seed=episode_log.derive_reset_seed(0,tid,ep),
                        passed=passed,checks={'accepted':passed})
                    rows.append(row)
                    cell={k:row[k] for k in ('cell_id','task_id','scene_id','target','policy_id','episode','reset_seed')}
                    cell.update(reset_state_id=f'{tid}__seed0__ep{ep}',passed=passed,executed=passed,
                        outcome='diagnostic_pass' if passed else 'build_failure',checks={'camera':True} if passed else None,
                        camera_metrics=None,workspace_metrics=None,disambiguation_metrics=None,scorer_metrics=None,
                        reset_provenance={'synthetic':'already validated by producer'} if passed else None)
                    cells.append(cell)
    chain={'summary':summary,'rows':rows,'planned_unavailable':[],'scenes':scenes,
        'menagerie':{'root':cam['menagerie_root'],'commit':'d'*40}}
    checked={'gate':gate,'cells':cells,'manifest_sha256':'b'*64}
    monkeypatch.setattr(camera,'validate_automatic_chain',lambda *a,**k:chain)
    monkeypatch.setattr(camera,'validate_automatic_camera_output',lambda **k:checked)
    monkeypatch.setattr(camera,'_evidence_root',lambda:tmp_path)
    monkeypatch.setattr(camera,'_git_snapshot',lambda c:{'commit':c,'dirty':False})
    return cam,Path(config_path),output,protocol,chain,checked


def test_frozen_full_selection_uses_qualification_only(full_fixture):
    cam,_,_,_,chain,checked=full_fixture
    matrix,states,pilot,selected,cells=adapter.full_qualified_policy_selection(cam,chain,checked)
    assert matrix['scene_ids']==[f'{i:010x}' for i in range(4)]
    assert matrix['qualified_scenes']==5 and len(states)==80 and len(pilot)==20
    assert len(selected['rows'])==len(cells)==160
    assert {s.scene_id for s in states if s.reset_state_id in pilot}=={'0000000000','0000000001'}
    for row in chain['rows']:row['success']=False
    for row in checked['cells']:row['success']=True
    assert adapter.full_qualified_policy_selection(cam,chain,checked)[0]==matrix


@pytest.mark.parametrize('damage',['missing','duplicate','unavailable','scope'])
def test_full_selection_rejects_population_drift(full_fixture,damage):
    cam,_,_,_,chain,checked=full_fixture
    if damage=='missing':chain['rows'].pop()
    elif damage=='duplicate':checked['cells'][-1]=checked['cells'][0]
    elif damage=='unavailable':chain['planned_unavailable']=[{}]
    else:cam['study_scope']=camera.AUTOMATIC_SCOPE
    with pytest.raises(ValueError):adapter.full_qualified_policy_selection(cam,chain,checked)


def test_insufficient_qualified_rooms_is_not_run_without_substitution(full_fixture):
    cam,_,_,_,chain,checked=full_fixture
    for row in checked['cells']:
        if row['scene_id'] in {'0000000003','0000000004'}:row['passed']=False
    result,states,pilot,_,_=adapter.full_qualified_policy_selection(cam,chain,checked)
    assert result['status']=='NOT_RUN' and result['qualified_scenes']==3 and states==pilot==[]


def config_for(full_fixture):
    cam,path,out,_,chain,checked=full_fixture
    matrix,states,pilot,selected,_=adapter.full_qualified_policy_selection(cam,chain,checked)
    scenes=[]
    for sid,data in selected['scenes'].items():
        b=data['task_bundle']
        scenes.append({'id':sid,'menagerie_root':cam['menagerie_root'],'tasks_json':b['planning_tasks'],
            'task_freeze_manifest':str(Path(b['planning_tasks']).parent/'manifest.json'),
            'construction_variants':{v:{'tasks_json':b['variant_tasks'][a],'factory_dir':b['factories'][a],
                'scene_xml':b['scene_xml'][a]} for a,v in [('A0','fixed_single_path'),('A4','agentic')]}})
    config={'paper_mode':False,'policy':'pi05_droid_jointpos','study_scope':'automatic_full_policy_engineering',
        'jitter':screen.JITTER_XY_M,'jitter_first_episode':True,'scenes':scenes,
        'full_qualification_selection':matrix,'pilot_reset_ids':pilot,
        'cpu_reset_eligibility':{'kind':'automatic_full','camera_config':{'path':str(path),'sha256':screen._sha256(path)},
            'bundle':str(out),'gate_sha256':screen._sha256(out/'gate.json'),'source_commit':'a'*40},
        'treatments':[{'id':f'{a}_raster','scene':v,'collision':'full_room','observation':'raster'}
            for a,v in [('a0','fixed_single_path'),('a4','agentic')]],
        'contract':{'policy':{'kind':'real','checkpoint_hash':'c'*64},'horizon_s':16}}
    return config,states


def test_full_adapter_returns160_canonical_eligible_rows(full_fixture,tmp_path):
    cfg,states=config_for(full_fixture)
    indexed=adapter.load_reset_eligibility(cfg,adapter_spec(cfg),states,root=tmp_path)
    assert len(indexed)==160 and all(r['passed'] for r in indexed.values())
    assert all(r['kind']=='automatic_full' for r in indexed.values())


@pytest.mark.parametrize('damage',['matrix','pilot','bank','scene','camera','kind'])
def test_full_adapter_does_not_accept_selected_identity_drift(full_fixture,tmp_path,damage):
    cfg,states=config_for(full_fixture)
    if damage=='matrix':cfg['full_qualification_selection']['tasks'].pop()
    elif damage=='pilot':cfg['pilot_reset_ids'][0]=states[-1].reset_state_id
    elif damage=='bank':states.pop()
    elif damage=='scene':cfg['scenes'].pop()
    elif damage=='camera':cfg['scenes'][0]['construction_variants']['agentic']['scene_xml']='other'
    else:cfg['cpu_reset_eligibility']['kind']='automatic_compact'
    with pytest.raises((ValueError,KeyError)):adapter.load_reset_eligibility(cfg,adapter_spec(cfg),states,root=tmp_path)


def test_existing_preparer_writes_full80_definitions_without_rewriting_task_bundles(full_fixture,tmp_path,monkeypatch):
    cam,path,output,_,chain,checked=full_fixture
    monkeypatch.setattr(preparer,'load_harness_spec',adapter_spec)
    monkeypatch.setattr(preparer,'_policy_contract',lambda *a,**k:{'policy':{'kind':'real','id':'pi05_droid_jointpos',
        'checkpoint_path':'checkpoint','checkpoint_hash':'c'*64},'horizon_s':16})
    out=tmp_path/'outputs/icra2027/full-scripted/harness/full_scripted_smoke'
    cfg,states,receipt=preparer.build_config(camera_config_path=path,camera_config_sha256=screen._sha256(path),
        camera_output=output,expected_code_commit='a'*40,out=out,openpi_root='openpi',openpi_commit='b'*40,
        mode='scripted',freeze_id='full-scripted')
    assert len(states)==80 and receipt['planned_episodes']==160 and len(cfg['pilot_reset_ids'])==20
    assert cfg['study_scope']=='e4_full_scripted_smoke' and cfg['cpu_reset_eligibility']['kind']=='automatic_full'
    assert len(cfg['scenes'])==4 and all(len(chain['scenes'][s['id']]['suites']['A0']['tasks'])==7 for s in cfg['scenes'])
    assert not out.exists()


def test_pilot_validation_projection_retains_full_persisted_contract(full_fixture):
    from robo.eval import harness_runner as runner, harness_validation as validation
    cfg,states=config_for(full_fixture)
    cfg['contract']['reset_ids']=[s.reset_state_id for s in states]
    spec=adapter_spec(cfg); original=copy.deepcopy(cfg)
    projected=runner.full_pilot_validation_spec(spec,states)
    assert cfg==original and len(spec.raw['contract']['reset_ids'])==80
    assert projected.raw['contract']['reset_ids']==cfg['pilot_reset_ids']
    # Exercise the actual canonical manifest validator's reset coverage check.
    result=validation.validate_treatment_manifests([],projected,set(cfg['pilot_reset_ids']))
    assert not any('differ from contract.reset_ids' in v for v in result['violations'])
    assert any('missing' in v for v in result['violations'])  # No fabricated rows.


@pytest.mark.parametrize('damage',['foreign','duplicate','missing','full_bank','declared','scope'])
def test_pilot_projection_rejects_invalid_original_bank(full_fixture,damage):
    from robo.eval import harness_runner as runner
    cfg,states=config_for(full_fixture);cfg['contract']['reset_ids']=[s.reset_state_id for s in states]
    if damage=='foreign':cfg['pilot_reset_ids'][0]='foreign'
    elif damage=='duplicate':cfg['pilot_reset_ids'][0]=cfg['pilot_reset_ids'][1]
    elif damage=='missing':cfg['pilot_reset_ids'].pop()
    elif damage=='full_bank':states.pop()
    elif damage=='declared':cfg['contract']['reset_ids'].pop()
    else:cfg['cpu_reset_eligibility']['kind']='automatic_compact'
    with pytest.raises(ValueError,match='full frozen bank'):
        runner.full_pilot_validation_spec(adapter_spec(cfg),states)


@pytest.fixture
def real_pilot(full_fixture,tmp_path,monkeypatch):
    from robo.eval import harness_runner as runner, harness_validation as validation
    cfg,states=config_for(full_fixture);cfg['contract']['reset_ids']=[s.reset_state_id for s in states]
    spec=adapter_spec(cfg);out=tmp_path/'real-stage';out.mkdir()
    put(out/'harness_config.yaml',cfg)
    rows=[];traces={}
    for state in states:
        for arm in ('a0','a4'):
            identity=f'{arm}_{state.reset_state_id}';task=state.task_id
            manifest=put(out/f'{identity}.json',{'git':{'commit':'a'*40,'dirty':False},
                'outcome':'task_failure','contract':{'policy':{'id':'pi05_droid_jointpos','kind':'real'}}})
            trace=put(out/f'{identity}.trace',[]);traces[trace]=[{'t':0,'stages':{task:{}}}]
            rows.append(dict(treatment_id=f'{arm}_raster',reset_state_id=state.reset_state_id,
                task_id=task,outcome='task_failure',ticks=1,manifest_path=manifest,trace_path=trace))
    monkeypatch.setattr(preparer,'_prepared',lambda *a:(cfg,states,spec,{}))
    monkeypatch.setattr(validation,'read_jsonl',lambda *a:rows)
    def saved(selected,projected,ids,**kw):
        assert set(projected.raw['contract']['reset_ids'])==ids
        return validation.validate_records(selected,projected,ids)
    monkeypatch.setattr(validation,'validate_saved_treatment_records',saved)
    monkeypatch.setattr(runner,'_preflight_resolved_scenes',lambda *a:({}, {}, {}))
    monkeypatch.setattr(runner,'_validate_record_inputs_against_current',lambda *a:[])
    monkeypatch.setattr(episode_log,'read_timeseries',lambda path:traces[str(path)])
    return out,cfg,rows,traces


def test_real_pilot_gate_preserves_failures_and_resumes_same_full_ledger(real_pilot):
    out,cfg,rows,_=real_pilot
    gate=preparer._real_pilot_receipt(out,'a'*40)
    assert gate['planned_episodes']==40 and gate['full_planned_episodes']==160
    assert gate['success_required'] is False and all(r['outcome']=='task_failure' for r in rows)
    put(out/'real_pilot_gate.json',gate)
    assert preparer.validate_real_pilot_before_full(out,'a'*40)==gate
    # Remaining 120 rows never alter the original forty-row pilot receipt.
    nonpilot=next(r for r in rows if r['reset_state_id'] not in cfg['pilot_reset_ids'])
    nonpilot['outcome']='success'
    assert preparer.validate_real_pilot_before_full(out,'a'*40)==gate


@pytest.mark.parametrize('damage',['missing','duplicate','foreign','crash','zero_ticks','trace','scorer','source','dirty','policy','outcome','gate','symlink'])
def test_real_pilot_requires_complete_unchanged_runtime_evidence(real_pilot,damage):
    out,cfg,rows,traces=real_pilot
    gate=preparer._real_pilot_receipt(out,'a'*40);put(out/'real_pilot_gate.json',gate)
    row=next(r for r in rows if r['reset_state_id'] in cfg['pilot_reset_ids'])
    if damage=='missing':rows.remove(row)
    elif damage=='duplicate':rows.append(copy.deepcopy(row))
    elif damage=='foreign':rows.append({**row,'reset_state_id':'unplanned'})
    elif damage=='crash':row['outcome']='crash'
    elif damage=='zero_ticks':row['ticks']=0
    elif damage=='trace':traces[row['trace_path']].append({'t':1,'stages':{row['task_id']:{}}})
    elif damage=='scorer':traces[row['trace_path']][0]['stages']={}
    elif damage in {'source','dirty','policy','outcome'}:
        p=Path(row['manifest_path']);m=json.loads(p.read_text())
        if damage=='source':m['git']['commit']='b'*40
        elif damage=='dirty':m['git']['dirty']=True
        elif damage=='policy':m['contract']['policy']['id']='scripted_sinusoid'
        else:m['outcome']='success'
        p.write_text(json.dumps(m))
    elif damage=='gate':put(out/'real_pilot_gate.json',{})
    else:
        p=out/'real_pilot_gate.json';p.rename(out/'other');p.symlink_to(out/'other')
    with pytest.raises(ValueError):preparer.validate_real_pilot_before_full(out,'a'*40)


def test_canonical_runner_pilot_validates_full_bank_and_skips_unreleased_cells(full_fixture,tmp_path,monkeypatch):
    from robo.eval import harness_runner as runner, harness_validation as validation
    cfg,states=config_for(full_fixture);cfg['contract']['reset_ids']=[s.reset_state_id for s in states]
    cfg['contract']['runtime_dependencies']={'mujoco_menagerie':{'root':'test-menagerie'}}
    spec=adapter_spec(cfg);out=tmp_path/'run-pilot';calls=[]
    rows=[dict(treatment_id=a,reset_state_id=s.reset_state_id,outcome='task_failure')
          for s in states if s.reset_state_id in cfg['pilot_reset_ids'] for a in spec.treatments]
    monkeypatch.setattr(runner,'load_harness_spec',lambda *a:spec)
    scenes={s['id']:s for s in cfg['scenes']}
    monkeypatch.setattr(runner,'_preflight_resolved_scenes',lambda *a:({s:{} for s in scenes},scenes,{}))
    monkeypatch.setattr(runner.legacy,'_get_or_plan_reset_states',lambda *a:states)
    monkeypatch.setattr(runner,'_validate_planned_resets',lambda bank,*a:calls.append(('full_bank',len(bank))))
    monkeypatch.setattr(adapter,'load_reset_eligibility',lambda config,spec,bank,**kw:
        {(a,s.reset_state_id):{'passed':True} for s in bank for a in spec.treatments})
    monkeypatch.setattr(runner,'_completed',lambda *a:{(r['treatment_id'],r['reset_state_id']):r for r in rows})
    monkeypatch.setattr(runner,'read_jsonl',lambda *a:rows)
    monkeypatch.setattr(runner.manifest_hash,'git_snapshot',lambda *a:{'commit':'a'*40,'dirty':False})
    monkeypatch.setattr(runner.legacy,'_frozen_config_hashes',lambda:('controller','camera','absolute_joint_position',8))
    monkeypatch.setattr(runner.legacy,'_policy_checkpoint_hash',lambda *a:'c'*64)
    monkeypatch.setattr(runner,'expected_server_identity_from_config',lambda *a:{})
    monkeypatch.setattr(runner.legacy,'get_policy',lambda *a,**kw:pytest.fail('pilot must not release pending full cells'))
    monkeypatch.setattr(runner.legacy,'build_env',lambda *a,**kw:pytest.fail('must not rerun completed pilot'))
    monkeypatch.setattr(runner,'_validate_record_inputs_against_current',lambda *a:[])
    def validate(records,projected,ids,**kw):
        assert projected.raw['contract']['reset_ids']==cfg['pilot_reset_ids']
        calls.append(('pilot_validate',len(records),len(ids)))
        return validation.validate_records(records,projected,ids)
    monkeypatch.setattr(runner,'validate_saved_treatment_records',validate)
    monkeypatch.setattr(runner,'generate_main_table',lambda *a,**kw:pytest.fail('pilot must not generate final table'))
    result=runner.run_matrix(cfg,out,execution_stage='pilot')
    assert calls==[('full_bank',80),('pilot_validate',40,20)]
    assert result['full_planned_episodes']==160 and result['pilot_planned_episodes']==40
    assert result['full_complete'] is False and result['main_table'] is None
    assert not (out/'validation.json').exists() and not (out/'paper_tables').exists()
    assert len(json.loads((out/'resolved_harness_config.json').read_text())['contract']['reset_ids'])==80
    # Same runner cannot release full execution without the original pilot receipt.
    with pytest.raises(ValueError,match='original fixed real pilot integrity'):
        runner.run_matrix(cfg,out)
