import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robo.eval import e4_camera_scorer_gate as gate
from robo.eval import e4_candidate_screen as screen
from run.icra2027 import e4_compact_harness as compact


def cpu_rows():
    return [dict(cell_id=f'{arm.lower()}__{scene}__{task}__seed0__ep{ep}',scene_id=scene,
        task_id=f'{scene}__{task}',policy_id=arm,episode=ep,
        reset_seed=compact.elog.derive_reset_seed(0,f'{scene}__{task}',ep),target='obj_1001',
        passed=False,checks={'construction_endpoints_accepted':False})
        for scene,task,_ in compact.TASKS for arm in compact.ARMS for ep in range(5)]


def suites():
    return {arm:dict(robot={'base_pos':[0,0,.75],'base_yaw':0},table={},ext_cam={},
        exclude_objects=[],tasks=[],scene_xml=arm) for arm in compact.ARMS}


@pytest.mark.parametrize('field',['robot','table','ext_cam','exclude_objects','tasks'])
def test_automatic_pair_drift_is_rejected(field):
    s=suites();gate._automatic_paired_fields(s)
    s['A4'][field]='changed'
    with pytest.raises(gate.CameraScorerGateError,match=field):gate._automatic_paired_fields(s)


@pytest.fixture
def fake_chain(tmp_path,monkeypatch):
    code={'commit':'a'*40,'dirty':False,'code_root':str(gate.CODE_ROOT)}
    config={'schema_version':1,'study_scope':gate.AUTOMATIC_SCOPE,'paper_ready':False,
        'render_backend':'osmesa','freeze_id':'automatic-test','producer_commit':'a'*40,
        'qualifier_screen_id':'cpu-test','protocol':{'path':str(tmp_path/'protocol.yaml'),'sha256':'b'*64},
        'menagerie_root':str(tmp_path),'openpi_resize_identity':{'pinned':True}}
    menagerie={'root':str(tmp_path),'files':[],'hash':'pinned'}
    rows=cpu_rows();sources={}
    for scene in compact.SCENES:
        file=tmp_path/(scene+'.json');file.write_text(json.dumps({'menagerie':{k:v for k,v in menagerie.items() if k!='files'}}))
        sources[scene]={'gate.json':{'path':str(file),'sha256':compact.sha(file)}}
    monkeypatch.setattr(gate,'_git_snapshot',lambda _:code)
    monkeypatch.setattr(gate,'_menagerie_snapshot',lambda *_:menagerie)
    monkeypatch.setattr(gate,'_openpi_snapshot',lambda:{'pinned':True})
    monkeypatch.setattr(compact,'checked_protocol',lambda *_:{'fixed':True})
    monkeypatch.setattr(compact,'qualifier_handoff',lambda *_a,**_k:(copy.deepcopy(rows),sources))
    prepared={}
    for scene in compact.SCENES:
        paths={}
        for arm,suite in suites().items():
            f=tmp_path/f'{scene}-{arm}.json';f.write_text(json.dumps(suite));paths[arm]=str(f)
        prepared[scene]={'task_bundle':{'variant_tasks':paths,'bundle_manifest_sha256':scene},'factories':{a:tmp_path for a in compact.ARMS}}
    monkeypatch.setattr(screen,'_load_prepare',lambda **kw:prepared[kw['scene_id']])
    return config,rows,prepared


def test_automatic_chain_reuses_full_qualifier_and_pinned_closures(fake_chain,tmp_path):
    config,rows,_=fake_chain
    result=gate.validate_automatic_chain(config,root=tmp_path)
    assert result['rows']==rows and result['summary']['planned']==40
    assert result['summary']['eligible']==0 and set(result['scenes'])==set(compact.SCENES)


@pytest.mark.parametrize('field',['study_scope','schema_version','paper_ready','render_backend','openpi_resize_identity'])
def test_automatic_scope_and_resize_change_rejected(fake_chain,tmp_path,field):
    config,_,_=fake_chain;config[field]='changed'
    with pytest.raises(gate.CameraScorerGateError):gate.validate_automatic_chain(config,root=tmp_path)


def test_automatic_missing_task_or_qualifier_failure_rejected(fake_chain,tmp_path,monkeypatch):
    config,_,prepared=fake_chain
    Path(prepared[compact.SCENES[0]]['task_bundle']['variant_tasks']['A4']).unlink()
    with pytest.raises(FileNotFoundError):gate.validate_automatic_chain(config,root=tmp_path)
    def reject(*a,**k):raise ValueError('tampered canonical qualifier')
    monkeypatch.setattr(compact,'qualifier_handoff',reject)
    with pytest.raises(ValueError,match='tampered'):gate.validate_automatic_chain(config,root=tmp_path)


@pytest.fixture
def published(tmp_path,monkeypatch):
    config={'freeze_id':'automatic-test','menagerie_root':str(tmp_path)}
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    code={'commit':'a'*40,'dirty':False}
    before={'rows':cpu_rows(),'scenes':{},'summary':{'planned':40}}
    monkeypatch.setattr(gate,'_git_snapshot',lambda _:code)
    monkeypatch.setattr(gate,'_evidence_root',lambda:tmp_path)
    monkeypatch.setattr(gate,'validate_automatic_chain',lambda *a,**k:copy.deepcopy(before))
    monkeypatch.setenv('MUJOCO_GL','osmesa');monkeypatch.setenv('PYOPENGL_PLATFORM','osmesa')
    from robo.envs import pi05_env
    def no_env(*a,**k):raise AssertionError('rejected CPU arm constructed a renderer')
    monkeypatch.setattr(pi05_env,'DroidSimEnv',no_env)
    result=gate.run_automatic_gate(config_path=config_path,expected_code_commit='a'*40)
    output=tmp_path/'outputs/icra2027/automatic-test/harness/automatic_camera_scorer'
    return config_path,output,result,before


def test_rejected_population_seals_40_null_cells_without_renderer(published):
    config,output,result,_=published
    assert result['planned_cells']==40 and result['build_failures']==40
    assert result['executed_cells']==0 and not result['policy_launch_allowed']
    checked=gate.validate_automatic_camera_output(config_path=config,output=output,expected_code_commit='a'*40)
    assert len(checked['cells'])==40
    assert all(r['camera_metrics'] is None and r['reset_provenance'] is None for r in checked['cells'])
    with pytest.raises(gate.CameraScorerGateError,match='reuse'):
        gate.run_automatic_gate(config_path=config,expected_code_commit='a'*40)


@pytest.mark.parametrize('target',['cells.jsonl','gate.json','manifest.json','seal.json','config'])
def test_sealed_camera_or_config_tamper_rejected(published,target):
    config,output,_,_=published
    p=config if target=='config' else output/target;p.write_text(p.read_text()+' ')
    with pytest.raises((gate.CameraScorerGateError,ValueError)):
        gate.validate_automatic_camera_output(config_path=config,output=output,expected_code_commit='a'*40)


@pytest.mark.parametrize('mutation',['missing','duplicate','telemetry','reset'])
def test_selected_cell_denominator_and_unexecuted_telemetry_rejected(published,mutation):
    _,out,_,before=published
    cells=[json.loads(l) for l in (out/'cells.jsonl').read_text().splitlines()]
    if mutation=='missing':cells.pop()
    if mutation=='duplicate':cells[-1]=copy.deepcopy(cells[0])
    if mutation=='telemetry':cells[0]['reset_provenance']={'invented':True}
    if mutation=='reset':cells[0]['reset_seed']+=1
    with pytest.raises(gate.CameraScorerGateError):gate._automatic_camera_summary(cells,before['rows'])


def test_diagnostic_error_cannot_acquire_fake_telemetry(published):
    _,out,_,before=published
    cells=[json.loads(l) for l in (out/'cells.jsonl').read_text().splitlines()]
    before['rows'][0]['passed']=True
    cells[0].update(outcome='diagnostic_error',error='RuntimeError: renderer failed')
    assert gate._automatic_camera_summary(cells,before['rows'])['diagnostic_error_cells']==1
    cells[0]['camera_metrics']=[]
    with pytest.raises(gate.CameraScorerGateError):gate._automatic_camera_summary(cells,before['rows'])


@pytest.fixture
def real_reset(tmp_path):
    if os.environ.get('MUJOCO_GL')!='osmesa':pytest.skip('run with the documented OSMesa environment')
    from robo.rigs.pi05_rig import lookat_quat
    p=tmp_path/'scene.xml';p.write_text('''<mujoco><option timestep="0.0016666666666666666"/><worldbody><light pos="0 0 3"/><geom type="plane" size="3 3 .1"/><body name="obj_1001" pos=".50 -.15 .79"><freejoint/><geom type="box" size=".02 .02 .02" mass=".03" rgba=".7 .15 .1 1"/></body><body name="obj_1002" pos=".8 -.2 .79"><freejoint/><geom type="box" size=".02 .02 .02" mass=".03" rgba=".1 .15 .7 1"/></body></worldbody></mujoco>''')
    suite={'scene_xml':str(p),'robot':{'base_pos':[0,0,.75],'base_yaw':0},'table':{'cx':.5,'cy':0.,'hx':.4,'hy':.4,'top_z':.75,'thickness_m':.02},'ext_cam':{'mode':'world','pos':[1.3,-1.2,1.6],'target':[.5,0,.8],'fovy':68},'exclude_objects':[]}
    suite['ext_cam']['quat_wxyz']=lookat_quat(suite['ext_cam']['pos'],suite['ext_cam']['target']).tolist()
    task={'task_id':'synthetic__obj_1001_to_region','target':'obj_1001','target_label':'mouse','task_family':'object_to_region','any_instance':False,'receptacle':None,'region':{'cx':.55,'cy':.2,'hx':.08,'hy':.08,'zlo':.72,'zhi':.85},'instructions':{'default':'move the mouse to the left side of the table'}}
    rows=[{'name':'obj_1001','label':'mouse','dims':np.array([.04]*3),'mass':.03,'tier':'A','drift':0.0}]
    rows.append({'name':'obj_1002','label':'cup','dims':np.array([.04]*3),'mass':.03,'tier':'A','drift':0.0})
    kwargs=dict(scene_xml=suite['scene_xml'],base_pos=suite['robot']['base_pos'],base_yaw=0,table_box=suite['table'],ext_cam=suite['ext_cam'],exclude_objects=(),menagerie_root=gate.MENAGERIE_COPY_SOURCE_ROOT)
    e=gate._build_headless_droid_env(**kwargs)
    seed=compact.elog.derive_reset_seed(0,task['task_id'],0)
    e.reset(settle_s=1.5,jitter_body=task['target'],jitter_xy=.01,jitter_uniform_draw=np.random.RandomState(seed).random_sample(2),reset_seed=seed)
    v=e.last_reset_provenance
    row={'cell_id':'a0_synthetic','policy_id':'A0','task_id':task['task_id'],'scene_id':'synthetic','episode':0,'reset_seed':seed,'target':task['target'],'reset_jitter':v['jitter'],'settle_protocol':v['settle_protocol'],'settle_contract_checks':gate._settle_contract_checks(v),'stability':screen._runtime_stability(env=e,task=task,provenance=v),'workspace':screen._runtime_workspace(env=e,task=task,suite=suite,task_rows=rows)}
    row['qualifier']=gate.disambiguation_metric(e,task,suite,rows,{k:row[k] for k in
        ('cell_id','policy_id','task_id','scene_id','episode','reset_seed','target')})
    row['passed']=all(row['settle_contract_checks'].values()) and row['stability']['passed'] and row['workspace']['passed']
    assert row['passed']
    return kwargs,task,suite,rows,row


def test_real_osmesa_reset_render_and_scorer_use_canonical_producers(real_reset,tmp_path):
    from robo.envs.pi05_env import DroidSimEnv
    kwargs,task,suite,rows,row=real_reset
    env=DroidSimEnv(**kwargs,render_wh=(640,360))
    try:
        result=gate._automatic_render_cell(env,task,suite,rows,row,tmp_path/'rendered')
        assert result['executed'] and result['checks']['exact_cpu_reset_replay']
        assert len(result['camera_metrics'])==2 and result['scorer_metrics']['passed']
        assert len(result['scorer_metrics']['negative_replays'])==4
        assert result['scorer_metrics']['oracle_validated'] is False
        assert result['reset_provenance']['settle_protocol']['step_count']==900
        assert list((tmp_path/'rendered').rglob('*.png'))
        # Visibility may fail for this synthetic fixed rig; no threshold or camera tuning.
        assert result['passed']==all(result['checks'].values())
    finally:env.renderer.close()


def test_real_reset_stability_drift_fails_before_diagnostics(real_reset,tmp_path):
    from robo.envs.pi05_env import DroidSimEnv
    kwargs,task,suite,rows,row=real_reset
    row['stability']['passed']=False
    env=DroidSimEnv(**kwargs,render_wh=(640,360))
    try:
        with pytest.raises(gate.CameraScorerGateError,match='reset differs'):
            gate._automatic_render_cell(env,task,suite,rows,row,tmp_path/'bad')
        assert not (tmp_path/'bad').exists()
    finally:env.renderer.close()


def test_eligible_renderer_exception_retains_cell_and_remaining_denominator(published,monkeypatch,tmp_path):
    config,_,_,before=published
    before['rows'][0]['passed']=True
    monkeypatch.setattr(gate,'validate_automatic_chain',lambda *a,**k:copy.deepcopy(before))
    # Missing resolved scene becomes one typed diagnostic error, never a successful measurement.
    cfg=json.loads(config.read_text());cfg['freeze_id']='exception-test'
    path=tmp_path/'exception.json';path.write_text(json.dumps(cfg))
    result=gate.run_automatic_gate(config_path=path,expected_code_commit='a'*40)
    assert result['planned_cells']==40 and result['build_failures']==39
    assert result['diagnostic_error_cells']==1 and result['executed_cells']==0
    output=tmp_path/'outputs/icra2027/exception-test/harness/automatic_camera_scorer'
    checked=gate.validate_automatic_camera_output(config_path=path,output=output,expected_code_commit='a'*40)
    assert checked['cells'][0]['outcome']=='diagnostic_error'
    assert checked['cells'][0]['reset_provenance'] is None
