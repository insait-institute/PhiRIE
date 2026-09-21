import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from run.icra2027 import e4_compact_qualification as m


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def identity(path):
    path = Path(path)
    return dict(path=str(path), size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def chain(tmp_path, monkeypatch):
    stage = tmp_path/m.MATERIALIZATION_FREEZE
    protocol = write(tmp_path/'protocol.json', {})
    source = dict(e3_root=str(tmp_path/'agentic'),protocol=identity(protocol),scenes={})
    config = write(tmp_path/'original_config.json',dict(source=source))
    contract = write(stage/'contract/freeze_manifest.json',dict(code=dict(commit=m.MATERIALIZATION_COMMIT,dirty=False),freeze_id=stage.name))
    scenes={}
    for scene,count in zip(m.compact.SCENES,(7,10)):
        descriptor=dict(schema_version=1,scene_id=scene)
        source['scenes'][scene]={'descriptor':descriptor}
        d=write(stage/'materialization'/scene/'automatic_scene_descriptor.json',descriptor)
        variants={p:identity(write(stage/'materialization'/scene/p/'materialization_manifest.json',{})) for p in m.compact.ARMS}
        scenes[scene]=dict(descriptor=identity(d),variants=variants)
    write(config,dict(source=source))
    for scene,count in zip(m.compact.SCENES,(7,10)):
        receipt=write(stage/'materialization'/scene/'result.json',dict(source=source,scene_id=scene,
            materializer_commit=m.MATERIALIZATION_COMMIT,status='PASS',paper_ready=False,
            planned_objects=count,planned_manipulation_episodes=20,stage_config=identity(config),
            variants={p:dict(manifest=member) for p,member in scenes[scene]['variants'].items()}))
        scenes[scene]['receipt']=identity(receipt)
    handoff=write(stage/'materialization/handoff.json',dict(status='PASS',paper_ready=False,
        materializer_commit=m.MATERIALIZATION_COMMIT,planned_manipulation_episodes=40,
        fixed_task_ids=['fixed'],scenes=scenes,source=source))
    calls=[]
    def validate(path,**kw):
        calls.append(str(path))
        return {'roster':{'job_count':7 if kw['expected_scene_id']==m.compact.SCENES[0] else 10}}
    api=SimpleNamespace(identity=identity,read=lambda p:json.loads(Path(p).read_text()),
        FIXED_TASKS=['fixed'],SCENES=dict(zip(m.compact.SCENES,(7,10))),contract_digest=lambda c:m.MATERIALIZATION_CONTRACT,
        resource=lambda c,n:identity(config),inspect_source=lambda p:dict(status='READY',source=source),
        materializer=SimpleNamespace(validate_materialized_factory=validate))
    monkeypatch.setattr(m,'_api',lambda:api)
    monkeypatch.setattr(m,'_validate_original_variants',lambda requests,runtime:[validate(r['factory'],expected_scene_id=r['scene'],expected_policy_id=r['arm']) for r in requests])
    monkeypatch.setattr(m.compact,'checked_protocol',lambda *a:None)
    return handoff,api,calls,source


def test_original_products_authenticated_without_copy(chain):
    path,api,calls,source=chain
    result=m.inspect_materialization(path)
    assert result['status']=='READY' and result['source']['canonical']==source
    assert len(calls)==4 and all('/'+m.MATERIALIZATION_FREEZE+'/materialization/' in p for p in calls)


@pytest.mark.parametrize('kind',['scope','commit','count','scene','e0','receipt','descriptor','variant','source'])
def test_materialization_tamper_rejected(chain,kind):
    path,api,calls,source=chain
    h=api.read(path);scene=m.compact.SCENES[0]
    if kind=='scope':h['paper_ready']=True
    elif kind=='commit':h['materializer_commit']='f'*40
    elif kind=='count':h['planned_manipulation_episodes']=30
    elif kind=='scene':h['scenes'].pop(scene)
    elif kind=='e0':write(path.parent.parent/'contract/freeze_manifest.json',dict(code=dict(commit='wrong',dirty=False),freeze_id='original'))
    elif kind in ('receipt','descriptor'):Path(h['scenes'][scene][kind]['path']).write_text('{}')
    elif kind=='variant':Path(h['scenes'][scene]['variants']['A0']['path']).write_text('changed')
    elif kind=='source':h['source']['e3_root']='other'
    write(path,h)
    with pytest.raises(ValueError):m.inspect_materialization(path)


@pytest.fixture
def execution(tmp_path,monkeypatch):
    root=tmp_path;freeze='new';stage=root/'outputs/icra2027'/freeze;stage.mkdir(parents=True)
    protocol=write(root/'protocol',{})
    handoff=write(root/'handoff',dict(scenes={s:dict(descriptor=dict(path=f'{s}.json')) for s in m.compact.SCENES}))
    config=dict(freeze_id=freeze,source=dict(handoff=dict(path=str(handoff)),canonical=dict(e3_root='canonical',protocol=identity(protocol))),
        menagerie_root='menagerie',openpi_resize_identity={})
    reviewed=dict(status='READY')
    monkeypatch.setattr(m,'validate_stage',lambda *a:(copy.deepcopy(config),copy.deepcopy(reviewed)))
    monkeypatch.setattr(m.screen,'evidence_root',lambda:root)
    monkeypatch.setattr(m,'_api',lambda:SimpleNamespace(read=lambda p:json.loads(Path(p).read_text())))
    calls=[]
    def call(name, result):
        def inner(**kw):calls.append((name,kw));return result
        return inner
    monkeypatch.setattr(m.screen,'prepare_automatic_candidates',call('population',{}))
    monkeypatch.setattr(m.screen,'prepare_automatic_task_suites',call('tasks',{}))
    monkeypatch.setattr(m.screen,'qualify_scene',call('qualify',{'planned':180,'passed':0}))
    monkeypatch.setattr(m.screen,'_validate_qualifier_output',call('validate',{'metric_rows':[{}]*180}))
    monkeypatch.delenv('SLURM_ARRAY_JOB_ID',raising=False)
    monkeypatch.delenv('SLURM_GPUS_ON_NODE',raising=False)
    return stage,config,reviewed,calls


def test_delegates_full_population_without_task_filter(execution):
    stage,c,r,calls=execution
    result=m.run(config_path='c',stage_root=stage,expected_commit='exact',scene_id=m.compact.SCENES[0])
    assert [x[0] for x in calls]==['population','tasks','qualify']
    assert calls[0][1]['export'] is True and calls[0][1]['e3_root']=='canonical'
    assert calls[-1][1]['automatic_population'] is True
    assert result=={'planned':180,'passed':0}
    assert all('tasks' not in kw and 'task_ids' not in kw for _,kw in calls)


def test_resume_revalidates_existing_qualifier(execution):
    stage,c,r,calls=execution;scene=m.compact.SCENES[0]
    for member in (f'automatic_candidates/{scene}/population',f'scene_prepares/{scene}',f'scene_qualifiers/{scene}'):
        (stage/member).mkdir(parents=True)
    result=m.run(config_path='c',stage_root=stage,expected_commit='exact',scene_id=scene)
    assert [x[0] for x in calls]==['validate'] and result['planned_cells']==180


def test_waiting_has_no_side_effects(execution):
    stage,c,r,calls=execution;r['status']='WAITING'
    result=m.run(config_path='c',stage_root=stage,expected_commit='exact',scene_id=m.compact.SCENES[0])
    assert result['status']=='WAITING' and not calls


@pytest.mark.parametrize('env',[('SLURM_ARRAY_JOB_ID','1'),('SLURM_GPUS_ON_NODE','1')])
def test_no_array_or_gpu(execution,monkeypatch,env):
    stage,c,r,calls=execution;monkeypatch.setenv(*env)
    with pytest.raises(ValueError):m.run(config_path='c',stage_root=stage,expected_commit='exact',scene_id=m.compact.SCENES[0])
    assert not calls


def test_invalid_scene_not_substituted(execution):
    stage,c,r,calls=execution
    with pytest.raises(ValueError):m.run(config_path='c',stage_root=stage,expected_commit='exact',scene_id='replacement')
    assert not calls


def test_camera_retains_failed_gate(execution,monkeypatch):
    stage,c,r,calls=execution
    def writer(**kw):
        write(kw['out'],{})
    monkeypatch.setattr(m.camera,'write_automatic_camera_config',writer)
    monkeypatch.setattr(m.camera,'run_automatic_gate',lambda **kw:dict(planned_cells=40,camera_scorer_pass=False))
    result=m.run(config_path='c',stage_root=stage,expected_commit='exact',camera_phase=True)
    assert result==dict(planned_cells=40,camera_scorer_pass=False) and not calls


def test_existing_camera_config_drift_rejected(execution):
    stage,c,r,calls=execution;write(stage/'automatic_camera_config.json',{})
    with pytest.raises(ValueError,match='config drift'):
        m.run(config_path='c',stage_root=stage,expected_commit='exact',camera_phase=True)


@pytest.fixture
def stage_validation(tmp_path,monkeypatch):
    freeze='new';root=tmp_path;stage=root/'outputs/icra2027'/freeze;stage.mkdir(parents=True)
    python=str(Path(__import__('sys').executable).resolve())
    c=dict(schema_version=1,scope=m.SCOPE,paper_ready=False,freeze_id=freeze,source={'handoff':{'path':'h'}},
        python=python,menagerie_root='model',menagerie={'root':'model'},openpi_resize_identity={'sha':'pinned'})
    config=write(root/'config',c)
    e0=write(stage/'contract/freeze_manifest.json',dict(freeze_id=freeze,code=dict(commit='exact',dirty=False)))
    api=SimpleNamespace(read=lambda p:json.loads(Path(p).read_text()),contract_digest=lambda c:None,
        identity=identity,resource=lambda c,n:identity(config) if n.endswith('_config') else identity(python))
    monkeypatch.setattr(m,'_api',lambda:api)
    monkeypatch.setattr(m.screen,'_code_snapshot',lambda expected:{'commit':expected})
    monkeypatch.setattr(m.screen,'evidence_root',lambda:root)
    monkeypatch.setattr(m,'inspect_materialization',lambda p:dict(status='READY',source=c['source']))
    monkeypatch.setattr(m.camera,'_menagerie_snapshot',lambda *a:dict(root='model',files={}))
    monkeypatch.setattr(m.camera,'_openpi_snapshot',lambda:dict(sha='pinned'))
    return config,stage,e0,c,api


def test_stage_exact_e0_source_and_models(stage_validation):
    config,stage,e0,c,api=stage_validation
    assert m.validate_stage(config,stage,'exact')[1]['status']=='READY'


@pytest.mark.parametrize('mutation',['scope','paper','source','dirty','freeze','model','resize','python','config_hash'])
def test_stage_drift_rejected(stage_validation,mutation):
    path,stage,e0,c,api=stage_validation;contract=api.read(e0)
    if mutation=='scope':c['scope']='other'
    if mutation=='paper':c['paper_ready']=True
    if mutation=='source':contract['code']['commit']='other'
    if mutation=='dirty':contract['code']['dirty']=True
    if mutation=='freeze':contract['freeze_id']='other'
    if mutation=='model':c['menagerie']['root']='other'
    if mutation=='resize':c['openpi_resize_identity']['sha']='other'
    if mutation=='python':c['python']='other'
    if mutation=='config_hash':api.resource=lambda c,n:{'sha256':'bad'}
    write(path,c);write(e0,contract)
    with pytest.raises(ValueError):m.validate_stage(path,stage,'exact')


@pytest.mark.parametrize('mutation',['commit','dirty','split_root','manifest_commit'])
def test_historical_validator_rejects_aliases(tmp_path,monkeypatch,mutation):
    requests=[]
    root=tmp_path/'oldcode';root.mkdir()
    for i in range(2):
        factory=tmp_path/f'f{i}'
        write(factory/'materialization_manifest.json',{'provenance':dict(code_root=str(root) if mutation!='split_root' or i==0 else 'other',
            validator_commit=m.MATERIALIZATION_COMMIT if mutation!='manifest_commit' else 'wrong')})
        requests.append(dict(factory=str(factory),scene='scene',arm='A0'))
    def check(args,**kw):
        if 'rev-parse' in args:return 'wrong' if mutation=='commit' else m.MATERIALIZATION_COMMIT
        return ' M tracked' if mutation=='dirty' else ''
    monkeypatch.setattr(m.subprocess,'check_output',check)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:pytest.fail('must reject before historical execution'))
    with pytest.raises(ValueError):m._validate_original_variants(requests,{'path':'python'})


def test_historical_validator_uses_original_interpreter_and_worktree(tmp_path,monkeypatch):
    root=tmp_path/'oldcode';root.mkdir();factory=tmp_path/'factory'
    write(factory/'materialization_manifest.json',{'provenance':dict(code_root=str(root),validator_commit=m.MATERIALIZATION_COMMIT)})
    monkeypatch.setattr(m.subprocess,'check_output',lambda args,**kw:m.MATERIALIZATION_COMMIT if 'rev-parse' in args else '')
    monkeypatch.setattr(m.screen,'evidence_root',lambda:tmp_path)
    seen=[]
    def run(args,**kw):
        seen.append((args,kw));return SimpleNamespace(stdout='[{"roster":{"job_count":7}}]')
    monkeypatch.setattr(m.subprocess,'run',run)
    rows=m._validate_original_variants([dict(factory=str(factory),scene='scene',arm='A0')],{'path':'original-python'})
    args,kw=seen[0]
    assert args[0]=='original-python' and kw['cwd']==root
    assert 'PYTHONPATH' not in kw['env'] and kw['check'] is True and rows[0]['roster']['job_count']==7
