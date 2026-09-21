"""No model or metric execution: test reuse-only materialization admission."""
import json
from pathlib import Path
import pytest
from run.icra2027 import e2_full_materialization as m

@pytest.fixture
def case(tmp_path,monkeypatch):
    scene='13c3e046d7';second='1ada7a0617';root=tmp_path/'output'
    config=dict(freeze_id='20260906-abcdef1-v1',scene_ids=[scene,second],pilot_scene=scene,
                e3_root='sealed_e3',descriptors={s:{'scene_id':s} for s in [scene,second]})
    contract=dict(code={'commit':'a'*40})
    p=dict(population={scene:1,second:1},resolved_jobs={'path':'jobs'})
    cp=tmp_path/'config.json';ep=tmp_path/'e0.json'
    cp.write_text(json.dumps(config));ep.write_text(json.dumps(contract))
    monkeypatch.setattr(m,'context',lambda *_:(config,contract,p,root))
    monkeypatch.setattr(m.protocol,'read',lambda _:{'scenes':[dict(scene_id=s,jobs=[dict(object_slot='obj_1000')]) for s in [scene,second]]})
    def materialize(**kwargs):
        m.write_new(kwargs['out']/'materialization_manifest.json',dict(roster=dict(object_slots=['obj_1000'],job_count=1,accepted_count=0)))
    monkeypatch.setattr(m.materializer,'materialize_factory_variant',materialize)
    monkeypatch.setattr(m.materializer,'validate_materialized_factory',lambda *_,**__: {'valid':True})
    monkeypatch.delenv('SLURM_ARRAY_JOB_ID',raising=False);monkeypatch.delenv('SLURM_JOB_GPUS',raising=False)
    return cp,ep,scene,second,root,config,contract


def test_pilot_and_next_share_canonical_producer(case):
    cp,ep,scene,second,out,c,e=case
    r=m.run(cp,ep,scene)
    assert r['planned_objects']==1 and r['accepted_objects']==0
    assert m.checked_result(c,e,out,scene)[0]==r
    assert m.run(cp,ep,second)['status']=='PASS'
    with pytest.raises(FileExistsError):m.run(cp,ep,scene)


def test_no_scene_substitution_or_full_before_pilot(case):
    cp,ep,scene,second,*_=case
    with pytest.raises(ValueError,match='outside'):m.run(cp,ep,'ffffffff00')
    with pytest.raises((ValueError,FileNotFoundError)):m.run(cp,ep,second)


def test_failed_materializer_has_typed_receipt_and_no_retry(case,monkeypatch):
    cp,ep,scene,_,out,*_=case
    def bad(**_):raise ValueError('source artifact changed')
    monkeypatch.setattr(m.materializer,'materialize_factory_variant',bad)
    with pytest.raises(ValueError,match='source artifact'):m.run(cp,ep,scene)
    result=json.loads((out/'failures'/f'{scene}.json').read_text())
    assert result['planned_objects']==1 and result['status']=='FAILED'
    assert not (out/'handoff'/scene).exists()
    with pytest.raises(FileExistsError):m.run(cp,ep,scene)


def test_materialized_roster_cannot_omit_abstention(case,monkeypatch):
    cp,ep,scene,*_=case
    def wrong(**kwargs):m.write_new(kwargs['out']/'materialization_manifest.json',dict(roster=dict(object_slots=[],job_count=0,accepted_count=0)))
    monkeypatch.setattr(m.materializer,'materialize_factory_variant',wrong)
    with pytest.raises(ValueError,match='roster'):m.run(cp,ep,scene)


def test_tampered_handoff_not_a_pilot(case):
    cp,ep,scene,second,out,c,e=case;m.run(cp,ep,scene)
    (out/'handoff'/scene/'result.json').write_text('{}')
    with pytest.raises(ValueError):m.run(cp,ep,second)

@pytest.mark.parametrize('key',['SLURM_ARRAY_JOB_ID','SLURM_JOB_GPUS'])
def test_no_array_or_gpu(case,monkeypatch,key):
    cp,ep,scene,*_=case;monkeypatch.setenv(key,'1')
    with pytest.raises(ValueError,match='ordinary CPU'):m.run(cp,ep,scene)

@pytest.mark.parametrize('change',['none','missing_scene','swapped_pilot','descriptor','denominator','source','runtime'])
def test_context_binds_exact_missing_factory_admission(tmp_path,monkeypatch,change):
    import sys
    roster=[f'{i:010x}' for i in range(50)];missing=roster[:9]
    source=dict(source=dict(protocol={'path':'tasks'},e3_root='original_e3',scenes={s:dict(descriptor={'scene_id':s}) for s in roster}))
    tasks=dict(qualification_tasks=[{'scene_id':s} for s in roster[9:]])
    p=dict(scene_ids=roster,factory_execution={'path':'execution'})
    c=dict(schema_version=1,scope=m.SCOPE,paper_ready=False,freeze_id='20260906-abcdef1-v1',full_protocol={'path':'protocol'},scene_ids=missing.copy(),pilot_scene=missing[0],policy_id='A4',planned_full_scenes=50,planned_full_objects=1871,planned_full_views=400,e3_root='original_e3',descriptors={s:{'scene_id':s} for s in missing},python=sys.executable)
    if change=='missing_scene':c['scene_ids'].pop()
    elif change=='swapped_pilot':c['pilot_scene']=missing[1]
    elif change=='descriptor':c['descriptors'][missing[0]]={'scene_id':missing[1]}
    elif change=='denominator':c['planned_full_objects']=1800
    elif change=='source':c['e3_root']='another_e3'
    elif change=='runtime':c['python']='/usr/bin/other_python'
    monkeypatch.setattr(m,'_bound_contract',lambda *_:(c,{}))
    monkeypatch.setattr(m,'_checked',lambda ref:Path(ref['path']))
    monkeypatch.setattr(m.protocol,'validate_protocol',lambda _:p)
    monkeypatch.setattr(m.protocol,'read',lambda ref: source if ref['path']=='execution' else tasks)
    monkeypatch.setattr(m.e3,'_validate_cli_execution',lambda *_,**__:None)
    if change=='none':assert m.context(tmp_path/'config',tmp_path/'e0')[0]==c
    else:
        with pytest.raises(ValueError):m.context(tmp_path/'config',tmp_path/'e0')
