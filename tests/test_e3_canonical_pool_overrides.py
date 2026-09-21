import copy
import json
from pathlib import Path

import pytest
import yaml

from run.icra2027 import e3_fresh_canonical_config as builder


@pytest.fixture
def declared(tmp_path,monkeypatch):
    monkeypatch.setattr(builder.e3,'CODE_ROOT',tmp_path)
    monkeypatch.setattr(builder.e3,'REPOSITORY_ROOT',tmp_path)
    scenes=sorted(builder.RVG_OVERRIDE_SCENES)+[f'{i:010x}' for i in range(48)]
    roster=tmp_path/'roster.yaml';roster.write_text(yaml.safe_dump({'population':{'scene_ids':scenes}}))
    records=[]
    for s in sorted(builder.RVG_OVERRIDE_SCENES):
        old=tmp_path/'old';new=tmp_path/'new'
        directory=old/'rvg_initial'/s;directory.mkdir(parents=True)
        for name in ('proposal_pool.json','input_manifest.json','proposal_records.jsonl'):(directory/name).write_text('{}')
        audit=old/'terminal'/f'{s}.json';audit.parent.mkdir(exist_ok=True);audit.write_text('{}')
        records.append(dict(scene_id=s,tool='reconviagen',path=str(new/'rvg_initial'/s/'proposal_pool.json'),
            freeze_root=str(new),source_commit='a'*40,original_pool=dict(path=str(directory/'proposal_pool.json'),
            sha256=builder.e3.sha256_file(directory/'proposal_pool.json'),freeze_root=str(old),
            terminal_audit=dict(path=str(audit),sha256=builder.e3.sha256_file(audit)))))
    manifest=tmp_path/'overrides.yaml';manifest.write_text(yaml.safe_dump(dict(schema_version=1,overrides=records)))
    return roster,scenes,manifest,records


@pytest.mark.parametrize('change',['unknown','missing','duplicate','wrong_tool','wrong_path','wrong_original'])
def test_override_population_and_paths_fail_closed(declared,change):
    _,scenes,manifest,records=declared
    if change=='unknown':records[0]['scene_id']=scenes[-1]
    if change=='missing':records.pop()
    if change=='duplicate':records[1]=copy.deepcopy(records[0])
    if change=='wrong_tool':records[0]['tool']='trellis'
    if change=='wrong_path':records[0]['path']=records[1]['path']
    if change=='wrong_original':records[0]['original_pool']['path']=records[0]['path']
    manifest.write_text(yaml.safe_dump(dict(schema_version=1,overrides=records)))
    with pytest.raises(ValueError):builder.load_pool_overrides(manifest,scenes)


def test_missing_selected_override_never_falls_back_to_old_pool(declared,tmp_path,monkeypatch):
    roster,scenes,manifest,records=declared
    calls=[]
    def missing(d,t,r,**kw):
        calls.append(r)
        return [str(r/'proposal_pool.json')] if r.parent==tmp_path/'new/rvg_initial' else []
    monkeypatch.setattr(builder,'_missing_inputs',missing)
    monkeypatch.setattr(builder,'_source_payload',lambda *a,**kw:pytest.fail('must not normalize incomplete replacement'))
    result=builder.cohort_readiness(roster,tmp_path/'d',tmp_path/'t',tmp_path/'old/rvg_initial',
        trellis_freeze_root=tmp_path/'tf',rvg_freeze_root=tmp_path/'old',pool_override_manifest=manifest)
    assert result['status']=='WAITING_REAL_INITIAL_POOLS'
    assert result['scenes_with_required_files']==48 and not result['config_written']
    assert set(result['missing_inputs'])==builder.RVG_OVERRIDE_SCENES
    assert all(tmp_path/'old/rvg_initial'/s not in calls for s in builder.RVG_OVERRIDE_SCENES)


def test_mixed_roots_preserve_all_jobs_and_runtime_histories(declared,tmp_path,monkeypatch):
    from agents.orchestrator import automatic_inventory
    roster,scenes,manifest,records=declared;captured=[]
    monkeypatch.setattr(builder,'_missing_inputs',lambda *a,**kw:[])
    def source(d,t,r,**kw):
        scene=d.name
        if scene in builder.RVG_OVERRIDE_SCENES:
            assert r==tmp_path/'new/rvg_initial'/scene
            assert kw['rvg_freeze_root']==str(tmp_path/'new')
            assert 'rvg_terminal_audit' not in kw
        else:
            assert r==tmp_path/'old/rvg_initial'/scene
            assert kw['rvg_freeze_root']==tmp_path/'old'
        n=1822 if scene==scenes[0] else 1
        return dict(scene_id=scene,initial_pools={}),[dict(job_id=f'{scene}:auto:{1000+i}',prepared=True) for i in range(n)]
    def history(record,source,jobs):
        return dict(scene_id=source['scene_id'],tool='reconviagen',pools=[record['original_pool'],{'path':record['path'],'sha256':'b'*64}])
    def normalize(payload,*args):
        captured.append(payload)
        return {},{'counts':{'jobs':1871}},{'initial_pool_complete':True},''
    monkeypatch.setattr(builder,'_source_payload',source);monkeypatch.setattr(builder,'override_runtime_history',history)
    monkeypatch.setattr(automatic_inventory,'build_payloads',normalize)
    result=builder.cohort_readiness(roster,tmp_path/'d',tmp_path/'t',tmp_path/'old/rvg_initial',
        trellis_freeze_root=tmp_path/'tf',rvg_freeze_root=tmp_path/'old',pool_override_manifest=manifest)
    assert result['planned_jobs']==1871 and result['planned_policy_object_rows']==9355
    assert len(captured[0]['automatic_sources'])==50
    assert len(result['runtime_accounting']['process_histories'])==2
    assert captured[0]['pool_override_manifest']['sha256']==builder.e3.sha256_file(manifest)
    assert not result['config_written']


@pytest.mark.parametrize('change',['source','freeze','path'])
def test_override_producer_identity_mismatch_rejected_before_original_reuse(declared,change):
    _,_,_,records=declared;record=records[0]
    selected=dict(path=record['path'],producer_provenance=dict(source_commit=record['source_commit'],freeze_id='new'))
    if change=='source':selected['producer_provenance']['source_commit']='b'*40
    if change=='freeze':selected['producer_provenance']['freeze_id']='old'
    if change=='path':selected['path']=record['original_pool']['path']
    with pytest.raises(ValueError,match='source/freeze identity'):
        builder.override_runtime_history(record,dict(scene_id=record['scene_id'],initial_pools={'reconviagen':selected}),[])


def test_original_failed_pool_tamper_rejected(declared):
    _,_,_,records=declared;record=records[0]
    Path(record['original_pool']['path']).write_text('tampered')
    selected=dict(path=record['path'],producer_provenance=dict(source_commit=record['source_commit'],freeze_id='new'))
    with pytest.raises(ValueError,match='provenance changed'):
        builder.override_runtime_history(record,dict(scene_id=record['scene_id'],initial_pools={'reconviagen':selected}),[])
