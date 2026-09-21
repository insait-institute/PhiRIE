import copy
import json
from pathlib import Path

import pytest

from run.icra2027 import e3_pilot_evaluation as audit


def fixture_rows(matched=False):
    evidence = {'settle_stable':False,'settle_drift_m':0.12}
    proposal = {'proposal_id':'j:trellis','job_id':'j','evidence':evidence,
                'input_hashes':{'visible_observation':'a'*64}}
    reference = {'status':'matched' if matched else 'unmatched',
                 'evaluation_surface':{'sha256':'b'*64} if matched else None}
    metric = (dict(f1_20=0.1,cd_cm=5.,collapse=False,evaluation_surface_sha256='b'*64,
                   registration_surface_sha256='a'*64,**evidence) if matched else
              dict(f1_20=None,cd_cm=None,collapse=None,geometry_evaluation_status='unmatched',**evidence))
    ledger = [{'job_id':'j','policy_id':p,'terminal_action':'accept' if p!='A4' else 'abstain',
               'selected_proposal_id':'j:trellis' if p!='A4' else None} for p in audit.e3.POLICY_IDS]
    rows = [dict(r,accepted=r['terminal_action']=='accept',metrics=metric if r['selected_proposal_id'] else None,
                 physical_metrics=evidence if r['selected_proposal_id'] else None) for r in ledger]
    shard = dict(job_count=1,geometry_reference_jobs=int(matched),geometry_unmatched_jobs=int(not matched),
                 proposal_metrics={'j:trellis':metric},rows=rows,source='frozen')
    return shard,{'proposals':[proposal]},ledger,{'j':reference},{'source':'frozen'}


@pytest.mark.parametrize('matched',[False,True])
def test_complete_rows_keep_unmatched_and_abstained_jobs(matched):
    audit._check_evaluation_rows(*fixture_rows(matched))


@pytest.mark.parametrize('field',['source','job_count','geometry_reference_jobs','geometry_unmatched_jobs'])
def test_wrong_identity_or_denominator_rejected(field):
    values=fixture_rows();values[0][field]='changed'
    with pytest.raises(ValueError):audit._check_evaluation_rows(*values)


@pytest.mark.parametrize('mutation',['drop','duplicate','outcome','geometry','probe','proposal'])
def test_metric_rows_cannot_drop_failures_or_change_controller(mutation):
    values=fixture_rows();shard=values[0]
    if mutation=='drop':shard['rows'].pop()
    if mutation=='duplicate':shard['rows'][-1]=copy.deepcopy(shard['rows'][0])
    if mutation=='outcome':shard['rows'][-1]['accepted']=True
    if mutation=='geometry':shard['proposal_metrics']['j:trellis']['f1_20']=0.9
    if mutation=='probe':shard['proposal_metrics']['j:trellis']['settle_stable']=True
    if mutation=='proposal':shard['proposal_metrics']['unexpected']={}
    with pytest.raises(ValueError):audit._check_evaluation_rows(*values)


@pytest.mark.parametrize('field',['evaluation_surface_sha256','registration_surface_sha256'])
def test_geometry_reference_drift_rejected(field):
    values=fixture_rows(True);values[0]['proposal_metrics']['j:trellis'][field]='c'*64
    with pytest.raises(ValueError,match='surface binding'):audit._check_evaluation_rows(*values)


def test_registration_surface_may_not_be_evaluation_surface():
    values=fixture_rows(True)
    values[3]['j']['evaluation_surface']['sha256']='a'*64
    values[0]['proposal_metrics']['j:trellis']['evaluation_surface_sha256']='a'*64
    with pytest.raises(ValueError,match='surface binding'):audit._check_evaluation_rows(*values)


@pytest.fixture
def retry(tmp_path,monkeypatch):
    import numpy as np
    monkeypatch.setattr(audit.e3,'checked_repo_path',lambda p,*a,**kw:Path(p))
    parent={'proposal_id':'j:trellis','job_id':'j','tool':'trellis','source_up_hypothesis':'+z',
            'artifact_paths':{'transform':str(tmp_path/'initial.npy')}}
    paths={role:tmp_path/name for role,name in [('retry','retry.json'),('transform','new.npy'),('registration','registration.json')]}
    paths['retry'].write_text(json.dumps({'action':'registration_signed_source_up_restart',
        'proposal_id':'j:retry','parent_proposal_id':'j:trellis','source_up_hypothesis':'-x'}))
    np.save(paths['transform'],np.eye(4))
    paths['registration'].write_text(json.dumps({'source_up_hypothesis':'-x','T':np.eye(4).tolist()}))
    proposal={'proposal_id':'j:retry','job_id':'j','tool':'registration_retry','parent_proposal_ids':['j:trellis'],'source_up_hypothesis':'-x',
        'artifact_paths':{k:str(p) for k,p in paths.items()},
        'artifact_hashes':{k:audit.e3.sha256_file(p) for k,p in paths.items()},
        'artifact_sizes':{k:p.stat().st_size for k,p in paths.items()}}
    return {'proposals':[parent,proposal]},paths


def test_retry_requires_new_artifact_and_declared_action(retry):
    assert audit._check_retry_artifacts(retry[0])=={'j'}


@pytest.mark.parametrize('mutation',['parent','job','tamper','same_path','action'])
def test_invalid_retry_is_not_counted(retry,mutation):
    control,paths=retry;proposal=control['proposals'][1]
    if mutation=='parent':proposal['parent_proposal_ids']=['j:retry']
    if mutation=='job':proposal['job_id']='other'
    if mutation=='tamper':paths['transform'].write_bytes(b'tampered')
    if mutation=='same_path':control['proposals'][0]['artifact_paths']['transform']=str(paths['transform'])
    if mutation=='action':
        value=json.loads(paths['retry'].read_text());value['action']='reread_same_file'
        paths['retry'].write_text(json.dumps(value))
        proposal['artifact_hashes']['retry']=audit.e3.sha256_file(paths['retry'])
        proposal['artifact_sizes']['retry']=paths['retry'].stat().st_size
    with pytest.raises(ValueError):audit._check_retry_artifacts(control)


@pytest.mark.parametrize('mutation',['same_axis','registration_axis','transform'])
def test_renamed_action_or_mismatched_transform_is_not_retry(retry,mutation):
    control,paths=retry;p=control['proposals'][1]
    if mutation=='same_axis':p['source_up_hypothesis']='+z'
    else:
        value=json.loads(paths['registration'].read_text())
        if mutation=='registration_axis':value['source_up_hypothesis']='-y'
        else:value['T'][0][0]=3.
        paths['registration'].write_text(json.dumps(value))
        p['artifact_hashes']['registration']=audit.e3.sha256_file(paths['registration'])
        p['artifact_sizes']['registration']=paths['registration'].stat().st_size
    with pytest.raises(ValueError):audit._check_retry_artifacts(control)


def test_selected_tree_authenticates_actual_files_and_empty_directories(tmp_path):
    root=tmp_path/'selected';(root/'A0').mkdir(parents=True);(root/'A4').mkdir()
    p=root/'A0/object.json';p.write_text('{}')
    seal={'selected_asset_members':{'A0/object.json':{'sha256':audit.e3.sha256_file(p),'size_bytes':2}},
          'selected_asset_directories':['A0','A4']}
    assert audit._check_selected_tree(root,seal)==[p]
    p.write_text('{"edited":true}')
    with pytest.raises(ValueError,match='changed'):audit._check_selected_tree(root,seal)
    p.unlink()
    with pytest.raises(ValueError,match='population'):audit._check_selected_tree(root,seal)


def test_selected_tree_rejects_symlink_and_extra_directory(tmp_path):
    root=tmp_path/'selected';root.mkdir()
    seal={'selected_asset_members':{},'selected_asset_directories':[]}
    (root/'extra').mkdir()
    with pytest.raises(ValueError,match='population'):audit._check_selected_tree(root,seal)
    (root/'extra').rmdir();(root/'link').symlink_to(tmp_path)
    with pytest.raises(ValueError,match='symlink'):audit._check_selected_tree(root,seal)


def test_reference_binding_validates_every_scene_surface(tmp_path,monkeypatch):
    path=tmp_path/'surfaces/scene/gt_4.npy';path.parent.mkdir(parents=True);path.write_bytes(b'surface')
    row={'status':'matched','matched_gt_id':4,'evaluation_surface':audit.identity(path)}
    audit._check_reference_bindings([row],tmp_path,'scene')
    with pytest.raises(ValueError,match='scene/path'):audit._check_reference_bindings([row],tmp_path,'other')
    row['status']='unmatched'
    with pytest.raises(ValueError,match='invented'):audit._check_reference_bindings([row],tmp_path,'scene')
    row.update(matched_gt_id=None,evaluation_surface=None)
    audit._check_reference_bindings([row],tmp_path,'scene')


@pytest.mark.parametrize('mode,scenes,jobs',[('pilot',1,15),('full',49,1871),('full',50,1870)])
def test_partial_cohort_cannot_start_audit_or_touch_gt(tmp_path,monkeypatch,mode,scenes,jobs):
    import yaml
    config=tmp_path/'matching.yaml';config.write_text(yaml.safe_dump(dict(mode=mode,planned_scenes=scenes,planned_jobs=jobs)))
    monkeypatch.setattr(audit.matching,'validate_construction',lambda _:pytest.fail('incomplete audit reached GT boundary'))
    with pytest.raises(ValueError,match='50/1871'):audit.audit_full(config,tmp_path/'contract.json')


def test_completed_audit_cannot_be_overwritten(tmp_path,monkeypatch):
    import yaml
    root=tmp_path/'outputs/icra2027/eval';root.mkdir(parents=True)
    target=root/'independent_evaluation_completion_audit.json';target.write_text('original')
    monkeypatch.setattr(audit.e3,'REPOSITORY_ROOT',tmp_path)
    config=tmp_path/'matching.yaml';config.write_text(yaml.safe_dump(dict(mode='full',planned_scenes=50,planned_jobs=1871,freeze_id='eval')))
    with pytest.raises(FileExistsError):audit.audit_full(config,tmp_path/'contract.json')
    assert target.read_text()=='original'


@pytest.fixture
def full_flow(tmp_path,monkeypatch):
    """Small artifacts, complete50/1871/9355 roster, mocked existing producer."""
    import yaml
    root=tmp_path/'outputs/icra2027';construction=root/'construction/agentic';out=root/'evaluation'
    pub=out/'agentic';pub.mkdir(parents=True)
    config_path=tmp_path/'matching.yaml';contract=tmp_path/'contract.json';contract.write_text('{}')
    (tmp_path/'construction_jobs.yaml').write_text('{}');(tmp_path/'construction_policies.yaml').write_text('{}')
    scenes=[];inventory=[];bindings=[];references=[];controls={};evaluations={};events=[]
    for index in range(50):
        scene=f'{index:010x}';jobs=[{'job_id':f'{scene}/obj_{j}'} for j in range(58 if index==0 else 37)]
        inventory.append({'scene_id':scene,'jobs':jobs})
        scenes.append({'scene_id':scene,'construction_root':str(construction),'gt_inputs':{}})
        bindings.append({'scene_id':scene,'construction_root':str(construction),'controller_shard_sha256':'sealed'})
        refs=[{'scene_id':scene,'job_id':j['job_id'],'status':'unmatched','matched_gt_id':None,'evaluation_surface':None} for j in jobs]
        references.extend(refs)
        ledger=[{'scene_id':scene,'job_id':j['job_id'],'policy_id':p,'terminal_action':'abstain',
                 'selected_proposal_id':None,'retry_produced':False} for j in jobs for p in audit.e3.POLICY_IDS]
        d=construction/'control'/scene;d.mkdir(parents=True)
        (d/'job_ledger.jsonl').write_text('\n'.join(json.dumps(r) for r in ledger))
        (d/'seal.json').write_text('{}');(d/'controller_shard.json').write_text('{}')
        controls[scene]=(d,{'proposals':[]},{'members':{'controller_shard.json':'sealed'}})
        d=pub/'evaluation'/scene;d.mkdir(parents=True);(d/'seal.json').write_text('{}');(d/'eval_shard.json').write_text('{}')
        evaluations[scene]=dict(job_count=len(jobs),geometry_reference_jobs=0,geometry_unmatched_jobs=len(jobs),
            proposal_metrics={},rows=[dict(r,accepted=False,metrics=None,physical_metrics=None) for r in ledger],
            evaluation_freeze_id='evaluation',construction_freeze_id='construction',freeze_id='construction',evaluation_code_commit='c'*40)
    config=dict(mode='full',planned_scenes=50,planned_jobs=1871,freeze_id='evaluation',scenes=scenes,
                agentic_uncertainty=copy.deepcopy(audit.e3.AGENTIC_UNCERTAINTY_PROTOCOL))
    config_path.write_text(yaml.safe_dump(config))
    refpath=out/'evaluation_matching/evaluation_references.json';refpath.parent.mkdir();refpath.write_text('{}')
    (refpath.parent/'seal.json').write_text('{}')
    for shard in evaluations.values():shard['external_evaluation_manifest_sha256']=audit.e3.sha256_file(refpath)
    manifest=dict(planned_scenes=50,planned_jobs=1871,code_commit='c'*40,scenes=bindings,rows=references,
                  config_sha256=audit.e3.sha256_file(config_path))
    payload=dict(counts={'scenes':50,'jobs_per_policy':1871,'policy_object_rows':9355,'genuine_retry_jobs':0},
        rows=[{'policy_id':p,'runtime_accounting_status':'NOT_RUN'} for p in audit.e3.POLICY_IDS],
        headline_eligible=False,paper_ready=False,runtime_scope='existing_scope',created_utc='original')
    (pub/'agentic_ablation.json').write_text(json.dumps(payload))
    (pub/'runtime_accounting.json').write_text('{}');(pub/'selected_assets').mkdir()
    for name in ('agentic_paired_uncertainty.json','agentic_paired_uncertainty.csv'):
        (pub/name).write_text('synthetic producer bytes; uncertainty arithmetic tested separately')
    members=('agentic_ablation.json','runtime_accounting.json',
             'agentic_paired_uncertainty.json','agentic_paired_uncertainty.csv')
    seal={'members':{name:audit.e3.sha256_file(pub/name) for name in members},
          'selected_asset_members':{},'selected_asset_directories':[]}
    (pub/'aggregate_seal.json').write_text(json.dumps(seal))
    monkeypatch.setattr(audit.e3,'REPOSITORY_ROOT',tmp_path)
    monkeypatch.setattr(audit.e3,'_validate_cli_execution',lambda *a,**k:events.append('E0'))
    monkeypatch.setattr(audit.matching,'validate_construction',lambda _:events.append('construction'))
    monkeypatch.setattr(audit.e3,'_load_inventory',lambda *a,**k:({'scenes':inventory,'counts':{'scenes':50,'jobs':1871,'policy_object_rows':9355}},{}))
    monkeypatch.setattr(audit.e3,'_load_control_scene',lambda _,s:controls[s])
    def load(*a,**k):
        assert events==['E0','construction'];events.append('GT');return {},manifest
    monkeypatch.setattr(audit.matching,'load_references',load)
    monkeypatch.setattr(audit.e3,'validate_ledger',lambda _:None)
    monkeypatch.setattr(audit.e3,'_load_eval_scene',lambda _,s,**k:evaluations[s])
    monkeypatch.setattr(audit.e3,'checked_repo_path',lambda p,*a,**k:Path(p))
    def aggregate(*args,evaluation_root,uncertainty_protocol):
        assert uncertainty_protocol==audit.e3.AGENTIC_UNCERTAINTY_PROTOCOL
        assert len(list((evaluation_root/'evaluation').iterdir()))==50
        events.append('canonical_aggregate_replay')
        for name in (*members,'aggregate_seal.json'):
            (evaluation_root/name).write_bytes((pub/name).read_bytes())
        return dict(payload,created_utc='replayed')
    monkeypatch.setattr(audit.e3,'run_aggregate',aggregate)
    monkeypatch.setattr(audit.e3,'_atomic_write_json',lambda p,v:p.write_text(json.dumps(v)))
    return config_path,contract,out,evaluations,events


def test_full_flow_authenticates_before_gt_and_replays_complete_only(full_flow):
    config,contract,out,_,events=full_flow
    result=audit.audit_full(config,contract)
    assert result['status']=='PASS' and result['terminal_policy_rows']==9355
    assert result['matched_jobs']==0 and result['unmatched_jobs']==1871
    assert events==['E0','construction','GT','canonical_aggregate_replay']
    assert result['paper_ready'] is False and result['claim_gate']=='NOT_RUN'
    assert (out/'independent_evaluation_completion_audit.json').is_file()


def test_full_flow_rejects_changed_last_scene_before_aggregate(full_flow):
    config,contract,out,shards,events=full_flow
    shards['0000000031']['rows'][-1]['accepted']=True
    with pytest.raises(ValueError,match='controller decision'):audit.audit_full(config,contract)
    assert 'canonical_aggregate_replay' not in events
    assert not (out/'independent_evaluation_completion_audit.json').exists()


def test_full_flow_rejects_unplanned_evaluation_child_before_gt(full_flow):
    config,contract,out,_,events=full_flow
    (out/'agentic/evaluation/extra').mkdir()
    with pytest.raises(ValueError,match='directory differs'):audit.audit_full(config,contract)
    assert events==['E0','construction']


def test_full_flow_rejects_reference_from_undeclared_scene(full_flow,monkeypatch):
    config,contract,out,_,events=full_flow
    original=audit.matching.load_references
    def changed(*args,**kwargs):
        references,manifest=original(*args,**kwargs)
        manifest['rows'].append({'scene_id':'unknown','job_id':'extra'})
        return references,manifest
    monkeypatch.setattr(audit.matching,'load_references',changed)
    with pytest.raises(ValueError,match='roster identity'):audit.audit_full(config,contract)
    assert 'canonical_aggregate_replay' not in events
    assert not (out/'independent_evaluation_completion_audit.json').exists()


@pytest.mark.parametrize("mutation", ["missing", "seed"])
def test_full_flow_uncertainty_protocol_drift_rejected_before_gt(full_flow, mutation):
    config, contract, out, _, events = full_flow
    value = audit.yaml.safe_load(config.read_text())
    if mutation == 'missing':
        del value['agentic_uncertainty']
    else:
        value['agentic_uncertainty']['seed'] = 1
    config.write_text(audit.yaml.safe_dump(value))
    with pytest.raises(ValueError, match='uncertainty protocol'):
        audit.audit_full(config, contract)
    assert events == []
    assert not (out/'independent_evaluation_completion_audit.json').exists()


def test_full_flow_missing_uncertainty_seal_member_fails_closed(full_flow):
    config, contract, out, _, _ = full_flow
    path = out/'agentic/aggregate_seal.json'
    seal = json.loads(path.read_text())
    del seal['members']['agentic_paired_uncertainty.csv']
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match='uncertainty outputs'):
        audit.audit_full(config, contract)
    assert not (out/'independent_evaluation_completion_audit.json').exists()
