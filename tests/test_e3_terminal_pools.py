import copy
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
import pytest

from agents.orchestrator.automatic_inventory import terminal_pool_rows
from run.icra2027 import e3_fresh_canonical_config as builder
from robo.manifest.hash import canonical_hash


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(builder.e3, 'REPOSITORY_ROOT', tmp_path)
    root=tmp_path/'freeze'; directory=root/'trellis_initial/09c1414f1b';directory.mkdir(parents=True)
    rows=[];audited=[]
    for index in range(4):
        prepared=index!=3
        status='available' if index<2 else 'generation_failed' if prepared else 'unavailable'
        row=dict(job_id=f'09c1414f1b:auto:{1000+index}',proposal_id=f'proposal-{index}',
            automatic_instance_id=str(1000+index),prepared=prepared,status=status,
            reason=None if index<2 else 'initial_tool_not_run' if prepared else 'preparation_unavailable',
            runtime={'status':'generated','seed':42,'wall_s':1.0} if index<2 else None,artifacts={})
        errors=[];verified=[]
        if index<2:
            for name in ('trellis_mesh.ply','trellis_gs.ply'):
                path=directory/str(index)/name;path.parent.mkdir(exist_ok=True)
                vertices=np.zeros(1,dtype=[('x','f4'),('y','f4'),('z','f4'),('opacity','f4')])
                if index==1 and name=='trellis_gs.ply':vertices['opacity']=np.inf
                PlyData([PlyElement.describe(vertices,'vertex')],text=False).write(str(path))
                identity={'path':str(path),'bytes':path.stat().st_size,'sha256':builder.e3.sha256_file(path)}
                row['artifacts'][name]=identity
                verified.append({'artifact':name,**identity})
                if index==1 and name=='trellis_gs.ply':
                    errors.append({'artifact':name,'reason':'nonfinite_numeric_field','field':'opacity',
                                   'nan_count':0,'positive_inf_count':1,'negative_inf_count':0})
        norm='available_verified' if index==0 else 'artifact_invalid' if index==1 else 'unavailable'
        reason=None if index==0 else 'artifact_validation_failed' if index==1 else 'no_per_object_completion_record_process_exit_1' if index==2 else 'preparation_unavailable'
        audited.append(dict(job_id=row['job_id'],proposal_id=row['proposal_id'],producer_status=status,
            normalized_status=norm,reason=reason,per_object_attempt_proven=index<2,
            artifact_validation_errors=errors,verified_artifacts=verified))
        rows.append(row)
    hashes={name:'d'*64 for name in builder.DISCOVERY_FILES}
    manifest=dict(code_commit='a'*40,config_sha256='b'*64,source_discovery_hashes=hashes,
                  source_gaussian_training_provenance=builder.FRESH)
    dump(directory/'input_manifest.json',manifest)
    records=directory/'proposal_records.jsonl';records.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    pool=dict(code_commit='a'*40,freeze_id='freeze',paper_ready=False,exit_code=1,planned_jobs=4,rows=rows,
        source_gaussian_training_provenance=builder.FRESH,source_discovery_hashes=hashes,
        input_manifest_sha256=builder.e3.sha256_file(directory/'input_manifest.json'),
        proposal_records_sha256=builder.e3.sha256_file(records))
    dump(directory/'proposal_pool.json',pool)
    dump(directory/'execution_status.json',{'status':'FAIL'})
    (directory/'generation.log').write_text('producer failure')
    contract=dict(freeze_id='freeze',code={'commit':'a'*40,'dirty':False},
                  resource_inventory=[{'kind':'experiment_config','sha256':'b'*64}])
    contract['contract_sha256']=canonical_hash(contract);dump(root/'contract/freeze_manifest.json',contract)
    audit=dict(schema_version=1,paper_ready=False,source_checkout_clean=True,independent_source_binding_pass=True,
        producer_source_commit='a'*40,producer_freeze_id='freeze',scene_id='09c1414f1b',config_sha256='b'*64,
        source_discovery_hashes=hashes,planned_jobs=4,producer_process_exit_code=1,job_id='123',
        scheduler='123|FAILED|1:0|hala|10\n',rows=audited,
        counts={'available_verified':1,'artifact_invalid':1,'unavailable':2},process_failure_classification='producer_failed')
    for name in ('input_manifest.json','proposal_pool.json','proposal_records.jsonl','execution_status.json','generation.log'):
        audit[name.replace('.','_')+'_sha256']=builder.e3.sha256_file(directory/name)
    path=root/'terminal_audit/audit.json';dump(path,audit)
    spec=dict(path=str(path),sha256=builder.e3.sha256_file(path),freeze_root=str(root))
    return directory,pool,manifest,spec,audit


def test_terminal_adapter_keeps_good_artifact_and_every_failed_job_without_mutation(terminal):
    directory,pool,manifest,spec,audit=terminal
    original=copy.deepcopy(pool);digest=builder.e3.sha256_file(directory/'proposal_pool.json')
    normalized,proof=terminal_pool_rows(directory,pool,manifest,spec)
    assert [r['status'] for r in normalized]==['available','unavailable','unavailable','unavailable']
    assert [r['job_id'] for r in normalized]==[r['job_id'] for r in pool['rows']]
    assert normalized[1]['reason']=='artifact_validation_failed' and not normalized[1]['artifacts']
    assert normalized[2]['runtime'] is None
    assert normalized[2]['reason']=='no_per_object_completion_record_process_exit_1'
    assert pool==original and builder.e3.sha256_file(directory/'proposal_pool.json')==digest
    assert proof['producer_process_exit_code']==1
    jobs=[{'job_id':row['job_id'],'prepared':row['prepared']} for row in pool['rows']]
    result=builder.authenticate_pool(directory,manifest['source_discovery_hashes'],jobs,
        freeze_root=spec['freeze_root'],terminal_audit=spec['path'])
    assert result['sha256']==digest and result['terminal_audit']==spec
    assert result['producer_provenance']['source_commit']=='a'*40
    assert not (directory/'postrun_audit.json').exists()


@pytest.mark.parametrize('change',['forged_available','lost_job','wrong_scene','wrong_source','wrong_config',
    'wrong_producer_exit','not_terminal','fake_success','wrong_counts','wrong_attempt','wrong_error_count',
    'wrong_artifact_hash','changed_log','changed_ply','forged_e0'])
def test_terminal_audit_cannot_hide_or_relabel_failures(terminal,change):
    directory,pool,manifest,spec,audit=terminal
    if change=='forged_available':audit['rows'][1].update(normalized_status='available_verified',reason=None,artifact_validation_errors=[])
    elif change=='lost_job':audit['rows'].pop()
    elif change=='wrong_scene':audit['scene_id']='38d58a7a31'
    elif change=='wrong_source':audit['producer_source_commit']='e'*40
    elif change=='wrong_config':audit['config_sha256']='e'*64
    elif change=='wrong_producer_exit':audit['producer_process_exit_code']=0
    elif change=='not_terminal':audit['scheduler']='123|RUNNING|0:0|hala|10\n'
    elif change=='fake_success':audit['scheduler']='123|COMPLETED|0:0|hala|10\n'
    elif change=='wrong_counts':audit['counts']['unavailable']=1
    elif change=='wrong_attempt':audit['rows'][2]['per_object_attempt_proven']=True
    elif change=='wrong_error_count':audit['rows'][1]['artifact_validation_errors'][0]['positive_inf_count']=0
    elif change=='wrong_artifact_hash':audit['rows'][0]['verified_artifacts'][0]['sha256']='0'*64
    elif change=='changed_log':(directory/'generation.log').write_text('different error')
    elif change=='changed_ply':Path(pool['rows'][0]['artifacts']['trellis_gs.ply']['path']).write_bytes(b'invalid')
    else:
        path=Path(spec['freeze_root'])/'contract/freeze_manifest.json';c=json.loads(path.read_text());c['code']['dirty']=True;dump(path,c)
    dump(Path(spec['path']),audit);spec['sha256']=builder.e3.sha256_file(spec['path'])
    with pytest.raises(ValueError):terminal_pool_rows(directory,pool,manifest,spec)


def test_terminal_audit_hash_is_frozen_separately(terminal):
    directory,pool,manifest,spec,audit=terminal
    Path(spec['path']).write_text('{}')
    with pytest.raises(ValueError,match='audit hash'):terminal_pool_rows(directory,pool,manifest,spec)


def test_missing_postrun_can_only_be_replaced_by_explicit_existing_terminal_audit(terminal):
    directory,pool,manifest,spec,audit=terminal
    missing=builder._missing_inputs(directory/'discovery',directory,directory,trellis_terminal_audit=spec['path'])
    assert str(directory/'postrun_audit.json') in missing  # RVG still lacks its audit.
    missing=builder._missing_inputs(directory/'discovery',directory,directory,trellis_terminal_audit=spec['path'],rvg_terminal_audit=spec['path'])
    assert str(directory/'postrun_audit.json') not in missing


def test_multiple_audit_batches_preserve_exact_paths_and_reject_conflicts(tmp_path):
    roots=[tmp_path/'batch1',tmp_path/'batch2']
    for root in roots:root.mkdir()
    scene='09c1414f1b'
    path=roots[1]/(scene+'.json');path.write_text('{}')
    assert builder.terminal_audit_for_scene(roots,scene)==str(path)
    first=roots[0]/path.name;first.write_text('{}')
    assert builder.terminal_audit_for_scene(roots,scene)==str(first)
    first.write_text('{"different":true}')
    with pytest.raises(ValueError,match='conflicting terminal audits'):
        builder.terminal_audit_for_scene(roots,scene)
    assert builder.terminal_audit_for_scene(roots,'38d58a7a31') is None


def test_unknown_raw_producer_status_is_not_converted_into_valid_unavailable(terminal):
    directory,pool,manifest,spec,audit=terminal
    pool['rows'][0]['status']='unknown_producer_state'
    audit['rows'][0]['producer_status']='unknown_producer_state'
    (directory/'proposal_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pool['rows']))
    pool['proposal_records_sha256']=builder.e3.sha256_file(directory/'proposal_records.jsonl')
    dump(directory/'proposal_pool.json',pool)
    for name in ('proposal_pool.json','proposal_records.jsonl'):
        audit[name.replace('.','_')+'_sha256']=builder.e3.sha256_file(directory/name)
    dump(Path(spec['path']),audit);spec['sha256']=builder.e3.sha256_file(spec['path'])
    with pytest.raises(ValueError,match='unknown original producer status'):
        terminal_pool_rows(directory,pool,manifest,spec)


@pytest.fixture
def rvg_terminal(terminal,monkeypatch):
    from run.icra2027 import e3_rvg_generation_pilot as rvg
    from run.icra2027.e3_rvg_terminal_audit import failure_rows,CLOSURE
    from run.icra2027.e3_auto_discovery_pilot import identity
    directory,pool,manifest,spec,audit=terminal
    new=directory.parent.parent/'rvg_initial'/directory.name;new.parent.mkdir();directory.rename(new);directory=new
    (directory/'execution_status.json').unlink()  # Original RVG never wrote this file.
    for row in pool['rows']:
        row.update(status='generation_failed' if row['prepared'] else 'unavailable',runtime=None,artifacts={})
    config=directory/'producer.yaml';config.write_text('scene_id: 09c1414f1b\n')
    manifest['config_sha256']=builder.e3.sha256_file(config)
    dump(directory/'input_manifest.json',manifest)
    (directory/'proposal_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pool['rows']))
    pool.update(input_manifest_sha256=builder.e3.sha256_file(directory/'input_manifest.json'),
                proposal_records_sha256=builder.e3.sha256_file(directory/'proposal_records.jsonl'))
    for name in ('view_manifest.json','views_receipt.json'):dump(directory/name,{'synthetic':name})
    pool.update(view_manifest_sha256=builder.e3.sha256_file(directory/'view_manifest.json'),
                views_receipt_sha256=builder.e3.sha256_file(directory/'views_receipt.json'))
    dump(directory/'execution_claim.json',dict(code_commit=pool['code_commit'],input_manifest_sha256=pool['input_manifest_sha256'],views_receipt_sha256=pool['views_receipt_sha256']))
    dump(directory/'runtime.json',{'job_id':'123'});dump(directory/'proposal_pool.json',pool)
    contract_path=Path(spec['freeze_root'])/'contract/freeze_manifest.json';contract=json.loads(contract_path.read_text())
    contract['resource_inventory'][0]['sha256']=manifest['config_sha256'];contract.pop('contract_sha256')
    contract['contract_sha256']=canonical_hash(contract);dump(contract_path,contract)
    audit.update(config_sha256=manifest['config_sha256'],producer_config=identity(config),rows=failure_rows(pool),counts={'unavailable':4})
    for name in CLOSURE:audit[name.replace('.','_')+'_sha256']=builder.e3.sha256_file(directory/name)
    dump(Path(spec['path']),audit);spec['sha256']=builder.e3.sha256_file(spec['path'])
    monkeypatch.setattr(rvg,'read_manifest',lambda *_:(manifest,{}))
    monkeypatch.setattr(rvg,'validate_view_receipt',lambda *_:{})
    return directory,pool,manifest,spec,audit


def test_rvg_terminal_retains_pre_generation_jobs_without_fabricated_status(rvg_terminal):
    directory,pool,manifest,spec,audit=rvg_terminal
    normalized,proof=terminal_pool_rows(directory,pool,manifest,spec)
    assert len(normalized)==4 and all(r['status']=='unavailable' for r in normalized)
    assert proof['counts']=={'unavailable':4}
    assert not (directory/'execution_status.json').exists()


@pytest.mark.parametrize('name',['execution_claim.json','view_manifest.json','views_receipt.json','runtime.json'])
def test_rvg_terminal_rejects_changed_original_closure(rvg_terminal,name):
    directory,pool,manifest,spec,audit=rvg_terminal
    (directory/name).write_text('{}')
    with pytest.raises(ValueError,match='source closure differs'):
        terminal_pool_rows(directory,pool,manifest,spec)


def test_rvg_terminal_cannot_bypass_actual_mask_validation(rvg_terminal,monkeypatch):
    from run.icra2027 import e3_rvg_generation_pilot as rvg
    directory,pool,manifest,spec,audit=rvg_terminal
    def reject(*args):raise ValueError('actual source mask bytes changed')
    monkeypatch.setattr(rvg,'validate_view_receipt',reject)
    with pytest.raises(ValueError,match='mask bytes changed'):
        terminal_pool_rows(directory,pool,manifest,spec)


def test_pre_generation_audit_cannot_relabel_partially_generated_pool(terminal):
    from run.icra2027.e3_rvg_terminal_audit import failure_rows
    from run.icra2027.e3_auto_discovery_pilot import PilotError
    with pytest.raises(PilotError,match='generated or attempted'):
        failure_rows(terminal[1])
