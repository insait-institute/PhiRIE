import copy
import json
from pathlib import Path
import pytest
import yaml

from run.icra2027 import e3_fresh_canonical_config as builder
from run.icra2027 import e3_fresh_canonical_phase as launcher
from robo.manifest.hash import canonical_hash


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_waiting_pool_never_writes_fake_config(tmp_path):
    result=builder.prepare(tmp_path/'discovery',tmp_path/'trellis',tmp_path/'rvg',
                           freeze_id='future',config_directory=tmp_path/'configs')
    assert result['status']=='WAITING_REAL_INITIAL_POOLS'
    assert result['config_written'] is False and not (tmp_path/'configs').exists()


@pytest.fixture
def pool(tmp_path,monkeypatch):
    monkeypatch.setattr(builder.e3,'REPOSITORY_ROOT',tmp_path)
    directory=tmp_path/'freeze/trellis_initial';directory.mkdir(parents=True)
    hashes={name:'d'*64 for name in builder.DISCOVERY_FILES}
    jobs=[{'job_id':'scene:auto:1000','prepared':False},
          {'job_id':'scene:auto:1001','prepared':True}]
    rows=[dict(j,status='unavailable' if not j['prepared'] else 'available',
               reason='preparation_unavailable' if not j['prepared'] else None) for j in jobs]
    manifest=dict(code_commit='a'*40,config_sha256='b'*64,
        source_gaussian_training_provenance=builder.FRESH,source_discovery_hashes=hashes)
    dump(directory/'input_manifest.json',manifest)
    (directory/'proposal_records.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    value=dict(code_commit='a'*40,freeze_id='freeze',exit_code=0,paper_ready=False,
        input_manifest_sha256=builder.e3.sha256_file(directory/'input_manifest.json'),
        proposal_records_sha256=builder.e3.sha256_file(directory/'proposal_records.jsonl'),
        source_gaussian_training_provenance=builder.FRESH,source_discovery_hashes=hashes,
        planned_jobs=2,rows=rows)
    dump(directory/'proposal_pool.json',value)
    audit=dict(state='COMPLETED',exit_code='0:0',producer_source_commit='a'*40,
        proposal_pool_sha256=builder.e3.sha256_file(directory/'proposal_pool.json'),
        input_manifest_sha256=value['input_manifest_sha256'])
    dump(directory/'postrun_audit.json',audit)
    contract=dict(freeze_id='freeze',code={'commit':'a'*40,'dirty':False},
        resource_inventory=[{'kind':'experiment_config','sha256':'b'*64}])
    contract['contract_sha256']=canonical_hash(contract)
    dump(directory.parent/'contract/freeze_manifest.json',contract)
    return directory,hashes,jobs


def test_completed_pool_preserves_missing_job(pool):
    directory,hashes,jobs=pool
    result=builder.authenticate_pool(directory,hashes,jobs)
    assert result['sha256']==builder.e3.sha256_file(directory/'proposal_pool.json')


@pytest.mark.parametrize('row',['42|COMPLETED|0:0|140','43|COMPLETED|0:0|140',
                                '42|FAILED|1:0|5'])
def test_rvg_embedded_scheduler_receipt_binds_actual_job(pool,row):
    directory,hashes,jobs=pool
    path=directory/'postrun_audit.json';audit=json.loads(path.read_text())
    audit.pop('state');audit.pop('exit_code')
    audit['source_commit']=audit.pop('producer_source_commit')
    audit.update(job_id='42',sacct='JobID|State|ExitCode|Elapsed\n'+row+'\n')
    dump(path,audit)
    if row.startswith('42|COMPLETED|0:0'):
        builder.authenticate_pool(directory,hashes,jobs)
    else:
        with pytest.raises(ValueError):builder.authenticate_pool(directory,hashes,jobs)


@pytest.mark.parametrize('change',['dropped_job','unknown_source','different_views','unattempted','failed_job','bad_e0','changed_records'])
def test_resealed_or_partial_pool_fails_closed(pool,change):
    directory,hashes,jobs=pool
    p=json.loads((directory/'proposal_pool.json').read_text())
    if change=='dropped_job':p['rows'].pop()
    elif change=='unknown_source':p['source_gaussian_training_provenance']='UNKNOWN'
    elif change=='different_views':p['source_discovery_hashes']['all_jobs_manifest.json']='e'*64
    elif change=='unattempted':p['rows'][1]['reason']='initial_tool_not_run'
    elif change=='failed_job':p['exit_code']=1
    elif change=='bad_e0':
        cpath=directory.parent/'contract/freeze_manifest.json'
        c=json.loads(cpath.read_text());c['contract_sha256']='0'*64;dump(cpath,c)
    else:(directory/'proposal_records.jsonl').write_text('{}\n')
    dump(directory/'proposal_pool.json',p)
    a=json.loads((directory/'postrun_audit.json').read_text())
    a['proposal_pool_sha256']=builder.e3.sha256_file(directory/'proposal_pool.json')
    dump(directory/'postrun_audit.json',a)
    with pytest.raises(ValueError):builder.authenticate_pool(directory,hashes,jobs)


@pytest.mark.parametrize('phase',['observe','control'])
def test_phase_guard_forbids_gt_and_original_rgb(phase):
    guard=launcher.guard_for(phase,{'scene_id':'09c1414f1b'})
    for path in ['/data/ScanNetpp/data/09c1414f1b/scans/mesh.ply',
                 '/data/ScanNetpp/data/09c1414f1b/dslr/resized_undistorted_images/a.JPG']:
        with pytest.raises(ValueError):guard('open',(path,'r',0))
    with pytest.raises(ValueError):guard('socket.connect',())
    guard('open',('${SIMANY_ROOT:-$PWD}/outputs/raw_mesh.ply','r',0))


def test_control_forbids_fresh_source_gaussian_but_observe_can_read_it():
    path='${SIMANY_ROOT:-$PWD}/outputs/fresh/scene.ply'
    cfg={'scene_id':'09c1414f1b'}
    with pytest.raises(ValueError):launcher.guard_for('control',cfg,[path])('open',(path,'r',0))
    launcher.guard_for('observe',cfg,[path])('open',(path,'r',0))


@pytest.fixture
def phase_config(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'CODE',tmp_path)
    monkeypatch.setattr(launcher.e3,'CODE_ROOT',tmp_path)
    monkeypatch.setattr(launcher.e3,'REPOSITORY_ROOT',tmp_path)
    (tmp_path/'policies.yaml').write_text('frozen')
    (tmp_path/'runtime.yaml').write_text('runtime')
    root=tmp_path/'freeze';root.mkdir()
    config=dict(schema_version=1,scope='fresh_canonical_engineering',paper_ready=False,
        freeze_id='freeze',planned_jobs=1,planned_policy_object_rows=5,object_slots=['obj_1000'],
        scene_id='09c1414f1b',jobs_config='must_not_read_jobs.yaml',policies_config='policies.yaml',
        policies_sha256=launcher.e3.sha256_file(tmp_path/'policies.yaml'),
        observation_runtime_config='runtime.yaml',observation_runtime_config_sha256=launcher.e3.sha256_file(tmp_path/'runtime.yaml'))
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    calls=[]
    monkeypatch.setattr(launcher.e3,'_validate_cli_execution',lambda *a,**kw:calls.append(kw) or {'code':{'commit':'x'}})
    jobs={'counts':{'scenes':1,'jobs':1,'policy_object_rows':5},
          'scenes':[{'scene_id':'09c1414f1b','jobs':[{'object_slot':'obj_1000'}]}]}
    monkeypatch.setattr(launcher.e3,'_load_inventory',lambda *a,**kw:(jobs,{}))
    return path,root,config,calls,jobs


def test_observe_control_authenticate_executor_without_reopening_jobs(phase_config):
    path,root,config,calls,jobs=phase_config
    for phase in ('observe','control'):
        launcher.validate(path,root,phase)
        assert calls[-1]['require_inventory'] is True
        assert config['jobs_config'] not in calls[-1]['config_paths']
    assert not (path.parent/config['jobs_config']).exists()


@pytest.mark.parametrize('change',['foreign_code','population','scene','policy'])
def test_phase_rejects_source_or_population_drift(phase_config,monkeypatch,change):
    path,root,config,calls,jobs=phase_config
    if change=='foreign_code':monkeypatch.setattr(launcher.e3,'CODE_ROOT',Path('/foreign'))
    elif change=='population':jobs['counts']['jobs']=0
    elif change=='scene':jobs['scenes'][0]['scene_id']='abcdef1234'
    else:(path.parent/'policies.yaml').write_text('tuned')
    with pytest.raises(ValueError):launcher.validate(path,root,'control')
