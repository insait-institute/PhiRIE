import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robo.eval import agentic_ablation as e3
from robo.eval import agentic_runtime_accounting as accounting
from robo.manifest.hash import canonical_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def pool_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(e3, 'checked_repo_path', lambda p, *a, **kw: Path(p))
    monkeypatch.setattr(e3, '_verify_sealed_directory', lambda *a, **kw: {})
    def make(name, tool='trellis', wall=10, exit_code=0, parent=None):
        root = tmp_path/name
        kind = 'trellis_initial' if tool == 'trellis' else 'rvg_initial'
        directory = root/kind/'scene'
        config = {'freeze_id':name, 'output_scene_id':'scene'}
        if parent:
            config['frozen_view_replay'] = {
                'source_config':parent['config'], 'source_contract':parent['contract'],
                'producer_commit':'source'}
        cp = write(root/'config.yaml', config)
        rows = [{'job_id':'scene:auto:1000','tool':tool,'status':'generation_failed' if exit_code else 'available'}]
        rp = directory/'proposal_records.jsonl';rp.parent.mkdir(parents=True);rp.write_text(json.dumps(rows[0])+'\n')
        manifest = write(directory/'input_manifest.json', {'code_commit':'source','config_sha256':cp['sha256'],'jobs':[{'job_id':r['job_id']} for r in rows]})
        contract = {'freeze_id':name,'code':{'commit':'source','dirty':False},'resource_inventory':[
            {'kind':'experiment_config','sha256':cp['sha256'],'resolved_path':cp['path']}]}
        contract['contract_sha256'] = canonical_hash(contract)
        ca = write(root/'contract/freeze_manifest.json', contract)
        anchor = write(directory/'proposal_pool.json', {'freeze_id':name,'code_commit':'source',
            'input_manifest_sha256':manifest['sha256'], 'proposal_records_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),
            'rows':rows,'planned_jobs':1,'wall_s':wall,'exit_code':exit_code,'generation_performed':True,
            'source_discovery_hashes':{'source':'fixed'}})
        return {'anchor':anchor,'config':cp,'contract':ca}
    return make


def setup(tmp_path, pools):
    out = tmp_path/'canonical'/'agentic'
    write(out/'input_inventory/inventory_audit.json', {'initial_pools':pools})
    control = {'scene':{'proposals':[{'job_id':'scene/obj_1000','tool':'trellis','wall_s':2},
        {'job_id':'scene/obj_1000','tool':'reconviagen','wall_s':3},
        {'job_id':'scene/obj_1000','tool':'registration_retry','wall_s':5}],
        'tool_failures':[{'job_id':'scene/obj_1000','tool':'registration_retry','wall_s':7}]}}
    ledger = [{'scene_id':'scene','job_id':'scene/obj_1000','policy_id':p} for p in e3.POLICY_IDS]
    return out,control,ledger


def test_process_overhead_failed_work_and_shared_policy_attribution(pool_factory,tmp_path):
    t=pool_factory('trellis');r=pool_factory('rvg','reconviagen',20,exit_code=1)
    out,c,l=setup(tmp_path,{'trellis':t['anchor'],'reconviagen':r['anchor']})
    result=accounting.account_runtime(out,{},c,l)
    assert [r['attributed_algorithm_phase_wall_s'] for r in result['rows']]==[12,35,35,47,47]
    assert len(result['generation_processes'])==2
    assert result['generation_processes'][1]['exit_code']==1
    assert result['actual_new_wave_incremental_wall_s'] is None
    assert result['cache_materialization_overhead_wall_s'] is None


def test_recovery_charges_original_failure_once(pool_factory,tmp_path):
    t=pool_factory('trellis');old=pool_factory('old','reconviagen',4,exit_code=1)
    new=pool_factory('new','reconviagen',20,parent=old)
    out,c,l=setup(tmp_path,{'trellis':t['anchor'],'reconviagen':new['anchor']})
    jobs={'runtime_accounting':{'schema_version':1,'process_histories':[
        {'scene_id':'scene','tool':'reconviagen','pools':[old['anchor'],new['anchor']]}]}}
    result=accounting.account_runtime(out,jobs,c,l)
    assert result['rows'][1]['attributed_generation_process_wall_s']==34
    assert len(result['generation_processes'])==3
    assert result['missing_evidence']==[]


def test_missing_recovery_ancestry_is_unknown_not_zero(pool_factory,tmp_path):
    t=pool_factory('trellis');old=pool_factory('old','reconviagen',4,exit_code=1)
    new=pool_factory('new','reconviagen',20,parent=old)
    out,c,l=setup(tmp_path,{'trellis':t['anchor'],'reconviagen':new['anchor']})
    result=accounting.account_runtime(out,{},c,l)
    assert result['rows'][0]['runtime_accounting_status']=='PASS'
    assert result['rows'][1]['attributed_algorithm_phase_wall_s'] is None
    assert result['rows'][1]['runtime_accounting_status']=='NOT_RUN'
    assert result['missing_evidence'][0]['reason']=='unanchored_recovery_process_history'


@pytest.mark.parametrize('wall',[None,-1,float('nan'),float('inf'),True,0])
def test_missing_invalid_timing_cannot_become_zero(pool_factory,tmp_path,wall):
    t=pool_factory('trellis',wall=wall);r=pool_factory('rvg','reconviagen')
    out,c,l=setup(tmp_path,{'trellis':t['anchor'],'reconviagen':r['anchor']})
    result=accounting.account_runtime(out,{},c,l)
    assert all(row['attributed_algorithm_phase_wall_s'] is None for row in result['rows'])
    assert result['missing_evidence']


def test_pool_tamper_rejected(pool_factory,tmp_path):
    t=pool_factory('trellis');r=pool_factory('rvg','reconviagen');p=Path(t['anchor']['path'])
    pool=json.loads(p.read_text());pool['wall_s']=1;p.write_text(json.dumps(pool))
    out,c,l=setup(tmp_path,{'trellis':t['anchor'],'reconviagen':r['anchor']})
    with pytest.raises(ValueError,match='content changed'):accounting.account_runtime(out,{},c,l)


@pytest.mark.parametrize('change',['duplicate','unrelated','wrong_hash','duplicate_history'])
def test_ambiguous_or_double_counted_history_rejected(pool_factory,tmp_path,change):
    t=pool_factory('trellis');r=pool_factory('rvg','reconviagen');extra=pool_factory('extra','reconviagen')
    out,c,l=setup(tmp_path,{'trellis':t['anchor'],'reconviagen':r['anchor']})
    anchors=[r['anchor']]
    if change=='duplicate':anchors*=2
    if change=='unrelated':anchors.append(extra['anchor'])
    if change=='wrong_hash':anchors=[dict(r['anchor'],sha256='0'*64)]
    histories=[{'scene_id':'scene','tool':'reconviagen','pools':anchors}]
    if change=='duplicate_history':histories*=2
    with pytest.raises(ValueError):accounting.account_runtime(out,{'runtime_accounting':{'schema_version':1,'process_histories':histories}},c,l)


def test_source_config_tamper_rejected(pool_factory,tmp_path):
    t=pool_factory('trellis');r=pool_factory('rvg','reconviagen')
    Path(t['config']['path']).write_text('{}')
    out,c,l=setup(tmp_path,{'trellis':t['anchor'],'reconviagen':r['anchor']})
    with pytest.raises(ValueError,match='content changed'):accounting.account_runtime(out,{},c,l)


def test_missing_failed_retry_timing_marks_only_retry_policies_unknown(pool_factory,tmp_path):
    t=pool_factory('trellis');r=pool_factory('rvg','reconviagen')
    out,c,l=setup(tmp_path,{'trellis':t['anchor'],'reconviagen':r['anchor']})
    del c['scene']['tool_failures'][0]['wall_s']
    result=accounting.account_runtime(out,{},c,l)
    assert [r['runtime_accounting_status'] for r in result['rows']]==['PASS']*3+['NOT_RUN']*2
    assert result['rows'][3]['canonical_proposal_and_retry_wall_s'] is None


def test_selected_producer_anchor_conflict_rejected(pool_factory,tmp_path):
    t=pool_factory('trellis');r=pool_factory('rvg','reconviagen')
    out,c,l=setup(tmp_path,{'trellis':dict(t['anchor'],code_commit='forged'),'reconviagen':r['anchor']})
    with pytest.raises(ValueError,match='producer identity'):accounting.account_runtime(out,{},c,l)


def test_resealed_pool_with_different_input_roster_rejected(pool_factory,tmp_path):
    t=pool_factory('trellis');r=pool_factory('rvg','reconviagen');p=Path(t['anchor']['path'])
    pool=json.loads(p.read_text());pool['rows'][0]['job_id']='scene:auto:1001'
    rp=p.parent/'proposal_records.jsonl';rp.write_text(json.dumps(pool['rows'][0])+'\n')
    pool['proposal_records_sha256']=hashlib.sha256(rp.read_bytes()).hexdigest();t['anchor']=write(p,pool)
    out,c,l=setup(tmp_path,{'trellis':t['anchor'],'reconviagen':r['anchor']})
    with pytest.raises(ValueError,match='records differ'):accounting.account_runtime(out,{},c,l)
