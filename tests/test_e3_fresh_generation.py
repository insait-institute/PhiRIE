import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from run.icra2027 import e3_fresh_generation_contract as contract
from run.icra2027 import e3_trellis_generation_pilot as runner
from run.icra2027.e3_auto_discovery_pilot import identity, sha, PilotError
from robo.manifest.hash import canonical_hash


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def e0(path, commit, config, resource):
    value={'code':{'commit':commit,'dirty':False},'freeze_id':path.parent.parent.name,
           'resource_inventory':[{'id':resource,'sha256':sha(config)}]}
    value['contract_sha256']=canonical_hash(value)
    dump(path,value)
    return identity(path)


def fresh_source(tmp_path, monkeypatch, count=3):
    source=tmp_path/'new/auto_discovery_pilot';source.mkdir(parents=True)
    proof={'status':contract.FRESH,'independent_heldout_evaluation':False}
    cfg=tmp_path/'discovery.yaml'
    cfg.write_text(yaml.safe_dump({'freeze_id':'new','scene_id':'scene'}))
    manifest={'source_gaussian_training_provenance':contract.FRESH,'config_sha256':sha(cfg),
              'code_commit':'fresh','gaussian_provenance':proof,'boundary':{},'input_images':{},'metadata':{}}
    summary={'source_gaussian_training_provenance':contract.FRESH,'code_commit':'fresh',
             'freeze_id':'new','scene_id':'scene','gaussian_provenance':proof}
    for name,value in [('input_manifest.json',manifest),('pilot_summary.json',summary),
                       ('output_hashes.json',{}),('postrun_audit.json',{}),('all_jobs_manifest.json',{})]:
        dump(source/name,value)
    c={'source_pilot':str(source),'source_gaussian_training_provenance':contract.FRESH,
       'source_summary_sha256':sha(source/'pilot_summary.json'),'source_output_hashes_sha256':sha(source/'output_hashes.json'),
       'source_discovery_hashes':{name:sha(source/name) for name in contract.DISCOVERY_FILES},
       'source_discovery_config':identity(cfg),'source_discovery_commit':'fresh',
       'source_discovery_contract':e0(source.parent/'contract/freeze_manifest.json','fresh',cfg,'e3_auto_pilot_config')}
    monkeypatch.setattr(contract,'validate_gaussian_provenance',lambda *a:copy.deepcopy(proof))
    jobs=[{'job_id':f'scene:auto:{1000+i}','automatic_instance_id':1000+i,'prepared':i<count-1,
           'output_index':i,'frame':'train.jpg'} for i in range(count)]
    for job in jobs:
        if job['prepared']:
            crop=source/f'crop-{job["output_index"]}.png';crop.write_bytes(f'crop{job["output_index"]}'.encode())
            job['input']=identity(crop);job['object_metadata']={'index':job['output_index']}
    loader=lambda _p:copy.deepcopy(jobs)
    return c,jobs,loader


@pytest.mark.parametrize('count',[0,3,7])
def test_fresh_source_proof_preserves_dynamic_denominator(tmp_path,monkeypatch,count):
    c,jobs,loader=fresh_source(tmp_path,monkeypatch,count)
    actual,binding=contract.discovery_binding(c,loader)
    assert actual==jobs and len(actual)==count
    assert binding['source_discovery_hashes']==c['source_discovery_hashes']
    assert binding['source_gaussian_training_provenance']==contract.FRESH


@pytest.mark.parametrize('change',['missing_hash','extra_hash','input','status','proof','source','config','e0'])
def test_fresh_source_negative_drift(tmp_path,monkeypatch,change):
    c,jobs,loader=fresh_source(tmp_path,monkeypatch)
    source=Path(c['source_pilot'])
    if change=='missing_hash':c['source_discovery_hashes'].pop('all_jobs_manifest.json')
    elif change=='extra_hash':c['source_discovery_hashes']['anything']='0'*64
    elif change=='input':dump(source/'input_manifest.json',{})
    elif change=='status':c['source_gaussian_training_provenance']='UNKNOWN'
    elif change=='proof':monkeypatch.setattr(contract,'validate_gaussian_provenance',lambda *a:{'status':'UNKNOWN'})
    elif change=='source':c['source_discovery_commit']='other'
    elif change=='config':Path(c['source_discovery_config']['path']).write_text('edited')
    elif change=='e0':Path(c['source_discovery_contract']['path']).write_text('{}')
    with pytest.raises((PilotError,KeyError)):
        contract.discovery_binding(c,loader)


def reuse_fixture(tmp_path):
    old=tmp_path/'old/trellis_initial';old.mkdir(parents=True)
    new_crop=tmp_path/'new.png';new_crop.write_bytes(b'crop')
    old_crop=old/'input.png';old_crop.write_bytes(b'crop')
    old_job={'job_id':'old:auto:1001','proposal_id':'old-proposal','prepared':True,'output_index':0,'input':identity(old_crop)}
    artifacts={}
    for name in contract.RAW_FILES:
        path=old/'construction/objects/obj_00'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
        artifacts[name]=identity(path)
    old_row={'job_id':old_job['job_id'],'proposal_id':'old-proposal','status':'available','tool':'trellis','seed':42,
             'input_sha256':sha(old_crop),'artifacts':artifacts,
             'runtime':{'seed':42,'status':'generated','model_source':'checkpoint','wall_s':2.0}}
    prior={'jobs':[old_job],'pool':{'rows':[old_row],'source_gaussian_training_provenance':'UNKNOWN'},
           'spec':{'directory':str(old),'producer_commit':'original'},'algorithm_files':{'source':'digest'},'runtime':{'name':'old-GPU'},'reuse_eligible':True}
    jobs=[{'job_id':'new:auto:1017','proposal_id':'new-proposal','automatic_instance_id':1017,
           'prepared':True,'input':identity(new_crop),'frame':'new-train.jpg'},
          {'job_id':'new:auto:1018','proposal_id':'missing-proposal','automatic_instance_id':1018,'prepared':False}]
    c={'models':{'trellis_snapshot':{'path':'checkpoint'}},'runtime_sha256':'runtime','crop_comparison_sha256':'comparison'}
    m={'jobs':jobs,'source_discovery_hashes':{name:'digest' for name in contract.DISCOVERY_FILES}}
    comparison={'rows':[{'job_id':jobs[0]['job_id'],'exact_matching_prior_jobs':[old_job['job_id']]},
                        {'job_id':jobs[1]['job_id'],'exact_matching_prior_jobs':[]}]}
    return c,m,prior,comparison


def test_raw_reuse_across_ids_preserves_original_generator_and_missing(tmp_path):
    c,m,prior,comparison=reuse_fixture(tmp_path)
    rows,receipts=contract.reuse_rows(c,m,prior,comparison)
    assert len(rows)==2 and len(receipts)==1
    assert rows[0]['job_id']=='new:auto:1017' and rows[0]['original_generator_commit']=='original'
    assert rows[1]['reason']=='preparation_unavailable'
    assert receipts[0]['original_gaussian_training_provenance']=='UNKNOWN'
    assert receipts[0]['source_gaussian_training_provenance']==contract.FRESH
    assert receipts[0]['generation_performed'] is False
    assert rows[0]['artifacts']==prior['pool']['rows'][0]['artifacts']


@pytest.mark.parametrize('change',['new_crop','old_crop','raw_output','raw_missing','seed','tool','checkpoint','comparison'])
def test_reuse_drift_never_creates_receipt(tmp_path,change):
    c,m,prior,comparison=reuse_fixture(tmp_path)
    row=prior['pool']['rows'][0]
    if change=='new_crop':Path(m['jobs'][0]['input']['path']).write_bytes(b'changed')
    elif change=='old_crop':Path(prior['jobs'][0]['input']['path']).write_bytes(b'changed')
    elif change=='raw_output':Path(row['artifacts']['trellis_mesh.ply']['path']).write_bytes(b'changed')
    elif change=='raw_missing':row['artifacts'].pop('trellis_gs.ply')
    elif change=='seed':row['runtime']['seed']=9
    elif change=='tool':row['tool']='reconviagen'
    elif change=='checkpoint':row['runtime']['model_source']='other'
    elif change=='comparison':comparison['rows'][0]['exact_matching_prior_jobs']=[]
    with pytest.raises(PilotError):contract.reuse_rows(c,m,prior,comparison)


def test_unmatched_prepared_is_not_run_not_fabricated_failure(tmp_path):
    c,m,prior,comparison=reuse_fixture(tmp_path)
    path=Path(m['jobs'][0]['input']['path']);path.write_bytes(b'new crop')
    m['jobs'][0]['input']=identity(path);comparison['rows'][0]['exact_matching_prior_jobs']=[]
    rows,receipts=contract.reuse_rows(c,m,prior,comparison)
    assert not receipts and rows[0]['reason']=='initial_tool_not_run'


def test_version_only_runtime_never_qualifies_even_exact_crop(tmp_path):
    c,m,prior,comparison=reuse_fixture(tmp_path)
    eligible,reason=contract.runtime_reuse_eligibility(c,{}, {})
    assert not eligible and reason=='original_independently_anchored_runtime_bytes_unavailable'
    prior['reuse_eligible']=eligible
    rows,receipts=contract.reuse_rows(c,m,prior,comparison)
    assert not receipts and rows[0]['reason']=='initial_tool_not_run'


def test_self_declared_complete_runtime_is_not_evidence(tmp_path):
    executable=tmp_path/'python';executable.write_bytes(b'python-runtime')
    proof=tmp_path/'runtime.json'
    dump(proof,{'scope':'complete_raw_producer_runtime','files':[identity(executable)]})
    config={'python':str(executable),'runtime_byte_identity':identity(proof)}
    old=copy.deepcopy(config);inputs={'runtime_byte_identity':copy.deepcopy(config['runtime_byte_identity'])}
    eligible,reason=contract.runtime_reuse_eligibility(config,old,inputs)
    assert not eligible and reason=='unsupported_original_runtime_byte_manifest_producer_or_schema'


def test_current_algorithm_change_ineligible_but_old_anchor_drift_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(contract,'CODE',tmp_path)
    expected={}
    for name in contract.ALGORITHM_FILES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'code');expected[name]=sha(path)
    c={'raw_reuse':{'algorithm_files':expected}}
    monkeypatch.setattr(contract.subprocess,'check_output',lambda *a,**k:b'code')
    assert contract.algorithm_identity(c,'old')['current_matches'] is True
    (tmp_path/'models/s4_trellis.py').write_bytes(b'changed')
    assert contract.algorithm_identity(c,'old')['current_matches'] is False
    c['raw_reuse']['algorithm_files']['models/s4_trellis.py']='0'*64
    with pytest.raises(PilotError,match='historical'):contract.algorithm_identity(c,'old')


def prior_source_fixture(tmp_path,monkeypatch):
    root=tmp_path/'old/trellis_initial';root.mkdir(parents=True)
    source=tmp_path/'discovery';source.mkdir()
    for name in ('pilot_summary.json','output_hashes.json'):dump(source/name,{})
    runtime={'python':'synthetic','packages':[['torch','synthetic']]}
    digest=hashlib.sha256(json.dumps(runtime,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    config={'freeze_id':'old','models':{'trellis_snapshot':{'path':'checkpoint'}},'seed':42,
            'python':'python','runtime_sha256':digest,'source_pilot':str(source),
            'source_summary_sha256':sha(source/'pilot_summary.json'),
            'source_output_hashes_sha256':sha(source/'output_hashes.json')}
    cfg=tmp_path/'old.yaml';cfg.write_text(yaml.safe_dump(config))
    jobs=[{'job_id':'scene:auto:1000','prepared':False}]
    inputs={'code_commit':'original','paper_ready':False,'config_sha256':sha(cfg),'seed':42,
            'jobs':[dict(jobs[0],proposal_id='old:scene:auto:1000:trellis:initial:seed42')]}
    dump(root/'input_manifest.json',inputs)
    pool={'code_commit':'original','freeze_id':'old','exit_code':0,'paper_ready':False,
          'input_manifest_sha256':sha(root/'input_manifest.json'),'planned_jobs':1,'rows':[{'job_id':jobs[0]['job_id']}]}
    dump(root/'proposal_pool.json',pool)
    audit={'state':'COMPLETED','exit_code':'0:0','producer_source_commit':'original',
           'input_manifest_sha256':sha(root/'input_manifest.json'),'proposal_pool_sha256':sha(root/'proposal_pool.json'),
           'config_sha256':sha(cfg)}
    dump(root/'postrun_audit.json',audit);dump(root/'python_runtime.json',dict(runtime,runtime_sha256=digest))
    dump(root/'runtime.json',{'name':'original-gpu'})
    current=copy.deepcopy(config)
    current['raw_reuse']={'directory':str(root),'producer_commit':'original','config':identity(cfg),
       'contract':e0(root.parent/'contract/freeze_manifest.json','original',cfg,'e3_trellis_config'),
       'anchors':{name:identity(root/name) for name in contract.PRIOR_FILES}}
    monkeypatch.setattr(contract,'algorithm_identity',lambda *_: {'files':{'source':'fixed'},'current_matches':True})
    return current,lambda _:copy.deepcopy(jobs)


def test_original_pool_authentication_preserves_runtime_ineligibility(tmp_path,monkeypatch):
    config,loader=prior_source_fixture(tmp_path,monkeypatch)
    result=contract.prior_trellis_source(config,loader)
    assert result['reuse_eligible'] is False


@pytest.mark.parametrize('change',['seed','commit','anchor','prior_input'])
def test_original_pool_recipe_or_receipt_drift_rejected(tmp_path,monkeypatch,change):
    config,loader=prior_source_fixture(tmp_path,monkeypatch)
    if change=='seed':config['seed']=7
    elif change=='commit':config['raw_reuse']['producer_commit']='other'
    elif change=='anchor':config['raw_reuse']['anchors']['proposal_pool.json']['sha256']='0'*64
    elif change=='prior_input':loader=lambda _:[{'job_id':'different','prepared':False}]
    with pytest.raises(PilotError):contract.prior_trellis_source(config,loader)


@pytest.mark.parametrize('key,value',[('models',{'different':'model'}),('runtime_sha256','changed')])
def test_declared_current_recipe_difference_is_reuse_ineligible(tmp_path,monkeypatch,key,value):
    config,loader=prior_source_fixture(tmp_path,monkeypatch)
    config[key]=value
    result=contract.prior_trellis_source(config,loader)
    assert not result['reuse_eligible']
    assert f'current_{key}_differs_from_original' in result['reuse_ineligible_reasons']


def test_legitimate_common_helper_change_keeps_normal_generation_action(tmp_path,monkeypatch):
    algorithm_check=contract.algorithm_identity
    config,loader=prior_source_fixture(tmp_path,monkeypatch)
    code=tmp_path/'current';expected={}
    for name in contract.ALGORITHM_FILES:
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'original');expected[name]=sha(path)
    config['raw_reuse']['algorithm_files']=expected
    (code/'agents/core/common.py').write_bytes(b'original plus export helper')
    monkeypatch.setattr(contract,'CODE',code)
    monkeypatch.setattr(contract,'historical_algorithm_identity',lambda _:dict(expected))
    monkeypatch.setattr(contract,'algorithm_identity',algorithm_check)
    monkeypatch.setattr(runner,'source_jobs',loader)
    job={'job_id':'new:auto:1000','automatic_instance_id':1000,'prepared':True,'input':{'sha256':'new'}}
    monkeypatch.setattr(contract,'discovery_binding',lambda *_:([copy.deepcopy(job)],
        {'source_gaussian_training_provenance':contract.FRESH,'source_discovery_hashes':{}}))
    monkeypatch.setattr(contract,'validate_crop_comparison',lambda *_:{'rows':[{'job_id':job['job_id'],'exact_matching_prior_jobs':[]}]})
    rows,binding,reused,receipts=runner.planned_jobs(config,tmp_path/'fresh/trellis_initial')
    assert rows[0]['generation_action']=='generate' and not receipts
    assert 'current_raw_algorithm_differs_from_original' in binding['raw_reuse_eligibility']['reasons']
    config['raw_reuse']['anchors']['proposal_pool.json']['sha256']='0'*64
    with pytest.raises(PilotError,match='source anchor changed'):
        runner.planned_jobs(config,tmp_path/'fresh/trellis_initial')


def test_staging_excludes_reused_jobs_but_combined_records_keep_all(tmp_path):
    c,m,prior,comparison=reuse_fixture(tmp_path)
    rows,receipts=contract.reuse_rows(c,m,prior,comparison)
    m['jobs'][0]['generation_action']='reuse';m['jobs'][1]['generation_action']='unavailable'
    m['reused_rows']=[rows[0]]
    path=tmp_path/'receipt.json';dump(path,receipts[0]);m['raw_reuse_receipts']=[identity(path)]
    dump(tmp_path/'construction/objects/objects.json',[])
    runner.validate_staged(tmp_path,m)
    combined=runner.collect_records(tmp_path,m,0)
    assert len(combined)==2 and combined[0]['raw_reuse_receipt']==identity(path)
    assert combined[1]['reason']=='preparation_unavailable'
    assert not any(runner.needs_generation(j) for j in m['jobs'])


def test_plan_stages_only_unmatched_and_freezes_reuse_receipt(tmp_path,monkeypatch):
    c,m,prior,comparison=reuse_fixture(tmp_path)
    crop=tmp_path/'unmatched.png';crop.write_bytes(b'new-object')
    unmatched={'job_id':'new:auto:1019','automatic_instance_id':1019,'prepared':True,'output_index':1,
               'input':identity(crop),'frame':'train.jpg','object_metadata':{'index':1}}
    m['jobs'].append(unmatched)
    comparison['rows'].append({'job_id':unmatched['job_id'],'exact_matching_prior_jobs':[]})
    c.update(raw_reuse={'declared':True},python='python',source_pilot=str(tmp_path),seed=42)
    dump(tmp_path/'pilot_summary.json',{})
    monkeypatch.setattr(contract,'discovery_binding',lambda *_:(copy.deepcopy(m['jobs']),
        {'source_gaussian_training_provenance':contract.FRESH,'source_discovery_hashes':m['source_discovery_hashes']}))
    monkeypatch.setattr(contract,'prior_trellis_source',lambda *_:dict(prior,reuse_ineligible_reason=None))
    monkeypatch.setattr(contract,'validate_crop_comparison',lambda *_:comparison)
    cfg=tmp_path/'config.json';dump(cfg,c)
    monkeypatch.setattr(runner,'context',lambda *_:(c,'source'))
    monkeypatch.setattr(runner,'import_smoke',lambda _: {})
    monkeypatch.setattr(runner,'validate_models',lambda _: {})
    monkeypatch.setattr(runner,'runtime_identity',lambda _: ({},'runtime'))
    monkeypatch.setattr(runner,'targeted_runtime_identity',lambda _: {'scope':'targeted','files':[]})
    root=tmp_path/'freeze';root.mkdir()
    runner.plan(cfg,root)
    out=root/'trellis_initial';planned=json.loads((out/'input_manifest.json').read_text())
    assert [j['generation_action'] for j in planned['jobs']]==['reuse','unavailable','generate']
    assert json.loads((out/'construction/objects/objects.json').read_text())==[{'index':1}]
    assert len(planned['raw_reuse_receipts'])==1 and len(planned['jobs'])==3
    runner.validate_source_binding(c,out,planned)
    runner.validate_staged(out,planned)
    changed=copy.deepcopy(planned);changed['jobs'][0]['generation_action']='generate'
    with pytest.raises(PilotError,match='config-bound source'):runner.validate_source_binding(c,out,changed)


def test_execution_claim_blocks_second_publication_without_gpu(tmp_path,monkeypatch):
    cfg=tmp_path/'config.json';dump(cfg,{'models':{}})
    out=tmp_path/'trellis_initial';out.mkdir()
    dump(out/'input_manifest.json',{'code_commit':'source','config_sha256':sha(cfg),'jobs':[]})
    dump(out/'model_stats.json',{})
    monkeypatch.setattr(runner,'context',lambda *_:({'models':{}},'source'))
    monkeypatch.setattr(runner,'validate_source_binding',lambda *_:None)
    monkeypatch.setattr(runner,'validate_staged',lambda *_:None)
    calls=[];monkeypatch.setattr(runner,'publish_pool',lambda *a,**k:calls.append(k))
    runner.execute(cfg,tmp_path)
    assert calls==[{'generation_performed':False}]
    with pytest.raises(FileExistsError):runner.execute(cfg,tmp_path)
    assert len(calls)==1


def test_partial_failure_records_missing_units_without_overwrite(tmp_path,monkeypatch):
    cfg=tmp_path/'config.json';dump(cfg,{'models':{}})
    out=tmp_path/'trellis_initial';out.mkdir()
    jobs=[{'job_id':f'j{i}','automatic_instance_id':i,'proposal_id':f'p{i}','prepared':i<2,
           'output_index':i,'input':{'sha256':'input'},'frame':'train.jpg'} for i in range(3)]
    dump(out/'input_manifest.json',{'code_commit':'source','config_sha256':sha(cfg),'jobs':jobs})
    monkeypatch.setattr(runner,'context',lambda *_:({'models':{}},'source'))
    monkeypatch.setattr(runner,'validate_source_binding',lambda *_:None)
    monkeypatch.setattr(runner,'validate_staged',lambda *_:None)
    def fail(*_):
        for name in runner.OUTPUTS:
            path=out/'construction/objects/obj_00'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'output')
        dump(out/'producer_records/object_00.json',{'seed':42,'status':'generated'})
        raise RuntimeError('synthetic partial producer failure')
    monkeypatch.setattr(runner,'_execute_claimed',fail)
    with pytest.raises(RuntimeError):runner.execute(cfg,tmp_path)
    status=json.loads((out/'execution_status.json').read_text())
    assert status['status']=='FAIL' and status['planned_jobs']==3
    assert status['available_job_ids']==['j0'] and status['missing_prepared_job_ids']==['j1']
    assert status['in_place_resume_supported'] is False and not (out/'proposal_pool.json').exists()
    before=sha(out/'execution_status.json')
    with pytest.raises(FileExistsError):runner.execute(cfg,tmp_path)
    assert sha(out/'execution_status.json')==before


@pytest.mark.parametrize('producer_exit', [0, 7])
def test_generation_branch_passes_frozen_paths_to_worker(tmp_path,monkeypatch,producer_exit):
    """Exercise claimed execution through worker dispatch, without a GPU/model."""
    cfg=tmp_path/'frozen config.yaml';cfg.write_text('{}')
    out=tmp_path/'trellis_initial';out.mkdir();dump(out/'model_stats.json',{})
    c={'models':{name:{'path':str(tmp_path/name)} for name in
                 ('trellis_source','trellis_snapshot','dinov2_source')},'python':'frozen-python'}
    monkeypatch.setenv('SLURM_JOB_ID','test-job');monkeypatch.setenv('SLURMD_NODENAME','hala')
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**k:'{}')
    monkeypatch.setattr(runner,'validate_gpu_memory',lambda _:None)
    commands=[]
    def call(command,**kwargs):
        commands.append((command,kwargs));return producer_exit
    monkeypatch.setattr(runner.subprocess,'call',call)
    published=[];monkeypatch.setattr(runner,'publish_pool',lambda *a,**k:published.append((a,k)))
    if producer_exit:
        with pytest.raises(PilotError,match='generation failed'):
            runner._execute_claimed(c,'source',out,{'jobs':[{'prepared':True}]},cfg,tmp_path)
    else:
        runner._execute_claimed(c,'source',out,{'jobs':[{'prepared':True}]},cfg,tmp_path)
    command,kwargs=commands[0]
    assert command[command.index('--config')+1]==str(cfg.resolve())
    assert command[command.index('--freeze-root')+1]==str(tmp_path.resolve())
    assert command[-2:]==['--phase','worker'] and kwargs['cwd']==runner.CODE
    assert kwargs['env']['PYTHONPATH']==str(runner.CODE)
    assert published[0][0][4]==producer_exit
