"""Missing preparation must remain a real planned job through canonical E3."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from robo.eval import agentic_ablation as e3


def inventory():
    scene_id='09c1414f1b'
    job={'freeze_id':'freeze','job_id':scene_id+'/obj_1000','scene_id':scene_id,
         'object_slot':'obj_1000','artifact_paths':{},'artifact_hashes':{},
         'construction_evidence':{'observation_status':'unavailable',
          'reason_code':'preparation_unavailable','source_frame':None,'bbox_px':None,
          'mask_provenance':None}}
    identity={'path':'/not-read','size_bytes':1,'sha256':'a'*64}
    jobs={'schema_version':2,'freeze_id':'freeze','study_scope':'automatic_training_only_engineering',
          'counts':{'scenes':1,'jobs':1,'policy_object_rows':5},
          'source_contract':{'jobs_sha256':'b'*64,'jobs_hash_method':'canonical_structured_sha256',
            'contract_sha256':'c'*64,'contract_hash_method':'canonical_json_sha256',
            'contract_freeze_id':'freeze','code_commit':'d'*40},
          'observation_protocol':dict(e3.OBSERVATION_PROTOCOL),
          'scenes':[{'scene_id':scene_id,'source_scene_gaussian':identity,
            'camera_artifacts':{'intrinsics':identity,'poses':identity},'jobs':[job]}]}
    rows=[]
    for tool in ('trellis','reconviagen'):
        rows.append({**{k:job[k] for k in ('freeze_id','job_id','scene_id','object_slot')},
            'proposal_id':job['job_id']+':'+tool,'tool_id':tool,'availability':'typed_unavailable',
            'artifact_paths':{},'artifact_hashes':{},'artifact_sizes':{},
            'construction_evidence':{'typed_failure':'preparation_unavailable','missing_artifacts':['raw_mesh','raw_gaussian']},
            'raw_generator_provenance':{'artifact_bytes_hash_frozen':False,'generator_commit':None,
                'checkpoint_identity_recorded':False,'runtime_manifest_recorded':False,'claim_status':'not_run'}})
    proposals={'schema_version':2,'freeze_id':'freeze','counts':{'jobs':1,'trellis_available':0,
                'reconviagen_available':0,'reconviagen_typed_unavailable':1},'proposals':rows}
    return jobs,proposals


def test_missing_preparation_reaches_all_five_terminal_rows_without_model_or_observation(tmp_path,monkeypatch):
    jobs,proposals=inventory()
    e3._validate_controller_inventory_schema(jobs,proposals)
    monkeypatch.setattr(e3,'_load_inventory',lambda *_a,**_k:copy.deepcopy((jobs,proposals)))
    def forbidden(*a,**kw): raise AssertionError('unavailable job invoked registration')
    monkeypatch.setattr(e3,'_process_candidate',forbidden)
    out=tmp_path/'agentic'
    observed=e3.run_observe('freeze',out,'09c1414f1b')
    assert observed['renderer_runtime'] is None
    assert len(observed['records'])==1
    assert observed['records'][0]['observation_point_count'] is None
    assert not list(out.rglob('*.npz'))
    policies=yaml.safe_load((e3.REPOSITORY_ROOT/'configs/experiments/icra2027/agentic_policies.yaml').read_text())
    result=e3.run_control(policies,{'code':{'commit':'d'*40}},'freeze',out,'09c1414f1b')
    assert result['job_count']==1 and result['ledger_row_count']==5
    rows=[json.loads(line) for line in (out/'control/09c1414f1b/job_ledger.jsonl').read_text().splitlines()]
    assert {r['policy_id'] for r in rows}=={'A0','A1','A2','A3','A4'}
    assert all(r['selected_proposal_id'] is None and r['terminal_action'] in {'reject','abstain'} for r in rows)
    assert not result['proposals']


@pytest.mark.parametrize('mutation',['fake_rgba','fake_frame','fake_mask','invalid_reason','legacy_version','available_proposal'])
def test_missing_job_cannot_masquerade_as_prepared(mutation):
    jobs,proposals=inventory();job=jobs['scenes'][0]['jobs'][0]
    if mutation=='fake_rgba':job['artifact_paths']['rgba']='fake.png'
    if mutation=='fake_frame':job['construction_evidence']['source_frame']='test.png'
    if mutation=='fake_mask':job['construction_evidence']['mask_provenance']='gt_projection'
    if mutation=='invalid_reason':job['construction_evidence']['reason_code']='low_final_f1'
    if mutation=='legacy_version':jobs['schema_version']=1
    if mutation=='available_proposal':
        row=proposals['proposals'][0];row['availability']='available'
        row['artifact_paths']={'raw_mesh':'m','raw_gaussian':'g'}
        row['artifact_hashes']={'raw_mesh':'a'*64,'raw_gaussian':'b'*64}
        row['artifact_sizes']={'raw_mesh':1,'raw_gaussian':1};row['construction_evidence']={}
    with pytest.raises(ValueError):e3._validate_controller_inventory_schema(jobs,proposals)


def test_missing_observation_cannot_claim_measured_zero_even_with_resealed_json(tmp_path,monkeypatch):
    jobs,proposals=inventory()
    monkeypatch.setattr(e3,'_load_inventory',lambda *_a,**_k:(jobs,proposals))
    out=tmp_path/'agentic';e3.run_observe('freeze',out,'09c1414f1b')
    directory=out/'observations/09c1414f1b';p=directory/'manifest.json'
    data=json.loads(p.read_text());data['records'][0]['observation_point_count']=0
    p.write_text(json.dumps(data))
    seal=json.loads((directory/'seal.json').read_text())
    seal['members']['manifest.json']=seal['controller_members']['manifest.json']=e3.sha256_file(p)
    (directory/'seal.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError,match='fabricated measured'):
        e3._load_observation_scene(out,'09c1414f1b')

@pytest.fixture
def automatic_sources(tmp_path,monkeypatch):
    from run.icra2027 import e3_trellis_generation_pilot as inputs
    scene='09c1414f1b';source=tmp_path/'discovery';source.mkdir()
    identities={role:{'path':f'/fixed/{role}','bytes':1,'sha256':letter*64}
                for role,letter in zip(('gaussian','intrinsics','poses'),'abc')}
    manifest={'gaussian':identities['gaussian'],'metadata':{
        'nerfstudio/transforms_undistorted.json':identities['intrinsics'],
        'colmap/images.txt':identities['poses']}}
    for name in ('pilot_summary.json','output_hashes.json','postrun_audit.json'):
        (source/name).write_text('{}')
    (source/'input_manifest.json').write_text(json.dumps(manifest))
    originals=[];pool_rows=[]
    for i in range(6):
        prepared=i<3;crop=tmp_path/f'crop{i}.png';crop.write_bytes(bytes([i]))
        original={'job_id':f'{scene}:auto:{1000+i}','automatic_instance_id':1000+i,'prepared':prepared}
        if prepared:
            original.update(input={'path':str(crop),'sha256':e3.sha256_file(crop)},frame='train.png',output_index=i,
                object_metadata={'bbox_px':[0,0,1,1],'gt_object_id':1000+i,'label':'NOT_FOR_CONTROLLER'})
        originals.append(original)
        artifacts={}
        if prepared:
            for name in ('trellis_mesh.ply','trellis_gs.ply'):
                path=tmp_path/f'{i}-{name}';path.write_bytes(b'fixture')
                artifacts[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':e3.sha256_file(path)}
        pool_rows.append({**{k:original[k] for k in ('job_id','automatic_instance_id','prepared')},
            'tool':'trellis','seed':42,'shared_initial_policy_rows':['A1','A2','A3','A4'],
            'input_sha256':e3.sha256_file(crop),'status':'available' if prepared else 'unavailable',
            'artifacts':artifacts,'runtime':{'status':'generated','wall_s':1.0} if prepared else None})
    monkeypatch.setattr(inputs,'source_jobs',lambda _:copy.deepcopy(originals))
    generated=tmp_path/'generated';generated.mkdir();mp=generated/'input_manifest.json'
    mp.write_text(json.dumps({'code_commit':'d'*40,'models':{'trellis':{'sha256':'e'*64}}}))
    pp=generated/'proposal_pool.json';pp.write_text(json.dumps({'code_commit':'d'*40,'freeze_id':'old',
        'input_manifest_sha256':e3.sha256_file(mp),'rows':pool_rows,'planned_jobs':6}))
    config={'study_scope':'automatic_training_only_engineering','paper_ready':False,
      'automatic_sources':{'scene_id':scene,'planned_jobs':6,'discovery_directory':str(source),
        'discovery_hashes':{p.name:e3.sha256_file(p) for p in source.iterdir()},
        'scene_sources':{k:{'path':v['path'],'size_bytes':v['bytes'],'sha256':v['sha256']} for k,v in identities.items()},
        'initial_pools':{'trellis':{'path':str(pp),'sha256':e3.sha256_file(pp)},
                         'reconviagen':{'status':'not_run','reason_code':'initial_tool_not_run'}}}}
    return config,{'contract_sha256':'f'*64,'freeze_id':'freeze','code':{'commit':'e'*40}}


def test_automatic_inventory_keeps_all_six_and_sanitizes_metadata(automatic_sources,tmp_path):
    config,contract=automatic_sources
    audit=e3.run_inventory(config,contract,'freeze',tmp_path/'agentic')
    jobs,proposals=e3._load_inventory(tmp_path/'agentic',controller_safe=True)
    assert jobs['counts']=={'scenes':1,'jobs':6,'policy_object_rows':30}
    assert len(proposals['proposals'])==12 and proposals['counts']['trellis_available']==3
    assert sum(e3._unavailable_observation(j) for j in jobs['scenes'][0]['jobs'])==3
    text=json.dumps(jobs)+json.dumps(proposals)
    assert 'gt_object_id' not in text and 'NOT_FOR_CONTROLLER' not in text
    assert audit['paper_ready'] is False and audit['evaluation_references']==[]
    with pytest.raises((ValueError,FileExistsError)):
        e3.run_inventory(config,contract,'freeze',tmp_path/'agentic')


@pytest.mark.parametrize('mutation',['lost_job','duplicate_job','different_crop','different_tool','different_seed','missing_runtime','changed_mesh','changed_scene_source'])
def test_automatic_inventory_rejects_source_or_pairing_drift(automatic_sources,mutation):
    from agents.orchestrator.automatic_inventory import build_payloads
    config,contract=automatic_sources;s=config['automatic_sources'];spec=s['initial_pools']['trellis']
    pp=Path(spec['path']);pool=json.loads(pp.read_text());row=pool['rows'][0]
    if mutation=='lost_job':pool['rows'].pop()
    if mutation=='duplicate_job':pool['rows'].append(copy.deepcopy(row))
    if mutation=='different_crop':row['input_sha256']='0'*64
    if mutation=='different_tool':row['tool']='reconviagen'
    if mutation=='different_seed':row['seed']=0
    if mutation=='missing_runtime':row['runtime']=None
    if mutation=='changed_mesh':Path(row['artifacts']['trellis_mesh.ply']['path']).write_bytes(b'changed')
    if mutation=='changed_scene_source':s['scene_sources']['gaussian']['path']='/other/gaussian'
    pp.write_text(json.dumps(pool));spec['sha256']=e3.sha256_file(pp)
    with pytest.raises(ValueError):build_payloads(config,contract,'freeze')

@pytest.mark.parametrize('mutation',['job_count','proposal_count','duplicate_tool','wrong_object'])
def test_automatic_denominator_and_tool_pair_binding(mutation):
    jobs,proposals=inventory()
    if mutation=='job_count':jobs['counts']['jobs']=0
    if mutation=='proposal_count':proposals['counts']['trellis_available']=1
    if mutation=='duplicate_tool':proposals['proposals'][1]['tool_id']='trellis'
    if mutation=='wrong_object':proposals['proposals'][1]['object_slot']='obj_9999'
    with pytest.raises(ValueError,match='automatic'):
        e3._validate_controller_inventory_schema(jobs,proposals)


@pytest.mark.parametrize('count',[0,3,7])
def test_fresh_inventory_uses_complete_population_not_legacy_six(automatic_sources,monkeypatch,count):
    from agents.orchestrator.automatic_inventory import build_payloads
    from run.icra2027 import e3_trellis_generation_pilot as inputs
    config,contract=automatic_sources;source=config['automatic_sources']
    directory=Path(source['discovery_directory']);scene=source['scene_id']
    originals=[{'job_id':f'{scene}:auto:{1000+i}','automatic_instance_id':1000+i,
                'prepared':False} for i in range(count)]
    monkeypatch.setattr(inputs,'source_jobs',lambda _:copy.deepcopy(originals))
    source['planned_jobs']=count
    for tool in source['initial_pools']:
        source['initial_pools'][tool]={'status':'not_run','reason_code':'initial_tool_not_run'}
    manifest_path=directory/'input_manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest.update(scene_id=scene,source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY')
    manifest_path.write_text(json.dumps(manifest))
    complete=directory/'all_jobs_manifest.json'
    complete.write_text(json.dumps({'scene_id':scene,'planned_jobs':count}))
    source['discovery_hashes']={p.name:e3.sha256_file(p) for p in directory.iterdir()}
    jobs,proposals,audit,_=build_payloads(config,contract,'freeze')
    assert jobs['counts']['jobs']==count and jobs['counts']['policy_object_rows']==count*5
    assert len(proposals['proposals'])==count*2
    assert audit['source_gaussian_training_provenance']=='FRESH_OFFICIAL_TRAIN_ONLY'
    assert audit['paper_ready'] is False and audit['initial_pool_complete'] is False
    source['discovery_hashes'].pop('all_jobs_manifest.json')
    with pytest.raises(ValueError,match='all-jobs anchor'):
        build_payloads(config,contract,'freeze')


def test_source_scene_binding_cannot_be_changed_even_when_population_matches(automatic_sources,monkeypatch):
    from agents.orchestrator.automatic_inventory import build_payloads
    from run.icra2027 import e3_trellis_generation_pilot as inputs
    config,contract=automatic_sources
    rows=inputs.source_jobs(None);rows[0]['job_id']='another_scene:auto:1000'
    monkeypatch.setattr(inputs,'source_jobs',lambda _:rows)
    with pytest.raises(ValueError,match='scene binding'):
        build_payloads(config,contract,'freeze')


def _sealed_fresh_source(tmp_path, count=1):
    """Real source_jobs path, with complete NPZ population and authenticated seal."""
    import numpy as np
    from run.icra2027.e3_auto_discovery_pilot import summarize_instances
    scene='09c1414f1b';directory=tmp_path/'real-discovery'
    (directory/'construction/objects').mkdir(parents=True)
    identities={k:{'path':'/test-only/'+k,'bytes':1,'sha256':c*64}
                for k,c in zip(('gaussian','intrinsics','poses'),'abc')}
    def write(name,value):
        path=directory/name;path.write_text(json.dumps(value));return e3.sha256_file(path)
    objects=[]
    if count:
        objects=[{'index':0,'gt_object_id':1000,'frame':'train.jpg','bbox_px':[0,0,1,1]}]
        crop=directory/'construction/objects/obj_00/rgba.png'
        crop.parent.mkdir();crop.write_bytes(b'same-single-crop')
    write('construction/objects/objects.json',objects)
    labels=['book']*count
    np.savez(directory/'construction/auto_instances.npz',labels=np.asarray(labels,dtype=str))
    stages=[{'stage':s,'exit_code':0} for s in ['render','fuse','discover','prepare']]
    stages.append({'stage':'refine','exit_code':0} if count else
                  {'stage':'refine','status':'NOT_RUN','reason':'empty_prepared_population'})
    input_hash=write('input_manifest.json',dict(scene_id=scene,
        source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY',
        planned_stages=[s['stage'] for s in stages],boundary={'training_frames':['train.jpg']},
        gaussian=identities['gaussian'],metadata={
            'nerfstudio/transforms_undistorted.json':identities['intrinsics'],
            'colmap/images.txt':identities['poses']}))
    rows=summarize_instances(labels,objects)
    summary=dict(scene_id=scene,paper_ready=False,freeze_id='fresh-discovery',code_commit='a'*40,
        stages=stages,rows=rows,discovered_instances=count,prepared_instances=len(objects),
        source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY')
    summary_hash=write('pilot_summary.json',summary)
    index={str(p.relative_to(directory/'construction')):{'sha256':e3.sha256_file(p)}
           for p in (directory/'construction').rglob('*') if p.is_file()}
    output_hash=write('output_hashes.json',index)
    all_hash=write('all_jobs_manifest.json',dict(kind='complete_automatic_discovery_jobs',schema_version=1,
        paper_ready=False,rows=rows,planned_jobs=count,scene_id=scene,freeze_id='fresh-discovery',code_commit='a'*40,
        source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY',input_manifest_sha256=input_hash,
        summary_sha256=summary_hash,output_hashes_sha256=output_hash))
    write('postrun_audit.json',dict(summary_sha256=summary_hash,output_hashes_sha256=output_hash,
        all_jobs_manifest_sha256=all_hash))
    source=dict(scene_id=scene,planned_jobs=count,discovery_directory=str(directory),
        discovery_hashes={p.name:e3.sha256_file(p) for p in directory.glob('*.json')},
        scene_sources={k:dict(path=v['path'],size_bytes=v['bytes'],sha256=v['sha256']) for k,v in identities.items()},
        initial_pools={tool:dict(status='not_run',reason_code='initial_tool_not_run') for tool in ('trellis','reconviagen')})
    return dict(study_scope='automatic_training_only_engineering',paper_ready=False,automatic_sources=source), {
        'contract_sha256':'f'*64,'freeze_id':'freeze','code':{'commit':'e'*40}}


@pytest.mark.parametrize('count',[0,1,7])
def test_real_fresh_source_complete_population_integration(tmp_path,count):
    from agents.orchestrator.automatic_inventory import build_payloads
    config,contract=_sealed_fresh_source(tmp_path,count)
    jobs,proposals,audit,_=build_payloads(config,contract,'freeze')
    assert jobs['counts']['jobs']==count and len(proposals['proposals'])==2*count
    assert audit['source_gaussian_training_provenance']=='FRESH_OFFICIAL_TRAIN_ONLY'
    assert audit['initial_pool_complete'] is False


@pytest.mark.parametrize('mutation',['unknown_summary','missing_all_jobs_seal','missing_both','wrong_all_jobs_seal'])
def test_real_fresh_source_cannot_bypass_complete_seal(tmp_path,mutation):
    from agents.orchestrator.automatic_inventory import build_payloads
    config,contract=_sealed_fresh_source(tmp_path,0)
    source=config['automatic_sources'];directory=Path(source['discovery_directory'])
    summary_path=directory/'pilot_summary.json';summary=json.loads(summary_path.read_text())
    audit_path=directory/'postrun_audit.json';audit=json.loads(audit_path.read_text())
    if mutation in ('unknown_summary','missing_both'):
        summary['source_gaussian_training_provenance']='UNKNOWN'
        summary_path.write_text(json.dumps(summary));audit['summary_sha256']=e3.sha256_file(summary_path)
    if mutation in ('missing_all_jobs_seal','missing_both'):audit.pop('all_jobs_manifest_sha256')
    if mutation=='wrong_all_jobs_seal':audit['all_jobs_manifest_sha256']='0'*64
    audit_path.write_text(json.dumps(audit))
    source['discovery_hashes']={p.name:e3.sha256_file(p) for p in directory.glob('*.json')}
    with pytest.raises(ValueError,match='provenance differs|all-jobs seal'):
        build_payloads(config,contract,'freeze')


@pytest.mark.parametrize('tool',['trellis','reconviagen'])
@pytest.mark.parametrize('mutation',['none','legacy_provenance','legacy_pool','missing_binding','different_binding'])
def test_fresh_pool_requires_complete_discovery_receipt_not_only_equal_rgba(tmp_path,tool,mutation):
    from agents.orchestrator.automatic_inventory import build_payloads
    from run.icra2027.e3_trellis_generation_pilot import source_jobs
    config,contract=_sealed_fresh_source(tmp_path)
    source=config['automatic_sources'];origin=source_jobs(source['discovery_directory'])[0]
    directory=tmp_path/'generation';directory.mkdir()
    artifact_names=('trellis_mesh.ply','trellis_gs.ply') if tool=='trellis' else ('rvg_mesh.ply','rvg_gs.ply')
    artifacts={}
    for name in artifact_names:
        p=directory/name;p.write_bytes(b'unchanged-raw-proposal')
        artifacts[name]={'path':str(p),'bytes':p.stat().st_size,'sha256':e3.sha256_file(p)}
    provenance='FRESH_OFFICIAL_TRAIN_ONLY'
    manifest=dict(code_commit='d'*40,models={tool:{'sha256':'e'*64}},
        source_gaussian_training_provenance=provenance,source_discovery_hashes=copy.deepcopy(source['discovery_hashes']))
    if mutation=='legacy_provenance':manifest['source_gaussian_training_provenance']='UNKNOWN'
    if mutation=='missing_binding':manifest.pop('source_discovery_hashes')
    if mutation=='different_binding':manifest['source_discovery_hashes']['input_manifest.json']='0'*64
    mp=directory/'input_manifest.json';mp.write_text(json.dumps(manifest))
    row=dict(job_id=origin['job_id'],automatic_instance_id=1000,prepared=True,tool=tool,seed=42,
        shared_initial_policy_rows=['A1','A2','A3','A4'],input_sha256=origin['input']['sha256'],
        status='available',artifacts=artifacts,runtime={'status':'generated','wall_s':1.0})
    pool=dict(code_commit='d'*40,freeze_id='fresh-generation',input_manifest_sha256=e3.sha256_file(mp),
        planned_jobs=1,rows=[row],source_gaussian_training_provenance='UNKNOWN' if mutation=='legacy_pool' else provenance)
    pp=directory/'proposal_pool.json';pp.write_text(json.dumps(pool))
    source['initial_pools'][tool]={'path':str(pp),'sha256':e3.sha256_file(pp)}
    if mutation=='none':
        _,proposals,_,_=build_payloads(config,contract,'freeze')
        assert proposals['counts'][tool+'_available']==1
    else:
        with pytest.raises(ValueError,match='complete discovery source binding'):
            build_payloads(config,contract,'freeze')


@pytest.mark.parametrize('status,reason,complete',[
    ('available',None,True),
    ('generation_failed','producer_exception',True),
    ('unavailable','fewer_than_two_usable_training_views',True),
    ('unavailable','initial_tool_not_run',False),
])
def test_initial_pool_completeness_requires_each_prepared_tool_terminal(automatic_sources,tmp_path,status,reason,complete):
    from agents.orchestrator.automatic_inventory import build_payloads
    config,contract=automatic_sources;source=config['automatic_sources']
    trellis_path=Path(source['initial_pools']['trellis']['path'])
    trellis=json.loads(trellis_path.read_text())
    rvg=copy.deepcopy(trellis)
    for row in rvg['rows']:
        row['tool']='reconviagen'
        row['artifacts']={name.replace('trellis_','rvg_'):value for name,value in row['artifacts'].items()}
    rvg_path=trellis_path.parent/'rvg_pool.json';rvg_path.write_text(json.dumps(rvg))
    source['initial_pools']['reconviagen']={'path':str(rvg_path),'sha256':e3.sha256_file(rvg_path)}
    row=trellis['rows'][0];row.update(status=status,reason=reason)
    if status!='available':
        row['artifacts']={};row['runtime']={'status':'generation_failed','wall_s':1.0} if status=='generation_failed' else None
    trellis_path.write_text(json.dumps(trellis))
    source['initial_pools']['trellis']['sha256']=e3.sha256_file(trellis_path)
    _,_,audit,_=build_payloads(config,contract,'freeze')
    assert audit['initial_pool_complete'] is complete
