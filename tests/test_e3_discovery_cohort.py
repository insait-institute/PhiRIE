"""Complete TRAIN/full-scene routing without GPU or evaluation reads."""
import copy
import json
import os
from pathlib import Path
import runpy
import sys

import pytest
import yaml

from run.icra2027 import e3_auto_discovery_pilot as auto
from run.icra2027.e3_gaussian_train_only import cohort_scene_ids


@pytest.fixture
def config():
    result = yaml.safe_load((auto.CODE/'configs/experiments/icra2027/e3_discovery_cohort.yaml').read_text())
    result['scene_id'] = cohort_scene_ids(result)[0]
    # This is a synthetic current-source contract, not an update to the frozen
    # historical experiment. Optional renderer APIs legitimately change bytes.
    result['unchanged_producer_files'] = {
        name: auto.sha(auto.CODE/name) for name in result['unchanged_producer_files']}
    return result


def test_historical_producer_hash_cannot_be_reused_after_source_change(config):
    roster = yaml.safe_load((auto.CODE/config['scene_roster']).read_text())
    config['unchanged_producer_files']['agents/discover/derive_mesh_from_splat.py'] = '0' * 64
    with pytest.raises(ValueError, match='frozen discovery producer code changed'):
        auto.validate_config(config, roster)


@pytest.mark.parametrize('change', [None,'scene','limit','stride','vocab','prompts','reuse','gaussian','roster'])
def test_full_population_contract(config,change):
    roster = yaml.safe_load((auto.CODE/config['scene_roster']).read_text())
    assert len(cohort_scene_ids(config)) == 50
    if change == 'scene': config['scene_id'] = '0000000000'
    elif change == 'limit': config['max_train_frames'] = 48
    elif change == 'stride': config['discovery_stride'] = 6
    elif change == 'vocab': config['full_vocabulary'] = 0
    elif change == 'prompts': config['discovery_prompts'].pop()
    elif change == 'reuse': config['crop_comparison_reference'] = {'anything':'legacy'}
    elif change == 'gaussian': config['source_gaussian_training_provenance'] = 'UNKNOWN'
    elif change == 'roster': config['scene_roster'] = config['fidelity_roster']['path']
    if change:
        with pytest.raises(ValueError): auto.validate_config(config,roster)
    else: auto.validate_config(config,roster)


def test_cohort_destination_is_per_scene_and_rejects_symlink(config,tmp_path):
    assert auto.destination_for(config,tmp_path) == tmp_path/'auto_discovery_pilot'/config['scene_id']
    elsewhere=tmp_path/'elsewhere';elsewhere.mkdir()
    (tmp_path/'auto_discovery_pilot').symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(ValueError): auto.destination_for(config,tmp_path)


def test_gaussian_layout_requires_scene_full_receipts(config):
    prefix,phase,required=auto.gaussian_source_layout(config)
    assert prefix.endswith(config['scene_id']) and phase=='train-full'
    assert len(required)==7 and all(config['scene_id'] in p for p in required if not p.startswith('contract/'))
    assert not any('train-pilot' in p for p in required)


@pytest.mark.parametrize('stage',['render','discover'])
def test_actual_worker_routes_all_train_before_unchanged_stride(config,tmp_path,monkeypatch,stage):
    config.update(dataset_root=str(tmp_path/'dataset'))
    destination=auto.destination_for(config,tmp_path)
    monkeypatch.setattr(auto,'load_context',lambda *a:(config,'source'))
    monkeypatch.setattr(auto,'checked_manifest',lambda *a:{'boundary':{'training_frames':['a.jpg','b.jpg']},
        'sam3_checkpoint':{},'sam3_source':{}})
    monkeypatch.setattr(auto,'validate_staged_inputs',lambda *a:None)
    monkeypatch.setattr(auto,'validate_cohort_runtime',lambda *a:None)
    monkeypatch.setattr(auto,'validate_input_stats',lambda *a:None)
    monkeypatch.setattr(auto,'cohort_resources',lambda *a:({},{}))
    (tmp_path/'contract').mkdir();(tmp_path/'contract/discovery_resources.json').write_text('{}')
    from run.icra2027 import e3_gaussian_train_only as gaussian
    monkeypatch.setattr(gaussian,'require_published_full_source',lambda *a:None)
    monkeypatch.setenv('SLURM_JOB_ID','fixture')
    monkeypatch.setenv('SLURMD_NODENAME','hala')
    for key,value in {'SIMANY_ROOT':str(auto.CODE),'SIMANY_OUT':str(destination/'construction'),
        'SIMANY_AUTO':'1','SIMANY_FULL':'1','SIMANY_MESH_SRC':'derived','SIMANY_NO_GT':'1',
        'SIMANY_SCENE':config['scene_id'],'SIMANY_SCANNETPP_ROOT':str(destination/'inputs'),
        'SIMANY_SPLATS_ROOT':str(destination/'inputs/splats'),'PYTHONNOUSERSITE':'1',
        'SIMANY_SAM3_CKPT':config['sam3_checkpoint']['path']}.items(): monkeypatch.setenv(key,value)
    monkeypatch.setattr(sys,'addaudithook',lambda *a:None)
    calls=[]
    monkeypatch.setattr(runpy,'run_module',lambda module,**kw:calls.append((module,list(sys.argv))))
    monkeypatch.setattr(sys,'argv',[])
    monkeypatch.setattr(sys,'executable',config['render_python'] if stage=='render' else config['sam3_python'])
    auto.worker(tmp_path/'config',tmp_path,stage,config['scene_id'])
    assert len(calls)==1 and '--train-split' in calls[0][1]
    assert '--max-train-frames' not in calls[0][1]
    original=sys.executable
    monkeypatch.setattr(sys,'executable','/foreign/python')
    with pytest.raises(ValueError,match='interpreter'):auto.worker(tmp_path/'config',tmp_path,stage,config['scene_id'])
    monkeypatch.setattr(sys,'executable',original)
    monkeypatch.setenv('SIMANY_SCENE','foreign')
    with pytest.raises(ValueError,match='environment'):auto.worker(tmp_path/'config',tmp_path,stage,config['scene_id'])


def test_all_train_plan_stages_every_training_view_and_no_test(config,tmp_path,monkeypatch):
    from robo.eval.fidelity_replacements import _tree_inventory
    config=copy.deepcopy(config)
    source=tmp_path/'dataset/data'/config['scene_id']; dslr=source/'dslr'
    images=dslr/'resized_undistorted_images';images.mkdir(parents=True)
    train=[f'{i:03d}.jpg' for i in range(60)]; test=['test.jpg']
    for name in train+test:(images/name).write_bytes(name.encode())
    (dslr/'train_test_lists.json').write_text(json.dumps({'train':train,'test':test}))
    (dslr/'colmap').mkdir();(dslr/'colmap/images.txt').write_text(''.join(
        f'{i+1} 1 0 0 0 0 0 0 1 {name}\n0 0 -1\n' for i,name in enumerate(train+test)))
    (dslr/'nerfstudio').mkdir();(dslr/'nerfstudio/transforms_undistorted.json').write_text('{}')
    sam=tmp_path/'sam';sam.mkdir();(sam/'source.py').write_text('pass')
    weights=tmp_path/'weights';weights.write_bytes(b'weights')
    gaussian=tmp_path/'gaussian';gaussian.write_bytes(b'gaussian')
    config.update(dataset_root=str(tmp_path/'dataset'),gaussian=str(gaussian),gaussian_sha256=auto.sha(gaussian),
        sam3_checkpoint={'path':str(weights),'bytes':7,'sha256':auto.sha(weights)},
        sam3_source={'path':str(sam),'tree_sha256':_tree_inventory(sam,label='test')['tree_sha256']})
    monkeypatch.setattr(auto,'load_context',lambda *a:(config,'source'))
    monkeypatch.setattr(auto,'bind_cohort_gaussian',lambda c:c)
    monkeypatch.setattr(auto,'validate_gaussian_provenance',lambda *a:{'status':'FRESH_OFFICIAL_TRAIN_ONLY'})
    monkeypatch.setattr(auto,'validate_cohort_runtime',lambda *a:None)
    cfg=tmp_path/'config';cfg.write_text('frozen');freeze=tmp_path/'freeze';freeze.mkdir()
    auto.plan(cfg,freeze,config['scene_id'])
    destination=auto.destination_for(config,freeze)
    manifest=json.loads((destination/'input_manifest.json').read_text())
    assert manifest['boundary']['training_frames']==train and len(manifest['input_images'])==60
    assert manifest['full_vocabulary']==1 and len(manifest['discovery_prompts'])==30
    staged=destination/'inputs/data'/config['scene_id']/'dslr/resized_undistorted_images'
    assert sorted(p.name for p in staged.iterdir())==train and not (staged/'test.jpg').exists()
    assert manifest['boundary']['max_train_frames'] is None
    auto.validate_staged_inputs(destination,config,manifest)
    with pytest.raises(FileExistsError):auto.plan(cfg,freeze,config['scene_id'])


def test_all_fifty_status_rows_retain_unknown_and_failed_denominators(config,tmp_path,monkeypatch):
    scenes=cohort_scene_ids(config)
    monkeypatch.setattr(auto,'load_context',lambda *a:(config,'source'))
    cfg=tmp_path/'config';cfg.write_text(yaml.safe_dump(config))
    failed=auto.destination_for(config,tmp_path)
    auto.write_new(failed/'unit_failure.json',{'error':'missing Gaussian full receipt'})
    result=auto.cohort_status(cfg,tmp_path,tmp_path/'snapshot.json')
    assert [r['scene_id'] for r in result['rows']]==scenes
    assert result['planned_scenes']==50 and result['completed_scenes']==0
    assert result['rows'][0]['state']=='BLOCKED'
    assert all(row['planned_jobs'] is None for row in result['rows'])
    with pytest.raises(FileExistsError):auto.cohort_status(cfg,tmp_path,tmp_path/'snapshot.json')


@pytest.mark.parametrize('damage',[None,'seal','source','config','scene_override'])
def test_cohort_exact_source_e0_context(config,tmp_path,monkeypatch,damage):
    from robo.manifest.hash import canonical_hash
    config.pop('scene_id')
    config['freeze_id']='fixture'
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    root=tmp_path/'fixture';(root/'contract').mkdir(parents=True)
    contract={'code':{'commit':'a'*40,'dirty':False},'freeze_id':'fixture',
        'resource_inventory':[{'id':'e3_auto_pilot_config','sha256':auto.sha(path)}]}
    contract['contract_sha256']=canonical_hash(contract)
    if damage=='seal':contract['contract_sha256']='0'*64
    if damage=='source':
        contract['code']['commit']='b'*40
        contract['contract_sha256']=canonical_hash({k:v for k,v in contract.items() if k!='contract_sha256'})
    if damage=='config':path.write_text(path.read_text()+'# changed\n')
    if damage=='scene_override':
        config['scene_id']='09c1414f1b';path.write_text(yaml.safe_dump(config))
    (root/'contract/freeze_manifest.json').write_text(json.dumps(contract))
    monkeypatch.setattr(auto.subprocess,'check_output',lambda command,**kw:'' if 'status' in command else 'a'*40)
    if damage:
        with pytest.raises(ValueError):auto.load_context(path,root,'09c1414f1b')
    else:
        result,commit=auto.load_context(path,root,'09c1414f1b')
        assert result['scene_id']=='09c1414f1b' and commit=='a'*40


@pytest.fixture
def gaussian_proof(config,tmp_path,monkeypatch):
    from agents.recon import colmap_poses as poses
    from robo.manifest.hash import canonical_hash
    config=copy.deepcopy(config);config['dataset_root']=str(tmp_path/'dataset')
    root=tmp_path/'sourcefreeze';prefix,phase,required=auto.gaussian_source_layout(config);d=root/prefix
    dslr=Path(config['dataset_root'])/'data'/config['scene_id']/'dslr'
    frames=[];images={};metadata={}
    for index in range(60):
        name=f'{index:03d}.jpg';p=dslr/'resized_undistorted_images'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
        images[name]=auto.identity(p)
        frames.append({'name':name,'source_rgb':str(p),'image':poses.file_identity(p)})
    for name in ('colmap/images.txt','nerfstudio/transforms_undistorted.json','train_test_lists.json'):
        p=dslr/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture camera/split')
        metadata[name]=auto.identity(p)
    inputs={'frames':frames,'source_metadata':{name:{'path':item['path'],'sha256':item['sha256']} for name,item in metadata.items()},
        'boundary':{'max_train_frames':None},'initialization_selection':{'max_frames':48}}
    initialization={'n_points':12}
    auto.write_new(d/'scene/training_inputs.json',inputs)
    auto.write_new(d/'initialization/init_manifest.json',initialization)
    code={'commit':'d'*40,'dirty':False};base={'code':code,'status':'PASS','config':{'sha256':'e'*64},
        'scene_id':config['scene_id'],'scope':'fresh_train_only_gaussian_population'}
    auto.write_new(d/'prepare_receipt.json',{**base,'input_manifest':poses.file_identity(d/'scene/training_inputs.json')})
    auto.write_new(d/'triangulate_receipt.json',{**base,'prepare_receipt':poses.file_identity(d/'prepare_receipt.json'),
        'init_manifest':poses.file_identity(d/'initialization/init_manifest.json')})
    report={'iters':15000,'seed':42,'independent_heldout_evaluation':False,'n_gaussians':5,
        'provenance':{'initialization':initialization,'official_test_images_read':0}}
    auto.write_new(d/phase/'train_report.json',report)
    (d/phase/'scene.ply').write_bytes(b'fixture gaussian')
    auto.write_new(d/f'{phase}_receipt.json',{**base,'training_report':report,
        'prepare_receipt':poses.file_identity(d/'prepare_receipt.json'),
        'triangulate_receipt':poses.file_identity(d/'triangulate_receipt.json'),
        'artifacts':{'scene.ply':poses.file_identity(d/phase/'scene.ply')}})
    contract={'freeze_id':root.name,'code':code,'resource_inventory':[{'id':'e3_gaussian_config','sha256':'e'*64}]}
    contract['contract_sha256']=canonical_hash(contract)
    auto.write_new(root/'contract/freeze_manifest.json',contract)
    config['gaussian_cohort']={'freeze_root':str(root),'producer_commit':'d'*40,
        'contract':poses.file_identity(root/'contract/freeze_manifest.json'),'config_sha256':'e'*64}
    monkeypatch.setattr(poses,'validate_training_initialization',lambda *a:(inputs,initialization))
    return auto.bind_cohort_gaussian(config), {'training_frames':[r['name'] for r in frames]}, images, metadata


@pytest.mark.parametrize('damage',[None,'source','anchor','roster','rgb','metadata','config'])
def test_full_gaussian_receipt_chain_binds_completed_same_scene_train_full(gaussian_proof,damage):
    c,b,images,metadata=gaussian_proof
    if damage=='source':c['gaussian_provenance']['producer_commit']='0'*40
    elif damage=='anchor':next(iter(c['gaussian_provenance']['anchors'].values()))['sha256']='0'*64
    elif damage=='roster':b['training_frames'].pop()
    elif damage=='rgb':images[b['training_frames'][0]]['sha256']='0'*64
    elif damage=='metadata':metadata['colmap/images.txt']['sha256']='0'*64
    elif damage=='config':c['gaussian_cohort']['config_sha256']='0'*64
    if damage:
        with pytest.raises(ValueError):auto.validate_gaussian_provenance(c,b,images,metadata)
    else:
        proof=auto.validate_gaussian_provenance(c,b,images,metadata)
        assert len(proof['training_frames'])==60 and proof['training_iterations']==15000
        assert proof['status']=='FRESH_OFFICIAL_TRAIN_ONLY' and not proof['independent_heldout_evaluation']


def test_other_scene_cannot_bind_completed_gaussian_unit(gaussian_proof):
    c,*_=gaussian_proof;c['scene_id']='0000000000'
    with pytest.raises(FileNotFoundError):auto.bind_cohort_gaussian(c)


@pytest.mark.parametrize('damage',[None,'producer','status','roster','image_path','metadata_path','source_receipt'])
def test_resealed_worker_plan_cannot_change_gaussian_source_binding(gaussian_proof,tmp_path,damage):
    c,b,images,metadata=gaussian_proof
    proof=auto.validate_gaussian_provenance(c,b,images,metadata)
    b['max_train_frames']=None
    cfg=tmp_path/'config';cfg.write_text('frozen config')
    destination=tmp_path/'worker';destination.mkdir()
    manifest={'code_commit':'active','config_sha256':auto.sha(cfg),'scope':auto.COHORT_SCOPE,
        'scene_id':c['scene_id'],'freeze_id':c['freeze_id'],'gaussian':auto.identity(c['gaussian']),
        'gaussian_provenance':proof,'boundary':b,'input_images':images,'metadata':metadata,
        'source_gaussian_training_provenance':'FRESH_OFFICIAL_TRAIN_ONLY'}
    if damage=='producer':manifest['gaussian_provenance']['producer_commit']='0'*40
    elif damage=='status':manifest['source_gaussian_training_provenance']='UNKNOWN'
    elif damage=='roster':manifest['boundary']['training_frames'].pop()
    elif damage=='image_path':manifest['input_images'][b['training_frames'][0]]['path']='/foreign/train.jpg'
    elif damage=='metadata_path':manifest['metadata']['colmap/images.txt']['path']='/foreign/images.txt'
    elif damage=='source_receipt':
        prefix,phase,_=auto.gaussian_source_layout(c)
        path=Path(c['gaussian_cohort']['freeze_root'])/prefix/f'{phase}_receipt.json'
        row=json.loads(path.read_text());row['status']='FAIL';path.write_text(json.dumps(row))
    auto.write_new(destination/'input_manifest.json',manifest)
    auto.write_new(destination/'plan_receipt.json',{'scene_id':c['scene_id'],'code_commit':'active',
        'config_sha256':auto.sha(cfg),'input_manifest_sha256':auto.sha(destination/'input_manifest.json')})
    if damage:
        with pytest.raises(ValueError):auto.checked_manifest(c,'active',cfg,destination)
    else:assert auto.checked_manifest(c,'active',cfg,destination)==manifest


def test_worker_cheap_stats_reject_changed_rgb_and_checkpoint(tmp_path):
    image=tmp_path/'train.jpg';image.write_bytes(b'original')
    checkpoint=tmp_path/'weights';checkpoint.write_bytes(b'weights')
    manifest={'input_images':{'train.jpg':auto.identity(image)},'metadata':{},
              'gaussian':auto.identity(image),'sam3_checkpoint':auto.identity(checkpoint)}
    auto.validate_input_stats(manifest)
    image.write_bytes(b'changed RGB')
    with pytest.raises(ValueError,match='input changed'):auto.validate_input_stats(manifest)
    manifest['input_images']={'train.jpg':auto.identity(image)};manifest['gaussian']=auto.identity(image)
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError,match='input changed'):auto.validate_input_stats(manifest)


@pytest.mark.parametrize('field',['path','sha256','bytes','tree_sha256'])
def test_resealed_resource_cache_cannot_replace_frozen_checkpoint_or_source(tmp_path,field):
    from robo.eval.fidelity_replacements import _tree_inventory
    weights=tmp_path/'weights';weights.write_bytes(b'weights')
    source=tmp_path/'source';source.mkdir();(source/'source.py').write_text('pass')
    config={'sam3_checkpoint':{'path':str(weights),'sha256':auto.sha(weights),'bytes':7},
            'sam3_source':{'path':str(source),'tree_sha256':_tree_inventory(source,label='test')['tree_sha256']}}
    auto.cohort_resources(config,tmp_path,'source')
    path=tmp_path/'contract/discovery_resources.json';record=json.loads(path.read_text())
    if field=='tree_sha256':record['source'][field]='0'*64
    else:record['checkpoint'][field]={'path':'/foreign/weights','sha256':'0'*64,'bytes':8}[field]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='frozen config'):auto.cohort_resources(config,tmp_path,'source')
