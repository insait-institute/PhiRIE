import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import pytest

from run.icra2027 import e3_rvg_generation_pilot as rvg
from run.icra2027 import e3_fresh_rvg_config as builder
from run.icra2027.e3_auto_discovery_pilot import identity,sha,PilotError
from run.icra2027.e3_fresh_generation_contract import FRESH,DISCOVERY_FILES


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def views_fixture(tmp_path,monkeypatch,count=3):
    out=tmp_path/'out';out.mkdir();images=tmp_path/'images';images.mkdir()
    for name,color in [('a.png',30),('b.png',90)]:Image.fromarray(np.full((4,4,3),color,np.uint8)).save(images/name)
    boundary={'boundary':{'training_frames':['a.png','b.png']},
              'input_images':{name:identity(images/name) for name in ['a.png','b.png']},
              'metadata':{'colmap/cameras.txt':{'sha256':'camera'},'colmap/images.txt':{'sha256':'poses'},'train_test_lists.json':{'sha256':'split'}}}
    source=tmp_path/'source';source.mkdir()
    source_hashes={}
    objects=[{'index':0,'gt_object_id':1000}] if count else []
    for name in rvg.CONSTRUCTION_INPUTS:
        path=out/'construction'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(json.dumps(objects).encode() if name=='objects/objects.json' else name.encode())
        source_hashes[name]=identity(path)
    dump(source/'output_hashes.json',source_hashes)
    c={'source_pilot':str(source),'scene_id':'scene','source_input_manifest_sha256':'input',
       'source_gaussian_training_provenance':FRESH,'source_discovery_hashes':{n:'hash' for n in DISCOVERY_FILES},'models':{}}
    jobs=[{'job_id':f'scene:auto:{1000+i}','automatic_instance_id':1000+i,'prepared':i==0,'output_index':0 if i==0 else None,
           'input':{'sha256':'initial_crop'}} for i in range(count)]
    m={'jobs':jobs,'source_gaussian_training_provenance':FRESH,'source_discovery_hashes':c['source_discovery_hashes']}
    dump(out/'input_manifest.json',m);dump(out/'pinned_models.json',{'models':{}})
    views=[('a.png',(0,0,2,2),np.ones((2,2),bool)),('b.png',(1,1,3,3),np.ones((2,2),bool))]
    common=types.ModuleType('agents.core.common');common.IMAGES_DIR=images
    common.load_intrinsics=lambda:(np.eye(3),4,4,{})
    common.load_colmap_w2c=lambda:{n:np.eye(4) for n in ['a.png','b.png']}
    common.make_raycast_scene=lambda:object()
    common.load_instances=lambda:[{'object_id':1000}]
    import agents.core
    monkeypatch.setitem(sys.modules,'agents.core.common',common);monkeypatch.setattr(agents.core,'common',common,raising=False)
    align=types.ModuleType('agents.assets.s5_align');align.scene_mesh_arrays=lambda:(np.zeros((3,3)),np.zeros((1,3),int))
    monkeypatch.setitem(sys.modules,'agents.assets.s5_align',align)
    model=types.ModuleType('models.s4_reconviagen');model.collect_views=lambda *_:views
    monkeypatch.setitem(sys.modules,'models.s4_reconviagen',model)
    import agents.discover.training_views as training
    monkeypatch.setattr(training,'select_training_views',lambda frames,*_:(frames,{'training_frames':['a.png','b.png']}))
    return c,out,m,boundary,views,images


def seal_views(c,out,m,boundary):
    dump(out/'views_receipt.json',dict(status='PASS',input_manifest=identity(out/'input_manifest.json'),
        view_manifest=identity(out/'view_manifest.json'),source_binding=rvg.view_source_binding(c,boundary),paper_ready=False))


@pytest.mark.parametrize('count',[0,3,7])
def test_actual_view_manifest_preserves_full_dynamic_denominator(tmp_path,monkeypatch,count):
    c,out,m,boundary,views,images=views_fixture(tmp_path,monkeypatch,count)
    _,_,record=rvg.collect_frozen_views(c,out,boundary,write=True)
    seal_views(c,out,m,boundary)
    rvg.validate_view_receipt(c,out,m,boundary)
    assert len(record['all_jobs'])==count and len(record['rows'])==min(1,count)
    assert record['source_discovery_hashes']==c['source_discovery_hashes']
    rvg.collect_frozen_views(c,out,boundary,write=False)


@pytest.mark.parametrize('change',['test_frame','mixed_image_root','changed_mask'])
def test_actual_view_input_boundary_rejects_test_mixed_or_changed_mask(tmp_path,monkeypatch,change):
    c,out,m,boundary,views,images=views_fixture(tmp_path,monkeypatch)
    rvg.freeze_object_views(out,0,views,boundary,4,4,images,write=True)
    if change=='test_frame':views[0]=('test.png',views[0][1],views[0][2])
    elif change=='mixed_image_root':images=tmp_path/'another_scene/images'
    else:views[0]=('a.png',views[0][1],np.zeros((2,2),bool))
    with pytest.raises(PilotError):rvg.freeze_object_views(out,0,views,boundary,4,4,images,write=False)


def test_resealed_view_manifest_cannot_replace_collector_recomputation(tmp_path,monkeypatch):
    c,out,m,boundary,views,images=views_fixture(tmp_path,monkeypatch)
    rvg.collect_frozen_views(c,out,boundary,write=True)
    edited=json.loads((out/'view_manifest.json').read_text())
    edited['rows'][0]['views'].reverse();dump(out/'view_manifest.json',edited)
    seal_views(c,out,m,boundary)  # Attacker also updates the local hash receipt.
    with pytest.raises(PilotError,match='source-derived recomputation'):
        rvg.collect_frozen_views(c,out,boundary,write=False)


@pytest.mark.parametrize('change',['drop_job','mixed_mesh','mixed_source_hashes','gt_directory'])
def test_staged_source_rejects_dropped_jobs_and_mixed_geometry(tmp_path,monkeypatch,change):
    c,out,m,boundary,views,images=views_fixture(tmp_path,monkeypatch)
    binding={k:m[k] for k in ['source_gaussian_training_provenance','source_discovery_hashes']}
    monkeypatch.setattr(rvg,'source_boundary',lambda _:(copy.deepcopy(m['jobs']),boundary,binding))
    if change=='gt_directory':
        scans=out/'inputs/data/scene/scans';scans.mkdir(parents=True)
    else:monkeypatch.setattr(rvg,'validate_staged_inputs',lambda *_:None)
    if change=='drop_job':edited=copy.deepcopy(m);edited['jobs'].pop();dump(out/'input_manifest.json',edited)
    elif change=='mixed_mesh':(out/'construction/derived_mesh.ply').write_bytes(b'another-scene')
    elif change=='mixed_source_hashes':edited=copy.deepcopy(m);edited['source_discovery_hashes']={};dump(out/'input_manifest.json',edited)
    with pytest.raises(PilotError):rvg.read_manifest(c,out)


def test_actual_consumed_rvg_pngs_match_rgb_masks_not_initial_rgba(tmp_path,monkeypatch):
    c,out,m,boundary,views,images=views_fixture(tmp_path,monkeypatch)
    _,_,record=rvg.collect_frozen_views(c,out,boundary,write=True)
    directory=out/'construction/objects/obj_00/rvg';directory.mkdir(parents=True)
    for i,(name,(u0,v0,u1,v1),mask) in enumerate(views):
        rgb=np.asarray(Image.open(images/name).convert('RGB'))
        Image.fromarray(np.dstack([rgb[v0:v1,u0:u1],(mask*255).astype(np.uint8)])).save(directory/f'view_{i:02d}.png')
    consumed=rvg.verify_consumed_views(out,0,record['rows'][0]['views'])
    assert len(consumed)==2 and consumed[0]['source_mask_sha256']==record['rows'][0]['views'][0]['mask_sha256']
    for name in ['rvg_mesh.ply','rvg_gs.ply']:(directory/name).write_bytes(b'generated')
    runtime=dict(status='generated',seed=42,view_manifest_sha256=sha(out/'view_manifest.json'),
                 source_discovery_hashes=m['source_discovery_hashes'],consumed_views=consumed)
    dump(out/'producer_records/object_00.json',runtime)
    rows=rvg.collect_records(out,m,'fresh',0)
    assert len(rows)==3 and rows[0]['status']=='available'
    altered=copy.deepcopy(runtime);altered['source_discovery_hashes']={}
    dump(out/'producer_records/object_00.json',altered)
    with pytest.raises(PilotError,match='actual multiview'):rvg.collect_records(out,m,'fresh',0)
    Image.fromarray(np.zeros((2,2,4),np.uint8)).save(directory/'view_00.png')
    with pytest.raises(PilotError,match='consumed RGB/mask'):rvg.verify_consumed_views(out,0,record['rows'][0]['views'])


def test_rvg_builder_waits_without_fake_population_or_model_calls(tmp_path):
    output=tmp_path/'config.yaml'
    report=builder.prepare(tmp_path/'source','missing','1'*40,'missing',output_config=output)
    assert report['status']=='WAITING_DISCOVERY' and report['planned_jobs'] is None
    assert not report['model_calls_performed'] and not output.exists()


def test_rvg_rejects_single_rgba_reuse_configuration():
    with pytest.raises(PilotError,match='actual multiview'):
        rvg.source_boundary({'raw_reuse':{'old':'pool'}})


@pytest.mark.parametrize('allowed',[True,False])
def test_actual_execution_command_branch_without_gpu_or_model_calls(tmp_path,monkeypatch,allowed):
    root=tmp_path/'freeze';out=root/'rvg_initial';out.mkdir(parents=True)
    config=tmp_path/'config.yaml';config.write_text('synthetic')
    models={}
    for name in ['rvg_source','dinov2_source']:
        path=tmp_path/name;path.mkdir();models[name]={'path':str(path)}
    c={'models':models,'python':'synthetic-python'}
    m={'code_commit':'source','config_sha256':sha(config),'jobs':[]}
    dump(out/'input_manifest.json',m);dump(out/'model_stats.json',{});dump(out/'views_receipt.json',{})
    monkeypatch.setattr(rvg,'context',lambda *_:(c,'source'))
    monkeypatch.setattr(rvg,'read_manifest',lambda *_:(m,{}))
    monkeypatch.setattr(rvg,'validate_view_receipt',lambda *_:{'rows':[{'available':True}]})
    monkeypatch.setenv('SLURM_JOB_ID','synthetic-job')
    monkeypatch.setenv('SLURMD_NODENAME','hala' if allowed else 'unapproved-node')
    probes=[];calls=[];published=[]
    def probe(command,**kwargs):
        probes.append(command)
        return json.dumps({'name':'synthetic GPU','free_bytes':64*1024**3,'total_bytes':64*1024**3})
    monkeypatch.setattr(rvg.subprocess,'check_output',probe)
    monkeypatch.setattr(rvg.subprocess,'call',lambda command,**kwargs:calls.append(command) or 0)
    monkeypatch.setattr(rvg,'publish_pool',lambda *args,**kwargs:published.append(kwargs))
    if not allowed:
        with pytest.raises(PilotError,match='unauthorized'):rvg.execute(config,root)
        assert not probes and not calls and not published
    else:
        rvg.execute(config,root)
        assert len(probes)==len(calls)==1
        assert calls[0][-6:]==['--config',str(config.resolve()),'--freeze-root',str(root.resolve()),'--phase','worker']
        assert published==[{'generation_performed':True}]


def test_config_bound_cpu_view_replay_uses_exact_masks_without_raycast(tmp_path,monkeypatch):
    from run.icra2027 import e3_rvg_frozen_views as replay
    c,out,m,boundary,views,images=views_fixture(tmp_path,monkeypatch)
    _,_,original=rvg.collect_frozen_views(c,out,boundary,write=True)
    seal_views(c,out,m,boundary)
    replay_out=tmp_path/'new_out';(replay_out/'construction/objects').mkdir(parents=True)
    (replay_out/'construction/objects/objects.json').write_bytes((out/'construction/objects/objects.json').read_bytes())
    monkeypatch.setattr(replay,'authenticate_source',lambda *_:(out,original))
    common=sys.modules['agents.core.common']
    common.make_raycast_scene=lambda:pytest.fail('frozen mask replay must not raycast')
    sys.modules['models.s4_reconviagen'].collect_views=lambda *_:pytest.fail('frozen mask replay must not reselect')
    configured={**c,'frozen_view_replay':{'mode':replay.MODE}}
    _,data,record=rvg.collect_frozen_views(configured,replay_out,boundary,write=True)
    assert len(data)==1 and record['all_jobs']==original['all_jobs']
    for actual,old in zip(record['rows'][0]['views'],original['rows'][0]['views']):
        assert actual['mask_sha256']==old['mask_sha256']
        assert actual['rgba_pixels_sha256']==old['rgba_pixels_sha256']
        assert actual['bbox']==old['bbox'] and actual['frame']==old['frame']
    rvg.collect_frozen_views(configured,replay_out,boundary,write=False)
    mask=Path(record['rows'][0]['views'][0]['mask_path']);changed=np.load(mask);changed[0,0]=False
    np.save(mask,changed,allow_pickle=False)
    with pytest.raises(PilotError,match='mask differs'):
        rvg.collect_frozen_views(configured,replay_out,boundary,write=False)


def test_cpu_replay_cannot_reseal_changed_source_view_receipt(tmp_path,monkeypatch):
    from run.icra2027 import e3_rvg_frozen_views as replay
    c,out,m,boundary,views,images=views_fixture(tmp_path,monkeypatch)
    rvg.collect_frozen_views(c,out,boundary,write=True);seal_views(c,out,m,boundary)
    config=tmp_path/'producer.yaml';config.write_text('contract_resource_id: test\nfreeze_id: old\n')
    c['frozen_view_replay']=dict(mode=replay.MODE,producer_commit='a'*40,
        source_config=identity(config),source_contract={'placeholder':True},
        view_manifest=identity(out/'view_manifest.json'),views_receipt=identity(out/'views_receipt.json'))
    monkeypatch.setattr(replay,'checked_contract',lambda *args:{'freeze_id':'old'})
    altered=json.loads((out/'view_manifest.json').read_text());altered['rows'][0]['views'].reverse()
    dump(out/'view_manifest.json',altered);seal_views(c,out,m,boundary)
    with pytest.raises(PilotError,match='source anchor changed'):
        replay.authenticate_source(c,boundary)


def test_frozen_cpu_source_authenticates_e0_and_rejects_model_recipe_drift(tmp_path,monkeypatch):
    import yaml
    from robo.manifest.hash import canonical_hash
    from run.icra2027 import e3_rvg_frozen_views as replay
    c,out,m,boundary,views,images=views_fixture(tmp_path,monkeypatch)
    original={**c,'contract_resource_id':'rvg-original','freeze_id':'original','seed':42,'runtime_sha256':'runtime'}
    source_config=tmp_path/'original.yaml';source_config.write_text(yaml.safe_dump(original))
    m.update(code_commit='a'*40,config_sha256=sha(source_config));dump(out/'input_manifest.json',m)
    rvg.collect_frozen_views(c,out,boundary,write=True);seal_views(c,out,m,boundary)
    contract=dict(freeze_id='original',code=dict(commit='a'*40,dirty=False),
                  resource_inventory=[dict(id='rvg-original',sha256=sha(source_config))])
    contract['contract_sha256']=canonical_hash(contract)
    contract_path=tmp_path/'original/contract/freeze_manifest.json';dump(contract_path,contract)
    config={**original,'freeze_id':'new','frozen_view_replay':dict(mode=replay.MODE,producer_commit='a'*40,
        source_config=identity(source_config),source_contract=identity(contract_path),
        view_manifest=identity(out/'view_manifest.json'),views_receipt=identity(out/'views_receipt.json'))}
    monkeypatch.setattr(rvg,'output_directory',lambda *_:out)
    monkeypatch.setattr(rvg,'read_manifest',lambda *_:(m,boundary))
    source,record=replay.authenticate_source(config,boundary)
    assert source==out and len(record['all_jobs'])==3
    config['models']={'different_checkpoint':{'path':'changed'}}
    with pytest.raises(PilotError,match='construction/model recipe'):
        replay.authenticate_source(config,boundary)
