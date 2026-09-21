import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image
import pytest
import torch

from agents.edit import inpaint_fill as fill, inpaint_masks as masks, inpaint_qwen as erase
from robo.eval import e3_factory_materializer, agentic_ablation as e3
from robo.manifest.hash import canonical_hash


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def identity(path):return masks._identity(path)


def make_contract(path,context):
    value=dict(freeze_id='new',code=dict(commit='a'*40,dirty=False),resource_inventory=[
        dict(resolved_path=str(context),hash_method='content_sha256',sha256=identity(context)['sha256'])])
    value['contract_sha256']=canonical_hash(value);write(path,value)


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    monkeypatch.setattr(e3,'_validate_cli_execution',lambda *a,**k:None)
    monkeypatch.setattr(sys,'addaudithook',lambda _:None)
    monkeypatch.setattr(fill,'public_runtime',lambda _:{'mock':True})
    images=tmp_path/'dataset/scene/dslr/images';images.mkdir(parents=True)
    rgb=images/'TRAIN.png';Image.new('RGB',(8,8),(180,180,180)).save(rgb)
    camera=write(tmp_path/'camera.json',dict(fl_x=4,fl_y=4,cx=4,cy=4,w=8,h=8))
    poses=write(tmp_path/'poses.txt',{})
    monkeypatch.setattr(fill.C,'load_colmap_w2c',lambda _:{'TRAIN.png':np.eye(4)})
    splat=tmp_path/'source.ply';splat.write_bytes(b'original authenticated Gaussian')
    boundary=write(tmp_path/'discovery/input_manifest.json',dict(gaussian=identity(splat),
        metadata={'nerfstudio/transforms_undistorted.json':identity(camera),'colmap/images.txt':identity(poses)},
        input_images={'TRAIN.png':identity(rgb)}))
    generation=write(tmp_path/'generation.json',dict(source_pilot=str(boundary.parent),
        source_discovery_hashes={'input_manifest.json':identity(boundary)['sha256']}))
    prepcontext=write(tmp_path/'prep_context.json',dict(generation_config=identity(generation)))
    prep=tmp_path/'prep';obj=prep/'obj_1000';obj.mkdir(parents=True)
    np.save(prep/'removal_union_idx.npy',np.array([0],dtype=np.int64))
    write(obj/'plane.json',dict(origin=[0,0,1],normal=[0,0,1],u=[1,0,0],v=[0,1,0],
        footprint_uv=[[-.01,-.01],[.01,-.01],[.01,.01],[-.01,.01]]))
    write(obj/'views.json',[dict(frame='TRAIN.png',w2c=np.eye(4).tolist())])
    mask=tmp_path/'mask.png';Image.new('L',(8,8),255).save(mask)
    painted=tmp_path/'painted.png';Image.new('RGB',(8,8),(220,220,220)).save(painted)
    state=dict(object_slot='obj_1000',automatic_instance_id=1000,terminal_action='accept',plane_status='FIT',view_status='VIEWS_SELECTED')
    objects=[dict(index=1000,automatic_instance_id=1000,aabb=[[-.05,-.05,.95],[.05,.05,1.05]])]
    factory=tmp_path/'factory';write(factory/'objects/objects.json',objects)
    row=dict(object_slot='obj_1000',automatic_instance_id=1000,view_index=0,frame='TRAIN.png',status='MASK_READY',
        mask_identity=identity(mask),output_identity=identity(painted),erasure_status='ERASED',final_pixels=64)
    seal=write(tmp_path/'erasure/seal.json',{})
    mb=dict(preparation=dict(directory=str(prep),context_identity=identity(prepcontext)),source_factory=str(factory),
        train_images={'TRAIN.png':identity(rgb)},rows=[copy.deepcopy(row)],result={'objects':[state]},objects=objects)
    bundle=dict(seal_identity=identity(seal),context={'scene_id':'0000000001'},
        result={'status':'COMPLETE','rows':[row]},mask_bundle=mb)
    monkeypatch.setattr(erase,'validate_public_erasure',lambda _:bundle,raising=False)
    context=write(tmp_path/'context.json',dict(schema_version=1,scope='automatic_train_only_background_fill',freeze_id='new',
        scene_id='0000000001',erasure_seal=identity(seal),python=sys.executable,runtime={'mock':True},algorithm=fill.PUBLIC_ALGORITHM))
    contract=tmp_path/'contract.json';make_contract(contract,context)
    return context,contract,bundle,tmp_path


def test_explicit_source_paths_and_primary_target_preserved(setup):
    context,contract,bundle,_=setup
    p=fill._validate_public_context(context,contract)
    assert p['accepted_objects']==1 and p['blocked']==[]
    assert p['erasures'][('obj_1000',0)]==bundle['result']['rows'][0]['output_identity']
    assert p['output']!=p['preparation'] and p['factory']!=p['output']


@pytest.mark.parametrize('kind', ['missing', 'drift_after_start'])
def test_overlay_identity_is_required_before_and_after_execution(setup, monkeypatch, kind):
    context, contract, bundle, root = setup
    expected = {'mock': True, 'dependency_overlay': {'tree_sha256': 'frozen'}}
    value = json.loads(context.read_text()); value['runtime'] = expected
    write(context, value); make_contract(contract, context)
    bundle['mask_bundle']['result']['objects'][0]['plane_status'] = 'NO_PLANE'
    calls = []
    def observed(_):
        calls.append(True)
        if kind == 'missing': return {'mock': True}
        return expected if len(calls) == 1 else {
            'mock': True, 'dependency_overlay': {'tree_sha256': 'changed'}}
    monkeypatch.setattr(fill, 'public_runtime', observed)
    with pytest.raises(ValueError, match='public fill runtime changed'):
        fill.run_public(context, contract)
    assert not list(root.rglob('clean_background.ply'))


@pytest.mark.parametrize('kind',['failed_erasure','seal_identity','TEST_camera','pose_drift','plane_nan','source_gaussian','context_drift'])
def test_invalid_sources_fail_closed(setup,kind):
    context,contract,bundle,tmp=setup
    if kind=='failed_erasure':bundle['result']['status']='FAILED'
    elif kind=='seal_identity':bundle['seal_identity']['sha256']='0'*64
    elif kind=='TEST_camera':write(tmp/'prep/obj_1000/views.json',[dict(frame='TEST.png',w2c=np.eye(4).tolist())])
    elif kind=='pose_drift':write(tmp/'prep/obj_1000/views.json',[dict(frame='TRAIN.png',w2c=(np.eye(4)*2).tolist())])
    elif kind=='plane_nan':
        plane=json.loads((tmp/'prep/obj_1000/plane.json').read_text());plane['origin'][0]=float('nan');write(tmp/'prep/obj_1000/plane.json',plane)
    elif kind=='source_gaussian':(tmp/'source.ply').write_bytes(b'changed')
    else:
        c=json.loads(context.read_text());c['algorithm']['iterations']=1499;write(context,c)
    with pytest.raises(ValueError):fill._validate_public_context(context,contract)


@pytest.mark.parametrize('reason',['NO_PLANE','NO_VIEW','PRIMARY_VIEW_UNAVAILABLE'])
def test_unfillable_accepted_objects_are_explicit_not_raw_fallback(setup,monkeypatch,reason):
    context,contract,bundle,tmp=setup
    state=bundle['mask_bundle']['result']['objects'][0]
    if reason=='NO_PLANE':state['plane_status']=reason
    elif reason=='NO_VIEW':state['view_status']=reason
    else:bundle['result']['rows'][0]['erasure_status']='EMPTY_MASK'
    monkeypatch.setattr(fill,'_fill',lambda _:pytest.fail('unfillable object entered optimizer'))
    result=fill.run_public(context,contract)
    output=tmp/'outputs/icra2027/new/fidelity/background_fill/0000000001'
    assert result['status']=='BLOCKED_UNFILLABLE_ACCEPTED' and not result['cleaned_background_created']
    assert result['planned_objects']==result['accepted_objects']==1 and result['planned_views']==1
    assert not (output/'clean_background.ply').exists()
    assert result['blocked'][0]['reason']==reason
    checked=fill.validate_public_fill(output)
    assert checked['clean_background'] is None


def test_no_accepted_removal_is_exact_source_identity(setup,monkeypatch):
    context,contract,bundle,tmp=setup
    bundle['mask_bundle']['result']['objects'][0]['terminal_action']='abstain'
    bundle['result']['rows']=[];bundle['mask_bundle']['rows']=[]
    np.save(tmp/'prep/removal_union_idx.npy',np.array([],dtype=np.int64))
    monkeypatch.setattr(fill,'_fill',lambda _:pytest.fail('zero accepted objects entered optimizer'))
    result=fill.run_public(context,contract)
    path=tmp/'outputs/icra2027/new/fidelity/background_fill/0000000001/clean_background.ply'
    assert result['status']=='NO_REMOVAL' and result['clean_background_is_unchanged_source']
    assert path.read_bytes()==(tmp/'source.ply').read_bytes() and result['planned_objects']==1
    assert fill.validate_public_fill(path.parent)['clean_background']==identity(path)


def test_midrun_input_drift_retains_failure_without_seal(setup,monkeypatch):
    context,contract,_,tmp=setup
    def drift(p):
        (p['output']/'partial.bin').write_bytes(b'partial')
        (tmp/'source.ply').write_bytes(b'changed')
        return {}
    monkeypatch.setattr(fill,'_fill',drift)
    monkeypatch.setattr(fill,'_validate_fill_products',lambda *a:None)
    with pytest.raises(ValueError):fill.run_public(context,contract)
    output=tmp/'outputs/icra2027/new/fidelity/background_fill/0000000001'
    assert not output.exists() and output.with_name(output.name+'.failed_partial').is_dir()
    failure=json.loads(output.with_name(output.name+'.failure.json').read_text())
    assert failure['planned_objects']==failure['planned_views']==1 and failure['status']=='FAILED'
    (tmp/'source.ply').write_bytes(b'original authenticated Gaussian')
    with pytest.raises(FileExistsError):fill.run_public(context,contract)


@pytest.mark.parametrize('suffix',['','.claim.json','.failure.json','.failed_partial'])
def test_existing_outputs_and_orphan_failures_cannot_be_overwritten(setup,suffix):
    context,contract,_,tmp=setup
    output=tmp/'outputs/icra2027/new/fidelity/background_fill'/('0000000001'+suffix)
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):fill.run_public(context,contract)


def cpu_fill_runtime(monkeypatch):
    def rasterization(**kw):
        color=torch.sigmoid(kw['colors'][:,0,:].mean(dim=0))
        return color[None,None,None,:].expand(1,kw['height'],kw['width'],3),None,None
    monkeypatch.setitem(sys.modules,'gsplat',types.SimpleNamespace(rasterization=rasterization))
    for name in ('tensor','ones','zeros','full'):
        original=getattr(torch,name)
        def call(*a,_original=original,**kw):
            if kw.get('device')=='cuda':kw['device']='cpu'
            return _original(*a,**kw)
        monkeypatch.setattr(torch,name,call)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*a,**k:self)
    original_to=torch.Tensor.to
    def to(self,*a,**kw):
        a=tuple('cpu' if isinstance(x,str) and x=='cuda' else x for x in a)
        return original_to(self,*a,**kw)
    monkeypatch.setattr(torch.Tensor,'to',to)
    n=20
    gs=dict(means=torch.tensor(np.column_stack([np.linspace(-.2,.2,n),np.zeros(n),np.ones(n)]),dtype=torch.float32),
        quats=torch.tensor([[1,0,0,0]]*n,dtype=torch.float32),scales=torch.full((n,3),.01),
        opacities=torch.full((n,),.8),sh=torch.zeros((n,1,3)),sh_degree=0)
    monkeypatch.setattr(fill.C,'load_gaussians',lambda _:gs)
    monkeypatch.setattr(fill.C,'render_view',lambda *a:(np.full((8,8,3),.5),None,None))


def test_existing_full1500_step_recipe_runs_with_cpu_renderer_double(setup,monkeypatch):
    context,contract,_,tmp=setup;cpu_fill_runtime(monkeypatch)
    result=fill.run_public(context,contract)
    stats=result['construction_diagnostics']
    output=tmp/'outputs/icra2027/new/fidelity/background_fill/0000000001'
    assert stats['iterations']==1500 and stats['primary_frames']==['TRAIN.png']
    assert stats['removal_gaussians']==1 and stats['fill_gaussians']>0 and stats['filled_objects']==1
    from plyfile import PlyData
    vertex=PlyData.read(output/'clean_background.ply')['vertex']
    assert len(vertex)==stats['source_gaussians']-stats['removal_gaussians']+stats['fill_gaussians']
    assert all(np.isfinite(vertex[name]).all() for name in vertex.data.dtype.names)
    summary=json.loads((output/'fill_stats_summary.json').read_text())
    assert summary['scope']=='TRAIN_erasure_target_diagnostic' and summary['independent_evaluation'] is False
    assert fill.validate_public_fill(output)['result']['accepted_objects']==1


@pytest.mark.parametrize('kind',['horizon','dropped_object'])
def test_output_cannot_promote_incomplete_fill(kind,tmp_path):
    diagnostics=dict(iterations=1500,filled_objects=1)
    if kind=='horizon':diagnostics['iterations']=1
    else:diagnostics['filled_objects']=0
    with pytest.raises(ValueError):fill._validate_fill_products(tmp_path,diagnostics,1)


def test_runtime_drift_retains_failure_roster(setup,monkeypatch):
    context,contract,_,tmp=setup
    monkeypatch.setattr(fill,'public_runtime',lambda _:{'changed':True})
    monkeypatch.setattr(fill,'_fill',lambda _:pytest.fail('drifted runtime entered optimizer'))
    with pytest.raises(ValueError,match='runtime changed'):fill.run_public(context,contract)
    failure=tmp/'outputs/icra2027/new/fidelity/background_fill/0000000001.failure.json'
    assert json.loads(failure.read_text())['planned_objects']==1


def test_no_removal_copy_corruption_is_not_promoted(setup,monkeypatch):
    import shutil
    context,contract,bundle,tmp=setup
    bundle['mask_bundle']['result']['objects'][0]['terminal_action']='abstain'
    bundle['result']['rows']=[];bundle['mask_bundle']['rows']=[]
    np.save(tmp/'prep/removal_union_idx.npy',np.array([],dtype=np.int64))
    def corrupt(source,destination):Path(destination).write_bytes(b'x'*Path(source).stat().st_size)
    monkeypatch.setattr(shutil,'copyfile',corrupt)
    with pytest.raises(ValueError,match='copied Gaussian differs'):fill.run_public(context,contract)
    output=tmp/'outputs/icra2027/new/fidelity/background_fill/0000000001'
    assert not (output/'seal.json').exists()
    assert json.loads(output.with_name(output.name+'.failure.json').read_text())['status']=='FAILED'


@pytest.mark.parametrize('kind',['population','promoted','artifact','unsealed'])
def test_fill_consumer_rejects_resealed_false_claims(setup,kind):
    context,contract,bundle,tmp=setup
    bundle['mask_bundle']['result']['objects'][0]['plane_status']='NO_PLANE'
    fill.run_public(context,contract)
    output=tmp/'outputs/icra2027/new/fidelity/background_fill/0000000001'
    path=output/'public_fill.json';result=json.loads(path.read_text())
    if kind=='population':result['planned_objects']=0
    elif kind=='promoted':result['status']='COMPLETE';result['cleaned_background_created']=True
    elif kind=='artifact':result['artifacts']['fake.ply']={'path':'fake'}
    else:(output/'unsealed.txt').write_text('extra')
    write(path,result)
    seal=json.loads((output/'seal.json').read_text())
    seal['members']['public_fill.json']=identity(path)['sha256'];write(output/'seal.json',seal)
    with pytest.raises((ValueError,KeyError)):fill.validate_public_fill(output)
