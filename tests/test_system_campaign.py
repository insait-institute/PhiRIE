"""CPU contract and numerical tests. Not GPU-model or policy performance tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from PIL import Image
from robo.campaign.core import save,load,receipt,digest,local_model
from robo.campaign.data import validate_inventory,freeze,catalog,gallery
from robo.campaign.runner import validate_tasks,prepare,worker,launch,collect
from robo.campaign.models import masked_composite,reserve_remote,inpaint
from robo.campaign.semantics import instances,normalized
from robo.campaign.visual import backward_warp,compose,protected_mask,transform_points
from robo.campaign.assets import factor_affine,bridge


def scene(**changes):
    return dict(dataset='robocasa',release='pinned',scene_id='s1',group_id='layout1',split='test',
                source_index='official index',license_checked=True,**changes)


def test_save_atomic_no_overwrite(tmp_path):
    p=tmp_path/'x.json';save(p,{'x':1})
    with pytest.raises(FileExistsError):save(p,{'x':2})
    assert load(p)=={'x':1}


def test_inventory_family_leak():
    a=scene();b={**a,'scene_id':'s2','split':'train'}
    with pytest.raises(ValueError):validate_inventory([a,b])


def test_inventory_cross_dataset_leak():
    a={**scene(),'physical_scene_id':'home1'};b={**a,'dataset':'scannetpp','split':'dev'}
    with pytest.raises(ValueError):validate_inventory([a,b])


def test_inventory_does_not_use_scores():
    with pytest.raises(ValueError):validate_inventory([{**scene(),'success':.9}])


def test_scene_is_not_policy_ready():
    with pytest.raises(ValueError):validate_inventory([{**scene(),'policy_ready':True}])


def test_freeze_reproducible(tmp_path):
    p=tmp_path/'scenes.jsonl';save(p,[{**scene(),'scene_id':f's{i}'} for i in range(6)],jsonl=True)
    a=freeze(p,tmp_path/'a',counts={'robocasa':{'test':3}})
    b=freeze(p,tmp_path/'b',counts={'robocasa':{'test':3}})
    assert a==b and len(a)==3


def test_freeze_insufficient(tmp_path):
    p=tmp_path/'i.jsonl';save(p,[scene()],jsonl=True)
    with pytest.raises(ValueError):freeze(p,tmp_path/'a',counts={'robocasa':{'test':2}})


def test_gallery(tmp_path):
    im=tmp_path/'im.png';Image.new('RGB',(20,20)).save(im)
    p=tmp_path/'i.jsonl';save(p,[{**scene(),'thumbnail':str(im)}],jsonl=True)
    gallery(p,tmp_path/'gallery');assert (tmp_path/'gallery/index.html').exists()


def task(tid='a',**kw):
    return dict(id=tid,kind='command',environment='cpu',split='dev',sensor='not_applicable',
                params={'argv':['echo','test'],'output_manifest':'{out}/done.json'},inputs={},**kw)


def test_cycles_rejected():
    a,b=task('a'),task('b');a['inputs']={'x':{'task':'b','artifact':'image'}};b['inputs']={'x':{'task':'a','artifact':'image'}}
    with pytest.raises(ValueError):validate_tasks([a,b])


def test_rgb_gt_leak(tmp_path):
    p=tmp_path/'x';p.write_text('x');a=task();a.update(sensor='rgb_video',scale_source='known_robot',camera_source='estimated')
    a['inputs']={'x':{**receipt(p),'role':'gt_depth'}}
    with pytest.raises(ValueError):validate_tasks([a])


def test_raw_video_with_known_cameras_rejected():
    a=task();a.update(sensor='rgb_video',scale_source='estimated',camera_source='reference')
    with pytest.raises(ValueError):validate_tasks([a])


def test_native_requires_gates():
    a=task();a['params']['purpose']='native_policy'
    with pytest.raises(ValueError):validate_tasks([a])


def test_real_command_worker_and_no_repeat(tmp_path):
    source=tmp_path/'repo';source.mkdir();subprocess.run(['git','init',str(source)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=test','-c','user.email=test@example.com','commit','--allow-empty','-m','init'],check=True,capture_output=True)
    runtime={'execution_ready':True,'source_root':str(source),'environments':{'cpu':{'python':sys.executable,'sbatch_args':[]}}}
    code="from pathlib import Path;import json,hashlib; p=Path(__import__('sys').argv[1]);q=p/'value.txt';q.write_text('measured-artifact');(p/'done.json').write_text(json.dumps(dict(artifacts=dict(value=dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest())))))"
    a=task();a['params']['argv']=[sys.executable,'-c',code,'{out}']
    rp=tmp_path/'r.json';tp=tmp_path/'t.jsonl';save(rp,runtime);save(tp,[a],jsonl=True)
    root=tmp_path/'bundle';prepare(tp,rp,root)
    assert len(launch(root,submit=False))==1
    assert not (root/'jobs').exists()
    result=worker(root,'a');assert result['status']=='COMPLETE'
    with pytest.raises(FileExistsError):worker(root,'a')
    assert collect(root,tmp_path/'col')['all_complete']


def test_model_manifest(tmp_path):
    p=tmp_path/'weights';p.mkdir();(p/'w').write_text('weight');m=tmp_path/'m.json'
    save(m,{'files':{'w':receipt(p/'w')['sha256']}})
    assert local_model({'checkpoint':str(p),'checkpoint_manifest':str(m)})==str(p)
    (p/'w').write_text('bad')
    with pytest.raises(ValueError):local_model({'checkpoint':str(p),'checkpoint_manifest':str(m)})


def test_mask_preservation():
    a=np.zeros((5,5,3),np.uint8);b=np.full_like(a,255);m=np.zeros((5,5),bool);m[2,2]=1
    c=masked_composite(a,b,m);assert np.array_equal(c[~m],a[~m]) and (c[m]==255).all()


def test_mask_bad_grid():
    with pytest.raises(ValueError):masked_composite(np.zeros((3,3,3)),np.zeros((4,4,3)),np.ones((3,3)))


def test_api_budget_and_repeat(tmp_path):
    a=tmp_path/'approval.json';save(a,{'approved':True,'data_upload_permitted':True,'models':['gemini-x'],
          'reserved_usd_per_call':.2,'max_calls':1,'max_usd':1,'ledger':str(tmp_path/'cost.jsonl')})
    c={'approval_file':str(a),'model_id':'gemini-x'};reserve_remote(c,'r1')
    with pytest.raises(RuntimeError):reserve_remote(c,'r1')
    with pytest.raises(PermissionError):reserve_remote(c,'r2')


def test_api_not_approved(tmp_path):
    p=tmp_path/'p';save(p,{'approved':False})
    with pytest.raises(PermissionError):reserve_remote({'approval_file':str(p),'model_id':'x'},'r')


def test_semantic_components_ids():
    xyz=np.array([[0,0,0],[.01,0,0],[1,0,0],[1.01,0,0]])
    f=np.array([[1.,0]]*4); ids=np.array([101,202,303,404])
    r=instances(xyz,f,[1,0],ids,min_size=2,radius=.03)
    assert sorted(r[0]['source_gaussian_ids']+r[1]['source_gaussian_ids'])==[101,202,303,404]
    assert len(r)==2


def test_empty_query():
    assert instances(np.zeros((1,3)),[[1,0]],[-1,0],[10],min_size=1)==[]


def test_zero_feature_rejected():
    with pytest.raises(ValueError):normalized([[0,0]])


def warp_inputs():
    rgb=np.arange(5*5*3,dtype=np.uint8).reshape(5,5,3);depth=np.ones((5,5));ids=np.zeros((5,5),int)
    K=np.eye(3);T=np.eye(4)
    return rgb,depth,ids,K,T


def test_warp_identity():
    rgb,d,ids,K,T=warp_inputs();w,v=backward_warp(rgb,d,ids,d,ids,K,K,T,T,{}, {})
    assert v.all() and np.array_equal(w,rgb)


def test_warp_occlusion_reject():
    rgb,d,ids,K,T=warp_inputs();w,v=backward_warp(rgb,d*.5,ids,d,ids,K,K,T,T,{}, {})
    assert not v.any()


def test_warp_instance_mismatch():
    rgb,d,ids,K,T=warp_inputs();_,v=backward_warp(rgb,d,ids+2,d,ids,K,K,T,T,{}, {})
    assert not v.any()


def test_moving_object_warp():
    rgb,d,ids,K,T=warp_inputs();ids[:]=1;current=T.copy();current[0,3]=1
    w,v=backward_warp(rgb,d,ids,d,ids,K,K,T,T,{'1':current},{'1':T})
    assert not v[:,0].any() and v[:,1:].all() and np.array_equal(w[:,1:],rgb[:,:-1])


def test_unknown_motion_invalid():
    rgb,d,ids,K,T=warp_inputs();ids[:]=1
    _,v=backward_warp(rgb,d,ids,d,ids,K,K,T,T,{}, {})
    assert not v.any()


def test_bad_transform():
    T=np.eye(4);T[0,0]=2
    with pytest.raises(ValueError):transform_points(T,np.zeros((1,3)))


def test_protection_and_bound():
    a=np.zeros((4,4,3),np.uint8);b=np.full_like(a,255);p=np.zeros((4,4),bool);p[1,1]=1
    r=compose(a,b,p,np.ones((4,4)),max_residual=20,strength=.5)
    assert (r[1,1]==0).all() and r.max()==10


def test_thin_robot_protected():
    ids=np.zeros((5,5),int);robot=np.zeros((5,5),bool);robot[:,2]=1
    assert protected_mask(ids,robot,np.zeros_like(robot))[:,2].all()


def test_bridge_affine_world_equivalent():
    T=np.eye(4);T[:3,:3]=np.diag([2,3,4]);T[:3,3]=[1,2,3];R,S=factor_affine(T)
    x=np.random.default_rng(1).normal(size=(20,3))
    assert np.allclose(x@T[:3,:3].T+T[:3,3],(x@S.T)@R[:3,:3].T+R[:3,3])


def test_bridge_actual_mesh(tmp_path):
    import trimesh
    mesh=tmp_path/'m.obj';trimesh.creation.box().export(mesh)
    pl=tmp_path/'placement.json';ph=tmp_path/'physics.json'
    save(pl,{'source':'method_estimated','uses_reference_pose':False,'T_world_from_asset':np.eye(4).tolist()})
    save(ph,{'mass_kg':.1,'friction':.6});out=tmp_path/'asset';out.mkdir()
    bridge({'collision_mode':'preserve_supplied','collision_inputs':['collision'],'method_label':'official-adapted'},
           {'mesh':str(mesh),'collision':str(mesh),'placement':str(pl),'physics':str(ph)},out)
    assert load(out/'aligned.json')['scale']==1 and (out/'collision/part_000.obj').exists()


def test_trainable_mask_identity():
    import torch
    from robo.campaign.learned import StateResidual
    torch.set_num_threads(1);model=StateResidual(width=4);x=torch.rand(1,16,8,8);x[:,15]=1
    output=model(x);assert torch.equal(output,x[:,:3])
    output.mean().backward()


def test_training_leak_rejected():
    from robo.campaign.learned import validate_pairs
    r={'sample_id':'a','scene_group':'g','split':'train','input_state_sha256':'a','target_state_sha256':'b','physical_geometry_equal':True}
    with pytest.raises(ValueError):validate_pairs([r],[{**r,'sample_id':'b','split':'dev'}])


def test_missing_requested_dataset_is_not_silently_skipped(tmp_path):
    p=tmp_path/'inventory.jsonl';save(p,[scene()],jsonl=True)
    with pytest.raises(ValueError):freeze(p,tmp_path/'a',counts={'behavior':{'test':1}})


def test_single_checkpoint_manifest_must_name_file(tmp_path):
    p=tmp_path/'w.pt';p.write_text('w');m=tmp_path/'m.json';save(m,{'files':{'wrong.pt':receipt(p)['sha256']}})
    with pytest.raises(ValueError):local_model({'checkpoint':str(p),'checkpoint_manifest':str(m)})


def test_hidden_mesh_rejected_for_constructor(tmp_path):
    p=tmp_path/'mesh';p.write_text('x');t=task();t.update(kind='generator',sensor='ideal_rgbd')
    t['inputs']={'x':{**receipt(p),'role':'gt_mesh'}}
    with pytest.raises(ValueError):validate_tasks([t])


def test_matrix_actual_bindings_and_no_cartesian_policy(tmp_path):
    from robo.campaign.matrix import build
    im=tmp_path/'im.png';Image.new('RGB',(16,16)).save(im)
    inventory=tmp_path/'scenes.jsonl';save(inventory,[{**scene(),'bindings':{'objects':[{'object_id':'mug','images':[receipt(im)]}]}}],jsonl=True)
    model=tmp_path/'model.json';save(model,{'generators':{'t':{'enabled':True,'environment':'t','params':{'backend':'trellis'}}}})
    build(inventory,model,tmp_path/'matrix',stages=['generator'])
    from robo.campaign.core import rows
    tasks=rows(tmp_path/'matrix/tasks.jsonl');assert len(tasks)==1 and tasks[0]['params']['images']==['image0']


def test_video_requires_train_capture(tmp_path):
    from robo.campaign.video import run
    # Failure occurs before calling external ffmpeg/COLMAP/train.
    source=tmp_path/'src';source.mkdir();subprocess.run(['git','init',str(source)],check=True,capture_output=True)
    subprocess.run(['git','-C',str(source),'-c','user.name=t','-c','user.email=t@example.com','commit','--allow-empty','-m','init'],check=True,capture_output=True)
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    with pytest.raises(ValueError):run({'source_root':str(source),'source_commit':commit,'capture_role':'heldout'}, {}, tmp_path)


def test_learned_inference_actual_checkpoint(tmp_path):
    import torch
    from robo.campaign.learned import StateResidual,predict
    model=StateResidual(width=4)
    p=tmp_path/'weights.pt';torch.save({'state_dict':model.state_dict(),'width':4,'bound':32/255},p)
    m=tmp_path/'manifest.json';save(m,{'files':{p.name:receipt(p)['sha256']}})
    a=np.full((8,8,3),123,np.uint8);p_mask=np.eye(8,dtype=bool)
    fields={'depth':np.ones((8,8)),'normals':np.zeros((8,8,3)),'confidence':np.ones((8,8))}
    result=predict({'checkpoint':str(p),'checkpoint_manifest':str(m),'device':'cpu'},a,a,fields,p_mask)
    assert np.array_equal(result,a)


def test_asset_output_collision_closure(tmp_path):
    import trimesh
    mesh=tmp_path/'m.obj';trimesh.creation.box().export(mesh)
    pl=tmp_path/'p.json';ph=tmp_path/'ph.json'
    save(pl,{'source':'method_estimated','uses_reference_pose':False,'T_world_from_asset':np.eye(4).tolist()});save(ph,{'mass_kg':1,'friction':.5})
    out=tmp_path/'out';out.mkdir()
    bridge({'collision_mode':'preserve_supplied','collision_inputs':['part'],'method_label':'baseline-adapted'},
           {'mesh':str(mesh),'part':str(mesh),'placement':str(pl),'physics':str(ph)},out)
    assert 'collision/part_000.obj' in load(out/'asset.json')['output_assets']


def test_color_cache_actual_sparse_solver():
    from robo.campaign.color_cache import solve_residual
    expected=np.array([[.02,.03,.04],[-.01,0,.01]])
    delta,diag=solve_residual(np.array([0,1]),np.array([0,1]),np.ones(2),expected,np.ones(2),2,ridge=1e-8)
    assert np.allclose(delta,expected,atol=1e-6) and len(diag)==3


def test_color_cache_no_confidence_no_update():
    from robo.campaign.color_cache import solve_residual
    delta,_=solve_residual(np.array([0]),np.array([0]),np.ones(1),np.ones((1,3)),np.zeros(1),1)
    assert np.array_equal(delta,np.zeros((1,3)))


def test_color_cache_invalid_responsibilities():
    from robo.campaign.color_cache import solve_residual
    with pytest.raises(ValueError):solve_residual(np.array([0]),np.array([0]),np.array([2.]),np.ones((1,3)),np.ones(1),1)
