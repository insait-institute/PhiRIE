#!/usr/bin/env python3
"""Bounded initial RVG proposals via existing training-view collection/run_rvg."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml
CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE))
from run.icra2027.e3_trellis_generation_pilot import context,source_jobs,validate_models,runtime_identity,cohort_models
from run.icra2027.e3_auto_discovery_pilot import sha,identity,write_new,PilotError,validate_gpu_memory,validate_staged_inputs,enforce_read_boundary
from run.icra2027.e3_fresh_generation_contract import discovery_binding,FRESH

CONSTRUCTION_INPUTS=('derived_mesh.ply','auto_instances.npz','objects/objects.json')


def scene_id(c):
    return c.get('scene_id','09c1414f1b')


def output_directory(c, root):
    """Keep each complete scene denominator isolated under one shared freeze."""
    out = Path(root) / 'rvg_initial'
    scope = c.get('output_scene_id')
    if scope is not None:
        if not re.fullmatch(r'[a-f0-9]{10}', str(scope)) or scope != scene_id(c):
            raise PilotError('RVG output scene differs from frozen source scene')
        out = out / scope
    return out


def source_boundary(c):
    if c.get('raw_reuse') or c.get('reuse_source'):
        raise PilotError('RVG requires actual multiview production; raw reuse is unsupported')
    source=Path(c['source_pilot'])
    for name,key in [('pilot_summary.json','source_summary_sha256'),('output_hashes.json','source_output_hashes_sha256'),('input_manifest.json','source_input_manifest_sha256')]:
        if sha(source/name)!=c[key]:raise PilotError(f'source identity differs: {name}')
    rows,binding=discovery_binding(c,source_jobs)
    boundary=json.loads((source/'input_manifest.json').read_text())
    if boundary.get('scene_id','09c1414f1b')!=scene_id(c):raise PilotError('RVG source scene differs')
    return rows,boundary,binding


def env_for(c,out):
    model=c['models']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',PYTHONPATH=str(CODE),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        SIMANY_ROOT=str(CODE),SIMANY_OUT=str(out/'construction'),SIMANY_SCENE=scene_id(c),SIMANY_AUTO='1',SIMANY_MESH_SRC='derived',SIMANY_NO_GT='1',
        SIMANY_SCANNETPP_ROOT=str(out/'inputs'),SIMANY_RVG_DIR=model['rvg_source']['path'],SIMANY_RVG_PINNED_CONFIG=str(out/'pinned_models.json'),
        TORCH_HOME=str(Path(model['dinov2_source']['path']).parent.parent))
    env.pop('PYTHONHOME',None)
    return env


def read_manifest(c,out):
    rows,boundary,binding=source_boundary(c)
    m=json.loads((out/'input_manifest.json').read_text())
    if m['jobs']!=rows:raise PilotError('RVG jobs differ from config-bound complete source')
    for key,value in binding.items():
        if m.get(key)!=value:raise PilotError('RVG staged source provenance differs')
    validate_staged_inputs(out,{'scene_id':scene_id(c)},boundary)
    source=Path(c['source_pilot']);hashes=json.loads((source/'output_hashes.json').read_text())
    for name in CONSTRUCTION_INPUTS:
        if sha(out/'construction'/name)!=hashes[name]['sha256']:raise PilotError(f'staged construction input changed: {name}')
    if json.loads((out/'pinned_models.json').read_text())!={'models':c['models']}:
        raise PilotError('pinned loader mapping differs')
    return m,boundary


def setup_read_guard(c,out,boundary):
    source_scene=(Path('/data/ScanNetpp/data')/scene_id(c)).resolve()
    allowed=set(boundary['boundary']['training_frames'])
    def guard(event,args):
        if event=='socket.connect':raise PilotError('offline pinned generation forbids network access')
        enforce_read_boundary(event,args,forbidden_scene=source_scene,
            image_root=(source_scene/'dslr/resized_undistorted_images').resolve(),allowed_images=allowed)
    sys.addaudithook(guard)


def freeze_object_views(out,idx,views,boundary,W,H,image_root,*,write):
    """Hash the actual RGB/mask inputs; selection stays with collect_views."""
    import numpy as np
    from PIL import Image
    if len(views)>12 or len({v[0] for v in views})!=len(views):
        raise PilotError('RVG collected view count/uniqueness differs')
    records=[]
    for k,(name,bbox,mask) in enumerate(views):
        if name not in boundary['boundary']['training_frames']:
            raise PilotError('RVG view outside frozen training boundary')
        image=Path(image_root)/name
        expected=boundary['input_images'][name]
        if image.resolve()!=Path(expected['path']) or sha(image)!=expected['sha256']:
            raise PilotError('RVG actual view image source differs')
        u0,v0,u1,v1=map(int,bbox);mask=np.asarray(mask)
        if (list(bbox)!=[u0,v0,u1,v1] or not (0<=u0<u1<=W and 0<=v0<v1<=H)
                or mask.dtype!=np.bool_ or mask.shape!=(v1-v0,u1-u0)):
            raise PilotError('RVG collected bbox/mask schema differs')
        rgb=np.asarray(Image.open(image).convert('RGB'))
        if rgb.shape!=(H,W,3):raise PilotError('RVG RGB/camera dimensions differ')
        rgba=np.dstack([rgb[v0:v1,u0:u1],(mask*255).astype(np.uint8)])
        mask_path=out/'frozen_views'/f'obj_{idx:02d}'/f'mask_{k:02d}.npy'
        if write:
            mask_path.parent.mkdir(parents=True,exist_ok=True)
            with mask_path.open('xb') as f:np.save(f,mask,allow_pickle=False)
        elif not np.array_equal(np.load(mask_path,allow_pickle=False),mask):
            raise PilotError('RVG view mask differs from source-derived recomputation')
        records.append(dict(frame=name,bbox=[u0,v0,u1,v1],mask_path=str(mask_path),
            mask_sha256=sha(mask_path),image_path=str(image.resolve()),image_sha256=sha(image),
            rgba_shape=list(rgba.shape),rgba_pixels_sha256=hashlib.sha256(rgba.tobytes()).hexdigest()))
    return records


def view_source_binding(c,boundary):
    source=Path(c['source_pilot']);hashes=json.loads((source/'output_hashes.json').read_text())
    return dict(source_input_manifest_sha256=c['source_input_manifest_sha256'],
        source_gaussian_training_provenance=c.get('source_gaussian_training_provenance','UNKNOWN'),
        source_discovery_hashes=c.get('source_discovery_hashes',{}),
        construction_hashes={name:hashes[name]['sha256'] for name in CONSTRUCTION_INPUTS},
        camera_and_split_hashes={name:record['sha256'] for name,record in boundary['metadata'].items()})


def verify_consumed_views(out,idx,expected):
    import numpy as np
    from PIL import Image
    directory=out/'construction/objects'/f'obj_{idx:02d}/rvg'
    paths=sorted(directory.glob('view_*.png'))
    if [p.name for p in paths]!=[f'view_{i:02d}.png' for i in range(len(expected))]:
        raise PilotError('RVG consumed PNG view population differs')
    records=[]
    for path,view in zip(paths,expected):
        rgba=np.asarray(Image.open(path).convert('RGBA'))
        if list(rgba.shape)!=view['rgba_shape'] or hashlib.sha256(rgba.tobytes()).hexdigest()!=view['rgba_pixels_sha256']:
            raise PilotError('RVG consumed RGB/mask differs from frozen actual inputs')
        records.append(dict(frame=view['frame'],source_image_sha256=view['image_sha256'],
            source_mask_sha256=view['mask_sha256'],rgba_pixels_sha256=view['rgba_pixels_sha256'],png=identity(path)))
    return records


def collect_frozen_views(c,out,boundary,*,write):
    if "frozen_view_replay" in c:
        from run.icra2027.e3_rvg_frozen_views import collect
        return collect(c,out,boundary,write=write)
    import numpy as np
    from PIL import Image
    from agents.core import common as C
    from agents.assets.s5_align import scene_mesh_arrays
    from models.s4_reconviagen import collect_views
    from agents.discover.training_views import select_training_views
    K,W,H,_=C.load_intrinsics()
    frames,training=select_training_views(sorted(C.load_colmap_w2c().items()),out/'inputs/data'/scene_id(c)/'dslr/train_test_lists.json',1,boundary['boundary'].get('max_train_frames',48))
    if training['training_frames']!=boundary['boundary']['training_frames']:raise PilotError('RVG training-view population changed')
    verts,faces=scene_mesh_arrays();scene=C.make_raycast_scene()
    instances={x['object_id']:x for x in C.load_instances()}
    objects=json.loads((out/'construction/objects/objects.json').read_text())
    rows=[];view_data={}
    for obj in objects:
        idx=obj['index'];views=collect_views(instances[obj['gt_object_id']],K,W,H,frames,scene,verts,faces,12)
        data=freeze_object_views(out,idx,views,boundary,W,H,C.IMAGES_DIR,write=write)
        rows.append({'object_index':idx,'automatic_instance_id':obj['gt_object_id'],'views':data,'available':len(views)>=2})
        view_data[idx]=(obj,instances[obj['gt_object_id']],views)
    jobs=json.loads((out/'input_manifest.json').read_text())['jobs']
    record={'training_frames':training['training_frames'],'n_views_max':12,'rows':rows,
        'all_jobs':[{'job_id':j['job_id'],'prepared':j['prepared'],'output_index':j.get('output_index')} for j in jobs],
        **view_source_binding(c,boundary)}
    if write:write_new(out/'view_manifest.json',record)
    elif record!=json.loads((out/'view_manifest.json').read_text()):raise PilotError('RVG view manifest differs from source-derived recomputation')
    return (K,W,H,frames,scene,verts,faces),view_data,record


def validate_view_receipt(c,out,m,boundary):
    receipt=json.loads((out/'views_receipt.json').read_text())
    expected=dict(status='PASS',input_manifest=identity(out/'input_manifest.json'),
                  view_manifest=identity(out/'view_manifest.json'),
                  source_binding=view_source_binding(c,boundary),paper_ready=False)
    if receipt!=expected:raise PilotError('RVG CPU view receipt differs')
    record=json.loads((out/'view_manifest.json').read_text())
    for key,value in expected['source_binding'].items():
        if record.get(key)!=value:raise PilotError('RVG multiview source binding differs')
    if record['all_jobs']!=[{'job_id':j['job_id'],'prepared':j['prepared'],'output_index':j.get('output_index')} for j in m['jobs']]:
        raise PilotError('RVG view denominator differs')
    prepared=[j for j in m['jobs'] if j['prepared']]
    if sorted((r['object_index'],r['automatic_instance_id']) for r in record['rows'])!=sorted(
            (j['output_index'],j['automatic_instance_id']) for j in prepared):
        raise PilotError('RVG prepared view population differs')
    if record['training_frames']!=boundary['boundary']['training_frames'] or record['n_views_max']!=12:
        raise PilotError('RVG view training population/recipe differs')
    for row in record['rows']:
        if len(row['views'])>12 or row['available']!=(len(row['views'])>=2):
            raise PilotError('RVG actual view availability differs')
        for k,view in enumerate(row['views']):
            if view['frame'] not in boundary['boundary']['training_frames']:
                raise PilotError('RVG view outside frozen training boundary')
            source=boundary['input_images'][view['frame']]
            mask=out/'frozen_views'/f'obj_{row["object_index"]:02d}'/f'mask_{k:02d}.npy'
            if (view['image_path']!=source['path'] or view['image_sha256']!=source['sha256']
                    or sha(view['image_path'])!=source['sha256'] or view['mask_path']!=str(mask)
                    or mask.is_symlink() or sha(mask)!=view['mask_sha256']):
                raise PilotError('RVG frozen actual RGB/mask identity differs')
    return record


def views_worker(config_path,root,generate=False,verify_only=False):
    c,git=context(config_path,root);out=output_directory(c,root);m,boundary=read_manifest(c,out)
    if m['code_commit']!=git or m['config_sha256']!=sha(config_path):raise PilotError('RVG worker plan source/config differs')
    env=env_for(c,out)
    for key in ['SIMANY_ROOT','SIMANY_OUT','SIMANY_SCENE','SIMANY_AUTO','SIMANY_MESH_SRC','SIMANY_SCANNETPP_ROOT','SIMANY_RVG_DIR','SIMANY_RVG_PINNED_CONFIG']:
        if os.environ.get(key)!=env[key]:raise PilotError('RVG worker environment differs')
    setup_read_guard(c,out,boundary)
    ctx,objects,views=collect_frozen_views(c,out,boundary,write=not (generate or verify_only))
    if verify_only:
        validate_view_receipt(c,out,m,boundary)
        return
    if not generate:return
    validate_view_receipt(c,out,m,boundary)
    import torch
    from agents.assets.factory_hybrid import run_rvg
    holder={'pipe':None}
    for idx,(obj,instance,frozen_views) in objects.items():
        if len(frozen_views)<2:
            write_new(out/'producer_records'/f'object_{idx:02d}.json',dict(status='unavailable',reason='fewer_than_two_usable_training_views',views=len(frozen_views)))
            continue
        odir=out/'construction/objects'/f'obj_{idx:02d}';odir.mkdir(exist_ok=True)
        if (odir/'rvg').exists() or (out/'producer_records'/f'object_{idx:02d}.json').exists():
            raise PilotError('fresh RVG proposals refuse existing artifacts or producer records')
        start=time.monotonic()
        try:
            wall=run_rvg(holder,obj,odir,instance,ctx,12,precollected_views=frozen_views)
            expected=next(r['views'] for r in views['rows'] if r['object_index']==idx)
            record=dict(status='generated',seed=42,wall_s=time.monotonic()-start,producer_wall_s=wall,views=len(frozen_views),peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                view_manifest_sha256=sha(out/'view_manifest.json'),
                source_discovery_hashes=m.get('source_discovery_hashes',{}),
                consumed_views=verify_consumed_views(out,idx,expected))
        except Exception as exc:
            record=dict(status='generation_failed',object_index=idx,seed=42,error_type=type(exc).__name__,reason=type(exc).__name__+': '+str(exc),wall_s=time.monotonic()-start,views=len(frozen_views))
            write_new(out/'producer_records'/f'object_{idx:02d}.json',record)
            print(f'RVG object {idx}: generation_failed: {record["reason"]}')
            torch.cuda.empty_cache()
            continue
        write_new(out/'producer_records'/f'object_{idx:02d}.json',record)


def plan(config_path,root):
    c,git=context(config_path,root);rows,boundary,binding=source_boundary(c)
    smoke_command=[c['python'],'-c','import models.s4_reconviagen;from trellis.pipelines import TrellisVGGTTo3DPipeline;import dreamsim,spconv.pytorch,nvdiffrast.torch;from trellis.representations.mesh import cube2mesh;print("RVG strict imports PASS")']
    smoke=subprocess.run(smoke_command,env=env_for(c,output_directory(c,root)),cwd=CODE,text=True,capture_output=True)
    if smoke.returncode:raise PilotError(f'RVG CPU model imports failed: {smoke.stderr}')
    models=cohort_models(c,root,tool="rvg",validator=validate_models)
    out=output_directory(c,root);out.mkdir(parents=True)
    write_new(out/'import_smoke.json',dict(command=smoke_command,exit_code=smoke.returncode,stdout=smoke.stdout,stderr=smoke.stderr))
    source=Path(c['source_pilot']);shutil.copytree(source/'inputs',out/'inputs',symlinks=True)
    objects=out/'construction/objects';objects.mkdir(parents=True)
    source_hashes=json.loads((source/'output_hashes.json').read_text())
    for name in CONSTRUCTION_INPUTS:
        src=source/'construction'/name
        if sha(src)!=source_hashes[name]['sha256']:raise PilotError('source geometry/objects changed')
        shutil.copyfile(src,out/'construction'/name)
    write_new(out/'pinned_models.json',{'models':c['models']})
    write_new(out/'input_manifest.json',dict(code_commit=git,config_sha256=sha(config_path),paper_ready=False,jobs=rows,models=models,**binding))
    runtime,digest=runtime_identity(c['python']);write_new(out/'python_runtime.json',dict(runtime_sha256=digest,**runtime))
    command=[c['python'],str(Path(__file__).resolve()),'--config',str(Path(config_path).resolve()),'--freeze-root',str(Path(root).resolve()),'--phase','views']
    with (out/'views.log').open('x') as log:code=subprocess.call(command,env=env_for(c,out),cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
    write_new(out/'views_execution.json',dict(command=command,exit_code=code))
    if code:raise PilotError('CPU RVG view plan failed; preserve logs')
    write_new(out/'views_receipt.json',dict(status='PASS',input_manifest=identity(out/'input_manifest.json'),
        view_manifest=identity(out/'view_manifest.json'),source_binding=view_source_binding(c,boundary),paper_ready=False))
    validate_view_receipt(c,out,json.loads((out/'input_manifest.json').read_text()),boundary)
    stats={}
    for model in c['models'].values():
        p=Path(model['path']);paths=p.rglob('*') if p.is_dir() else [p]
        for f in paths:
            if f.is_file():s=f.stat();stats[str(f.resolve())]=[s.st_size,s.st_mtime_ns]
    write_new(out/'model_stats.json',stats)
    print(json.dumps({'planned_jobs':len(rows),'prepared_inputs':sum(r['prepared'] for r in rows),'view_plan':str(out/'view_manifest.json'),'paper_ready':False}))


def collect_records(out,m,freeze_id,code):
    rows=[]
    for job in m['jobs']:
        row=dict(job_id=job['job_id'],automatic_instance_id=job['automatic_instance_id'],prepared=job['prepared'],output_index=job.get('output_index'),proposal_id=f"{freeze_id}:{job['job_id']}:reconviagen:initial:seed42",tool='reconviagen',seed=42,paper_ready=False,shared_initial_policy_rows=['A1','A2','A3','A4'])
        if not job['prepared']:row.update(status='unavailable',reason='preparation_unavailable',artifacts={})
        else:
            row['input_sha256']=job['input']['sha256']
            idx=job['output_index'];rdir=out/'construction/objects'/f'obj_{idx:02d}/rvg';record=out/'producer_records'/f'object_{idx:02d}.json'
            runtime=json.loads(record.read_text()) if record.is_file() else None
            complete=runtime and runtime['status']=='generated' and runtime.get('seed')==42 and all((rdir/n).is_file() and (rdir/n).stat().st_size>0 for n in ['rvg_mesh.ply','rvg_gs.ply'])
            if complete and m.get('source_gaussian_training_provenance')==FRESH:
                views=json.loads((out/'view_manifest.json').read_text())
                expected=next(r['views'] for r in views['rows'] if r['object_index']==idx)
                if (runtime.get('view_manifest_sha256')!=sha(out/'view_manifest.json')
                        or runtime.get('source_discovery_hashes')!=m['source_discovery_hashes']
                        or runtime.get('consumed_views')!=verify_consumed_views(out,idx,expected)):
                    raise PilotError('RVG produced artifact lacks actual multiview source receipt')
            row.update(status='available' if complete else ('unavailable' if runtime and runtime['status']=='unavailable' else 'generation_failed'),reason=None if complete else (runtime.get('reason') if runtime else f'worker_exit_{code}'),runtime=runtime,artifacts={n:identity(rdir/n) for n in ['rvg_mesh.ply','rvg_gs.ply'] if (rdir/n).is_file()})
        rows.append(row)
    return rows


def execute(config_path,root):
    c,git=context(config_path,root);out=output_directory(c,root);m,boundary=read_manifest(c,out)
    if m['code_commit']!=git or m['config_sha256']!=sha(config_path):raise PilotError('RVG plan source/config differs')
    views=validate_view_receipt(c,out,m,boundary)
    stats=json.loads((out/'model_stats.json').read_text());actual={}
    for model in c['models'].values():
        p=Path(model['path']);paths=p.rglob('*') if p.is_dir() else [p]
        for f in paths:
            if f.is_file():s=f.stat();actual[str(f.resolve())]=[s.st_size,s.st_mtime_ns]
    if actual!=stats:raise PilotError('model files changed after CPU plan')
    write_new(out/'execution_claim.json',dict(code_commit=git,input_manifest_sha256=sha(out/'input_manifest.json'),
        views_receipt_sha256=sha(out/'views_receipt.json'),paper_ready=False))
    if not any(row['available'] for row in views['rows']):
        command=[c['python'],str(Path(__file__).resolve()),'--config',str(Path(config_path).resolve()),
                 '--freeze-root',str(Path(root).resolve()),'--phase','verify-views']
        with (out/'empty_views_revalidation.log').open('x') as log:
            verified=subprocess.call(command,env=env_for(c,out),cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
        if verified:raise PilotError('RVG empty-view population failed source-derived recomputation')
        (out/'producer_records').mkdir()
        for row in views['rows']:
            write_new(out/'producer_records'/f'object_{row["object_index"]:02d}.json',dict(status='unavailable',reason='fewer_than_two_usable_training_views',views=len(row['views'])))
        publish_pool(c,git,out,m,0,0.0,generation_performed=False)
        return
    node=os.environ.get('SLURMD_NODENAME','')
    if not os.environ.get('SLURM_JOB_ID') or not(node=='hala' or node.startswith(('gcp-','sof1-'))):raise PilotError('unauthorized GPU node')
    probe=subprocess.check_output([c['python'],'-c','import torch,json;assert torch.cuda.device_count()==1;f,t=torch.cuda.mem_get_info();print(json.dumps(dict(name=torch.cuda.get_device_name(),free_bytes=f,total_bytes=t)))'],text=True)
    gpu=json.loads(probe.strip().splitlines()[-1]);validate_gpu_memory(gpu)
    write_new(out/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],node=node,**gpu));(out/'producer_records').mkdir()
    command=[c['python'],str(Path(__file__).resolve()),'--config',str(Path(config_path).resolve()),'--freeze-root',str(Path(root).resolve()),'--phase','worker']
    start=time.monotonic()
    with (out/'generation.log').open('x') as log:code=subprocess.call(command,env=env_for(c,out),cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
    publish_pool(c,git,out,m,code,time.monotonic()-start,generation_performed=True)
    if code:raise PilotError('RVG failed; preserve complete per-object artifacts')


def publish_pool(c,git,out,m,code,wall_s,*,generation_performed):
    current,boundary=read_manifest(c,out)
    if current!=m:raise PilotError('RVG input manifest changed during generation')
    validate_view_receipt(c,out,m,boundary)
    freeze_id=c.get('freeze_id',out.parent.name)
    rows=collect_records(out,m,freeze_id,code)
    result=dict(code_commit=git,freeze_id=freeze_id,paper_ready=False,source_gaussian_training_provenance=m.get('source_gaussian_training_provenance','UNKNOWN'),initial_pool_complete=False,planned_jobs=len(rows),prepared_inputs=sum(r['prepared'] for r in rows),available_rvg=sum(r['status']=='available' for r in rows),wall_s=wall_s,exit_code=code,view_manifest_sha256=sha(out/'view_manifest.json'),views_receipt_sha256=sha(out/'views_receipt.json'),input_manifest_sha256=sha(out/'input_manifest.json'),rows=rows,generation_performed=generation_performed)
    if m.get('source_gaussian_training_provenance')==FRESH:result['source_discovery_hashes']=m['source_discovery_hashes']
    with (out/'proposal_records.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    result['proposal_records_sha256']=sha(out/'proposal_records.jsonl')
    write_new(out/'proposal_pool.json',result)
    print(json.dumps(result,indent=2))


def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--freeze-root',required=True);a.add_argument('--phase',choices=['plan','views','verify-views','run','worker'],required=True);args=a.parse_args()
    if args.phase=='plan':plan(args.config,args.freeze_root)
    elif args.phase=='run':execute(args.config,args.freeze_root)
    else:views_worker(args.config,args.freeze_root,generate=args.phase=='worker',verify_only=args.phase=='verify-views')
if __name__=='__main__':main()
