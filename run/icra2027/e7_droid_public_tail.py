"""Frozen public DROID GS -> TSDF -> SAM3 -> existing object contracts.

No proposal generation, simulator build, or reference-based source selection.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import time
import numpy as np
from agents.recon.droid_extract import input_identity,verify_input,prospective_environment
from agents.recon.colmap_poses import write_new_json
from robo.manifest.hash import canonical_hash,git_snapshot
from run.icra2027.e7_droid_gaussian import tree,verify_tree

CODE=Path(__file__).resolve().parents[2]
PROTOCOL={'render_stride':3,'discovery_stride':3,'render_scale':0.5,'alpha_min':0.6,
          'voxel_m':0.005,'sdf_trunc_m':0.02,'score_min':0.45,'min_frames_seen':2,
          'min_voxels':25,'merge_iou':0.25,'pixel_stride':4,'max_extent_m':0.9,
          'full_vocabulary':False,'pixel_gate_reference_height':1168,
          'min_bbox_reference_px':48,'min_mask_reference_px':400}
SCOPE='droid_public_gaussian_discovery_object_contracts'
CPU_PACKAGES=['numpy','scipy','open3d','trimesh',('PIL','Pillow'),('cv2','opencv-python'),'plyfile']
SOURCE_PROBE='''
import json,sys
from pathlib import Path
from run.icra2027.e7_droid_gaussian import context,validate_prepared,validate_training_report,verify_tree,input_identity
p,s,w=sys.argv[1:];c,code,e=context(p,s)
if w not in c['workspace_ids']:raise ValueError('outside original GS population')
d=Path(s)/'gaussian'/w;prep=json.loads((d/'prepare_receipt.json').read_text())
if prep['code']!=code or prep['config']!=input_identity(p) or prep['workspace_id']!=w:raise ValueError('GS preparation closure differs')
base={'source_commit':code['commit'],'contract_sha256':e['contract_sha256'],'preparation':input_identity(d/'prepare_receipt.json'),'workspace_id':w}
if prep['status']=='NOT_RUN':
 if prep['train_source']['eligible'] is not False or (d/'scene').exists():raise ValueError('invalid unavailable GS source')
 print('E7_GS_SOURCE_JSON='+json.dumps(dict(base,available=False,reason='sealed_TRAIN_fit_unavailable')));sys.exit(0)
validate_prepared(c,w,d,p,code)
r=json.loads((d/'train_receipt.json').read_text())
if r['code']!=code or r['config']!=input_identity(p) or r['workspace_id']!=w:raise ValueError('GS training source differs')
if r['status']=='FAIL':
 print('E7_GS_SOURCE_JSON='+json.dumps(dict(base,available=False,reason='GS_training_failed',training=input_identity(d/'train_receipt.json'))));sys.exit(0)
if r['status']!='PASS':raise ValueError('GS source not terminal')
verify_tree(d/'train',r['artifacts']);validate_training_report(r['training_report'],prep,30000)
print('E7_GS_SOURCE_JSON='+json.dumps(dict(base,available=True,scene_dir=str(d/'scene/data'/w),scene_artifacts=prep['scene_artifacts'],scene_root=str(d/'scene'),gaussian=input_identity(d/'train/scene.ply'),training=input_identity(d/'train_receipt.json'))))
'''


def validate_source(c):
    src=c['gaussian_source'];old=Path(src['code_root']);snap=git_snapshot(old)
    if old.resolve()!=old or snap['dirty'] or snap['commit']!=src['commit']:
        raise ValueError('original GS source changed')
    g=json.loads(verify_input(src['config']).read_text());e=json.loads(verify_input(src['contract']).read_text())
    payload={k:v for k,v in e.items() if k not in {'created_utc','environment','contract_sha256'}}
    bound=[x for x in e['resource_inventory'] if x['id']=='e7_gaussian_config']
    if (canonical_hash(payload)!=e['contract_sha256'] or e['code']['commit']!=src['commit'] or e['code']['dirty']
        or e['freeze_id']!=Path(src['stage_root']).name or len(bound)!=1 or bound[0]['sha256']!=src['config']['sha256']):
        raise ValueError('GS source E0/config closure differs')
    if (len(g['workspace_ids'])!=10 or c['workspace_ids']!=g['workspace_ids']
        or len(set(c['workspace_ids']))!=10 or c['pilot_workspace']!=g['pilot_workspace']):
        raise ValueError('original ten GS slots and first-IPRL pilot required')
    return g


def original_gaussian(c,wid):
    validate_source(c)
    if wid not in c['workspace_ids']:raise ValueError('unknown original workspace')
    src=c['gaussian_source'];env=prospective_environment(src['code_root']);env['SIMANY_EVIDENCE_ROOT']=c['evidence_root']
    command=[src['python'],'-c',SOURCE_PROBE,src['config']['path'],src['stage_root'],wid]
    raw=subprocess.check_output(command,cwd=src['code_root'],env=env,text=True)
    rows=[r.removeprefix('E7_GS_SOURCE_JSON=') for r in raw.splitlines() if r.startswith('E7_GS_SOURCE_JSON=')]
    if len(rows)!=1:raise ValueError('original GS validator must emit exactly one receipt')
    r=json.loads(rows[0])
    expected=json.loads(verify_input(src['contract']).read_text())
    if (r['source_commit']!=src['commit'] or r['workspace_id']!=wid
        or r['contract_sha256']!=expected['contract_sha256']):raise ValueError('wrong GS source receipt')
    return r


def verify_cpu_runtime(c):
    env=prospective_environment(CODE)
    command=[c['cpu_python'],'-c','import json;from agents.recon.droid_extract import runtime_manifest;print("E7_CPU_RUNTIME="+json.dumps(runtime_manifest('+repr(CPU_PACKAGES)+')))']
    raw=subprocess.check_output(command,cwd=CODE,env=env,text=True)
    rows=[r.removeprefix('E7_CPU_RUNTIME=') for r in raw.splitlines() if r.startswith('E7_CPU_RUNTIME=')]
    if len(rows)!=1 or json.loads(rows[0])!=json.loads(verify_input(c['cpu_runtime']).read_text()):
        raise ValueError('CPU mesh/crop runtime bytes changed')


def context(path,stage,*,runtime=True):
    path=Path(path).absolute();stage=Path(stage).absolute();c=json.loads(path.read_text());code=git_snapshot(CODE)
    if (c['scope']!=SCOPE or c['protocol']!=PROTOCOL or c['paper_ready'] is not False
        or c['generate_assets'] is not False or c['full_build'] is not False):
        raise ValueError('DROID public tail protocol/scope differs')
    if stage.resolve()!=stage or stage.name!=c['freeze_id'] or code['dirty'] or code['commit']=='nogit':
        raise ValueError('clean exact source and canonical output required')
    e=json.loads((stage/'contract/freeze_manifest.json').read_text());payload={k:v for k,v in e.items() if k not in {'created_utc','environment','contract_sha256'}}
    if canonical_hash(payload)!=e['contract_sha256'] or e['code']['commit']!=code['commit'] or e['code']['dirty'] or e['freeze_id']!=stage.name:
        raise ValueError('tail source/E0 differs')
    refs={'e7_tail_config':input_identity(path),'e7_tail_runtime':c['discovery_runtime'],'e7_tail_cpu_runtime':c['cpu_runtime']}
    for name,ref in refs.items():
        verify_input(ref);bound=[r for r in e['resource_inventory'] if r['id']==name]
        if len(bound)!=1 or bound[0]['sha256']!=ref['sha256']:raise ValueError('tail E0 resource differs: '+name)
    validate_source(c)
    model=json.loads(verify_input(c['discovery_runtime']).read_text())
    from robo.eval.fidelity_replacements import _tree_inventory
    checkpoint=model['sam3_checkpoint'];p=Path(checkpoint['path']);bound=[r for r in e['resource_inventory'] if r['id']=='sam3_checkpoint']
    if len(bound)!=1 or bound[0]['sha256']!=checkpoint['sha256'] or p.stat().st_size!=checkpoint['bytes'] or p.stat().st_mtime_ns!=checkpoint['mtime_ns']:
        raise ValueError('content-frozen SAM3 checkpoint changed')
    if _tree_inventory(Path(model['sam3_source']['path']),label='SAM3 source')['tree_sha256']!=model['sam3_source']['tree_sha256']:
        raise ValueError('SAM3 source changed')
    if runtime:
        from run.icra2027.e3_auto_discovery_pilot import validate_cohort_runtime
        validate_cohort_runtime(model);verify_cpu_runtime(c)
    return c,code,e,model


def phase_commands(c,runtime,source,dest):
    sd=source['scene_dir']
    return [('render',runtime['render_python'],'agents.discover.derive_mesh_from_splat',['render','--scene-dir',sd,'--splat-ply',source['gaussian']['path'],'--frame-stride','3']),
            ('fuse',c['cpu_python'],'agents.discover.derive_mesh_from_splat',['fuse','--scene-dir',sd]),
            ('discover',runtime['sam3_python'],'agents.discover.auto_segment',['--scene-dir',sd,'--out-dir',str(dest),'--mesh-path',str(dest/'derived_mesh.ply'),'--frame-stride','3']),
            ('prepare',c['cpu_python'],'agents.discover.factory_prepare',['--read-frames-out',str(dest/'object_read_frames.json')])]


def environment(c,runtime,source,dest,phase):
    env=prospective_environment(CODE)
    # Same metadata-only pixel scaling as the existing DROID launcher.
    calibration=json.loads((Path(source['scene_dir'])/'dslr/nerfstudio/transforms_undistorted.json').read_text());h=calibration['h']
    env.update(SIMANY_SCENE=source['workspace_id'],SIMANY_SCANNETPP_ROOT=source['scene_root'],SIMANY_OUT=str(dest),
               SIMANY_FULL='0',SIMANY_SAM3_CKPT=runtime['sam3_checkpoint']['path'],HF_HUB_OFFLINE='1',HF_HOME=runtime['hf_home'],
               SIMANY_MIN_BBOX_PX=str(max(6,round(48*h/1168))),SIMANY_MIN_MASK_PX=str(max(9,round(400*(h/1168)**2))))
    if phase=='render':
        env['PYTHONPATH']=str(CODE)+os.pathsep+runtime['renderer_dependency_root']
        env['TORCH_EXTENSIONS_DIR']=c['torch_extensions_root']
    elif phase=='discover':env['PYTHONPATH']=str(CODE)+os.pathsep+runtime['sam3_source']['path']
    return env


def enforce_public_read(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(args[0])).resolve()
    if (p.name in {'held_out_reference.json','alignment_evaluation.json','result.json'}
        or any(x.lower() in {'vault','hidden_gt','ground_truth','full_cpu_summary'} for x in p.parts)
        or p.is_relative_to('/data/ScanNetpp') or p.is_relative_to('/group/worldcept/PhiRIE/code/SimAny/data/recon_scenes/data')):
        raise PermissionError('public tail forbids reference/outcome/legacy scene reads: '+str(p))


def worker(config,stage,wid,phase):
    c,_,_,runtime=context(config,stage,runtime=False);source=original_gaussian(c,wid)
    if not source['available']:raise ValueError('unavailable GS cannot invoke downstream producer')
    dest=Path(stage)/'public_tail'/wid
    commands={n:(python,module,args) for n,python,module,args in phase_commands(c,runtime,source,dest)}
    if phase not in commands:raise ValueError('unknown producer phase')
    python,module,args=commands[phase]
    validate_worker_environment(c,runtime,source,dest,phase,python)
    # Existing producers may overwrite; own one immutable invocation per phase.
    with (dest/(phase+'.started')).open('x') as lock:lock.write(str(os.getpid()))
    sys.addaudithook(enforce_public_read)
    sys.argv=[module,*args];runpy.run_module(module,run_name='__main__')


def validate_worker_environment(c,runtime,source,dest,phase,python):
    expected=environment(c,runtime,source,dest,phase)
    keys={k for k in expected if k.startswith(('SIMANY_','PYTHON','OMP_','OPENBLAS_','MKL_','HF_'))}
    keys|={k for k in os.environ if k.startswith(('SIMANY_','PYTHON','SIMF_','LD_'))}
    if str(Path(sys.executable).absolute())!=str(Path(python).absolute()):
        raise ValueError('worker interpreter differs from frozen phase')
    if any(os.environ.get(k)!=expected.get(k) for k in keys):
        raise ValueError('worker environment differs from frozen public constructor')
    if phase in {'render','discover'}:
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not visible or ',' in visible:raise ValueError('one allocated GPU required')


def validate_public_products(source,dest):
    """Audit camera/read populations; no additional geometry metric."""
    meta=json.loads((Path(source['scene_dir'])/'dslr/nerfstudio/transforms_undistorted.json').read_text())
    names=sorted(Path(x['file_path']).name for x in meta['frames'])
    if len(names)!=len(set(names)) or not names:raise ValueError('invalid public RGB roster')
    views=sorted((dest/'mesh_derive').glob('view_*.npz'))
    expected=names[::PROTOCOL['render_stride']]
    if [p.name for p in views]!=[f'view_{i:04d}.npz' for i in range(len(expected))]:
        raise ValueError('render output roster differs')
    for p,name in zip(views,expected):
        with np.load(p,allow_pickle=False) as view:
            if str(view['fname'])!=name:raise ValueError('render camera identity differs')
    reads=json.loads((dest/'object_read_frames.json').read_text())
    if isinstance(reads,dict):reads=reads['frames']
    if not isinstance(reads,list) or len(reads)!=len(set(reads)) or not set(reads)<=set(names):
        raise ValueError('object crop read outside public RGB roster')


def validate_fused_mesh(dest):
    """Require an actual finite triangle mesh even if the TSDF process exits 0."""
    from plyfile import PlyData
    path=Path(dest)/'derived_mesh.ply'
    if path.is_symlink() or not path.is_file() or path.stat().st_size==0:
        raise ValueError('missing or empty fused mesh')
    mesh=PlyData.read(path)
    if 'vertex' not in mesh or 'face' not in mesh:
        raise ValueError('fused mesh lacks vertices or faces')
    vertices=np.column_stack([mesh['vertex'][axis] for axis in ('x','y','z')])
    faces=mesh['face']['vertex_indices']
    if len(vertices)<3 or len(faces)==0 or not np.isfinite(vertices).all():
        raise ValueError('empty or nonfinite fused mesh')
    for start in range(0,len(faces),65536):
        triangles=np.stack(faces[start:start+65536])
        if (triangles.ndim!=2 or triangles.shape[1]!=3 or triangles.dtype.kind not in 'iu'
            or np.any(triangles<0) or np.any(triangles>=len(vertices))):
            raise ValueError('invalid fused triangle indices')


def validate_output(path,stage,wid):
    c,code,e,runtime=context(path,stage)
    if wid not in c['workspace_ids']:raise ValueError('unknown output workspace')
    d=Path(stage)/'public_tail'/wid;p=d/'tail_receipt.json'
    if p.is_symlink():raise ValueError('receipt symlink forbidden')
    r=json.loads(p.read_text())
    if (r['scope']!=SCOPE or r['code']!=code or r['config']!=input_identity(path)
        or r['freeze_id']!=c['freeze_id'] or r['contract_sha256']!=e['contract_sha256']
        or r['workspace_id']!=wid or r['planned_workspaces']!=10
        or r['paper_ready'] is not False or r['full_build'] is not False):
        raise ValueError('tail output source/scope differs')
    actual=tree(d);actual.pop('tail_receipt.json')
    if actual!=r['artifacts_without_receipt']:raise ValueError('tail artifact inventory changed')
    source=original_gaussian(c,wid)
    if r['original_gaussian']!=source:raise ValueError('tail original source differs')
    if r['status']=='PASS':
        if source['available'] is not True:raise ValueError('successful tail lacks original GS')
        validate_fused_mesh(d)
        if [x['stage'] for x in r['stages']]!=['render','fuse','discover','prepare'] or any(x['returncode']!=0 for x in r['stages']):
            raise ValueError('successful tail phase population differs')
        for row,(name,python,module,args) in zip(r['stages'],phase_commands(c,runtime,source,d)):
            command=[python,'-m','run.icra2027.e7_droid_public_tail','--config',str(path),'--stage-root',str(stage),'--workspace',wid,'--worker',name]
            if (row['command']!=command or row['producer_module']!=module or row['producer_args']!=args
                or not math.isfinite(row['wall_s']) or row['wall_s']<0):raise ValueError('phase invocation/runtime differs')
        validate_public_products(source,d)
        if any(r[k]!=v for k,v in object_contracts(d).items()):raise ValueError('object denominator replay differs')
    elif r['status']=='NOT_RUN':
        if source['available'] is not False or r['stages'] or r['reason']!=source['reason'] or any(r[k] is not None for k in ('jobs','discovered_instances','prepared_instances')):
            raise ValueError('unavailable tail fabricated execution')
    elif r['status']!='FAIL' or not r.get('error'):raise ValueError('invalid terminal tail state')
    return r


def object_contracts(dest):
    """Use the existing discovery denominator/crop-index aggregator."""
    from run.icra2027.e3_auto_discovery_pilot import summarize_instances
    with np.load(dest/'auto_instances.npz',allow_pickle=False) as data:labels=data['labels'].tolist()
    objects=json.loads((dest/'objects/objects.json').read_text())
    return {'discovered_instances':len(labels),'prepared_instances':len(objects),'jobs':summarize_instances(labels,objects)}


PHASES=('render','fuse','discover','prepare')


def scheduler_guard(gpu=False):
    node=os.environ.get('SLURMD_NODENAME','')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_ARRAY_JOB_ID') or not (node.lower()=='hala' or node.startswith(('gcp','sof1'))):
        raise ValueError('ordinary allowed-node job required')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu and (not visible or ',' in visible):raise ValueError('one allocated GPU required')
    allocated=(any(os.environ.get(k,'').strip() not in {'','N/A','NoDevFiles','-1'}
                   for k in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS'))
               or os.environ.get('SLURM_GPUS_ON_NODE','').strip() not in {'','0','N/A','NoDevFiles','-1'})
    if not gpu and (allocated or (visible and visible not in {'NoDevFiles','-1'})):
        raise ValueError('CPU phase must not reserve a GPU')


def receipt_header(c,code,e,path,wid):
    return {'schema_version':1,'scope':SCOPE,'freeze_id':c['freeze_id'],'code':code,
            'contract_sha256':e['contract_sha256'],'config':input_identity(path),'workspace_id':wid,
            'job_id':os.environ.get('SLURM_JOB_ID'),'hostname':socket.gethostname(),
            'paper_ready':False,'full_build':False,'planned_workspaces':10}


def validate_header(r,c,code,e,path,wid):
    expected=receipt_header(c,code,e,path,wid)
    for k,v in expected.items():
        if k not in {'job_id','hostname'} and r.get(k)!=v:raise ValueError('stage receipt source/config/scope differs')


def plan(path,stage,wid):
    c,code,e,_=context(path,stage);scheduler_guard()
    if wid not in c['workspace_ids']:raise ValueError('unknown original workspace')
    source=original_gaussian(c,wid)
    d=Path(stage)/'public_tail'/wid;d.parent.mkdir(parents=True,exist_ok=True);d.mkdir()
    r=receipt_header(c,code,e,path,wid)
    r.update(original_gaussian=source,status='READY' if source['available'] else 'NOT_RUN',
             reason=None if source['available'] else source['reason'])
    write_new_json(d/'plan_receipt.json',r)
    return r


def validate_plan(path,stage,wid):
    c,code,e,runtime=context(path,stage)
    if wid not in c['workspace_ids']:raise ValueError('unknown original workspace')
    d=Path(stage)/'public_tail'/wid;p=d/'plan_receipt.json'
    if p.is_symlink():raise ValueError('plan symlink forbidden')
    r=json.loads(p.read_text());validate_header(r,c,code,e,path,wid)
    source=original_gaussian(c,wid)
    if r['original_gaussian']!=source or r['status']!=('READY' if source['available'] else 'NOT_RUN'):
        raise ValueError('planned original source/status changed')
    return c,code,e,runtime,r


def validate_phase(path,stage,wid,phase):
    c,code,e,runtime,plan_record=validate_plan(path,stage,wid)
    d=Path(stage)/'public_tail'/wid;p=d/(phase+'_receipt.json')
    if p.is_symlink():raise ValueError('phase receipt symlink forbidden')
    r=json.loads(p.read_text());validate_header(r,c,code,e,path,wid)
    if r['phase']!=phase or r['plan']!=input_identity(d/'plan_receipt.json'):
        raise ValueError('phase plan identity differs')
    if r['status'] not in {'PASS','FAIL'}:raise ValueError('phase must be terminal')
    for name,ref in r['artifacts'].items():
        if Path(ref['path'])!=d/name or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('phase artifact path differs')
        verify_input(ref)
    if r['status']=='PASS':
        row=r['execution'];_,python,module,args=next(x for x in phase_commands(c,runtime,plan_record['original_gaussian'],d) if x[0]==phase)
        command=[python,'-m','run.icra2027.e7_droid_public_tail','--config',str(path),'--stage-root',str(stage),'--workspace',wid,'--worker',phase]
        if row['stage']!=phase or row['returncode']!=0 or row['command']!=command or row['producer_module']!=module or row['producer_args']!=args:
            raise ValueError('phase execution differs')
        if phase=='fuse':validate_fused_mesh(d)
    elif not r.get('error'):raise ValueError('failed phase missing reason')
    return r


def run_phase(path,stage,wid,phase):
    if phase not in PHASES:raise ValueError('unknown producer phase')
    c,code,e,runtime,planned=validate_plan(path,stage,wid);scheduler_guard(phase in {'render','discover'})
    if planned['status']!='READY':raise ValueError('unavailable source cannot execute a phase')
    if wid!=c['pilot_workspace'] and validate_output(path,stage,c['pilot_workspace'])['status']!='PASS':
        raise ValueError('fixed first-IPRL tail pilot must pass first')
    for previous in PHASES[:PHASES.index(phase)]:
        if validate_phase(path,stage,wid,previous)['status']!='PASS':raise ValueError('predecessor phase failed')
    d=Path(stage)/'public_tail'/wid
    with (d/(phase+'.lock')).open('x') as lock:lock.write(str(os.getpid()))
    source=planned['original_gaussian'];r=receipt_header(c,code,e,path,wid)
    r.update(phase=phase,plan=input_identity(d/'plan_receipt.json'),status='RUNNING')
    name,python,module,args=next(row for row in phase_commands(c,runtime,source,d) if row[0]==phase)
    command=[python,'-m','run.icra2027.e7_droid_public_tail','--config',str(path),'--stage-root',str(stage),'--workspace',wid,'--worker',phase]
    start=time.monotonic()
    try:
        with (d/(phase+'.log')).open('x') as log:
            process=subprocess.run(command,cwd=CODE,env=environment(c,runtime,source,d,phase),stdout=log,stderr=subprocess.STDOUT)
        r['execution']={'stage':phase,'command':command,'producer_module':module,'producer_args':args,'returncode':process.returncode,'wall_s':time.monotonic()-start}
        if process.returncode:raise RuntimeError('public producer failed: '+phase)
        if phase=='fuse':validate_fused_mesh(d)
        validate_plan(path,stage,wid)  # Source/runtime bytes again before PASS.
        r['status']='PASS'
    except BaseException as exc:
        r.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        r['wall_s']=time.monotonic()-start;r['artifacts']=tree(d)
        write_new_json(d/(phase+'_receipt.json'),r)
    return r


def finalize(path,stage,wid):
    c,code,e,_,planned=validate_plan(path,stage,wid);scheduler_guard()
    d=Path(stage)/'public_tail'/wid
    with (d/'finalize.lock').open('x') as lock:lock.write(str(os.getpid()))
    source=planned['original_gaussian'];r=receipt_header(c,code,e,path,wid)
    r.update(original_gaussian=source,stages=[],status='RUNNING',discovered_instances=None,prepared_instances=None,jobs=None)
    start=time.monotonic()
    try:
        if planned['status']=='NOT_RUN':
            r.update(status='NOT_RUN',reason=source['reason']);return r
        for phase in PHASES:
            row=validate_phase(path,stage,wid,phase)
            if row['status']!='PASS':raise ValueError('tail has a failed producer phase')
            r['stages'].append(row['execution'])
        validate_public_products(source,d);r.update(object_contracts(d));r['status']='PASS'
    except BaseException as exc:
        r.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        r['wall_s']=time.monotonic()-start;r['artifacts_without_receipt']=tree(d)
        write_new_json(d/'tail_receipt.json',r)
    return r


def execute(path,stage,wid):
    """CPU planning compatibility entrypoint; never reserves GPU work."""
    r=plan(path,stage,wid)
    return finalize(path,stage,wid) if r['status']=='NOT_RUN' else r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--stage-root',required=True)
    p.add_argument('--workspace',required=True);p.add_argument('--worker');p.add_argument('--phase',choices=['plan',*PHASES,'finalize','validate'],default='plan');a=p.parse_args()
    if a.worker:worker(a.config,a.stage_root,a.workspace,a.worker)
    elif a.phase=='validate':print(json.dumps(validate_output(a.config,a.stage_root,a.workspace),indent=2))
    elif a.phase=='plan':print(json.dumps(execute(a.config,a.stage_root,a.workspace),indent=2))
    elif a.phase=='finalize':print(json.dumps(finalize(a.config,a.stage_root,a.workspace),indent=2))
    else:print(json.dumps(run_phase(a.config,a.stage_root,a.workspace,a.phase),indent=2))

if __name__=='__main__':main()
