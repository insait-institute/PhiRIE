"""Frozen RGB-only E6 reconstruction orchestration using existing producers."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import socket
from datetime import datetime,timezone

from agents.recon.colmap_poses import file_identity, write_new_json
from robo.manifest.hash import canonical_hash, git_snapshot, hash_checkpoint_path
from robo.eval.freeze import CANONICAL_FREEZE_ID
from robo.eval import e4_candidate_screen as sealed

CODE=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('SIMANY_ROOT',CODE))
PUBLIC=ROOT/'data/recon_scenes/data'
SCENES=('behavior_task0020','behavior_task0011','behavior_task0023','behavior_task0027','behavior_task0045','behavior_task0002')
CONDITIONS=('clean','mild','severe')
RULE='lexicographic_uniform48_replace_last_with_present_query_anchor_v1'


def read(path):return json.loads(Path(path).read_text())

def identity(path):
    path=Path(path)
    return {'path':str(path),**file_identity(path)}


def select_frames(names, anchor, limit=48):
    if limit!=48 or names!=sorted(set(names)) or len(names)<2:
        raise ValueError('fixed48 rule requires sorted unique names and at least two frames')
    if any(Path(n).name!=n or not n.endswith('.jpg') for n in names):
        raise ValueError('RGB roster contains unsafe/non-JPEG name')
    selected=names[:] if len(names)<=limit else [names[i*(len(names)-1)//(limit-1)] for i in range(limit)]
    if anchor in names and anchor not in selected:
        selected[-1]=anchor
    return sorted(selected)


def public_roster(*, public_root=PUBLIC, query_root=None):
    """Read only RGB and frozen query declarations, never old camera/depth files."""
    query_root=Path(query_root or CODE/'configs/experiments/icra2027/public_task_queries')
    # Query validation intentionally does not read legacy extrinsics/depth.
    import importlib.util
    spec=importlib.util.spec_from_file_location('e6_query_contract',query_root/'validate.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    sources,queries,expansion=[read(query_root/n) for n in ('sources.json','queries.json','condition_expansion.json')]
    validator.validate(sources,queries,expansion,verify_source_bytes=False)
    source_index={s['source_id']:s for row in sources['scenes'] for s in row['sources']}
    rows=[];public_root=Path(public_root).resolve(strict=True)
    for scene in SCENES:
        q=[x for x in queries['queries'] if x['scene_id']==scene]
        anchors={x['source_state_anchor'] for x in q}
        if len(q)!=4 or len(anchors)!=1:raise ValueError('immutable query state differs')
        anchor=Path(source_index[next(iter(anchors))]['relative_path']).name
        for condition in CONDITIONS:
            name=scene+('' if condition=='clean' else '_'+condition)
            rgb=sealed.sealed_cpu._regular_directory(public_root/name/'dslr/resized_undistorted_images',root=public_root,label='public RGB')
            names=sorted(p.name for p in rgb.iterdir() if p.suffix=='.jpg')
            selected=select_frames(names,anchor)
            frames=[]
            for name in names:
                path=sealed.sealed_cpu._regular_file(rgb/name,root=public_root,label='public RGB frame')
                frames.append({'name':name,'selected':name in selected,**identity(path)})
            rows.append({'scene_id':scene,'condition_id':condition,'source_frame':anchor,
                'source_frame_present':anchor in names,'frame_count':len(names),'frames':frames,
                'query_hashes':{x['task_id']:x['query_sha256'] for x in q}})
    return {'schema_version':1,'scope':'e6_rgb_only_reconstruction_inputs','selection_rule':RULE,
        'public_root':str(public_root),'planned_conditions':18,'planned_query_rows':72,
        'legacy_depth_pose_splat_inputs':False,'query_sources':[identity(query_root/n) for n in
        ('sources.json','queries.json','condition_expansion.json','validate.py')],'rows':rows}


def validate_roster(roster, *, public_root=PUBLIC):
    if (roster.get('scope')!='e6_rgb_only_reconstruction_inputs' or roster.get('selection_rule')!=RULE
        or roster.get('planned_conditions')!=18 or roster.get('planned_query_rows')!=72
        or roster.get('legacy_depth_pose_splat_inputs') is not False):
        raise ValueError('RGB-only scope/population changed')
    if [(r['scene_id'],r['condition_id']) for r in roster['rows']]!=[(s,c) for s in SCENES for c in CONDITIONS]:
        raise ValueError('condition roster changed')
    root=Path(public_root).resolve()
    if roster['public_root']!=str(root):raise ValueError('public root changed')
    query_root=CODE/'configs/experiments/icra2027/public_task_queries'
    expected_sources=[identity(query_root/n) for n in ('sources.json','queries.json','condition_expansion.json','validate.py')]
    if roster['query_sources']!=expected_sources:raise ValueError('immutable query package differs')
    queries=read(query_root/'queries.json')['queries']
    source_index={s['source_id']:s for scene in read(query_root/'sources.json')['scenes'] for s in scene['sources']}
    for row in roster['rows']:
        name=row['scene_id']+('' if row['condition_id']=='clean' else '_'+row['condition_id'])
        rgb=root/name/'dslr/resized_undistorted_images'
        names=[f['name'] for f in row['frames']]
        if names!=sorted(p.name for p in rgb.iterdir() if p.suffix=='.jpg'):
            raise ValueError('complete public frame roster differs')
        q=[q for q in queries if q['scene_id']==row['scene_id']]
        anchor=Path(source_index[q[0]['source_state_anchor']]['relative_path']).name
        if row['source_frame']!=anchor or row['query_hashes']!={x['task_id']:x['query_sha256'] for x in q}:
            raise ValueError('query identity/source anchor changed')
        if row['frame_count']!=len(names) or row['source_frame_present']!=(row['source_frame'] in names):
            raise ValueError('source view/count differs')
        selected=select_frames(names,row['source_frame'])
        if len(row['query_hashes'])!=4:raise ValueError('query denominator changed')
        for f in row['frames']:
            if Path(f['path'])!=rgb/f['name'] or f['selected']!=(f['name'] in selected):
                raise ValueError('RGB source/selection changed')
            sealed.sealed_cpu._regular_file(Path(f['path']),root=root,label='public RGB frame')
    return roster


def validate(config_path, stage_root, *, selected=None):
    config=read(config_path);stage=Path(stage_root).resolve()
    code=git_snapshot(CODE)
    if code['dirty'] or code['commit']!=config['source_commit']:
        raise ValueError('E6 requires exact clean source')
    tier=config.get('tier')
    expected_scene=SCENES[2] if tier=='smoke' else SCENES[0]
    if (tier not in {'smoke','pilot'} or config['scope']!='e6_rgb_only_reconstruction_'+tier or config['freeze_id']!=stage.name
        or config['max_frames']!=48 or config['seed']!=0 or config['gs_iters']!=15000
        or config['execution_scene']!=expected_scene or config['fallback_allowed'] is not False):
        raise ValueError('frozen E6 scope/pilot/RNG differs')
    if not CANONICAL_FREEZE_ID.fullmatch(stage.name):raise ValueError('noncanonical freeze ID')
    sealed.sealed_cpu._inside(stage,root=ROOT,label='E6 freeze')
    if stage.parent!=ROOT/'outputs/icra2027':raise ValueError('stage outside canonical outputs')
    contract=read(stage/'contract/freeze_manifest.json')
    payload={k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
    if canonical_hash(payload)!=contract['contract_sha256'] or contract['code']['commit']!=code['commit'] or contract['code']['dirty']:
        raise ValueError('E0 source/digest mismatch')
    if contract['freeze_id']!=stage.name:raise ValueError('E0 stage mismatch')
    refs={'e6_reconstruction_config':identity(Path(config_path).resolve()),
          'e6_rgb_roster':config['roster'],'e6_runtime':config['runtime']}
    for name,ref in refs.items():
        bound=[r for r in contract['resource_inventory'] if r['id']==name]
        if len(bound)!=1 or bound[0]['sha256']!=ref['sha256'] or Path(bound[0]['resolved_path'])!=Path(ref['path']):
            raise ValueError('E0 input binding differs: '+name)
        if identity(ref['path'])!=ref:raise ValueError('sealed input changed: '+name)
    roster=validate_roster(read(config['roster']['path']),public_root=PUBLIC)
    if tier=='smoke':
        clean=[r for r in roster['rows'] if r['condition_id']=='clean']
        smallest=min(clean,key=lambda r:(r['frame_count'],SCENES.index(r['scene_id'])))['scene_id']
        if smallest!=expected_scene or config.get('selection_rationale')!='smallest_positive_clean_frame_count_then_frozen_scene_order':
            raise ValueError('engineering smoke selection changed')
    runtime=read(config['runtime']['path'])
    sources={s['id']:s for s in runtime['sources']}
    checkpoints={f['id']:f for f in runtime['checkpoints']}
    if (runtime['omega_source']!=sources['omega']['path'] or runtime['da3_source']!=sources['da3']['path']
        or runtime['omega_checkpoint']!=checkpoints['omega_checkpoint']['path']
        or Path(runtime['da3_checkpoint'])!=Path(checkpoints['da3_checkpoint']['path']).parent):
        raise ValueError('runtime model/source paths are not bound')
    bound_files={f['path'] for f in runtime['files']}
    if any(str(Path(runtime[k]).resolve()) not in bound_files for k in ('python','gs_python')):
        raise ValueError('runtime interpreter is not bound')
    for tree in runtime['sources']:
        snapshot=git_snapshot(tree['path'])
        if snapshot['dirty'] or snapshot['commit']!=tree['commit']:
            raise ValueError('dependency source changed')
    for f in runtime['files']:
        if identity(f['path'])!={k:f[k] for k in ('path','sha256','size_bytes')}:
            raise ValueError('runtime dependency changed')
    for package in runtime.get('packages',{}).values():
        if hash_checkpoint_path(Path(package['path']).parent)!=package['tree_stat_fingerprint']:
            raise ValueError('runtime installed package fingerprint changed')
    verify_gs_runtime(runtime,stage)
    for f in runtime['checkpoints']:
        # E0 seals the content hash. Immutable file-stat identity rejects later
        # changes without repeatedly rereading multi-GB weights at each stage.
        p=Path(f['path']);st=p.stat()
        if st.st_size!=f['size_bytes'] or st.st_mtime_ns!=f['mtime_ns']:
            raise ValueError('checkpoint changed since content freeze')
        bound=[r for r in contract['resource_inventory'] if r['id']==f['id']]
        if len(bound)!=1 or bound[0]['sha256']!=f['sha256'] or Path(bound[0]['resolved_path'])!=p:
            raise ValueError('checkpoint E0 hash differs')
    if selected is not None:
        if selected not in CONDITIONS or (tier=='smoke' and selected!='clean'):raise ValueError('unknown or unplanned condition')
        row=next(r for r in roster['rows'] if r['scene_id']==expected_scene and r['condition_id']==selected)
        for f in row['frames']:
            if identity(f['path'])!={k:f[k] for k in ('path','sha256','size_bytes')}:
                raise ValueError('public RGB changed')
        return config,runtime,row
    return config,runtime,roster


def verify_gs_runtime(runtime,stage):
    observed=json.loads(subprocess.check_output([runtime['gs_python'],'-m',
        'run.icra2027.e3_gaussian_train_only','--runtime-identity'],cwd=CODE,
        env=environment(runtime,stage,'gaussian'),text=True))
    if observed!=runtime['gs_runtime_identity']:
        raise ValueError('canonical Gaussian runtime identity differs')


def commands(config, runtime, destination, frames):
    dest=Path(destination);scene_name=dest.name
    scene=dest/'reconstruction/data'/scene_name
    return [
      ('omega',runtime['python'],['-m','models.vggt_scene','--images-dir',str(frames),'--out',str(dest/'recon.npz'),
        '--backend','omega','--omega-source',runtime['omega_source'],'--omega-checkpoint',runtime['omega_checkpoint'],'--seed','0']),
      ('metricize',runtime['python'],['-m','agents.recon.metricize','--recon',str(dest/'recon.npz'),
        '--images-dir',str(frames),'--out',str(dest/'recon_metric.npz'),'--checkpoint',runtime['da3_checkpoint']]),
      ('scene',runtime['python'],['-m','agents.recon.make_scene_dir','--recon',str(dest/'recon_metric.npz'),
        '--frames-dir',str(frames),'--scene',scene_name,'--root',str(dest/'reconstruction')]),
      ('gaussian',runtime['gs_python'],['-m','agents.recon.gsplat_train','--scene-dir',str(scene),
        '--init-ply',str(scene/'init_points.ply'),'--out',str(dest/'gaussian/scene.ply'),
        '--iters','15000','--seed','0'])]


def environment(runtime, destination, kind):
    env=os.environ.copy()
    for k in list(env):
        if k.startswith(('SIMANY_','SIMF_')) or k in {'PYTHONHOME','PYTHONPATH'}:env.pop(k)
    extra=runtime['gs_overlay'] if kind=='gaussian' else runtime['da3_overlay']+os.pathsep+runtime['da3_source']+'/src'
    env.update(PYTHONPATH=str(CODE)+os.pathsep+extra,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
        PYTHONHASHSEED='0',SIMANY_ROOT=str(CODE),SIMANY_OUT=str(destination),SIMANY_AUTO='1',
        SIMANY_MESH_SRC='derived',TMPDIR=str(ROOT/'.tmp/icra2027'),XDG_CACHE_HOME=str(ROOT/'.cache/icra2027/e6-runtime'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(ROOT/'.cache/icra2027/e6-runtime/huggingface'),
        TORCH_EXTENSIONS_DIR=runtime['torch_extensions_root'],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        MKL_NUM_THREADS='4',TORCH_CUDA_ARCH_LIST='8.6;9.0+PTX')
    return env


def run(config_path, stage_root, condition):
    config,runtime,row=validate(config_path,stage_root,selected=condition)
    parent=Path(stage_root)/'audit/public_reconstruction';parent.mkdir(parents=True,exist_ok=True)
    dest=parent/(config['execution_scene']+'_'+condition)
    if dest.exists():raise FileExistsError('refusing existing E6 reconstruction')
    dest.mkdir();frames=dest/'frames';frames.mkdir();start=time.monotonic();records=[]
    started=datetime.now(timezone.utc).isoformat()
    for f in row['frames']:
        if f['selected']:
            shutil.copyfile(f['path'],frames/f['name'])
            if file_identity(frames/f['name'])!={k:f[k] for k in ('sha256','size_bytes')}:raise ValueError('RGB copy differs')
    write_new_json(dest/'input_manifest.json',row)
    failed=None
    for name,python,args in commands(config,runtime,dest,frames):
        t=time.monotonic()
        with (dest/(name+'.log')).open('x') as log:
            result=subprocess.run([python,*args],cwd=CODE,env=environment(runtime,dest,name),stdout=log,stderr=subprocess.STDOUT)
        records.append({'stage':name,'command':[python,*args],'returncode':result.returncode,'wall_s':time.monotonic()-t})
        write_new_json(dest/(name+'_execution.json'),records[-1])
        if result.returncode:failed=name;break
    validate(config_path,stage_root,selected=condition)
    if not failed:
        validate_products(dest,row)
    report={'schema_version':1,'scope':config['scope'],'source_commit':config['source_commit'],
        'freeze_id':config['freeze_id'],'scene_id':config['execution_scene'],'condition_id':condition,
        'stage_status':'FAIL' if failed else 'PASS','terminal_reconstruction_status':'failed' if failed else 'complete',
        'failed_stage':failed,'failure_type':'producer_nonzero_exit' if failed else None,
        'stage_records':records,'wall_s':time.monotonic()-start,'planned_queries':4,
        'started_utc':started,'hostname':socket.gethostname(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'source_frame_present':row['source_frame_present'],'model_fallback_used':False,
        'legacy_depth_pose_splat_read':False,'feature_rows_written':0,'paper_ready':False,
        'limitation':'RGB reconstruction stage only; no full twin, query grounding, labels, policy or task-local claim.'}
    members={str(p.relative_to(dest)):file_identity(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    sealed._publish_bundle(dest/'terminal',manifest_kind='e6_public_rgb_reconstruction',
        payloads={'gate.json':(json.dumps(report,indent=2)+'\n').encode()},
        manifest_fields={'source_commit':config['source_commit'],'output_members':members,'paper_ready':False})
    return report


def validate_products(destination,row):
    """Independent schema checks over actual producer outputs; no reference data."""
    import numpy as np
    d=Path(destination);names=sorted(f['name'] for f in row['frames'] if f['selected'])
    for filename in ('recon.npz','recon_metric.npz'):
        with np.load(d/filename,allow_pickle=False) as x:
            if list(x['names'])!=names or x['w2c'].shape!=(len(names),4,4) or x['K'].shape!=(3,3):
                raise ValueError('reconstruction camera/frame schema differs')
            if x['points'].ndim!=2 or x['points'].shape[1]!=3 or len(x['points'])<1:
                raise ValueError('empty/malformed predicted geometry')
            for key in x.files:
                if x[key].dtype.kind in 'fci' and not np.isfinite(x[key]).all():
                    raise ValueError('nonfinite reconstruction')
            if filename=='recon_metric.npz' and (float(x['metric_scale'])<=0 or x['T_align'].shape!=(4,4)):
                raise ValueError('invalid predicted metric frame')
    train=read(d/'gaussian/train_report.json')
    if (train['seed']!=0 or train['iters']!=15000 or train['independent_heldout_evaluation'] is not False
        or set(train['gradient_train_frames'])&set(train['internal_diagnostic_frames'])
        or sorted(train['gradient_train_frames']+train['internal_diagnostic_frames'])!=names):
        raise ValueError('Gaussian producer split/RNG differs')
    if not (d/'gaussian/scene.ply').is_file():raise ValueError('Gaussian artifact missing')


def validate_output(config_path,stage_root,condition):
    config,runtime,row=validate(config_path,stage_root,selected=condition)
    dest=Path(stage_root)/'audit/public_reconstruction'/(config['execution_scene']+'_'+condition)
    bundle=sealed._validate_bundle(dest/'terminal',root=ROOT,expected_kind='e6_public_rgb_reconstruction')
    expected=bundle['manifest']['output_members']
    actual={str(p.relative_to(dest)):file_identity(p) for p in sorted(dest.rglob('*'))
            if p.is_file() and 'terminal' not in p.relative_to(dest).parts}
    if actual!=expected:raise ValueError('reconstruction output bytes changed')
    gate=read(dest/'terminal/gate.json')
    if read(dest/'input_manifest.json')!=row:raise ValueError('published input roster differs')
    plan=commands(config,runtime,dest,dest/'frames')
    records=gate['stage_records']
    if len(records)>len(plan) or any(r['stage']!=name or r['command']!=[python,*args]
            for r,(name,python,args) in zip(records,plan)):
        raise ValueError('executed commands differ from fixed producer chain')
    if (gate['source_commit']!=config['source_commit'] or gate['freeze_id']!=config['freeze_id']
        or gate['scene_id']!=config['execution_scene'] or gate['condition_id']!=condition or gate['planned_queries']!=4
        or gate['paper_ready'] is not False or gate['model_fallback_used'] is not False
        or gate['feature_rows_written']!=0 or gate['legacy_depth_pose_splat_read'] is not False
        or gate['scope']!=config['scope'] or gate['source_frame_present']!=row['source_frame_present']):
        raise ValueError('reconstruction gate scope differs')
    if gate['stage_status']=='PASS':
        if (len(gate['stage_records'])!=4 or any(r['returncode'] for r in gate['stage_records'])
            or gate['terminal_reconstruction_status']!='complete' or gate['failed_stage'] is not None
            or gate['failure_type'] is not None):
            raise ValueError('PASS requires all canonical stages')
        validate_products(dest,row)
    elif gate['stage_status']=='FAIL':
        rows=gate['stage_records']
        if (not rows or rows[-1]['returncode']==0 or any(r['returncode'] for r in rows[:-1])
            or gate['failure_type']!='producer_nonzero_exit' or gate['terminal_reconstruction_status']!='failed'
            or gate['failed_stage']!=rows[-1]['stage']):
            raise ValueError('invalid typed failure receipt')
    else:raise ValueError('unknown terminal stage status')
    return gate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--stage-root')
    p.add_argument('--condition',choices=CONDITIONS);p.add_argument('--roster-out');p.add_argument('--validate',action='store_true')
    p.add_argument('--validate-output',action='store_true')
    a=p.parse_args()
    if a.roster_out:write_new_json(a.roster_out,public_roster());return
    if a.validate_output:print(json.dumps(validate_output(a.config,a.stage_root,a.condition),indent=2));return
    if a.validate:validate(a.config,a.stage_root,selected=a.condition);print('E6_PREFLIGHT=PASS');return
    result=run(a.config,a.stage_root,a.condition)
    print(json.dumps(result,indent=2))
    if result['stage_status']=='FAIL':raise SystemExit(3)

if __name__=='__main__':main()
