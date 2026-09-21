"""Public reconstruction-to-grounding bridge using existing render/fuse/SAM3 producers."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import numpy as np
from run.icra2027 import e6_public_reconstruction as shared
from robo.certification import public_grounding as grounding

CODE=Path(__file__).resolve().parents[2]
CONSTRUCTION_CODE='/group/worldcept/PhiRIE/code/SimAny-wt/e6-public-full'
CONSTRUCTION_SHA='b2c827aa5c675ae8578a6bd1aec5c4a8f0402d99'
KEYS={'schema_version','scope','freeze_id','source_commit','construction','discovery_runtime','rgb_runtime','protocol'}
RUNTIME_KEYS={'sam3_source','sam3_checkpoint','sam3_python','render_python','renderer_dependency_root',
              'renderer_dependency_tree_sha256','runtime','hf_home'}


def validate(config_path, stage):
    config_path=Path(config_path).resolve(strict=True);stage=Path(stage).resolve(strict=True)
    cfg=shared.read(config_path);snap=shared.git_snapshot(CODE)
    if (set(cfg)!=KEYS or cfg['schema_version']!=1 or cfg['scope']!='e6_public_grounding_v1'
        or cfg['source_commit']!=snap['commit'] or snap['dirty'] or cfg['protocol']!=grounding.PROTOCOL):
        raise ValueError('grounding source/schema/protocol differs')
    if (stage.parent!=shared.ROOT/'outputs/icra2027' or stage.name!=cfg['freeze_id']
        or config_path.parent!=stage or not shared.CANONICAL_FREEZE_ID.fullmatch(stage.name)):
        raise ValueError('grounding stage outside frozen output scope')
    refs={name:cfg[name] for name in ('construction','discovery_runtime','rgb_runtime')}
    for name,ref in refs.items():
        if shared.identity(ref['path'])!=ref:raise ValueError('grounding input changed: '+name)
    contract=shared.read(stage/'contract/freeze_manifest.json')
    payload={k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
    if (shared.canonical_hash(payload)!=contract['contract_sha256'] or contract['code']['commit']!=cfg['source_commit']
        or contract['code']['dirty'] or contract['freeze_id']!=stage.name):raise ValueError('grounding E0 source/digest differs')
    for name,ref in {'config':shared.identity(config_path),**refs}.items():
        hits=[r for r in contract['resource_inventory'] if r['id']=='e6_grounding_'+name]
        if len(hits)!=1 or hits[0]['sha256']!=ref['sha256'] or hits[0]['resolved_path']!=ref['path']:
            raise ValueError('grounding E0 binding differs: '+name)
    source=shared.read(cfg['construction']['path'])
    if source['source_commit']!=CONSTRUCTION_SHA or source['scope']!='e6_public_rgb_full18':
        raise ValueError('requires original full RGB constructor')
    old=shared.git_snapshot(CONSTRUCTION_CODE)
    if old['commit']!=CONSTRUCTION_SHA or old['dirty']:raise ValueError('original constructor source changed')
    runtime=shared.read(cfg['discovery_runtime']['path'])
    if set(runtime)!=RUNTIME_KEYS:raise ValueError('discovery runtime schema differs')
    from run.icra2027.e3_auto_discovery_pilot import validate_cohort_runtime
    from robo.eval.fidelity_replacements import _tree_inventory
    validate_cohort_runtime(runtime)
    if _tree_inventory(Path(runtime['sam3_source']['path']),label='SAM3 source')['tree_sha256']!=runtime['sam3_source']['tree_sha256']:
        raise ValueError('SAM3 source changed')
    checkpoint=runtime['sam3_checkpoint'];p=Path(checkpoint['path'])
    if p.stat().st_size!=checkpoint['bytes'] or p.stat().st_mtime_ns!=checkpoint['mtime_ns']:
        raise ValueError('SAM3 checkpoint changed since content freeze')
    hits=[r for r in contract['resource_inventory'] if r['id']=='sam3_checkpoint']
    if len(hits)!=1 or hits[0]['sha256']!=checkpoint['sha256'] or hits[0]['resolved_path']!=str(p):
        raise ValueError('SAM3 content checkpoint not bound by E0')
    rgb=shared.read(cfg['rgb_runtime']['path'])
    if source['runtime']!=cfg['rgb_runtime']:raise ValueError('RGB/render runtime changed')
    return cfg,runtime,rgb,source


def original_unit(cfg,rgb,scene,condition):
    """Replay original clean constructor validation; never relabel its source."""
    source=Path(cfg['construction']['path'])
    command=[rgb['python'],'-m','run.icra2027.e6_public_reconstruction_full','--config',str(source),
             '--stage-root',str(source.parent),'--scene',scene,'--condition',condition,'--validate-output']
    env=shared.environment(rgb,source.parent,'validation');env['PYTHONPATH']=CONSTRUCTION_CODE
    env['SIMANY_ROOT']=CONSTRUCTION_CODE
    result=subprocess.run(command,cwd=CONSTRUCTION_CODE,env=env,text=True,capture_output=True)
    if result.returncode not in {0,3}:raise ValueError('original constructor validation failed: '+result.stderr[-1500:])
    gate=json.loads(result.stdout)
    if (result.returncode==0)!=(gate['stage_status']=='PASS'):raise ValueError('constructor terminal status differs')
    unit=source.parent/'audit/public_reconstruction'/(scene+'_'+condition)
    origin=Path(gate['origin']) if gate['mode']=='reused_original' else unit
    shared.sealed.sealed_cpu._inside(origin,root=shared.ROOT,label='public construction origin')
    return gate,origin,dict(command=command,returncode=result.returncode,terminal_manifest=shared.identity(unit/'terminal/manifest.json'))


def commands(runtime,rgb,origin,dest):
    scene=origin/'reconstruction/data'/origin.name
    return [
        ('render',runtime['render_python'],'agents.discover.derive_mesh_from_splat',
            ['render','--frame-stride','1','--scene-dir',str(scene),'--splat-ply',str(origin/'gaussian/scene.ply')]),
        ('fuse',rgb['python'],'agents.discover.derive_mesh_from_splat',['fuse','--scene-dir',str(scene)]),
        ('discover',runtime['sam3_python'],'agents.discover.auto_segment',
            ['--frame-stride','1','--scene-dir',str(scene),'--out-dir',str(dest),'--mesh-path',str(dest/'derived_mesh.ply')])]


def enforce_public_read(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    path=Path(os.fsdecode(args[0])).resolve()
    bad={'vault','oracle','hidden_gt','ground_truth'}
    if (any(p.lower() in bad for p in path.parts) or path.is_relative_to('/data/ScanNetpp')
        or path.is_relative_to(shared.ROOT/'data/recon_scenes/data')):
        raise PermissionError('public grounding forbids reference/legacy scene reads: '+str(path))


def worker(config_path,stage,scene,condition,phase):
    cfg,runtime,rgb,source=validate(config_path,stage)
    node=os.environ.get('SLURMD_NODENAME','')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_ARRAY_JOB_ID') or not (node.lower()=='hala' or node.startswith(('gcp','sof1'))):
        raise ValueError('grounding workers require ordinary jobs on an allowed node')
    if phase=='fuse':require_cpu_allocation()
    gate,origin,_=original_unit(cfg,rgb,scene,condition)
    if gate['stage_status']!='PASS':raise ValueError('rejected reconstruction cannot invoke grounding producer')
    dest=Path(stage)/'audit/public_grounding'/(scene+'_'+condition)
    *_,expected=plan_context(config_path,stage,scene,condition)
    plan_ref=checked_plan(dest,expected)
    claim=shared.read(dest/(phase+'_claim.json'))
    if claim!={'stage':phase,'plan_manifest':plan_ref,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}:
        raise ValueError('worker lacks exact resource-phase claim')
    target={'render':dest/'mesh_derive','fuse':dest/'derived_mesh.ply','discover':dest/'auto_instances.npz'}.get(phase)
    if target is None or target.exists():raise FileExistsError('worker resource phase already has output')
    planned={n:(module,args) for n,_,module,args in commands(runtime,rgb,origin,dest)}
    if phase not in planned:raise ValueError('unknown grounding phase')
    module,args=planned[phase]
    sys.addaudithook(enforce_public_read)
    sys.argv=[module,*args]
    runpy.run_module(module,run_name='__main__')


def associate(origin,dest,queries):
    from plyfile import PlyData
    from agents.core import common as C
    mesh=PlyData.read(dest/'derived_mesh.ply')['vertex']
    vertices=np.stack([np.asarray(mesh[k]) for k in ('x','y','z')],axis=1)
    with np.load(dest/'auto_instances.npz',allow_pickle=False) as data:
        objects=[dict(id=f'auto_{i:04d}',label=str(label),vertices=np.asarray(data[f'vert_idx_{i}']).copy(),
                      score=float(data['scores'][i]),n_frames=int(data['n_frames'][i])) for i,label in enumerate(data['labels'])]
    for obj in objects:
        if not np.isfinite(obj['score']) or not 0<=obj['score']<=1 or obj['n_frames']<2:
            raise ValueError('malformed automatic discovery evidence')
    scene=origin/'reconstruction/data'/origin.name
    K,W,H,_=C.load_intrinsics(scene/'dslr/nerfstudio/transforms_undistorted.json')
    frames={}
    for path in sorted((dest/'mesh_derive').glob('view_*.npz')):
        with np.load(path,allow_pickle=False) as data:
            name=str(data['fname'].item())
            if name in frames:raise ValueError('duplicate rendered source frame')
            frames[name]=(path,{key:np.asarray(data[key]).copy() for key in ('depth','alpha','w2c')})
    input_row=shared.read(origin/'input_manifest.json')
    selected={x['name'] for x in input_row['frames'] if x['selected']}
    if set(frames)!=selected:raise ValueError('all-selected render frame roster differs')
    frame=frames.get(input_row['source_frame']);projected=[]
    surface=None
    if frame is not None:
        path,data=frame;kh=K.copy();kh[0]*=data['depth'].shape[1]/W;kh[1]*=data['depth'].shape[0]/H
        for obj in objects:
            projected.append(dict(id=obj['id'],label=obj['label'],projection=grounding.projected_instance(
                vertices,obj['vertices'],kh,data['w2c'],data['depth'],data['alpha'])))
        surface=lambda annotation:grounding.region_surface(annotation,data['depth'],data['alpha'],kh,data['w2c'])
    grounded=[grounding.ground_query(q,source_present=frame is not None,projected_instances=projected,surface=surface) for q in queries]
    for q in grounded:
        q['source_render']=shared.identity(frame[0]) if frame else None
    population=[dict(id=o['id'],label=o['label'],score=o['score'],n_frames=o['n_frames'],
                     aabb=[vertices[o['vertices']].min(0).tolist(),vertices[o['vertices']].max(0).tolist()]) for o in objects]
    return dict(discovered_objects=population,queries=grounded,robot_frame=None,physics_verified=False)


def execute(config_path,stage,scene,condition):
    cfg,runtime,rgb,source=validate(config_path,stage)
    roster=shared.read(source['roster']['path'])
    rows=[r for r in roster['rows'] if (r['scene_id'],r['condition_id'])==(scene,condition)]
    if len(rows)!=1:raise ValueError('unknown planned unit')
    querypath=Path(CONSTRUCTION_CODE)/'configs/experiments/icra2027/public_task_queries/queries.json'
    queries=[q for q in shared.read(querypath)['queries'] if q['scene_id']==scene]
    if len(queries)!=4 or {q['task_id']:q['query_sha256'] for q in queries}!=rows[0]['query_hashes']:
        raise ValueError('public query identities differ')
    gate,origin,reference=original_unit(cfg,rgb,scene,condition)
    if gate['stage_status']=='PASS':
        raise ValueError('use --prepare, separate --phase jobs, and --finalize for successful constructions')
    dest=Path(stage)/'audit/public_grounding'/(scene+'_'+condition)
    shared.sealed.sealed_cpu._inside(dest,root=shared.ROOT,label='grounding output')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.mkdir()
    records=[];failed=None
    return finish_result(config_path,stage,scene,condition,cfg,gate,origin,reference,queries,dest,records,failed)


def finish_result(config_path,stage,scene,condition,cfg,gate,origin,reference,queries,dest,records,failed):
    if gate['stage_status']!='PASS' or failed:
        result=dict(discovered_objects=None,queries=[grounding.ground_query(q,source_present=False,projected_instances=[],unavailable_reason='upstream_reconstruction_rejected' if gate['stage_status']!='PASS' else 'grounding_producer_failed') for q in queries],robot_frame=None,physics_verified=False)
        for q in result['queries']:
            q['construction_unavailable_reason']='upstream_reconstruction_rejected' if gate['stage_status']!='PASS' else 'grounding_producer_failed'
    else:result=associate(origin,dest,queries)
    result['virtual_robot_declaration']=grounding.declare_virtual_robot(result['discovered_objects']) if condition=='clean' and result['discovered_objects'] is not None else None
    # Degraded conditions do not independently re-place the robot. A later
    # authenticated clean-to-condition SE3 declaration is required first.
    validate(config_path,stage)
    result.update(schema_version=1,scope=cfg['scope'],source_commit=cfg['source_commit'],freeze_id=cfg['freeze_id'],
                  scene_id=scene,condition_id=condition,planned_queries=4,planned_cohort_queries=72,
                  stage_status='FAIL' if failed or gate['stage_status']!='PASS' else 'PASS',
                  original_construction=reference,stage_records=records,failed_stage=failed,
                  feature_rows_written=0,paper_ready=False,protocol=grounding.PROTOCOL)
    members={str(p.relative_to(dest)):shared.file_identity(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    shared.sealed._publish_bundle(dest/'terminal',manifest_kind='e6_public_grounding',
        payloads={'gate.json':(json.dumps(result,indent=2)+'\n').encode()},
        manifest_fields={'source_commit':cfg['source_commit'],'output_members':members,'paper_ready':False})
    return result



PHASES=('render','fuse','discover')


def require_cpu_allocation():
    if (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS')
        or os.environ.get('SLURM_GPUS_ON_NODE','0') not in ('','0')):
        raise ValueError('CPU stage may not reserve a GPU')


def plan_context(config_path,stage,scene,condition):
    cfg,runtime,rgb,source=validate(config_path,stage)
    gate,origin,reference=original_unit(cfg,rgb,scene,condition)
    row=next((r for r in shared.read(source['roster']['path'])['rows']
              if (r['scene_id'],r['condition_id'])==(scene,condition)),None)
    if row is None:raise ValueError('unknown planned grounding unit')
    querypath=Path(CONSTRUCTION_CODE)/'configs/experiments/icra2027/public_task_queries/queries.json'
    queries=[q for q in shared.read(querypath)['queries'] if q['scene_id']==scene]
    if len(queries)!=4 or {q['task_id']:q['query_sha256'] for q in queries}!=row['query_hashes']:
        raise ValueError('public query roster changed')
    dest=Path(stage)/'audit/public_grounding'/(scene+'_'+condition)
    shared.sealed.sealed_cpu._inside(dest,root=shared.ROOT,label='grounding output')
    plan=dict(source_commit=cfg['source_commit'],config=shared.identity(config_path),freeze_id=cfg['freeze_id'],
              scene_id=scene,condition_id=condition,original_construction=reference,origin=str(origin),
              upstream_status=gate['stage_status'],query_hashes=row['query_hashes'],planned_queries=4,
              phases=list(PHASES),resource_classes={'render':'gpu','fuse':'cpu','discover':'gpu','finalize':'cpu'})
    return cfg,runtime,rgb,gate,origin,reference,queries,dest,plan


def prepare(config_path,stage,scene,condition):
    *_,dest,plan=plan_context(config_path,stage,scene,condition)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.mkdir()
    shared.sealed._publish_bundle(dest/'plan',manifest_kind='e6_public_grounding_plan',
        payloads={'gate.json':(json.dumps(plan,indent=2)+'\n').encode()},
        manifest_fields={'source_commit':plan['source_commit'],'paper_ready':False})
    return plan


def checked_plan(dest,expected):
    shared.sealed._validate_bundle(dest/'plan',root=shared.ROOT,expected_kind='e6_public_grounding_plan')
    if shared.read(dest/'plan/gate.json')!=expected:raise ValueError('grounding input plan changed')
    return shared.identity(dest/'plan/manifest.json')


def phase_files(dest,phase):
    paths=[dest/(phase+'.log'),dest/(phase+'_claim.json')]
    if phase=='render':paths+=sorted((dest/'mesh_derive').glob('*.npz'))
    elif phase=='fuse':paths+=[dest/'derived_mesh.ply']
    elif phase=='discover':paths+=[dest/'auto_instances.npz']
    else:raise ValueError('unknown resource phase')
    return {str(p.relative_to(dest)):shared.file_identity(p) for p in paths if p.is_file()}


def checked_phase(dest,phase,plan_ref,predecessors):
    b=shared.sealed._validate_bundle(dest/('phase_'+phase),root=shared.ROOT,expected_kind='e6_public_grounding_phase')
    gate=shared.read(dest/('phase_'+phase)/'gate.json')
    if (gate['stage']!=phase or gate['plan_manifest']!=plan_ref or gate['predecessors']!=predecessors
        or gate['resource_class']!=('cpu' if phase=='fuse' else 'gpu')
        or gate['output_members']!=phase_files(dest,phase)):
        raise ValueError('grounding phase binding/output changed')
    return gate,shared.identity(dest/('phase_'+phase)/'manifest.json')


def run_phase(config_path,stage,scene,condition,phase):
    if phase not in PHASES:raise ValueError('unknown grounding resource phase')
    cfg,runtime,rgb,gate,origin,reference,queries,dest,expected=plan_context(config_path,stage,scene,condition)
    plan_ref=checked_plan(dest,expected)
    if gate['stage_status']!='PASS':raise ValueError('rejected construction cannot execute a resource phase')
    predecessors={}
    for previous in PHASES[:PHASES.index(phase)]:
        record,ref=checked_phase(dest,previous,plan_ref,dict(predecessors))
        if record['returncode']:raise ValueError('failed predecessor prevents downstream resource use')
        predecessors[previous]=ref
    target={'render':dest/'mesh_derive','fuse':dest/'derived_mesh.ply','discover':dest/'auto_instances.npz'}[phase]
    if target.exists() or (dest/(phase+'.log')).exists() or (dest/('phase_'+phase)).exists() or (dest/(phase+'_claim.json')).exists():
        raise FileExistsError('resource phase already has output; no overwrite/resubmission')
    if phase=='fuse':require_cpu_allocation()
    name,python,module,args=next(row for row in commands(runtime,rgb,origin,dest) if row[0]==phase)
    env=shared.environment(rgb,dest,'gaussian' if phase=='render' else phase)
    env.update(SIMANY_NO_GT='1',SIMANY_FULL='1',SIMANY_SAM3_CKPT=runtime['sam3_checkpoint']['path'])
    if phase=='discover':env['PYTHONPATH']=str(CODE)+os.pathsep+runtime['sam3_source']['path']
    command=[python,'-m','run.icra2027.e6_public_grounding','--config',str(config_path),'--stage-root',str(stage),
             '--scene',scene,'--condition',condition,'--worker',phase]
    shared.write_new_json(dest/(phase+'_claim.json'),dict(stage=phase,plan_manifest=plan_ref,slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    t=time.monotonic()
    with (dest/(phase+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
    validate(config_path,stage)
    record=dict(stage=phase,command=command,producer_module=module,producer_args=args,returncode=result.returncode,
        wall_s=time.monotonic()-t,plan_manifest=plan_ref,predecessors=predecessors,output_members=phase_files(dest,phase),
        resource_class='cpu' if phase=='fuse' else 'gpu',slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    shared.sealed._publish_bundle(dest/('phase_'+phase),manifest_kind='e6_public_grounding_phase',
        payloads={'gate.json':(json.dumps(record,indent=2)+'\n').encode()},
        manifest_fields={'source_commit':cfg['source_commit'],'paper_ready':False})
    return record


def finalize(config_path,stage,scene,condition):
    require_cpu_allocation()
    cfg,runtime,rgb,gate,origin,reference,queries,dest,expected=plan_context(config_path,stage,scene,condition)
    plan_ref=checked_plan(dest,expected);records=[];predecessors={};failed=None
    if gate['stage_status']=='PASS':
        for phase in PHASES:
            record,ref=checked_phase(dest,phase,plan_ref,dict(predecessors));records.append(record)
            predecessors[phase]=ref
            if record['returncode']:failed=phase;break
    return finish_result(config_path,stage,scene,condition,cfg,gate,origin,reference,queries,dest,records,failed)


def validate_output(config_path,stage,scene,condition):
    cfg,runtime,rgb,source=validate(config_path,stage)
    old,origin,reference=original_unit(cfg,rgb,scene,condition)
    dest=Path(stage)/'audit/public_grounding'/(scene+'_'+condition)
    bundle=shared.sealed._validate_bundle(dest/'terminal',root=shared.ROOT,expected_kind='e6_public_grounding')
    actual={str(p.relative_to(dest)):shared.file_identity(p) for p in sorted(dest.rglob('*'))
            if p.is_file() and 'terminal' not in p.relative_to(dest).parts}
    if actual!=bundle['manifest']['output_members']:raise ValueError('grounding output bytes changed')
    result=shared.read(dest/'terminal/gate.json')
    for key,value in dict(schema_version=1,scope=cfg['scope'],source_commit=cfg['source_commit'],freeze_id=cfg['freeze_id'],
            scene_id=scene,condition_id=condition,planned_queries=4,planned_cohort_queries=72,
            feature_rows_written=0,paper_ready=False,protocol=grounding.PROTOCOL,original_construction=reference).items():
        if result.get(key)!=value:raise ValueError('grounding output identity differs: '+key)
    querypath=Path(CONSTRUCTION_CODE)/'configs/experiments/icra2027/public_task_queries/queries.json'
    queries=[q for q in shared.read(querypath)['queries'] if q['scene_id']==scene]
    records=result['stage_records'];planned=commands(runtime,rgb,origin,dest)
    if len(records)>3:raise ValueError('unexpected grounding actions')
    for record,(phase,python,module,args) in zip(records,planned):
        command=[python,'-m','run.icra2027.e6_public_grounding','--config',str(config_path),'--stage-root',str(stage),
                 '--scene',scene,'--condition',condition,'--worker',phase]
        if (record['stage']!=phase or record['command']!=command or record['producer_module']!=module
            or record['producer_args']!=args or not np.isfinite(record['wall_s']) or record['wall_s']<0):
            raise ValueError('grounding action receipt differs')
    if old['stage_status']=='PASS':
        *_,expected_plan=plan_context(config_path,stage,scene,condition)
        plan_ref=checked_plan(dest,expected_plan);predecessors={}
        for record in records:
            observed,ref=checked_phase(dest,record['stage'],plan_ref,dict(predecessors))
            if observed!=record:raise ValueError('terminal action differs from resource phase receipt')
            predecessors[record['stage']]=ref
    if old['stage_status']!='PASS':
        if records or result['stage_status']!='FAIL' or result['failed_stage'] is not None:
            raise ValueError('rejected upstream cannot execute grounding')
        reason='upstream_reconstruction_rejected'
    elif result['stage_status']=='PASS':
        if len(records)!=3 or any(r['returncode'] for r in records) or result['failed_stage'] is not None:
            raise ValueError('grounding PASS requires complete existing producer chain')
        reason=None
    elif result['stage_status']=='FAIL':
        if (not records or records[-1]['returncode']==0 or any(r['returncode'] for r in records[:-1])
            or result['failed_stage']!=records[-1]['stage']):raise ValueError('malformed grounding failure')
        reason='grounding_producer_failed'
    else:raise ValueError('unknown grounding terminal state')
    if reason:
        expected=dict(discovered_objects=None,queries=[grounding.ground_query(q,source_present=False,projected_instances=[],unavailable_reason=reason)
                      for q in queries],robot_frame=None,physics_verified=False)
        for q in expected['queries']:q['construction_unavailable_reason']=reason
    else:expected=associate(origin,dest,queries)
    for key,value in expected.items():
        if result[key]!=value:raise ValueError('grounding evidence replay differs: '+key)
    virtual=grounding.declare_virtual_robot(expected['discovered_objects']) if condition=='clean' and expected['discovered_objects'] is not None else None
    if result.get('virtual_robot_declaration')!=virtual:raise ValueError('virtual declaration differs')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--stage-root',required=True)
    p.add_argument('--scene');p.add_argument('--condition');p.add_argument('--validate',action='store_true');p.add_argument('--worker');p.add_argument('--validate-output',action='store_true');p.add_argument('--prepare',action='store_true');p.add_argument('--phase',choices=PHASES);p.add_argument('--finalize',action='store_true')
    a=p.parse_args()
    if a.validate:validate(a.config,a.stage_root);print('E6_GROUNDING_PREFLIGHT=PASS');return
    if a.worker:worker(a.config,a.stage_root,a.scene,a.condition,a.worker);return
    if a.prepare:print(json.dumps(prepare(a.config,a.stage_root,a.scene,a.condition),indent=2));return
    if a.phase:
        result=run_phase(a.config,a.stage_root,a.scene,a.condition,a.phase);print(json.dumps(result,indent=2))
        if result['returncode']:raise SystemExit(3)
        return
    result=(finalize if a.finalize else validate_output if a.validate_output else execute)(a.config,a.stage_root,a.scene,a.condition);print(json.dumps(result,indent=2))
    if result['stage_status']!='PASS':raise SystemExit(3)

if __name__=='__main__':main()
