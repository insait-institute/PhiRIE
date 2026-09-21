#!/usr/bin/env python3
"""Initial TRELLIS/TRELLIS.2 proposals through the same provenance runner."""
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
import yaml
CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE))
from run.icra2027.e3_auto_discovery_pilot import identity, sha, write_new, validate_gpu_memory, summarize_instances, PilotError
from robo.eval.fidelity_replacements import _tree_inventory

from run.icra2027.e3_generator_backend import generator, validate_config, environment, TRELLIS2_OUTPUTS

OUTPUTS = ('trellis_mesh.ply','trellis_gs.ply','mesh_sim.ply','mesh_sim.obj')

def output_directory(config, root):
    out = Path(root) / (generator(config) + '_initial')
    scene = config.get('output_scene_id')
    if scene is not None:
        import re
        if not isinstance(scene, str) or not re.fullmatch(r'[a-f0-9]{10}', scene):
            raise PilotError('invalid generation output scene')
        if Path(config['source_pilot']).name != scene:
            raise PilotError('output scene differs from discovery input')
        out /= scene
    if out.absolute() != out.resolve():
        raise PilotError('generation output cannot contain symlinks')
    return out


def context(config_path,root):
    c=yaml.safe_load(Path(config_path).read_text());root=Path(root)
    validate_config(c)
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=CODE,text=True).strip()
    contract=json.loads((root/'contract/freeze_manifest.json').read_text())
    if dirty or contract['code']['dirty'] or contract['code']['commit']!=git:
        raise PilotError('exact clean source contract required')
    if c['freeze_id']!=root.name or contract['freeze_id']!=root.name or c['paper_ready'] is not False or c['seed']!=42:
        raise PilotError('frozen engineering scope/seed differs')
    resource=[r for r in contract['resource_inventory'] if r['id']==c.get('contract_resource_id','e3_trellis_config')]
    if len(resource)!=1 or resource[0]['sha256']!=sha(config_path):raise PilotError('config identity differs')
    _,runtime_hash=runtime_identity(c['python'])
    if runtime_hash!=c['runtime_sha256']:raise PilotError('frozen Python runtime differs')
    return c,git

def runtime_identity(python):
    command = "import json,sys;from importlib.metadata import distributions;print(json.dumps(dict(python=sys.version,packages=sorted((d.metadata['Name'],d.version) for d in distributions())),sort_keys=True))"
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(CODE))
    env.pop('PYTHONHOME',None)
    payload = json.loads(subprocess.check_output([python,'-c',command],env=env,text=True))
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return payload,digest


def targeted_runtime_identity(python, *, extra_packages=()):
    """Cheap new-run byte audit, explicitly insufficient as a full runtime seal."""
    command = '''import json,sys,importlib.util,importlib.metadata
from pathlib import Path
paths={str(Path(sys.executable).resolve())}
for module,package in [('torch','torch'),('numpy','numpy'),('PIL','Pillow'),('open3d','open3d'),('xformers','xformers'),('spconv','spconv-cu120'),('nvdiffrast','nvdiffrast'),('trimesh','trimesh')]:
 spec=importlib.util.find_spec(module)
 if spec is not None and spec.origin:paths.add(str(Path(spec.origin).resolve()))
 try:d=importlib.metadata.distribution(package)
 except importlib.metadata.PackageNotFoundError:continue
 for f in d.files or []:
  if str(f).endswith('.dist-info/RECORD'):paths.add(str(Path(d.locate_file(f)).resolve()))
print(json.dumps(sorted(paths)))'''
    command = command.replace("('trimesh','trimesh')]:", "('trimesh','trimesh')] + " + repr(list(extra_packages)) + ":")
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE))
    env.pop('PYTHONHOME',None)
    paths=json.loads(subprocess.check_output([python,'-c',command],env=env,text=True))
    return dict(scope='targeted_executable_package_records_and_entrypoints',
                complete_raw_runtime_identity=False,files=[identity(p) for p in paths])


def runtime_bytes_identity(c):
    if generator(c) == 'trellis2':
        return targeted_runtime_identity(c['python'], extra_packages=[
            ('cumesh','cumesh'), ('flex_gemm','flex-gemm'), ('o_voxel','o-voxel'),
            ('flash_attn','flash-attn'), ('transformers','transformers')])
    return targeted_runtime_identity(c['python'])


def import_smoke(c):
    if generator(c) == 'trellis2':
        command = [c['python'], '-c', '''import os, importlib.util
import models.s4_trellis2 as producer
import torch, numpy, trimesh
producer.validate_sources(os.environ['SIMANY_TRELLIS2_DIR'],os.environ['SIMANY_TRELLIS2_MODEL'],os.environ['SIMANY_DINOV3_MODEL'],os.environ['SIMANY_SS_DECODER'],os.environ['SIMANY_TRELLIS2_SOURCE_COMMIT'])
for module in ('cumesh','flex_gemm','o_voxel','flash_attn','open3d','transformers'):
 assert importlib.util.find_spec(module) is not None, module
print("TRELLIS.2 CPU source/dependency preflight PASS; CUDA import requires allocated GPU")''']
        env = environment(c, CODE)
        env['PYTHONPATH'] += os.pathsep + c['models']['trellis2_source']['path']
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        if result.returncode: raise PilotError('TRELLIS.2 import smoke failed: '+result.stderr)
        return dict(command=command, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    models=c['models']
    env=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),SIMANY_ROOT=str(CODE),
        SIMANY_TRELLIS_DIR=models['trellis_source']['path'],SIMANY_TRELLIS_MODEL=models['trellis_snapshot']['path'],
        SIMANY_DINOV2_REPO=models['dinov2_source']['path'],TORCH_HOME=str(Path(models['dinov2_source']['path']).parent.parent),HF_HUB_OFFLINE='1')
    env.pop('PYTHONHOME',None)
    command=[c['python'],'-c','import models.s4_trellis;from trellis.pipelines import TrellisImageTo3DPipeline;import open3d;import spconv.pytorch;import nvdiffrast.torch;from trellis.models.structured_latent_vae import decoder_mesh;from trellis.representations.mesh import cube2mesh;print("strict producer imports PASS")']
    result=subprocess.run(command,env=env,text=True,capture_output=True)
    if result.returncode:raise PilotError(f'CPU producer import smoke failed: {result.stderr}')
    return dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)


def source_jobs(source):
    source=Path(source)
    summary=json.loads((source/'pilot_summary.json').read_text())
    manifest=json.loads((source/'input_manifest.json').read_text())
    audit=json.loads((source/'postrun_audit.json').read_text())
    if audit['summary_sha256']!=sha(source/'pilot_summary.json') or summary['paper_ready'] is not False:
        raise PilotError('source diagnostic identity differs')
    provenance=manifest.get('source_gaussian_training_provenance','UNKNOWN')
    if provenance!=summary.get('source_gaussian_training_provenance','UNKNOWN'):
        raise PilotError('source Gaussian provenance differs between manifest and summary')
    if provenance!='UNKNOWN' and 'all_jobs_manifest_sha256' not in audit:
        raise PilotError('fresh discovery requires complete all-jobs seal')
    objects=json.loads((source/'construction/objects/objects.json').read_text())
    if [s['stage'] for s in summary['stages']] != manifest['planned_stages']:
        raise PilotError('source stage roster differs')
    for stage in summary['stages']:
        if stage.get('exit_code') == 0:
            continue
        if stage == {'stage':'refine','status':'NOT_RUN','reason':'empty_prepared_population'} and not objects:
            continue
        raise PilotError('source stage failed or was unexpectedly skipped')
    hashes=json.loads((source/'output_hashes.json').read_text())
    if sha(source/'output_hashes.json')!=audit['output_hashes_sha256']:raise PilotError('source output index differs')
    if sha(source/'construction/objects/objects.json')!=hashes['objects/objects.json']['sha256']:
        raise PilotError('source objects metadata changed')
    instances=source/'construction/auto_instances.npz'
    if sha(instances)!=hashes['auto_instances.npz']['sha256']:
        raise PilotError('source complete discovery instances changed')
    import numpy as np
    with np.load(instances,allow_pickle=False) as data:
        labels=data['labels'].tolist()
    expected=summarize_instances(labels,objects)
    if summary['rows']!=expected or summary['discovered_instances']!=len(labels) or summary['prepared_instances']!=len(objects):
        raise PilotError('complete automatic discovery denominator differs')
    scene_id=summary.get('scene_id','09c1414f1b')
    if 'all_jobs_manifest_sha256' in audit:
        path=source/'all_jobs_manifest.json'
        if sha(path)!=audit['all_jobs_manifest_sha256']:
            raise PilotError('complete all-jobs seal changed')
        all_jobs=json.loads(path.read_text())
        if (all_jobs.get('kind')!='complete_automatic_discovery_jobs'
                or all_jobs.get('schema_version')!=1 or all_jobs.get('paper_ready') is not False
                or all_jobs['rows']!=expected or type(all_jobs['planned_jobs']) is not int
                or all_jobs['planned_jobs']!=len(labels)
                or all_jobs['scene_id']!=scene_id or all_jobs['freeze_id']!=summary['freeze_id']
                or all_jobs['code_commit']!=summary['code_commit']
                or all_jobs['source_gaussian_training_provenance']!=summary['source_gaussian_training_provenance']
                or summary['source_gaussian_training_provenance']!=manifest['source_gaussian_training_provenance']
                or all_jobs['input_manifest_sha256']!=sha(source/'input_manifest.json')
                or all_jobs['summary_sha256']!=sha(source/'pilot_summary.json')
                or all_jobs['output_hashes_sha256']!=sha(source/'output_hashes.json')):
            raise PilotError('complete all-jobs manifest binding differs')
    elif summary.get('source_gaussian_training_provenance','UNKNOWN')!='UNKNOWN':
        raise PilotError('fresh discovery requires complete all-jobs seal')
    allowed=set(manifest['boundary']['training_frames'])
    by_index={o['index']:o for o in objects};rows=[]
    for row in summary['rows']:
        record=dict(row)
        record['job_id']=f"{scene_id}:auto:{row['automatic_instance_id']}"
        if row['prepared']:
            index=row['output_index'];obj=by_index[index]
            if obj['gt_object_id']!=row['automatic_instance_id'] or obj['frame'] not in allowed:
                raise PilotError('prepared input identity/frame differs')
            rel=f'objects/obj_{index:02d}/rgba.png';path=source/'construction'/rel
            if sha(path)!=hashes[rel]['sha256']:raise PilotError('source crop changed')
            record.update(input=identity(path),frame=obj['frame'],object_metadata=obj)
        rows.append(record)
    return rows

def validate_models(c):
    records={}
    for name,spec in c['models'].items():
        path=Path(spec['path'])
        if 'tree_sha256' in spec:
            record=_tree_inventory(path,label=name)
            if record['tree_sha256']!=spec['tree_sha256']:raise PilotError(f'model tree differs: {name}')
        else:
            record=identity(path)
            if record['sha256']!=spec['sha256']:raise PilotError(f'model weight differs: {name}')
        records[name]=record
    return records


def cohort_models(config, root, *, tool="trellis", validator=None):
    validator = validator or validate_models
    if tool not in {'trellis', 'rvg', 'trellis2'}:raise PilotError('unknown model cache tool')
    if config.get('output_scene_id') is None:
        return validator(config)
    path = Path(root)/'contract'/f'{tool}_model_resources.json'
    binding = {'models':config['models'], 'runtime_sha256':config['runtime_sha256'],
               'python':config['python']}
    def stats():
        result = {}
        for spec in config['models'].values():
            p = Path(spec['path'])
            for f in (p.rglob('*') if p.is_dir() else [p]):
                if f.is_file():
                    st=f.stat();result[str(f.resolve())]=[st.st_size,st.st_mtime_ns]
        return result
    if path.exists():
        record=json.loads(path.read_text())
        if record['binding'] != binding or record['stats'] != stats():
            raise PilotError('shared generation model identity changed')
        for name, expected in config['models'].items():
            key='tree_sha256' if 'tree_sha256' in expected else 'sha256'
            if record['models'][name][key] != expected[key]:
                raise PilotError('shared generation model hash differs')
        return record['models']
    before=stats(); models=validator(config); after=stats()
    if before != after:raise PilotError('model changed during initial hash pass')
    write_new(path,dict(binding=binding,models=models,stats=after))
    return models


def planned_jobs(c, out):
    """Derive reuse/generate actions from frozen inputs, never previous quality."""
    from run.icra2027.e3_fresh_generation_contract import (
        discovery_binding, validate_crop_comparison, prior_trellis_source, reuse_rows,
    )
    validate_config(c)
    rows, binding = discovery_binding(c, source_jobs)
    tool = generator(c)
    for row in rows:
        row['proposal_id'] = f"{c.get('freeze_id', Path(out).parent.name)}:{row['job_id']}:{tool}:initial:seed42"
    reused, receipts = [], []
    if c.get('raw_reuse'):
        comparison = validate_crop_comparison(c, rows)
        prior = prior_trellis_source(c, source_jobs)
        reused, receipts = reuse_rows(c, dict(jobs=rows, **binding), prior, comparison)
        available = {r['job_id'] for r in reused if r['status'] == 'available'}
        for row in rows:
            row['generation_action'] = ('reuse' if row['job_id'] in available else
                                        'generate' if row['prepared'] else 'unavailable')
        binding['raw_reuse_eligibility']={'eligible':prior['reuse_eligible'],
            'reason':prior['reuse_ineligible_reason'],
            'reasons':prior.get('reuse_ineligible_reasons',[]),
            'runtime_identity_requirement':'independently anchored original runtime bytes; version equality alone is insufficient'}
    return rows, binding, reused, receipts


def needs_generation(job):
    return job['prepared'] and job.get('generation_action', 'generate') == 'generate'

def plan(config_path,root):
    c,git=context(config_path,root);out=output_directory(c,root)
    rows,binding,reused,receipts=planned_jobs(c,out)
    smoke=import_smoke(c);models=cohort_models(c,root,tool=generator(c));out.parent.mkdir(parents=True,exist_ok=True);out.mkdir()
    write_new(out/'import_smoke.json',smoke)
    packages,runtime_hash=runtime_identity(c['python'])
    write_new(out/'python_runtime.json',dict(runtime_sha256=runtime_hash,**packages))
    runtime_bytes=None
    if binding['source_gaussian_training_provenance']=='FRESH_OFFICIAL_TRAIN_ONLY':
        path=out/'targeted_runtime_bytes.json';write_new(path,runtime_bytes_identity(c))
        runtime_bytes=identity(path)
    receipt_identities=[]
    for index,receipt in enumerate(receipts):
        path=out/'raw_reuse_receipts'/f'{index:04d}.json'
        write_new(path,receipt);receipt_identities.append(identity(path))
    objects=[]
    for row in rows:
        if not needs_generation(row):continue
        obj=row['object_metadata'];objects.append(obj)
        target=out/'construction/objects'/f"obj_{obj['index']:02d}/rgba.png"
        target.parent.mkdir(parents=True);shutil.copyfile(row['input']['path'],target)
        if sha(target)!=row['input']['sha256']:raise PilotError('copied crop differs')
    write_new(out/'construction/objects/objects.json',objects)
    write_new(out/'input_manifest.json',dict(code_commit=git,config_sha256=sha(config_path),
      source_pilot=c['source_pilot'],source_summary_sha256=sha(Path(c['source_pilot'])/'pilot_summary.json'),
      paper_ready=False,seed=42,jobs=rows,models=models,**binding,
      **({'generator': 'trellis2', 'native_gaussian': False, 'pipeline_type':c['pipeline_type']} if generator(c)=='trellis2' else {}),
      reused_rows=[r for r in reused if r['status']=='available'],raw_reuse_receipts=receipt_identities,
      targeted_runtime_bytes=runtime_bytes,
      **({'runtime_byte_identity':c['runtime_byte_identity']} if c.get('runtime_byte_identity') else {})))
    print(json.dumps({'planned_jobs':len(rows),'prepared_inputs':sum(r['prepared'] for r in rows),
                     'generate_inputs':len(objects),'reused_inputs':len(receipts),'paper_ready':False}))

def validate_source_binding(c, out, manifest):
    """Rebuild expected jobs from config-bound source bytes, not mutable staging."""
    if generator(manifest) != generator(c):
        raise PilotError('staged generator differs from config-bound source')
    if generator(c) == 'trellis2' and (manifest.get('native_gaussian') is not False
            or manifest.get('pipeline_type') != c['pipeline_type']):
        raise PilotError('staged TRELLIS.2 capabilities or resolution differ')
    expected,binding,reused,receipts=planned_jobs(c,out)
    if manifest['jobs']!=expected:
        raise PilotError('staged manifest jobs differ from config-bound source')
    for key,value in binding.items():
        if manifest.get(key, 'UNKNOWN' if key=='source_gaussian_training_provenance' else None)!=value:
            raise PilotError('staged source provenance differs')
    if manifest.get('reused_rows',[])!=[r for r in reused if r['status']=='available']:
        raise PilotError('staged reused rows differ')
    from run.icra2027.e3_fresh_generation_contract import checked_identity
    runtime_bytes=manifest.get('targeted_runtime_bytes')
    if binding['source_gaussian_training_provenance']=='FRESH_OFFICIAL_TRAIN_ONLY':
        if not runtime_bytes:raise PilotError('fresh runtime byte audit absent')
        runtime=json.loads(checked_identity(runtime_bytes).read_text())
        if runtime!=runtime_bytes_identity(c):raise PilotError('targeted runtime bytes differ')
    anchors=manifest.get('raw_reuse_receipts',[])
    if len(anchors)!=len(receipts):raise PilotError('reuse receipt denominator differs')
    for index,(anchor,receipt) in enumerate(zip(anchors,receipts)):
        if (Path(anchor['path'])!=Path(out)/'raw_reuse_receipts'/f'{index:04d}.json'
                or json.loads(checked_identity(anchor).read_text())!=receipt):
            raise PilotError('frozen reuse receipt differs')


def validate_staged(out,manifest):
    expected=[]
    for row in manifest['jobs']:
        if needs_generation(row):
            expected.append(row['object_metadata']);p=out/'construction/objects'/f"obj_{row['output_index']:02d}/rgba.png"
            if sha(p)!=row['input']['sha256']:raise PilotError('staged generation crop differs')
    if json.loads((out/'construction/objects/objects.json').read_text())!=expected:
        raise PilotError('staged objects roster differs')

def worker(config_path,root):
    import runpy
    c,git=context(config_path,root);out=output_directory(c,root)
    m=json.loads((out/'input_manifest.json').read_text());validate_source_binding(c,out,m);validate_staged(out,m)
    expected = environment(c, CODE, out)
    keys = ['SIMANY_OUT','SIMANY_ROOT','SIMANY_GENERATION_RECORDS']
    keys += (['SIMANY_TRELLIS2_DIR','SIMANY_TRELLIS2_MODEL','SIMANY_DINOV3_MODEL','SIMANY_SS_DECODER','SIMANY_TRELLIS2_SOURCE_COMMIT','SIMANY_TRELLIS2_PIPELINE_TYPE','SIMANY_TRELLIS2_SEED'] if generator(c)=='trellis2' else ['SIMANY_TRELLIS_DIR','SIMANY_TRELLIS_MODEL','SIMANY_DINOV2_REPO'])
    if any(os.environ.get(k)!=expected[k] for k in keys):raise PilotError('generation environment differs')
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if Path('/data/ScanNetpp') in p.parents or Path('/data/ScanNetppv2_gsplat') in p.parents:
            raise PilotError('pure generation may not read original scene or Gaussian')
    sys.addaudithook(guard)
    runpy.run_module('models.s4_'+generator(c),run_name='__main__')

def collect_records(out,manifest,exit_code):
    rows=[]
    tool = generator(manifest)
    outputs = TRELLIS2_OUTPUTS if tool=='trellis2' else OUTPUTS
    reused={r['job_id']:r for r in manifest.get('reused_rows',[])}
    for job in manifest['jobs']:
        if job.get('generation_action')=='reuse':
            row=dict(reused[job['job_id']])
            receipt=[r for r in manifest['raw_reuse_receipts']
                     if json.loads(Path(r['path']).read_text())['new_job_id']==job['job_id']]
            if len(receipt)!=1:raise PilotError('reused proposal lacks unique handoff receipt')
            row['raw_reuse_receipt']=receipt[0];rows.append(row);continue
        row={k:job[k] for k in ('job_id','automatic_instance_id','proposal_id','prepared')}
        row.update(tool=tool,seed=42,paper_ready=False,shared_initial_policy_rows=[] if tool=='trellis2' else ['A1','A2','A3','A4'])
        if tool=='trellis2': row.update(native_gaussian=False, study_scope='trellis2_mesh_engineering', full_twin_ready=False)
        if not job['prepared']:
            row.update(status='unavailable',reason='preparation_unavailable',artifacts={})
        else:
            index=job['output_index'];odir=out/'construction/objects'/f'obj_{index:02d}'
            record=out/'producer_records'/f'object_{index:02d}.json'
            runtime=json.loads(record.read_text()) if record.is_file() else None
            complete=(runtime is not None and runtime.get('status')=='generated' and runtime.get('seed')==42
                      and all((odir/name).is_file() and (odir/name).stat().st_size for name in outputs))
            if tool=='trellis2' and complete:
                complete=(runtime.get('generator')=='trellis2'
                          and runtime.get('input_sha256')==job['input']['sha256']
                          and runtime.get('pipeline_type')==manifest['pipeline_type'])
                for name in outputs:
                    spec=runtime.get('artifacts',{}).get(name,{})
                    path=odir/name
                    if spec.get('sha256')!=sha(path) or spec.get('bytes')!=path.stat().st_size:
                        complete=False
            reason=None if complete else ((runtime or {}).get('reason') or f'producer_exit_{exit_code}_or_incomplete_artifacts')
            row.update(status='available' if complete else 'generation_failed',reason=reason,
                       input_sha256=job['input']['sha256'],frame=job['frame'],
                       artifacts={name:identity(odir/name) for name in outputs if (odir/name).is_file()},
                       runtime=runtime)
        rows.append(row)
    return rows

def execute(config_path,root):
    c,git=context(config_path,root);out=output_directory(c,root);m=json.loads((out/'input_manifest.json').read_text())
    if m['code_commit']!=git or m['config_sha256']!=sha(config_path):raise PilotError('input plan differs')
    validate_source_binding(c,out,m)
    validate_staged(out,m)
    # Exclusive execution claim survives failure; never overwrite a partial run.
    write_new(out/'execution_claim.json',dict(code_commit=git,config_sha256=sha(config_path),
              input_manifest_sha256=sha(out/'input_manifest.json'),paper_ready=False))
    try:
        _execute_claimed(c,git,out,m,config_path,root)
    except BaseException as error:
        write_attempt_status(out,m,'FAIL',f'{type(error).__name__}: {error}')
        raise
    else:
        write_attempt_status(out,m,'PASS',None)


def write_attempt_status(out,manifest,status,error):
    rows=collect_records(out,manifest,0 if status=='PASS' else 1)
    write_new(out/'execution_status.json',dict(status=status,error=error,paper_ready=False,
        planned_jobs=len(rows),available_job_ids=[r['job_id'] for r in rows if r['status']=='available'],
        missing_prepared_job_ids=[r['job_id'] for r in rows if r['prepared'] and r['status']!='available'],
        execution_claim_sha256=sha(out/'execution_claim.json'),
        input_manifest_sha256=sha(out/'input_manifest.json'),in_place_resume_supported=False,
        resume_requirement='New frozen attempt required. Existing per-object artifacts need independently validated input, producer, runtime and artifact proof before reuse; no overwrite or automatic regeneration is authorized by this status.'))


def _execute_claimed(c,git,out,m,config_path,root):
    # One model hash pass was performed in the CPU plan. Runtime checks file
    # sizes/mtimes against a stat snapshot captured immediately after it.
    stats=json.loads((out/'model_stats.json').read_text())
    current_paths=set()
    for model in c['models'].values():
        p=Path(model['path']);paths=p.rglob('*') if p.is_dir() else [p]
        current_paths.update(str(f.resolve()) for f in paths if f.is_file())
    if current_paths!=set(stats):raise PilotError('model tree topology changed since plan')
    for path,expected in stats.items():
        st=Path(path).stat()
        if [st.st_size,st.st_mtime_ns]!=expected:raise PilotError(f'model changed since plan: {path}')
    if not any(needs_generation(job) for job in m['jobs']):
        publish_pool(c,git,out,m,0,0.0,[],generation_performed=False)
        return
    node=os.environ.get('SLURMD_NODENAME','')
    if not os.environ.get('SLURM_JOB_ID') or not(node=='hala' or node.startswith(('gcp-','sof1-'))):raise PilotError('unauthorized node')
    probe=subprocess.check_output([c['python'],'-c','import torch,json;assert torch.cuda.device_count()==1;f,t=torch.cuda.mem_get_info();print(json.dumps(dict(name=torch.cuda.get_device_name(),free_bytes=f,total_bytes=t,torch=torch.__version__,cuda=torch.version.cuda)))'],text=True)
    gpu=json.loads(probe.strip().splitlines()[-1]);validate_gpu_memory(gpu)
    write_new(out/'runtime.json',dict(job_id=os.environ['SLURM_JOB_ID'],node=node,**gpu))
    (out/'producer_records').mkdir()
    env = environment(c, CODE, out)
    if generator(c)=='trellis2':
        for name in ('torch_extensions','triton','inductor','numba','flexgemm'):
            (out/'runtime_cache'/name).mkdir(parents=True, exist_ok=True)
    command=[c['python'],str(Path(__file__).resolve()),'--config',str(Path(config_path).resolve()),'--freeze-root',str(Path(root).resolve()),'--phase','worker']
    started=time.monotonic()
    with (out/'generation.log').open('x') as log:code=subprocess.call(command,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT)
    publish_pool(c,git,out,m,code,time.monotonic()-started,command,generation_performed=True)
    if code:raise PilotError('generation failed; preserve completed per-object artifacts')


def publish_pool(c,git,out,m,code,wall_s,command,*,generation_performed):
    validate_source_binding(c,out,m)
    rows=collect_records(out,m,code)
    status=m.get('source_gaussian_training_provenance','UNKNOWN')
    fresh=status=='FRESH_OFFICIAL_TRAIN_ONLY'
    result=dict(paper_ready=False,source_gaussian_training_provenance=status,code_commit=git,freeze_id=c['freeze_id'],planned_jobs=len(rows),prepared_inputs=sum(r['prepared'] for r in rows),available_trellis=sum(r['status']=='available' for r in rows),initial_pool_complete=False,input_manifest_sha256=sha(out/'input_manifest.json'),rvg_status='initial_tool_not_run' if fresh else 'unavailable_missing_pinned_dependencies',rvg_rows=[dict(job_id=r['job_id'],status='unavailable',reason=('initial_tool_not_run' if fresh else 'missing_pinned_dependencies') if r['prepared'] else 'preparation_unavailable') for r in rows],exit_code=code,wall_s=wall_s,command=command,rows=rows,generation_performed=generation_performed,reused_trellis=sum(r.get('generation_mode')=='validated_raw_reuse' for r in rows))
    if generator(c)=='trellis2':
        result['generator']='trellis2'
        result['available_trellis2']=result.pop('available_trellis')
        result.pop('reused_trellis')
        result.pop('rvg_status'); result.pop('rvg_rows')
        result.update(native_gaussian=False, full_twin_ready=False, study_scope='trellis2_mesh_engineering')
    if fresh:
        result['source_discovery_hashes']=m['source_discovery_hashes']
    with (out/'proposal_records.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    result['proposal_records_sha256']=sha(out/'proposal_records.jsonl')
    write_new(out/'proposal_pool.json',result)
    print(json.dumps(result,indent=2))

def main():
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--freeze-root',required=True);a.add_argument('--phase',required=True,choices=['plan','run','worker','audit','mesh-probe']);args=a.parse_args()
    if args.phase=='plan':
        plan(args.config,args.freeze_root)
        c=yaml.safe_load(Path(args.config).read_text());stats={}
        for model in c['models'].values():
            p=Path(model['path']);paths=sorted(p.rglob('*')) if p.is_dir() else [p]
            for f in paths:
                if f.is_file():s=f.stat();stats[str(f.resolve())]=[s.st_size,s.st_mtime_ns]
        write_new(output_directory(c,args.freeze_root)/'model_stats.json',stats)
    elif args.phase=='run':execute(args.config,args.freeze_root)
    elif args.phase=='audit':
        from run.icra2027.e3_generator_backend import audit_trellis2
        audit_trellis2(args.config,args.freeze_root)
    elif args.phase=='mesh-probe':
        from run.icra2027.e3_trellis2_mesh_probe import probe
        probe(args.config,args.freeze_root)
    else:worker(args.config,args.freeze_root)
if __name__=='__main__':main()
