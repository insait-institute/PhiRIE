"""Source-bound launcher; canonical E3 owns every inventory/observation/decision."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import yaml

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE))
from robo.eval import agentic_ablation as e3


def scene_slots(config):
    if config.get('schema_version') == 1:
        mapping = {config['scene_id']: config['object_slots']}
    elif config.get('schema_version') == 2:
        mapping = config['scene_object_slots']
        if (not isinstance(mapping, dict) or list(mapping) != config['scene_ids']
                or type(config['planned_scenes']) is not int
                or len(mapping) != config['planned_scenes']):
            raise ValueError('canonical cohort scene roster differs')
    else:
        raise ValueError('unsupported canonical execution schema')
    for scene, slots in mapping.items():
        e3._require_scene_id(scene)
        if not isinstance(slots, list) or len(set(slots)) != len(slots):
            raise ValueError('canonical object slots must be unique within each scene')
        for slot in slots: e3._require_object_slot(slot)
    if sum(map(len, mapping.values())) != config['planned_jobs']:
        raise ValueError('canonical scene populations differ from planned jobs')
    return mapping


def selected_scene(config, phase, scene_id):
    slots = scene_slots(config)
    if phase in {'observe', 'control'}:
        selected = scene_id or (config['scene_id'] if config['schema_version'] == 1 else None)
        if selected not in slots:
            raise ValueError('scene phase requires an explicit frozen scene')
        return selected
    if scene_id is not None:
        raise ValueError('global phase may not select a scene subset')
    return None


def validate(config_path, root, phase, scene_id=None):
    if e3.CODE_ROOT != CODE:
        raise ValueError('canonical implementation differs from launcher checkout')
    config_path=e3._checked_code_path(config_path,'phase config')
    config=yaml.safe_load(config_path.read_text())
    root=e3.checked_repo_path(root,'canonical freeze',kind='dir')
    if (config.get('schema_version') not in {1,2} or config.get('scope')!='fresh_canonical_engineering'
            or config.get('paper_ready') is not False or config['freeze_id']!=root.name
            or type(config['planned_jobs']) is not int or config['planned_jobs']<0
            or config['planned_policy_object_rows']!=5*config['planned_jobs']):
        raise ValueError('canonical engineering population/scope differs')
    slots = scene_slots(config)
    selected_scene(config, phase, scene_id)
    paths=[config_path,config['policies_config'],config['observation_runtime_config']]
    if phase=='inventory':paths.append(config['jobs_config'])
    contract=e3._validate_cli_execution(root/'contract/freeze_manifest.json',root.name,
        root/'agentic',config_paths=paths,require_inventory=phase in {'observe','control'})
    if (e3.sha256_file(CODE/config['policies_config'])!=config['policies_sha256']
            or e3.sha256_file(CODE/config['observation_runtime_config'])!=config['observation_runtime_config_sha256']):
        raise ValueError('frozen policy/runtime recipe differs')
    if phase in {'observe','control'}:
        jobs,_=e3._load_inventory(root/'agentic',controller_safe=True)
        if (jobs['counts']!={'scenes':len(slots),'jobs':config['planned_jobs'],
                           'policy_object_rows':config['planned_policy_object_rows']}
                or [s['scene_id'] for s in jobs['scenes']]!=list(slots)
                or any([j['object_slot'] for j in s['jobs']] != slots[s['scene_id']]
                       for s in jobs['scenes'])):
            raise ValueError('sealed canonical population differs from phase config')
    return config,root,contract


def guard_for(phase, config, gaussian_paths=()):
    gaussian_paths={Path(p).resolve() for p in gaussian_paths}
    def guard(event,args):
        if event=='socket.connect':raise ValueError('canonical engineering phase forbids network access')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        # Block every scene's GT/original RGB, including scenes outside this shard.
        data = Path('/data/ScanNetpp/data')
        if data in path.parents:
            parts = path.relative_to(data).parts
            if len(parts) >= 2 and parts[1] == 'scans':
                raise ValueError('evaluation surfaces may not enter canonical construction')
            if len(parts) >= 3 and parts[1:3] == ('dslr','resized_undistorted_images'):
                raise ValueError('canonical phase consumes frozen crops, never original RGB')
        if phase=='control' and (Path('/data/ScanNetpp') in path.parents
                                or Path('/data/ScanNetppv2_gsplat') in path.parents
                                or path in gaussian_paths):
            raise ValueError('control consumes sealed observations, not original scene')
    return guard


def run(config_path,root,phase,scene_id=None):
    if os.environ.get('SLURM_ARRAY_JOB_ID'):
        raise ValueError('canonical phases require ordinary individual jobs')
    config,root,contract=validate(config_path,root,phase,scene_id)
    scene = selected_scene(config, phase, scene_id)
    slots = scene_slots(config)
    if scene is not None and os.environ.get('SIMANY_SCENE') != scene:
        raise ValueError('runtime scene environment differs from selected frozen scene')
    observation=yaml.safe_load((CODE/config['observation_runtime_config']).read_text())
    if phase in {'observe','preflight'}:
        from run.icra2027.e3_gaussian_train_only import verify_runtime
        observation_runtime=verify_runtime(observation)
    else:observation_runtime=None
    from run.icra2027.e3_trellis_generation_pilot import runtime_identity
    control_runtime,digest=runtime_identity(config['control_python'])
    if digest!=config['control_runtime_sha256']:
        raise ValueError('frozen control runtime differs')
    required_python=config['observation_python'] if phase=='observe' else config['control_python']
    if Path(sys.executable).resolve()!=Path(required_python).resolve():
        raise ValueError('phase interpreter differs from frozen recipe')
    gpu=None
    if phase in {'observe','control'}:
        node=os.environ.get('SLURMD_NODENAME','').lower()
        if not os.environ.get('SLURM_JOB_ID') or not (node=='hala' or node.startswith(('gcp','sof1'))):
            raise ValueError('canonical phase requires an allowed Slurm node')
        if phase=='observe':
            import torch
            from run.icra2027.e3_auto_discovery_pilot import validate_gpu_memory
            if torch.cuda.device_count()!=1:raise ValueError('exactly one GPU required')
            free,total=torch.cuda.mem_get_info()
            gpu=dict(name=torch.cuda.get_device_name(),free_bytes=free,total_bytes=total)
            validate_gpu_memory(gpu)
        elif os.environ.get('SLURM_GPUS_ON_NODE','0') not in {'','0'}:
            raise ValueError('control must request CPU resources only')
    command=['--jobs',config['jobs_config'],'--policies',config['policies_config'],
        '--contract-manifest',str(root/'contract/freeze_manifest.json'),'--freeze-id',root.name,
        '--out',str(root/'agentic'),'--'+phase]
    if scene is not None:command.extend(['--scene-id',scene])
    receipt=dict(schema_version=1,paper_ready=False,phase=phase,scene_id=scene,
        code_commit=contract['code']['commit'],contract_sha256=contract['contract_sha256'],
        config_sha256=e3.sha256_file(config_path),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        node=os.environ.get('SLURMD_NODENAME'),planned_jobs=len(slots[scene]) if scene else config['planned_jobs'],
        planned_policy_object_rows=5*len(slots[scene]) if scene else config['planned_policy_object_rows'],
        control_runtime_sha256=digest,observation_runtime=observation_runtime,gpu=gpu,command=command)
    receipt_root = root
    if config['schema_version'] == 2 and scene is not None:
        receipt_root = root/'execution_receipts'/scene
        receipt_root = e3.checked_repo_path(receipt_root,'scene receipt root',must_exist=False)
        receipt_root.mkdir(parents=True,exist_ok=True)
    def write(name,value):
        path=e3.checked_repo_path(receipt_root/name,'phase receipt',must_exist=False)
        with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    write(phase+'_execution_start.json',receipt)
    if phase=='preflight':
        write(phase+'_execution_result.json',dict(status='PASS',paper_ready=False))
        return 0
    if phase in {'observe','control'}:
        inventory,_=e3._load_inventory(root/'agentic',controller_safe=True)
        sys.addaudithook(guard_for(phase,config,
            [s['source_scene_gaussian']['path'] for s in inventory['scenes']]))
    start=time.monotonic()
    try:
        code=e3.main(command)
        if code:raise RuntimeError(f'canonical producer exited {code}')
        if phase=='control':
            directory,shard,_=e3._load_control_scene(root/'agentic',scene)
            rows=[json.loads(line) for line in (directory/'job_ledger.jsonl').read_text().splitlines()]
            if len(rows)!=5*len(slots[scene]):
                raise ValueError('canonical control lost planned terminal rows')
        write(phase+'_execution_result.json',dict(status='PASS',exit_code=code,
            wall_s=time.monotonic()-start,paper_ready=False))
        return code
    except Exception as error:
        write(phase+'_execution_failure.json',dict(status='FAIL',type=type(error).__name__,
            reason=str(error),wall_s=time.monotonic()-start,paper_ready=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--freeze-root',required=True)
    parser.add_argument('--phase',required=True,choices=['preflight','inventory','observe','control'])
    parser.add_argument('--scene-id')
    args=parser.parse_args();return run(args.config,args.freeze_root,args.phase,args.scene_id)

if __name__=='__main__':raise SystemExit(main())
