"""Resolve immutable constructor receipts and dispatch ordinary canonical engines."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from robo.roundtrip.matrix import (read_rows,save_new,sha,canonical_hash,commands_for_units,
                                  validate_terminal,IDENTITY_FIELDS)
from robo.roundtrip.local_policy_instance import preflight_engine
from robo.manifest.hash import git_snapshot

METHODS=['REF_NATIVE','B0_FIXED_NATIVE','B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE']
DISCOVERY_FAILURES=('no automatic target mask with usable TRAIN depth',
                    'automatic target lacks predeclared multiview depth confirmation')


def b0_bindings(binding,build_root):
    """Only a proven shared discovery failure can propagate to all arms."""
    root=Path(build_root)/binding['instance_slot_id'];config=root/'build_config.json'
    if not config.exists():return []
    c=json.loads(config.read_text())
    for key in ('canonical_instance_id','capture_manifest_sha256','cohort_id'):
        if c.get(key)!=binding[key]:raise ValueError('B0 acquisition binding differs: '+key)
    manifest=root/'build_manifest.json';failure=root/'build_failure.json'
    if manifest.exists() and failure.exists():raise ValueError('contradictory B0 build terminals')
    base={'canonical_instance_id':binding['canonical_instance_id']}
    if manifest.exists():
        m=json.loads(manifest.read_text())
        if m.get('status')!='BUILT':return []
        for key in ('canonical_instance_id','capture_manifest_sha256','cohort_id'):
            if m.get(key)!=binding[key]:raise ValueError('B0 manifest binding differs')
        for rel,digest in m.get('source_hashes',{}).items():
            relpath=Path(rel)
            if relpath.is_absolute() or '..' in relpath.parts or sha(root/relpath)!=digest:raise ValueError('B0 artifact hash changed')
        obj=root/'construction/objects/obj_00'
        if not all((obj/n).is_file() for n in ('aligned.json','physics.json')):raise ValueError('BUILT B0 lacks native object contract')
        return [{**base,'controller_method':METHODS[1],'accepted':True,'terminal_status':'BUILT',
                 'object_dir':str(obj.resolve()),'build_manifest':str(manifest.resolve()),'build_manifest_sha256':sha(manifest)}]
    if not failure.exists():return []
    f=json.loads(failure.read_text());log=root/'segment.log';runtime=root/'segment_runtime.json'
    if f.get('status')!='construction_unavailable' or f.get('phase')!='segment' or not log.exists() or not runtime.exists():return []
    if json.loads(runtime.read_text()).get('exit_code')!=1:return []
    reason=next((r for r in DISCOVERY_FAILURES if ('ValueError: '+r) in log.read_text()),None)
    if reason is None:return []  # environment/runtime failures are unmeasured, never inferred build loss
    evidence={str(p.resolve()):sha(p) for p in (failure,config,log,runtime)}
    return [{**base,'controller_method':method,'accepted':False,'terminal_status':'BUILD_FAILED','object_dir':None,
             'build_manifest':str(failure.resolve()),'build_manifest_sha256':sha(failure),'source_evidence':evidence,
             'failure':reason,'propagation_rule':'all arms require the same frozen automatic TRAIN discovery; no initial pool exists'} for method in METHODS[1:]]


def read_context_bindings(paths):
    rows=[]
    for path in paths:
        for row in read_rows(path):
            if row['controller_method'] not in METHODS[2:]:raise ValueError('unexpected context method')
            if sha(row['build_manifest'])!=row['build_manifest_sha256']:raise ValueError('context manifest changed')
            rows.append(row)
    return rows


def corrected_b0(override,original,binding):
    """Accept only a hash-proven dimensions metadata repair; no geometry edits."""
    if override.get('controller_method')!='B0_FIXED_NATIVE' or override.get('accepted') is not True:
        raise ValueError('metadata override must be an accepted B0 asset')
    for key in ('canonical_instance_id','capture_manifest_sha256','instance_slot_id'):
        if override.get(key)!=binding[key]:raise ValueError('metadata override identity differs')
    for key in ('parent_build_manifest','build_manifest','metadata_repair_receipt'):
        if sha(override[key])!=override[key+'_sha256']:raise ValueError('metadata override lineage changed')
    if original['build_manifest_sha256']!=override['parent_build_manifest_sha256']:
        raise ValueError('metadata override parent is not the frozen B0 build')
    proof=json.loads(Path(override['metadata_repair_receipt']).read_text())
    if (proof.get('kind')!='implementation_metadata_correction' or proof.get('changed_files')!=['aligned.json']
        or proof.get('changed_fields')!=['world_dims'] or proof.get('referenced_surface_equal') is not True
        or proof.get('geometry_transform_collision_physics_unchanged') is not True or proof.get('outcome_input_used') is not False):
        raise ValueError('override is not a proven metadata-only correction')
    old=Path(original['object_dir']);new=Path(override['object_dir'])
    a=json.loads((old/'aligned.json').read_text());b=json.loads((new/'aligned.json').read_text())
    if {k:v for k,v in a.items() if k!='world_dims'}!={k:v for k,v in b.items() if k!='world_dims'}:
        raise ValueError('metadata correction changed alignment or transform')
    parents=proof['parent_hashes'];children=proof['child_hashes']
    if set(parents)!=set(children):raise ValueError('metadata repair file roster changed')
    required={'aligned.json','physics.json','mesh_sim.obj','mesh_sim.ply'}
    required.update(str(p.relative_to(old)) for p in (old/'collision').glob('*.obj'))
    if not required.issubset(parents):raise ValueError('metadata repair missing native geometry/physics proof')
    for rel in parents:
        path=Path(rel)
        if path.is_absolute() or '..' in path.parts:raise ValueError('metadata proof path escape')
        if sha(old/path)!=parents[rel] or sha(new/path)!=children[rel]:raise ValueError('corrected metadata artifact changed')
        if rel!='aligned.json' and parents[rel]!=children[rel]:raise ValueError('geometry or physics changed in metadata repair')
    return override


def nonrollout(unit,binding):
    status=binding['terminal_status']
    if status not in ('BUILD_FAILED','ABSTAINED') or binding.get('accepted') is not False:raise ValueError('explicit nonrollout binding required')
    terminal={**{k:unit[k] for k in IDENTITY_FIELDS},'unit_id':unit['unit_id'],
        'planned_unit_sha256':canonical_hash(unit),'terminal_status':status,'executed':False,'success':None,
        'failure':binding.get('failure',status),'construction_binding':binding,'source_code':git_snapshot()}
    return validate_terminal(unit,terminal)


def dispatch_ready(planned,bindings,b0_root,context_paths,out,*,worker_root,worker_source,admission,max_jobs=48,hold_new=False,b0_overrides=()):
    if max_jobs<=0:raise ValueError('positive bounded job limit required')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);worker_root=Path(worker_root).resolve()
    contexts=read_context_bindings(context_paths);results=[];submissions=0
    for binding in bindings:
        instance=binding['canonical_instance_id'];units=[u for u in planned if u['canonical_instance_id']==instance]
        if len(units)!=50:raise ValueError('TEST requires50 planned units per canonical instance')
        intent=out/instance/'submission_intent.json'
        if intent.exists():continue  # scheduler ambiguity is not permission to duplicate
        builds=b0_bindings(binding,b0_root);by_method={b['controller_method']:b for b in builds}
        replacements=[r for r in b0_overrides if r['canonical_instance_id']==instance]
        if len(replacements)>1:raise ValueError('duplicate B0 metadata overrides')
        if replacements:
            if 'B0_FIXED_NATIVE' not in by_method:raise ValueError('metadata correction lacks verified original B0 build')
            by_method['B0_FIXED_NATIVE']=corrected_b0(replacements[0],by_method['B0_FIXED_NATIVE'],binding)
        for b in contexts:
            if b['canonical_instance_id']!=instance:continue
            method=b['controller_method']
            if method in by_method:
                if by_method[method]['terminal_status']=='BUILD_FAILED' and b['terminal_status']=='BUILD_FAILED':continue
                raise ValueError('duplicate or contradictory construction bindings')
            by_method[method]=b
        if set(by_method)!=set(METHODS[1:]):continue
        current=out/instance;current.mkdir(parents=True,exist_ok=True)
        for unit in units:
            b=by_method.get(unit['controller_method'])
            if b is None or b['terminal_status'] not in ('BUILD_FAILED','ABSTAINED'):continue
            terminal=nonrollout(unit,b);path=worker_root/unit['unit_id']/'terminal.json'
            if path.exists():validate_terminal(unit,json.loads(path.read_text()))
            else:save_new(path,terminal)
        commands=commands_for_units(units,list(by_method.values()),host='localhost',port=8017,
            python_native='/group/worldcept/PhiRIE/code/SimAny-wt/sr0-native/.venv-native/bin/python')
        preflight_engine(units,commands,worker_root,METHODS,10)
        # Immutable per-instance snapshot; canonical planned config paths stay fixed.
        if not (current/'planned_units.jsonl').exists():
            save_new(current/'planned_units.jsonl',units,jsonl=True);save_new(current/'commands.json',commands)
            save_new(current/'build_bindings.jsonl',list(by_method.values()),jsonl=True)
        argv=['bash','run/roundtrip/local_policy_instance.sh','--planned',str(current/'planned_units.jsonl'),
            '--commands',str(current/'commands.json'),'--instance-id',instance,'--worker-root',str(worker_root),
            '--out',str(current/'endpoint'),'--expected-resets','10','--engine-protocol','per_canonical_engine_v1','--admission',str(Path(admission).resolve())]
        script=current/'engine.sbatch'
        if not script.exists():script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(Path(worker_source).resolve()))+'\nexec '+shlex.join(argv)+'\n')
        cmd=['sbatch','--parsable','--partition=debug','--qos=debug','--constraint=zone-sof1|zone-gcp-eu1',
             '--nodes=1','--ntasks=1','--gres=gpu:a6000:1','--cpus-per-task=8','--mem=64G','--time=01:30:00',
             '--job-name=n1-testengine-'+binding['instance_slot_id'][-8:],'--output='+str(current/'engine-%j.log'),str(script)]
        if hold_new:cmd.insert(1,'--hold')
        save_new(intent,{'canonical_instance_id':instance,'state':'SUBMITTING','argv':cmd,'source_code':git_snapshot()})
        proc=subprocess.run(cmd,capture_output=True,text=True)
        receipt={'canonical_instance_id':instance,'state':'SUBMITTED' if proc.returncode==0 else 'SUBMIT_FAILED',
            'job_id':proc.stdout.strip().split(';')[0] if proc.returncode==0 else None,'returncode':proc.returncode,'stderr':proc.stderr,
            'planned_units':50,'executable_units':len(commands),'worker_source':str(Path(worker_source).resolve())}
        save_new(current/'submission.json',receipt);results.append(receipt);submissions+=1
        if submissions>=max_jobs:break
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('planned','bindings','b0-root','out','worker-root','worker-source','admission'):p.add_argument('--'+key,required=True)
    p.add_argument('--context-bindings',action='append',default=[]);p.add_argument('--max-jobs',type=int,default=48)
    p.add_argument('--hold-new',action='store_true');p.add_argument('--b0-overrides')
    a=p.parse_args(argv)
    print(json.dumps(dispatch_ready(read_rows(a.planned),read_rows(a.bindings),a.b0_root,a.context_bindings,a.out,
        worker_root=a.worker_root,worker_source=a.worker_source,admission=a.admission,max_jobs=a.max_jobs,hold_new=a.hold_new,b0_overrides=read_rows(a.b0_overrides) if a.b0_overrides else ()),indent=2))
if __name__=='__main__':main()
