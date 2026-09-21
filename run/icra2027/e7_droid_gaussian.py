"""Source-bound DROID GS continuation; robot evidence is sealed TRAIN only.

This orchestrates the existing scene exporter and Gaussian trainer. It never
opens CPU outcome or held-out kinematic evaluation files.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from agents.recon.colmap_poses import write_new_json
from agents.recon.droid_extract import input_identity, verify_input, prospective_environment
from robo.manifest.hash import canonical_hash, git_snapshot
from run.icra2027.e3_gaussian_train_only import runtime_identity

CODE = Path(__file__).resolve().parents[2]
SCOPE = 'droid_sealed_train_fit_gaussian_continuation'
# This executes in the ORIGINAL CPU source/runtime, on its original CPU host.
# No outcome/reference parser is imported or invoked.
SOURCE_PROBE = '''
import json,sys
from pathlib import Path
from robo.eval.real_world_records import validate_prospective
from agents.recon.align_to_traj import validate_train_fit
from agents.recon.droid_extract import input_identity,verify_input
config,stage,code,workspace=sys.argv[1:]
c,e=validate_prospective(config,stage,code)
cap=[r for r in c['captures'] if r['workspace_id']==workspace]
if len(cap)!=1: raise ValueError('workspace outside original fixed cohort')
u=Path(stage)/'real_world/workspaces'/workspace
base={'capture':cap[0],'contract_sha256':e['contract_sha256'],'source_commit':e['code']['commit']}
if not (u/'alignment/seal.json').exists():
 print('E7_TRAIN_SOURCE_JSON='+json.dumps(dict(base,eligible=False,reason='sealed_TRAIN_fit_unavailable')));sys.exit(0)
fit=validate_train_fit(u/'alignment')
if fit['plan']!=cap[0]['plan']: raise ValueError('fit belongs to another original slot')
ex=json.loads((u/'extraction/extraction_manifest.json').read_text())
if ex['plan']!=cap[0]['plan'] or ex['train_trajectory']!=fit['train']: raise ValueError('original extraction closure differs')
for ref in ex['frames']: verify_input(ref)
train=json.loads(Path(fit['train']['path']).read_text())
if fit['train']['path']!=str(u/'extraction/train_trajectory.json'): raise ValueError('TRAIN path differs')
print('E7_TRAIN_SOURCE_JSON='+json.dumps({'eligible':True,'extraction':ex,'extraction_identity':input_identity(u/'extraction/extraction_manifest.json'),'capture':cap[0],'fit':fit,'seal':input_identity(u/'alignment/seal.json'),'fit_identity':input_identity(u/'alignment/fit.json'),'contract_sha256':e['contract_sha256'],'source_commit':e['code']['commit']}))
'''


def tree(directory):
    directory = Path(directory).absolute()
    if directory.resolve() != directory or not directory.is_dir():
        raise ValueError('artifact directory must be canonical and real')
    result = {}
    for p in sorted(directory.rglob('*')):
        if p.is_symlink():
            raise ValueError('artifact symlink forbidden')
        if p.is_file():
            result[str(p.relative_to(directory))] = input_identity(p)
    return result


def verify_tree(directory, identities):
    if tree(directory) != identities:
        raise ValueError('artifact inventory changed')


def source_config(config):
    src = config['cpu_source']
    old = Path(src['code_root'])
    if old.resolve() != old or git_snapshot(old)['commit'] != src['commit'] or git_snapshot(old)['dirty']:
        raise ValueError('original CPU source changed')
    c = json.loads(verify_input(src['config']).read_text())
    contract = json.loads(verify_input(src['contract']).read_text())
    payload = {k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
    if (canonical_hash(payload) != contract['contract_sha256'] or contract['code']['commit'] != src['commit']
            or contract['code']['dirty'] or contract['freeze_id'] != Path(src['stage_root']).name):
        raise ValueError('original CPU E0 differs')
    bound = [x for x in contract['configs'] if x['field'] == 'real_world_config']
    if len(bound) != 1 or bound[0]['source_content_sha256'] != src['config']['sha256']:
        raise ValueError('original CPU config not E0 bound')
    ids = [r['workspace_id'] for r in c['captures']]
    if len(ids) != 10 or len(set(ids)) != 10 or config['workspace_ids'] != ids:
        raise ValueError('all original ten slots must remain in order')
    if config['pilot_workspace'] != ids[0]:
        raise ValueError('first original IPRL pilot must remain fixed')
    rule = json.loads(verify_input(config['continuation_rule']).read_text())
    if (rule['planned_workspace_ids'] != ids or rule['training_iterations'] != 30000
            or rule['all_original_slots_retained'] is not True or rule['threshold_or_view_policy_tuning'] is not False
            or rule['config'] != {k:src['config'][k] for k in ('path','sha256')}):
        raise ValueError('outcome-independent continuation rule differs')
    return c, contract


def context(config_path, stage):
    c = json.loads(Path(config_path).read_text())
    if (c.get('scope') != SCOPE or c.get('paper_ready') is not False
            or (c.get('iters'),c.get('smoke_iters'),c.get('seed'),c.get('holdout_every')) != (30000,16,0,10)
            or c.get('full_build') != 'NOT_RUN' or c.get('independent_image_fidelity') is not False):
        raise ValueError('fixed GS recipe/scope changed')
    stage = Path(stage).absolute()
    if stage.resolve() != stage or stage.name != c['freeze_id']:
        raise ValueError('GS stage path/freeze differs')
    code = git_snapshot(CODE)
    e = json.loads((stage/'contract/freeze_manifest.json').read_text())
    payload = {k:v for k,v in e.items() if k not in {'created_utc','environment','contract_sha256'}}
    if (code['dirty'] or code['commit']=='nogit' or e['code']['commit'] != code['commit']
            or e['code']['dirty'] or e['freeze_id'] != stage.name or canonical_hash(payload)!=e['contract_sha256']):
        raise ValueError('exact clean GS E0 required')
    bound = [r for r in e['resource_inventory'] if r['id']=='e7_gaussian_config']
    if len(bound)!=1 or bound[0]['sha256']!=input_identity(config_path)['sha256']:
        raise ValueError('GS config not E0 bound')
    source_config(c)
    return c,code,e


def require_cpu_terminal(c, workspace=None):
    if set(c['cpu_submissions']) != set(c['workspace_ids']):
        raise ValueError('all original CPU attempt identities required')
    jobs=[]
    for wid,identity in c['cpu_submissions'].items():
        r=json.loads(verify_input(identity).read_text())
        if r['source_commit']!=c['cpu_source']['commit'] or r['freeze_id']!=Path(c['cpu_source']['stage_root']).name:
            raise ValueError('CPU submission source/freeze differs')
        if r.get('workspace_id',c['pilot_workspace'])!=wid:
            raise ValueError('CPU submission belongs to another slot')
        jobs.append(str(r['job_id']))
    if len(set(jobs))!=10 or any(not j.isdecimal() for j in jobs):
        raise ValueError('ten unique CPU job IDs required')
    raw=subprocess.check_output(['sacct','-j',','.join(jobs),'-nP','--format=JobIDRaw,State'],text=True)
    states={parts[0]:parts[1].split()[0] for line in raw.splitlines() if len(parts:=line.split('|'))>=2}
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}
    required=jobs if workspace is None else [str(json.loads(verify_input(c['cpu_submissions'][workspace]).read_text())['job_id'])]
    if any(states.get(j) not in terminal for j in required):
        raise ValueError('CPU cohort still running; unavailable TRAIN fits cannot yet be classified')
    return states


def train_source(c, workspace):
    if workspace not in c['workspace_ids']:
        raise ValueError('workspace outside frozen population')
    s = c['cpu_source']
    if os.environ.get('SLURMD_NODENAME',socket.gethostname()) != s['validation_host']:
        raise ValueError('original TRAIN numerical replay requires original CPU host')
    env = prospective_environment(s['code_root'])
    env['SIMANY_EVIDENCE_ROOT'] = c['evidence_root']
    raw = subprocess.check_output([s['python'],'-c',SOURCE_PROBE,s['config']['path'],s['stage_root'],s['code_root'],workspace],
        cwd=s['code_root'],env=env,text=True)
    rows = [r.removeprefix('E7_TRAIN_SOURCE_JSON=') for r in raw.splitlines() if r.startswith('E7_TRAIN_SOURCE_JSON=')]
    if len(rows)!=1:
        raise ValueError('original TRAIN validator must emit exactly one receipt')
    out=json.loads(rows[0]);cpu,e=source_config(c)
    if (out['source_commit']!=s['commit'] or out['contract_sha256']!=e['contract_sha256']
            or out['capture']!=next(r for r in cpu['captures'] if r['workspace_id']==workspace)):
        raise ValueError('original TRAIN receipt closure differs')
    return out


def validate_source_receipt(c, workspace, source):
    cpu,e=source_config(c)
    cap=next(r for r in cpu['captures'] if r['workspace_id']==workspace)
    if (source['capture']!=cap or source['source_commit']!=c['cpu_source']['commit']
            or source['contract_sha256']!=e['contract_sha256'] or source['eligible'] is not True):
        raise ValueError('prepared source belongs to another CPU slot or contract')
    u=Path(c['cpu_source']['stage_root'])/'real_world/workspaces'/workspace
    expected={'fit_identity':u/'alignment/fit.json','seal':u/'alignment/seal.json',
              'extraction_identity':u/'extraction/extraction_manifest.json'}
    for key,path in expected.items():
        if Path(source[key]['path'])!=path:
            raise ValueError('prepared original source path differs')
        verify_input(source[key])
    fit=json.loads(Path(source['fit_identity']['path']).read_text())
    extraction=json.loads(Path(source['extraction_identity']['path']).read_text())
    seal=json.loads(Path(source['seal']['path']).read_text())
    if (source['fit']!=fit or source['extraction']!=extraction or fit['plan']!=cap['plan']
            or extraction['plan']!=cap['plan'] or extraction['train_trajectory']!=fit['train']
            or seal!={'fit_sha256':source['fit_identity']['sha256'],'fit_digest':canonical_hash(fit)}):
        raise ValueError('embedded prepared source payload differs from original sealed bytes')
    for key,path in {'train':u/'extraction/train_trajectory.json','recon':u/'recon.npz','aligned':u/'alignment/recon_base.npz'}.items():
        if Path(fit[key]['path'])!=path:
            raise ValueError('prepared fit artifact belongs to another workspace')
        verify_input(fit[key])
    for identity in extraction['frames']:
        path=Path(identity['path'])
        if path.parent!=u/'extraction/wrist_frames':
            raise ValueError('extracted RGB belongs to another workspace')
        verify_input(identity)
    return source


def training_environment(c, workspace):
    env=prospective_environment(CODE)
    env.update(PYTHONPATH=str(CODE)+os.pathsep+c['training_dependency_root'],
        TORCH_EXTENSIONS_DIR=c['torch_extensions_root'],SIMANY_SCENE=workspace,
        SIMANY_ROOT=str(CODE),SIMANY_EVIDENCE_ROOT=c['evidence_root'])
    return env


def verify_training_runtime(c, workspace):
    raw=subprocess.check_output([c['training_python'],'-m','run.icra2027.e7_droid_gaussian','--runtime-identity'],
        cwd=CODE,env=training_environment(c,workspace),text=True)
    if json.loads(raw)!=c['training_runtime_identity']:
        raise ValueError('GS runtime closure changed')
    if canonical_hash(tree(c['training_dependency_root']))!=c['training_dependency_tree_sha256']:
        raise ValueError('GS dependency overlay bytes changed')


def validate_prepared(c, workspace, destination, config_path, code):
    receipt=json.loads((destination/'prepare_receipt.json').read_text())
    if (receipt['status']!='PASS' or receipt['code']!=code or receipt['config']!=input_identity(config_path)
            or receipt['workspace_id']!=workspace):
        raise ValueError('same-source successful preparation required')
    source=validate_source_receipt(c,workspace,receipt['train_source']);fit=source['fit']
    if source['source_commit']!=c['cpu_source']['commit']:
        raise ValueError('original fit source differs')
    for key in ('fit_identity','seal','extraction_identity'):
        verify_input(source[key])
    for key in ('plan','train','recon','aligned'):
        verify_input(fit[key])
    for identity in receipt['public_rgb'].values():
        verify_input(identity)
    verify_tree(destination/'scene',receipt['scene_artifacts'])
    return receipt


def validate_training_report(report, prepared, iterations):
    names=set(prepared['public_rgb'])
    train=report['gradient_train_frames'];diagnostic=report['internal_diagnostic_frames']
    if (report['iters']!=iterations or report['seed']!=0 or report['independent_heldout_evaluation'] is not False
            or report['bitwise_determinism_claimed'] is not False or report['n_gaussians']<=0
            or not math.isfinite(report['wall_s']) or report['wall_s']<=0
            or not math.isfinite(report['psnr_internal_diagnostic'])
            or set(train)&set(diagnostic) or set(train)|set(diagnostic)!=names
            or len(train)!=len(set(train)) or len(diagnostic)!=len(set(diagnostic))
            or (report['n_train'],report['n_holdout'])!=(len(train),len(diagnostic))):
        raise ValueError('canonical training report differs from fixed recipe/input roster')


def run(config_path, stage, workspace, phase):
    c,code,e=context(config_path,stage)
    if workspace not in c['workspace_ids'] or phase not in {'prepare','train-smoke','train'}:
        raise ValueError('unknown phase/workspace')
    if phase=='train-smoke' and workspace!=c['pilot_workspace']:
        raise ValueError('native smoke only uses first preregistered slot')
    dest=Path(stage)/'gaussian'/workspace
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/(phase+'.lock')).open('x') as f:f.write(str(os.getpid()))
    start=time.monotonic();r={'schema_version':1,'scope':SCOPE,'workspace_id':workspace,'phase':phase,'code':code,
        'config':input_identity(config_path),'freeze_id':c['freeze_id'],'contract_sha256':e['contract_sha256'],
        'status':'RUNNING','job_id':os.environ.get('SLURM_JOB_ID'),'hostname':socket.gethostname(),
        'full_build':False,'paper_ready':False,'independent_image_fidelity':False}
    try:
        if phase=='prepare':
            src=train_source(c,workspace)  # MUST precede RGB/model input materialization.
            r['train_source']=src
            if not src['eligible']:
                require_cpu_terminal(c,workspace)
                r.update(status='NOT_RUN',train_gate='UNAVAILABLE',reason=src['reason'])
                return r
            import numpy as np
            rec=dict(np.load(src['fit']['aligned']['path'],allow_pickle=False))
            frames=Path(c['cpu_source']['stage_root'])/'real_world/workspaces'/workspace/'extraction/wrist_frames'
            names=[str(n) for n in rec['names']]
            if not names or len(names)!=len(set(names)) or any(Path(n).name!=n for n in names):
                raise ValueError('invalid registered RGB roster')
            r['public_rgb']={n:input_identity(frames/n) for n in names}
            original_frames={Path(x['path']).name:x for x in src['extraction']['frames']}
            if any(original_frames.get(n)!=v for n,v in r['public_rgb'].items()):
                raise ValueError('registered RGB differs from original extraction')
            # Existing exporter has overwrite semantics, so own an exclusive root.
            root=dest/'scene';root.mkdir()
            from agents.recon.make_scene_dir import write_scene_dir
            write_scene_dir(rec,frames,workspace,root)
            r['scene_artifacts']=tree(root)
        else:
            prepared=validate_prepared(c,workspace,dest,config_path,code)
            pilot=Path(stage)/'gaussian'/c['pilot_workspace']
            if phase=='train':
                required=pilot/'train-smoke_receipt.json' if workspace==c['pilot_workspace'] else pilot/'train_receipt.json'
                gate=json.loads(required.read_text())
                if gate['status']!='PASS' or gate['code']!=code or gate['config']!=r['config']:
                    raise ValueError('native smoke/full-recipe pilot integrity must pass first')
                verify_tree(required.parent/gate['phase'],gate['artifacts'])
            verify_training_runtime(c,workspace)
            node=os.environ.get('SLURMD_NODENAME','')
            if not (node.lower()=='hala' or node.startswith(('gcp','sof1'))):
                raise ValueError('unauthorized GPU host')
            visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
            if not visible or ',' in visible:
                raise ValueError('exactly one allocated GPU required')
            r['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True)
            out=dest/phase;out.mkdir()
            sd=dest/'scene/data'/workspace
            command=[c['training_python'],'-m','agents.recon.gsplat_train','--scene-dir',str(sd),
                '--init-ply',str(sd/'init_points.ply'),'--out',str(out/'scene.ply'),'--iters',str(c['smoke_iters'] if phase=='train-smoke' else c['iters']),
                '--holdout-every','10','--seed','0']
            r['command']=command
            with (out/'trainer.log').open('x') as log:
                subprocess.run(command,cwd=CODE,env=training_environment(c,workspace),stdout=log,stderr=subprocess.STDOUT,check=True)
            verify_training_runtime(c,workspace)  # Reauthenticate runtime before publishing success.
            r['training_report']=json.loads((out/'train_report.json').read_text())
            validate_training_report(r['training_report'],prepared,c['smoke_iters'] if phase=='train-smoke' else c['iters'])
            r['artifacts']=tree(out)
            validate_prepared(c,workspace,dest,config_path,code)
        if context(config_path,stage)[1]!=code:
            raise ValueError('source changed during phase')
        r['status']='PASS'
    except BaseException as exc:
        r.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        r['wall_s']=time.monotonic()-start
        write_new_json(dest/(phase+'_receipt.json'),r)
    return r


def main():
    if sys.argv[1:]==['--runtime-identity']:
        print(json.dumps(runtime_identity(),sort_keys=True));return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--stage-root',required=True)
    p.add_argument('--workspace',required=True);p.add_argument('--phase',choices=['prepare','train-smoke','train'],required=True)
    a=p.parse_args();print(json.dumps(run(a.config,a.stage_root,a.workspace,a.phase)))

if __name__=='__main__':main()
