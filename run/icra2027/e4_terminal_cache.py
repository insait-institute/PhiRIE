"""Resumable per-scene original-source terminal validation; no new measurements."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket

from run.icra2027 import e4_planning_terminal as terminal

MODE='sealed_scene_cache_v1'
SCOPE='full_discovery_terminal_scene_validation'
PHASES=('first','independent')
api=terminal.api
screen=terminal.screen


def _checked(ref):
    path=Path(ref['path'])
    if not path.is_absolute() or path.resolve(strict=True)!=path:raise ValueError('cached identity aliases another path')
    if api.identity(ref['path'])!=ref:raise ValueError('cached source evidence changed')
    return Path(ref['path'])


def _context(config,code):
    stage=Path(config['scene_cache_root']).parent
    if (config.get('replay_mode')!=MODE or Path(config['scene_cache_root'])!=stage/'scene_validation'
            or stage.name!=config['freeze_id']):raise ValueError('scene cache mode/path differs')
    e0=api.read(stage/'contract/freeze_manifest.json');api.contract_digest(e0)
    config_path=api.resource(e0,'e4_full_terminal_config')['path']
    actual,_=terminal._validate_full_stage(config_path,stage,code['commit'],stage/'full_qualification',code)
    if actual!=config:raise ValueError('scene cache config differs from stage E0')
    source=config['source']
    for key in ('config','e0','runtime','environment','osmesa_environment'):_checked(source[key])
    if terminal.full_source_contract(source['config']['path'],Path(source['e0']['path']).parent.parent,
                                     source['code_commit'])!=source:raise ValueError('scene cache original source differs')
    _checked(config['compatibility'])
    if config['cpu_feature_mode']!=terminal.ISA_MODE:raise ValueError('scene cache ISA differs')
    return stage,api.identity(config_path),api.identity(stage/'contract/freeze_manifest.json')


def _metadata(result,config,sid):
    source=api.read(_checked(config['source']['config']))
    protocol=api.read(_checked(source['source']['protocol']))
    if (result['source_config']!=source or result['protocol']!=protocol
            or set(result['scenes'])!={sid} or sid not in config['jobs']
            or any(row['scene_id']!=sid for row in result['cells'])):
        raise ValueError('cached scene/source/protocol population differs')
    scene=result['scenes'][sid]
    if scene['job']!=config['jobs'][sid]:raise ValueError('cached scene scheduler attempt differs')
    for path,identity in scene['evidence'].items():
        if path!=identity['path']:raise ValueError('cached evidence path differs')
        _checked(identity)
    for attempt in config.get('prior_attempts',{}).get(sid,[]):
        for identity in attempt['logs'].values():_checked(identity)
    return result


def _pass(config,code,sid,phase,context):
    stage,config_id,e0_id=context
    path=stage/'scene_validation'/sid/phase
    bundle=screen._validate_bundle(path,root=screen.evidence_root(),expected_kind=SCOPE)
    manifest=bundle['manifest']
    expected=dict(code=code,freeze_id=stage.name,scene_id=sid,phase=phase,config=config_id,E0=e0_id)
    if (any(manifest.get(k)!=v for k,v in expected.items())
            or set(manifest['files'])!={'result.json','process.json'}):
        raise ValueError('cached replay source/stage/scene binding differs')
    result=api.read(path/'result.json');process=api.read(path/'process.json')
    request=dict(config,_scene_ids=[sid]);request_hash=hashlib.sha256(json.dumps(request).encode()).hexdigest()
    worker=Path(terminal.__file__).with_name('e4_full_terminal_worker.py')
    if (process['returncode']!=0 or json.loads(process['stdout'])!=result
            or process['request_sha256']!=request_hash or process['cwd']!=config['source']['code_root']
            or process['worker']!=api.identity(worker)
            or process['command']!=terminal._full_worker_command(config,worker)
            or not math.isfinite(process['runtime_seconds']) or process['runtime_seconds']<0
            or not str(process['job_id']).isdigit() or process['hostname'] not in config['cache_hosts']):
        raise ValueError('cached canonical process receipt differs')
    _host(config,process['hostname'])
    return _metadata(result,config,sid),dict(manifest=api.identity(path/'manifest.json'),seal=api.identity(path/'seal.json'))


def _host(config,hostname):
    if hostname not in config['cache_hosts']:raise ValueError('unproven cache execution host')
    proof=api.read(_checked(config['cache_hosts'][hostname]))
    if (proof.get('source_commit')!=config['source']['code_commit']
            or proof.get('freeze_id')!=config['source']['freeze_id']
            or proof.get('disabled_features')!=terminal.ISA_MODE
            or proof.get('acceptance_tolerances_changed') is not False
            or proof.get('allowed_host',proof.get('hardware_host'))!=hostname):
        raise ValueError('cache host compatibility receipt differs')
    for key in ('whole_pilot_replay','process','original_envelope'):
        if key in proof:_checked(proof[key])


def produce_scene(*,config_path,stage_root,expected_code_commit,scene_id):
    code=screen._code_snapshot(expected_code_commit);config=api.read(config_path)
    context=_context(config,code);stage,config_id,e0_id=context
    if Path(stage_root)!=stage or scene_id not in config['jobs']:raise ValueError('cache stage/scene differs')
    hostname=socket.gethostname();_host(config,hostname)
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdigit():raise ValueError('scene replay requires an ordinary scheduler job')
    if os.environ.get('NPY_DISABLE_CPU_FEATURES')!=terminal.ISA_MODE:raise ValueError('scene replay ISA is not fixed')
    first=None
    for phase in PHASES:
        path=stage/'scene_validation'/scene_id/phase
        if path.exists():
            result,_=_pass(config,code,scene_id,phase,context)
        else:
            process={'job_id':job,'hostname':hostname}
            attempt=stage/'scene_validation_attempts'/scene_id/(job+'-'+phase+'.json')
            attempt.parent.mkdir(parents=True,exist_ok=True)
            if attempt.exists():raise FileExistsError('scene replay attempt already exists')
            try:
                result=terminal._full_replay(config,scene_ids=[scene_id],process_receipt=process)
                _metadata(result,config,scene_id)
                if first is not None and result!=first:raise ValueError('independent scene replay differs')
                if screen._code_snapshot(expected_code_commit)!=code:raise ValueError('scene replay code changed')
            except Exception as exc:
                process['failure']=type(exc).__name__+': '+str(exc)
                with attempt.open('x') as f:json.dump(process,f,indent=2,allow_nan=False)
                raise
            with attempt.open('x') as f:json.dump(process,f,indent=2,allow_nan=False)
            screen._publish_bundle(path,manifest_kind=SCOPE,payloads={'result.json':screen._json_bytes(result),
                'process.json':screen._json_bytes(process)},manifest_fields=dict(code=code,freeze_id=stage.name,
                scene_id=scene_id,phase=phase,config=config_id,E0=e0_id))
            result,_=_pass(config,code,scene_id,phase,context)
        if first is not None and result!=first:raise ValueError('independent scene replay differs')
        first=result
    return {'status':'PASS','scene_id':scene_id,'freeze_id':stage.name,'policy_executed':0}


def compose(config,code):
    context=_context(config,code);stage=context[0]
    root=stage/'scene_validation'
    if not root.is_dir() or {p.name for p in root.iterdir()}!=set(config['jobs']):
        raise ValueError('scene validation cache must contain exactly full scene roster')
    result=None;proofs={}
    for sid in sorted(config['jobs']):
        if {p.name for p in (root/sid).iterdir()}!=set(PHASES):raise ValueError('scene validation pass roster differs')
        first,a=_pass(config,code,sid,'first',context)
        second,b=_pass(config,code,sid,'independent',context)
        if first!=second:raise ValueError('independent scene replay differs')
        if result is None:result=dict(source_config=first['source_config'],protocol=first['protocol'],scenes={},cells=[])
        if first['source_config']!=result['source_config'] or first['protocol']!=result['protocol']:
            raise ValueError('scene cache shared source differs')
        result['scenes'].update(first['scenes']);result['cells'].extend(first['cells'])
        proofs[sid]={'first':a,'independent':b}
    result['cells'].sort(key=lambda r:r['cell_id']);result['validation_cache']=proofs
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','freeze-root','expected-code-commit','scene-id'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(produce_scene(config_path=a.config,stage_root=a.freeze_root,
        expected_code_commit=a.expected_code_commit,scene_id=a.scene_id),sort_keys=True))

if __name__=='__main__':main()
