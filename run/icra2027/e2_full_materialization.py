"""Materialize only E2 factories omitted by E4; reuse the canonical producer."""
from pathlib import Path
import argparse
import json
import os
import time
import yaml
from agents.edit.inpaint_masks import _identity, _checked, _bound_contract, _sealed
from robo.eval import agentic_ablation as e3
from robo.eval import e3_factory_materializer as materializer
from run.icra2027 import e2_full_factorized_protocol as protocol
from run.icra2027.e2_raw_room import write_new

SCOPE = 'full_e2_missing_factory_materialization'


def context(config_path, contract_path):
    cp, ep = Path(config_path).absolute(), Path(contract_path).absolute()
    c, contract = _bound_contract(cp, ep)
    p = protocol.validate_protocol(_checked(c['full_protocol']))
    source = protocol.read(p['factory_execution'])
    tasks = protocol.read(source['source']['protocol'])
    missing = [s for s in p['scene_ids'] if not any(q['scene_id'] == s for q in tasks['qualification_tasks'])]
    if (c['schema_version'] != 1 or c['scope'] != SCOPE or c['paper_ready'] is not False
            or c['scene_ids'] != missing or len(missing) != 9 or c['pilot_scene'] != missing[0]
            or c['policy_id'] != 'A4' or c['planned_full_scenes'] != 50
            or c['planned_full_objects'] != 1871 or c['planned_full_views'] != 400
            or c['e3_root'] != source['source']['e3_root']
            or c['descriptors'] != {s:source['source']['scenes'][s]['descriptor'] for s in missing}):
        raise ValueError('missing-factory protocol/source/roster differs')
    if Path(c['python']).resolve() != Path(__import__('sys').executable).resolve():
        raise ValueError('materialization interpreter differs')
    out = e3.REPOSITORY_ROOT/'outputs/icra2027'/c['freeze_id']/'fidelity/materialization'
    e3._validate_cli_execution(ep,c['freeze_id'],out,config_paths=[cp])
    return c, contract, p, out


def checked_result(c, contract, out, scene):
    directory, seal = _sealed(out/'handoff'/scene)
    r = json.loads((directory/'result.json').read_text())
    if (r['scope'] != SCOPE or r['status'] != 'PASS' or r['scene_id'] != scene
            or r['freeze_id'] != c['freeze_id'] or r['code_commit'] != contract['code']['commit']
            or yaml.safe_load(_checked(r['config']).read_text()) != c
            or json.loads(_checked(r['contract']).read_text()) != contract
            or r['paper_ready'] is not False or r['evaluation_geometry_read'] is not False):
        raise ValueError('materialization handoff binding differs')
    factory = out/'factories'/scene/'A4'
    if _checked(r['manifest']) != factory/'materialization_manifest.json':
        raise ValueError('materialization handoff factory differs')
    checked = materializer.validate_materialized_factory(factory,expected_scene_id=scene,expected_policy_id='A4')
    if checked != r['validation']:
        raise ValueError('materialization changed after handoff')
    return r, seal


def run(config_path, contract_path, scene):
    if os.environ.get('SLURM_ARRAY_JOB_ID') or os.environ.get('SLURM_JOB_GPUS'):
        raise ValueError('ordinary CPU materialization only')
    c, contract, p, out = context(config_path,contract_path)
    if scene not in c['scene_ids']:raise ValueError('scene outside missing factory cohort')
    if scene != c['pilot_scene']:checked_result(c,contract,out,c['pilot_scene'])
    target = out/'factories'/scene/'A4'; handoff=out/'handoff'/scene
    claim=out/'claims'/f'{scene}.json'
    if any(x.exists() or x.is_symlink() for x in (target,handoff,claim)):
        raise FileExistsError('materialization attempt already exists')
    start=time.monotonic()
    write_new(claim,dict(config=_identity(Path(config_path).absolute()),contract=_identity(Path(contract_path).absolute())))
    descriptor=out/'descriptors'/f'{scene}.json'
    write_new(descriptor,c['descriptors'][scene])
    try:
        materializer.materialize_factory_variant(e3_root=c['e3_root'],scene_id=scene,policy_id='A4',
            out=target,automatic_scene_contract=descriptor)
        validation=materializer.validate_materialized_factory(target,expected_scene_id=scene,expected_policy_id='A4')
        if context(config_path,contract_path)[:3] != (c,contract,p):
            raise ValueError('source changed during materialization')
        manifest=json.loads((target/'materialization_manifest.json').read_text())
        expected=[j['object_slot'] for s in protocol.read(p['resolved_jobs'])['scenes'] if s['scene_id']==scene for j in s['jobs']]
        if manifest['roster']['object_slots']!=expected or manifest['roster']['job_count']!=p['population'][scene]:
            raise ValueError('materialized original object roster differs')
        result=dict(schema_version=1,scope=SCOPE,status='PASS',freeze_id=c['freeze_id'],scene_id=scene,
            code_commit=contract['code']['commit'],config=_identity(Path(config_path).absolute()),
            contract=_identity(Path(contract_path).absolute()),manifest=_identity(target/'materialization_manifest.json'),
            validation=validation,planned_objects=manifest['roster']['job_count'],
            accepted_objects=manifest['roster']['accepted_count'],wall_s=time.monotonic()-start,
            evaluation_geometry_read=False,paper_ready=False)
        write_new(handoff/'result.json',result)
        write_new(handoff/'seal.json',dict(schema_version=1,members={'result.json':_identity(handoff/'result.json')['sha256']}))
        return result
    except Exception as exc:
        write_new(out/'failures'/f'{scene}.json',dict(status='FAILED',scene_id=scene,
            planned_objects=p['population'][scene],error_type=type(exc).__name__,error=str(exc),
            wall_s=time.monotonic()-start,paper_ready=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--contract',required=True)
    parser.add_argument('--scene');parser.add_argument('--context-only',action='store_true')
    a=parser.parse_args()
    if a.context_only:
        c,_,_,_=context(a.config,a.contract);print(json.dumps(dict(status='PASS',scenes=c['scene_ids'],pilot_scene=c['pilot_scene'])))
    else:print(json.dumps(run(a.config,a.contract,a.scene),sort_keys=True))

if __name__=='__main__':main()
