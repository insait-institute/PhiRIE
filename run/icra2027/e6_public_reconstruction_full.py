"""Full RGB cohort admission/publication around the unchanged 7c measurement producer.

This wrapper is not a new reconstruction producer. Fresh subprocesses execute the
same pinned modules/arguments/environment as the authenticated smoke and pilot.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from run.icra2027 import e6_public_reconstruction as shared

CODE = Path(__file__).resolve().parents[2]
PRODUCER_SHA = '7c76b2c39ceb515237d8eac490fbf76813092569'
CONFIG_KEYS = {'schema_version', 'scope', 'freeze_id', 'source_commit', 'measurement_producer',
               'reference_configs', 'roster', 'runtime', 'recipe_sha256', 'reuse'}
OLD_KEYS = {'schema_version', 'tier', 'scope', 'freeze_id', 'source_commit', 'max_frames', 'seed',
            'gs_iters', 'execution_scene', 'fallback_allowed', 'roster', 'runtime'}
REUSED = [('behavior_task0023', 'clean'), *[('behavior_task0020', c) for c in shared.CONDITIONS]]


def producer(ref):
    if set(ref) != {'path', 'commit'} or ref['commit'] != PRODUCER_SHA:
        raise ValueError('measurement producer must be explicitly pinned 7c')
    path = Path(ref['path']).resolve(strict=True)
    shared.sealed.sealed_cpu._inside(path, root=shared.ROOT.parent, label='producer checkout')
    snap = shared.git_snapshot(path)
    if snap['dirty'] or snap['commit'] != PRODUCER_SHA:
        raise ValueError('measurement producer source changed')
    spec = importlib.util.spec_from_file_location('e6_pinned_measurement', path/'run/icra2027/e6_public_reconstruction.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe(old, runtime, api):
    """Exhaustive legacy config partition; unknown fields never silently disappear."""
    allowed = OLD_KEYS | ({'selection_rationale'} if old.get('tier') == 'smoke' else set())
    if set(old) != allowed or old['schema_version'] != 1 or old['source_commit'] != PRODUCER_SHA:
        raise ValueError('unknown or changed measurement config fields')
    if (old['tier'] not in {'smoke', 'pilot'} or old['scope'] != 'e6_rgb_only_reconstruction_'+old['tier']
        or old['max_frames'] != 48 or old['seed'] != 0 or old['gs_iters'] != 15000
        or old['fallback_allowed'] is not False):
        raise ValueError('measurement recipe changed')
    if old['tier'] == 'smoke' and old.get('selection_rationale') != 'smallest_positive_clean_frame_count_then_frozen_scene_order':
        raise ValueError('smoke admission rationale changed')
    # Destination placeholders affect paths only; scene identity is separately bound per unit.
    plan = api.commands(old, runtime, Path('/MEASUREMENT_DEST'), Path('/MEASUREMENT_FRAMES'))
    return {'schema_version': 1, 'measurement_source_commit': PRODUCER_SHA,
            'runtime': old['runtime'], 'commands': plan, 'selection_rule': api.RULE,
            'max_frames': 48, 'seed': 0, 'gs_iters': 15000, 'fallback_allowed': False,
            'object_ids': None, 'controller': None, 'camera': 'RGB-predicted Omega and DA3 metric frame',
            'metric': 'pinned producer construction validation; no held-out metric',
            'treatment': 'public RGB condition only',
            'excluded_config_fields': {
                'schema_version': 'validated exactly before partition',
                'tier': 'engineering admission label; all measurement arguments identical',
                'scope': 'engineering admission label; all measurement arguments identical',
                'freeze_id': 'publication identity only; original receipts remain authoritative',
                'execution_scene': 'bound separately in exact per-unit input roster',
                'roster': 'complete roster content and per-unit frame identities compared separately',
                'selection_rationale': 'smoke admission only, validated exact; no measurement effect'}}


def replay(api, config_path, condition=None):
    runtime = shared.read(shared.read(config_path)['runtime']['path'])
    args = [runtime['python'], '-m', 'run.icra2027.e6_public_reconstruction', '--config', str(config_path),
            '--stage-root', str(Path(config_path).parent)]
    args += ['--validate'] if condition is None else ['--validate-output', '--condition', condition]
    result = subprocess.run(args, cwd=api.CODE, env=api.environment(runtime, Path(config_path).parent, 'validation'),
                            text=True, capture_output=True)
    if result.returncode:
        raise ValueError('original 7c validation failed: '+result.stdout[-2000:]+result.stderr[-2000:])
    return {'command': args, 'cwd': str(api.CODE), 'returncode': result.returncode,
            'stdout_sha256': shared.hashlib.sha256(result.stdout.encode()).hexdigest()}


def validate(config_path, stage, *, require_e0=True):
    config_path = Path(config_path).resolve(strict=True); stage = Path(stage).resolve(strict=True)
    cfg = shared.read(config_path)
    if set(cfg) != CONFIG_KEYS or cfg['schema_version'] != 1 or cfg['scope'] != 'e6_public_rgb_full18':
        raise ValueError('full cohort config schema differs')
    snap = shared.git_snapshot(CODE)
    if snap['dirty'] or snap['commit'] != cfg['source_commit']:
        raise ValueError('wrapper requires exact clean source')
    if (stage.parent != shared.ROOT/'outputs/icra2027' or stage.name != cfg['freeze_id']
        or not shared.CANONICAL_FREEZE_ID.fullmatch(stage.name) or config_path.parent != stage):
        raise ValueError('full stage path/freeze differs')
    api = producer(cfg['measurement_producer'])
    for ref in [cfg['roster'], cfg['runtime'], *cfg['reference_configs']]:
        if shared.identity(ref['path']) != ref:
            raise ValueError('input identity changed')
    roster = api.validate_roster(shared.read(cfg['roster']['path']))
    runtime = shared.read(cfg['runtime']['path'])
    refs = cfg['reference_configs']
    if len(refs) != 2: raise ValueError('requires original smoke and pilot configs')
    old_configs = [shared.read(r['path']) for r in refs]
    if [c['tier'] for c in old_configs] != ['smoke', 'pilot']:
        raise ValueError('reference config order differs')
    if [c['execution_scene'] for c in old_configs] != [api.SCENES[2], api.SCENES[0]]:
        raise ValueError('reference scene differs')
    for ref, old in zip(refs, old_configs):
        if old['runtime'] != cfg['runtime'] or shared.read(old['roster']['path']) != roster:
            raise ValueError('runtime or complete input roster changed')
        if shared.canonical_hash(recipe(old, runtime, api)) != cfg['recipe_sha256']:
            raise ValueError('measurement-affecting recipe differs')
        replay(api, ref['path'])
    if [(r['scene_id'], r['condition_id']) for r in cfg['reuse']] != REUSED:
        raise ValueError('reuse set differs from predeclared four units')
    for row in cfg['reuse']:
        if set(row) != {'scene_id', 'condition_id', 'config', 'terminal_manifest', 'terminal_seal'}:
            raise ValueError('reuse declaration schema differs')
        ref = refs[0 if row['scene_id'] == api.SCENES[2] else 1]
        if row['config'] != ref: raise ValueError('reuse config binding differs')
        unit = Path(ref['path']).parent/'audit/public_reconstruction'/(row['scene_id']+'_'+row['condition_id'])
        for field, filename in [('terminal_manifest', 'manifest.json'), ('terminal_seal', 'seal.json')]:
            if row[field] != shared.identity(unit/'terminal'/filename):
                raise ValueError('reuse source seal changed')
    if require_e0:
        contract = shared.read(stage/'contract/freeze_manifest.json')
        payload = {k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}}
        if (shared.canonical_hash(payload) != contract['contract_sha256'] or contract['freeze_id'] != stage.name
            or contract['code']['commit'] != cfg['source_commit'] or contract['code']['dirty']):
            raise ValueError('full E0 source/digest differs')
        bindings = {'e6_full_config': shared.identity(config_path), 'e6_rgb_roster': cfg['roster'],
                    'e6_runtime': cfg['runtime'], 'e6_original_smoke': refs[0], 'e6_original_pilot': refs[1]}
        for name, ref in bindings.items():
            hits = [r for r in contract['resource_inventory'] if r['id'] == name]
            if len(hits) != 1 or hits[0]['sha256'] != ref['sha256'] or hits[0]['resolved_path'] != ref['path']:
                raise ValueError('full E0 input binding differs: '+name)
        for ref in runtime['checkpoints']:
            hits = [r for r in contract['resource_inventory'] if r['id'] == ref['id']]
            if len(hits) != 1 or hits[0]['sha256'] != ref['sha256'] or hits[0]['resolved_path'] != ref['path']:
                raise ValueError('full checkpoint E0 binding differs')
    return cfg, runtime, roster, api


def unit_row(roster, scene, condition):
    rows = [r for r in roster['rows'] if r['scene_id'] == scene and r['condition_id'] == condition]
    if len(rows) != 1: raise ValueError('unit outside frozen 18-condition cohort')
    return rows[0]


def execute(config_path, stage, scene, condition):
    cfg, runtime, roster, api = validate(config_path, stage)
    row = unit_row(roster, scene, condition)
    dest = Path(stage)/'audit/public_reconstruction'/(scene+'_'+condition)
    shared.sealed.sealed_cpu._inside(dest, root=shared.ROOT, label='full reconstruction destination')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir()  # no overwrite, including a previous partial attempt
    started = time.monotonic()
    ref = next((r for r in cfg['reuse'] if (r['scene_id'],r['condition_id']) == (scene,condition)), None)
    if ref:
        receipt = replay(api, ref['config']['path'], condition)
        origin = Path(ref['config']['path']).parent/'audit/public_reconstruction'/dest.name
        gate = shared.read(origin/'terminal/gate.json')
        if gate['stage_status'] != 'PASS': raise ValueError('reuse requires validated completed original')
        report = {'mode': 'reused_original', 'origin': str(origin), 'original_validation': receipt,
                  'original_terminal_manifest': ref['terminal_manifest'], 'original_terminal_seal': ref['terminal_seal'],
                  'stage_status': 'PASS', 'stage_records': gate['stage_records'], 'measurement_wall_s': gate['wall_s']}
    else:
        frames = dest/'frames'; frames.mkdir()
        for f in row['frames']:
            if shared.identity(f['path']) != {k:f[k] for k in ('path','sha256','size_bytes')}:
                raise ValueError('public RGB changed')
            if f['selected']:
                shutil.copyfile(f['path'], frames/f['name'])
                if shared.file_identity(frames/f['name']) != {k:f[k] for k in ('sha256','size_bytes')}:
                    raise ValueError('RGB copy differs')
        records = []; failed = None
        for name, python, args in api.commands({}, runtime, dest, frames):
            t = time.monotonic()
            with (dest/(name+'.log')).open('x') as log:
                result = subprocess.run([python,*args], cwd=api.CODE, env=api.environment(runtime,dest,name),
                                        stdout=log, stderr=subprocess.STDOUT)
            record = {'stage':name, 'command':[python,*args], 'returncode':result.returncode, 'wall_s':time.monotonic()-t}
            records.append(record); shared.write_new_json(dest/(name+'_execution.json'),record)
            if result.returncode: failed = name; break
        if not failed: api.validate_products(dest,row)
        report = {'mode':'fresh_pinned_producer', 'stage_status':'FAIL' if failed else 'PASS',
                  'failed_stage':failed, 'failure_type':'producer_nonzero_exit' if failed else None,
                  'stage_records':records, 'measurement_wall_s':time.monotonic()-started}
    validate(config_path, stage)
    shared.write_new_json(dest/'input_manifest.json',row)
    report.update(schema_version=1, scope=cfg['scope'], wrapper_source_commit=cfg['source_commit'],
                  measurement_producer=cfg['measurement_producer'], recipe_sha256=cfg['recipe_sha256'],
                  freeze_id=cfg['freeze_id'], scene_id=scene, condition_id=condition, planned_queries=4,
                  feature_rows_written=0, paper_ready=False, model_fallback_used=False,
                  legacy_depth_pose_splat_read=False, hostname=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    members = {str(p.relative_to(dest)):shared.file_identity(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    shared.sealed._publish_bundle(dest/'terminal',manifest_kind='e6_public_rgb_full_unit',
        payloads={'gate.json':(json.dumps(report,indent=2)+'\n').encode()},
        manifest_fields={'wrapper_source_commit':cfg['source_commit'],'measurement_producer':cfg['measurement_producer'],
                         'output_members':members,'paper_ready':False})
    return report


def validate_output(config_path, stage, scene, condition):
    cfg,runtime,roster,api = validate(config_path,stage)
    row = unit_row(roster,scene,condition); dest = Path(stage)/'audit/public_reconstruction'/(scene+'_'+condition)
    bundle = shared.sealed._validate_bundle(dest/'terminal',root=shared.ROOT,expected_kind='e6_public_rgb_full_unit')
    actual = {str(p.relative_to(dest)):shared.file_identity(p) for p in sorted(dest.rglob('*'))
              if p.is_file() and 'terminal' not in p.relative_to(dest).parts}
    if actual != bundle['manifest']['output_members'] or shared.read(dest/'input_manifest.json') != row:
        raise ValueError('full unit content differs')
    g = shared.read(dest/'terminal/gate.json')
    for key,value in {'wrapper_source_commit':cfg['source_commit'],'measurement_producer':cfg['measurement_producer'],
        'recipe_sha256':cfg['recipe_sha256'],'freeze_id':cfg['freeze_id'],'scene_id':scene,'condition_id':condition,
        'planned_queries':4,'feature_rows_written':0,'paper_ready':False,'model_fallback_used':False,
        'legacy_depth_pose_splat_read':False,'scope':cfg['scope']}.items():
        if g.get(key) != value: raise ValueError('full unit gate identity differs: '+key)
    ref = next((r for r in cfg['reuse'] if (r['scene_id'],r['condition_id']) == (scene,condition)),None)
    if ref:
        replay(api,ref['config']['path'],condition)
        origin = Path(ref['config']['path']).parent/'audit/public_reconstruction'/dest.name
        old = shared.read(origin/'terminal/gate.json')
        if (g['mode'] != 'reused_original' or g['origin'] != str(origin) or old['stage_status'] != 'PASS'
            or g['stage_status'] != 'PASS' or g['stage_records'] != old['stage_records']
            or g['measurement_wall_s'] != old['wall_s'] or g['original_terminal_manifest'] != ref['terminal_manifest']
            or g['original_terminal_seal'] != ref['terminal_seal']):
            raise ValueError('original measurement attribution differs')
    else:
        plan = api.commands({},runtime,dest,dest/'frames'); records = g['stage_records']
        if (g['mode'] != 'fresh_pinned_producer' or not records or len(records)>4
            or any(r['stage']!=n or r['command']!=[p,*a] for r,(n,p,a) in zip(records,plan))):
            raise ValueError('pinned measurement execution differs')
        if g['stage_status'] == 'PASS':
            if len(records)!=4 or any(r['returncode'] for r in records) or g['failed_stage'] is not None or g['failure_type'] is not None:
                raise ValueError('PASS requires complete producer chain')
            api.validate_products(dest,row)
        elif g['stage_status'] == 'FAIL':
            if (records[-1]['returncode']==0 or any(r['returncode'] for r in records[:-1])
                or g['failed_stage'] != records[-1]['stage'] or g['failure_type'] != 'producer_nonzero_exit'):
                raise ValueError('typed producer failure differs')
        else: raise ValueError('unknown terminal status')
    return g


def publish_cohort(config_path, stage, output):
    """Immutable status snapshot; unavailable units remain explicit in 18/72."""
    cfg, runtime, roster, api = validate(config_path, stage)
    output = Path(output).resolve()
    shared.sealed.sealed_cpu._inside(output, root=Path(stage), label='cohort snapshot')
    if output.exists(): raise FileExistsError('cohort snapshots are immutable')
    rows = []
    for row in roster['rows']:
        scene, condition = row['scene_id'], row['condition_id']
        unit = Path(stage)/'audit/public_reconstruction'/(scene+'_'+condition)
        if (unit/'terminal').exists():
            gate = validate_output(config_path,stage,scene,condition)
            status = gate['stage_status']
            terminal = shared.identity(unit/'terminal/manifest.json')
            reason = gate.get('failure_type')
        else:
            status = 'INCOMPLETE' if unit.exists() else 'NOT_RUN'
            terminal = None
            reason = 'unsealed_partial_output' if unit.exists() else 'not_executed'
        rows.append(dict(scene_id=scene,condition_id=condition,status=status,reason=reason,
                         planned_queries=4,terminal_manifest=terminal,source_frame_present=row['source_frame_present']))
    report = dict(schema_version=1,scope='e6_public_rgb_full_cohort',wrapper_source_commit=cfg['source_commit'],
                  measurement_producer=cfg['measurement_producer'],freeze_id=cfg['freeze_id'],
                  planned_conditions=18,planned_query_rows=72,feature_rows_written=0,paper_ready=False,
                  counts={s:sum(r['status']==s for r in rows) for s in ('PASS','FAIL','INCOMPLETE','NOT_RUN')},
                  units=rows)
    shared.sealed._publish_bundle(output,manifest_kind='e6_public_rgb_full_cohort',
        payloads={'gate.json':(json.dumps(report,indent=2)+'\n').encode()},
        manifest_fields={'wrapper_source_commit':cfg['source_commit'],'paper_ready':False})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--stage-root',required=True)
    p.add_argument('--cohort-output'); p.add_argument('--scene'); p.add_argument('--condition'); p.add_argument('--validate',action='store_true')
    p.add_argument('--validate-output',action='store_true'); a=p.parse_args()
    if a.cohort_output:
        print(json.dumps(publish_cohort(a.config,a.stage_root,a.cohort_output),indent=2)); return
    if a.validate:
        validate(a.config,a.stage_root); print('E6_FULL_PREFLIGHT=PASS'); return
    fn=validate_output if a.validate_output else execute
    result=fn(a.config,a.stage_root,a.scene,a.condition); print(json.dumps(result,indent=2))
    if result['stage_status']=='FAIL': raise SystemExit(3)

if __name__=='__main__': main()
