"""New canonical E1 drop measurements on authenticated historical A4 exports.

Only the first two prospectively fixed scenes are admitted. This adapter does
not run construction, inspect evaluation GT, or replace the historical drop
protocol. Runtime remains component-scoped and independent F1 is NOT_RUN.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from robo.eval import construction_metrics as metrics
from robo.eval import e4_candidate_screen as screen
from run.icra2027 import e4_compact_materialization as shared
from run.icra2027.e4_planning_terminal import _exact_code

CODE = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = '553db0575241a6858921b6d648d24c29392161fd'
SOURCE_FREEZE = '20260906-0e7579a-v1'
SOURCE_CONTRACT = 'b963de90d9e277a9aa8c48ff2776240a5d3c675ee72411abc371f8d19022aea7'
PILOT = ['09c1414f1b', '0d2ee665be']
SCOPE = 'e1_current_automatic_splat_canonical_drop_pilot'

# This executes in the original, immutable export producer checkout. Its
# canonical validators authenticate the selected body roster and collision
# assets; their prior settle values are never used as E1 measurements.
SOURCE_SCRIPT = '''import contextlib,json,sys
from pathlib import Path
from robo.eval import e4_candidate_screen as s,e3_factory_materializer as m
r=json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
    s._code_snapshot(r['code_commit']);root=s.evidence_root()
    fs={p:Path(r['stage'])/'automatic_candidates'/r['scene_id']/'materialized'/p for p in s.POLICIES}
    reports={p:m.validate_materialized_factory(fs[p],expected_scene_id=r['scene_id'],expected_policy_id=p,repository_root=root) for p in s.POLICIES}
    report=reports['A4'];roster=report['roster']
    export=s.validate_full_room_export(fs['A4'],scene_id=r['scene_id'],policy='A4',root=root,
      expected_object_slots=roster['accepted_slots'],expected_discovered_slots=roster['object_slots'],automatic_factories=fs)
print(json.dumps({'factory':str(fs['A4']),'materialization':report,'export':export}))
'''


def _checked(identity):
    if shared.identity(identity['path']) != identity:
        raise ValueError('E1 referenced artifact changed')
    return Path(identity['path'])


def _write(path, value):
    shared.e3._atomic_write_json(path, value)


def source_contract(config):
    src = config['source']
    source_e0 = _checked(src['e0'])
    if source_e0 != screen.evidence_root()/'outputs/icra2027'/SOURCE_FREEZE/'contract/freeze_manifest.json':
        raise ValueError('E1 original source E0 path differs')
    contract = shared.read(source_e0)
    if (shared.contract_digest(contract) != SOURCE_CONTRACT
            or contract['freeze_id'] != SOURCE_FREEZE
            or contract['code']['commit'] != SOURCE_COMMIT
            or contract['code']['dirty'] is not False
            or shared.resource(contract, 'e4_full_qualification_config') != src['config']):
        raise ValueError('E1 original export source contract differs')
    code = _exact_code(contract['code']['repository'], SOURCE_COMMIT)
    original = shared.read(_checked(src['config']))
    original_python = shared.resource(contract, 'qualification_python')
    if str(Path(original_python['path']).resolve()) != original['python']:
        raise ValueError('E1 original validator Python differs')
    if original['freeze_id'] != SOURCE_FREEZE:
        raise ValueError('E1 original export freeze differs')
    protocol = shared.read(_checked(original['source']['protocol']))
    scenes = [row['scene_id'] for row in protocol['source_populations']]
    if len(scenes) != 50 or scenes != sorted(set(scenes)) or scenes[:2] != PILOT or config['planned_scenes'] != scenes:
        raise ValueError('E1 original full scene denominator differs')
    return code, original



def measurement_runtime():
    import numpy
    import numpy.linalg.linalg
    import numpy.core._multiarray_umath
    import pybullet
    paths = {'e1_python':Path(sys.executable).resolve(), 'e1_pybullet':Path(pybullet.__file__),
        'e1_drop_implementation':CODE/'agents/eval/factory_report.py',
        'e1_link_pose_implementation':CODE/'robo/sim/s7_sim.py',
        'e1_numpy_entrypoint':Path(numpy.__file__),
        'e1_numpy_norm_implementation':Path(numpy.linalg.linalg.__file__),
        'e1_numpy_core_binary':Path(numpy.core._multiarray_umath.__file__)}
    return {name:shared.identity(path) for name,path in paths.items()}


def prepare_config(*, freeze_id, out):
    root=screen.evidence_root()
    _exact_code(CODE, subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip())
    stage=root/'outputs/icra2027'/screen._validated_screen_id(freeze_id)
    if not stage.is_dir() or stage.name==SOURCE_FREEZE:
        raise ValueError('E1 requires a separately reserved stage freeze')
    source_e0=root/'outputs/icra2027'/SOURCE_FREEZE/'contract/freeze_manifest.json'
    original_contract=shared.read(source_e0)
    source_config=shared.resource(original_contract,'e4_full_qualification_config')
    original=shared.read(source_config['path'])
    protocol=shared.read(original['source']['protocol']['path'])
    config=dict(schema_version=1,scope=SCOPE,freeze_id=freeze_id,paper_ready=False,
        measurement_scope=metrics.MEASUREMENT_SCOPE,regime=metrics.REQUIRED_REGIMES[3],
        planned_scenes=[r['scene_id'] for r in protocol['source_populations']],pilot_scenes=PILOT,
        source=dict(e0=shared.identity(source_e0),config=source_config),runtime=measurement_runtime())
    source_contract(config)
    _write(out,config)
    return config


def validate_stage(config_path, stage, expected_commit):
    code = _exact_code(CODE, expected_commit)
    config = shared.read(config_path)
    root = screen.evidence_root();declared_stage=Path(stage);stage=declared_stage.resolve(strict=True)
    if declared_stage != stage:raise ValueError('E1 stage path must be canonical and absolute')
    if (config.get('scope') != SCOPE or config.get('schema_version') != 1
            or config.get('pilot_scenes') != PILOT or config.get('paper_ready') is not False
            or config.get('measurement_scope') != metrics.MEASUREMENT_SCOPE
            or config.get('regime') != metrics.REQUIRED_REGIMES[3]
            or stage != root/'outputs/icra2027'/config['freeze_id']):
        raise ValueError('E1 pilot scope, source or path differs')
    contract = shared.read(stage/'contract/freeze_manifest.json');shared.contract_digest(contract)
    if (contract['freeze_id'] != config['freeze_id'] or contract['code']['commit'] != expected_commit
            or contract['code']['dirty'] is not False
            or Path(contract['code']['repository']) != CODE
            or shared.resource(contract, 'e1_drop_config') != shared.identity(config_path)):
        raise ValueError('E1 pilot E0/config differs')
    for name, identity in config['runtime'].items():
        if shared.resource(contract, name) != identity:
            raise ValueError('E1 drop runtime resource differs')
    actual = measurement_runtime()
    if actual != config['runtime']:
        raise ValueError('E1 actual measurement runtime differs')
    source_contract(config)
    return config, code


def authenticate_export(config, scene_id, log_dir):
    if scene_id not in PILOT:
        raise ValueError('E1 scene is not the fixed pilot')
    code, original = source_contract(config)
    stage = _checked(config['source']['e0']).parent.parent
    environment = dict(os.environ, SIMANY_EVIDENCE_ROOT=str(screen.evidence_root()),
                       SIMANY_SCENE=scene_id, SIMANY_AUTO='1', SIMANY_NO_GT='1', SIMANY_MESH_SRC='derived')
    environment.pop('PYTHONPATH', None)
    request = dict(code_commit=SOURCE_COMMIT, stage=str(stage), scene_id=scene_id)
    completed = subprocess.run([original['python'], '-c', SOURCE_SCRIPT], cwd=code['code_root'],
        input=json.dumps(request), text=True, capture_output=True, env=environment)
    (Path(log_dir)/'original_validator.stdout').write_text(completed.stdout)
    (Path(log_dir)/'original_validator.stderr').write_text(completed.stderr)
    if completed.returncode:
        raise ValueError(f'original export validator failed: exit {completed.returncode}')
    return json.loads(completed.stdout)


def export_bodies(authenticated):
    factory = Path(authenticated['factory'])
    manifest_path = factory/'sim_export/isaac_manifest.json'
    manifest_id = shared.identity(manifest_path)
    expected_manifest = authenticated['export']['artifacts']['isaac_manifest']
    if any(manifest_id[k] != expected_manifest[k] for k in ('sha256','size_bytes')):
        raise ValueError('E1 Isaac manifest changed after authentication')
    materialization_path = factory/'materialization_manifest.json'
    if shared.identity(materialization_path)['sha256'] != authenticated['materialization']['manifest_sha256']:
        raise ValueError('E1 materialization manifest changed after authentication')
    output_members = shared.read(materialization_path)['output_members']
    manifest = shared.read(manifest_path)
    roster = authenticated['materialization']['roster']
    rows = manifest['objects']
    if sorted(row['name'] for row in rows) != sorted(roster['accepted_slots']):
        raise ValueError('E1 exported URDF roster differs')
    bodies = []
    for row in sorted(rows, key=lambda value: value['name']):
        urdf = Path(row['urdf'])
        urdf.relative_to(factory)
        members = [shared.identity(urdf)]
        for mesh in ET.parse(urdf).getroot().findall('.//mesh'):
            name = mesh.get('filename')
            if not name:
                raise ValueError('E1 URDF mesh path is missing')
            path = Path(name)
            if not path.is_absolute():path = urdf.parent/path
            path = Path(os.path.abspath(path));path.relative_to(factory)
            members.append(shared.identity(path))
        for member in members:
            relative = str(Path(member['path']).relative_to(factory))
            if output_members.get(relative) != {k:member[k] for k in ('sha256','size_bytes')}:
                raise ValueError('E1 URDF/mesh lacks authenticated materialization membership')
        bodies.append(dict(object_slot=row['name'], urdf=str(urdf),
                           members=list({x['path']:x for x in members}.values())))
    return bodies


def measure_bodies(bodies, physics, drop):
    """Invoke the existing measurement once for every authenticated body."""
    slots = [row['object_slot'] for row in bodies]
    if len(slots) != len(set(slots)):
        raise ValueError('E1 duplicate drop body')
    for body in bodies:
        for member in body['members']:_checked(member)
    rows = []
    connection = physics.connect(physics.DIRECT)
    if connection < 0:raise RuntimeError('PyBullet DIRECT connection failed')
    try:
        for body in bodies:
            start = time.monotonic()
            try:
                result = drop(physics, Path(body['urdf']))
            except Exception as exc:
                # A failed body stays attempted, with no invented drift/stability.
                rows.append(dict(object_slot=body['object_slot'], status='ERROR', measurement=None,
                                 error=dict(type=type(exc).__name__, message=str(exc)),
                                 runtime_seconds=time.monotonic()-start))
            else:
                rows.append(dict(object_slot=body['object_slot'], status='COMPLETE', measurement=result,
                                 error=None, runtime_seconds=time.monotonic()-start))
    finally:
        physics.disconnect(connection)
    for body in bodies:
        for member in body['members']:_checked(member)
    return rows


def publish_authenticated_drop(config, code, authenticated, output, scene_id):
    """Shared post-admission publisher; caller owns source/export authentication."""
    expected_commit=code['commit']
    bodies = export_bodies(authenticated)
    import pybullet
    from agents.eval.factory_report import drop_test
    start = time.monotonic()
    rows = measure_bodies(bodies, pybullet, drop_test)
    elapsed = time.monotonic()-start
    _exact_code(CODE, expected_commit)
    roster = authenticated['materialization']['roster']
    complete = all(row['status']=='COMPLETE' for row in rows)
    runtime = dict.fromkeys(metrics.RUNTIME_COMPONENTS)
    runtime['construction_drop_verification'] = elapsed
    record = dict(freeze_id=config['freeze_id'],regime=config['regime'],scene_id=scene_id,
        scene_status='success' if bodies else 'empty',input_instances=roster['job_count'],
        controller_accepted_instances=roster['accepted_count'],accepted_instances=len(bodies),
        f1_20=None,f1_weight=0,stable_instances=sum(row['measurement']['stable'] for row in rows) if complete else None,
        tested_instances=len(rows) if complete else None,runtime_minutes=None,build_commit=expected_commit,
        build_manifest_path=str(output/'build_manifest.json'),failure_reason='' if bodies else 'no accepted export bodies',
        source_artifact_hash=shared.canonical_hash(authenticated),record_valid=False,
        validity_reasons=['independent_geometry_evaluation_not_run','full_runtime_not_measured']+([] if complete else ['drop_measurement_error']),
        geometry_reference_status='NOT_RUN',measurement_scope=config['measurement_scope'],runtime_components_seconds=runtime)
    report=dict(schema_version=1,scope=config['scope'],status='PASS' if complete else 'PARTIAL',paper_ready=False,
        producer_code=code,source=config['source'],scene_id=scene_id,planned_scenes=config['planned_scenes'],
        pilot_scenes=PILOT,measurement_scope=config['measurement_scope'],authenticated_export=authenticated,
        bodies=bodies,measurements=rows,attempted_bodies=len(rows),completed_bodies=sum(r['status']=='COMPLETE' for r in rows),
        host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),record=record)
    return publish_drop_report(output, report)


def publish_drop_report(output, report):
    record=report['record'];expected_commit=report['producer_code']['commit']
    _write(output/'drop_report.json',report)
    _write(output/'build_manifest.json',{**{k:record[k] for k in ('freeze_id','regime','scene_id','build_commit','source_artifact_hash','record_valid','validity_reasons')},'record':record})
    # Existing producer remains authoritative; incomplete components are watermarked.
    aggregated=metrics.aggregate([record],allow_preliminary=True,smoke=True)
    _write(output/'construction_record_aggregate.json',aggregated)
    _write(output/'seal.json',dict(schema_version=1,source_commit=expected_commit,
        members={name:shared.identity(output/name) for name in ('drop_report.json','build_manifest.json','construction_record_aggregate.json')}))
    return report


def execute(config_path, stage, expected_commit, scene_id):
    from run.icra2027.e4_full_qualification import cpu_guard
    cpu_guard()
    config, code = validate_stage(config_path, stage, expected_commit)
    if scene_id not in PILOT:raise ValueError('E1 scene is not the fixed pilot')
    output = Path(stage)/'construction_drop'/scene_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        authenticated = authenticate_export(config, scene_id, output)
        return publish_authenticated_drop(config, code, authenticated, output, scene_id)
    except Exception as exc:
        _write(output/'failure.json',dict(status='FAIL',scene_id=scene_id,paper_ready=False,
            error_type=type(exc).__name__,error=str(exc),measurement_scope=config['measurement_scope'],
            source=config['source'],producer_code=code))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--stage',required=True)
    parser.add_argument('--expected-commit',required=True);parser.add_argument('--scene',required=True)
    args=parser.parse_args();result=execute(args.config,args.stage,args.expected_commit,args.scene)
    print(json.dumps({k:result[k] for k in ('status','scene_id','attempted_bodies','completed_bodies')}))


if __name__=='__main__':main()
