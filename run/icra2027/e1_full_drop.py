"""Full-roster E1 admission around existing materializers, exports and drop metrics.

No model, evaluation, or policy execution is introduced. Every scene receives
an export/drop or typed terminal record; room rejection is never converted into
an isolated-body export success.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from run.icra2027 import e1_current_drop as base
from run.icra2027 import e4_compact_materialization as api
from robo.eval import e3_factory_materializer as materializer
from robo.eval import e4_candidate_screen as screen

CODE=Path(__file__).resolve().parents[2]
SCOPE='e1_full_automatic_splat_verified_room_export_and_canonical_drop'
PILOT_SOURCE='6b3e57c9c3efa29cc5ce9b3d019bfb8381938473'
PILOT_FREEZE='20260906-860f3fc-v1'
PILOT_CONTRACT='3f3ed55900d70581ae89f837fa776567aabf3068ce1aadea0fe8c7a1c6bca06f'
E2_SOURCE='87ef5e2ab6e039394c184a35533291171b7a8a67'
E2_FREEZE='20260906-02e9d33-v2'
REJECTION='room collision re-enters the protected object carve'
EXPORT_FILES=('scene.xml','isaac_manifest.json','mujoco_settle.json','room_collision_report.json')
ROOM_SCOPE='A4_bodies_only_after_complete_paired_static_full_room_export_validation'
EXPORT_FIXED_ENVIRONMENT = dict(
    NPY_DISABLE_CPU_FEATURES='AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL',
    LP_NUM_THREADS='4',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
    MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa')


def export_environment():
    environment={key:os.environ.get(key) for key in screen.EXPORT_SUBPROCESS_ENVIRONMENT_FIELDS}
    if any(environment[key]!=value for key,value in EXPORT_FIXED_ENVIRONMENT.items()):
        raise ValueError('full E1 requires fixed compatible exporter numerical environment')
    if not environment['LD_LIBRARY_PATH']:
        raise ValueError('full E1 requires explicit OSMesa library search path')
    return environment



def _source_population(config):
    code, original=base.source_contract(config)
    populations={}
    for sid in config['planned_scenes']:
        source=original['source']['scenes'][sid]
        path=base._checked(source['identities']['ledger'])
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        selected=[r for r in rows if r['policy_id']=='A4']
        slots=[r['object_slot'] for r in selected]
        if (len(selected)!=source['planned_objects'] or len(set(slots))!=len(slots)
                or any(r['scene_id']!=sid for r in selected)):
            raise ValueError('full E1 original object denominator differs')
        populations[sid]=dict(input_instances=len(selected),controller_accepted_instances=sum(r['terminal_action']=='accept' for r in selected),
            accepted_slots=sorted(r['object_slot'] for r in selected if r['terminal_action']=='accept'),
            ledger=source['identities']['ledger'],descriptor=source['descriptor'])
    if len(populations)!=50 or sum(r['input_instances'] for r in populations.values())!=1871:
        raise ValueError('full E1 must retain all50/1871')
    return populations,original,code


def _pilot_admission(config):
    stage=screen.evidence_root()/'outputs/icra2027'/PILOT_FREEZE
    e0=api.read(base._checked(config['pilot_admission']['e0']))
    if (Path(config['pilot_admission']['e0']['path'])!=stage/'contract/freeze_manifest.json'
            or api.contract_digest(e0)!=PILOT_CONTRACT or e0['code']['commit']!=PILOT_SOURCE or e0['code']['dirty'] is not False):
        raise ValueError('full E1 original pilot E0 differs')
    base._exact_code(e0['code']['repository'],PILOT_SOURCE)
    qa=api.read(base._checked(config['pilot_admission']['qa']))
    if qa.get('status')!='PASS' or qa.get('producer_gate')!='PILOT_PASSED' or qa.get('pilot_scenes')!=base.PILOT:
        raise ValueError('full E1 lacks fixed two-scene pilot integrity')
    for unit in qa['units']:
        for name in ('seal','source_report','premeasurement','postseal_QA'):base._checked(unit[name])
        seal=api.read(unit['seal']['path'])
        for identity in seal['members'].values():base._checked(identity)
        report=api.read(unit['source_report']['path'])
        if report['producer_code']['commit']!=PILOT_SOURCE or report['status']!='PASS':
            raise ValueError('full E1 pilot measurement integrity differs')
    if [u['scene_id'] for u in qa['units']]!=base.PILOT:
        raise ValueError('full E1 pilot scenes omitted or reordered')


def _anchors(factory):
    return {name:api.identity(factory/name) for name in ('materialization_manifest.json','seal.json',
        *(f'sim_export/{name}' for name in EXPORT_FILES))}


def plan_units(config):
    populations,original,_=_source_population(config)
    root=screen.evidence_root();old=root/'outputs/icra2027'/base.SOURCE_FREEZE
    units={}
    for sid,population in populations.items():
        factory=old/'automatic_candidates'/sid/'materialized/A4'
        rejection=old/'dispatch'/f'{sid}_export_rejection.json'
        if not population['controller_accepted_instances']:
            mode='no_accepted_bodies';anchors={}
        elif rejection.is_file():
            mode='original_paired_room_rejection';anchors={'rejection':api.identity(rejection)}
        elif all((factory/'sim_export'/name).is_file() for name in EXPORT_FILES):
            mode='original_export';anchors=_anchors(factory)
        else:
            mode='fresh_export';anchors={}
        units[sid]=dict(population=population,mode=mode,anchors=anchors,
            original_factory=str(factory),e2_materialization=None)
        if mode=='fresh_export':
            e2=root/'outputs/icra2027'/E2_FREEZE/'fidelity/materialization/factories'/sid/'A4'
            if e2.is_dir():
                units[sid]['e2_materialization']={'manifest':api.identity(e2/'materialization_manifest.json'),
                    'seal':api.identity(e2/'seal.json')}
    return units


def prepare_config(*,freeze_id,out):
    base._exact_code(CODE,subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip())
    root=screen.evidence_root();stage=root/'outputs/icra2027'/screen._validated_screen_id(freeze_id)
    if not stage.is_dir() or freeze_id in {base.SOURCE_FREEZE,PILOT_FREEZE,E2_FREEZE}:
        raise ValueError('full E1 requires a fresh reserved stage')
    old=root/'outputs/icra2027'/base.SOURCE_FREEZE
    e0=api.read(old/'contract/freeze_manifest.json');source_config=api.resource(e0,'e4_full_qualification_config')
    original=api.read(source_config['path'])
    protocol=api.read(original['source']['protocol']['path'])
    pilot=root/'outputs/icra2027'/PILOT_FREEZE
    config=dict(schema_version=1,scope=SCOPE,freeze_id=freeze_id,paper_ready=False,planned_objects=1871,
        measurement_scope=base.metrics.MEASUREMENT_SCOPE,room_export_requirement=ROOM_SCOPE,regime=base.metrics.REQUIRED_REGIMES[3],
        planned_scenes=[r['scene_id'] for r in protocol['source_populations']],pilot_scenes=base.PILOT,
        pilot_admission={'e0':api.identity(pilot/'contract/freeze_manifest.json'),'qa':api.identity(pilot/'dispatch/independent_pilot_qa.json')},
        source=dict(e0=api.identity(old/'contract/freeze_manifest.json'),config=source_config),runtime=base.measurement_runtime(),export_environment=export_environment(),
        prior_pilot_drop_reuse=False,prior_pilot_remeasurement_reason='new source/full-cohort admission; unchanged cheap canonical drop implementation')
    config['units']=plan_units(config);_pilot_admission(config)
    base._write(out,config);return config


def validate_stage(config_path,stage,expected_commit):
    code=base._exact_code(CODE,expected_commit);config=api.read(config_path)
    root=screen.evidence_root();stage=Path(stage)
    if (stage.resolve(strict=True)!=stage or stage!=root/'outputs/icra2027'/config['freeze_id']
            or config.get('scope')!=SCOPE or config.get('schema_version')!=1 or config.get('paper_ready') is not False
            or config.get('room_export_requirement')!=ROOM_SCOPE or config.get('planned_objects')!=1871
            or config.get('measurement_scope')!=base.metrics.MEASUREMENT_SCOPE or config.get('regime')!=base.metrics.REQUIRED_REGIMES[3]
            or config.get('pilot_scenes')!=base.PILOT or config.get('prior_pilot_drop_reuse') is not False):
        raise ValueError('full E1 scope/source/path differs')
    e0=api.read(stage/'contract/freeze_manifest.json');api.contract_digest(e0)
    if (e0['freeze_id']!=config['freeze_id'] or e0['code']['commit']!=expected_commit or e0['code']['dirty'] is not False
            or Path(e0['code']['repository'])!=CODE or api.resource(e0,'e1_full_drop_config')!=api.identity(config_path)):
        raise ValueError('full E1 E0/config differs')
    if config.get('export_environment')!=export_environment():raise ValueError('full E1 actual exporter environment differs from frozen config')
    if config['runtime']!=base.measurement_runtime():raise ValueError('full E1 measurement runtime changed')
    for name,identity in config['runtime'].items():
        if api.resource(e0,name)!=identity:raise ValueError('full E1 runtime E0 binding differs')
    _pilot_admission(config)
    populations,original,original_code=_source_population(config)
    _validate_units(config,populations,Path(config['source']['e0']['path']).parent.parent)
    return config,code,original,original_code



def _validate_units(config,populations,original_stage):
    if list(config['units'])!=config['planned_scenes'] or any(config['units'][sid]['population']!=p for sid,p in populations.items()):
        raise ValueError('full E1 unit roster or original population differs')
    for sid,unit in config['units'].items():
        expected_factory=original_stage/'automatic_candidates'/sid/'materialized/A4'
        if unit['original_factory']!=str(expected_factory):raise ValueError('full E1 original factory path differs')
        expected_keys={'original_export':{'materialization_manifest.json','seal.json',*(f'sim_export/{n}' for n in EXPORT_FILES)},
            'original_paired_room_rejection':{'rejection'},'fresh_export':set(),'no_accepted_bodies':set()} if unit['mode']=='original_export' else {
            'original_paired_room_rejection':{'rejection'},'fresh_export':set(),'no_accepted_bodies':set()}
        if unit['mode'] not in expected_keys or set(unit['anchors'])!=expected_keys[unit['mode']]:
            raise ValueError('full E1 original anchor roster differs')
        rejection_path=original_stage/'dispatch'/f'{sid}_export_rejection.json'
        if unit['population']['controller_accepted_instances'] and rejection_path.is_file() and unit['mode']!='original_paired_room_rejection':
            raise ValueError('full E1 cannot rerun or hide original paired-room rejection')
        if unit['mode']=='fresh_export' and all((expected_factory/'sim_export'/name).is_file() for name in EXPORT_FILES):
            raise ValueError('full E1 must authenticate existing export instead of regenerating it')
        if unit['mode']=='original_export' and unit['anchors']!=_anchors(expected_factory):
            raise ValueError('full E1 original export identities changed')
        if unit['mode']=='original_paired_room_rejection' and unit['anchors']['rejection']['path']!=str(original_stage/'dispatch'/f'{sid}_export_rejection.json'):
            raise ValueError('full E1 original rejection path differs')
        if unit['mode'] not in {'no_accepted_bodies','original_paired_room_rejection','original_export','fresh_export'}:
            raise ValueError('full E1 unknown export mode')
        if (unit['mode']=='no_accepted_bodies')!=(unit['population']['controller_accepted_instances']==0):
            raise ValueError('full E1 no-body closure hides accepted objects')
        for identity in unit['anchors'].values():base._checked(identity)
        if unit['e2_materialization']:
            comparison=unit['e2_materialization']
            expected=screen.evidence_root()/'outputs/icra2027'/E2_FREEZE/'fidelity/materialization/factories'/sid/'A4'
            if set(comparison)!={'manifest','seal'} or comparison['manifest']['path']!=str(expected/'materialization_manifest.json') or comparison['seal']['path']!=str(expected/'seal.json'):
                raise ValueError('full E1 E2 comparison path/roster differs')
            for identity in comparison.values():base._checked(identity)
            manifest=api.read(comparison['manifest']['path']);seal=api.read(comparison['seal']['path'])
            if (manifest['scene_id']!=sid or manifest['policy_id']!='A4' or manifest['provenance']['materializer_commit']!=E2_SOURCE
                    or manifest['provenance']['materializer_dirty'] is not False
                    or seal['members']['materialization_manifest.json']!={k:comparison['manifest'][k] for k in ('sha256','size_bytes')}):
                raise ValueError('full E1 E2 comparison source/seal differs')

def _original_export(config,unit,scene,output,original,original_code):
    request=dict(code_commit=base.SOURCE_COMMIT,stage=str(Path(config['source']['e0']['path']).parent.parent),scene_id=scene)
    env=dict(os.environ,SIMANY_EVIDENCE_ROOT=str(screen.evidence_root()),SIMANY_SCENE=scene,SIMANY_AUTO='1',SIMANY_NO_GT='1',SIMANY_MESH_SRC='derived')
    env.pop('PYTHONPATH',None)
    result=subprocess.run([original['python'],'-c',base.SOURCE_SCRIPT],cwd=original_code['code_root'],input=json.dumps(request),text=True,capture_output=True,env=env)
    (output/'original_validator.stdout').write_text(result.stdout);(output/'original_validator.stderr').write_text(result.stderr)
    if result.returncode:raise ValueError(f'original full-room validator failed: exit {result.returncode}')
    for identity in unit['anchors'].values():base._checked(identity)
    authenticated=json.loads(result.stdout)
    if authenticated['factory']!=unit['original_factory']:raise ValueError('original factory path changed')
    return authenticated



def _replay_original_rejection(config,unit,scene,output,original,original_code):
    script='''import contextlib,json,sys
from pathlib import Path
from robo.eval import e4_candidate_screen as s
r=json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
 s._code_snapshot(r['code_commit']);root=s.evidence_root()
 fs={p:Path(r['stage'])/'automatic_candidates'/r['scene_id']/'materialized'/p for p in s.POLICIES}
 context=s._automatic_export_context(fs,scene_id=r['scene_id'],root=root)
 try:s.validate_full_room_export(fs['A0'],scene_id=r['scene_id'],policy='A0',root=root,expected_object_slots=context['rosters']['A0']['accepted_slots'],expected_discovered_slots=context['object_slots'],automatic_factories=fs)
 except s.CandidateScreenError as exc:
  if str(exc)!=r['reason']:raise
 else:raise ValueError('claimed original room rejection does not replay')
print(json.dumps({'status':'PASS','observed_rejection':r['reason'],'source_commit':r['code_commit'],'scene_id':r['scene_id']}))
'''
    request=dict(code_commit=base.SOURCE_COMMIT,stage=str(Path(config['source']['e0']['path']).parent.parent),scene_id=scene,reason=REJECTION)
    env=dict(os.environ,SIMANY_EVIDENCE_ROOT=str(screen.evidence_root()),SIMANY_SCENE=scene,SIMANY_AUTO='1',SIMANY_NO_GT='1',SIMANY_MESH_SRC='derived')
    env.pop('PYTHONPATH',None)
    result=subprocess.run([original['python'],'-c',script],cwd=original_code['code_root'],input=json.dumps(request),text=True,capture_output=True,env=env)
    (output/'original_rejection.stdout').write_text(result.stdout);(output/'original_rejection.stderr').write_text(result.stderr)
    if result.returncode:raise ValueError('original room rejection failed exact-source replay')
    expected=dict(status='PASS',observed_rejection=REJECTION,source_commit=base.SOURCE_COMMIT,scene_id=scene)
    if json.loads(result.stdout)!=expected:raise ValueError('original rejection replay identity differs')
    return dict(result=expected,stdout=api.identity(output/'original_rejection.stdout'),stderr=api.identity(output/'original_rejection.stderr'))


def _fresh_export(config,unit,scene,stage,code,original):
    descriptor=stage/'descriptors'/f'{scene}.json'
    descriptor.parent.mkdir(parents=True,exist_ok=True);base._write(descriptor,unit['population']['descriptor'])
    factories={p:stage/'automatic_candidates'/scene/'materialized'/p for p in screen.POLICIES}
    for policy in screen.POLICIES:
        materializer.materialize_factory_variant(e3_root=original['source']['e3_root'],scene_id=scene,policy_id=policy,
            out=factories[policy],automatic_scene_contract=descriptor)
    # Materialization copies existing selected artifacts; it never generates an
    # object. Compare E2's immutable output-member set when available, retaining
    # each producer's own source manifest rather than relabeling the old one.
    if unit['e2_materialization']:
        old=api.read(base._checked(unit['e2_materialization']['manifest']))
        current=api.read(factories['A4']/'materialization_manifest.json')
        if (old['provenance']['materializer_commit']!=E2_SOURCE or old['provenance']['materializer_dirty'] is not False
                or old['scene_id']!=scene or old['policy_id']!='A4' or old['output_members']!=current['output_members']
                or old['e3_code_commit']!=current['e3_code_commit'] or old['e3_root']!=current['e3_root']):
            raise ValueError('E2 reusable asset bytes differ from canonical E1 materialization')
    execution=screen._run_full_room_export(factories['A4'],scene_id=scene,root=screen.evidence_root(),
        common_carve_factories=[factories[p] for p in screen.POLICIES],automatic=True,
        subprocess_environment=config['export_environment'])
    report=materializer.validate_materialized_factory(factories['A4'],expected_scene_id=scene,
        expected_policy_id='A4',repository_root=screen.evidence_root())
    roster=report['roster']
    export=screen.validate_full_room_export(factories['A4'],scene_id=scene,policy='A4',root=screen.evidence_root(),
        expected_object_slots=roster['accepted_slots'],expected_discovered_slots=roster['object_slots'],automatic_factories=factories)
    return dict(factory=str(factories['A4']),materialization=report,export=export,export_execution=execution)


def _terminal(config,code,unit,scene,output,kind,reason,verification=None):
    base._exact_code(CODE,code['commit'])
    population=unit['population'];zero=kind=='no_accepted_bodies';rejected=kind in {'original_paired_room_rejection','measured_room_export_rejection'}
    record=dict(freeze_id=config['freeze_id'],regime=config['regime'],scene_id=scene,
        scene_status='empty' if zero else 'failed',input_instances=population['input_instances'],
        controller_accepted_instances=population['controller_accepted_instances'],accepted_instances=0 if zero or rejected else None,
        f1_20=None,f1_weight=0,stable_instances=0 if zero or rejected else None,tested_instances=0 if zero or rejected else None,
        runtime_minutes=None,build_commit=code['commit'],build_manifest_path=str(output/'build_manifest.json'),failure_reason=reason,
        source_artifact_hash=api.canonical_hash(unit),record_valid=False,validity_reasons=[kind,'independent_geometry_evaluation_not_run','full_runtime_not_measured'],
        geometry_reference_status='NOT_RUN',measurement_scope=config['measurement_scope'],runtime_components_seconds=dict.fromkeys(base.metrics.RUNTIME_COMPONENTS))
    report=dict(schema_version=1,scope=SCOPE,status='EMPTY' if zero else 'NOT_RUN',paper_ready=False,
        producer_code=code,source=config['source'],scene_id=scene,planned_scenes=config['planned_scenes'],pilot_scenes=base.PILOT,
        measurement_scope=config['measurement_scope'],room_export_requirement=ROOM_SCOPE,
        full_room_export_status='NOT_APPLICABLE_NO_ACCEPTED_BODIES' if zero else 'BLOCKED_BY_OBSERVED_PAIRED_STATIC_REJECTION' if kind=='original_paired_room_rejection' else 'REJECTED_FULL_ROOM_EXPORT' if rejected else 'UNAVAILABLE',
        terminal_kind=kind,terminal_reason=reason,verification=verification,source_unit=unit,authenticated_export=None,bodies=[],measurements=[],attempted_bodies=0,completed_bodies=0,
        host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),record=record)
    return base.publish_drop_report(output,report)


def execute(config_path,stage,expected_commit,scene):
    from run.icra2027.e4_full_qualification import cpu_guard
    cpu_guard();config,code,original,original_code=validate_stage(config_path,stage,expected_commit)
    if scene not in config['units']:raise ValueError('full E1 scene outside original roster')
    stage=Path(stage);unit=config['units'][scene];output=stage/'construction_drop'/scene
    output.mkdir(parents=True,exist_ok=False)
    if unit['mode']=='no_accepted_bodies':return _terminal(config,code,unit,scene,output,unit['mode'],'original A4 accepted no bodies')
    if unit['mode']=='original_paired_room_rejection':
        receipt=api.read(base._checked(unit['anchors']['rejection']))
        if (receipt.get('source_commit')!=base.SOURCE_COMMIT or receipt.get('scene_id')!=scene
                or receipt.get('classification')!='construction_artifact_collision_integrity_failure' or receipt.get('exact_error')!=REJECTION):
            raise ValueError('original paired-room rejection receipt differs')
        for path,identity in receipt['source_artifacts'].items():base._checked(dict(path=path,**identity))
        verification=_replay_original_rejection(config,unit,scene,output,original,original_code)
        return _terminal(config,code,unit,scene,output,unit['mode'],REJECTION,verification)
    try:
        authenticated=(_original_export(config,unit,scene,output,original,original_code) if unit['mode']=='original_export'
            else _fresh_export(config,unit,scene,stage,code,original))
        roster=authenticated['materialization']['roster']
        if (roster['job_count']!=unit['population']['input_instances'] or roster['accepted_slots']!=unit['population']['accepted_slots']):
            raise ValueError('full E1 authenticated export lost planned objects')
        authenticated['full_room_admission']=dict(status='PASS',requirement=ROOM_SCOPE,source_mode=unit['mode'],source_unit=unit)
        return base.publish_authenticated_drop(config,code,authenticated,output,scene)
    except (ValueError,OSError,RuntimeError,subprocess.CalledProcessError,screen.CandidateScreenError) as exc:
        base._write(output/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),source_unit=unit,source_commit=expected_commit))
        if (output/'drop_report.json').exists():raise
        kind='measured_room_export_rejection' if str(exc)==REJECTION else 'export_or_measurement_unavailable'
        return _terminal(config,code,unit,scene,output,kind,str(exc))



def aggregate_full(config_path,stage,expected_commit):
    """Generate the full declared ladder, retaining four unexecuted input rows."""
    config,code,_,_=validate_stage(config_path,stage,expected_commit)
    stage=Path(stage);records=[];source_reports=[]
    for scene in config['planned_scenes']:
        directory=stage/'construction_drop'/scene
        seal=api.read(directory/'seal.json')
        if seal['source_commit']!=expected_commit:raise ValueError('full E1 unit source differs')
        members=seal.get('members',{})
        if set(members)!={'drop_report.json','build_manifest.json','construction_record_aggregate.json'}:
            raise ValueError('full E1 unit seal member roster differs')
        for name,identity in members.items():
            path=directory/name
            if identity.get('path')!=str(path) or path.is_symlink() or path.resolve(strict=True)!=path:
                raise ValueError('full E1 unit seal member path differs')
            base._checked(identity)
        report=api.read(directory/'drop_report.json')
        if (report['scope']!=SCOPE or report['scene_id']!=scene or report['producer_code']!=code
                or report['planned_scenes']!=config['planned_scenes'] or report['record']['input_instances']!=config['units'][scene]['population']['input_instances']):
            raise ValueError('full E1 terminal unit lost source or population')
        record=report['record'];population=config['units'][scene]['population']
        if (record['controller_accepted_instances']!=population['controller_accepted_instances']
                or record['geometry_reference_status']!='NOT_RUN' or record['f1_20'] is not None or record['f1_weight']!=0
                or record['runtime_minutes'] is not None):
            raise ValueError('full E1 promoted unmeasured geometry/runtime or changed controller coverage')
        if report['authenticated_export'] is not None:
            if report['authenticated_export'].get('full_room_admission',{}).get('requirement')!=ROOM_SCOPE:
                raise ValueError('full E1 cannot promote isolated body export')
            if config['units'][scene]['mode']=='fresh_export':
                receipt=report['authenticated_export'].get('export_execution') or {}
                if (receipt.get('requested_environment')!=config['export_environment'] or not receipt.get('calls')
                        or any(call.get('returncode')!=0 or any(call.get('environment',{}).get(k)!=v
                            for k,v in config['export_environment'].items()) for call in receipt['calls'])):
                    raise ValueError('full E1 fresh exporter subprocess receipt differs')
            if base.export_bodies(report['authenticated_export'])!=report['bodies']:
                raise ValueError('full E1 exported body identities changed')
            for body in report['bodies']:
                for identity in body['members']:base._checked(identity)
            if ([b['object_slot'] for b in report['bodies']]!=[r['object_slot'] for r in report['measurements']]
                    or record['accepted_instances']!=len(report['bodies']) or report['attempted_bodies']!=len(report['bodies'])
                    or report['completed_bodies']!=sum(r['status']=='COMPLETE' for r in report['measurements'])):
                raise ValueError('full E1 dropped or duplicated physical attempts')
            complete=all(r['status']=='COMPLETE' for r in report['measurements'])
            expected_stable=sum(r['measurement']['stable'] for r in report['measurements']) if complete else None
            if record['stable_instances']!=expected_stable or record['tested_instances']!=(len(report['measurements']) if complete else None):
                raise ValueError('full E1 conditional stability differs from measured body records')
        else:
            unit=config['units'][scene];kind=report.get('terminal_kind');record=report['record']
            if report.get('source_unit')!=unit or report.get('room_export_requirement')!=ROOM_SCOPE:
                raise ValueError('full E1 terminal source or room scope differs')
            if kind=='no_accepted_bodies':
                if unit['mode']!='no_accepted_bodies' or record['accepted_instances']!=0 or record['tested_instances']!=0 or record['scene_status']!='empty':
                    raise ValueError('full E1 false empty-scene closure')
            elif kind=='original_paired_room_rejection':
                verification=report.get('verification') or {}
                if unit['mode']!=kind or record['accepted_instances']!=0 or record['tested_instances']!=0 or report['terminal_reason']!=REJECTION:
                    raise ValueError('full E1 original room rejection was promoted')
                if verification.get('result')!=dict(status='PASS',observed_rejection=REJECTION,source_commit=base.SOURCE_COMMIT,scene_id=scene):
                    raise ValueError('full E1 room rejection lacks original-source replay')
                for key in ('stdout','stderr'):base._checked(verification[key])
            elif kind=='measured_room_export_rejection':
                if record['accepted_instances']!=0 or record['tested_instances']!=0 or report['terminal_reason']!=REJECTION:
                    raise ValueError('full E1 measured room rejection was promoted')
                failure=api.read(directory/'failure.json')
                if failure.get('error')!=REJECTION or failure.get('source_unit')!=unit:raise ValueError('full E1 measured rejection lacks source failure')
            elif kind=='export_or_measurement_unavailable':
                if record['accepted_instances'] is not None or record['tested_instances'] is not None:
                    raise ValueError('full E1 unavailable measurement has fabricated counts')
            else:raise ValueError('full E1 unknown terminal kind')
            if report['measurements'] or report['bodies'] or report['attempted_bodies']!=0:
                raise ValueError('full E1 terminal fabricated a physical measurement')
        if base.metrics.aggregate([report['record']],allow_preliminary=True,smoke=True)!=api.read(directory/'construction_record_aggregate.json'):
            raise ValueError('full E1 canonical unit aggregate differs')
        records.append(report['record']);source_reports.append(api.identity(directory/'seal.json'))
    if sum(r['input_instances'] for r in records)!=1871:raise ValueError('full E1 denominator must remain1871')
    destination=stage/'construction';destination.mkdir(exist_ok=False)
    for regime_index,regime in enumerate(base.metrics.REQUIRED_REGIMES):
        if regime==config['regime']:continue
        for scene in config['planned_scenes']:
            manifest=destination/'unexecuted_inputs'/str(regime_index)/f'{scene}.json'
            record=dict(freeze_id=config['freeze_id'],regime=regime,scene_id=scene,scene_status='unavailable',input_instances=None,
                controller_accepted_instances=None,accepted_instances=None,f1_20=None,f1_weight=0,stable_instances=None,tested_instances=None,
                runtime_minutes=None,build_commit=expected_commit,build_manifest_path=str(manifest),failure_reason='input_family_not_run',
                source_artifact_hash=api.canonical_hash(dict(config=api.identity(config_path),regime=regime,scene_id=scene,status='NOT_RUN')),
                record_valid=False,validity_reasons=['input_family_not_run'],geometry_reference_status='NOT_RUN',
                measurement_scope=config['measurement_scope'],runtime_components_seconds=dict.fromkeys(base.metrics.RUNTIME_COMPONENTS))
            base._write(manifest,{**{k:record[k] for k in ('freeze_id','regime','scene_id','build_commit','source_artifact_hash','record_valid','validity_reasons')},'record':record})
            records.append(record)
    records=sorted(records,key=lambda r:(base.metrics.REQUIRED_REGIMES.index(r['regime']),r['scene_id']))
    if len(records)!=250:raise ValueError('full E1 ladder must preserve250 planned cells')
    base._write(destination/'scene_records.json',records)
    base.shared.e3._atomic_write_text(destination/'scene_records.csv',base.metrics._csv_text(records))
    result=base.metrics.generate(destination/'scene_records.json',destination/'table',allow_preliminary=True)
    base._write(destination/'seal.json',dict(source_commit=expected_commit,config=api.identity(config_path),planned_scene_regime_cells=250,
        measured_regimes=1,paper_ready=False,source_unit_seals=source_reports,
        members={name:api.identity(destination/name) for name in ('scene_records.json','scene_records.csv','table/construction_table.json','table/construction_table.csv','table/construction_table.tex')}))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--stage',required=True)
    p.add_argument('--expected-commit',required=True)
    actions=p.add_mutually_exclusive_group(required=True);actions.add_argument('--scene');actions.add_argument('--aggregate',action='store_true')
    a=p.parse_args()
    if a.aggregate:
        r=aggregate_full(a.config,a.stage,a.expected_commit);print(json.dumps(dict(paper_ready=r['paper_ready'],rows=len(r['rows']))))
    else:
        r=execute(a.config,a.stage,a.expected_commit,a.scene);print(json.dumps({k:r[k] for k in ('status','scene_id','attempted_bodies','completed_bodies')}))

if __name__=='__main__':main()
