"""Full-cohort E4 source binding and canonical CPU qualification, without policy execution."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

from robo.eval import e4_candidate_screen as screen
from robo.eval import e4_camera_scorer_gate as camera
from robo.eval import e3_factory_materializer as materializer
from run.icra2027 import e4_compact_materialization as shared

SCOPE = 'e4_full_discovery_budget_qualification'
SOURCE_COMMIT = '0a8b4caa6005dd9201b58ee028f11c26ef06c34a'
SOURCE_FREEZE = '20260905-859f51d-v1'
SOURCE_CONTRACT = 'f022d499b60cf84e4453f261d431120ae4ac3f98836692499b6b97e5693872ce'
CONTROL_DRIVER = 'bad63f1c6bf3b4e28db96390eeb51ee4ceb4c2491080ae53eff35dc5f9e4bb4b'
PLANNING_UNQUALIFIED = 'automatic task planning has no prepared size-admissible target; all declared pairs remain unqualified'


def inspect_source(protocol_path, e3_root):
    """Consume the original complete construction audit; never load evaluation."""
    root = screen.evidence_root()
    e3_root = materializer.e3.checked_repo_path(e3_root, 'full E3 construction', kind='dir')
    contract_path = e3_root.parent/'contract/freeze_manifest.json'
    contract = shared.read(contract_path)
    if (e3_root.parent.name != SOURCE_FREEZE or contract['freeze_id'] != SOURCE_FREEZE
            or shared.contract_digest(contract) != SOURCE_CONTRACT
            or contract['code']['commit'] != SOURCE_COMMIT or contract['code']['dirty'] is not False):
        raise ValueError('original full construction E0 differs')
    original = Path(contract['code']['repository'])
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=original,text=True).strip() != SOURCE_COMMIT
            or subprocess.check_output(['git','status','--porcelain'],cwd=original,text=True).strip()):
        raise ValueError('original full construction source is not exact and clean')
    jobs, _, audit, inventory = materializer._verify_inventory(e3_root)
    config_id = shared.resource(contract, 'agentic_fresh_jobs')
    config = shared.read(config_id['path'])
    if (jobs['freeze_id'] != SOURCE_FREEZE or jobs['counts'] != {'scenes':50,'jobs':1871,'policy_object_rows':9355}
            or jobs['source_contract']['code_commit'] != SOURCE_COMMIT
            or jobs['source_contract']['contract_sha256'] != SOURCE_CONTRACT
            or jobs['source_contract']['jobs_sha256'] != shared.canonical_hash(config)):
        raise ValueError('original complete inventory differs')
    protocol = shared.read(protocol_path)
    first = protocol['source_populations'][0]
    population = dict(e3_root=str(e3_root),e3_freeze_id=SOURCE_FREEZE,e3_producer_commit=SOURCE_COMMIT,
        scene_id=first['scene_id'],counts={'planned_objects':first['planned_objects']},
        pairs=[{k:q[k] for k in ('target','receptacle','task_family')} for q in first['queries']])
    screen.qualification_selection(protocol_path,population,root=root)
    dispatch = e3_root.parent/'dispatch'
    controls_path = dispatch/'full50_control_integrity_gate.json'
    observations_path = dispatch/'full50_observation_integrity_gate.json'
    missing = [str(p) for p in (controls_path,observations_path) if not p.is_file()]
    if missing:
        return dict(status='WAITING',dependencies=missing,paper_ready=False)
    controls, observations = shared.read(controls_path), shared.read(observations_path)
    driver_id = shared.identity(dispatch/'audit_full50_controls.py')
    if driver_id['sha256'] != CONTROL_DRIVER or controls.get('audit_driver_sha256') != CONTROL_DRIVER:
        raise ValueError('original complete control audit implementation differs')
    for payload, scope in ((controls,'complete_construction_integrity_only'),
                           (observations,'complete_training_observation_integrity_only')):
        if any(payload.get(k) != v for k,v in dict(status='PASS',scope=scope,freeze_id=SOURCE_FREEZE,
                source_commit=SOURCE_COMMIT,planned_scenes=50,planned_jobs=1871,
                evaluation_geometry_read=False,paper_ready=False).items()):
            raise ValueError('full construction audit scope or population differs')
    if controls.get('contract_sha256') != SOURCE_CONTRACT or controls.get('planned_policy_rows') != 9355:
        raise ValueError('complete control contract/denominator differs')
    scenes = [s['scene_id'] for s in jobs['scenes']]
    if ([s['scene_id'] for s in controls['scenes']] != scenes
            or [s['scene_id'] for s in observations['records']] != scenes
            or [s['scene_id'] for s in config['automatic_sources']] != scenes):
        raise ValueError('complete audit/descriptor scene roster differs')
    sources = {}
    for planned, control, observation, descriptor in zip(jobs['scenes'],controls['scenes'],
            observations['records'],config['automatic_sources'],strict=True):
        sid=planned['scene_id'];count=len(planned['jobs'])
        materializer._automatic_scene_audit(jobs,audit,sid)
        if control['planned_jobs'] != count or control['policy_rows'] != count*5 or observation['planned_jobs'] != count:
            raise ValueError('scene audit population differs')
        paths = {'control_seal':(e3_root/'control'/sid/'seal.json',control['seal_sha256']),
            'controller_shard':(e3_root/'control'/sid/'controller_shard.json',control['shard_sha256']),
            'ledger':(e3_root/'control'/sid/'job_ledger.jsonl',control['ledger_sha256']),
            'observation_seal':(e3_root/'observations'/sid/'seal.json',observation['seal_sha256'])}
        identities={name:shared.identity(path) for name,(path,_) in paths.items()}
        if any(identities[name]['sha256'] != expected for name,(_,expected) in paths.items()):
            raise ValueError('original audited construction member changed')
        sources[sid]=dict(identities=identities,planned_objects=count,
            descriptor={'schema_version':1,**{k:descriptor[k] for k in ('scene_id','discovery_directory','discovery_hashes')}})
    return dict(status='READY',paper_ready=False,source=dict(e3_root=str(e3_root),
        e0=shared.identity(contract_path),jobs_config=config_id,inventory=inventory,
        protocol=shared.identity(protocol_path),control_audit=shared.identity(controls_path),
        observation_audit=shared.identity(observations_path),control_driver=driver_id,scenes=sources))


def prepare_config(*, protocol_path, e3_root, freeze_id, menagerie_root, out):
    reviewed=inspect_source(protocol_path,e3_root)
    if reviewed['status'] != 'READY':return reviewed
    screen._validated_screen_id(freeze_id)
    model=camera._menagerie_snapshot(Path(menagerie_root),camera.EXPECTED_MENAGERIE_COMMIT)
    protocol=shared.read(protocol_path)
    # Pilot is chosen solely from preregistered task IDs, before qualification.
    pilot=next(r['scene_id'] for r in protocol['source_populations'] if
               any(q['scene_id']==r['scene_id'] for q in protocol['qualification_tasks']))
    config=dict(schema_version=1,scope=SCOPE,paper_ready=False,freeze_id=freeze_id,
        source=reviewed['source'],python=str(Path(sys.executable).resolve()),
        menagerie_root=str(Path(menagerie_root).resolve()),menagerie={k:v for k,v in model.items() if k!='files'},
        openpi_resize_identity=camera._openpi_snapshot(),pilot_scene=pilot,
        planned_scenes=50,planned_objects=1871,planned_semantic_queries=269,
        planned_qualification_cells=2690,input_semantic_queries=6155,budget_exclusions=5886)
    with Path(out).open('x') as f:json.dump(config,f,indent=2,sort_keys=True);f.write('\n')
    return config


def validate_stage(config_path, stage_root, expected_commit):
    code=screen._code_snapshot(expected_commit);config=shared.read(config_path)
    stage_root=Path(stage_root).resolve(strict=True)
    contract=shared.read(stage_root/'contract/freeze_manifest.json');shared.contract_digest(contract)
    if (config.get('scope')!=SCOPE or config.get('schema_version')!=1 or config.get('paper_ready') is not False
            or stage_root != screen._experiment_root(screen.evidence_root(),config['freeze_id'])
            or contract['freeze_id']!=config['freeze_id'] or contract['code']['commit']!=code['commit']
            or contract['code']['dirty'] is not False
            or shared.resource(contract,'e4_full_qualification_config')!=shared.identity(config_path)):
        raise ValueError('full qualification stage config/source/E0 differs')
    runtime=shared.resource(contract,'qualification_python')
    if Path(runtime['path']).resolve()!=Path(sys.executable).resolve() or config['python']!=str(Path(sys.executable).resolve()):
        raise ValueError('qualification Python differs')
    reviewed=inspect_source(config['source']['protocol']['path'],config['source']['e3_root'])
    if reviewed.get('status')!='READY' or reviewed['source']!=config['source']:
        raise ValueError('sealed full construction closure differs')
    model=camera._menagerie_snapshot(Path(config['menagerie_root']),camera.EXPECTED_MENAGERIE_COMMIT)
    if {k:v for k,v in model.items() if k!='files'}!=config['menagerie'] or camera._openpi_snapshot()!=config['openpi_resize_identity']:
        raise ValueError('qualification model/resize identity changed')
    protocol=shared.read(config['source']['protocol']['path'])
    expected=dict(planned_scenes=50,planned_objects=1871,planned_semantic_queries=269,
                  planned_qualification_cells=2690,input_semantic_queries=6155,budget_exclusions=5886)
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('full stage denominator differs')
    pilot=next(r['scene_id'] for r in protocol['source_populations'] if any(q['scene_id']==r['scene_id'] for q in protocol['qualification_tasks']))
    if config['pilot_scene']!=pilot:raise ValueError('pilot differs from original task ordering')
    return config,protocol


def cpu_guard():
    if os.environ.get('SLURM_ARRAY_JOB_ID'):raise ValueError('ordinary jobs required')
    if any(os.environ.get(k,'') not in ('','-1','NoDevFiles') for k in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES')):
        raise ValueError('qualification requires CPU-only allocation')
    if os.environ.get('SLURM_GPUS_ON_NODE','0') not in ('','0'):raise ValueError('qualification allocated a GPU')


def validate_selected_cells(measured, tasks):
    """Bind canonical cells to the exact predeclared task/arm/reset identities."""
    expected = {(task, arm, episode,
                 f'{arm.lower()}__{task}__seed{screen.BASE_SEED}__ep{episode}')
                for task in tasks for arm in screen.POLICIES for episode in range(screen.EPISODES)}
    rows = measured['metric_rows']
    actual = [(r['task_id'], r['policy_id'], r['episode'], r['cell_id']) for r in rows]
    summaries = [r['task_id'] for r in measured['gate']['task_summaries']]
    if (len(actual) != len(expected) or set(actual) != expected
            or len(summaries) != len(tasks) or set(summaries) != set(tasks)):
        raise ValueError('canonical qualifier omitted or replaced predeclared task/reset cells')


def run(*,config_path,stage_root,expected_commit,scene_id):
    cpu_guard();config,protocol=validate_stage(config_path,stage_root,expected_commit)
    if scene_id not in config['source']['scenes']:raise ValueError('scene outside frozen population')
    stage=Path(stage_root);common=dict(screen_id=config['freeze_id'],scene_id=scene_id,expected_commit=expected_commit)
    tasks=sorted(q['task_id'] for q in protocol['qualification_tasks'] if q['scene_id']==scene_id)
    output=stage/'qualification_handoff'/scene_id
    if output.exists():raise FileExistsError('sealed scene handoff cannot be overwritten')
    if scene_id!=config['pilot_scene'] and tasks:
        pilot=stage/'qualification_handoff'/config['pilot_scene']
        bundle=screen._validate_bundle(pilot,root=screen.evidence_root(),expected_kind=SCOPE)
        prior=screen._read_json_member(bundle['directory'],'result.json')
        if prior.get('status')!='QUALIFICATION_COMPLETE' or prior.get('config')!=shared.identity(config_path):
            raise ValueError('original real qualifier pilot integrity required before remaining scenes')
        pilot_measured=screen._validate_qualifier_output(root=screen.evidence_root(),screen_id=config['freeze_id'],
            scene_id=config['pilot_scene'],expected_commit=expected_commit)
        pilot_tasks=sorted(q['task_id'] for q in protocol['qualification_tasks'] if q['scene_id']==config['pilot_scene'])
        validate_selected_cells(pilot_measured,pilot_tasks)
    result=dict(schema_version=1,scope=SCOPE,scene_id=scene_id,source_commit=expected_commit,
        freeze_id=config['freeze_id'],config=shared.identity(config_path),selected_task_ids=tasks,
        planned_qualification_cells=10*len(tasks),policy_executed=0,policy_success=None,
        camera=None,rollout_ledger=None,paper_ready=False)
    if not tasks:
        result.update(status='NOT_RUN',reason='no_queries_in_original_protocol',qualifier=None)
    else:
        descriptor=stage/'descriptors'/f'{scene_id}.json';descriptor.parent.mkdir(parents=True,exist_ok=True)
        expected=config['source']['scenes'][scene_id]['descriptor']
        if descriptor.exists():
            if shared.read(descriptor)!=expected:raise ValueError('descriptor changed')
        else:
            with descriptor.open('x') as f:json.dump(expected,f,sort_keys=True)
        population=stage/'automatic_candidates'/scene_id/'population'
        if not population.exists():
            screen.prepare_automatic_candidates(**common,e3_root=config['source']['e3_root'],
                automatic_scene_descriptor=descriptor,qualification_protocol=config['source']['protocol']['path'],export=True)
        try:
            if not (stage/'scene_prepares'/scene_id).exists():screen.prepare_automatic_task_suites(**common)
        except screen.CandidateScreenError as exc:
            if str(exc)!=PLANNING_UNQUALIFIED:raise
            result.update(status='PLANNING_UNQUALIFIED',reason=str(exc),qualifier=None,reset_definitions=None)
        else:
            if not (stage/'scene_qualifiers'/scene_id).exists():
                screen.qualify_scene(**common,automatic_population=True,menagerie_root=config['menagerie_root'],
                    expected_menagerie_commit=camera.EXPECTED_MENAGERIE_COMMIT)
            measured=screen._validate_qualifier_output(root=screen.evidence_root(),**common)
            validate_selected_cells(measured,tasks)
            result.update(status='QUALIFICATION_COMPLETE',qualifier=shared.identity(stage/'scene_qualifiers'/scene_id/'manifest.json'),
                prerequisite_cells=measured['gate']['cell_count'],simulator_cells=measured['gate']['exact_900_step_cells'],
                strict_pass_task_ids=measured['gate']['strict_pass_task_ids'])
    after,_=validate_stage(config_path,stage_root,expected_commit)
    if after!=config:raise ValueError('qualification source changed during execution')
    screen._publish_bundle(output,manifest_kind=SCOPE,payloads={'result.json':screen._json_bytes(result)},
        manifest_fields={'code':screen._code_snapshot(expected_commit),'freeze_id':config['freeze_id'],'scene_id':scene_id})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inspect',action='store_true');p.add_argument('--prepare-config',action='store_true')
    for name in ('protocol','e3-root','freeze-id','menagerie-root','out','config','freeze-root','expected-code-commit','scene-id'):
        p.add_argument('--'+name)
    a=p.parse_args()
    if a.inspect:r=inspect_source(a.protocol,a.e3_root)
    elif a.prepare_config:r=prepare_config(protocol_path=a.protocol,e3_root=a.e3_root,freeze_id=a.freeze_id,menagerie_root=a.menagerie_root,out=a.out)
    else:r=run(config_path=a.config,stage_root=a.freeze_root,expected_commit=a.expected_code_commit,scene_id=a.scene_id)
    print(json.dumps(r,indent=2,sort_keys=True))

if __name__=='__main__':main()
