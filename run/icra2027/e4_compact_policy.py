"""Stage the canonical compact or full policy harness from sealed evidence.

This preparer never produces rollout rows or measured reset telemetry. The only
execution shortcut is a wholly rejected matrix, handled by harness_runner itself.
"""
from __future__ import annotations

import argparse
import copy
import json
import getpass
import sys
from pathlib import Path

import yaml

from robo.eval import e4_camera_scorer_gate as camera
from robo.eval import e4_candidate_screen as screen
from robo.eval import episode_log
from robo.eval.e4_reset_eligibility import load_reset_eligibility
from robo.eval.harness_spec import load_harness_spec
from robo.policy.control_contract import FROZEN_CONTROL_CONTRACT, FROZEN_FIELDS_PATH
from robo.policy.registry import PolicyRegistry
from robo.policy.runtime_identity import expected_server_identity_from_config, verify_clean_git_checkout
from run.icra2027 import e4_compact_harness as compact

CODE_ROOT = Path(__file__).resolve().parents[2]


def _policy_contract(chain, *, openpi_root, openpi_commit, cache_root):
    """Read existing policy/rig/control sources; do not infer a service identity."""
    registry = PolicyRegistry.from_config_dir()
    entry = registry.get('pi05_droid_jointpos')
    registry.verify_checkpoint_hash(entry.id)
    if registry.validate_control_contract(entry.id):
        raise ValueError('compact policy differs from frozen control contract')
    if entry.client_kind != 'pi05_server' or entry.status == 'unavailable':
        raise ValueError('compact protocol requires its available real policy')
    resize=chain['summary']['openpi_resize_identity']
    if Path(openpi_root).resolve()!=Path(resize['root']).resolve() or openpi_commit!=resize['commit']:
        raise ValueError('policy OpenPI differs from authenticated camera preprocessing')
    verify_clean_git_checkout(openpi_root, openpi_commit, owner='compact OpenPI')
    menagerie = chain['menagerie']
    source_root=Path(menagerie['source_root'])
    verify_clean_git_checkout(source_root, menagerie['source_commit'], owner='compact Menagerie')
    source_files=[row for name in ('franka_emika_panda','robotiq_2f85')
        for row in camera._tree_inventory(source_root/name,relative_to=source_root)]
    if source_files!=menagerie['files']:
        raise ValueError('policy Menagerie Git source differs from camera runtime rig bytes')
    # Require the actual authoritative file so the legacy contract's fallback
    # defaults can never silently become a newly sealed experiment declaration.
    if not FROZEN_FIELDS_PATH.is_file():
        raise ValueError('frozen control/camera source is missing')
    frozen=yaml.safe_load(FROZEN_FIELDS_PATH.read_text())
    if not all(isinstance(frozen.get(key),dict) for key in ('control','cameras','robot','rubric')):
        raise ValueError('frozen control/camera source is malformed')
    template = yaml.safe_load((CODE_ROOT/'configs/experiments/icra2027/harness.yaml').read_text())
    contract = copy.deepcopy(template['contract'])
    contract['policy'] = dict(id=entry.id, kind='real', checkpoint_path=entry.checkpoint_path,
        checkpoint_hash=entry.checkpoint_hash, checkpoint_hash_kind='tree_path_size_mtime_sha256',
        training_config=entry.training_config)
    from robo.policy.sampling_contract import sampling_contract
    contract['policy']['sampling'] = sampling_contract(entry.training_config)
    contract['runtime_dependencies'] = {
        'openpi': {'root': str(Path(openpi_root).resolve()), 'commit': openpi_commit},
        'mujoco_menagerie': {'root': str(source_root), 'commit': menagerie['source_commit']},
        'checkpoint_cache_root': str(cache_root)}
    control = FROZEN_CONTROL_CONTRACT
    contract['action_convention'] = control.env_action_convention
    contract['control_rate_hz'] = control.rate_hz
    contract['horizon_s'] = control.horizon_seconds
    contract['controller'] = {key:getattr(control,key) for key in (
        'physics_dt','substeps_per_tick','action_dim','gripper_binarize_threshold',
        'per_tick_joint_delta_clamp_rad')}
    contract['cameras']['by_scene'] = {
        scene: {'ext_cam':data['suites']['A0']['ext_cam']}
        for scene,data in chain['scenes'].items()}
    contract['cameras']['image_preprocessing'] = entry.image_preprocessing.model_dump(mode='json')
    contract['robot']['by_scene'] = {
        scene:data['suites']['A0']['robot'] for scene,data in chain['scenes'].items()}
    contract['task_instruction'] = 'Frozen per-task default instruction from authenticated planning suite'
    return contract


def build_config(*, camera_config_path, camera_config_sha256, camera_output,
                 expected_code_commit, out, openpi_root, openpi_commit, host='localhost', port=8000,
                 mode='real', scripted_output=None, scripted_gate_sha256=None, freeze_id=None):
    """Authenticate before constructing the compact40 or full160-cell matrix."""
    root = camera._evidence_root()
    code = camera._git_snapshot(expected_code_commit)
    config_path = camera._regular_file(Path(camera_config_path),root=root,label='compact camera config')
    if screen._sha256(config_path) != camera_config_sha256:
        raise ValueError('compact camera config SHA256 differs')
    original = json.loads(config_path.read_text())
    chain = camera.validate_automatic_chain(original,root=root)
    checked = camera.validate_automatic_camera_output(config_path=config_path,
        output=camera_output,expected_code_commit=expected_code_commit)
    is_full=original.get('study_scope')==camera.FULL_AUTOMATIC_SCOPE
    selection=None;pilot_ids=None
    if is_full:
        from robo.eval.e4_reset_eligibility import full_qualified_policy_selection
        selection,states,pilot_ids,chain,_=full_qualified_policy_selection(original,chain,checked)
        if selection['status']!='READY_FOR_FROZEN_PILOT':
            raise ValueError('full qualified policy matrix NOT_RUN: '+json.dumps(selection,sort_keys=True))
    else:
        protocol = compact.checked_protocol(original['protocol']['path'],original['protocol']['sha256'])
        states = compact.definitions(protocol)
    expected_resets=80 if is_full else 20
    out = Path(out).resolve()
    if not freeze_id or freeze_id==original['freeze_id']:
        raise ValueError('policy stage requires a separately reserved freeze ID')
    camera._validated_id(freeze_id,label='policy stage freeze')
    required_parent = root/'outputs/icra2027'/freeze_id/'harness'
    if mode not in {'real','scripted'}:
        raise ValueError('unsupported compact policy mode')
    expected_name = ('full_policy' if mode=='real' else 'full_scripted_smoke') if is_full else ('compact_policy' if mode=='real' else 'compact_scripted_smoke')
    if out.parent != required_parent.resolve() or out.name != expected_name:
        raise ValueError('compact output must be the camera freeze harness/compact_policy directory')
    if len(states)!=expected_resets or len({s.reset_state_id for s in states})!=expected_resets:
        raise ValueError('compact canonical reset denominator differs')
    contract = _policy_contract(chain,openpi_root=openpi_root,openpi_commit=openpi_commit,
                                 cache_root=Path('/scratch')/getpass.getuser()/'icra2027'/original['freeze_id']/'policy-checkpoint-cache')
    real_policy=copy.deepcopy(contract['policy'])
    if mode=='scripted':
        contract['policy']={'id':'scripted_sinusoid','kind':'scripted_smoke',
            'checkpoint_hash':None,'checkpoint_hash_kind':'not_applicable'}
    contract['reset_ids'] = [s.reset_state_id for s in states]
    scenes = []
    for scene,data in chain['scenes'].items():
        bundle=data['task_bundle']
        scenes.append({'id':scene,'tasks_json':bundle['planning_tasks'],
            'task_freeze_manifest':str(Path(bundle['planning_tasks']).parent/'manifest.json'),
            'menagerie_root':chain['menagerie']['root'],
            'construction_variants':{variant:{'factory_dir':bundle['factories'][arm],
                'tasks_json':bundle['variant_tasks'][arm],'scene_xml':bundle['scene_xml'][arm]}
                for arm,variant in [('A0','fixed_single_path'),('A4','agentic')]}})
    config={'schema_version':1,'freeze_id':freeze_id,'paper_mode':False,'paper_ready':False,
        'study_scope':(('automatic_full_policy_engineering' if mode=='real' else 'e4_full_scripted_smoke') if is_full
            else ('automatic_compact_policy_engineering' if mode=='real' else 'e4_compact_scripted_smoke')),'out_dir':str(out),
        'policy':contract['policy']['id'],'checkpoint_path':contract['policy'].get('checkpoint_path'),
        'host':host,'port':int(port),'seeds':[0],'episodes':5,'jitter':screen.JITTER_XY_M,
        'jitter_first_episode':True,'variant':'default','horizon_s':contract['horizon_s'],
        'video':True,'open_loop_horizon':FROZEN_CONTROL_CONTRACT.chunk_size,
        'contract':contract,'scenes':scenes,
        'treatments':[{'id':f'{arm}_raster','scene':variant,'collision':'full_room','observation':'raster'}
            for arm,variant in [('a0','fixed_single_path'),('a4','agentic')]],
        'comparisons':[{'id':'construction','axis':'scene','baseline':'a0_raster',
                       'treatments':['a0_raster','a4_raster']}],
        'cpu_reset_eligibility':{'kind':'automatic_full' if is_full else 'automatic_compact',
            'camera_config':{'path':str(config_path),'sha256':camera_config_sha256},
            'bundle':str(Path(camera_output).resolve()),
            'gate_sha256':screen._sha256(Path(camera_output)/'gate.json'),
            'source_commit':expected_code_commit}}
    if is_full:config.update(full_qualification_selection=selection,pilot_reset_ids=pilot_ids)
    spec=load_harness_spec(config)
    indexed=load_reset_eligibility(config,spec,states,root=root)
    if mode=='real':
        expected_server_identity_from_config(config)  # Local files only; no connection.
    if len(indexed)!=2*expected_resets:
        raise ValueError('compact eligibility denominator differs')
    smoke=None
    if mode=='real' and any(item['passed'] for item in indexed.values()):
        if not scripted_output or not scripted_gate_sha256:
            raise ValueError('eligible real-policy preparation requires sealed canonical scripted smoke')
        smoke=validate_scripted_smoke(scripted_output,expected_code_commit,
            expected_sha256=scripted_gate_sha256,real_config=config)
        config['scripted_smoke']={'out':str(Path(scripted_output).resolve()),
                                  'gate_sha256':scripted_gate_sha256}
    return config,states,{'freeze_id':freeze_id,'mode':mode,'expected_real_policy':real_policy,'scripted_smoke':smoke,'code':code,'upstream':chain['summary'],
        'camera_manifest_sha256':checked['manifest_sha256'],
        'planned_reset_definitions':expected_resets,'planned_episodes':2*expected_resets,
        'eligible_episodes':sum(item['passed'] for item in indexed.values()),
        'failure_counts':{kind:sum(item['failure_type']==kind for item in indexed.values())
            for kind in ('cpu_reset_validity_failure','camera_scorer_failure')},
        'measured_reset_telemetry':False,'policy_invoked':False,'paper_ready':False,
        'policy_launch_allowed':False,
        'remaining_gate':'Scripted runtime integrity first; live real-policy identity/warmup is verified by canonical harness before actions.'}


def prepare(**kwargs):
    out=Path(kwargs['out']).resolve()
    config,states,receipt=build_config(**kwargs)
    with camera._atomic_directory(out) as staging:
        (staging/'harness_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        episode_log.save_reset_states(states,staging/'reset_states.json')
        receipt['files']={name:screen._sha256(staging/name)
                          for name in ('harness_config.yaml','reset_states.json')}
        (staging/'preparation_receipt.json').write_bytes(screen._json_bytes(receipt))
        (staging/'preparation_seal.json').write_bytes(screen._json_bytes({
            'preparation_receipt_sha256':screen._sha256(staging/'preparation_receipt.json')}))
        stage_config=_stage_freeze_config(config,out,receipt)
        (staging/'stage_freeze.yaml').write_text(yaml.safe_dump(stage_config,sort_keys=False))
        camera._fsync_tree(staging)
    return receipt



def _stage_freeze_config(config,out,receipt):
    """Use the existing E0 schema and resource producer without new hash rules."""
    from robo.eval import freeze
    value=yaml.safe_load((CODE_ROOT/'configs/experiments/icra2027/freeze.yaml').read_text())
    value.update(freeze_id=config['freeze_id'],code_commit=receipt['code']['commit'],
        mode='smoke',git_dirty=False,paper_repository=str(camera._evidence_root().parent/'SimAnyRoom'))
    for field in freeze.CONFIG_FIELDS:
        value[field]=str((CODE_ROOT/value[field]).resolve())
    value['harness_config']=str(out/'harness_config.yaml')
    dependencies=config['contract']['runtime_dependencies']
    declaration=config['cpu_reset_eligibility']
    paths={'harness_config':out/'harness_config.yaml','reset_bank':out/'reset_states.json',
        'camera_config':declaration['camera_config']['path'],
        'camera_gate':Path(declaration['bundle'])/'gate.json',
        'preparation_receipt':out/'preparation_receipt.json','python':sys.executable,
        'openpi':dependencies['openpi']['root'],
        'menagerie_git':dependencies['mujoco_menagerie']['root'],
        'frozen_fields':FROZEN_FIELDS_PATH,'preparer':Path(__file__)}
    if config.get('scripted_smoke'):
        paths['scripted_gate']=Path(config['scripted_smoke']['out'])/'scripted_smoke_gate.json'
    value['input_roots']=[{'id':key,'path':str(path),'kind':'metadata','required':True}
                          for key,path in paths.items()]
    value['checkpoint_roots']=[{'id':'fixed_real_policy','path':receipt['expected_real_policy']['checkpoint_path'],
                                'kind':'checkpoint','required':True}]
    value['hardware']={'accelerator':'cpu' if receipt['mode']=='scripted' or not receipt['eligible_episodes'] else 'gpu',
                       'purpose':'compact canonical engineering stage'}
    return value


def _validate_stage_e0(contract,config,out,expected_code_commit):
    from robo.eval import freeze
    from robo.manifest.hash import canonical_hash
    contract=Path(contract).resolve()
    if contract!=(out.parent.parent/'contract').resolve():
        raise ValueError('policy E0 must belong to the separately reserved stage freeze')
    manifest=json.loads((contract/'freeze_manifest.json').read_text())
    digest=canonical_hash({k:v for k,v in manifest.items() if k not in {'created_utc','environment','contract_sha256'}})
    if (manifest.get('contract_sha256')!=digest or manifest.get('freeze_id')!=config['freeze_id']
            or manifest['code'].get('commit')!=expected_code_commit or manifest['code'].get('dirty') is not False
            or manifest.get('paper_ready') is not False):
        raise ValueError('policy E0 checksum/source/freeze differs')
    if 'PREFLIGHT=PASS' not in (contract/'preflight.log').read_text():
        raise ValueError('policy E0 lacks exact-source preflight PASS')
    receipt=json.loads((out/'preparation_receipt.json').read_text())
    required=_stage_freeze_config(config,out,receipt)
    for category in ('input_roots','checkpoint_roots'):
        for entry in required[category]:
            current=freeze._inventory_resource(entry,category=category,repo_root=CODE_ROOT)
            matches=[r for r in manifest['resource_inventory'] if r.get('category')==category and r.get('id')==entry['id']]
            if len(matches)!=1 or any(matches[0].get(k)!=current[k] for k in (
                    'resolved_path','exists','type','size_bytes','sha256','hash_method')):
                raise ValueError(f"policy E0 input/runtime identity differs: {entry['id']}")


def _prepared(out, expected_code_commit):
    out=Path(out).resolve()
    camera._git_snapshot(expected_code_commit)
    receipt=json.loads((out/'preparation_receipt.json').read_text())
    seal=json.loads((out/'preparation_seal.json').read_text())
    if seal!={'preparation_receipt_sha256':screen._sha256(out/'preparation_receipt.json')}:
        raise ValueError('compact preparation receipt changed')
    if receipt['code']['commit']!=expected_code_commit:
        raise ValueError('compact preparation source changed')
    for name in ('harness_config.yaml','reset_states.json'):
        if screen._sha256(out/name)!=receipt['files'][name]:
            raise ValueError('compact prepared config/reset bank changed')
    config=yaml.safe_load((out/'harness_config.yaml').read_text())
    if Path(config['out_dir']).resolve()!=out:
        raise ValueError('compact prepared harness output differs')
    states=episode_log.load_reset_states(out/'reset_states.json')
    spec=load_harness_spec(config)
    eligibility=load_reset_eligibility(config,spec,states,root=camera._evidence_root())
    if len(eligibility)!=(160 if config['cpu_reset_eligibility']['kind']=='automatic_full' else 40):
        raise ValueError('compact prepared denominator differs')
    return config,states,spec,eligibility


def _scripted_receipt(out, expected_code_commit):
    """Validate the existing canonical ledger and traces; define no new scorer."""
    from robo.eval import harness_runner, harness_validation
    out=Path(out).resolve()
    config,states,spec,eligibility=_prepared(out,expected_code_commit)
    if config['policy']!='scripted_sinusoid' or config['study_scope'] not in {'e4_compact_scripted_smoke','e4_full_scripted_smoke'}:
        raise ValueError('task smoke requires canonical scripted policy scope')
    ledger=out/harness_runner.LEDGER_NAME
    records=harness_validation.read_jsonl(ledger)
    planned={s.reset_state_id for s in states}
    validation=harness_validation.validate_saved_treatment_records(records,spec,planned,
        manifest_root=camera._evidence_root())
    resolved,_,planning=harness_runner._preflight_resolved_scenes(config,spec,out)
    violations=harness_runner._validate_record_inputs_against_current(records,resolved,planning)
    if len(records)!=2*len(states) or not validation['ok'] or violations:
        raise ValueError('canonical scripted ledger failed denominator/source/paired validation')
    files={str(ledger):screen._sha256(ledger)}
    executed=0
    for record in records:
        eligible=eligibility[(record['treatment_id'],record['reset_state_id'])]
        if not eligible['passed']:
            continue  # The canonical validator authenticates original prebuild evidence.
        if record['outcome'] not in {'success','task_failure'} or record.get('ticks',0)<=0:
            raise ValueError('eligible scripted reset lacks completed runtime/scorer smoke')
        manifest_path=camera._regular_file(Path(record['manifest_path']),root=out,label='scripted manifest')
        trace_path=camera._regular_file(Path(record['trace_path']),root=out,label='scripted trace')
        manifest=json.loads(manifest_path.read_text())
        if manifest['git']['commit']!=expected_code_commit or manifest['git']['dirty']:
            raise ValueError('scripted runtime source differs')
        if (manifest.get('outcome')!=record['outcome'] or manifest['contract']['policy']['id']!='scripted_sinusoid'
                or manifest['contract']['policy']['kind']!='scripted_smoke'):
            raise ValueError('scripted ledger/manifest policy or outcome differs')
        ticks=episode_log.read_timeseries(trace_path)
        if len(ticks)!=record['ticks'] or any(tick.get('t')!=i or not isinstance(tick.get('stages'),dict)
                or record['task_id'] not in tick['stages'] for i,tick in enumerate(ticks)):
            raise ValueError('scripted trace lacks canonical tick/scorer integrity')
        files.update({str(manifest_path):screen._sha256(manifest_path),str(trace_path):screen._sha256(trace_path)})
        executed+=1
    return {'schema_version':1,'study_scope':config['study_scope'],
        'source_commit':expected_code_commit,'planned_episodes':2*len(states),'executed_episodes':executed,
        'runtime_scorer_integrity_passed':True,'paper_ready':False,
        'success_required':False,'config_sha256':screen._sha256(out/'harness_config.yaml'),
        'reset_bank_sha256':screen._sha256(out/'reset_states.json'),'files':files}


def validate_scripted_smoke(out, expected_code_commit, *, expected_sha256, real_config):
    out=Path(out).resolve()
    gate=out/'scripted_smoke_gate.json'
    if screen._sha256(gate)!=expected_sha256:
        raise ValueError('scripted smoke gate SHA256 differs')
    fresh=_scripted_receipt(out,expected_code_commit)
    if json.loads(gate.read_text())!=fresh:
        raise ValueError('scripted smoke source/artifact drift')
    scripted=yaml.safe_load((out/'harness_config.yaml').read_text())
    def paired(config):
        value=copy.deepcopy(config)
        for key in ('policy','checkpoint_path','study_scope','out_dir','scripted_smoke','freeze_id'):
            value.pop(key,None)
        value['contract'].pop('policy',None)
        return value
    if paired(scripted)!=paired(real_config):
        raise ValueError('scripted/real frozen task, geometry, camera, reset or control drift')
    return fresh


def _real_pilot_receipt(out, expected_code_commit):
    """Validate the fixed first forty canonical rows without requiring success."""
    from robo.eval import harness_runner, harness_validation
    out=Path(out).resolve();config,states,spec,eligibility=_prepared(out,expected_code_commit)
    if config['cpu_reset_eligibility']['kind']!='automatic_full' or config['policy']!='pi05_droid_jointpos':
        raise ValueError('real pilot requires the full frozen learned-policy matrix')
    ids=set(config['pilot_reset_ids'])
    all_records=harness_validation.read_jsonl(out/harness_runner.LEDGER_NAME)
    full_keys={(t,state.reset_state_id) for state in states for t in spec.treatments}
    actual_keys=[(r.get('treatment_id'),r.get('reset_state_id')) for r in all_records]
    if len(actual_keys)!=len(set(actual_keys)) or not set(actual_keys)<=full_keys:
        raise ValueError('real pilot ledger contains duplicate or foreign full-matrix cells')
    records=[r for r in all_records if r['reset_state_id'] in ids]
    pilot_spec=harness_runner.full_pilot_validation_spec(spec,states)
    valid=harness_validation.validate_saved_treatment_records(records,pilot_spec,ids,manifest_root=camera._evidence_root())
    resolved,_,planning=harness_runner._preflight_resolved_scenes(config,spec,out)
    if (len(ids)!=20 or len(records)!=40 or not valid['ok']
            or harness_runner._validate_record_inputs_against_current(records,resolved,planning)):
        raise ValueError('fixed real pilot canonical identity/coverage validation failed')
    files={}
    for record in records:
        if record['outcome'] not in {'success','task_failure'} or record.get('ticks',0)<=0:
            raise ValueError('real pilot lacks completed policy/runtime evidence')
        for field in ('manifest_path','trace_path'):
            path=camera._regular_file(Path(record[field]),root=out,label='real pilot artifact')
            files[str(path)]=screen._sha256(path)
        manifest=json.loads(Path(record['manifest_path']).read_text())
        if (manifest.get('git',{}).get('commit')!=expected_code_commit
                or manifest.get('git',{}).get('dirty') is not False):
            raise ValueError('real pilot runtime source differs')
        if (manifest.get('outcome')!=record['outcome']
                or manifest.get('contract',{}).get('policy',{}).get('id')!='pi05_droid_jointpos'
                or manifest['contract']['policy'].get('kind')!='real'):
            raise ValueError('real pilot ledger/manifest policy or outcome differs')
        trace=episode_log.read_timeseries(Path(record['trace_path']))
        if (len(trace)!=record['ticks'] or any(tick.get('t')!=i or not isinstance(tick.get('stages'),dict)
                or record['task_id'] not in tick['stages'] for i,tick in enumerate(trace))):
            raise ValueError('real pilot trace/ledger tick or scorer integrity differs')
    return dict(schema_version=1,study_scope='full_qualified_real_policy_pilot',source_commit=expected_code_commit,
        planned_episodes=40,full_planned_episodes=160,success_required=False,paper_ready=False,
        config_sha256=screen._sha256(out/'harness_config.yaml'),pilot_reset_ids=config['pilot_reset_ids'],
        records_sha256=camera._canonical_hash(records),files=files)


def validate_real_pilot_before_full(out, expected_code_commit):
    out=Path(out);path=out/'real_pilot_gate.json'
    if not path.is_file() or path.is_symlink():raise ValueError('full policy release requires original fixed real pilot integrity')
    replay=_real_pilot_receipt(out,expected_code_commit)
    if json.loads(path.read_text())!=replay:raise ValueError('real pilot artifact gate changed')
    return replay


def execute(*, out, expected_code_commit, mode, contract):
    """Dispatch exclusively to harness_runner; source/eligibility errors are fatal."""
    from robo.eval import harness_runner
    out=Path(out).resolve()
    declared=yaml.safe_load((out/'harness_config.yaml').read_text())
    _validate_stage_e0(contract,declared,out,expected_code_commit)
    config,states,spec,eligibility=_prepared(out,expected_code_commit)
    if mode=='prebuild-only':
        if any(item['passed'] for item in eligibility.values()):
            raise ValueError('prebuild-only cannot launch eligible cells')
    elif mode=='scripted':
        if config['policy']!='scripted_sinusoid' or config['study_scope'] not in {'e4_compact_scripted_smoke','e4_full_scripted_smoke'}:
            raise ValueError('scripted execution requires its explicit smoke scope')
    elif mode in {'real','real-pilot'}:
        if mode=='real-pilot' and config['cpu_reset_eligibility']['kind']!='automatic_full':
            raise ValueError('real pilot subset requires full qualified protocol')
        if config['policy']!='pi05_droid_jointpos':
            raise ValueError('real execution requires the fixed protocol policy')
        if any(item['passed'] for item in eligibility.values()):
            smoke=config.get('scripted_smoke')
            if not smoke:
                raise ValueError('eligible real execution lacks canonical scripted smoke')
            validate_scripted_smoke(smoke['out'],expected_code_commit,
                expected_sha256=smoke['gate_sha256'],real_config=config)
        expected_server_identity_from_config(config)
    else:
        raise ValueError('unknown compact execution mode')
    result=harness_runner.run_matrix(config,out,resume=True,**({'execution_stage':'pilot'} if mode=='real-pilot' else {}))
    if mode=='real-pilot':
        gate=_real_pilot_receipt(out,expected_code_commit)
        path=out/'real_pilot_gate.json'
        if path.exists():
            if json.loads(path.read_text())!=gate:raise ValueError('real pilot gate changed')
        else:
            with path.open('xb') as stream:stream.write(screen._json_bytes(gate))
    if mode=='scripted':
        gate=_scripted_receipt(out,expected_code_commit)
        path=out/'scripted_smoke_gate.json'
        if path.exists():
            if json.loads(path.read_text())!=gate:
                raise ValueError('refusing to overwrite a different scripted smoke gate')
        else:
            with path.open('xb') as stream:
                stream.write(screen._json_bytes(gate))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--camera-config-path',required=True)
    prep.add_argument('--camera-config-sha256',required=True)
    prep.add_argument('--camera-output',required=True)
    prep.add_argument('--openpi-root',required=True)
    prep.add_argument('--openpi-commit',required=True)
    prep.add_argument('--host',default='localhost')
    prep.add_argument('--port',type=int,default=8000)
    prep.add_argument('--freeze-id',required=True)
    prep.add_argument('--mode',choices=['real','scripted'],default='real')
    prep.add_argument('--scripted-output')
    prep.add_argument('--scripted-gate-sha256')
    run=sub.add_parser('run')
    run.add_argument('--contract',required=True)
    run.add_argument('--mode',choices=['prebuild-only','scripted','real','real-pilot'],required=True)
    for command in (prep,run):
        command.add_argument('--expected-code-commit',required=True)
        command.add_argument('--out',required=True)
    args=vars(parser.parse_args());command=args.pop('command')
    result=prepare(**args) if command=='prepare' else execute(**args)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
