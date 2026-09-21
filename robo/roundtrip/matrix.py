"""Immutable native cohort planning and one-writer collection around existing runners.

Planning slots are NOT canonical instance identities. Resolve them with N0's
sealed native bundles before an episode can be dispatched. No job arrays.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import yaml
from robo.manifest.hash import canonical_hash, git_snapshot

IDENTITY_FIELDS = ('cohort_id','canonical_instance_id','reset_id','policy_id',
 'controller_method','scope','sensor_regime','renderer','execution_protocol')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save_new(path, value, *, jsonl=False):
    """Atomic no-overwrite publication within the destination filesystem."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.partial.{os.getpid()}')
    data=(''.join(json.dumps(x,sort_keys=True,allow_nan=False)+'\n' for x in value) if jsonl
          else json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    with temporary.open('x') as f:f.write(data);f.flush();os.fsync(f.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink()


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def enumerate_plan(config):
    if config.get('kind')!='native_matrix_plan' or config.get('schema_version')!=1:
        raise ValueError('requires native_matrix_plan schema1, not per-unit native spec')
    if config.get('split') not in ('development','test'):raise ValueError('declare cohort split')
    layouts=config['layout_ids'];seeds=config['instance_generation_seeds'];resets=config['reset_perturbations'];methods=config['controller_methods']
    for name,values in [('layouts',layouts),('generation seeds',seeds),('methods',methods),('reset IDs',[r['reset_id'] for r in resets])]:
        if not values or len(values)!=len(set(values)):raise ValueError('duplicate or empty '+name)
    if any(isinstance(x,bool) or not isinstance(x,int) for x in layouts+seeds):raise ValueError('integer layout/generation seeds required')
    for reset in resets:
        t=reset['translation_world_m']
        if len(t)!=3 or any(not __import__('math').isfinite(float(x)) for x in t):raise ValueError('finite world translation required')
        if not isinstance(reset['policy_rng_seed'],int):raise ValueError('policy RNG seed required')
    slots=[];planned=[]
    for layout in layouts:
        for task,horizon in config['tasks'].items():
            if not isinstance(horizon,int) or horizon<=0:raise ValueError('native task horizon unresolved')
            for generation_seed in seeds:
                slot={'cohort_id':config['cohort_id'],'layout_id':layout,'style_id':config['style_id'],
                      'task_id':task,'native_horizon':horizon,'generation_seed':generation_seed,'split':config['split']}
                slot['instance_slot_id']='slot-'+canonical_hash(slot)[:20];slots.append(slot)
                for reset in resets:
                    for method in methods:
                        row={**slot,'canonical_instance_id':None,'reset_id':reset['reset_id'],
                             'policy_rng_seed':reset['policy_rng_seed'],'translation_world_m':reset['translation_world_m'],
                             **{k:config[k] for k in ('policy_id','scope','sensor_regime','renderer','execution_protocol')},
                             'controller_method':method,'terminal_status':'UNRESOLVED_INSTANCE','executed':None,'success':None}
                        row['plan_unit_id']='plan-'+canonical_hash({k:v for k,v in row.items() if k not in ('terminal_status','executed','success')})[:24]
                        planned.append(row)
    return slots,planned


def resolve_plan(config, bindings):
    """Bind every slot to one actual N0-validated canonical manifest/reset bank."""
    from robo.roundtrip.identity import validate_canonical_instance,validate_reset_bank
    slots,planned=enumerate_plan(config);by_slot={b['instance_slot_id']:b for b in bindings}
    if len(by_slot)!=len(bindings) or set(by_slot)!={s['instance_slot_id'] for s in slots}:
        raise ValueError('bindings must cover each declared slot once; unavailable slots cannot be replaced')
    ids=set()
    for slot in slots:
        b=by_slot[slot['instance_slot_id']];m=json.loads(Path(b['canonical_manifest']).read_text());bank=json.loads(Path(b['reset_bank']).read_text())
        validate_canonical_instance(m,b['bundle_dir']);validate_reset_bank(bank,m)
        actual=m['canonical_instance_id']
        if actual in ids:raise ValueError('duplicate actual instance in independent build roster')
        ids.add(actual);b['_manifest']=m;b['_bank']=bank
        expected_resets=[]
        for r in config['reset_perturbations']:
            delta=[[1.,0.,0.,float(r['translation_world_m'][0])],[0.,1.,0.,float(r['translation_world_m'][1])],[0.,0.,1.,float(r['translation_world_m'][2])],[0.,0.,0.,1.]]
            expected_resets.append({'reset_id':r['reset_id'],'delta_world':delta,'policy_rng_seed':r['policy_rng_seed']})
        if bank['resets']!=expected_resets:raise ValueError('reset bank differs from predeclared perturbation roster')
        # Native task/layout/style are authoritative even when seed integers match.
        for key in ('layout_id','style_id','task_id'):
            observed=m['identity'].get('task' if key=='task_id' else key)
            if observed!=slot[key]:raise ValueError('native instance differs from planned '+key)
    rows=[]
    for row in planned:
        b=by_slot[row['instance_slot_id']]
        bound={**row,'canonical_instance_id':b['_manifest']['canonical_instance_id'],
               'canonical_manifest':str(Path(b['canonical_manifest']).resolve()),'canonical_manifest_sha256':b['_manifest']['manifest_sha256'],'canonical_manifest_file_sha256':sha(b['canonical_manifest']),
               'reset_bank':str(Path(b['reset_bank']).resolve()),'reset_bank_sha256':sha(b['reset_bank']),'reset_contract_sha256':b['_bank']['reset_contract_sha256'],
               'bundle_dir':str(Path(b['bundle_dir']).resolve()),'terminal_status':'NOT_SCHEDULED'}
        bound['unit_id']='unit-'+canonical_hash({k:bound[k] for k in IDENTITY_FIELDS})[:32]
        rows.append(bound)
    return rows


def canonical_reset_roster_hash(rows):
    fields=('canonical_instance_id','canonical_manifest_sha256','reset_contract_sha256','instance_slot_id')
    identities={}
    for u in rows:
        entry={k:u[k] for k in fields};key=u['canonical_instance_id']
        if key in identities and identities[key]!=entry:raise ValueError('conflicting canonical reset roster')
        identities[key]=entry
    return canonical_hash([identities[k] for k in sorted(identities)])


def write_resolved_configs(planned,bindings,out,*,worker_root,host='localhost',port=8017,python_native='/group/worldcept/PhiRIE/code/SimAny-wt/sr0-native/.venv-native/bin/python',builds=(),reference_units=(),test_admission=None):
    """Emit N5 v2 configs and executable existing-runner commands, no rollout."""
    import copy
    from robo.roundtrip.spec import validate_spec
    by_slot={b['instance_slot_id']:b for b in bindings};objects={}
    for build in builds:
        key=(build['canonical_instance_id'],build['controller_method'])
        if key in objects:raise ValueError('duplicate method build binding')
        objects[key]=build['object_dir']
    out=Path(out);worker_root=Path(worker_root).resolve();rows=[];commands={}
    for unit in planned:
        binding=by_slot[unit['instance_slot_id']]
        if not binding.get('native_config'):raise ValueError('sealed instance native_config missing')
        native_path=Path(binding['native_config']);c=copy.deepcopy(json.loads(native_path.read_text()))
        # Acquisition config is identity-bound through the N0 policy fingerprint.
        manifest=json.loads(Path(unit['canonical_manifest']).read_text())
        if canonical_hash(c['policy'])!=manifest['identity']['policy_sha256']:raise ValueError('native acquisition policy changed')
        for key in ('platform','robot','camera_size'):
            if c[key]!=manifest['identity'][key]:raise ValueError('native acquisition configuration changed: '+key)
        for key in ('task_id','layout_id','style_id'):
            if c['instance'][key]!=unit[key]:raise ValueError('native acquisition instance changed: '+key)
        c.update(schema_version=2,policy_id=unit['policy_id'],renderer=unit['renderer'],cohort_id=unit['cohort_id'],canonical_instance_id=unit['canonical_instance_id'],
            reset_id=unit['reset_id'],policy_rng_seed=unit['policy_rng_seed'],scope=unit['scope'],
            controller_method=unit['controller_method'],execution_protocol=unit['execution_protocol'],
            canonical_manifest_sha256=unit['canonical_manifest_sha256'],reset_contract_sha256=unit['reset_contract_sha256'])
        if unit['split']=='test':
            if test_admission is None:
                ref=next((r for r in reference_units if r['canonical_instance_id']==unit['canonical_instance_id'] and r['reset_id']==unit['reset_id']),None)
                inherited=None if ref is None else json.loads(Path(ref['config_path']).read_text()).get('test_admission')
                if inherited is None:raise ValueError('resolved TEST execution admission required')
                c['test_admission']=inherited
                protocol=json.loads(Path(ref['config_path']).read_text()).get('policy_engine_protocol')
                if protocol is not None:c['policy_engine_protocol']=protocol
            else:
                if test_admission.get('kind')!='resolved_test_execution_admission' or test_admission.get('ready') is not True:
                    raise ValueError('resolved TEST execution admission is not ready')
                if test_admission.get('roster_sha256')!=canonical_reset_roster_hash(planned):raise ValueError('resolved TEST roster differs from admission')
                c['test_admission']={k:test_admission[k] for k in ('roster_sha256','dev_gate_sha256','thresholds_sha256')}
                protocol=test_admission.get('policy_engine_protocol')
                if protocol is not None:
                    if protocol!='per_canonical_engine_v1':raise ValueError('unknown TEST engine protocol')
                    c['policy_engine_protocol']=protocol
        validate_spec(c);config_path=out/'configs'/(unit['unit_id']+'.json');save_new(config_path,c)
        rows.append({**unit,'config_path':str(config_path.resolve()),'config_sha256':canonical_hash(c),
            'native_config_source':str(native_path.resolve()),'native_config_source_sha256':sha(native_path),
            'result_path_planned':str(worker_root/unit['unit_id']/'runner'/'episode'/'result.json')})
    commands=commands_for_units([*reference_units,*rows],builds,host=host,port=port,python_native=python_native)
    commands={r['unit_id']:commands[r['unit_id']] for r in rows if r['unit_id'] in commands}
    save_new(out/'planned_units.jsonl',rows,jsonl=True);save_new(out/'commands.json',commands)
    return rows,commands


def extend_methods(planned,methods,bindings,out,*,worker_root,builds=(),host='localhost',port=8017):
    """Append declared treatment arms without rewriting or rerunning existing units."""
    import copy
    if not methods or len(set(methods))!=len(methods):raise ValueError('duplicate or empty extension methods')
    present={u['controller_method'] for u in planned}
    if present.intersection(methods):raise ValueError('extension method already exists')
    refs=[u for u in planned if u['controller_method']=='REF_NATIVE']
    keys=[(u['canonical_instance_id'],u['reset_id']) for u in refs]
    if not refs or len(keys)!=len(set(keys)):raise ValueError('unique matched reference roster required')
    rows=[]
    for ref in refs:
        for method in methods:
            u=copy.deepcopy(ref)
            for k in ('config_path','config_sha256','native_config_source','native_config_source_sha256','result_path_planned'):
                u.pop(k,None)
            u.update(controller_method=method,terminal_status='NOT_SCHEDULED',executed=None,success=None)
            u['unit_id']='unit-'+canonical_hash({k:u[k] for k in IDENTITY_FIELDS})[:32]
            u['plan_unit_id']='plan-'+canonical_hash({k:u[k] for k in IDENTITY_FIELDS})[:24]
            rows.append(u)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    emitted,commands=write_resolved_configs(rows,bindings,out/'extension',worker_root=worker_root,
        builds=builds,host=host,port=port,reference_units=refs)
    combined=[*planned,*emitted]
    save_new(out/'planned_units.jsonl',combined,jsonl=True);save_new(out/'commands.json',commands)
    save_new(out/'extension_receipt.json',{'existing_planned_units_sha256':canonical_hash(planned),
        'existing_units_unchanged':True,'added_methods':methods,'added_units':len(emitted),'total_units':len(combined)})
    return combined,commands


def commands_for_units(rows,builds=(),*,host='localhost',port=8017,python_native='/group/worldcept/PhiRIE/code/SimAny-wt/sr0-native/.venv-native/bin/python'):
    objects={};commands={}
    for b in builds:
        key=(b['canonical_instance_id'],b['controller_method'])
        if key in objects:raise ValueError('duplicate method build binding')
        status=b.get('terminal_status')
        if status in ('ABSTAINED','BUILD_FAILED'):
            if b.get('accepted') is True:raise ValueError('accepted build contradicts terminal status')
            objects[key]=None
            continue
        if b.get('accepted') is False:raise ValueError('unaccepted build requires explicit terminal status')
        objects[key]=b['object_dir']
    references={(r['canonical_instance_id'],r['reset_id']):r for r in rows if r['controller_method']=='REF_NATIVE'}
    for unit in rows:
        argv=[python_native,'-m','robo.roundtrip.paired','--config',unit['config_path'],
              '--canonical-reference',unit['bundle_dir'],'--canonical-manifest',unit['canonical_manifest'],
              '--reset-bank',unit['reset_bank'],'--out',str(Path(unit['result_path_planned']).parent.parent),
              '--host',host,'--port',str(port)]
        requires=[]
        if unit['controller_method']!='REF_NATIVE':
            obj=objects.get((unit['canonical_instance_id'],unit['controller_method']))
            if obj is None:continue  # planned unit persists; missing build is not a fabricated outcome
            ref=references[(unit['canonical_instance_id'],unit['reset_id'])]
            ref_result=Path(ref['result_path_planned'])
            argv+=['--reference-episode',str(ref_result.parent),'--object-dir',str(Path(obj).resolve())]
            requires=[str(ref_result),str(Path(obj).resolve()/'aligned.json'),str(Path(obj).resolve()/'physics.json')]
        commands[unit['unit_id']]={'planned_unit_sha256':canonical_hash(unit),'argv':argv,
            'result_relative_path':'runner/episode/result.json','requires_files':requires}
    return commands


def load_acquisition_source(config,slot,source):
    """Validate an earlier acquisition before resuming its exact native instance."""
    from robo.roundtrip.identity import validate_canonical_instance,validate_reset_bank
    source=Path(source).resolve(strict=True)
    run=json.loads((source/'acquisition_manifest.json').read_text())
    if run['config_sha256']!=canonical_hash(config) or run['slot']!=slot:
        raise ValueError('resume source differs from declared acquisition slot/config')
    manifest=json.loads((source/'private/canonical_instance.json').read_text())
    bank=json.loads((source/'private/reset_bank.json').read_text())
    validate_canonical_instance(manifest,source/'private/canonical');validate_reset_bank(bank,manifest)
    return {'root':source,'manifest':manifest,'bank':bank,
        'config':json.loads((source/'native_config.json').read_text()),
        'source_files':{name:sha(source/name) for name in ['acquisition_manifest.json','native_config.json',
            'private/canonical_instance.json','private/reset_bank.json','private/canonical/scene.xml','private/canonical/canonical_state.json']}}


def validate_test_capture_admission(config,path):
    """TEST capture requires frozen DEV outcomes/settings and all task controls."""
    if not path:raise ValueError('TEST capture requires admission receipt')
    receipt=json.loads(Path(path).read_text())
    if receipt.get('kind')!='native_test_admission' or receipt.get('capture_ready') is not True:
        raise ValueError('TEST capture admission is not ready')
    if receipt.get('config_sha256')!=canonical_hash(config):raise ValueError('TEST config differs from admission')
    code=git_snapshot()
    if code['dirty'] or code['commit']!=receipt.get('source_commit'):raise ValueError('TEST source differs from clean admitted commit')
    for item in [receipt['core_status'],*receipt['identity_gates'].values(),*receipt['frozen_inputs']]:
        if sha(item['path'])!=item['sha256']:raise ValueError('TEST admission evidence changed')
    status=json.loads(Path(receipt['core_status']['path']).read_text())
    groups={g['controller_method']:g for g in status['groups']}
    for method in ('REF_NATIVE','B0_FIXED_NATIVE'):
        g=groups.get(method,{})
        if g.get('planned')!=40 or g.get('terminal')!=40 or g.get('unmeasured')!=0:
            raise ValueError('DEV40 core arm is not complete')
    if set(receipt['identity_gates'])!=set(config['tasks']):raise ValueError('missing native task family gate')
    for gate in receipt['identity_gates'].values():
        if json.loads(Path(gate['path']).read_text()).get('passed') is not True:raise ValueError('native task identity gate failed')
    if not receipt['frozen_inputs']:raise ValueError('construction/capture settings must be frozen')
    return receipt


def acquire_instance(config,slot_id,out,*,adapter_factory=None,identity_factory=None,reset_factory=None,camera_pose_reader=None,width=1280,height=720,source_instance=None,test_admission=None,capture_view_plan=None):
    """Generate exactly one declared native instance and its static TRAIN capture.

    This privileged acquisition runs no policy and never chooses by success.
    Constructor access is limited to public/capture_id; canonical data is private.
    """
    import copy
    import numpy as np
    from robo.roundtrip.capture import CaptureSpec,capture_static
    from robo.roundtrip.capture_native import native_camera_poses,smoke_view_plan
    dev_capture=config.get('dev_scope_capture')
    if config.get('split')=='test' and (dev_capture is not None or capture_view_plan is not None):
        raise ValueError('custom capture trajectories are DEV-only; TEST protocol is frozen')
    if (dev_capture is None)!=(capture_view_plan is None):
        raise ValueError('DEV scope capture requires explicit trajectory config and callable')
    if dev_capture is not None:
        if (not isinstance(dev_capture,dict) or dev_capture.get('trajectory_id')!='cabinet_camera_pitch_v1' or
                dev_capture.get('counts')!={'train':12,'dev':0,'test':2} or
                dev_capture.get('scope')!='L1_cabinet_bottom_support_component'):
            raise ValueError('unsupported frozen DEV scope capture protocol')
    slots,_=enumerate_plan(config);matches=[s for s in slots if s['instance_slot_id']==slot_id]
    if len(matches)!=1:raise ValueError('acquisition slot not in frozen plan')
    if config.get('split')=='test':
        validate_test_capture_admission(config,test_admission)
        expected={'trajectory_id':'native_camera_local_translations_v1','counts':{'train':6,'dev':0,'test':2},
                  'width':width,'height':height,'scope':'L0_target_only'}
        if config.get('capture')!=expected:raise ValueError('TEST capture settings differ from supported frozen trajectory')
    slot=matches[0];base=yaml.safe_load(Path(config['base_config']).read_text())
    base=copy.deepcopy(base);base['instance'].update(task_id=slot['task_id'],layout_id=slot['layout_id'],style_id=slot['style_id'],split=slot['split'])
    base['horizon']=slot['native_horizon'];base['reset_seeds']=[slot['generation_seed']]
    # Bind the successful official checkpoint receipt in every new instance.
    base['policy']['checkpoint_receipt_sha256']=config['checkpoint_receipt_sha256']
    reused=load_acquisition_source(config,slot,source_instance) if source_instance is not None else None
    if reused is not None and reused['config']!=base:raise ValueError('resumed native configuration differs')
    if adapter_factory is None:
        from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
        adapter_factory=RoboCasaAdapter
    if identity_factory is None or reset_factory is None:
        from robo.roundtrip.identity import create_canonical_instance,create_reset_bank
        identity_factory=identity_factory or create_canonical_instance;reset_factory=reset_factory or create_reset_bank
    code=git_snapshot()
    if code.get('dirty') is not False or code.get('commit')=='nogit':raise ValueError('acquisition requires clean committed source')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    save_new(out/'acquisition_manifest.json',{'config':config,'config_sha256':canonical_hash(config),'slot':slot,'source_code':code,'policy_invoked':False,'reuse_source':str(reused['root']) if reused else None,'reuse_source_files':reused['source_files'] if reused else None})
    save_new(out/'native_config.json',base);started=time.monotonic();adapter=None
    try:
        adapter=adapter_factory(base);adapter.reset_from_spec({'seed':slot['generation_seed']})
        private=out/'private';private.mkdir(mode=0o700);bundle=private/'canonical'
        if reused is None:
            adapter.export_reference_for_evaluator(bundle)
            instance=identity_factory(bundle,base)
        else:
            import shutil
            bundle.mkdir(mode=0o700)
            for name in ('scene.xml','canonical_state.json','body_bindings.json'):
                source=reused['root']/'private/canonical'/name
                if source.exists():shutil.copyfile(source,bundle/name)
            instance=reused['manifest']
        save_new(private/'canonical_instance.json',instance)
        perturbations=[]
        for r in config['reset_perturbations']:
            delta=np.eye(4);delta[:3,3]=r['translation_world_m']
            perturbations.append({'reset_id':r['reset_id'],'delta_world':delta.tolist(),'policy_rng_seed':r['policy_rng_seed']})
        bank=reset_factory(instance,perturbations)
        if reused is not None and bank!=reused['bank']:raise ValueError('resumed reset bank differs')
        save_new(private/'reset_bank.json',bank)
        # Native reset may leave fixture-derived transforms stale. The official
        # XML/state restore refreshes them while restoring integration warmstart.
        # Never change static tolerances or regenerate a failed native instance.
        state=json.loads((bundle/'canonical_state.json').read_text())
        adapter.import_xml((bundle/'scene.xml').read_text(),canonical_state=state)
        from robo.roundtrip.capture_native import _check_canonical_restore
        _check_canonical_restore(adapter,state)
        if not np.array_equal(adapter.get_state()['integration_state'],state['integration_state']):
            raise ValueError('canonical capture restore changed integration state')
        trajectory='native_camera_local_translations_v1' if dev_capture is None else dev_capture['trajectory_id']
        counts={'train':6,'dev':0,'test':2} if dev_capture is None else dev_capture['counts']
        capture_id='c-'+canonical_hash({'instance':instance['canonical_instance_id'],'trajectory':trajectory,'width':width,'height':height})[:16]
        frames=(capture_view_plan or smoke_view_plan)((camera_pose_reader or native_camera_poses)(adapter))
        capture=capture_static(adapter,spec=CaptureSpec(capture_id=capture_id,width=width,height=height,counts=counts,robot_mode='parked'),
            frames=frames,public_out=out/'public',vault=private/'capture',robot_config={'robot':base['robot'],'action_convention':'native RoboCasa configured controller'},
            task_instruction=adapter.get_policy_observation()['annotation.human.task_description'])
        public_manifest=Path(capture['public'])/'capture_manifest.json'
        binding={'instance_slot_id':slot_id,'canonical_instance_id':instance['canonical_instance_id'],
            'canonical_manifest':str((private/'canonical_instance.json').resolve()),'bundle_dir':str(bundle.resolve()),
            'reset_bank':str((private/'reset_bank.json').resolve()),'capture_public':str(Path(capture['public']).resolve()),
            'capture_manifest_sha256':sha(public_manifest),'native_config':str((out/'native_config.json').resolve()),
            'cohort_id':config['cohort_id'],'elapsed_s':time.monotonic()-started,'scope':'L0_target_only',
            'capture_scope':('6TRAIN/2heldout target DEV trajectory; not whole-room coverage' if dev_capture is None else dev_capture['scope']),
            'trajectory_uses_hidden_object_poses':False,'camera_assistance':'known native camera poses; privileged acquisition aid',
            'policy_invoked':False,'reuse_source':str(reused['root']) if reused else None,
            'canonical_cache_refresh':'official import_xml preserving exact integration state'}
        if dev_capture is not None:
            binding.update(dev_scope_capture=dev_capture,scope='L1_cabinet_bottom_support_component',
                frame_plan_sha256=canonical_hash(frames),whole_cabinet_coverage='NOT_ESTABLISHED')
        if reused is not None:
            for name,digest in reused['source_files'].items():
                if sha(reused['root']/name)!=digest:raise ValueError('resume source changed during capture')
        if git_snapshot()!=code:raise ValueError('source changed during acquisition')
        save_new(out/'binding.json',binding);return binding
    except Exception as exc:
        save_new(out/'acquisition_failure.json',{'error_type':type(exc).__name__,'reason':str(exc),'policy_invoked':False,'elapsed_s':time.monotonic()-started});raise
    finally:
        if adapter is not None:adapter.close()


def inventory_pilot(pilot,config):
    """Read all declared original references, including every policy failure."""
    from robo.roundtrip.paired import load_reference_bundle
    rows=[];seen=set()
    for seed in config['reset_seeds']:
        b=load_reference_bundle(Path(pilot)/f'episode_seed{seed}',Path(pilot)/f'canonical_seed{seed}',config=config,reset_seed=seed)
        metadata=b['state']['native_metadata'];identity=canonical_hash({'xml_sha256':b['provenance']['canonical_files']['scene.xml'],
             'canonical_state_semantic_sha256':b['provenance']['canonical_state_semantic_sha256']})
        if identity in seen:raise ValueError('original pilot has duplicate canonical instance')
        seen.add(identity)
        rows.append({'original_seed':seed,'canonical_bundle_id':'legacy-'+identity,
            'canonical_reference':b['provenance']['canonical_reference'],'reference_episode':b['provenance']['reference_episode'],
            'provenance':b['provenance'],'native_success':b['result']['success'],'ticks':b['result']['ticks'],
            'instruction':metadata['lang'],'objects':[{'role':o['name'],'info':o.get('info')} for o in metadata['object_cfgs']],
            'asset_training_split':[o.get('info',{}).get('split','unknown') for o in metadata['object_cfgs']],
            'capture_reuse_rule':'exact XML/state/capture binding required; same integer seed alone insufficient',
            'capture_required':seed!=0,'existing_seed0_capture_bound':seed==0})
    return rows


def validate_terminal(unit,terminal):
    if terminal.get('unit_id')!=unit['unit_id'] or terminal.get('planned_unit_sha256')!=canonical_hash(unit):
        raise ValueError('terminal shard identity/config drift')
    for key in IDENTITY_FIELDS:
        if terminal.get(key)!=unit[key]:raise ValueError('terminal identity mismatch: '+key)
    status=terminal.get('terminal_status');executed=terminal.get('executed');success=terminal.get('success')
    if status not in ('RECORDED','BUILD_FAILED','ABSTAINED','ENVIRONMENT_FAILED','CODE_FAILED','RESOURCE_FAILED'):
        raise ValueError('unknown terminal status')
    if status=='RECORDED':
        if executed is not True or not isinstance(success,bool):raise ValueError('recorded result requires execution and native boolean')
        result=terminal.get('result',{})
        if result.get('executed')!=executed or result.get('success')!=success:raise ValueError('canonical result outcome mismatch')
        if unit.get('config_sha256') and result.get('config_sha256')!=unit['config_sha256']:raise ValueError('canonical runner config differs from planned unit')
        if terminal.get('result_path') and sha(terminal['result_path'])!=terminal['result_sha256']:raise ValueError('result bytes changed')
    elif executed is not False or success is not None:
        raise ValueError('nonexecuted failure must not invent a native policy success/failure')
    return terminal


def adjudicate_collision_failures(shards, shard_paths, manifest_path):
    """Correct proven artifact failures without changing original worker shards.

    This narrow post-run classification does not repair an asset or infer a
    native outcome. Valid meshes, external failures, and executed trials cannot
    be admitted by this path.
    """
    import inspect
    import re
    import numpy as np
    import trimesh
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema_version')!=1 or manifest.get('kind')!='native_collision_failure_adjudication':
        raise ValueError('unsupported terminal adjudication contract')
    if manifest.get('trimesh_version')!=trimesh.__version__:
        raise ValueError('adjudication must use the frozen importer trimesh version')
    def bound(record):
        if (not isinstance(record,dict) or not isinstance(record.get('path'),str)
                or not re.fullmatch('[0-9a-f]{64}',str(record.get('sha256','')))):
            raise ValueError('adjudication evidence is unbound')
        path=Path(record['path']).resolve()
        if sha(path)!=record['sha256']:raise ValueError('adjudication evidence changed')
        return path
    audit=bound(manifest['independent_audit']);audit_record=json.loads(audit.read_text())
    required={'schema_version':1,'kind':'independent_original_collision_import_audit',
              'classification':'METHOD_CONSTRUCTION_IMPORT_VALIDITY_FAILURE',
              'adjudication_supported':True,'source_dirty':False,'geometry_repaired':False,
              'threshold_changed':False,'new_policy_outcomes':0,
              'proposed_status':'BUILD_FAILED','proposed_executed':False,'proposed_success':None}
    if any(key not in audit_record or audit_record[key]!=value for key,value in required.items()):
        raise ValueError('unsupported independent collision audit conclusion')
    source=audit_record.get('source_commit','')
    if not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('independent audit source is unbound')
    if (audit_record.get('trimesh_version')!=trimesh.__version__ or
            audit_record.get('numpy_version')!=np.__version__ or
            Path(audit_record.get('python','')).resolve()!=Path(sys.executable).resolve()):
        raise ValueError('independent audit runtime differs from importer runtime')
    convex_path=bound(audit_record.get('trimesh_convex_source'))
    if (convex_path!=Path(inspect.getsourcefile(trimesh.convex)).resolve()):
        raise ValueError('independent audit convex implementation differs')
    importer_path=bound(audit_record.get('importer_source'))
    try:
        repo=Path(subprocess.check_output(['git','-C',str(importer_path.parent),'rev-parse','--show-toplevel'],text=True).strip())
        relative=importer_path.relative_to(repo).as_posix()
        if relative!='robo/roundtrip/importers/robocasa.py':raise ValueError('wrong importer source path')
        committed=subprocess.check_output(['git','-C',str(repo),'show',source+':'+relative])
    except (subprocess.CalledProcessError,ValueError) as exc:
        raise ValueError('independent audit importer/source commit is unavailable') from exc
    if hashlib.sha256(committed).hexdigest()!=audit_record['importer_source']['sha256']:
        raise ValueError('independent audit importer does not match original source commit')
    build_path=bound(audit_record.get('build_manifest'));build=json.loads(build_path.read_text())
    if build.get('canonical_instance_id')!=audit_record.get('canonical_instance_id'):
        raise ValueError('independent audit construction canonical differs')
    config_path=bound(audit_record.get('constructor_config'))
    if (config_path!=build_path.parent/'build_config.json' or
            build.get('source_hashes',{}).get('build_config.json')!=audit_record['constructor_config']['sha256'] or
            canonical_hash(json.loads(config_path.read_text()))!=build.get('config_sha256')):
        raise ValueError('independent audit construction config differs')
    audited_part=bound(audit_record.get('collision_part'))
    try:relative_part=audited_part.relative_to(build_path.parent).as_posix()
    except ValueError as exc:raise ValueError('collision is outside original construction') from exc
    if (build.get('source_hashes',{}).get(relative_part)!=audit_record['collision_part']['sha256'] or
            audit_record.get('construction_source_hashes_verified',{}).get(str(audited_part))!=audit_record['collision_part']['sha256']):
        raise ValueError('collision bytes are not bound to original construction')
    audited_units=audit_record.get('units',[])
    audit_by_id={row['unit_id']:row for row in audited_units}
    if len(audit_by_id)!=len(audited_units):raise ValueError('duplicate independently audited unit')
    seen=set();result=dict(shards)
    for entry in manifest['entries']:
        uid=entry['unit_id']
        if uid in seen or uid not in shards:raise ValueError('duplicate or unplanned adjudication')
        seen.add(uid);original=shards[uid]
        unit_audit=audit_by_id.get(uid)
        if (unit_audit is None or unit_audit.get('original_status')!='CODE_FAILED' or
                unit_audit.get('proposed_status')!='BUILD_FAILED' or
                unit_audit.get('executed') is not False or unit_audit.get('success') is not None):
            raise ValueError('unit lacks supported independent adjudication')
        if (original.get('source_code',{}).get('commit')!=source or
                original.get('source_code',{}).get('dirty') is not False or
                original.get('canonical_instance_id')!=audit_record['canonical_instance_id'] or
                original.get('reset_id')!=unit_audit.get('reset_id')):
            raise ValueError('original terminal source/canonical/reset differs from audit')
        terminal_path=bound(entry['original_terminal'])
        if terminal_path!=shard_paths[uid].resolve():raise ValueError('adjudication original shard differs')
        if (original['terminal_status']!='CODE_FAILED' or original['executed'] is not False
                or original['success'] is not None or original.get('result') is not None
                or original.get('returncode')!=1):
            raise ValueError('only unexecuted generic import errors may be adjudicated')
        log=bound(entry['worker_log']);part=bound(entry['collision_part'])
        if (bound(unit_audit.get('terminal'))!=terminal_path or
                bound(unit_audit.get('worker_log'))!=log or part!=audited_part or
                entry['collision_part']['sha256']!=audit_record['collision_part']['sha256']):
            raise ValueError('adjudication entry differs from independently audited evidence')
        planned_path=bound(unit_audit.get('planned_unit'));planned=json.loads(planned_path.read_text())
        unit_config=bound(unit_audit.get('config'))
        argv=original.get('argv',[])
        if (planned_path!=terminal_path.parent/'planned_unit.json' or
                canonical_hash(planned)!=original.get('planned_unit_sha256') or
                Path(planned.get('config_path','')).resolve()!=unit_config or
                canonical_hash(json.loads(unit_config.read_text()))!=planned.get('config_sha256') or
                planned.get('config_sha256')!=unit_audit.get('config_canonical_sha256') or
                argv.count('--config')!=1 or
                Path(argv[argv.index('--config')+1]).resolve()!=unit_config):
            raise ValueError('adjudication planned config differs from original runner')
        if log!=terminal_path.parent/'worker.log':raise ValueError('adjudication worker log differs')
        argv=original.get('argv',[])
        if argv.count('--object-dir')!=1:raise ValueError('original object directory is not bound')
        directory=Path(argv[argv.index('--object-dir')+1]).resolve()
        if directory!=Path(audit_record.get('object_dir','')).resolve():raise ValueError('audit object directory differs')
        if part.parent!=directory/'collision' or not part.name.startswith('part_') or part.suffix!='.obj':
            raise ValueError('adjudicated collision is not the original supplied asset')
        expected=f'ValueError: collision part is not a closed convex volume: {part.name}'
        if audit_record.get('exact_importer_exception')!=expected:raise ValueError('independent audit error differs')
        if expected not in log.read_text().splitlines():raise ValueError('original import failure not established')
        if original.get('result_path') or Path(original.get('result_path_planned', terminal_path.parent/'runner/episode/result.json')).exists():
            raise ValueError('adjudication cannot replace an existing policy result')
        mesh=trimesh.load(part,process=False,force='mesh')
        if mesh.is_watertight and mesh.is_convex and mesh.volume>0:
            raise ValueError('original collision passes the unchanged importer predicate')
        proof={'manifest':{'path':str(manifest_path),'sha256':sha(manifest_path)},
               'independent_audit':{'path':str(audit),'sha256':sha(audit)},
               **{k:entry[k] for k in ('original_terminal','worker_log','collision_part')}}
        result[uid]={**original,'terminal_status':'BUILD_FAILED','original_terminal_status':'CODE_FAILED',
                     'failure_classification':'invalid_supplied_collision_volume',
                     'failure':'Supplied construction collision fails unchanged native import validity; no policy executed.',
                     'classification_adjudication':proof}
    if not seen:raise ValueError('empty adjudication is not a classification audit')
    return result


def collect(planned,shard_root,out,*,adjudications=None):
    """One exclusive merger, all planned units retained; missing outcomes null."""
    by_id={u['unit_id']:u for u in planned}
    if len(by_id)!=len(planned):raise ValueError('duplicate planned unit')
    shards={};shard_paths={}
    for path in sorted(Path(shard_root).glob('*/terminal.json')):
        terminal=json.loads(path.read_text());uid=terminal.get('unit_id')
        if uid not in by_id or uid in shards:raise ValueError('unexpected or duplicate terminal shard')
        shards[uid]=validate_terminal(by_id[uid],terminal);shard_paths[uid]=path
    if adjudications:
        shards=adjudicate_collision_failures(shards,shard_paths,adjudications)
        for uid,terminal in shards.items():validate_terminal(by_id[uid],terminal)
    rows=[]
    for unit in planned:
        t=shards.get(unit['unit_id']);rows.append({**unit,**(t or {'terminal_status':'NOT_SCHEDULED','executed':None,'success':None})})
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    save_new(out/'episode_ledger.jsonl',rows,jsonl=True)
    groups=[]
    for method in sorted({u['controller_method'] for u in planned}):
        selected=[r for r in rows if r['controller_method']==method];n=len(selected)
        terminal=sum(r['terminal_status']!='NOT_SCHEDULED' for r in selected);measured=sum(r['terminal_status'] in ('RECORDED','BUILD_FAILED','ABSTAINED') for r in selected);executed=sum(r['executed'] is True for r in selected);success=sum(r['success'] is True for r in selected)
        groups.append({'controller_method':method,'planned':n,'terminal':terminal,'unmeasured':n-measured,
            'executed':executed,'successes':success,'success_per_planned':success/n if measured==n else None,
            'executed_per_planned':executed/n,'success_per_executed':success/executed if executed else None})
    with (out/'coverage.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(groups[0]) if groups else ['controller_method']);writer.writeheader();writer.writerows(groups)
    save_new(out/'failed_units.jsonl',[r for r in rows if r['terminal_status'] not in ('RECORDED','NOT_SCHEDULED') or r['success'] is False],jsonl=True)
    save_new(out/'status.json',{'planned':len(rows),'terminal':len(shards),'complete':len(shards)==len(rows),'groups':groups})
    if adjudications:
        save_new(out/'adjudication_receipt.json',{'manifest_path':str(Path(adjudications).resolve()),
            'manifest_sha256':sha(adjudications),'original_shards_modified':False,
            'classified_units':sum('classification_adjudication' in r for r in rows),
            'native_success_values_changed':False})
    return rows


def worker(unit,argv,out,*,result_relative_path="episode/result.json"):
    """Execute a predeclared existing runner command in one private unit shard."""
    if not isinstance(argv,list) or not argv or any(not isinstance(v,str) for v in argv):raise ValueError('explicit argv required')
    out=Path(out)
    result_relative_path=Path(result_relative_path)
    if result_relative_path.is_absolute() or '..' in result_relative_path.parts:raise ValueError('result path must stay in worker shard')
    out.mkdir(parents=True,exist_ok=False)
    save_new(out/'planned_unit.json',unit);started=time.monotonic()
    with (out/'worker.log').open('x') as log:
        completed=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=False)
    result_path=out/result_relative_path;result=json.loads(result_path.read_text()) if result_path.exists() else None
    receipt={**{k:unit[k] for k in IDENTITY_FIELDS},'unit_id':unit['unit_id'],'planned_unit_sha256':canonical_hash(unit),
        'argv':argv,'source_code':git_snapshot(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'returncode':completed.returncode,'elapsed_s':time.monotonic()-started}
    if result is not None and result.get('executed') is True and isinstance(result.get('success'),bool):
        receipt.update(terminal_status='RECORDED',executed=True,success=result['success'],result=result,
                       result_path=str(result_path.resolve()),result_sha256=sha(result_path))
    else:
        # No policy outcome is inferred from a failed process or absent result.
        receipt.update(terminal_status='CODE_FAILED',executed=False,success=None,result=result,
                       failure='runner failed or did not produce a canonical executed result; inspect worker.log')
    validate_terminal(unit,receipt);save_new(out/'terminal.json',receipt)
    return receipt


def run_instance(planned,commands,*,instance_id,method,out,expected_resets=5,scope=None):
    """One bounded instance/arm shard, sequential native episodes, warm policy remote."""
    selected=[u for u in planned if u['canonical_instance_id']==instance_id and u['controller_method']==method and (scope is None or u['scope']==scope)]
    if len(selected)!=expected_resets or len({u['reset_id'] for u in selected})!=expected_resets:
        raise ValueError('instance worker requires exactly the declared reset roster')
    out=Path(out).resolve();completed=[];pending=[]
    for unit in selected:
        uid=unit['unit_id'];shard=out/uid
        if Path(unit['result_path_planned']).resolve()!=shard/'runner/episode/result.json':
            raise ValueError('worker root differs from frozen result paths')
        if (shard/'terminal.json').exists():
            completed.append(validate_terminal(unit,json.loads((shard/'terminal.json').read_text())));continue
        command=commands.get(uid)
        if command is None or any(not Path(p).is_file() for p in command.get('requires_files',[])):
            pending.append(uid);continue
        if command['planned_unit_sha256']!=canonical_hash(unit):raise ValueError('instance command binding differs')
        completed.append(worker(unit,command['argv'],shard,result_relative_path=command.get('result_relative_path','runner/episode/result.json')))
    return {'canonical_instance_id':instance_id,'controller_method':method,'planned':expected_resets,
        'terminal':len(completed),'pending_unit_ids':pending,'executed':sum(t['executed'] is True for t in completed),
        'successes':sum(t['success'] is True for t in completed)}


def dispatch(planned,commands,out,*,sbatch_args,max_jobs):
    """Bounded ordinary jobs; unique submission intent prevents duplicate resume."""
    if max_jobs<=0:raise ValueError('positive bounded submission limit required')
    if any('--array' in a or a=='-a' or a.startswith('-a') and not a.startswith('--') for a in sbatch_args):raise ValueError('job arrays are prohibited')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);submitted=[]
    for unit in planned:
        uid=unit['unit_id'];shard=out/'workers'/uid;receipt=out/'dispatch'/f'{uid}.json'
        if (shard/'terminal.json').exists():validate_terminal(unit,json.loads((shard/'terminal.json').read_text()));continue
        if receipt.exists():continue  # unknown scheduler state is not permission to duplicate
        if uid not in commands:continue
        command=commands[uid]
        if command.get('planned_unit_sha256')!=canonical_hash(unit):raise ValueError('launch command unit binding mismatch')
        if any(not Path(p).is_file() for p in command.get('requires_files',[])):continue
        save_new(receipt,{'unit_id':uid,'state':'SUBMITTING','command':command,'source_code':git_snapshot()})
        upath=out/'dispatch'/f'{uid}.unit.json';save_new(upath,unit)
        cpath=out/'dispatch'/f'{uid}.command.json';save_new(cpath,command['argv'])
        script=out/'dispatch'/f'{uid}.sh'
        with script.open('x') as f:f.write('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join([sys.executable,'-m','robo.roundtrip.matrix','--phase','worker','--unit',str(upath.resolve()),'--command',str(cpath.resolve()),'--out',str(shard.resolve()),'--result-relative-path',command.get('result_relative_path','episode/result.json')])+'\n')
        proc=subprocess.run(['sbatch','--parsable',*sbatch_args,str(script.resolve())],capture_output=True,text=True)
        terminal={'unit_id':uid,'state':'SUBMITTED' if proc.returncode==0 else 'SUBMIT_FAILED','job_id':proc.stdout.strip().split(';')[0] if proc.returncode==0 else None,'stderr':proc.stderr,'argv':['sbatch','--parsable',*sbatch_args,str(script.resolve())]}
        save_new(out/'dispatch'/f'{uid}.submitted.json',terminal);submitted.append(terminal)
        if len(submitted)>=max_jobs:break
    return submitted


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',required=True,choices=['dry-run','resolve','inventory-pilot','dispatch','collect','status','worker','acquire','commands','run-instance','extend'])
    ap.add_argument('--adjudications',help='Immutable evidence for proven unexecuted collision-asset failures; collect only')
    ap.add_argument('--admission');ap.add_argument('--method',action='append',default=[]);ap.add_argument('--instance-id');ap.add_argument('--controller-method');ap.add_argument('--expected-resets',type=int,default=5);ap.add_argument('--source-instance');ap.add_argument('--worker-root');ap.add_argument('--host',default='localhost');ap.add_argument('--port',type=int,default=8017);ap.add_argument('--builds');ap.add_argument('--slot-id');ap.add_argument('--width',type=int,default=1280);ap.add_argument('--height',type=int,default=720);ap.add_argument('--config');ap.add_argument('--out',required=True);ap.add_argument('--bindings');ap.add_argument('--planned');ap.add_argument('--shards');ap.add_argument('--pilot');ap.add_argument('--commands');ap.add_argument('--max-jobs',type=int,default=1);ap.add_argument('--sbatch-arg',action='append',default=[]);ap.add_argument('--unit');ap.add_argument('--command');ap.add_argument('--result-relative-path',default='episode/result.json')
    a=ap.parse_args(argv);out=Path(a.out)
    if a.phase=='extend':
        result,_=extend_methods(read_rows(a.planned),a.method,read_rows(a.bindings),out,worker_root=a.worker_root,builds=read_rows(a.builds) if a.builds else (),host=a.host,port=a.port)
    elif a.phase=='commands':
        result=commands_for_units(read_rows(a.planned),read_rows(a.builds) if a.builds else (),host=a.host,port=a.port);save_new(out/'commands.json',result)
    elif a.phase=='run-instance':result=run_instance(read_rows(a.planned),json.loads(Path(a.commands).read_text()),instance_id=a.instance_id,method=a.controller_method,out=out,expected_resets=a.expected_resets)
    elif a.phase=='worker':result=worker(json.loads(Path(a.unit).read_text()),json.loads(Path(a.command).read_text()),out,result_relative_path=a.result_relative_path)
    elif a.phase in ('collect','status'):
        result=collect(read_rows(a.planned),a.shards,out,adjudications=a.adjudications)
    elif a.phase=='dispatch':result=dispatch(read_rows(a.planned),json.loads(Path(a.commands).read_text()),out,sbatch_args=a.sbatch_arg,max_jobs=a.max_jobs)
    elif a.phase=='acquire':result=acquire_instance(yaml.safe_load(Path(a.config).read_text()),a.slot_id,out,width=a.width,height=a.height,source_instance=a.source_instance,test_admission=a.admission)
    else:
        config=yaml.safe_load(Path(a.config).read_text())
        out.mkdir(parents=True,exist_ok=False)
        if a.phase=='inventory-pilot':result=inventory_pilot(a.pilot,config);save_new(out/'original_pilot_instances.jsonl',result,jsonl=True)
        elif a.phase=='dry-run':
            slots,units=enumerate_plan(config);save_new(out/'instance_slots.jsonl',slots,jsonl=True);save_new(out/'planned_slots.jsonl',units,jsonl=True)
            result={'canonical_instances_planned':len(slots),'episodes_planned':len(units),'resolved_instances':0,'execution_ready':False};save_new(out/'cohort_manifest.json',{'config':config,'config_sha256':canonical_hash(config),**result})
        else:
            bindings=read_rows(a.bindings);result=resolve_plan(config,bindings)
            if a.worker_root:
                result,_=write_resolved_configs(result,bindings,out,worker_root=a.worker_root,host=a.host,port=a.port,builds=read_rows(a.builds) if a.builds else (),test_admission=json.loads(Path(a.admission).read_text()) if a.admission else None)
            else:save_new(out/'planned_units.jsonl',result,jsonl=True)
    print(json.dumps({'phase':a.phase,'output':str(out),'rows':len(result) if isinstance(result,list) else result}));return 0

if __name__=='__main__':raise SystemExit(main())
