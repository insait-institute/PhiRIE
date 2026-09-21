"""Versioned L1 collision repair with unchanged target and explicit native context.

The preserved B3 destination is never overwritten. Diagnostic settling restores
all integration bytes. No task success, native target pose or native shape is an
optimization input. This module does not run a policy or certify task retention.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
from robo.manifest.hash import canonical_hash, git_snapshot
from robo.roundtrip.identity import file_hash
from robo.roundtrip.scope_bundle import artifact_hashes
from robo.roundtrip.system_verification import lineage_alternative
from agents.orchestrator.job_graph import descendants, validate_ledger

CONFIG = dict(schema_version=1, tier='DEV', scope='L1_target_destination',
    max_calls_per_dependency=2, settle_seconds=.5, max_drift_m=.02,
    max_rotation_deg=15., max_penetration_m=.005,
    bm_order=['reselect_candidate','regenerate_collision'],
    regeneration='per_part_convex_hull_no_whole_object_hull_v1',
    action_arguments='current_destination_artifact; other initial generator relative to original B3 lineage')
TEST_CONFIG=dict(CONFIG,schema_version=2,tier='TEST')
METHODS={'B4':'B4_ROOM_REPAIR_NATIVE','BM':'BM_BUDGET_MATCHED_NATIVE'}
EDGES=[('destination','scope_import'),('scope_import','native_contacts'),
       ('scope_import','retained_body_settle'),('native_contacts','acceptance'),
       ('retained_body_settle','acceptance')]


def validate_config(config):
    expected=TEST_CONFIG if config.get('schema_version')==2 else CONFIG
    if canonical_hash(config)!=canonical_hash(expected):
        raise ValueError('L1 version/tier/action bank/settings are frozen; TEST requires schema2')


def _json(path):return json.loads(Path(path).read_text())
def _write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def _source(path):return dict(path=str(Path(path).resolve()),sha256=file_hash(path))
def _checked(source):
    if file_hash(source['path'])!=source['sha256']:raise ValueError('source receipt changed')
    return _json(source['path'])
def _roles(bundle):
    entities=bundle['entities']
    if len(entities)!=2 or {e['object_id'] for e in entities}!={'target','destination'}:
        raise ValueError('L1 requires exactly target and destination identities')
    return {e['object_id']:e for e in entities}
def _closure(entity):
    hashes=artifact_hashes(entity['object_dir'])
    if hashes!=entity['artifact_hashes']:raise ValueError('scope artifact closure changed')
    return hashes

def next_action(method, measured, used):
    if method not in METHODS:raise ValueError('unknown scope method')
    if method=='BM':return next((a for a in CONFIG['bm_order'] if a not in used),None)
    if measured['passed']:return None
    order=(['regenerate_collision','reselect_candidate'] if 'invalid_collision' in measured['reason_codes']
           else ['reselect_candidate','regenerate_collision'])
    return next((a for a in order if a not in used),None)


def regenerate_collision(source,out):
    """Hull each supplied part independently; never bridge the cavity globally."""
    import trimesh
    source,out=Path(source),Path(out)
    before=artifact_hashes(source)
    shutil.copytree(source,out)
    parts=[]
    for name in sorted(k for k in before if k.startswith('collision/')):
        mesh=trimesh.load(source/name,force='mesh',process=False)
        if not np.isfinite(mesh.vertices).all() or len(mesh.vertices)<4:
            raise ValueError('collision regeneration requires finite volumetric input')
        hull=mesh.convex_hull
        hull.export(out/name)
        actual=trimesh.load(out/name,force='mesh',process=False)
        if not actual.is_watertight or not actual.is_convex or actual.volume<=0:
            raise ValueError('regenerated exported part still fails strict convexity')
        parts.append(dict(part=name,input_sha256=before[name],output_sha256=file_hash(out/name),
            input_volume=float(mesh.volume),output_volume=float(actual.volume),
            input_is_convex=bool(mesh.is_convex),output_is_convex=bool(actual.is_convex)))
    after=artifact_hashes(out)
    if any(before[k]!=after[k] for k in ('aligned.json','physics.json','mesh_sim.obj')):
        raise ValueError('collision action altered visual/pose/physics prior')
    if before==after:raise ValueError('collision action produced no new artifact')
    return dict(source_artifact_hashes=before,result_artifact_hashes=after,parts=parts,
                surface_preservation='NOT_ASSUMED',metadata_correction=False,whole_object_hull=False)


def measure_scope(xml,state,entities,config,*,scope=None):
    """Compile actual exports, sample all substep contacts and restore reset."""
    import mujoco
    from robo.roundtrip.scope import import_scope
    from robo.roundtrip.scope_contacts import bind_contact_inventory,sample_contacts
    from robo.roundtrip.adapters.integration_state import remap_integration_state
    validate_config(config)
    scope=scope or config['scope']
    if scope not in ('L0_target_only',config['scope']):raise ValueError('unsupported diagnostic scope')
    try:imported,manifest=import_scope(xml,entities=entities,scope=scope)
    except ValueError as exc:
        if 'collision part is not a closed convex volume' not in str(exc):raise
        return dict(passed=False,reason_codes=['invalid_collision'],import_error=str(exc),
                    native_policy_success=None,contacts=None,state_restored_byte_exact=True),None
    return measure_imported_scope(xml,state,imported,manifest,config)


def measure_imported_scope(xml,state,imported,manifest,config):
    """Evaluate a sealed imported scene with the existing unchanged settle bank."""
    import mujoco
    from robo.roundtrip.scope_contacts import bind_contact_inventory,sample_contacts
    from robo.roundtrip.adapters.integration_state import remap_integration_state
    validate_config(config)
    scope=manifest['scope']
    if scope=='DEV_partial_observed_workspace' and config['tier']!='DEV':
        raise ValueError('partial observed workspace is DEV engineering only')
    if manifest['source_xml_sha256']!=__import__('hashlib').sha256(xml.encode()).hexdigest() or manifest['imported_xml_sha256']!=__import__('hashlib').sha256(imported.encode()).hexdigest():
        raise ValueError('sealed scope XML hash differs')
    source=mujoco.MjModel.from_xml_string(xml);model=mujoco.MjModel.from_xml_string(imported)
    data=mujoco.MjData(model);kind=mujoco.mjtState.mjSTATE_INTEGRATION
    vector=remap_integration_state(source,model,state['integration_state'])
    mujoco.mj_setState(model,data,vector,kind)
    for name,pose in manifest['replaced_joints'].items():
        adr=model.jnt_qposadr[model.joint(name).id];data.qpos[adr:adr+7]=pose
    saved=np.empty(mujoco.mj_stateSize(model,kind));mujoco.mj_getState(model,data,saved,kind)
    inventory=bind_contact_inventory(model,manifest['entities'],manifest.get('workspace_bounds_world_m'))
    inventory['sampling']='every MuJoCo substep during bounded diagnostic settle, including initial forward'
    replaced={r['body_name'] for r in manifest['entities']}
    body_ids=sorted({int(model.jnt_bodyid[i]) for i in range(model.njnt)
                     if model.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE})
    rows=[];trajectory=[];reasons=[];all_native_rows=[]
    try:
        mujoco.mj_forward(model,data)
        start={b:data.xpos[b].copy() for b in body_ids};quats={b:data.xquat[b].copy() for b in body_ids}
        max_drift={b:0. for b in body_ids};max_rotation={b:0. for b in body_ids}
        steps=max(1,round(config['settle_seconds']/model.opt.timestep))
        for step in range(steps+1):
            contacts=sample_contacts(model,data,inventory)
            for c in contacts['contacts']:rows.append(dict(step=step,**c))
            if scope=='DEV_partial_observed_workspace':
                all_native_rows.extend(dict(step=step,**c) for c in sample_contacts(model,data,inventory,include_all=True)['contacts'])
            trajectory.append(dict(step=step,positions={model.body(b).name:data.xpos[b].tolist() for b in body_ids}))
            for b in body_ids:
                max_drift[b]=max(max_drift[b],float(np.linalg.norm(data.xpos[b]-start[b])))
                angle=float(np.degrees(2*np.arccos(np.clip(abs(np.dot(quats[b],data.xquat[b])),0,1))))
                max_rotation[b]=max(max_rotation[b],angle)
            if step<steps:mujoco.mj_step(model,data)
        finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        penetration=max((max(0.,-r['distance_m']) for r in rows),default=0.)
        bodies=[dict(body_name=model.body(b).name,classification='reconstructed' if model.body(b).name in replaced else 'retained_reference',
                     max_drift_m=max_drift[b],max_rotation_deg=max_rotation[b]) for b in body_ids]
        if not finite:reasons.append('nonfinite_after_settle')
        if penetration>config['max_penetration_m']:reasons.append('scope_penetration')
        if any(r['max_drift_m']>config['max_drift_m'] for r in bodies):reasons.append('scope_settle_translation')
        if any(r['max_rotation_deg']>config['max_rotation_deg'] for r in bodies):reasons.append('scope_settle_rotation')
        report=dict(passed=not reasons,reason_codes=reasons,engine='mujoco',engine_version=mujoco.__version__,
            oracle_context=True,context_scope=scope+'_retained_room',finite=finite,
            contacts=rows,max_penetration_m=penetration,bodies=bodies,trajectory=trajectory,
            settle_steps=steps,timestep_s=float(model.opt.timestep),native_policy_success=None,
            independent_fidelity='NOT_RUN',robot_IK='NOT_RUN',destination_accessibility='NOT_CERTIFIED',
            integration_mapping='robo.roundtrip.adapters.integration_state.remap_integration_state')
        if scope=='DEV_partial_observed_workspace':
            report.update(all_native_substep_contacts=all_native_rows,
                outside_workspace_contact_events=sum(r['outside_declared_workspace'] is True for r in all_native_rows),
                absence_certification='only sampled native integrator states in this bounded settle; not a policy trajectory')
    finally:
        mujoco.mj_setState(model,data,saved,kind);mujoco.mj_forward(model,data);mujoco.mj_setState(model,data,saved,kind)
    restored=np.empty_like(saved);mujoco.mj_getState(model,data,restored,kind)
    report['state_restored_byte_exact']=bool(np.array_equal(saved,restored))
    if not report['state_restored_byte_exact']:raise RuntimeError('scope probe reset restoration failed')
    return report,manifest


def validate_scope_repair(path, *, original_scope_bundle, config):
    """Validate an accepted repair independent of a fresh baseline episode path."""
    receipt=_json(path);original=(_json(original_scope_bundle) if isinstance(original_scope_bundle,(str,Path)) else original_scope_bundle)
    bank=receipt['action_bank'];validate_config(bank['config'])
    if (bank['config_sha256']!=canonical_hash(bank['config']) or bank['max_calls_per_dependency']!=2 or
        receipt.get('schema_version')!=1 or receipt.get('tier')!=bank['config']['tier'] or receipt.get('scope')!=CONFIG['scope'] or
        receipt.get('method')!=config.get('controller_method') or METHODS.get(receipt.get('method_key'))!=receipt.get('method') or
        receipt.get('source',{}).get('dirty') is not False or
        config.get('instance',{}).get('split')!=('test' if bank['config']['tier']=='TEST' else 'development')):
        raise ValueError('repair method/tier/action bank differs')
    roles=_roles(original)
    if bank['config']['tier']=='TEST':
        from robo.roundtrip.scope_bundle import validate_target_build_binding
        if receipt['method_key']!='B4':raise ValueError('TEST scope matrix has no BM arm')
        if roles['target'].get('constructor_method')!=config['controller_method']:
            raise ValueError('scope repair changed target construction method')
        validate_target_build_binding(receipt['target_build_binding'],config,roles['target']['object_dir'],original['capture_manifest_sha256'])
    for entity in roles.values():_closure(entity)
    for key in ('canonical_instance_id','canonical_manifest_sha256','capture_manifest_sha256','cohort_id'):
        if receipt.get(key)!=original.get(key):raise ValueError('repair canonical/capture identity changed')
    if config.get('canonical_instance_id')!=receipt['canonical_instance_id'] or config.get('scope')!=receipt['scope']:
        raise ValueError('repair requested for different canonical/scope')
    sealed=_checked(receipt['input_scope_bundle']);sealed_roles=_roles(sealed)
    for name in roles:
        for key in ('artifact_hashes','native_role','role','component_kind'):
            if roles[name].get(key)!=sealed_roles[name].get(key):raise ValueError('original scope role/artifact identity changed')
    if receipt['frozen_target_artifact_hashes']!=roles['target']['artifact_hashes']:
        raise ValueError('repair changed frozen target')
    pool=_checked(receipt['original_B3_pool'])
    if receipt['original_B3_pool']['sha256']!=original['destination_selection']['candidate_pool_sha256']:
        raise ValueError('repair original B3 pool differs')
    if pool['canonical_instance_id']!=receipt['canonical_instance_id']:raise ValueError('repair pool canonical differs')
    actions=receipt['actions']
    if len(actions)!=receipt['actual_calls'] or len(actions)>2 or len({a['action_id'] for a in actions})!=len(actions):
        raise ValueError('repair action budget/identity invalid')
    prior=roles['destination']['artifact_hashes'];used=[]
    before=_checked(receipt['verification_before']);measured=before
    for action in actions:
        expected=next_action(receipt['method_key'],measured,used)
        if action['kind']!=expected or action['kind'] in used or action['dependency_id']!='destination':
            raise ValueError('repair scheduler/action bank changed')
        used.append(action['kind'])
        if action['source_artifact_hashes']!=prior:raise ValueError('repair proposal parent differs')
        if action['invalidated_dependencies']!=sorted(descendants(EDGES,'destination')):
            raise ValueError('repair dependency refresh incomplete')
        if artifact_hashes(action['source_object_dir'])!=prior or artifact_hashes(action['result_object_dir'])!=action['result_artifact_hashes']:
            raise ValueError('repair intermediate artifact closure changed')
        result=action['result_artifact_hashes']
        if result==prior:raise ValueError('repair reread identical artifact')
        if action['kind']=='regenerate_collision' and any(result[k]!=prior[k] for k in ('aligned.json','physics.json','mesh_sim.obj')):
            raise ValueError('collision action changed visual/pose/physics')
        if action['kind']=='reselect_candidate':
            alternative=lineage_alternative(pool)
            if alternative is None or action['details']['alternative_proposal_id']!=alternative['proposal_id']:
                raise ValueError('repair alternative does not match frozen generator lineage')
            matches=[r for r in pool['outcomes'] if r['selected_proposal_id']==alternative['proposal_id']]
            if not matches or any(matches[0]['artifact_hashes'].get(k)!=v for k,v in result.items()):
                raise ValueError('repair alternative artifact differs from pool')
        prior=result;measured=_checked(action['verification_after'])
    after=_checked(receipt['verification_after'])
    if canonical_hash(after)!=canonical_hash(measured) or not after['passed'] or not after['state_restored_byte_exact']:
        raise ValueError('repair final evidence missing/failed')
    if receipt.get('accepted') is not True or receipt.get('terminal_status')!='READY':
        raise ValueError('scope repair abstained or failed')
    selected=receipt['destination_selected'];_closure(selected)
    if selected['artifact_hashes']!=prior:raise ValueError('selected repair differs from final action')
    return receipt


def run_scope_verification(scope_bundle,canonical_reference,body_names,config,out,method='B4'):
    from robo.roundtrip.scope_bundle import selected_destination
    validate_config(config)
    if method not in METHODS:raise ValueError('unknown scope method')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    original=_json(scope_bundle);roles=_roles(original)
    baseline=_json(original['baseline_config'])
    expected_split='test' if config['tier']=='TEST' else 'development'
    if baseline['instance']['split']!=expected_split:raise ValueError('L1 scope tier differs from frozen admission')
    if config['tier']=='TEST':
        from robo.roundtrip.scope_bundle import validate_target_build_binding
        if method!='B4' or baseline['controller_method']!=METHODS[method]:
            raise ValueError('TEST L1 repair requires its own B4 target; no BM scope arm')
        validate_target_build_binding(original['target_build_binding'],baseline,roles['target']['object_dir'],original['capture_manifest_sha256'])
    for entity in roles.values():_closure(entity)
    if set(body_names)!=set(roles):raise ValueError('native evaluator body mapping incomplete')
    pool_path=Path(original['destination_selection']['candidate_pool_path'])
    pool=_json(pool_path)
    build_path=original['destination_build_manifest']
    selected,hashes,_=selected_destination(pool_path,build_path,_json(build_path),original['destination_selection']['rvg_receipt_path'])
    if hashes!=roles['destination']['artifact_hashes']:raise ValueError('scope original B3 selection differs')
    from robo.roundtrip.identity import validate_canonical_instance
    reference=Path(canonical_reference);native_manifest=_json(reference.parent/'canonical_instance.json')
    validate_canonical_instance(native_manifest,reference)
    if (native_manifest['canonical_instance_id']!=original['canonical_instance_id'] or
            native_manifest['manifest_sha256']!=original['canonical_manifest_sha256'] or native_manifest['dataset_split']!=expected_split):
        raise ValueError('scope probe canonical identity/tier differs')
    xml=(reference/'scene.xml').read_text();state=_json(reference/'canonical_state.json')
    entities=[dict(body_name=body_names[e['object_id']],object_id=e['object_id'],role=e['role'],object_dir=e['object_dir'],
        **({'component_kind':e['component_kind']} if e.get('component_kind') else {})) for e in original['entities']]
    control,_=measure_scope(xml,state,[e for e in entities if e['object_id']=='target'],config,scope='L0_target_only')
    _write(out/'retained_destination_L0_control.json',control)
    current=copy.deepcopy(roles['destination']);current['proposal_id']=original['destination_selection']['selected_proposal_id']
    before,manifest=measure_scope(xml,state,entities,config);_write(out/'verification_before.json',before)
    measured=before;actions=[];used=[];error=None
    alternative=lineage_alternative(pool)
    if alternative:
        alt=Path(alternative['object_dir']);alternative['object_dir']=str(pool_path.parent/alt.name) if alt.is_relative_to('/output') else str(alt)
    for index in range(config['max_calls_per_dependency']):
        action=next_action(method,measured,used)
        if action is None:break
        action_dir=out/f'action_{index:02d}';action_dir.mkdir();started=time.monotonic()
        parent=current['proposal_id'];prior=current['artifact_hashes']
        proposal=f'{parent}:L1:{method}:{index}:{action}'
        try:
            if action=='regenerate_collision':
                details=regenerate_collision(current['object_dir'],action_dir/'object')
                candidate=action_dir/'object'
            else:
                if alternative is None:raise ValueError('missing other-initial-generator candidate')
                candidate=Path(alternative['object_dir']);hashes=artifact_hashes(candidate)
                expected=[r for r in pool['outcomes'] if r['selected_proposal_id']==alternative['proposal_id']]
                if not expected or any(expected[0]['artifact_hashes'].get(k)!=v for k,v in hashes.items()):
                    raise ValueError('alternative candidate source closure changed')
                details=dict(alternative_proposal_id=alternative['proposal_id'],source_pool_sha256=file_hash(pool_path))
            hashes=artifact_hashes(candidate)
            if hashes==prior:raise ValueError('repair action produced identical artifacts')
            updated=copy.deepcopy(entities)
            next(e for e in updated if e['object_id']=='destination')['object_dir']=str(candidate.resolve())
            next_measure,next_manifest=measure_scope(xml,state,updated,config)
            _write(action_dir/'verification_after.json',next_measure)
            row=dict(action_id=proposal,kind=action,dependency_id='destination',proposal_id=proposal,parent_proposal_id=parent,
                source_object_dir=current['object_dir'],result_object_dir=str(candidate.resolve()),
                source_artifact_hashes=prior,result_artifact_hashes=hashes,wall_s=time.monotonic()-started,
                invalidated_dependencies=sorted(descendants(EDGES,'destination')),details=details,
                verification_after=_source(action_dir/'verification_after.json'))
            actions.append(row);used.append(action);entities=updated;measured=next_measure;manifest=next_manifest
            current=dict(object_dir=str(candidate.resolve()),artifact_hashes=hashes,proposal_id=proposal,parent_proposal_id=parent)
        except Exception as exc:
            error=dict(action_id=proposal,kind=action,type=type(exc).__name__,reason=str(exc),wall_s=time.monotonic()-started)
            _write(action_dir/'failure.json',error);break
    _write(out/'verification_after.json',measured)
    accepted=measured['passed'] and error is None
    ledger=[dict(policy_id=method,job_id=original['canonical_instance_id']+':destination',proposal_id=current['proposal_id'],
        selected_proposal_id=current['proposal_id'] if accepted else None,terminal_action='accept' if accepted else 'abstain',
        reason_codes=measured['reason_codes'],actions=actions,failed_action=error)]
    validate_ledger(ledger)
    with (out/'repair_ledger.jsonl').open('x') as f:
        for row in ledger:f.write(json.dumps(row,allow_nan=False)+'\n')
    report=dict(schema_version=1,tier=config['tier'],scope=config['scope'],method=METHODS[method],method_key=method,
        **{key:original[key] for key in ('canonical_instance_id','canonical_manifest_sha256','capture_manifest_sha256','cohort_id')},
        input_scope_bundle=_source(scope_bundle),original_B3_pool=_source(pool_path),
        frozen_target_artifact_hashes=roles['target']['artifact_hashes'],target_build_binding=original.get('target_build_binding'),
        target_method=roles['target'].get('constructor_method'),destination_original=roles['destination'],
        destination_selected=current,action_bank=dict(config=config,config_sha256=canonical_hash(config),max_calls_per_dependency=2),
        actual_calls=len(actions)+(error is not None),completed_calls=len(actions),actions=actions,failed_action=error,
        accepted=bool(accepted),terminal_status='READY' if accepted else ('BUILD_FAILED' if error else 'ABSTAINED'),
        verification_before=_source(out/'verification_before.json'),verification_after=_source(out/'verification_after.json'),
        canonical_sources={name:_source(reference/name) for name in ('scene.xml','canonical_state.json')},
        retained_destination_L0_control=_source(out/'retained_destination_L0_control.json'),
        control_usage='diagnostic attribution only; not a repair scheduler or threshold input',
        body_names=body_names,source=git_snapshot(Path(__file__).resolve().parents[2]),
        oracle_context=True,native_policy_success=None,independent_physical_validation='NOT_RUN',
        limitation='Own finite-contact/settle gate only; no IK, certified opening, fidelity or native-success guarantee.')
    _write(out/'build_manifest.json',report)
    if manifest is not None:_write(out/'import_manifest.json',manifest)
    if accepted:
        admission=copy.deepcopy(baseline);admission.update(scope=config['scope'],controller_method=METHODS[method])
        validate_scope_repair(out/'build_manifest.json',original_scope_bundle=original,config=admission)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scope-bundle','canonical-reference','body-names','config','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--method',choices=METHODS,required=True)
    a=p.parse_args(argv)
    config=_json(a.config)
    report=run_scope_verification(a.scope_bundle,a.canonical_reference,_json(a.body_names),config,a.out,a.method)
    print(json.dumps({k:report[k] for k in ('method','terminal_status','accepted','actual_calls')},allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
