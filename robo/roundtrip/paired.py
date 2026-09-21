"""One-target native same-policy comparison bound to an exported REF instance.

Privileged evaluation only. Construction must finish before reading this bundle.
Seed equality alone is insufficient to bind a RoboCasa scene instance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from robo.manifest.hash import canonical_hash, git_snapshot


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_reference_bundle(reference_episode, canonical_reference, *, config, reset_seed):
    reference = Path(reference_episode).resolve(strict=True)
    canonical = Path(canonical_reference).resolve(strict=True)
    # The existing REF producer exports immediately before each episode. It did
    # not yet place XML hashes in result.json; enforce its sibling convention.
    if (canonical.parent != reference.parent or canonical.name != f'canonical_seed{reset_seed}'
            or reference.name != f'episode_seed{reset_seed}'):
        raise ValueError('canonical bundle must be the REF producer sibling for this seed')
    result = json.loads((reference/'result.json').read_text())
    if result.get('execution_kind') != 'closed_loop_visual_policy' or result.get('treatment_id') != 'REF_NATIVE':
        raise ValueError('paired source must be a native learned-policy REF episode')
    if result.get('config_sha256') != canonical_hash(config):
        raise ValueError('paired native config differs from REF')
    if result.get('reset_seed') != reset_seed or reset_seed not in config['reset_seeds']:
        raise ValueError('paired reset differs from REF or declared roster')
    state = json.loads((canonical/'canonical_state.json').read_text())
    initial = json.loads((reference/'initial_state.json').read_text())
    if canonical_hash(state) != canonical_hash(initial):
        raise ValueError('canonical state does not match the REF episode initial state')
    required = {'integration_state','controller_state','observable_timing','native_metadata'}
    if not required.issubset(state):
        raise ValueError('canonical state lacks full native restoration fields')
    xml = (canonical/'scene.xml').read_text()
    provenance = {'reference_episode':str(reference), 'canonical_reference':str(canonical),
        'reset_seed':reset_seed, 'config_sha256':canonical_hash(config),
        'reference_files':{name:_sha(reference/name) for name in
                           ['result.json','initial_state.json','actions.json']},
        'canonical_files':{name:_sha(canonical/name) for name in ['canonical_state.json','scene.xml']},
        'canonical_state_semantic_sha256':canonical_hash(state),
        'xml_binding':'immediate pre-episode export in original REF producer sibling directory',
        'reference_result_has_independent_xml_hash':False}
    return {'state':state, 'xml':xml, 'result':result, 'provenance':provenance}


def prepare_paired_adapter(adapter, bundle, *, object_dir=None, object_id='target', role='obj'):
    """Bind Python task roles to canonical metadata before choosing the target.

    Both imports use the saved REF integration/controller state. The generated
    target pose is supplied only by its frozen construction receipt.
    """
    from robo.roundtrip.importers.robocasa import import_reconstructed_object, rebind_native_object
    adapter.reset_from_spec({'seed':bundle['provenance']['reset_seed']})
    adapter.import_xml(bundle['xml'], canonical_state=bundle['state'])
    if object_dir is None:
        return {'scope':'native_canonical_import_control', 'canonical':bundle['provenance']}
    obj = adapter.native.objects[role]
    if len(obj.joints) != 1:
        raise ValueError('bounded rigid comparison requires one native free joint')
    joint, body = obj.joints[0], obj.root_body
    xml, imported = import_reconstructed_object(bundle['xml'], body_name=body,
        object_dir=object_dir, object_id=object_id)
    adapter.import_xml(xml, canonical_state=bundle['state'], replaced_joint=joint,
        estimated_pose=imported['position_m']+imported['quaternion_wxyz'])
    imported['binding'] = rebind_native_object(adapter.native, object_name=role, receipt=imported)
    imported['canonical'] = bundle['provenance']
    return imported



def prepare_scope_adapter(adapter,bundle,scope_bundle,baseline_import):
    """Import two frozen estimated rigid objects; native role lookup is evaluator-only."""
    from robo.roundtrip.scope import import_scope,bind_scope_objects
    prepare_paired_adapter(adapter,bundle)
    entities=[];roles={}
    for entity in scope_bundle['entities']:
        if scope_bundle.get('scope')=='DEV_partial_observed_workspace' and entity.get('component_kind'):
            fixture=adapter.native.get_fixture(entity['native_role'])
            if fixture.name!=entity['native_role'] or fixture.root_body!=entity['body_name']:
                raise ValueError('exact sealed workspace fixture binding differs')
            entities.append(dict(body_name=fixture.root_body,object_id=entity['object_id'],object_dir=entity['object_dir'],role=entity['role'],component_kind=entity['component_kind']))
            roles[entity['object_id']]=fixture.name
            continue
        if entity.get('component_kind') in ('sink_basin','cabinet_bottom_shelf'):
            attribute='sink' if entity['component_kind']=='sink_basin' else 'cab'
            if entity.get('native_role')!=attribute or entity.get('native_binding_kind')!='fixture_task_attribute':
                raise ValueError('static destination requires declared native task fixture binding')
            fixture=getattr(adapter.native,attribute)
            entities.append(dict(body_name=fixture.root_body,object_id=entity['object_id'],
                object_dir=entity['object_dir'],role=entity['role'],component_kind=entity['component_kind']))
            # Exact fixture name avoids random substring role lookup.
            roles[entity['object_id']]=fixture.name
            continue
        obj=adapter.native.objects.get(entity['native_role'])
        if obj is None or len(obj.joints)!=1:
            raise ValueError('L1 requires two existing native movable rigid object roles')
        entities.append(dict(body_name=obj.root_body,object_id=entity['object_id'],
            object_dir=entity['object_dir'],role=entity['role']))
        roles[entity['object_id']]=entity['native_role']
    if scope_bundle.get('scope')=='DEV_partial_observed_workspace':
        from robo.roundtrip.workspace_import import import_observed_workspace
        from robo.roundtrip.scope_bundle import artifact_hashes
        for e in entities:e['artifact_hashes']=artifact_hashes(e['object_dir'])
        coverage=json.loads(Path(scope_bundle['workspace_coverage']['path']).read_text())
        xml,manifest=import_observed_workspace(bundle['xml'],base_entities=entities[:2],support_entities=entities[2:],coverage=coverage,bounds=coverage['workspace_bounds_world_m'])
    else:
        xml,manifest=import_scope(bundle['xml'],entities=entities,scope='L1_target_destination')
    target=next(r for r in manifest['entities'] if r['role']=='target')
    # Same files are necessary but not sufficient: importer physics/appearance
    # must also match the already-executed L0 target before native mutation.
    keys=('source_hashes','position_m','quaternion_wxyz','scale','bounds_local_m','mass_kg',
          'inertia_kg_m2','physics_prior','contact_prior','geometry_inflation_m','appearance',
          'visual_geoms','contact_geoms')
    if any(canonical_hash(target.get(k))!=canonical_hash(baseline_import.get(k)) for k in keys):
        raise ValueError('L1 import changes target artifact or target physics/appearance')
    fixture_components=[r for r in manifest['entities'] if r.get('import_route')=='static_fixture_component_v1']
    kwargs={'fixture_components':fixture_components} if fixture_components else {}
    adapter.import_xml(xml,canonical_state=bundle['state'],replaced_joints=manifest['replaced_joints'],**kwargs)
    manifest['binding']=bind_scope_objects(adapter.native,manifest=manifest,native_role_names=roles)
    manifest['canonical']=bundle['provenance']
    from robo.roundtrip.scope_contacts import bind_adapter_contacts
    bind_adapter_contacts(adapter,manifest)
    return manifest


def run_paired_episode(adapter, policy, *, bundle, config, object_dir, out_dir,
                       treatment_id='FIXED', object_id='target', role='obj'):
    """Use the canonical learned-policy runner, with unchanged config and RNG."""
    from robo.eval.harness_runner import run_native_episode
    if not getattr(policy, 'is_visual_policy', False):
        raise ValueError('paired closed-loop requires a learned visual policy')
    if canonical_hash(policy.metadata) != canonical_hash(bundle['result']['policy_identity']):
        raise ValueError('policy/checkpoint/runtime identity differs from REF')
    if canonical_hash(config) != bundle['provenance']['config_sha256']:
        raise ValueError('config changed after canonical bundle validation')
    if not object_dir or treatment_id == 'REF_NATIVE':
        raise ValueError('reconstructed comparison requires an object and distinct treatment')
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=False)
    receipt = {'schema_version':1, 'execution_kind':'closed_loop_visual_policy',
        'treatment_axis':'reconstructed_target_asset_bundle',
        'replacement_scope':'target_only_oracle_context_diagnostic',
        'reference':bundle['provenance'], 'policy_identity':policy.metadata,
        'source_code':git_snapshot(), 'planned_comparison_episodes':1,
        'executed_comparison_episodes':0, 'state':'PREPARING'}
    def save():
        (out/'pair_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        imported = prepare_paired_adapter(adapter,bundle,object_dir=object_dir,object_id=object_id,role=role)
        (out/'import_receipt.json').write_text(json.dumps(imported,indent=2)+'\n')
        result = run_native_episode(adapter,policy,config=config,
            reset_seed=bundle['provenance']['reset_seed'],out_dir=out/'episode',
            treatment_id=treatment_id,execution_kind='closed_loop_visual_policy')
        receipt.update(state='RECORDED', executed_comparison_episodes=int(result['executed']),
            comparison_result_sha256=_sha(out/'episode'/'result.json'),
            reference_success=bundle['result'].get('success'), comparison_success=result.get('success'),
            comparison_error=result.get('error'))
        save()
        return result
    except Exception as exc:
        receipt.update(state='FAILED', failure=f'{type(exc).__name__}: {exc}')
        save()
        raise


def run_resolved_episode(adapter, policy, *, config, canonical_reference, canonical_manifest,
                         reset_bank, out_dir, object_dir=None, reference_episode=None,
                         object_id='target', role='obj', replay_actions=None, actions_sha256=None,
                         scope_bundle=None, baseline_config=None, observed_surface_receipt=None,
                         control_b0_build=None, control_capture=None):
    """Version-2 native worker with explicit L0 construction or L0-to-L1 scope pairing."""
    from robo.roundtrip.spec import validate_spec, paired_config_hash
    from robo.roundtrip.identity import validate_canonical_instance, validate_reset_bank
    from robo.eval.harness_runner import run_native_episode
    validate_spec(config)
    if config['schema_version']!=2 or config['scope'] not in ('L0_target_only','L1_target_destination','DEV_partial_observed_workspace') or config.get('renderer','native')!='native':
        raise ValueError('resolved worker supports bounded v2 L0/L1 native observations only')
    observed=config['controller_method']=='OBSERVED_SURFACE_NATIVE'
    control_binding=None
    if observed:
        if config['scope']!='L0_target_only' or object_dir is None or any(x is None for x in (observed_surface_receipt,control_b0_build,control_capture)):
            raise ValueError('observed native control requires sealed receipt, source B0 build and public capture')
        from robo.roundtrip.observed_native import validate_observed_control
        control_binding=validate_observed_control(observed_surface_receipt,object_dir,control_b0_build,control_capture,config)
    elif any(x is not None for x in (observed_surface_receipt,control_b0_build,control_capture)):
        raise ValueError('observed control provenance cannot label another method')
    partial=config['scope']=='DEV_partial_observed_workspace'
    l1=config['scope']=='L1_target_destination' or partial
    scope_record=None
    if l1:
        if scope_bundle is None or baseline_config is None or reference_episode is None or object_dir is not None or replay_actions is not None or role!='obj' or object_id!='target':
            raise ValueError('L1 requires sealed scope bundle and L0 baseline, no separate target or replay')
        from robo.roundtrip.scope_bundle import validate_bundle
        scope_record=json.loads(Path(scope_bundle).read_text())
        if partial:
            from robo.roundtrip.workspace_policy import validate_workspace_bundle
            validate_workspace_bundle(scope_record,config,reference_episode,baseline_config)
        else:
            validate_bundle(scope_record,config,reference_episode,baseline_config)
    elif scope_bundle is not None or baseline_config is not None:
        raise ValueError('scope-only arguments cannot alter an L0 comparison')
    directory=Path(canonical_reference)
    manifest=json.loads(Path(canonical_manifest).read_text())
    bank=json.loads(Path(reset_bank).read_text())
    validate_canonical_instance(manifest,directory);validate_reset_bank(bank,manifest)
    if (config['canonical_instance_id']!=manifest['canonical_instance_id'] or
            config['canonical_manifest_sha256']!=manifest['manifest_sha256'] or
            config['reset_contract_sha256']!=bank['reset_contract_sha256']):
        raise ValueError('resolved unit canonical/reset identity mismatch')
    if manifest['identity']['policy_sha256']!=canonical_hash(config['policy']):
        raise ValueError('resolved policy differs from canonical acquisition contract')
    resets=[r for r in bank['resets'] if r['reset_id']==config['reset_id']]
    if len(resets)!=1 or resets[0]['policy_rng_seed']!=config['policy_rng_seed']:
        raise ValueError('resolved unit reset or RNG differs from bank')
    metadata=policy.metadata
    from robo.roundtrip.policy_engine import validate_engine
    validate_engine(config,metadata)
    if metadata.get('checkpoint_receipt_sha256')!=config['policy']['checkpoint_receipt_sha256']:
        raise ValueError('live policy checkpoint differs from resolved contract')
    method=config['controller_method'];reference=None
    if l1:
        reference=json.loads((Path(reference_episode)/'result.json').read_text())
        if canonical_hash(reference.get('policy_identity'))!=canonical_hash(metadata):
            raise ValueError('L0 baseline policy runtime differs')
    elif method=='REF_NATIVE':
        if replay_actions is not None:raise ValueError('native action replay must be REF_IMPORT_CONTROL')
        if object_dir is not None:raise ValueError('REF must use native assets')
    else:
        if not reference_episode or (method!='REF_IMPORT_CONTROL' and not object_dir):
            raise ValueError('reconstructed worker requires build and REF')
        if method=='REF_IMPORT_CONTROL' and (object_dir is not None or replay_actions is None):
            raise ValueError('REF import control requires reference actions and native assets')
        reference=json.loads((Path(reference_episode)/'result.json').read_text())
        if reference.get('native_schema_version')!=2 or reference.get('controller_method')!='REF_NATIVE':
            raise ValueError('v2 comparison requires a v2 REF, not a legacy RNG episode')
        if reference.get('comparison_contract_sha256')!=paired_config_hash(config):
            raise ValueError('REF frozen fields/reset/protocol differ')
        if canonical_hash(reference.get('policy_identity'))!=canonical_hash(metadata):
            raise ValueError('REF policy runtime differs')
    if reference is not None:
        validate_engine(config,metadata,reference)
    if replay_actions is not None:
        if (config['execution_protocol']!='full_horizon_feedback_diagnostic' or reference is None or
                reference.get('full_horizon_completed') is not True or reference.get('ticks')!=config['horizon']):
            raise ValueError('full-horizon replay requires a completed full-H native reference')
        if (Path(replay_actions).resolve()!=(Path(reference_episode)/'actions.json').resolve() or
                _sha(replay_actions)!=actions_sha256 or actions_sha256!=reference.get('actions_sha256')):
            raise ValueError('replay action source/hash differs from full-H REF')
        from robo.roundtrip.replay import _action_array
        if len(_action_array(json.loads(Path(replay_actions).read_text())))!=config['horizon']:
            raise ValueError('full-H action source length differs from native horizon')
    binding={'canonical_instance_id':manifest['canonical_instance_id'],
        'canonical_manifest_sha256':manifest['manifest_sha256'],
        'scene_xml_sha256':manifest['identity']['scene_xml_sha256'],
        'state_sha256':manifest['identity']['state_sha256'],
        'asset_closure_sha256':manifest['identity']['asset_closure_sha256']}
    if reference is not None and reference.get('canonical_reference')!=binding:
        raise ValueError('REF direct XML/state/asset binding differs')
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=False)
    receipt={'schema_version':2,'canonical_reference':binding,'reset':resets[0],
        'config_sha256':canonical_hash(config),'comparison_contract_sha256':paired_config_hash(config),
        'source_code':git_snapshot(),'planned_comparison_episodes':1,'executed_comparison_episodes':0,
        'state':'PREPARING','method':method,'reference_result_sha256':None if reference is None else _sha(Path(reference_episode)/'result.json')}
    def save():(out/'pair_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if observed:
        receipt['construction_control']=control_binding
        adapter.control_binding=control_binding
    if l1:
        receipt.update(treatment_axis='replacement_scope',scope_bundle_sha256=_sha(scope_bundle),
            system_variant=scope_record['system_variant'],baseline_scope='L0_target_only',
            scope_role_constructors={r['object_id'] if partial else r['role']:r['constructor_method'] for r in scope_record['entities']})
    save()
    try:
        bundle={'state':json.loads((directory/'canonical_state.json').read_text()),
            'xml':(directory/'scene.xml').read_text(),
            'provenance':{'reset_seed':config['reset_seeds'][0],'canonical_reference':binding}}
        if l1:
            baseline_import=json.loads((Path(reference_episode).parent/'import_receipt.json').read_text())
            imported=prepare_scope_adapter(adapter,bundle,scope_record,baseline_import)
            adapter.scope_binding={k:receipt[k] for k in ('treatment_axis','scope_bundle_sha256','system_variant','baseline_scope','scope_role_constructors','reference_result_sha256')}

        else:
            imported=prepare_paired_adapter(adapter,bundle,object_dir=object_dir,object_id=object_id,role=role)
        (out/'import_receipt.json').write_text(json.dumps(imported,indent=2)+'\n')
        reset_receipt=adapter.apply_paired_reset(resets[0],role=role)
        (out/'reset_receipt.json').write_text(json.dumps(reset_receipt,indent=2)+'\n')
        adapter.canonical_binding=binding
        if replay_actions is None:
            result=run_native_episode(adapter,policy,config=config,reset_seed=config['policy_rng_seed'],
                out_dir=out/'episode',treatment_id=method,execution_kind='closed_loop_visual_policy')
        else:
            from robo.roundtrip.replay import run_replay_episode
            result=run_replay_episode(adapter,config=config,reset_seed=config['policy_rng_seed'],
                out_dir=out/'episode',treatment_id=method,actions_path=replay_actions,
                expected_actions_sha256=actions_sha256,
                source_identity={'reference_result_sha256':_sha(Path(reference_episode)/'result.json'),
                                 'canonical_reference':binding,'source_policy_identity':metadata})
        receipt.update(state='RECORDED',executed_comparison_episodes=int(result['executed']),
            comparison_result_sha256=_sha(out/'episode'/'result.json'),
            comparison_success=result.get('success'),comparison_error=result.get('error'))
        save();return result
    except Exception as exc:
        receipt.update(state='FAILED',failure=f'{type(exc).__name__}: {exc}');save();raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','canonical-reference','out']:
        parser.add_argument('--'+name,required=True)
    for name in ['reference-episode','object-dir','canonical-manifest','reset-bank']:
        parser.add_argument('--'+name)
    parser.add_argument('--reset-seed',type=int)
    parser.add_argument('--replay-actions');parser.add_argument('--actions-sha256')
    parser.add_argument('--scope-bundle');parser.add_argument('--baseline-config')
    parser.add_argument('--observed-surface-receipt');parser.add_argument('--control-b0-build');parser.add_argument('--control-capture')
    parser.add_argument('--treatment-id',default='FIXED')
    parser.add_argument('--object-id',default='target'); parser.add_argument('--role',default='obj')
    parser.add_argument('--host',default='localhost'); parser.add_argument('--port',type=int,default=8017)
    args=parser.parse_args(argv)
    from robo.roundtrip.spec import load_spec
    from robo.roundtrip.adapters.robocasa import RoboCasaAdapter
    from robo.roundtrip.native_policy import NativePolicy
    config=load_spec(args.config)
    if config['schema_version']==1:
        if args.reference_episode is None or args.reset_seed is None or args.object_dir is None:
            parser.error('v1 requires reference-episode, reset-seed and object-dir')
        bundle=load_reference_bundle(args.reference_episode,args.canonical_reference,config=config,reset_seed=args.reset_seed)
    elif not args.canonical_manifest or not args.reset_bank:
        parser.error('v2 requires canonical-manifest and reset-bank')
    policy=NativePolicy(args.host,args.port,replan_steps=5)
    adapter=RoboCasaAdapter(config)
    try:
        if config['schema_version']==2:
            result=run_resolved_episode(adapter,policy,config=config,canonical_reference=args.canonical_reference,
                canonical_manifest=args.canonical_manifest,reset_bank=args.reset_bank,object_dir=args.object_dir,
                reference_episode=args.reference_episode,out_dir=args.out,object_id=args.object_id,role=args.role,
                replay_actions=args.replay_actions,actions_sha256=args.actions_sha256,
                scope_bundle=args.scope_bundle,baseline_config=args.baseline_config,
                observed_surface_receipt=args.observed_surface_receipt,control_b0_build=args.control_b0_build,control_capture=args.control_capture)
        else:
            result=run_paired_episode(adapter,policy,bundle=bundle,config=config,object_dir=args.object_dir,
                out_dir=args.out,treatment_id=args.treatment_id,object_id=args.object_id,role=args.role)
        print(json.dumps(result,indent=2))
        return 0 if result.get('error') is None else 2
    finally:
        adapter.close();policy.close()


if __name__=='__main__':
    raise SystemExit(main())
