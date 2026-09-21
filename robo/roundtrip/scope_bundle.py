"""Sealed estimated role handles for an explicit L0-to-L1 scope contrast."""
from pathlib import Path
import argparse
import copy
import json
from robo.manifest.hash import canonical_hash
from robo.roundtrip.identity import file_hash


def artifact_hashes(directory):
    directory=Path(directory)
    paths=[directory/n for n in ('aligned.json','physics.json','mesh_sim.obj')]
    parts=sorted((directory/'collision').glob('part_*.obj'))
    if not parts:raise ValueError('scope role has no frozen CoACD parts')
    return {str(p.relative_to(directory)):file_hash(p) for p in paths+parts}


def validate_scope_pair(config,baseline_config,baseline_result):
    from robo.roundtrip.spec import validate_spec
    validate_spec(config);validate_spec(baseline_config)
    if (config['scope']!='L1_target_destination' or baseline_config['scope']!='L0_target_only' or
        config['instance']['task_id'] not in ('PickPlaceSinkToCounter','PickPlaceCounterToSink','PickPlaceCounterToCabinet') or config['controller_method'] not in ('B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE') or
        baseline_config['controller_method']!=config['controller_method'] or config['execution_protocol']!='primary_native'):
        raise ValueError('bounded L1 contrast requires supported native sink task L0/L1 B3 target, primary protocol')
    allowed={'scope','replacement_scope'}
    if canonical_hash({k:v for k,v in config.items() if k not in allowed})!=canonical_hash({k:v for k,v in baseline_config.items() if k not in allowed}):
        raise ValueError('scope contrast changes another frozen field')
    if (baseline_result.get('native_schema_version')!=2 or baseline_result.get('execution_kind')!='closed_loop_visual_policy' or
        baseline_result.get('controller_method')!=config['controller_method'] or baseline_result.get('scope')!='L0_target_only' or
        baseline_result.get('config_sha256')!=canonical_hash(baseline_config) or not baseline_result.get('executed') or
        baseline_result.get('error') is not None):
        raise ValueError('scope baseline must be a completed matching canonical L0 episode')
    # Success is deliberately not an admission requirement.


def selected_destination(pool_path,build_path,build,rvg_receipt=None):
    """Consume only the frozen A3 outcome; never select using native outcomes."""
    from robo.roundtrip.shared_candidates import proposal_prefix
    pool_path=Path(pool_path).resolve();pool=json.loads(pool_path.read_text())
    if (pool.get('schema_version')!=2 or pool.get('planned_objects')!=1 or
        pool.get('controller')!='agents.orchestrator.controller.run_policies' or
        pool.get('isolated_probe_is_native_context_evidence') is not False or
        pool.get('object_role') not in ('receptacle','support') or pool.get('object_role')!=build.get('object_role') or
        pool.get('b0_manifest_sha256')!=file_hash(build_path)):
        raise ValueError('destination pool requires its sealed B0 receptacle source')
    for key in ('canonical_instance_id','cohort_id','capture_manifest_sha256'):
        if pool.get(key)!=build[key]:raise ValueError('destination pool canonical/capture identity differs')
    prefix=proposal_prefix(build,{'capture_manifest_sha256':build['capture_manifest_sha256']})
    candidates=pool['initial_candidates']+([pool['retry_candidate']] if pool.get('retry_candidate') else [])
    ids=[c['proposal_id'] for c in candidates]
    if len(ids)!=len(set(ids)) or not all(pid.startswith(prefix+':') for pid in ids):
        raise ValueError('destination pool proposal role/dependency identity differs')
    rows=[r for r in pool['outcomes'] if r.get('native_method')=='B3_AGENT_NATIVE']
    if len(rows)!=1 or rows[0].get('policy_id')!='A3':raise ValueError('destination pool requires one A3/B3 outcome')
    row=rows[0]
    if row.get('terminal_action')!='accept' or row.get('selected_proposal_id') not in ids:
        raise ValueError('B3 destination rejected, abstained or failed; no B0 fallback')
    selected=Path(row['object_dir'])
    if selected.is_relative_to('/output'):
        selected=(pool_path.parent.parent/selected.relative_to('/output')).resolve()
    else:selected=(selected if selected.is_absolute() else pool_path.parent/selected).resolve()
    if not selected.is_relative_to(pool_path.parent):raise ValueError('selected destination escapes its frozen pool')
    hashes=row['artifact_hashes']
    for name,digest in hashes.items():
        member=Path(name);path=selected/member
        if member.is_absolute() or '..' in member.parts or path.is_symlink() or file_hash(path)!=digest:
            raise ValueError('B3 destination artifact closure changed')
    for name,digest in artifact_hashes(selected).items():
        if hashes.get(name)!=digest:raise ValueError('B3 destination imported artifact is not sealed')
    ledger=pool_path.parent/'selection_ledger.jsonl'
    ledger_rows=[json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    if ledger_rows!=pool['outcomes']:raise ValueError('destination selection ledger differs from pool')
    rvg=Path(rvg_receipt).resolve() if rvg_receipt is not None else pool_path.parent.parent/'rvg/rvg_receipt.json'
    rr=json.loads(rvg.read_text())
    if file_hash(rvg)!=pool['rvg_receipt_sha256'] or rr.get('b0_manifest_sha256')!=pool['b0_manifest_sha256']:
        raise ValueError('destination generator receipt differs from pool')
    binding=dict(candidate_pool_path=str(pool_path),candidate_pool_sha256=file_hash(pool_path),
        selection_ledger_sha256=file_hash(ledger),rvg_receipt_path=str(rvg),rvg_receipt_sha256=file_hash(rvg),
        selected_proposal_id=row['selected_proposal_id'],reason_codes=row['reason_codes'],retry_invoked=row['retry_invoked'],
        native_method='B3_AGENT_NATIVE',construction_evidence_scope='isolated observed-object evidence; not native context verification')
    return selected,artifact_hashes(selected),binding


def seal_bundle(baseline_episode,baseline_config,target_dir,destination_build_manifest,destination_candidate_pool=None,destination_rvg_receipt=None,*,destination_repair_receipt=None,_construction_only=False,target_build_binding=None):
    baseline_config=Path(baseline_config);config=json.loads(baseline_config.read_text())
    l1=copy.deepcopy(config);l1.update(scope='L1_target_destination',replacement_scope='target_destination')
    target=Path(target_dir).resolve();hashes=artifact_hashes(target)
    if _construction_only:
        from robo.roundtrip.spec import validate_spec
        validate_spec(config)
        if baseline_episode is not None or config.get('schema_version')!=2 or config.get('scope')!='L0_target_only' or config.get('controller_method') not in ('B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE') or destination_repair_receipt is not None:
            raise ValueError('construction-only scope requires unexecuted frozen L0 per-method input')
        if destination_candidate_pool is None:raise ValueError('construction input requires a real B3 destination selection')
    else:
        baseline_episode=Path(baseline_episode)
        result=json.loads((baseline_episode/'result.json').read_text());validate_scope_pair(l1,config,result)
        imported=json.loads((baseline_episode.parent/'import_receipt.json').read_text())
        if {str(target/k):v for k,v in hashes.items()}!=imported['source_hashes']:
            raise ValueError('L1 target artifacts are not identical to executed L0 target')
    path=Path(destination_build_manifest).resolve();build=json.loads(path.read_text());declared=Path(build['object_dir'])
    # Isolated constructors mount this exact artifact root at /output.
    if declared.is_relative_to('/output'):
        destination=(path.parent/declared.relative_to('/output')).resolve()
    else:
        destination=(declared if declared.is_absolute() else path.parent/declared).resolve()
    if not destination.is_relative_to(path.parent):raise ValueError('scope component path escapes its sealed build root')
    if (build.get('status')!='BUILT' or build.get('built_objects')!=1 or build.get('component_only') is not True or
        build.get('object_role') not in ('receptacle','support') or build.get('declared_scope')!='L1_target_destination' or
        build.get('canonical_instance_id')!=config['canonical_instance_id'] or build.get('cohort_id')!=config['cohort_id'] or
        build.get('method')!='B0_fixed_trellis'):
        raise ValueError('destination requires a matching sealed B0 receptacle component')
    destination_hashes=artifact_hashes(destination)
    component_kind=build.get('component_kind')
    task=config['instance']['task_id']
    if ((task=='PickPlaceCounterToSink' and component_kind!='sink_basin') or
            (task=='PickPlaceSinkToCounter' and component_kind is not None) or
            (task=='PickPlaceCounterToCabinet' and component_kind!='cabinet_bottom_shelf') or
            (build.get('object_role')=='support')!=(component_kind=='cabinet_bottom_shelf')):
        raise ValueError('destination component does not match frozen public task role')
    for name,digest in destination_hashes.items():
        key=str((destination/name).relative_to(path.parent))
        if build['source_hashes'].get(key)!=digest:raise ValueError('destination artifact changed after build freeze')
    selection=None
    if destination_candidate_pool is not None:
        destination,destination_hashes,selection=selected_destination(destination_candidate_pool,path,build,destination_rvg_receipt)
    elif destination_rvg_receipt is not None:raise ValueError('RVG source binding requires a B3 destination pool')
    bundle=dict(schema_version=1,scope='L1_target_destination',canonical_instance_id=config['canonical_instance_id'],
        canonical_manifest_sha256=config['canonical_manifest_sha256'],cohort_id=config['cohort_id'],
        treatment_axis='replacement_scope',system_variant='hybrid_scope_diagnostic_target_B3_destination_B0',
        baseline_config=str(baseline_config.resolve()),baseline_config_sha256=file_hash(baseline_config),
        baseline_episode=None if _construction_only else str(baseline_episode.resolve()),baseline_result_sha256=None if _construction_only else file_hash(baseline_episode/'result.json'),
        baseline_import_receipt_sha256=None if _construction_only else file_hash(baseline_episode.parent/'import_receipt.json'),
        entities=[dict(object_id='target',native_role='obj',role='target',object_dir=str(target),constructor_method=config['controller_method'],artifact_hashes=hashes),
                  dict(object_id='destination',native_role='container',role=build['object_role'],object_dir=str(destination),constructor_method='B0_fixed_trellis',artifact_hashes=destination_hashes)],
        destination_build_manifest=str(path),destination_build_manifest_sha256=file_hash(path),
        capture_manifest_sha256=build['capture_manifest_sha256'],absolute_scorer_correspondence=None,
        claim_scope='hybrid role constructor; binding-specific native scope diagnostic')
    if selection is not None:
        bundle.update(system_variant='B3_target_B3_destination_L1',destination_selection=selection,
            claim_scope='B3 selection for both reconstructed roles; binding-specific native L1 scope; retained room/support oracle context')
        bundle['entities'][1]['constructor_method']='B3_AGENT_NATIVE'
    if component_kind=='sink_basin':
        bundle['entities'][1].update(native_role='sink',component_kind=component_kind,
                                    native_binding_kind='fixture_task_attribute')
        bundle.update(retained_fixture_context=['native fixture frame', 'native goal regions',
            'faucet articulation', 'fixture child accessories'],
            static_destination_scope='direct basin shell physical geometry only',
            native_goal_region_policy='fixed privileged native task rubric; no inferred generated interior')
    elif component_kind=='cabinet_bottom_shelf':
        bundle['entities'][1].update(native_role='cab',component_kind=component_kind,
                                    native_binding_kind='fixture_task_attribute')
        bundle.update(retained_fixture_context=['native fixture frame','all native goal regions',
            'native-open doors','cabinet walls and ceiling','upper shelves'],
            static_destination_scope='bottom cabinet support surface only',
            native_success_may_use_retained_upper_shelves=True,
            native_goal_region_policy='all original cabinet levels and full-bbox rubric unchanged; bottom support component diagnostic')
    if _construction_only:
        bundle.update(kind='unexecuted_scope_construction_input',policy_invoked=False,baseline_executed=False,
            construction_input_config_sha256=canonical_hash(config),target_method=config['controller_method'])
        if target_build_binding is not None:
            bundle['target_build_binding']=validate_target_build_binding(target_build_binding,config,target,build['capture_manifest_sha256'])
        elif config['controller_method']!='B3_AGENT_NATIVE' or config['instance'].get('split')=='test':
            raise ValueError('construction input requires typed target method/source proof')
        if config['controller_method']!='B3_AGENT_NATIVE':
            bundle.update(system_variant=config['controller_method']+'_target_B3_destination_construction_input',claim_scope='unexecuted per-method target plus B3 destination before context verification')
    if config['controller_method']!='B3_AGENT_NATIVE' and not _construction_only:
        if destination_repair_receipt is None or selection is None:
            raise ValueError('B4/BM scope requires explicit destination repair receipt and original B3 pool')
        from robo.roundtrip.scope_verification import validate_scope_repair
        repaired=validate_scope_repair(destination_repair_receipt,original_scope_bundle=bundle,config=l1)
        selected=repaired['destination_selected'];directory=Path(selected['object_dir']).resolve()
        hashes=artifact_hashes(directory)
        if any(selected['artifact_hashes'].get(k)!=v for k,v in hashes.items()):raise ValueError('repaired destination artifacts differ')
        bundle['entities'][1].update(object_dir=str(directory),artifact_hashes=hashes,constructor_method=config['controller_method'])
        bundle.update(destination_repair={'path':str(Path(destination_repair_receipt).resolve()),'sha256':file_hash(destination_repair_receipt)},
            system_variant=config['controller_method']+'_target_and_destination_L1',
            claim_scope='declared bounded scope verification/repair; native binding-specific outcomes')
    elif destination_repair_receipt is not None:
        raise ValueError('B3 scope cannot silently consume an extra repair action')
    return bundle


def validate_bundle(bundle,config,baseline_episode,baseline_config):
    if bundle.get('schema_version')!=1 or bundle.get('scope')!='L1_target_destination':raise ValueError('invalid scope bundle')
    if bundle.get('kind')=='unexecuted_scope_construction_input':raise ValueError('construction input is not a sealed executed-baseline scope comparison')
    baseline_episode=Path(baseline_episode);baseline_config=Path(baseline_config)
    if (file_hash(baseline_episode/'result.json')!=bundle['baseline_result_sha256'] or
        file_hash(baseline_episode.parent/'import_receipt.json')!=bundle['baseline_import_receipt_sha256'] or
        file_hash(baseline_config)!=bundle['baseline_config_sha256'] or
        file_hash(bundle['destination_build_manifest'])!=bundle['destination_build_manifest_sha256']):
        raise ValueError('scope source receipt changed')
    expected=seal_bundle(baseline_episode,baseline_config,bundle['entities'][0]['object_dir'],bundle['destination_build_manifest'],
        bundle.get('destination_selection',{}).get('candidate_pool_path'),bundle.get('destination_selection',{}).get('rvg_receipt_path'),
        destination_repair_receipt=bundle.get('destination_repair',{}).get('path'))
    if canonical_hash(expected)!=canonical_hash(bundle):raise ValueError('scope bundle differs from sealed actual artifacts')
    validate_scope_pair(config,json.loads(baseline_config.read_text()),json.loads((baseline_episode/'result.json').read_text()))
    return bundle



def validate_target_build_binding(row,config,target,capture_hash):
    """Validate the original accepted per-method target; abstention cannot be repaired by scope."""
    from robo.roundtrip.system_verification import asset_identity
    target=Path(target).resolve();method=config['controller_method']
    if (row.get('canonical_instance_id')!=config['canonical_instance_id'] or row.get('controller_method')!=method or
            row.get('accepted') is not True or row.get('terminal_status')!='READY' or
            Path(row.get('object_dir','')).resolve()!=target):
        raise ValueError('target method/identity/acceptance differs; inherit original failure')
    source=Path(row['binding_source']);manifest=Path(row['build_manifest'])
    if file_hash(source)!=row['binding_source_sha256'] or file_hash(manifest)!=row['build_manifest_sha256']:
        raise ValueError('target binding source changed')
    source_rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    fields=('canonical_instance_id','controller_method','accepted','terminal_status','object_dir','build_manifest','build_manifest_sha256')
    if sum(all(r.get(k)==row.get(k) for k in fields) for r in source_rows)!=1:raise ValueError('target binding not in frozen source')
    m=json.loads(manifest.read_text())
    if (m.get('method')!={'B3_AGENT_NATIVE':'B3','B4_ROOM_REPAIR_NATIVE':'B4'}[method] or m.get('accepted') is not True or
            m.get('capture_manifest_sha256')!=capture_hash or Path(m.get('selected_asset','')).resolve()!=target or
            m.get('selected_asset_identity')!=asset_identity(target)):
        raise ValueError('target construction method/artifact/source differs')
    return copy.deepcopy(row)


def prepare_construction_bundle(config_path,target_dir,destination_build_manifest,destination_candidate_pool,destination_rvg_receipt=None,*,target_build_binding=None):
    """Freeze real role artifacts before N3 verification; no policy result is fabricated."""
    return seal_bundle(None,config_path,target_dir,destination_build_manifest,destination_candidate_pool,destination_rvg_receipt,_construction_only=True,target_build_binding=target_build_binding)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['baseline-episode','baseline-config','target-dir','destination-build-manifest','out']:p.add_argument('--'+k,required=True)
    p.add_argument('--destination-candidate-pool')
    p.add_argument('--destination-rvg-receipt')
    a=p.parse_args(argv);bundle=seal_bundle(a.baseline_episode,a.baseline_config,a.target_dir,a.destination_build_manifest,a.destination_candidate_pool,a.destination_rvg_receipt)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(bundle,f,indent=2);f.write('\n')
    config=json.loads(Path(a.baseline_config).read_text());config.update(scope='L1_target_destination',replacement_scope='target_destination')
    with out.with_suffix('.config.json').open('x') as f:json.dump(config,f,indent=2);f.write('\n')
    print(out)


if __name__=='__main__':main()
