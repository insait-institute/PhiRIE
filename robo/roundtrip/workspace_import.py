"""Transactional partial DEV workspace integration with explicit L2 abstention.

Only post-freeze evaluator role bindings may name native bodies. The output is
an engineering scene, never a complete L2 trial while inventory is unresolved.
"""
import hashlib
from robo.roundtrip.scope import import_scope
from robo.roundtrip.fixture_scope import import_fixture_component
from robo.roundtrip.scope_bundle import artifact_hashes


def import_observed_workspace(xml,*,base_entities,support_entities,coverage,bounds):
    if (coverage.get('kind')!='TRAIN_sampled_workspace_coverage' or
            coverage.get('native_asset_access') is not False or
            coverage.get('heldout_access') is not False or coverage.get('recipe',{}).get('tier')!='DEV'):
        raise ValueError('sealed TRAIN-only DEV coverage required')
    if coverage['workspace_bounds_world_m']!=bounds:raise ValueError('workspace bounds changed')
    if {e['object_id'] for e in support_entities}!={'source_support','destination_support'}:
        raise ValueError('both declared workspace support identities required')
    if len(support_entities)!=2 or any(e['role']!='support' for e in support_entities):
        raise ValueError('workspace supports must be explicit distinct entities')
    all_entities=base_entities+support_entities
    if len({e['body_name'] for e in all_entities})!=len(all_entities):raise ValueError('overlapping workspace native bindings')
    for e in all_entities:
        if artifact_hashes(e['object_dir'])!=e['artifact_hashes']:raise ValueError('workspace artifact closure differs')
    def kwargs(e):return {k:v for k,v in e.items() if k!='artifact_hashes'}
    out,manifest=import_scope(xml,entities=[kwargs(e) for e in base_entities],scope='L1_target_destination',workspace_bounds_world_m=bounds)
    for e in support_entities:
        out,r=import_fixture_component(out,**kwargs(e));manifest['entities'].append(r)
        manifest['inventory'].append(dict(native_component_body=r['body_name'],classification='reconstructed_direct_physical_geoms',
            reconstruction_object_id=r['object_id'],removed_native_geoms=r['removed_original_geoms'],retained_reference_components=r['component_roster']))
    manifest.update(scope='DEV_partial_observed_workspace',requested_scope='L2_task_workspace',
        import_status='IMPORTED_NOT_EXECUTED',L2_READY=False,scope_admission='COMPLETENESS_NOT_ESTABLISHED',controller_decision=None,
        source_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),imported_xml_sha256=hashlib.sha256(out.encode()).hexdigest(),
        residual_occupied_components=coverage['residual_components'],unknown_samples=coverage['unknown_samples'],
        failure_classification='missing construction evidence for complete obstacle inventory',
        observed_support_limitation='partial TRAIN surfaces; replacing source basin direct geometry does not recover unseen walls',
        retained_native_inventory='all unreplaced bodies/world geoms listed; no obstacle absence certificate',
        primary_scope_episode=False)
    return out,manifest
