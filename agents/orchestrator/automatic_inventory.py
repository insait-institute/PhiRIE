"""Normalize source-bound automatic jobs for the canonical E3 controller.

Inventory-only: model/runtime receipts stay in the audit sidecar. The controller
receives no labels, source scores, GT-shaped metadata keys, or evaluation data.
"""
from __future__ import annotations
import json
import math
from pathlib import Path


def terminal_unavailable_reason(original, exit_code):
    """Keep failure reasons truthful about the original per-object record."""
    if not original['prepared']:
        return 'preparation_unavailable'
    runtime = original.get('runtime')
    if runtime is None:
        return 'no_per_object_completion_record_process_exit_'+str(exit_code)
    status = runtime.get('status')
    if status == 'generation_failed':
        return 'producer_reported_generation_failure'
    if status == 'unavailable':
        return 'view_eligibility_unavailable'
    if status == 'generated':
        return 'producer_reported_incomplete_generation'
    raise ValueError('unknown terminal per-object runtime status')


def terminal_pool_rows(directory, pool, manifest, spec):
    """Authenticate terminal failures and quarantine invalid artifacts in memory.

    This is an inventory adapter, not a generator or acceptance policy. Raw
    producer records remain authoritative and are never rewritten or relabeled.
    """
    import copy
    from collections import Counter
    import numpy as np
    from plyfile import PlyData
    from robo.eval import agentic_ablation as e3
    from robo.manifest.hash import canonical_hash

    root = e3.checked_repo_path(spec['freeze_root'], 'terminal producer freeze', kind='dir')
    directory = e3.checked_repo_path(directory, 'terminal pool directory', kind='dir')
    relative = directory.relative_to(root)
    if len(relative.parts) != 2 or relative.parts[0] not in {'trellis_initial','rvg_initial'}:
        raise ValueError('terminal pool must belong to its declared per-scene producer freeze')
    scene = e3._require_scene_id(relative.parts[1])
    path = e3.checked_repo_path(spec['path'], 'terminal producer audit', kind='file')
    path.relative_to(root)
    if e3.sha256_file(path) != spec['sha256']:
        raise ValueError('terminal audit hash differs')
    audit = json.loads(path.read_text())
    contract_path = e3.checked_repo_path(root/'contract/freeze_manifest.json', 'terminal producer E0', kind='file')
    contract = json.loads(contract_path.read_text())
    digest = canonical_hash({k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}})
    if (digest != contract['contract_sha256'] or contract['code']['dirty'] is not False
            or contract['code']['commit'] != pool['code_commit']
            or contract['freeze_id'] != pool['freeze_id'] or root.name != pool['freeze_id']
            or not any(r.get('kind') == 'experiment_config' and r.get('sha256') == manifest['config_sha256']
                       for r in contract['resource_inventory'])):
        raise ValueError('terminal producer E0/code/config binding differs')
    if (audit.get('schema_version') != 1 or audit.get('paper_ready') is not False
            or audit.get('source_checkout_clean') is not True
            or audit.get('independent_source_binding_pass') is not True
            or audit['producer_source_commit'] != pool['code_commit']
            or manifest['code_commit'] != pool['code_commit']
            or audit['producer_freeze_id'] != pool['freeze_id']
            or audit['scene_id'] != scene or audit['config_sha256'] != manifest['config_sha256']
            or audit['source_discovery_hashes'] != manifest['source_discovery_hashes']
            or audit['planned_jobs'] != pool['planned_jobs']
            or audit['producer_process_exit_code'] != pool['exit_code']):
        raise ValueError('terminal audit producer identity differs')
    closure = ('input_manifest.json','proposal_pool.json','proposal_records.jsonl','execution_status.json','generation.log')
    if relative.parts[0] == 'rvg_initial':
        closure = ('input_manifest.json','proposal_pool.json','proposal_records.jsonl',
                   'execution_claim.json','generation.log','runtime.json',
                   'view_manifest.json','views_receipt.json')
    for name in closure:
        member = e3.checked_repo_path(directory/name, 'terminal source closure', kind='file')
        if e3.sha256_file(member) != audit[name.replace('.','_')+'_sha256']:
            raise ValueError('terminal audit source closure differs: '+name)
    if relative.parts[0] == 'rvg_initial':
        from run.icra2027 import e3_rvg_generation_pilot as rvg
        from run.icra2027.e3_fresh_generation_contract import checked_identity
        import yaml
        config_path = checked_identity(audit['producer_config'])
        if e3.sha256_file(config_path) != manifest['config_sha256']:
            raise ValueError('RVG terminal producer config differs')
        config = yaml.safe_load(config_path.read_text())
        original_manifest, boundary = rvg.read_manifest(config, directory)
        if original_manifest != manifest:
            raise ValueError('RVG terminal staged manifest differs')
        rvg.validate_view_receipt(config, directory, manifest, boundary)
        if (pool.get('view_manifest_sha256') != audit['view_manifest_json_sha256']
                or pool.get('views_receipt_sha256') != audit['views_receipt_json_sha256']):
            raise ValueError('RVG terminal view closure differs')
        claim = json.loads((directory/'execution_claim.json').read_text())
        runtime = json.loads((directory/'runtime.json').read_text())
        if (claim.get('code_commit') != pool['code_commit']
                or claim.get('input_manifest_sha256') != pool['input_manifest_sha256']
                or claim.get('views_receipt_sha256') != pool['views_receipt_sha256']
                or str(runtime.get('job_id')) != str(audit['job_id'])):
            raise ValueError('RVG terminal execution claim differs')
    if (pool['input_manifest_sha256'] != audit['input_manifest_json_sha256']
            or pool['proposal_records_sha256'] != audit['proposal_records_jsonl_sha256']
            or [json.loads(line) for line in (directory/'proposal_records.jsonl').read_text().splitlines()] != pool['rows']):
        raise ValueError('terminal original proposal records differ')
    scheduler = [line.split('|') for line in audit['scheduler'].splitlines()
                 if line.split('|')[0] == str(audit['job_id'])]
    if len(scheduler) != 1 or len(scheduler[0]) < 3 or scheduler[0][1] not in {'COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY'}:
        raise ValueError('terminal audit lacks terminal scheduler evidence')
    if scheduler[0][1] == 'COMPLETED' and (scheduler[0][2] != '0:0' or pool['exit_code'] != 0):
        raise ValueError('completed terminal producer exit differs')
    if [r['job_id'] for r in audit['rows']] != [r['job_id'] for r in pool['rows']]:
        raise ValueError('terminal audit lost or reordered planned jobs')
    normalized = []
    for original, row in zip(pool['rows'], audit['rows']):
        if original['status'] not in {'available','generation_failed','unavailable'}:
            raise ValueError('unknown original producer status')
        if (row['proposal_id'] != original['proposal_id'] or row['producer_status'] != original['status']
                or row['per_object_attempt_proven'] != bool(original.get('runtime'))):
            raise ValueError('terminal audit changed original job attribution')
        errors, verified = [], []
        for name, identity in original['artifacts'].items():
            artifact = e3.checked_repo_path(identity['path'], 'terminal artifact', must_exist=False)
            artifact.relative_to(directory)
            if not artifact.is_file():
                errors.append({'artifact':name,'reason':'missing'}); continue
            if artifact.stat().st_size != identity['bytes'] or e3.sha256_file(artifact) != identity['sha256']:
                errors.append({'artifact':name,'reason':'hash_or_size_mismatch'}); continue
            verified.append({'artifact':name, 'path':str(artifact), 'sha256':identity['sha256'], 'bytes':identity['bytes']})
            if name.endswith('.ply'):
                try:
                    vertices = PlyData.read(str(artifact))['vertex'].data
                    if not len(vertices): errors.append({'artifact':name,'reason':'empty_vertices'})
                    for field in vertices.dtype.names:
                        values = vertices[field]
                        if np.issubdtype(values.dtype,np.number) and not np.isfinite(values).all():
                            errors.append({'artifact':name,'reason':'nonfinite_numeric_field','field':field,
                                'nan_count':int(np.isnan(values).sum()),'positive_inf_count':int(np.isposinf(values).sum()),
                                'negative_inf_count':int(np.isneginf(values).sum())})
                except Exception as error:
                    errors.append({'artifact':name,'reason':'ply_parse_failure','exception':repr(error)})
        if original['status'] == 'available':
            runtime = original.get('runtime')
            if not runtime or runtime.get('status') != 'generated' or runtime.get('seed') != 42:
                errors.append({'reason':'missing_valid_generated_runtime'})
            status = 'artifact_invalid' if errors else 'available_verified'
            reason = 'artifact_validation_failed' if errors else None
        else:
            status = 'unavailable'
            reason = terminal_unavailable_reason(original, pool['exit_code'])
        if (row['normalized_status'] != status or row['reason'] != reason
                or sorted(row['artifact_validation_errors'],key=lambda r:json.dumps(r,sort_keys=True)) != sorted(errors,key=lambda r:json.dumps(r,sort_keys=True))
                or sorted(row['verified_artifacts'],key=lambda r:json.dumps(r,sort_keys=True)) != sorted(verified,key=lambda r:json.dumps(r,sort_keys=True))):
            raise ValueError('terminal per-object artifact audit does not replay exactly')
        converted = copy.deepcopy(original)
        if status != 'available_verified':
            converted.update(status='unavailable', reason=reason, artifacts={})
        normalized.append(converted)
    if dict(Counter(r['normalized_status'] for r in audit['rows'])) != audit['counts']:
        raise ValueError('terminal audit count denominator differs')
    return normalized, {'path':str(path), 'sha256':spec['sha256'], 'freeze_root':str(root),
        'counts':audit['counts'], 'scheduler':audit['scheduler'],
        'producer_process_exit_code':pool['exit_code'],
        'process_failure_classification':audit['process_failure_classification']}


def build_payloads(jobs, contract, freeze_id):
    """Combine the complete frozen scene roster through the same single-scene producer."""
    from robo.eval import agentic_ablation as e3

    sources = jobs.get('automatic_sources')
    if isinstance(sources, dict):
        return _build_single(jobs, contract, freeze_id)
    if not isinstance(sources, list) or not sources:
        raise ValueError('automatic sources must be a scene declaration or nonempty scene list')
    population = jobs.get('population', {})
    for key in ('scene_roster_config_file_sha256', 'scene_roster_sha256'):
        e3._require_sha256(population.get(key), key)
    scene_ids = e3._scene_roster(jobs)
    if any(not isinstance(source, dict) for source in sources) or [s.get('scene_id') for s in sources] != scene_ids:
        raise ValueError('automatic source list must preserve the complete frozen scene roster and order')
    expected = {key: population.get(key) for key in (
        'planned_scenes', 'planned_jobs_per_policy', 'planned_policy_object_rows')}
    if any(type(value) is not int or value < 0 for value in expected.values()):
        raise ValueError('automatic population counts must be nonnegative integers')
    bundles = [_build_single({**jobs, 'automatic_sources': source}, contract, freeze_id)
               for source in sources]
    result = dict(bundles[0][0])
    result['scenes'] = [scene for bundle in bundles for scene in bundle[0]['scenes']]
    result['counts'] = {'scenes': len(scene_ids),
        'jobs': sum(bundle[0]['counts']['jobs'] for bundle in bundles),
        'policy_object_rows': sum(bundle[0]['counts']['policy_object_rows'] for bundle in bundles)}
    if result['counts'] != {'scenes': expected['planned_scenes'],
        'jobs': expected['planned_jobs_per_policy'], 'policy_object_rows': expected['planned_policy_object_rows']}:
        raise ValueError('complete automatic population differs from frozen scene/object/row counts')
    result['source_contract'] = {**result['source_contract'], 'jobs_sha256': e3._canonical_digest(jobs)}
    proposals = {**bundles[0][1], 'proposals': [row for bundle in bundles for row in bundle[1]['proposals']],
        'counts': {key: sum(bundle[1]['counts'][key] for bundle in bundles) for key in bundles[0][1]['counts']}}
    e3._validate_controller_inventory_schema(result, proposals)
    e3._check_no_forbidden_fields(result); e3._check_no_forbidden_fields(proposals)
    provenance = {bundle[2]['source_gaussian_training_provenance'] for bundle in bundles}
    audit = {'schema_version': 2, 'freeze_id': freeze_id, 'paper_ready': False,
        'source_gaussian_training_provenance': next(iter(provenance)) if len(provenance) == 1 else 'MIXED',
        'initial_pool_complete': all(bundle[2]['initial_pool_complete'] for bundle in bundles),
        'evaluation_status': 'NOT_RUN', 'evaluation_references': [],
        'counts': result['counts'], 'scene_roster': scene_ids,
        'scene_audits': {scene_id: bundle[2] for scene_id, bundle in zip(scene_ids, bundles)},
        'runtime_accounting_status': bundles[0][2]['runtime_accounting_status']}
    return result, proposals, audit, ''.join(bundle[3] for bundle in bundles)


def _build_single(jobs, contract, freeze_id):
    from robo.eval import agentic_ablation as e3
    from run.icra2027.e3_trellis_generation_pilot import source_jobs

    if jobs.get('study_scope') != 'automatic_training_only_engineering' or jobs.get('paper_ready') is not False:
        raise ValueError('automatic inventory is engineering only')
    source = jobs['automatic_sources']
    directory = e3.checked_repo_path(source['discovery_directory'], 'automatic discovery source', kind='dir')
    required_source_members = {'pilot_summary.json', 'output_hashes.json', 'input_manifest.json', 'postrun_audit.json'}
    if 'all_jobs_manifest.json' in source['discovery_hashes']:
        required_source_members.add('all_jobs_manifest.json')
    for name, digest in source['discovery_hashes'].items():
        if name not in required_source_members:
            raise ValueError('undeclared automatic source member')
        if e3.sha256_file(directory/name) != digest:
            raise ValueError('automatic discovery source changed')
    if set(source['discovery_hashes']) != required_source_members:
        raise ValueError('automatic source closure incomplete')
    original = source_jobs(directory)
    manifest = json.loads((directory/'input_manifest.json').read_text())
    scene_id = e3._require_scene_id(source['scene_id'])
    provenance = manifest.get('source_gaussian_training_provenance', 'UNKNOWN')
    planned = source['planned_jobs']
    if type(planned) is not int or planned < 0 or len(original) != planned:
        raise ValueError('predeclared pilot scene or automatic denominator changed')
    if provenance == 'FRESH_OFFICIAL_TRAIN_ONLY':
        if 'all_jobs_manifest.json' not in required_source_members:
            raise ValueError('fresh automatic population requires a frozen all-jobs anchor')
        complete = json.loads((directory/'all_jobs_manifest.json').read_text())
        if (complete['scene_id'] != scene_id or complete['planned_jobs'] != planned
                or manifest.get('scene_id') != scene_id):
            raise ValueError('fresh automatic population scene or denominator differs')
        # source_jobs authenticates this manifest against the producer completion
        # seal and actual segmentation population, including unprepared instances.
    elif provenance != 'UNKNOWN' or scene_id != '09c1414f1b' or planned != 6:
        raise ValueError('legacy automatic pilot identity changed')
    if any(r['job_id'] != f"{scene_id}:auto:{r['automatic_instance_id']}" for r in original):
        raise ValueError('automatic source job scene binding differs')
    by_job = {r['job_id']: r for r in original}
    pools = {}
    pool_audit = {}
    for tool in ('trellis', 'reconviagen'):
        spec = source['initial_pools'][tool]
        if spec.get('status') == 'not_run':
            if set(spec) != {'status', 'reason_code'} or spec['reason_code'] != 'initial_tool_not_run':
                raise ValueError('unrun initial tool must be explicitly typed')
            pools[tool] = {}
            pool_audit[tool] = dict(spec)
            continue
        path = e3.checked_repo_path(spec['path'], 'initial proposal pool', kind='file')
        if e3.sha256_file(path) != spec['sha256']:
            raise ValueError('initial proposal pool changed')
        pool = json.loads(path.read_text())
        generation_manifest_path = path.parent / 'input_manifest.json'
        if e3.sha256_file(generation_manifest_path) != pool['input_manifest_sha256']:
            raise ValueError('generation input/model manifest changed')
        generation_manifest = json.loads(generation_manifest_path.read_text())
        if generation_manifest['code_commit'] != pool['code_commit'] or not generation_manifest['models']:
            raise ValueError('generation source/checkpoint provenance incomplete')
        if provenance == 'FRESH_OFFICIAL_TRAIN_ONLY':
            # A single equal RGBA does not authenticate RVG's other views and
            # geometry-derived masks. Fresh proposals need the complete source
            # closure in their generation receipt; legacy pools fail closed.
            if (generation_manifest.get('source_gaussian_training_provenance') != provenance
                    or pool.get('source_gaussian_training_provenance') != provenance
                    or generation_manifest.get('source_discovery_hashes') != source['discovery_hashes']):
                raise ValueError('fresh generation lacks exact complete discovery source binding')
        normalized_rows = pool['rows']
        terminal_provenance = None
        if 'terminal_audit' in spec:
            normalized_rows, terminal_provenance = terminal_pool_rows(path.parent, pool, generation_manifest, spec['terminal_audit'])
        rows = {r['job_id']: r for r in normalized_rows}
        if len(rows) != len(pool['rows']) or set(rows) != set(by_job) or pool['planned_jobs'] != len(original):
            raise ValueError('initial proposal pool lost or duplicated planned jobs')
        for job_id, row in rows.items():
            origin = by_job[job_id]
            if (row['automatic_instance_id'], row['prepared']) != (origin['automatic_instance_id'], origin['prepared']):
                raise ValueError('initial proposal job binding differs')
            if not row['prepared'] and row['status'] == 'available':
                raise ValueError('unprepared job cannot have a generated proposal')
            if row['prepared'] and row['input_sha256'] != origin['input']['sha256']:
                raise ValueError('generation input differs from automatic crop')
            if row.get('tool') != tool or row.get('seed') != 42 or row.get('shared_initial_policy_rows') != ['A1','A2','A3','A4']:
                raise ValueError('initial tool/seed/shared-pool treatment differs')
            if row['status'] == 'available':
                runtime = row.get('runtime')
                if not isinstance(runtime, dict) or runtime.get('status') != 'generated' or not math.isfinite(runtime.get('wall_s', float('nan'))) or runtime['wall_s'] < 0:
                    raise ValueError('available generation lacks measured runtime receipt')
            if row['status'] not in {'available', 'generation_failed', 'unavailable'}:
                raise ValueError('unknown initial proposal status')
        pools[tool] = rows
        pool_audit[tool] = {'path': str(path), 'sha256': spec['sha256'],
                            'code_commit': pool['code_commit'], 'freeze_id': pool['freeze_id']}
        if terminal_provenance is not None:
            pool_audit[tool]['terminal_audit'] = terminal_provenance
    scene_sources = {}
    for role, identity in source['scene_sources'].items():
        if role not in {'gaussian', 'intrinsics', 'poses'}:
            raise ValueError('unexpected scene source role')
        # Content identities are checked again by the canonical observation phase.
        e3._exact_keys(identity, {'path', 'size_bytes', 'sha256'}, 'automatic scene source')
        e3._require_sha256(identity['sha256'], role)
        scene_sources[role] = dict(identity)
    if set(scene_sources) != {'gaussian','intrinsics','poses'}:
        raise ValueError('automatic scene observation sources incomplete')
    # The engineering pilot must use the very same Gaussian and camera sources
    # that generated discovery, never quietly swap in a later reconstruction.
    expected = {'gaussian': manifest['gaussian'],
                'intrinsics': manifest['metadata']['nerfstudio/transforms_undistorted.json'],
                'poses': manifest['metadata']['colmap/images.txt']}
    for role, identity in scene_sources.items():
        source_identity = expected[role]
        if (identity['sha256'], identity['size_bytes'], identity['path']) != (source_identity['sha256'], source_identity['bytes'], source_identity['path']):
            raise ValueError('observation source differs from discovery construction')
    normalized_jobs=[];normalized_proposals=[];audit_jobs=[]
    for origin in original:
        slot=e3._require_object_slot(f"obj_{origin['automatic_instance_id']}")
        job_id=f'{scene_id}/{slot}'
        prepared=origin['prepared']
        evidence={'observation_status':'available' if prepared else 'unavailable',
            'reason_code':None if prepared else 'preparation_unavailable',
            'source_frame':origin['frame'] if prepared else None,
            'bbox_px':origin['object_metadata']['bbox_px'] if prepared else None,
            'mask_provenance':'automatic_training_only' if prepared else None}
        rgba=e3.checked_repo_path(origin['input']['path'],'automatic RGBA',kind='file') if prepared else None
        job={'freeze_id':freeze_id,'scene_id':scene_id,'object_slot':slot,'job_id':job_id,
             'artifact_paths':{'rgba':rgba.relative_to(e3.REPOSITORY_ROOT).as_posix()} if prepared else {},
             'artifact_hashes':{'rgba':origin['input']['sha256']} if prepared else {},
             'construction_evidence':evidence}
        normalized_jobs.append(job)
        audit_jobs.append({'job_id':job_id,'source_job_id':origin['job_id'],
                           'automatic_instance_id':origin['automatic_instance_id'],
                           'prepared_output_index':origin.get('output_index')})
        for tool in ('trellis','reconviagen'):
            row=pools[tool].get(origin['job_id'])
            available=bool(prepared and row and row['status']=='available')
            artifacts={}
            if available:
                names=('trellis_mesh.ply','trellis_gs.ply') if tool=='trellis' else ('rvg_mesh.ply','rvg_gs.ply')
                for role,name in zip(('raw_mesh','raw_gaussian'),names):
                    identity=row['artifacts'][name]
                    path=e3.checked_repo_path(identity['path'],f'generated {role}',kind='file')
                    if path.stat().st_size != identity['bytes'] or e3.sha256_file(path)!=identity['sha256']:
                        raise ValueError('generated artifact identity changed')
                    artifacts[role]={'path':path.relative_to(e3.REPOSITORY_ROOT).as_posix(),
                                     'size_bytes':identity['bytes'],'sha256':identity['sha256']}
            reason=('preparation_unavailable' if not prepared else
                    'initial_tool_not_run' if row is None else row.get('reason') or 'generation_failed')
            normalized_proposals.append({**{k:job[k] for k in ('freeze_id','job_id','scene_id','object_slot')},
                'proposal_id':f'{job_id}:{tool}','tool_id':tool,
                'availability':'available' if available else 'typed_unavailable',
                'artifact_paths':{k:v['path'] for k,v in artifacts.items()},
                'artifact_hashes':{k:v['sha256'] for k,v in artifacts.items()},
                'artifact_sizes':{k:v['size_bytes'] for k,v in artifacts.items()},
                'construction_evidence':{} if available else {'typed_failure':reason,'missing_artifacts':['raw_mesh','raw_gaussian']},
                'raw_generator_provenance':{'artifact_bytes_hash_frozen':available,
                    'generator_commit':pool_audit[tool].get('code_commit') if available else None,
                    'checkpoint_identity_recorded':available,'runtime_manifest_recorded':available,
                    'claim_status':'engineering_only' if available else 'not_run'}})
    result={'schema_version':2,'freeze_id':freeze_id,'study_scope':jobs['study_scope'],
        'counts':{'scenes':1,'jobs':len(original),'policy_object_rows':len(original)*5},
        'source_contract':{'jobs_sha256':e3._canonical_digest(jobs),'jobs_hash_method':'canonical_structured_sha256',
            'contract_sha256':contract['contract_sha256'],'contract_hash_method':'canonical_json_sha256',
            'contract_freeze_id':contract['freeze_id'],'code_commit':contract['code']['commit']},
        'observation_protocol':dict(e3.OBSERVATION_PROTOCOL),'scenes':[{'scene_id':scene_id,
            'source_scene_gaussian':scene_sources['gaussian'],
            'camera_artifacts':{k:scene_sources[k] for k in ('intrinsics','poses')},'jobs':normalized_jobs}]}
    proposals={'schema_version':2,'freeze_id':freeze_id,'counts':{'jobs':len(original),
        'trellis_available':sum(r['tool_id']=='trellis' and r['availability']=='available' for r in normalized_proposals),
        'reconviagen_available':sum(r['tool_id']=='reconviagen' and r['availability']=='available' for r in normalized_proposals),
        'reconviagen_typed_unavailable':sum(r['tool_id']=='reconviagen' and r['availability']=='typed_unavailable' for r in normalized_proposals)},
        'proposals':normalized_proposals}
    e3._validate_controller_inventory_schema(result,proposals)
    e3._check_no_forbidden_fields(result);e3._check_no_forbidden_fields(proposals)
    audit={'schema_version':2,'freeze_id':freeze_id,'paper_ready':False,
        'source_gaussian_training_provenance':provenance,
        'initial_pool_complete':all('status' not in pool_audit[t] and all(
            not row['prepared'] or row.get('reason') != 'initial_tool_not_run'
            for row in pools[t].values()) for t in pools),
        'evaluation_status':'NOT_RUN','evaluation_references':[],
        'runtime_accounting_status':'generation receipts retained; aggregate must join generation overhead before paper use',
        'source_discovery':source['discovery_hashes'],'initial_pools':pool_audit,'jobs':audit_jobs}
    hashes=''.join(f"{digest}  {row['artifact_paths'][role]}\n" for row in normalized_jobs+normalized_proposals
                   for role,digest in sorted(row['artifact_hashes'].items()))
    return result,proposals,audit,hashes
