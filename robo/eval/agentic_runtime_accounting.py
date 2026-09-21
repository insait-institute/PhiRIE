"""Aggregate-only runtime attribution from sealed raw generation processes.

No controller input or decision is changed. Pool process time includes startup,
failed work and invalid outputs; object runtime is never added to that process
again. Reported costs are method attribution, not elapsed fleet or capture time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from robo.manifest.hash import canonical_hash

TOOLS = ('trellis', 'reconviagen')


def _identity(anchor):
    from robo.eval import agentic_ablation as e3
    path = Path(anchor['path'])
    if not path.is_absolute():
        raise ValueError('runtime provenance path must be absolute')
    e3._reject_symlink_components(path)
    if not path.is_file():
        raise ValueError('runtime provenance file is missing: ' + str(path))
    if e3.sha256_file(path) != anchor['sha256']:
        raise ValueError('runtime provenance content changed: ' + str(path))
    return path


def _pool(anchor, scene, tool):
    """Authenticate small provenance closure, without rehashing model/assets."""
    from robo.eval import agentic_ablation as e3
    path = _identity(anchor)
    e3.checked_repo_path(path, 'runtime pool', kind='file')
    expected_kind = 'trellis_initial' if tool == 'trellis' else 'rvg_initial'
    if path.name != 'proposal_pool.json' or path.parent.name != scene or path.parent.parent.name != expected_kind:
        raise ValueError('runtime pool scene/tool path differs')
    root = path.parent.parent.parent
    pool = json.loads(path.read_text())
    for field in ('code_commit', 'freeze_id'):
        if field in anchor and anchor[field] != pool[field]:
            raise ValueError('selected runtime producer identity differs')
    manifest_path = _identity(dict(path=str(path.parent/'input_manifest.json'), sha256=pool['input_manifest_sha256']))
    manifest = json.loads(manifest_path.read_text())
    records = _identity(dict(path=str(path.parent/'proposal_records.jsonl'), sha256=pool['proposal_records_sha256']))
    if ([r['job_id'] for r in pool['rows']] != [j['job_id'] for j in manifest['jobs']]
            or pool['rows'] != [json.loads(line) for line in records.read_text().splitlines() if line.strip()]):
        raise ValueError('runtime pool records differ')
    contract_path = e3.checked_repo_path(root/'contract/freeze_manifest.json', 'generation runtime E0', kind='file')
    contract = json.loads(contract_path.read_text())
    digest = canonical_hash({k:v for k,v in contract.items() if k not in {'created_utc','environment','contract_sha256'}})
    if (digest != contract['contract_sha256'] or contract['freeze_id'] != root.name
            or pool['freeze_id'] != root.name or contract['code']['dirty'] is not False
            or manifest['code_commit'] != pool['code_commit'] or pool['code_commit'] != contract['code']['commit']):
        raise ValueError('runtime generation E0/source differs')
    candidates = [r for r in contract['resource_inventory'] if r.get('kind') == 'experiment_config'
                  and r.get('sha256') == manifest['config_sha256']]
    if len(candidates) != 1:
        raise ValueError('runtime generation config binding ambiguous or missing')
    config_path = _identity(dict(path=candidates[0]['resolved_path'], sha256=manifest['config_sha256']))
    config = yaml.safe_load(config_path.read_text())
    if (str(config.get('output_scene_id', config.get('scene_id'))) != scene
            or config['freeze_id'] != root.name or pool['planned_jobs'] != len(pool['rows'])
            or len({r['job_id'] for r in pool['rows']}) != len(pool['rows'])
            or any(not r['job_id'].startswith(scene+':auto:') or r['tool'] != tool for r in pool['rows'])):
        raise ValueError('runtime scene/job/tool denominator differs')
    # Recovery ancestry is declared by the already frozen generator config.
    parents = set()
    replay = config.get('frozen_view_replay')
    if replay:
        source_config_path = _identity(replay['source_config'])
        original = yaml.safe_load(source_config_path.read_text())
        source_contract_path = _identity(replay['source_contract'])
        original_contract = json.loads(source_contract_path.read_text())
        if original_contract['code']['commit'] != replay['producer_commit'] or original['freeze_id'] != original_contract['freeze_id']:
            raise ValueError('runtime recovery parent source differs')
        parents.add(str(source_contract_path.parent.parent / expected_kind / scene / 'proposal_pool.json'))
    reuse = config.get('raw_reuse')
    if reuse:
        parents.add(str(_identity(reuse['anchors']['proposal_pool.json'])))
    for receipt_anchor in manifest.get('raw_reuse_receipts', []):
        receipt = json.loads(_identity(receipt_anchor).read_text())
        parents.add(str(_identity(receipt['original_source_anchors']['anchors']['proposal_pool.json'])))
    value = pool.get('wall_s')
    missing = None
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        value = None
        missing = 'missing_or_invalid_generation_process_timing'
    elif pool.get('generation_performed') is True and value == 0:
        value = None
        missing = 'zero_timing_for_executed_generation'
    if type(pool.get('generation_performed')) is not bool:
        value = None
        missing = 'missing_generation_execution_status'
    object_times = [r.get('runtime', {}).get('wall_s') for r in pool['rows']
                    if isinstance(r.get('runtime'), dict) and r['runtime'].get('status') == 'generated'
                    and r.get('generation_mode') != 'validated_raw_reuse']
    object_sum = sum(object_times) if all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in object_times) else None
    if type(pool.get('exit_code')) is not int:
        raise ValueError('runtime generation process is not terminal')
    return dict(path=str(path), sha256=anchor['sha256'], freeze_id=pool['freeze_id'],
                code_commit=pool['code_commit'], config_sha256=manifest['config_sha256'],
                input_manifest_sha256=pool['input_manifest_sha256'], contract_sha256=digest,
                planned_jobs=pool['planned_jobs'], job_ids=[r['job_id'] for r in pool['rows']],
                discovery_hashes=pool.get('source_discovery_hashes'),
                wall_s=value, missing_reason=missing, exit_code=pool['exit_code'],
                completed_object_runtime_records=len(object_times), completed_object_wall_s=object_sum,
                process_residual_wall_s=None if value is None or object_sum is None else value-object_sum,
                process_residual_scope='startup_and_process_overhead_plus_work_without_completed_object_timers',
                generation_performed=pool.get('generation_performed'),
                required_parent_pool_paths=sorted(parents),
                producer_status_counts={s:sum(r['status']==s for r in pool['rows']) for s in sorted({r['status'] for r in pool['rows']})})


def account_runtime(out_dir, jobs, control_by_scene, ledger):
    """Return a JSON sidecar. Optional histories live in E0-bound jobs YAML.

    runtime_accounting: {schema_version: 1, process_histories: [
      {scene_id: ..., tool: reconviagen, pools: [{path: ..., sha256: ...}, ...]}]}
    A history must include the selected pool once and its complete ancestry.
    Unknown timing/ancestry produces null+NOT_RUN; conflicting identities raise.
    """
    from robo.eval import agentic_ablation as e3
    e3._verify_sealed_directory(out_dir/'input_inventory', controller_safe=False)
    audit_path = out_dir/'input_inventory/inventory_audit.json'
    audit = json.loads(audit_path.read_text())
    scene_ids = sorted(control_by_scene)
    if 'scene_audits' in audit:
        audits = audit['scene_audits']
    elif len(scene_ids) == 1:
        audits = {scene_ids[0]: audit}
    else:
        audits = {}
    spec = jobs.get('runtime_accounting', {'schema_version': 1, 'process_histories': []})
    if set(spec) != {'schema_version','process_histories'} or spec['schema_version'] != 1:
        raise ValueError('runtime history schema differs')
    histories = {}
    for history in spec['process_histories']:
        if set(history) != {'scene_id','tool','pools'}:
            raise ValueError('runtime history keys differ')
        key = (str(history['scene_id']), history['tool'])
        if key in histories or key[0] not in scene_ids or key[1] not in TOOLS:
            raise ValueError('runtime history duplicated or outside population')
        histories[key] = history['pools']
    processes = {}; scene_tools = {}; missing = []
    for scene in scene_ids:
        for tool in TOOLS:
            selected = audits.get(scene, {}).get('initial_pools', {}).get(tool)
            key = (scene, tool)
            if selected is None or selected.get('status') == 'not_run':
                scene_tools[key] = None
                missing.append(dict(scene_id=scene, tool=tool, reason='initial_generation_not_run_or_unrecorded'))
                continue
            anchors = histories.get(key, [selected])
            paths = [str(Path(a['path'])) for a in anchors]
            if len(paths) != len(set(paths)) or paths.count(selected['path']) != 1:
                raise ValueError('runtime process double counted or selected pool omitted')
            if next(a['sha256'] for a in anchors if a['path']==selected['path']) != selected['sha256']:
                raise ValueError('selected runtime pool differs from sealed inventory')
            entries = [_pool(a, scene, tool) for a in anchors]
            by_path = {p['path']:p for p in entries}
            current = by_path[selected['path']]
            expected_jobs = {f'{scene}/obj_{raw.rsplit(":auto:",1)[1]}' for raw in current['job_ids']}
            for policy in e3.POLICY_IDS:
                actual_jobs = [r['job_id'] for r in ledger if r['scene_id']==scene and r['policy_id']==policy]
                if len(actual_jobs) != len(expected_jobs) or set(actual_jobs) != expected_jobs:
                    raise ValueError('runtime pool differs from canonical planned jobs')
            if any(p['job_ids'] != current['job_ids'] or p['discovery_hashes'] != current['discovery_hashes'] for p in entries):
                raise ValueError('runtime recovery changed original planned inputs')
            reachable = set(); active = set(); absent = set()
            def visit(path):
                if path in active: raise ValueError('runtime recovery ancestry cycle')
                if path in reachable: return
                if path not in by_path: absent.add(path); return
                active.add(path)
                for parent in by_path[path]['required_parent_pool_paths']: visit(parent)
                active.remove(path); reachable.add(path)
            visit(selected['path'])
            if set(paths) != reachable:
                raise ValueError('runtime history has unrelated process or duplicate cost')
            for p in entries:
                if p['path'] in processes and processes[p['path']] != p:
                    raise ValueError('runtime process identity conflict')
                processes[p['path']] = p
                if p['missing_reason']: missing.append(dict(scene_id=scene, tool=tool, path=p['path'], reason=p['missing_reason']))
            if absent: missing.append(dict(scene_id=scene, tool=tool, reason='unanchored_recovery_process_history', paths=sorted(absent)))
            scene_tools[key] = None if absent or any(p['wall_s'] is None for p in entries) else sum(p['wall_s'] for p in entries)
    rows = []
    for policy in e3.POLICY_IDS:
        invoked = TOOLS[:1] if policy == 'A0' else TOOLS
        values = [scene_tools[s,t] for s in scene_ids for t in invoked]
        generation = None if any(v is None for v in values) else sum(values)
        invoked_canonical = set(invoked) | ({'registration_retry'} if policy in ('A3','A4') else set())
        bad_canonical = [dict(scene_id=s, policy_id=policy, job_id=p.get('job_id'), reason='missing_or_invalid_canonical_timing')
                         for s,shard in control_by_scene.items()
                         for p in shard['proposals'] + shard.get('tool_failures', [])
                         if p.get('tool') in invoked_canonical and (type(p.get('wall_s')) not in (int,float)
                            or not math.isfinite(p['wall_s']) or p['wall_s'] < 0)]
        missing.extend(bad_canonical)
        canonical = None if bad_canonical else sum(e3._policy_runtime_seconds(policy, r['job_id'], control_by_scene[r['scene_id']], r)
                        for r in ledger if r['policy_id']==policy)
        rows.append(dict(policy_id=policy, planned_scenes=len(scene_ids),
                         canonical_proposal_and_retry_wall_s=canonical,
                         attributed_generation_process_wall_s=generation,
                         attributed_algorithm_phase_wall_s=None if generation is None or canonical is None else generation+canonical,
                         runtime_accounting_status='NOT_RUN' if generation is None or canonical is None else 'PASS'))
    return dict(schema_version=1, scope='attributed_initial_generation_plus_canonical_registration_physics_and_retries',
                construction_freeze_id=out_dir.parent.name, inventory_audit_sha256=e3.sha256_file(audit_path),
                rows=rows, generation_processes=list(processes.values()), missing_evidence=missing,
                actual_new_wave_incremental_wall_s=None, cache_materialization_overhead_wall_s=None,
                incremental_runtime_status='NOT_RUN_separate_execution_timing_not_measured',
                exclusions=['discovery','capture','source_Gaussian_training','observation_export','evaluation','scheduler_queue','cache_materialization'],
                accounting_rules=['Each original and recovery process charged once per scene and invoked tool, including failures and invalid/unavailable outputs.',
                                  'Per-object generator wall_s is contained in process wall_s and is never added again.',
                                  'A0 charges TRELLIS; A1-A4 charge both generators. Shared physical processes across policies are method attribution, never fleet elapsed time.',
                                  'Existing canonical per-policy registration/physics/retry accounting is preserved.'],
                paper_ready=False)
