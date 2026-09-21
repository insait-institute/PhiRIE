"""Authenticate full E4 coverage for paper formatting without replaying physics."""
from collections import Counter
import json
import math
from pathlib import Path

from run.icra2027 import e4_planning_terminal as terminal

MEMBERS = {'gate.json', 'scene_receipts.json', 'planned_qualification_cells.jsonl'}
FIXED = dict(planned_scenes=50, planned_objects=1871, input_semantic_queries=6155,
             budget_exclusions=5886, planned_semantic_queries=269,
             planned_qualification_cells=2690, policy_target_episodes=160)


def require(condition, message):
    if not condition:
        raise ValueError('full qualification paper source: ' + message)


def checked(ref):
    path = Path(ref['path'])
    require(path.is_absolute() and path.resolve(strict=True) == path, 'aliased metadata path')
    actual = terminal.api.identity(path)
    require(actual['sha256'] == ref['sha256']
            and ('size_bytes' not in ref or actual['size_bytes'] == ref['size_bytes']),
            'metadata bytes changed')
    return path


def validate_metadata(gate, scenes, cells, protocol, config):
    """Recount logical records; this function performs no metric or physics work."""
    require(gate['scope'] == terminal.FULL_SCOPE and gate['schema_version'] == 1
            and gate['status'] == 'PASS' and all(gate[k] == v for k, v in FIXED.items()),
            'scope or fixed denominators differ')
    require(gate['paper_ready'] is False and gate['headline_eligible'] is False
            and gate['claim_gate'] == gate['camera_status'] == 'NOT_RUN'
            and gate['policy_success'] is None and gate['rollout_ledger'] is None
            and gate['instantiated_policy_episodes'] == gate['policy_executed'] == 0,
            'unsupported policy or scientific promotion')
    tasks = {q['task_id']: q['scene_id'] for q in protocol['qualification_tasks']}
    require(len(tasks) == len(protocol['qualification_tasks']) == 269,
            'original task roster differs')
    require(len(scenes) == 50 and set(scenes) == set(config['jobs'])
            and all(s['status'] in terminal.FULL_STATES for s in scenes.values()),
            'scene roster or status differs')
    expected = {(sid, tid, arm, ep) for tid, sid in tasks.items()
                for arm in ('A0', 'A4') for ep in range(5)}
    keys = [(r['scene_id'], r['task_id'], r['policy_id'], r['episode']) for r in cells]
    require(len(keys) == len(set(keys)) == 2690 and set(keys) == expected,
            'planned cells omitted, duplicated or substituted')
    for row in cells:
        require(row['cell_id'] == f"{row['policy_id'].lower()}__{row['task_id']}__seed0__ep{row['episode']}"
                and row['policy_executed'] is False and row['policy_success'] is None,
                'logical identity or unmeasured policy telemetry differs')
        if row['source_kind'] == 'canonical_qualifier':
            evidence = row['qualification_evidence']
            require(evidence['cell_id'] == row['cell_id'] and type(evidence['passed']) is bool
                    and row['qualification_state'] == ('PASS' if evidence['passed'] else 'FAIL'),
                    'canonical prerequisite evidence differs')
        else:
            require(row['source_kind'] == scenes[row['scene_id']]['status'].lower()
                    and row['qualification_state'] == 'NOT_RUN'
                    and all(row.get(k) is None for k in
                            ('qualification_evidence', 'reset_definition', 'camera', 'physics')),
                    'unavailable cell has fabricated measurements')
    for sid, scene in scenes.items():
        selected = sorted(t for t, s in tasks.items() if s == sid)
        rows = [r for r in cells if r['scene_id'] == sid]
        measured = [r for r in rows if r['source_kind'] == 'canonical_qualifier']
        require(scene['selected_task_ids'] == selected and scene['planned_cells'] == len(rows)
                and scene['qualified_cells'] == sum(r['qualification_state'] == 'PASS' for r in measured)
                and type(scene['actual_900_step_cells']) is int
                and 0 <= scene['actual_900_step_cells'] <= len(measured)
                and bool(measured) == (scene['status'] == 'QUALIFICATION_COMPLETE')
                and scene['policy_executed'] == 0 and scene['policy_success'] is None,
                'scene summary differs from logical cells')
        require(scene['status'] != 'SOURCE_NO_QUERIES' or not selected,
                'nonempty scene relabeled source-only')
        strict = scene['strict_pass_task_ids']
        require(len(strict) == len(set(strict)) and set(strict) <= set(selected)
                and all(sum(r['task_id'] == task and r['qualification_state'] == 'PASS'
                            for r in measured) == 10 for task in strict),
                'strict task membership differs')
    counts = dict(scene_status_counts=dict(Counter(s['status'] for s in scenes.values())),
        cell_state_counts=dict(Counter(r['qualification_state'] for r in cells)),
        cell_source_counts=dict(Counter(r['source_kind'] for r in cells)),
        prerequisite_checked_cells=sum(r['source_kind'] == 'canonical_qualifier' for r in cells),
        actual_900_step_cells=sum(s['actual_900_step_cells'] for s in scenes.values()),
        qualified_cells=sum(s['qualified_cells'] for s in scenes.values()),
        strict_pass_task_ids=sorted(t for s in scenes.values() for t in s['strict_pass_task_ids']))
    require(all(gate[k] == v for k, v in counts.items()), 'aggregate counts differ')
    attempts = list(config['jobs'].values()) + [r for rows in config.get('prior_attempts', {}).values() for r in rows]
    require(all(type(r['elapsed_seconds']) is int and r['elapsed_seconds'] >= 0 for r in attempts)
            and gate['scheduler_elapsed_seconds'] == sum(r['elapsed_seconds'] for r in attempts)
            and gate['prior_attempts'] == config.get('prior_attempts', {}),
            'failed-attempt runtime attribution differs')


def _contexts(qa, gate, out):
    """Validate existing E0/source metadata APIs, without invoking worker/replay."""
    source = qa['source']
    code = terminal._exact_code(source['repository'], source['commit'])
    require(source['dirty'] is False and gate['producer_code'] == code, 'producer source differs')
    config_path = checked(qa['config']); e0_path = checked(qa['E0'])
    require(gate['audit_config'] == qa['config'] and gate['audit_E0'] == qa['E0'],
            'stage config/E0 identity differs')
    config, stage = terminal._validate_full_stage(config_path, e0_path.parent.parent,
                                                   source['commit'], out, code)
    e0 = terminal.api.read(e0_path)
    require(terminal.api.resource(e0, 'terminal_driver') == qa['driver'],
            'independent replay driver is not E0-bound')
    original = gate['source']
    require(config['source'] == original and terminal.full_source_contract(
            original['config']['path'], Path(original['e0']['path']).parent.parent,
            original['code_commit']) == original, 'original qualification source differs')
    expected = dict(qualification_config=original['config'], qualification_E0=original['e0'],
                    qualification_python=original['runtime'],
                    qualification_environment=original['environment'],
                    compatibility_envelope=config['compatibility'])
    require(qa['source_evidence'] == expected, 'original evidence identity differs')
    for ref in [*expected.values(), original['osmesa_environment']]:
        checked(ref)
    original_config = terminal.api.read(original['config']['path'])
    protocol = terminal.api.read(checked(original_config['source']['protocol']))
    require(set(original_config['source']['scenes']) == set(config['jobs'])
            and all(original_config[k] == v for k, v in FIXED.items() if k != 'policy_target_episodes'),
            'original scene roster or denominators differ')
    return config, protocol, stage


def validate_source(spec, path, gate, audit_path, qa):
    """Audit SHA is pinned by the paper freeze; geometry is never recomputed."""
    require(checked(spec) == path and checked(spec['completion_audit']) == audit_path,
            'source path differs')
    out = path.parent
    require(path.name == 'gate.json' and out.name == 'full_qualification'
            and audit_path.name == 'completion_audit.json' and qa['full_output'] == str(out),
            'unexpected source entrypoint')
    require(qa['schema_version'] == 1 and qa['status'] == 'PASS'
            and qa['scope'] == terminal.FULL_SCOPE and qa['paper_ready'] is False
            and qa['headline_eligible'] is False and qa['claim_gate'] == 'NOT_RUN',
            'independent audit scope or promotion differs')
    require(set(qa['members']) == MEMBERS, 'audited member roster differs')
    for name, ref in qa['members'].items():
        require(checked(ref) == out/name, 'audited member location differs')
    for name in ('manifest', 'seal'):
        require(checked(qa[name]) == out/(name+'.json'), 'bundle metadata location differs')
    bundle = terminal.screen._validate_bundle(out, root=terminal.screen.evidence_root(),
                                             expected_kind=terminal.FULL_SCOPE)
    require(qa['validated_gate'] == {**gate, 'manifest_sha256': bundle['manifest_sha256'],
                                    'seal_sha256': bundle['seal_sha256']},
            'independent original-source replay result differs')
    checked(qa['driver'])
    process = terminal.api.read(checked(qa['validation_process']))
    require(process['returncode'] == 0 and process['command'] == qa['command']
            and process['runtime_seconds'] == qa['runtime_seconds']
            and json.loads(process['stdout']) == qa['validated_gate'],
            'independent validation process receipt differs')
    config, protocol, stage = _contexts(qa, gate, out)
    require(bundle['manifest']['code'] == gate['producer_code']
            and bundle['manifest']['freeze_id'] == gate['freeze_id'] == stage.name,
            'bundle source/freeze differs')
    scenes = terminal.api.read(out/'scene_receipts.json')
    cells = [json.loads(line) for line in (out/'planned_qualification_cells.jsonl').read_text().splitlines()]
    validate_metadata(gate, scenes, cells, protocol, config)
    require(isinstance(qa['command'], list) and 'validate-full' in qa['command']
            and str(out) in qa['command'] and qa['source']['commit'] in qa['command']
            and str(qa['job_id']).isdigit() and isinstance(qa['hostname'], str)
            and isinstance(qa['runtime_seconds'], (int, float))
            and math.isfinite(qa['runtime_seconds']) and qa['runtime_seconds'] >= 0,
            'independent replay execution receipt differs')
    return dict(producer_commit=qa['source']['commit'], audit_E0=qa['E0'],
                audit_config=qa['config'], manifest=qa['manifest'], seal=qa['seal'],
                verification_scope='Pinned original-source replay plus metadata-only logical coverage recount')
