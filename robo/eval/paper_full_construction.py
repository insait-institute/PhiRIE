"""Authenticate scoped E1 export/drop evidence, never promote incomplete Table I.

This is a read-only paper adapter. It authenticates existing source contracts,
sealed per-body attempts and canonical aggregation; it executes no physics.
The aggregate seal is supplied as ``completion_audit`` by paper_pipeline.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path

from run.icra2027 import e1_full_drop as producer

api = producer.api
base = producer.base
metrics = base.metrics
MEMBERS = {'scene_records.json', 'scene_records.csv', 'table/construction_table.json',
           'table/construction_table.csv', 'table/construction_table.tex'}
UNIT_MEMBERS = {'drop_report.json', 'build_manifest.json', 'construction_record_aggregate.json'}


def require(condition, message):
    if not condition:
        raise ValueError('full construction paper source: ' + message)


def checked(ref):
    path = Path(ref['path'])
    require(path.is_absolute() and path.resolve(strict=True) == path, 'aliased artifact path')
    actual = api.identity(path)
    require(actual['sha256'] == ref['sha256']
            and ('size_bytes' not in ref or actual['size_bytes'] == ref['size_bytes']),
            'artifact bytes changed')
    return path


def _contexts(audit, stage):
    """Validate frozen source/runtime identities without importing physics engines."""
    config_path = checked(audit['config'])
    config = api.read(config_path)
    e0_path = stage/'contract/freeze_manifest.json'
    e0 = api.read(e0_path)
    api.contract_digest(e0)
    code = base._exact_code(e0['code']['repository'], audit['source_commit'])
    require(e0['code']['dirty'] is False and e0['code']['dirty_override_for_smoke'] is False
            and e0['code']['commit'] == audit['source_commit']
            and e0['freeze_id'] == config['freeze_id'] == stage.name
            and api.resource(e0, 'e1_full_drop_config') == audit['config'],
            'source E0/config/freeze differs')
    require(config_path == stage/'execution.json' and config['scope'] == producer.SCOPE
            and config['schema_version'] == 1 and config['paper_ready'] is False
            and config['room_export_requirement'] == producer.ROOM_SCOPE
            and config['measurement_scope'] == metrics.MEASUREMENT_SCOPE
            and config['planned_objects'] == 1871 and config['regime'] == metrics.REQUIRED_REGIMES[3]
            and config['pilot_scenes'] == base.PILOT and config['prior_pilot_drop_reuse'] is False,
            'declared full cohort or measurement scope differs')
    require(set(config['runtime']) == {'e1_python', 'e1_pybullet', 'e1_drop_implementation',
                'e1_link_pose_implementation', 'e1_numpy_entrypoint',
                'e1_numpy_norm_implementation', 'e1_numpy_core_binary'},
            'measurement runtime identity roster differs')
    for name, ref in config['runtime'].items():
        checked(ref)
        require(api.resource(e0, name) == ref, 'measurement runtime not E0-bound')
    for name, value in producer.EXPORT_FIXED_ENVIRONMENT.items():
        require(config['export_environment'][name] == value, 'export environment differs')
    producer._pilot_admission(config)
    populations, _, _ = producer._source_population(config)
    producer._validate_units(config, populations, Path(config['source']['e0']['path']).parent.parent)
    return config, code, api.identity(e0_path)


def validate_unit(report, unit, config, code):
    """Check measured/terminal telemetry without reclassifying drop results."""
    sid = report['scene_id']; record = report['record']; population = unit['population']
    require(report['schema_version'] == 1 and report['scope'] == producer.SCOPE
            and report['producer_code'] == code and report['paper_ready'] is False
            and report['source'] == config['source']
            and report['planned_scenes'] == config['planned_scenes']
            and report['measurement_scope'] == config['measurement_scope'],
            'unit source or scope differs')
    require(record['scene_id'] == sid and record['freeze_id'] == config['freeze_id']
            and record['build_commit'] == code['commit'] and record['regime'] == config['regime']
            and record['measurement_scope'] == config['measurement_scope']
            and record['input_instances'] == population['input_instances']
            and record['controller_accepted_instances'] == population['controller_accepted_instances']
            and record['record_valid'] is False and record['geometry_reference_status'] == 'NOT_RUN'
            and record['f1_20'] is None and record['f1_weight'] == 0 and record['runtime_minutes'] is None,
            'unit denominator or unsupported scientific promotion')
    require({'independent_geometry_evaluation_not_run', 'full_runtime_not_measured'}
            <= set(record['validity_reasons']), 'incomplete Table I limitations removed')
    export = report['authenticated_export']
    if export is not None:
        require(export['full_room_admission'] == dict(status='PASS', requirement=producer.ROOM_SCOPE,
                    source_mode=unit['mode'], source_unit=unit)
                and unit['mode'] in {'original_export', 'fresh_export'},
                'export lacks full-room artifact admission')
        require(base.export_bodies(export) == report['bodies'], 'exported body identity differs')
        roster = export['materialization']['roster']
        require(roster['job_count'] == population['input_instances']
                and roster['accepted_slots'] == population['accepted_slots'],
                'export lost planned/accepted object identities')
        if unit['mode'] == 'fresh_export':
            execution = export.get('export_execution') or {}
            require(execution.get('requested_environment') == config['export_environment']
                    and execution.get('calls') and all(c.get('returncode') == 0
                        and all(c.get('environment', {}).get(k) == v
                            for k, v in config['export_environment'].items()) for c in execution['calls']),
                    'fresh exporter execution receipt differs')
        bodies = report['bodies']; rows = report['measurements']
        slots = [b['object_slot'] for b in bodies]
        require(len(slots) == len(set(slots)) and slots == [r['object_slot'] for r in rows]
                and report['attempted_bodies'] == report['completed_bodies'] == len(bodies)
                and report['status'] == 'PASS' and record['scene_status'] == 'success'
                and record['accepted_instances'] == record['tested_instances'] == len(bodies),
                'drop attempts missing, duplicated or incomplete')
        for row in rows:
            require(row['status'] == 'COMPLETE' and row['error'] is None
                    and type(row['measurement']['stable']) is bool
                    and type(row['measurement']['sunk']) is bool
                    and math.isfinite(row['measurement']['drift_m'])
                    and row['measurement']['drift_m'] >= 0
                    and math.isfinite(row['runtime_seconds']) and row['runtime_seconds'] >= 0,
                    'drop measurement is missing or invalid')
        require(record['stable_instances'] == sum(r['measurement']['stable'] for r in rows)
                and record['source_artifact_hash'] == api.canonical_hash(export),
                'conditional drop count or source hash differs')
    else:
        kind = report['terminal_kind']
        require(kind in {'no_accepted_bodies', 'original_paired_room_rejection'}
                and unit['mode'] == kind and report['source_unit'] == unit
                and report['room_export_requirement'] == producer.ROOM_SCOPE
                and record['accepted_instances'] == record['tested_instances'] == record['stable_instances'] == 0
                and report['bodies'] == report['measurements'] == []
                and report['attempted_bodies'] == report['completed_bodies'] == 0
                and record['source_artifact_hash'] == api.canonical_hash(unit),
                'terminal source changed or missing measurements promoted')
        if kind == 'no_accepted_bodies':
            require(population['controller_accepted_instances'] == 0 and record['scene_status'] == 'empty'
                    and report['status'] == 'EMPTY', 'false empty-scene closure')
        else:
            verification = report['verification']
            require(report['status'] == 'NOT_RUN' and record['scene_status'] == 'failed'
                    and report['terminal_reason'] == producer.REJECTION
                    and verification['result'] == dict(status='PASS', observed_rejection=producer.REJECTION,
                        source_commit=base.SOURCE_COMMIT, scene_id=sid), 'room rejection hidden or changed')
            for name in ('stdout', 'stderr'):
                checked(verification[name])


def validate_source(spec, path, table, audit_path, audit):
    """Return verified provenance; all displayed values remain original table fields."""
    path = Path(path); audit_path = Path(audit_path)
    require(checked(spec) == path and checked(spec['completion_audit']) == audit_path,
            'source entrypoint differs')
    out = path.parent.parent; stage = out.parent
    require(path == out/'table/construction_table.json' and audit_path == out/'seal.json'
            and out.name == 'construction' and stage.parent.name == 'icra2027',
            'unexpected aggregate or audit location')
    require(api.read(path) == table and api.read(audit_path) == audit,
            'payload differs from pinned source bytes')
    require(audit['paper_ready'] is False and audit['planned_scene_regime_cells'] == 250
            and audit['measured_regimes'] == 1 and set(audit['members']) == MEMBERS,
            'aggregate scope or member roster differs')
    for name, ref in audit['members'].items():
        require(checked(ref) == out/name, 'aggregate member location differs')
    config, code, e0_ref = _contexts(audit, stage)
    require(len(config['planned_scenes']) == len(set(config['planned_scenes'])) == 50,
            'full scene denominator differs')
    require([checked(ref) for ref in audit['source_unit_seals']] ==
            [stage/'construction_drop'/sid/'seal.json' for sid in config['planned_scenes']],
            'source scene seals missing, duplicated or substituted')
    measured = []
    for sid, ref in zip(config['planned_scenes'], audit['source_unit_seals']):
        directory = Path(ref['path']).parent; seal = api.read(ref['path'])
        require(seal['source_commit'] == code['commit'] and set(seal['members']) == UNIT_MEMBERS,
                'unit seal producer or members differ')
        for name, member in seal['members'].items():
            require(checked(member) == directory/name, 'unit member location differs')
        report = api.read(directory/'drop_report.json')
        require(report['scene_id'] == sid and report['record']['build_manifest_path'] == str(directory/'build_manifest.json'),
                'unit scene identity differs')
        validate_unit(report, config['units'][sid], config, code)
        record = report['record']
        require(api.read(directory/'build_manifest.json')['record'] == record,
                'unit manifest differs from measurement record')
        require(metrics.aggregate([record], allow_preliminary=True, smoke=True) ==
                api.read(directory/'construction_record_aggregate.json'), 'unit canonical aggregate differs')
        measured.append(record)
    records = api.read(out/'scene_records.json')
    keys = [(r['regime'], r['scene_id']) for r in records]
    require(len(keys) == len(set(keys)) == 250 and set(keys) ==
            {(regime, sid) for regime in metrics.REQUIRED_REGIMES for sid in config['planned_scenes']},
            'declared input ladder cells lost or duplicated')
    require([r for r in records if r['regime'] == config['regime']] == measured,
            'full records differ from sealed scene measurements')
    for record in records:
        if record['regime'] != config['regime']:
            require(record['validity_reasons'] == ['input_family_not_run'] and record['record_valid'] is False
                    and all(record[k] is None for k in ('input_instances', 'controller_accepted_instances',
                        'accepted_instances', 'stable_instances', 'tested_instances', 'f1_20', 'runtime_minutes')),
                    'unexecuted input family promoted')
    require(table['schema_version'] == 3 and table['paper_ready'] is False
            and table['preliminary_override'] is True and table['smoke'] is False
            and all(r['valid_for_paper'] is False for r in table['rows'])
            and table['rows'] == metrics.aggregate(records, allow_preliminary=True),
            'canonical table changed or incomplete table promoted')
    entries = sorted((r['build_manifest_path'], api.identity(r['build_manifest_path'])['sha256']) for r in records)
    digest = hashlib.sha256(''.join(f'{sha}  {name}\n' for name, sha in entries).encode()).hexdigest()
    provenance = table['provenance']
    e0 = api.read(e0_ref['path'])
    construction_configs = [r for r in e0['configs'] if r['field'] == 'construction_config']
    require(len(construction_configs) == 1 and provenance['construction_config_source_sha256'] ==
            construction_configs[0]['source_content_sha256'], 'construction regime config identity differs')
    require(provenance['scene_records_sha256'] == audit['members']['scene_records.json']['sha256']
            and provenance['build_manifest_count'] == 250 and provenance['build_manifest_set_sha256'] == digest,
            'canonical source-record/manifest provenance differs')
    return dict(producer_commit=code['commit'], audit_E0=e0_ref, audit_config=audit['config'],
                seal=api.identity(audit_path), selected_row_index=3, paper_ready=False,
                table_i_complete=False, manipulation_claim='NOT_RUN',
                scene_status_counts=dict(Counter(r['scene_status'] for r in measured)),
                verification_scope='Sealed full-roster export and isolated canonical drop only; no physics replay; incomplete Table I remains invalid')
