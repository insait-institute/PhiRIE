"""Read-only E6 engineering closure authentication; never estimates performance."""
import json
from pathlib import Path
import yaml
from robo.manifest.hash import canonical_hash, git_snapshot
from robo.eval import audit_labels
from robo.eval import build_task_support_dataset as canonical
from robo.certification import public_feature_bridge as bridge

PRODUCER = '152446d6874155e3a2115759c96e9a369f59236a'
METRICS = ('AUROC', 'AUPRC', 'Brier', 'Risk@80', 'Risk@60')
BASELINES = ('global_psnr_db', 'global_f1_score')
MEMBERS = {'labels.csv', 'features_with_validity.csv', 'task_local_features_and_labels.csv'}


def require(condition, message):
    if not condition:
        raise ValueError('task-support closure: ' + message)


def checked(ref):
    p = Path(ref['path'])
    require(p.is_absolute() and p.resolve(strict=True) == p, 'source path uses an alias')
    require(canonical._sha256(p) == ref['sha256'], 'source bytes changed: ' + str(p))
    if 'size_bytes' in ref:
        require(p.stat().st_size == ref['size_bytes'], 'source size changed')
    return p


def read(path):
    return json.loads(Path(path).read_text())


def contract(path, config_ref):
    value = read(path)
    require(canonical_hash({k: v for k, v in value.items()
                           if k not in {'created_utc', 'environment', 'contract_sha256'}})
            == value['contract_sha256'], 'E0 digest changed')
    code = Path(value['code']['repository'])
    snapshot = git_snapshot(code)
    require(code.is_absolute() and code.resolve(strict=True) == code
            and snapshot['commit'] == value['code']['commit'] == PRODUCER
            and not snapshot['dirty'] and value['code']['dirty'] is False,
            'original feature source/E0 changed')
    require(value['freeze_id'] == path.parent.parent.name, 'E0 stage differs')
    matched = [r for r in value['resource_inventory']
               if r.get('resolved_path') == config_ref['path']]
    require(len(matched) == 1 and matched[0]['hash_method'] == 'content_sha256'
            and matched[0]['sha256'] == config_ref['sha256'], 'E0 config binding changed')
    checked(config_ref)
    # Rehash metadata/config inputs; no model, private reference or vault access.
    for r in value['resource_inventory']:
        if r.get('hash_method') == 'content_sha256':
            checked({'path': r['resolved_path'], 'sha256': r['sha256']})
    for r in value['configs']:
        checked({'path': r['resolved_path'], 'sha256': r['source_content_sha256']})
    return value, code


def sealed_members(directory, members):
    for name, digest in members.items():
        path = directory / name
        require(path.resolve(strict=True).is_relative_to(directory), 'seal path escapes output')
        checked({'path': str(path), 'sha256': digest})


def validate_build_flag(feature, bundle):
    """Bind the published failure denominator to the authenticated source."""
    flag = float(feature['scene_build_failed'])
    require(flag in (0., 1.), 'constructor failure flag is not binary')
    require(bool(flag) == (bundle['build_status'] == 'failed'),
            'constructor failure flag differs from authenticated bundle')


def _public_sources(full_config, features):
    """Use the existing producer's exact source/18-unit validation and adapter."""
    require(full_config['source_commit'] == PRODUCER and full_config['tier'] == 'full',
            'feature input source or tier differs')
    # The externally pinned full-feature QA already records the original
    # expensive validation. Authenticate exactly those source/config/gate bytes;
    # no geometry arithmetic, producer CLI or model is replayed for formatting.
    contexts = [bridge._reference_context(ref) for ref in full_config['references']]
    require([(c[3]['scene_id'], c[3]['condition_id']) for c in contexts]
            == [(s, c) for s in bridge.shared.SCENES for c in bridge.shared.CONDITIONS],
            'complete fixed original public gate roster differs')
    require(all(ref['producer'] == full_config['references'][0]['producer']
                and ref['config'] == full_config['references'][0]['config']
                for ref in full_config['references']), 'mixed original source/config')
    cache = {c[4]: c[3] for c in contexts}
    require(len(full_config['references']) == 18, 'public grounding denominator differs')
    expected = []
    for ref in full_config['references']:
        gate = bridge.validate_reference(ref, cache)
        expected.extend((full_config['freeze_id'], q['scene_id'], q['task_id'], gate['condition_id'])
                        for q in gate['original_gate']['queries'])
    require(expected == [tuple(r[k] for k in canonical.JOIN_KEY) for r in features],
            'feature rows differ from original public queries')
    for row in features:
        bundle = read(row['build_manifest_path'])
        bridge.authenticate_bundle(bundle, cache)
        validate_build_flag(row, bundle)
        require(tuple(bundle[k] for k in canonical.JOIN_KEY)
                == tuple(row[k] for k in canonical.JOIN_KEY), 'bundle query differs')


def validate_closure(qa_path, *, expected_sha256):
    """Return counts only after authenticating the complete failed scientific gate.

    The caller must pin ``expected_sha256`` in the paper freeze. No files are
    published and no label, baseline, probability or metric is newly measured.
    """
    qa_path = checked({'path': str(qa_path), 'sha256': expected_sha256})
    qa = read(qa_path)
    require(qa_path.name == 'validity_independent_qa.json', 'unexpected closure entrypoint')
    stage = qa_path.parent
    require(qa['schema_version'] == 1 and qa['status'] == 'PASS'
            and qa['scope'] == 'artifact_integrity_and_missing_evidence_validity_closure'
            and qa['source']['commit'] == PRODUCER and qa['source']['dirty'] is False,
            'closure scope/source differs')
    require(all(qa.get(k) is False for k in ['paper_ready', 'reference_data_opened',
            'rollouts_invoked', 'predictions_generated'])
            and qa['predictive_claim_gate'] == 'FAIL'
            and all(qa[k] == 'NOT_RUN' for k in ['loso_status', 'reference_measurement_status',
                                               'baseline_measurement_status'])
            and qa['metrics'] == dict.fromkeys(METRICS), 'unsupported claim or measurement')
    refs = ['execution_config', 'contract', 'execution_receipt', 'feature_seal',
            'full_feature_qa', 'measurements', 'label_join_manifest', 'qa_script']
    paths = {k: checked(qa[k]) for k in refs}
    require(paths['execution_config'] == stage/'execution.json'
            and paths['contract'] == stage/'contract/freeze_manifest.json'
            and paths['execution_receipt'] == stage/'join_execution_receipt.json'
            and paths['measurements'] == stage/'measurements_not_run.json'
            and paths['label_join_manifest'] == stage/'audit/validity_closure/label_join_manifest.json',
            'closure stage binding differs')
    cfg = read(paths['execution_config']); execution = read(paths['execution_receipt'])
    e0, code = contract(paths['contract'], qa['execution_config'])
    require(cfg['source_commit'] == PRODUCER and cfg['freeze_id'] == qa['freeze_id'] == stage.name
            and cfg['predictions_allowed'] is False and cfg['paper_predictive_claim'] is False
            and cfg['feature_seal'] == qa['feature_seal'] and cfg['measurements'] == qa['measurements'],
            'execution config scope differs')
    require(execution['source'] == qa['source'] and execution['cwd'] == str(code)
            and execution['contract_sha256'] == e0['contract_sha256']
            and execution['config'] == qa['execution_config']
            and execution['returncode'] == cfg['expected_returncode'] == 2
            and execution['command'] == cfg['command'], 'canonical join execution differs')
    require('LOSO blocked: every training/test fold requires both labels'
            in checked(execution['stderr']).read_text(), 'canonical LOSO blocker absent')
    require(checked(execution['stdout']).read_text() == '', 'unexpected join output')
    features_dir = paths['feature_seal'].parent
    full_stage = features_dir.parent.parent
    full_qa = read(paths['full_feature_qa'])
    require(paths['full_feature_qa'] == full_stage/'full_independent_qa.json'
            and features_dir == full_stage/'audit/features'
            and full_stage.name == qa['feature_data_freeze_id'] == cfg['feature_data_freeze_id']
            and full_qa['source'] == qa['source'] and full_qa['status'] == 'PASS'
            and full_qa['feature_seal'] == qa['feature_seal'], 'full feature source binding differs')
    full_e0, full_code = contract(full_stage/'contract/freeze_manifest.json', full_qa['config'])
    require(full_code == code, 'feature and label source differ')
    for module in [canonical, audit_labels, bridge]:
        path = Path(module.__file__).resolve()
        relative = path.relative_to(Path(__file__).resolve().parents[2])
        require(canonical._sha256(path) == canonical._sha256(code/relative),
                'canonical verifier differs from original producer')
    seal = read(paths['feature_seal'])
    require(seal['source_kind'] == 'real' and seal['query_rows'] == 72
            and seal['feature_generation_read_labels'] is False and seal['paper_ready'] is False
            and seal['driver_sha256'] == canonical._sha256(Path(canonical.__file__))
            and seal['evaluation_baseline_columns'] == list(BASELINES), 'feature seal scope differs')
    sealed_members(features_dir, seal['members'])
    require({str(p.relative_to(features_dir)) for p in features_dir.rglob('*') if p.is_file()}
            == set(seal['members']) | {'feature_seal.json'}, 'unexpected unsealed feature artifact')
    for ref in seal['source_refs']:
        checked(ref)
    feature_config = read(features_dir/'config_resolved.json')
    require(seal['config_sha256'] == canonical._sha256(features_dir/'config_resolved.json')
            and seal['freeze_id'] == full_stage.name and feature_config['tier'] == 'full',
            'feature configuration differs')
    features = canonical._read_csv(features_dir/'features_unlabeled.csv')
    closure = paths['label_join_manifest'].parent
    manifest = read(paths['label_join_manifest'])
    require(set(manifest['members']) == MEMBERS and qa['members'] == manifest['members']
            and {p.name for p in closure.iterdir()} == MEMBERS | {'label_join_manifest.json'},
            'unexpected prediction/metric or missing closure artifact')
    require(not any((stage/'audit'/name).exists() or (full_stage/'audit'/name).exists()
                    for name in ['predictions', 'table', 'metrics', 'heldout_predictions.csv', 'fold_models.json', 'risk_coverage.csv', 'risk_coverage.pdf']),
            'unexpected prediction/metric artifact')
    sealed_members(closure, manifest['members'])
    measurements = read(paths['measurements'])
    protocol_path = checked(cfg['protocol']); protocol = yaml.safe_load(protocol_path.read_text())
    require(manifest['protocol_sha256'] == seal['label_protocol_sha256'] == cfg['protocol']['sha256']
            and protocol['label_producer_sha256'] == canonical._sha256(Path(audit_labels.__file__))
            and protocol['label_definition'] == 'construction_validity', 'label protocol differs')
    require(manifest['feature_seal_sha256'] == qa['feature_seal']['sha256']
            and manifest['measurement_sha256'] == qa['measurements']['sha256']
            and measurements['feature_seal_sha256'] == qa['feature_seal']['sha256']
            and measurements['freeze_id'] == manifest['freeze_id'] == seal['freeze_id']
            and measurements['measurement_status'] == 'NOT_RUN'
            and all(measurements[k] is False for k in ['reference_data_opened', 'rollouts_invoked',
                                                      'baseline_measurements_invoked']), 'measurement scope differs')
    labels = canonical._read_csv(closure/'labels.csv')
    joined = canonical._read_csv(closure/'task_local_features_and_labels.csv')
    intermediate = canonical._read_csv(closure/'features_with_validity.csv')
    keys = lambda rows: [tuple(r[k] for k in canonical.JOIN_KEY) for r in rows]
    require(len(features) == len(set(keys(features))) == 72
            and keys(features) == keys(labels) == keys(joined) == keys(intermediate)
            == keys(measurements['rows']), 'complete unique 72-query denominator differs')
    thresholds = audit_labels.ValidityThresholds(**protocol['thresholds'])
    null_fields = list(thresholds.__dict__) + list(protocol['required_booleans']) + list(BASELINES)
    for feature, label, row, middle, record in zip(features, labels, joined, intermediate, measurements['rows']):
        require(record['measurement_status'] == 'NOT_RUN'
                and record['measurement_scope'] == 'all_required_task_invariants'
                and all(record.get(k, 'absent') is None for k in null_fields), 'unmeasured field is not explicit null')
        require(float(feature['robot_frame_missing']) == float(feature['geometry_policy_cameras_missing']) == 1,
                'missing-frame closure applied to supported input')
        result = audit_labels.label_record(record, thresholds)
        reasons = list(result['invalid_reasons'])
        if float(feature['scene_build_failed']):
            reasons.append('construction_failed')
        reasons += ['construction_robot_frame_missing', 'construction_policy_cameras_missing',
                    'robot_alignment_failed_or_missing']
        require(label['invalid_label'] == '1' and label['invalid_reasons'] == ';'.join(reasons),
                'canonical missing-evidence label differs')
        expected = {**feature, 'invalid_label': label['invalid_label'], 'invalid_reasons': label['invalid_reasons']}
        require(middle == expected and row == {**expected, **dict.fromkeys(BASELINES, '')},
                'joined feature/label/baseline changed')
    scenes = sorted({r['scene_id'] for r in features})
    folds = [{'heldout_scene': scene, 'training_has_both_labels': False,
              'heldout_has_both_labels': False} for scene in scenes]
    require(len(scenes) == 6 and manifest['fold_health'] == qa['fold_health'] == folds
            and manifest['loso_admissible'] is False and manifest['paper_ready'] is False,
            'LOSO fold health/claim differs')
    expected_counts = {'planned_conditions': 18, 'planned_queries': 72, 'invalid_queries': 72,
                       'valid_queries': 0, 'failed_constructor_query_rows':
                       sum(int(float(r['scene_build_failed'])) for r in features),
                       'missing_robot_frames': 72, 'missing_policy_cameras': 72}
    require(all(qa[k] == v for k, v in expected_counts.items()) and manifest['query_rows'] == 72,
            'closure aggregate counts differ')
    _public_sources(read(checked(full_qa['config'])), features)
    return {'schema_version': 1, 'scope': 'task_support_missing_evidence_engineering',
            'status': 'PASS', 'rows': [{'queries': 72, 'constructor_failed_queries':
            expected_counts['failed_constructor_query_rows'], 'invalid_queries': 72,
            'estimable_folds': 0, 'planned_folds': 6}], 'claim_gate': 'FAIL', 'loso_status': 'NOT_RUN',
            'paper_ready': False, 'predictive_performance': False, 'reference_measurements': False,
            'policy_success': False, 'metrics': dict.fromkeys(METRICS),
            'source': {'path': str(qa_path), 'sha256': expected_sha256},
            'feature_seal': qa['feature_seal'], 'label_join_manifest': qa['label_join_manifest'],
            'producer_commit': PRODUCER, 'contract_sha256': e0['contract_sha256'],
            'feature_contract_sha256': full_e0['contract_sha256']}
