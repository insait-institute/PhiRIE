import copy

import pytest

from robo.eval import agentic_ablation as e3


@pytest.fixture
def paired():
    jobs = {'s1': ['s1/o1', 's1/o2'], 's2': ['s2/o1', 's2/o2']}
    rows = []
    for scene, ids in jobs.items():
        for job in ids:
            for index, policy in enumerate(e3.POLICY_IDS):
                accepted = not (policy == 'A4' and job.endswith('o2'))
                metric = dict(f1_20=0.1 + index * 0.1, cd_cm=5. - index * 0.5,
                              evaluation_surface_sha256='a' * 64)
                if policy == 'A2' and job == 's2/o1':
                    metric = dict(f1_20=None, cd_cm=None, geometry_evaluation_status='unmatched')
                rows.append(dict(scene_id=scene, job_id=job, policy_id=policy,
                                 accepted=accepted, metrics=metric if accepted else None,
                                 physical_metrics={'settle_stable': index % 2 == 0} if accepted else None))
    return rows, jobs, copy.deepcopy(e3.AGENTIC_UNCERTAINTY_PROTOCOL)


def find(result, contrast, metric):
    return next(r for r in result['rows'] if r['contrast'] == contrast and r['metric'] == metric)


def test_all_fixed_contrasts_coverage_first_and_common_geometry(paired):
    result = e3._paired_uncertainty(*paired)
    assert len(result['rows']) == 16
    assert result['claim_gate'] == 'NOT_RUN' and result['paper_ready'] is False
    coverage = find(result, 'A4-A0', 'build_coverage')
    assert coverage['planned_jobs'] == coverage['paired_jobs'] == 4
    assert coverage['delta'] == -0.5 and coverage['ci95'] == [-0.5, -0.5]
    geometry = find(result, 'A4-A0', 'f1_20')
    assert geometry['paired_jobs'] == 2 and geometry['excluded_pair_jobs'] == 2
    assert geometry['delta'] == pytest.approx(0.4)
    assert geometry['selection_conditioned'] is True
    assert find(result, 'A2-A1', 'f1_20')['paired_jobs'] == 3
    assert find(result, 'A2-A1', 'cd_cm')['paired_job_ids_sha256'] == find(result, 'A2-A1', 'f1_20')['paired_job_ids_sha256']
    stability = find(result, 'A2-A1', 'stable_fraction')
    assert stability['paired_jobs'] == 4  # unmatched geometry does not discard a physical probe
    assert stability['delta'] == 1.0 and stability['independent_physical_validation'] is False


def test_deterministic_shared_scene_bootstrap(paired):
    first = e3._paired_uncertainty(*paired)
    assert first == e3._paired_uncertainty(*paired)
    assert first['protocol']['samples'] == 2000 and first['protocol']['seed'] == 0


def test_zero_or_single_supported_scene_does_not_invent_precision(paired):
    rows, jobs, protocol = paired
    for row in rows:
        if row['policy_id'] == 'A4' and row['scene_id'] == 's2':
            row.update(accepted=False, metrics=None, physical_metrics=None)
    result = e3._paired_uncertainty(rows, jobs, protocol)
    quality = find(result, 'A4-A0', 'f1_20')
    assert quality['paired_scenes'] == 1 and quality['delta'] == pytest.approx(.4)
    assert quality['ci95'] == [None, None] and quality['status'] == 'NOT_ESTIMABLE'
    for row in rows:
        if row['policy_id'] == 'A4':
            row.update(accepted=False, metrics=None, physical_metrics=None)
    quality = find(e3._paired_uncertainty(rows, jobs, protocol), 'A4-A0', 'f1_20')
    assert quality['paired_jobs'] == 0 and quality['delta'] is None
    assert quality['ci95'] == [None, None] and quality['excluded_pair_jobs'] == 4


def test_scene_clusters_preserve_object_weighting_and_within_scene_dependence():
    jobs = {'big': ['b1', 'b2', 'b3'], 'small': ['s1']}
    rows = [dict(scene_id=scene, job_id=job, policy_id=policy,
                 accepted=policy == 'A1' and scene == 'big', metrics=None, physical_metrics=None)
            for scene, ids in jobs.items() for job in ids for policy in e3.POLICY_IDS]
    result = find(e3._paired_uncertainty(rows, jobs, e3.AGENTIC_UNCERTAINTY_PROTOCOL), 'A1-A0', 'build_coverage')
    assert result['delta'] == .75  # pooled object pairs, not an equal-scene point mean
    assert result['ci95'] == [0., 1.]  # whole scenes resample together


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'scene', 'surface', 'accepted_type'])
def test_pairing_or_geometry_binding_drift_fails_closed(paired, mutation):
    rows, jobs, protocol = paired
    if mutation == 'missing': rows.pop()
    if mutation == 'duplicate': rows[-1] = copy.deepcopy(rows[0])
    if mutation == 'scene': rows[0]['scene_id'] = 'other'
    if mutation == 'surface': rows[1]['metrics']['evaluation_surface_sha256'] = 'b' * 64
    if mutation == 'accepted_type': rows[0]['accepted'] = 1
    with pytest.raises(ValueError): e3._paired_uncertainty(rows, jobs, protocol)


@pytest.mark.parametrize('field,value', [('samples', 10), ('seed', 1), ('confidence', .9), ('contrasts', [['A0', 'A4']])])
def test_uncertainty_protocol_cannot_be_tuned(paired, field, value):
    rows, jobs, protocol = paired; protocol[field] = value
    with pytest.raises(ValueError, match='predeclared'): e3._paired_uncertainty(rows, jobs, protocol)


def test_uncertainty_option_cannot_open_config_during_construction(monkeypatch):
    monkeypatch.setattr(e3, '_load_yaml_code', lambda *a, **k: pytest.fail('construction opened statistics config'))
    with pytest.raises(SystemExit):
        e3.main(['--jobs', 'jobs', '--policies', 'policies', '--contract-manifest', 'contract',
                 '--freeze-id', '20260905-1234567-v1', '--out', 'out', '--scene-id', '13c3e046d7',
                 '--control', '--uncertainty-config', 'matching.yaml'])


def test_real_aggregate_seals_supplement_without_changing_main_values(tmp_path, monkeypatch, paired):
    import json
    rows, jobs_by_scene, protocol = paired
    resolved = {'counts': {'scenes': 2, 'jobs': 4, 'policy_object_rows': 20},
                'scenes': [{'scene_id': scene, 'jobs': [{'job_id': job} for job in ids]}
                           for scene, ids in jobs_by_scene.items()]}
    jobs = {'population': {'planned_scenes': 2, 'planned_jobs_per_policy': 4,
                           'planned_policy_object_rows': 20}, 'study_scope': 'synthetic'}
    policies = {'reporting_contract': {'minimum_genuine_retry_jobs_for_headline_claim': 1,
                                       'main_columns': ['policy_id', 'build_coverage']}}
    construction = tmp_path/'construction'
    controls = {}
    for scene in jobs_by_scene:
        directory = construction/scene
        for policy in e3.POLICY_IDS:
            (directory/'selected_assets'/policy).mkdir(parents=True)
        ledger = [dict(row, object_slot=row['job_id'], retry_required=False,
                       retry_produced=False, terminal_action='accept' if row['accepted'] else 'abstain')
                  for row in rows if row['scene_id'] == scene]
        (directory/'job_ledger.jsonl').write_text('\n'.join(json.dumps(r) for r in ledger))
        controls[scene] = (directory, {'policy_config_sha256': e3._canonical_digest(policies)}, {})
    monkeypatch.setattr(e3, '_load_inventory', lambda *a, **k: (resolved, {}))
    monkeypatch.setattr(e3, '_load_control_scene', lambda _, scene: controls[scene])
    monkeypatch.setattr(e3, '_load_eval_scene', lambda _, scene, **k: {
        'evaluation_freeze_id': 'evaluation', 'construction_freeze_id': 'construction',
        'rows': [r for r in rows if r['scene_id'] == scene]})
    monkeypatch.setattr(e3, 'validate_ledger', lambda _: None)
    monkeypatch.setattr(e3, '_aggregate_rows', lambda *a: [dict(policy_id=p, build_coverage=.5) for p in e3.POLICY_IDS])
    monkeypatch.setattr(e3, '_coverage_sweep', lambda *a: [])
    monkeypatch.setattr(e3, '_utc_now', lambda: 'fixed')
    base = tmp_path/'evaluation/base'; base.mkdir(parents=True)
    supplement = tmp_path/'evaluation/supplement'; supplement.mkdir()
    e3.run_aggregate(jobs, policies, 'construction', construction, evaluation_root=base)
    e3.run_aggregate(jobs, policies, 'construction', construction, evaluation_root=supplement,
                     uncertainty_protocol=protocol)
    for name in json.loads((base/'aggregate_seal.json').read_text())['members']:
        assert (base/name).read_bytes() == (supplement/name).read_bytes()
    seal = json.loads((supplement/'aggregate_seal.json').read_text())
    for name in ('agentic_paired_uncertainty.json', 'agentic_paired_uncertainty.csv'):
        assert seal['members'][name] == e3.sha256_file(supplement/name)
    assert len(json.loads((supplement/'agentic_paired_uncertainty.json').read_text())['rows']) == 16
    with pytest.raises(FileExistsError):
        e3.run_aggregate(jobs, policies, 'construction', construction, evaluation_root=supplement,
                         uncertainty_protocol=protocol)
