"""Nested producer E0 bindings and complete-roster readiness, no model calls."""
import copy
import json
import shutil

import pytest
import yaml

from run.icra2027 import e3_fresh_canonical_config as builder
from tests.test_e3_fresh_canonical import pool, dump


@pytest.fixture
def nested_pool(pool):
    directory, hashes, jobs = pool
    scene = '09c1414f1b'
    # Rebind fixture job IDs before nesting; all file receipts are updated.
    for job in jobs:
        job['job_id'] = job['job_id'].replace('scene:', scene + ':')
    value = json.loads((directory/'proposal_pool.json').read_text())
    for row in value['rows']:
        row['job_id'] = row['job_id'].replace('scene:', scene + ':')
    records = directory/'proposal_records.jsonl'
    records.write_text(''.join(json.dumps(row)+'\n' for row in value['rows']))
    value['proposal_records_sha256'] = builder.e3.sha256_file(records)
    dump(directory/'proposal_pool.json', value)
    audit = json.loads((directory/'postrun_audit.json').read_text())
    audit['proposal_pool_sha256'] = builder.e3.sha256_file(directory/'proposal_pool.json')
    dump(directory/'postrun_audit.json', audit)
    nested = directory/scene
    nested.mkdir()
    for path in list(directory.iterdir()):
        if path.is_file(): shutil.move(str(path), nested/path.name)
    return nested, hashes, jobs, directory.parent


def test_nested_pool_requires_explicit_matching_freeze(nested_pool):
    directory, hashes, jobs, root = nested_pool
    with pytest.raises(ValueError):
        builder.authenticate_pool(directory, hashes, jobs)
    receipt = builder.authenticate_pool(directory, hashes, jobs, freeze_root=root)
    binding = receipt['producer_provenance']
    assert binding['source_commit'] == 'a'*40
    assert binding['freeze_id'] == root.name
    assert binding['config_sha256'] == 'b'*64
    assert binding['contract_file_sha256'] == builder.e3.sha256_file(root/'contract/freeze_manifest.json')
    assert binding['proposal_records_sha256'] == builder.e3.sha256_file(directory/'proposal_records.jsonl')


@pytest.mark.parametrize('change', ['foreign_root', 'wrong_scene', 'wrong_freeze_name', 'symlink_contract'])
def test_nested_pool_rejects_substitution(nested_pool, change):
    directory, hashes, jobs, root = nested_pool
    if change == 'foreign_root':
        root = root.parent/'foreign'; root.mkdir()
    elif change == 'wrong_scene':
        jobs = copy.deepcopy(jobs); jobs[0]['job_id'] = '5748ce6f01:auto:1000'
    elif change == 'wrong_freeze_name':
        renamed = root.with_name('another-freeze'); root.rename(renamed)
        directory = renamed/directory.relative_to(root); root = renamed
    else:
        contract = root/'contract/freeze_manifest.json'
        real = contract.with_name('real.json'); contract.rename(real); contract.symlink_to(real)
    with pytest.raises(ValueError):
        builder.authenticate_pool(directory, hashes, jobs, freeze_root=root)


@pytest.fixture
def roster(tmp_path, monkeypatch):
    monkeypatch.setattr(builder.e3, 'CODE_ROOT', tmp_path)
    path = tmp_path/'roster.yaml'
    scenes = [f'{i:010x}' for i in range(50)]
    path.write_text(yaml.safe_dump({'population': {'scene_ids': scenes}}))
    return path, scenes


def test_partial_cohort_reports_all_missing_without_config_writes(roster, tmp_path):
    path, scenes = roster
    result = builder.cohort_readiness(path, tmp_path/'discovery', tmp_path/'trellis', tmp_path/'rvg',
        trellis_freeze_root=tmp_path/'tf', rvg_freeze_root=tmp_path/'rf')
    assert result['status'] == 'WAITING_REAL_INITIAL_POOLS'
    assert result['config_written'] is False
    assert result['planned_scenes'] == 50 and result['scenes_with_required_files'] == 0
    assert list(result['missing_inputs']) == scenes
    assert all(len(missing) == 13 for missing in result['missing_inputs'].values())
    assert sorted(p.name for p in tmp_path.iterdir()) == ['roster.yaml']


@pytest.mark.parametrize('change', ['truncated', 'duplicate', 'invalid_scene'])
def test_cohort_rejects_changed_roster_shape(roster, tmp_path, change):
    path, scenes = roster
    if change == 'truncated': scenes.pop()
    elif change == 'duplicate': scenes[-1] = scenes[0]
    else: scenes[-1] = '../escape'
    path.write_text(yaml.safe_dump({'population': {'scene_ids': scenes}}))
    with pytest.raises(ValueError):
        builder.cohort_readiness(path, tmp_path/'d', tmp_path/'t', tmp_path/'r',
            trellis_freeze_root=tmp_path/'tf', rvg_freeze_root=tmp_path/'rf')


def test_complete_cohort_delegates_exact_population_to_canonical_normalizer(roster, tmp_path, monkeypatch):
    from agents.orchestrator import automatic_inventory
    path, scenes = roster
    monkeypatch.setattr(builder, '_missing_inputs', lambda *args: [])
    roots = {'trellis_freeze_root': tmp_path/'tf', 'rvg_freeze_root': tmp_path/'rf'}
    calls = []
    def source(discovery, trellis, rvg, **kwargs):
        assert kwargs == roots
        scene = discovery.name
        assert trellis.name == rvg.name == scene
        return {'scene_id': scene, 'initial_pools': {}, 'planned_jobs': 2}, [
            {'job_id': scene+':auto:1000', 'prepared': True},
            {'job_id': scene+':auto:1001', 'prepared': False}]
    def normalize(payload, contract, freeze):
        calls.append(payload)
        assert builder.e3._scene_roster(payload) == scenes
        assert [s['scene_id'] for s in payload['automatic_sources']] == scenes
        assert contract['freeze_id'] == freeze == 'config-validation'
        return {}, {'counts': {'jobs': 100}}, {'initial_pool_complete': True}, ''
    monkeypatch.setattr(builder, '_source_payload', source)
    monkeypatch.setattr(automatic_inventory, 'build_payloads', normalize)
    result = builder.cohort_readiness(path, tmp_path/'d', tmp_path/'t', tmp_path/'r', **roots)
    assert result['status'] == 'READY_FOR_RESERVED_FREEZE'
    assert result['planned_jobs'] == 100 and result['prepared_inputs'] == 50
    assert result['planned_policy_object_rows'] == 500
    assert calls[0]['population']['planned_jobs_per_policy'] == 100
    assert result['paper_ready'] is False and result['config_written'] is False


def test_cohort_cli_rejects_config_write_request(monkeypatch):
    monkeypatch.setattr('sys.argv', ['builder', '--discovery', 'd', '--trellis', 't', '--rvg', 'r',
        '--cohort-roster', 'roster', '--trellis-freeze-root', 'tf', '--rvg-freeze-root', 'rf',
        '--config-directory', 'must-not-write'])
    with pytest.raises(SystemExit) as error: builder.main()
    assert error.value.code == 2


def test_cohort_writer_preserves_per_scene_slots_and_unused_reservation(tmp_path, monkeypatch):
    from pathlib import Path
    source_code = builder.CODE
    monkeypatch.setattr(builder, 'CODE', tmp_path)
    monkeypatch.setattr(builder.e3, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(builder, 'runtime_identity', lambda python: ({}, 'a'*64))
    for name in (builder.POLICIES, builder.RUNTIME, 'configs/experiments/icra2027/agentic_automatic_freeze.yaml'):
        destination = tmp_path/name; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_code/name, destination)
    scenes = [f'{i:010x}' for i in range(50)]
    bundles = [({'scene_id':scene}, [{'automatic_instance_id':1000}, {'automatic_instance_id':1001}]) for scene in scenes]
    payload = {'automatic_sources':[s for s, _ in bundles],
               'population':{'scene_roster_config':str(tmp_path/'roster.yaml')}}
    result = {'planned_jobs':100, 'planned_policy_object_rows':500}
    destination = tmp_path/'configs/cohort'
    with pytest.raises(ValueError, match='unused canonical reservation'):
        builder._write_configs(payload,bundles,result,'fresh',destination)
    assert not destination.exists()
    (tmp_path/'outputs/icra2027/.freeze_ids/fresh').mkdir(parents=True)
    result = builder._write_configs(payload,bundles,result,'fresh',destination)
    execution = yaml.safe_load(Path(result['files']['execution']).read_text())
    assert execution['schema_version'] == 2 and execution['planned_scenes'] == 50
    assert execution['scene_ids'] == scenes
    assert all(slots == ['obj_1000','obj_1001'] for slots in execution['scene_object_slots'].values())
    assert execution['planned_policy_object_rows'] == 500
    frozen = yaml.safe_load(Path(result['files']['freeze']).read_text())
    assert any(row['id'] == 'canonical_scene_roster' for row in frozen['input_roots'])
    with pytest.raises(FileExistsError):
        builder._write_configs(payload,bundles,result,'fresh',destination)
