"""Prospective FK independence, immutable fit and unchanged residual metrics."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from agents.recon import droid_extract as extract
from agents.recon import align_to_traj as align
from agents.eval import droid_alignment_eval as evaluate
from robo.eval import real_world_records as records
from robo.manifest.hash import canonical_hash


def dump(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    import robo.manifest.hash as hashes
    monkeypatch.setattr(hashes, 'git_snapshot', lambda *a: {'commit': 'a' * 40, 'dirty': False, 'branch': 'test'})
    episode = tmp_path / 'episode'; (episode / 'recordings/MP4').mkdir(parents=True)
    raw = {'metadata': episode / 'metadata_test.json', 'trajectory': episode / 'trajectory.h5',
           'video': episode / 'recordings/MP4/123.mp4'}
    for key, p in raw.items(): p.write_bytes(key.encode())
    plan = {'schema_version': 1, 'scope': 'droid_train_only_alignment', 'episode': str(episode),
        'camera': 'wrist', 'serial': '123', 'inputs': {k: extract.input_identity(p) for k, p in raw.items()},
        'split': extract.prospective_split(121, 122)}
    plan_path = dump(tmp_path / 'plan.json', plan)
    indices = np.arange(122)
    positions = np.column_stack((np.sin(indices / 9), np.cos(indices / 13), indices / 50)) * .1
    fk = np.repeat(np.eye(4)[None], 122, axis=0); fk[:, :3, 3] = positions
    ids = plan['split']['rgb_indices']
    w2c = np.repeat(np.eye(4)[None], len(ids), axis=0)
    # True correspondence offset+1, metric scale2, translation(.3,.2,.1).
    w2c[:, :3, 3] = -(positions[np.asarray(ids) + 1] - [.3, .2, .1]) / 2
    rec = {'names': np.array([f'frame_{i:06d}.jpg' for i in ids]), 'w2c': w2c,
        'K': np.array([[100., 0, 320], [0, 100, 240], [0, 0, 1]]), 'frame_wh': np.array([640, 480]),
        'points': np.column_stack((np.linspace(-.1, .1, 100), np.zeros(100), np.ones(100))),
        'points_rgb': np.ones((100, 3), np.uint8), 'K_depth': np.eye(3),
        'depth': np.ones((len(ids), 2, 2), np.float16)}
    recon = tmp_path / 'recon.npz'; np.savez_compressed(recon, **rec)
    train = {'schema_version': 1, 'role': 'train_only', 'plan': extract.input_identity(plan_path),
        'fk_by_index': {str(i): fk[i].tolist() for i in plan['split']['train_fk_indices']}}
    train_path = dump(tmp_path / 'train.json', train)
    return {'root': tmp_path, 'plan': plan, 'plan_path': plan_path, 'rec': rec, 'recon': recon,
            'train': train, 'train_path': train_path, 'fk': fk, 'fit': tmp_path / 'fit'}


def fit_reference(f, shift=0):
    report = align.align_train_only(f['recon'], f['train_path'], f['plan_path'], f['fit'])
    fk = f['fk'].copy(); fk[:, 0, 3] += shift
    ref = {'schema_version': 1, 'role': 'held_out_only', 'plan': extract.input_identity(f['plan_path']),
        'fit': extract.input_identity(f['fit'] / 'fit.json'),
        'fk_by_index': {str(i): fk[i].tolist() for i in f['plan']['split']['held_out_fk_indices']}}
    return report, dump(f['root'] / 'reference.json', ref)


@pytest.mark.parametrize('n', [92, 100, 121, 167, 205, 211, 213, 297, 337, 401])
def test_metadata_split_disjoint_for_every_offset(n):
    s = extract.prospective_split(n, n + 1)
    assert len(s['rgb_indices']) <= 240
    assert set(s['train_fk_indices']).isdisjoint(s['held_out_fk_indices'])
    assert max(s['train_video_indices']) < min(s['held_out_video_indices']) - 4
    assert len(s['excluded_guard_indices']) == 4


@pytest.mark.parametrize('n,t', [(0, 1), (5, 6), (120, 100), (100.0, 101)])
def test_invalid_or_too_small_metadata_rejected(n, t):
    with pytest.raises(ValueError): extract.prospective_split(n, t)


def test_offset_selection_and_real_numpy_end_to_end(fixture):
    f = fixture; report, ref = fit_reference(f)
    assert report['solution']['frame_offset'] == 1
    assert report['solution']['scale'] == pytest.approx(2)
    assert align.validate_train_fit(f['fit']) == report
    out = f['root'] / 'eval.json'
    result = evaluate.evaluate_sealed_fit(f['fit'], ref, out)
    assert result['gate']['passed'] is True
    assert result['held_out_metrics']['center_rms_m'] < 1e-12
    assert result['evaluated_reference_frames'] == result['planned_reference_frames']
    assert evaluate.evaluate_sealed_fit(f['fit'], ref) == result
    with pytest.raises(FileExistsError): evaluate.evaluate_sealed_fit(f['fit'], ref, out)
    with pytest.raises(FileExistsError): align.align_train_only(f['recon'], f['train_path'], f['plan_path'], f['fit'])


def test_reference_changes_never_change_train_fit_or_offset(fixture):
    f = fixture; report, ref = fit_reference(f, shift=.25)
    result = evaluate.evaluate_sealed_fit(f['fit'], ref)
    assert result['held_out_metrics']['center_rms_m'] == pytest.approx(.25)
    assert result['gate']['passed'] is False
    assert align.validate_train_fit(f['fit']) == report


def test_train_rejects_full_fk_and_reference_indices(fixture):
    f = fixture
    leaked = copy.deepcopy(f['train']); leaked['full'] = {'c2w_base': f['fk'].tolist()}
    with pytest.raises(ValueError, match='sparse TRAIN'): align.train_solution(f['rec'], leaked, f['plan'])
    leaked = copy.deepcopy(f['train'])
    leaked['fk_by_index'][str(f['plan']['split']['held_out_fk_indices'][0])] = np.eye(4).tolist()
    with pytest.raises(ValueError, match='FK roster'): align.train_solution(f['rec'], leaked, f['plan'])


def test_offset_union_leakage_plan_rejected(fixture):
    plan = copy.deepcopy(fixture['plan'])
    plan['split']['held_out_fk_indices'].append(plan['split']['train_fk_indices'][0])
    with pytest.raises(ValueError, match='leakage'): extract.validate_frame_plan(plan)


def test_missing_fit_blocks_reference_read(fixture, monkeypatch):
    reads = []
    monkeypatch.setattr(extract, 'read_planned_fk', lambda *a: reads.append(a))
    with pytest.raises(FileNotFoundError):
        extract.export_reference(fixture['plan_path'], fixture['fit'], fixture['root'] / 'reference.json')
    assert reads == []


def test_missing_fit_blocks_evaluator_reference_read(fixture, monkeypatch):
    read = Path.read_text
    ref = fixture['root'] / 'reference.json'; ref.write_text('FORBIDDEN')
    def guarded(path, *a, **kw):
        assert path != ref, 'reference touched before fit closure'
        return read(path, *a, **kw)
    monkeypatch.setattr(Path, 'read_text', guarded)
    with pytest.raises(FileNotFoundError): evaluate.evaluate_sealed_fit(fixture['fit'], ref)


def test_tampered_fit_rejected_even_after_rehash(fixture):
    f = fixture; report, ref = fit_reference(f)
    report['solution']['frame_offset'] = 0
    dump(f['fit'] / 'fit.json', report)
    dump(f['fit'] / 'seal.json', {'fit_sha256': extract.input_identity(f['fit'] / 'fit.json')['sha256'],
                                'fit_digest': canonical_hash(report)})
    with pytest.raises(ValueError, match='replay differs'): align.validate_train_fit(f['fit'])


def test_tampered_aligned_geometry_rejected_after_rehash(fixture):
    f = fixture; report, _ = fit_reference(f)
    path = f['fit'] / 'recon_base.npz'; arrays = dict(np.load(path)); arrays['points'] += 1
    np.savez_compressed(path, **arrays)
    report['aligned'] = extract.input_identity(path)
    dump(f['fit'] / 'fit.json', report)
    dump(f['fit'] / 'seal.json', {'fit_sha256': extract.input_identity(f['fit'] / 'fit.json')['sha256'],
                                'fit_digest': canonical_hash(report)})
    with pytest.raises(ValueError, match='aligned reconstruction differs'): align.validate_train_fit(f['fit'])


def test_raw_input_tamper_rejected(fixture):
    Path(fixture['plan']['inputs']['trajectory']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity changed'): extract.validate_frame_plan(fixture['plan'])


def test_input_symlink_rejected(fixture):
    link = fixture['root'] / 'link'; link.symlink_to(fixture['train_path'])
    with pytest.raises(ValueError, match='symlink'): extract.input_identity(link)


def test_reference_role_or_roster_cannot_be_replaced(fixture):
    f = fixture; _, path = fit_reference(f)
    ref = json.loads(path.read_text()); ref['fk_by_index'] = f['train']['fk_by_index']; dump(path, ref)
    with pytest.raises(ValueError, match='reference roster'): evaluate.evaluate_sealed_fit(f['fit'], path)


def test_missing_registered_reference_frames_retained(fixture):
    f = fixture; ids = f['plan']['split']['held_out_video_indices']; missing = ids[-3:]
    keep = [i for i, n in enumerate(f['rec']['names']) if int(Path(n).stem.split('_')[1]) not in missing]
    for name in ('names', 'w2c', 'depth'): f['rec'][name] = f['rec'][name][keep]
    np.savez_compressed(f['recon'], **f['rec'])
    _, ref = fit_reference(f); result = evaluate.evaluate_sealed_fit(f['fit'], ref)
    assert result['planned_reference_frames'] == len(ids)
    assert result['evaluated_reference_frames'] == len(ids) - 3
    assert result['missing_reference_frames'] == missing


def test_missing_train_pose_coverage_rejected(fixture):
    f = fixture; rec = copy.deepcopy(f['rec'])
    for key in ('w2c', 'names', 'depth'): rec[key] = rec[key][-20:]
    with pytest.raises(ValueError, match='TRAIN pose coverage'): align.train_solution(rec, f['train'], f['plan'])


def test_degenerate_train_rejected(fixture):
    f = fixture; f['rec']['w2c'][:, :3, 3] = 0
    with pytest.raises(ValueError, match='degenerate TRAIN'): align.train_solution(f['rec'], f['train'], f['plan'])


def test_numpy_runtime_drift_rejected(fixture):
    f = fixture; report, _ = fit_reference(f)
    report['runtime']['numpy_version'] = 'wrong'
    dump(f['fit'] / 'fit.json', report)
    dump(f['fit'] / 'seal.json', {'fit_sha256': extract.input_identity(f['fit'] / 'fit.json')['sha256'],
                                'fit_digest': canonical_hash(report)})
    with pytest.raises(ValueError, match='NumPy runtime'): align.validate_train_fit(f['fit'])


def test_all_ten_missing_workspaces_stay_not_run(fixture, monkeypatch):
    f = fixture
    captures = [{'workspace_id': 'scene' + str(i), 'capture_id': str(i),
                 'plan': extract.input_identity(f['plan_path'])} for i in range(10)]
    config = {'captures': captures, 'freeze_id': 'test'}
    monkeypatch.setattr(records, 'validate_prospective', lambda *a: (config, {'code': {'commit': 'a' * 40}}))
    out = f['root'] / 'summary'
    records.summarize_prospective(None, f['root'], out, f['root'])
    rows = json.loads((out / 'workspaces.json').read_text())['rows']
    assert len(rows) == 10
    assert all(r['execution_status'] == 'NOT_RUN' for r in rows)
    assert all(r['translation_cm'] is None and r['runtime_minutes'] is None for r in rows)
    assert all(r['reconstruction_success'] is False and r['full_build_success'] is False for r in rows)


def test_h5_constructor_reads_only_train_indices(fixture, monkeypatch):
    import sys
    from types import SimpleNamespace
    f = fixture; seen = []
    class Dataset:
        shape = (122, 6)
        def __getitem__(self, indices):
            seen.append(indices)
            assert indices == f['plan']['split']['train_fk_indices']
            assert set(indices).isdisjoint(f['plan']['split']['held_out_fk_indices'])
            return np.zeros((len(indices), 6))
        def __array__(self, *args):
            raise AssertionError('full H5 dataset materialized')
    class File:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, key):
            assert key == 'observation/camera_extrinsics/123_left'
            return Dataset()
    monkeypatch.setitem(sys.modules, 'h5py', SimpleNamespace(File=lambda *a: File()))
    result = extract.read_planned_fk(f['plan'], 'train')
    assert set(result) == {str(i) for i in seen[0]}


def test_runtime_retains_venv_invocation_path(tmp_path, monkeypatch):
    binary = tmp_path / 'python-real'; binary.write_bytes(b'python')
    venv = tmp_path / 'venv-python'; venv.symlink_to(binary)
    invoked = []
    def output(command, **kwargs):
        invoked.append(command)
        return 'E7_RUNTIME_JSON=' + json.dumps({'invocation': str(venv), 'python': extract.input_identity(binary), 'packages': {}})
    monkeypatch.setattr(records.subprocess, 'check_output', output)
    runtime = records._runtime_identity(venv, ['numpy'])
    assert invoked[0][0] == str(venv)
    assert runtime['invocation'] == str(venv)
    assert runtime['python']['path'] == str(binary)


def test_source_drift_rejected(fixture, monkeypatch):
    import robo.manifest.hash as hashes
    f = fixture; fit_reference(f)
    monkeypatch.setattr(hashes, 'git_snapshot', lambda *a: {'commit': 'b' * 40, 'dirty': False})
    with pytest.raises(ValueError, match='original clean source'): align.validate_train_fit(f['fit'])


def test_reference_export_parser_preserves_diagnostic_stdout(fixture, monkeypatch):
    f = fixture; report, _ = fit_reference(f)
    monkeypatch.setattr(extract.subprocess, 'check_output', lambda *a, **k:
        '[simany] diagnostic\nE7_FIT_JSON=' + json.dumps(report) + '\n')
    monkeypatch.setattr(extract, 'read_planned_fk', lambda plan, role: {
        str(i): f['fk'][i].tolist() for i in plan['split']['held_out_fk_indices']})
    out = f['root'] / 'reference_export.json'
    extract.export_reference(f['plan_path'], f['fit'], out, validator_python=report['runtime']['invocation'])
    assert json.loads(out.read_text())['fit'] == extract.input_identity(f['fit'] / 'fit.json')


def test_failed_workspace_not_replaced_or_executed_again(fixture, monkeypatch):
    f = fixture
    capture = {'workspace_id': 'failed_original', 'capture_id': 'original',
               'plan': extract.input_identity(f['plan_path'])}
    config = {'captures': [capture], 'freeze_id': 'test', 'runtime': {}}
    contract = {'code': {'commit': 'a' * 40}, 'contract_sha256': 'b' * 64, 'freeze_id': 'test'}
    monkeypatch.setattr(records, 'validate_prospective', lambda *a: (config, contract))
    unit = f['root'] / 'real_world/workspaces/failed_original'; unit.mkdir(parents=True)
    config_path = dump(f['root'] / 'execution.json', config)
    contract['configs'] = [{'field': 'real_world_config', 'source_content_sha256': extract.input_identity(config_path)['sha256']}]
    dump(unit / 'attempt.json', {'config': extract.input_identity(config_path), 'source_commit': 'a' * 40,
        'contract_sha256': 'b' * 64, 'workspace_id': capture['workspace_id'], 'runtime': {}})
    result = {'capture': capture, 'contract_sha256': contract['contract_sha256'],
        'artifacts': {'attempt.json': extract.input_identity(unit / 'attempt.json')},
        'alignment_evaluation': None, 'status': 'FAILED', 'runtime_seconds': 7,
        'freeze_id': 'test', 'source_commit': 'a' * 40, 'reconstruction_success': False,
        'full_build_success': False, 'gaussian_training_status': 'NOT_RUN', 'full_build_status': 'NOT_RUN',
        'failure': {'reason': 'TRAIN center RMS exceeds unchanged0.10m constructor gate'}}
    dump(unit / 'result.json', result)
    out = f['root'] / 'summary'
    records.summarize_prospective(None, f['root'], out, f['root'])
    row = json.loads((out / 'workspaces.json').read_text())['rows'][0]
    assert row['execution_status'] == 'FAILED'
    assert row['cpu_stage_runtime_seconds'] == 7 and row['runtime_minutes'] is None
    assert row['translation_cm'] is None


def test_full_cpu_requires_first_pilot_not_selected_winner(tmp_path, monkeypatch):
    config = {'tier': 'pilot_then_full', 'captures': [{'workspace_id': 'first'}, {'workspace_id': 'second'}]}
    checked = []
    def validate(path, capture, contract, config):
        checked.append((path, capture)); return {'status': 'COMPLETE',
            'alignment_evaluation': {'held_out_metrics': {'n_frames': 5}, 'gate': {'passed': False}}}
    monkeypatch.setattr(records, 'validate_prospective_unit', validate)
    records.require_cpu_pilot(config, {}, tmp_path, 'first')
    assert checked == []
    records.require_cpu_pilot(config, {}, tmp_path, 'second')
    assert checked[0][1]['workspace_id'] == 'first'
    # Scientific negative on held-out error does not trigger replacement/tuning.
    assert checked[0][0] == tmp_path / 'real_world/workspaces/first/result.json'


def test_full_cpu_rejects_unattempted_or_incomplete_pilot(tmp_path, monkeypatch):
    config = {'tier': 'pilot_then_full', 'captures': [{'workspace_id': 'first'}]}
    with pytest.raises(FileNotFoundError): records.require_cpu_pilot(config, {}, tmp_path, 'second')
    monkeypatch.setattr(records, 'validate_prospective_unit', lambda *a: {'status': 'FAILED', 'alignment_evaluation': None})
    with pytest.raises(records.RecordError, match='first-IPRL'): records.require_cpu_pilot(config, {}, tmp_path, 'second')


@pytest.fixture
def stage_fixture(fixture, monkeypatch):
    from robo.eval import agentic_ablation
    f = fixture; root = f['root']; stage = root / '20260906-aaaaaaa-v1'; (stage / 'contract').mkdir(parents=True)
    capture = {'workspace_id': 'original0', 'capture_id': 'episode', 'episode': 'episode', 'source': 'droid'}
    captures = [{**capture, 'workspace_id': 'original' + str(i)} for i in range(10)]
    original = dump(root / 'original.json', {'raw_root': str(root)})
    monkeypatch.setattr(records, 'load_roster', lambda *a: captures)
    monkeypatch.setattr(agentic_ablation, 'checked_repo_path', lambda p, *a, **k: p)
    runtime = {'invocation': 'fake_python', 'python': extract.input_identity(original), 'packages': {}}
    monkeypatch.setattr(records, '_runtime_identity', lambda *a: runtime)
    config = {'mode': 'prospective_cpu_alignment', 'paper_ready': False, 'pose_backend': 'colmap',
        'fallback_allowed': False, 'seed': 0, 'cpu_threads': 8, 'source_commit': 'inherit',
        'gs_training': 'NOT_RUN', 'full_build': 'NOT_RUN', 'tier': 'pilot_then_full', 'freeze_id': stage.name,
        'execution_workspace': captures[0]['workspace_id'], 'original_config': extract.input_identity(original),
        'captures': [{**c, 'plan': extract.input_identity(f['plan_path'])} for c in captures],
        'runtime': {'h5': runtime, 'sfm': runtime, 'ffmpeg': extract.input_identity(original),
                    'ffprobe': extract.input_identity(original), 'taskset': extract.input_identity(original)}}
    config_path = dump(root / 'execution.json', config)
    contract = {'freeze_id': stage.name, 'code': {'commit': 'a' * 40, 'dirty': False},
        'configs': [{'field': 'real_world_config', 'source_content_sha256': extract.input_identity(config_path)['sha256']}]}
    contract['contract_sha256'] = canonical_hash(contract)
    dump(stage / 'contract/freeze_manifest.json', contract)
    return f, stage, config_path, config, contract


def test_actual_contract_digest_and_all_ten_roster_validation(stage_fixture):
    f, stage, path, config, contract = stage_fixture
    assert records.validate_prospective(path, stage, f['root']) == (config, contract)


@pytest.mark.parametrize('field,value', [('seed', 1), ('pose_backend', 'omega'), ('fallback_allowed', True),
                                       ('execution_workspace', 'original1'), ('paper_ready', True)])
def test_prospective_frozen_config_drift_rejected(stage_fixture, field, value):
    f, stage, path, config, contract = stage_fixture
    config[field] = value; dump(path, config)
    with pytest.raises(records.RecordError): records.validate_prospective(path, stage, f['root'])


def test_e0_digest_tamper_rejected_before_input_read(stage_fixture, monkeypatch):
    f, stage, path, config, contract = stage_fixture
    contract['code']['commit'] = 'b' * 40; dump(stage / 'contract/freeze_manifest.json', contract)
    monkeypatch.setattr(extract, 'validate_frame_plan', lambda *a, **k: pytest.fail('raw plan read before E0 closure'))
    with pytest.raises(records.RecordError, match='E0 digest'): records.validate_prospective(path, stage, f['root'])


def test_shortened_capture_roster_rejected_even_if_e0_rebound(stage_fixture):
    f, stage, path, config, contract = stage_fixture
    config['captures'].pop(); dump(path, config)
    contract['configs'][0]['source_content_sha256'] = extract.input_identity(path)['sha256']
    contract.pop('contract_sha256'); contract['contract_sha256'] = canonical_hash(contract)
    dump(stage / 'contract/freeze_manifest.json', contract)
    with pytest.raises(records.RecordError, match='ten-episode roster'): records.validate_prospective(path, stage, f['root'])


def test_prospective_environment_removes_injected_code_and_scene(monkeypatch, tmp_path):
    for name in ('PYTHONHOME', 'PYTHONPATH', 'SIMANY_OUT', 'SIMF_SCENE', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'OMP_DYNAMIC'):
        monkeypatch.setenv(name, '/untrusted')
    env = extract.prospective_environment(tmp_path)
    assert env['PYTHONPATH'] == str(tmp_path)
    assert env['PYTHONNOUSERSITE'] == '1' and env['PYTHONHASHSEED'] == '0'
    assert env['SIMANY_AUTO'] == '1' and env['SIMANY_MESH_SRC'] == 'derived'
    assert all(name not in env for name in ('PYTHONHOME', 'SIMANY_OUT', 'SIMF_SCENE', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'OMP_DYNAMIC'))


def test_runtime_drift_is_compared_by_complete_package_manifest(stage_fixture, monkeypatch):
    f, stage, path, config, contract = stage_fixture
    monkeypatch.setattr(records, '_runtime_identity', lambda *a: {'invocation': 'fake_python', 'packages': {'numpy': {'files': 'changed'}}})
    with pytest.raises(records.RecordError, match='runtime changed'): records.validate_prospective(path, stage, f['root'])
