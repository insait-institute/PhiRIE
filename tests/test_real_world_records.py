import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from robo.eval.real_world_metrics import generate, summarize_construction
from robo.eval.real_world_records import (RecordError, build_record, fingerprint,
                                         load_roster, strict_full_build, validate_alignment)


def test_roster_preserves_failed_launch_and_rejects_drift(tmp_path):
    path = tmp_path / 'roster.yaml'
    path.write_text('batch: [good, CLVR]\n')
    config = {'mode': 'legacy_audit', 'paper_ready': False,
              'droid': {'roster_path': 'roster.yaml', 'roster_sha256': fingerprint(path)['sha256'],
                        'batches': ['batch'], 'launches': [
                            {'episode': 'good', 'workspace_id': 'good'},
                            {'episode': 'CLVR', 'workspace_id': 'failed'}]},
              'phone': {'captures': []}}
    assert len(load_roster(config, tmp_path)) == 2
    dropped = copy.deepcopy(config)
    dropped['droid']['launches'].pop()
    with pytest.raises(RecordError, match='preserve every'):
        load_roster(dropped, tmp_path)
    config['paper_ready'] = True
    with pytest.raises(RecordError, match='cannot be promoted'):
        load_roster(config, tmp_path)
    config['paper_ready'] = False
    path.write_text('batch: [good]\n')
    with pytest.raises(RecordError, match='hash mismatch'):
        load_roster(config, tmp_path)


def test_strict_build_missing_manifest_and_holdout_leakage():
    checks = dict.fromkeys(('metric_reconstruction', 'discovery', 'accepted_movable',
                           'registered_assets', 'collision_physics', 'background',
                           'full_room_collision', 'robot_alignment_or_not_applicable',
                           'simulator_load_and_settle', 'immutable_build_manifest'), True)
    assert strict_full_build(checks)
    checks.pop('immutable_build_manifest')
    assert not strict_full_build(checks)
    row = {'translation_cm': 1., 'rotation_deg': 2., 'independent_alignment': {
        'held_out_ids': ['f2'], 'construction_ids': ['f1'], 'selection_ids': ['f2'],
        'reference_sha256': 'abc'}}
    with pytest.raises(RecordError, match='disjoint'):
        validate_alignment(row)
    row['independent_alignment']['selection_ids'] = ['f1']
    validate_alignment(row)
    row['translation_cm'] = float('nan')
    with pytest.raises(RecordError, match='finite'):
        validate_alignment(row)


def test_failed_capture_keeps_denominator_and_null_alignment(tmp_path):
    output = tmp_path / 'outputs'
    output.mkdir()
    video = tmp_path / 'phone.mp4'
    video.write_bytes(b'fixture')
    (output / 'video2sim_1.log').write_text(
        f'video2sim: {video} -> scene=phone out={output / "video_phone"}\nABORT: metricize\n')
    capture = {'source': 'phone', 'capture_id': 'phone', 'workspace_id': 'video_phone',
               'scene_id': 'phone', 'video': 'phone.mp4', 'job_ids': [1]}
    row, manifest = build_record(capture, {'artifact_root': str(tmp_path)}, 'test')
    assert row['accepted_objects'] is None
    assert not row['reconstruction_success'] and not row['full_build_success']
    assert row['translation_cm'] is None and row['alignment_pass'] is None
    assert not row['paper_ready'] and manifest['launch_attempts'][0]['errors']
    result = summarize_construction([row])[0]
    assert result['workspaces'] == 1 and result['full_builds'] == 0
    assert result['accepted_object_records'] == 0 and result['alignment_evaluated'] == 0
    assert result['alignment_pass_rate'] is None
    with pytest.raises(ValueError, match='duplicate'):
        summarize_construction([row, row])
    src = tmp_path / 'rows.json'
    src.write_text(json.dumps({'rows': [row]}))
    assert generate(src, None, tmp_path / 'table')['paper_ready'] is False
    capture['scene_id'] = 'changed'
    with pytest.raises(RecordError, match='identity mismatch'):
        build_record(capture, {'artifact_root': str(tmp_path)}, 'test')


def test_reconstructed_artifacts_still_not_strict_full_build(tmp_path):
    output = tmp_path / 'outputs'
    output.mkdir()
    raw = tmp_path / 'raw/LAB/success/episode'
    raw.mkdir(parents=True)
    (raw / 'trajectory.h5').write_bytes(b'fixture')
    scene = 'droid_lab'
    directory = output / scene
    recon = directory / 'recon'
    recon.mkdir(parents=True)
    np.savez(recon / 'recon.npz', w2c=np.eye(4)[None], K=np.eye(3))
    (recon / 'align_report.json').write_text('{}')
    (recon / 'held_out_eval.json').write_text(json.dumps({'held_out_metrics': {
        'center_rms_m': .02, 'rotation_residual_deg': {'median': 3}}}))
    (output / 'droid_recon_2.log').write_text(f'droid_recon: {raw} -> scene={scene} out={directory}\nDONE ->\n')
    transforms = tmp_path / 'data/recon_scenes/data' / scene / 'dslr/nerfstudio'
    transforms.mkdir(parents=True)
    (transforms / 'transforms_undistorted.json').write_text('{"frames":[{}]}')
    splats = tmp_path / 'data/recon_scenes/splats'
    splats.mkdir()
    (splats / f'{scene}.ply').write_bytes(b'fixture')
    (splats / f'{scene}_train_report.json').write_text('{"n_gaussians":1}')
    capture = {'source': 'droid', 'capture_id': 'LAB/success/episode', 'episode': 'LAB/success/episode',
               'workspace_id': scene, 'scene_id': scene, 'job_ids': [2]}
    row, _ = build_record(capture, {'artifact_root': str(tmp_path), 'raw_root': str(tmp_path / 'raw')}, 'test')
    assert row['reconstruction_success'] and not row['full_build_success']
    assert row['legacy_alignment_translation_cm'] == 2
    assert row['legacy_alignment_rotation_deg'] == 3
    assert row['translation_cm'] is None and row['rotation_deg'] is None


def test_contract_source_config_and_no_overwrite(tmp_path, monkeypatch):
    from robo.eval.real_world_records import run
    config = tmp_path / 'config.yaml'
    config.write_text('mode: legacy_audit\npaper_ready: false\n')
    out = tmp_path / 'freeze/real_world'
    monkeypatch.setattr('robo.eval.real_world_records.subprocess.check_output',
                        lambda cmd, **kwargs: 'source\n' if cmd[-1] == 'HEAD' else '')
    with pytest.raises(RecordError, match='matching E0'):
        run(config, out, 'f', tmp_path)
    contract = out.parent / 'contract/freeze_manifest.json'
    contract.parent.mkdir(parents=True)
    contract.write_text(json.dumps({'freeze_id': 'f', 'code': {'commit': 'source'},
                                   'configs': [{'field': 'real_world_config', 'source_content_sha256': 'wrong'}]}))
    with pytest.raises(RecordError, match='differs from E0'):
        run(config, out, 'f', tmp_path)
    out.mkdir()
    with pytest.raises(RecordError, match='refusing to overwrite'):
        run(config, out, 'f', tmp_path)
