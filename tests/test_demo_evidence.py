import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from agents.orchestrator.artifact import sha256_file
from interface.demo_agentic import DemoError, resolve, evidence_comparison_image
from tests.test_demo_manifest import fixture


def evidence_fixture(tmp_path, retry=False):
    path, config = fixture(tmp_path)
    events = []
    for index, (policy, tool) in enumerate([('A0', 'trellis'), ('A2', 'reconviagen'), ('A3', 'registration_retry')]):
        directory = tmp_path / policy
        directory.mkdir()
        mesh = directory / 'mesh.obj'
        mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
        transform = np.eye(4)
        transform[0, 3] = index * .1
        np.save(directory / 'transform.npy', transform)
        (directory / 'registration.json').write_text(json.dumps({'T': transform.tolist()}))
        values = {'symmetric_clipped_registration_residual_m': .05,
                  'support_overlap_fraction': .11, 'settle_stable': index == 2}
        (directory / 'probe.json').write_text(json.dumps({'settle_stable': index == 2}))
        (directory / 'evidence.json').write_text(json.dumps({
            'input_hashes': {'raw_mesh': sha256_file(mesh)}, 'raw_values': values}))
        artifacts = {'raw_mesh': mesh, 'transform': directory / 'transform.npy',
                     'registration': directory / 'registration.json',
                     'probe': directory / 'probe.json', 'evidence': directory / 'evidence.json'}
        proposal = f'scene/obj:{tool}'
        if index == 2:
            rp = directory / 'retry.json'
            rp.write_text(json.dumps({'action': 'registration_signed_source_up_restart',
                'proposal_id': proposal, 'parent_proposal_id': 'scene/obj:reconviagen'}))
            artifacts['retry'] = rp
        event = {'freeze_id': 'fixture-v1', 'scene_id': 'scene', 'object_slot': 'obj',
                 'job_id': 'scene/obj', 'policy_id': policy, 'proposal_id': proposal,
                 'retry_invoked': index == 2, 'retry_produced': index == 2,
                 'reason_codes': ['bounded_retry' if index == 2 else 'construction_evidence_selection'],
                 'support_label': 'unsupported', 'selected_asset': {'tool': tool,
                 'artifact_paths': {k: str(v) for k, v in artifacts.items()},
                 'artifact_hashes': {k: sha256_file(v) for k, v in artifacts.items()}}}
        ep = directory / 'event.json'
        ep.write_text(json.dumps(event))
        events.append({'path': str(ep), 'sha256': sha256_file(ep)})
    source = config['sources'][0]
    selected = events[2 if retry else 1]
    source.update(kind='selection', **selected,
                  evidence_comparison={'kind': 'registration_retry' if retry else 'initial_selection',
                                       'records': events[1:] if retry else events[:2]})
    path.write_text(yaml.safe_dump(config))
    return path, config, events


def rewrite_event(path, config, reference, edit):
    p = Path(reference['path'])
    value = json.loads(p.read_text())
    edit(value)
    p.write_text(json.dumps(value))
    digest = sha256_file(p)
    reference['sha256'] = digest
    if config['sources'][0]['path'] == str(p):
        config['sources'][0]['sha256'] = digest
    path.write_text(yaml.safe_dump(config))


@pytest.mark.parametrize('retry', [False, True])
def test_real_evidence_comparison_and_unsupported_verdict(tmp_path, retry):
    path, _, _ = evidence_fixture(tmp_path, retry)
    source = resolve(path)['sources'][0]
    assert source['event']['support_label'] == 'unsupported'
    assert len(source['comparison']['candidates']) == 2
    assert source['comparison']['candidates'][-1]['values']['settle_stable'] is retry


@pytest.mark.parametrize('damage', ['remove_mesh', 'tamper_evidence'])
def test_comparison_artifact_damage_fails_closed(tmp_path, damage):
    path, _, _ = evidence_fixture(tmp_path)
    p = tmp_path / 'A0' / ('mesh.obj' if damage == 'remove_mesh' else 'evidence.json')
    p.unlink() if damage == 'remove_mesh' else p.write_text('{}')
    with pytest.raises(DemoError, match='missing source or SHA256'):
        resolve(path)


def test_probe_cannot_be_relabelled_as_stable(tmp_path):
    path, config, _ = evidence_fixture(tmp_path)
    ref = config['sources'][0]['evidence_comparison']['records'][0]
    p = tmp_path / 'A0/evidence.json'
    data = json.loads(p.read_text()); data['raw_values']['settle_stable'] = True
    p.write_text(json.dumps(data))
    rewrite_event(path, config, ref, lambda event: event['selected_asset']['artifact_hashes'].update(evidence=sha256_file(p)))
    with pytest.raises(DemoError, match='probe verdict disagree'):
        resolve(path)


def test_retry_parent_identity_is_enforced(tmp_path):
    path, config, _ = evidence_fixture(tmp_path, True)
    ref = config['sources'][0]['evidence_comparison']['records'][1]
    p = tmp_path / 'A3/retry.json'
    data = json.loads(p.read_text()); data['parent_proposal_id'] = 'wrong-parent'
    p.write_text(json.dumps(data))
    rewrite_event(path, config, ref, lambda event: event['selected_asset']['artifact_hashes'].update(retry=sha256_file(p)))
    with pytest.raises(DemoError, match='real parent'):
        resolve(path)


def test_actual_mesh_comparison_renders_on_mujoco(tmp_path):
    path, _, _ = evidence_fixture(tmp_path)
    source = resolve(path)['sources'][0]
    image = evidence_comparison_image(source)
    assert image.size == (1920, 1080)
    checks = source['comparison']['rendering_checks']
    assert len(checks) == 2
    assert checks[0]['camera_distance'] == pytest.approx(checks[1]['camera_distance'], rel=1e-6)
    for check in checks:
        assert not check['clipping']
        assert check['all_union_vertices'] == 8
        assert max(map(abs, check['projected_ndc_min'] + check['projected_ndc_max'])) < .601
    # Real blue mesh surfaces must be present in both panels, independently of text.
    for x in (135, 980):
        pixels = np.asarray(image)[272:772, x:x+780]
        assert np.count_nonzero((pixels[:, :, 2] > pixels[:, :, 0] + 20) & (pixels[:, :, 2] > 60)) > 100


def test_retry_cannot_reuse_transform_under_new_bytes(tmp_path):
    path, config, _ = evidence_fixture(tmp_path, True)
    ref = config['sources'][0]['evidence_comparison']['records'][1]
    matrix = np.load(tmp_path / 'A2/transform.npy')
    matrix[0, 3] = .125
    np.save(tmp_path / 'A3/transform.npy', matrix.astype(np.float32))
    registration = tmp_path / 'A3/registration.json'
    registration.write_text(json.dumps({'T': matrix.astype(np.float32).tolist()}))
    # Use an exactly representable transform so equality cannot be hidden by dtype.
    parent = matrix.astype(np.float64)
    np.save(tmp_path / 'A2/transform.npy', parent)
    (tmp_path / 'A2/registration.json').write_text(json.dumps({'T': parent.tolist()}))
    for index, policy in [(0, 'A2'), (1, 'A3')]:
        reference = config['sources'][0]['evidence_comparison']['records'][index]
        def edit(event):
            event['selected_asset']['artifact_hashes'].update(
                transform=sha256_file(tmp_path / policy / 'transform.npy'),
                registration=sha256_file(tmp_path / policy / 'registration.json'))
        rewrite_event(path, config, reference, edit)
    with pytest.raises(DemoError, match='new registration'):
        resolve(path)
