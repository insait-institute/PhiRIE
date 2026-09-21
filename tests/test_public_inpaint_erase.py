import copy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from agents.edit.inpaint_qwen import erase_masked_view, erase_public_views


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def bundle(tmp_path):
    image = np.full((64, 80, 3), 30, np.uint8)
    mask = np.zeros((64, 80), np.uint8)
    mask[20:40, 30:50] = 255
    source, masks = tmp_path / 'TRAIN.png', tmp_path / 'mask.png'
    Image.fromarray(image).save(source)
    Image.fromarray(mask).save(masks)
    return {'train_images': {'TRAIN.png': identity(source)},
            'objects': [{'index': 1001}, {'index': 1002}],
            'result': {'objects': [{'object_slot': 'obj_1001', 'terminal_action': 'accept', 'plane_status': 'FIT'},
                                   {'object_slot': 'obj_1002', 'terminal_action': 'abstain'}]},
            'rows': [{'object_slot': 'obj_1001', 'automatic_instance_id': 1001,
                      'view_index': 0, 'frame': 'TRAIN.png', 'status': 'MASK_READY',
                      'mask_identity': identity(masks), 'projected_pixels': 200,
                      'final_pixels': 400}]}


def white_model(image, mask):
    assert image.mode == 'RGB' and mask.mode == 'L'
    return Image.new('RGB', image.size, color='white')


def test_preserves_exact_exterior_and_original_arrays():
    full = np.zeros((64, 80, 3), np.uint8)
    mask = np.zeros((64, 80), bool); mask[20:40, 30:50] = True
    out = erase_masked_view(full, mask, white_model)
    assert np.array_equal(full, np.zeros_like(full))
    assert np.array_equal(out[~mask], full[~mask])
    assert out[mask].max() == 255
    assert out[mask].sum() > 0


def test_empty_mask_never_invokes_model():
    def forbidden(*args):
        pytest.fail('empty mask called model')
    assert erase_masked_view(np.zeros((20, 20, 3), np.uint8), np.zeros((20, 20), bool), forbidden) is None


@pytest.mark.parametrize('position', [(0, 0), (0, 19), (19, 0), (19, 19)])
def test_single_border_pixel_preserves_exterior(position):
    full = np.full((20, 20, 3), 20, np.uint8)
    mask = np.zeros((20, 20), bool); mask[position] = True
    out = erase_masked_view(full, mask, white_model)
    assert np.array_equal(out[~mask], full[~mask])


@pytest.mark.parametrize('bad', ['dtype', 'shape', 'mask_type'])
def test_bad_input_is_not_enhanced(bad):
    full = np.zeros((20, 20, 3), np.uint8); mask = np.ones((20, 20), bool)
    if bad == 'dtype': full = full.astype(float)
    if bad == 'shape': mask = mask[:10]
    if bad == 'mask_type': mask = mask.astype(np.uint8)
    with pytest.raises(ValueError): erase_masked_view(full, mask, white_model)


def test_model_failure_propagates_without_fallback():
    def failure(*args): raise RuntimeError('model failed')
    with pytest.raises(RuntimeError, match='model failed'):
        erase_masked_view(np.zeros((20, 20, 3), np.uint8), np.ones((20, 20), bool), failure)


@pytest.mark.parametrize('response', [None, Image.new('L', (64, 64)), Image.new('RGB', (10, 10))])
def test_malformed_prediction_fails(response):
    with pytest.raises(ValueError, match='malformed'):
        erase_masked_view(np.zeros((20, 20, 3), np.uint8), np.ones((20, 20), bool), lambda *args: response)


def test_planned_object_denominator_and_input_hashes_preserved(tmp_path):
    source = bundle(tmp_path); before = copy.deepcopy(source)
    result = erase_public_views(source, tmp_path / 'out', white_model)
    assert source == before
    assert result['planned_objects'] == 2 and result['erased_views'] == 1
    assert result['objects'][1]['terminal_action'] == 'abstain'
    row = result['rows'][0]
    assert row['outside_mask_equal'] is True
    assert identity(Path(row['output_identity']['path'])) == row['output_identity']
    assert identity(Path(source['train_images']['TRAIN.png']['path'])) == source['train_images']['TRAIN.png']
    with pytest.raises(FileExistsError): erase_public_views(source, tmp_path / 'out', white_model)


def test_failed_views_stay_in_denominator_and_have_no_output(tmp_path):
    def failure(*args): raise RuntimeError('unavailable enhancer')
    result = erase_public_views(bundle(tmp_path), tmp_path / 'out', failure)
    assert result['planned_views'] == result['enhancer_failed_views'] == 1
    assert result['erased_views'] == 0
    assert result['rows'][0]['output_identity'] is None
    assert not list((tmp_path / 'out').rglob('*.png'))


def test_empty_views_retained_without_model(tmp_path):
    source = bundle(tmp_path)
    source['rows'][0].update(status='EMPTY_PROJECTED_MASK', projected_pixels=0, mask_identity=None, final_pixels=0)
    result = erase_public_views(source, tmp_path / 'out', lambda *args: pytest.fail('model called'))
    assert result['empty_mask_views'] == result['planned_views'] == 1
    assert result['erased_views'] == 0


@pytest.mark.parametrize('drift', ['test_frame', 'pixels', 'duplicate', 'path', 'bytes'])
def test_source_roster_drift_fails_before_inference(tmp_path, drift):
    source = bundle(tmp_path)
    row = source['rows'][0]
    if drift == 'test_frame': row['frame'] = 'TEST.png'
    if drift == 'pixels': row['final_pixels'] = 12
    if drift == 'duplicate': source['rows'].append(copy.deepcopy(row))
    if drift == 'path': row['object_slot'] = 'obj_../other'
    if drift == 'bytes': Path(row['mask_identity']['path']).write_bytes(b'changed')
    with pytest.raises((ValueError, RuntimeError)):
        erase_public_views(source, tmp_path / 'out', white_model)


def public_context(tmp_path, monkeypatch):
    import json
    import sys
    import yaml
    import agents.edit.inpaint_qwen as erase
    import agents.edit.inpaint_masks as masks
    from robo.eval import agentic_ablation as e3, e3_factory_materializer
    data = bundle(tmp_path)
    frozen = tmp_path / 'outputs/icra2027/20260906-aaaaaaa-v1'
    (frozen / 'contract').mkdir(parents=True)
    contract = frozen / 'contract/freeze_manifest.json'
    contract.write_text(json.dumps({'contract_sha256': 'contract', 'code': {'commit': 'a' * 40}}))
    old = tmp_path / 'old_masks'; old.mkdir()
    seal = old / 'seal.json'; seal.write_text('{}')
    checkpoint = tmp_path / 'checkpoint'; checkpoint.write_bytes(b'fake_model')
    context = {'schema_version': 1, 'scope': 'automatic_train_only_erasure',
               'freeze_id': frozen.name, 'scene_id': 'a' * 10, 'mask_seal': identity(seal),
               'backend': 'lama_cpu', 'checkpoint': identity(checkpoint),
               'runtime': {'fake': 'unit_test'}, 'algorithm': erase.PUBLIC_ALGORITHM}
    data.update(context={'scene_id': 'a' * 10}, seal_identity=identity(seal))
    path = tmp_path / 'context.yaml'; path.write_text(yaml.safe_dump(context))
    monkeypatch.setattr(e3, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(e3, '_validate_cli_execution', lambda *a, **k: None)
    monkeypatch.setattr(masks, 'validate_public_masks', lambda directory: data, raising=False)
    def bound(context_path, contract_path, *, producer_commit=None):
        c, e0 = yaml.safe_load(context_path.read_text()), json.loads(contract_path.read_text())
        if producer_commit is not None: assert producer_commit == e0['code']['commit']
        return c, e0
    monkeypatch.setattr(masks, '_bound_contract', bound)
    monkeypatch.setattr(erase, 'public_runtime', lambda: {'fake': 'unit_test'})
    monkeypatch.setattr(erase, 'load_lama', lambda: white_model)
    # Audit hooks cannot be removed from a process; boundary enforcement is
    # covered by the source-mask validator tests and exercised in real jobs.
    monkeypatch.setattr(sys, 'addaudithook', lambda guard: None)
    return path, contract, frozen, data


def test_atomic_publication_binds_final_paths_and_does_not_mutate_masks(tmp_path, monkeypatch):
    import json
    from agents.edit.inpaint_qwen import run_public
    path, contract, frozen, data = public_context(tmp_path, monkeypatch)
    before = identity(tmp_path / 'old_masks/seal.json')
    result = run_public(path, contract)
    destination = frozen / 'fidelity/erasure' / ('a' * 10)
    assert result['status'] == 'COMPLETE'
    row = result['rows'][0]
    assert Path(row['output_identity']['path']).is_relative_to(destination)
    assert identity(Path(row['output_identity']['path'])) == row['output_identity']
    manifest = json.loads((destination / 'seal.json').read_text())
    for member, digest in manifest['members'].items():
        assert hashlib.sha256((destination / member).read_bytes()).hexdigest() == digest
    assert identity(tmp_path / 'old_masks/seal.json') == before
    with pytest.raises(FileExistsError): run_public(path, contract)


def test_checkpoint_failure_never_loads_or_falls_back(tmp_path, monkeypatch):
    import agents.edit.inpaint_qwen as erase
    path, contract, _, _ = public_context(tmp_path, monkeypatch)
    (tmp_path / 'checkpoint').write_bytes(b'changed')
    monkeypatch.setattr(erase, 'load_lama', lambda: pytest.fail('loaded bad checkpoint'))
    with pytest.raises((ValueError, RuntimeError)): erase.run_public(path, contract)


def test_load_failure_retains_attempt_and_partial(tmp_path, monkeypatch):
    import json
    import agents.edit.inpaint_qwen as erase
    path, contract, frozen, _ = public_context(tmp_path, monkeypatch)
    def failure(): raise RuntimeError('load failure')
    monkeypatch.setattr(erase, 'load_lama', failure)
    with pytest.raises(RuntimeError, match='load failure'): erase.run_public(path, contract)
    dest = frozen / 'fidelity/erasure' / ('a' * 10)
    assert not dest.exists()
    assert dest.with_name(dest.name + '.failed_partial').is_dir()
    failed = json.loads(dest.with_name(dest.name + '.failure.json').read_text())
    assert failed['fallback_attempted'] is False
    assert failed['planned_objects'] == 2 and failed['planned_views'] == 1
    assert len(failed['planned_rows']) == 1
    with pytest.raises(FileExistsError): erase.run_public(path, contract)


def test_runtime_drift_prevents_loading(tmp_path, monkeypatch):
    import agents.edit.inpaint_qwen as erase
    path, contract, _, _ = public_context(tmp_path, monkeypatch)
    monkeypatch.setattr(erase, 'public_runtime', lambda: {'changed': True})
    monkeypatch.setattr(erase, 'load_lama', lambda: pytest.fail('loaded drifted runtime'))
    with pytest.raises(ValueError, match='runtime differs'): erase.run_public(path, contract)


def test_no_applicable_masks_never_loads_model(tmp_path, monkeypatch):
    import agents.edit.inpaint_qwen as erase
    path, contract, _, data = public_context(tmp_path, monkeypatch)
    data['rows'] = []
    monkeypatch.setattr(erase, 'load_lama', lambda: pytest.fail('unnecessary model load'))
    result = erase.run_public(path, contract)
    assert result['planned_objects'] == 2 and result['planned_views'] == result['erased_views'] == 0


def test_no_plane_views_retained_without_loading(tmp_path, monkeypatch):
    import agents.edit.inpaint_qwen as erase
    path, contract, _, data = public_context(tmp_path, monkeypatch)
    data['result']['objects'][0]['plane_status'] = 'NO_PLANE'
    monkeypatch.setattr(erase, 'load_lama', lambda: pytest.fail('no-plane model load'))
    result = erase.run_public(path, contract)
    assert result['planned_views'] == result['no_plane_views'] == 1
    assert result['erased_views'] == 0


@pytest.mark.parametrize('axis', ['backend', 'algorithm'])
def test_declared_treatment_cannot_fallback(tmp_path, monkeypatch, axis):
    import yaml
    import agents.edit.inpaint_qwen as erase
    path, contract, _, _ = public_context(tmp_path, monkeypatch)
    config = yaml.safe_load(path.read_text())
    if axis == 'backend': config['backend'] = 'auto'
    else: config['algorithm']['outside_mask'] = 'may_change'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='fallback is prohibited'): erase.run_public(path, contract)


@pytest.mark.parametrize('artifact', ['.failure.json', '.failed_partial'])
def test_retained_failure_refuses_new_attempt_even_without_claim(tmp_path, monkeypatch, artifact):
    import agents.edit.inpaint_qwen as erase
    path, contract, frozen, _ = public_context(tmp_path, monkeypatch)
    dest = frozen / 'fidelity/erasure' / ('a' * 10)
    dest.parent.mkdir(parents=True)
    retained = dest.with_name(dest.name + artifact)
    if artifact.endswith('.json'): retained.write_text('old failure')
    else: retained.mkdir()
    with pytest.raises(FileExistsError): erase.run_public(path, contract)


@pytest.mark.parametrize('drift', ['context', 'contract', 'runtime'])
def test_mid_inference_drift_cannot_publish(tmp_path, monkeypatch, drift):
    import agents.edit.inpaint_qwen as erase
    path, contract, frozen, _ = public_context(tmp_path, monkeypatch)
    def mutate(image, mask):
        if drift == 'context': path.write_text(path.read_text() + '\n')
        elif drift == 'contract': contract.write_text(contract.read_text() + '\n')
        else: monkeypatch.setattr(erase, 'public_runtime', lambda: {'changed': True})
        return white_model(image, mask)
    monkeypatch.setattr(erase, 'load_lama', lambda: mutate)
    with pytest.raises((ValueError, RuntimeError)): erase.run_public(path, contract)
    assert not (frozen / 'fidelity/erasure' / ('a' * 10)).exists()


@pytest.mark.parametrize('no_views', [False, True])
def test_fill_consumer_authenticates_complete_or_empty_handoff(tmp_path, monkeypatch, no_views):
    import agents.edit.inpaint_qwen as erase
    path, contract, frozen, data = public_context(tmp_path, monkeypatch)
    if no_views: data['rows'] = []
    result = erase.run_public(path, contract)
    destination = frozen / 'fidelity/erasure' / ('a' * 10)
    checked = erase.validate_public_erasure(destination)
    assert checked['result'] == result
    assert checked['mask_bundle'] == data


@pytest.mark.parametrize('tamper', ['failed', 'denominator', 'exterior'])
def test_fill_consumer_rejects_resealed_semantic_tamper(tmp_path, monkeypatch, tamper):
    import json
    import agents.edit.inpaint_qwen as erase
    path, contract, frozen, _ = public_context(tmp_path, monkeypatch)
    result = erase.run_public(path, contract)
    destination = frozen / 'fidelity/erasure' / ('a' * 10)
    if tamper == 'failed': result['status'] = 'FAILED'
    elif tamper == 'denominator': result['planned_objects'] = 1
    else:
        output = Path(result['rows'][0]['output_identity']['path'])
        with Image.open(output) as im: pixels = np.asarray(im).copy()
        pixels[0, 0] = 250
        Image.fromarray(pixels).save(output)
        result['rows'][0]['output_identity'] = identity(output)
    (destination / 'erasure.json').write_text(json.dumps(result))
    seal = json.loads((destination / 'seal.json').read_text())
    for name in seal['members']:
        seal['members'][name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    (destination / 'seal.json').write_text(json.dumps(seal))
    with pytest.raises(ValueError): erase.validate_public_erasure(destination)
