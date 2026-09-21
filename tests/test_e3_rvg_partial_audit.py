"""Partial producer failure audit: no model calls or metric evaluation."""
import copy
import json
from collections import Counter

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import pytest

from run.icra2027 import e3_rvg_terminal_audit as audit
from run.icra2027.e3_auto_discovery_pilot import identity, sha, PilotError


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def partial(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.e3, 'REPOSITORY_ROOT', tmp_path)
    out = tmp_path/'pool'; out.mkdir()
    rgba = np.full((2, 2, 4), 255, dtype=np.uint8)
    import hashlib
    view = dict(frame='frame', image_sha256='a'*64, mask_sha256='b'*64,
                rgba_shape=list(rgba.shape), rgba_pixels_sha256=hashlib.sha256(rgba.tobytes()).hexdigest())
    dump(out/'view_manifest.json', {'rows':[{'object_index':i, 'views':[view]} for i in range(4)]})
    manifest = {'source_discovery_hashes':{'input':'c'*64}}
    rows = []
    for i in range(4):
        row = dict(job_id=f'job-{i}', proposal_id=f'proposal-{i}', output_index=i,
                   prepared=i != 3, status='available' if i == 0 else 'generation_failed' if i < 3 else 'unavailable',
                   reason=None if i == 0 else 'typed failure' if i == 1 else 'worker_exit_1', artifacts={}, runtime=None)
        if i == 0:
            directory = out/'construction/objects'/f'obj_{i:02d}/rvg'; directory.mkdir(parents=True)
            Image.fromarray(rgba).save(directory/'view_00.png')
            for name in ('rvg_mesh.ply', 'rvg_gs.ply'):
                vertices = np.zeros(2, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('opacity', 'f4')])
                PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(directory/name))
                row['artifacts'][name] = identity(directory/name)
            row['runtime'] = dict(status='generated', seed=42, wall_s=1.0,
                view_manifest_sha256=sha(out/'view_manifest.json'), source_discovery_hashes=manifest['source_discovery_hashes'],
                consumed_views=audit.rvg.verify_consumed_views(out, i, [view]))
        elif i == 1:
            row['runtime'] = dict(status='generation_failed', wall_s=0.1, reason='RuntimeError: input.numel() == 0')
        if row['runtime'] is not None:
            dump(out/'producer_records'/f'object_{i:02d}.json', row['runtime'])
        rows.append(row)
    pool = {'rows':rows, 'exit_code':1, 'planned_jobs':4}
    return out, pool, manifest


def test_partial_keeps_success_typed_failure_unknown_suffix_and_preparation(partial):
    out, pool, manifest = partial
    before = copy.deepcopy(pool)
    rows = audit.terminal_rows(out, pool, manifest)
    assert len(rows) == 4 and pool == before
    assert [r['normalized_status'] for r in rows] == ['available_verified', 'unavailable', 'unavailable', 'unavailable']
    assert [r['attempt_classification'] for r in rows] == ['completed_generation', 'proven_failed_attempt', 'unknown_unattempted_suffix', 'preparation_unavailable']
    assert [r['generation_attempt_proven'] for r in rows] == [True, True, False, False]
    assert rows[1]['reason'] == 'producer_reported_generation_failure'
    assert rows[2]['reason'] == 'no_per_object_completion_record_process_exit_1'
    assert rows[1]['producer_runtime_record']['sha256'] == sha(out/'producer_records/object_01.json')


@pytest.mark.parametrize('change', ['runtime', 'extra_record', 'png', 'source', 'runtime_nan', 'runtime_negative'])
def test_partial_rejects_runtime_and_consumed_image_tamper(partial, change):
    out, pool, manifest = partial
    if change == 'runtime':
        dump(out/'producer_records/object_00.json', {})
    elif change == 'extra_record':
        dump(out/'producer_records/object_02.json', {})
    elif change == 'png':
        Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8)).save(out/'construction/objects/obj_00/rvg/view_00.png')
    elif change == 'source':
        manifest['source_discovery_hashes'] = {'input':'d'*64}
    else:
        pool['rows'][1]['runtime']['wall_s'] = float('nan') if change == 'runtime_nan' else -1
        dump(out/'producer_records/object_01.json', pool['rows'][1]['runtime'])
    with pytest.raises((PilotError, ValueError)):
        audit.terminal_rows(out, pool, manifest)


@pytest.mark.parametrize('change', ['hash', 'nonfinite', 'empty'])
def test_partial_quarantines_bad_artifact_without_losing_denominator(partial, change):
    out, pool, manifest = partial
    from pathlib import Path
    path = Path(pool['rows'][0]['artifacts']['rvg_gs.ply']['path'])
    if change == 'hash':
        path.write_bytes(b'changed')
    else:
        vertices = np.zeros(0 if change == 'empty' else 2, dtype=[('x', 'f4'), ('opacity', 'f4')])
        if change == 'nonfinite': vertices['opacity'] = np.inf
        PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(path))
        pool['rows'][0]['artifacts']['rvg_gs.ply'] = identity(path)
    rows = audit.terminal_rows(out, pool, manifest)
    assert len(rows) == 4 and rows[0]['normalized_status'] == 'artifact_invalid'
    expected = {'hash':'hash_or_size_mismatch', 'nonfinite':'nonfinite_numeric_field', 'empty':'empty_vertices'}[change]
    assert rows[0]['artifact_validation_errors'][0]['reason'] == expected


def test_zero_generation_behavior_remains_identical(partial):
    out, pool, manifest = partial
    for path in (out/'producer_records').iterdir(): path.unlink()
    for row in pool['rows']:
        row.update(status='generation_failed' if row['prepared'] else 'unavailable', runtime=None, artifacts={})
    legacy = audit.failure_rows(pool)
    rows = audit.terminal_rows(out, pool, manifest)
    assert [{key: row[key] for key in legacy[0]} for row in rows] == legacy


def test_partial_report_replays_canonical_adapter(partial, monkeypatch):
    from robo.manifest.hash import canonical_hash
    out, pool, manifest = partial
    root = out.parent/'freeze'
    directory = root/'rvg_initial/6115eddb86'; directory.parent.mkdir(parents=True)
    # Keep artifact paths stable by placing the fixture below the required freeze.
    import shutil
    shutil.copytree(out, directory)
    for row in pool['rows']:
        for record in row['artifacts'].values(): record['path'] = record['path'].replace(str(out), str(directory))
        if row.get('runtime') and row['runtime']['status'] == 'generated':
            views = json.loads((directory/'view_manifest.json').read_text())['rows'][0]['views']
            row['runtime']['consumed_views'] = audit.rvg.verify_consumed_views(directory, 0, views)
            dump(directory/'producer_records/object_00.json', row['runtime'])
    config = directory/'producer.yaml'; config.write_text('scene_id: 6115eddb86\n')
    manifest.update(code_commit='a'*40, config_sha256=sha(config))
    dump(directory/'input_manifest.json', manifest)
    (directory/'proposal_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pool['rows']))
    pool.update(code_commit='a'*40, freeze_id='freeze', input_manifest_sha256=sha(directory/'input_manifest.json'),
                proposal_records_sha256=sha(directory/'proposal_records.jsonl'),
                view_manifest_sha256=sha(directory/'view_manifest.json'))
    dump(directory/'views_receipt.json', {'synthetic':True})
    pool['views_receipt_sha256'] = sha(directory/'views_receipt.json')
    dump(directory/'proposal_pool.json', pool)
    dump(directory/'execution_claim.json', {k:pool[k] for k in ('code_commit','input_manifest_sha256','views_receipt_sha256')})
    dump(directory/'runtime.json', {'job_id':'123'})
    (directory/'generation.log').write_text('RuntimeError: input.numel() == 0')
    contract = dict(freeze_id='freeze', code={'commit':'a'*40, 'dirty':False},
                    resource_inventory=[{'kind':'experiment_config', 'sha256':sha(config)}])
    contract['contract_sha256'] = canonical_hash(contract)
    dump(root/'contract/freeze_manifest.json', contract)
    rows = audit.terminal_rows(directory, pool, manifest)
    report = dict(schema_version=1, paper_ready=False, source_checkout_clean=True, independent_source_binding_pass=True,
        producer_source_commit='a'*40, producer_freeze_id='freeze', scene_id='6115eddb86', config_sha256=sha(config),
        producer_config=identity(config), source_discovery_hashes=manifest['source_discovery_hashes'], planned_jobs=4,
        producer_process_exit_code=1, job_id='123', scheduler='123|FAILED|1:0|hala|10\n', rows=rows,
        counts=dict(Counter(r['normalized_status'] for r in rows)), process_failure_classification='tool_generation_empty_sparse_coordinates')
    report.update({name.replace('.','_')+'_sha256':sha(directory/name) for name in audit.CLOSURE})
    path = root/'terminal_audit/audit.json'; dump(path, report)
    monkeypatch.setattr(audit.rvg, 'read_manifest', lambda *_:(manifest, {}))
    monkeypatch.setattr(audit.rvg, 'validate_view_receipt', lambda *_:{})
    normalized, _ = audit.terminal_pool_rows(directory, pool, manifest, dict(path=str(path), sha256=sha(path), freeze_root=str(root)))
    assert [r['status'] for r in normalized] == ['available', 'unavailable', 'unavailable', 'unavailable']
    assert normalized[1]['runtime']['status'] == 'generation_failed'
    assert normalized[2]['runtime'] is None
    report['rows'][1]['reason'] = 'no_per_object_completion_record_process_exit_1'
    dump(path, report)
    with pytest.raises(ValueError, match='artifact audit does not replay'):
        audit.terminal_pool_rows(directory, pool, manifest, dict(path=str(path), sha256=sha(path), freeze_root=str(root)))


def test_known_runtime_cannot_claim_missing_record(partial):
    from agents.orchestrator.automatic_inventory import terminal_unavailable_reason
    _, pool, _ = partial
    assert terminal_unavailable_reason(pool['rows'][1], 1) == 'producer_reported_generation_failure'
    pool['rows'][1]['runtime'] = {'status':'unavailable', 'reason':'fewer_than_two_usable_training_views'}
    assert terminal_unavailable_reason(pool['rows'][1], 1) == 'view_eligibility_unavailable'



def test_completed_invalid_pool_has_artifact_failure_classification(partial):
    from pathlib import Path
    out, pool, manifest = partial
    path = Path(pool['rows'][0]['artifacts']['rvg_gs.ply']['path'])
    vertices = np.zeros(1, dtype=[('x', 'f4')]); vertices['x'] = np.inf
    PlyData([PlyElement.describe(vertices, 'vertex')]).write(str(path))
    pool['rows'][0]['artifacts']['rvg_gs.ply'] = identity(path)
    pool['exit_code'] = 0
    rows = audit.terminal_rows(out, pool, manifest)
    assert rows[0]['normalized_status'] == 'artifact_invalid'
    assert audit.failure_classification(pool, rows, '') == 'artifact_validation_failure'


@pytest.mark.parametrize('all_generated', [False, True])
def test_completed_pool_without_invalid_artifacts_requires_normal_audit(partial, all_generated):
    out, pool, manifest = partial
    if all_generated:
        pool['rows'] = pool['rows'][:1]
        pool['planned_jobs'] = 1
        (out/'producer_records/object_01.json').unlink()
    pool['exit_code'] = 0
    rows = audit.terminal_rows(out, pool, manifest)
    with pytest.raises(PilotError, match='requires the normal postrun audit'):
        audit.failure_classification(pool, rows, '')


def test_mask_replay_failure_keeps_distinct_classification(partial):
    out, pool, manifest = partial
    rows = audit.terminal_rows(out, pool, manifest)
    assert audit.failure_classification(pool, rows, 'RVG view mask differs from source-derived recomputation') == 'environment_view_mask_replay_incompatibility'
