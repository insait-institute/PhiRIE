"""Regression gates for opt-in TRELLIS.2 in the existing proposal runner."""
import copy
import json
from pathlib import Path

import pytest

from run.icra2027 import e3_trellis_generation_pilot as runner
from run.icra2027.e3_generator_backend import (
    TRELLIS2_OUTPUTS, environment, generator, validate_config,
)


def config(tmp_path):
    models = {key: {'path': str(tmp_path/key), 'sha256': '0'*64} for key in (
        'trellis2_source', 'trellis2_snapshot', 'dinov3_snapshot',
        'ss_decoder_config', 'ss_decoder_weight')}
    return dict(generator='trellis2', models=models, upstream_commit='a'*40,
                pipeline_type='512', study_scope='trellis2_mesh_engineering',
                native_gaussian=False, seed=42)


def test_opt_in_output_identity_cannot_alias_frozen_trellis(tmp_path):
    assert generator({}) == 'trellis'
    assert runner.output_directory({}, tmp_path) == tmp_path/'trellis_initial'
    c = config(tmp_path)
    assert runner.output_directory(c, tmp_path) == tmp_path/'trellis2_initial'
    c.update(output_scene_id='38d58a7a31', source_pilot='/discovery/38d58a7a31')
    assert runner.output_directory(c, tmp_path).name == '38d58a7a31'
    with pytest.raises(runner.PilotError, match='unknown'):
        generator({'generator': 'trellis3'})


@pytest.mark.parametrize('change,message', [
    ({'raw_reuse': {'anything': True}}, 'cannot reuse'),
    ({'native_gaussian': True}, 'no native Gaussian'),
    ({'pipeline_type': 'auto'}, 'pipeline_type'),
    ({'study_scope': 'automatic_training_only_engineering'}, 'separate mesh'),
    ({'upstream_commit': 'main'}, 'pinned'),
])
def test_invalid_new_treatment_fails_closed(tmp_path, change, message):
    c = config(tmp_path)
    validate_config(c)
    c.update(change)
    with pytest.raises(runner.PilotError, match=message):
        validate_config(c)


def test_explicit_environment_ignores_inherited_v1_model(tmp_path, monkeypatch):
    monkeypatch.setenv('SIMANY_TRELLIS_MODEL', '/wrong-v1-model')
    c = config(tmp_path)
    env = environment(c, tmp_path, tmp_path/'out')
    assert env['SIMANY_TRELLIS2_MODEL'] == c['models']['trellis2_snapshot']['path']
    assert env['SIMANY_TRELLIS2_PIPELINE_TYPE'] == '512'
    assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'


def test_same_discovery_population_distinct_proposal_ids(tmp_path, monkeypatch):
    c = config(tmp_path)
    c['freeze_id'] = 'new-freeze'
    jobs = [dict(job_id='scene:auto:1', prepared=True),
            dict(job_id='scene:auto:2', prepared=False)]
    from run.icra2027 import e3_fresh_generation_contract as shared
    monkeypatch.setattr(shared, 'discovery_binding', lambda c, loader: (copy.deepcopy(jobs), {}))
    rows, _, _, _ = runner.planned_jobs(c, tmp_path)
    assert [r['job_id'] for r in rows] == [r['job_id'] for r in jobs]
    assert all(':trellis2:initial:seed42' in r['proposal_id'] for r in rows)
    assert rows[1]['prepared'] is False


def test_denominator_failures_and_no_borrowed_gaussian(tmp_path):
    rows = [dict(job_id=f'j{i}', automatic_instance_id=1000+i,
                 proposal_id=f'p{i}', prepared=i<2, output_index=i,
                 input={'sha256': 'input'}, frame='train.jpg') for i in range(3)]
    obj = tmp_path/'construction/objects/obj_00'
    obj.mkdir(parents=True)
    for name in TRELLIS2_OUTPUTS:
        (obj/name).write_bytes(b'artifact')
    records = tmp_path/'producer_records'
    records.mkdir()
    (records/'object_00.json').write_text(json.dumps(dict(status='generated', seed=42, generator='trellis2',input_sha256='input',pipeline_type='512',artifacts={name:dict(sha256=runner.sha(obj/name),bytes=(obj/name).stat().st_size) for name in TRELLIS2_OUTPUTS})))
    manifest = dict(generator='trellis2', pipeline_type='512', jobs=rows)
    result = runner.collect_records(tmp_path, manifest, 1)
    assert [r['status'] for r in result] == ['available','generation_failed','unavailable']
    assert all(r['tool']=='trellis2' and r['native_gaussian'] is False for r in result)
    assert all(r['shared_initial_policy_rows']==[] and not r['full_twin_ready'] for r in result)
    assert 'trellis_gs.ply' not in result[0]['artifacts']
    (obj/'trellis2_pbr.npz').unlink()
    assert runner.collect_records(tmp_path, manifest, 0)[0]['status']=='generation_failed'


def test_manifest_generator_swap_rejected_before_source_read(tmp_path):
    c = config(tmp_path)
    with pytest.raises(runner.PilotError, match='staged generator'):
        runner.validate_source_binding(c, tmp_path, {'generator':'trellis','jobs':[]})
