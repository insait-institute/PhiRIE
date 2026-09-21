import copy
import json

import pytest

from run.icra2027.e2_fresh_readiness import (
    check_content_disjoint, check_pool, disjoint_evaluation,
)
from run.icra2027.e3_auto_discovery_pilot import identity
from run.icra2027.e3_fresh_generation_contract import FRESH


def test_uses_existing_camera_selector_and_entire_train_boundary():
    split = {'train': ['train.jpg', 'unused_train.jpg'],
             'test': [f'test{i}.jpg' for i in range(11)]}
    names = disjoint_evaluation(split, split['train'] + split['test'], ['train.jpg'], ['train.jpg'])
    assert names == ['test0.jpg', 'test1.jpg', 'test2.jpg', 'test4.jpg',
                     'test5.jpg', 'test7.jpg', 'test8.jpg', 'test10.jpg']
    assert not set(names) & set(split['train'])


@pytest.mark.parametrize('change', ['split_overlap', 'generator_test', 'duplicate_train', 'missing_cameras'])
def test_rejects_view_leakage_and_missing_cameras(change):
    split = {'train': ['a.jpg'], 'test': [f'test{i}.jpg' for i in range(8)]}
    training, generation, cameras = ['a.jpg'], ['a.jpg'], split['test']
    if change == 'split_overlap': split['train'].append('test0.jpg')
    if change == 'generator_test': generation.append('test0.jpg')
    if change == 'duplicate_train': training.append('a.jpg')
    if change == 'missing_cameras': cameras = cameras[:-1]
    with pytest.raises(ValueError):
        disjoint_evaluation(split, cameras, training, generation)


def test_rejects_same_pixels_under_different_filename():
    with pytest.raises(ValueError, match='duplicate'):
        check_content_disjoint(['samehash'], ['samehash'])
    check_content_disjoint(['testhash'], ['trainhash'])


def fixture_pool(tmp_path, *, mutation=None):
    jobs = [{'job_id': 's:auto:1', 'prepared': False}, {'job_id': 's:auto:2', 'prepared': True}]
    source = {'source-file': 'sourcehash'}
    common = dict(source_gaussian_training_provenance=FRESH, source_discovery_hashes=source,
                  code_commit='c'*40)
    inputs = dict(common, jobs=copy.deepcopy(jobs))
    if mutation == 'dropped_input': inputs['jobs'].pop()
    path = tmp_path/'input_manifest.json';path.write_text(json.dumps(inputs))
    pool = dict(common, rows=copy.deepcopy(jobs), planned_jobs=len(jobs),
                input_manifest_sha256=identity(path)['sha256'])
    if mutation == 'mixed_source': pool['source_discovery_hashes'] = {'other': 'hash'}
    if mutation == 'unknown': pool['source_gaussian_training_provenance'] = 'UNKNOWN'
    if mutation == 'dropped_pool': pool['rows'].pop()
    if mutation == 'changed_prepared': pool['rows'][0]['prepared'] = True
    if mutation == 'count_drift': pool['planned_jobs'] = 6
    path = tmp_path/'proposal_pool.json';path.write_text(json.dumps(pool))
    audit = dict(common, producer_source_commit=common['code_commit'],
                 input_manifest_sha256=pool['input_manifest_sha256'],
                 proposal_pool_sha256=identity(path)['sha256'])
    if mutation == 'wrong_receipt': audit['proposal_pool_sha256'] = '0'*64
    (tmp_path/'postrun_audit.json').write_text(json.dumps(audit))
    spec = {name: identity(tmp_path/name) for name in
            ('proposal_pool.json','input_manifest.json','postrun_audit.json')}
    return spec, jobs, source


def test_retains_failed_discovery_denominator(tmp_path):
    spec, jobs, source = fixture_pool(tmp_path)
    pool, _ = check_pool(spec, jobs, source)
    assert len(pool['rows']) == 2 and pool['rows'][0]['prepared'] is False


@pytest.mark.parametrize('mutation', ['mixed_source', 'unknown', 'dropped_input', 'dropped_pool',
                                     'changed_prepared', 'count_drift', 'wrong_receipt'])
def test_rejects_pool_source_population_or_receipt_drift(tmp_path, mutation):
    spec, jobs, source = fixture_pool(tmp_path, mutation=mutation)
    with pytest.raises(ValueError): check_pool(spec, jobs, source)


def test_rejects_modified_anchored_pool(tmp_path):
    spec, jobs, source = fixture_pool(tmp_path)
    with (tmp_path/'proposal_pool.json').open('a') as f: f.write(' ')
    with pytest.raises(ValueError, match='anchor changed'):
        check_pool(spec, jobs, source)
