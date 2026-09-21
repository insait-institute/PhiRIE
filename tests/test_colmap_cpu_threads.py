"""Actual pycolmap resource options preserve all default scientific settings."""
import copy
import json

import pytest

from agents.recon import colmap_poses as producer


@pytest.mark.parametrize('threads', [1, 2, 8])
def test_native_options_change_only_thread_counts(threads):
    pycolmap = pytest.importorskip('pycolmap')
    actual = producer.cpu_thread_options(threads)
    defaults = (pycolmap.FeatureExtractionOptions(), pycolmap.FeatureMatchingOptions(), pycolmap.IncrementalPipelineOptions())
    for index, (configured, default) in enumerate(zip(actual, defaults)):
        expected = copy.deepcopy(default.todict())
        expected['num_threads'] = threads
        if index == 2: expected['mapper']['num_threads'] = threads
        assert configured.todict() == expected


@pytest.mark.parametrize('threads', [-1, 0, None, True, 1.5])
def test_reject_automatic_or_invalid_thread_counts(threads):
    pytest.importorskip('pycolmap')
    with pytest.raises(ValueError, match='positive integer'): producer.cpu_thread_options(threads)


def test_actual_producer_forwards_native_options(tmp_path, monkeypatch):
    pycolmap = pytest.importorskip('pycolmap')
    monkeypatch.setattr('os.sched_getaffinity', lambda _: set(range(8)))
    calls = {}
    monkeypatch.setattr(pycolmap, 'extract_features', lambda *a, **k: calls.update(extract=k))
    monkeypatch.setattr(pycolmap, 'match_sequential', lambda *a, **k: calls.update(match=k))
    class Reconstruction:
        def num_reg_images(self): return 10
    def mapping(*args, **kwargs):
        calls['mapping'] = kwargs; return {0: Reconstruction()}
    monkeypatch.setattr(pycolmap, 'incremental_mapping', mapping)
    producer.run_sfm(tmp_path / 'rgb', tmp_path, num_threads=8)
    assert calls['extract']['extraction_options'].num_threads == 8
    assert calls['match']['matching_options'].num_threads == 8
    assert calls['mapping']['options'].num_threads == 8
    assert calls['mapping']['options'].mapper.num_threads == 8
    assert calls['match']['pairing_options'].overlap == producer.SEQ_OVERLAP
    record = json.loads((tmp_path / 'thread_options.json').read_text())
    assert set(record['effective_threads'].values()) == {8}
    assert record['gpu_enabled'] is False
    with pytest.raises(FileExistsError): producer.run_sfm(tmp_path / 'rgb', tmp_path, num_threads=8)


def test_declared_threads_cannot_exceed_affinity(tmp_path, monkeypatch):
    pytest.importorskip('pycolmap')
    monkeypatch.setattr('os.sched_getaffinity', lambda _: {1, 2})
    with pytest.raises(ValueError, match='affinity'): producer.run_sfm(tmp_path / 'rgb', tmp_path, num_threads=8)


def test_legacy_calls_keep_original_library_defaults(tmp_path, monkeypatch):
    pycolmap = pytest.importorskip('pycolmap'); calls = []
    def capture(*a, **kw): calls.append(kw)
    monkeypatch.setattr(pycolmap, 'extract_features', capture)
    monkeypatch.setattr(pycolmap, 'match_sequential', capture)
    monkeypatch.setattr(pycolmap, 'incremental_mapping', lambda *a, **kw: {})
    with pytest.raises(SystemExit, match='no reconstruction'): producer.run_sfm(tmp_path / 'rgb', tmp_path)
    assert 'extraction_options' not in calls[0] and 'matching_options' not in calls[1]
    assert not (tmp_path / 'thread_options.json').exists()


def test_native_sift_creates_only_eight_workers(tmp_path):
    """Regression for32GB OOM caused by hardware-wide native SIFT pools."""
    import subprocess
    import sys
    from pathlib import Path
    from agents.recon.droid_extract import prospective_environment
    pytest.importorskip('pycolmap'); pytest.importorskip('PIL')
    code = '''import sys,json,resource,pycolmap,numpy as np
from pathlib import Path
from PIL import Image
from agents.recon.colmap_poses import cpu_thread_options
p=Path(sys.argv[1]);images=p/'images';images.mkdir();rng=np.random.RandomState(0)
for i in range(4):Image.fromarray(rng.randint(0,256,(384,384,3),dtype=np.uint8)).save(images/f'{i}.jpg')
options,_,_=cpu_thread_options(8)
pycolmap.extract_features(p/'database.db',images,extraction_options=options,device=pycolmap.Device.cpu)
print(json.dumps({'workers':options.num_threads,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
'''
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=root,
        env=prospective_environment(root), capture_output=True, text=True, check=True)
    assert result.stderr.count('Creating SIFT CPU feature extractor') == 8
    assert json.loads(result.stdout)['workers'] == 8
