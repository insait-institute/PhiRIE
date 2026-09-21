import json
from pathlib import Path
import pytest
from tests.test_roundtrip_gaussian_build import capture
from robo.roundtrip.gaussian_build import sha
from robo.roundtrip.gaussian_eval import sealed_views


@pytest.fixture
def config(capture,tmp_path):
    build=tmp_path/'sealed-build';(build/'pilot').mkdir(parents=True)
    # Hash-only source fixture; unit tests never rasterize this synthetic file.
    ply=build/'pilot/scene.ply';ply.write_bytes(b'frozen synthetic source')
    digest=sha(capture/'capture_manifest.json')
    (build/'gs_build_manifest.json').write_text(json.dumps({'status':'BUILT','source_gaussian':'MEASURED',
        'capture_manifest_sha256':digest,'phases':{'pilot':{'scene_ply_sha256':sha(ply)}}}))
    vault=tmp_path/'private'/capture.name
    return {'source_build':str(build),'source_build_manifest_sha256':sha(build/'gs_build_manifest.json'),
        'source_ply_sha256':sha(ply),'public_capture':str(capture),'public_capture_manifest_sha256':digest,
        'private_capture':str(vault),'private_capture_receipt_sha256':sha(vault/'capture_receipt.json'),'planned_views':2}


def test_exact_original_two_heldouts(config):
    _,rows,_=sealed_views(config)
    assert [r['frame_id'] for r in rows]==['f000006','f000007']


def test_changed_source_gs_rejected(config):
    (Path(config['source_build'])/'pilot/scene.ply').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Gaussian bytes'):sealed_views(config)


def test_changed_evaluation_camera_rejected(config):
    p=Path(config['private_capture'])/'test/cameras.jsonl';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='camera bytes'):sealed_views(config)


def test_narrowed_heldout_count_rejected(config):
    config['planned_views']=1
    with pytest.raises(ValueError,match='exactly the two'):sealed_views(config)
