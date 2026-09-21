import json
from pathlib import Path

import numpy as np
import pytest

from robo.roundtrip.capture import CaptureSpec, capture_static
from robo.roundtrip.gaussian_build import prepare, validate_initialization, sampled_points, sha


class CameraAdapter:
    def get_named_body_state(self): return {'secret_native_asset': [0, 0, 0, 1, 0, 0, 0]}
    def get_state(self): return {'secret_native_state': 42}
    def render_capture(self, camera, *, width, height):
        T = np.eye(4); T[0, 3] = camera['offset']
        return {'rgb': np.full((height,width,3), 127, np.uint8),
                'depth_m': np.full((height,width),2.,np.float32),
                'K': [[10.,0.,width/2],[0.,10.,height/2],[0.,0.,1.]],
                'T_world_from_camera': T, 'timestamp': 0., 'depth_semantics':'camera_z'}


@pytest.fixture
def capture(tmp_path):
    pytest.importorskip('plyfile', reason='optional Gaussian initialization serialization runtime')
    result = capture_static(CameraAdapter(), spec=CaptureSpec(capture_id='c-0123456789abcdef',
        width=16,height=16,counts={'train':6,'dev':0,'test':2}),
        frames=[{'frame_id':f'f{i:06}','split':'train' if i<6 else 'test',
                 'camera':{'offset':i*.1}} for i in range(8)],
        public_out=tmp_path/'public',vault=tmp_path/'private',robot_config={'robot':'known_robot'},task_instruction='Move object')
    return Path(result['public'])


def test_train_only_initialization_no_private_reads(capture,tmp_path,monkeypatch):
    original=Path.open
    def guarded(self,*args,**kwargs):
        if str(tmp_path/'private') in str(self): raise AssertionError('heldout/private read')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    out=tmp_path/'build';result=prepare(capture,out,stride=4)
    inputs,init=validate_initialization(out/'scene',out/'init_points.ply',out/'init_manifest.json',capture)
    assert result==init and init['n_points']>=4
    assert len(inputs['frames'])==6 and init['heldout_frames_read']==0
    assert 'secret_native' not in json.dumps(inputs)
    assert init['metric_scale_fitted'] is False
    with pytest.raises(FileExistsError): prepare(capture,out,stride=4)


def test_backprojection_metric_camera_z_and_rotation():
    K=np.array([[2.,0.,1.],[0.,2.,1.],[0.,0.,1.]])
    T=np.eye(4);T[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];T[:3,3]=[3,4,5]
    xyz,_=sampled_points(np.full((3,3),2.,np.float32),np.zeros((3,3,3),np.uint8),K,T,1)
    assert np.allclose(xyz[0],[4,3,7])
    recovered=(xyz-T[:3,3])@T[:3,:3]
    assert np.allclose(recovered[:,2],2.)
    assert np.linalg.norm(recovered[0])>2.


def test_reject_changed_initialization_bytes(capture,tmp_path):
    out=tmp_path/'build';prepare(capture,out,stride=4)
    with (out/'init_points.ply').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='bytes changed'):
        validate_initialization(out/'scene',out/'init_points.ply',out/'init_manifest.json',capture)


def test_reject_staged_pose_change(capture,tmp_path):
    out=tmp_path/'build';prepare(capture,out,stride=4)
    p=out/'scene/native_training_inputs.json';x=json.loads(p.read_text());x['frames'][0]['w2c'][0][3]+=1;p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='bytes changed'):
        validate_initialization(out/'scene',out/'init_points.ply',out/'init_manifest.json',capture)


def reseal_camera_rows(capture,rows):
    p=capture/'train/cameras.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    m=capture/'capture_manifest.json';x=json.loads(m.read_text());x['files']['train/cameras.jsonl']=sha(p);m.write_text(json.dumps(x))


def test_reject_hidden_frame_path_even_resealed_camera_row(capture,tmp_path):
    rows=[json.loads(l) for l in (capture/'train/cameras.jsonl').read_text().splitlines()]
    rows[0]['rgb']='../private/test/rgb/f000006.png';reseal_camera_rows(capture,rows)
    with pytest.raises(ValueError,match='escapes'):
        prepare(capture,tmp_path/'build',stride=4)


def test_reject_unsupported_mixed_intrinsics(capture,tmp_path):
    rows=[json.loads(l) for l in (capture/'train/cameras.jsonl').read_text().splitlines()]
    rows[0]['K'][0][0]+=1;reseal_camera_rows(capture,rows)
    with pytest.raises(ValueError,match='mixed K'):
        prepare(capture,tmp_path/'build',stride=4)


def test_reject_ray_range_semantics(capture,tmp_path):
    p=capture/'capture_manifest.json';x=json.loads(p.read_text());x['depth_semantics']='ray_range_meters';p.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='metric RGB-D'):
        prepare(capture,tmp_path/'build',stride=4)


def test_reject_extra_staged_heldout(capture,tmp_path):
    out=tmp_path/'build';prepare(capture,out,stride=4)
    (out/'scene/dslr/resized_undistorted_images/f000006.png').write_bytes(b'not TRAIN')
    with pytest.raises(ValueError,match='extra/missing'):
        validate_initialization(out/'scene',out/'init_points.ply',out/'init_manifest.json',capture)
