import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from robo.roundtrip.capture import (CaptureError, CaptureSpec, backproject_camera_z,
    capture_static, constructor_command, validate_camera, validate_public_capture)


class Adapter:
    def __init__(self, motion=False, depth=True, semantics="camera_z"):
        self.pose = np.array([1., 2., 3., 1., 0., 0., 0.])
        self.motion, self.depth, self.semantics = motion, depth, semantics
        self.frames = 0

    def get_named_body_state(self):
        return {"private_native_asset_model42": self.pose.copy()}

    def get_state(self):
        return {"private_registry_id": "native_model42", "pose": self.pose.copy()}

    def render_capture(self, camera, *, width, height):
        self.frames += 1
        if self.motion:
            self.pose[0] += 0.001
        T = np.eye(4)
        T[0, 3] = camera["x"]
        K = np.array([[2.+camera.get("k", 0), 0., width/2], [0., 2., height/2], [0., 0., 1.]])
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 7, 13]  # asymmetric RGB witness
        return dict(rgb=rgb, depth_m=np.full((height,width), 2., dtype=np.float32) if self.depth else None,
                    K=K, T_world_from_camera=T, timestamp=0., depth_semantics=self.semantics)


def frames():
    return [dict(frame_id=f"f{i:06}", split="train" if i < 6 else "test",
                 camera={"x": i*0.1, "native_name": "private-camera", "k": i%2}) for i in range(8)]


def spec(**kwargs):
    return CaptureSpec(capture_id="c-0123456789abcdef", width=4, height=3,
                       counts={"train":6,"dev":0,"test":2}, **kwargs)


def capture(tmp_path, *, adapter=None, frame_rows=None, capture_spec=None):
    return capture_static(adapter or Adapter(), spec=capture_spec or spec(),
                          frames=frames() if frame_rows is None else frame_rows,
                          public_out=tmp_path/"public", vault=tmp_path/"private",
                          robot_config={"robot":"known_robot"}, task_instruction="Put the object on the counter")


def test_static_capture_train_only_private_holdout_and_deterministic(tmp_path):
    first=capture(tmp_path/"one")
    second=capture(tmp_path/"two")
    assert first["manifest"] == second["manifest"]
    pub, private=Path(first["public"]),Path(first["private"])
    assert validate_public_capture(pub)["train_frames"] == 6
    assert len(list((pub/"train/rgb").glob("*.png"))) == 6
    assert len(list((private/"test/rgb").glob("*.png"))) == 2
    assert not (pub/"test").exists()
    for p in pub.rglob("*"):
        if p.suffix in {".json", ".jsonl", ".txt"}:
            assert "private" not in p.read_text()
            assert "native_model42" not in p.read_text()
    receipt=json.loads((private/"capture_receipt.json").read_text())
    assert receipt["counts"] == {"train":6,"dev":0,"test":2}
    from PIL import Image
    assert list(np.asarray(Image.open(pub/"train/rgb/f000000.png"))[0,0]) == [255,7,13]


def test_plane_backprojection_z_not_range_and_inverse():
    K=np.array([[2.,0.,1.],[0.,2.,1.],[0.,0.,1.]])
    T=np.eye(4); T[:3,:3]=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]];T[:3,3]=[3.,4.,5.]
    points,valid=backproject_camera_z(np.full((3,3),2.,dtype=np.float32),K,T)
    homogeneous=np.concatenate([points,np.ones((3,3,1))],axis=2)
    camera=homogeneous @ np.linalg.inv(T).T
    assert np.allclose(camera[...,2],2.)
    assert np.allclose(camera[0,0,:3],[-1.,-1.,2.])
    assert np.linalg.norm(camera[0,0,:3]) > 2.
    assert valid.all()


@pytest.mark.parametrize("change", ["scale", "reflection", "last_row", "K"])
def test_invalid_camera(change):
    K=np.array([[2.,0.,1.],[0.,2.,1.],[0.,0.,1.]])
    T=np.eye(4)
    if change=="scale":T[0,0]=1000.
    elif change=="reflection":T[0,0]=-1.
    elif change=="last_row":T[3,3]=2.
    else:K[0,0]=0.
    with pytest.raises(CaptureError):validate_camera(K,T,width=3,height=3)


def test_mixed_state_invalidates_capture_and_preserves_attempt(tmp_path):
    with pytest.raises(CaptureError,match="mixed-state"):
        capture(tmp_path,adapter=Adapter(motion=True))
    assert (tmp_path/"private"/spec().capture_id/"capture_failure.json").is_file()
    assert not (tmp_path/"public"/spec().capture_id/"capture_manifest.json").exists()
    with pytest.raises(CaptureError,match="already exists"):
        capture(tmp_path)


def test_intermediate_motion_rejected_even_if_final_state_would_restore(tmp_path):
    class Moving(Adapter):
        def render_capture(self,*a,**kw):
            value=super().render_capture(*a,**kw)
            self.pose[0]=1.001 if self.frames==1 else 1.
            return value
    with pytest.raises(CaptureError,match="mixed-state"):
        capture(tmp_path,adapter=Moving())


def test_duplicate_frame_ids_across_splits(tmp_path):
    rows=frames();rows[-1]["frame_id"]=rows[0]["frame_id"]
    with pytest.raises(CaptureError,match="duplicate frame"):
        capture(tmp_path,frame_rows=rows)
    assert not (tmp_path/"public").exists()


def test_duplicate_pose_across_splits(tmp_path):
    rows=frames();rows[-1]["camera"]=copy.deepcopy(rows[0]["camera"])
    with pytest.raises(CaptureError,match="duplicate camera"):
        capture(tmp_path,frame_rows=rows)


@pytest.mark.parametrize("adapter", [Adapter(depth=False), Adapter(semantics="ray_range")])
def test_missing_depth_or_wrong_semantics_rejected(tmp_path,adapter):
    with pytest.raises(CaptureError):capture(tmp_path,adapter=adapter)


@pytest.mark.parametrize("depth", [np.ones((3,3),dtype=np.uint16), np.full((3,3),np.nan), -np.ones((3,3))])
def test_depth_units_type_and_invalid_pixels(depth):
    with pytest.raises(CaptureError):backproject_camera_z(depth,np.eye(3),np.eye(4))


def test_zero_missing_depth_not_geometry():
    points,mask=backproject_camera_z(np.array([[0.,2.],[2.,2.]]),np.eye(3),np.eye(4))
    assert not mask[0,0]
    assert mask.sum()==3


def test_rgb_only_does_not_serialize_depth(tmp_path):
    result=capture(tmp_path,capture_spec=spec(sensor_regime="posed_rgb"))
    pub=Path(result['public'])
    assert not list(pub.rglob("*.npy"))
    assert not any('depth' in p.name for p in pub.rglob('*'))
    assert validate_public_capture(pub)['depth_semantics'] is None


def test_known_robot_config_not_native_metadata(tmp_path):
    with pytest.raises(CaptureError,match="undeclared public metadata"):
        capture_static(Adapter(),spec=spec(),frames=frames(),public_out=tmp_path/'public',vault=tmp_path/'private',
                       robot_config={'native_asset_id':'secret'},task_instruction='Put object away')


def test_private_id_in_instruction_rejected(tmp_path):
    with pytest.raises(CaptureError,match="private identifier"):
        capture_static(Adapter(),spec=spec(),frames=frames(),public_out=tmp_path/'public',vault=tmp_path/'private',
                       robot_config={},task_instruction='Put native_model42 away',forbidden_public_tokens=['native_model42'])


def test_vault_must_be_outside_public_tree(tmp_path):
    with pytest.raises(CaptureError,match="must not overlap"):
        capture_static(Adapter(),spec=spec(),frames=frames(),public_out=tmp_path,vault=tmp_path/'private',
                       robot_config={},task_instruction='Put object away')


def test_hash_mutation_and_extra_heldout_rejected(tmp_path):
    result=capture(tmp_path); pub=Path(result['public'])
    (pub/'train/rgb/f000000.png').write_bytes(b'changed')
    with pytest.raises(CaptureError,match="hash mismatch"):validate_public_capture(pub)
    other=capture(tmp_path/'other');path=Path(other['public']);(path/'test').mkdir();(path/'test/secret.txt').write_text('GT')
    with pytest.raises(CaptureError,match="roster differs"):validate_public_capture(path)


def test_constructor_forbidden_ancestor_mount_rejected(tmp_path):
    result=capture(tmp_path);out=tmp_path/'out';out.mkdir()
    with pytest.raises(CaptureError,match="private/native"):
        constructor_command(bubblewrap='/usr/bin/true',public_capture=result['public'],output=out,
            runtime_mounts={str(tmp_path):'/deps'},forbidden_roots=[tmp_path/'private'],argv=['/usr/bin/python3'])


def test_constructor_real_kernel_boundary(tmp_path):
    bwrap=os.environ.get('SIMANY_TEST_BWRAP')
    if not bwrap:
        pytest.skip('set SIMANY_TEST_BWRAP for actual user-namespace isolation test')
    result=capture(tmp_path)
    native=tmp_path/'native_assets';native.mkdir();(native/'secret_mesh.obj').write_text('reference mesh')
    code=tmp_path/'worker';code.mkdir()
    script='''import os,pathlib,socket
assert not pathlib.Path(%r).exists()
assert not pathlib.Path(%r).exists()
assert pathlib.Path('/capture/train/cameras.jsonl').is_file()
assert not pathlib.Path('/capture/test').exists()
try: pathlib.Path('/capture/forbidden_write').write_text('bad')
except OSError: pass
else: raise AssertionError('public capture writable')
assert 'SR1_SECRET_TOKEN' not in os.environ
s=socket.socket();s.settimeout(.1)
try: s.connect(('1.1.1.1',443))
except OSError: pass
else: raise AssertionError('network escaped namespace')
pathlib.Path('/output/result.json').write_text('{"isolation":"PASS"}')
''' % (str(tmp_path/'private'),str(native/'secret_mesh.obj'))
    (code/'worker.py').write_text(script)
    out=tmp_path/'out';out.mkdir()
    runtime={p:p for p in ['/usr','/lib','/lib64'] if Path(p).exists()};runtime[str(code)]='/worker'
    command=constructor_command(bubblewrap=bwrap,public_capture=result['public'],output=out,
        runtime_mounts=runtime,forbidden_roots=[tmp_path/'private',native],argv=['/usr/bin/python3','/worker/worker.py'])
    completed=subprocess.run(command,text=True,capture_output=True,env={**os.environ,'SR1_SECRET_TOKEN':'secret'},close_fds=True,timeout=15)
    assert completed.returncode==0,completed.stderr
    assert json.loads((out/'result.json').read_text())=={'isolation':'PASS'}


def test_cuda_device_allocation_validation(monkeypatch):
    from robo.roundtrip.capture import _cuda_device_paths
    from types import SimpleNamespace
    import stat
    monkeypatch.setenv('SLURM_JOB_ID','123')
    monkeypatch.setenv('SLURM_JOB_GPUS','2-3')
    monkeypatch.delenv('SLURM_STEP_GPUS',raising=False)
    monkeypatch.setattr(os,'lstat',lambda path:SimpleNamespace(st_mode=stat.S_IFCHR | 0o666))
    paths=['/dev/nvidia2','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
    assert _cuda_device_paths(paths)==paths
    with pytest.raises(CaptureError,match='outside'):_cuda_device_paths(['/dev/nvidia0'])
    monkeypatch.setenv('SLURM_STEP_GPUS','3')
    with pytest.raises(CaptureError,match='outside'):_cuda_device_paths(['/dev/nvidia2'])


@pytest.mark.parametrize('path',['/dev','/dev/null','/dev/dri/renderD128','/private/native_asset','/dev/nvidia0/../../private'])
def test_cuda_rejects_arbitrary_paths(monkeypatch,path):
    from robo.roundtrip.capture import _cuda_device_paths
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURM_JOB_GPUS','0')
    monkeypatch.delenv('SLURM_STEP_GPUS',raising=False)
    with pytest.raises(CaptureError,match='individual NVIDIA'):_cuda_device_paths([path])


@pytest.mark.parametrize('kind',['file','symlink','directory'])
def test_cuda_rejects_non_character_nodes(monkeypatch,kind):
    from robo.roundtrip.capture import _cuda_device_paths
    from types import SimpleNamespace
    import stat
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURM_JOB_GPUS','0')
    monkeypatch.delenv('SLURM_STEP_GPUS',raising=False)
    mode={'file':stat.S_IFREG,'symlink':stat.S_IFLNK,'directory':stat.S_IFDIR}[kind]
    monkeypatch.setattr(os,'lstat',lambda path:SimpleNamespace(st_mode=mode | 0o777))
    with pytest.raises(CaptureError,match='real character device'):_cuda_device_paths(['/dev/nvidia0'])


def test_cuda_requires_allocation_and_no_uuid_fallback(monkeypatch):
    from robo.roundtrip.capture import _cuda_device_paths
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    assert _cuda_device_paths([])==[]
    with pytest.raises(CaptureError,match='Slurm allocation'):_cuda_device_paths(['/dev/nvidia0'])
    monkeypatch.setenv('SLURM_JOB_ID','123');monkeypatch.setenv('SLURM_JOB_GPUS','GPU-abcd')
    monkeypatch.delenv('SLURM_STEP_GPUS',raising=False)
    with pytest.raises(CaptureError,match='indices unavailable'):_cuda_device_paths(['/dev/nvidia0'])
