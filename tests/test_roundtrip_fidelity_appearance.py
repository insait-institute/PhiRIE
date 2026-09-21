import json
import numpy as np
import pytest
from PIL import Image
from robo.roundtrip import fidelity_native as module


def fixture(tmp_path,monkeypatch):
    public=tmp_path/'public/c-test';private=tmp_path/'private/capture/c-test'
    public.mkdir(parents=True);(private/'test').mkdir(parents=True)
    (public/'capture_manifest.json').write_text('{}')
    Image.fromarray(np.zeros((8,8,3),dtype=np.uint8)).save(private/'test/test.png')
    train=dict(frame_id='train',T_world_from_camera=np.eye(4).tolist())
    T=np.eye(4);T[0,3]=.1
    row=dict(frame_id='test',rgb='test/test.png',rgb_sha256=module.file_hash(private/'test/test.png'),K=np.eye(3).tolist(),T_world_from_camera=T.tolist())
    (private/'test/cameras.jsonl').write_text(json.dumps(row)+'\n')
    (private/'capture_plan.json').write_text(json.dumps(dict(spec=dict(counts=dict(test=1),width=8,height=8),frames=[dict(frame_id='test',split='test',camera=dict(T_world_from_camera=T.tolist()))])))
    import robo.roundtrip.build as build
    monkeypatch.setattr(build,'read_train',lambda p: ({},[train]))
    binding=dict(capture_public=str(public),bundle_dir=str(tmp_path/'private/canonical'),capture_manifest_sha256=module.file_hash(public/'capture_manifest.json'))
    return binding,private,train


def test_frozen_heldout_and_overlap_rejection(tmp_path,monkeypatch):
    binding,private,train=fixture(tmp_path,monkeypatch)
    rows,spec=module.checked_heldout(binding)
    assert len(rows)==1 and rows[0]['frame_id']=='test' and spec['width']==8
    train['T_world_from_camera']=rows[0]['T_world_from_camera']
    with pytest.raises(ValueError,match='overlap'):module.checked_heldout(binding)


def test_changed_heldout_image_rejected(tmp_path,monkeypatch):
    binding,private,train=fixture(tmp_path,monkeypatch)
    Image.fromarray(np.ones((8,8,3),dtype=np.uint8)).save(private/'test/test.png')
    with pytest.raises(ValueError,match='RGB changed'):module.checked_heldout(binding)


def test_missing_heldout_retained_as_failure(tmp_path,monkeypatch):
    binding,private,train=fixture(tmp_path,monkeypatch)
    (private/'test/cameras.jsonl').write_text('')
    with pytest.raises(ValueError,match='roster'):module.checked_heldout(binding)


def test_render_only_warmstart_may_refresh():
    mujoco=pytest.importorskip('mujoco')
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    data=mujoco.MjData(model);mask=mujoco.mjtState.mjSTATE_INTEGRATION
    def state():
        v=np.empty(mujoco.mj_stateSize(model,mask));mujoco.mj_getState(model,data,v,mask);return v
    before=state();data.qacc_warmstart[:]=1
    assert module.static_render_state(model,before,state())
    data.qpos[0]+=.001
    with pytest.raises(ValueError,match='physical/control'):module.static_render_state(model,before,state())


def test_perfect_image_psnr_preserved_without_floor():
    a=np.zeros((8,8,3));values={'psnr':float('inf'),'ssim':1.,'lpips':0.}
    result=module.serializable_appearance(values,a,a)
    assert result['psnr'] is None and result['psnr_status']=='POSITIVE_INFINITY' and result['exact_rgb_match']
    json.dumps(result,allow_nan=False)
    with pytest.raises(ValueError,match='inconsistent'):module.serializable_appearance(values,a,a+1)
    with pytest.raises(ValueError,match='inconsistent'):module.serializable_appearance(dict(values,psnr=float('nan')),a,a)


def test_train_render_prelude_preserves_frozen_order_and_camera(tmp_path,monkeypatch):
    from robo.roundtrip.fidelity_native import checked_train_render_prelude,file_hash
    import robo.roundtrip.build as build
    public=tmp_path/'public/capture-one';public.mkdir(parents=True);(public/'capture_manifest.json').write_text('{}')
    bundle=tmp_path/'private/canonical';bundle.mkdir(parents=True)
    camera={'native_name':'camera','T_world_from_camera':np.eye(4).tolist()}
    train=[dict(frame_id='f0',T_world_from_camera=np.eye(4).tolist())]
    monkeypatch.setattr(build,'read_train',lambda p:({},train))
    path=bundle.parent/'capture'/public.name/'capture_plan.json';path.parent.mkdir(parents=True)
    frames=[dict(frame_id='f0',split='train',camera=camera),dict(frame_id='f1',split='test',camera=camera)]
    path.write_text(json.dumps(dict(frames=frames)))
    binding=dict(capture_public=str(public),bundle_dir=str(bundle),capture_manifest_sha256=file_hash(public/'capture_manifest.json'))
    prefix,receipt=checked_train_render_prelude(binding)
    assert prefix==frames[:1] and receipt['evaluation_rgb_accessed_for_prelude'] is False
    path.write_text(json.dumps(dict(frames=list(reversed(frames)))))
    with pytest.raises(ValueError,match='prefix order'):checked_train_render_prelude(binding)
    frames[0]['camera']['T_world_from_camera'][0][3]=1.;path.write_text(json.dumps(dict(frames=frames)))
    with pytest.raises(ValueError,match='calibration'):checked_train_render_prelude(binding)
