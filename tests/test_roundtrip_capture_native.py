import copy
import json
from pathlib import Path

import numpy as np
import pytest

from robo.manifest.hash import canonical_hash
from robo.roundtrip.capture import CaptureError
from robo.roundtrip import capture_native as c


def config():
    return {'schema_version':1,'execution_ready':True,'benchmark':'robocasa_native',
        'platform':{'robocasa_commit':'a'*40,'robosuite_commit':'b'*40,'mujoco_version':'3.3.1'},
        'instance':{'task_id':'PickPlaceCounterToSink','layout_id':11,'style_id':11,'split':'development'},
        'robot':'PandaOmron','horizon':600,'reset_seeds':[0,1,2,3,4],
        'replacement_scope':'target_only','oracle_context':True,'sensor_regime':'ideal_rgbd_posed'}


def identity(tmp_path,cfg):
    folder=tmp_path/'identity';folder.mkdir()
    report={'passed':True,'initial_objects_equal':True,'initial_observation_diff':{},'u1_initial_observation_diff':{},
        'u0':[{'observation_diff':{},'state_max_abs':0.,'predicate_equal':True} for _ in range(10)],
        'u1':[{'observation_diff':{},'state_max_abs':0.,'predicate_equal':True} for _ in range(10)]}
    (folder/'identity_report.json').write_text(json.dumps(report))
    (folder/'run_manifest.json').write_text(json.dumps({'phase':'identity','config':cfg,'config_sha256':canonical_hash(cfg),'code':{'dirty':False}}))
    return folder/'identity_report.json'


def camera_poses():
    left=np.eye(4);right=np.eye(4);right[0,3]=1.
    return dict(zip(c.NATIVE_CAMERAS,[left,right]))


def test_plan_is_deterministic_split_and_declared_offsets():
    plan=c.smoke_view_plan(camera_poses())
    assert plan==c.smoke_view_plan(camera_poses())
    assert [p['split'] for p in plan]==['train']*6+['test']*2
    assert len({p['frame_id'] for p in plan})==8
    assert [p['camera']['T_world_from_camera'][0][3] for p in plan[:3]]==[-.12,0.,.12]
    assert all(p['camera']['T_world_from_camera'][1][3]==.18 for p in plan[6:])


def test_translation_uses_camera_frame():
    poses=camera_poses();poses[c.NATIVE_CAMERAS[0]][:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    p=c.smoke_view_plan(poses)[0]['camera']['T_world_from_camera']
    assert np.allclose(np.array(p)[:3,3],[0,-.12,0])


def test_coincident_cameras_rejected_without_view_search():
    with pytest.raises(CaptureError,match='duplicate poses'):
        c.smoke_view_plan({name:np.eye(4) for name in c.NATIVE_CAMERAS})


@pytest.mark.parametrize('change',['flag','step','config','truncated'])
def test_identity_controls_fail_closed(tmp_path,change):
    cfg=config();path=identity(tmp_path,cfg);r=json.loads(path.read_text())
    if change=='flag':r['passed']=False
    elif change=='step':r['u1'][3]['state_max_abs']=.01
    elif change=='truncated':r['u0']=[]
    else:cfg['horizon']=100
    path.write_text(json.dumps(r))
    with pytest.raises(CaptureError):c.check_identity_report(path,cfg)


class FakeAdapter:
    def __init__(self,config):self.closed=False;self.reset=None;self.render_calls=0
    def reset_from_spec(self,reset):self.reset=reset
    def get_policy_observation(self):return {'annotation.human.task_description':'Put the cup in the sink'}
    def get_named_body_state(self):return {'private_body':[0.,0.,0.,1.,0.,0.,0.]}
    def get_state(self):return {'qpos':[0.,1.],'native_asset_id':'private42'}
    def render_capture(self,camera,*,width,height):
        self.render_calls+=1
        return {'rgb':np.zeros((height,width,3),np.uint8),'depth_m':np.ones((height,width),np.float32),
            'K':np.array([[3.,0.,width/2],[0.,3.,height/2],[0.,0.,1.]]),
            'T_world_from_camera':np.array(camera['T_world_from_camera']),'timestamp':0.}
    def export_reference_for_evaluator(self,out):
        out.mkdir();(out/'scene.xml').write_text('<mujoco/>');(out/'canonical_state.json').write_text('{}')
    def environment_lock(self):return {'private_native_runtime':'fake'}
    def close(self):self.closed=True


def test_collector_delegates_capture_and_keeps_native_data_private(tmp_path,monkeypatch):
    cfg=config();path=identity(tmp_path,cfg);adapters=[]
    monkeypatch.setattr(c,'git_snapshot',lambda:{'commit':'c'*40,'dirty':False})
    def factory(cfg):a=FakeAdapter(cfg);adapters.append(a);return a
    out=tmp_path/'run';pub=tmp_path/'public';vault=tmp_path/'private'
    r=c.collect_native_smoke(config=cfg,identity_report=path,capture_id='c-1234567890abcdef',out=out,
        public_out=pub,vault=vault,width=4,height=3,adapter_factory=factory,camera_pose_reader=lambda _:camera_poses())
    assert r['status']=='PASS' and r['view_counts']=={'train':6,'dev':0,'test':2}
    assert adapters[0].reset=={'seed':0} and adapters[0].render_calls==8 and adapters[0].closed
    manifest=json.loads((out/'run_manifest.json').read_text())
    assert manifest['trajectory']['native_object_poses_used_for_camera_path'] is False
    assert manifest['trajectory']['native_fixture_bounds_used'] is False
    for p in pub.rglob('*'):
        if p.suffix in {'.json','.jsonl','.txt'}:
            assert 'private' not in p.read_text() and 'robot0_agentview' not in p.read_text()
    assert (vault/'c-1234567890abcdef/reference_hashes.json').exists()


def test_collector_rejects_dirty_source_before_native_loading(tmp_path,monkeypatch):
    cfg=config();path=identity(tmp_path,cfg)
    monkeypatch.setattr(c,'git_snapshot',lambda:{'commit':'c'*40,'dirty':True})
    with pytest.raises(CaptureError,match='clean source'):
        c.collect_native_smoke(config=cfg,identity_report=path,capture_id='c-1234567890abcdef',out=tmp_path/'run',
            public_out=tmp_path/'public',vault=tmp_path/'private')
    assert not (tmp_path/'run').exists()


def test_nonpredeclared_layout_rejected(tmp_path):
    cfg=config();cfg['instance']['layout_id']=12
    with pytest.raises(CaptureError,match='layout11'):
        c.collect_native_smoke(config=cfg,identity_report=tmp_path/'missing',capture_id='c-1234567890abcdef',out=tmp_path/'run',
            public_out=tmp_path/'public',vault=tmp_path/'private')


def canonical(tmp_path,cfg):
    pilot=tmp_path/'pilot';directory=pilot/'canonical_seed0';directory.mkdir(parents=True)
    state={'qpos':[0.,1.],'qvel':[0.,0.],'act':[],'time':0.,
        'native_metadata':{'layout_id':11,'style_id':11,'lang':'Put the cup in the sink'}}
    (directory/'canonical_state.json').write_text(json.dumps(state))
    (directory/'scene.xml').write_text('<mujoco model="private_native42"/>')
    (pilot/'run_manifest.json').write_text(json.dumps({'phase':'pilot','config':cfg,
        'config_sha256':canonical_hash(cfg),'code':{'dirty':False}}))
    return directory,state


def test_canonical_reference_imported_before_camera_planning(tmp_path,monkeypatch):
    cfg=config();path=identity(tmp_path,cfg);directory,state=canonical(tmp_path,cfg)
    monkeypatch.setattr(c,'git_snapshot',lambda:{'commit':'c'*40,'dirty':False})
    class CanonicalAdapter(FakeAdapter):
        def __init__(self,cfg):super().__init__(cfg);self.imported=None
        def import_xml(self,xml,*,canonical_state):
            self.imported=(xml,canonical_state)
        def native_success(self):return False
        def get_state(self):return state
    a=CanonicalAdapter(cfg)
    def poses(adapter):
        assert adapter.imported is not None
        assert 'private_native42' in adapter.imported[0]
        assert adapter.imported[1]==state
        return camera_poses()
    report=c.collect_native_smoke(config=cfg,identity_report=path,capture_id='c-1234567890abcdef',out=tmp_path/'run',
        public_out=tmp_path/'public',vault=tmp_path/'private',width=4,height=3,
        adapter_factory=lambda cfg:a,camera_pose_reader=poses,canonical_reference=directory)
    assert report['reference_policy_initial_state_bound'] is True
    assert report['canonical_reference']['state']['sha256']
    assert report['canonical_reference']['xml']['sha256']
    for p in (tmp_path/'public').rglob('*'):
        if p.suffix in {'.json','.jsonl','.txt'}:
            assert 'private_native42' not in p.read_text()
            assert 'canonical_seed0' not in p.read_text()


@pytest.mark.parametrize('change',['task','layout','missing_state','instruction','config'])
def test_canonical_reference_validation(tmp_path,change):
    cfg=config();directory,state=canonical(tmp_path,cfg)
    if change=='task':state['native_metadata']['task_id']='DifferentTask'
    elif change=='layout':state['native_metadata']['layout_id']=12
    elif change=='missing_state':del state['qvel']
    elif change=='instruction':state['native_metadata']['lang']=''
    else:cfg['horizon']=100
    (directory/'canonical_state.json').write_text(json.dumps(state))
    with pytest.raises(CaptureError):c.load_canonical_reference(directory,cfg)


@pytest.mark.parametrize('change',['terminal','instruction','state'])
def test_canonical_restore_mismatch_rejected(tmp_path,change):
    _,state=canonical(tmp_path,config())
    class Bad:
        def native_success(self):return change=='terminal'
        def get_policy_observation(self):return {'annotation.human.task_description':'wrong' if change=='instruction' else state['native_metadata']['lang']}
        def get_state(self):return {**state,'qpos':[1.,2.]} if change=='state' else state
    with pytest.raises(CaptureError):c._check_canonical_restore(Bad(),state)


@pytest.mark.parametrize('seed',[1,2,3,4])
def test_all_original_pilot_seeds_can_bind_canonical_identity(tmp_path,seed):
    cfg=config();directory,state=canonical(tmp_path,cfg)
    target=directory.with_name(f'canonical_seed{seed}');directory.rename(target)
    result=c.load_canonical_reference(target,cfg,reference_seed=seed)
    assert result['state']==state
    with pytest.raises(CaptureError):c.load_canonical_reference(target,cfg,reference_seed=0)


@pytest.mark.parametrize('seed',[-1,5,True])
def test_capture_rejects_undeclared_pilot_seed(tmp_path,seed):
    with pytest.raises(CaptureError,match='outside the declared'):
        c.collect_native_smoke(config=config(),identity_report='unused',capture_id='c-1234567890abcdef',
            out=tmp_path/'run',public_out=tmp_path/'public',vault=tmp_path/'private',reference_seed=seed)


def test_nonzero_capture_requires_canonical_bundle(tmp_path):
    with pytest.raises(CaptureError,match='sealed canonical'):
        c.collect_native_smoke(config=config(),identity_report='unused',capture_id='c-1234567890abcdef',
            out=tmp_path/'run',public_out=tmp_path/'public',vault=tmp_path/'private',reference_seed=3)
