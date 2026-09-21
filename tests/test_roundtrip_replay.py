import json
from pathlib import Path

import numpy as np
import pytest

from robo.roundtrip.replay import FixedActionPolicy, compare_relative_trajectories, run_replay_episode, sha256, load_frame_contract


def pose(position=(0, 0, 0), angle=0):
    h=np.radians(angle)/2
    return [*position, np.cos(h), 0, 0, np.sin(h)]


def initial(p=None, time=0):
    return {'time':time,'object_states':{'obj':pose() if p is None else p}}


def ticks(poses, start=0, dt=.05):
    return [{'tick':i,'simulation_time_s':start+(i+1)*dt,'objects':{'obj':p}}
            for i,p in enumerate(poses)]


def compare(ri,rt,ci,ct,n=1,marker=(0,0,0),**kwargs):
    return compare_relative_trajectories(ri,rt,ci,ct,role='obj',marker_world_m=marker,planned_steps=n,**kwargs)


def test_different_native_origins_and_basis_do_not_create_fake_error():
    # Same physical rigid motion, comparison body has local origin [2,0,0]
    # and a constant 45degree basis rotation. Naive body xyz subtraction is wrong.
    r=compare(initial(),ticks([pose((1,0,0),90)]),
              initial(pose((2,0,0),45)),ticks([pose((1,2,0),135)]),marker=(1,0,0))
    assert r['relative_displacement_rmse_cm']==pytest.approx(0,abs=1e-12)
    assert r['relative_rotation_increment_mean_deg']==pytest.approx(0,abs=1e-5)
    assert r['position_rmse_cm'] is None
    assert r['absolute_frame_correspondence']=='NOT_ESTABLISHED'


def test_known_standard_displacement_rmse():
    r=compare(initial(),ticks([pose(),pose()]),initial(),
              ticks([pose((.03,0,0)),pose((0,.04,0))]),n=2)
    assert r['relative_displacement_rmse_cm']==pytest.approx(np.sqrt((3**2+4**2)/2))
    assert r['relative_full_horizon_final_displacement_error_cm']==pytest.approx(4)


def test_world_rotation_increment_known_angle_and_sign_equivalence():
    a=pose(angle=30);b=pose(angle=120)
    r=compare(initial(),ticks([pose(angle=0)]),initial(a),ticks([b]))
    assert r['relative_rotation_increment_mean_deg']==pytest.approx(90)
    b[3:]=[-q for q in b[3:]]
    r2=compare(initial(),ticks([pose(angle=0)]),initial(a),ticks([b]))
    assert r2['relative_rotation_increment_mean_deg']==pytest.approx(90)


def test_initial_offset_is_removed_only_in_explicit_relative_diagnostic():
    r=compare(initial(),ticks([pose((1,0,0))]),initial(pose((7,0,0))),ticks([pose((8,0,0))]))
    assert r['relative_displacement_rmse_cm']==pytest.approx(0)
    assert r['relative_diagnostic_removes_initial_placement_error']
    assert r['position_rmse_cm'] is None and r['final_position_error_cm'] is None


def test_crash_prefix_has_coverage_without_padded_final_error():
    r=compare(initial(),ticks([pose(),pose()]),initial(),ticks([pose((.1,0,0))]),n=2)
    assert r['matched_steps']==1 and r['matched_step_coverage']==.5
    assert r['retained_duration_s']==.05
    assert r['relative_displacement_rmse_cm']==pytest.approx(10)
    assert r['relative_full_horizon_final_displacement_error_cm'] is None
    assert r['relative_prefix_final_displacement_error_cm']==pytest.approx(10)


def test_missing_target_remains_unmeasured():
    r=compare(initial(),ticks([pose()]),{'time':0,'object_states':{}},[])
    assert r['comparison_missing_target']=='missing_initial_target'
    assert r['matched_steps']==0 and r['relative_displacement_rmse_cm'] is None


def test_missing_target_mid_trace_stops_prefix():
    ct=ticks([pose(),pose()]);ct[1]['objects']={}
    r=compare(initial(),ticks([pose(),pose()]),initial(),ct,n=2)
    assert r['matched_steps']==1 and r['comparison_missing_target']=='missing_target_in_trace'


@pytest.mark.parametrize('bad',['timestamps','quaternion','roster','convention','too_long'])
def test_invalid_trace_contract_rejected(bad):
    rt=ticks([pose()]);ct=ticks([pose()]);kw={};n=1
    if bad=='timestamps':ct[0]['simulation_time_s']=.10
    elif bad=='quaternion':ct[0]['objects']['obj'][3]=2
    elif bad=='roster':ct[0]['tick']=1
    elif bad=='convention':kw['quaternion_convention']='xyzw'
    else:ct.append({'tick':1,'simulation_time_s':.10,'objects':{'obj':pose()}})
    with pytest.raises(ValueError):compare(initial(),rt,initial(),ct,n=n,**kw)


def test_absolute_sim_clock_offsets_do_not_change_relative_timegrid():
    r=compare(initial(time=10),ticks([pose()],start=10),initial(time=50),ticks([pose()],start=50))
    assert r['matched_steps']==1


def test_replay_source_is_not_visual_and_returns_exact_independent_action_copies():
    a=np.arange(24,dtype=float).reshape(2,12);p=FixedActionPolicy(a,source_identity={'kind':'fixture'})
    assert p.is_visual_policy is False
    with pytest.raises(RuntimeError):p.infer({})
    p.reset(0);first=p.infer({});first[:]=999
    assert np.array_equal(p.infer({'untrusted_target_pose':'ignored'}),a[1])
    with pytest.raises(IndexError):p.infer({})
    p.reset(42);assert np.array_equal(p.infer({}),a[0])


@pytest.mark.parametrize('actions',[[],[[0]*8],[[float('nan')]*12]])
def test_invalid_action_source_rejected(actions):
    with pytest.raises(ValueError):FixedActionPolicy(actions,source_identity={})


def test_run_delegates_only_to_canonical_harness_and_refuses_early_stop(tmp_path,monkeypatch):
    from robo.eval import harness_runner
    src=tmp_path/'bank.json';src.write_text(json.dumps(np.arange(36).reshape(3,12).tolist()))
    calls=[]
    def fake(adapter,policy,**kw):
        calls.append(kw);policy.reset(kw['reset_seed']);out=kw['out_dir'];out.mkdir()
        actions=[policy.infer({}).tolist() for _ in range(kw['config']['horizon'])]
        (out/'actions.json').write_text(json.dumps(actions));return {'ticks':len(actions),'error':None}
    monkeypatch.setattr(harness_runner,'run_native_episode',fake)
    r=run_replay_episode(object(),actions_path=src,config={'horizon':600},reset_seed=0,
                         out_dir=tmp_path/'episode',treatment_id='B0',expected_actions_sha256=sha256(src))
    assert r['ticks']==3 and calls[0]['execution_kind']=='fixed_action_replay'
    assert calls[0]['config']['horizon']==3
    assert calls[0]['config']['native_policy_horizon']==600
    assert calls[0]['config']['replay_contract']['early_success_termination'] is False
    def early(adapter,policy,**kw):
        out=kw['out_dir'];out.mkdir();(out/'actions.json').write_text('[]');return {'ticks':0,'error':None}
    monkeypatch.setattr(harness_runner,'run_native_episode',early)
    with pytest.raises(ValueError,match='terminated early'):
        run_replay_episode(object(),actions_path=src,config={'horizon':600},reset_seed=0,
                            out_dir=tmp_path/'early',treatment_id='B0',expected_actions_sha256=sha256(src))


def test_reference_action_hash_drift_rejected_before_rollout(tmp_path):
    src=tmp_path/'bank.json';src.write_text(json.dumps([[0]*12]))
    with pytest.raises(ValueError,match='content changed'):
        run_replay_episode(None,actions_path=src,config={'horizon':600},reset_seed=0,
                            out_dir=tmp_path/'no_output',treatment_id='B0',expected_actions_sha256='0'*64)
    assert not (tmp_path/'no_output').exists()


def test_frame_marker_requires_public_source_hash(tmp_path):
    src=tmp_path/'train_points.npy';np.save(src,np.zeros((3,3)))
    c={'schema_version':1,'kind':'relative_public_train_marker','selection_rule':'TRAIN target pointcloud centroid',
       'marker_world_m':[0,0,0],'public_source':{'path':str(src),'sha256':sha256(src)}}
    p=tmp_path/'frame.json';p.write_text(json.dumps(c));assert load_frame_contract(p)==c
    src.write_bytes(b'changed')
    with pytest.raises(ValueError):load_frame_contract(p)


def test_marker_must_equal_public_cloud_centroid_and_refuses_overwrite(tmp_path):
    from robo.roundtrip.replay import write_public_marker_contract
    src=tmp_path/'cloud.npz';np.savez(src,points_world_m=[[0,1,2],[2,3,4]])
    dest=tmp_path/'frame.json';c=write_public_marker_contract(src,dest)
    assert c['marker_world_m']==[1,2,3]
    assert load_frame_contract(dest)==c
    with pytest.raises(FileExistsError):write_public_marker_contract(src,dest)
    c['marker_world_m']=[0,0,0];dest.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='differs from'):load_frame_contract(dest)


class NativeFixture:
    """Only robot actions advance physics; no object setter is exposed."""
    def __init__(self, *, crash_step=None, success_steps=(1,), failed_observation=False):
        from types import SimpleNamespace
        self.t=0;self.crash_step=crash_step;self.success_steps=success_steps
        self.failed_observation=failed_observation;self.actions=[]
        self.native=SimpleNamespace(control_freq=20,sim=SimpleNamespace(data=SimpleNamespace(time=0.,qpos=np.zeros(2),qvel=np.zeros(2))))
    def get_state(self):return initial()
    def get_policy_observation(self):
        if self.failed_observation and self.t==self.crash_step:raise RuntimeError('renderer lost after crash')
        return {'rgb':np.zeros((8,8,3),dtype=np.uint8)}
    def tracked_objects(self):return {'obj':pose((self.t*.001,0,0))}
    def native_stage_state(self):return {'native_task_success':self.native_success()}
    def native_success(self):return self.t in self.success_steps
    def step_native_action(self,action):
        if self.t+1==self.crash_step:
            self.t+=1;raise RuntimeError('environment crashed')
        self.t+=1;self.native.sim.data.time=self.t*.05;self.actions.append(action.copy())
        return {},0,self.native_success(),{}


def fixture_config():
    return {'horizon':4,'instance':{'task_id':'fixture','layout_id':11,'style_id':11},
            'video_image_keys':['rgb'],'replacement_scope':'target_only','oracle_context':True,
            'sensor_regime':'ideal_rgbd_posed'}


def test_canonical_replay_ignores_early_success_and_records_final_failure(tmp_path,monkeypatch):
    from robo.eval.harness_runner import run_native_episode
    import imageio.v2 as imageio
    from types import SimpleNamespace
    monkeypatch.setattr(imageio,'get_writer',lambda *a,**k:SimpleNamespace(append_data=lambda x:None,close=lambda:None))
    a=NativeFixture(success_steps=(1,));actions=np.arange(48,dtype=float).reshape(4,12)
    r=run_native_episode(a,FixedActionPolicy(actions,source_identity={}),config=fixture_config(),
                         reset_seed=0,out_dir=tmp_path/'episode',treatment_id='REPLAY',execution_kind='fixed_action_replay')
    assert r['ticks']==4 and r['horizon']==4 and r['success'] is False
    assert np.array_equal(np.asarray(a.actions),actions)
    from robo.eval.episode_log import read_timeseries
    trace=read_timeseries(tmp_path/'episode/trace.json.gz')
    assert trace[0]['native_predicates']['native_task_success'] is True
    assert trace[-1]['native_predicates']['native_task_success'] is False


def test_canonical_replay_reports_success_at_fixed_horizon(tmp_path,monkeypatch):
    from robo.eval.harness_runner import run_native_episode
    import imageio.v2 as imageio
    from types import SimpleNamespace
    monkeypatch.setattr(imageio,'get_writer',lambda *a,**k:SimpleNamespace(append_data=lambda x:None,close=lambda:None))
    r=run_native_episode(NativeFixture(success_steps=(4,)),FixedActionPolicy(np.zeros((4,12)),source_identity={}),
                         config=fixture_config(),reset_seed=0,out_dir=tmp_path/'episode',
                         treatment_id='REPLAY',execution_kind='fixed_action_replay')
    assert r['ticks']==4 and r['success'] is True
    assert r['outcome']=='success'


def test_canonical_replay_crash_and_video_error_preserve_prefix_ledger(tmp_path,monkeypatch):
    from robo.eval.harness_runner import run_native_episode,LEDGER_NAME
    import imageio.v2 as imageio
    from types import SimpleNamespace
    monkeypatch.setattr(imageio,'get_writer',lambda *a,**k:SimpleNamespace(append_data=lambda x:None,close=lambda:None))
    r=run_native_episode(NativeFixture(crash_step=3,success_steps=(),failed_observation=True),
                         FixedActionPolicy(np.zeros((4,12)),source_identity={}),config=fixture_config(),
                         reset_seed=0,out_dir=tmp_path/'episode',treatment_id='REPLAY',execution_kind='fixed_action_replay')
    assert r['ticks']==2 and r['success'] is None
    assert 'environment crashed' in r['error'] and 'renderer lost' in r['video_error']
    assert len(json.loads((tmp_path/'episode/actions.json').read_text()))==2
    assert (tmp_path/'episode'/LEDGER_NAME).is_file()
