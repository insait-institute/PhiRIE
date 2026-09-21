"""Actual MuJoCo geometry for the existing two-family diagnostic producers."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from robo.eval import e4_candidate_screen as screen
from robo.eval import e4_camera_scorer_gate as camera
from robo.tasks.pi05_tasks import TaskScorer


TASK = {'task_id':'synthetic__obj_00_into_obj_01','task_family':'object_to_receptacle',
        'target':'obj_00','target_label':'mouse','any_instance':False,
        'receptacle':'obj_01','receptacle_dims':[.22,.22,.10],
        'instructions':{'default':'put the mouse in the tray'}}
SUITE = {'robot':{'base_pos':[0,0,0],'base_yaw':0.0}}


def geometry(solid=False):
    import mujoco
    container = '<geom type="box" size=".11 .11 .05"/>' if solid else '''
       <geom type="box" pos="0 0 -.035" size=".11 .11 .01"/>
       <geom type="box" pos=".10 0 0" size=".01 .11 .035"/>
       <geom type="box" pos="-.10 0 0" size=".01 .11 .035"/>
       <geom type="box" pos="0 .10 0" size=".09 .01 .035"/>
       <geom type="box" pos="0 -.10 0" size=".09 .01 .035"/>'''
    model=mujoco.MjModel.from_xml_string(f'''<mujoco><option timestep="0.0016666666666666666"/><worldbody>
      <geom type="plane" pos="0 0 .755" size="2 2 .1"/>
      <body name="obj_00" pos=".45 -.20 .80"><freejoint/><geom size=".01"/></body>
      <body name="obj_01" pos=".50 .20 .80"><freejoint/>{container}</body>
    </worldbody></mujoco>''')
    class Env:
        free_bodies=['obj_00','obj_01']
        def __init__(self):
            self.model=model;self.data=mujoco.MjData(model);self.info={'ext_cam':'ext'}
            mujoco.mj_forward(model,self.data)
            self.start_pose={name:self.body_pose(name) for name in self.free_bodies}
        def body_pose(self,name):
            bid=self.model.body(name).id
            return self.data.xpos[bid].copy(),self.data.xquat[bid].copy()
        def grasped(self,name):return False
        def at_rest(self,name):return True
        def lifted(self,name,min_dz=.05):
            return self.body_pose(name)[0][2]>self.start_pose[name][0][2]+min_dz
    return Env()


def footprints(env):
    return {name:screen._world_xy_rectangle(*env.body_pose(name),np.array(dims))
            for name,dims in [('obj_00',[.02]*3),('obj_01',TASK['receptacle_dims'])]}


def test_receptacle_resolver_uses_actual_body_pose_and_exact_existing_scorer():
    env=geometry();task=copy.deepcopy(TASK)
    result=screen.task_destination(task,env)
    assert result['destination_region']==TaskScorer(env,task).region
    assert result['receptacle_position_world_m']==pytest.approx([.5,.2,.8])
    assert result['goal_collision_status']=='NOT_RUN' and not result['cavity_verified']
    assert task==TASK
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('mutation',['missing','same','both','negative_dims','nan_dims','wrong_family'])
def test_bad_destination_is_rejected(mutation):
    task=copy.deepcopy(TASK)
    if mutation=='missing':task['receptacle']='obj_99'
    if mutation=='same':task['receptacle']='obj_00'
    if mutation=='both':task['region']={}
    if mutation=='negative_dims':task['receptacle_dims'][0]=-1
    if mutation=='nan_dims':task['receptacle_dims'][0]=float('nan')
    if mutation=='wrong_family':task['task_family']='object_to_region'
    with pytest.raises(screen.CandidateScreenError):screen.task_destination(task,geometry())


@pytest.mark.parametrize('solid',[False,True])
def test_actual_goal_geometry_never_gets_cavity_credit_from_scorer_box(solid):
    import mujoco
    env=geometry(solid);current=footprints(env)
    result=screen._workspace_from_footprints(current=current,task=TASK,suite=SUITE)
    assert result['checks']['goal_clear_of_all_non_target_bodies']
    assert all(v for k,v in result['checks'].items() if k!='receptacle_goal_collision_verified')
    assert result['passed'] is False
    assert result['task_destination']['goal_collision_status']=='NOT_RUN'
    # Demonstrate why the score box cannot certify physics: identical task/
    # dimensions, but the actual solid model overlaps a target at its center.
    env.data.qpos[:3]=[.5,.2,.8];mujoco.mj_forward(env.model,env.data)
    penetrates=any(c.dist<0 for c in env.data.contact[:env.data.ncon])
    assert penetrates is solid
    json.dumps(result,allow_nan=False)


def test_goal_exempts_only_named_receptacle_not_other_obstacles():
    current=footprints(geometry())
    current['obj_02']=screen._world_xy_rectangle(np.array([.5,.2,.8]),np.array([1.,0,0,0]),np.array([.03]*3))
    result=screen._workspace_from_footprints(current=current,task=TASK,suite=SUITE)
    assert not result['checks']['goal_clear_of_all_non_target_bodies']
    assert [x['slot'] for x in result['obstacle_clearance']['blocking_footprints']]==['obj_02']


def test_camera_workspace_reports_family_and_keeps_initial_success_negative(monkeypatch):
    env=geometry()
    monkeypatch.setattr(camera,'_camera_pose',lambda *_:{})
    monkeypatch.setattr(camera,'_project_with_depth',lambda *_:{'visible':True})
    result=camera.workspace_metric(env,TASK,SUITE,{'dims':[.02]*3},np.ones((360,640)),{})
    assert result['task_destination']['task_family']=='object_to_receptacle'
    assert all(v for k,v in result['checks'].items() if k!='receptacle_goal_collision_verified')
    assert result['passed'] is True
    assert result['task_destination']['goal_collision_status']=='PASS'
    import mujoco
    env.data.qpos[:3]=[.5,.2,.8];mujoco.mj_forward(env.model,env.data)
    result=camera.workspace_metric(env,TASK,SUITE,{'dims':[.02]*3},np.ones((360,640)),{})
    assert not result['checks']['target_initially_outside_goal_region']


def test_canonical_receptacle_scorer_replay_keeps_all_four_negative_paths_and_state():
    env=geometry()
    before=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.time)
    result=camera.scorer_replay(env,TASK,{})
    assert result['passed'] is True and len(result['negative_replays'])==4
    assert result['task_destination']['cavity_verified'] is False
    assert result['oracle_validated'] is False
    assert np.array_equal(env.data.qpos,before[0]) and np.array_equal(env.data.qvel,before[1])
    assert env.data.time==before[2] and not env.grasped('obj_00')


def test_paired_roles_preserve_missing_destination_denominator():
    target={'name':'obj_00','label':'mouse','tier':'A','dims':np.array([.02]*3),'mass':.02,'drift':0.}
    dest={'name':'obj_01','label':'tray','tier':'A','dims':np.array([.22,.22,.10]),'mass':.2,'drift':0.}
    rows={arm:{'obj_00':target,'obj_01':dest} for arm in screen.POLICIES}
    evidence=screen.paired_receptacle_evidence(TASK,rows)
    assert evidence['role_constraints_passed'] and not evidence['qualification_passed']
    del rows['A0']['obj_01']
    evidence=screen.paired_receptacle_evidence(TASK,rows)
    assert set(evidence['by_policy'])=={'A0','A4'}
    assert evidence['by_policy']['A0']['reasons']==['receptacle_not_accepted_in_construction_arm']
    assert evidence['by_policy']['A4']['role_constraints_passed']


def test_task_family_matches_canonical_ledger_and_ambiguous_declaration_fails():
    from robo.eval.harness_runner import _task_family
    assert _task_family(TASK)==screen.task_destination(TASK,geometry())["task_family"]
    bad=copy.deepcopy(TASK);bad["task_family"]="object_to_region"
    with pytest.raises(screen.CandidateScreenError):
        screen.paired_receptacle_evidence(bad,{"A0":{},"A4":{}})


@pytest.mark.parametrize("solid",[False,True])
def test_fixed_interior_probe_uses_actual_collision_settle_and_restores_all_state(solid):
    env=geometry(solid);before=env.data.qpos.copy()
    current=footprints(env)
    probe=screen.receptacle_goal_probe(env,TASK)
    assert probe["passed"] is (not solid)
    assert probe["checks"]["initial_witness_has_no_penetration"] is (not solid)
    assert probe["step_count"]==900
    assert probe["state_before_sha256"]==probe["state_after_sha256"]
    assert np.array_equal(env.data.qpos,before)
    result=screen._workspace_from_footprints(current=current,task=TASK,suite=SUITE,goal_probe=probe)
    assert result["passed"] is (not solid)
    assert result["task_destination"]["goal_collision_status"]==("FAIL" if solid else "PASS")
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("mutation",["original_pose","placement","count","speed","contact","state_hash","task","quaternion","extra_field"])
def test_probe_replay_rejects_changed_inputs_or_derived_evidence(mutation):
    env=geometry();current=footprints(env);probe=screen.receptacle_goal_probe(env,TASK)
    if mutation=="original_pose":probe["original_positions_m"]["obj_01"][0]+=.1
    if mutation=="placement":probe["witness_position_m"][0]+=.1
    if mutation=="count":probe["step_count"]=899
    if mutation=="speed":probe["settled_linear_velocity_m_s"]["obj_00"][0]=.06
    if mutation=="contact":probe["initial_target_contacts"]=[{"bodies":["obj_00","obj_01"],"geom_ids":[0,1],"distance_m":-.1}]
    if mutation=="state_hash":probe["state_after_sha256"]="0"*64
    if mutation=="task":probe["receptacle"]="obj_99"
    if mutation=="quaternion":probe["probe_start_quaternions_wxyz"]["obj_00"]=[0,1,0,0]
    if mutation=="extra_field":probe["unfrozen_threshold"]=100
    with pytest.raises(screen.CandidateScreenError):
        screen._workspace_from_footprints(current=current,task=TASK,suite=SUITE,goal_probe=probe)


def test_probe_restores_complete_data_even_if_step_raises(monkeypatch):
    import mujoco
    env=geometry();before=env.data.qpos.copy();time=env.data.time
    def fail(*_):raise RuntimeError("injected step failure")
    monkeypatch.setattr(mujoco,"mj_step",fail)
    with pytest.raises(RuntimeError,match="injected"):
        screen.receptacle_goal_probe(env,TASK)
    assert np.array_equal(env.data.qpos,before) and env.data.time==time


@pytest.mark.parametrize('solid',[False,True])
def test_canonical_runtime_workspace_consumes_probe_without_altering_reset(solid):
    env=geometry(solid)
    rows=[{'name':'obj_00','dims':np.array([.02]*3)},
          {'name':'obj_01','dims':np.array(TASK['receptacle_dims'])}]
    before=env.data.qpos.copy()
    result=screen._runtime_workspace(env=env,task=TASK,suite=SUITE,task_rows=rows)
    assert result['passed'] is (not solid)
    assert np.array_equal(env.data.qpos,before)


@pytest.mark.parametrize('invalid',['disabled_collision','wrong_timestep','nonfinite_state'])
def test_probe_rejects_invalid_physical_query(invalid):
    env=geometry()
    if invalid=='disabled_collision':
        body=env.model.body('obj_01')
        for index in range(int(body.geomadr[0]),int(body.geomadr[0])+int(body.geomnum[0])):
            env.model.geom_contype[index]=env.model.geom_conaffinity[index]=0
    if invalid=='wrong_timestep':env.model.opt.timestep=.002
    if invalid=='nonfinite_state':env.data.qvel[0]=float('nan')
    with pytest.raises(screen.CandidateScreenError):screen.receptacle_goal_probe(env,TASK)
