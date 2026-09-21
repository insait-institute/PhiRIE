import json
from pathlib import Path

import numpy as np
import pytest

from robo.roundtrip import system_verification as sv


@pytest.fixture
def fixture(tmp_path):
    import mujoco
    import trimesh
    asset=tmp_path/'asset';asset.mkdir();(asset/'collision').mkdir()
    mesh=trimesh.creation.box([.02,.02,.02]);mesh.export(asset/'mesh_sim.obj');mesh.export(asset/'collision/part_00.obj')
    T=np.eye(4);T[:3,3]=[0,0,.04]
    (asset/'aligned.json').write_text(json.dumps({'T':T.tolist(),'scale':1.0}))
    (asset/'physics.json').write_text(json.dumps({'mass_kg':.1,'friction':.5}))
    xml='<mujoco><option timestep=".002"/><worldbody><geom name="floor" type="plane" size="1 1 .1"/><body name="obj_main" pos="0 0 .04"><freejoint name="obj_joint"/><geom type="box" size=".01 .01 .01"/></body></worldbody></mujoco>'
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);state=np.empty(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION)
    xp=tmp_path/'scene.xml';xp.write_text(xml);sp=tmp_path/'state.json';sp.write_text(json.dumps({'integration_state':state.tolist()}))
    capture=tmp_path/'public';capture.mkdir();(capture/'capture_manifest.json').write_text('{}')
    return asset,xp,sp,capture


def support(*args):
    return {'available':True,'correction_z_m':-.029,'source':'public_TRAIN_depth_horizontal_support_proxy'}


def test_actual_collider_drift_and_reset_restoration(fixture):
    asset,xp,sp,_=fixture
    report,_,_=sv.load_and_measure(xp.read_text(),json.loads(sp.read_text()),asset,'obj_main',sv.DEFAULT_CONFIG)
    assert report['settle_drift_m']>.02 and not report['passed']
    assert report['probe_state_restored_byte_exact'] and report['settle_contacts']
    assert report['actual_collision_parts'] and report['native_task_success'] is None


def test_real_pose_change_dependencies_and_v1_no_repair(fixture,monkeypatch,tmp_path):
    asset,xp,sp,capture=fixture
    monkeypatch.setattr(sv,'observe_support',support)
    for method in ('V1','B4','BM','B3'):
        out=tmp_path/method
        result=sv.run(xml_path=xp,state_path=sp,asset=asset,capture=capture,config=sv.DEFAULT_CONFIG,out=out,method=method)
        assert result['actual_calls'] == (1 if method in ('B4','BM') else 0)
        assert result['accepted'] == (method!='V1')
        if result['actual_calls']:
            change=json.loads((out/'repair_ledger.jsonl').read_text())
            assert change['before_T']!=change['after_T']
            assert 'gs_transform' in change['invalidated_dependencies']
            assert json.loads((out/'verification_after.json').read_text())['settle_drift_m'] < .003
    assert json.loads((asset/'aligned.json').read_text())['T'][2][3]==.04


def test_irreparable_preserved_and_bound_refused(fixture,monkeypatch,tmp_path):
    asset,xp,sp,capture=fixture
    monkeypatch.setattr(sv,'observe_support',lambda *a:{'available':False})
    result=sv.run(xml_path=xp,state_path=sp,asset=asset,capture=capture,config=sv.DEFAULT_CONFIG,out=tmp_path/'bad')
    assert result['accepted'] is False and result['actual_calls']==0
    with pytest.raises(ValueError):sv.validate_config(dict(sv.DEFAULT_CONFIG,max_calls=3))
    with pytest.raises(ValueError):sv.validate_config(dict(sv.DEFAULT_CONFIG,max_support_correction_m=.5))


def test_public_depth_support_without_native_shape(fixture,monkeypatch):
    asset,_,_,capture=fixture
    depth=np.ones((40,40),dtype=np.float32);np.save(capture/'depth.npy',depth)
    T=np.eye(4);T[:3,:3]=np.diag([1,-1,-1]);T[2,3]=1
    row={'depth_m':'depth.npy','K':[[200,0,20],[0,200,20],[0,0,1]],'T_world_from_camera':T.tolist()}
    monkeypatch.setattr(sv,'read_train',lambda p:({'capture_id':'public_fixture'},[row]))
    config=dict(sv.DEFAULT_CONFIG,minimum_support_points=3)
    result=sv.observe_support(capture,asset,config)
    assert result['available'] and abs(result['support_z_world_m'])<1e-10
    assert result['correction_z_m']==pytest.approx(-.029)


def test_bm_fixed_order_two_calls_and_b4_stops_when_valid(fixture,monkeypatch,tmp_path):
    import shutil
    asset,xp,sp,capture=fixture
    alt=tmp_path/'alternative';shutil.copytree(asset,alt)
    data=json.loads((alt/'aligned.json').read_text());data['T'][2][3]=.05
    (alt/'aligned.json').write_text(json.dumps(data))
    monkeypatch.setattr(sv,'observe_support',support)
    b4=sv.run(xml_path=xp,state_path=sp,asset=asset,capture=capture,config=sv.DEFAULT_CONFIG,out=tmp_path/'adaptive',method='B4',alternatives=[alt])
    bm=sv.run(xml_path=xp,state_path=sp,asset=asset,capture=capture,config=sv.DEFAULT_CONFIG,out=tmp_path/'matched',method='BM',alternatives=[alt])
    assert b4['actual_calls']==1 and bm['actual_calls']==2
    actions=[json.loads(x)['action'] for x in (tmp_path/'matched/repair_ledger.jsonl').read_text().splitlines()]
    assert actions==sv.DEFAULT_CONFIG['bm_order']


def test_bounded_pose_api_and_noop_are_not_repairs(fixture,tmp_path):
    asset,*_=fixture
    for dz in (0,.5,float('nan')):
        with pytest.raises(ValueError):sv.corrected_asset(asset,tmp_path/'bad',{'available':True,'correction_z_m':dz})
    assert not (tmp_path/'bad').exists()


def test_failure_specific_scheduler_and_frozen_tie_order():
    probe=lambda reasons: {'passed':False,'reason_codes':reasons}
    assert sv.next_context_action('B4',probe(['initial_penetration']),set())==('support_correction','initial_penetration')
    assert sv.next_context_action('B4',probe(['public_observation_residual']),set())==('reselect_candidate','public_observation_residual')
    assert sv.next_context_action('B4',probe(['initial_penetration','public_observation_residual']),set())==('reselect_candidate','public_observation_residual')
    assert sv.next_context_action('B4',probe(['initial_penetration','public_observation_residual']),{'reselect_candidate'})==('support_correction','initial_penetration')
    assert sv.next_context_action('B4',probe(['native_destination_access']),{'reselect_candidate'})==(None,None)
    assert sv.next_context_action('B4',{'passed':True,'reason_codes':[]},set())==(None,None)
    assert sv.next_context_action('BM',{'passed':True,'reason_codes':[]},set())==('reselect_candidate','fixed_bank_order')


def test_other_generator_bank_follows_retry_parent_and_rejects_ambiguity():
    pool={'initial_candidates':[{'proposal_id':'T','tool':'trellis'},{'proposal_id':'R','tool':'reconviagen'}],
          'retry_candidate':{'proposal_id':'retry','parent_proposal_ids':['T']},
          'outcomes':[{'native_method':'B0_FIXED_NATIVE','selected_proposal_id':'T','object_dir':'/output/T'},
                      {'native_method':'B1_FIXED_PRIORITY','selected_proposal_id':'R','object_dir':'/output/R'},
                      {'native_method':'B3_AGENT_NATIVE','selected_proposal_id':'retry','object_dir':'/output/retry'}]}
    assert sv.lineage_alternative(pool)['tool']=='reconviagen'
    pool['outcomes'][-1].update(selected_proposal_id='R',object_dir='/output/R')
    assert sv.lineage_alternative(pool)['tool']=='trellis'
    pool['outcomes'][-1]['selected_proposal_id']='unknown'
    with pytest.raises(ValueError,match='lineage'):sv.lineage_alternative(pool)
    pool['outcomes'][-1]['selected_proposal_id']='retry'
    pool['retry_candidate']['parent_proposal_ids']=['T','R']
    with pytest.raises(ValueError,match='lineage'):sv.lineage_alternative(pool)
