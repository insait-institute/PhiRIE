import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robo.eval import e4_camera_scorer_gate as gate


def _config():
    return json.loads((Path(gate.CODE_ROOT) / 'configs/experiments/icra2027/e4_winner_camera.json').read_text())


@pytest.mark.parametrize('mutation', ['producer', 'cpu_gate', 'task_manifest', 'factory'])
def test_winner_input_drift_fails_before_render(mutation):
    config = _config()
    if mutation == 'producer':
        config['producer_commit'] = '0' * 40
    elif mutation == 'factory':
        config['factory_closure_sha256']['A4'] = '0' * 64
    else:
        config['inputs'][mutation]['sha256'] = '0' * 64
    with pytest.raises((gate.CameraScorerGateError, ValueError)):
        gate.validate_winner_chain(config, root=gate.EXPECTED_EVIDENCE_ROOT)


def test_winner_changed_cpu_cell_rejected(monkeypatch):
    if not gate.EXPECTED_EVIDENCE_ROOT.is_dir():
        pytest.skip('sealed evidence root is unavailable')
    from robo.eval import harness_validation
    original = harness_validation.read_jsonl
    def changed(path):
        rows = original(path)
        if Path(path).name == 'metrics.jsonl':
            rows[5]['reset_seed'] += 1
        return rows
    monkeypatch.setattr(harness_validation, 'read_jsonl', changed)
    producer_root = json.loads((gate.EXPECTED_EVIDENCE_ROOT / _config()['inputs']['cpu_gate']['path']).read_text())['code']['code_root']
    if not Path(producer_root).is_dir():
        pytest.skip('historical winner-chain integration requires its original producer checkout')
    monkeypatch.setattr(gate, 'CODE_ROOT', Path(producer_root))
    with pytest.raises(RuntimeError, match="binding differs"):
        gate.validate_winner_chain(_config(), root=gate.EXPECTED_EVIDENCE_ROOT)


def test_runtime_reset_drift_fails_without_scorer_or_render(tmp_path):
    env = SimpleNamespace(reset=lambda **_: {}, last_reset_provenance={'jitter': 'changed'})
    row = {'passed': True, 'reset_seed': 1, 'reset_jitter': {'uniform_draw_0_1': [.5, .5]}}
    with pytest.raises(gate.CameraScorerGateError, match='reset differs'):
        gate._winner_render_cell(env, {'target':'obj_05'}, {}, [], row,
                                 {'reset_provenance':{}}, tmp_path)


def test_winner_invalid_arms_never_instantiate_env_and_keep_denominator(monkeypatch, tmp_path):
    from robo.envs import pi05_env
    from robo.tasks import pi05_tasks
    config = _config(); config['freeze_id'] = 'winner-test'
    config_path = tmp_path/'config.json'; config_path.write_text(json.dumps(config))
    manifest = tmp_path/'oldmanifest.json'; manifest.write_text('{}')
    calls=[]
    class Env:
        def __init__(self, scene, *_args, **_kwargs):
            calls.append(scene)
            self.renderer=SimpleNamespace(close=lambda:None)
    monkeypatch.setattr(pi05_env, 'DroidSimEnv', Env)
    monkeypatch.setattr(pi05_tasks, '_load_objects', lambda _:[])
    rows=[];records=[]
    for policy in ('A0','A4'):
        for ep in range(5):
            rows.append({'policy_id':policy,'cell_id':f'{policy}-{ep}', 'task_id':'task',
                         'episode':ep,'passed':policy=='A4','checks':{'stable':policy=='A4'}})
            records.append({'treatment_id':f'{policy.lower()}_raster','reset_state_id':f'task__seed0__ep{ep}',
                            'manifest_path':str(manifest)})
    suite={'scene_xml':'A4','robot':{'base_pos':[0,0,0],'base_yaw':0},'table':{},'ext_cam':{},
           'exclude_objects':[],'tasks':[{'task_id':'task'}]}
    upstream={'rows':rows,'records':records,'suites':{'A4':suite},'factories':{'A4':tmp_path},'summary':{}}
    monkeypatch.setattr(gate,'validate_winner_chain',lambda *_args,**_kwargs:copy.deepcopy(upstream))
    monkeypatch.setattr(gate,'_git_snapshot',lambda _: {'commit':'a'*40,'dirty':False})
    monkeypatch.setattr(gate,'_evidence_root',lambda:tmp_path)
    monkeypatch.setattr(gate,'_gpu_runtime_attestation',lambda _: {"qualifier_preflight_gate_sha256":config["inputs"]["cpu_gate"]["sha256"]})
    def diagnostic(_env,_task,_suite,_task_rows,row,_original,_staging):
        return {'cell_id':row['cell_id'],'executed':True,'passed':False,'camera_metrics':[],
                'workspace_metrics':{},'disambiguation_metrics':{},'scorer_metrics':{}}
    monkeypatch.setattr(gate,'_winner_render_cell',diagnostic)
    result=gate.run_winner_gate(config_path=config_path,expected_code_commit='a'*40)
    assert calls==['A4']*5
    assert result['planned_cells']==10 and result['executed_cells']==5
    assert result['build_failures']==5 and not result['real_policy_infra_smoke_allowed']
    cells=[json.loads(x) for x in (tmp_path/'outputs/icra2027/winner-test/harness/camera_scorer/cells.jsonl').read_text().splitlines()]
    assert len(cells)==10
    assert len(list((tmp_path/'outputs/icra2027/winner-test/harness/camera_scorer/cell_records').glob('*.json')))==10
    assert all(r['camera_metrics'] is None and r['reset_provenance'] is None for r in cells[:5])


def test_singleton_exemption_cannot_hide_distractor_or_missing_target(monkeypatch,tmp_path):
    env=SimpleNamespace(_task_rows=[])
    monkeypatch.setattr(gate,'_eligible_same_label_objects',lambda *_args:['obj_05','obj_06'])
    with pytest.raises(gate.CameraScorerGateError,match='singleton'):
        gate.camera_metric(env=env,raw=np.zeros((360,640,3),dtype=np.uint8),
            segmentation=None,camera_role='exterior',camera_name='ext',target='obj_05',distractors=[],
            expected_ext_cam={},output_dir=tmp_path,artifact_root=tmp_path,row_identity={},
            singleton_task={'target':'obj_05','any_instance':False,'instructions':{'default':'move the mouse'}})


def test_authenticated_singleton_still_requires_target_pixels(monkeypatch,tmp_path):
    env=SimpleNamespace(_task_rows=[])
    task={'target':'obj_05','any_instance':False,'instructions':{'default':'move the mouse'}}
    monkeypatch.setattr(gate,'_eligible_same_label_objects',lambda *_args:['obj_05'])
    monkeypatch.setattr(gate,'_camera_pose',lambda *_args:{})
    monkeypatch.setattr(gate,'exterior_camera_replay',lambda *_args:{'passed':True})
    monkeypatch.setattr(gate,'_policy_resize_rgb',lambda _raw:np.zeros((224,224,3),dtype=np.uint8))
    mask=np.zeros((360,640),dtype=bool)
    monkeypatch.setattr(gate,'semantic_segmentation',lambda *_args,**_kwargs:(mask.astype(np.uint8),{'target':mask}))
    raw=np.random.RandomState(0).randint(0,256,size=(360,640,3),dtype=np.uint8)
    kwargs=dict(env=env,raw=raw,segmentation=None,camera_role='exterior',camera_name='ext',target='obj_05',
                distractors=[],expected_ext_cam={},output_dir=tmp_path,artifact_root=tmp_path,row_identity={},singleton_task=task)
    result=gate.camera_metric(**kwargs)
    assert result['checks']['every_same_label_distractor_visible_at_policy_input']
    assert not result['checks']['target_visible_at_policy_input'] and not result['passed']
    mask[100:120,200:240]=True
    result=gate.camera_metric(**kwargs)
    assert result['passed'] and result['distractor_visibility_applicable'] is False
    kwargs['singleton_task']=None
    result=gate.camera_metric(**kwargs)
    assert not result['passed']  # legacy default still requires a distractor


def test_base_offset_world_conversion_matches_rig_and_rejects_pose_drift(monkeypatch):
    from robo.rigs import pi05_rig as rig
    suite={'robot':{'base_pos':[1.,2.,3.],'base_yaw':np.pi/2},
           'ext_cam':{'pos':[1.,0.,.5],'target':[0.,0.,0.],'fovy':68.}}
    original=copy.deepcopy(suite)
    expected=gate._winner_expected_world_camera(suite)
    assert expected['pos']==pytest.approx([1.,3.,3.5])
    assert expected['quat_wxyz']==pytest.approx(rig.lookat_quat([1.,3.,3.5],[0.,0.,0.]))
    assert suite==original
    import mujoco
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,np.asarray(expected['quat_wxyz']))
    observed={'position_world_m':expected['pos'],'rotation_camera_to_world':rotation.reshape(3,3).tolist(),'fovy_deg':68.}
    monkeypatch.setattr(gate,'_camera_pose',lambda *_args:copy.deepcopy(observed))
    env=SimpleNamespace(info={'ext_cam':'ext'})
    assert gate.exterior_camera_replay(env,expected)['passed']
    observed['position_world_m']=[2.,3.,3.5]
    assert not gate.exterior_camera_replay(env,expected)['passed']


@pytest.mark.parametrize('inside', [True, False])
def test_workspace_real_numpy_geometry_serializes_without_changing_predicate(monkeypatch, inside):
    env=SimpleNamespace(info={'ext_cam':'ext'},
        body_pose=lambda _name:(np.array([.5 if inside else .1,0.,.8]),np.array([1.,0.,0.,0.])))
    monkeypatch.setattr(gate,'_camera_pose',lambda *_args:{'position_world_m':[0.,0.,2.],
        'rotation_camera_to_world':np.eye(3).tolist(),'fovy_deg':68.})
    task={'target':'obj_05','receptacle':None,'region':{'cx':.5,'cy':0.,'hx':.2,'hy':.2,'zlo':.7,'zhi':1.}}
    row=gate.workspace_metric(env,task,{'robot':{'base_pos':[0.,0.,0.],'base_yaw':0.}},
        {'dims':[.05,.05,.05]},np.full((360,640),10.,dtype=float),{})
    encoded=json.dumps(row,allow_nan=False)
    assert json.loads(encoded)['initially_in_goal_region'] is inside
    assert row['checks']['target_initially_outside_goal_region'] is (not inside)


def test_winner_failed_staging_preserves_completed_cells_without_publication(tmp_path):
    output=tmp_path/'camera_scorer'
    with pytest.raises(RuntimeError):
        with gate._atomic_directory(output,preserve_failed=True) as staging:
            gate._write_json(staging/'complete_cell.json',{'passed':False})
            raise RuntimeError('later serializer failure')
    assert not output.exists()
    failed=list(tmp_path.glob('.camera_scorer.failed.*'))
    assert len(failed)==1
    assert json.loads((failed[0]/'complete_cell.json').read_text())=={'passed':False}


def test_table_camera_producer_is_world_declared_and_translation_equivariant():
    from robo.eval.e4_candidate_screen import _suite_for_plan
    plan={'selected':{'table':{'cx':3.,'cy':2.,'hx':1.,'hy':1.,'top_z':.7,'thickness_m':.02},
                      'base_pos':[3.1,1.2,.7],'base_yaw':1.2,'tasks':[]}}
    a=_suite_for_plan(scene_id='scene',scene_xml=Path('/scene.xml'),plan=plan,rows_by_policy={'A0':{},'A4':{}})
    assert a['ext_cam']['mode']=='world'
    assert a['ext_cam']['pos']==[3.,3.,1.45]
    assert gate._winner_expected_world_camera(a)['pos']==a['ext_cam']['pos']
    moved=copy.deepcopy(plan)
    moved['selected']['table']['cx']+=10.
    moved['selected']['base_pos'][0]+=10.
    b=_suite_for_plan(scene_id='scene',scene_xml=Path('/scene.xml'),plan=moved,rows_by_policy={'A0':{},'A4':{}})
    assert np.asarray(b['ext_cam']['pos'])-np.asarray(a['ext_cam']['pos'])==pytest.approx([10.,0.,0.])
    assert b['ext_cam']['quat_wxyz']==pytest.approx(a['ext_cam']['quat_wxyz'])
    reference=copy.deepcopy(a)
    for key in ('mode','frame','quat_wxyz'):reference['ext_cam'].pop(key)
    gate._assert_common_camera_correction(a,reference)
    changed=copy.deepcopy(a);changed['ext_cam']['fovy']=70.
    with pytest.raises(gate.CameraScorerGateError,match='position/target/FOV'):
        gate._assert_common_camera_correction(changed,reference)
    changed=copy.deepcopy(a);changed['robot']['base_yaw']+=.1
    with pytest.raises(gate.CameraScorerGateError,match='another task/robot/table'):
        gate._assert_common_camera_correction(changed,reference)
