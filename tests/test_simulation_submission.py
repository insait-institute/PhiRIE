"""CPU checks for the no-physical-trial submission follow-up.

Native asset/policy execution still requires the pinned cluster environment.
"""
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robo.roundtrip import mechanism_followup as follow
from robo.roundtrip import object_fidelity as appearance
from robo.roundtrip import scorer_sensitivity as sensitivity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def fake_core(monkeypatch):
    core = SimpleNamespace(sha=digest, canonical_hash=canonical,
        IDENTITY_FIELDS=('cohort_id','canonical_instance_id','reset_id','policy_id',
                         'controller_method','scope','sensor_regime','renderer','execution_protocol'))
    monkeypatch.setattr(follow, '_core', lambda: core)
    return core


def row(method='REF_NATIVE', iid='scene-a', reset='r0'):
    return dict(cohort_id='old',canonical_instance_id=iid,reset_id=reset,policy_id='frozen',
                controller_method=method,scope='L0_target_only',sensor_regime='ideal_rgbd_posed',
                renderer='native',execution_protocol='primary_native', policy_rng_seed=42,
                layout_id=1, style_id=1, task_id='PickPlaceCounterToSink', split='test',
                native_horizon=600, result_path='old/result.json', result={'success':True},
                executed=True,success=True,config_path='old.json',config_sha256='old')


def test_followup_preserves_all_resets_not_only_successes(fake_core):
    refs=[row(reset='r0'),dict(row(reset='r1'),success=False)]
    units=follow.followup_rows(refs,'new')
    assert len(units)==10
    assert len({u['unit_id'] for u in units})==10
    assert {u['reset_id'] for u in units}=={'r0','r1'}
    assert all(u['success'] is None and u['executed'] is None and 'result_path' not in u for u in units)
    assert [u['controller_method'] for u in units[:5]]==list(follow.METHODS)


@pytest.mark.parametrize('source,new_id',[([], 'new'),([row()], 'old'),([row(),row()], 'new'),
    ([dict(row(),scope='L1_target_destination')],'new')])
def test_followup_rejects_invalid_roster(fake_core,source,new_id):
    with pytest.raises(ValueError):follow.followup_rows(source,new_id)


def make_pool(tmp_path, selected='initial'):
    obj=tmp_path/'selected_00';(obj/'collision').mkdir(parents=True)
    for name in ('aligned.json','physics.json','mesh_sim.obj','collision/part_00.obj'):
        (obj/name).write_text('{}' if name.endswith('.json') else 'mesh bytes')
    pool=dict(schema_version=2,canonical_instance_id='scene-a',capture_manifest_sha256='capture',
              initial_candidates=[{'proposal_id':'initial'}],
              outcomes=[dict(native_method='B1_FIXED_PRIORITY',selected_proposal_id=selected,
                             object_dir='/former/root/selected_00',
                             artifact_hashes={str(p.relative_to(obj)):digest(p) for p in obj.rglob('*') if p.is_file()})])
    path=tmp_path/'candidate_pool.json';path.write_text(json.dumps(pool));return path,obj


def test_pool_binding_reuses_exact_frozen_initial(fake_core,tmp_path):
    path,obj=make_pool(tmp_path)
    b=follow.pool_binding(path,'B1_FIXED_PRIORITY','capture')
    assert b['accepted'] is True and Path(b['object_dir'])==obj
    assert b['selected_proposal_id']=='initial'


def test_pool_cannot_use_retry_for_B1(fake_core,tmp_path):
    path,_=make_pool(tmp_path,'retry')
    with pytest.raises(ValueError,match='initial proposals'):follow.pool_binding(path,'B1_FIXED_PRIORITY','capture')


def test_pool_rejects_changed_mesh(fake_core,tmp_path):
    path,obj=make_pool(tmp_path);(obj/'mesh_sim.obj').write_text('changed')
    with pytest.raises(ValueError,match='artifact changed'):follow.pool_binding(path,'B1_FIXED_PRIORITY','capture')


def test_pool_rejects_wrong_capture(fake_core,tmp_path):
    path,_=make_pool(tmp_path)
    with pytest.raises(ValueError,match='capture mismatch'):follow.pool_binding(path,'B1_FIXED_PRIORITY','wrong')


def test_pool_index_conflicting_snapshots(fake_core,tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
    p,_=make_pool(a);q,_=make_pool(b)
    value=json.loads(q.read_text());value['different']=True;q.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='different target pools'):follow.pool_index([tmp_path])


def engine_rows(tmp_path,protocol):
    result=[]
    for i,m in enumerate(follow.METHODS):
        p=tmp_path/f'{i}.json';p.write_text(json.dumps({'mechanism_followup':protocol}))
        result.append(dict(row(m),config_path=str(p)))
    return result


def test_exact_new_engine_roster(tmp_path):
    assert follow.expected_engine_methods(engine_rows(tmp_path,follow.PROTOCOL))==list(follow.METHODS)


def test_legacy_engine_roster_unchanged(tmp_path):
    assert 'B4_ROOM_REPAIR_NATIVE' in follow.expected_engine_methods(engine_rows(tmp_path,None))


def test_new_protocol_cannot_be_legacy_pilot(tmp_path):
    with pytest.raises(ValueError):follow.expected_engine_methods(engine_rows(tmp_path,follow.PROTOCOL),pilot=True)


def test_mixed_engine_protocol_rejected(tmp_path):
    units=engine_rows(tmp_path,follow.PROTOCOL);Path(units[0]['config_path']).write_text('{}')
    with pytest.raises(ValueError):follow.expected_engine_methods(units)


def test_scope_drift_rejected(tmp_path):
    units=engine_rows(tmp_path,follow.PROTOCOL);units[0]['scope']='L1_target_destination'
    with pytest.raises(ValueError):follow.expected_engine_methods(units)


def test_native_collision_check_is_not_a_stability_gate(tmp_path):
    trimesh=pytest.importorskip('trimesh');(tmp_path/'collision').mkdir()
    trimesh.creation.box().export(tmp_path/'collision/part_00.obj')
    assert follow.invalid_collision_parts(tmp_path)==[]
    (tmp_path/'collision/part_01.obj').write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    assert len(follow.invalid_collision_parts(tmp_path))==1


def test_crop_is_fixed_from_reference_and_clipped():
    mask=np.zeros((80,100),bool);mask[1:4,1:4]=True
    assert appearance.crop_from_mask(mask)==(0,0,32,32)
    mask[:]=False;mask[77:,97:]=True
    assert appearance.crop_from_mask(mask)==(68,48,100,80)


def test_empty_reference_visibility_is_not_a_perfect_prediction():
    with pytest.raises(ValueError):appearance.crop_from_mask(np.zeros((40,40),bool))


def test_segmentation_channel_order():
    seg=np.array([[[5,17],[17,5],[5,18],[-1,-1]]],np.int32)
    assert appearance.segmentation_mask(seg,[17],5).tolist()==[[True,False,False,False]]


def test_segmentation_invalid_shape():
    with pytest.raises(ValueError):appearance.segmentation_mask(np.zeros((3,3,3)),[1],5)


def test_infinite_psnr_preserved():
    assert appearance.encode_psnr(float('inf'))=={'masked_psnr':None,'masked_psnr_status':'POSITIVE_INFINITY'}
    with pytest.raises(ValueError):appearance.encode_psnr(float('nan'))


def triangle():
    return np.array([[0.,0,0],[.03,0,0],[0,.03,0]]),np.array([[0,1,2]])


def test_surface_centroid():
    v,f=triangle();np.testing.assert_allclose(sensitivity.surface_centroid(v,f),[.01,.01,0])


def test_physical_world_shape_not_body_origin_defines_centroid():
    v,f=triangle();grip=[.26,0,0]
    a=sensitivity.geometric_diagnostics(v,f,grip,[0,0,0])
    b=sensitivity.geometric_diagnostics(v,f,grip,[.08,0,0])
    assert a['gripper_to_visual_centroid_m']==b['gripper_to_visual_centroid_m']
    assert a['origin_retreat_pass']!=b['origin_retreat_pass']
    assert a['official_success_modified'] is False


def test_rotated_goal_basis_and_union():
    region=np.array([[0.,0,0],[0,2,0],[-3,0,0],[0,0,4]])
    assert sensitivity.points_in_regions(np.array([[-1,1,2],[1,1,2]]),[region]).tolist()==[True,False]
    assert sensitivity.points_in_regions(np.array([[0.,0,0]]),[]) is None


def test_degenerate_region_rejected():
    with pytest.raises(ValueError):sensitivity.points_in_regions(np.array([[0.,0,0]]),[np.zeros((4,3))])


def test_trace_gzip_no_object_pose_playback(tmp_path):
    path=tmp_path/'trace.json.gz'
    with gzip.open(path,'wt') as f:json.dump([{'tick':0},{'tick':1}],f)
    assert len(sensitivity.load_trace(path))==2
    with gzip.open(path,'wt') as f:json.dump([{'tick':1}],f)
    with pytest.raises(ValueError):sensitivity.load_trace(path)


def test_actual_spec_admits_new_method_names_only_in_protocol():
    spec=pytest.importorskip('robo.roundtrip.spec')
    assert {'B1_FIXED_PRIORITY','B2_EVIDENCE'} <= spec.METHODS


def test_actual_table_computes_component_contrasts():
    tables=pytest.importorskip('robo.eval.native_scale_tables')
    units=[]
    for layout in (1,2):
        for method in follow.METHODS:
            u=row(method,iid=f'i{layout}');u.update(layout_id=layout, measured=True,
                service_success=method!='B0_FIXED_NATIVE',success=method!='B0_FIXED_NATIVE',
                terminal_status='RECORDED',unit_id=f'{layout}-{method}')
            units.append(u)
    _,comparisons,_=tables.tables(units)
    contrasts={(r['method'],r['reference']) for r in comparisons}
    assert ('B1_FIXED_PRIORITY','B0_FIXED_NATIVE') in contrasts
    assert ('B2_EVIDENCE','B1_FIXED_PRIORITY') in contrasts
    assert ('B3_AGENT_NATIVE','B2_EVIDENCE') in contrasts
