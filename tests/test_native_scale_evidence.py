import json
from pathlib import Path
import pytest
from robo.eval.native_scale_evidence import Sources,context_tables,runtime_tables,full_horizon_tables
from robo.eval.native_scale_tables import _sha


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def unit(iid='i',method='B4_ROOM_REPAIR_NATIVE'):
    return dict(canonical_instance_id=iid,controller_method=method,cohort_id='c',scope='L0_target_only',sensor_regime='ideal',renderer='native',execution_protocol='primary_native',split='DEV',policy_id='p')


@pytest.fixture
def context(tmp_path):
    root=tmp_path/'context';m=root/'slot/B4/build_manifest.json'
    manifest=dict(source_code={'dirty':False},method='B4',accepted=False,actual_calls=1,
                  timing={'total_wall_s':2.,'extra_actions_wall_s':1.},capture_manifest_sha256='capture')
    put(m,manifest)
    probe=dict(settle_drift_m=.04,settle_rotation_deg=20.,initial_penetration_m=.001,public_object_surface_guard={'residual_m':.01},passed=False)
    for name in ('verification_before.json','verification_after.json'):put(m.parent/name,probe)
    summary=dict(instance_slot_id='slot',method='B4',accepted=False,actual_calls=1,before_drift_m=.04,after_drift_m=.04,before_rotation_deg=20.,after_rotation_deg=20.,before_penetration_m=.001,after_penetration_m=.001,after_public_residual_m=.01,total_wall_s=2.,extra_actions_wall_s=1.,native_policy_success=None)
    put(root/'context_results.json',[summary])
    receipt=dict(canonical_instance_id='i',method='B4',state='COMPLETE',instance_slot_id='slot',build_manifest=str(m),build_manifest_sha256=_sha(m),accepted=False,binding={'canonical_instance_id':'i','capture_manifest_sha256':'capture'})
    put(root/'execution_receipt.json',[receipt,dict(canonical_instance_id='j',method='B4',state='BUILD_FAILED')])
    arm=dict(unit(),method='B4_ROOM_REPAIR_NATIVE',planned=10,executed=0,measured=10,success_per_planned=0.,success_per_executed=None)
    return root,arm


def test_context_keeps_upstream_denominator_and_independent_stability_null(context):
    root,arm=context
    rows,details=context_tables([unit(),unit('j')],[arm],[root],Sources())
    assert rows[0]['instances_planned']==2 and rows[0]['instances_probed']==1
    assert rows[0]['instances_accepted']==0 and rows[0]['upstream_build_failed']==1
    assert rows[0]['independent_room_stability'] is None and rows[0]['whole_method_seconds'] is None
    assert details[0]['native_policy_success'] is None


@pytest.mark.parametrize('kind',['duplicate','missing','different','policy'])
def test_context_rejects_corrupt_or_missing_aggregate(context,kind):
    root,arm=context;p=root/'context_results.json';rows=json.loads(p.read_text())
    if kind=='duplicate':rows*=2
    if kind=='missing':rows=[]
    if kind=='different':rows[0]['after_drift_m']=0.
    if kind=='policy':rows[0]['native_policy_success']=True
    put(p,rows)
    with pytest.raises(ValueError):context_tables([unit(),unit('j')],[arm],[root],Sources())


def test_runtime_only_measured_stages_and_no_whole_method_sum(tmp_path):
    root=tmp_path/'b0';put(root/'build_manifest.json',{'canonical_instance_id':'i'})
    put(root/'segment_runtime.json',{'exit_code':0,'wall_s':2.})
    rows,details=runtime_tables([unit(),unit('j')],[{'b0_build':str(root)}],Sources())
    segment=next(r for r in rows if r['stage']=='B0_segment_wrapper')
    assert segment['timing_coverage']==.5 and segment['conditional_seconds_per_timed_instance']==2.
    assert all(r['whole_method_seconds'] is None for r in rows)
    assert next(r for r in rows if r['stage']=='separate_CoACD')['measured_seconds_sum'] is None
    with pytest.raises(ValueError,match='duplicate'):runtime_tables([unit()],[{'b0_build':str(root)}]*2,Sources())
    put(root/'segment_runtime.json',{'exit_code':0,'wall_s':-1.})
    with pytest.raises(ValueError,match='negative'):runtime_tables([unit()],[{'b0_build':str(root)}],Sources())


def test_full_horizon_hash_budget_and_unknown_absolute_error(tmp_path):
    result=put(tmp_path/'episode/result.json',{'horizon':2,'ticks':2,'reset_state_id':'r','reset_seed':0,'scene_id':'s','task_id':'t','sensor_regime':'ideal','policy_identity':{'checkpoint':'frozen'},'config_sha256':'config','native_predicates':{'native_task_success':False}})
    actions=put(tmp_path/'episode/actions.json',[[1],[2]])
    row=dict(seed=0,method='REF_NATIVE',directory=str(result.parent),result_sha256=_sha(result),actions_sha256=_sha(actions),planned_steps=2,recorded_steps=2,success_at_horizon=False,success_ever=False,full_horizon_completed=True,execution_kind='closed_loop_visual_policy')
    payload=dict(protocol='full_horizon_feedback_diagnostic',source_receipts=[],rows=[row],planned_episodes=1,counts={'REF_NATIVE':dict(planned=1,executed=1,completed_full_horizon=1,success_ever=0,success_at_horizon=0)},claim_scope='DEV')
    path=put(tmp_path/'summary.json',payload)
    counts,rows=full_horizon_tables([path],Sources());assert counts[0]['absolute_position_error_cm'] is None
    payload['rows'][0]['position_rmse_cm']=0.;put(path,payload)
    with pytest.raises(ValueError,match='absolute'):full_horizon_tables([path],Sources())
    payload['rows'][0].pop('position_rmse_cm');payload['rows']*=2;put(path,payload)
    with pytest.raises(ValueError,match='duplicate'):full_horizon_tables([path],Sources())


def test_missing_context_is_unknown_not_zero(context):
    root,arm=context
    row=dict(unit('z'),method=arm['method'],planned=5,executed=0,measured=0,success_per_planned=None,success_per_executed=None)
    actual=context_tables([unit(),unit('j'),unit('z')],[row],[root],Sources())[0][0]
    # arm population includes all same-block units; incomplete decisions preserve null acceptance.
    assert actual['instances_accepted'] is None


def test_abstention_cannot_appear_as_executed_native_episode(context):
    root,arm=context
    with pytest.raises(ValueError,match='abstained'):
        context_tables([dict(unit(),executed=True),unit('j')],[arm],[root],Sources())


@pytest.fixture
def horizon_v2(tmp_path):
    labels=('REF_NATIVE','REF_IMPORT_CONTROL','B0_REPLAY','B0_LEARNED','OBSERVED_REPLAY','OBSERVED_LEARNED')
    rows=[]
    for label in labels:
        kind='fixed_action_replay' if label.endswith('REPLAY') or label=='REF_IMPORT_CONTROL' else 'closed_loop_visual_policy'
        result=put(tmp_path/label/'result.json',dict(horizon=2,ticks=2,reset_state_id='r0',reset_seed=0,scene_id='s',task_id='t',sensor_regime='ideal',policy_identity={'checkpoint_receipt_sha256':'checkpoint'},config_sha256=label,comparison_contract_sha256='contract',canonical_instance_id='native-new',canonical_manifest_sha256='canon',reset_contract_sha256='reset',reset_id='r0',native_schema_version=2,policy_rng_seed=0,execution_protocol='full_horizon_feedback_diagnostic',native_predicates={'native_task_success':False}))
        actions=put(result.parent/'actions.json',[[1],[2]])
        rows.append(dict(label=label,directory=str(result.parent),result_sha256=_sha(result),actions_sha256=_sha(actions),planned_steps=2,recorded_steps=2,success_at_horizon=False,success_ever=False,full_horizon_completed=True,execution_kind=kind))
    progress=put(tmp_path/'progress.json',{})
    return put(tmp_path/'summary.json',dict(schema_version=2,planned_canonical_instances=1,planned_episodes=6,executed_episodes=6,completed_full_horizon=6,rows=rows,canonical_instance_id='native-new',reset_id='r0',config_comparison_sha256='contract',checkpoint_receipt_sha256='checkpoint',source_progress=str(progress),source_progress_sha256=_sha(progress),claim_scope='single DEV canonical'))


def test_v2_keeps_canonical_group_and_frozen_comparison_not_treatment_config(horizon_v2):
    counts,rows=full_horizon_tables([horizon_v2],Sources())
    assert len(counts)==6 and len(rows)==6
    assert {r['seed'] for r in rows}=={'native-new/r0'}
    assert all(r['position_rmse_cm'] is None for r in rows)
    assert 'OBSERVED_SURFACE_NATIVE_REPLAY' in {r['method'] for r in rows}
    alias=put(horizon_v2.with_name('alias.json'),json.loads(horizon_v2.read_text()))
    with pytest.raises(ValueError,match='duplicate'):full_horizon_tables([horizon_v2,alias],Sources())


@pytest.mark.parametrize('bad',['missing','duplicate','contract','canonical'])
def test_v2_rejects_invalid_declared_roster_or_identity(horizon_v2,bad):
    d=json.loads(horizon_v2.read_text())
    if bad=='missing':d['rows'].pop()
    if bad=='duplicate':d['rows'][-1]=d['rows'][0]
    if bad=='contract':d['config_comparison_sha256']='wrong'
    if bad=='canonical':d['canonical_instance_id']='wrong'
    put(horizon_v2,d)
    with pytest.raises(ValueError):full_horizon_tables([horizon_v2],Sources())


def test_split_runtime_sources_charge_original_reused_stages(tmp_path):
    root=tmp_path/'b0';put(root/'build_manifest.json',{'canonical_instance_id':'i'})
    put(root/'segment_runtime.json',dict(exit_code=0,wall_s=2.))
    rvg=put(tmp_path/'gpu/rvg_receipt.json',dict(wall_s=10.,producer_wall_s=7.))
    pool=put(tmp_path/'cpu/candidate_pool.json',dict(b0_manifest_sha256=_sha(root/'build_manifest.json'),rvg_receipt_sha256=_sha(rvg),selection_wall_s=3.))
    records=[dict(b0_build=str(root),candidate_pool=str(pool),rvg_receipt=str(rvg))]
    rows,details=runtime_tables([unit(),unit('j')],records,Sources())
    values={r['stage']:r for r in rows}
    assert values['B0_segment_wrapper']['measured_seconds_sum']==2.
    assert values['shared_RVG_wrapper']['measured_seconds_sum']==10.
    assert values['shared_RVG_inference_nested']['measured_seconds_sum']==7.
    assert values['shared_selection']['measured_seconds_sum']==3.
    assert values['B0_segment_wrapper']['required_by_methods']==['B0_FIXED_NATIVE','B3_AGENT_NATIVE','B4_ROOM_REPAIR_NATIVE','BM_BUDGET_MATCHED_NATIVE']
    assert all(r['reuse_discount_applied'] is False and r['whole_method_seconds'] is None for r in rows)
    put(rvg,dict(wall_s=0.,producer_wall_s=0.))
    with pytest.raises(ValueError,match='source bytes changed'):runtime_tables([unit()],records,Sources())


@pytest.mark.parametrize('bad',[None,'budget','method'])
def test_context_binding_uses_raw_calls_and_preserves_failure_denominator(context,monkeypatch,bad):
    from robo.eval.native_scale_evidence import context_binding_tables
    import robo.eval.native_scale_construction as construction
    root,arm=context;m=root/'slot/B4/build_manifest.json';manifest=json.loads(m.read_text());manifest.update(maximum_calls=2,action_bank_contract={'maximum_calls_per_target_dependency':2})
    if bad=='budget':manifest['actual_calls']=3
    if bad=='method':manifest['method']='BM'
    put(m,manifest);plan=put(root/'plan.json',dict(binding=dict(unit(),capture_manifest_sha256='capture')))
    row=dict(terminal_status='ABSTAINED',accepted=False,build_manifest=str(m),build_manifest_sha256=_sha(m),plan_path=str(plan),plan_sha256=_sha(plan))
    monkeypatch.setattr(construction,'construction_decisions',lambda units,paths:({('i',arm['method']):row,('j',arm['method']):dict(terminal_status='BUILD_FAILED',propagation_rule='same automatic discovery')},{}))
    if bad:
        with pytest.raises(ValueError):context_binding_tables([unit(),unit('j')],[arm],[],Sources())
    else:
        rows,details=context_binding_tables([unit(),unit('j')],[arm],[],Sources())
        assert rows[0]['extra_tool_calls']==1 and rows[0]['upstream_build_failed']==1
        assert rows[0]['instances_accepted']==0 and rows[0]['independent_room_stability'] is None
        assert details[0]['algorithm_required_context'] is True


def test_failed_construction_runtime_is_charged_without_later_stages(tmp_path,monkeypatch):
    import robo.eval.native_scale_construction as construction
    root=tmp_path/'failure';failure=put(root/'build_failure.json',dict(status='construction_unavailable'))
    timing=put(root/'segment_runtime.json',dict(exit_code=1,wall_s=4.))
    row=dict(terminal_status='BUILD_FAILED',build_manifest=str(failure),build_manifest_sha256=_sha(failure),source_evidence={str(timing):_sha(timing)})
    monkeypatch.setattr(construction,'construction_decisions',lambda u,p:({('i','B0_FIXED_NATIVE'):row},{}))
    record=dict(b0_build=str(root),canonical_instance_id='i')
    with pytest.raises(ValueError,match='bound constructor failure'):runtime_tables([unit()],[record],Sources())
    rows,_=runtime_tables([unit()],[record],Sources(),['snapshot'])
    segment=next(r for r in rows if r['stage']=='B0_segment_wrapper')
    assert segment['measured_seconds_sum']==4. and segment['failed_builds_with_timing']==1
    assert next(r for r in rows if r['stage']=='B0_generate_wrapper')['measured_seconds_sum'] is None
