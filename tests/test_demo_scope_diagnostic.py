"""Fail-closed presentation pairing; negative outcomes must remain visible."""
from copy import deepcopy
import pytest
from interface.demo_native_progress import _validate_scope_rows


@pytest.fixture
def evidence():
    diagnosis={'kind':'first_predeclared_scope_r0_diagnostic','canonical_instance_id':'native-a'}
    base=dict(canonical_instance_id='native-a',canonical_manifest_sha256='canonical',
        reset_id='r0',reset_contract_sha256='reset',policy_rng_seed=0,policy_identity_sha256='policy',
        policy_engine={'engine_id':'one','process_uuid':'one'},horizon=600,execution_protocol='primary_native',
        renderer='native',sensor_regime='ideal_rgbd_posed',task_id='PickPlaceCounterToSink',
        execution_kind='closed_loop_visual_policy',executed=True,error=None,video_error=None,
        initial_state_sha256='same',first_success_step=None)
    entries=[]
    for label,ticks,success in [('REF',327,True),('L0_B3',420,True),('L1_B3',600,False)]:
        row=dict(base,ticks=ticks,success=success,first_success_step=ticks if success else None,
            controller_method='REF_NATIVE' if label=='REF' else 'B3_AGENT_NATIVE',
            scope='L1_target_destination' if label=='L1_B3' else 'L0_target_only')
        if label=='REF':row['initial_state_sha256']='native-different'
        source={k:row[k] for k in ('ticks','success','first_success_step','initial_state_sha256')}
        source.update(engine=deepcopy(row['policy_engine']),reset_state_id='r0',
            max_penetration_contact={'distance_m':-.01,'object_ids':[None,'destination'],'geom_names':['robot0_link','destination_part']})
        entries.append({'label':label,'result':row,'diagnostic':source})
    return diagnosis,entries


def test_three_outcomes_retained_with_reference_pose_difference(evidence):
    pair=_validate_scope_rows(*evidence)
    assert pair['L1_B3']['result']['success'] is False
    assert len(pair)==3


@pytest.mark.parametrize('key,value',[
    ('policy_engine',{'engine_id':'other'}),('reset_id','r1'),('policy_rng_seed',1),
    ('policy_identity_sha256','other'),('canonical_manifest_sha256','other'),
    ('initial_state_sha256','other'),('scope','L0_target_only'),('controller_method','B4_ROOM_REPAIR_NATIVE'),
    ('error','timeout'),('video_error','failed'),('executed',False)])
def test_mismatched_or_incomplete_episode_rejected(evidence,key,value):
    diagnosis,entries=evidence;entries[2]['result'][key]=value
    with pytest.raises(ValueError):_validate_scope_rows(diagnosis,entries)


@pytest.mark.parametrize('operation',['missing','duplicate','altered_outcome','missing_contact','unrelated_contact'])
def test_missing_duplicate_or_unsupported_diagnosis_rejected(evidence,operation):
    diagnosis,entries=evidence
    if operation=='missing':entries.pop()
    elif operation=='duplicate':entries[2]=deepcopy(entries[1])
    elif operation=='altered_outcome':entries[2]['diagnostic']['success']=True
    elif operation=='missing_contact':entries[2]['diagnostic']['max_penetration_contact']=None
    elif operation=='unrelated_contact':entries[2]['diagnostic']['max_penetration_contact']['geom_names']=['floor','part']
    with pytest.raises(ValueError):_validate_scope_rows(diagnosis,entries)
