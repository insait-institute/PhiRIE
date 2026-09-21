"""Certified non-invocation preserves expected identity without inventing telemetry."""
import copy
from types import SimpleNamespace
import pytest
from robo.eval.harness_runner import _runtime_contract
from robo.eval.harness_validation import _validate_e4_runtime_policy_manifest

POLICY={'id':'pi05_droid_jointpos','kind':'real','checkpoint_hash':'ac'*32,'training_config':'pi05_droid'}

def manifest():
    return {'treatment_id':'a0','outcome':'build_failure','contract':{
        'policy':copy.deepcopy(POLICY),'policy_checkpoint_hash':'ac'*32,
        'runtime_policy_checkpoint_fingerprint':'ac'*32,
        'policy_execution':'not_invoked_prebuild'}}

def test_only_certified_noninvocation_avoids_live_server_requirement():
    m=manifest()
    assert _validate_e4_runtime_policy_manifest(m,{},POLICY,certified_prebuild=True)==[]
    assert _validate_e4_runtime_policy_manifest(m,{},POLICY)

@pytest.mark.parametrize('mutation',['checkpoint','server','telemetry','outcome','training'])
def test_prebuild_identity_drift_and_forged_execution_rejected(mutation):
    m=manifest()
    if mutation=='checkpoint':m['contract']['runtime_policy_checkpoint_fingerprint']='bc'*32
    if mutation=='server':m['contract']['policy']['server_identity']={'claimed_verified':True}
    if mutation=='telemetry':m['reset_provenance']={'fake':True}
    if mutation=='outcome':m['outcome']='success'
    if mutation=='training':m['contract']['policy']['training_config']='changed'
    assert _validate_e4_runtime_policy_manifest(m,{},POLICY,certified_prebuild=True)


def test_runtime_contract_never_claims_server_for_uninvoked_policy():
    state=SimpleNamespace(ep=0,reset_state_id='t__seed0__ep0',scene_id='scene',
        task_id='t',base_seed=0,reset_seed=17)
    kwargs=dict(config={'policy':POLICY['id'],'contract':{'policy':copy.deepcopy(POLICY),
        'runtime_dependencies':{'declared':True}}},state=state,controller_hash='controller',
        camera_hash='camera',action_convention='absolute_joint_position',action_dim=8,
        policy_hash='ac'*32,task_instruction='move object')
    with pytest.raises(ValueError,match='lacks verified server'):_runtime_contract(**kwargs)
    result=_runtime_contract(**kwargs,policy_not_invoked=True)
    assert result['policy_execution']=='not_invoked_prebuild'
    assert result['policy']==POLICY
    with pytest.raises(ValueError,match='must not claim'):
        _runtime_contract(**kwargs,policy_not_invoked=True,server_identity={'fake':True})
    changed=copy.deepcopy(kwargs);changed['policy_hash']='bc'*32
    with pytest.raises(ValueError,match='checkpoint hash differs'):
        _runtime_contract(**changed,policy_not_invoked=True)
