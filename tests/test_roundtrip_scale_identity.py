import copy
import json
from pathlib import Path
import numpy as np
import pytest
from robo.roundtrip.identity import (create_canonical_instance,validate_canonical_instance,
    create_reset_bank,validate_reset_bank,assert_unique_instances,perturb_pose)
from robo.roundtrip.spec import load_spec,validate_spec


@pytest.fixture
def native_bundle(tmp_path):
    config=load_spec(Path(__file__).parents[1]/'configs/experiments/sim_recon_sim/reference.yaml')
    state={k:[] for k in ['integration_state','controller_state','qpos','qvel']}
    state.update(observable_timing={'camera':{'phase':.01}},native_metadata={'object_id':'one'},object_states={})
    (tmp_path/'canonical_state.json').write_text(json.dumps(state))
    (tmp_path/'shape.obj').write_text('test immutable asset')
    (tmp_path/'scene.xml').write_text('<mujoco><asset><mesh file="shape.obj"/></asset></mujoco>')
    return tmp_path,config


def test_canonical_assets_state_and_duplicate_instances(native_bundle):
    directory,config=native_bundle;m=create_canonical_instance(directory,config)
    assert validate_canonical_instance(m,directory)==m['canonical_instance_id']
    with pytest.raises(ValueError,match='duplicate'):assert_unique_instances([m,m])
    (directory/'shape.obj').write_text('changed bytes')
    with pytest.raises(ValueError,match='asset closure'):validate_canonical_instance(m,directory)


def test_same_seed_different_metadata_is_different_instance(native_bundle):
    directory,config=native_bundle;a=create_canonical_instance(directory,config)
    path=directory/'canonical_state.json';state=json.loads(path.read_text());state['native_metadata']['object_id']='two'
    path.write_text(json.dumps(state));b=create_canonical_instance(directory,config)
    assert a['canonical_instance_id']!=b['canonical_instance_id']
    with pytest.raises(ValueError,match='XML/state'):validate_canonical_instance(a,directory)


def test_resets_separate_from_generation_and_reject_gt_snap(native_bundle):
    directory,config=native_bundle;m=create_canonical_instance(directory,config)
    record={'reset_id':'r0','delta_world':np.eye(4).tolist(),'policy_rng_seed':0}
    bank=create_reset_bank(m,[record]);assert validate_reset_bank(bank,m)
    with pytest.raises(ValueError,match='duplicate'):create_reset_bank(m,[record,record])
    with pytest.raises(ValueError,match='absolute GT'):create_reset_bank(m,[dict(record,reference_pose=[0]*7)])
    bank['resets'][0]['policy_rng_seed']=3
    with pytest.raises(ValueError):validate_reset_bank(bank,m)


def test_common_world_reset_preserves_estimated_error():
    delta=np.eye(4);delta[:2,:2]=[[0,-1],[1,0]];delta[:3,3]=[.01,.02,0]
    reference=perturb_pose([0,0,0,1,0,0,0],delta)
    estimated=perturb_pose([.03,.04,0,1,0,0,0],delta)
    assert np.linalg.norm(np.asarray(reference[:3])-estimated[:3])==pytest.approx(.05)
    assert not np.array_equal(reference,estimated)


def test_identity_and_translation_reset_preserve_quaternion_bytes():
    pose=np.array([-.0,.2,.3,.7071067811865475,0,0,-.7071067811865475])
    assert np.asarray(perturb_pose(pose,np.eye(4))).tobytes()==pose.tobytes()
    delta=np.eye(4);delta[0,3]=.01
    assert np.asarray(perturb_pose(pose,delta))[3:].tobytes()==pose[3:].tobytes()


def v2_config(native_bundle):
    _,c=native_bundle;c=copy.deepcopy(c)
    c.update(schema_version=2,cohort_id='dev',canonical_instance_id='native-a',reset_id='r0',policy_rng_seed=2,
             canonical_manifest_sha256='a'*64,reset_contract_sha256='b'*64,scope='L0_target_only',
             controller_method='REF_NATIVE',execution_protocol='primary_native')
    c['policy']['checkpoint_receipt_sha256']='c'*64
    return c


@pytest.mark.parametrize('bad',['horizon','test','checkpoint','threshold','phase','reset'])
def test_schema2_rejects_unresolved_or_changed_contract(native_bundle,bad):
    c=v2_config(native_bundle);validate_spec(c)
    if bad=='horizon':c['instance']['task_id']='PickPlaceSinkToCounter'
    elif bad=='test':c['instance']['split']='test'
    elif bad=='checkpoint':c['policy']['checkpoint_revision']='wrong'
    elif bad=='threshold':c['native_success_threshold_overrides']={'retreat':.20}
    elif bad=='phase':c['execution_protocol']='success_ever'
    else:del c['reset_id']
    with pytest.raises(ValueError):validate_spec(c)


def test_schema1_unchanged_and_schema2_task_horizon(native_bundle):
    validate_spec(native_bundle[1]);c=v2_config(native_bundle)
    c['instance']['task_id']='PickPlaceSinkToCounter';c['horizon']=900
    validate_spec(c)


def test_prospective_cabinet_uses_official_horizon(native_bundle):
    c=v2_config(native_bundle);c['instance']['task_id']='PickPlaceCounterToCabinet';c['horizon']=750
    validate_spec(c)
    c['horizon']=600
    with pytest.raises(ValueError,match='horizon'):validate_spec(c)
