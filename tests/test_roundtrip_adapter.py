from pathlib import Path
import copy
import numpy as np
import pytest
from robo.roundtrip.spec import load_spec, validate_spec, assert_frozen_pair
from robo.roundtrip.reference import compare
from robo.roundtrip.adapters.robocasa import RoboCasaAdapter

CONFIG=Path(__file__).parents[1]/'configs/experiments/sim_recon_sim/reference.yaml'

def test_planning_template_rejected():
    with pytest.raises(ValueError,match='resolved native schema'):
        load_spec(Path(__file__).parents[1]/'plan/icra2027/11_sim_recon_sim/experiment.template.yaml')

def test_native_pair_rejects_policy_camera_horizon_drift():
    a=load_spec(CONFIG)
    for key,value in [('robot','Panda'),('horizon',100),('camera_size',[224,224])]:
        b=copy.deepcopy(a);b[key]=value
        with pytest.raises(ValueError,match='paired-field drift'):assert_frozen_pair(a,b)
    b=copy.deepcopy(a);b['build_manifest']='new_reconstructed_manifest';assert_frozen_pair(a,b)

def test_unresolved_pin_and_test_cohort_rejected():
    c=load_spec(CONFIG);c['platform']['robocasa_commit']='latest'
    with pytest.raises(ValueError,match='unresolved'):validate_spec(c)
    c=load_spec(CONFIG);c['instance']['split']='test'
    with pytest.raises(ValueError,match='DEV only'):validate_spec(c)

def test_uint8_difference_does_not_wrap():
    result=compare({'rgb':np.array([0],np.uint8)},{'rgb':np.array([255],np.uint8)})
    assert result['rgb']['max_abs']==255

def test_adapter_rejects_wrong_actions_before_native_step():
    a=RoboCasaAdapter(load_spec(CONFIG))
    for act in [np.zeros(8),np.full(12,np.nan)]:
        with pytest.raises(ValueError,match='finite12D'):a.step_native_action(act)

def test_canonical_native_ledger_rejects_replay_mislabel(tmp_path):
    from robo.eval.harness_runner import run_native_episode
    with pytest.raises(ValueError,match='cannot be labeled'):
        run_native_episode(None,object(),config=load_spec(CONFIG),reset_seed=0,
            out_dir=tmp_path/'episode',treatment_id='REF_NATIVE')
    assert not (tmp_path/'episode').exists()
