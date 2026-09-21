import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from robo.roundtrip.spec import load_spec


class Adapter:
    def __init__(self,success=(1,)):
        self.t=0;self.success_steps=success;self.canonical_binding={'scene_xml_sha256':'a'*64,'state_sha256':'b'*64}
        self.native=NS(control_freq=20,sim=NS(data=NS(time=0.,qpos=np.zeros(2),qvel=np.zeros(2))))
    def get_state(self):return {'time':0,'object_states':{}}
    def source_xml(self):return '<mujoco/>'
    def get_policy_observation(self):return {'rgb':np.zeros((4,4,3),dtype=np.uint8)}
    def tracked_objects(self):return {}
    def native_success(self):return self.t in self.success_steps
    def native_stage_state(self):return {'native_task_success':self.native_success()}
    def step_native_action(self,a):
        self.t+=1;self.native.sim.data.time=self.t*.05
        return {},0,self.native_success(),{}


class Policy:
    is_visual_policy=True
    metadata={'checkpoint':'same'}
    def bind_stream(self,**kw):self.binding=kw
    def reset(self,seed):self.seed=seed
    def infer(self,obs):return np.zeros(12)


@pytest.fixture
def config(monkeypatch):
    c=load_spec(Path(__file__).parents[1]/'configs/experiments/sim_recon_sim/reference.yaml')
    c.update(schema_version=2,cohort_id='dev',canonical_instance_id='native-a',reset_id='reset0',
        policy_rng_seed=42,scope='L0_target_only',controller_method='REF_NATIVE',
        execution_protocol='primary_native',canonical_manifest_sha256='c'*64,reset_contract_sha256='d'*64,
        horizon=4,video_image_keys=['rgb'])
    c['policy']['checkpoint_receipt_sha256']='e'*64
    monkeypatch.setitem(__import__('robo.roundtrip.spec',fromlist=['NATIVE_HORIZONS']).NATIVE_HORIZONS,'PickPlaceCounterToSink',4)
    import imageio.v2 as imageio
    monkeypatch.setattr(imageio,'get_writer',lambda *a,**k:NS(append_data=lambda x:None,close=lambda:None))
    return c


def run(config,tmp_path,**kwargs):
    from robo.eval.harness_runner import run_native_episode
    p=Policy();a=Adapter(**kwargs)
    record=run_native_episode(a,p,config=config,reset_seed=0,out_dir=tmp_path/'episode',treatment_id='REF_NATIVE')
    return record,p


def test_primary_native_stops_on_first_success(config,tmp_path):
    r,p=run(config,tmp_path)
    assert r['ticks']==1 and r['success'] is True
    assert r['first_success_step']==1 and r['success_at_horizon'] is None
    assert p.seed==42 and p.binding['canonical_instance_id']=='native-a'
    assert r['reset_id']=='reset0' and r['canonical_reference']['scene_xml_sha256']=='a'*64


def test_full_horizon_uses_real_policy_after_success_no_success_ever_relabel(config,tmp_path):
    config['execution_protocol']='full_horizon_feedback_diagnostic'
    r,_=run(config,tmp_path)
    assert r['ticks']==4 and r['success'] is False and r['success_at_horizon'] is False
    assert r['success_ever'] is True and r['first_success_step']==1
    assert r['full_horizon_completed'] is True
    assert len(json.loads((tmp_path/'episode/actions.json').read_text()))==4


def test_same_seed_other_instance_or_protocol_has_unique_episode_id(config,tmp_path):
    a,_=run(config,tmp_path/'one');c=copy.deepcopy(config);c['canonical_instance_id']='native-b'
    b,_=run(c,tmp_path/'two');c['execution_protocol']='full_horizon_feedback_diagnostic'
    d,_=run(c,tmp_path/'three')
    assert len({a['episode_id'],b['episode_id'],d['episode_id']})==3


@pytest.mark.parametrize('protocol',['primary_native','full_horizon_feedback_diagnostic'])
@pytest.mark.parametrize('versioned_engine',[False,True])
def test_resolved_ref_and_reconstructed_workers_bind_same_canonical_reset(config,tmp_path,monkeypatch,protocol,versioned_engine):
    from robo.roundtrip.identity import create_canonical_instance,create_reset_bank
    from robo.roundtrip.paired import run_resolved_episode
    import robo.roundtrip.paired as paired
    config['execution_protocol']=protocol
    directory=tmp_path/'canonical';directory.mkdir()
    state={'integration_state':[],'controller_state':[],'observable_timing':{},
           'native_metadata':{},'qpos':[],'qvel':[],'object_states':{}}
    (directory/'canonical_state.json').write_text(json.dumps(state));(directory/'scene.xml').write_text('<mujoco/>')
    manifest=create_canonical_instance(directory,config)
    bank=create_reset_bank(manifest,[{'reset_id':'reset0','delta_world':np.eye(4).tolist(),'policy_rng_seed':42}])
    config.update(canonical_instance_id=manifest['canonical_instance_id'],canonical_manifest_sha256=manifest['manifest_sha256'],reset_contract_sha256=bank['reset_contract_sha256'])
    mp=tmp_path/'canonical_instance.json';mp.write_text(json.dumps(manifest))
    bp=tmp_path/'reset_bank.json';bp.write_text(json.dumps(bank))
    monkeypatch.setattr(paired,'prepare_paired_adapter',lambda *a,**k:{'scope':'target_only'})
    calls=[]
    class ResolvedAdapter(Adapter):
        def apply_paired_reset(self,reset,**kwargs):calls.append(reset);return reset
    class BoundPolicy(Policy):
        metadata={'checkpoint_receipt_sha256':'e'*64}
    if versioned_engine:
        from robo.manifest.hash import canonical_hash
        config['policy_engine_protocol']='per_canonical_engine_v1'
        BoundPolicy.metadata['policy_engine']=dict(engine_id='fixed-engine',engine_protocol='per_canonical_engine_v1',
            canonical_instance_id=config['canonical_instance_id'],process_uuid='first-process',runtime={},runtime_fingerprint=canonical_hash({}))
    common=dict(canonical_reference=directory,canonical_manifest=mp,reset_bank=bp)
    ref=run_resolved_episode(ResolvedAdapter(),BoundPolicy(),config=config,out_dir=tmp_path/'ref',**common)
    config['controller_method']='B0_FIXED_NATIVE'
    cmp=run_resolved_episode(ResolvedAdapter(),BoundPolicy(),config=config,out_dir=tmp_path/'cmp',
        object_dir='frozen-build',reference_episode=tmp_path/'ref/episode',**common)
    assert ref['canonical_reference']==cmp['canonical_reference']
    assert ref['comparison_contract_sha256']==cmp['comparison_contract_sha256']
    assert ref['episode_id']!=cmp['episode_id']
    assert calls==[bank['resets'][0],bank['resets'][0]]
    if versioned_engine:
        assert ref['policy_engine']==cmp['policy_engine']==BoundPolicy.metadata['policy_engine']
    if protocol=='full_horizon_feedback_diagnostic':
        replay=run_resolved_episode(ResolvedAdapter(),BoundPolicy(),config=config,out_dir=tmp_path/'replay',
            object_dir='frozen-build',reference_episode=tmp_path/'ref/episode',
            replay_actions=tmp_path/'ref/episode/actions.json',actions_sha256=ref['actions_sha256'],**common)
        assert replay['ticks']==4 and replay['execution_kind']=='fixed_action_replay'
        if versioned_engine:assert replay['policy_engine']==ref['policy_engine']
        assert replay['success_ever'] is True and replay['success_at_horizon'] is False
        assert json.loads((tmp_path/'ref/episode/actions.json').read_text())==json.loads((tmp_path/'replay/episode/actions.json').read_text())
        (tmp_path/'ref/episode/actions.json').write_text('[]')
        with pytest.raises(ValueError,match='source/hash'):
            run_resolved_episode(ResolvedAdapter(),BoundPolicy(),config=config,out_dir=tmp_path/'mutated',
                object_dir='frozen-build',reference_episode=tmp_path/'ref/episode',
                replay_actions=tmp_path/'ref/episode/actions.json',actions_sha256=ref['actions_sha256'],**common)
        assert not (tmp_path/'mutated').exists()
    if versioned_engine:
        BoundPolicy.metadata['policy_engine']['process_uuid']='restarted-process'
        prior=len(calls)
        with pytest.raises(ValueError,match='policy runtime differs'):
            run_resolved_episode(ResolvedAdapter(),BoundPolicy(),config=config,out_dir=tmp_path/'other-engine',
                object_dir='frozen-build',reference_episode=tmp_path/'ref/episode',**common)
        assert len(calls)==prior and not (tmp_path/'other-engine').exists()
    config['policy_rng_seed']=43
    with pytest.raises(ValueError,match='reset or RNG'):
        run_resolved_episode(ResolvedAdapter(),BoundPolicy(),config=config,out_dir=tmp_path/'drift',
            object_dir='frozen-build',reference_episode=tmp_path/'ref/episode',**common)


def test_canonical_ledger_retains_explicit_hybrid_scope_metadata(config,tmp_path):
    from robo.eval.harness_runner import run_native_episode
    a=Adapter();a.scope_binding={'system_variant':'hybrid_scope_diagnostic_target_B3_destination_B0','treatment_axis':'replacement_scope'}
    r=run_native_episode(a,Policy(),config=config,reset_seed=0,out_dir=tmp_path/'episode',treatment_id='REF_NATIVE')
    assert r['scope_treatment']==a.scope_binding
    assert json.loads((tmp_path/'episode/result.json').read_text())['scope_treatment']==a.scope_binding


def test_observed_method_cannot_run_without_control_provenance(config,tmp_path):
    from robo.roundtrip.paired import run_resolved_episode
    config['controller_method']='OBSERVED_SURFACE_NATIVE'
    with pytest.raises(ValueError,match='sealed receipt'):
        run_resolved_episode(Adapter(),Policy(),config=config,canonical_reference='unused',canonical_manifest='unused',reset_bank='unused',out_dir=tmp_path/'no_output',object_dir='unverified')
    assert not (tmp_path/'no_output').exists()
