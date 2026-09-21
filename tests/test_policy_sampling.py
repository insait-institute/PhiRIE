from __future__ import annotations
import copy
from types import SimpleNamespace
import numpy as np
import pytest

from robo.policy.sampling_contract import (
    sampling_contract, validate_contract, request, noise_and_receipt,
    ExplicitNoisePolicy, REQUEST_KEY, RECEIPT_KEY, validate_episode_receipts,
)
from robo.policy.clients.pi05_client import Pi05PolicyClient
from robo.policy.runtime_identity import PolicyRuntimeIdentityError


def test_stateless_noise_is_paired_order_independent_and_domain_isolated():
    contract = sampling_contract('pi05_droid_jointpos')
    envelope = request(contract, seed=91, chunk=0, domain='episode')
    first, receipt = noise_and_receipt(envelope, contract)
    for other in (request(contract, seed=0, chunk=0, domain='warmup'),
                  request(contract, seed=92, chunk=0, domain='episode'),
                  request(contract, seed=91, chunk=1, domain='episode')):
        noise, other_receipt = noise_and_receipt(other, contract)
        assert not np.array_equal(noise, first)
        assert other_receipt['noise_sha256'] != receipt['noise_sha256']
    again, again_receipt = noise_and_receipt(envelope, contract)
    np.testing.assert_array_equal(first, again)
    assert receipt == again_receipt
    assert first.dtype == np.float32 and first.shape == (15, 32)


@pytest.mark.parametrize('field,value', [('version', True), ('version', 2), ('shape', [15, 8]),
    ('numpy_version', 'wrong'), ('algorithm', 'different'), ('dtype', 'float64')])
def test_contract_drift_fails(field, value):
    contract = sampling_contract('pi05_droid_jointpos')
    contract[field] = value
    with pytest.raises(ValueError):
        validate_contract(contract, 'pi05_droid_jointpos')


@pytest.mark.parametrize('seed,chunk,domain', [(True,0,'episode'),(-1,0,'episode'),
    (2**32,0,'episode'),(1,-1,'episode'),(1,0,'warmup'),(0,0,'treatment_A4')])
def test_malformed_seed_chunk_or_domain_fails(seed, chunk, domain):
    with pytest.raises(ValueError):
        request(sampling_contract('pi05_droid_jointpos'), seed=seed, chunk=chunk, domain=domain)


class ToyPolicy:
    def __init__(self):
        self.noises = []
    def infer(self, obs, *, noise):
        assert REQUEST_KEY not in obs
        self.noises.append(noise.copy())
        actions = noise[:, :8] * .01 + np.asarray(obs['observation/joint_position']).mean()
        actions[:, 7] = .5
        return {'actions': actions}


def make_client(monkeypatch, *, fail_after_infer=False, wrong_receipt=False):
    contract = sampling_contract('pi05_droid_jointpos')
    toy = ToyPolicy()
    wrapper = ExplicitNoisePolicy(toy, contract)
    class Wire:
        failed = False
        def infer(self, obs):
            result = wrapper.infer(obs)
            if fail_after_infer and not self.failed:
                self.failed = True
                raise ConnectionError('reply lost after actual inference')
            if wrong_receipt:
                result[RECEIPT_KEY]['noise_sha256'] = '0'*64
            return result
    wire = Wire()
    client = Pi05PolicyClient(connector=lambda: wire, retry_sleep_s=0)
    # Exercise all production sampling methods while substituting only identity
    # transport and image resize, neither of which participates in RNG.
    client._sampling = contract
    monkeypatch.setattr('robo.policy.clients.pi05_client._resize_with_pad', lambda im,h,w: im)
    return client, toy


def obs(offset=0):
    return {'observation/exterior_image_1_left': np.zeros((2,2,3), dtype=np.uint8),
        'observation/wrist_image_left': np.zeros((2,2,3),dtype=np.uint8),
        'observation/joint_position': np.full(7, offset),
        'observation/gripper_position': np.zeros(1)}


def test_client_retry_after_lost_response_preserves_noise_and_episode_sequence(monkeypatch):
    client, toy = make_client(monkeypatch, fail_after_infer=True)
    client.begin_episode(38)
    a = client(obs(), 'pick')
    assert len(toy.noises) == 2
    np.testing.assert_array_equal(toy.noises[0], toy.noises[1])
    receipt = copy.deepcopy(client.sampling_receipts)
    client.warmup(obs(), 'warmup')
    client.begin_episode(38)
    b = client(obs(), 'pick')
    np.testing.assert_array_equal(a,b)
    assert receipt == client.sampling_receipts
    for _ in range(15):
        client(obs(), 'pick')
    assert [r['chunk'] for r in client.sampling_receipts] == [0,1]


def test_same_noise_does_not_force_actions_or_success_to_match(monkeypatch):
    client, toy = make_client(monkeypatch)
    client.begin_episode(38)
    a = client(obs(0), 'pick')
    client.begin_episode(38)
    b = client(obs(.01), 'pick')
    np.testing.assert_array_equal(toy.noises[0], toy.noises[1])
    assert not np.array_equal(a,b)


def test_missing_begin_episode_and_tampered_receipt_fail_before_action(monkeypatch):
    client, toy = make_client(monkeypatch, wrong_receipt=True)
    with pytest.raises(PolicyRuntimeIdentityError, match='begin_episode'):
        client(obs(), 'pick')
    assert not toy.noises
    client.begin_episode(1)
    with pytest.raises(PolicyRuntimeIdentityError, match='receipt'):
        client(obs(), 'pick')
    assert not client.sampling_receipts


def test_wrapper_rejects_absent_or_changed_request_before_policy():
    contract = sampling_contract('pi05_droid_jointpos')
    toy = ToyPolicy()
    wrapper = ExplicitNoisePolicy(toy, contract)
    with pytest.raises(ValueError):
        wrapper.infer(obs())
    envelope = request(contract, seed=1, chunk=0, domain='episode')
    envelope['contract']['shape'] = [15,8]
    with pytest.raises(ValueError):
        wrapper.infer({**obs(), REQUEST_KEY: envelope})
    assert not toy.noises


def test_receipt_validation_rejects_seed_drift_missing_chunks_and_fake_completion():
    contract = sampling_contract('pi05_droid_jointpos')
    receipts = [noise_and_receipt(request(contract, seed=7, chunk=i, domain='episode'), contract)[1] for i in range(2)]
    validate_episode_receipts(receipts,contract,seed=7,ticks=16,completed=True,chunk_size=15)
    for bad,seed,ticks in [(receipts,8,16),(receipts[:1],7,16),(receipts,7,0)]:
        with pytest.raises(ValueError):
            validate_episode_receipts(bad,contract,seed=seed,ticks=ticks,completed=True,chunk_size=15)


def test_sampling_capability_and_serving_code_are_authenticated(tmp_path):
    from tests.test_policy_runtime_identity import _git_repo, _checkpoint
    from robo.policy.runtime_identity import build_server_identity, validate_server_identity
    from robo.manifest.hash import canonical_hash
    upstream = tmp_path/'openpi'
    commit = _git_repo(upstream)
    checkpoint, fingerprint = _checkpoint(tmp_path/'weights')
    contract = sampling_contract('pi05_droid_jointpos')
    identity = build_server_identity(policy_id='pi05_droid_jointpos', checkpoint_path=checkpoint,
        checkpoint_fingerprint=fingerprint, training_config='pi05_droid_jointpos',
        openpi_root=upstream, openpi_commit=commit,port=8100,sampling=contract)
    assert validate_server_identity(identity,expected=identity) == identity
    bad = copy.deepcopy(identity)
    bad['policy']['sampling']['implementation_sha256']['bound_server.py'] = '0'*64
    bad['identity_sha256'] = canonical_hash({k:v for k,v in bad.items() if k!='identity_sha256'})
    with pytest.raises(PolicyRuntimeIdentityError,match='sampling'):
        validate_server_identity(bad)
    legacy = copy.deepcopy(identity)
    legacy['policy'].pop('sampling')
    legacy['identity_sha256'] = canonical_hash({k:v for k,v in legacy.items() if k!='identity_sha256'})
    with pytest.raises(PolicyRuntimeIdentityError, match='exact harness'):
        validate_server_identity(legacy,expected=identity)


def test_canonical_runner_passes_the_exact_frozen_reset_seed_before_actions():
    from robo.eval import harness_runner
    seen = []
    class Env:
        def reset(self, **kwargs):
            self.last_reset_provenance = {'jitter': harness_runner._planned_jitter(
                reset_seed=kwargs['reset_seed'],jitter_xy=kwargs['jitter_xy'],target=kwargs['jitter_body'])}
    class Policy:
        def begin_episode(self, seed):
            seen.append(seed)
            raise ValueError('intentional stop before action')
        def reset(self):
            raise AssertionError('seed would be lost')
    result, ticks, _, _ = harness_runner._run_episode(Env(),{'target':'object'},Policy(),None,
        episode_id='A4',prompt='pick',horizon_s=1,reset_seed=123,jitter_xy=.01,
        policy_timeout_s=1,capture_video=False)
    assert seen == [123] and not ticks
    assert result['outcome'] == 'environment_crash'
    assert 'intentional stop' in result['error']


def test_paired_sampling_compares_algorithm_but_allows_early_termination():
    from robo.eval.harness_validation import validate_manifest_pair
    contract = sampling_contract('pi05_droid_jointpos')
    receipts = [noise_and_receipt(request(contract,seed=6,chunk=i,domain='episode'),contract)[1] for i in range(2)]
    a = {'contract': {'policy': {'sampling':contract},'reset_seed':6},
         'policy_sampling_receipts':receipts,'policy_sampling_ticks':16}
    b = copy.deepcopy(a)
    b['policy_sampling_receipts'] = receipts[:1]
    b['policy_sampling_ticks'] = 5
    assert not validate_manifest_pair(a,b,set())
    b['contract']['policy']['sampling']['algorithm'] = 'wrong'
    assert validate_manifest_pair(a,b,set()) == ['contract.policy.sampling.algorithm']


def test_runtime_manifest_requires_authentic_chunk_receipts(tmp_path):
    from tests.test_policy_runtime_identity import _git_repo, _checkpoint
    from robo.policy.runtime_identity import build_server_identity
    from robo.eval.harness_validation import _validate_e4_runtime_policy_manifest
    upstream=tmp_path/'openpi'
    commit=_git_repo(upstream)
    checkpoint,fingerprint=_checkpoint(tmp_path/'weights')
    sampling=sampling_contract('pi05_droid_jointpos')
    identity=build_server_identity(policy_id='pi05_droid_jointpos',checkpoint_path=checkpoint,
        checkpoint_fingerprint=fingerprint,training_config='pi05_droid_jointpos',
        openpi_root=upstream,openpi_commit=commit,port=8100,sampling=sampling)
    expected={'id':'pi05_droid_jointpos','kind':'real','checkpoint_hash':fingerprint,
        'training_config':'pi05_droid_jointpos','sampling':sampling}
    receipt=noise_and_receipt(request(sampling,seed=6,chunk=0,domain='episode'),sampling)[1]
    manifest={'treatment_id':'A0','outcome':'task_failure',
        'contract':{'policy':{**expected,'server_identity':identity},
            'runtime_policy_checkpoint_fingerprint':fingerprint,
            'policy_checkpoint_hash':fingerprint,'reset_seed':6},
        'policy_sampling_ticks':1,'policy_sampling_receipts':[receipt]}
    assert not _validate_e4_runtime_policy_manifest(manifest,identity,expected)
    for defect in ('seed','missing','ticks'):
        bad=copy.deepcopy(manifest)
        if defect=='seed': bad['policy_sampling_receipts'][0]['seed']=7
        elif defect=='missing': bad.pop('policy_sampling_receipts')
        else: bad['policy_sampling_ticks']=16
        assert any('sampling evidence' in e for e in _validate_e4_runtime_policy_manifest(bad,identity,expected))


def test_second_episode_reset_failure_cannot_copy_previous_noise_receipts(monkeypatch):
    from robo.eval import harness_runner
    client, _ = make_client(monkeypatch)
    client.begin_episode(1)
    client(obs(),'pick')
    first_row_receipts=client.sampling_receipts
    assert first_row_receipts[0]['seed']==1
    class BrokenReset:
        def reset(self, **kwargs):
            raise RuntimeError('second reset fails before a measurement')
    result,ticks,_,provenance=harness_runner._run_episode(
        BrokenReset(),{'target':'object'},client,None,episode_id='second',
        prompt='pick',horizon_s=1,reset_seed=2,jitter_xy=.01,
        policy_timeout_s=1,capture_video=False)
    assert result['outcome']=='environment_crash' and result['ticks']==0
    assert not ticks and provenance is None and client.sampling_receipts==[]
    assert first_row_receipts[0]['seed']==1
    assert client._sampling_seed==2


@pytest.mark.parametrize('horizon',[1,14,16,15.0,True])
def test_sampling_rejects_noncanonical_client_chunk_size(horizon):
    identity={'policy':{'sampling':sampling_contract('pi05_droid_jointpos'),
                        'training_config':'pi05_droid_jointpos'}}
    with pytest.raises(ValueError,match='15-step'):
        Pi05PolicyClient(expected_server_identity=identity,open_loop_horizon=horizon)


@pytest.mark.parametrize('rows',[0,1,14,16])
def test_sampled_action_chunk_must_have_exact_declared_row_count(monkeypatch,rows):
    client,_=make_client(monkeypatch)
    contract=client._sampling
    envelope=request(contract,seed=1,chunk=0,domain='episode')
    receipt=noise_and_receipt(envelope,contract)[1]
    monkeypatch.setattr(client,'_infer_retry',lambda req:{'actions':np.zeros((rows,8)),RECEIPT_KEY:receipt})
    client.begin_episode(1)
    with pytest.raises(ValueError,match='exactly 15 rows'):
        client(obs(),'pick')
    assert client._chunk is None
