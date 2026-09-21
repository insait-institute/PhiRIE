from concurrent.futures import ThreadPoolExecutor
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
from robo.roundtrip.native_policy import NativePolicy,chunk_seed,POLICY_CONFIG,UPSTREAM_COMMIT,CHECKPOINT_REVISION
from robo.roundtrip.native_policy_server import SeededPolicy


class Client:
    def get_server_metadata(self):
        return {'policy_config':POLICY_CONFIG,'openpi_commit':UPSTREAM_COMMIT,
            'checkpoint_revision':CHECKPOINT_REVISION,'checkpoint_receipt_sha256':'content',
            'rng_protocol':'simany-sr0-policy-v1/per-chunk-jax-key'}
    def infer(self,obs):
        seed=obs['_simany_rng_seed'];rng=np.random.default_rng(seed)
        return {'actions':rng.normal(size=(50,12)),'simany_rng_seed':seed}


def policy(monkeypatch,instance):
    monkeypatch.setattr('robo.roundtrip.native_policy.pack_observation',lambda o:{})
    p=NativePolicy('test',0,client=Client())
    p.bind_stream(canonical_instance_id=instance,reset_id='r0',policy_rng_seed=42);p.reset(999)
    return p


def test_interleaved_streams_and_resume_have_identical_actions(monkeypatch):
    a=policy(monkeypatch,'a');b=policy(monkeypatch,'b')
    interleaved_a=[];interleaved_b=[]
    for _ in range(8):interleaved_b.append(b.infer({}));interleaved_a.append(a.infer({}))
    aa=policy(monkeypatch,'a');bb=policy(monkeypatch,'b')
    assert np.array_equal(interleaved_a,[aa.infer({}) for _ in range(8)])
    assert np.array_equal(interleaved_b,[bb.infer({}) for _ in range(8)])
    restored=policy(monkeypatch,'other');restored.load_state_dict(a.state_dict())
    assert np.array_equal([a.infer({}) for _ in range(8)],[restored.infer({}) for _ in range(8)])
    assert not np.array_equal(interleaved_a,interleaved_b)


def test_stream_seed_depends_on_instance_reset_and_separate_policy_seed():
    values={chunk_seed(seed,chunk,instance_id=instance,reset_id=reset)
            for seed in [0,1] for chunk in [0,1] for instance in ['a','b'] for reset in ['r0','r1']}
    assert len(values)==16
    with pytest.raises(ValueError):chunk_seed(0,-1)


def test_seed_assignment_and_inference_atomic_across_threads(monkeypatch):
    monkeypatch.setitem(sys.modules,'jax',NS(random=NS(key=lambda seed:seed)))
    class Model:
        def infer(self,obs):
            time.sleep(.002)
            return {'observed_key':self._rng}
    service=SeededPolicy(Model())
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(lambda i:service.infer({'_simany_rng_seed':i}),range(12)))
    assert [row['observed_key'] for row in rows]==list(range(12))
