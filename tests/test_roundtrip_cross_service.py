from types import SimpleNamespace as NS
import numpy as np
import pytest
import robo.roundtrip.native_policy as native
from robo.roundtrip.policy_interleaving import compare_services


def factory(change=False):
    def make(host,port):
        def infer(obs):
            actions=np.full((50,12),obs['_simany_rng_seed']%17,dtype=np.float32)
            if change and host=='new':actions[49,11]+=.01
            return {'actions':actions,'simany_rng_seed':obs['_simany_rng_seed']}
        return NS(metadata={'same':'checkpoint'},_client=NS(infer=infer),close=lambda:None)
    return make


def test_cross_service_compares_all_fifty_actions_not_only_executed_prefix(monkeypatch):
    monkeypatch.setattr(native,'pack_observation',lambda obs:{})
    r=compare_services('old',1,'new',2,policy_factory=factory())
    assert r['passed'] and r['model_chunk_requests']==12 and r['native_episodes']==0
    r=compare_services('old',1,'new',2,policy_factory=factory(True))
    assert not r['passed'] and all(x['max_abs_difference']>0 for x in r['rows'])


def test_cross_service_diagnostic_saves_complete_arrays_and_reference_repeat(monkeypatch,tmp_path):
    monkeypatch.setattr(native,'pack_observation',lambda obs:{'image':np.zeros((2,2),dtype=np.uint8)})
    r=compare_services('old',1,'new',2,policy_factory=factory(),artifact_dir=tmp_path/'arrays',repeat_reference=True)
    assert r['passed'] and r['model_chunk_requests']==18
    assert len(list((tmp_path/'arrays').glob('*.npy')))==18
    assert all(x['reference_repeat_byte_exact'] and len(set(x['input_sha256']))==1 for x in r['rows'])
