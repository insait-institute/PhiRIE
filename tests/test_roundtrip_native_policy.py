import numpy as np
import pytest

from robo.roundtrip.native_policy import (
    NativePolicy, STATE_KEYS, CAMERA_KEYS, POLICY_CONFIG,
    UPSTREAM_COMMIT, CHECKPOINT_REVISION, chunk_seed, pack_observation,
)


def observation():
    return {**{k: np.zeros(n) for k, n in zip(STATE_KEYS, [3, 4, 3, 4, 2])},
            **{k: np.full((256, 256, 3), i, dtype=np.uint8) for i, k in enumerate(CAMERA_KEYS.values())},
            'annotation.human.task_description': 'Pick the object from the counter and place it in the sink.'}


class Client:
    def __init__(self):
        self.calls = []
    def get_server_metadata(self):
        return {'policy_config': POLICY_CONFIG, 'openpi_commit': UPSTREAM_COMMIT,
                'checkpoint_revision': CHECKPOINT_REVISION, 'checkpoint_receipt_sha256': 'test-fixture',
                'rng_protocol': 'simany-sr0-policy-v1/per-chunk-jax-key'}
    def infer(self, obs):
        self.calls.append(obs)
        return {'actions': np.arange(50*12).reshape(50, 12), 'simany_rng_seed': obs['_simany_rng_seed']}


def test_official_observation_order_and_three_cameras():
    o = observation()
    for i, k in enumerate(STATE_KEYS): o[k].fill(i)
    p = pack_observation(o)
    assert p['observation/state'].tolist() == [0]*3+[1]*4+[2]*3+[3]*4+[4]*2
    for i, k in enumerate(CAMERA_KEYS):
        assert p[k].shape == (224, 224, 3)
        assert np.all(p[k] == i)
    assert p['prompt'] == o['annotation.human.task_description']


def test_reset_clears_chunk_and_paired_seed_repeats():
    c = Client(); p = NativePolicy('localhost', 8017, client=c)
    p.reset(3)
    actions = [p.infer(observation()) for _ in range(6)]
    assert len(c.calls) == 2
    assert c.calls[0]['_simany_rng_seed'] != c.calls[1]['_simany_rng_seed']
    assert np.array_equal(actions[4], np.arange(48, 60))
    first_seed = c.calls[0]['_simany_rng_seed']
    p.reset(3)
    assert np.array_equal(p.infer(observation()), actions[0])
    assert c.calls[-1]['_simany_rng_seed'] == first_seed


def test_requires_reset():
    with pytest.raises(RuntimeError): NativePolicy('localhost', 8017, client=Client()).infer(observation())


@pytest.mark.parametrize('change', ['missing_camera', 'wrong_dtype', 'wrong_state', 'nonfinite', 'empty_prompt'])
def test_invalid_observation_fails_closed(change):
    o = observation()
    if change == 'missing_camera': del o['video.robot0_agentview_right']
    elif change == 'wrong_dtype': o['video.robot0_agentview_left'] = np.zeros((256, 256, 3), dtype=float)
    elif change == 'wrong_state': o[STATE_KEYS[0]] = np.zeros(4)
    elif change == 'nonfinite': o[STATE_KEYS[0]][0] = np.nan
    else: o['annotation.human.task_description'] = ''
    with pytest.raises((ValueError, KeyError)): pack_observation(o)


def test_wrong_checkpoint_rejected():
    c = Client(); c.get_server_metadata = lambda: {'policy_config': 'pi05_droid'}
    with pytest.raises(ValueError): NativePolicy('localhost', 8017, client=c)


@pytest.mark.parametrize('seed', [-1, True, 1.5])
def test_bad_seed(seed):
    with pytest.raises(ValueError): chunk_seed(seed, 0)


def test_bad_server_action_or_seed():
    c = Client(); p = NativePolicy('localhost', 8017, client=c); p.reset(0)
    c.infer = lambda obs: {'actions': np.zeros((50, 12)), 'simany_rng_seed': -1}
    with pytest.raises(ValueError): p.infer(observation())
    c.infer = lambda obs: {'actions': np.zeros((50, 8)), 'simany_rng_seed': obs['_simany_rng_seed']}
    with pytest.raises(ValueError): p.infer(observation())
