"""Stateless, paired action noise for the existing OpenPI inference API."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

REQUEST_KEY = 'simany_sampling_request'
RECEIPT_KEY = 'simany_sampling_receipt'
ALGORITHM = 'sha256-pcg64-standard-normal-f32-v1'


def sampling_contract(training_config: str) -> dict:
    dimensions = {'pi05_droid_jointpos': 32, 'pi05_droid_jointpos_sim': 8}
    if training_config not in dimensions:
        raise ValueError('unsupported deterministic sampling training config')
    return {'version': 1, 'algorithm': ALGORITHM, 'numpy_version': np.__version__,
            'shape': [15, dimensions[training_config]], 'dtype': 'float32',
            'seed_source': 'canonical_reset_seed', 'chunk_start': 0,
            'implementation_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in ('sampling_contract.py', 'bound_server.py', 'clients/pi05_client.py')}}


def validate_contract(value: Mapping, training_config: str) -> dict:
    if json.dumps(dict(value), sort_keys=True) != json.dumps(sampling_contract(training_config), sort_keys=True):
        raise ValueError('sampling algorithm/version/shape/environment differs')
    return copy.deepcopy(dict(value))


def request(contract: Mapping, *, seed: int, chunk: int, domain: str) -> dict:
    for key, value in [('seed', seed), ('chunk', chunk)]:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
            raise ValueError(f'sampling {key} must be a uint32 integer')
    if domain not in {'episode', 'warmup'}:
        raise ValueError('sampling domain must be episode or warmup')
    if domain == 'warmup' and (seed != 0 or chunk != 0):
        raise ValueError('warmup sampling must use its fixed independent domain')
    return {'contract': copy.deepcopy(dict(contract)), 'seed': seed, 'chunk': chunk,
            'domain': domain}


def noise_and_receipt(value: Mapping, contract: Mapping) -> tuple[np.ndarray, dict]:
    if not isinstance(value, Mapping) or set(value) != {'contract', 'seed', 'chunk', 'domain'}:
        raise ValueError('missing or malformed sampling request')
    expected = request(contract, seed=value['seed'], chunk=value['chunk'], domain=value['domain'])
    if dict(value) != expected:
        raise ValueError('request sampling contract differs')
    if contract.get('numpy_version') != np.__version__ or contract.get('algorithm') != ALGORITHM:
        raise ValueError('runtime noise implementation differs from frozen declaration')
    payload = json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(payload).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest, 'big')))
    noise = rng.standard_normal(tuple(contract['shape']), dtype=np.float32)
    receipt = {**expected, 'noise_sha256': hashlib.sha256(noise.astype('<f4', copy=False).tobytes()).hexdigest()}
    return noise, receipt


class ExplicitNoisePolicy:
    """Adapt request metadata to public Policy.infer(noise=), without RNG mutation."""
    def __init__(self, policy, contract: Mapping):
        self.policy = policy
        self.contract = copy.deepcopy(dict(contract))

    def infer(self, obs: Mapping) -> dict:
        obs = dict(obs)
        sampling = obs.pop(REQUEST_KEY, None)
        noise, receipt = noise_and_receipt(sampling, self.contract)
        result = dict(self.policy.infer(obs, noise=noise))
        if RECEIPT_KEY in result:
            raise ValueError('upstream policy returned a reserved sampling receipt')
        result[RECEIPT_KEY] = receipt
        return result


def validate_episode_receipts(receipts, contract, *, seed: int, ticks: int,
                              completed: bool, chunk_size: int) -> None:
    if not isinstance(receipts, list):
        raise ValueError('episode sampling receipts are missing')
    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0:
        raise ValueError('sampling evidence ticks must be a nonnegative integer')
    needed = (ticks + chunk_size - 1) // chunk_size
    if len(receipts) < needed or len(receipts) > needed + (0 if completed else 1):
        raise ValueError('sampling receipt count differs from executed ticks')
    for chunk, receipt in enumerate(receipts):
        _, expected = noise_and_receipt(request(contract, seed=seed, chunk=chunk, domain='episode'), contract)
        if receipt != expected:
            raise ValueError('episode sampling receipt seed/chunk/noise differs')
