"""Prospective process binding; legacy cross-service identity gates stay strict."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import uuid
from robo.manifest.hash import canonical_hash

PROTOCOL = 'per_canonical_engine_v1'


def engine_metadata(engine_id, protocol, canonical_instance_id):
    if protocol != PROTOCOL or not engine_id or not canonical_instance_id:
        raise ValueError('policy engine requires declared protocol, unique ID and canonical instance')
    import jax
    root=Path(__file__).resolve().parents[2]
    runtime={'python':platform.python_version(),
        'packages':{name:importlib.metadata.version(name) for name in ('jax','jaxlib','flax','numpy')},
        'devices':[dict(platform=d.platform,kind=d.device_kind,id=d.id) for d in jax.devices()],
        'environment':{k:v for k,v in os.environ.items() if k.startswith(('JAX_','XLA_')) or k in ('CUDA_VISIBLE_DEVICES','CUDA_HOME','CUDA_PATH','LD_LIBRARY_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'source_hashes':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('robo/roundtrip/native_policy_server.py','robo/roundtrip/native_policy.py','robo/roundtrip/policy_engine.py')}}
    return dict(engine_id=engine_id,engine_protocol=protocol,canonical_instance_id=canonical_instance_id,
        process_uuid=str(uuid.uuid4()),runtime=runtime,runtime_fingerprint=canonical_hash(runtime))


def validate_engine(config, metadata, reference=None):
    declared=config.get('policy_engine_protocol');engine=metadata.get('policy_engine')
    if declared is None:
        if engine is not None:raise ValueError('versioned engine requires an explicit experiment protocol')
        return None
    if declared!=PROTOCOL or not isinstance(engine,dict) or engine.get('engine_protocol')!=declared:
        raise ValueError('policy engine protocol mismatch')
    if engine.get('canonical_instance_id')!=config['canonical_instance_id']:
        raise ValueError('policy engine canonical instance differs')
    if not engine.get('engine_id') or not engine.get('process_uuid') or engine.get('runtime_fingerprint')!=canonical_hash(engine.get('runtime')):
        raise ValueError('policy engine identity or runtime fingerprint missing')
    if config.get('policy_engine_id',engine['engine_id'])!=engine['engine_id']:
        raise ValueError('policy engine ID differs from planned engine')
    if reference is not None and reference.get('policy_identity',{}).get('policy_engine')!=engine:
        raise ValueError('reference came from a different policy engine process')
    return engine
