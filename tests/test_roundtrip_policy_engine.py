import copy
import pytest
from robo.manifest.hash import canonical_hash
from robo.roundtrip.policy_engine import PROTOCOL,validate_engine


def fixture():
    config=dict(policy_engine_protocol=PROTOCOL,canonical_instance_id='native-a',policy_engine_id='planned-engine')
    engine=dict(engine_id='planned-engine',engine_protocol=PROTOCOL,canonical_instance_id='native-a',
        process_uuid='actual-process-one',runtime={'jax':'frozen'},runtime_fingerprint=canonical_hash({'jax':'frozen'}))
    return config,dict(policy_engine=engine)


def test_legacy_metadata_unchanged_and_versioned_protocol_required():
    assert validate_engine({}, {}) is None
    config,metadata=fixture()
    with pytest.raises(ValueError,match='explicit'):validate_engine({},metadata)
    with pytest.raises(ValueError,match='protocol'):validate_engine(config,{})


def test_restarting_same_named_engine_cannot_reuse_reference():
    config,metadata=fixture();reference={'policy_identity':copy.deepcopy(metadata)}
    assert validate_engine(config,metadata,reference)==metadata['policy_engine']
    metadata['policy_engine']['process_uuid']='actual-process-two'
    with pytest.raises(ValueError,match='different policy engine'):validate_engine(config,metadata,reference)


def test_canonical_and_runtime_binding_fail_closed():
    config,metadata=fixture();config['canonical_instance_id']='native-b'
    with pytest.raises(ValueError,match='canonical'):validate_engine(config,metadata)
    config['canonical_instance_id']='native-a';metadata['policy_engine']['runtime']['jax']='changed'
    with pytest.raises(ValueError,match='fingerprint'):validate_engine(config,metadata)
