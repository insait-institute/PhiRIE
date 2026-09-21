import json
from pathlib import Path
import pytest
from run.icra2027 import e3_trellis_generation_pilot as runner
from run.icra2027.e3_auto_discovery_pilot import PilotError, identity


def test_scene_output_isolated_and_rejects_cross_scene(tmp_path):
    scene='38d58a7a31';c={'output_scene_id':scene,'source_pilot':str(tmp_path/'discovery'/scene)}
    assert runner.output_directory(c,tmp_path/'freeze') == tmp_path/'freeze/trellis_initial'/scene
    c['output_scene_id']='5748ce6f01'
    with pytest.raises(PilotError,match='differs'):runner.output_directory(c,tmp_path/'freeze')
    c['output_scene_id']='../escape'
    with pytest.raises(PilotError,match='invalid'):runner.output_directory(c,tmp_path/'freeze')


def test_models_hash_once_but_reject_mutation(tmp_path,monkeypatch):
    p=tmp_path/'weight';p.write_bytes(b'original')
    config={'output_scene_id':'38d58a7a31','models':{'model':{'path':str(p),'sha256':identity(p)['sha256']}},'runtime_sha256':'runtime','python':'python'}
    (tmp_path/'contract').mkdir();calls=[]
    def validate(c):calls.append(1);return {'model':identity(p)}
    monkeypatch.setattr(runner,'validate_models',validate)
    first=runner.cohort_models(config,tmp_path)
    assert runner.cohort_models(config,tmp_path)==first and len(calls)==1
    p.write_bytes(b'modified model')
    with pytest.raises(PilotError,match='changed'):runner.cohort_models(config,tmp_path)


def test_scene_directory_cannot_be_redirected(tmp_path):
    other=tmp_path/'elsewhere';other.mkdir()
    (tmp_path/'trellis_initial').symlink_to(other,target_is_directory=True)
    with pytest.raises(PilotError,match='symlinks'):runner.output_directory({},tmp_path)
