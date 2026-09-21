import json
import pytest
from robo.roundtrip.continuation import aggregate,mappings,sha


def test_requires_complete_accounting_and_unique_bindings(tmp_path):
    (tmp_path/'run_manifest.json').write_text(json.dumps({'config':{'reset_seeds':[0,1]}}))
    with pytest.raises(ValueError,match='exactly once'):aggregate(tmp_path,{0:tmp_path},{})
    with pytest.raises(ValueError,match='duplicate'):mappings(['0=a','0=b'])


def test_build_failure_is_not_an_executed_native_failure(tmp_path):
    pilot=tmp_path/'pilot';pilot.mkdir();(pilot/'run_manifest.json').write_text(json.dumps({'config':{'reset_seeds':[0]}}))
    ref=pilot/'episode_seed0';ref.mkdir()
    r=dict(execution_kind='closed_loop_visual_policy',reset_seed=0,config_sha256='x',policy_identity={'checkpoint_receipt_sha256':'y'},executed=True,success=False,ticks=600,horizon=600,error=None)
    (ref/'result.json').write_text(json.dumps(r));failure=tmp_path/'failure.json'
    failure.write_text(json.dumps(dict(status='construction_unavailable',built_objects=0,phase='segment')))
    report=aggregate(pilot,{}, {0:failure});b0=report['rows'][1]
    assert b0['native_success'] is None and b0['executed'] is False and b0['service_success'] is False
    assert report['summary']['B0_FIXED_NATIVE']['success_per_executed'] is None
    failure.write_text(json.dumps(dict(status='not_scheduled',built_objects=0,phase='segment')))
    with pytest.raises(ValueError,match='actual construction failure'):aggregate(pilot,{}, {0:failure})
