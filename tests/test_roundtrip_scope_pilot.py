import json
from types import SimpleNamespace as NS
import pytest
from robo.roundtrip import scope_pilot


def test_scope_pilot_requires_allocated_gpu(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(ValueError,match='allocated'):scope_pilot.run(NS())


def test_dead_engine_preserves_three_planned_episodes_without_native_fallback(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','test');monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    c=tmp_path/'config.json';c.write_text(json.dumps(dict(scope='L0_target_only',controller_method='B3_AGENT_NATIVE',canonical_instance_id='native-a',instance={'task_id':'PickPlaceSinkToCounter'})))
    b=tmp_path/'build.json';b.write_text(json.dumps(dict(canonical_instance_id='native-a',object_role='receptacle')))
    monkeypatch.setattr(scope_pilot,'validate_spec',lambda x:None)
    monkeypatch.setattr(scope_pilot,'selected_destination',lambda *x:None)
    monkeypatch.setattr(scope_pilot,'owns_listening_port',lambda *x:False)
    monkeypatch.setattr(scope_pilot.subprocess,'Popen',lambda *a,**kw:NS(pid=123,poll=lambda:1))
    out=tmp_path/'run';args=NS(baseline_config=str(c),destination_build_manifest=str(b),destination_candidate_pool='pool',destination_rvg_receipt='rvg',out=str(out))
    with pytest.raises(RuntimeError,match='before readiness'):scope_pilot.run(args)
    failure=json.loads((out/'failure.json').read_text())
    assert failure['planned_episodes']==3 and failure['completed_rows']==[]
    assert not (out/'REF').exists() and not (out/'L1_B3').exists()
