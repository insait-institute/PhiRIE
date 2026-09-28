import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('e1_source_audit',Path(__file__).resolve().parents[1]/'run/icra2027/e1_audit_current_sources.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_all_five_regimes_preserve_unmeasured_values():
    rows=[row for scene in m.PAPER_SCENE_IDS for row in m.scene_plan(scene,{'inputs':1},{'accept':1},{'state':'sealed'})]
    assert len(rows)==250 and len({(r['scene_id'],r['regime']) for r in rows})==250
    assert all(not r['table_i_record_admissible'] and not r['reuse_admitted'] for r in rows)
    assert all(all(v is None for v in r['table_i_measurements'].values()) for r in rows)
    assert sum(r['controller_evidence'] is not None for r in rows)==50
    assert {r['regime'] for r in rows if r['controller_evidence']}=={'Auto discovery + splat-fused mesh'}


def test_controller_acceptance_never_promoted_to_simulation_yield():
    rows=m.scene_plan('s',{}, {'accepted_instances':100,'stable_instances':100}, {})
    auto=rows[3]
    assert auto['table_i_measurements']['accepted_instances'] is None
    assert auto['table_i_measurements']['stable_instances'] is None
    assert auto['table_i_measurements']['runtime_minutes'] is None


def test_sealed_metadata_and_tamper(tmp_path):
    f=tmp_path/'gate.json';f.write_text(json.dumps({'scene':'s'}))
    (tmp_path/'seal.json').write_text(json.dumps({'members':{'gate.json':m.identity(f)}}))
    assert m.sealed_metadata(tmp_path,'gate.json')['payload']=={'scene':'s'}
    f.write_text('{}')
    with pytest.raises(ValueError,match='source hash differs'):m.sealed_metadata(tmp_path,'gate.json')


def test_unsealed_inflight_metadata_is_not_completed(tmp_path):
    (tmp_path/'gate.json').write_text('{}')
    assert m.sealed_metadata(tmp_path,'gate.json') is None


def test_missing_source_fails(tmp_path):
    with pytest.raises(FileNotFoundError):m.identity(tmp_path/'missing')
