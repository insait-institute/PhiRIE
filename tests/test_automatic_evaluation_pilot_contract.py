"""Pilot declaration changes only the matching manifest's population contract."""
import json
from pathlib import Path

import pytest
import yaml

from agents.eval import automatic_matching_manifest as matching


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    e3 = matching.e3
    monkeypatch.setattr(e3, 'REPOSITORY_ROOT', tmp_path)
    roster = [str(s) for s in yaml.safe_load((Path(__file__).resolve().parents[1] /
        'configs/experiments/icra2027/construction_regimes.yaml').read_text())['population']['scene_ids']]
    scene = '13c3e046d7'
    slots = {s: [f'obj_{1000+i}' for i in range(1 if s == scene else 2)] for s in roster}
    slots[roster[-1]] = [f'obj_{1000+i}' for i in range(1871-sum(len(v) for s,v in slots.items() if s!=roster[-1]))]
    execution = dict(schema_version=2, scope='fresh_canonical_engineering',paper_ready=False,
        freeze_id='source',scene_ids=roster,scene_object_slots=slots,planned_scenes=50,
        planned_jobs=1871,planned_policy_object_rows=9355,
        pilot=dict(selection_rule='smallest_positive_population_then_scene_id',scene_id=scene,planned_jobs=1))
    path = tmp_path/'execution.yaml';path.write_text(yaml.safe_dump(execution,sort_keys=False))
    anchor = matching._absolute_identity(path)
    root = tmp_path/'source/agentic';root.mkdir(parents=True)
    contract = dict(freeze_id='source',code=dict(commit='a'*40,dirty=False),resource_inventory=[
        dict(resolved_path=str(path),hash_method='content_sha256',sha256=anchor['sha256'])])
    contract['contract_sha256'] = matching.canonical_hash(contract)
    contract_path = root.parent/'contract/freeze_manifest.json';contract_path.parent.mkdir()
    contract_path.write_text(json.dumps(contract))
    jobs = dict(freeze_id='source',counts=dict(scenes=50,jobs=1871,policy_object_rows=9355),
        source_contract=dict(contract_sha256=contract['contract_sha256'],code_commit='a'*40),
        scenes=[dict(scene_id=s,jobs=[dict(object_slot=slot) for slot in values]) for s,values in slots.items()])
    monkeypatch.setattr(e3,'_load_inventory',lambda *a,**kw:(jobs,{}))
    config = dict(protocol=matching.PROTOCOL,paper_ready=False,mode='pilot',planned_scenes=1,
        planned_jobs=1,pilot_execution_config=anchor,dataset_root='/data/ScanNetpp',
        scenes=[dict(scene_id=scene,planned_jobs=1,construction_root=str(root))])
    roster_path = tmp_path/'roster.yaml';roster_path.write_text(yaml.safe_dump({'population':{'scene_ids':roster}}))
    config['population_roster'] = matching._absolute_identity(roster_path)
    return config, roster, execution, jobs, contract_path


def test_predeclared_smallest_input_pilot_is_accepted(pilot):
    config,roster,*_ = pilot
    assert matching._pilot_from_execution(config,roster) == (['13c3e046d7'],1)


@pytest.mark.parametrize('mutation',['undeclared_scene','count','count_bool','roster','hash','source','inventory','inventory_order','inventory_identity','unbound_config','altered_rule'])
def test_pilot_contract_drift_rejected(pilot, mutation):
    config,roster,execution,jobs,contract_path = pilot
    if mutation=='undeclared_scene':config['scenes'][0]['scene_id']='ac48a9b736'
    elif mutation=='count':config['planned_jobs']=2
    elif mutation=='count_bool':config['planned_jobs']=True
    elif mutation=='roster':roster.reverse()
    elif mutation=='hash':config['pilot_execution_config']['sha256']='0'*64
    elif mutation=='source':jobs['source_contract']['code_commit']='b'*40
    elif mutation=='inventory':jobs['scenes'][0]['jobs'].pop()
    elif mutation=='inventory_order':jobs['scenes'].reverse()
    elif mutation=='inventory_identity':jobs['scenes'][0]['jobs'][0]['object_slot']='obj_9999'
    elif mutation=='unbound_config':
        contract=json.loads(contract_path.read_text());contract['resource_inventory']=[]
        contract_path.write_text(json.dumps(contract))
    else:
        execution['pilot']['selection_rule']='pick_the_best_result'
        path=Path(config['pilot_execution_config']['path']);path.write_text(yaml.safe_dump(execution,sort_keys=False))
        config['pilot_execution_config']=matching._absolute_identity(path)
    with pytest.raises(ValueError):matching._pilot_from_execution(config,roster)


def test_new_pilot_still_requires_control_seal_before_gt(pilot, monkeypatch):
    config,*_=pilot
    def unsealed(*a,**kw):raise ValueError('controller seal missing')
    def forbidden(*a,**kw):pytest.fail('GT accessed before construction seal')
    monkeypatch.setattr(matching.e3,'_load_control_scene',unsealed)
    from agents.eval import eval_vs_gt
    monkeypatch.setattr(eval_vs_gt,'load_scene_gt',forbidden)
    with pytest.raises(ValueError,match='controller seal missing'):
        matching.validate_construction(config)


def test_legacy_pilot_does_not_silently_accept_new_scene(pilot):
    config,*_=pilot
    config.pop('pilot_execution_config')
    with pytest.raises(ValueError,match='predeclared evaluation roster'):
        matching.validate_construction(config)
