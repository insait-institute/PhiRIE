"""Explicit frozen qualification budget retains the complete discovery population."""
import copy
import json
from pathlib import Path
import pytest
import yaml
from robo.eval import e4_candidate_screen as screen
from robo.eval import e4_task_freeze as freezer
from robo.eval import e3_factory_materializer as materializer
from run.icra2027 import e4_full_protocol as full
from tests.test_e4_automatic_task_freeze import automatic_bundle_inputs
from tests.test_e4_automatic_candidates import population
from tests.test_e4_automatic_materializer import automatic_factory_input


def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    return full.original.identity(p)


@pytest.fixture
def budget(tmp_path,monkeypatch):
    scenes=['1000000000'] + [f'{i:010x}' for i in range(1, 50)];populations=[];inventories=[]
    for i,scene in enumerate(scenes):
        count=37 if i<49 else 58
        rows=[{'automatic_instance_id':1000+j,'label':'book' if j<8 else 'wall'} for j in range(count)]
        ref=put(tmp_path/scene/'all_jobs_manifest.json',{'scene_id':scene,'planned_jobs':count,'rows':rows,'source_gaussian_training_provenance':full.original.canonical.FRESH})
        audit=put(tmp_path/scene/'postrun_audit.json',{'all_jobs_manifest_sha256':ref['sha256']})
        populations.append({'scene_id':scene,'planned_objects':count,'all_jobs_manifest':ref,'discovery_audit':audit,'queries':full.original.semantic_pairs(scene,rows)})
        inventories.append({'scene_id':scene,'jobs':[{} for _ in rows]})
    cohort=put(tmp_path/'cohort.json',{'population':{'scene_ids':scenes}})
    refs={'source_cohort':cohort,'role_definition':full.original.identity(screen.CODE_ROOT/'robo/tasks/pi05_tasks.py')}
    protocol=full.protocol_from_population(populations,references=refs);p=tmp_path/'protocol.yaml';p.write_text(yaml.safe_dump(protocol))
    counts={'scenes':50,'jobs':1871,'policy_object_rows':9355}
    inv={'schema_version':2,'freeze_id':'original','source_contract':{'code_commit':'old'},'scenes':inventories,'counts':counts}
    audit={'counts':counts,'scene_roster':scenes,'scene_audits':{r['scene_id']:{'source_discovery':{'all_jobs_manifest.json':r['all_jobs_manifest']['sha256'],'postrun_audit.json':r['discovery_audit']['sha256']}} for r in populations}}
    monkeypatch.setattr(materializer,'_verify_inventory',lambda *a:(inv,None,audit,None))
    queries=populations[0]['queries'];population={'scene_id':scenes[0],'e3_root':str(tmp_path),'e3_freeze_id':'original','e3_producer_commit':'old','counts':{'planned_objects':37},'pairs':[{k:q[k] for k in ['target','receptacle','task_family']} for q in queries]}
    return tmp_path,p,protocol,population,inv


def test_complete_population_then_fixed_budget(budget):
    root,p,_,population,_=budget;r=screen.qualification_selection(p,population,root=root)
    assert r['input_semantic_queries']==8 and len(r['selected_task_ids'])==4 and len(r['excluded_task_ids'])==4
    assert r['full_input_semantic_queries']==400 and r['full_qualification_queries']==200 and r['qualification_cells']==40
    assert len(population['pairs'])==8


@pytest.mark.parametrize('change',['missing_selection','duplicate_selection','swap_selection','drop_scene','truncated_inventory','wrong_source','dropped_pair','source_tamper','changed_scope'])
def test_budget_and_source_mutations_rejected(budget,change):
    root,p,protocol,population,inv=budget
    if change=='missing_selection':protocol['qualification_tasks'].pop()
    if change=='duplicate_selection':protocol['qualification_tasks'][1]=protocol['qualification_tasks'][0]
    if change=='swap_selection':protocol['qualification_tasks'][0]=protocol['source_populations'][0]['queries'][-1]
    if change=='drop_scene':protocol['source_populations'].pop()
    if change=='truncated_inventory':inv['scenes'].pop()
    if change=='wrong_source':inv['source_contract']['code_commit']='other'
    if change=='dropped_pair':population['pairs'].pop()
    if change=='source_tamper':Path(protocol['source_populations'][0]['all_jobs_manifest']['path']).write_text('{}')
    if change=='changed_scope':protocol['paper_ready']=True
    p.write_text(yaml.safe_dump(protocol))
    with pytest.raises((ValueError,screen.CandidateScreenError)):screen.qualification_selection(p,population,root=root)


def test_protocol_symlink_rejected(budget):
    root,p,_,population,_=budget;link=root/'alias.yaml';link.symlink_to(p)
    with pytest.raises(screen.sealed_cpu.PilotGateError):screen.qualification_selection(link,population,root=root)


def test_budget_preserves_full_geometry_placement():
    from tests.test_e4_automatic_task_planning import source_population
    population=source_population();before=copy.deepcopy(population)
    whole=screen.plan_automatic_population_tasks(population)
    ids=[t['task']['task_id'] for t in whole['selected']['tasks']][:2]
    selected=screen.plan_automatic_population_tasks(population,qualification_task_ids=ids)
    assert population==before
    for key in ['table','base_pos','base_yaw']:
        assert selected['selected'][key]==whole['selected'][key]
    assert selected['source_geometry_sha256']==whole['source_geometry_sha256']
    assert len(selected['selected']['tasks'])==2 and selected['input_semantic_queries']==whole['candidate_count']
    with pytest.raises(screen.CandidateScreenError):screen.plan_automatic_population_tasks(population,qualification_task_ids=ids+ids)


@pytest.fixture
def budget_bundle(automatic_bundle_inputs,monkeypatch):
    root,kwargs,candidates=automatic_bundle_inputs
    original=json.loads(Path(kwargs['automatic_population']).read_text());ids=sorted(f"{original['scene_id']}__{q['target']}_to_{q['receptacle'] or 'region'}" for q in original['pairs'])
    protocol=root/'budget.yaml';protocol.write_text('synthetic protocol validated separately')
    selection={'protocol':screen._identity(protocol,root=root),'selected_task_ids':ids[:4],'excluded_task_ids':ids[4:]}
    monkeypatch.setattr(screen,'qualification_selection',lambda *a,**k:copy.deepcopy(selection))
    gate={**original,'qualification_selection':copy.deepcopy(selection)}
    destination=root/'budget_population'
    screen._publish_bundle(destination,manifest_kind='e4_automatic_candidate_population',payloads={'gate.json':screen._json_bytes(gate),'pairs.json':screen._json_bytes(gate['pairs'])},manifest_fields={'scene_id':gate['scene_id'],'screen_id':'budget','code':gate['code']})
    kwargs['automatic_population']=destination/'gate.json'
    for p in candidates.values():
        value=json.loads(p.read_text());value['tasks']=[t for t in value['tasks'] if t['task_id'] in selection['selected_task_ids']];p.write_text(json.dumps(value))
    return root,kwargs,candidates,selection


def test_budget_task_freeze_preserves_population_and_legacy_schema(budget_bundle):
    root,kwargs,_,selection=budget_bundle
    manifest=freezer.freeze_task_bundle(**kwargs)
    assert manifest['schema_version']==2 and manifest['logical_task_ids']==selection['selected_task_ids']
    gate=json.loads(Path(kwargs['automatic_population']).read_text());assert len(gate['pairs'])==9
    checked=freezer.validate_task_bundle(kwargs['out']/'manifest.json',expected_scene_id='09c1414f1b',repository_root=root)
    assert checked['logical_task_ids']==selection['selected_task_ids'] and checked['planned_pair_arm_rows']==8


def test_budget_is_not_max_tasks_shortcut(budget_bundle):
    _,kwargs,_,_=budget_bundle;kwargs['max_tasks']=4
    with pytest.raises(ValueError,match='cannot truncate'):freezer.freeze_task_bundle(**kwargs)


def test_selected_candidate_cannot_be_dropped(budget_bundle):
    _,kwargs,candidates,_=budget_bundle;p=candidates['A4'];value=json.loads(p.read_text());value['tasks'].pop();p.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='every declared pair'):freezer.freeze_task_bundle(**kwargs)


def test_same_counts_different_discovery_source_rejected(budget,monkeypatch):
    root,p,_,population,inv=budget
    _,_,audit,_=materializer._verify_inventory(None)
    audit['scene_audits'][population['scene_id']]['source_discovery']['all_jobs_manifest.json']='other'
    with pytest.raises(screen.CandidateScreenError,match='different discovery'):screen.qualification_selection(p,population,root=root)


def test_yaml_integer_scene_ids_use_canonical_normalization(budget):
    root, p, protocol, population, _ = budget
    cohort_path = Path(protocol['input_references']['source_cohort']['path'])
    cohort = json.loads(cohort_path.read_text())
    cohort['population']['scene_ids'][0] = 1000000000
    protocol['input_references']['source_cohort'] = put(cohort_path, cohort)
    p.write_text(yaml.safe_dump(protocol))
    receipt = screen.qualification_selection(p, population, root=root)
    assert receipt['scene_id'] == '1000000000'
    assert receipt['qualification_queries'] == 4
