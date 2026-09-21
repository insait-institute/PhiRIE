"""Source-only planning keeps every automatic pair and canonical task semantics."""
import copy
from pathlib import Path
import numpy as np
import pytest
from robo.eval import e4_candidate_screen as screen
from robo.tasks import pi05_tasks


def source_population():
    labels=['plant pot','plant pot','book','book','book']
    boxes=[[[0,0,1.2],[.4,.3,1.5]], [[.4,0,1.4],[.78,.51,1.56]],
           [[0,0,.9275],[.275,.135,1.0385]], [[.5,.1,1.2],[.629,.195,1.234]],
           [[-.5,-.3,.3],[-.255,-.078,.55]]]
    objects={f'obj_{1000+i}':dict(label=label,aabb=boxes[i],
        prepared_output_index={1:0,2:1}.get(i)) for i,label in enumerate(labels)}
    pairs=[dict(candidate_id=f'09c1414f1b/obj_{target}/{dest or "region"}',
        target=f'obj_{target}',receptacle=dest,
        task_family='object_to_receptacle' if dest else 'object_to_region')
        for target in [1002,1003,1004] for dest in [None,'obj_1000','obj_1001']]
    return dict(scene_id='09c1414f1b',objects=objects,pairs=pairs)


def test_complete_source_roster_uses_existing_placement_without_constructed_metrics(monkeypatch):
    source=source_population();original=copy.deepcopy(source)
    def forbidden(*args,**kwargs):raise AssertionError('construction outcome or GT must not drive planning')
    monkeypatch.setattr(pi05_tasks,'_is_graspable',forbidden)
    monkeypatch.setattr(pi05_tasks,'_pick_scan_camera',forbidden)
    plan=screen.plan_automatic_population_tasks(source)
    assert source==original
    assert plan['candidate_count']==9 and plan['planned_pair_arm_rows']==18
    assert plan['member_slots']==['obj_1002'] and plan['prepared_slots']==['obj_1001','obj_1002']
    assert plan['selected']['table']['thickness_m']==.020
    assert plan['construction_eligibility_checked'] is False
    tasks=[x['task'] for x in plan['selected']['tasks']]
    assert len({t['task_id'] for t in tasks})==9
    assert sum(t['receptacle'] is None for t in tasks)==3
    assert next(t for t in tasks if t['receptacle']=='obj_1001')['receptacle_dims']==pytest.approx([.38,.51,.16])
    suites=screen._automatic_suites_for_plan(scene_id=source['scene_id'],
        factories={'A0':Path('/a0'),'A4':Path('/a4')},plan=plan)
    assert {k:v for k,v in suites['A0'].items() if k!='scene_xml'}=={k:v for k,v in suites['A4'].items() if k!='scene_xml'}
    assert suites['A0']['exclude_objects']==[] and suites['A0']['ext_cam']['mode']=='world'
    for t in tasks:
        if 'region' in t:
            assert t['region']['zlo']==plan['selected']['table']['top_z']-.02


@pytest.mark.parametrize('mutation',['unprepared','nonfinite','oversize'])
def test_source_planning_fails_closed_without_changing_population(mutation):
    source=source_population()
    if mutation=='unprepared':source['objects']['obj_1002']['prepared_output_index']=None
    elif mutation=='nonfinite':source['objects']['obj_1002']['aabb'][0][0]=float('nan')
    else:source['objects']['obj_1002']['aabb'][1][0]=.5
    with pytest.raises(screen.CandidateScreenError):screen.plan_automatic_population_tasks(source)
    assert len(source['pairs'])==9

from tests.test_e4_automatic_task_freeze import automatic_bundle_inputs
from tests.test_e4_automatic_candidates import population
from tests.test_e4_automatic_materializer import automatic_factory_input


def test_automatic_task_preparer_reuses_canonical_freezer_and_qualifier(automatic_bundle_inputs,monkeypatch):
    import json
    from robo.eval import e4_camera_scorer_gate as camera
    root,kwargs,_=automatic_bundle_inputs
    # Isolate the already-tested source placement from this all-unavailable
    # construction fixture, to exercise the real closure/freezer/qualifier path.
    plan=screen.plan_automatic_population_tasks(source_population())
    monkeypatch.setattr(screen,'plan_automatic_population_tasks',lambda _:copy.deepcopy(plan))
    def forbidden(*a,**k):raise AssertionError('failed construction must never build an environment')
    monkeypatch.setattr(camera,'_build_headless_droid_env',forbidden)
    result=screen.prepare_automatic_task_suites(screen_id='tasks',scene_id='09c1414f1b',expected_commit='d'*40)
    assert result['candidate_count']==9 and result['planned_pair_arm_rows']==18
    prepared=screen._load_prepare(root=root,screen_id='tasks',scene_id='09c1414f1b',expected_commit='d'*40)
    contracts=screen._prepared_task_contracts(prepared)
    rows=screen._qualify_task_suites(scene_id='09c1414f1b',task_bundle=prepared['task_bundle'],
        factories=prepared['factories'],menagerie_root=root/'unused')
    replay=screen._replay_qualifier_metrics(rows,scene_id='09c1414f1b',expected_task_contracts=contracts)
    assert replay['cell_count']==90 and replay['exact_900_step_cells']==0
    assert replay['strict_pass_task_count']==0
    assert all(r['workspace'] is None for r in rows)
    monkeypatch.setattr(camera,'_menagerie_snapshot',lambda path,commit:{
        'root':str(root/'unused'),'commit':'b'*40,'files':{}})
    gate=screen.qualify_scene(screen_id='tasks',scene_id='09c1414f1b',expected_commit='d'*40,
        menagerie_root=root/'unused',expected_menagerie_commit='b'*40,automatic_population=True)
    assert gate['cell_count']==90 and gate['exact_900_step_cells']==0
    assert gate['strict_pass_task_count']==0 and not gate['gpu_launch_allowed']
    # Exercise the persisted artifact loader, not only the pure row replay.
    validated=screen._validate_qualifier_output(root=root,screen_id='tasks',
        scene_id='09c1414f1b',expected_commit='d'*40)
    assert len(validated['metric_rows'])==90
    assert validated['gate']['exact_900_step_cells']==0
    assert not validated['gate']['gpu_launch_allowed']
    with pytest.raises(screen.CandidateScreenError,match='outside'):
        screen.qualify_scene(screen_id='tasks',scene_id='09c1414f1b',expected_commit='d'*40,
            menagerie_root=root/'unused',expected_menagerie_commit='b'*40)
    with pytest.raises(screen.CandidateScreenError,match="overwrite"):
        screen.prepare_automatic_task_suites(screen_id='tasks',scene_id='09c1414f1b',expected_commit='d'*40)
    # A changed planning rule cannot reuse the sealed task bundle or CPU inputs.
    plan['selected']['base_pos'][0]+=.01
    with pytest.raises(screen.CandidateScreenError,match='planning differs'):
        screen._load_prepare(root=root,screen_id='tasks',scene_id='09c1414f1b',expected_commit='d'*40)


@pytest.mark.parametrize('interrupt_at',['task_freeze','scene_prepare'])
def test_missing_only_prepare_resume_preserves_published_bundles(automatic_bundle_inputs,monkeypatch,interrupt_at):
    from robo.eval import e4_task_freeze as freezer
    root,_,_=automatic_bundle_inputs
    plan=screen.plan_automatic_population_tasks(source_population())
    monkeypatch.setattr(screen,'plan_automatic_population_tasks',lambda _:copy.deepcopy(plan))
    original_publish=screen._publish_bundle;original_freeze=freezer.freeze_task_bundle
    def stop(*a,**k):raise RuntimeError('simulated interruption')
    if interrupt_at=='task_freeze':monkeypatch.setattr(freezer,'freeze_task_bundle',stop)
    else:
        def publish(path,**kw):
            if path.parent.name=='scene_prepares':stop()
            return original_publish(path,**kw)
        monkeypatch.setattr(screen,'_publish_bundle',publish)
    args=dict(screen_id='tasks',scene_id='09c1414f1b',expected_commit='d'*40)
    with pytest.raises(RuntimeError,match='interruption'):screen.prepare_automatic_task_suites(**args)
    experiment=screen._experiment_root(root,'tasks')
    paths=[p for folder in ('candidate_suites','task_freezes') for p in (experiment/folder).rglob('*') if p.is_file()]
    before={p:p.read_bytes() for p in paths}
    monkeypatch.setattr(freezer,'freeze_task_bundle',original_freeze)
    monkeypatch.setattr(screen,'_publish_bundle',original_publish)
    result=screen.prepare_automatic_task_suites(**args)
    assert result['candidate_count']==9 and all(p.read_bytes()==data for p,data in before.items())
    assert screen._load_prepare(root=root,**args)['task_bundle']['logical_task_ids']


def test_partial_candidate_source_drift_rejected(automatic_bundle_inputs,monkeypatch):
    from robo.eval import e4_task_freeze as freezer
    root,_,_=automatic_bundle_inputs
    plan=screen.plan_automatic_population_tasks(source_population())
    monkeypatch.setattr(screen,'plan_automatic_population_tasks',lambda _:copy.deepcopy(plan))
    def stop(**kwargs):raise RuntimeError('simulated interruption')
    monkeypatch.setattr(freezer,'freeze_task_bundle',stop)
    args=dict(screen_id='tasks',scene_id='09c1414f1b',expected_commit='d'*40)
    with pytest.raises(RuntimeError):screen.prepare_automatic_task_suites(**args)
    plan['selected']['base_pos'][0]+=.01
    with pytest.raises(screen.CandidateScreenError,match='partial candidate'):
        screen.prepare_automatic_task_suites(**args)


def test_partial_task_freeze_from_other_candidate_paths_rejected(automatic_bundle_inputs,monkeypatch):
    from robo.eval import e4_task_freeze as freezer
    root,_,_=automatic_bundle_inputs
    plan=screen.plan_automatic_population_tasks(source_population())
    monkeypatch.setattr(screen,'plan_automatic_population_tasks',lambda _:copy.deepcopy(plan))
    original=freezer.freeze_task_bundle;captured={}
    def stop(**kwargs):captured.update(kwargs);raise RuntimeError('simulated interruption')
    monkeypatch.setattr(freezer,'freeze_task_bundle',stop)
    args=dict(screen_id='tasks',scene_id='09c1414f1b',expected_commit='d'*40)
    with pytest.raises(RuntimeError):screen.prepare_automatic_task_suites(**args)
    for name in ('a0_candidates','a4_candidates'):
        alternate=root/f'alternate_{name}.json';alternate.write_bytes(Path(captured[name]).read_bytes())
        captured[name]=alternate
    original(**captured)  # Valid task bundle, but belongs to a different input path.
    monkeypatch.setattr(freezer,'freeze_task_bundle',original)
    with pytest.raises(screen.CandidateScreenError,match='partial task freeze'):
        screen.prepare_automatic_task_suites(**args)
