"""Canonical task bundles retain complete automatic jobs, including absent bodies."""
import copy
import json
import shutil
from pathlib import Path
import pytest

from robo.eval import e4_task_freeze as freeze
from robo.eval import e4_candidate_screen as screen
from tests.test_e4_automatic_candidates import population
from tests.test_e4_automatic_materializer import automatic_factory_input
from tests.test_e4_collision_repair import _write_minimal_repaired_export
from tests.test_e3_factory_materializer import _write_json


@pytest.fixture
def automatic_bundle_inputs(population,monkeypatch):
    root,out,descriptor=population
    monkeypatch.setattr(freeze,'REPOSITORY_ROOT',root)
    monkeypatch.setattr(freeze,'_git_snapshot',lambda:dict(code_root=str(freeze.CODE_ROOT),commit='d'*40,dirty=False,status=[]))
    def synthetic_export(factory,*,scene_id,root,common_carve_factories,automatic=False):
        assert automatic
        factories=dict(zip(screen.POLICIES,common_carve_factories,strict=True))
        context=screen._automatic_export_context(factories,scene_id=scene_id,root=root)
        source,collision=_write_minimal_repaired_export(root/('synthetic-'+factory.name))
        for name in ('sim','sim_export'):shutil.copytree(source/name,factory/name)
        xml=factory/'sim_export/scene.xml';xml.write_text(xml.read_text().replace(str(source),str(factory)))
        collision['background_carve'].update(support_clip_source='automatic_instance_aabb_bottom_plus_5mm',
            discovered_slots=context['object_slots'],carved_slots=[],hull_count=0,
            source_mesh_sha256=context['source_mesh_sha256'])
        collision['collision_exclusion']['hull_count']=0
        _write_json(factory/'sim_export/room_collision_report.json',collision)
    monkeypatch.setattr(screen,'_run_full_room_export',synthetic_export)
    result=screen.prepare_automatic_candidates(screen_id='tasks',scene_id='09c1414f1b',
        e3_root=out,automatic_scene_descriptor=descriptor,expected_commit='d'*40,export=True)
    directory=root/'outputs/icra2027/tasks/automatic_candidates/09c1414f1b'
    factories={p:directory/'materialized'/p for p in screen.POLICIES}
    tasks=[]
    for pair in result['pairs']:
        tasks.append(dict(task_id=f"09c1414f1b__{pair['target']}_to_"+(pair['receptacle'] or 'region'),
            target=pair['target'],receptacle=pair['receptacle'],
            region=None if pair['receptacle'] else dict(cx=.2,cy=0,hx=.1,hy=.1),
            instructions={'default':'move the book to the declared destination'}))
    candidates={}
    for p in screen.POLICIES:
        suite=dict(scene='09c1414f1b',scene_xml=str(factories[p]/'sim_export/scene.xml'),tasks=tasks,
            robot=dict(base_pos=[0,0,.5],base_yaw=0),table=dict(cx=0,cy=0,hx=.5,hy=.5,top_z=.5),
            ext_cam=dict(mode='world',pos=[0,1,1]),exclude_objects=[],time_limit_s=16.)
        candidates[p]=root/f'{p}-candidates.json';_write_json(candidates[p],suite)
    kwargs=dict(scene_id='09c1414f1b',a0_factory=factories['A0'],a4_factory=factories['A4'],
        a0_candidates=candidates['A0'],a4_candidates=candidates['A4'],planning_source='A4',
        out=root/'task-bundle',repository_root=root,automatic_population=directory/'population/gate.json')
    return root,kwargs,candidates


def test_canonical_automatic_task_freeze_preserves_nine_pairs_with_zero_accepted(automatic_bundle_inputs):
    root,kwargs,_=automatic_bundle_inputs
    manifest=freeze.freeze_task_bundle(**kwargs)
    assert manifest['schema_version']==2 and manifest['e3_claim_status'] is None
    assert len(manifest['logical_task_ids'])==9
    validated=freeze.validate_task_bundle(root/'task-bundle/manifest.json',expected_scene_id='09c1414f1b',repository_root=root)
    assert len(validated['logical_task_ids'])==9
    a0=json.loads((root/'task-bundle/a0_tasks.json').read_text())
    a4=json.loads((root/'task-bundle/a4_tasks.json').read_text())
    assert {k:v for k,v in a0.items() if k!='scene_xml'}=={k:v for k,v in a4.items() if k!='scene_xml'}


@pytest.mark.parametrize('mutation',['drop_task','swap_endpoint','truncate','unsealed_population'])
def test_automatic_task_freeze_rejects_population_drift(automatic_bundle_inputs,mutation):
    root,kwargs,candidates=automatic_bundle_inputs
    if mutation=='truncate':kwargs['max_tasks']=1
    elif mutation=='unsealed_population':
        path=kwargs['automatic_population'];data=json.loads(path.read_text());data['pairs'].pop();_write_json(path,data)
    else:
        for path in candidates.values():
            suite=json.loads(path.read_text())
            if mutation=='drop_task':suite['tasks'].pop()
            else:suite['tasks'][0]['target']='obj_1000'
            _write_json(path,suite)
    with pytest.raises((ValueError, screen.CandidateScreenError)):freeze.freeze_task_bundle(**kwargs)
    assert not (root/'task-bundle').exists()


def test_automatic_cpu_qualifier_records_all_ninety_failed_cells_without_env(automatic_bundle_inputs,monkeypatch):
    from robo.eval import e4_camera_scorer_gate as camera
    root,kwargs,_=automatic_bundle_inputs
    freeze.freeze_task_bundle(**kwargs)
    bundle=freeze.validate_task_bundle(root/'task-bundle/manifest.json',expected_scene_id='09c1414f1b',repository_root=root)
    def forbidden(*a,**k):raise AssertionError('invalid construction arm instantiated an environment')
    monkeypatch.setattr(camera,'_build_headless_droid_env',forbidden)
    rows=screen._qualify_task_suites(scene_id='09c1414f1b',task_bundle=bundle,
        factories={p:Path(bundle['factories'][p]) for p in screen.POLICIES},menagerie_root=root/'unused-robot')
    assert len(rows)==90 and len({r['cell_id'] for r in rows})==90
    assert all(r['failure_type']=='construction_endpoint_unavailable' and not r['passed'] for r in rows)
    assert all(r['reset_jitter'] is None and r['workspace'] is None and r['settle_protocol'] is None for r in rows)
    broken=copy.deepcopy(bundle);broken['construction_rosters']['A0']['accepted_slots']=['obj_1002']
    with pytest.raises(screen.CandidateScreenError,match='partitions'):
        screen._qualify_task_suites(scene_id='09c1414f1b',task_bundle=broken,
            factories={p:Path(bundle['factories'][p]) for p in screen.POLICIES},menagerie_root=root/'unused')


def test_automatic_cpu_failure_replay_rejects_fabricated_reset_and_keeps_denominator(automatic_bundle_inputs):
    root,kwargs,_=automatic_bundle_inputs
    freeze.freeze_task_bundle(**kwargs)
    bundle=freeze.validate_task_bundle(root/'task-bundle/manifest.json',expected_scene_id='09c1414f1b',repository_root=root)
    rows=screen._qualify_task_suites(scene_id='09c1414f1b',task_bundle=bundle,
        factories={p:Path(bundle['factories'][p]) for p in screen.POLICIES},menagerie_root=root/'unused')
    suite=json.loads(Path(bundle['variant_tasks']['A0']).read_text())
    contracts={task['task_id']:{'target':task['target'],'construction_failures_by_policy':{
        p:screen._automatic_construction_failures(task,bundle,p) for p in screen.POLICIES}}
        for task in suite['tasks']}
    replay=screen._replay_qualifier_metrics(rows,scene_id='09c1414f1b',expected_task_contracts=contracts)
    assert replay['cell_count']==90 and replay['exact_900_step_cells']==0
    assert replay['strict_pass_task_count']==0
    assert all(t['maximum_room_drift_m'] is None for t in replay['task_summaries'])
    for mutation in ('telemetry','owner','seed','drop'):
        changed=copy.deepcopy(rows)
        if mutation=='telemetry':changed[0]['reset_jitter']={'applied':True}
        elif mutation=='owner':changed[0]['construction_failures'][0]['terminal_action']='accept'
        elif mutation=='seed':changed[0]['reset_seed']+=1
        else:changed.pop()
        with pytest.raises(screen.CandidateScreenError):
            screen._replay_qualifier_metrics(changed,scene_id='09c1414f1b',expected_task_contracts=contracts)


def test_task_requirement_failures_remain_distinct_from_missing_construction():
    import numpy as np
    task={'target':'obj_1002','receptacle':None}
    roster={'object_slots':['obj_1002'],'accepted_slots':['obj_1002'],'rejected_slots':[],'abstained_slots':[]}
    bundle={'automatic_population':{},'construction_rosters':{p:roster for p in screen.POLICIES}}
    row=dict(name='obj_1002',label='book',construction_eligible=True,tier=None,
             dims=np.array([.2,.1,.05]),mass=.1,drift=.039213)
    observed={p:{'obj_1002':row} for p in screen.POLICIES}
    assert screen._automatic_construction_failures(task,bundle,'A0')==[]
    assert screen._automatic_task_requirement_failures(task,bundle,'A0',observed)==['export_target_drift_not_below_30mm']
    broken=copy.deepcopy(observed);broken['A0']={}
    with pytest.raises(screen.CandidateScreenError,match='missing its validated'):
        screen._automatic_task_requirement_failures(task,bundle,'A0',broken)
