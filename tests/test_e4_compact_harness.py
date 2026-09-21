import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

MODULE = Path(__file__).resolve().parents[1]/'run/icra2027/e4_compact_harness.py'
spec = importlib.util.spec_from_file_location('compact_harness', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def protocol(tmp_path):
    tasks=[dict(scene_id=s,task_id=f'{s}__{t}',task_family=f,target='obj_1001',
                receptacle=None if f=='object_to_region' else ('obj_1000' if s==m.SCENES[0] else 'obj_1005'))
           for s,t,f in m.TASKS]
    queries=copy.deepcopy(tasks)
    queries += [dict(scene_id=m.SCENES[i%2],task_id=f'{m.SCENES[i%2]}__zzz{i}',
                     task_family='object_to_region') for i in range(24)]
    p=dict(schema_version=1,study_scope='e4_compact_canonical_engineering',paper_ready=False,
           no_substitution_after_qualification=True,no_policy_outcome_selection=True,
           no_full_cohort_replacement=True,episodes_per_task_arm=5,reset_base_seed=0,
           jitter_xy_m=.01,planned_manipulation_episodes=40,construction_arms=['A0','A4'],
           first_real_policy='pi05_droid_jointpos',planned_qualification_queries=28,
           planned_qualification_cells=280,population=dict(scene_ids=list(m.SCENES),
           planned_objects=17,planned_policy_object_rows=85),
           fixed_manipulation_tasks=tasks,selected_semantic_queries=queries)
    path=tmp_path/'protocol.yaml';path.write_text(yaml.safe_dump(p))
    return path,p


def test_plan_uses_canonical_ids_without_measured_state(protocol,tmp_path):
    path,p=protocol;out=tmp_path/'handoff'
    report=m.prepare(path,m.sha(path),out)
    states=m.elog.load_reset_states(out/'planned_reset_definitions.json')
    cells=json.loads((out/'planned_episode_cells.json').read_text())
    assert len(states)==20 and len(cells)==40
    assert len({r['episode_id'] for r in cells})==40
    for s in states:
        assert s.reset_seed==m.elog.derive_reset_seed(0,s.task_id,s.ep)
        paired=[r for r in cells if r['reset_state_id']==s.reset_state_id]
        assert len(paired)==2 and paired[0]['reset_seed']==paired[1]['reset_seed']
    assert all(c['outcome'] is None and c['qualification_state']=='NOT_RUN' for c in cells)
    assert report['actual_reset_bank'] is None and report['measured_initial_poses'] is None
    assert report['policy_launch_allowed'] is False and report['harness_config'] is None
    assert not (out/'reset_states.json').exists() and not (out/'harness_ledger.jsonl').exists()
    with pytest.raises(FileExistsError):m.prepare(path,m.sha(path),out)


@pytest.mark.parametrize('mutation', ['hash','task','policy','seed','jitter','episodes','population','substitute','queries'])
def test_protocol_drift_rejected(protocol,tmp_path,mutation):
    path,p=protocol;expected=m.sha(path)
    if mutation=='hash':path.write_text(path.read_text()+'\n')
    else:
        if mutation=='task':p['fixed_manipulation_tasks'][0]['task_id']='replacement'
        if mutation=='policy':p['first_real_policy']='scripted_sinusoid'
        if mutation=='seed':p['reset_base_seed']=1
        if mutation=='jitter':p['jitter_xy_m']=.02
        if mutation=='episodes':p['episodes_per_task_arm']=4
        if mutation=='population':p['population']['planned_objects']=16
        if mutation=='substitute':p['no_substitution_after_qualification']=False
        if mutation=='queries':p['selected_semantic_queries'].pop()
        path.write_text(yaml.safe_dump(p));expected=m.sha(path)
    with pytest.raises(ValueError):m.prepare(path,expected,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def qualifier_fixture(monkeypatch,tmp_path,p,mutate=None):
    from robo.eval import e4_candidate_screen as screen
    called=[]
    monkeypatch.setattr(screen,'_load_prepare',lambda **kw:{'gate':{'automatic_population':{}}})
    monkeypatch.setattr(screen,'_code_snapshot',lambda commit: {'commit':commit})
    def validate(**kwargs):
        called.append(kwargs['scene_id']);scene=kwargs['scene_id'];directory=tmp_path/scene;directory.mkdir()
        rows=[]
        for q in p['selected_semantic_queries']:
            if q['scene_id']!=scene:continue
            for arm in m.ARMS:
                for ep in range(5):
                    rows.append(dict(scene_id=scene,task_id=q['task_id'],policy_id=arm,episode=ep,
                        reset_seed=m.elog.derive_reset_seed(0,q['task_id'],ep),
                        cell_id=f"{arm.lower()}__{q['task_id']}__seed0__ep{ep}",passed=False,
                        failure_type='construction_endpoint_unavailable',checks={'construction_endpoints_accepted':False},
                        reset_jitter=None))
        if mutate=='missing':rows=rows[1:]
        if mutate=='duplicate':rows.append(rows[0])
        if mutate=='source':raise ValueError('sealed source changed')
        for name in ('gate.json','manifest.json','seal.json','metrics.jsonl'):(directory/name).write_text('{}')
        return dict(bundle={'directory':directory},metric_rows=rows)
    monkeypatch.setattr(screen,'_validate_qualifier_output',validate)
    return called


def test_canonical_full_qualifier_replay_preserves_failed_selected_tasks(protocol,tmp_path,monkeypatch):
    path,p=protocol;called=qualifier_fixture(monkeypatch,tmp_path,p)
    out=tmp_path/'handoff';r=m.prepare(path,m.sha(path),out,screen_id='freeze',expected_commit='a'*40)
    assert called==list(m.SCENES) and r['qualification_replay']=='PASS'
    cells=json.loads((out/'planned_episode_cells.json').read_text())
    assert len(cells)==40 and all(c['qualification_state']=='FAIL' for c in cells)
    assert all(c['outcome'] is None and c['rollout_state']=='NOT_RUN' for c in cells)
    assert all(c['source_failure_type']=='construction_endpoint_unavailable' for c in cells)
    assert r['policy_launch_allowed'] is False


@pytest.mark.parametrize('mutation',['missing','duplicate','source'])
def test_qualifier_missing_duplicates_source_drift_fail_closed(protocol,tmp_path,monkeypatch,mutation):
    path,p=protocol;qualifier_fixture(monkeypatch,tmp_path,p,mutation)
    with pytest.raises(ValueError):
        m.prepare(path,m.sha(path),tmp_path/'bad',screen_id='freeze',expected_commit='a'*40)
    assert not (tmp_path/'bad').exists()


def test_source_commit_required_for_qualification(protocol,tmp_path):
    path,_=protocol
    with pytest.raises(ValueError):m.prepare(path,m.sha(path),tmp_path/'bad',screen_id='freeze')
