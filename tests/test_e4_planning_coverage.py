"""Typed NOT_RUN planning coverage; never synthesize a rollout/task definition."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from run.icra2027 import e4_compact_harness as compact
from run.icra2027 import e4_planning_terminal as real_terminal
from robo.eval import e4_candidate_screen as screen


@pytest.fixture
def terminal(tmp_path,monkeypatch):
    tasks=[dict(scene_id=s,task_id=f'{s}__{t}',task_family=f,target='obj_1001',
                receptacle=None if f=='object_to_region' else ('obj_1000' if s==compact.SCENES[0] else 'obj_1005'))
           for s,t,f in compact.TASKS]
    queries=copy.deepcopy(tasks)
    for scene,count in zip(compact.SCENES,(16,8),strict=True):
        queries += [dict(scene_id=scene,task_id=f'{scene}__zzz{i}',task_family='object_to_region') for i in range(count)]
    p=dict(schema_version=1,study_scope='e4_compact_canonical_engineering',paper_ready=False,
           no_substitution_after_qualification=True,no_policy_outcome_selection=True,
           no_full_cohort_replacement=True,episodes_per_task_arm=5,reset_base_seed=0,
           jitter_xy_m=.01,planned_manipulation_episodes=40,construction_arms=['A0','A4'],
           first_real_policy='pi05_droid_jointpos',planned_qualification_queries=28,
           planned_qualification_cells=280,population=dict(scene_ids=list(compact.SCENES),
           planned_objects=17,planned_policy_object_rows=85),
           fixed_manipulation_tasks=tasks,selected_semantic_queries=queries)
    protocol=tmp_path/'protocol.yaml';protocol.write_text(yaml.safe_dump(p))
    code={'code_root':str(Path(compact.__file__).resolve().parents[2]),'commit':'a'*40,'dirty':False}
    receipt=dict(schema_version=1,scope='automatic_compact_planning_applicability_audit',
        status='PASS',applicability_status='FAIL',execution_status='NOT_RUN',
        paper_ready=False,headline_eligible=False,producer_code=code,
        protocol={'path':str(protocol),'sha256':compact.sha(protocol)},
        source={'code_commit':'b'*40,'freeze_id':'original-fixture'},
        planned_semantic_pairs=28,planned_episodes=40,
        scenes={compact.SCENES[0]:{'semantic_pairs':18,'planning_status':'FAIL','reason':'no_prepared_size_admissible_target'},
                compact.SCENES[1]:{'semantic_pairs':10,'planning_status':'PASS','reason':None,
                    'qualifier':{'planned_cells':100,'passed_cells':0,'fixed_selected_cells':20,'fixed_passed_cells':0}}})
    path=tmp_path/'terminal.json';path.write_text(json.dumps(receipt))
    calls=[]
    def validate(path,*,expected_producer_commit):
        calls.append((str(path),expected_producer_commit))
        return json.loads(Path(path).read_text())
    # External terminal source validation owns historical population replay;
    # these tests exercise the actual coverage boundary and sealed output files.
    monkeypatch.setitem(sys.modules,'run.icra2027.e4_planning_terminal',SimpleNamespace(validate_terminal_receipt=validate))
    monkeypatch.setattr(screen,'_code_snapshot',lambda expected:code if expected=='a'*40 else None)
    return SimpleNamespace(root=tmp_path,protocol=protocol,p=p,path=path,receipt=receipt,calls=calls)


def prepare(f,out=None,**kw):
    return compact.prepare(f.protocol,compact.sha(f.protocol),out or f.root/'coverage',
        evidence_root=f.root,expected_commit='a'*40,
        planning_terminal_path=f.path,planning_terminal_sha256=compact.sha(f.path),**kw)


def test_preserves_all40_definitions_and_28_queries_without_outcomes(terminal):
    f=terminal; report=prepare(f)
    out=f.root/'coverage'
    checked=screen._validate_bundle(out,root=f.root,expected_kind='e4_planning_unqualified_coverage')
    cells=json.loads((out/'planned_episode_cells.json').read_text())
    states=compact.elog.load_reset_states(out/'planned_reset_definitions.json')
    assert len(cells)==40 and len(states)==20
    assert len({c['episode_id'] for c in cells})==40
    assert report['semantic_pairs_by_scene']==dict(zip(compact.SCENES,(18,10),strict=True))
    assert report['planned_semantic_pairs']==28 and report['planned_qualification_cells']==280
    assert report['source_qualification_summaries'][compact.SCENES[1]]==f.receipt['scenes'][compact.SCENES[1]]['qualifier']
    assert report['source_qualification_summaries'][compact.SCENES[0]] is None
    assert len(f.calls)==2  # Receipt is reauthenticated before atomic publication.
    for state in states:
        pair=[c for c in cells if c['reset_state_id']==state.reset_state_id]
        assert len(pair)==2 and {c['construction_policy'] for c in pair}=={'A0','A4'}
        assert all(c['reset_seed']==compact.elog.derive_reset_seed(0,state.task_id,state.ep) for c in pair)
    for c in cells:
        assert c['execution_status']==c['rollout_state']=='NOT_RUN'
        assert c['policy_execution']=='not_invoked_prebuild'
        for field in ('task_definition','reset_provenance','camera_diagnostics','outcome','success','score',
                      'grasp','lift','place','ticks','policy_latency_ms','observation_latency_ms'):
            assert c[field] is None
        if c['scene_id']==compact.SCENES[0]:
            assert c['task_applicability_status']=='FAIL' and c['source_failure_type']=='planning_unqualified'
        else:
            assert c['scene_planning_status']=='PASS' and c['task_applicability_status']=='NOT_RUN'
            assert c['source_failure_type']=='paired_matrix_prerequisite_unavailable'
    assert report['claim_gate']=='NOT_RUN' and report['applicability_gate']=='FAIL'
    assert report['rollout_ledger'] is report['harness_config'] is report['actual_reset_bank'] is None
    assert report['paper_ready'] is report['policy_launch_allowed'] is False
    assert not (out/'rollouts.jsonl').exists() and not (out/'reset_states.json').exists()
    assert checked['manifest']['paper_ready'] is False


@pytest.mark.parametrize('mutation',['claim','execution','protocol','source','count','scene_count','missing_scene','reason','no_failure','boolean_version'])
def test_changed_or_promoted_terminal_receipt_is_rejected(terminal,mutation):
    f=terminal;r=copy.deepcopy(f.receipt)
    if mutation=='claim':r['paper_ready']=True
    elif mutation=='execution':r['execution_status']='PASS'
    elif mutation=='protocol':r['protocol']['sha256']='0'*64
    elif mutation=='source':r['producer_code']['commit']='c'*40
    elif mutation=='count':r['planned_episodes']=39
    elif mutation=='scene_count':r['scenes'][compact.SCENES[0]]['semantic_pairs']=17
    elif mutation=='missing_scene':r['scenes'].pop(compact.SCENES[1])
    elif mutation=='reason':r['scenes'][compact.SCENES[0]]['reason']='new_threshold_after_results'
    elif mutation=='no_failure':r['scenes'][compact.SCENES[0]].update(planning_status='PASS',reason=None)
    else:r['schema_version']=True
    f.path.write_text(json.dumps(r))
    with pytest.raises(ValueError):prepare(f)
    assert not (f.root/'coverage').exists()


def test_changed_receipt_hash_and_mixed_qualifier_mode_are_rejected(terminal):
    f=terminal; old=compact.sha(f.path);f.path.write_text(f.path.read_text()+'\n')
    with pytest.raises(ValueError,match='hash differs'):
        compact.prepare(f.protocol,compact.sha(f.protocol),f.root/'coverage',evidence_root=f.root,
            expected_commit='a'*40,planning_terminal_path=f.path,planning_terminal_sha256=old)
    with pytest.raises(ValueError,match='cannot mix'):
        prepare(f,screen_id='old-screen')


def test_published_coverage_is_immutable_and_tampering_is_detected(terminal):
    f=terminal;prepare(f);out=f.root/'coverage'
    before=(out/'planned_episode_cells.json').read_bytes()
    with pytest.raises(FileExistsError):prepare(f)
    assert (out/'planned_episode_cells.json').read_bytes()==before
    cells=json.loads(before);cells[0]['success']=True
    (out/'planned_episode_cells.json').write_text(json.dumps(cells))
    with pytest.raises(screen.CandidateScreenError,match='member changed'):
        screen._validate_bundle(out,root=f.root,expected_kind='e4_planning_unqualified_coverage')


def test_output_cannot_escape_evidence_root(terminal):
    f=terminal
    with pytest.raises(screen.sealed_cpu.PilotGateError,match='escapes'):
        prepare(f,out=f.root.parent/'outside-evidence-coverage')
    link=f.root/'symlink';link.symlink_to(f.root,target_is_directory=True)
    with pytest.raises(screen.sealed_cpu.PilotGateError,match='symlink'):
        prepare(f,out=link/'coverage')


def test_real_terminal_validation_boundary_preserves_qualifier_and_rejects_tamper(terminal,monkeypatch):
    f=terminal;code=f.receipt['producer_code']
    observed={'protocol':f.receipt['protocol'],'scenes':f.receipt['scenes']}
    monkeypatch.setattr(real_terminal,'_replay',lambda source:copy.deepcopy(observed))
    monkeypatch.setattr(real_terminal,'_exact_code',lambda root,expected:code)
    monkeypatch.setitem(sys.modules,'run.icra2027.e4_planning_terminal',real_terminal)
    receipt=real_terminal._payload(f.receipt['source'],code)
    f.path.write_text(json.dumps(receipt))
    report=prepare(f)
    assert report['source_qualification_summaries'][compact.SCENES[1]]['fixed_passed_cells']==0
    altered=copy.deepcopy(receipt)
    altered['scenes'][compact.SCENES[1]]['qualifier']['fixed_passed_cells']=20
    f.path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='exact source replay'):
        prepare(f,out=f.root/'tampered-coverage')
