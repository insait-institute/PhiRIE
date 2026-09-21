"""Terminal applicability is authenticated coverage, never invented rollout data."""
import copy
import json
import pytest
from run.icra2027 import e4_planning_terminal as terminal


@pytest.fixture
def observed(monkeypatch):
    from run.icra2027 import e4_compact_harness as compact
    monkeypatch.setattr(compact,'checked_protocol',lambda *a: {})
    result={'protocol':{'path':'protocol.yaml','sha256':'a'*64},'scenes':{
        '27dd4da69e':{'semantic_pairs':18,'planning_status':'FAIL','reason':terminal.REASON,
                      'task_definition':None,'reset_bank':None,'camera':None,'rollout':None,'qualifier':None},
        '40aec5fffa':{'semantic_pairs':10,'planning_status':'PASS','reason':None,
                      'task_definition':None,'reset_bank':None,'camera':None,'rollout':None,
                      'qualifier':{'planned_cells':100,'passed_cells':0,'fixed_selected_cells':20,'fixed_passed_cells':0}}}}
    monkeypatch.setattr(terminal,'_replay',lambda source:copy.deepcopy(result))
    return result


def test_applicability_failure_is_not_execution_failure(observed):
    r=terminal._payload({}, {'code_root':'root','commit':'a'*40,'dirty':False})
    assert r['status']=='PASS' and r['applicability_status']=='FAIL'
    assert r['execution_status']=='NOT_RUN' and r['claim_gate']=='FAIL'
    assert r['planned_episodes']==40 and r['planned_semantic_pairs']==28
    assert not r['paper_ready'] and not r['headline_eligible']
    for s in r['scenes'].values():
        assert all(s[k] is None for k in ('task_definition','reset_bank','camera','rollout'))
    assert r['scenes']['40aec5fffa']['planning_status']=='PASS'


@pytest.mark.parametrize('change', ['drop_scene','drop_pair','no_failure'])
def test_no_terminal_label_without_full_negative_population(observed,change):
    if change=='drop_scene':observed['scenes'].pop('40aec5fffa')
    elif change=='drop_pair':observed['scenes']['27dd4da69e']['semantic_pairs']=17
    else:observed['scenes']['27dd4da69e']['planning_status']='PASS'
    with pytest.raises(ValueError):terminal._payload({}, {})


@pytest.mark.parametrize('field,value',[
    ('execution_status','COMPLETE'),('planned_episodes',20),('headline_eligible',True),
    ('applicability_status','PASS'),('claim_gate','PASS'),('paper_ready',True)])
def test_receipt_exact_replay_rejects_status_and_denominator_tamper(tmp_path,monkeypatch,observed,field,value):
    code={'code_root':'original','commit':'a'*40,'dirty':False}
    monkeypatch.setattr(terminal,'_exact_code',lambda root,expected:code)
    r=terminal._payload({},code);r[field]=value
    p=tmp_path/'receipt.json';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='exact source replay'):
        terminal.validate_terminal_receipt(p,expected_producer_commit='a'*40)


def test_receipt_source_replay_preserves_real_qualifier(tmp_path,monkeypatch,observed):
    code={'code_root':'original','commit':'a'*40,'dirty':False}
    monkeypatch.setattr(terminal,'_exact_code',lambda root,expected:code)
    r=terminal._payload({},code);p=tmp_path/'receipt.json';p.write_text(json.dumps(r))
    assert terminal.validate_terminal_receipt(p,expected_producer_commit='a'*40)==r
    r['scenes']['40aec5fffa']['qualifier']['fixed_passed_cells']=20
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):terminal.validate_terminal_receipt(p,expected_producer_commit='a'*40)



def test_original_validator_failure_is_not_reclassified(tmp_path,monkeypatch):
    import subprocess
    source={'config':{'path':'config'},'e0':{'path':str(tmp_path/'contract/freeze_manifest.json')},
            'code_commit':'a'*40,'runtime':{'path':'python'},'code_root':str(tmp_path)}
    monkeypatch.setattr(terminal,'source_contract',lambda *args:source)
    monkeypatch.setattr(terminal.screen,'evidence_root',lambda:tmp_path)
    def fail(*args,**kwargs):
        assert kwargs['cwd']==str(tmp_path) and 'PYTHONPATH' not in kwargs['env']
        raise subprocess.CalledProcessError(1,['original-validator'],stderr='unrelated source failure')
    monkeypatch.setattr(terminal.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):terminal._replay(source)
