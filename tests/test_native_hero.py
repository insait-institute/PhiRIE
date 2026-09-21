import copy
import pytest
from interface.demo_native_hero import choose_repair,retry_evidence,repair_playback,result_note,repair_slot
from interface.demo_native_progress import native_tier


@pytest.fixture
def retry_pool(tmp_path):
    import json
    from agents.orchestrator.artifact import sha256_file
    root=tmp_path/'selection';root.mkdir()
    def write(path,value):path.write_text(json.dumps(value));return sha256_file(path)
    before=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
    after=copy.deepcopy(before);after[0][3]=.1
    def asset(name,transform):
        directory=root/name;directory.mkdir()
        for filename in ('mesh_sim.ply','mesh_sim.obj','physics.json'):(directory/filename).write_text(filename)
        write(directory/'aligned.json',dict(T=transform,scale=1.))
        return dict(object_dir='/output/selection/'+name,artifact_hashes={p.name:sha256_file(p) for p in directory.iterdir()})
    parent=asset('selected_00',before);retry=asset('selected_01',after)
    evidence=dict(symmetric_clipped_registration_residual_m=.01)
    pool=dict(schema_version=2,source_commit='a'*40,
      initial_candidates=[dict(proposal_id='trellis',tool='trellis'),dict(proposal_id='rvg',tool='reconviagen')],
      retry_candidate=dict(proposal_id='retry',tool='registration_retry',parent_proposal_ids=['rvg'],evidence=evidence),
      outcomes=[dict(native_method='B0_FIXED_NATIVE',selected_proposal_id='trellis',artifact_hashes={'aligned.json':'unrelated-B0'}),
        dict(parent,native_method='B1_FIXED_PRIORITY',selected_proposal_id='rvg'),
        dict(retry,native_method='B3_AGENT_NATIVE',retry_invoked=True,selected_proposal_id='retry')])
    action=root/'registration_retry';action.mkdir()
    write(action/'registration.json',dict(T=after,scale=1.,source_up_hypothesis='-y',**evidence))
    write(action/'evidence.json',dict(producer='agents.orchestrator.runtime.align_and_probe',producer_commit='a'*40,
        raw_values=evidence,input_hashes={'mesh':parent['artifact_hashes']['mesh_sim.ply']}))
    path=root/'candidate_pool.json';write(path,pool)
    return pool,path,write


def test_retry_binds_declared_rvg_parent_not_b0(retry_pool):
    pool,path,_=retry_pool;used={}
    assert retry_evidence(pool,path,source_files=used)[0]['proposal_id']=='rvg'
    assert str(path.parent/'registration_retry/registration.json') in used
    assert str(path.parent/'selected_01/aligned.json') in used


@pytest.mark.parametrize('defect,match',[
    ('unchanged_transform','unchanged parent'),('selected_transform','does not implement'),
    ('changed_artifact','artifact changed'),('wrong_mesh','declared parent'),
    ('wrong_evidence','action/evidence'),('wrong_producer','action/evidence'),
    ('wrong_scale','does not implement'),('empty_parent','exactly one'),
    ('duplicate_id','new retry'),('wrong_selection','did not select'),
    ('foreign_directory','outside its pool'),('pool_source','source file')])
def test_retry_rejects_false_action_or_selected_artifact(retry_pool,defect,match):
    import json
    pool,path,write=retry_pool;root=path.parent;reg=root/'registration_retry/registration.json';ev=root/'registration_retry/evidence.json'
    if defect=='unchanged_transform':
        # Different serialized bytes/new proposal ID and B0 hash are insufficient.
        before=json.loads((root/'selected_00/aligned.json').read_text())
        r=json.loads(reg.read_text());r['T']=before['T'];write(reg,r)
        before['unrelated_metadata']='new bytes'
        pool['outcomes'][2]['artifact_hashes']['aligned.json']=write(root/'selected_01/aligned.json',before)
    elif defect=='selected_transform':
        r=json.loads((root/'selected_01/aligned.json').read_text());r['T'][0][3]=.2
        pool['outcomes'][2]['artifact_hashes']['aligned.json']=write(root/'selected_01/aligned.json',r)
    elif defect=='changed_artifact':(root/'selected_01/mesh_sim.obj').write_text('changed')
    elif defect in ('wrong_mesh','wrong_evidence','wrong_producer'):
        e=json.loads(ev.read_text())
        if defect=='wrong_mesh':e['input_hashes']['mesh']='wrong'
        elif defect=='wrong_evidence':e['raw_values']={}
        else:e['producer_commit']='b'*40
        write(ev,e)
    elif defect=='wrong_scale':
        r=json.loads(reg.read_text());r['scale']=2.;write(reg,r)
    elif defect=='empty_parent':pool['retry_candidate']['parent_proposal_ids']=[]
    elif defect=='duplicate_id':pool['retry_candidate']['proposal_id']='rvg'
    elif defect=='wrong_selection':pool['outcomes'][2]['selected_proposal_id']='rvg'
    elif defect=='foreign_directory':pool['outcomes'][2]['object_dir']='/unrelated/selection/selected_01'
    elif defect=='pool_source':pool['source_commit']='changed'
    if defect!='pool_source':write(path,pool)
    with pytest.raises(ValueError,match=match):retry_evidence(pool,path)


def test_retry_action_sidecars_must_enter_release_closure(retry_pool):
    from agents.orchestrator.artifact import sha256_file
    pool,path,_=retry_pool;root=path.parent
    sources={str(root/'registration_retry'/name):sha256_file(root/'registration_retry'/name)
             for name in ('registration.json','evidence.json')}
    def checked(p,digest=None):
        expected=digest or sources.get(str(p))
        if expected is None or sha256_file(p)!=expected:raise ValueError('unbound action sidecar')
    assert retry_evidence(pool,path,checked=checked)[1]['proposal_id']=='retry'
    del sources[str(root/'registration_retry/registration.json')]
    with pytest.raises(ValueError,match='unbound action sidecar'):retry_evidence(pool,path,checked=checked)


def test_repair_selection_ignores_policy_outcomes():
    planned=[{'canonical_instance_id':i} for i in ['a','b']]
    rows=[dict(canonical_instance_id=i,method='B4_ROOM_REPAIR_NATIVE',accepted=True,actual_calls=1,native_policy_success=s) for i,s in [('b',True),('a',False)]]
    assert choose_repair(planned,rows)['canonical_instance_id']=='a'
    rows[1]['native_policy_success']=True;rows[0]['native_policy_success']=False
    assert choose_repair(planned,rows)['canonical_instance_id']=='a'
    rows[1]['actual_calls']=0
    assert choose_repair(planned,rows)['canonical_instance_id']=='b'


def test_native_split_and_whole_failed_repair_playback():
    assert native_tier({'split':'test'}) == 'TEST'
    assert native_tier({'split':'development'}) == 'DEV'
    with pytest.raises(ValueError,match='declared'):
        native_tier({'split':'best_successes'})
    # A full failed900-action episode fits without discarding its final actions.
    assert repair_playback({'split':'test'},{'ticks':900,'success':False}) == (2,22.5)
    assert repair_playback({'split':'test'},{'ticks':900,'success':True}) == (2,22.5)
    assert repair_playback({'split':'dev'},{'ticks':400}) == (1,20)
    with pytest.raises(ValueError,match='never truncate'):
        repair_playback({'split':'test'},{'ticks':1000})


def test_test_result_caption_does_not_recycle_development_negative():
    rows=[dict(method='B3_AGENT_NATIVE',executed=300,planned=480,successes_observed=90),
          dict(method='B4_ROOM_REPAIR_NATIVE',executed=120,planned=480,successes_observed=80)]
    assert 'loses coverage' in result_note(rows,'TEST')
    rows[1]['successes_observed']=100
    assert 'loses coverage' not in result_note(rows,'TEST')
    assert 'does not establish preservation' in result_note(rows,'TEST')


def test_incomplete_test_cannot_become_complete_hero(monkeypatch,tmp_path):
    import interface.demo_native_hero as hero
    monkeypatch.setattr(hero,'resolve_pair',lambda _:({'state':'INCOMPLETE'},{'split':'test'},{}))
    out=tmp_path/'demo'
    with pytest.raises(ValueError,match='complete native release'):
        hero.render(tmp_path,out)
    assert not out.exists()


def test_repair_slot_resolves_only_frozen_canonical_identity():
    rows=[dict(canonical_instance_id='canonical-a',instance_slot_id='slot-a')]
    repair=dict(canonical_instance_id='canonical-a')
    assert repair_slot(rows,repair)=='slot-a'
    with pytest.raises(ValueError,match='changes canonical'):
        repair_slot(rows,dict(repair,instance_slot_id='slot-b'))
    with pytest.raises(ValueError,match='unique'):
        repair_slot(rows+[dict(canonical_instance_id='canonical-a',instance_slot_id='slot-b')],repair)
