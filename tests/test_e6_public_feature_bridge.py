from copy import deepcopy
import json
from pathlib import Path
import pytest
from robo.certification import public_feature_bridge as b
from robo.certification.task_graph import build_graph
from robo.eval import build_task_support_dataset as producer


def fixture():
    query=dict(scene_id='behavior_task0020',task_id='q1',task_family='place_on_surface_region',query_sha256='a'*64,
        roles=dict(manipulated_object=dict(status='resolved',selected_object_id='chosen',candidates=[dict(object_id='chosen',overlap_iou=.3)]),
                   target=dict(status='observed_region',region_id='q1:target',selected_object_id=None,metric_surface=dict(aabb=[[0,0,0],[1,1,0]],physical_volume_known=False))))
    gate=dict(scene_id=query['scene_id'],condition_id='mild',source_commit=b.PRODUCER_SHA,
        original_gate=dict(queries=[query],robot_frame=None,physics_verified=False,feature_rows_written=0,paper_ready=False,
          stage_status='PASS',discovered_objects=[dict(id='chosen',label='cup',aabb=[[0,0,0],[.1,.1,.1]]),
                                                 dict(id='distractor',label='cup',aabb=[[.2,0,0],[.3,.1,.1]])]))
    ref=dict(producer=dict(path='original',commit=b.PRODUCER_SHA),config=dict(path='config',sha256='c'*64,size_bytes=1),
             gate=dict(path='gate',sha256='d'*64,size_bytes=1))
    return gate,ref


def test_exact_public_selected_id_has_no_language_reranking():
    gate,ref=fixture();bundle=b.bundle_from_gate(gate,ref,freeze_id='freeze',task_id='q1')
    producer._public_payload(bundle)
    bundle['task']['language']={'instruction':'move a different cup'}
    graph,features=producer._construction_features(bundle)
    hypotheses=graph['role_resolutions']['manipulated_object']['hypotheses']
    assert [h['object_id'] for h in hypotheses]==['chosen']
    assert hypotheses[0]['confidence']==.3
    assert features['robot_frame_missing']==features['geometry_policy_cameras_missing']==1
    assert graph['role_resolutions']['target']['status']=='unresolved'
    assert graph['role_resolutions']['support']['hypotheses']==[]
    assert not graph['edges']


@pytest.mark.parametrize('status',['unresolved','ambiguous'])
def test_missing_public_role_cannot_be_recovered_by_category(status):
    gate,ref=fixture();q=gate['original_gate']['queries'][0]
    q['roles']['manipulated_object'].update(status=status,selected_object_id=None,reason='public_evidence_insufficient')
    bundle=b.bundle_from_gate(gate,ref,freeze_id='freeze',task_id='q1')
    graph,features=producer._construction_features(bundle)
    assert graph['role_resolutions']['manipulated_object']['hypotheses']==[]
    assert features['geometry_required_role_missing']==1


def test_failed_constructor_retains_null_geometry_and_denominator():
    gate,ref=fixture();gate['original_gate'].update(stage_status='FAIL',discovered_objects=None)
    gate['original_gate']['queries'][0]['roles']['manipulated_object'].update(status='unresolved',selected_object_id=None)
    bundle=b.bundle_from_gate(gate,ref,freeze_id='freeze',task_id='q1')
    assert bundle['build_status']=='failed' and bundle['scene']['objects'] is None
    _,f=producer._construction_features(bundle)
    assert f['scene_build_failed']==f['robot_frame_missing']==1


@pytest.mark.parametrize('change',['object','query','role','camera','robot','physics','region','rubric'])
def test_authenticated_bundle_rejects_invented_or_changed_evidence(monkeypatch,change):
    gate,ref=fixture();bundle=b.bundle_from_gate(gate,ref,freeze_id='freeze',task_id='q1')
    monkeypatch.setattr(b,'validate_reference',lambda reference,cache=None:deepcopy(gate))
    b.authenticate_bundle(bundle)
    if change=='object':bundle['task']['construction_role_bindings']['manipulated_object']['object_id']='distractor'
    elif change=='query':bundle['task']['query_sha256']='b'*64
    elif change=='role':bundle['task']['unresolved_roles'].pop('target')
    elif change=='camera':bundle['scene']['cameras'][0]['pos']=[0,0,0]
    elif change=='robot':bundle['scene']['robot']={'base_pos':[0,0,0]}
    elif change=='physics':bundle['physics_verified']=True
    elif change=='region':bundle['task']['public_role_evidence']['target']['metric_surface']['physical_volume_known']=True
    else:bundle['task']['rubric']={'role_refs':{'manipulated_object':'distractor'}}
    with pytest.raises(ValueError,match='exact authenticated'):b.authenticate_bundle(bundle)


def test_bindings_without_authenticated_source_rejected():
    gate,ref=fixture();bundle=b.bundle_from_gate(gate,ref,freeze_id='freeze',task_id='q1');bundle.pop('public_grounding')
    with pytest.raises(ValueError,match='authenticated'):b.authenticate_bundle(bundle)


def test_graph_rejects_role_and_source_alias_conflicts():
    gate,ref=fixture();bundle=b.bundle_from_gate(gate,ref,freeze_id='freeze',task_id='q1')
    bundle['task']['rubric']['role_refs']={'manipulated_object':'chosen'}
    with pytest.raises(ValueError,match='aliases'):build_graph(bundle['scene'],bundle['task'])
    bundle['task']['rubric']={};bundle['task']['unresolved_roles']['manipulated_object']='missing'
    with pytest.raises(ValueError,match='contradictory'):build_graph(bundle['scene'],bundle['task'])


def test_old_graph_without_public_binding_remains_usable():
    scene={'scene_id':'s','objects':[],'robot':None,'cameras':[]}
    task={'task_id':'t','roles':['manipulated_object'],'unresolved_roles':{'manipulated_object':'missing'}}
    assert build_graph(scene,task)['role_resolutions']['manipulated_object']['hypotheses']==[]


def test_reference_requires_exact_producer_and_hash(monkeypatch,tmp_path):
    gate,ref=fixture();root=tmp_path/'evidence';root.mkdir();code=tmp_path/'source';code.mkdir()
    monkeypatch.setattr(b.shared,'ROOT',root)
    ref['producer']['path']=str(code)
    monkeypatch.setattr(b.shared,'git_snapshot',lambda p:dict(commit='wrong',dirty=False))
    with pytest.raises(ValueError,match='source differs'):b.validate_reference(ref)
    monkeypatch.setattr(b.shared,'git_snapshot',lambda p:dict(commit=b.PRODUCER_SHA,dirty=False,branch='irrelevant'))
    path=root/'config';path.write_text('{}');ref['config']=b.shared.identity(path);path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='bytes changed'):b.validate_reference(ref)


def test_public_firewall_rejects_gt_and_benchmark_aliases():
    for value in ({'gt_pose':[0,0,0]}, {'rubric':{'role_refs':{'target':'oracle'}}}):
        with pytest.raises(ValueError,match='forbidden'):producer._public_payload(value)


def test_complete72_bridge_uses_canonical_feature_producer(monkeypatch,tmp_path):
    import csv
    monkeypatch.delenv('SIMANY_EVIDENCE_ROOT',raising=False)
    monkeypatch.setattr(producer,'ROOT',tmp_path)
    monkeypatch.setattr(b.shared,'git_snapshot',lambda path:dict(commit='source',dirty=False))
    original,base=fixture();gates={};refs=[]
    for index,(scene,condition) in enumerate((s,c) for s in b.shared.SCENES for c in b.shared.CONDITIONS):
        gate=deepcopy(original);gate['scene_id']=scene;gate['condition_id']=condition
        queries=[]
        for i in range(4):
            q=deepcopy(original['original_gate']['queries'][0]);q.update(scene_id=scene,task_id=f'q{i}',query_sha256=str(i)*64)
            q['roles']['target']['region_id']=f'q{i}:target';queries.append(q)
        gate['original_gate']['queries']=queries
        if index<4:
            gate['original_gate'].update(stage_status='FAIL',discovered_objects=None)
            for q in queries:q['roles']['manipulated_object'].update(status='unresolved',selected_object_id=None)
        ref=deepcopy(base);ref['gate']['path']=str(index);refs.append(ref);gates[str(index)]=gate
    monkeypatch.setattr(b,'validate_reference',lambda reference,cache=None:deepcopy(gates[reference['gate']['path']]))
    monkeypatch.setattr(b,'validate_references',lambda references,**kwargs:{})
    cfg=dict(schema_version=1,source_commit='source',freeze_id='20260906-abcdef0-v1',tier='full',label_protocol_sha256='e'*64,references=refs)
    config=tmp_path/'bridge.json';config.write_text(json.dumps(cfg))
    result=b.prepare_inputs(config,tmp_path/'bundles');assert result['planned_queries']==72
    output=producer.generate_features(tmp_path/'bundles/features_config.json',tmp_path/'features')
    assert output['query_rows']==72
    rows=list(csv.DictReader((tmp_path/'features/features_unlabeled.csv').open()))
    assert sum(float(r['scene_build_failed']) for r in rows)==16
    assert all(float(r['robot_frame_missing'])==float(r['geometry_policy_cameras_missing'])==1 for r in rows)
    assert len({r['task_id'] for r in rows})==4
    seal=json.loads((tmp_path/'features/feature_seal.json').read_text())
    assert seal['feature_generation_read_labels'] is False
    with pytest.raises(FileExistsError):b.prepare_inputs(config,tmp_path/'bundles')
    cfg['references']=refs[:-1];config.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='complete fixed'):b.prepare_inputs(config,tmp_path/'missing')


def batch_fixture(monkeypatch,tmp_path):
    from types import SimpleNamespace
    rgb=tmp_path/'rgb.json';rgb.write_text(json.dumps({'python':'pinned-python'}))
    measurement=tmp_path/'measurement.json';measurement.write_text(json.dumps({'rgb_runtime':{'path':str(rgb)}}))
    config=tmp_path/'execution.json';config.write_text(json.dumps({'fresh_config':{'path':str(measurement)}}))
    refs=[];gates=[]
    for c in b.shared.CONDITIONS:
        refs.append({'producer':{'path':'code','commit':b.PRODUCER_SHA},'config':{'path':str(config),'sha256':'f'*64},'gate':{'path':c,'sha256':c}})
        gates.append(dict(scene_id='behavior_task0020',condition_id=c,marker=c))
    def context(ref):
        g=next(g for g in gates if g['condition_id']==ref['gate']['path'])
        return tmp_path,config,tmp_path,g,b.shared.canonical_hash(ref)
    monkeypatch.setattr(b,'_reference_context',context)
    monkeypatch.setattr(b.shared,'environment',lambda *args:{})
    calls=[]
    def run(command,**kwargs):
        calls.append(command);return SimpleNamespace(returncode=0,stdout=json.dumps(gates),stderr='')
    monkeypatch.setattr(b.subprocess,'run',run)
    return refs,gates,calls


def test_batch_validates_context_once_and_every_unit_each_invocation(monkeypatch,tmp_path):
    refs,gates,calls=batch_fixture(monkeypatch,tmp_path)
    cache=b.validate_references(refs,tier='pilot')
    assert len(cache)==3 and len(calls)==1
    program=calls[0][-1]
    assert program.count('d.validate(')==1 and 'd._validate_output' in program
    assert all(repr(('behavior_task0020',c)) in program for c in b.shared.CONDITIONS)
    b.validate_references(refs,tier='pilot');assert len(calls)==2
    changed=deepcopy(refs);changed[0]['gate']['sha256']='new'
    assert b.shared.canonical_hash(changed[0]) not in cache


@pytest.mark.parametrize('change',['omit','reorder','add','mix_source','mix_config'])
def test_batch_rejects_roster_and_identity_drift(monkeypatch,tmp_path,change):
    refs,gates,calls=batch_fixture(monkeypatch,tmp_path)
    if change=='omit':refs=refs[:-1]
    elif change=='reorder':refs=list(reversed(refs))
    elif change=='add':refs.append(refs[0])
    elif change=='mix_source':refs[1]['producer']['commit']='wrong'
    else:refs[1]['config']['sha256']='changed'
    with pytest.raises(ValueError,match='cannot'):b.validate_references(refs,tier='pilot')
    assert not calls


def test_batch_rejects_changed_original_result(monkeypatch,tmp_path):
    from types import SimpleNamespace
    refs,gates,calls=batch_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(b.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=0,stdout=json.dumps(gates[::-1]),stderr=''))
    with pytest.raises(ValueError,match='batch validation differs'):b.validate_references(refs,tier='pilot')
