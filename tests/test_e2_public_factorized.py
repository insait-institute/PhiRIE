import copy
import json
from pathlib import Path

import pytest

from run.icra2027 import e2_public_factorized as pilot,e2_raw_room as raw
from tests.test_e2_raw_room import fixture_plan,common_mock


def original(tmp_path,scene):
    tmp_path.mkdir(parents=True,exist_ok=True)
    p=fixture_plan(tmp_path);p.update(scene_id=scene,camera_bank={'sha256':'frozen'},original_camera_plan={'sha256':'frozen'},
        source_gaussian_training_provenance='FRESH_OFFICIAL_TRAIN_ONLY')
    return dict(plan=p)


def test_compact_pair_manifest_keeps_all_missing_scene_views(tmp_path,monkeypatch):
    from robo.eval import fidelity_metrics as metric
    c=dict(freeze_id='new',metric_source={'code_commit':'a'*40});contract={'code':{'commit':'b'*40}}
    results={}
    for scene in pilot.SCENES:
        r=original(tmp_path/scene,scene);p=r['plan'];common,_=common_mock(p)
        rawviews=raw.export_raw(p,tmp_path/scene/'raw',common)
        composite=(raw.export_raw(p,tmp_path/scene/'auto',common,gaussian=object(),method_slug='factorized_auto_discovery')
            if scene==pilot.SCENES[0] else [])
        results[scene]=dict(raw_source=dict(r,bundle_identity={'sha256':'c'*64}),source_fill={'sha256':'d'*64},
            raw_views=rawviews,composite_views=composite,status='COMPLETE' if composite else 'BLOCKED_UNFILLABLE_ACCEPTED',
            accepted_objects=1 if composite else 4)
    manifest,coverage=pilot.make_manifest(c,contract,results)
    assert len(coverage)==4 and sum(x['planned_views'] for x in coverage if x['method']==pilot.AUTO)==16
    auto=[r for r in coverage if r['method']==pilot.AUTO]
    assert [r['available_views'] for r in auto]==[8,0] and len(auto[1]['missing_views'])==8
    assert sum(r['planned_objects'] for r in auto)==17
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    class FakeLPIPS:
        error=None;provenance={}
        def __init__(self,*a,**kw):pass
        def __call__(self,*a,**kw):return .2
    monkeypatch.setattr(metric,'LPIPSEvaluator',FakeLPIPS);monkeypatch.setattr(metric,'REPOSITORY_ROOT',tmp_path)
    table=metric.evaluate_manifest(path,tmp_path/'metrics',bootstrap_samples=10)
    counts={r['method']:r['n_images'] for r in table['rows'] if r['method'] in (raw.METHOD,pilot.AUTO)}
    assert counts[raw.METHOD]==16 and counts[pilot.AUTO]==8 and table['paper_ready'] is False
    results.pop(pilot.SCENES[1])
    with pytest.raises(ValueError,match='denominator'):pilot.make_manifest(c,contract,results)


@pytest.mark.parametrize('kind',['scene','population','gaussian','TRAIN','TEST','camera'])
def test_scoped_source_boundary_cannot_drift(tmp_path,kind):
    scene=pilot.SCENES[0];r=original(tmp_path,scene);p=r['plan']
    fill=dict(result={'planned_objects':7},source=dict(context={'scene_id':scene},
        source_boundary={'gaussian':copy.deepcopy(p['gaussian'])},train_images={'train.jpg':{}}))
    pilot.validate_scene(fill,r,scene)
    if kind=='scene':p['scene_id']=pilot.SCENES[1]
    elif kind=='population':fill['result']['planned_objects']=6
    elif kind=='gaussian':p['gaussian']['sha256']='wrong'
    elif kind=='TRAIN':p['optimization_input_frames']=[]
    elif kind=='TEST':p['evaluation_images'][0]['frame']='train.jpg'
    else:p.pop('camera_bank')
    with pytest.raises(ValueError):pilot.validate_scene(fill,r,scene)


def test_raw_copy_uses_authenticated_exact_bytes_and_never_overwrites(tmp_path):
    p=original(tmp_path/'original',pilot.SCENES[0])['plan'];common,_=common_mock(p)
    bundle=tmp_path/'bundle';raw.export_raw(p,bundle,common)
    record={'bundle':str(bundle),'bundle_identity':raw.identity(bundle/'manifest.json')}
    views=pilot.copy_raw(record,tmp_path/'copy')
    assert len(views)==8
    assert all(Path(v['render_path']).read_bytes()==(bundle/'input_scene_gaussian'/Path(v['render_path']).name).read_bytes() for v in views)
    with pytest.raises(FileExistsError):pilot.copy_raw(record,tmp_path/'copy')
    (bundle/'input_scene_gaussian/test0.png').write_bytes(b'altered')
    with pytest.raises(ValueError,match='copy differs'):pilot.copy_raw(record,tmp_path/'bad-copy')


@pytest.mark.parametrize('kind',['dirty','commit'])
def test_archived_source_must_still_be_exact_clean_commit(monkeypatch,tmp_path,kind):
    monkeypatch.setattr(pilot,'git_snapshot',lambda _:{'dirty':kind=='dirty','commit':'a'*40})
    with pytest.raises(ValueError,match='source changed'):
        pilot.source_check(dict(code_root=str(tmp_path),code_commit='b'*40 if kind=='commit' else 'a'*40))


def test_scene_render_refuses_previous_attempt_before_reading_images(monkeypatch,tmp_path):
    c={'fill_seals':{}};out=tmp_path/'stage';scene=pilot.SCENES[0]
    claim=out/'scenes'/f'{scene}.claim.json';claim.parent.mkdir(parents=True);claim.write_text('{}')
    monkeypatch.setattr(pilot,'context',lambda *a:(c,{},out))
    monkeypatch.setattr(pilot,'validate_public_fill',lambda _:pytest.fail('attempt reused construction'))
    with pytest.raises(FileExistsError):pilot.render('config','contract',scene)


def test_metrics_cannot_start_with_partial_scene_population(monkeypatch,tmp_path):
    c={'freeze_id':'f'};out=tmp_path/'stage'
    monkeypatch.setattr(pilot,'context',lambda *a:(c,{},out))
    def partial(*a):raise FileNotFoundError('second scene not sealed')
    monkeypatch.setattr(pilot,'scene_result',partial)
    with pytest.raises(FileNotFoundError):pilot.metrics('config','contract')
    assert not (out/'metrics').exists()


@pytest.mark.parametrize('kind',[None,'config_identity','contract_identity','view_id','original_camera'])
def test_scene_consumer_binds_config_contract_and_exact_views(tmp_path,monkeypatch,kind):
    from robo.manifest.hash import canonical_hash
    from agents.edit.inpaint_masks import _identity
    from robo.eval import agentic_ablation as e3
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    scene=pilot.SCENES[0];source=original(tmp_path/'original',scene);p=source['plan']
    directory=tmp_path/'scene';directory.mkdir();common,_=common_mock(p)
    views=raw.export_raw(p,directory/'raw',common)
    fillseal=tmp_path/'fillseal.json';fillseal.write_text('{}')
    c=dict(freeze_id='new',fill_seals={scene:_identity(fillseal)},raw_source={'frozen':'source'})
    config=tmp_path/'config.json';config.write_text(json.dumps(c))
    contract=dict(freeze_id='new',code={'commit':'a'*40,'dirty':False},resource_inventory=[
        dict(resolved_path=str(config),hash_method='content_sha256',sha256=_identity(config)['sha256'])])
    contract['contract_sha256']=canonical_hash(contract)
    cp=tmp_path/'contract.json';cp.write_text(json.dumps(contract))
    fill=dict(seal_identity=_identity(fillseal),result={'planned_objects':7},source=dict(context={'scene_id':scene},
        source_boundary={'gaussian':copy.deepcopy(p['gaussian'])},train_images={'train.jpg':{}},blocked=[{'reason':'NO_PLANE'}]))
    monkeypatch.setattr(pilot,'validate_public_fill',lambda _:fill)
    monkeypatch.setattr(pilot,'audit_raw',lambda *a:source)
    result=dict(scope=pilot.SCOPE,freeze_id='new',scene_id=scene,code_commit='a'*40,source_fill=c['fill_seals'][scene],
        planned_objects=7,planned_views=8,paper_ready=False,full_e3_gt_access=False,construction_modified=False,
        config_identity=_identity(config),contract_identity=_identity(cp),status='BLOCKED_UNFILLABLE_ACCEPTED',
        raw_source=copy.deepcopy(source),raw_views=views,composite_views=[])
    if kind in ('config_identity','contract_identity'):
        other=tmp_path/'other.json';other.write_text('{}');result[kind]=_identity(other)
    elif kind=='view_id':result['raw_views'][0]['view_id']='different.jpg'
    elif kind=='original_camera':result['raw_source']['plan']['evaluation_images'][0]['w2c'][0][3]=123
    (directory/'result.json').write_text(json.dumps(result));pilot.seal_output(directory)
    if kind is None:assert pilot.scene_result(directory,c,contract,scene)[0]['planned_views']==8
    else:
        with pytest.raises(ValueError):pilot.scene_result(directory,c,contract,scene)


def test_metric_runtime_probe_uses_consumer_when_metric_source_predates_helper(tmp_path,monkeypatch):
    import sys
    from agents.edit.inpaint_masks import _identity
    consumer=tmp_path/'consumer';metric_source=tmp_path/'old_metric'
    for root in (consumer,metric_source):
        module=root/'robo/eval/fidelity_metrics.py';module.parent.mkdir(parents=True);module.write_text('# same frozen evaluator bytes\n')
    helper=consumer/'run/icra2027/e2_raw_room.py';helper.parent.mkdir(parents=True)
    helper.write_text('import json; print(json.dumps({"origin":"consumer runtime probe"}))\n')
    assert not (metric_source/'run/icra2027/e2_raw_room.py').exists()
    weights=tmp_path/'weights';weights.write_bytes(b'pinned')
    c=dict(metric_source={'module':_identity(metric_source/'robo/eval/fidelity_metrics.py')},
        execution={'python':{'metrics':sys.executable},'torch_home':str(tmp_path),
            'lpips_backbone':_identity(weights)})
    monkeypatch.setattr(raw,'CODE',consumer)
    assert pilot.metric_runtime(c)=={'origin':'consumer runtime probe'}
    (consumer/'robo/eval/fidelity_metrics.py').write_text('changed')
    with pytest.raises(ValueError,match='implementation differ'):pilot.metric_runtime(c)


@pytest.mark.parametrize('kind',[None,'scene_seal','policy_source','missing_scene','nested_reuse'])
def test_metrics_reuse_calls_original_validator_with_original_config_and_e0(tmp_path,monkeypatch,kind):
    from agents.edit.inpaint_masks import _identity
    old=tmp_path/'old';old.mkdir();config=old/'config.json';config.write_text('{}');contract=old/'E0.json';contract.write_text('{}')
    seals={}
    for scene in pilot.SCENES:
        path=old/scene/'seal.json';path.parent.mkdir();path.write_text('{}');seals[scene]=_identity(path)
    immutable=('scope','scene_ids','planned_objects','planned_views_per_scene','seed','bootstrap_samples',
        'bootstrap_seed','paper_ready','full_e3_gt_access','fill_seals','raw_source','metric_source','execution')
    original={k:'same' for k in immutable};original['execution']={}
    c=copy.deepcopy(original);c['render_source']=dict(code_root=str(old),code_commit='a'*40,
        config=_identity(config),contract=_identity(contract),python='pinned-python',scene_seals=copy.deepcopy(seals))
    payload={'results':{s:{'original_scene':s} for s in pilot.SCENES},'seals':seals};calls=[]
    monkeypatch.setattr(pilot,'source_check',lambda _:old)
    monkeypatch.setattr(pilot,'_bound_contract',lambda *a,**kw:(original,{}))
    monkeypatch.setattr(raw,'environment',lambda *a:{})
    def execute(command,**kw):calls.append((command,kw));return json.dumps(payload)
    monkeypatch.setattr(pilot.subprocess,'check_output',execute)
    if kind=='scene_seal':payload['seals']=copy.deepcopy(seals);payload['seals'][pilot.SCENES[0]]['sha256']='other'
    elif kind=='policy_source':c['fill_seals']='changed'
    elif kind=='missing_scene':payload['results'].pop(pilot.SCENES[1])
    elif kind=='nested_reuse':original['render_source']={}
    if kind is None:
        results,actual=pilot.collect_scenes(c,{},tmp_path/'new')
        assert actual==seals and list(results)==list(pilot.SCENES)
        command,kwargs=calls[0]
        assert command[-2:]==[str(config),str(contract)] and 'p.scene_result' in command[2]
        assert kwargs['cwd']==old and kwargs['env']['PYTHONPATH']==str(old)
    else:
        with pytest.raises(ValueError):pilot.collect_scenes(c,{},tmp_path/'new')
