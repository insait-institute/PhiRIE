"""CPU-only contract tests: no real TEST images, model calls, or new metrics."""
import copy
import json
from pathlib import Path

import pytest

from agents.edit.inpaint_masks import _identity
from robo.manifest.hash import canonical_hash
from run.icra2027 import e2_public_factorized as consumer
from run.icra2027 import e2_full_factorized_protocol as protocol
from run.icra2027.e2_full_preparation import ADMISSION_SHA


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return _identity(path)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(consumer.e3, 'REPOSITORY_ROOT', tmp_path)
    monkeypatch.setattr(consumer, 'git_snapshot', lambda _: dict(dirty=False, commit=consumer.PREPARATION_SOURCE))
    ids = [f'{n:010d}' for n in range(50)]
    counts = {s: 37 for s in ids}; counts[ids[-1]] = 58
    p = dict(scene_ids=ids, population=counts, pilot_scene=ids[0])
    protocol_ref = write(tmp_path/'protocol.json', p)
    monkeypatch.setattr(protocol, 'validate_protocol', lambda _: p)
    admission = write(tmp_path/'admission.json', {})
    # The real anchor is checked independently; mock only the fixture admission.
    monkeypatch.setattr(__import__('run.icra2027.e2_full_preparation', fromlist=['x']), 'ADMISSION_SHA', admission['sha256'])
    root = tmp_path/'outputs/icra2027/preparation/fidelity/preparation_inventory'
    config_path = tmp_path/'producer/config.json'; contract_path = tmp_path/'E0.json'
    contexts = {s: write(tmp_path/'producer'/f'{s}.json', {}) for s in ids[1:]}
    pilot = dict(directory=str(tmp_path/'pilot'), seal=write(tmp_path/'pilot/seal.json', {}),
                 context=write(tmp_path/'pilot/context.json', {}), contract=write(tmp_path/'pilot/E0.json', {}),
                 producer_commit='a'*40)
    pc = dict(freeze_id='preparation', full_protocol=protocol_ref, background_admission=admission,
              contexts=contexts, pilot_preparation=pilot)
    config_ref = write(config_path, pc)
    contract = dict(freeze_id='preparation', code=dict(commit=consumer.PREPARATION_SOURCE, dirty=False),
                    resource_inventory=[dict(resolved_path=str(config_path), hash_method='content_sha256', sha256=config_ref['sha256'])])
    contract['contract_sha256'] = canonical_hash(contract)
    contract_ref = write(contract_path, contract)
    rows = []
    for i, sid in enumerate(ids):
        state = 'STRUCTURALLY_FILLABLE' if i < 36 else 'BLOCKED_UNFILLABLE_ACCEPTED' if i < 46 else 'NO_ACCEPTED_OBJECTS'
        directory = tmp_path/'outputs/icra2027/preparation/fidelity/removal'/sid/'inpaint'
        prep = pilot if i == 0 else dict(directory=str(directory), seal=write(directory/'seal.json', {}),
              context=contexts[sid], contract=contract_ref, producer_commit=consumer.PREPARATION_SOURCE)
        rows.append(dict(scene_id=sid, planned_objects=counts[sid], accepted_objects=int(i<46),
            planned_training_views=3 if i<46 else 0, planned_test_views=8,
            status=state, blocked_objects=[dict(object_slot='obj_01', reasons=['NO_PLANE'])] if 36<=i<46 else [],
            missing_test_views=8 if 36<=i<46 else None, preparation=prep, paper_ready=False))
    inventory = dict(schema_version=1, scope='full_e2_training_preparation', status='PASS', freeze_id='preparation',
        config=config_ref, contract=contract_ref, full_protocol=protocol_ref, background_admission=admission,
        planned_scenes=50, planned_objects=1871, planned_test_views=400, rows=rows,
        mask_fill_inference_invoked=False, TEST_quality_used_for_admission=False, paper_ready=False)
    write(root/'inventory.json', inventory); consumer.seal_output(root)
    seal = _identity(root/'seal.json'); monkeypatch.setattr(consumer, 'PREPARATION_SEAL', seal['sha256'])
    c = dict(scope=consumer.FULL_SCOPE, freeze_id='evaluation', scene_ids=ids, planned_objects=counts,
        full_protocol=protocol_ref, preparation_inventory=dict(code_root=str(tmp_path/'producer'),
        code_commit=consumer.PREPARATION_SOURCE, config=config_ref, contract=contract_ref, seal=seal),
        fill_seals={r['scene_id']: write(tmp_path/'fill'/r['scene_id']/'seal.json', {})
                    for r in rows if r['status'] != 'BLOCKED_UNFILLABLE_ACCEPTED'})
    return c, p, inventory, root


def test_full_inventory_authenticates_exact50_without_models_or_test_reads(tmp_path, monkeypatch):
    c, p, inventory, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(consumer, '_preparation', lambda *a: pytest.fail('metadata gate reopened a scene'))
    assert consumer.full_inventory(c) == (p, inventory)
    assert len(c['fill_seals']) == 40 and sum(c['planned_objects'].values()) == 1871


@pytest.mark.parametrize('kind', ['source', 'missing', 'reseal', 'config', 'contract', 'roster', 'count', 'fill_missing', 'fill_blocked', 'legacy_pair'])
def test_full_inventory_drift_fails_closed(tmp_path, monkeypatch, kind):
    c, p, inventory, root = fixture(tmp_path, monkeypatch)
    if kind == 'source': c['preparation_inventory']['code_commit'] = 'b'*40
    elif kind == 'missing': (root/'inventory.json').unlink()
    elif kind == 'reseal':
        inventory['rows'][36]['status'] = 'STRUCTURALLY_FILLABLE'
        write(root/'inventory.json', inventory); (root/'seal.json').unlink(); consumer.seal_output(root)
        c['preparation_inventory']['seal'] = _identity(root/'seal.json')
    elif kind == 'config': Path(c['preparation_inventory']['config']['path']).write_text('{}')
    elif kind == 'contract': Path(c['preparation_inventory']['contract']['path']).write_text('{}')
    elif kind == 'roster': c['scene_ids'] = c['scene_ids'][:-1]
    elif kind == 'count': c['planned_objects'] = dict(c['planned_objects'], **{p['scene_ids'][0]:36})
    elif kind == 'fill_missing': c['fill_seals'].pop(p['scene_ids'][0])
    elif kind == 'fill_blocked': c['fill_seals'][p['scene_ids'][36]] = next(iter(c['fill_seals'].values()))
    else: c['paired_protocol'] = {'legacy':True}
    with pytest.raises((ValueError, FileNotFoundError)):
        consumer.full_inventory(c)


def source_fixture(tmp_path, monkeypatch, state):
    c, p, inventory, _ = fixture(tmp_path, monkeypatch)
    row = next(r for r in inventory['rows'] if r['status'] == state and r['scene_id'] != p['pilot_scene'])
    sid = row['scene_id']; prep_spec = row['preparation']
    objects = [dict(object_slot='obj_01', terminal_action='accept', plane_status='FIT', selected_frames=['train.jpg'], projected_mask_pixels=[4])]
    if state == 'BLOCKED_UNFILLABLE_ACCEPTED': objects[0]['plane_status'] = 'NO_PLANE'
    if state == 'NO_ACCEPTED_OBJECTS': objects = []
    report = dict(planned_objects=p['population'][sid], accepted_objects=len(objects), objects=objects)
    prep = dict(directory=prep_spec['directory'], seal_identity=prep_spec['seal'], context_identity=prep_spec['context'],
        contract_identity=prep_spec['contract'], report=report, jobs=[{}]*row['planned_training_views'],
        source_factory=str(tmp_path/'factory'), train_images={'train.jpg':{}})
    slots = [f'obj_{i:02d}' for i in range(p['population'][sid])]
    p['resolved_jobs'] = write(tmp_path/'jobs.json', dict(scenes=[dict(scene_id=sid, jobs=[dict(object_slot=s) for s in slots])]))
    factory = write(tmp_path/'factory/materialization_manifest.json', dict(e3_code_commit=protocol.CONSTRUCTION_COMMIT,
        e3_freeze_id=protocol.CONSTRUCTION_FREEZE, scene_id=sid, policy_id='A4', roster=dict(job_count=len(slots), object_slots=slots)))
    gaussian = tmp_path/'gaussian.ply'; gaussian.write_bytes(b'source gaussian')
    boundary = write(tmp_path/'generation/input_manifest.json', dict(gaussian=_identity(gaussian)))
    generation = write(tmp_path/'generation/config.json', dict(source_pilot=str(tmp_path/'generation'),
        source_discovery_hashes={'input_manifest.json':boundary['sha256']}))
    prep_spec['context'] = write(Path(prep_spec['context']['path']), dict(scene_id=sid, materialization_manifest=factory,
                                                                     generation_config=generation))
    prep['context_identity'] = prep_spec['context']
    fill = dict(directory=tmp_path/'fill'/sid, seal_identity=c['fill_seals'].get(sid),
        result=dict(planned_objects=row['planned_objects'], accepted_objects=row['accepted_objects'],
                    status='NO_REMOVAL' if not objects else 'COMPLETE'),
        source=dict(factory=Path(prep['source_factory']), blocked=[], erasure=dict(mask_bundle=dict(preparation=prep))))
    monkeypatch.setattr(consumer, 'full_inventory', lambda _: (p, inventory))
    monkeypatch.setattr(consumer, '_preparation', lambda *a: prep)
    monkeypatch.setattr(consumer, 'validate_public_fill', lambda *a: fill)
    return c, sid, row, prep, fill


def test_train_blocker_never_invokes_fill_or_renderer(tmp_path, monkeypatch):
    c, sid, row, prep, _ = source_fixture(tmp_path, monkeypatch, 'BLOCKED_UNFILLABLE_ACCEPTED')
    monkeypatch.setattr(consumer, 'validate_public_fill', lambda *a: pytest.fail('blocked TRAIN scene opened fill'))
    result = consumer.construction(c, sid)
    assert result['source']['blocked'] == row['blocked_objects']
    assert result['result']['status'] == 'BLOCKED_UNFILLABLE_ACCEPTED'
    assert sid not in c['fill_seals']


@pytest.mark.parametrize('state', ['STRUCTURALLY_FILLABLE', 'NO_ACCEPTED_OBJECTS'])
@pytest.mark.parametrize('kind', [None, 'prep_seal', 'source_factory', 'accepted', 'status', 'blocked', 'missing_fill'])
def test_actual_fill_requires_same_preparation_and_zero_semantics(tmp_path, monkeypatch, state, kind):
    c, sid, row, prep, fill = source_fixture(tmp_path, monkeypatch, state)
    fill['source']['erasure']['mask_bundle']['preparation'] = copy.deepcopy(prep)
    if kind == 'prep_seal': fill['source']['erasure']['mask_bundle']['preparation']['seal_identity']['sha256'] = 'wrong'
    elif kind == 'source_factory': fill['source']['factory'] = tmp_path/'other'
    elif kind == 'accepted': fill['result']['accepted_objects'] += 1
    elif kind == 'status': fill['result']['status'] = 'COMPLETE' if state == 'NO_ACCEPTED_OBJECTS' else 'NO_REMOVAL'
    elif kind == 'blocked': fill['source']['blocked'] = [{'reason':'NO_PLANE'}]
    elif kind == 'missing_fill': Path(c['fill_seals'][sid]['path']).unlink()
    if kind is None: assert consumer.construction(c, sid)['result'] == fill['result']
    else:
        with pytest.raises((ValueError, FileNotFoundError)): consumer.construction(c, sid)


def test_relabeling_train_blocker_or_population_is_rejected(tmp_path, monkeypatch):
    c, sid, row, prep, fill = source_fixture(tmp_path, monkeypatch, 'BLOCKED_UNFILLABLE_ACCEPTED')
    row['status'] = 'STRUCTURALLY_FILLABLE'
    with pytest.raises(ValueError, match='relabeled'): consumer.construction(c, sid)


def test_full_common_quality_keeps_all400_coverage_and_zero_object_scenes(tmp_path, monkeypatch):
    c, p, inventory, _ = fixture(tmp_path, monkeypatch)
    c['metric_source'] = {'code_commit':'a'*40}
    monkeypatch.setattr(consumer, 'full_inventory', lambda _: (p, inventory))
    results = {}
    for row in inventory['rows']:
        sid = row['scene_id']; views = [dict(view_id=f'test{n}.jpg', render_path=str(tmp_path/sid/f'test{n}.png'),
                                          gt_path=str(tmp_path/'gt'/sid/f'test{n}.png')) for n in range(8)]
        results[sid] = dict(raw_source=dict(plan=dict(evaluation_images=[{'frame':v['view_id']} for v in views],
            optimization_input_frames=['train.jpg']), bundle_identity={'sha256':'c'*64}),
            raw_views=views, composite_views=[] if row['blocked_objects'] else views,
            source_fill=c['fill_seals'].get(sid), status='BLOCKED_UNFILLABLE_ACCEPTED' if row['blocked_objects'] else 'COMPLETE',
            accepted_objects=row['accepted_objects'],source_preparation=row['preparation'],
            background_status='NO_REMOVAL' if row['status']=='NO_ACCEPTED_OBJECTS' else row['status'])
    manifest, coverage = consumer.make_manifest(c, {'code':{'commit':'b'*40}}, results)
    for method, available in ((consumer.raw.METHOD,400),(consumer.AUTO,320)):
        rows = [r for r in coverage if r['method']==method]
        assert sum(r['planned_views'] for r in rows)==400
        assert sum(r['available_views'] for r in rows)==available
        assert sum(r['analysis_views'] for r in rows)==320
        assert sum(r['planned_objects'] for r in rows)==1871
        assert sum(r['n_views'] for r in manifest['room_methods'][method]['records'])==320
    assert 'one-scene' not in manifest['provenance']['inference_scope']
    assert len([r for r in coverage if not r['accepted_objects']])==8
    results.pop(p['scene_ids'][-1])
    with pytest.raises(ValueError, match='denominator'): consumer.make_manifest(c, {'code':{'commit':'b'*40}}, results)


@pytest.mark.parametrize('kind', [None, 'matrix', 'frame', 'gaussian', 'TRAIN', 'bank', 'outside', 'producer', 'raw_config', 'raw_freeze'])
def test_full_raw_reuse_calls_original_source_then_binds_original_cameras(tmp_path, monkeypatch, kind):
    from tests.test_e2_raw_room import fixture_plan
    sid='0000000001'; plan=fixture_plan(tmp_path)
    original=write(tmp_path/'original-plan.json', plan)
    bank=write(tmp_path/'camera-bank.json', dict(scenes=[dict(scene_id=sid,plan=original)]))
    protocol_ref=write(tmp_path/'protocol.json', {})
    p=dict(scene_ids=[sid],camera_bank=bank,evaluation_frames={sid:[v['frame'] for v in plan['evaluation_images']]})
    monkeypatch.setattr(protocol,'validate_protocol',lambda _:p)
    result=dict(plan=copy.deepcopy(plan));result['plan']['original_camera_plan']=original
    spec=dict(code_root=str(tmp_path),code_commit='a'*40,python='archived-python',freeze_root=str(tmp_path/'freeze'),
        config=write(tmp_path/'raw-config.json',{}),contract=write(tmp_path/'freeze/contract/freeze_manifest.json',{}))
    monkeypatch.setattr(consumer,'RAW_SOURCE',spec['code_commit'])
    monkeypatch.setattr(consumer,'RAW_CONFIG',spec['config']['sha256'])
    monkeypatch.setattr(consumer,'RAW_FREEZE',Path(spec['freeze_root']).name)
    calls=[]
    monkeypatch.setattr(consumer,'source_check',lambda _:tmp_path)
    def execute(command,**kwargs):calls.append(command);return json.dumps(result)
    monkeypatch.setattr(consumer.subprocess,'check_output',execute)
    if kind=='matrix':result['plan']['evaluation_images'][0]['w2c'][0][3]=99
    elif kind=='frame':result['plan']['evaluation_images'][0]['frame']='different.jpg'
    elif kind=='gaussian':result['plan']['gaussian']['sha256']='changed'
    elif kind=='TRAIN':result['plan']['optimization_input_frames']=[]
    elif kind=='bank':result['plan']['original_camera_plan']['sha256']='changed'
    elif kind=='outside':sid='outside'
    elif kind=='producer':spec['code_commit']='different'
    elif kind=='raw_config':spec['config']['sha256']='different'
    elif kind=='raw_freeze':spec['freeze_root']=str(tmp_path/'other-freeze')
    if kind is None:
        assert consumer.audit_raw(spec,sid,full_protocol=protocol_ref)==result
        assert calls[0][0]=='archived-python' and 'r.check_render_receipt' in calls[0][2]
    else:
        with pytest.raises(ValueError):consumer.audit_raw(spec,sid,full_protocol=protocol_ref)
    if kind=='outside':assert not calls


@pytest.mark.parametrize('kind', [None,'relabeled','reference','both_references','raw_render'])
def test_full_scene_seal_rejects_hidden_blockers_and_different_reference_pixels(tmp_path,monkeypatch,kind):
    from tests.test_e2_public_factorized import original
    from tests.test_e2_raw_room import common_mock
    monkeypatch.setattr(consumer.e3,'REPOSITORY_ROOT',tmp_path)
    sid='0000000001';source=original(tmp_path/'original',sid);p=source['plan']
    directory=tmp_path/'scene';directory.mkdir();common,_=common_mock(p)
    views=consumer.raw.export_raw(p,directory/'raw',common)
    blocked=kind in (None,'relabeled')
    composite=[] if blocked else consumer.raw.export_raw(p,directory/'composite',common)
    source['manifest']={'room_methods':{consumer.raw.METHOD:{'records':[{'views':copy.deepcopy(views)}]}}}
    prep={'sealed':'preparation'};fillseal=write(tmp_path/'fill/seal.json',{})
    c=dict(scope=consumer.FULL_SCOPE,freeze_id='new',scene_ids=[sid],planned_objects={sid:37},
           fill_seals={} if blocked else {sid:fillseal},raw_source={})
    config=write(tmp_path/'config.json',c)
    contract=dict(freeze_id='new',code=dict(commit='a'*40,dirty=False),resource_inventory=[
        dict(resolved_path=config['path'],hash_method='content_sha256',sha256=config['sha256'])])
    contract['contract_sha256']=canonical_hash(contract);cp=write(tmp_path/'contract.json',contract)
    fill=dict(preparation=prep,result=dict(planned_objects=37,accepted_objects=1,
        status='BLOCKED_UNFILLABLE_ACCEPTED' if blocked else 'COMPLETE'),source=dict(context={'scene_id':sid},
        source_boundary={'gaussian':p['gaussian']},train_images={'train.jpg':{}},blocked=[{'reason':'NO_PLANE'}] if blocked else []))
    monkeypatch.setattr(consumer,'construction',lambda *a:fill)
    monkeypatch.setattr(consumer,'original_raw',lambda *a:source)
    r=dict(scope=consumer.FULL_SCOPE,freeze_id='new',scene_id=sid,code_commit='a'*40,source_fill=None if blocked else fillseal,
        planned_objects=37,accepted_objects=1,planned_views=8,paper_ready=False,full_e3_gt_access=False,construction_modified=False,
        source_preparation=prep,background_status=fill['result']['status'],config_identity=config,contract_identity=cp,
        status='BLOCKED_UNFILLABLE_ACCEPTED' if blocked else 'COMPLETE',raw_source=source,raw_views=views,composite_views=composite)
    if kind=='relabeled':r['status']='FAILED'
    elif kind in ('reference','both_references'):
        gt=Path(composite[0]['gt_path']);gt.write_bytes(b'different but resealed reference');composite[0]['gt_sha256']=consumer.e3.sha256_file(gt)
        if kind=='both_references':
            raw_gt=Path(views[0]['gt_path']);raw_gt.write_bytes(gt.read_bytes())
            views[0]['gt_sha256']=consumer.e3.sha256_file(raw_gt)
            assert views[0]['gt_sha256']==composite[0]['gt_sha256']
    elif kind=='raw_render':
        rgb=Path(views[0]['render_path']);rgb.write_bytes(b'resealed wrong raw copy')
        views[0]['render_sha256']=consumer.e3.sha256_file(rgb)
    write(directory/'result.json',r);consumer.seal_output(directory)
    if kind is None:assert consumer.scene_result(directory,c,contract,sid)[0]==r
    else:
        with pytest.raises(ValueError):consumer.scene_result(directory,c,contract,sid)

@pytest.mark.parametrize('action,valid', [('VERIFIED_REUSE',True),('VERIFIED_REUSE',False),('UNKNOWN',True)])
def test_original_worker_routes_declared_reuse_through_original_validator(tmp_path,monkeypatch,capsys,action,valid):
    import sys,types
    from run import icra2027
    bank=tmp_path/'bank.json';bank.write_text(json.dumps({'rows':[{'scene_id':'0d2ee665be','action':action}]}))
    refs={'render_receipt':{'path':'original/render.json'},'execution_receipt':{'path':'original/execution.json'}}
    config=dict(reuse_bank={'path':str(bank)},runtime={},python={},renderer_dependency_root='overlay',lpips_backbone={},torch_home='cache')
    calls=[]
    def checked(*args):
        calls.append('checked_reuse_source')
        if not valid:raise ValueError('original source seal changed')
        return {},refs,{}
    def imported(*args):calls.append('reused_evaluation_manifest');return {'authenticated':True}
    fake=types.SimpleNamespace(context=lambda *a:(config,{'commit':'source'},tmp_path,{}),
        checked_plan=lambda *a:{'freeze_id':'evaluation'},checked_identity=lambda ref:Path(ref['path']),
        checked_reuse_source=checked,reused_evaluation_manifest=imported,
        identity=lambda p:{'path':str(p)},reuse_source_fingerprint=lambda c:'unchanged')
    monkeypatch.setattr(icra2027,'e2_raw_room',fake)
    monkeypatch.setattr(sys,'argv',['worker','config','freeze','0d2ee665be'])
    if action!='VERIFIED_REUSE' or not valid:
        with pytest.raises(ValueError):exec(consumer.RAW_AUDIT_PROGRAM,{})
    else:
        exec(consumer.RAW_AUDIT_PROGRAM,{})
        result=json.loads(capsys.readouterr().out)
        assert calls==['checked_reuse_source','reused_evaluation_manifest']
        assert result['execution_receipt']==refs['execution_receipt']
        assert result['manifest']=={'authenticated':True}
