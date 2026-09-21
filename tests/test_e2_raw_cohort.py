"""The raw cohort keeps fixed TEST views and uses the canonical per-view evaluator."""
import copy,json
from pathlib import Path
import numpy as np
import pytest,yaml
from PIL import Image
from run.icra2027 import e2_raw_room as raw
from run.icra2027 import e3_auto_discovery_pilot as auto
from agents.recon.colmap_poses import file_identity
from tests.test_e2_raw_room import fixture_plan,common_mock


def config(mode='smoke'):
    return yaml.safe_load((raw.CODE/f'configs/experiments/icra2027/e2_raw_cohort_{mode}.yaml').read_text())


def test_population_and_receipt_routes():
    c=config();assert raw.cohort_scenes(c)==list(raw.SMOKE_SCENES)
    assert len(raw.cohort_scenes(config('full')))==50
    assert raw.receipt_root(dict(c,scene_id='a'),Path('/root'))==Path('/root/raw_room/a')
    for key,value in [('planned_scenes',2),('mode','pilot'),('smoke_scenes',['different'])]:
        changed=dict(c,**{key:value})
        with pytest.raises(ValueError):raw.cohort_scenes(changed)


@pytest.mark.parametrize('n',[2,50])
def test_canonical_cohort_combines_per_view_records_without_scene_mean(n,tmp_path,monkeypatch):
    from robo.eval import fidelity_metrics as metrics
    scenes=[f'scene{i}' for i in range(n)];manifests=[]
    for i,scene in enumerate(scenes):
        d=tmp_path/scene;d.mkdir();plan=fixture_plan(d);plan['scene_id']=scene
        common,_=common_mock(plan);raw.export_raw(plan,d/'bundle',common)
        manifests.append(raw.canonical_manifest(plan,d/'bundle'))
    manifest=raw.combine_cohort_manifests(manifests,scenes=scenes,freeze_id='fixture',code_commit='a'*40)
    class FakeLPIPS:
        error=None;provenance={}
        def __init__(self,*a,**kw):pass
        def __call__(self,*a,**kw):return .25
    monkeypatch.setattr(metrics,'LPIPSEvaluator',FakeLPIPS)
    monkeypatch.setattr(metrics,'REPOSITORY_ROOT',tmp_path)
    p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest))
    result=metrics.evaluate_manifest(p,tmp_path/'table',bootstrap_samples=10)
    row=next(r for r in result['rows'] if r['method']==raw.METHOD)
    assert row['n_images']==n*8 and row['n_scenes']==n
    assert row['metric_samples']==dict(psnr=n*8,ssim=n*8,lpips=n*8)
    assert len(result['rows'])==8 and result['paper_ready'] is False
    assert all(r['psnr'] is None for r in result['rows'] if r['method']!=raw.METHOD)
    for mutation in ('missing','source','scene','extra_method'):
        bad=copy.deepcopy(manifests)
        if mutation=='missing':bad.pop()
        elif mutation=='source':bad[0]['provenance']['code_commit']='other'
        elif mutation=='scene':bad[0]['room_methods'][raw.METHOD]['records'][0]['scene_id']='other'
        else:
            key=next(k for k in bad[0]['room_methods'] if k!=raw.METHOD)
            bad[0]['room_methods'][key]['records']=[{}]
        with pytest.raises(ValueError):raw.combine_cohort_manifests(bad,scenes=scenes,freeze_id='fixture',code_commit='a'*40)


@pytest.fixture
def train_source(tmp_path,monkeypatch):
    c=config();c['scene_id']=raw.SMOKE_SCENES[0]
    d=tmp_path/'source';d.mkdir();plan=fixture_plan(d)
    train=d/'train.jpg';Image.fromarray(np.full((8,10,3),17,dtype=np.uint8)).save(train)
    split=d/'train_test_lists.json';split.write_text(json.dumps({'train':['train.jpg'],'test':[r['frame'] for r in plan['evaluation_images']]}))
    image_root=d/'resized_undistorted_images';image_root.mkdir()
    for row in plan['evaluation_images']:(image_root/row['frame']).write_bytes(Path(row['path']).read_bytes())
    (image_root/'train.jpg').write_bytes(train.read_bytes())
    poses=d/'images.txt';poses.write_text('synthetic poses')
    source={'frames':[{'name':'train.jpg','source_rgb':str(train),'image':file_identity(train)}],
        'boundary':{'max_train_frames':None,'training_frames':['train.jpg']},'calibration':plan['calibration'],
        'source_metadata':{'colmap/images.txt':dict(path=str(poses),**file_identity(poses)),
                           'train_test_lists.json':dict(path=str(split),**file_identity(split))}}
    inputs=tmp_path/'unit/scene';inputs.mkdir(parents=True);(inputs/'training_inputs.json').write_text(json.dumps(source))
    c['gaussian_cohort']['freeze_root']=str(tmp_path)
    bound=dict(c,gaussian=plan['gaussian']['path'],gaussian_sha256=plan['gaussian']['sha256'],gaussian_provenance={})
    monkeypatch.setattr(auto,'bind_cohort_gaussian',lambda c:bound)
    monkeypatch.setattr(auto,'gaussian_source_layout',lambda c:('unit','train-full',set()))
    from agents.recon import colmap_poses
    monkeypatch.setattr(colmap_poses,'pose_headers',lambda p:{r['frame']:np.eye(4) for r in plan['evaluation_images']})
    calls=[];monkeypatch.setattr(auto,'validate_gaussian_provenance',lambda *a:calls.append(a))
    return c,source,inputs,train,image_root,calls


def test_complete_train_source_plan_uses_exact_eight_disjoint_test_views(train_source):
    c,source,inputs,train,image_root,calls=train_source
    plan=raw.cohort_plan_payload(c,{'commit':'a'*40},{'contract_sha256':'b'*64},full_validation=True)
    assert len(plan['evaluation_images'])==8 and len(calls)==1
    assert plan['optimization_input_frames']==['train.jpg']
    assert plan['selected_rgb_training_byte_overlap']==[] and not plan['paper_ready']


@pytest.mark.parametrize('mutation',['overlap','training_limit','metadata','gaussian'])
def test_source_plan_rejects_leakage_or_provenance_drift(train_source,mutation):
    c,source,inputs,train,image_root,calls=train_source
    if mutation=='overlap':(image_root/'test0.jpg').write_bytes(train.read_bytes())
    elif mutation=='training_limit':
        source['boundary']['max_train_frames']=48;(inputs/'training_inputs.json').write_text(json.dumps(source))
    elif mutation=='metadata':Path(source['source_metadata']['colmap/images.txt']['path']).write_bytes(b'drift')
    else:(image_root.parent/'scene.ply').write_bytes(b'drift')
    with pytest.raises(ValueError):raw.cohort_plan_payload(c,{'commit':'a'*40},{'contract_sha256':'b'*64},full_validation=True)


def test_full_requires_source_bound_two_scene_smoke_and_exact_metric_counts(tmp_path,monkeypatch):
    from run.icra2027 import e3_gaussian_train_only as gaussian
    monkeypatch.setattr(gaussian,'require_published_full_source',lambda _:None)
    smoke=config();smoke['freeze_id']='smoke';smoke_path=tmp_path/'smoke.yaml';smoke_path.write_text(yaml.safe_dump(smoke))
    root=tmp_path/'smoke';out=tmp_path/'local/raw_room';(out/'aggregate/table').mkdir(parents=True)
    table={'paper_ready':False,'lpips_backend_error':None,'validation':{'lpips_provenance_complete':True},
           'rows':[{'method':raw.METHOD,'n_scenes':2,'n_images':16,'metric_samples':dict(psnr=16,ssim=16,lpips=16)}]}
    table_path=out/'aggregate/table/fidelity_table.json';table_path.write_text(json.dumps(table))
    coverage={'planned_scenes':2,'planned_views':16,'completed_scenes':2,'rows':[{'scene_id':s,'status':'PASS'} for s in raw.SMOKE_SCENES]}
    coverage_path=out/'aggregate/coverage.json';coverage_path.write_text(json.dumps(coverage))
    receipt={'status':'PASS','code_commit':'a'*40,'config_sha256':raw.sha(smoke_path),'contract_sha256':'b'*64,
             'scenes':list(raw.SMOKE_SCENES),'n_images':16,'table':auto.identity(table_path),'coverage':auto.identity(coverage_path)}
    (root/'raw_room').mkdir(parents=True);receipt_path=root/'raw_room/aggregate_receipt.json';receipt_path.write_text(json.dumps(receipt))
    c=dict(smoke,mode='full',freeze_id='full',smoke_dependency={'config':auto.identity(smoke_path),'freeze_root':str(root)})
    code={'commit':'a'*40};monkeypatch.setattr(raw,'context',lambda *a:(smoke,code,out,{'contract_sha256':'b'*64}))
    assert raw.require_cohort_smoke(c,code)==receipt
    for field,value in [('code_commit','other'),('n_images',8),('scenes',[raw.SMOKE_SCENES[0]])]:
        bad=dict(receipt,**{field:value});receipt_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):raw.require_cohort_smoke(c,code)
    table['rows'][0]['metric_samples']['lpips']=8;table_path.write_text(json.dumps(table))
    receipt['table']=auto.identity(table_path);receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='sixteen'):raw.require_cohort_smoke(c,code)


def test_full_driver_runs_render_only_and_preserves_scene_argument(tmp_path,monkeypatch):
    from types import SimpleNamespace
    c=config('full');scene='38d58a7a31';out=tmp_path/'local';out.mkdir();root=tmp_path/'root'
    c['scene_id']=scene
    monkeypatch.setattr(raw,'context',lambda *a:(c,{'commit':'a'*40},out,{}))
    monkeypatch.setattr(raw,'checked_plan',lambda *a:{})
    gates=[];monkeypatch.setattr(raw,'require_cohort_smoke',lambda *a:gates.append(True))
    monkeypatch.setattr(raw,'gpu_identity',lambda:{'node':'hala'})
    monkeypatch.setattr(raw,'environment',lambda *a:{})
    commands=[]
    def subprocess_run(command,**kw):
        commands.append(command);(out/'bundle').mkdir();(out/'bundle/manifest.json').write_text('{}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(raw.subprocess,'run',subprocess_run)
    path=tmp_path/'config.yaml';path.write_text('fixed')
    raw.run(path,root,scene)
    assert gates==[True] and len(commands)==1
    assert commands[0][commands[0].index('--phase')+1]=='render'
    assert commands[0][commands[0].index('--scene-id')+1]==scene
    receipt=json.loads((root/'raw_room'/scene/'execution_receipt.json').read_text())
    assert receipt['scene_id']==scene and receipt['status']=='PASS' and 'table' not in receipt


def camera_bank_fixture(train_source, tmp_path, monkeypatch):
    c, source, inputs, train, image_root, calls = train_source
    c = dict(c, mode='full', freeze_id='original-full')
    scene = c['scene_id']
    monkeypatch.setattr(raw, 'cohort_scenes', lambda _: [scene])
    config_path = tmp_path/'original.yaml'
    original_config = {k:v for k,v in c.items() if k != 'scene_id'}
    config_path.write_text(yaml.safe_dump(original_config))
    root = tmp_path/'original-full'
    contract = dict(freeze_id=root.name, code={'commit':'a'*40, 'dirty':False},
        resource_inventory=[dict(id='e2_raw_room_config',sha256=raw.sha(config_path))])
    contract['contract_sha256'] = raw.canonical_hash(contract)
    raw.write_new(root/'contract/freeze_manifest.json', contract)
    plan = raw.cohort_plan_payload(c, contract['code'], contract)
    plan_path = tmp_path/'old-local'/scene/'plan.json';raw.write_new(plan_path,plan)
    receipt = dict(status='PASS',code_commit='a'*40,scene_id=scene,
        config_sha256=raw.sha(config_path),native_train_source_validation=True,
        plan=raw.identity(plan_path))
    raw.write_new(root/'raw_room'/scene/'plan_receipt.json',receipt)
    bank_path = tmp_path/'bank.json'
    bank_identity = raw.build_camera_bank(config_path,root,bank_path)
    return dict(c,freeze_id='new-full',camera_bank=bank_identity), plan, bank_path, plan_path


def test_frozen_camera_bank_exact_values_survive_runtime_recomputation(train_source,tmp_path,monkeypatch):
    c, original, bank, original_path = camera_bank_fixture(train_source,tmp_path,monkeypatch)
    from agents.recon import colmap_poses
    # Values regenerated by a different NumPy/BLAS must never replace bank bytes.
    changed = np.eye(4); changed[0,0] += 1e-15
    monkeypatch.setattr(colmap_poses,'pose_headers',lambda _: {
        row['frame']:changed.copy() for row in original['evaluation_images']})
    actual = raw.cohort_plan_payload(c,{'commit':'b'*40},{'contract_sha256':'c'*64})
    assert actual['evaluation_images'] == original['evaluation_images']
    assert actual['evaluation_images'][0]['w2c'][0][0] != changed[0,0]
    assert actual['original_camera_plan'] == raw.identity(original_path)
    assert actual['camera_bank'] == raw.identity(bank)
    assert actual['code_commit'] == 'b'*40 and actual['freeze_id'] == 'new-full'
    common, calls = common_mock(actual)
    raw.export_raw(actual,tmp_path/'new-bundle',common)
    assert len(calls) == 8


@pytest.mark.parametrize('mutation',['bank_bytes','old_plan_bytes','old_receipt_bytes',
    'wrong_scene','calibration','matrix','source_commit'])
def test_camera_bank_rejects_integrity_drift(train_source,tmp_path,monkeypatch,mutation):
    c, original, bank_path, plan_path = camera_bank_fixture(train_source,tmp_path,monkeypatch)
    bank = json.loads(bank_path.read_text())
    if mutation == 'bank_bytes':bank_path.write_text(bank_path.read_text()+' ')
    elif mutation == 'old_plan_bytes':plan_path.write_text(plan_path.read_text()+' ')
    elif mutation == 'old_receipt_bytes':
        p=Path(bank['scenes'][0]['plan_receipt']['path']);p.write_text(p.read_text()+' ')
    elif mutation in {'wrong_scene','source_commit'}:
        if mutation=='wrong_scene':bank['scenes'][0]['scene_id']='outside-roster'
        else:bank['source_commit']='f'*40
        bank_path.write_text(json.dumps(bank));c['camera_bank']=raw.identity(bank_path)
    else:
        if mutation=='calibration':original['calibration']['width'] += 1
        else:original['evaluation_images'][0]['w2c'][3][3] = 2
        plan_path.write_text(json.dumps(original));bank['scenes'][0]['plan']=raw.identity(plan_path)
        receipt_path=Path(bank['scenes'][0]['plan_receipt']['path'])
        receipt=json.loads(receipt_path.read_text());receipt['plan']=raw.identity(plan_path)
        receipt_path.write_text(json.dumps(receipt));bank['scenes'][0]['plan_receipt']=raw.identity(receipt_path)
        bank_path.write_text(json.dumps(bank));c['camera_bank']=raw.identity(bank_path)
    with pytest.raises(ValueError):
        raw.cohort_plan_payload(c,{'commit':'b'*40},{'contract_sha256':'c'*64})


def test_camera_bank_builder_refuses_overwrite_and_unvalidated_plan(train_source,tmp_path,monkeypatch):
    c, original, bank_path, plan_path = camera_bank_fixture(train_source,tmp_path,monkeypatch)
    bank=json.loads(bank_path.read_text())
    config_path=bank['source_config']['path'];root=Path(bank['source_contract']['path']).parents[1]
    with pytest.raises(FileExistsError):raw.build_camera_bank(config_path,root,bank_path)
    p=Path(bank['scenes'][0]['plan_receipt']['path']);receipt=json.loads(p.read_text())
    receipt['native_train_source_validation']=False;p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='provenance'):
        raw.build_camera_bank(config_path,root,tmp_path/'bad-bank.json')


def reuse_fixture(train_source,tmp_path,monkeypatch):
    c, original, bank_path, plan_path = camera_bank_fixture(train_source,tmp_path,monkeypatch)
    camera=json.loads(bank_path.read_text());old_config=Path(camera['source_config']['path'])
    root=Path(camera['source_contract']['path']).parents[1];old_out=plan_path.parent
    common,_=common_mock(original);raw.export_raw(original,old_out/'bundle',common)
    artifact=raw.identity(old_out/'bundle/manifest.json')
    render=dict(status='PASS',code_commit=original['code_commit'],freeze_id=original['freeze_id'],
        config_sha256=raw.sha(old_config),plan_sha256=raw.sha(plan_path),artifact=artifact)
    raw.write_new(old_out/'render_receipt.json',render)
    raw.write_new(root/'raw_room'/original['scene_id']/'execution_receipt.json',dict(
        status='PASS',code_commit=original['code_commit'],freeze_id=original['freeze_id'],
        scene_id=original['scene_id'],artifact=artifact,stages=[dict(stage='render',exit_code=0)]))
    monkeypatch.setattr(raw,'reuse_source_fingerprint',lambda _:dict(renderer='unchanged',metric='unchanged'))
    reuse_path=tmp_path/'reuse-bank.json';raw.build_reuse_bank(old_config,root,bank_path,reuse_path)
    c['reuse_bank']=raw.identity(reuse_path)
    code={'commit':'b'*40};contract={'contract_sha256':'c'*64}
    plan=raw.cohort_plan_payload(c,code,contract)
    out=tmp_path/'new-output';out.mkdir();raw.write_new(out/'plan.json',plan)
    config_path=tmp_path/'new-config.yaml';config_path.write_text(yaml.safe_dump({k:v for k,v in c.items() if k!='scene_id'}))
    monkeypatch.setattr(raw,'context',lambda *args:(c,code,out,contract))
    monkeypatch.setattr(raw,'checked_plan',lambda *args:plan)
    return c,code,plan,out,config_path,reuse_path,old_out


def test_reuse_copies_exact_pixels_preserves_producer_and_runs_canonical_metrics(train_source,tmp_path,monkeypatch):
    c,code,plan,out,config_path,reuse_path,old_out=reuse_fixture(train_source,tmp_path,monkeypatch)
    raw.import_reused_scene(config_path,tmp_path/'root',c['scene_id'])
    manifest=raw.reused_evaluation_manifest(c,code,out,plan,config_path)
    record=manifest['room_methods'][raw.METHOD]['records'][0]
    assert record['freeze_id']=='new-full'
    assert record['render_producer']['code_commit']=='a'*40
    assert record['render_producer']['freeze_id']=='original-full'
    assert json.loads((out/'bundle/manifest.json').read_text())['plan']['code_commit']=='a'*40
    assert not (out/'render_receipt.json').exists() and not (out/'execution_claim.json').exists()
    assert raw.sha(out/'bundle/original_manifest.json')==raw.sha(old_out/'bundle/manifest.json')
    for directory in ('input_scene_gaussian','reference'):
        for p in (old_out/'bundle'/directory).glob('*.png'):
            assert raw.sha(p)==raw.sha(out/'bundle'/directory/p.name)
    combined=raw.combine_cohort_manifests([manifest],scenes=[c['scene_id']],freeze_id=c['freeze_id'],code_commit=code['commit'])
    from robo.eval import fidelity_metrics as metrics
    class FakeLPIPS:
        error=None;provenance={}
        def __init__(self,*a,**kw):pass
        def __call__(self,*a,**kw):return .25
    monkeypatch.setattr(metrics,'LPIPSEvaluator',FakeLPIPS)
    monkeypatch.setattr(metrics,'REPOSITORY_ROOT',tmp_path)
    p=out/'evaluation.json';p.write_text(json.dumps(combined))
    result=metrics.evaluate_manifest(p,out/'table',bootstrap_samples=10)
    row=next(r for r in result['rows'] if r['method']==raw.METHOD)
    assert row['n_images']==8 and row['metric_samples']==dict(psnr=8,ssim=8,lpips=8)
    with pytest.raises(FileExistsError):raw.import_reused_scene(config_path,tmp_path/'root',c['scene_id'])


@pytest.mark.parametrize('mutation',['producer','runtime','camera','gaussian','recipe',
    'pixel','execution','fingerprint','action','copied_pixel','import_producer'])
def test_reuse_rejects_provenance_or_treatment_drift(train_source,tmp_path,monkeypatch,mutation):
    c,code,plan,out,config_path,reuse_path,old_out=reuse_fixture(train_source,tmp_path,monkeypatch)
    bank=json.loads(reuse_path.read_text())
    if mutation in {'copied_pixel','import_producer'}:
        raw.import_reused_scene(config_path,tmp_path/'root',c['scene_id'])
        if mutation=='copied_pixel':next((out/'bundle/input_scene_gaussian').glob('*.png')).write_bytes(b'drift')
        else:
            p=out/'reuse_receipt.json';r=json.loads(p.read_text());r['producer_commit']='b'*40;p.write_text(json.dumps(r))
        with pytest.raises(ValueError):raw.reused_evaluation_manifest(c,code,out,plan,config_path)
        return
    if mutation=='producer':bank['source_commit']='f'*40
    elif mutation=='runtime':c=copy.deepcopy(c);c['runtime']['render']['version']='other-runtime'
    elif mutation=='camera':plan['evaluation_images'][0]['w2c'][0][0]+=1e-15
    elif mutation=='gaussian':plan['gaussian']['sha256']='f'*64
    elif mutation=='recipe':plan['render_recipe']['scale']=.5
    elif mutation=='pixel':next((old_out/'bundle/input_scene_gaussian').glob('*.png')).write_bytes(b'drift')
    elif mutation=='execution':
        p=Path(bank['rows'][0]['execution_receipt']['path']);p.write_text(p.read_text()+' ')
    elif mutation=='fingerprint':bank['source_fingerprint']['renderer']='changed'
    elif mutation=='action':bank['rows'][0]['action']='RENDER'
    if mutation in {'producer','fingerprint','action'}:
        reuse_path.write_text(json.dumps(bank));c['reuse_bank']=raw.identity(reuse_path)
    with pytest.raises(ValueError):raw.checked_reuse_source(c,code,plan)


def test_reuse_rejects_resealed_import_pixel_declarations(train_source,tmp_path,monkeypatch):
    c,code,plan,out,config_path,reuse_path,old_out=reuse_fixture(train_source,tmp_path,monkeypatch)
    raw.import_reused_scene(config_path,tmp_path/'root',c['scene_id'])
    manifest_path=out/'bundle/manifest.json';manifest=json.loads(manifest_path.read_text())
    artifact=manifest['artifacts'][0];pixel=out/'bundle'/artifact['relative_path'];pixel.write_bytes(b'replacement')
    artifact['sha256']=raw.sha(pixel);artifact['bytes']=pixel.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    receipt_path=out/'reuse_receipt.json';receipt=json.loads(receipt_path.read_text())
    receipt['artifact']=raw.identity(manifest_path);receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='frozen original bytes'):
        raw.reused_evaluation_manifest(c,code,out,plan,config_path)
