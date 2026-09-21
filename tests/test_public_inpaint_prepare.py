import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
import yaml

from agents.edit import inpaint_prepare as prepare


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def identity(path):
    import hashlib
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def factory(tmp_path):
    directory = tmp_path/'factory'; directory.mkdir()
    first = trimesh.creation.box(extents=[.2, .2, .2]); first.apply_translation([0, 0, 2])
    second = first.copy(); second.apply_translation([1, 0, 0])
    mesh = trimesh.util.concatenate([first, second]); mesh.export(directory/'derived_mesh.ply')
    np.savez(directory/'auto_instances.npz', labels=['a', 'b'], scores=[1., 1.],
             vert_idx_0=np.arange(8), vert_idx_1=np.arange(8, 16))
    trimesh.points.PointCloud(mesh.vertices).export(directory/'scene.ply')
    objects = []
    for index, action in [(1000, 'accept'), (1001, 'abstain')]:
        objects.append(dict(index=index, automatic_instance_id=index, instance_namespace='automatic', label='item'))
        transform = np.eye(4); transform[:3, 3] = [index-1000, 0, 2]
        write_json(directory/f'objects/obj_{index}/aligned.json', dict(T=transform.tolist(),
            construction_eligible=action == 'accept', terminal_action=action,
            rejected=None if action == 'accept' else 'e3_policy_abstain', reason_codes=['fixture']))
        trimesh.creation.box(extents=[.2, .2, .2]).export(directory/f'objects/obj_{index}/trellis_mesh.ply')
    write_json(directory/'objects/objects.json', objects)
    write_json(directory/'cameras.json', dict(fl_x=8, fl_y=8, cx=4, cy=4, w=8, h=8))
    (directory/'inpaint').mkdir()
    return directory


def public(factory):
    return dict(factory=factory, inpaint=factory/'inpaint', splat=factory/'scene.ply',
                intrinsics=factory/'cameras.json', poses=factory/'poses.txt', training_frames=['train.jpg'])


def test_public_no_plane_no_view_and_abstentions_stay_in_denominator(factory, monkeypatch):
    training = np.eye(4); training[2, 3] = -10  # no object in front of this TRAIN camera
    monkeypatch.setattr(prepare.C, 'load_colmap_w2c', lambda _: {'train.jpg': training, 'test.jpg': np.eye(4)})
    monkeypatch.setattr(prepare.C, 'load_instances', lambda: pytest.fail('GT-dispatch loader used'))
    monkeypatch.setattr(prepare.C, 'make_raycast_scene', lambda: pytest.fail('implicit mesh path used'))
    result = prepare._prepare(public(factory))
    assert (result['planned_objects'], result['accepted_objects'], result['abstained_objects']) == (2, 1, 1)
    assert result['no_plane_objects'] == result['no_view_objects'] == 1
    assert result['objects'][1]['preparation_status'] == 'NOT_APPLICABLE'
    assert json.loads((factory/'inpaint/obj_1000/views.json').read_text()) == []
    assert not (factory/'inpaint/obj_1001').exists()
    assert not (factory/'inpaint/obj_1000/plane.json').exists()


def test_zero_accepted_is_explicit_and_removes_nothing(factory, monkeypatch):
    path = factory/'objects/obj_1000/aligned.json'; alignment = json.loads(path.read_text())
    alignment.update(terminal_action='reject', construction_eligible=False, rejected='e3_policy_reject')
    write_json(path, alignment)
    monkeypatch.setattr(prepare.C, 'load_colmap_w2c', lambda _: {'train.jpg': np.eye(4)})
    result = prepare._prepare(public(factory))
    assert result['status'] == 'NO_ACCEPTED_OBJECTS'
    assert result['planned_objects'] == 2 and result['accepted_objects'] == 0
    assert result['rejected_objects'] == result['abstained_objects'] == 1
    assert result['removed_gaussians'] == 0
    assert np.load(factory/'inpaint/removal_union_idx.npy').size == 0


def test_raw_mesh_transform_applied_exactly_once_with_frozen_sample_seed(factory):
    directory = factory/'objects/obj_1000'
    mesh = trimesh.creation.box(); mesh.apply_translation([1, 0, 0]); mesh.export(directory/'trellis_mesh.ply')
    wrong = mesh.copy(); wrong.apply_translation([100, 0, 0]); wrong.export(directory/'mesh_sim.obj')
    transform = np.eye(4); transform[:3, :3] *= 2; transform[0, 3] = 10
    expected, _ = trimesh.sample.sample_surface(trimesh.load(directory/'trellis_mesh.ply', process=False), 5000, seed=0)
    expected = expected*2 + [10, 0, 0]
    np.random.seed(123)
    actual = prepare._sample_asset(directory, {'T': transform.tolist()}, public=True)
    np.testing.assert_array_equal(actual, expected)
    np.random.seed(456)
    np.testing.assert_array_equal(prepare._sample_asset(directory, {'T': transform.tolist()}, public=True), actual)


@pytest.mark.parametrize('mutation', ['alias', 'duplicate', 'missing', 'wrong_index'])
def test_public_identity_never_falls_back_to_gt_alias(factory, mutation):
    path = factory/'objects/objects.json'; rows = json.loads(path.read_text())
    if mutation == 'alias': rows[0]['gt_object_id'] = rows[0]['automatic_instance_id']
    if mutation == 'duplicate': rows[1] = rows[0]
    if mutation == 'missing': rows.pop()
    if mutation == 'wrong_index': rows[0]['index'] = 0
    write_json(path, rows)
    with pytest.raises(ValueError):
        prepare._public_objects(factory, [{'object_id': 1000}, {'object_id': 1001}])


def test_malformed_accepted_mesh_is_fatal(factory, monkeypatch):
    trimesh.points.PointCloud([[0, 0, 0]]).export(factory/'objects/obj_1000/trellis_mesh.ply')
    monkeypatch.setattr(prepare.C, 'load_colmap_w2c', lambda _: {'train.jpg': np.eye(4)})
    with pytest.raises(ValueError, match='accepted asset'):
        prepare._prepare(public(factory))


def test_missing_train_camera_is_not_replaced_with_test(factory, monkeypatch):
    monkeypatch.setattr(prepare.C, 'load_colmap_w2c', lambda _: {'test.jpg': np.eye(4)})
    with pytest.raises(ValueError, match='unregistered'):
        prepare._prepare(public(factory))


@pytest.fixture
def source_context(tmp_path, monkeypatch):
    from robo.eval import agentic_ablation as e3
    from robo.eval import e3_factory_materializer as materializer
    from run.icra2027 import e3_fresh_generation_contract as fresh
    root = tmp_path/'outputs/icra2027/new'
    factory = tmp_path/'old_freeze/scene/A4'; factory.mkdir(parents=True)
    discovery = tmp_path/'discovery'; discovery.mkdir()
    pool_dir = tmp_path/'pool'; pool_dir.mkdir()
    generation = write_json(tmp_path/'generation.yaml', {'source_pilot': str(discovery)})
    input_manifest = write_json(pool_dir/'input_manifest.json', {'config_sha256': identity(generation)['sha256']})
    pool = write_json(pool_dir/'proposal_pool.json', {'input_manifest_sha256': identity(input_manifest)['sha256']})
    frame = tmp_path/'data/scene/dslr/resized_undistorted_images/train.jpg'
    frame.parent.mkdir(parents=True); frame.write_bytes(b'authentic fixture rgb bytes')
    split = write_json(frame.parent.parent/'train_test_lists.json', {'train':['train.jpg'], 'test':['test.jpg']})
    intrinsics = write_json(frame.parent.parent/'nerfstudio/transforms_undistorted.json', {})
    poses = write_json(frame.parent.parent/'colmap/images.txt', {})
    gaussian = write_json(discovery/'scene.ply', {})
    metadata = {'train_test_lists.json':identity(split), 'nerfstudio/transforms_undistorted.json':identity(intrinsics), 'colmap/images.txt':identity(poses)}
    boundary = dict(scene_id='0000000001', boundary={'training_frames':['train.jpg']}, metadata=metadata,
                    input_images={'train.jpg': identity(frame)}, gaussian=identity(gaussian))
    boundary_path = write_json(discovery/'input_manifest.json', boundary)
    descriptor = write_json(root/'descriptor.json', {'discovery_directory':str(discovery), 'discovery_hashes':{'source':'sealed'}})
    manifest = write_json(factory/'materialization_manifest.json', {'e3_root':str(tmp_path/'construction'),
        'source_scene':{'source_gaussian_training_provenance':fresh.FRESH, 'automatic_scene_descriptor':identity(descriptor)}})
    (tmp_path/'construction').mkdir()
    context = dict(schema_version=1, scope='automatic_train_only_removal_preparation', freeze_id='new', scene_id='0000000001', policy_id='A4',
                   materialization_manifest=identity(manifest), generation_config=identity(generation), seed=0, algorithm=dict(prepare.PUBLIC_ALGORITHM))
    context_path=write_json(tmp_path/'context.yaml',context)
    scene = {'source_scene_gaussian':dict(path=str(gaussian),size_bytes=gaussian.stat().st_size,sha256=identity(gaussian)['sha256']),
             'camera_artifacts':{key:dict(path=str(path),size_bytes=path.stat().st_size,sha256=identity(path)['sha256']) for key,path in [('intrinsics',intrinsics),('poses',poses)]}}
    events=[]
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    monkeypatch.setattr(e3,'_validate_cli_execution',lambda *a,**k:events.append(('E0',k['config_paths'])))
    monkeypatch.setattr(prepare,'_validate_source_factory',lambda *a:events.append(('materializer',{})))
    monkeypatch.setattr(materializer,'_verify_inventory',lambda _: ({},{},{},{}))
    monkeypatch.setattr(materializer,'_automatic_scene_audit',lambda *a: {'initial_pools':{'trellis':identity(pool)}})
    monkeypatch.setattr(e3,'_scene_inventory',lambda *a:scene)
    monkeypatch.setattr(fresh,'discovery_binding',lambda *a:([], {'source_discovery_hashes':{'source':'sealed'},
        'gaussian_provenance':{'status':fresh.FRESH,'training_frames':['train.jpg']}}))
    return context_path, factory, boundary_path, frame, events


def test_public_context_reuses_source_validators_and_e0_binding(source_context):
    context, factory, _, _, events=source_context
    result=prepare._validate_public_context(context,'contract')
    assert result['factory']==factory and result['training_frames']==['train.jpg']
    assert events[0]==('E0',[context]) and events[1][0]=='materializer'


@pytest.mark.parametrize('mutation',['train_rgb','test_frame','seed','generation','empty_output'])
def test_public_context_drift_and_existing_empty_output_fail_closed(source_context,mutation):
    context, factory, boundary_path, image, _=source_context
    if mutation=='train_rgb':image.write_bytes(b'changed')
    elif mutation=='test_frame':
        boundary=json.loads(boundary_path.read_text());boundary['boundary']['training_frames']=['test.jpg'];write_json(boundary_path,boundary)
    elif mutation in {'seed','generation'}:
        value=json.loads(context.read_text())
        if mutation=='seed':value['seed']=1
        else:
            changed=write_json(context.parent/'another.yaml',{'source_pilot':'other'})
            value['generation_config']=identity(changed)
        write_json(context,value)
    else:(factory.parents[2]/'outputs/icra2027/new/fidelity/removal/0000000001/inpaint').mkdir(parents=True)
    with pytest.raises((ValueError,FileExistsError)):
        prepare._validate_public_context(context,'contract')


def test_public_projects_only_selected_train_views(factory, monkeypatch):
    write_json(factory/'cameras.json', dict(fl_x=32, fl_y=32, cx=16, cy=16, w=32, h=32))
    monkeypatch.setattr(prepare.C, 'load_colmap_w2c', lambda _: {'train.jpg': np.eye(4), 'test.jpg': np.eye(4)})
    result=prepare._prepare(public(factory))
    assert result['objects'][0]['selected_frames']==['train.jpg']
    masks=np.load(factory/'inpaint/obj_1000/proj_masks.npz',allow_pickle=False)
    assert masks['frames'].tolist()==['train.jpg'] and masks['masks'].any()


@pytest.mark.parametrize('fail',[False,True])
def test_public_publication_is_atomic_sealed_and_never_overwrites(tmp_path,monkeypatch,fail):
    import sys
    from robo.eval import agentic_ablation as e3
    factory=tmp_path/'factory';factory.mkdir()
    image=factory/'data/scene/dslr/resized_undistorted_images/train.jpg'
    image.parent.mkdir(parents=True);image.write_bytes(b'rgb')
    preparation=tmp_path/'new/removal';preparation.mkdir(parents=True)
    context=dict(factory=factory,preparation_root=preparation,source_validation={},training_frames=['train.jpg'],context_sha256='c'*64,
                 materialization_sha256='m'*64,gaussian_provenance={'status':'FRESH_OFFICIAL_TRAIN_ONLY'},
                 boundary={'input_images':{'train.jpg':identity(image)}})
    monkeypatch.setattr(prepare,'_validate_public_context',lambda *a:dict(context))
    monkeypatch.setattr(e3,'REPOSITORY_ROOT',tmp_path)
    hooks=[];monkeypatch.setattr(sys,'addaudithook',hooks.append)
    def produce(value):
        (value['inpaint']/'partial.dat').write_bytes(b'actual partial bytes')
        if fail:raise ValueError('malformed accepted asset')
        return {'status':'NO_ACCEPTED_OBJECTS','planned_objects':2,'accepted_objects':0}
    monkeypatch.setattr(prepare,'_prepare',produce)
    if fail:
        with pytest.raises(ValueError,match='malformed'):prepare.run_public('context','contract')
        assert not (preparation/'inpaint').exists()
        assert (preparation/'inpaint_failed_partial/partial.dat').read_bytes()==b'actual partial bytes'
        assert json.loads((preparation/'inpaint_public_failure.json').read_text())['status']=='FAILED'
    else:
        result=prepare.run_public('context','contract')
        assert result['cleaned_background_created'] is False and result['paper_ready'] is False
        seal=json.loads((preparation/'inpaint/seal.json').read_text())
        assert seal['members']['partial.dat']==identity(preparation/'inpaint/partial.dat')['sha256']
    assert sorted(p.name for p in factory.iterdir()) == ['data']
    with pytest.raises(FileExistsError):prepare.run_public('context','contract')
    hooks[0]('open',(str(image),'r'))
    with pytest.raises(ValueError):hooks[0]('open',(str(image.with_name('test.jpg')),'r'))
    with pytest.raises(ValueError):hooks[0]('open',(str(image.parent.parent.parent/'scans/mesh.ply'),'r'))
    with pytest.raises(ValueError):hooks[0]('socket.connect',())



def test_public_support_plane_uses_same_ring_and_trim_algorithm(factory, monkeypatch):
    from scipy.spatial import Delaunay
    grid=np.linspace(-.3,.3,55)
    x,y=np.meshgrid(grid,grid)
    vertices=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,1.9)])
    plane=trimesh.Trimesh(vertices=vertices,faces=Delaunay(vertices[:,:2]).simplices,process=False)
    mesh=trimesh.load(factory/'derived_mesh.ply',process=False)
    trimesh.util.concatenate([mesh,plane]).export(factory/'derived_mesh.ply')
    camera=np.eye(4);camera[2,3]=-10
    monkeypatch.setattr(prepare.C,'load_colmap_w2c',lambda _:{'train.jpg':camera})
    result=prepare._prepare(public(factory))
    assert result['objects'][0]['plane_status']=='FIT'
    assert result['no_plane_objects']==result['plane_trim_not_converged_objects']==0
    value=json.loads((factory/'inpaint/obj_1000/plane.json').read_text())
    assert value['trim_ok'] is True
    np.testing.assert_allclose(value['normal'],[0,0,1],atol=1e-6)



def test_selected_subpixel_view_keeps_explicit_empty_mask_count(factory, monkeypatch):
    monkeypatch.setattr(prepare.C, 'load_colmap_w2c', lambda _: {'train.jpg': np.eye(4)})
    result=prepare._prepare(public(factory))
    assert result['objects'][0]['selected_frames']==['train.jpg']
    assert result['objects'][0]['projected_mask_pixels']==[0]
    assert result['empty_projected_mask_views']==1


@pytest.mark.parametrize('mutation',[None,'dirty','head','receipt','commit'])
def test_archived_factory_validation_keeps_recorded_source(tmp_path,monkeypatch,mutation):
    import subprocess
    from types import SimpleNamespace
    from robo.eval import e3_factory_materializer as materializer
    source=tmp_path/'old_source';source.mkdir()
    factory=tmp_path/'old_factory';factory.mkdir()
    manifest_path=write_json(factory/'materialization_manifest.json',{})
    provenance=dict(code_root=str(source),materializer_commit='a'*40,validator_commit='a'*40,
                    materializer_dirty=False,validator_dirty=False)
    if mutation=='commit':provenance['materializer_commit']='b'*40
    def git(args,**kwargs):
        assert kwargs['cwd']==source
        if args[1]=='rev-parse':return ('b'*40 if mutation=='head' else 'a'*40)+'\n'
        return ' M changed\n' if mutation=='dirty' else ''
    calls=[]
    monkeypatch.setattr(subprocess,'check_output',git)
    def run(args,**kwargs):
        calls.append((args,kwargs))
        assert kwargs['cwd']==source and 'PYTHONPATH' not in kwargs['env']
        assert json.loads(kwargs['input'])=={'factory':str(factory),'scene':'0000000001'}
        return SimpleNamespace(stdout=json.dumps(dict(scene_id='0000000001',policy_id='A0' if mutation=='receipt' else 'A4',
            validator_commit='a'*40,manifest_sha256=identity(manifest_path)['sha256'])))
    monkeypatch.setattr(subprocess,'run',run)
    monkeypatch.setattr(materializer,'validate_materialized_factory',lambda *a,**k:pytest.fail('source was relabeled to current validator'))
    if mutation:
        with pytest.raises(ValueError):prepare._validate_source_factory(factory,{'provenance':provenance},'0000000001')
    else:
        report=prepare._validate_source_factory(factory,{'provenance':provenance},'0000000001')
        assert report['validator_commit']=='a'*40 and len(calls)==1


def test_repository_relative_descriptor_resolves_independently_of_execution_cwd(source_context,monkeypatch):
    from robo.eval import agentic_ablation as e3
    context,factory,_,_,_=source_context
    manifest_path=factory/'materialization_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    descriptor=manifest['source_scene']['automatic_scene_descriptor']
    descriptor['path']=str(Path(descriptor['path']).relative_to(e3.REPOSITORY_ROOT))
    write_json(manifest_path,manifest)
    value=json.loads(context.read_text());value['materialization_manifest']=identity(manifest_path);write_json(context,value)
    execution=context.parent/'another_worktree';execution.mkdir();monkeypatch.chdir(execution)
    assert not Path(descriptor['path']).exists()
    result=prepare._validate_public_context(context,'contract')
    assert result['factory']==factory and result['training_frames']==['train.jpg']
