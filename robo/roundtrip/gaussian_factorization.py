"""Native TRAIN RGB-D adapter for the existing public removal preparation.

This prepares an observed mesh/automatic-instance factory, then calls the
existing inpaint producer. It neither completes background nor renders motion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import os

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from robo.roundtrip.gaussian_build import capture_rows, sha, save
from robo.roundtrip.capture import constructor_command,slurm_device_minors


RECIPE = {'tsdf_voxel_m': .005, 'tsdf_truncation_m': .02,
          'instance_vertex_distance_m': .03, 'instance_min_vertices': 10,
          'instance_mapping': 'existing_auto_segment_nearest_observed_points',
          'frame_selection': 'all_sealed_train', 'background_completion': 'NOT_RUN'}


def checked_file(root, manifest, relative):
    path = root/relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or sha(path) != manifest['source_hashes'].get(relative)):
        raise ValueError('unbound source object artifact: '+relative)
    return path


def source_inputs(capture, source_gs, source_object):
    capture, source_gs, source_object = map(Path, (capture, source_gs, source_object))
    public, cameras = capture_rows(capture)
    digest = sha(capture/'capture_manifest.json')
    gs = json.loads((source_gs/'gs_build_manifest.json').read_text())
    obj = json.loads((source_object/'build_manifest.json').read_text())
    if (gs.get('status') != 'BUILT' or gs.get('capture_manifest_sha256') != digest
            or obj.get('status') != 'BUILT' or obj.get('capture_manifest_sha256') != digest
            or obj.get('tier') != 'DEV' or obj.get('method') != 'B0_fixed_trellis'):
        raise ValueError('same-capture frozen DEV source GS and fixed B0 required')
    ply = source_gs/'pilot/scene.ply'
    if sha(ply) != gs['phases']['pilot']['scene_ply_sha256']:
        raise ValueError('source GS bytes changed')
    needed = ['discovery/discovery_manifest.json', 'discovery/observation_points.npy',
              'construction/objects/obj_00/aligned.json',
              'construction/objects/obj_00/trellis_mesh.ply',
              'construction/objects/obj_00/trellis_gs.ply']
    paths = {key: checked_file(source_object, obj, key) for key in needed}
    discovery = json.loads(paths[needed[0]].read_text())
    if (discovery.get('capture_manifest_sha256') != digest
            or discovery.get('read_splits') != ['train']
            or discovery.get('mask_source') != 'automatic_sam3'):
        raise ValueError('automatic TRAIN discovery lineage required')
    ids = {row['frame_id'] for row in cameras}
    for candidate in discovery['accepted_candidates']:
        if candidate.split('-m')[0] not in ids:
            raise ValueError('non-TRAIN discovery mask')
        paths['discovery/'+candidate+'.png'] = checked_file(source_object, obj, 'discovery/'+candidate+'.png')
    points = np.load(paths[needed[1]], allow_pickle=False)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 10
            or not np.isfinite(points).all() or sha(paths[needed[1]]) != discovery['observation_sha256']):
        raise ValueError('invalid frozen observed target points')
    candidate_path = checked_file(source_object, obj, 'discovery/candidates.json')
    paths['discovery/candidates.json'] = candidate_path
    candidates = {r['candidate_id']:r for r in json.loads(candidate_path.read_text())}
    scores = [candidates[key]['score'] for key in discovery['accepted_candidates']]
    if not scores or not np.isfinite(scores).all(): raise ValueError('missing finite discovery scores')
    discovery = dict(discovery, adapter_mean_sam_score=float(np.mean(scores)))
    return public, cameras, ply, discovery, points, paths


def observed_vertex_indices(vertices, points):
    """Same nearest-vertex/3cm rule as agents.discover.auto_segment."""
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all()):
        raise ValueError('empty or nonfinite observed scene mesh')
    distance, index = cKDTree(vertices).query(points[::max(len(points)//20000, 1)], k=1)
    result = np.unique(index[distance < RECIPE['instance_vertex_distance_m']])
    if len(result) < RECIPE['instance_min_vertices']:
        raise ValueError('observed target has fewer than ten mapped mesh vertices')
    return result


def prepare(capture, source_gs, source_object, out):
    import open3d as o3d
    from agents.discover.derive_mesh_from_splat import new_tsdf_volume, integrate_rgbd
    from agents.edit.inpaint_prepare import _prepare, PUBLIC_ALGORITHM
    public, rows, ply, discovery, points, paths = source_inputs(capture, source_gs, source_object)
    capture, source_gs, source_object, out = map(Path, (capture, source_gs, source_object, out))
    out.mkdir(parents=True, exist_ok=False)
    factory = out/'factory'; factory.mkdir()
    K = np.asarray(rows[0]['K']); width, height = public['width'], public['height']
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0,0], K[1,1], K[0,2], K[1,2])
    volume = new_tsdf_volume(RECIPE['tsdf_voxel_m'], RECIPE['tsdf_truncation_m'])
    poses = []
    for index, row in enumerate(rows):
        rgb = np.asarray(Image.open(capture/row['rgb']).convert('RGB'))
        depth = np.load(capture/row['depth_m'], allow_pickle=False).astype(np.float32)
        w2c = np.linalg.inv(row['T_world_from_camera'])
        integrate_rgbd(volume, rgb, depth, intrinsic, w2c)
        q = Rotation.from_matrix(w2c[:3,:3]).as_quat()[[3,0,1,2]]
        poses += [' '.join(map(str, [index+1, *q, *w2c[:3,3], 1, row['frame_id']+'.png'])), '0 0 -1']
    mesh = volume.extract_triangle_mesh()
    vertices = np.asarray(mesh.vertices)
    if not len(mesh.triangles): raise ValueError('TRAIN TSDF has no observed faces')
    target = observed_vertex_indices(vertices, points)
    if not o3d.io.write_triangle_mesh(str(factory/'derived_mesh.ply'), mesh):
        raise RuntimeError('observed mesh write failed')
    np.savez_compressed(factory/'auto_instances.npz', labels=np.array([discovery['target_prompt']]),
                        scores=np.array([discovery['adapter_mean_sam_score']]), vert_idx_0=target)
    # Existing auto-instance contract uses 1000+k; this is an explicit adapter
    # alias of observed_0, not a native/GT object identifier or new proposal.
    asset = factory/'objects/obj_1000'; asset.mkdir(parents=True)
    for name in ('trellis_mesh.ply', 'trellis_gs.ply'):
        shutil.copyfile(paths['construction/objects/obj_00/'+name], asset/name)
    aligned = json.loads(paths['construction/objects/obj_00/aligned.json'].read_text())
    aligned.update(terminal_action='accept', construction_eligible=True, rejected=False,
                   acceptance_source='fixed_B0_build_manifest; not agentic verification')
    save(asset/'aligned.json', aligned)
    save(factory/'objects/objects.json', [{'index':1000, 'automatic_instance_id':1000,
         'instance_namespace':'automatic', 'label':discovery['target_prompt'],
         'source_object_id':discovery['object_id'], 'centroid':points.mean(axis=0).tolist()}])
    calibration = factory/'intrinsics.json'
    save(calibration, dict(fl_x=K[0,0], fl_y=K[1,1], cx=K[0,2], cy=K[1,2], w=width, h=height))
    camera_path = factory/'images.txt'; camera_path.write_text('\n'.join(poses)+'\n')
    params = dict(factory=factory, intrinsics=calibration, poses=camera_path,
                  splat=ply, inpaint=out/'preparation', training_frames=[r['frame_id']+'.png' for r in rows])
    result = _prepare(public=params)
    source_inputs(capture, source_gs, source_object)  # Detect changed inputs before sealing.
    receipt = dict(schema_version=1, scope='native_DEV_source_GS_removal_preparation',
        status='PREPARED', capture_manifest_sha256=sha(capture/'capture_manifest.json'),
        source_gs_manifest_sha256=sha(source_gs/'gs_build_manifest.json'), source_ply_sha256=sha(ply),
        source_object_manifest_sha256=sha(source_object/'build_manifest.json'),
        input_artifacts={k:sha(v) for k,v in paths.items()}, recipe=RECIPE,
        removal_recipe=PUBLIC_ALGORITHM, automatic_identity_map={'1000':discovery['object_id']},
        observed_mesh_vertices=len(vertices), observed_mesh_faces=len(mesh.triangles),
        target_mesh_vertices=len(target), preparation=result,
        heldout_frames_read=0, native_assets_read=0, clean_background='NOT_RUN',
        object_motion='NOT_RUN', robot_occlusion='NOT_RUN', policy_arm=False,
        output_files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    save(out/'factorization_preparation.json', receipt)
    return receipt


def isolated_command(code,capture,source_gs,source_object,out,python,bubblewrap,forbidden,
                     *,phase='prepare',preparation=None,preparation_sha256=None,
                     checkpoint=None,checkpoint_sha256=None,erasure=None,erasure_sha256=None,
                     dependency_root=None):
    """CPU-only actual allowlist: observed inputs plus audited source modules."""
    code,source_gs,source_object,python=map(Path,(code,source_gs,source_object,python))
    _,_,ply,_,_,objects=source_inputs(capture,source_gs,source_object)
    names=['agents/__init__.py','agents/core/__init__.py','agents/core/common.py',
           'agents/discover/__init__.py','agents/discover/derive_mesh_from_splat.py',
           'agents/edit/__init__.py','agents/edit/inpaint_prepare.py',
           'agents/assets/__init__.py','agents/assets/s5_align.py',
           'robo/__init__.py','robo/roundtrip/capture.py',
           'robo/roundtrip/gaussian_build.py','robo/roundtrip/gaussian_factorization.py']
    mounts={str(code/name):'/code/'+name for name in names}
    envroot=python.absolute().parent.parent
    if any(list(envroot.glob('lib/python*/site-packages/'+name)) for name in ['robocasa','robosuite','omnigibson']):
        raise ValueError('constructor environment contains native reference APIs')
    mounts[str(envroot)]=str(envroot)
    mounts[str(source_gs/'gs_build_manifest.json')]='/source_gs/gs_build_manifest.json'
    mounts[str(ply)]='/source_gs/pilot/scene.ply'
    mounts[str(source_object/'build_manifest.json')]='/source_object/build_manifest.json'
    mounts.update({str(path):'/source_object/'+relative for relative,path in objects.items()})
    if phase not in {'prepare','erase','fill'}:raise ValueError('unregistered native factorization phase')
    devices=[]
    if phase in {'erase','fill'}:
        sealed_preparation(preparation,preparation_sha256)
        mounts[str(preparation)]='/prepared'
    if phase=='erase':
        if sha(checkpoint)!=checkpoint_sha256:raise ValueError('erasure checkpoint differs')
        mounts[str(checkpoint)]='/checkpoint/big-lama.pt'
        for name in ['agents/edit/inpaint_masks.py','agents/edit/inpaint_qwen.py']:
            mounts[str(code/name)]='/code/'+name
    if phase=='fill':
        sealed_erasure(erasure,erasure_sha256,preparation_sha256)
        mounts[str(erasure)]='/erasure'
        mounts[str(code/'agents/edit/inpaint_fill.py')]='/code/agents/edit/inpaint_fill.py'
        extension=envroot/'lib/python3.10/site-packages/gsplat/csrc.so'
        if not extension.is_file() or dependency_root is None:
            raise ValueError('precompiled gsplat extension and explicit dependency runtime required')
        mounts[str(dependency_root)]='/dependencies'
        minors=slurm_device_minors(os.environ)
        if len(minors)!=1:raise ValueError('one allocated fill GPU required')
        devices=[f'/dev/nvidia{next(iter(minors))}','/dev/nvidiactl','/dev/nvidia-uvm']
        if Path('/dev/nvidia-uvm-tools').exists():devices.append('/dev/nvidia-uvm-tools')
        if Path('/opt/nvidia-driver/lib').exists():mounts['/opt/nvidia-driver/lib']='/opt/nvidia-driver/lib'
    for path in ['/usr/lib','/usr/lib64','/etc/ld.so.cache']:
        if Path(path).exists():mounts[path]=path
    bootstrap='''import importlib.util,json,pathlib,sys
sys.path.insert(0,'/dependencies');sys.path.insert(0,'/code')
for name in ['robocasa','robosuite','omnigibson','robo.roundtrip.adapters']:
 assert importlib.util.find_spec(name) is None, 'native API exposed'
for path in json.loads(sys.argv[1]):
 assert not pathlib.Path(path).exists(), 'private tree exposed'
from robo.roundtrip.gaussian_factorization import prepare,erase,fill,save
save(pathlib.Path('/output/isolation_preflight.json'),{'status':'PASS','private_roots_unavailable':True,'native_api_unavailable':True,'network':'unshared'})
if sys.argv[2]=='prepare':prepare('/capture','/source_gs','/source_object','/output/prepared')
elif sys.argv[2]=='erase':
 options=json.loads(sys.argv[3])
 result=erase('/capture','/source_gs','/source_object','/prepared',options['preparation_sha256'],
              '/checkpoint/big-lama.pt',options['checkpoint_sha256'],'/output/erasure')
 if result['status']!='COMPLETE':sys.exit(2)
else:
 import torch,gsplat,hashlib
 from gsplat import csrc
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 save(pathlib.Path('/output/gpu_runtime.json'),{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
  'gsplat_extension_sha256':hashlib.sha256(pathlib.Path(csrc.__file__).read_bytes()).hexdigest()})
 options=json.loads(sys.argv[3])
 fill('/capture','/source_gs','/source_object','/prepared',options['preparation_sha256'],
      '/erasure',options['erasure_sha256'],'/output/fill')
'''
    argv=constructor_command(bubblewrap=bubblewrap,public_capture=capture,output=out,
        runtime_mounts=mounts,forbidden_roots=forbidden,device_paths=devices,argv=[str(python),'-s','-c',bootstrap,
        json.dumps([str(Path(p).resolve()) for p in forbidden]),phase,
        json.dumps(dict(preparation_sha256=preparation_sha256,checkpoint_sha256=checkpoint_sha256,
                        erasure_sha256=erasure_sha256))])
    env={'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4',
         'OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','SIMANY_ROOT':'/code',
         'SIMANY_SCENE':'opaque_train','SIMANY_MESH_SRC':'derived','SIMANY_NO_GT':'1'}
    if phase=='fill':env['CUDA_VISIBLE_DEVICES']=os.environ['CUDA_VISIBLE_DEVICES']
    extra=['--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64']
    for key,value in env.items():extra+=['--setenv',key,value]
    argv[argv.index('--chdir'):argv.index('--chdir')]=extra
    return argv,mounts


def sealed_preparation(directory,expected_sha256):
    directory=Path(directory);path=directory/'factorization_preparation.json'
    if sha(path)!=expected_sha256:raise ValueError('preparation receipt changed')
    record=json.loads(path.read_text())
    if record['scope']!='native_DEV_source_GS_removal_preparation' or record['status']!='PREPARED':
        raise ValueError('successful native preparation required')
    for name,digest in record['output_files'].items():
        target=directory/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or target.is_symlink()
                or not target.resolve().is_relative_to(directory.resolve()) or sha(target)!=digest):
            raise ValueError('preparation output bytes changed')
    return record


def erase(capture,source_gs,source_object,preparation,preparation_sha256,
          checkpoint,checkpoint_sha256,out,*,model=None):
    """Reuse cached SAM3 masks and existing strict LaMa erasure, never fallback."""
    import torch
    from agents.edit.inpaint_masks import _merge_mask,PUBLIC_ALGORITHM as MASK_RECIPE
    from agents.edit.inpaint_qwen import load_lama,erase_masked_view,PUBLIC_ALGORITHM as ERASE_RECIPE
    public,rows,_,discovery,_,paths=source_inputs(capture,source_gs,source_object)
    preparation,capture,checkpoint,out=map(Path,(preparation,capture,checkpoint,out))
    prepared=sealed_preparation(preparation,preparation_sha256)
    if (prepared['capture_manifest_sha256']!=sha(capture/'capture_manifest.json')
            or prepared['source_object_manifest_sha256']!=sha(Path(source_object)/'build_manifest.json')
            or sha(checkpoint)!=checkpoint_sha256):
        raise ValueError('erasure source/checkpoint identity differs')
    out.mkdir(parents=True,exist_ok=False)
    os.environ['LAMA_MODEL']=str(checkpoint)
    torch.set_num_threads(4);torch.manual_seed(0)
    if model is None:model=load_lama()
    by_frame={row['frame_id']+'.png':row for row in rows}
    planned=[]
    for obj in prepared['preparation']['objects']:
        slot=obj['object_slot'];directory=preparation/'preparation'/slot
        views=json.loads((directory/'views.json').read_text())
        masks=np.load(directory/'proj_masks.npz',allow_pickle=False)
        if list(masks['frames'])!=[v['frame'] for v in views]:raise ValueError('prepared mask camera roster differs')
        for index,view in enumerate(views):
            if view['frame'] not in by_frame:raise ValueError('non-TRAIN erasure view')
            row=dict(object_slot=slot,view_index=index,frame=view['frame'],status='NOT_RUN',
                     mask=None,erased_rgb=None,outside_mask_equal=None,failure=None)
            planned.append(row)
            if obj['plane_status']=='NO_PLANE':row['status']='NOT_APPLICABLE_NO_PLANE';continue
            projected=masks['masks'][index].astype(bool)
            if not projected.any():row['status']='EMPTY_MASK';continue
            cached=[np.asarray(Image.open(path))>127 for key,path in paths.items()
                    if key.startswith('discovery/'+view['frame'][:-4]+'-m') and key.endswith('.png')]
            candidates=np.stack(cached) if cached else np.zeros((0,*projected.shape),dtype=bool)
            merged,count=_merge_mask(projected,candidates)
            frame=by_frame[view['frame']]
            full=np.asarray(Image.open(capture/frame['rgb']).convert('RGB'))
            destination=out/slot;destination.mkdir(exist_ok=True)
            mp=destination/f'mask_{index}.png';Image.fromarray((merged*255).astype(np.uint8)).save(mp)
            row.update(mask=str(mp.relative_to(out)),mask_sha256=sha(mp),
                       cached_candidates=len(cached),merged_candidates=count,
                       projected_pixels=int(projected.sum()),final_pixels=int(merged.sum()),
                       train_rgb_sha256=sha(capture/frame['rgb']))
            try:
                image=erase_masked_view(full,merged,model)
                if image is None or not np.array_equal(image[~merged],full[~merged]):
                    raise ValueError('erasure changed unmasked TRAIN pixels')
                output=destination/f'inpainted_{index}.png';Image.fromarray(image).save(output)
                row.update(status='ERASED',erased_rgb=str(output.relative_to(out)),
                           erased_rgb_sha256=sha(output),outside_mask_equal=True)
            except Exception as error:
                row.update(status='ENHANCER_FAILED',failure={'type':type(error).__name__,'reason':str(error)})
    sealed_preparation(preparation,preparation_sha256)
    source_inputs(capture,source_gs,source_object)
    if sha(checkpoint)!=checkpoint_sha256:raise ValueError('erasure checkpoint changed during execution')
    result=dict(schema_version=1,scope='native_DEV_train_only_LaMa_erasure',
        status='COMPLETE' if all(r['status']=='ERASED' for r in planned) else 'INCOMPLETE',
        backend='lama_cpu',checkpoint_sha256=checkpoint_sha256,
        preparation_sha256=preparation_sha256,capture_manifest_sha256=sha(capture/'capture_manifest.json'),
        mask_recipe=MASK_RECIPE,erasure_recipe=ERASE_RECIPE,mask_inference='reuse frozen automatic TRAIN SAM3 masks',
        planned_objects=prepared['preparation']['planned_objects'],planned_views=len(planned),
        erased_views=sum(r['status']=='ERASED' for r in planned),rows=planned,
        clean_background='NOT_RUN',object_motion='NOT_RUN',robot_occlusion='NOT_RUN',
        heldout_frames_read=0,native_assets_read=0,policy_arm=False,
        output_files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    save(out/'native_erasure.json',result)
    return result


def sealed_erasure(directory,expected_sha256,preparation_sha256):
    directory=Path(directory);path=directory/'native_erasure.json'
    if sha(path)!=expected_sha256:raise ValueError('erasure receipt changed')
    result=json.loads(path.read_text())
    if (result['scope']!='native_DEV_train_only_LaMa_erasure' or result['status']!='COMPLETE'
            or result['preparation_sha256']!=preparation_sha256
            or result['planned_views']!=len(result['rows']) or result['erased_views']!=result['planned_views']
            or not result['rows'] or any(r['status']!='ERASED' or not r['outside_mask_equal'] for r in result['rows'])):
        raise ValueError('incomplete erasure cannot enter Gaussian fill')
    keys=[(r['object_slot'],r['view_index']) for r in result['rows']]
    if len(set(keys))!=len(keys):raise ValueError('duplicate erased view')
    for name,digest in result['output_files'].items():
        target=directory/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or target.is_symlink()
                or not target.resolve().is_relative_to(directory.resolve()) or sha(target)!=digest):
            raise ValueError('erasure output bytes changed')
    for row in result['rows']:
        if (result['output_files'].get(row['mask'])!=row['mask_sha256']
                or result['output_files'].get(row['erased_rgb'])!=row['erased_rgb_sha256']):
            raise ValueError('erasure view not in sealed output roster')
    return result


def fill(capture,source_gs,source_object,preparation,preparation_sha256,
         erasure,erasure_sha256,out):
    """Call existing fixed 1500-step TRAIN-erasure fill; source GS stays frozen."""
    import torch
    from agents.edit.inpaint_fill import _fill,_validate_fill_products,PUBLIC_ALGORITHM
    public,rows,ply,_,points,_=source_inputs(capture,source_gs,source_object)
    prepared=sealed_preparation(preparation,preparation_sha256)
    erased=sealed_erasure(erasure,erasure_sha256,preparation_sha256)
    if erased['capture_manifest_sha256']!=sha(Path(capture)/'capture_manifest.json'):
        raise ValueError('fill capture identity differs')
    if (prepared['preparation']['accepted_objects']!=1 or prepared['preparation']['no_plane_objects']
            or prepared['preparation']['no_view_objects']):
        raise ValueError('all fixed native objects require support and prepared views')
    out,preparation,erasure=map(Path,(out,preparation,erasure));out.mkdir(parents=True,exist_ok=False)
    # Original preparation stays immutable. The older adapter omitted aabb,
    # required by existing mask-vote carving. Add only that observed-point
    # metadata to a new factory copy, never native or hidden geometry.
    factory=out/'factory';shutil.copytree(preparation/'factory/objects',factory/'objects',
                                       ignore=shutil.ignore_patterns('objects.json'))
    objects=json.loads((preparation/'factory/objects/objects.json').read_text())
    objects[0]['aabb']=[points.min(axis=0).tolist(),points.max(axis=0).tolist()]
    save(factory/'objects/objects.json',objects)
    outputs=out/'background';outputs.mkdir()
    mapping={r['frame_id']+'.png':r for r in rows}
    train={name:{'path':str(Path(capture)/r['rgb'])} for name,r in mapping.items()}
    masks={};erasures={}
    for row in erased['rows']:
        if row['frame'] not in mapping:raise ValueError('non-TRAIN fill target')
        key=(row['object_slot'],row['view_index'])
        masks[key]={'path':str(erasure/row['mask'])};erasures[key]={'path':str(erasure/row['erased_rgb'])}
        actual=np.asarray(Image.open(train[row['frame']]['path']).convert('RGB'))
        target=np.asarray(Image.open(erasures[key]['path']).convert('RGB'))
        mask=np.asarray(Image.open(masks[key]['path']))>127
        if sha(train[row['frame']]['path'])!=row['train_rgb_sha256'] or not np.array_equal(actual[~mask],target[~mask]):
            raise ValueError('erasure target changed unmasked TRAIN RGB')
    np.random.seed(0);torch.manual_seed(0)
    diagnostics=_fill(public=dict(factory=factory,preparation=preparation/'preparation',output=outputs,
        intrinsics=preparation/'factory/intrinsics.json',splat=ply,masks=masks,erasures=erasures,train_images=train))
    _validate_fill_products(outputs,diagnostics,prepared['preparation']['accepted_objects'])
    sealed_preparation(preparation,preparation_sha256);sealed_erasure(erasure,erasure_sha256,preparation_sha256)
    source_inputs(capture,source_gs,source_object)
    result=dict(schema_version=1,scope='native_DEV_train_only_Gaussian_background_completion',status='BUILT',
        preparation_sha256=preparation_sha256,erasure_sha256=erasure_sha256,source_ply_sha256=sha(ply),
        capture_manifest_sha256=sha(Path(capture)/'capture_manifest.json'),recipe=PUBLIC_ALGORITHM,
        metadata_extension='observed target AABB added to separate factory copy',
        diagnostics=diagnostics,clean_background_sha256=sha(outputs/'clean_background.ply'),
        heldout_frames_read=0,native_assets_read=0,object_motion='NOT_RUN',robot_occlusion='NOT_RUN',policy_arm=False,
        output_files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    save(out/'native_background.json',result)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--capture',required=True,type=Path)
    p.add_argument('--source-gs',required=True,type=Path);p.add_argument('--source-object',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    print(json.dumps(prepare(a.capture,a.source_gs,a.source_object,a.out)))


if __name__ == '__main__': main()
