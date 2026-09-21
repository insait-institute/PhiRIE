"""TRAIN-only native RGB-D adaptation for the existing gsplat trainer.

This is source-scene appearance construction, not a movable-object/background
factorization or a policy observation arm. Native assets and heldouts are never
inputs. Metric depth and known OpenCV camera poses are declared assistance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from robo.roundtrip.capture import (validate_public_capture, validate_camera,
                                   constructor_command, slurm_device_minors)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def capture_rows(capture):
    capture = Path(capture)
    manifest = validate_public_capture(capture)
    if (manifest['sensor_regime'] != 'ideal_rgbd'
            or manifest['depth_semantics'] != 'camera_z_meters_zero_missing'
            or manifest['camera_convention'] != 'OpenCV_x_right_y_down_z_forward'
            or manifest['transform_convention'] != 'T_world_from_camera'
            or manifest['color_space'] != 'sRGB' or manifest['distortion'] != 'none_undistorted'):
        raise ValueError('source GS requires explicitly declared posed metric RGB-D')
    rows = [json.loads(line) for line in (capture/'train/cameras.jsonl').read_text().splitlines()]
    if len(rows) < 2:
        raise ValueError('at least two TRAIN views required by existing trainer')
    for row in rows:
        fid = row['frame_id']
        if not re.fullmatch(r'f[0-9]{6}', fid):
            raise ValueError('opaque TRAIN frame IDs required')
        for key, suffix, hash_key in [('rgb', f'train/rgb/{fid}.png', 'rgb_sha256'),
                                      ('depth_m', f'train/depth_m/{fid}.npy', 'depth_sha256')]:
            if row.get(key) != suffix or manifest['files'].get(suffix) != row.get(hash_key):
                raise ValueError('camera row escapes its sealed TRAIN file/hash')
    # The existing trainer currently has one K per scene. Fail explicitly rather
    # than silently substituting the first calibration for a different camera.
    if any(row['K'] != rows[0]['K'] for row in rows):
        raise ValueError('existing source-scene trainer requires shared K; mixed K unsupported')
    return manifest, rows


def sampled_points(depth, rgb, K, T, stride):
    """Pixel centers, optical-axis z in meters, OpenCV c2w; no native geometry."""
    if (depth.shape != rgb.shape[:2] or depth.ndim != 2
            or not np.issubdtype(depth.dtype, np.floating)
            or not np.isfinite(depth).all() or (depth < 0).any()):
        raise ValueError('invalid metric camera-z depth')
    h, w = depth.shape
    K, T = validate_camera(K, T, width=w, height=h)
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    valid = z > 0
    xyz = np.stack(((u-K[0, 2])*z/K[0, 0], (v-K[1, 2])*z/K[1, 1], z), -1)
    world = xyz @ T[:3, :3].T + T[:3, 3]
    return world[valid], rgb[::stride, ::stride][valid]


def prepare(capture, out, *, stride=8, voxel_m=0.005):
    from PIL import Image
    from plyfile import PlyData, PlyElement
    if type(stride) is not int or stride < 1 or not np.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError('positive fixed sampling stride and metric voxel required')
    capture, out = Path(capture), Path(out)
    manifest, rows = capture_rows(capture)
    out.mkdir(parents=True, exist_ok=False)
    scene = out/'scene'; images = scene/'dslr/resized_undistorted_images'
    images.mkdir(parents=True)
    points, colors, frames = [], [], []
    for row in rows:
        rgb = np.asarray(Image.open(capture/row['rgb']).convert('RGB'))
        if rgb.shape != (manifest['height'], manifest['width'], 3):
            raise ValueError('TRAIN RGB dimensions differ from declared camera')
        depth = np.load(capture/row['depth_m'], allow_pickle=False)
        xyz, color = sampled_points(depth, rgb, row['K'], row['T_world_from_camera'], stride)
        points.append(xyz); colors.append(color)
        name = row['frame_id']+'.png'
        shutil.copyfile(capture/row['rgb'], images/name)
        frames.append({'name': name, 'w2c': np.linalg.inv(row['T_world_from_camera']).tolist(),
                       'source_frame': row, 'image_sha256': sha(images/name)})
    xyz, rgb = np.concatenate(points), np.concatenate(colors)
    # Deterministic first observed sample per metric voxel, never quality-ranked.
    _, keep = np.unique(np.floor(xyz/voxel_m).astype(np.int64), axis=0, return_index=True)
    keep.sort(); xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) < 4 or not np.isfinite(xyz).all():
        raise ValueError('insufficient finite TRAIN RGB-D initialization')
    ply = np.empty(len(xyz), dtype=[(k, 'f4') for k in 'xyz']+
                   [(k, 'u1') for k in ('red', 'green', 'blue')])
    for i, key in enumerate('xyz'): ply[key] = xyz[:, i]
    for i, key in enumerate(('red', 'green', 'blue')): ply[key] = rgb[:, i]
    PlyData([PlyElement.describe(ply, 'vertex')]).write(str(out/'init_points.ply'))
    inputs = {'kind': 'native_public_train_rgbd_v1', 'capture_id': manifest['capture_id'],
              'capture_manifest_sha256': sha(capture/'capture_manifest.json'),
              'calibration': {'K': rows[0]['K'], 'width': manifest['width'], 'height': manifest['height']},
              'frames': frames, 'heldout_frames_read': 0, 'native_assets_read': 0,
              'input_assistance': 'known OpenCV poses and metric camera-z depth',
              'robot_mode': manifest['robot_mode']}
    save(scene/'native_training_inputs.json', inputs)
    initialization = {'kind': 'native_train_rgbd_initialization_v1',
                      'source_input_sha256': sha(scene/'native_training_inputs.json'),
                      'capture_manifest_sha256': inputs['capture_manifest_sha256'],
                      'init_ply_sha256': sha(out/'init_points.ply'), 'n_points': len(xyz),
                      'stride': stride, 'voxel_m': voxel_m,
                      'selection': 'all_train_frames_regular_pixel_stride_first_sample_per_voxel',
                      'frame_ids': [r['frame_id'] for r in rows], 'heldout_frames_read': 0,
                      'native_assets_read': 0, 'metric_scale_fitted': False}
    save(out/'init_manifest.json', initialization)
    validate_initialization(scene, out/'init_points.ply', out/'init_manifest.json', capture)
    return initialization


def validate_initialization(scene, init_ply, init_manifest, capture):
    from plyfile import PlyData
    scene, init_ply, init_manifest, capture = map(Path, (scene, init_ply, init_manifest, capture))
    manifest, rows = capture_rows(capture)
    inputs_path = scene/'native_training_inputs.json'
    inputs, init = json.loads(inputs_path.read_text()), json.loads(init_manifest.read_text())
    if (init.get('kind') != 'native_train_rgbd_initialization_v1'
            or inputs.get('kind') != 'native_public_train_rgbd_v1'
            or init.get('heldout_frames_read') != 0 or init.get('native_assets_read') != 0
            or inputs.get('heldout_frames_read') != 0 or inputs.get('native_assets_read') != 0
            or init.get('metric_scale_fitted') is not False):
        raise ValueError('unrecognized native TRAIN-only initialization provenance')
    if (sha(inputs_path) != init['source_input_sha256']
            or sha(capture/'capture_manifest.json') != init['capture_manifest_sha256']
            or inputs['capture_manifest_sha256'] != init['capture_manifest_sha256']
            or sha(init_ply) != init['init_ply_sha256'] or init_ply.is_symlink()):
        raise ValueError('source/init manifest or PLY bytes changed')
    if inputs['capture_id'] != manifest['capture_id'] or [r['source_frame'] for r in inputs['frames']] != rows:
        raise ValueError('TRAIN camera lineage changed')
    if (init.get('frame_ids') != [row['frame_id'] for row in rows]
            or init.get('selection') != 'all_train_frames_regular_pixel_stride_first_sample_per_voxel'
            or type(init.get('stride')) is not int or init['stride'] < 1
            or not np.isfinite(init.get('voxel_m', float('nan'))) or init['voxel_m'] <= 0
            or init.get('n_points', 0) < 4
            or len(PlyData.read(str(init_ply))['vertex']) != init['n_points']):
        raise ValueError('initialization point count or sampling declaration changed')
    expected = {'K': rows[0]['K'], 'width': manifest['width'], 'height': manifest['height']}
    if inputs['calibration'] != expected:
        raise ValueError('calibration changed')
    names = [row['frame_id']+'.png' for row in rows]
    images = scene/'dslr/resized_undistorted_images'
    if sorted(p.name for p in images.iterdir()) != sorted(names):
        raise ValueError('extra/missing staged TRAIN image')
    for frame, row in zip(inputs['frames'], rows):
        path = images/frame['name']
        if (frame['name'] != row['frame_id']+'.png' or path.is_symlink()
                or sha(path) != row['rgb_sha256'] or sha(path) != frame['image_sha256']
                or not np.array_equal(np.asarray(frame['w2c']), np.linalg.inv(row['T_world_from_camera']))):
            raise ValueError('staged image or camera changed')
    return inputs, init


def train_source(capture, out, *, smoke_iters=16, pilot_iters=500, stride=8, voxel_m=0.005):
    """Run the canonical trainer twice; real smoke precedes the bounded DEV pilot."""
    if smoke_iters != 16 or pilot_iters != 500:
        raise ValueError('first DEV source-GS recipe is frozen at16 smoke/500 pilot iterations')
    from agents.recon.gsplat_train import main as train
    out = Path(out); started = time.monotonic()
    init = prepare(capture, out, stride=stride, voxel_m=voxel_m)
    reports = {}
    for phase, iters in [('smoke', smoke_iters), ('pilot', pilot_iters)]:
        dest = out/phase; dest.mkdir()
        train(['--scene-dir', str(out/'scene'), '--init-ply', str(out/'init_points.ply'),
               '--native-init-manifest', str(out/'init_manifest.json'), '--native-capture', str(capture),
               '--out', str(dest/'scene.ply'), '--iters', str(iters), '--holdout-every', '10', '--seed', '42'])
        reports[phase] = {'scene_ply_sha256': sha(dest/'scene.ply'),
                          'train_report_sha256': sha(dest/'train_report.json')}
    result = {'kind': 'native_source_scene_gs_dev_v1', 'status': 'BUILT',
              'capture_manifest_sha256': init['capture_manifest_sha256'],
              'initialization': init, 'phases': reports, 'elapsed_s': time.monotonic()-started,
              'trainer': 'agents.recon.gsplat_train', 'heldout_evaluation': 'NOT_RUN',
              'source_gaussian': 'MEASURED', 'factorized_gaussian': 'NOT_RUN',
              'policy_observation_arm': 'NOT_RUN', 'harmonizer': 'NOT_RUN', 'paper_ready': False,
              'limitation': 'Six TRAIN views; static parked robot remains baked; no removal, completion, or movable GS yet.'}
    save(out/'gs_build_manifest.json', result)
    return result


def isolated_command(*, code, capture, out, python, dependency_root, bubblewrap, forbidden, cpu=False):
    """Reuse SR1's actual empty-root allowlist; do not mount native API code."""
    code, out, python, dependency_root = map(Path, (code, out, python, dependency_root))
    env_root = python.absolute().parent.parent
    site = env_root/'lib/python3.10/site-packages'
    if not site.is_dir():
        raise ValueError('existing gsplat Python3.10 runtime required')
    if not (site/'gsplat/csrc.so').is_file():
        raise ValueError('precompiled gsplat extension required; implicit JIT build is not admitted')
    for native in ('robocasa', 'robosuite', 'omnigibson'):
        if (site/native).exists():
            raise ValueError('GS runtime exposes native reference package')
    names = ('agents/__init__.py', 'agents/core/__init__.py', 'agents/core/common.py',
             'agents/recon/__init__.py', 'agents/recon/gsplat_train.py',
             'robo/__init__.py', 'robo/roundtrip/capture.py', 'robo/roundtrip/gaussian_build.py')
    mounts = {}
    for name in names:
        path = code/name
        if not path.is_file() or path.is_symlink():
            raise ValueError('missing/symlink constructor source')
        mounts[str(path)] = '/code/'+name
    mounts[str(env_root)] = str(env_root)
    mounts[str(dependency_root)] = '/dependencies'
    for name in ('/usr/lib', '/usr/lib64', '/opt/nvidia-driver/lib', '/etc/ld.so.cache'):
        if Path(name).exists(): mounts[name] = name
    devices = []
    if not cpu:
        minors = slurm_device_minors(os.environ)
        if len(minors) != 1: raise ValueError('one allocated training GPU required')
        devices = [f'/dev/nvidia{next(iter(minors))}', '/dev/nvidiactl', '/dev/nvidia-uvm']
        if Path('/dev/nvidia-uvm-tools').exists(): devices.append('/dev/nvidia-uvm-tools')
    bootstrap = '''import importlib.util,json,os,pathlib,runpy,sys
sys.path.insert(0,'/dependencies');sys.path.insert(0,'/code')
for module in ('robocasa','robosuite','omnigibson','robo.roundtrip.adapters'):
 assert importlib.util.find_spec(module) is None, 'native API exposed: '+module
for path in json.loads(sys.argv[1]):
 assert not pathlib.Path(path).exists(), 'private tree exposed'
receipt={'native_api_unavailable':True,'private_roots_unavailable':True,'network':'unshared','cpu':sys.argv[2]=='cpu'}
if sys.argv[2]!='cpu':
 import torch,gsplat,hashlib
 from gsplat import csrc as extension
 assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'isolated CUDA allocation unavailable'
 receipt.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,gsplat=gsplat.__version__,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],precompiled_extension_sha256=hashlib.sha256(pathlib.Path(extension.__file__).read_bytes()).hexdigest())
 assert torch.arange(8,device='cuda').sum().item()==28
pathlib.Path('/output/isolation_preflight.json').write_text(json.dumps(receipt,indent=2))
sys.argv=['gaussian_build','--capture','/capture','--out','/output/build','--phase','prepare' if sys.argv[2]=='cpu' else 'train']
runpy.run_module('robo.roundtrip.gaussian_build',run_name='__main__')
'''
    command = constructor_command(bubblewrap=bubblewrap, public_capture=capture, output=out,
        runtime_mounts=mounts, forbidden_roots=forbidden, device_paths=devices,
        argv=[str(python), '-s', '-c', bootstrap, json.dumps([str(Path(p).resolve()) for p in forbidden]), 'cpu' if cpu else 'gpu'])
    idx = command.index('--chdir')
    env = {'PYTHONNOUSERSITE':'1', 'PYTHONDONTWRITEBYTECODE':'1', 'PYTHONHASHSEED':'42',
           'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4',
           'SIMANY_ROOT':'/code', 'SIMANY_SCENE':'opaque_native_train', 'SIMANY_MESH_SRC':'derived',
           'TORCH_EXTENSIONS_DIR':'/output/runtime_cache/torch', 'CUDA_CACHE_PATH':'/output/runtime_cache/cuda'}
    if not cpu: env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    extra = ['--symlink','usr/lib','/lib','--symlink','usr/lib64','/lib64']
    for key,value in env.items(): extra += ['--setenv',key,value]
    command[idx:idx] = extra
    return command, mounts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'train', 'isolated-prepare', 'isolated-train'], default='train')
    parser.add_argument('--code', type=Path)
    parser.add_argument('--training-python', type=Path)
    parser.add_argument('--dependency-root', type=Path)
    parser.add_argument('--bubblewrap', type=Path)
    parser.add_argument('--forbidden', action='append', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.phase.startswith('isolated-'):
        if not all((args.code,args.training_python,args.dependency_root,args.bubblewrap,args.forbidden,args.receipt)):
            raise ValueError('explicit runtime and private/native forbidden roots required')
        args.out.mkdir(parents=True, exist_ok=False)
        command,mounts = isolated_command(code=args.code,capture=args.capture,out=args.out,
            python=args.training_python,dependency_root=args.dependency_root,bubblewrap=args.bubblewrap,
            forbidden=args.forbidden,cpu=args.phase=='isolated-prepare')
        save(args.receipt, {'command':command,'source_hashes':{v:sha(k) for k,v in mounts.items() if v.startswith('/code/')},
            'capture_manifest_sha256':sha(args.capture/'capture_manifest.json'),'mounts':mounts,
            'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'paper_ready':False})
        raise SystemExit(subprocess.run(command).returncode)
    result = prepare(args.capture, args.out) if args.phase == 'prepare' else train_source(args.capture, args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
