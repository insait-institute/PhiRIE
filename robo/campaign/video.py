"""RGB-only capture-video -> COLMAP -> reference 3DGS, with explicit scale status.

Only a TRAIN capture video is accepted. Held-out poses/views are evaluator-side.
The chosen COLMAP executable and official Gaussian source must be pinned/admitted.
Metric calibration is an additional measured step, never recovered from hidden GT.
"""
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from .core import load, receipt, save


def run(config, inputs, out):
    from .models import source
    src = source(config); out = Path(out).absolute()
    if config.get('capture_role') != 'train_only' or config.get('camera_source') != 'estimated':
        raise ValueError('raw video frontend requires TRAIN capture and estimated cameras')
    # Upstream convert.py assembles shell commands; refuse unsafe paths rather than
    # pretending our subprocess argument list makes those nested commands safe.
    for p in (out, Path(src['root']), Path(config['colmap'])):
        if not re.fullmatch(r'[A-Za-z0-9_./+-]+', str(p)):
            raise ValueError('upstream COLMAP converter requires shell-safe absolute paths')
    fps = float(config.get('fps', 3)); count = int(config.get('max_frames', 180))
    if not 0 < fps <= 60 or not 4 <= count <= 2000:
        raise ValueError('declare bounded frame extraction budget')
    scene = out/'colmap_scene'; frames = scene/'input'; frames.mkdir(parents=True)
    commands = [[config.get('ffmpeg', 'ffmpeg'), '-nostdin', '-v', 'error', '-i', inputs['video'],
                 '-vf', f'fps={fps}', '-frames:v', str(count), str(frames/'%06d.png')],
                [sys.executable, str(Path(src['root'])/'convert.py'), '-s', str(scene),
                 '--colmap_executable', config['colmap']]]
    iterations = int(config.get('iterations', 30000))
    if iterations < 1: raise ValueError('positive GS iteration budget required')
    commands.append([sys.executable, str(Path(src['root'])/'train.py'), '-s', str(scene),
                     '-m', str(out/'gaussian_model'), '--iterations', str(iterations),
                     '--save_iterations', str(iterations), '--disable_viewer'])
    save(out/'commands.json', {'commands': commands, 'source': src, 'video': receipt(inputs['video']),
                              'heldout_access': False, 'estimated_metric_scale': False})
    for index, cmd in enumerate(commands):
        with (out/f'stage-{index}.log').open('x') as log:
            subprocess.run(cmd, cwd=src['root'], check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=float(config.get('timeout_s', 14400)))
        if index == 0 and len(list(frames.glob('*.png'))) < 4:
            raise ValueError('insufficient extracted views')
    ply = out/'gaussian_model'/'point_cloud'/f'iteration_{iterations}'/'point_cloud.ply'
    if not ply.is_file(): raise ValueError('official Gaussian trainer did not publish requested iteration')
    from plyfile import PlyData
    vertices = PlyData.read(ply)['vertex'].data
    if not len(vertices) or not np.isfinite(np.column_stack([vertices[k] for k in ('x','y','z')])).all():
        raise ValueError('invalid reconstructed Gaussian positions')
    np.save(out/'gaussian_ids.npy', np.arange(len(vertices), dtype=np.int64))
    save(out/'reconstruction.json', {'source':src, 'video':receipt(inputs['video']),
        'camera_source':'estimated_from_rgb', 'sensor':'rgb_video', 'metric_scale':'NOT_ESTABLISHED',
        'units':'arbitrary SfM gauge; must bind an observation-derived scale before physical import',
        'gaussians':receipt(ply), 'gaussian_ids':receipt(out/'gaussian_ids.npy'),
        'sfm_model':str(scene/'sparse'/'0'), 'training_views':len(list(frames.glob('*.png'))),
        'claim':'RGB Gaussian reconstruction only; physical room construction NOT_RUN'})
    return {'gaussians':ply, 'gaussian_ids':out/'gaussian_ids.npy','reconstruction':out/'reconstruction.json'}
