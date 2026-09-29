"""Render a saved MuJoCo shooting trace using the original GS scene and 3D VFX.

Requires the scene-specific refined GS/LaMa artifacts and contact rollout from
the holding/shooting capture. This is a replay renderer, not a dataset installer.
"""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from gsplat import rasterization

from agents.core import common as C
from physicalview.phiview_scene import FlyCamera, subset
from physicalview.projectile_effects import particles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--rollout', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Read-only physics replay: never rerun capture code or rewrite source assets.
    model = mujoco.MjModel.from_xml_path(str(args.rollout/'shooting-scene.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bottle_id = model.body('obj_19').id
    original_position = data.xpos[bottle_id].copy()
    original_rotation = data.xmat[bottle_id].reshape(3, 3).copy()
    trace = json.loads((args.rollout/'shooting-trace.json').read_text())
    frames, summary = trace['frames'], trace['summary']
    states = np.load(args.rollout/'shooting-states.npz')
    assert len({h['bullet'] for h in summary['hits']}) >= 2
    original = C.load_gaussians()
    mask = np.ones(len(original['means']), bool)
    mask[np.load(args.assets/'removal-indices.npy')] = False

    def load_gs(path):
        data = np.load(path)
        return {k: int(v) if k == 'sh_degree' else torch.as_tensor(v, dtype=torch.float32, device='cuda') for k, v in data.items()}

    background = C.cat_gaussians([subset(original, torch.as_tensor(mask, device='cuda')), load_gs(args.assets/'table-fill-gaussians.npz')])
    obj = load_gs(args.assets/'refined-bottle-gaussians.npz')
    camera = np.load(args.assets/'render-camera.npz')
    center = np.asarray(summary['start_position'])
    pos = center+(camera['position']-center)*.77
    look = center+[0, .09, -.04]
    direction = look-pos; direction /= np.linalg.norm(direction)
    cam = FlyCamera(pos, float(np.arctan2(direction[1], direction[0])), float(np.arcsin(direction[2])), 55)
    width, height = 1752, 1168
    wc, intr = cam.matrices((width, height))
    viewmats = torch.as_tensor(wc, dtype=torch.float32, device='cuda')[None]
    ks = torch.as_tensor(intr, dtype=torch.float32, device='cuda')[None]
    receipts = {'scene': 'fb5a96b1a2', 'resolution': [width, height], 'device': torch.cuda.get_device_name(),
                'w2c': wc.tolist(), 'K': intr.tolist(), 'physics_summary': summary,
                'method': 'Original GS room and refined original GS bottle, fitted 3D LaMa table fill, real MuJoCo two-shot trajectory; decorative world-space tracer curls and contact sparks',
                'frames': []}

    def render(i):
        row = frames[i]
        data.qpos[:] = states['qpos'][i]; data.qvel[:] = states['qvel'][i]
        data.time = row['time']
        mujoco.mj_forward(model, data)
        transform = np.eye(4)
        transform[:3, :3] = data.xmat[bottle_id].reshape(3, 3)@original_rotation.T
        transform[:3, 3] = data.xpos[bottle_id]-transform[:3, :3]@original_position
        effects = particles(frames, summary['hits'], row['time'])
        parts = [background, C.transform_gaussians(obj, transform)]
        if len(effects['means']):
            count = len(effects['means'])
            gs = {k: torch.as_tensor(effects[k], device='cuda') for k in ('means', 'scales', 'opacities')}
            gs['quats'] = torch.tensor([1., 0, 0, 0], device='cuda').repeat(count, 1)
            gs['sh'] = torch.as_tensor((effects['colors']-.5)/.28209479177387814, device='cuda')[:, None, :]
            gs['sh_degree'] = 0
            parts.append(gs)
        g = C.cat_gaussians(parts)
        with torch.inference_mode():
            pixels, _, _ = rasterization(means=g['means'], quats=g['quats'], scales=g['scales'], opacities=g['opacities'], colors=g['sh'], sh_degree=3,
                viewmats=viewmats, Ks=ks, width=width, height=height, near_plane=.01, far_plane=100., packed=False, render_mode='RGB+ED', rasterize_mode='antialiased')
        rgb = (pixels[0, :, :, :3].clamp(0, 1).cpu().numpy()*255+.5).astype(np.uint8)
        # Projectile heads are luminous GS particles centered on the measured
        # collision spheres. Avoid painting an unlit mesh over their bright cores.
        receipts['frames'].append({'source_tick': i, 'simulation_time': row['time'], 'effect_particles': len(effects['means'])})
        return rgb

    indices = [0, 52, 58, 62, 69, 76, 80, 100, 150, 360] if args.preview else list(range(160))+list(range(160, len(frames), 4))
    proc = clean_proc = None
    if not args.preview:
        proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.out/'shooting-two-bullets.mp4')], stdin=subprocess.PIPE)
        clean_proc = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.out/'shooting-two-bullets-clean.mp4')], stdin=subprocess.PIPE)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
    try:
        for n, i in enumerate(indices):
            rgb = render(i)
            if args.preview or i in (0, 58, 62, 69, 76, 80, 100, 156, 360):
                Image.fromarray(rgb).save(args.out/f'frame-{i:03d}.png')
            if proc:
                clean_proc.stdin.write(rgb.tobytes())
                picture = Image.fromarray(rgb)
                layer = Image.new('RGBA', picture.size)
                draw = ImageDraw.Draw(layer)
                draw.rounded_rectangle((36, 32, 708, 135), radius=16, fill=(12, 18, 26, 205))
                draw.text((59, 47), 'PHIVIEW  /  TWO-SHOT DEMO', font=title_font, fill=(247, 249, 252))
                draw.text((60, 91), '0.25x SLOW MOTION' if i < 160 else '1x  /  SETTLE', font=font, fill=(183, 197, 211))
                draw.text((1270, 48), 'MUJOCO CONTACTS + GS', font=font, fill=(247, 249, 252), stroke_width=2, stroke_fill=(15, 20, 25))
                picture = Image.alpha_composite(picture.convert('RGBA'), layer).convert('RGB')
                proc.stdin.write(picture.tobytes())
            if n % 30 == 0: print('rendered', n, '/', len(indices), flush=True)
    finally:
        if proc:
            proc.stdin.close()
            clean_proc.stdin.close()
            exits = [proc.wait(), clean_proc.wait()]
            if any(exits): raise RuntimeError(f'Video encoder failed: {exits}')
    receipts.update({'output_fps': 30, 'output_frames': len(indices), 'output_seconds': len(indices)/30, 'time_remapping': 'source ticks 0..159 at 0.25x, remaining ticks at 1x; no reversed or repeated impacts'})
    (args.out/'render-receipts.json').write_text(json.dumps(receipts, indent=2))


if __name__ == '__main__':
    main()
