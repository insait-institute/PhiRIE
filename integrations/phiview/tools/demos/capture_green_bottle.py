"""Capture real GS frames, manual placement, robot contact and projectile physics.

Run in the compatible GPU environment. The destination must be new; source scene
and session files are copied, and no scientific artifacts are overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
from physicalview.phiview import Demo, json_default


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default)+'\n')


def capture(args):
    args.out.mkdir(parents=True, exist_ok=False)
    runtime, package = args.out/'runtime', args.out/'green-bottle-demo'
    runtime.mkdir(); package.mkdir()
    shutil.copy2(args.session/'active-scene.json', runtime/'active-scene.json')
    shutil.copytree(args.session/'interactive_objects', runtime/'interactive_objects')
    d = Demo(SimpleNamespace(out=str(runtime), config=str(args.config), scene=args.scene,
        width=1752, height=1168, fps=30, demo=True, no_demo_prepare=False))
    receipts = {}
    try:
        d.render(); d.preparation.advance()
        if d.preparation.status['state'] != 'ready':
            raise RuntimeError('Prepare the green bottle in the source session first')
        name = d.selected
        qa, _ = d.physics.addresses(name)
        original_position = d.physics.data.qpos[qa:qa+3].copy()
        source_camera_position = d.camera.position.copy()

        def view(wide=False):
            d.camera.position = original_position+(source_camera_position-original_position)*(.96 if wide else .72)
            target = original_position+[0, -.12, .20 if wide else .05]
            direction = target-d.camera.position; direction /= np.linalg.norm(direction)
            d.camera.yaw = float(np.arctan2(direction[1], direction[0]))
            d.camera.pitch = float(np.arcsin(direction[2])); d.camera.fov = 60 if wide else 58
            d.camera_view = 'free'; d.mode = 'simulation'; d.highlight = False

        def image():
            # Paper frames contain no selection overlays; geometry is rendered by GS.
            selected = d.selected; d.selected = None
            try:
                return d.render()
            finally:
                d.selected = selected

        def shot(filename, method, **extra):
            Image.fromarray(image()).save(package/filename)
            w2c, K = d.camera.matrices(d.wh)
            receipts[filename] = {'method': method, 'simulation_time': float(d.physics.data.time),
                'position': d.physics.data.qpos[d.physics.addresses(name)[0]:][:3].copy(),
                'w2c': w2c, 'K': K, **extra}

        view()
        shot('01-original-position.png', 'Original captured Gaussian position')
        for i, offset in enumerate(([0, -.35, 0], [.24, -.23, 0], [-.20, -.15, 0]), 2):
            d.physics.reset(); d.selected = name; d.render()
            mask, _, w2c, K = d.frames[d.frame_id]
            y, x = np.argwhere(mask == d.scene.ids[name])[len(np.argwhere(mask == d.scene.ids[name]))//2]
            begin = d.execute({'op': 'move_begin', 'frame': d.frame_id, 'x': float(x/d.wh[0]), 'y': float(y/d.wh[1])})
            g = d.object_drag.active
            desired = original_position+offset
            world = desired-g['offset']
            camera = w2c[:3, :3]@world+w2c[:3, 3]
            pixel = K@camera; xy = pixel[:2]/pixel[2]/np.asarray(d.wh)
            d.execute({'op': 'move', 'drag': begin['drag'], 'x': float(xy[0]), 'y': float(xy[1])})
            d.execute({'op': 'move_end', 'drag': begin['drag']})
            actual = d.physics.data.qpos[d.physics.addresses(name)[0]:][:3]
            np.testing.assert_allclose(actual, desired, atol=1e-6)
            shot(f'{i:02d}-moved-position.png', 'Manual Move object control; not a policy action', offset_m=offset)

        # A separate contact-driven robot rollout. Never attach/teleport the object
        # to the gripper. IK produces actuator targets; MuJoCo moves the bottle.
        d.physics.reset(); view(wide=True); d.selected = name
        d.policy.choose('scripted_ik'); d.execute({'op': 'robot'})
        d.physics.running = True
        for _ in range(30): d.physics.step(1/30)
        d.physics.running = False
        robot_start = d.physics.data.qpos[d.physics.addresses(name)[0]:][:3].copy()
        d.execute({'op': 'robot_command', 'command': 'push to right'})
        trajectory = []; best = 0.; contact_frame = False
        for i in range(150):
            d.physics.step(1/30)
            p = d.physics.data.qpos[d.physics.addresses(name)[0]:][:3].copy()
            delta = float(np.linalg.norm(p-robot_start))
            contacts = d.physics.robot_status.get('target_contact_steps', 0)
            trajectory.append({'step': i, 'time': float(d.physics.data.time), 'position': p,
                               'displacement_m': delta, 'contact_steps': contacts})
            if contacts and not contact_frame:
                shot('05-robot-contact.png', 'Scripted IK with physical robot-object contacts; GS object and room',
                     robot=dict(d.physics.robot_status))
                contact_frame = True
            if contacts and delta > best+.005:
                best = delta
                shot('06-robot-moving-bottle.png', 'Scripted IK push; contact-driven bottle motion; GS object and room',
                     displacement_m=delta, robot=dict(d.physics.robot_status))
        assert contact_frame and best > .03, d.physics.robot_status
        dump(package/'robot-trajectory.json', trajectory)
        robot_result = {'displacement_m': best, 'controller': 'Scripted IK (no learned policy)',
                        'robot': dict(d.physics.robot_status), 'success': None}

        # Reconstruct physics without a robot for a clear shooting demonstration.
        from physicalview.phiview_sim import DemoPhysics
        if d.physics.renderer: d.physics.renderer.close()
        d.physics = DemoPhysics(d.state, runtime)
        d.physics.enable([name]); d.selected = name; view()
        d.physics.running = True
        for _ in range(30): d.physics.step(1/30)
        d.physics.running = False
        shooting_start = d.physics.data.qpos[d.physics.addresses(name)[0]:][:3].copy()
        frames, fps, fire_frame = 180, 30, 30
        video = package/'07-shooting-bottle.mp4'
        argv = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{d.wh[0]}x{d.wh[1]}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'medium', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
        process = subprocess.Popen(argv, stdin=subprocess.PIPE)
        shooting = []
        try:
            for i in range(frames):
                if i == fire_frame:
                    # Aim slightly above the center to compensate projectile gravity.
                    speed = 12.
                    distance = float(np.linalg.norm(shooting_start-d.camera.position))
                    aim = shooting_start+[0, 0, .5*9.81*((distance-.12)/speed)**2]
                    d.physics.shoot(d.camera.position, aim-d.camera.position, speed)
                if i >= fire_frame: d.physics.step(1/fps)
                frame = image(); process.stdin.write(frame.tobytes())
                p = d.physics.data.qpos[d.physics.addresses(name)[0]:][:3].copy()
                shooting.append({'frame': i, 'time': float(d.physics.data.time), 'position': p,
                    'displacement_m': float(np.linalg.norm(p-shooting_start)), 'contacts': list(d.physics.events)})
                if i in (fire_frame, fire_frame+12, frames-1):
                    Image.fromarray(frame).save(package/f'shooting-frame-{i:03d}.png')
        finally:
            process.stdin.close()
            if process.wait() != 0: raise RuntimeError('Video encoding failed')
        hits = [e for e in d.physics.events if any('phiview_ball_' in g for g in e['geoms'])
                and any(name in g for g in e['geoms'])]
        assert hits, 'No recorded projectile-bottle contact'
        dump(package/'shooting-trajectory.json', shooting)
        receipts[video.name] = {'method': 'Actual MuJoCo projectile contacts with GS-rendered bottle and room',
            'fps': fps, 'frames': frames, 'seconds': frames/fps, 'resolution': d.wh,
            'fire_frame': fire_frame, 'projectile_bottle_contacts': hits,
            'max_displacement_m': max(x['displacement_m'] for x in shooting)}
        dump(package/'capture-receipts.json', receipts)
        manifest = {'scene': args.scene, 'target': name, 'label': 'Green spray bottle',
            'source': d.manifest, 'physics': d.physics.parameters(name), 'robot_result': robot_result,
            'scope': 'Real server GS and MuJoCo renders. Curated SAM3 target, partial visible geometry, unmeasured physics, existing GT-referenced room supports. No manipulation success benchmark.',
            'files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted(package.iterdir()) if p.is_file()]}
        dump(package/'manifest.json', manifest)
        (package/'README.md').write_text('''# Green bottle demo — fb5a96b1a2

Open the PNG files directly and play `07-shooting-bottle.mp4` (6 seconds, 30 FPS).

- `01-original-position.png`: original bottle position.
- `02`–`04`: three positions produced with the Move object control.
- `05-robot-contact.png`: robot contacting the bottle.
- `06-robot-moving-bottle.png`: bottle moved by a scripted IK push and physical contacts.
- `07-shooting-bottle.mp4`: an actual simulated projectile hitting the bottle.

The bottle and room are rendered from the original full Gaussian scene on the GPU.
The robot and projectile are MuJoCo meshes composited using depth. No image synthesis,
mesh replacement for the bottle, or selection overlays are used in these captures.
The robot demonstration uses scripted IK and existing GT-referenced support geometry;
it is not a learned π0.5 success result. The bottle is a partial observed Gaussian
surface with an approximate convex collision proxy and unmeasured mass/friction.
Moving it can expose reconstruction gaps; these images do not claim scene inpainting.
Receipts contain camera matrices, positions, controller metadata and collision traces.
''')
        shutil.copy2(Path(__file__), package/'capture_green_bottle.py')
        manifest['files'] = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                             for p in sorted(package.iterdir()) if p.is_file() and p.name != 'manifest.json']
        dump(package/'manifest.json', manifest)
        zip_path = args.out/'green-bottle-demo.zip'
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for p in sorted(package.iterdir()): archive.write(p, 'green-bottle-demo/'+p.name)
        with zipfile.ZipFile(zip_path) as archive: assert archive.testzip() is None
        print(json.dumps({'package': str(package), 'zip': str(zip_path), 'bytes': zip_path.stat().st_size,
                          'robot_displacement_m': best, 'shooting_contacts': len(hits)}), flush=True)
    finally:
        d.policy.close(); d.audit.close()
        if d.physics.renderer: d.physics.renderer.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--session', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--scene', default='fb5a96b1a2_factory')
    capture(ap.parse_args())
