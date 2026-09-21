"""PhysicalView viewer for native simulator snapshots; no policy dependencies."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--port', type=int, default=8092)
    args = ap.parse_args()
    server = viser.ViserServer(host='0.0.0.0', port=args.port)
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## SimAnyRoom native reconstruction\nB3 target-only reconstruction in the original native room.\nRecorded actions are re-executed in MuJoCo; this is a presentation replay, not a new policy evaluation.')
    status = server.gui.add_markdown('Loading native scene…')
    frame = server.gui.add_image(np.zeros((480, 640, 3), dtype=np.uint8), label='Native textured camera')
    for action in ['Pause', 'Play', 'Reset']:
        button = server.gui.add_button(action)
        @button.on_click
        def command(event, action=action):
            p = args.root / 'command.tmp'
            p.write_text(json.dumps({'action': action, 'nonce': time.time_ns()}))
            p.replace(args.root / 'command.json')
    handles = {}
    while not (args.root / 'meshes.json').exists():
        time.sleep(.5)
    for item in json.loads((args.root / 'meshes.json').read_text()):
        mesh = trimesh.load(args.root / item['path'], force='mesh')
        body = item['body']
        handles[body] = server.scene.add_frame('/native/' + str(body), show_axes=False)
        server.scene.add_mesh_trimesh('/native/' + str(body) + '/mesh', mesh)
    @server.on_client_connect
    def connected(client):
        target = np.array(json.loads((args.root / 'import_receipt.json').read_text())['position_m'])
        client.camera.position = tuple(target + np.array([1.1, -1.5, 1.0]))
        client.camera.look_at = tuple(target)
    while True:
        try:
            state = json.loads((args.root / 'live.json').read_text())
            for body, pose in state['poses'].items():
                if int(body) in handles:
                    handles[int(body)].position = tuple(pose[:3])
                    handles[int(body)].wxyz = tuple(pose[3:])
            frame.image = np.asarray(Image.open(args.root / 'frame.jpg').convert('RGB'))
            status.content = f"Step {state['step']}/{state['horizon']} · MuJoCo time {state['time']:.2f}s · native success {state['success']}\n\n3D: colored geometry; image: original native materials. Drag to orbit, scroll to zoom."
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        time.sleep(.12)


if __name__ == '__main__':
    main()
