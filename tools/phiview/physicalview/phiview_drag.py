"""Explicit manual placement of a selected rigid body from image-space drags."""
from __future__ import annotations

import numpy as np


def plane_point(w2c, K, wh, xy, z):
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1):
        raise ValueError('Drag point outside image')
    c2w = np.linalg.inv(w2c)
    direction = c2w[:3, :3] @ np.linalg.solve(K, [xy[0]*wh[0], xy[1]*wh[1], 1.])
    if abs(direction[2]) < 1e-6:
        raise ValueError('Use a view looking down toward the object to move it')
    distance = (z-c2w[2, 3])/direction[2]
    if distance <= 0:
        raise ValueError('Drag plane is behind the camera')
    return c2w[:3, 3]+distance*direction


class ObjectDrag:
    def __init__(self, demo):
        self.demo = demo
        self.active = None
        self.generation = 0

    def begin(self, frame_id, xy):
        d = self.demo
        name = d.selected_required()
        if name not in d.physics.available:
            raise ValueError('Make the selected object simulatable first')
        frame = d.frames.get(frame_id)
        if frame is None:
            raise ValueError('Displayed frame expired; start dragging again')
        mask, _, w2c, K = frame
        h, w = mask.shape
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy >= 1):
            raise ValueError('Drag point outside image')
        if int(mask[int(xy[1]*h), int(xy[0]*w)]) != d.scene.ids[name]:
            raise ValueError('Drag directly on the selected object')
        qa, _ = d.physics.addresses(name)
        position = d.physics.data.qpos[qa:qa+3].copy()
        hit = plane_point(w2c, K, (w, h), xy, position[2])
        d.policy.stop(); d.physics.running = False; d.physics.plan = []
        d.physics.enable([name]); d.mode = 'simulation'
        d.keys = []; d.look[:] = 0
        self.generation += 1
        self.active = {'id': self.generation, 'name': name, 'position': position,
                       'offset': position-hit, 'w2c': w2c, 'K': K, 'wh': (w, h)}
        return self.generation

    def move(self, token, xy):
        g = self.active
        if not g or token != g['id']:
            return  # A delayed packet must never move an object after release.
        d = self.demo
        position = plane_point(g['w2c'], g['K'], g['wh'], xy, g['position'][2])+g['offset']
        if np.linalg.norm(position-g['position']) > 2:
            raise ValueError('Drag is limited to two metres; release and drag again')
        qa, da = d.physics.addresses(g['name'])
        d.physics.data.qpos[qa:qa+3] = position
        d.physics.data.qvel[da:da+6] = 0
        import mujoco
        mujoco.mj_forward(d.physics.model, d.physics.data)
        d.physics.running = False

    def end(self, token=None):
        if token is None or (self.active and token == self.active['id']):
            self.active = None
