"""Server-side orbit, pan and dolly for an image-only mouse controller."""
from __future__ import annotations

import numpy as np


def navigate(camera, message, depth=None):
    # Validate the whole gesture before changing any camera state.
    orbit = np.asarray(message.get('orbit', [0, 0]), dtype=float)
    pan = np.asarray(message.get('pan', [0, 0]), dtype=float)
    zoom = float(message.get('zoom', 0))
    if (orbit.shape != (2,) or pan.shape != (2,) or not np.isfinite(orbit).all()
            or not np.isfinite(pan).all() or not np.isfinite(zoom)):
        raise ValueError('Invalid camera gesture')
    orbit, pan = np.clip(orbit, -1, 1), np.clip(pan, -1, 1)
    distance = camera.orbit_distance
    if distance is None:
        distance = 2.
        if depth is not None:
            h, w = depth.shape
            sample = depth[max(0, h//2-3):h//2+4, max(0, w//2-3):w//2+4]
            valid = sample[np.isfinite(sample) & (sample > .05)]
            if valid.size:
                distance = float(np.median(valid))
    distance = float(np.clip(distance, .05, 100))
    target = camera.position + camera.forward()*distance
    yaw = camera.yaw - float(orbit[0])*2*np.pi
    pitch = float(np.clip(camera.pitch-float(orbit[1])*2*np.pi, -np.pi/2+.02, np.pi/2-.02))
    forward = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.])
    up = np.cross(right, forward)
    # Deltas are fractions of image height, independent of browser resolution.
    target += (-pan[0]*right + pan[1]*up)*2*distance*np.tan(np.deg2rad(camera.fov)/2)
    distance = float(np.clip(distance*np.exp(np.clip(zoom, -1, 1)*.8), .05, 100))
    camera.position = target-forward*distance
    camera.yaw, camera.pitch, camera.orbit_distance = yaw, pitch, distance
