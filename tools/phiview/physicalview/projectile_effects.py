"""World-space tracer wakes and contact sparks for server-side GS rendering.

Effects consume a physics trace and never modify it. The bright core follows
the measured trajectory; curled side wisps and sparks are decorative particles,
not collision geometry. All returned positions are in scene coordinates, so
the Gaussian renderer supplies normal scene occlusion.
"""
from __future__ import annotations

import numpy as np

PALETTE = ((1., .38, .035), (.035, .72, 1.))


def particles(frames, impacts, now, *, lifetime=.28):
    """Return numpy Gaussian attributes, using only samples at or before now."""
    xyz, scales, colors, alpha = [], [], [], []

    def add(points, radius, rgb, opacity):
        points = np.asarray(points).reshape(-1, 3)
        count = len(points)
        xyz.extend(points)
        scales.extend(np.broadcast_to(np.asarray(radius), (count,)))
        colors.extend(np.broadcast_to(np.asarray(rgb), (count, 3)))
        alpha.extend(np.broadcast_to(np.asarray(opacity), (count,)))

    histories = {}
    for frame in frames:
        if now-lifetime <= frame['time'] <= now+1e-8:
            for name, point in frame['projectiles'].items():
                histories.setdefault(name, []).append((frame['time'], point))
    hit_times = {h['bullet']: h['time'] for h in impacts}
    for name, samples in histories.items():
        if now > hit_times.get(name, float('inf'))+.45:
            continue
        color = np.asarray(PALETTE[int(name.rsplit('_', 1)[1]) % len(PALETTE)])
        for (t0, p0), (t1, p1) in zip(samples[:-1], samples[1:]):
            p0, p1 = np.asarray(p0), np.asarray(p1)
            delta = p1-p0
            count = min(32, max(2, int(np.linalg.norm(delta)/.003)+1))
            u = np.linspace(0, 1, count, endpoint=False)
            age = now-(t0+(t1-t0)*u)
            fade = np.maximum(0, 1-age/lifetime)**1.3
            points = p0+u[:, None]*delta
            add(points, .003, color*1.6+.12, fade*.85)
            add(points, .009, color, fade*.17)
            # A widening corkscrew wake produces the requested flicking tail.
            axis = delta/max(np.linalg.norm(delta), 1e-9)
            side = np.cross(axis, [0, 0, 1.])
            side /= max(np.linalg.norm(side), 1e-9)
            up = np.cross(axis, side)
            radius = .022*np.sin(np.pi*np.clip(age/lifetime, 0, 1))
            phase = age*105+t0*11
            curl = points+radius[:, None]*(np.sin(phase)[:, None]*side+np.cos(phase)[:, None]*up)
            add(curl, .0025, color*1.2, fade*.50)
        # The head remains centered on the actual collision projectile.
        point = np.asarray(samples[-1][1])
        add(point, .020, color, .26)
        add(point, .010, [1.7, 1.65, 1.5], .92)

    for hit in impacts:
        age = now-hit['time']
        if not 0 <= age <= .22:
            continue
        index = int(hit['bullet'].rsplit('_', 1)[1])
        color = np.asarray(PALETTE[index % len(PALETTE)])
        rng = np.random.default_rng(401+index)
        directions = rng.normal(size=(44, 3))
        directions[:, 2] = np.abs(directions[:, 2])*.8
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        speed = rng.uniform(.4, 1.5, size=(44, 1))
        fade = (1-age/.22)**1.5
        points = np.asarray(hit['position'])+directions*speed*age+[0, 0, -1.5*age**2]
        add(points, .003, color*1.8+.2, fade*.9)
        add(points, .010, color, fade*.13)
        if age < .045:
            add(hit['position'], .025+age*.6, [1.8, 1.6, 1.1], .65*(1-age/.045))

    return {'means': np.asarray(xyz, dtype=np.float32).reshape(-1, 3),
            'scales': np.repeat(np.asarray(scales, dtype=np.float32)[:, None], 3, axis=1),
            'colors': np.asarray(colors, dtype=np.float32).reshape(-1, 3),
            'opacities': np.asarray(alpha, dtype=np.float32)}
