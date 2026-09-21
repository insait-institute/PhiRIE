"""Scripted 'interactive session' demo (venv, CPU pybullet).

Fakes a user session on the full-vocabulary twin: settle -> kinematically
drag a monitor up/over (as if grabbed by a gizmo) -> release under gravity
-> shove a chair -> lift & drop a small object. Records 30 fps link poses
AND a smooth camera path (slerp between good DSLR views); the photoreal
render happens in gsplat_sim_render.py which accepts per-frame cameras.

Usage: demo_session.py --out poses.json
"""
import argparse
import json

import numpy as np

from agents.core import common as C
from robo.sim.s7_sim import build_background, com_pose, link_pose

HZ, FPS = 240, 30


def slerp_w2c(w2c_a, w2c_b, t):
    from scipy.spatial.transform import Rotation, Slerp
    c2w_a, c2w_b = np.linalg.inv(w2c_a), np.linalg.inv(w2c_b)
    rots = Rotation.from_matrix([c2w_a[:3, :3], c2w_b[:3, :3]])
    R = Slerp([0, 1], rots)([t])[0].as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = (1 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
    return np.linalg.inv(c2w)


def main():
    import pybullet as p

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    aligneds = [json.loads(
        (C.OUT / "objects" / f"obj_{m['index']:02d}" / "aligned.json").read_text())
        for m in objects]
    kept = [(m, a) for m, a in zip(objects, aligneds) if not a.get("rejected")]
    gts = {g["object_id"]: g for g in C.load_instances()}
    bg_path = build_background([m for m, _ in kept], [a for _, a in kept], gts)

    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / HZ)
    col = p.createCollisionShape(p.GEOM_MESH, fileName=str(bg_path),
                                 flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    p.changeDynamics(p.createMultiBody(0, col), -1, lateralFriction=0.6)

    bodies, labels = {}, {}
    for m, a in kept:
        name = f"obj_{m['index']:02d}"
        odir = C.OUT / "objects" / name
        ph = json.loads((odir / "physics.json").read_text())
        s, R, t = C.decompose_similarity(np.array(a["T"]))
        q = C.rot_to_quat_wxyz(R)
        bid = p.loadURDF(str(odir / "object.urdf"), basePosition=t,
                         baseOrientation=[q[1], q[2], q[3], q[0]],
                         flags=p.URDF_USE_INERTIA_FROM_FILE)
        p.changeDynamics(bid, -1, lateralFriction=ph["friction"],
                         restitution=ph["restitution"],
                         linearDamping=0.05, angularDamping=0.05)
        bodies[name] = bid
        labels[name] = m["label"]

    frames = []
    spf = HZ // FPS

    def record():
        fr = {}
        for n, b in bodies.items():
            lp, lq = link_pose(p, b)  # xyzw
            fr[n] = [lp, [lq[3], lq[0], lq[1], lq[2]]]  # -> wxyz for renderer
        frames.append(fr)

    def sim(seconds, freeze=None, zero_all=False):
        for _ in range(int(seconds * FPS)):
            for _ in range(spf):
                p.stepSimulation()
                if zero_all:
                    for b in bodies.values():
                        p.resetBaseVelocity(b, [0, 0, 0], [0, 0, 0])
                elif freeze:
                    for n in freeze:
                        p.resetBaseVelocity(bodies[n], [0, 0, 0], [0, 0, 0])
            record()

    def find(sub, k=0):
        hits = [n for n in bodies if sub in labels[n]]
        return hits[k] if len(hits) > k else None

    def drag(name, dpos, dyaw_deg, seconds):
        """Kinematic gizmo drag: smooth-step to offset pose."""
        bid = bodies[name]
        lp0, lq0 = link_pose(p, bid)
        from scipy.spatial.transform import Rotation
        for i in range(int(seconds * FPS)):
            t = (i + 1) / (seconds * FPS)
            t = 3 * t * t - 2 * t ** 3  # smoothstep
            lp = np.array(lp0) + t * np.array(dpos)
            rz = Rotation.from_euler("z", t * dyaw_deg, degrees=True)
            q0 = Rotation.from_quat(lq0)  # xyzw
            lq = (rz * q0).as_quat()
            p.resetBasePositionAndOrientation(
                bid, *com_pose(p, bid, lp.tolist(), lq.tolist()))
            p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0])
            for _ in range(spf):
                p.stepSimulation()
                p.resetBaseVelocity(bid, [0, 0, 0], [0, 0, 0])
            record()

    # ---- scripted session ---------------------------------------------------
    sim(1.2, zero_all=True)                       # settle

    mon = find("monitor") or find("keyboard")
    if mon:
        drag(mon, [0.15, 0.25, 0.35], 25, 2.5)    # grab & carry the monitor
        sim(2.5)                                   # release -> falls
    chair = find("chair")
    if chair:
        bid = bodies[chair]
        for i in range(int(0.5 * FPS)):
            for _ in range(spf):
                p.applyExternalForce(bid, -1, [90, 45, 0],
                                     p.getBasePositionAndOrientation(bid)[0],
                                     p.WORLD_FRAME)
                p.stepSimulation()
            record()
        sim(2.5)                                   # chair slides/topples
    small = find("mug") or find("bottle") or find("mouse")
    if small:
        drag(small, [0.0, 0.1, 0.3], 0, 1.2)      # lift a small object
        sim(2.3)                                   # drop

    # ---- camera path through 3 good DSLR views -----------------------------
    K, W, H, _ = C.load_intrinsics()
    cents = np.array([m["centroid"] for m, _ in kept])
    scored = []
    for fn, w2c in sorted(C.load_colmap_w2c().items()):
        pc = cents @ w2c[:3, :3].T + w2c[:3, 3]
        z = np.clip(pc[:, 2], 1e-6, None)
        u = pc[:, 0] / z * K[0, 0] + K[0, 2]
        v = pc[:, 1] / z * K[1, 1] + K[1, 2]
        n = int(((z > 0.3) & (u > 0) & (u < W) & (v > 0) & (v < H)).sum())
        scored.append((n, fn, w2c))
    scored.sort(key=lambda x: -x[0])
    picks = [scored[0]]
    for n, fn, w2c in scored[1:]:
        if all(np.linalg.norm(np.linalg.inv(w2c)[:3, 3]
                              - np.linalg.inv(pw)[:3, 3]) > 0.9
               for _, _, pw in picks):
            picks.append((n, fn, w2c))
        if len(picks) == 3:
            break
    keys = [w for _, _, w in picks] + [picks[0][2]]
    T = len(frames)
    cams = []
    for i in range(T):
        u = i / max(T - 1, 1) * (len(keys) - 1)
        a = min(int(u), len(keys) - 2)
        cams.append(slerp_w2c(keys[a], keys[a + 1], u - a).tolist())

    C.save_json(args.out, {"fps": FPS, "frames": frames, "cams": cams})
    print(f"[ds] {T} frames, actors: monitor={mon} chair={chair} "
          f"small={small}")


if __name__ == "__main__":
    main()
