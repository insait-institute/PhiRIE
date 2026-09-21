"""Physics backend for viewer.py (runs in the SimAny venv).

Reads {name: [pos_xyz, quat_wxyz]} start poses (link frame), simulates 4 s of
real gravity dynamics against the carved background mesh, writes a 30 fps
link-frame trajectory back as {fps, frames: [{name: [pos, quat_wxyz]}]}.
"""
import argparse
import json

import numpy as np

from agents.core import common as C
from robo.sim.s7_sim import build_background, link_pose

SECONDS, HZ, REC = 4.0, 240, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--traj", required=True)
    args = ap.parse_args()
    start = json.loads(open(args.poses).read())

    import pybullet as p

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
    bgb = p.createMultiBody(0, col)
    p.changeDynamics(bgb, -1, lateralFriction=0.6, restitution=0.05)

    bodies = {}
    for m, a in kept:
        name = f"obj_{m['index']:02d}"
        if name not in start:
            continue
        pos, q = start[name]
        ph = json.loads((C.OUT / "objects" / name / "physics.json").read_text())
        bid = p.loadURDF(str(C.OUT / "objects" / name / "object.urdf"),
                         basePosition=pos,
                         baseOrientation=[q[1], q[2], q[3], q[0]],
                         flags=p.URDF_USE_INERTIA_FROM_FILE)
        p.changeDynamics(bid, -1, lateralFriction=ph["friction"],
                         restitution=ph["restitution"],
                         linearDamping=0.04, angularDamping=0.04)
        bodies[name] = bid

    frames = []
    for i in range(int(SECONDS * HZ)):
        p.stepSimulation()
        if i % REC == 0:
            fr = {}
            for name, bid in bodies.items():
                pos, q = link_pose(p, bid)  # xyzw
                fr[name] = [pos, [q[3], q[0], q[1], q[2]]]  # -> wxyz
            frames.append(fr)
    json.dump({"fps": HZ / REC, "frames": frames}, open(args.traj, "w"))
    print(f"settled {len(bodies)} bodies, {len(frames)} frames")


if __name__ == "__main__":
    main()
