"""Render a MuJoCo rollout video from the exported scene.xml.

2 s settle -> lift a few objects and drop them -> 4 s dynamics.
Headless EGL rendering (MUJOCO_GL=egl on a GPU node).
Usage: mujoco_video.py [--xml PATH] [--out PATH]
"""
import argparse

import numpy as np

from agents.core import common as C


def main():
    import imageio.v2 as imageio
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(C.OUT / "sim_export" / "scene.xml"))
    ap.add_argument("--out", default=str(C.OUT / "sim_export" / "mujoco_rollout.mp4"))
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--dump-poses", default="")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    free_bodies = [model.body(i).name for i in range(model.nbody)
                   if model.body(i).jntnum[0] == 1 and
                   model.body(i).name.startswith("obj_")]
    centers = np.array([data.xpos[model.body(n).id] for n in free_bodies])
    lookat = centers.mean(axis=0)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = float(np.ptp(centers, axis=0).max() + 1.8)
    cam.azimuth, cam.elevation = 135, -30

    renderer = mujoco.Renderer(model, height=720, width=1280)
    fps, dt = 30, model.opt.timestep
    steps_per_frame = max(int(1 / (fps * dt)), 1)
    frames = []

    pose_log = []

    def rollout(seconds):
        for _ in range(int(seconds * fps)):
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
            # body frame == canonical asset frame (inertial origin at 0),
            # xquat is wxyz - exactly what the gsplat compositor consumes
            pose_log.append({n: [data.xpos[model.body(n).id].tolist(),
                                 data.xquat[model.body(n).id].tolist()]
                             for n in free_bodies})

    rollout(2.0)  # settle

    # lift the 3 largest objects 18 cm and give them a nudge
    picks = sorted(free_bodies,
                   key=lambda n: -model.body(n).subtreemass[0])[:3]
    for n in picks:
        b = model.body(n)
        adr = model.jnt_qposadr[b.jntadr[0]]
        data.qpos[adr + 2] += 0.18
        vadr = model.jnt_dofadr[b.jntadr[0]]
        data.qvel[vadr:vadr + 3] = np.random.RandomState(0).uniform(-0.15, 0.15, 3)
    mujoco.mj_forward(model, data)
    rollout(args.seconds - 2.0)

    imageio.mimwrite(args.out, frames, fps=fps, quality=8,
                     macro_block_size=None)
    if args.dump_poses:
        C.save_json(args.dump_poses, {"fps": fps, "frames": pose_log})
    print(f"[mv] wrote {args.out} ({len(frames)} frames, "
          f"lifted: {picks})")


if __name__ == "__main__":
    main()
