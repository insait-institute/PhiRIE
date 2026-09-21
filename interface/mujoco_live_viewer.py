"""Gaussian-native LIVE MuJoCo simulation in the browser (viser).

Server: MuJoCo steps the exported scene.xml in real time.
Client: background scene splat + each object rendered as ITS OWN gaussian
splat node whose pose is synced to the simulator every frame.
Controls: Run/Pause, Reset, Drop-an-object; while PAUSED, drag an object's
gizmo and its new pose is written back into MuJoCo qpos.

Run under the mini-viewer env (has viser + mujoco):
  SIMANY_SCENE=... SIMANY_OUT=.../<scene>_auto SIMANY_AUTO=1 \
  python mujoco_live_viewer.py --port 8091
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import viser

from agents.core import common as C  # noqa: E402


def splat_arrays_from(gs, extra_scale=1.0, max_splats=2_500_000):
    means = gs["means"].cpu().numpy() * extra_scale
    quats = gs["quats"].cpu().numpy().astype(np.float64)
    scales = gs["scales"].cpu().numpy().astype(np.float64) * extra_scale
    opac = gs["opacities"].cpu().numpy()
    rgb = np.clip(0.5 + 0.2820948 * gs["sh"][:, 0].cpu().numpy(), 0, 1)
    if len(means) > max_splats:
        w = opac * scales.mean(axis=1)
        idx = np.argsort(-w)[:max_splats]
        means, quats, scales, opac, rgb = (a[idx] for a in
                                           (means, quats, scales, opac, rgb))
    import utils3d
    R = utils3d.numpy.quaternion_to_matrix(quats)
    cov = (R * scales[:, None, :] ** 2) @ R.transpose(0, 2, 1)
    return (means.astype(np.float32), cov.astype(np.float32),
            rgb.astype(np.float32), opac.astype(np.float32).reshape(-1, 1))


def main():
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--xml", default=str(C.OUT / "sim_export" / "scene.xml"))
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos0 = data.qpos.copy()
    free = [model.body(i).name for i in range(model.nbody)
            if model.body(i).jntnum[0] == 1 and
            model.body(i).name.startswith("obj_")]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    try:
        print(f"[ml] SHARE URL: {server.request_share_url()}", flush=True)
    except Exception as e:
        print(f"[ml] share url failed: {e}", flush=True)

    clean = C.OUT / "inpaint" / "clean_background.ply"
    bg_src = clean if clean.exists() else C.SPLAT_PLY
    print(f"[ml] loading background splat: {bg_src.name}", flush=True)
    c, cov, rgb, o = splat_arrays_from(C.load_gaussians(bg_src, device="cpu"))
    server.scene.add_gaussian_splats("/background", centers=c,
                                     covariances=cov, rgbs=rgb, opacities=o)

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    nodes, gizmos = {}, {}
    for m in objects:
        name = f"obj_{m['index']:02d}"
        if name not in free:
            continue
        odir = C.OUT / "objects" / name
        al = json.loads((odir / "aligned.json").read_text())
        s = C.decompose_similarity(np.array(al["T"]))[0]
        gs = C.load_gaussians(odir / "trellis_gs.ply", device="cpu")
        c, cov, rgb, o = splat_arrays_from(gs, extra_scale=s)
        b = model.body(name).id
        nodes[name] = server.scene.add_gaussian_splats(
            f"/objs/{name}", centers=c, covariances=cov, rgbs=rgb,
            opacities=o, position=tuple(data.xpos[b]),
            wxyz=tuple(data.xquat[b]))
        gizmos[name] = server.scene.add_transform_controls(
            f"/giz/{name}", scale=0.22, line_width=1.5,
            position=tuple(data.xpos[b]), wxyz=tuple(data.xquat[b]),
            visible=False)
        print(f"[ml] {name} {m['label']}: {len(c)} gaussians", flush=True)

    with server.gui.add_folder("MuJoCo live"):
        gui_run = server.gui.add_checkbox("run physics", True)
        gui_giz = server.gui.add_checkbox("edit mode (pause + gizmos)", False)
        btn_reset = server.gui.add_button("Reset scene")
        btn_drop = server.gui.add_button("Lift & drop random object")
        status = server.gui.add_markdown("running")

    last_sent = {}

    def sync_from_sim(force=False):
        # delta-gated: settled bodies produce zero traffic (a naive 30Hz x
        # 18-node broadcast wedged the server and killed the share tunnel)
        for name in nodes:
            b = model.body(name).id
            pos, quat = data.xpos[b].copy(), data.xquat[b].copy()
            prev = last_sent.get(name)
            if not force and prev is not None and \
                    np.linalg.norm(pos - prev[0]) < 5e-4 and \
                    np.abs(quat - prev[1]).max() < 1e-3:
                continue
            last_sent[name] = (pos, quat)
            nodes[name].position = tuple(pos)
            nodes[name].wxyz = tuple(quat)
            gizmos[name].position = tuple(pos)
            gizmos[name].wxyz = tuple(quat)

    @gui_giz.on_update
    def _(_):
        edit = gui_giz.value
        gui_run.value = not edit
        for g in gizmos.values():
            g.visible = edit
        status.content = ("edit mode: drag gizmos, uncheck to resume"
                          if edit else "running")

    @btn_reset.on_click
    def _(_):
        data.qpos[:] = qpos0
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        sync_from_sim()
        status.content = "scene reset"

    @btn_drop.on_click
    def _(_):
        name = free[np.random.randint(len(free))]
        b = model.body(name)
        adr = model.jnt_qposadr[b.jntadr[0]]
        data.qpos[adr + 2] += 0.25
        vadr = model.jnt_dofadr[b.jntadr[0]]
        data.qvel[vadr:vadr + 6] = 0
        mujoco.mj_forward(model, data)
        status.content = f"dropped {name}"

    print(f"[ml] up on port {args.port} with {len(nodes)} gaussian bodies",
          flush=True)
    dt = model.opt.timestep
    RATE = 15  # Hz, browser sync
    while True:
        t0 = time.time()
        if not server.get_clients():
            time.sleep(0.25)
            continue
        if gui_giz.value:
            # write user-edited gizmo poses back into the simulator
            for name, g in gizmos.items():
                b = model.body(name)
                adr = model.jnt_qposadr[b.jntadr[0]]
                data.qpos[adr:adr + 3] = g.position
                data.qpos[adr + 3:adr + 7] = g.wxyz
                vadr = model.jnt_dofadr[b.jntadr[0]]
                data.qvel[vadr:vadr + 6] = 0
            mujoco.mj_forward(model, data)
            for name in nodes:
                bid = model.body(name).id
                nodes[name].position = tuple(data.xpos[bid])
                nodes[name].wxyz = tuple(data.xquat[bid])
        elif gui_run.value:
            for _ in range(max(int((1 / RATE) / dt), 1)):
                mujoco.mj_step(model, data)
            with server.atomic():
                sync_from_sim()
        time.sleep(max(0.0, 1 / RATE - (time.time() - t0)))


if __name__ == "__main__":
    main()
