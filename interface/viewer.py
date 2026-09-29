"""Interactive digital-twin editor (viser), multi-scene.

Discovers every result set under --outputs-root (any dir with
objects/objects.json), gives a scene dropdown, and per scene shows the
photoreal splat + sim-ready assets on drag gizmos. Buttons:
  - Run physics: current (user-edited) poses -> PyBullet in the main SimAny
    venv -> 30 fps playback in the viewer
  - Reset poses: back to the registered placement
  - Record take: samples object poses AND the connected browser's camera at
    30 fps while you drag / run physics; Stop & save writes a take json to
    outputs/demo_sessions/<result-set>/. Render it photoreal (objects as
    ORIGINAL scene splats) with:
      run_gs interface.demo_movie replay --take <take.json> --out <mp4>

Run under the mini-viewer env:
  .../mini-viewer/bin/python -m interface.viewer --outputs-root ./outputs
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

from agents.core import common as C  # noqa: E402  (scene-independent helpers only)

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = os.environ.get("SIMANY_PY", str(REPO_ROOT / ".venv" / "bin" / "python"))
SPLATS = Path(os.environ.get("SIMANY_SPLATS_ROOT", "/data/ScanNetppv2_gsplat/splats"))

_splat_cache = {}


def splat_arrays(key, ply_path, max_splats):
    if key in _splat_cache:
        return _splat_cache[key]
    gs = C.load_gaussians(ply_path, device="cpu")
    means = gs["means"].numpy()
    quats = gs["quats"].numpy().astype(np.float64)  # wxyz
    scales = gs["scales"].numpy().astype(np.float64)
    opac = gs["opacities"].numpy()
    rgb = np.clip(0.5 + 0.2820948 * gs["sh"][:, 0].numpy(), 0, 1)
    if len(means) > max_splats:
        # importance subsampling (opacity x size) instead of random: keeps
        # the load-bearing gaussians, visibly better at the same budget
        w = opac * scales.mean(axis=1)
        idx = np.argsort(-w)[:max_splats]
        means, quats, scales, opac, rgb = (a[idx] for a in
                                           (means, quats, scales, opac, rgb))
    import utils3d
    R = utils3d.numpy.quaternion_to_matrix(quats)
    cov = (R * scales[:, None, :] ** 2) @ R.transpose(0, 2, 1)
    out = (means.astype(np.float32), cov.astype(np.float32),
           rgb.astype(np.float32), opac.astype(np.float32).reshape(-1, 1))
    if len(_splat_cache) >= 3:  # keep RAM bounded
        _splat_cache.pop(next(iter(_splat_cache)))
    _splat_cache[key] = out
    return out


def discover(root):
    found = {}
    for d in sorted(Path(root).glob("*")):
        if (d / "objects" / "objects.json").exists():
            found[d.name] = d
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root",
                    default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-splats", type=int, default=800_000)
    args = ap.parse_args()

    results = discover(args.outputs_root)
    assert results, f"no result sets under {args.outputs_root}"
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    share_file = REPO_ROOT / "outputs" / "SHARE_URL.txt"

    def request_share():
        try:
            url = server.request_share_url()
            share_file.write_text(url + "\n")
            print(f"[viewer] SHARE URL: {url}", flush=True)
            return url
        except Exception as e:
            print(f"[viewer] share url failed: {e}", flush=True)
            return None

    def share_watchdog():
        # the share.viser.studio relay tunnel can die behind some networks;
        # probe it and re-tunnel automatically, latest URL in SHARE_URL.txt
        import urllib.request
        while True:
            time.sleep(90)
            url = (share_file.read_text().strip()
                   if share_file.exists() else None)
            ok = False
            if url:
                try:
                    with urllib.request.urlopen(url, timeout=8) as r:
                        ok = r.status == 200
                except Exception:
                    ok = False
            if not ok:
                try:
                    server.disconnect_share_url()
                except Exception:
                    pass
                request_share()

    request_share()
    import threading
    threading.Thread(target=share_watchdog, daemon=True).start()

    state = {"handles": [], "gizmos": {}, "home": {}, "out": None,
             "scene_id": None, "busy": False}
    rec = {"on": False, "frames": [], "cams": [], "home": {}, "meta": {}}

    with server.gui.add_folder("Digital twin"):
        prefer = ["578511c8a9_factory", "c50d2d1d42_factory"]
        default = next((p for p in prefer if p in results),
                       next(iter(results)))
        gui_scene = server.gui.add_dropdown("scene", tuple(results),
                                            initial_value=default)
        gui_bg = server.gui.add_checkbox("show background splat", True)
        gui_clean = server.gui.add_checkbox(
            "inpainted background (objects removed)", False)
        gui_giz = server.gui.add_checkbox("show gizmos", True)
        btn_reset = server.gui.add_button("Reset poses")
        btn_phys = server.gui.add_button("Run physics (4 s)")
        status = server.gui.add_markdown("loading ...")

    with server.gui.add_folder("Record demo take"):
        btn_rec = server.gui.add_button("● Start recording")
        btn_stop = server.gui.add_button("■ Stop & save", disabled=True)
        rec_status = server.gui.add_markdown(
            "records object poses + your camera at 30 fps")

    REC_FPS = 30

    def _cam_snapshot():
        clients = server.get_clients()
        if not clients:
            return None
        cam = clients[max(clients)].camera
        return {"position": [float(x) for x in cam.position],
                "wxyz": [float(x) for x in cam.wxyz],
                "fov": float(cam.fov), "aspect": float(cam.aspect),
                "look_at": [float(x) for x in cam.look_at],
                "up": [float(x) for x in cam.up_direction]}

    def _rec_loop():
        next_t = time.monotonic()
        while rec["on"]:
            fr = {n: [list(map(float, tc.position)),
                      list(map(float, tc.wxyz))]
                  for n, tc in state["gizmos"].items()}
            rec["frames"].append(fr)
            rec["cams"].append(_cam_snapshot())
            if len(rec["frames"]) % (5 * REC_FPS) == 0:
                rec_status.content = (f"recording ... "
                                      f"{len(rec['frames']) / REC_FPS:.0f}s")
            next_t += 1.0 / REC_FPS
            time.sleep(max(0.0, next_t - time.monotonic()))

    def _rec_save():
        take = {"fps": REC_FPS, "scene": rec["meta"].get("scene_id"),
                "out": rec["meta"].get("out"),
                "home": rec["home"], "frames": rec["frames"],
                "cams": rec["cams"]}
        d = REPO_ROOT / "outputs" / "demo_sessions" / rec["meta"]["name"]
        d.mkdir(parents=True, exist_ok=True)
        path = d / time.strftime("take_%Y%m%d_%H%M%S.json")
        path.write_text(json.dumps(take))
        return path, len(rec["frames"])

    def stop_recording(save=True):
        if not rec["on"]:
            return
        rec["on"] = False
        time.sleep(2.0 / REC_FPS)  # let the sampler thread exit
        if save and rec["frames"]:
            path, n = _rec_save()
            rec_status.content = (f"saved **{n / REC_FPS:.1f}s** "
                                  f"({n} frames) → `{path.name}`")
            print(f"[viewer] take saved: {path}", flush=True)
        btn_rec.disabled = False
        btn_stop.disabled = True

    @btn_rec.on_click
    def _(_):
        if rec["on"] or state["out"] is None:
            return
        if _cam_snapshot() is None:
            rec_status.content = "no browser client connected - open the viewer first"
            return
        rec.update(on=True, frames=[], cams=[],
                   home={n: [list(t), list(q)]
                         for n, (t, q) in state["home"].items()},
                   meta={"scene_id": state["scene_id"],
                         "out": str(state["out"]),
                         "name": state["out"].name})
        btn_rec.disabled = True
        btn_stop.disabled = False
        rec_status.content = "recording ... drag objects / run physics / move the camera"
        import threading as _th
        _th.Thread(target=_rec_loop, daemon=True).start()

    @btn_stop.on_click
    def _(_):
        stop_recording(save=True)

    def add_background(out, scene_id):
        clean = out / "inpaint" / "clean_background.ply"
        use_clean = gui_clean.value and clean.exists()
        key = f"{scene_id}{'_clean' if use_clean else ''}"
        ply = clean if use_clean else SPLATS / f"{scene_id}.ply"
        centers, cov, rgb, opac = splat_arrays(key, ply, args.max_splats)
        bg = server.scene.add_gaussian_splats(
            "/background", centers=centers, covariances=cov, rgbs=rgb,
            opacities=opac, visible=gui_bg.value)
        state["handles"].append(bg)
        state["bg"] = bg
        state["has_clean"] = clean.exists()
        return use_clean

    def load_scene(name):
        stop_recording(save=True)  # a scene switch ends any live take
        out = results[name]
        scene_id = name.replace("_factory", "")
        status.content = f"loading {name} ..."
        for h in state["handles"]:
            try:
                h.remove()
            except Exception:
                pass
        state.update(handles=[], gizmos={}, home={}, out=out,
                     scene_id=scene_id)
        add_background(out, scene_id)

        n = 0
        objects = json.loads((out / "objects" / "objects.json").read_text())
        for m in objects:
            oname = f"obj_{m['index']:02d}"
            odir = out / "objects" / oname
            if not (odir / "aligned.json").exists():
                continue
            al = json.loads((odir / "aligned.json").read_text())
            if al.get("rejected"):
                continue
            s, R, t = C.decompose_similarity(np.array(al["T"]))
            tm = trimesh.load(odir / "mesh_sim.ply", process=False)
            tm.vertices = tm.vertices * s
            q = C.rot_to_quat_wxyz(R)
            tc = server.scene.add_transform_controls(
                f"/objs/{oname}",
                scale=float(2.2 * np.max(np.ptp(tm.vertices, axis=0))),
                line_width=1.5, position=tuple(t), wxyz=tuple(q),
                visible=gui_giz.value)
            mh = server.scene.add_mesh_trimesh(f"/objs/{oname}/mesh", tm)
            state["handles"] += [tc, mh]
            state["gizmos"][oname] = tc
            state["home"][oname] = (tuple(t), tuple(q))
            n += 1
        extra = " · inpainted bg available" if state.get("has_clean") else ""
        status.content = (f"**{scene_id}** — {n} objects{extra}. "
                          "drag, then run physics")

    @gui_scene.on_update
    def _(_):
        if not state["busy"]:
            state["busy"] = True
            try:
                load_scene(gui_scene.value)
            finally:
                state["busy"] = False

    @gui_bg.on_update
    def _(_):
        if "bg" in state:
            state["bg"].visible = gui_bg.value

    @gui_clean.on_update
    def _(_):
        if state["busy"] or state["out"] is None:
            return
        state["busy"] = True
        try:
            status.content = "switching background ..."
            if "bg" in state:
                try:
                    state["bg"].remove()
                    state["handles"].remove(state["bg"])
                except Exception:
                    pass
            used = add_background(state["out"], state["scene_id"])
            status.content = ("inpainted background (objects removed)"
                              if used else
                              "original background"
                              + ("" if state.get("has_clean")
                                 else " — no inpainted ply for this scene"))
        finally:
            state["busy"] = False

    @gui_giz.on_update
    def _(_):
        for tc in state["gizmos"].values():
            tc.visible = gui_giz.value

    @btn_reset.on_click
    def _(_):
        for name, (t, q) in state["home"].items():
            state["gizmos"][name].position = t
            state["gizmos"][name].wxyz = q
        status.content = "poses reset"

    @btn_phys.on_click
    def _(_):
        if state["busy"]:
            return
        state["busy"] = True
        btn_phys.disabled = True
        status.content = "simulating ..."
        try:
            out = state["out"]
            poses = {n: [list(map(float, tc.position)),
                         list(map(float, tc.wxyz))]
                     for n, tc in state["gizmos"].items()}
            pin = out / "sim" / "viewer_poses.json"
            pin.parent.mkdir(exist_ok=True)
            pin.write_text(json.dumps(poses))
            traj_p = out / "sim" / "viewer_traj.json"
            # -m from the repo root: viewer_settle uses package imports
            r = subprocess.run(
                [VENV_PY, "-m", "robo.sim.viewer_settle",
                 "--poses", str(pin), "--traj", str(traj_p)],
                cwd=str(REPO_ROOT),
                env={**os.environ, "SIMANY_OUT": str(out),
                     "SIMANY_SCENE": state["scene_id"]},
                capture_output=True, text=True)
            if r.returncode != 0:
                status.content = f"physics failed: ...{r.stderr[-300:]}"
                return
            traj = json.loads(traj_p.read_text())
            dt = 1.0 / traj["fps"]
            for fr in traj["frames"]:
                t0 = time.time()
                with server.atomic():
                    for n, (pos, q) in fr.items():
                        if n in state["gizmos"]:
                            state["gizmos"][n].position = tuple(pos)
                            state["gizmos"][n].wxyz = tuple(q)
                time.sleep(max(0.0, dt - (time.time() - t0)))
            status.content = "physics done — poses are the settled state"
        finally:
            btn_phys.disabled = False
            state["busy"] = False

    load_scene(default)
    print(f"[viewer] up on port {args.port} with {len(results)} result sets",
          flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
