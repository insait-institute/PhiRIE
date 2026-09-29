"""Headless end-to-end smoke for SimAny Studio on a GPU node (no browser needed).

    source run/env.sh
    $STUDIO_PY -m physicalview.smoke --scene c50d2d1d42_factory --out outputs/studio/smoke

Steps (each recorded PASS/FAIL/SKIP in the JSON report, never aborting the run):
  1. config + GPU detection + compatibility matrix
  2. discover result sets; load the requested scene (splat, clean bg, objects, tasks)
  3. splat arrays for the background and every accepted object (client-side path)
  4. GPU Renderer: held-out camera render (PSNR vs the real photo when available),
     clean-bg composite with all objects, one proposal thumbnail
  5. RobotSession: reset the first task, IK to a nearby EE target, robot masks,
     raster + composite observations, a short scripted_sinusoid episode
  5b. server render stream (physicalview.streaming): a scene camera converted to the viser
     convention and an orbit camera rendered through ServerRenderStream.render_for_camera
     (non-black, JPEG-encodable, saved under --out); with the RobotSession alive the
     robot-composited frame must differ from the plain one
  6. JobManager: run one trivial local job to completion
  7. pipeline builders: build (do not run) one JobSpec per stage
  8. real viser server: start StudioApp on --port, check every tab built, stop
Images are written under --out for eyeballing.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from pathlib import Path

import numpy as np


def _save(path: Path, img: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(img).astype(np.uint8)).save(path)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))


class Report:
    def __init__(self) -> None:
        self.steps: dict[str, dict] = {}

    def run(self, name: str, fn, *, skip_if: str | None = None):
        if skip_if:
            self.steps[name] = {"status": "SKIP", "reason": skip_if}
            print(f"[smoke] SKIP {name}: {skip_if}", flush=True)
            return None
        t0 = time.time()
        try:
            out = fn()
            self.steps[name] = {"status": "PASS", "seconds": round(time.time() - t0, 2),
                                "info": out if isinstance(out, (dict, list, str, int, float)) else None}
            print(f"[smoke] PASS {name} ({time.time() - t0:.1f}s) {out if isinstance(out, (str, dict)) else ''}", flush=True)
            return out
        except Exception as exc:  # noqa: BLE001
            self.steps[name] = {"status": "FAIL", "seconds": round(time.time() - t0, 2),
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()[-3000:]}
            print(f"[smoke] FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            return None

    @property
    def ok(self) -> bool:
        return all(s["status"] != "FAIL" for s in self.steps.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="c50d2d1d42_factory")
    ap.add_argument("--out", default="outputs/studio/smoke")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--episode-s", type=float, default=2.0)
    ap.add_argument("--skip-robot", action="store_true")
    ap.add_argument("--skip-server", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    from physicalview.config import load_config
    from physicalview.gpu import compatibility_matrix, detect_gpu

    rep = Report()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    gpu = detect_gpu()
    rep.run("gpu", lambda: {"gpu": gpu.to_dict(), "matrix": compatibility_matrix(cfg, gpu)})

    # ---- scene ------------------------------------------------------------------------
    from physicalview import scene_state as SS
    holder: dict = {}

    def _discover():
        sets = SS.discover_result_sets(cfg)
        holder["sets"] = sets
        match = [r for r in sets if r.name == args.scene]
        if not match:
            raise FileNotFoundError(f"{args.scene} not among {len(sets)} result sets")
        holder["rs"] = match[0]
        return {"n_result_sets": len(sets), "kind": match[0].kind,
                "n_objects": match[0].n_objects, "n_accepted": match[0].n_accepted}
    rep.run("discover", _discover)

    def _load():
        st = SS.load_scene(cfg, holder["rs"])
        holder["state"] = st
        return {"cameras": len(st.cameras), "objects": len(st.objects),
                "accepted": sum(o.accepted for o in st.objects.values()),
                "clean_bg": st.clean_bg_gs is not None, "tasks": bool(st.tasks),
                "scene_xml": str(st.scene_xml) if st.scene_xml else None}
    rep.run("load_scene", _load, skip_if=None if "rs" in holder else "no result set")
    st = holder.get("state")

    # ---- splat arrays -------------------------------------------------------------------
    from physicalview import splats as SP

    def _splats():
        bg = SP.to_viser_arrays(st.clean_bg_gs or st.splat_gs, cfg.max_splats_background)
        n_obj = 0
        for oid, rec in st.objects.items():
            if not rec.accepted or rec.T_world is None:
                continue
            arr = SP.to_viser_arrays(SS.object_canonical_gs(st, oid), cfg.max_splats_object)
            SP.pose_arrays(arr, rec.T_world)
            n_obj += 1
        return {"bg_splats": len(bg), "objects_posed": n_obj}
    rep.run("splat_arrays", _splats, skip_if=None if st else "scene not loaded")

    # ---- renderer -----------------------------------------------------------------------
    from physicalview.render import RenderUnavailable, Renderer
    renderer = None

    def _render():
        nonlocal renderer
        renderer = Renderer(st, device="cuda", use_clean_bg=True)
        info: dict = {"background": renderer.background_kind}
        frame = sorted(st.cameras)[len(st.cameras) // 2]
        renderer.hidden = set(st.objects)             # background only vs the photo
        img = renderer.render_camera(frame, scale=0.5)
        _save(out / f"render_{frame}_bg.png", img)
        photo = None
        rs = st.result_set
        for cand in (rs.scene_dir / "dslr" / "resized_undistorted_images" / frame,) if rs.scene_dir else ():
            if cand.exists():
                from PIL import Image
                photo = np.asarray(Image.open(cand).convert("RGB").resize((img.shape[1], img.shape[0])))
        if photo is not None:
            renderer.set_background("raw") if st.splat_gs is not None else None
            raw = renderer.render_camera(frame, scale=0.5)
            info["psnr_raw_bg_vs_photo_db"] = round(_psnr(raw, photo), 2)
            _save(out / f"photo_{frame}.png", photo)
        renderer.hidden = set()
        if st.clean_bg_gs is not None:
            renderer.set_background("clean")
        comp = renderer.render_camera(frame, scale=0.5)
        _save(out / f"render_{frame}_composite.png", comp)
        info["composite_nonblack_frac"] = round(float((comp.max(axis=2) > 8).mean()), 3)
        acc = [o for o, r in st.objects.items() if r.accepted]
        if acc:
            thumb = renderer.thumbnail(acc[0])
            _save(out / f"thumb_{acc[0]}.png", thumb)
            info["thumbnail"] = acc[0]
        return info
    rep.run("renderer", _render, skip_if=None if (st and gpu.present) else "no scene or no GPU")

    # ---- robot -------------------------------------------------------------------------
    def _robot():
        from physicalview.robot import RobotSession, make_policy
        sess = RobotSession(st, cfg, render_wh=cfg.render_wh)
        holder["robot"] = sess
        task = st.tasks["tasks"][0]
        sess.reset(task, seed=0, jitter=0.0)
        info: dict = {"task": task["task_id"]}
        pos, quat = sess.ee_pose()
        ik = sess.ee_target(np.asarray(pos) + np.array([0.0, 0.0, 0.05]), quat)
        info["ik_converged"] = bool(ik.converged)
        info["ik_pos_err_mm"] = round(float(ik.pos_err) * 1000, 2)
        for _ in range(10):
            sess.hold_tick()
        obs = sess.obs("raster")
        _save(out / "obs_raster_ext.png", obs["observation/exterior_image_1_left"])
        _save(out / "obs_raster_wrist.png", obs["observation/wrist_image_left"])
        if renderer is not None:
            obs_c = sess.obs("composite", renderer=renderer)
            info["composite_mode"] = obs_c.get("_mode", "composite")
            _save(out / "obs_composite_ext.png", obs_c["observation/exterior_image_1_left"])
            _save(out / "obs_composite_wrist.png", obs_c["observation/wrist_image_left"])
        policy = make_policy(cfg, "scripted_sinusoid", "localhost", 0, sess.home)
        ticks = []
        stop = threading.Event()
        res = sess.run_episode(policy, task["instructions"]["default"], args.episode_s,
                               "composite" if renderer is not None else "raster",
                               on_tick=lambda t, imgs: ticks.append(t.sim_time),
                               stop_event=stop, renderer=renderer)
        info.update({"episode_outcome": res.outcome, "ticks": res.ticks, "score": res.score,
                     "wall_s": round(res.wall_s, 2), "error": res.error})
        return info
    rep.run("robot", _robot, skip_if=("--skip-robot" if args.skip_robot else
                                       None if (st and st.tasks and st.scene_xml) else "scene has no task suite/scene.xml"))

    # ---- server render stream ------------------------------------------------------------
    def _stream():
        import dataclasses
        import io
        from PIL import Image
        from physicalview.app import Context
        from physicalview.render import look_at_w2c
        from physicalview.splats import matrix_to_quat_wxyz
        from physicalview.streaming import ServerRenderStream, stream_size

        sess = holder.get("robot")
        if sess is not None and getattr(sess, "closed", False):
            sess = None
        sctx = Context(server=None, config=cfg, gpu=gpu, jobs=None)
        sctx.scene, sctx.renderer, sctx.robot = st, renderer, sess
        stream = ServerRenderStream(sctx, lambda: renderer, lambda: sess, cfg)
        info: dict = {"settings": dataclasses.asdict(stream.settings), "robot_session": sess is not None}

        def check(name: str, img: np.ndarray, wh) -> dict:
            assert img.dtype == np.uint8 and img.shape == (wh[1], wh[0], 3), f"{name}: shape {img.shape}"
            frac = float((img.max(axis=2) > 8).mean())
            assert frac > 0.2, f"{name}: frame mostly black ({frac:.3f} non-black)"
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=int(stream.jpeg_quality))
            _save(out / f"stream_{name}.png", img)
            return {"nonblack_frac": round(frac, 3), "jpeg_bytes": len(buf.getvalue())}

        # (a) a real scene camera expressed in the viser convention (position + c2w wxyz)
        frame = sorted(st.cameras)[len(st.cameras) // 2]
        c2w = np.linalg.inv(st.cameras[frame])
        fov = float(2 * np.arctan(st.H / (2 * st.K[1, 1]))) if (st.K is not None and st.H) else float(np.radians(60))
        aspect = (st.W / st.H) if (st.W and st.H) else 16 / 9
        wh = stream_size(960, aspect, 1280)
        stream.show_robot = False
        img_a = stream.render_for_camera(c2w[:3, 3], matrix_to_quat_wxyz(c2w[:3, :3]), fov, aspect, wh)
        info["scene_camera"] = {"frame": frame, "wh": list(wh), **check(f"scene_cam_{frame}", img_a, wh)}
        ref = renderer.render_camera(frame, scale=wh[0] / st.W) if st.W else None
        if ref is not None and ref.shape == img_a.shape:
            info["scene_camera"]["psnr_vs_render_camera_db"] = round(_psnr(img_a, ref), 2)
            assert info["scene_camera"]["psnr_vs_render_camera_db"] > 20, "viser-camera conversion disagrees with render_camera"

        # (b) an orbit camera: over the robot workspace when a sim exists, else the objects
        if sess is not None:
            base = np.asarray(sess.suite["robot"]["base_pos"], float)
            yaw = float(sess.suite["robot"].get("base_yaw", 0.0))
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
            target = base + Rz @ np.array([0.45, 0.0, 0.25])
            eye = base + Rz @ np.array([0.6, -1.5, 1.1])
        else:
            cs = [r.T_world[:3, 3] for r in st.objects.values() if r.accepted and r.T_world is not None]
            target = np.mean(cs, axis=0) if cs else np.zeros(3)
            eye = target + np.array([1.5, -1.5, 1.0])
        c2w_o = np.linalg.inv(look_at_w2c(eye, target))
        pos_o, q_o = c2w_o[:3, 3], matrix_to_quat_wxyz(c2w_o[:3, :3])
        wh_o = (1280, 720)
        img_b = stream.render_for_camera(pos_o, q_o, np.radians(60.0), 16 / 9, wh_o)
        info["orbit"] = check("orbit", img_b, wh_o)

        # (c) robot composite: same camera with the MuJoCo robot pass on
        if sess is not None:
            stream.show_robot = True
            img_c = stream.render_for_camera(pos_o, q_o, np.radians(60.0), 16 / 9, wh_o)
            info["robot_composite"] = check("orbit_robot", img_c, wh_o)
            changed = float((np.abs(img_c.astype(np.int16) - img_b.astype(np.int16)).max(axis=2) > 8).mean())
            info["robot_composite"]["changed_frac"] = round(changed, 4)
            assert changed > 0.002, f"robot-composited frame equals the plain one (changed {changed:.4f})"
            w2c_o = np.linalg.inv(c2w_o)
            K_o = np.array([[wh_o[1] / (2 * np.tan(np.radians(30))), 0, wh_o[0] / 2],
                            [0, wh_o[1] / (2 * np.tan(np.radians(30))), wh_o[1] / 2], [0, 0, 1]])
            rgb, mask, _, _ = stream._robot_pass(sess, w2c_o, K_o, wh_o)
            info["robot_composite"]["mask_frac"] = round(float((mask > 0).mean()), 4)
            assert (mask > 0).mean() > 0.002, "robot mask empty for the orbit camera"
            _save(out / "stream_orbit_robot_mask.png", np.repeat(mask[..., None], 3, axis=2))
        stream.stop()
        return info
    rep.run("server_render_stream", _stream,
            skip_if=None if (st and renderer is not None) else "no scene or no GPU renderer")

    # ---- jobs ---------------------------------------------------------------------------
    def _jobs():
        from physicalview.jobs import JobManager, JobSpec, JobState
        mgr = JobManager(cfg, gpu, jobs_root=out / "jobs")
        job = mgr.submit(JobSpec(name="smoke-echo", argv=[str(cfg.interpreter("studio") if cfg.interpreter("studio").exists() else "python3"), "-c", "print('studio smoke ok')"],
                                 env_key="studio", needs_gpu=False, where="local"))
        job = job.wait(timeout=60)
        tail = job.tail(5)
        mgr.shutdown()
        if job.state != JobState.SUCCEEDED:
            raise RuntimeError(f"job state {job.state}: {job.error} / {tail}")
        return {"state": str(job.state), "tail": tail.strip()[-80:], "where": job.where}
    rep.run("jobs", _jobs)

    def _pipeline():
        from physicalview import pipeline as P
        rs = holder["rs"]
        ctx = P.StageContext(cfg, rs.scene_id, rs.out_dir, auto=(rs.kind == "auto"),
                             scene_dir=rs.scene_dir,
                             images_dir=(rs.scene_dir / "dslr" / "resized_undistorted_images") if rs.scene_dir else None)
        built = {}
        built["generate:trellis"] = P.generate(ctx, "trellis", [0]).argv[-3:]
        built["generate:reconviagen"] = P.generate(ctx, "reconviagen", [0, 1]).argv[-3:]
        built["register"] = P.register(ctx, "signed_source_up", [0]).argv[-4:]
        built["export_mjcf"] = P.export_mjcf(ctx, "room").argv[-3:]
        built["tasks"] = P.tasks(ctx).argv[-3:]
        from physicalview.app import Selection
        chain = P.inpaint(ctx, Selection(kind="object", object_ids=[9]), "remove the mug completely", "lama")
        built["inpaint_chain"] = [s.name for s in chain]
        built["policy_server"] = P.policy_server(cfg, "pi05_droid_jointpos", 8000).argv[:3]
        return {k: [str(x) for x in v] if isinstance(v, (list, tuple)) else v for k, v in built.items()}
    rep.run("pipeline_builders", _pipeline, skip_if=None if "rs" in holder else "no result set")

    # ---- server -------------------------------------------------------------------------
    def _server():
        from physicalview.app import StudioApp
        app = StudioApp(cfg, port=args.port, host="127.0.0.1")
        status = dict(app.ctx.panel_status)
        info: dict = {"panels": status, "display_mode": app.ctx.display_mode,
                      "stream": app.ctx.stream is not None}
        try:
            bad = {k: v for k, v in status.items() if v != "built"}
            if bad:
                raise RuntimeError(f"panels not built: {bad}")
            if app.ctx.stream is None:
                raise RuntimeError("scene panel did not create ctx.stream")
            panel = getattr(app.ctx, "scene_panel", None)
            if panel is not None and "rs" in holder and gpu.present:
                # Headless display-mode check (no browser): load the scene through the real
                # panel and inspect viser's node registry for gaussian-splat nodes.
                registry = app.server.scene._handle_from_node_name

                def splat_nodes() -> list[str]:
                    return sorted(n for n, h in registry.items() if type(h).__name__ == "GaussianSplatHandle")

                panel._load(holder["rs"])       # synchronous here (the UI runs it in a daemon thread)
                info["server_mode"] = {"splat_nodes": splat_nodes(),
                                       "helper_nodes": sorted(n for n in registry if n.startswith("/helpers/")),
                                       "stream_active": app.ctx.stream.active}
                if panel.mode != "server" or app.ctx.display_mode != "server":
                    raise RuntimeError(f"expected server display mode, got {panel.mode}")
                if info["server_mode"]["splat_nodes"]:
                    raise RuntimeError(f"server mode created splat nodes: {info['server_mode']['splat_nodes']}")
                if not app.ctx.stream.active:
                    raise RuntimeError("server mode did not start the render stream")
                panel._set_display_mode("client")
                info["client_mode"] = {"n_splat_nodes": len(splat_nodes()), "stream_active": app.ctx.stream.active}
                if "/background" not in splat_nodes() or app.ctx.stream.active:
                    raise RuntimeError(f"client mode: splat nodes {splat_nodes()[:3]}..., stream active {app.ctx.stream.active}")
                panel._set_display_mode("server")
                if splat_nodes() or not app.ctx.stream.active:
                    raise RuntimeError(f"back to server mode: splat nodes left {splat_nodes()}")
                info["mode_switch_ok"] = True
            time.sleep(1.0)
        finally:
            app.shutdown()
        return info
    rep.run("viser_server", _server, skip_if="--skip-server" if args.skip_server else None)

    if holder.get("robot") is not None:
        try:
            holder["robot"].close()
        except Exception:  # noqa: BLE001
            pass
    report = {"scene": args.scene, "gpu": gpu.to_dict(), "ok": rep.ok, "steps": rep.steps,
              "out": str(out.resolve())}
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v["status"] for k, v in rep.steps.items()}, indent=1))
    print(f"[smoke] {'ALL PASS' if rep.ok else 'FAILURES'} -> {out / 'smoke_report.json'}", flush=True)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
