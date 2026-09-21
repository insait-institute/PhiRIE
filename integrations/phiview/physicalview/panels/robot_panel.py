"""Robot tab — see docs/ARCHITECTURE.md for the required controls.

CONTRACT: build(ctx) creates the tab's GUI and wires events. Implemented by the
owning agent; app.py shows a placeholder while this raises NotImplementedError.

Sections
  1. Task: task dropdown (ctx.scene.tasks) + "Author new task" (target / receptacle /
     instruction) + instruction variant + Reset (seed, jitter). Reset lazily creates
     ``ctx.robot = RobotSession(...)`` in a daemon thread.
  2. Policy: policy dropdown, server host/port, "Start policy server (Slurm)" (via
     physicalview.pipeline.policy_server + ctx.jobs), "Check server" (warmup latency),
     observation mode, horizon, Run / Stop, stage bar, score/outcome, two live images.
  3. Manual control: 7 joint sliders (limits from jnt_range), gripper slider, Home,
     "Apply targets" (sliders/gizmo write joint targets), an end-effector transform gizmo
     (drag -> IK -> targets, IK residual shown), "Physics running" (15 Hz hold_tick loop).

Every tick publishes ``robot.tick`` with {'body_poses', 'reset_poses', 'qpos', 'stages',
'sim_time'} so the scene panel can re-pose object splats / frames and the server-render
stream re-renders. Display modes (``ctx.display_mode``, topic ``display.mode_changed``):
in *client* mode the robot's visual geoms are mirrored as viser meshes (one node per geom,
created once per session, position/wxyz updated per tick, <= 15 Hz); in *server* mode no
robot meshes are created — the robot is composited into the streamed JPEG by
physicalview.streaming. "Show robot in 3D view" toggles either. The two observation
images are created once and updated in place.

viser is imported only inside build() so the module imports in CPU test environments.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from physicalview.robot import (
    EXT_KEY, GRIP_KEY, GUI_IMAGE_WH, JOINT_KEY, STAGE_NAMES, WRIST_KEY,
    RobotSession, author_task, make_policy,
)

VARIANTS = ("default", "vague", "specific")
OBS_MODES = ("raster", "composite photoreal")
AUTHOR_OPTION = "Author new task…"
NONE_OPTION = "none"
LOOP_HZ = 15.0
IMAGE_HZ = 5.0
SLIDER_HZ = 5.0
PANDA_FALLBACK_RANGES = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                                  [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
                                  [-2.8973, 2.8973]])
# DROID reset pose (robo/rigs/pi05_rig.py::PANDA_HOME); sliders start here because 0.0 is
# outside joint4's range.
PANDA_HOME = np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0])


@dataclass
class _State:
    session: RobotSession | None = None
    authored: dict[str, dict] = field(default_factory=dict)      # task_id -> task dict
    episode_thread: threading.Thread | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    busy: bool = False                                           # session being built
    syncing_sliders: bool = False
    last_image_t: float = 0.0
    last_slider_t: float = 0.0
    last_publish_t: float = 0.0
    robot_nodes: dict[int, Any] = field(default_factory=dict)    # geom id -> viser handle (client mode)
    robot_nodes_session: Any = None                              # session whose geoms are mirrored
    gizmo: Any = None
    gizmo_user_moving: bool = False
    closing: bool = False


def _stage_bar(stages: dict[str, bool]) -> str:
    return " · ".join(f"{'●' if stages.get(s) else '○'} {s}" for s in STAGE_NAMES)


def _quat_of_mat(xmat: np.ndarray) -> tuple[float, float, float, float]:
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(xmat, dtype=float).reshape(9))
    return tuple(float(v) for v in q)


def _geom_color(model, g: int) -> tuple[int, int, int]:
    rgba = model.geom_rgba[g]
    mat = int(model.geom_matid[g])
    if mat >= 0:
        rgba = model.mat_rgba[mat]
    rgb = np.clip(np.asarray(rgba[:3], dtype=float), 0, 1) * 255
    return tuple(int(v) for v in rgb)


def _mesh_arrays(model, g: int) -> tuple[np.ndarray, np.ndarray]:
    mid = int(model.geom_dataid[g])
    va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
    verts = np.asarray(model.mesh_vert[va:va + vn], dtype=np.float32)
    faces = np.asarray(model.mesh_face[fa:fa + fn], dtype=np.uint32)
    return verts, faces


def build(ctx) -> None:
    import viser  # noqa: F401 - only available in the studio env

    server = ctx.server
    gui = server.gui
    st = _State()
    cfg = ctx.config

    gui.add_markdown("### Robot — Franka + 2F-85 over the exported scene")
    status = gui.add_markdown("_Load a scene with a task suite (Export MJCF + Generate tasks)._")
    show_robot_cb = gui.add_checkbox("Show robot in 3D view", True,
                                     hint="client mode: robot geom meshes; server mode: robot composited into the stream")

    def say(msg: str) -> None:
        try:
            status.content = msg
        except Exception:  # noqa: BLE001
            pass
        ctx.log(f"[robot] {msg}")

    # ------------------------------------------------------------------ 1. Task --
    with gui.add_folder("Task"):
        task_dd = gui.add_dropdown("Task", options=["(no tasks)"], initial_value="(no tasks)")
        variant_dd = gui.add_dropdown("Instruction variant", options=list(VARIANTS), initial_value="default")
        prompt_md = gui.add_markdown("")
        with gui.add_folder("Author new task", expand_by_default=False):
            target_dd = gui.add_dropdown("Target object", options=["(none)"], initial_value="(none)")
            recep_dd = gui.add_dropdown("Receptacle", options=[NONE_OPTION], initial_value=NONE_OPTION)
            instr_txt = gui.add_text("Instruction (optional)", "")
            author_btn = gui.add_button("Add authored task")
        seed_num = gui.add_number("Seed", 0, min=0, max=10_000, step=1)
        jitter_sl = gui.add_slider("Target jitter (m)", 0.0, 0.10, 0.005, 0.0)
        reset_btn = gui.add_button("Reset episode (build sim if needed)", icon=viser.Icon.REFRESH)

    # ---------------------------------------------------------------- 2. Policy --
    with gui.add_folder("Policy"):
        policies = list(cfg.policies) or ["scripted_sinusoid"]
        policy_dd = gui.add_dropdown("Policy", options=policies, initial_value=policies[0])
        host_txt = gui.add_text("Server host", "localhost")
        port_num = gui.add_number("Server port", int(cfg.policy_server_port), min=1, max=65535, step=1)
        start_srv_btn = gui.add_button("Start policy server (Slurm)")
        check_btn = gui.add_button("Check server")
        obs_dd = gui.add_dropdown("Observation", options=list(OBS_MODES), initial_value=OBS_MODES[0])
        horizon_sl = gui.add_slider("Horizon (s)", 2.0, 120.0, 1.0, 32.0)
        run_btn = gui.add_button("Run episode", icon=viser.Icon.PLAYER_PLAY)
        stop_btn = gui.add_button("Stop", icon=viser.Icon.PLAYER_STOP)
        stage_md = gui.add_markdown(_stage_bar({}))
        score_md = gui.add_markdown("score — · outcome —")
        blank = np.zeros((GUI_IMAGE_WH[1], GUI_IMAGE_WH[0], 3), np.uint8)
        ext_img = gui.add_image(blank, label="exterior (policy input)", format="jpeg", jpeg_quality=70)
        wrist_img = gui.add_image(blank, label="wrist (policy input)", format="jpeg", jpeg_quality=70)

    # -------------------------------------------------------- 3. Manual control --
    with gui.add_folder("Manual control", expand_by_default=False):
        joint_sl = []
        for i in range(7):
            lo, hi = PANDA_FALLBACK_RANGES[i]
            q0 = float(np.clip(PANDA_HOME[i], lo, hi))
            joint_sl.append(gui.add_slider(f"joint{i + 1} (rad)", float(lo), float(hi), 0.005, q0))
        grip_sl = gui.add_slider("gripper (0 open → 1 closed)", 0.0, 1.0, 0.01, 0.0)
        home_btn = gui.add_button("Home")
        apply_cb = gui.add_checkbox("Apply targets (sliders / gizmo drive the arm)", False)
        gizmo_cb = gui.add_checkbox("End-effector gizmo", False)
        physics_cb = gui.add_checkbox("Physics running", True)
        ik_md = gui.add_markdown("IK: —")

    # ------------------------------------------------------------- task helpers --
    def all_tasks() -> dict[str, dict]:
        out: dict[str, dict] = {}
        scene = ctx.scene
        if scene is not None and scene.tasks:
            for t in scene.tasks.get("tasks", ()):
                out[t["task_id"]] = t
        out.update(st.authored)
        return out

    def current_task() -> dict | None:
        return all_tasks().get(task_dd.value)

    def refresh_task_options(select: str | None = None) -> None:
        ids = list(all_tasks())
        opts = ids + [AUTHOR_OPTION] if ids else ["(no tasks)", AUTHOR_OPTION]
        task_dd.options = opts
        if select in opts:
            task_dd.value = select
        elif task_dd.value not in opts:
            task_dd.value = opts[0]
        update_prompt()

    def update_prompt() -> None:
        t = current_task()
        if t is None:
            prompt_md.content = "_no task selected_"
            return
        ins = t.get("instructions", {})
        prompt_md.content = f"**prompt:** {ins.get(variant_dd.value, '')}"

    def refresh_object_options() -> None:
        scene = ctx.scene
        names = []
        if scene is not None and scene.objects:
            names = [f"{k} — {v.label}" for k, v in sorted(scene.objects.items()) if v.accepted]
        target_dd.options = names or ["(none)"]
        target_dd.value = target_dd.options[0]
        recep_dd.options = [NONE_OPTION] + names
        recep_dd.value = NONE_OPTION

    def _obj_id(label: str) -> str | None:
        if label in ("(none)", NONE_OPTION, ""):
            return None
        return label.split(" — ")[0]

    @variant_dd.on_update
    def _(_e) -> None:
        update_prompt()

    @task_dd.on_update
    def _(_e) -> None:
        if task_dd.value == AUTHOR_OPTION:
            say("Fill in 'Author new task' and press 'Add authored task'.")
        update_prompt()

    @author_btn.on_click
    def _(_e) -> None:
        scene = ctx.scene
        if scene is None:
            say("Load a scene first.")
            return
        tgt = _obj_id(target_dd.value)
        if tgt is None:
            say("Pick a target object.")
            return
        try:
            task = author_task(scene, tgt, _obj_id(recep_dd.value), None, instr_txt.value)
        except Exception as exc:  # noqa: BLE001
            say(f"author_task failed: {exc}")
            return
        st.authored[task["task_id"]] = task
        refresh_task_options(select=task["task_id"])
        say(f"authored {task['task_id']}")

    # --------------------------------------------------------- robot 3D mirror --
    # Server-render mode: the robot is part of the streamed JPEG (MuJoCo raster composited
    # over the splats), so no geom meshes are mirrored. Client mode: one mesh node per visual
    # geom, created ONCE per session (identity, not id(model)) and only re-posed per tick.
    def display_mode() -> str:
        return getattr(ctx, "display_mode", "client")

    def remove_robot_nodes() -> None:
        for h in st.robot_nodes.values():
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        st.robot_nodes.clear()
        st.robot_nodes_session = None

    def ensure_robot_nodes(sess: RobotSession) -> None:
        if display_mode() != "client":
            if st.robot_nodes:
                remove_robot_nodes()
            return
        if st.robot_nodes_session is sess and st.robot_nodes:
            return
        remove_robot_nodes()
        import mujoco
        m = sess.model
        visible = bool(show_robot_cb.value)
        for g in range(m.ngeom):
            bname = m.body(m.geom_bodyid[g]).name or ""
            if not bname.startswith("robot/"):
                continue
            if int(m.geom_group[g]) == 3 or m.geom_rgba[g][3] <= 0.0:
                continue  # collision geoms
            name = f"/robot/geoms/{g}"
            color = _geom_color(m, g)
            gtype = int(m.geom_type[g])
            size = np.asarray(m.geom_size[g], dtype=float)
            try:
                if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                    verts, faces = _mesh_arrays(m, g)
                    h = server.scene.add_mesh_simple(name, verts, faces, color=color, visible=visible)
                elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                    h = server.scene.add_box(name, color=color, dimensions=tuple(2 * size), visible=visible)
                elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                    h = server.scene.add_icosphere(name, radius=float(size[0]), color=color, visible=visible)
                elif gtype in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
                    import trimesh
                    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                        tm = trimesh.creation.capsule(radius=float(size[0]), height=2 * float(size[1]))
                    else:
                        tm = trimesh.creation.cylinder(radius=float(size[0]), height=2 * float(size[1]))
                    h = server.scene.add_mesh_simple(name, np.asarray(tm.vertices, np.float32),
                                                     np.asarray(tm.faces, np.uint32), color=color,
                                                     visible=visible)
                else:
                    continue
            except Exception:  # noqa: BLE001
                ctx.log(f"[robot] geom {g} not mirrored:\n{traceback.format_exc()}")
                continue
            st.robot_nodes[g] = h
        st.robot_nodes_session = sess
        ctx.log(f"[robot] client mode: {len(st.robot_nodes)} robot geom meshes mirrored in the 3D view")

    def ensure_gizmo() -> None:
        if st.gizmo is None:
            st.gizmo = server.scene.add_transform_controls("/robot/ee_gizmo", scale=0.15,
                                                           line_width=2.0, visible=False)

            @st.gizmo.on_update
            def _(_h) -> None:
                sess2 = st.session
                if sess2 is None or not gizmo_cb.value:
                    return
                st.gizmo_user_moving = True
                try:
                    res = sess2.ee_target(np.asarray(st.gizmo.position), np.asarray(st.gizmo.wxyz))
                    ik_md.content = (f"IK: {'ok' if res.converged else 'NOT converged'} · "
                                     f"pos {res.pos_err * 1000:.1f} mm · rot {np.degrees(res.rot_err):.1f}° · "
                                     f"{res.iters} it")
                    if res.converged:
                        sync_sliders_from(res.q, sess2.joint_targets()[1], force=True)
                except Exception as exc:  # noqa: BLE001
                    ik_md.content = f"IK error: {exc}"
                finally:
                    st.gizmo_user_moving = False

    def set_robot_visible() -> None:
        vis = bool(show_robot_cb.value)
        for h in st.robot_nodes.values():
            try:
                h.visible = vis
            except Exception:  # noqa: BLE001
                pass
        stream = getattr(ctx, "stream", None)
        if stream is not None:
            stream.show_robot = vis
        ctx.events.publish("display.invalidate")

    show_robot_cb.on_update(lambda _e: set_robot_visible())
    if getattr(ctx, "stream", None) is not None:
        ctx.stream.show_robot = bool(show_robot_cb.value)

    def on_display_mode(mode) -> None:
        sess = st.session
        if mode == "client" and sess is not None and not sess.closed:
            ensure_robot_nodes(sess)
            update_robot_nodes(sess)
        else:
            remove_robot_nodes()

    ctx.events.subscribe("display.mode_changed", on_display_mode)

    def update_robot_nodes(sess: RobotSession) -> None:
        d = sess.data
        with server.atomic():
            for g, h in st.robot_nodes.items():
                h.position = tuple(float(v) for v in d.geom_xpos[g])
                h.wxyz = _quat_of_mat(d.geom_xmat[g])
            if st.gizmo is not None and not gizmo_cb.value:
                pos, quat = sess.ee_pose()
                st.gizmo.position = tuple(float(v) for v in pos)
                st.gizmo.wxyz = tuple(float(v) for v in quat)

    def sync_sliders_from(q: np.ndarray, g: float, force: bool = False) -> None:
        now = time.time()
        if not force and now - st.last_slider_t < 1.0 / SLIDER_HZ:
            return
        st.last_slider_t = now
        st.syncing_sliders = True
        try:
            with server.atomic():
                for i, s in enumerate(joint_sl):
                    s.value = float(np.clip(q[i], s.min, s.max))
                grip_sl.value = float(np.clip(g, 0.0, 1.0))
        finally:
            st.syncing_sliders = False

    def publish_tick(sess: RobotSession, info, force: bool = False) -> None:
        now = time.time()
        if not force and now - st.last_publish_t < 1.0 / LOOP_HZ:
            return
        st.last_publish_t = now
        ctx.events.publish("robot.tick", {
            "body_poses": info.body_poses, "reset_poses": sess.body_reset_poses,
            "qpos": info.qpos, "gripper": info.gripper, "stages": info.stages,
            "sim_time": info.sim_time,
        })
        update_robot_nodes(sess)
        if not apply_cb.value and not st.gizmo_user_moving:
            sync_sliders_from(info.qpos, info.gripper)

    def show_images(imgs: dict, force: bool = False) -> None:
        now = time.time()
        if not force and now - st.last_image_t < 1.0 / IMAGE_HZ:
            return
        st.last_image_t = now
        try:
            ext_img.image = imgs["ext"]
            wrist_img.image = imgs["wrist"]
        except Exception:  # noqa: BLE001
            pass

    def show_obs(sess: RobotSession) -> None:
        obs = sess.obs("raster", None)
        from physicalview.robot import small_image
        show_images({"ext": small_image(obs[EXT_KEY]), "wrist": small_image(obs[WRIST_KEY])}, force=True)

    # ------------------------------------------------------------- session build --
    def apply_joint_ranges(sess: RobotSession) -> None:
        for i, s in enumerate(joint_sl):
            lo, hi = sess.joint_ranges[i]
            try:
                s.min, s.max = float(lo), float(hi)
            except Exception:  # noqa: BLE001 - older viser: sliders keep the Panda defaults
                pass

    def ensure_session() -> RobotSession | None:
        if st.session is not None and ctx.robot is st.session:
            return st.session
        scene = ctx.scene
        if scene is None:
            say("No scene loaded — load one in the Scene tab.")
            return None
        if not scene.tasks:
            say("This scene has no task suite: run **Export MJCF** and **Generate tasks** in the "
                "Generate tab (or author a task after exporting).")
            return None
        say("Building MuJoCo scene + rig…")
        old = st.session
        st.session = None
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001
                pass
        sess = RobotSession(scene, cfg, render_wh=tuple(cfg.render_wh))
        st.session = sess
        ctx.robot = sess
        apply_joint_ranges(sess)
        ensure_gizmo()
        ensure_robot_nodes(sess)
        say(f"sim ready: {len(sess.free_bodies)} free objects, {sess.model.ngeom} geoms"
            + (" (robot composited into the server render stream)" if display_mode() != "client" else ""))
        return sess

    def do_reset() -> None:
        if st.busy:
            say("busy…")
            return
        if st.episode_thread is not None and st.episode_thread.is_alive():
            say("stop the running episode first")
            return
        st.busy = True
        try:
            sess = ensure_session()
            if sess is None:
                return
            task = current_task()
            if task is None:
                tasks = all_tasks()
                task = next(iter(tasks.values()), None)
                refresh_task_options(select=task["task_id"] if task else None)
            info = sess.reset(task, seed=int(seed_num.value), jitter=float(jitter_sl.value))
            sync_sliders_from(info.qpos, 0.0, force=True)
            tick = sess.hold_tick()
            publish_tick(sess, tick, force=True)
            show_obs(sess)
            stage_md.content = _stage_bar(tick.stages)
            score_md.content = "score — · outcome —"
            say(f"reset {info.task_id or '(no task)'} seed={info.seed} jitter={info.jitter:.3f} m")
        except Exception as exc:  # noqa: BLE001
            say(f"reset failed: {exc}")
            ctx.log(traceback.format_exc())
        finally:
            st.busy = False

    @reset_btn.on_click
    def _(_e) -> None:
        threading.Thread(target=do_reset, daemon=True, name="robot-reset").start()

    # ---------------------------------------------------------- policy actions --
    @start_srv_btn.on_click
    def _(_e) -> None:
        try:
            from physicalview import pipeline
            gpu_type = (cfg.policy_server_gpu_types or [None])[0]
            spec = pipeline.policy_server(cfg, policy_dd.value, int(port_num.value), gpu_type)
            job = ctx.jobs.submit(spec)
            say(f"policy server job {job.id} submitted ({policy_dd.value}, port {int(port_num.value)}); "
                f"set 'Server host' to the node once it is RUNNING (Jobs tab).")
        except NotImplementedError:
            say("pipeline.policy_server is not implemented yet — start run/pi05_serve.sh manually.")
        except Exception as exc:  # noqa: BLE001
            say(f"policy server submit failed: {exc}")

    def do_check_server() -> None:
        try:
            from robo.policy.clients.pi05_client import Pi05PolicyClient
            sess = st.session
            if sess is not None:
                obs = sess.obs("raster", None)
            else:
                obs = {EXT_KEY: np.zeros((180, 320, 3), np.uint8), WRIST_KEY: np.zeros((180, 320, 3), np.uint8),
                       JOINT_KEY: np.zeros(7), GRIP_KEY: np.zeros(1)}
            client = Pi05PolicyClient(host=host_txt.value, port=int(port_num.value), retries=1, retry_sleep_s=0.0)
            say(f"checking {host_txt.value}:{int(port_num.value)} (first call may jit for minutes)…")
            t0 = time.time()
            client.warmup(obs, "warmup")
            t1 = time.time()
            client(obs, "warmup")
            say(f"server OK: warmup {t1 - t0:.1f} s, one inference {time.time() - t1:.2f} s")
        except Exception as exc:  # noqa: BLE001
            say(f"server check failed: {type(exc).__name__}: {exc}")

    @check_btn.on_click
    def _(_e) -> None:
        threading.Thread(target=do_check_server, daemon=True, name="robot-check").start()

    def run_episode_thread() -> None:
        sess = st.session
        task = current_task()
        if sess is None or task is None:
            say("Reset first (builds the sim) and pick a task.")
            return
        prompt = task["instructions"].get(variant_dd.value, task["instructions"]["default"])
        obs_mode = "composite" if obs_dd.value.startswith("composite") else "raster"
        renderer = ctx.renderer if obs_mode == "composite" else None
        if obs_mode == "composite" and renderer is None:
            say("no GPU renderer available — using raster observations")
        try:
            policy = make_policy(cfg, policy_dd.value, host_txt.value, int(port_num.value), sess.home)
        except Exception as exc:  # noqa: BLE001
            say(f"policy unavailable: {type(exc).__name__}: {exc}")
            return
        try:
            sess.reset(task, seed=int(seed_num.value), jitter=float(jitter_sl.value))
        except Exception as exc:  # noqa: BLE001
            say(f"reset failed: {exc}")
            return
        st.stop_event = threading.Event()
        say(f"episode: {policy_dd.value} · '{prompt}' · {obs_mode} · {horizon_sl.value:.0f} s "
            "(warmup may take minutes for a fresh server)")
        stage_md.content = _stage_bar({})
        score_md.content = "score 0.00 · outcome running"
        last_stage = {"txt": ""}

        def on_tick(info, imgs) -> None:
            publish_tick(sess, info)
            show_images(imgs)
            bar = _stage_bar(info.stages)
            if bar != last_stage["txt"]:
                last_stage["txt"] = bar
                stage_md.content = bar

        res = sess.run_episode(policy, prompt, float(horizon_sl.value), obs_mode, on_tick,
                               st.stop_event, renderer=renderer)
        stage_md.content = _stage_bar(res.stages)
        score_md.content = (f"score **{res.score:.2f}** · outcome **{res.outcome}** · {res.ticks} ticks · "
                            f"{res.wall_s:.1f} s" + (f" · error: {res.error}" if res.error else ""))
        say(f"episode done: {res.outcome} score={res.score:.2f} ticks={res.ticks}")

    @run_btn.on_click
    def _(_e) -> None:
        if st.episode_thread is not None and st.episode_thread.is_alive():
            say("an episode is already running")
            return
        if st.session is None:
            say("Reset first to build the simulation.")
            return
        st.episode_thread = threading.Thread(target=run_episode_thread, daemon=True, name="robot-episode")
        st.episode_thread.start()

    @stop_btn.on_click
    def _(_e) -> None:
        st.stop_event.set()
        say("stop requested")

    # ------------------------------------------------------------ manual control --
    def slider_targets() -> tuple[np.ndarray, float]:
        return np.array([s.value for s in joint_sl], dtype=float), float(grip_sl.value)

    def on_slider(_e) -> None:
        if st.syncing_sliders or not apply_cb.value:
            return
        sess = st.session
        if sess is None:
            return
        q, g = slider_targets()
        sess.set_joint_targets(q, g)

    for s in joint_sl:
        s.on_update(on_slider)
    grip_sl.on_update(on_slider)

    @home_btn.on_click
    def _(_e) -> None:
        sess = st.session
        if sess is None:
            return
        sess.set_joint_targets(sess.home, 0.0)
        sync_sliders_from(sess.home, 0.0, force=True)
        ik_md.content = "IK: — (home targets set)"

    @apply_cb.on_update
    def _(_e) -> None:
        sess = st.session
        if sess is not None and apply_cb.value:
            q, g = slider_targets()
            sess.set_joint_targets(q, g)

    @gizmo_cb.on_update
    def _(_e) -> None:
        sess = st.session
        if st.gizmo is None:
            return
        if sess is not None and gizmo_cb.value:
            pos, quat = sess.ee_pose()  # snap to the TCP when enabling
            st.gizmo.position = tuple(float(v) for v in pos)
            st.gizmo.wxyz = tuple(float(v) for v in quat)
        st.gizmo.visible = bool(gizmo_cb.value)

    def control_loop() -> None:
        period = 1.0 / LOOP_HZ
        while not st.closing:
            t0 = time.time()
            sess = st.session
            episode_running = st.episode_thread is not None and st.episode_thread.is_alive()
            if sess is None or st.busy or episode_running or not physics_cb.value or sess.closed:
                time.sleep(0.1)
                continue
            try:
                if not server.get_clients():
                    time.sleep(0.25)
                    continue
                info = sess.hold_tick()
                publish_tick(sess, info)
                if time.time() - st.last_image_t > 1.0 / IMAGE_HZ:
                    show_obs(sess)
                stage_md.content = _stage_bar(info.stages)
            except Exception as exc:  # noqa: BLE001
                ctx.log(f"[robot] control loop error: {exc}\n{traceback.format_exc()}")
                time.sleep(0.5)
            time.sleep(max(0.0, period - (time.time() - t0)))

    threading.Thread(target=control_loop, daemon=True, name="robot-control").start()

    # ------------------------------------------------------------------ events --
    def on_scene_loaded(_scene) -> None:
        if st.episode_thread is not None and st.episode_thread.is_alive():
            st.stop_event.set()
        old, st.session, ctx.robot = st.session, None, None
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001
                pass
        remove_robot_nodes()
        st.authored.clear()
        refresh_task_options()
        refresh_object_options()
        scene = ctx.scene
        if scene is not None and scene.tasks:
            n = len(scene.tasks.get("tasks", ()))
            say(f"scene {scene.result_set.name}: {n} tasks. Press Reset to build the sim.")
        else:
            say("scene has no task suite — run Export MJCF + Generate tasks in the Generate tab.")

    ctx.events.subscribe("scene.loaded", on_scene_loaded)
    if ctx.scene is not None:
        on_scene_loaded(ctx.scene)
    else:
        refresh_task_options()
