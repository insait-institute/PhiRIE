"""RobotSession: MuJoCo scene + Franka rig + policies for the Studio robot tab.

CONTRACT (robot agent). Built on robo.envs.pi05_env.DroidSimEnv (DO NOT fork it):

    session = RobotSession(state: SceneState, config, task: dict | None = None,
                           render_wh=(640,360), exclude_objects=None)
        Builds DroidSimEnv from state.tasks (scene_xml, robot base_pos/base_yaw, table,
        ext_cam, exclude_objects). If state.tasks is None, a task suite must be authored
        first (author_task below) or export_mjcf/tasks jobs run.
    session.reset(task, seed=0, jitter=0.0) -> ResetInfo (calls env.reset with the task's
        target jitter like robo.eval.pi05_eval.run_episode does)
    session.tick(action: np.ndarray(8)) -> TickInfo(qpos(7), gripper(1), body_poses
        {obj: (pos, quat_wxyz)}, contacts summary, stage flags via robo.tasks.pi05_tasks
        TaskScorer, sim_time)
    session.hold_tick() -> TickInfo (re-applies current joint targets; used to let
        physics settle while the user edits)
    session.set_joint_targets(q7, gripper01) / session.joint_targets()
    session.ee_target(pos, quat_wxyz|None) -> IKResult (physicalview.ik) and sets
        joint targets when converged
    session.obs(mode="raster"|"composite", renderer: Renderer|None) -> dict with the
        two DROID image keys + joint/gripper arrays (composite uses Renderer.
        composite_with_robot with mujoco robot masks from robo.rendering.mujoco_masks)
    session.run_episode(policy, prompt, horizon_s, obs_mode, on_tick=callable(TickInfo,
        obs_images), stop_event) -> EpisodeResult(success, score, stages, ticks, outcome,
        wall_s, error) — uses robo.eval.pi05_eval.run_episode semantics (stage scorer,
        success = place held) but streams ticks; policy is a robo.policy PolicyClient
        (Pi05PolicyClient(host, port) or ScriptedPolicyClient) created by
        make_policy(config, policy_id, host, port, home_pose).
    session.body_reset_poses -> {obj: (pos, quat)} captured after reset (for
        splats.frame_transform)
    session.robot_mask(cam_name) -> uint8 mask from robo.rendering.mujoco_masks
    session.locked() -> the session RLock (``with session.locked():``) for callers that
        touch model/data themselves (the server-render stream's MuJoCo pass)
    session.live_body_poses() -> (body_poses, reset_poses) under the lock
    session.robot_geom_ids() -> cached geom ids of the robot bodies (mask rendering)
    session.close()

    author_task(state, target_obj, receptacle_obj | None, region | None, instruction) ->
        task dict in the pi05_tasks.json schema (task_id, target, target_label,
        receptacle, region {cx,cy,hx,hy,zlo,zhi}, instructions {default,vague,specific}).

Threading: run_episode runs in a background thread owned by the caller; every public
method acquires the session lock so GUI sliders and the episode loop never step MuJoCo
concurrently.

Implementation notes (not part of the contract):
  * End effector for IK / the gizmo: the rig's ``robot/2f85/pinch`` SITE (Menagerie's
    pinch point, 0.145 m along the 2F-85 ``base`` body's +z). At the home pose the
    silicone pad bodies sit 0.112 m and the pad contact geoms span ~0.112-0.149 m along
    that same axis, so the pinch site already lies between the finger pads; the TCP
    offset is therefore zero (``TCP_OFFSET``). robo/eval/pi05_eval.py uses the same site
    for its ee-distance diagnostics.
  * ``reset`` mirrors pi05_eval.run_episode: ``env.reset(jitter_body=target if jitter>0
    else None, jitter_xy=jitter, rng=np.random.RandomState(seed))``.
  * ``run_episode`` does NOT reset (the panel resets with the user's seed/jitter first);
    it re-arms the scorer, warms the policy up on the first observation if needed, and
    calls ``policy.reset()``.
  * Outcome strings follow robo.eval.episode_log.Outcome exactly (``success``,
    ``task_failure``, ``policy_timeout``, ``safety_termination``, ``env_crash``) plus
    ``stopped`` when the caller's stop_event fires. See ``classify_episode_exception``.
  * Composite observations: raster from MuJoCo, photoreal from ``renderer.
    composite_with_robot`` with ``render.mujoco_camera_to_w2c_K`` and the exact
    segmentation robot mask; objects are posed with ``splats.body_pose_transform`` (the
    same rule as robo/rendering/pi05_render.py). Any failure falls back to raster and
    marks ``obs["_mode"] = "raster"``.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from physicalview.config import StudioConfig
from physicalview.ik import IKResult, mat_to_quat, solve_ik
from physicalview.scene_state import SceneState

log = logging.getLogger("studio.robot")

EXT_KEY = "observation/exterior_image_1_left"
WRIST_KEY = "observation/wrist_image_left"
JOINT_KEY = "observation/joint_position"
GRIP_KEY = "observation/gripper_position"
STAGE_NAMES = ("grasp", "lift", "hover", "place")
GUI_IMAGE_WH = (320, 180)

# Outcome vocabulary (robo.eval.episode_log.Outcome values + "stopped").
OUTCOME_SUCCESS = "success"
OUTCOME_TASK_FAILURE = "task_failure"
OUTCOME_POLICY_TIMEOUT = "policy_timeout"
OUTCOME_SAFETY = "safety_termination"
OUTCOME_ENV_CRASH = "env_crash"
OUTCOME_STOPPED = "stopped"


@dataclass
class TickInfo:
    qpos: np.ndarray
    gripper: float
    body_poses: dict[str, tuple[np.ndarray, np.ndarray]]
    stages: dict[str, bool]
    sim_time: float
    contacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResetInfo:
    task_id: str | None
    seed: int
    jitter: float
    qpos: np.ndarray
    body_poses: dict[str, tuple[np.ndarray, np.ndarray]]
    obs: dict


@dataclass
class EpisodeResult:
    success: bool
    score: float
    stages: dict[str, bool]
    ticks: int
    outcome: str
    wall_s: float
    error: str | None = None


class _PolicyCallError(Exception):
    """Wraps an exception raised inside policy(obs, prompt) so the episode loop can
    tell policy failures from environment failures when classifying the outcome."""

    def __init__(self, inner: BaseException):
        super().__init__(str(inner))
        self.inner = inner


def _synced(fn):
    """Decorator: run the method under the session's re-entrant lock."""
    def wrapper(self, *a, **kw):
        with self._lock:
            return fn(self, *a, **kw)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def small_image(img: np.ndarray, wh: tuple[int, int] = GUI_IMAGE_WH) -> np.ndarray:
    """Downscale an RGB uint8 image for the browser (cv2 > PIL > stride fallback)."""
    w, h = int(wh[0]), int(wh[1])
    if img.shape[1] <= w and img.shape[0] <= h:
        return np.ascontiguousarray(img)
    try:
        import cv2
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
    except Exception:  # noqa: BLE001
        sy = max(img.shape[0] // h, 1)
        sx = max(img.shape[1] // w, 1)
        return np.ascontiguousarray(img[::sy, ::sx])


def classify_episode_exception(exc: BaseException, *, from_policy: bool) -> str:
    """Map an exception raised during an episode onto an outcome string.

    Rules (in order):
      1. raised inside the policy call and it is a timeout / connection problem
         (``TimeoutError``, ``robo.eval.episode_log.PolicyTimeoutError``,
         ``ConnectionError``, ``OSError``, or a class whose name contains "Timeout" or
         "ConnectionClosed" — the websocket library's own exception types) ->
         ``policy_timeout``.
      2. ``robo.policy.control_contract.EnvActionShapeError`` (NaN/Inf or malformed
         action) and ``episode_log.SafetyTerminationError`` (non-finite simulator state)
         -> ``safety_termination``.
      3. everything else -> ``robo.eval.episode_log.classify_exception`` (which yields
         ``policy_timeout`` for TimeoutError-likes and ``env_crash`` otherwise).
    """
    try:
        from robo.eval import episode_log as EL
    except Exception:  # noqa: BLE001 - keep the session usable without robo.eval
        EL = None  # type: ignore[assignment]
    try:
        from robo.policy.control_contract import EnvActionShapeError
    except Exception:  # noqa: BLE001
        EnvActionShapeError = ()  # type: ignore[assignment]
    name = type(exc).__name__
    if from_policy:
        timeout_like = isinstance(exc, (TimeoutError, ConnectionError, OSError))
        if EL is not None and isinstance(exc, EL.PolicyTimeoutError):
            timeout_like = True
        if "Timeout" in name or "ConnectionClosed" in name:
            timeout_like = True
        if timeout_like:
            return OUTCOME_POLICY_TIMEOUT
    if EnvActionShapeError and isinstance(exc, EnvActionShapeError):
        return OUTCOME_SAFETY
    if EL is not None:
        if isinstance(exc, EL.SafetyTerminationError):
            return OUTCOME_SAFETY
        return EL.classify_exception(exc).value
    if isinstance(exc, TimeoutError):  # pragma: no cover - fallback without robo.eval
        return OUTCOME_POLICY_TIMEOUT
    return OUTCOME_ENV_CRASH


def _empty_stages() -> dict[str, bool]:
    return {s: False for s in STAGE_NAMES}


class RobotSession:
    """DroidSimEnv wrapper with joint/EE control, staged scoring and streaming episodes."""

    EE_SITE = "robot/2f85/pinch"
    TCP_OFFSET = np.zeros(3)  # pinch site already sits between the finger pads
    ROBOT_ROOTS = ("robot/",)

    def __init__(self, state: SceneState, config: StudioConfig, task: dict | None = None,
                 render_wh: tuple[int, int] = (640, 360), exclude_objects=None):
        import mujoco
        from robo.envs import pi05_env

        self._lock = threading.RLock()
        self.state = state
        self.config = config
        suite = state.tasks
        if suite is None:
            raise RuntimeError(
                f"scene {state.result_set.name} has no task suite (sim_export/pi05_tasks.json): "
                "run Export MJCF + Generate tasks in the Generate tab, or author one.")
        scene_xml = Path(state.scene_xml) if state.scene_xml else Path(suite["scene_xml"])
        if not scene_xml.exists():
            raise FileNotFoundError(f"scene.xml missing: {scene_xml} (run Export MJCF)")
        exclude = (tuple(exclude_objects) if exclude_objects is not None
                   else tuple(suite.get("exclude_objects", ())))
        self.suite = suite
        self.render_wh = (int(render_wh[0]), int(render_wh[1]))
        self.env = pi05_env.DroidSimEnv(
            str(scene_xml), suite["robot"]["base_pos"], suite["robot"]["base_yaw"],
            table_box=suite.get("table"), ext_cam=suite.get("ext_cam"),
            exclude_objects=exclude, render_wh=self.render_wh)
        self.model, self.data, self.info = self.env.model, self.env.data, self.env.info
        try:  # any_instance scoring (same as pi05_eval.main)
            from robo.tasks import pi05_tasks
            self.env._task_rows = pi05_tasks._load_objects(Path(state.result_set.out_dir))
        except Exception as exc:  # noqa: BLE001 - optional diagnostics only
            log.debug("task rows unavailable: %s", exc)
            self.env._task_rows = None

        m = self.model
        self.arm_joints = list(self.info["arm_joints"])
        self._jadr = np.array([m.joint(n).qposadr[0] for n in self.arm_joints])
        self._dadr = np.array([m.joint(n).dofadr[0] for n in self.arm_joints])
        self.joint_ranges = np.array([m.jnt_range[m.joint(n).id] for n in self.arm_joints])
        self.home = np.asarray(self.info["home"], dtype=float).copy()
        self._site_id = m.site(self.EE_SITE).id
        self._scratch = mujoco.MjData(m)
        self._seg_renderer = None
        self._robot_geom_ids: np.ndarray | None = None
        self._targets_q = self.home.copy()
        self._targets_g = 0.0
        self._scorer = None
        self.task: dict | None = None
        self.body_reset_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._last_obs: dict | None = None
        self.closed = False

        tasks = list(suite.get("tasks", ()))
        first = task if task is not None else (tasks[0] if tasks else None)
        self.reset(first, seed=0, jitter=0.0)

    # ------------------------------------------------------------------ helpers --
    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def locked(self) -> threading.RLock:
        """Context manager serialising MuJoCo access with the control loop / episode
        thread: ``with session.locked(): ...`` (re-entrant)."""
        return self._lock

    @_synced
    def live_body_poses(self) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]],
                                       dict[str, tuple[np.ndarray, np.ndarray]]]:
        """(current body poses, reset poses) of the free objects, for posing their
        gaussians with splats.frame_transform."""
        return self._body_poses(), dict(self.body_reset_poses)

    def robot_geom_ids(self) -> np.ndarray:
        """Geom ids of the robot (bodies under ROBOT_ROOTS), cached; used for masks."""
        return self._robot_geoms()

    @property
    def free_bodies(self) -> list[str]:
        return list(self.env.free_bodies)

    @property
    def scorer(self):
        return self._scorer

    def _body_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {b: self.env.body_pose(b) for b in self.env.free_bodies}

    def _stages(self) -> dict[str, bool]:
        if self._scorer is None:
            return _empty_stages()
        st = self._scorer.summary()["stages"]
        return {k: bool(st.get(k, False)) for k in STAGE_NAMES}

    def _contacts(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ncon": int(self.data.ncon)}
        if self.task is not None:
            tgt = self.task["target"]
            if tgt in self.env._body_geoms:
                sides = sorted(self.env.pad_contacts(tgt))
                out["target_pads"] = sides
                out["grasped"] = len(sides) == 2
                out["lifted"] = bool(self.env.lifted(tgt))
                out["at_rest"] = bool(self.env.at_rest(tgt))
        return out

    def _tick_info(self) -> TickInfo:
        return TickInfo(qpos=self.env.joint_position(),
                        gripper=float(self.env.gripper_position()[0]),
                        body_poses=self._body_poses(), stages=self._stages(),
                        sim_time=float(self.data.time), contacts=self._contacts())

    def _arm_scorer(self) -> None:
        if self.task is None:
            self._scorer = None
            return
        from robo.tasks import pi05_tasks
        self._scorer = pi05_tasks.TaskScorer(self.env, self.task)

    # ------------------------------------------------------------------ control --
    @_synced
    def reset(self, task: dict | None, seed: int = 0, jitter: float = 0.0) -> ResetInfo:
        """env.reset exactly like robo.eval.pi05_eval.run_episode, then re-arm scorer."""
        rng = np.random.RandomState(int(seed))
        jitter = float(jitter)
        if task is not None and task.get("target") not in self.env.free_bodies:
            raise KeyError(f"task target {task.get('target')!r} is not a free body in the sim "
                           f"(free: {self.env.free_bodies})")
        obs = self.env.reset(jitter_body=task["target"] if (task and jitter > 0) else None,
                             jitter_xy=jitter, rng=rng)
        self.task = task
        self._arm_scorer()
        self._targets_q = self.home.copy()
        self._targets_g = 0.0
        self.body_reset_poses = self._body_poses()
        self._last_obs = obs
        return ResetInfo(task_id=task.get("task_id") if task else None, seed=int(seed),
                         jitter=jitter, qpos=self.env.joint_position(),
                         body_poses=dict(self.body_reset_poses), obs=obs)

    @_synced
    def tick(self, action: np.ndarray) -> TickInfo:
        from robo.policy.control_contract import validate_env_action
        a = validate_env_action(action)
        self.env.apply_action(a)
        self._targets_q = a[:7].copy()
        self._targets_g = float(a[7])
        if not np.all(np.isfinite(self.data.qpos)):
            try:
                from robo.eval.episode_log import SafetyTerminationError
            except Exception:  # noqa: BLE001
                SafetyTerminationError = RuntimeError  # type: ignore[assignment]
            raise SafetyTerminationError("non-finite qpos after mj_step")
        if self._scorer is not None:
            self._scorer.update()
        return self._tick_info()

    @_synced
    def hold_tick(self) -> TickInfo:
        """Re-apply the current joint targets for one control tick (physics keeps running)."""
        return self.tick(np.concatenate([self._targets_q, [self._targets_g]]))

    @_synced
    def set_joint_targets(self, q7: np.ndarray, gripper01: float) -> None:
        q = np.asarray(q7, dtype=float).reshape(7)
        self._targets_q = np.clip(q, self.joint_ranges[:, 0], self.joint_ranges[:, 1])
        self._targets_g = float(np.clip(gripper01, 0.0, 1.0))

    @_synced
    def joint_targets(self) -> tuple[np.ndarray, float]:
        return self._targets_q.copy(), float(self._targets_g)

    @_synced
    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Current TCP pose (pinch site + TCP_OFFSET in the site frame), wxyz quaternion."""
        R = self.data.site_xmat[self._site_id].reshape(3, 3)
        pos = self.data.site_xpos[self._site_id] + R @ self.TCP_OFFSET
        return pos.copy(), mat_to_quat(R)

    @_synced
    def ee_target(self, pos, quat_wxyz=None, **ik_kw) -> IKResult:
        """Solve IK for the TCP and adopt the solution as joint targets when converged."""
        pos = np.asarray(pos, dtype=float).reshape(3)
        if quat_wxyz is not None and np.any(self.TCP_OFFSET != 0):
            import mujoco
            R = np.empty(9)
            mujoco.mju_quat2Mat(R, np.asarray(quat_wxyz, dtype=float))
            pos = pos - R.reshape(3, 3) @ self.TCP_OFFSET
        kw = dict(q_init=self._targets_q, scratch=self._scratch, iters=100)
        kw.update(ik_kw)
        res = solve_ik(self.model, self.data, self.EE_SITE, pos, quat_wxyz,
                       self._jadr, self._dadr, **kw)
        if res.converged:
            self.set_joint_targets(res.q, self._targets_g)
        return res

    # -------------------------------------------------------------- observation --
    def _robot_geoms(self) -> np.ndarray:
        if self._robot_geom_ids is None:
            try:
                from robo.rendering import mujoco_masks
                ids = set(mujoco_masks.robot_geom_ids(self.model, self.ROBOT_ROOTS))
            except Exception:  # noqa: BLE001 - robo unavailable: bodies named robot/... directly
                m = self.model
                ids = {g for g in range(m.ngeom)
                       if (m.body(int(m.geom_bodyid[g])).name or "").startswith(self.ROBOT_ROOTS)}
            self._robot_geom_ids = np.asarray(sorted(ids), dtype=np.int64)
        return self._robot_geom_ids

    @_synced
    def robot_mask(self, cam_name: str) -> np.ndarray:
        """uint8 {0,255} robot mask at render_wh from MuJoCo segmentation rendering."""
        import mujoco
        from robo.rendering import mujoco_masks
        if self._seg_renderer is None:
            w, h = self.render_wh
            self._seg_renderer = mujoco.Renderer(self.model, height=h, width=w)
            self._seg_renderer.enable_segmentation_rendering()
        self._seg_renderer.update_scene(self.data, camera=cam_name)
        seg = self._seg_renderer.render()
        return mujoco_masks.robot_mask_from_segmentation(seg, self._robot_geoms())

    def _object_poses_for_render(self) -> dict[str, np.ndarray]:
        from physicalview.splats import body_pose_transform
        poses = {}
        for b in self.env.free_bodies:
            rec = self.state.objects.get(b)
            if rec is None or rec.T_world is None:
                continue
            pos, quat = self.env.body_pose(b)
            poses[b] = body_pose_transform(rec.T_world, pos, quat)
        return poses

    @_synced
    def obs(self, mode: str = "raster", renderer=None) -> dict:
        obs = self.env.get_obs()
        obs["_mode"] = "raster"
        if mode == "composite" and renderer is not None:
            try:
                from physicalview.render import mujoco_camera_to_w2c_K
                poses = self._object_poses_for_render()
                for key, cam in ((EXT_KEY, self.info["ext_cam"]), (WRIST_KEY, self.info["wrist_cam"])):
                    raster = obs[key]
                    wh = (raster.shape[1], raster.shape[0])
                    w2c, K = mujoco_camera_to_w2c_K(self.model, self.data, cam, wh)
                    mask = self.robot_mask(cam)
                    if mask.shape[:2] != raster.shape[:2]:
                        mask = small_image(np.repeat(mask[..., None], 3, axis=2), wh)[..., 0]
                    obs[key] = np.asarray(
                        renderer.composite_with_robot(w2c, K, wh, poses, raster, mask), dtype=np.uint8)
                obs["_mode"] = "composite"
            except Exception as exc:  # noqa: BLE001 - photoreal is best effort
                log.warning("composite obs failed, falling back to raster: %s", exc)
                obs["_mode"] = "raster"
        elif mode == "composite":
            log.debug("composite requested without a renderer; raster used")
        self._last_obs = obs
        return obs

    # ------------------------------------------------------------------ episode --
    def run_episode(self, policy, prompt: str, horizon_s: float, obs_mode: str,
                    on_tick: Callable | None, stop_event, renderer=None) -> EpisodeResult:
        """Stream one closed-loop episode; see module notes for outcome rules."""
        from robo.rigs import pi05_rig as rig
        t_wall = time.time()
        ticks = 0
        outcome = OUTCOME_TASK_FAILURE
        error: str | None = None
        n_ticks = max(int(round(float(horizon_s) * rig.CONTROL_HZ)), 1)
        try:
            with self._lock:
                if self._scorer is None and self.task is not None:
                    self._arm_scorer()
                obs = self.obs(obs_mode, renderer)
            if not getattr(policy, "warmed_up", True):
                try:
                    policy.warmup(obs, prompt)
                except Exception as exc:  # noqa: BLE001
                    raise _PolicyCallError(exc) from exc
            policy.reset()
            for _ in range(n_ticks):
                if stop_event is not None and stop_event.is_set():
                    outcome = OUTCOME_STOPPED
                    break
                try:
                    action = policy(obs, prompt)
                except Exception as exc:  # noqa: BLE001
                    from robo.policy.control_contract import EnvActionShapeError
                    if isinstance(exc, EnvActionShapeError):
                        raise
                    raise _PolicyCallError(exc) from exc
                with self._lock:
                    info = self.tick(action)
                    obs = self.obs(obs_mode, renderer)
                    success = bool(self._scorer is not None and self._scorer.success)
                ticks += 1
                if on_tick is not None:
                    try:
                        on_tick(info, {"ext": small_image(obs[EXT_KEY]),
                                       "wrist": small_image(obs[WRIST_KEY])})
                    except Exception:  # noqa: BLE001 - a GUI hiccup must not end the episode
                        log.error("on_tick failed:\n%s", traceback.format_exc())
                if success:
                    outcome = OUTCOME_SUCCESS
                    break
        except _PolicyCallError as wrapped:
            outcome = classify_episode_exception(wrapped.inner, from_policy=True)
            error = f"{type(wrapped.inner).__name__}: {wrapped.inner}"
            log.error("episode policy failure (%s):\n%s", outcome, traceback.format_exc())
        except Exception as exc:  # noqa: BLE001
            outcome = classify_episode_exception(exc, from_policy=False)
            error = f"{type(exc).__name__}: {exc}"
            log.error("episode failure (%s):\n%s", outcome, traceback.format_exc())
        with self._lock:
            if self._scorer is not None:
                summ = self._scorer.summary()
                success = bool(summ["success"])
                score = float(summ["score"])
                stages = {k: bool(summ["stages"].get(k, False)) for k in STAGE_NAMES}
            else:
                success, score, stages = False, 0.0, _empty_stages()
        if success and error is None and outcome != OUTCOME_STOPPED:
            outcome = OUTCOME_SUCCESS
        return EpisodeResult(success=success, score=score, stages=stages, ticks=ticks,
                             outcome=outcome, wall_s=time.time() - t_wall, error=error)

    # ---------------------------------------------------------------- lifecycle --
    @_synced
    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for r in (getattr(self.env, "renderer", None), self._seg_renderer):
            try:
                if r is not None:
                    r.close()
            except Exception:  # noqa: BLE001
                pass
        self._seg_renderer = None


# ------------------------------------------------------------------ factories ----

def make_policy(config: StudioConfig, policy_id: str, host: str, port: int, home_pose):
    """PolicyClient from configs/policies via robo.policy.registry (kwargs per client_kind
    exactly as robo/eval/paired_runner.py::get_policy branches them)."""
    from robo.policy.registry import DEFAULT_CONFIG_DIR, PolicyRegistry
    cfg_dir = Path(config.repo_root) / "configs" / "policies"
    registry = PolicyRegistry.from_config_dir(cfg_dir if cfg_dir.is_dir() else DEFAULT_CONFIG_DIR)
    entry = registry.get(policy_id)
    if entry.client_kind == "scripted":
        kwargs = {"home": np.asarray(home_pose, dtype=float)}
    else:
        kwargs = {"host": host, "port": int(port)}
    return registry.make_client(policy_id, **kwargs)


def _label_of(state: SceneState, obj: str) -> str:
    rec = state.objects.get(obj) if state.objects else None
    if rec is not None and rec.label:
        return str(rec.label).lower()
    return obj


def _instructions(gname: str, receptacle_label: str | None, word: str | None,
                  instruction: str) -> dict[str, str]:
    """Same wording as robo/tasks/pi05_tasks.py::generate; a user instruction replaces
    the default variant only."""
    if receptacle_label:
        out = {"default": f"put {gname} in the {receptacle_label}",
               "vague": f"put {gname} away",
               "specific": f"pick up {gname} and place it inside the {receptacle_label}"}
    else:
        where = f"the {word} side of the table" if word else "the marked region"
        out = {"default": f"move {gname} to {where}",
               "vague": f"move {gname}",
               "specific": f"pick up {gname} and set it down on {where}"}
    if instruction and instruction.strip():
        out["default"] = instruction.strip()
    return out


def _target_center(state: SceneState, target_obj: str) -> np.ndarray | None:
    rec = state.objects.get(target_obj) if state.objects else None
    if rec is None:
        return None
    if rec.T_world is not None:
        return np.asarray(rec.T_world, dtype=float)[:3, 3].copy()
    aabb = rec.meta.get("aabb") if isinstance(rec.meta, dict) else None
    if aabb is not None:
        return np.asarray(aabb, dtype=float).mean(axis=0)
    return None


def _table_z(state: SceneState, target_obj: str) -> float:
    suite = state.tasks or {}
    table = suite.get("table")
    if table and "top_z" in table:
        return float(table["top_z"])
    c = _target_center(state, target_obj)
    rec = state.objects.get(target_obj) if state.objects else None
    if c is not None:
        dims = rec.world_dims if (rec is not None and rec.world_dims) else None
        return float(c[2] - (float(dims[2]) / 2 if dims else 0.0))
    raise ValueError("cannot derive the table height: no table in the suite and no aligned "
                     f"transform for {target_obj}")


def default_region(state: SceneState, target_obj: str, size_m: float = 0.30) -> dict:
    """A ``size_m`` box in front of the target on the table top.

    "In front" is judged from the robot base (suite robot.base_pos/base_yaw): the box
    centre is ``size_m`` beyond the target along the base heading when that stays on the
    table and inside the arm's frontal workspace (robo.tasks.pi05_tasks._in_workspace);
    otherwise the other three directions (towards the robot, left, right) are tried in
    that order and the first admissible one is used, falling back to "towards the robot".
    z-band: [table_z - 0.02, table_z + 0.30] like pi05_tasks' region tasks.
    """
    c = _target_center(state, target_obj)
    if c is None:
        raise ValueError(f"{target_obj} has no aligned transform; cannot place a region")
    top_z = _table_z(state, target_obj)
    suite = state.tasks or {}
    robot = suite.get("robot") or {}
    base = np.asarray(robot.get("base_pos", c), dtype=float)[:2]
    yaw = float(robot.get("base_yaw", 0.0))
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    left = np.array([-heading[1], heading[0]])
    table = suite.get("table")

    def on_table(p):
        if not table:
            return True
        return (abs(p[0] - table["cx"]) <= table["hx"] - size_m / 2 and
                abs(p[1] - table["cy"]) <= table["hy"] - size_m / 2)

    def in_ws(p):
        try:
            from robo.tasks.pi05_tasks import _in_workspace
            return bool(_in_workspace(p, base, yaw))
        except Exception:  # noqa: BLE001
            return True

    cands = [heading, -heading, left, -left]
    centre = c[:2] - size_m * heading
    for d in cands:
        p = c[:2] + size_m * d
        if on_table(p) and in_ws(p):
            centre = p
            break
    return {"cx": float(centre[0]), "cy": float(centre[1]),
            "hx": float(size_m / 2), "hy": float(size_m / 2),
            "zlo": float(top_z - 0.02), "zhi": float(top_z + 0.30)}


def author_task(state: SceneState, target_obj: str, receptacle_obj: str | None,
                region: dict | None, instruction: str) -> dict:
    """Build a task dict in the pi05_tasks.json schema for the current scene."""
    if receptacle_obj in ("", "none", "None"):
        receptacle_obj = None
    if receptacle_obj == target_obj:
        raise ValueError("receptacle must differ from the target")
    scene = state.result_set.name
    label = _label_of(state, target_obj)
    gname = f"the {label}"
    task: dict[str, Any] = {
        "target": target_obj, "target_label": label, "any_instance": False,
        "receptacle": None, "authored": "studio",
    }
    if receptacle_obj is not None:
        rlabel = _label_of(state, receptacle_obj)
        rec = state.objects.get(receptacle_obj) if state.objects else None
        dims = rec.world_dims if (rec is not None and rec.world_dims) else None
        if dims is None:
            raise ValueError(f"receptacle {receptacle_obj} has no world_dims (not aligned)")
        task.update(task_id=f"{scene}__{target_obj}_into_{receptacle_obj}__studio",
                    receptacle=receptacle_obj,
                    receptacle_dims=[float(x) for x in dims],
                    instructions=_instructions(gname, rlabel, None, instruction))
        if region:
            task["region"] = dict(region)
        return task

    reg = dict(region) if region else default_region(state, target_obj)
    top_z = None
    if "zlo" not in reg or "zhi" not in reg:
        top_z = _table_z(state, target_obj)
        reg.setdefault("zlo", float(top_z - 0.02))
        reg.setdefault("zhi", float(top_z + 0.30))
    for k in ("cx", "cy", "hx", "hy"):
        if k not in reg:
            raise ValueError(f"region is missing {k!r}")
    reg = {k: float(reg[k]) for k in ("cx", "cy", "hx", "hy", "zlo", "zhi")}
    # left/right wording relative to the robot heading, like pi05_tasks.generate
    word = None
    suite = state.tasks or {}
    robot = suite.get("robot")
    c = _target_center(state, target_obj)
    if robot is not None and c is not None:
        yaw = float(robot.get("base_yaw", 0.0))
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        side = float(np.dot(np.array([reg["cx"], reg["cy"]]) - c[:2], left))
        word = "left" if side >= 0 else "right"
    task.update(task_id=f"{scene}__{target_obj}_to_region__studio", region=reg,
                instructions=_instructions(gname, None, word, instruction))
    return task


__all__ = ["RobotSession", "TickInfo", "ResetInfo", "EpisodeResult", "make_policy",
           "author_task", "default_region", "classify_episode_exception", "small_image",
           "EXT_KEY", "WRIST_KEY", "JOINT_KEY", "GRIP_KEY", "STAGE_NAMES", "GUI_IMAGE_WH"]
