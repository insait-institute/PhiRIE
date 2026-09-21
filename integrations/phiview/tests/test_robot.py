"""Tests for physicalview.robot.

CPU part (always runs): author_task / default_region, make_policy (scripted), outcome
classification, image downscale, and a renderer-free rig build that checks the chosen TCP
(``robot/2f85/pinch``) lies between the finger pads.

GPU part (needs MUJOCO_GL=egl + a GPU; skipped otherwise): RobotSession over the real
c50d2d1d42_factory scene — reset, hold_tick, EE IK target, robot mask, and a 3 s scripted
episode with streamed ticks + stop_event.

    srun -p debug --gres=gpu:a6000:1 --mem=32G --time=00:20:00 bash -lc \
      'cd $REPO && MUJOCO_GL=egl PYTHONPATH=$PWD .venv/bin/python -m pytest -q tests/test_studio_robot.py'
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from physicalview import robot as R  # noqa: E402
from physicalview.config import load_config  # noqa: E402
from physicalview.scene_state import ObjectRecord, ResultSet, SceneState  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FACTORY = ROOT / "outputs" / "c50d2d1d42_factory"
SUITE_JSON = FACTORY / "sim_export" / "pi05_tasks.json"
HAVE_SCENE = SUITE_JSON.exists() and (FACTORY / "sim_export" / "scene.xml").exists()
HAVE_MENAGERIE = (ROOT / "third_party" / "mujoco_menagerie" / "robotiq_2f85" / "2f85.xml").exists()


# ------------------------------------------------------------------ fixtures --

def _record(name: str, index: int, label: str, T: np.ndarray, dims, aabb=None) -> ObjectRecord:
    return ObjectRecord(name=name, index=index, label=label,
                        meta={"index": index, "label": label,
                              "aabb": aabb if aabb is not None else [[0, 0, 0], [0, 0, 0]]},
                        aligned={"T": T.tolist(), "world_dims": list(dims), "tier": "A"},
                        accepted=True, rejected_reason=None, T_world=T, world_dims=list(dims),
                        proposals={}, chosen_source="trellis", hybrid=None, physics=None,
                        collision_parts=1, drop_test=None, artifacts={}, rgba_png=None,
                        dir=Path("/nonexistent") / name)


def _T(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


@pytest.fixture
def synthetic_state(tmp_path: Path) -> SceneState:
    """Hand-built scene: table top at z=0.75, robot at the origin heading +x, a mug
    0.45 m ahead, a bowl 0.2 m to its left, plus a far-away lamp (not on the table)."""
    suite = {
        "scene": "synthetic_factory",
        "scene_xml": str(tmp_path / "scene.xml"),
        "robot": {"base_pos": [0.0, 0.0, 0.75], "base_yaw": 0.0},
        "table": {"cx": 0.4, "cy": 0.0, "hx": 0.8, "hy": 0.6, "top_z": 0.75},
        "ext_cam": {"pos": [0.05, 0.57, 0.66], "target": None, "fovy": 68.0},
        "exclude_objects": [],
        "time_limit_s": 16.0,
        "tasks": [],
    }
    objs = {
        "obj_01": _record("obj_01", 1, "mug", _T(0.45, 0.0, 0.80), [0.10, 0.10, 0.10]),
        "obj_02": _record("obj_02", 2, "bowl", _T(0.45, 0.20, 0.78), [0.18, 0.18, 0.06]),
        "obj_03": _record("obj_03", 3, "lamp", _T(3.0, 3.0, 1.2), [0.2, 0.2, 0.5]),
    }
    rs = ResultSet(name="synthetic_factory", scene_id="synthetic", kind="factory",
                   out_dir=tmp_path, scene_dir=None, splat_ply=None, mesh_ply=None,
                   has_sim_export=True, has_tasks=True, n_objects=3, n_accepted=3)
    return SceneState(result_set=rs, K=np.eye(3), W=0, H=0, cameras={}, splat_gs=None,
                      clean_bg_gs=None, mesh=None, objects=objs, tasks=suite,
                      scene_xml=None, sim_export=tmp_path, timings={}, report=None)


# ------------------------------------------------------------ author_task (CPU)

def test_author_region_task_default_region_in_front(synthetic_state):
    task = R.author_task(synthetic_state, "obj_01", None, None, "")
    assert task["task_id"] == "synthetic_factory__obj_01_to_region__studio"
    assert task["target"] == "obj_01" and task["target_label"] == "mug"
    assert task["receptacle"] is None and task["any_instance"] is False
    reg = task["region"]
    assert set(reg) == {"cx", "cy", "hx", "hy", "zlo", "zhi"}
    # 0.3 m box: half sizes 0.15, centred 0.30 m beyond the mug along the +x heading
    assert reg["hx"] == pytest.approx(0.15) and reg["hy"] == pytest.approx(0.15)
    assert reg["cx"] == pytest.approx(0.75) and reg["cy"] == pytest.approx(0.0)
    assert reg["zlo"] == pytest.approx(0.75 - 0.02) and reg["zhi"] == pytest.approx(0.75 + 0.30)
    # the target's start position is NOT inside the region (otherwise hover is trivial)
    assert not (abs(0.45 - reg["cx"]) <= reg["hx"] and abs(0.0 - reg["cy"]) <= reg["hy"])
    ins = task["instructions"]
    assert set(ins) == {"default", "vague", "specific"}
    assert ins["vague"] == "move the mug"
    assert ins["default"].startswith("move the mug to the") and "side of the table" in ins["default"]
    assert ins["specific"].startswith("pick up the mug and set it down on the")
    json.dumps(task)  # serialisable like pi05_tasks.json


def test_author_region_task_falls_back_when_front_is_off_table(synthetic_state):
    # Put the mug near the far table edge: "in front" would leave the table (and the
    # 0.80 m reach), so the region must be placed elsewhere but still on the table.
    synthetic_state.objects["obj_01"].T_world = _T(1.05, 0.0, 0.80)
    task = R.author_task(synthetic_state, "obj_01", None, None, "")
    reg = task["region"]
    t = synthetic_state.tasks["table"]
    assert abs(reg["cx"] - t["cx"]) <= t["hx"] and abs(reg["cy"] - t["cy"]) <= t["hy"]
    assert np.hypot(reg["cx"], reg["cy"]) <= 0.80 + 1e-9


def test_author_task_user_instruction_and_explicit_region(synthetic_state):
    region = {"cx": 0.6, "cy": -0.3, "hx": 0.1, "hy": 0.1}
    task = R.author_task(synthetic_state, "obj_01", None, region, "slide the mug to the corner")
    assert task["instructions"]["default"] == "slide the mug to the corner"
    assert task["instructions"]["vague"] == "move the mug"
    assert task["region"]["cx"] == 0.6 and task["region"]["hx"] == 0.1
    assert task["region"]["zlo"] == pytest.approx(0.73) and task["region"]["zhi"] == pytest.approx(1.05)
    assert "right" in task["instructions"]["specific"]  # -y is the robot's right when heading +x


def test_author_receptacle_task(synthetic_state):
    task = R.author_task(synthetic_state, "obj_01", "obj_02", None, "")
    assert task["task_id"] == "synthetic_factory__obj_01_into_obj_02__studio"
    assert task["receptacle"] == "obj_02"
    assert task["receptacle_dims"] == pytest.approx([0.18, 0.18, 0.06])
    assert "region" not in task  # TaskScorer resolves the region from the receptacle pose
    assert task["instructions"] == {
        "default": "put the mug in the bowl",
        "vague": "put the mug away",
        "specific": "pick up the mug and place it inside the bowl",
    }
    with pytest.raises(ValueError):
        R.author_task(synthetic_state, "obj_01", "obj_01", None, "")
    # "none" from a dropdown means no receptacle
    assert R.author_task(synthetic_state, "obj_01", "none", None, "")["receptacle"] is None


def test_table_z_derived_from_target_when_suite_has_no_table(synthetic_state):
    synthetic_state.tasks.pop("table")
    reg = R.default_region(synthetic_state, "obj_01")
    # mug centre z 0.80, height 0.10 -> bottom 0.75
    assert reg["zlo"] == pytest.approx(0.73) and reg["zhi"] == pytest.approx(1.05)


# ------------------------------------------------------------- make_policy (CPU)

@pytest.mark.backend
def test_make_policy_scripted_matches_registry_client():
    from robo.policy.clients.scripted_client import ScriptedPolicyClient
    cfg = load_config()
    home = np.array([0.0, -np.pi / 5, 0.0, -4 * np.pi / 5, 0.0, 3 * np.pi / 5, 0.0])
    pol = R.make_policy(cfg, "scripted_sinusoid", "localhost", 8000, home)
    assert isinstance(pol, ScriptedPolicyClient)
    obs = {R.JOINT_KEY: home, R.GRIP_KEY: np.zeros(1)}
    pol.warmup(obs, "x")
    pol.reset()
    a = pol(obs, "x")
    assert a.shape == (8,) and np.all(np.isfinite(a))
    with pytest.raises(Exception):
        R.make_policy(cfg, "no_such_policy", "localhost", 8000, home)


# ---------------------------------------------------------- classification (CPU)

@pytest.mark.backend
def test_outcome_classification_rules():
    from robo.eval.episode_log import PolicyTimeoutError, SafetyTerminationError
    from robo.policy.control_contract import EnvActionShapeError

    class ConnectionClosedError(Exception):
        pass

    assert R.classify_episode_exception(TimeoutError("x"), from_policy=True) == "policy_timeout"
    assert R.classify_episode_exception(PolicyTimeoutError("x"), from_policy=True) == "policy_timeout"
    assert R.classify_episode_exception(ConnectionRefusedError(), from_policy=True) == "policy_timeout"
    assert R.classify_episode_exception(ConnectionClosedError(), from_policy=True) == "policy_timeout"
    assert R.classify_episode_exception(EnvActionShapeError("nan"), from_policy=True) == "safety_termination"
    assert R.classify_episode_exception(SafetyTerminationError("nan"), from_policy=False) == "safety_termination"
    assert R.classify_episode_exception(RuntimeError("boom"), from_policy=False) == "env_crash"
    assert R.classify_episode_exception(ValueError("bad chunk"), from_policy=True) == "env_crash"
    # env-side TimeoutError still maps through episode_log.classify_exception
    assert R.classify_episode_exception(TimeoutError("x"), from_policy=False) == "policy_timeout"


def test_small_image_downscales_and_passes_through():
    big = np.zeros((720, 1280, 3), np.uint8)
    out = R.small_image(big)
    assert out.shape == (180, 320, 3) and out.dtype == np.uint8
    tiny = np.zeros((90, 160, 3), np.uint8)
    assert R.small_image(tiny).shape == (90, 160, 3)


# ---------------------------------------------- renderer-free rig geometry (CPU)

@pytest.mark.skipif(not (HAVE_SCENE and HAVE_MENAGERIE), reason="c50d2d1d42_factory scene / menagerie missing")
def test_pinch_site_is_between_finger_pads_cpu():
    from robo.rigs import pi05_rig as rig
    suite = json.loads(SUITE_JSON.read_text())
    m, info = rig.build_scene_model(suite["scene_xml"], suite["robot"]["base_pos"],
                                    suite["robot"]["base_yaw"], table_box=suite["table"],
                                    ext_cam=suite["ext_cam"],
                                    exclude_objects=tuple(suite["exclude_objects"]))
    d = mujoco.MjData(m)
    d.qpos[[m.joint(n).qposadr[0] for n in info["arm_joints"]]] = info["home"]
    mujoco.mj_forward(m, d)
    pinch = d.site(R.RobotSession.EE_SITE).xpos
    base = d.body("robot/2f85/base")
    Rb = base.xmat.reshape(3, 3)
    z = Rb[:, 2]                                   # gripper approach axis
    pads = [g for g in range(m.ngeom) if "2f85" in (m.geom(g).name or "") and "pad" in m.geom(g).name]
    assert len(pads) >= 4
    along = [float(np.dot(d.geom_xpos[g] - base.xpos, z)) for g in pads]
    lateral = np.mean([d.geom_xpos[g] for g in pads], axis=0) - base.xpos
    lateral -= np.dot(lateral, z) * z
    p_along = float(np.dot(pinch - base.xpos, z))
    assert p_along == pytest.approx(0.145, abs=1e-6)
    assert min(along) - 0.02 <= p_along <= max(along) + 0.02     # within the pad span
    assert np.linalg.norm((pinch - base.xpos) - p_along * z) < 1e-6  # on the centre axis
    assert np.linalg.norm(lateral) < 5e-3                         # pads symmetric about it

    # IK on the full rig (CPU): a 5 cm move of the TCP from home round-trips.
    from physicalview import ik
    jadr = np.array([m.joint(n).qposadr[0] for n in info["arm_joints"]])
    dadr = np.array([m.joint(n).dofadr[0] for n in info["arm_joints"]])
    pos, quat = ik.ee_pose(m, d, R.RobotSession.EE_SITE)
    res = ik.solve_ik(m, d, R.RobotSession.EE_SITE, pos + [0.05, 0.0, -0.05], quat, jadr, dadr,
                      q_init=info["home"], iters=100)
    assert res.converged and res.pos_err < 1e-3 and res.rot_err < 1e-2


# --------------------------------------------------------- RobotSession (GPU/EGL)

def _egl_ok() -> bool:
    try:
        m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size="1"/></worldbody></mujoco>')
        r = mujoco.Renderer(m, 16, 16)
        r.close()
        return True
    except Exception:  # noqa: BLE001 - EGL raises exotic error types
        return False


def _load_state() -> SceneState:
    from physicalview import scene_state as SS
    suite = json.loads(SUITE_JSON.read_text())
    try:
        objects = SS.load_objects(FACTORY)
    except Exception:  # noqa: BLE001 - keep the test independent of the scene agent
        objects = {}
    rs = ResultSet(name=FACTORY.name, scene_id="c50d2d1d42", kind="factory", out_dir=FACTORY,
                   scene_dir=None, splat_ply=None, mesh_ply=None, has_sim_export=True,
                   has_tasks=True, n_objects=len(objects), n_accepted=len(objects))
    return SceneState(result_set=rs, K=np.eye(3), W=0, H=0, cameras={}, splat_gs=None,
                      clean_bg_gs=None, mesh=None, objects=objects, tasks=suite,
                      scene_xml=FACTORY / "sim_export" / "scene.xml", sim_export=FACTORY / "sim_export",
                      timings={}, report=None)


@pytest.fixture(scope="module")
def session():
    if not (HAVE_SCENE and HAVE_MENAGERIE):
        pytest.skip("c50d2d1d42_factory scene / menagerie missing")
    if not _egl_ok():
        pytest.skip("MuJoCo offscreen rendering unavailable (needs MUJOCO_GL=egl + GPU)")
    s = R.RobotSession(_load_state(), load_config(), render_wh=(320, 180))
    yield s
    s.close()


def test_session_reset_and_hold(session):
    task = session.suite["tasks"][0]
    info = session.reset(task, seed=3, jitter=0.02)
    assert info.task_id == task["task_id"]
    assert set(info.body_poses) == set(session.free_bodies)
    assert task["target"] in session.body_reset_poses
    np.testing.assert_allclose(info.qpos, session.home, atol=0.05)
    assert info.obs[R.EXT_KEY].shape == (180, 320, 3)
    # jitter is seeded: same seed -> same target pose; different seed -> different
    p1 = session.reset(task, seed=3, jitter=0.02).body_poses[task["target"]][0]
    p2 = session.reset(task, seed=4, jitter=0.02).body_poses[task["target"]][0]
    p3 = session.reset(task, seed=3, jitter=0.02).body_poses[task["target"]][0]
    np.testing.assert_allclose(p1, p3, atol=1e-6)
    assert np.linalg.norm(p1[:2] - p2[:2]) > 1e-4
    # hold_tick keeps the arm at home and reports all stages False
    for _ in range(10):
        ti = session.hold_tick()
    assert isinstance(ti, R.TickInfo)
    assert ti.stages == {"grasp": False, "lift": False, "hover": False, "place": False}
    assert ti.sim_time > 0 and ti.qpos.shape == (7,) and 0.0 <= ti.gripper <= 1.0
    assert "ncon" in ti.contacts and "grasped" in ti.contacts
    np.testing.assert_allclose(ti.qpos, session.home, atol=0.05)


def test_session_joint_and_ee_targets(session):
    session.reset(session.suite["tasks"][0], seed=0, jitter=0.0)
    q, g = session.joint_targets()
    np.testing.assert_allclose(q, session.home)
    assert g == 0.0
    # out-of-range targets are clipped to jnt_range
    session.set_joint_targets(np.full(7, 10.0), 2.0)
    q, g = session.joint_targets()
    assert np.all(q <= session.joint_ranges[:, 1] + 1e-12) and g == 1.0
    session.set_joint_targets(session.home, 0.0)
    # EE target: move the TCP 4 cm down and 3 cm forward, keep orientation
    pos0, quat0 = session.ee_pose()
    target = pos0 + np.array([0.03, 0.0, -0.04])
    res = session.ee_target(target, quat0)
    assert res.converged and res.pos_err < 1e-3 and res.rot_err < 1e-2
    q_t, _ = session.joint_targets()
    np.testing.assert_allclose(q_t, res.q)
    for _ in range(45):  # 3 s of settling (<= 0.2 rad/tick clamp)
        session.hold_tick()
    pos1, _ = session.ee_pose()
    assert np.linalg.norm(pos1 - target) < 0.02
    # unreachable: no exception, targets untouched
    res2 = session.ee_target(pos0 + np.array([3.0, 0.0, 0.0]), quat0)
    assert not res2.converged
    np.testing.assert_allclose(session.joint_targets()[0], q_t)


def test_session_obs_and_mask(session):
    obs = session.obs("raster", None)
    assert obs["_mode"] == "raster"
    assert obs[R.EXT_KEY].shape == (180, 320, 3) and obs[R.WRIST_KEY].shape == (180, 320, 3)
    assert obs[R.JOINT_KEY].shape == (7,) and obs[R.GRIP_KEY].shape == (1,)
    # composite without a renderer degrades to raster and says so
    assert session.obs("composite", None)["_mode"] == "raster"
    mask = session.robot_mask(session.info["wrist_cam"])
    assert mask.shape == (180, 320) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}
    assert (mask > 0).mean() > 0.02  # the gripper housing is visible in the wrist cam
    ext_mask = session.robot_mask(session.info["ext_cam"])
    assert (ext_mask > 0).any()


def test_session_scripted_episode_streams_ticks(session):
    from robo.rigs import pi05_rig as rig
    task = session.suite["tasks"][0]
    session.reset(task, seed=0, jitter=0.0)
    policy = R.make_policy(load_config(), "scripted_sinusoid", "localhost", 8000, session.home)
    seen = []

    def on_tick(info, imgs):
        assert isinstance(info, R.TickInfo)
        assert imgs["ext"].shape[0] <= 180 and imgs["ext"].shape[1] <= 320
        assert imgs["wrist"].dtype == np.uint8
        seen.append(info.sim_time)

    res = session.run_episode(policy, task["instructions"]["default"], 3.0, "raster",
                              on_tick, threading.Event())
    assert isinstance(res, R.EpisodeResult)
    assert res.ticks == int(3.0 * rig.CONTROL_HZ) == len(seen)
    assert res.outcome == "task_failure" and res.success is False  # sinusoid never places
    assert 0.0 <= res.score <= 1.0 and set(res.stages) == set(R.STAGE_NAMES)
    assert res.error is None and res.wall_s > 0
    assert np.all(np.diff(seen) > 0)
    assert policy.warmed_up


def test_session_episode_stop_event_and_policy_failure(session):
    task = session.suite["tasks"][0]
    session.reset(task, seed=0, jitter=0.0)
    policy = R.make_policy(load_config(), "scripted_sinusoid", "localhost", 8000, session.home)
    stop = threading.Event()
    n = {"k": 0}

    def on_tick(info, imgs):
        n["k"] += 1
        if n["k"] >= 5:
            stop.set()

    res = session.run_episode(policy, "x", 10.0, "raster", on_tick, stop)
    assert res.outcome == "stopped" and res.ticks == 5

    class Flaky:
        warmed_up = True

        def reset(self):
            pass

        def __call__(self, obs, prompt):
            raise TimeoutError("server gone")

    res2 = session.run_episode(Flaky(), "x", 2.0, "raster", None, threading.Event())
    assert res2.outcome == "policy_timeout" and res2.ticks == 0 and "TimeoutError" in res2.error

    class Nan:
        warmed_up = True

        def reset(self):
            pass

        def __call__(self, obs, prompt):
            return np.full(8, np.nan)

    res3 = session.run_episode(Nan(), "x", 2.0, "raster", None, threading.Event())
    assert res3.outcome == "safety_termination" and res3.ticks == 0
    # the session is still usable afterwards
    ti = session.hold_tick()
    assert np.all(np.isfinite(ti.qpos))
