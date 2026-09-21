"""Tests for robo/eval/paired_runner.py, robo/eval/episode_log.py, and
robo/eval/reference_scene.py (Task 09, plan/09_PAIRED_ROLLOUT_RUNNER.md).
Design rationale: docs/MUJOCO_PAIRED_PROTOCOL.md.

Every test below that builds a real `robo.envs.pi05_env.DroidSimEnv`
constructs a REAL, tiny, synthetic MuJoCo scene (two objects, a robot rig,
a floor) -- no mocked physics -- to keep runtime small while still
exercising the real env/scorer/policy code paths. This needs an actual GL
context (`mujoco.Renderer`, unconditionally constructed by DroidSimEnv):
on this cluster that means MUJOCO_GL=egl AND an active GPU allocation
(device nodes are root-only outside a slurm job), e.g.:

    srun -p debug --gres=gpu:a6000:1 -t 20:00 \\
        .venv/bin/python -m pytest -q tests/test_paired_runner.py

Run: .venv/bin/python -m pytest -q tests/test_paired_runner.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from robo.envs import pi05_env
from robo.eval import episode_log as elog
from robo.eval import paired_runner as pr
from robo.eval import reference_scene as refscene

pytestmark = pytest.mark.filterwarnings("ignore")


# ===================================================== synthetic fixture ==
# A hand-authored, minimal "SimAny factory build": two objects (a graspable
# "mug" and a "bowl" receptacle) sitting on a table in front of a robot
# base, small enough to build+step in well under a second, but real enough
# (real robot rig attach, real physics, real TaskScorer) to exercise the
# whole pipeline instead of mocking it.

_MUG_POS = (0.35, 0.05, 0.55)
_BOWL_POS = (0.35, -0.15, 0.53)


def _write_synthetic_factory(root: Path) -> dict:
    """Write objects.json/aligned.json/physics.json/object.urdf for two
    objects, plus a hand-authored "simany-like" scene.xml that places them
    at the EXACT SAME world pose aligned.json declares but with a
    DIFFERENT geom representation (capsule primitives) than
    robo.eval.reference_scene's box primitives -- so the "paired reset
    poses match" test is actually checking pose parity across two
    differently-shaped scenes, not comparing a module against itself.

    Returns a dict: factory_dir, tasks_json (path to the suite), scene_xml.
    """
    factory_dir = root / "synth_factory"
    objdir = factory_dir / "objects"
    objdir.mkdir(parents=True)

    specs = [
        (0, "obj_00", "mug", _MUG_POS, (0.08, 0.08, 0.10), 0.05, 0.6),
        (1, "obj_01", "bowl", _BOWL_POS, (0.12, 0.12, 0.06), 0.10, 0.5),
    ]

    objects_json = []
    for index, name, label, pos, dims, mass, friction in specs:
        odir = objdir / name
        odir.mkdir()
        T = [[1.0, 0.0, 0.0, pos[0]],
             [0.0, 1.0, 0.0, pos[1]],
             [0.0, 0.0, 1.0, pos[2]],
             [0.0, 0.0, 0.0, 1.0]]
        (odir / "aligned.json").write_text(json.dumps({
            "T": T, "world_dims": list(dims), "tier": "A", "rejected": False}))
        (odir / "physics.json").write_text(json.dumps({
            "mass_kg": mass, "friction": friction, "restitution": 0.3}))
        (odir / "object.urdf").write_text("<robot name='stub'/>")  # existence-only
        # robo/tasks/pi05_tasks.py::_load_objects (reused unmodified by
        # robo.eval.paired_runner.build_env to populate env._task_rows)
        # reads `aabb` unconditionally straight off the objects.json row
        # (a real factory build's SAM3/detector output, not aligned.json) --
        # a synthetic fixture has to supply one too, or that reuse crashes
        # with a KeyError before it ever reaches interesting test behavior.
        half = [d / 2.0 for d in dims]
        aabb = [[pos[i] - half[i] for i in range(3)],
                [pos[i] + half[i] for i in range(3)]]
        objects_json.append({"index": index, "label": label, "aabb": aabb})
    (objdir / "objects.json").write_text(json.dumps(objects_json))

    sim_export = factory_dir / "sim_export"
    sim_export.mkdir()
    scene_xml = sim_export / "scene.xml"
    scene_xml.write_text(_synthetic_simany_scene_xml())

    suite = {
        "scene": "synth_scene",
        "scene_xml": str(scene_xml),
        "robot": {"base_pos": [0.0, 0.0, 0.5], "base_yaw": 0.0},
        "table": {"cx": 0.5, "cy": 0.0, "hx": 0.4, "hy": 0.4, "top_z": 0.5},
        "ext_cam": {"pos": [0.05, 0.30, 0.35], "target": [0.4, 0.0, 0.55],
                    "fovy": 68.0},
        "exclude_objects": [],
        "time_limit_s": 6.0,
        "tasks": [{
            "task_id": "synth_scene__obj_00_into_obj_01",
            "target": "obj_00", "target_label": "mug", "any_instance": False,
            "receptacle": "obj_01", "receptacle_dims": [0.12, 0.12, 0.06],
            "instructions": {
                "default": "put the mug in the bowl",
                "vague": "put it away",
                "specific": "pick up the mug and place it in the bowl"},
        }],
    }
    tasks_json = sim_export / "pi05_tasks.json"
    tasks_json.write_text(json.dumps(suite, indent=1))
    return {"factory_dir": factory_dir, "tasks_json": tasks_json,
            "scene_xml": scene_xml, "suite": suite}


def _synthetic_simany_scene_xml() -> str:
    def body(name, pos, r, h, mass, friction, rgba):
        return f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="1 0 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="1e-5 1e-5 1e-5"/>
      <geom type="capsule" size="{r} {h}" contype="1" conaffinity="1"
            friction="{friction} 0.005 0.0001" rgba="{rgba}"/>
    </body>"""

    bodies = "\n".join([
        body("obj_00", _MUG_POS, 0.04, 0.03, 0.05, 0.6, "0.8 0.2 0.2 1"),
        body("obj_01", _BOWL_POS, 0.06, 0.02, 0.10, 0.5, "0.2 0.2 0.8 1"),
    ])
    return f"""<mujoco model="synth_scene">
  <compiler angle="radian" autolimits="true" balanceinertia="true"/>
  <option timestep="0.0016666667" integrator="implicitfast"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light directional="true" pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="0 0 0" size="5 5 1"
          friction="0.8 0.005 0.0001"/>
{bodies}
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired_runner_fixture")
    return _write_synthetic_factory(root)


def _scene_cfg(synth, **overrides):
    cfg = {"id": "synth_scene", "factory_dir": str(synth["factory_dir"]),
           "tasks_json": str(synth["tasks_json"]), "render_wh": (64, 64)}
    cfg.update(overrides)
    return cfg


def _base_config(synth, out_dir, **overrides):
    cfg = {
        "policy": "scripted_sinusoid",
        "seeds": [0],
        "episodes": 1,
        "horizon_s": 1.0,
        "jitter": 0.0,
        "variant": "default",
        "video": False,
        "policy_timeout_s": 5.0,
        "conditions": ["simany", "reference"],
        "scenes": [_scene_cfg(synth)],
        "out_dir": str(out_dir),
    }
    cfg.update(overrides)
    return cfg


# ============================================================ pure logic ==

def test_classify_exception_four_failure_categories_are_distinct():
    """Task 09 step 7, at the classification-table level: four DIFFERENT
    exception types map to four DIFFERENT Outcome values (a fifth,
    builtin TimeoutError, deliberately maps to the SAME value as
    PolicyTimeoutError -- that collapse is intentional, not a bug)."""
    mapping = {
        elog.BuildFailureError("no asset"): elog.Outcome.BUILD_FAILURE,
        elog.PolicyTimeoutError("too slow"): elog.Outcome.POLICY_TIMEOUT,
        elog.SafetyTerminationError("nan"): elog.Outcome.SAFETY_TERMINATION,
        RuntimeError("something else broke"): elog.Outcome.ENV_CRASH,
    }
    seen = set()
    for exc, expected in mapping.items():
        got = elog.classify_exception(exc)
        assert got == expected
        seen.add(got)
    assert len(seen) == 4, "the four exception types must map to 4 distinct outcomes"
    # builtin TimeoutError intentionally collapses onto POLICY_TIMEOUT
    assert elog.classify_exception(TimeoutError("hung")) == elog.Outcome.POLICY_TIMEOUT


def test_derive_reset_seed_is_pure_function_of_its_inputs():
    a = elog.derive_reset_seed(0, "task_a", 0)
    b = elog.derive_reset_seed(0, "task_a", 0)
    c = elog.derive_reset_seed(0, "task_a", 1)
    d = elog.derive_reset_seed(1, "task_a", 0)
    assert a == b  # same inputs -> same output, independent of call order
    assert a != c and a != d  # different ep / seed -> different draw
    assert 0 <= a < 2 ** 32


def test_episode_ledger_load_completed_ignores_truncated_last_line(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = elog.EpisodeLedger(path)
    rec = elog.EpisodeRecord(
        episode_id="simany__x", condition="simany", scene_id="s", task_id="t",
        reset_state_id="x", ep=0, base_seed=0, reset_seed=1,
        outcome=elog.Outcome.SUCCESS, success=True, score=1.0, stages={},
        ticks=1, error=None, video_path=None, timeseries_path=None,
        manifest_path=None, wall_s=0.1)
    ledger.append(rec)
    with open(path, "a") as f:
        f.write('{"episode_id": "simany__truncated", "condition": "sim')  # no newline
    completed = ledger.load_completed()
    assert set(completed) == {"simany__x"}


# ===================================================== reference scene ====

def test_build_reference_scene_xml_includes_all_valid_objects(synth):
    ref_path = synth["factory_dir"] / "reference_scene.xml"
    out_path, kept = refscene.build_reference_scene_xml(
        synth["factory_dir"], synth["suite"], ref_path)
    assert out_path.exists()
    assert sorted(kept) == ["obj_00", "obj_01"]
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(out_path))
    assert m.body("obj_00") is not None
    assert m.body("obj_01") is not None


def test_build_reference_scene_xml_raises_when_no_valid_objects(tmp_path):
    factory_dir = tmp_path / "empty_factory"
    (factory_dir / "objects").mkdir(parents=True)
    (factory_dir / "objects" / "objects.json").write_text("[]")
    with pytest.raises(ValueError):
        refscene.build_reference_scene_xml(
            factory_dir, {"scene": "empty"}, tmp_path / "ref.xml")


def test_paired_reset_poses_match_across_conditions(synth, tmp_path):
    """The core design-decision test: the "simany" scene (hand-authored
    fixture, capsule geoms) and the "reference" scene
    (robo.eval.reference_scene, box geoms) must place every object body at
    the EXACT SAME initial world pose, despite having totally different
    collision/visual representations -- this is what makes a paired
    comparison meaningful rather than accidental.
    """
    suite = synth["suite"]
    ref_path = tmp_path / "reference_scene.xml"
    ref_xml, _kept = refscene.build_reference_scene_xml(
        synth["factory_dir"], suite, ref_path)

    common = dict(base_pos=suite["robot"]["base_pos"],
                  base_yaw=suite["robot"]["base_yaw"],
                  table_box=suite["table"], ext_cam=suite["ext_cam"],
                  exclude_objects=(), render_wh=(64, 64))
    env_simany = pi05_env.DroidSimEnv(suite["scene_xml"], **common)
    env_reference = pi05_env.DroidSimEnv(str(ref_xml), **common)

    env_simany.reset(settle_s=0.0)
    env_reference.reset(settle_s=0.0)

    for name in ("obj_00", "obj_01"):
        pos_a, quat_a = env_simany.body_pose(name)
        pos_b, quat_b = env_reference.body_pose(name)
        assert np.allclose(pos_a, pos_b, atol=1e-6), (name, pos_a, pos_b)
        assert np.allclose(quat_a, quat_b, atol=1e-6), (name, quat_a, quat_b)


# ============================================================= run_matrix =

def test_deterministic_repeat_yields_identical_logs(synth, tmp_path):
    """Task 09 test 1: repeating one manifest (scripted policy, not a real
    server) yields identical deterministic logs across two independent
    runs."""
    config = _base_config(synth, tmp_path / "run_a", video=False)
    summary_a = pr.run_matrix(dict(config), tmp_path / "run_a")
    config_b = dict(config)
    summary_b = pr.run_matrix(config_b, tmp_path / "run_b")

    assert summary_a["coverage_by_outcome"] == summary_b["coverage_by_outcome"]

    ledger_a = elog.EpisodeLedger(tmp_path / "run_a" / "episode_ledger.jsonl").load_completed()
    ledger_b = elog.EpisodeLedger(tmp_path / "run_b" / "episode_ledger.jsonl").load_completed()
    assert set(ledger_a) == set(ledger_b)
    for episode_id, rec_a in ledger_a.items():
        rec_b = ledger_b[episode_id]
        for field in ("outcome", "success", "score", "stages", "ticks", "error"):
            assert getattr(rec_a, field) == getattr(rec_b, field), (episode_id, field)
        ts_a = elog.read_timeseries(rec_a.timeseries_path)
        ts_b = elog.read_timeseries(rec_b.timeseries_path)
        assert ts_a == ts_b, f"{episode_id}: per-tick time series diverged"


def test_coverage_includes_failed_builds(synth, tmp_path):
    """Task 09 test 4: a scene whose build is deliberately uninstantiable
    (its `scene_xml` points nowhere) must appear in the coverage
    denominator with outcome build_failure -- not vanish from it."""
    broken_suite = dict(synth["suite"])
    broken_suite["scene_xml"] = str(tmp_path / "does_not_exist.xml")
    broken_tasks_json = tmp_path / "broken_pi05_tasks.json"
    broken_tasks_json.write_text(json.dumps(broken_suite))

    scene_cfg = _scene_cfg(synth, tasks_json=str(broken_tasks_json))
    config = _base_config(synth, tmp_path / "run_broken", scenes=[scene_cfg])
    summary = pr.run_matrix(config, tmp_path / "run_broken")

    ledger = elog.EpisodeLedger(tmp_path / "run_broken" / "episode_ledger.jsonl")
    completed = ledger.load_completed()

    simany_records = [r for r in completed.values() if r.condition == "simany"]
    reference_records = [r for r in completed.values() if r.condition == "reference"]
    assert simany_records, "the broken condition must still produce a ledger row"
    assert all(r.outcome == elog.Outcome.BUILD_FAILURE for r in simany_records)
    assert all(r.error for r in simany_records)  # never a silent/empty error
    # "reference" doesn't depend on suite["scene_xml"] at all, so it must
    # build and run fine -- proving conditions fail independently, and a
    # broken simany build never silently swallows the reference episodes.
    assert reference_records
    assert not any(r.outcome == elog.Outcome.BUILD_FAILURE for r in reference_records)

    n_planned = len(elog.load_reset_states(tmp_path / "run_broken" / "reset_states.json"))
    assert summary["n_episode_records"] == 2 * n_planned  # both conditions, none dropped
    assert summary["coverage_by_outcome"]["build_failure"] == n_planned


def test_crash_and_restart_does_not_duplicate_or_reorder_episode_ids(
        synth, tmp_path, monkeypatch):
    """Task 09 test 3: a simulated mid-matrix crash (a controlled exception
    injected into `_finalize_episode`, well after `run_episode_core` has
    already produced a real, classified outcome -- modeling e.g. a disk
    error while writing artifacts, not an in-episode failure) must not
    duplicate or reorder episode_ids on restart; the pre-crash ledger
    prefix must be byte-identical after the restart completes the matrix.
    """
    out_dir = tmp_path / "run_crash"
    config = _base_config(synth, out_dir, episodes=2, video=False)

    real_finalize = pr._finalize_episode
    call_count = {"n": 0}

    def flaky_finalize(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise RuntimeError("simulated process crash mid-matrix")
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(pr, "_finalize_episode", flaky_finalize)
    with pytest.raises(RuntimeError, match="simulated process crash"):
        pr.run_matrix(config, out_dir)

    ledger_path = out_dir / "episode_ledger.jsonl"
    pre_crash_lines = ledger_path.read_text().splitlines()
    assert len(pre_crash_lines) == 2  # the 2 episodes before the 3rd call blew up

    monkeypatch.setattr(pr, "_finalize_episode", real_finalize)
    summary = pr.run_matrix(config, out_dir)  # restart, same config/out_dir

    reset_states = elog.load_reset_states(out_dir / "reset_states.json")
    expected_total = 2 * len(reset_states)  # 2 conditions
    all_lines = ledger_path.read_text().splitlines()
    assert len(all_lines) == expected_total, "no duplicate/missing lines after restart"

    episode_ids = [json.loads(line)["episode_id"] for line in all_lines]
    assert len(episode_ids) == len(set(episode_ids)), "an episode_id was duplicated"
    assert episode_ids[:2] == [json.loads(line)["episode_id"] for line in pre_crash_lines], (
        "restart must not rewrite/reorder the pre-crash ledger prefix")
    assert summary["n_episode_records"] == expected_total


def test_resume_skips_already_completed_episodes_without_rerunning(synth, tmp_path, monkeypatch):
    """A second run_matrix call against an out_dir that already finished
    must not re-invoke run_episode_core for any already-completed
    episode_id (not just "not duplicate the ledger line" -- literally
    never re-executes physics for it)."""
    out_dir = tmp_path / "run_resume"
    config = _base_config(synth, out_dir, video=False)
    pr.run_matrix(config, out_dir)

    calls = []
    real_run_episode_core = pr.run_episode_core

    def counting_run_episode_core(*args, **kwargs):
        calls.append(1)
        return real_run_episode_core(*args, **kwargs)

    monkeypatch.setattr(pr, "run_episode_core", counting_run_episode_core)
    pr.run_matrix(config, out_dir)
    assert calls == [], "a fully-completed matrix must not re-run any episode"


# ===================================================== outcome categories =

class _CrashPolicy:
    def reset(self):
        pass

    def __call__(self, obs, prompt):
        raise RuntimeError("policy blew up")


class _SlowPolicy:
    def __init__(self, home, sleep_s):
        self.home, self.sleep_s = home, sleep_s

    def reset(self):
        pass

    def __call__(self, obs, prompt):
        time.sleep(self.sleep_s)
        return np.concatenate([self.home, [0.0]])


class _NaNPolicy:
    def __init__(self, home):
        self.home = home

    def reset(self):
        pass

    def __call__(self, obs, prompt):
        a = np.concatenate([self.home, [0.0]])
        a[0] = np.nan
        return a


@pytest.fixture(scope="module")
def simany_env(synth):
    env, suite, factory_dir = pr.build_env(_scene_cfg(synth), "simany",
                                           synth["factory_dir"].parent)
    return env, suite


def test_outcome_env_crash_is_distinguishable(simany_env):
    env, suite = simany_env
    task = suite["tasks"][0]
    outcome, *_ = pr.run_episode_core(
        env, task, _CrashPolicy(), "test", horizon_s=1.0, reset_seed=0)
    assert outcome == elog.Outcome.ENV_CRASH


def test_outcome_policy_timeout_is_distinguishable(simany_env):
    env, suite = simany_env
    task = suite["tasks"][0]
    home = pi05_env.rig.PANDA_HOME
    outcome, *_ = pr.run_episode_core(
        env, task, _SlowPolicy(home, sleep_s=0.05), "test", horizon_s=1.0,
        reset_seed=0, policy_timeout_s=0.01)
    assert outcome == elog.Outcome.POLICY_TIMEOUT


def test_outcome_safety_termination_is_distinguishable(simany_env):
    env, suite = simany_env
    task = suite["tasks"][0]
    home = pi05_env.rig.PANDA_HOME
    outcome, *_ = pr.run_episode_core(
        env, task, _NaNPolicy(home), "test", horizon_s=1.0, reset_seed=0)
    assert outcome == elog.Outcome.SAFETY_TERMINATION


def test_outcome_build_failure_is_distinguishable(synth, tmp_path):
    bad_cfg = _scene_cfg(synth, tasks_json=str(tmp_path / "missing.json"))
    with pytest.raises(elog.BuildFailureError):
        pr.build_env(bad_cfg, "simany", tmp_path)


def test_outcome_task_failure_is_distinguishable(simany_env):
    """A real (scripted, deterministic) rollout too short to ever satisfy
    the rubric ends in TASK_FAILURE, not success, crash, or timeout."""
    env, suite = simany_env
    task = suite["tasks"][0]
    home = pi05_env.rig.PANDA_HOME

    class _NoOpPolicy:
        def reset(self):
            pass

        def __call__(self, obs, prompt):
            return np.concatenate([home, [0.0]])

    outcome, summary, *_ = pr.run_episode_core(
        env, task, _NoOpPolicy(), "test", horizon_s=0.5, reset_seed=0)
    assert outcome == elog.Outcome.TASK_FAILURE
    assert summary["success"] is False


def test_five_outcome_categories_are_pairwise_distinct():
    """The five failure/task categories Task 09 step 7 requires
    (build_failure, policy_timeout, safety_termination, env_crash,
    task_failure) really are five distinct enum values, not aliases."""
    values = {elog.Outcome.BUILD_FAILURE, elog.Outcome.POLICY_TIMEOUT,
              elog.Outcome.SAFETY_TERMINATION, elog.Outcome.ENV_CRASH,
              elog.Outcome.TASK_FAILURE}
    assert len(values) == 5


# ==================================================== degenerate diagnostic

def test_check_degenerate_flags_all_zero_but_preserves_records():
    records = [
        elog.EpisodeRecord(
            episode_id=f"simany__x{i}", condition="simany", scene_id="s",
            task_id="t", reset_state_id=f"x{i}", ep=i, base_seed=0,
            reset_seed=i, outcome=elog.Outcome.TASK_FAILURE, success=False,
            score=0.0, stages={}, ticks=10, error=None, video_path=None,
            timeseries_path=None, manifest_path=None, wall_s=0.1)
        for i in range(5)
    ]
    msg = elog.check_degenerate(records, min_n=4)
    assert msg is not None and "all zero" in msg.lower()
    assert len(records) == 5  # never discarded


def test_check_degenerate_silent_when_scores_vary():
    records = [
        elog.EpisodeRecord(
            episode_id=f"simany__x{i}", condition="simany", scene_id="s",
            task_id="t", reset_state_id=f"x{i}", ep=i, base_seed=0,
            reset_seed=i, outcome=elog.Outcome.TASK_FAILURE, success=False,
            score=(0.25 if i % 2 else 0.0), stages={}, ticks=10, error=None,
            video_path=None, timeseries_path=None, manifest_path=None, wall_s=0.1)
        for i in range(5)
    ]
    assert elog.check_degenerate(records, min_n=4) is None


def test_check_degenerate_ignores_build_failures_in_the_all_zero_check():
    """A run that is mostly build failures plus a genuinely mixed handful
    of scored episodes must NOT be flagged as degenerate just because a
    naive check counted the build failures' placeholder score=0.0 too."""
    records = [
        elog.EpisodeRecord(
            episode_id=f"simany__bf{i}", condition="simany", scene_id="s",
            task_id="t", reset_state_id=f"bf{i}", ep=i, base_seed=0,
            reset_seed=i, outcome=elog.Outcome.BUILD_FAILURE, success=False,
            score=0.0, stages={}, ticks=0, error="boom", video_path=None,
            timeseries_path=None, manifest_path=None, wall_s=0.0)
        for i in range(20)
    ] + [
        elog.EpisodeRecord(
            episode_id=f"simany__ok{i}", condition="simany", scene_id="s",
            task_id="t", reset_state_id=f"ok{i}", ep=i, base_seed=0,
            reset_seed=i, outcome=elog.Outcome.TASK_FAILURE, success=False,
            score=(0.25 if i % 2 else 0.0), stages={}, ticks=10, error=None,
            video_path=None, timeseries_path=None, manifest_path=None, wall_s=0.1)
        for i in range(4)
    ]
    assert elog.check_degenerate(records, min_n=4) is None
