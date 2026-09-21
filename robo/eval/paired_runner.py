"""robo.eval.paired_runner: mujoco_paired protocol matrix runner (Task 09,
plan/09_PAIRED_ROLLOUT_RUNNER.md). Design decisions (what an "official/
reference" condition even means without PolaRiS/Isaac Lab) are documented
in docs/MUJOCO_PAIRED_PROTOCOL.md.

Runs a policy x scene x condition x seed x episode matrix where "condition"
is one of:
  - "simany":           the SimAny reconstruction's own scene.xml
                        (robo/sim/export_mjcf.py's real output).
  - "reference":        the hand-built primitive-box stand-in for the same
                        scene (robo.eval.reference_scene), holding robot/
                        camera/control/rubric/object-initial-pose
                        byte-identical.
  - "simfoundry_repro": the SimFoundry-reproduction ablation's own
                        reconstruction of the same scene (baselines/
                        simfoundry_repro.py: one representative frame,
                        single-view TRELLIS, no GT), exported through the
                        SAME robo/sim/export_mjcf.py via
                        robo.eval.simfoundry_condition. Admissible only
                        where configs/experiments/icra_contract_v1.yaml's
                        mujoco_paired protocol lists it
                        (admissible_baselines).

Every other axis (robot base pose, camera, control contract, rubric,
per-episode reset seed) is generated ONCE up front
(`plan_reset_states`/`robo.eval.episode_log.ResetState`) and re-used
verbatim by both conditions -- Task 09 step 1's central invariant. Outcomes
are logged to an append-only `robo.eval.episode_log.EpisodeLedger` keyed by
a deterministic `episode_id`, which is what makes resume-at-episode-
granularity (step 5) and coverage-including-build-failures (step 2)
mechanical rather than ad hoc.

Fast-validation (plan/09, no real policy server needed):
    .venv/bin/python -m robo.eval.paired_runner \\
        --config configs/experiments/paired_smoke.yaml --episodes 2

Real matrix (via slurm): run/slurm/paired_matrix.sbatch.
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import yaml

from robo.envs import pi05_env
from robo.eval import episode_log as elog
from robo.eval import pi05_eval
from robo.eval import reference_scene as refscene
from robo.eval import simfoundry_condition as sfcond
from robo.manifest import hash as manifest_hash
from robo.manifest import io as manifest_io
from robo.manifest.schema import ObservationPreprocessing, RolloutManifest, StagedProgress
from robo.policy import control_contract as cc
from robo.tasks import pi05_tasks

ROOT = Path(__file__).resolve().parents[2]
FROZEN_FIELDS_PATH = ROOT / "configs" / "experiments" / "frozen_fields.yaml"

# NOTE: "simfoundry_repro" is a selectable third condition (see
# robo.eval.simfoundry_condition) but deliberately NOT part of the default
# CONDITIONS tuple `run_matrix` falls back to when a config declares
# neither its own `conditions:` list nor an explicit `--conditions` CLI
# override -- existing configs (e.g. configs/experiments/paired_smoke.yaml,
# which DOES declare `conditions: [simany, reference]` explicitly anyway)
# must not silently start attempting a baseline whose construction job may
# not exist/have finished for that scene. Opt in via `conditions:` in the
# config or `--conditions simany reference simfoundry_repro` on the CLI.
CONDITIONS = ("simany", "reference")
ALL_CONDITIONS = ("simany", "reference", "simfoundry_repro")
DEFAULT_POLICY_TIMEOUT_S = 20.0

# Task 08's registry owns checkpoint and control-contract validation.  A
# missing registry may retain the scripted smoke fallback, but a real policy
# is never directly constructed around registry/import/validation failures.
try:
    from robo.policy.registry import PolicyRegistry
    _HAVE_REGISTRY = True
except ImportError:
    PolicyRegistry = None  # noqa: N816
    _HAVE_REGISTRY = False


# --------------------------------------------------------------- utilities

def _resolve(root: Path, p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def _frozen_policies() -> dict:
    frozen = yaml.safe_load(FROZEN_FIELDS_PATH.read_text())
    return {p["id"]: p for p in frozen["policies"]}


def get_policy(policy_id: str, home_pose, *, host="localhost", port=8000,
              open_loop_horizon=15, expected_server_identity=None):
    """Construct a policy client for ``policy_id`` fail-closed.

    Registry, checkpoint, control-contract, and server-identity failures
    propagate to the caller.  Only the checkpoint-free scripted smoke policy
    has a compatibility path when the registry module itself is unavailable.
    """
    if _HAVE_REGISTRY:
        registry = PolicyRegistry.from_config_dir()
        entry = registry.get(policy_id)
        # Each client_kind's from_entry() takes a different kwarg set.
        if entry.client_kind == "scripted":
            client_kwargs = {"home": home_pose}
        else:
            client_kwargs = {
                "host": host, "port": port,
                "expected_server_identity": expected_server_identity,
            }
        client = registry.make_client(policy_id, **client_kwargs)
        if entry.client_kind != "scripted":
            # Websocket construction receives server metadata before any
            # inference.  Authenticate it here so callers cannot accidentally
            # turn an identity mismatch into an episode-level build failure.
            client.verify_server_identity()
        return client
    policies = _frozen_policies()
    if policy_id not in policies:
        raise ValueError(
            f"policy {policy_id!r} is not declared in {FROZEN_FIELDS_PATH}'s "
            f"`policies` list ({sorted(policies)})")
    entry = policies[policy_id]
    if entry.get("action_output") == "scripted_smoke_test_only":
        return pi05_eval.ScriptedPolicy(home_pose)
    raise RuntimeError(
        "robo.policy.registry is required for every real policy; refusing "
        "unverified direct ServerPolicy construction")


def _frozen_config_hashes():
    return pi05_eval._frozen_config_hashes()


def _policy_checkpoint_hash(policy_id: str, checkpoint_path: str = "") -> "str | None":
    """Mirrors robo/eval/pi05_eval.py::_policy_checkpoint_hash's semantics
    (scripted -> None; a real local path -> a real fingerprint) without needing to
    fake an argparse.Namespace to call that function directly."""
    if policy_id == "scripted_sinusoid":
        return None
    if checkpoint_path:
        p = Path(checkpoint_path)
        if p.exists():
            return manifest_hash.hash_checkpoint_path(p)
        raise FileNotFoundError(
            f"checkpoint_path for real policy {policy_id!r} does not exist: {p}")
    raise ValueError(
        f"real policy {policy_id!r} requires an explicit local checkpoint_path")


# ---------------------------------------------------------- reset planning

def plan_reset_states(scene_cfgs, seeds, episodes, task_filter="") -> "list[elog.ResetState]":
    """Generate the FULL reset-state plan once, from the scene configs'
    OWN task suites -- Task 09 step 1. Every condition later replays this
    exact list; nothing about a condition feeds back into this function.
    """
    states = []
    for sc in scene_cfgs:
        suite_path = _resolve(ROOT, sc["tasks_json"])
        suite = json.loads(suite_path.read_text())
        for task in suite["tasks"]:
            if task_filter and task_filter not in task["task_id"]:
                continue
            for seed in seeds:
                for ep in range(episodes):
                    rsid = f"{task['task_id']}__seed{seed}__ep{ep}"
                    states.append(elog.ResetState(
                        reset_state_id=rsid, scene_id=suite["scene"],
                        task_id=task["task_id"], ep=ep, base_seed=seed,
                        reset_seed=elog.derive_reset_seed(seed, task["task_id"], ep)))
    return states


def _get_or_plan_reset_states(out_dir, scene_cfgs, seeds, episodes, task_filter):
    path = Path(out_dir) / "reset_states.json"
    if path.exists():
        states = elog.load_reset_states(path)
        print(f"[paired_runner] reusing {len(states)} reset states already "
              f"persisted at {path} (Task 09 step 1: never resampled) -- any "
              f"--episodes/seeds on this invocation are ignored", flush=True)
        return states
    states = plan_reset_states(scene_cfgs, seeds, episodes, task_filter)
    elog.save_reset_states(states, path)
    print(f"[paired_runner] generated + persisted {len(states)} reset states "
          f"-> {path}", flush=True)
    return states


# ------------------------------------------------------------- env builder

def build_env(scene_cfg: dict, condition: str, work_dir):
    """Build one DroidSimEnv for (scene, condition). NEVER raises a bare
    exception -- always `elog.BuildFailureError`, so the caller can log a
    coverage failure for every reset_state that would have used this env
    instead of an unhandled crash (Task 09 step 2: "missing assets/
    uninstantiable states are coverage failures, not skips").

    Returns (env, suite_dict, factory_dir).
    """
    try:
        suite_path = _resolve(ROOT, scene_cfg["tasks_json"])
        if not suite_path.exists():
            raise FileNotFoundError(f"tasks suite not found: {suite_path}")
        suite = json.loads(suite_path.read_text())
        factory_dir = _resolve(ROOT, scene_cfg.get("factory_dir", suite_path.parent.parent))

        if condition == "simany":
            scene_xml = Path(suite["scene_xml"])
            if not scene_xml.exists():
                raise FileNotFoundError(
                    f"reconstructed scene_xml not found: {scene_xml}")
        elif condition == "reference":
            ref_path = Path(work_dir) / "scenes" / f"{suite['scene']}__reference_scene.xml"
            scene_xml, _kept = refscene.build_reference_scene_xml(factory_dir, suite, ref_path)
        elif condition == "simfoundry_repro":
            # A genuinely separate reconstruction of the same scene (see
            # robo.eval.simfoundry_condition's module docstring) -- its own
            # baseline_dir stands in for `factory_dir` from here on, so
            # `_scene_manifest_hash`'s object_inventory below reflects THIS
            # condition's own objects, not the SimAny factory build's.
            # `suite=suite` lets resolve_condition re-ground each task's
            # target/receptacle onto THIS build's own object ids
            # (robo.eval.task_regrounding) instead of assuming the SAME
            # obj_NN index names the same physical object in both builds.
            resolved = sfcond.resolve_condition(scene_cfg, suite=suite)
            scene_xml, factory_dir = resolved["scene_xml"], resolved["baseline_dir"]
            suite = resolved.get("suite", suite)
        else:
            raise ValueError(f"unknown condition {condition!r}")

        render_wh = tuple(scene_cfg.get("render_wh", (640, 360)))
        env = pi05_env.DroidSimEnv(
            str(scene_xml), suite["robot"]["base_pos"], suite["robot"]["base_yaw"],
            table_box=suite["table"], ext_cam=suite["ext_cam"],
            exclude_objects=tuple(suite.get("exclude_objects", ())),
            render_wh=render_wh,
            menagerie_root=scene_cfg.get("menagerie_root"),
            xml_dump=Path(work_dir) / "scenes" / f"{suite['scene']}__{condition}_rig.xml")
        env._task_rows = pi05_tasks._load_objects(factory_dir)
        return env, suite, factory_dir
    except elog.BuildFailureError:
        raise
    except Exception as e:  # noqa: BLE001 -- classified into one error type
        raise elog.BuildFailureError(
            f"{condition} build for scene {scene_cfg.get('id', '?')!r}: "
            f"{type(e).__name__}: {e}") from e


def _scene_manifest_hash(suite: dict, condition: str, factory_dir) -> str:
    """Same field set robo/eval/pi05_eval.py hashes for its own
    scene_manifest_hash (scene/scene_xml/robot/table/ext_cam/
    exclude_objects), plus condition-distinguishing fields for
    "reference"/"simfoundry_repro" -- so a "simany" hash from this module
    is directly comparable to one pi05_eval.py itself would produce, and
    the other two conditions' hashes are guaranteed to differ (asset
    identity/provenance is exactly what mujoco_paired is allowed to vary)
    while every FROZEN_ROLLOUT_FIELD (robo/manifest/schema.py) stays
    untouched.
    """
    fields = {
        "scene": suite["scene"], "condition": condition,
        "robot": suite["robot"], "table": suite["table"],
        "ext_cam": suite["ext_cam"],
        "exclude_objects": sorted(suite.get("exclude_objects", ())),
    }
    if condition == "simany":
        fields["scene_xml"] = suite["scene_xml"]
    elif condition == "simfoundry_repro":
        fields["scene_construction"] = "simfoundry_repro_v1"
        fields["object_inventory"] = sorted(
            r["object_id"] for r in sfcond.instance_inventory(factory_dir))
    else:
        fields["scene_construction"] = "reference_primitive_boxes_v1"
        fields["object_inventory"] = sorted(
            r["object_id"] for r in refscene.instance_inventory(factory_dir))
    return manifest_hash.canonical_hash(fields)


# --------------------------------------------------------------- one tick

def _tick_row(env, action, k, scorer):
    objects = {}
    for name in env.free_bodies:
        pos, quat = env.body_pose(name)
        objects[name] = {"pos": pos.tolist(), "quat_wxyz": quat.tolist()}
    return {
        "t": k,
        "joint_position": env.joint_position().tolist(),
        "gripper_position": env.gripper_position().tolist(),
        "action": np.asarray(action, dtype=float).tolist(),
        "objects": objects,
        "contacts": _current_contacts(env),
        "stages": {t: dict(st) for t, st in scorer.stages.items()},
    }


def _current_contacts(env):
    pairs = set()
    d = env.data
    for k in range(d.ncon):
        c = d.contact[k]
        b1 = env.model.body(env.model.geom_bodyid[c.geom1]).name
        b2 = env.model.body(env.model.geom_bodyid[c.geom2]).name
        pairs.add(tuple(sorted((b1, b2))))
    return sorted(pairs)


def _video_frame(obs):
    import cv2
    ext = obs["observation/exterior_image_1_left"]
    wr = obs["observation/wrist_image_left"]
    h = ext.shape[0]
    scale = h / wr.shape[0]
    wr = cv2.resize(wr, (int(wr.shape[1] * scale), h))
    return np.concatenate([ext, wr], axis=1)


# ------------------------------------------------------------- one episode

def run_episode_core(env, task, policy, prompt, horizon_s, reset_seed, *,
                     jitter_xy=0.0, policy_timeout_s=DEFAULT_POLICY_TIMEOUT_S,
                     capture_video=False):
    """One episode: reset -> tick loop -> outcome.

    Mirrors robo/eval/pi05_eval.py::run_episode's structure (same
    env/scorer calls: env.reset / policy.reset / TaskScorer / apply_action
    / get_obs / scorer.update) but:

      (a) uses a PER-EPISODE `np.random.RandomState(reset_seed)` instead of
          one RandomState threaded across an entire suite run, so the
          jitter draw depends only on `reset_seed` (Task 09 step 1) and
          not on call order -- see episode_log.derive_reset_seed;
      (b) records a full per-tick time series (joint/gripper/action/every
          object pose/contacts/stage booleans), not just the final staged
          score (Task 09 step 4);
      (c) classifies every failure into one of episode_log.Outcome's five
          non-success categories (Task 09 step 7) instead of a single
          collapsed "failure."

    Returns (outcome, summary_dict, ticks_series, frames_or_None,
             error_str_or_None, n_ticks, wall_s).
    """
    rng = np.random.RandomState(reset_seed)
    t0 = time.time()
    ticks_series = []
    frames = [] if capture_video else None
    k = -1
    try:
        obs = env.reset(jitter_body=task["target"] if jitter_xy > 0 else None,
                        jitter_xy=jitter_xy, rng=rng)
        policy.reset()
        scorer = pi05_tasks.TaskScorer(env, task)
        n_ticks = max(int(horizon_s * pi05_env.rig.CONTROL_HZ), 1)
        for k in range(n_ticks):
            tp0 = time.time()
            action = policy(obs, prompt)
            infer_s = time.time() - tp0
            if infer_s > policy_timeout_s:
                raise elog.PolicyTimeoutError(
                    f"policy call took {infer_s:.1f}s, over the "
                    f"{policy_timeout_s:.1f}s budget, at tick {k}")
            action = np.asarray(action, dtype=float)
            if not np.isfinite(action).all():
                raise elog.SafetyTerminationError(
                    f"non-finite action at tick {k}: {action.tolist()}")
            env.apply_action(action)
            obs = env.get_obs()
            if not np.isfinite(env.data.qpos).all():
                raise elog.SafetyTerminationError(
                    f"non-finite simulator state (qpos) at tick {k}")
            scorer.update()
            ticks_series.append(_tick_row(env, action, k, scorer))
            if capture_video:
                frames.append(_video_frame(obs))
            if scorer.success:
                break
        summary = scorer.summary()
        outcome = (elog.Outcome.SUCCESS if summary["success"]
                  else elog.Outcome.TASK_FAILURE)
        return outcome, summary, ticks_series, frames, None, (k + 1), time.time() - t0
    except Exception as e:  # noqa: BLE001 -- classified below, never swallowed
        outcome = elog.classify_exception(e)
        summary = {"success": False, "score": 0.0,
                   "stages": {"grasp": False, "lift": False,
                              "hover": False, "place": False},
                   "scored_target": None, "wrong_grasps": []}
        return (outcome, summary, ticks_series, frames,
               f"{type(e).__name__}: {e}", len(ticks_series), time.time() - t0)


# ------------------------------------------------------------- persistence

def _make_record(cond, st, outcome, summary, *, error, ticks, wall_s,
                 video_path=None, timeseries_path=None, manifest_path=None):
    return elog.EpisodeRecord(
        episode_id=elog.episode_id_for(cond, st.reset_state_id),
        condition=cond, scene_id=st.scene_id, task_id=st.task_id,
        reset_state_id=st.reset_state_id, ep=st.ep, base_seed=st.base_seed,
        reset_seed=st.reset_seed, outcome=outcome,
        success=bool(summary.get("success", False)),
        score=float(summary.get("score", 0.0)),
        stages=summary.get("stages", {}) or {},
        ticks=ticks, error=error, video_path=video_path,
        timeseries_path=timeseries_path, manifest_path=manifest_path,
        wall_s=wall_s)


def _log_immediate_failure(ledger, completed, cond, st, outcome, error):
    """Log a terminal outcome for a reset_state that never got to run an
    episode at all (e.g. its scene/env failed to build). Idempotent: a
    no-op if this episode_id is already in `completed` (resume safety)."""
    episode_id = elog.episode_id_for(cond, st.reset_state_id)
    if episode_id in completed:
        return completed[episode_id]
    rec = _make_record(cond, st, outcome, {}, error=error, ticks=0, wall_s=0.0)
    ledger.append(rec)
    completed[episode_id] = rec
    return rec


def _finalize_episode(out_dir, cond, st, task, suite, outcome, summary,
                      ticks_series, frames, error, ticks, wall_s, *,
                      git_snap, controller_hash, camera_hash,
                      action_convention, action_dim, factory_dir,
                      horizon_s, variant, policy_checkpoint_hash):
    """Write per-episode artifacts (timeseries, video, RolloutManifest)
    and return the ledger record. This is a free function (not a method)
    specifically so tests can monkeypatch it in isolation -- see
    tests/test_paired_runner.py's crash/restart test.
    """
    episode_id = elog.episode_id_for(cond, st.reset_state_id)
    ep_dir = Path(out_dir) / "episodes" / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    ts_path = None
    if ticks_series:
        ts_path = ep_dir / "timeseries.json.gz"
        elog.write_timeseries(ts_path, ticks_series)

    video_path = None
    if frames:
        import imageio.v2 as imageio
        video_path = ep_dir / "video.mp4"
        imageio.mimwrite(str(video_path), frames, fps=pi05_env.rig.CONTROL_HZ,
                         quality=8, macro_block_size=None)

    manifest_path = None
    try:
        scene_hash = _scene_manifest_hash(suite, cond, factory_dir)
        rollout_manifest = RolloutManifest(
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            git_dirty=git_snap["dirty"], scene_build_commit=git_snap["commit"],
            scene_manifest_hash=scene_hash,
            policy_checkpoint_hash=policy_checkpoint_hash,
            controller_config_hash=controller_hash, camera_config_hash=camera_hash,
            task_id=task["task_id"], initial_state_id=st.reset_state_id,
            rollout_seed=st.base_seed, scene_id=f"{suite['scene']}__{cond}",
            language=task["instructions"][variant],
            rubric_version=pi05_eval.RUBRIC_VERSION, horizon_s=horizon_s,
            observation_preprocessing=ObservationPreprocessing(
                mode="raster", resize_hw=(224, 224)),
            action_convention=action_convention, action_dim=action_dim,
            video_path=str(video_path) if video_path else None,
            # state_path and contact_path intentionally point at the SAME
            # per-tick file: each tick row already carries both the
            # object-pose state and the contact list, so writing two
            # copies of one file would be pure duplication, not more data.
            state_path=str(ts_path) if ts_path else None,
            contact_path=str(ts_path) if ts_path else None,
            success=bool(summary.get("success", False)),
            staged_progress=StagedProgress(**(summary.get("stages") or {
                "grasp": False, "lift": False, "hover": False, "place": False})),
            failure_label=(None if summary.get("success")
                          else pi05_eval._failure_label(summary)),
        )
        target_dir = Path(out_dir) / "manifests" / episode_id
        manifest_io.write_manifest(target_dir, rollout_manifest)
        manifest_path = str(target_dir / manifest_io.DEFAULT_MANIFEST_FILENAME)
    except Exception as e:  # noqa: BLE001 -- a manifest-write bug must not
        # eat a real episode result; log loudly and keep the episode data.
        print(f"[paired_runner] WARNING: manifest write failed for "
              f"{episode_id}: {type(e).__name__}: {e}", flush=True)

    return _make_record(cond, st, outcome, summary, error=error, ticks=ticks,
                        wall_s=wall_s,
                        video_path=str(video_path) if video_path else None,
                        timeseries_path=str(ts_path) if ts_path else None,
                        manifest_path=manifest_path)


# ---------------------------------------------------------------- matrix

def _write_summary(out_dir):
    ledger = elog.EpisodeLedger(Path(out_dir) / "episode_ledger.jsonl")
    completed = ledger.load_completed()
    by_outcome: dict = {}
    for rec in completed.values():
        by_outcome.setdefault(rec.outcome.value, []).append(rec.episode_id)
    summary = {
        "n_episode_records": len(completed),
        "coverage_by_outcome": {k: len(v) for k, v in by_outcome.items()},
        "episode_ids_by_outcome": by_outcome,
    }
    (Path(out_dir) / "matrix_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[paired_runner] {len(completed)} episode records -> "
          f"{summary['coverage_by_outcome']}", flush=True)
    return summary


def run_matrix(config: dict, out_dir, *, episodes_override=None,
              conditions=None, resume=True):
    """Execute the full policy x scene x condition x seed x episode matrix
    described by `config` (see configs/experiments/paired_smoke.yaml for
    the schema) into `out_dir`. Safe to re-invoke on the same `out_dir`:
    reset_states.json and episode_ledger.jsonl make every already-terminal
    episode a no-op resume (Task 09 step 5), never a resample.

    `conditions=None` (the default) reads `config["conditions"]`, falling
    back to running both (`CONDITIONS`) if the config doesn't declare one
    either -- an explicit `conditions=(...)` argument (the CLI's
    `--conditions`, or a caller like a test) always wins over both.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = tuple(conditions) if conditions is not None else tuple(
        config.get("conditions", CONDITIONS))

    scene_cfgs = config["scenes"]
    seeds = config.get("seeds", [0])
    episodes = episodes_override if episodes_override is not None else config.get("episodes", 1)
    task_filter = config.get("task_filter", "")
    policy_id = config["policy"]
    horizon_s = config.get("horizon_s", cc.FROZEN_CONTROL_CONTRACT.horizon_seconds)
    jitter_xy = config.get("jitter", 0.0)
    video = bool(config.get("video", False))
    variant = config.get("variant", "default")
    policy_timeout_s = config.get("policy_timeout_s", DEFAULT_POLICY_TIMEOUT_S)
    host, port = config.get("host", "localhost"), config.get("port", 8000)
    checkpoint_path = config.get("checkpoint_path", "")

    reset_states = _get_or_plan_reset_states(out_dir, scene_cfgs, seeds, episodes, task_filter)
    by_scene: dict = {}
    for s in reset_states:
        by_scene.setdefault(s.scene_id, []).append(s)

    ledger = elog.EpisodeLedger(out_dir / "episode_ledger.jsonl")
    completed = ledger.load_completed()

    git_snap = manifest_hash.git_snapshot()
    controller_hash, camera_hash, action_convention, action_dim = _frozen_config_hashes()
    policy_ckpt_hash = _policy_checkpoint_hash(policy_id, checkpoint_path)

    scene_cfg_by_id = {sc["id"]: sc for sc in scene_cfgs}
    home_pose = pi05_env.rig.PANDA_HOME

    for scene_id, states in by_scene.items():
        scene_cfg = scene_cfg_by_id.get(scene_id)
        if scene_cfg is None:
            for st in states:
                for cond in conditions:
                    _log_immediate_failure(
                        ledger, completed, cond, st, elog.Outcome.BUILD_FAILURE,
                        f"no scene config entry for scene_id={scene_id!r}")
            continue

        envs, suites, factory_dirs = {}, {}, {}
        for cond in conditions:
            try:
                env, suite, factory_dir = build_env(scene_cfg, cond, out_dir)
                envs[cond], suites[cond], factory_dirs[cond] = env, suite, factory_dir
            except elog.BuildFailureError as e:
                print(f"[paired_runner] BUILD_FAILURE scene={scene_id} "
                      f"condition={cond}: {e}", flush=True)
                for st in states:
                    _log_immediate_failure(ledger, completed, cond, st,
                                          elog.Outcome.BUILD_FAILURE, str(e))

        if not envs:
            continue

        policy = get_policy(policy_id, home_pose, host=host, port=port)
        warmed_up = False
        scene_records = []

        for cond, env in envs.items():
            suite, factory_dir = suites[cond], factory_dirs[cond]
            if hasattr(policy, "warmup") and not warmed_up:
                policy.warmup(env.reset(settle_s=0.1), "warmup")
                warmed_up = True
            tasks_by_id = {t["task_id"]: t for t in suite["tasks"]}

            for st in states:
                episode_id = elog.episode_id_for(cond, st.reset_state_id)
                if resume and episode_id in completed:
                    scene_records.append(completed[episode_id])
                    continue

                task = tasks_by_id.get(st.task_id)
                if task is None:
                    rec = _log_immediate_failure(
                        ledger, completed, cond, st, elog.Outcome.BUILD_FAILURE,
                        f"task_id {st.task_id!r} not present in the {cond} "
                        f"condition's task suite")
                    scene_records.append(rec)
                    continue

                prompt = task["instructions"][variant]
                (outcome, summary, ticks_series, frames, error, ticks,
                 wall_s) = run_episode_core(
                    env, task, policy, prompt, horizon_s, st.reset_seed,
                    jitter_xy=jitter_xy if st.ep > 0 else 0.0,
                    policy_timeout_s=policy_timeout_s, capture_video=video)

                rec = _finalize_episode(
                    out_dir, cond, st, task, suite, outcome, summary,
                    ticks_series, frames, error, ticks, wall_s,
                    git_snap=git_snap, controller_hash=controller_hash,
                    camera_hash=camera_hash, action_convention=action_convention,
                    action_dim=action_dim, factory_dir=factory_dir,
                    horizon_s=horizon_s, variant=variant,
                    policy_checkpoint_hash=policy_ckpt_hash)
                ledger.append(rec)
                completed[episode_id] = rec
                scene_records.append(rec)
                tag = rec.outcome.value if rec.outcome != elog.Outcome.SUCCESS else "OK"
                print(f"[paired_runner] {tag:16s} {episode_id} score={rec.score:.2f} "
                      f"ticks={rec.ticks}", flush=True)

        elog.check_degenerate(scene_records)

    return _write_summary(out_dir)


# ------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="matrix config yaml")
    ap.add_argument("--out", default=None, help="output dir (default: config's out_dir)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="episodes per (task, seed) -- ignored if this "
                         "--out already has a persisted reset_states.json")
    ap.add_argument("--conditions", nargs="+", default=None,
                    choices=list(ALL_CONDITIONS),
                    help="default: config's own `conditions` list, or "
                         "simany+reference if the config doesn't declare "
                         "one either -- pass simfoundry_repro explicitly "
                         "(here or in the config) to include the third "
                         "baseline condition")
    ap.add_argument("--policy", default=None, help="override config's `policy`")
    ap.add_argument("--host", default=None, help="override config's `host`")
    ap.add_argument("--port", type=int, default=None, help="override config's `port`")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-run every episode even if the ledger marks it complete")
    args = ap.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    for key, val in (("policy", args.policy), ("host", args.host), ("port", args.port)):
        if val is not None:
            config[key] = val

    out_dir = args.out or config.get("out_dir") or (
        ROOT / "outputs" / "paired_runs" / Path(args.config).stem)
    out_dir = _resolve(ROOT, out_dir)

    summary = run_matrix(config, out_dir, episodes_override=args.episodes,
                        conditions=(tuple(args.conditions) if args.conditions else None),
                        resume=not args.no_resume)
    print(f"[paired_runner] done -> {out_dir}")
    print(json.dumps(summary["coverage_by_outcome"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
