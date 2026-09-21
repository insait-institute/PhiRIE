"""Manifest-backed paired robot evaluation harness used by the SimAny paper.

This runner supersedes the original two-condition ``paired_runner`` for paper
experiments while reusing its tested environment, policy, task, reset-planning,
and trace helpers. It supports scene-construction, collision, and observation
interventions, with one persisted reset bank and fail-closed coverage accounting.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robo.envs import pi05_env
from robo.eval import episode_log as elog
from robo.eval import paired_runner as legacy
from robo.eval import pi05_eval
from robo.eval.harness_spec import HarnessSpec, TreatmentSpec, load_harness_spec
from robo.eval.harness_validation import (
    CODE_ROOT,
    E4_CONSTRUCTION_VARIANTS,
    EVIDENCE_ROOT,
    read_jsonl,
    task_definition_hash,
    validate_records,
    validate_resolved_scene_treatments,
    validate_saved_treatment_records,
)
from robo.eval.main_table import generate as generate_main_table
from robo.eval.observation_pipeline import ObservationPipeline, build_source
from robo.manifest import hash as manifest_hash
from robo.policy import control_contract as cc
from robo.policy.runtime_identity import expected_server_identity_from_config
from robo.rendering.harmonizer_client import (
    EnhancerError, IdentityEnhancer, SocketHarmonizerClient)
from robo.tasks import pi05_tasks

# ROOT remains the artifact-root compatibility alias.  Validation and imports
# execute from CODE_ROOT, while configs, sealed E3/E4 inputs, and outputs are
# resolved beneath the explicit SIMANY_EVIDENCE_ROOT selected by the launcher.
ROOT = EVIDENCE_ROOT
LEDGER_NAME = "harness_ledger.jsonl"


def _resolve(path: str | Path, base: Path | None = None) -> Path:
    if base is None:
        base = ROOT
    value = Path(path)
    return value if value.is_absolute() else base / value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _completed(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["treatment_id"], row["reset_state_id"]): row
            for row in read_jsonl(path)}


def _task_family(task: dict) -> str:
    if task.get("task_family"):
        return str(task["task_family"])
    if task.get("receptacle") or task.get("receptacle_body"):
        return "object_to_receptacle"
    if task.get("region") is not None:
        return "object_to_region"
    instruction = " ".join(str(v) for v in (task.get("instructions") or {}).values()).lower()
    return ("object_to_receptacle" if any(
        token in instruction for token in ("tray", "bowl", "basket", "plate", "box"))
        else "object_to_region")


def _scene_config_for_treatment(scene_cfg: dict, treatment: TreatmentSpec,
                                out_dir: Path) -> dict:
    """Apply implementation path overrides without relabeling a condition."""
    cfg = copy.deepcopy(scene_cfg)
    options = treatment.options
    scene_xml_override = cfg.get("scene_xml")

    construction_override: dict[str, Any] = {}
    if treatment.scene in E4_CONSTRUCTION_VARIANTS:
        variants = scene_cfg.get("construction_variants")
        if not isinstance(variants, dict):
            raise elog.BuildFailureError(
                f"scene {scene_cfg.get('id', '?')!r} requires "
                "construction_variants for E4 A0/A4 treatments")
        value = variants.get(treatment.scene)
        if not isinstance(value, dict):
            raise elog.BuildFailureError(
                f"scene {scene_cfg.get('id', '?')!r} has no construction "
                f"variant {treatment.scene!r}")
        construction_override = copy.deepcopy(value)
        missing = [key for key in ("factory_dir", "tasks_json", "scene_xml")
                   if not construction_override.get(key)]
        if missing:
            raise elog.BuildFailureError(
                f"scene {scene_cfg.get('id', '?')!r} construction variant "
                f"{treatment.scene!r} is missing {missing}")
        cfg.update(construction_override)
        scene_xml_override = construction_override["scene_xml"]

    # A variant may carry collision-specific exports of its own.  The
    # variant-local mapping wins over the historical scene-global mapping.
    collision_overrides: dict[str, Any] = {}
    global_collisions = scene_cfg.get("collision_variants", {})
    if isinstance(global_collisions, dict):
        value = global_collisions.get(treatment.collision, {})
        if isinstance(value, dict):
            collision_overrides.update(value)
    local_collisions = construction_override.get("collision_variants", {})
    if isinstance(local_collisions, dict):
        value = local_collisions.get(treatment.collision, {})
        if isinstance(value, dict):
            collision_overrides.update(value)
    cfg.update(collision_overrides)
    if collision_overrides.get("scene_xml"):
        scene_xml_override = collision_overrides["scene_xml"]
    elif collision_overrides.get("tasks_json"):
        # A replacement suite owns its embedded XML unless the same override
        # explicitly replaces scene_xml too.
        scene_xml_override = None

    # Retain the old treatment-global escape hatch.  New E4 configs should
    # use construction_variants so every room resolves independently.
    for field in ("factory_dir", "tasks_json", "scene_xml"):
        if options.get(field):
            cfg[field] = options[field]
    if options.get("scene_xml"):
        scene_xml_override = options["scene_xml"]
    elif options.get("tasks_json"):
        scene_xml_override = None
    if treatment.collision == "private_shims" and not (
        options.get("tasks_json") or collision_overrides or options.get("scene_xml")):
        raise elog.BuildFailureError(
            "private_shims requires an alternate tasks/XML path; refusing to "
            "reuse full-room collision under a different label")
    if scene_xml_override:
        source_path = _resolve(cfg["tasks_json"])
        source_suite = json.loads(source_path.read_text())
        resolved_xml = str(_resolve(scene_xml_override))
        if source_suite.get("scene_xml") == resolved_xml:
            # The paired E4 task freezer already sealed this exact
            # variant-specific suite.  Preserve its authenticated path rather
            # than replacing it with an unauthenticated byte copy.
            cfg["tasks_json"] = str(source_path)
            cfg["scene_xml"] = resolved_xml
        else:
            source_suite["scene_xml"] = resolved_xml
            suite_path = out_dir / "resolved_suites" / (
                f"{scene_cfg['id']}__{treatment.id}.json")
            _atomic_json(suite_path, source_suite)
            cfg["tasks_json"] = str(suite_path)
            cfg["scene_xml"] = source_suite["scene_xml"]
    else:
        cfg.pop("scene_xml", None)
    # ``legacy.build_env`` still resolves relative factory/task paths against
    # the checkout containing ``paired_runner.py``.  E4 deliberately executes
    # from a clean code worktree while reading sealed artifacts from ROOT
    # (SIMANY_EVIDENCE_ROOT), so hand the legacy builder canonical absolute
    # artifact paths.  The paired preflight below has already retained the
    # logical labels and will validate that these paths stay inside ROOT.
    for field in ("factory_dir", "tasks_json", "scene_xml"):
        if cfg.get(field):
            cfg[field] = str(_resolve(cfg[field]))
    return cfg


def _planning_suite(scene_cfg: dict) -> dict[str, Any]:
    path = _resolve(scene_cfg["tasks_json"])
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"task suite is not a mapping: {path}")
    return value


def _scene_configs_for_legacy_planner(
    scene_cfgs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Give the legacy reset planner evidence-root-absolute task suites.

    ``paired_runner.plan_reset_states`` predates split code/evidence roots and
    resolves relative paths against its own code checkout.  Copying only the
    planning path keeps the frozen user config byte-stable while preventing a
    clean E4 worktree from accidentally looking for sealed suites beneath its
    empty ``outputs/`` directory.
    """
    resolved = copy.deepcopy(scene_cfgs)
    for scene_cfg in resolved:
        tasks_json = scene_cfg.get("tasks_json")
        if tasks_json:
            scene_cfg["tasks_json"] = str(_resolve(tasks_json))
    return resolved


def _preflight_resolved_scenes(config: dict, spec: HarnessSpec,
                               out_dir: Path) -> tuple[
        dict[str, dict[str, dict]], dict[str, dict],
        dict[str, dict[str, dict]]]:
    """Resolve and validate the complete matrix before env/policy creation."""
    resolved_by_scene: dict[str, dict[str, dict]] = {}
    scene_cfg_by_logical_id: dict[str, dict] = {}
    planning_tasks_by_scene: dict[str, dict[str, dict]] = {}
    violations: list[str] = []

    for scene_cfg in config.get("scenes", ()):
        config_scene_id = str(scene_cfg.get("id", "?"))
        try:
            planning_suite = _planning_suite(scene_cfg)
            logical_scene_id = str(planning_suite["scene"])
        except (KeyError, OSError, TypeError, ValueError) as exc:
            violations.append(
                f"scene {config_scene_id!r} cannot load planning suite: "
                f"{type(exc).__name__}: {exc}")
            continue
        if logical_scene_id in scene_cfg_by_logical_id:
            violations.append(
                f"duplicate logical scene id {logical_scene_id!r} in scene configs")
            continue
        if any(treatment.scene in E4_CONSTRUCTION_VARIANTS
               for treatment in spec.treatments.values()
               ) and logical_scene_id != config_scene_id:
            violations.append(
                f"scene {config_scene_id!r} planning suite uses logical scene "
                f"{logical_scene_id!r}; E4 construction variants require them to match")

        current: dict[str, dict] = {}
        for treatment in spec.treatments.values():
            try:
                current[treatment.id] = _scene_config_for_treatment(
                    scene_cfg, treatment, out_dir)
            except Exception as exc:  # preflight aggregates every config defect
                violations.append(
                    f"scene {config_scene_id!r} treatment {treatment.id!r} "
                    f"cannot resolve: {type(exc).__name__}: {exc}")
        report = validate_resolved_scene_treatments(
            scene_cfg, current, spec, root=ROOT)
        violations.extend(report["violations"])
        for treatment_id, identities in report.get(
            "treatment_artifacts", {}
        ).items():
            if treatment_id in current:
                current[treatment_id]["_e4_input_artifacts"] = identities

        tasks: dict[str, dict] = {}
        task_values = planning_suite.get("tasks")
        if not isinstance(task_values, list):
            task_values = []
        for index, task in enumerate(task_values):
            if not isinstance(task, dict) or not task.get("task_id"):
                violations.append(
                    f"scene {config_scene_id!r} planning task[{index}] has no task_id")
                continue
            task_id = str(task["task_id"])
            if task_id in tasks:
                violations.append(
                    f"scene {config_scene_id!r} has duplicate logical task_id "
                    f"{task_id!r}")
            tasks[task_id] = task
        resolved_by_scene[logical_scene_id] = current
        scene_cfg_by_logical_id[logical_scene_id] = scene_cfg
        planning_tasks_by_scene[logical_scene_id] = tasks

    if violations:
        raise ValueError("harness scene preflight failed: " + " | ".join(violations))
    if not resolved_by_scene:
        raise ValueError("harness scene preflight failed: no resolvable scenes")
    return resolved_by_scene, scene_cfg_by_logical_id, planning_tasks_by_scene


def _planned_jitter(*, reset_seed: int, jitter_xy: float,
                    target: str | None) -> dict[str, Any]:
    """Replay the persisted reset seed into one explicit, arm-independent draw."""
    if isinstance(reset_seed, (bool, np.bool_)):
        raise ValueError("reset seed must be an integer")
    reset_seed = int(reset_seed)
    if not 0 <= reset_seed <= np.iinfo(np.uint32).max:
        raise ValueError("reset seed must be a uint32 value")
    jitter_xy = float(jitter_xy)
    if not np.isfinite(jitter_xy) or jitter_xy < 0.0:
        raise ValueError("jitter must be a finite non-negative value")
    draw = np.random.RandomState(reset_seed).random_sample(2)
    offset = (2.0 * draw - 1.0) * jitter_xy
    body = str(target) if target is not None and jitter_xy > 0.0 else None
    return {
        "algorithm": "numpy.random.RandomState.random_sample_then_affine",
        "reset_seed": reset_seed,
        "body": body,
        "max_abs_xy_m": jitter_xy,
        "uniform_draw_0_1": [float(value) for value in draw],
        "offset_xy_m": [float(value) for value in offset],
        "applied": bool(body is not None and jitter_xy > 0.0),
    }


def _reset_definition(state, *, task: dict | None,
                      jitter_xy: float) -> dict[str, Any]:
    target = task.get("target") if isinstance(task, dict) else None
    return {
        "reset_state_id": state.reset_state_id,
        "scene_id": state.scene_id,
        "task_id": state.task_id,
        "episode_index": state.ep,
        "base_seed": state.base_seed,
        "reset_seed": state.reset_seed,
        "jitter": _planned_jitter(
            reset_seed=state.reset_seed, jitter_xy=jitter_xy, target=target),
    }


def _validate_planned_resets(states, planning_tasks_by_scene: dict[str, dict[str, dict]],
                             strict_scene_ids: set[str]) -> None:
    violations: list[str] = []
    seen: set[str] = set()
    for state in states:
        if state.reset_state_id in seen:
            violations.append(f"duplicate reset_state_id {state.reset_state_id!r}")
        seen.add(state.reset_state_id)
        tasks = planning_tasks_by_scene.get(state.scene_id)
        if tasks is None:
            violations.append(
                f"reset {state.reset_state_id!r} references unknown logical scene "
                f"{state.scene_id!r}")
            continue
        if state.task_id not in tasks:
            violations.append(
                f"reset {state.reset_state_id!r} references unknown logical task "
                f"{state.task_id!r}")
        if state.scene_id not in strict_scene_ids:
            continue
        expected_id = (
            f"{state.task_id}__seed{state.base_seed}__ep{state.ep}")
        if state.reset_state_id != expected_id:
            violations.append(
                f"reset id mismatch for {state.task_id!r}: "
                f"expected {expected_id!r}, got {state.reset_state_id!r}")
        expected_seed = elog.derive_reset_seed(
            state.base_seed, state.task_id, state.ep)
        if state.reset_seed != expected_seed:
            violations.append(
                f"reset seed mismatch for {state.reset_state_id!r}: "
                f"expected {expected_seed}, got {state.reset_seed}")
    if violations:
        raise ValueError("harness reset preflight failed: " + " | ".join(violations))


def _enhancer_for(treatment: TreatmentSpec):
    if treatment.observation != "harmonizer_c":
        return None
    options = treatment.options.get("enhancer", {})
    backend = options.get("backend", "socket")
    if backend == "socket":
        socket_path = options.get("socket")
        if not socket_path:
            raise elog.BuildFailureError("harmonizer_c requires options.enhancer.socket")
        return SocketHarmonizerClient(
            str(socket_path), timeout_s=float(options.get("timeout_s", 30.0)))
    if backend == "identity":
        if not bool(options.get("allow_smoke_only", False)):
            raise elog.BuildFailureError(
                "identity enhancer is permitted only with allow_smoke_only=true")
        return IdentityEnhancer()
    raise elog.BuildFailureError(f"unknown enhancer backend {backend!r}")


def _classify_exception(exc: Exception) -> str:
    if isinstance(exc, EnhancerError):
        return "enhancer_failure"
    if isinstance(exc, elog.PolicyTimeoutError):
        return "policy_timeout"
    if isinstance(exc, elog.SafetyTerminationError):
        return "safety_termination"
    if isinstance(exc, elog.BuildFailureError):
        return "build_failure"
    return "environment_crash"


def _episode_id(treatment_id: str, reset_state_id: str) -> str:
    return f"{treatment_id}__{reset_state_id}"


def _run_episode(env, task, policy, pipeline: ObservationPipeline, *,
                 episode_id: str, prompt: str, horizon_s: float,
                 reset_seed: int, jitter_xy: float, policy_timeout_s: float,
                 capture_video: bool) -> tuple[
                     dict, list[dict], list[np.ndarray] | None, dict | None]:
    jitter = _planned_jitter(
        reset_seed=reset_seed, jitter_xy=jitter_xy, target=task["target"])
    ticks: list[dict] = []
    frames = [] if capture_video else None
    reset_provenance = None
    start = time.perf_counter()
    try:
        # Bind/clear sampling even when this episode fails during reset.
        if callable(getattr(policy, "begin_episode", None)):
            policy.begin_episode(reset_seed)
        env.reset(
            jitter_body=jitter["body"],
            jitter_xy=jitter["max_abs_xy_m"],
            reset_seed=jitter["reset_seed"],
            jitter_uniform_draw=jitter["uniform_draw_0_1"],
            jitter_offset_xy=jitter["offset_xy_m"],
        )
        reset_provenance = copy.deepcopy(
            getattr(env, "last_reset_provenance", None))
        if not isinstance(reset_provenance, dict):
            raise RuntimeError("environment did not publish reset provenance")
        if reset_provenance.get("jitter") != jitter:
            raise RuntimeError(
                "environment reset provenance differs from the planned jitter")
        if not callable(getattr(policy, "begin_episode", None)):
            policy.reset()
        pipeline.reset(episode_id)
        observation_result = pipeline.get_obs()
        obs = observation_result.observation
        scorer = pi05_tasks.TaskScorer(env, task)
        n_ticks = max(int(horizon_s * pi05_env.rig.CONTROL_HZ), 1)
        policy_latencies, observation_latencies = [], [observation_result.latency_ms]
        per_camera_meta = []
        if observation_result.per_camera:
            per_camera_meta.append(observation_result.per_camera)
        for tick_index in range(n_ticks):
            policy_start = time.perf_counter()
            action = policy(obs, prompt)
            policy_ms = (time.perf_counter() - policy_start) * 1000.0
            if policy_ms > 1000.0 * policy_timeout_s:
                raise elog.PolicyTimeoutError(
                    f"policy call took {policy_ms / 1000.0:.2f}s at tick {tick_index}")
            action = np.asarray(action, dtype=float)
            if not np.isfinite(action).all():
                raise elog.SafetyTerminationError("policy returned non-finite action")
            env.apply_action(action)
            observation_result = pipeline.get_obs()
            obs = observation_result.observation
            if not np.isfinite(env.data.qpos).all():
                raise elog.SafetyTerminationError("simulator qpos became non-finite")
            scorer.update()
            row = legacy._tick_row(env, action, tick_index, scorer)
            row["policy_latency_ms"] = policy_ms
            row["observation_latency_ms"] = observation_result.latency_ms
            row["observation_metadata"] = observation_result.per_camera
            ticks.append(row)
            policy_latencies.append(policy_ms)
            observation_latencies.append(observation_result.latency_ms)
            if observation_result.per_camera:
                per_camera_meta.append(observation_result.per_camera)
            if capture_video:
                frames.append(legacy._video_frame(obs))
            if scorer.success:
                break
        summary = scorer.summary()
        return ({
            "outcome": "success" if summary["success"] else "task_failure",
            "success": bool(summary["success"]),
            "score": float(summary.get("score", 0.0)),
            "stages": summary.get("stages", {}), "error": None,
            "wall_s": time.perf_counter() - start, "ticks": len(ticks),
            "policy_latency_ms": policy_latencies,
            "observation_latency_ms": observation_latencies,
            "per_camera_metadata": per_camera_meta}, ticks, frames,
            reset_provenance)
    except Exception as exc:
        return ({
            "outcome": _classify_exception(exc), "success": False, "score": 0.0,
            "stages": {"grasp": False, "lift": False,
                       "hover": False, "place": False},
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": time.perf_counter() - start, "ticks": len(ticks),
            "policy_latency_ms": [r.get("policy_latency_ms", 0.0) for r in ticks],
            "observation_latency_ms": [r.get("observation_latency_ms", 0.0)
                                       for r in ticks],
            "per_camera_metadata": [r.get("observation_metadata", {}) for r in ticks]},
            ticks, frames, reset_provenance)


def _write_episode_artifacts(out_dir: Path, record: dict, ticks: list[dict],
                             frames, manifest: dict) -> dict:
    episode_dir = out_dir / "episodes" / record["episode_id"]
    episode_dir.mkdir(parents=True, exist_ok=True)
    if ticks:
        trace_path = episode_dir / "timeseries.json.gz"
        elog.write_timeseries(trace_path, ticks)
        record["trace_path"] = str(trace_path)
    if frames:
        import imageio.v2 as imageio
        video_path = episode_dir / "video.mp4"
        imageio.mimwrite(str(video_path), frames, fps=pi05_env.rig.CONTROL_HZ,
                         quality=8, macro_block_size=None)
        record["video_path"] = str(video_path)
    manifest["artifacts"] = {
        "trace_path": record.get("trace_path"),
        "video_path": record.get("video_path")}
    manifest_path = episode_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    record["manifest_path"] = str(manifest_path)
    return record


def _failure_record(treatment: TreatmentSpec, state, error: str,
                    contract: dict | None = None,
                    task: dict | None = None) -> dict:
    record = {
        "episode_id": _episode_id(treatment.id, state.reset_state_id),
        "treatment_id": treatment.id, "scene_id": state.scene_id,
        "treatment": treatment.to_dict(),
        "task_id": state.task_id,
        "task_family": _task_family(task) if task is not None else "unknown",
        "reset_state_id": state.reset_state_id, "base_seed": state.base_seed,
        "reset_seed": state.reset_seed, "outcome": "build_failure",
        "success": False, "score": 0.0,
        "stages": {"grasp": False, "lift": False,
                   "hover": False, "place": False},
        "ticks": 0, "wall_s": 0.0, "error": error}
    if contract is not None:
        # A typed terminal failure still carries the exact frozen contract so
        # it can participate in pair validation rather than disappearing.
        record["contract"] = contract
    return record


def _episode_jitter(config, state):
    return float(config.get("jitter", 0.0)) if (
        state.ep > 0 or config.get("jitter_first_episode", False)) else 0.0


def _runtime_contract(*, config: dict, state, controller_hash: str,
                      camera_hash: str, action_convention: str,
                      action_dim: int, policy_hash: str | None,
                      task_instruction: str,
                      server_identity: dict | None = None,
                      policy_not_invoked: bool = False,
                      suite: dict | None = None,
                      task: dict | None = None,
                      construction_artifacts: dict | None = None) -> dict:
    effective_jitter = _episode_jitter(config, state)
    reset_definition = _reset_definition(
        state, task=task, jitter_xy=effective_jitter)
    paired_fields = {
        "task_definition_hash": (
            task_definition_hash(task) if task is not None else None),
        "reset_definition": reset_definition,
        "reset_definition_hash": manifest_hash.canonical_hash(reset_definition),
    }
    declared = config.get("contract")
    if isinstance(declared, dict):
        contract = copy.deepcopy(declared)
        policy = contract.setdefault("policy", {})
        policy_id = policy.get("id", config.get("policy"))
        policy["id"] = policy_id
        declared_policy_hash = policy.get("checkpoint_hash")
        if policy_not_invoked:
            if server_identity is not None or policy.get("server_identity") is not None:
                raise ValueError("uninvoked policy must not claim a verified server identity")
            contract["policy_execution"] = "not_invoked_prebuild"
        if policy.get("kind", "real") == "real":
            if declared_policy_hash != policy_hash:
                raise ValueError(
                    "declared policy checkpoint hash differs from current "
                    "checkpoint fingerprint")
            if not policy_not_invoked and contract.get("runtime_dependencies") and server_identity is None:
                raise ValueError(
                    "E4 real-policy contract lacks verified server identity")
            if server_identity is not None:
                policy["server_identity"] = copy.deepcopy(server_identity)
        contract.update({
            "policy_id": policy_id,
            "policy_checkpoint_hash": declared_policy_hash,
            "runtime_policy_checkpoint_fingerprint": policy_hash,
            "controller_config_hash": controller_hash,
            "camera_config_hash": camera_hash,
            "action_convention": action_convention,
            "action_dim": action_dim,
            "control_hz": pi05_env.rig.CONTROL_HZ,
            "control_rate_hz": pi05_env.rig.CONTROL_HZ,
            "horizon_s": float(config.get(
                "horizon_s", contract.get(
                    "horizon_s", cc.FROZEN_CONTROL_CONTRACT.horizon_seconds))),
            "task_instruction": task_instruction,
            "task_id": state.task_id,
            "reset_state_id": state.reset_state_id,
            "rollout_seed": state.base_seed,
            "reset_seed": state.reset_seed,
            "language_variant": config.get("variant", "default"),
            **paired_fields,
        })
        if construction_artifacts is not None:
            contract["construction_artifacts"] = copy.deepcopy(
                construction_artifacts)
        return contract
    suite = suite or {}
    return {
        "policy_id": config["policy"], "policy_checkpoint_hash": policy_hash,
        "controller_config_hash": controller_hash,
        "camera_config_hash": camera_hash,
        "robot": suite.get("robot"),
        "cameras": {"config_hash": camera_hash, "exterior": suite.get("ext_cam")},
        "controller": {"config_hash": controller_hash},
        "action_convention": action_convention, "action_dim": action_dim,
        "control_hz": pi05_env.rig.CONTROL_HZ,
        "control_rate_hz": pi05_env.rig.CONTROL_HZ,
        "horizon_s": float(config.get(
            "horizon_s", cc.FROZEN_CONTROL_CONTRACT.horizon_seconds)),
        "rubric_version": pi05_eval.RUBRIC_VERSION,
        "rubric": {"version": pi05_eval.RUBRIC_VERSION},
        "task_instruction": task_instruction,
        "task_id": state.task_id, "reset_state_id": state.reset_state_id,
        "rollout_seed": state.base_seed, "reset_seed": state.reset_seed,
        "language_variant": config.get("variant", "default"),
        **paired_fields,
        **({"construction_artifacts": copy.deepcopy(construction_artifacts)}
           if construction_artifacts is not None else {})}


def _planned_instruction(scene_cfg: dict | None, state, config: dict) -> str:
    fallback = str(config.get("contract", {}).get("task_instruction", "unknown"))
    if scene_cfg is None or not scene_cfg.get("tasks_json"):
        return fallback
    try:
        suite = json.loads(_resolve(scene_cfg["tasks_json"]).read_text())
        task = next(item for item in suite.get("tasks", ())
                    if item.get("task_id") == state.task_id)
        return str(task["instructions"][config.get("variant", "default")])
    except (KeyError, OSError, StopIteration, TypeError, ValueError):
        return fallback


def _make_manifest(*, treatment: TreatmentSpec, state, suite: dict,
                   factory_dir: Path, config: dict, result: dict,
                   controller_hash: str, camera_hash: str,
                   action_convention: str, action_dim: int,
                   policy_hash: str | None, git_snapshot: dict,
                   server_identity: dict | None,
                   task_instruction: str, task: dict,
                   construction_artifacts: dict | None = None,
                   reset_provenance: dict | None = None) -> dict:
    contract = _runtime_contract(
        config=config, state=state, controller_hash=controller_hash,
        camera_hash=camera_hash, action_convention=action_convention,
        action_dim=action_dim, policy_hash=policy_hash,
        server_identity=server_identity,
        task_instruction=task_instruction, suite=suite, task=task,
        construction_artifacts=construction_artifacts)
    sampling_evidence = {}
    if "policy_sampling_receipts" in result:
        sampling_evidence = {"policy_sampling_receipts": copy.deepcopy(result["policy_sampling_receipts"]),
                             "policy_sampling_ticks": result["ticks"]}
    scene_payload = {
        "scene": suite["scene"], "scene_xml": suite.get("scene_xml"),
        "robot": suite.get("robot"), "table": suite.get("table"),
        "ext_cam": suite.get("ext_cam"),
        "exclude_objects": sorted(suite.get("exclude_objects", ())),
        "treatment": treatment.to_dict()}
    return {
        "schema_version": 1, "manifest_kind": "harness_rollout",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_snapshot, "treatment_id": treatment.id,
        "treatment": treatment.to_dict(), "scene_id": state.scene_id,
        "scene_manifest_hash": manifest_hash.canonical_hash(scene_payload),
        "factory_dir": str(factory_dir), "contract": contract,
        "construction_artifacts": copy.deepcopy(construction_artifacts),
        "reset_provenance": copy.deepcopy(reset_provenance),
        "outcome": result["outcome"], **sampling_evidence}


def _validate_record_inputs_against_current(
    records: list[dict],
    resolved_by_scene: dict[str, dict[str, dict]],
    planning_tasks_by_scene: dict[str, dict[str, dict]],
) -> list[str]:
    """Bind resumed and newly written rows to current task/asset bytes."""
    violations: list[str] = []
    for record in records:
        scene_id = str(record.get("scene_id", ""))
        treatment_id = str(record.get("treatment_id", ""))
        task_id = str(record.get("task_id", ""))
        expected_cfg = resolved_by_scene.get(scene_id, {}).get(treatment_id)
        expected_task = planning_tasks_by_scene.get(scene_id, {}).get(task_id)
        if expected_cfg is None or expected_task is None:
            violations.append(
                f"record {record.get('episode_id')!r} no longer resolves to its "
                "current scene/treatment/task contract")
            continue
        manifest: dict[str, Any]
        manifest_path = record.get("manifest_path")
        if manifest_path:
            path = _resolve(manifest_path)
            try:
                manifest = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                violations.append(
                    f"record {record.get('episode_id')!r} manifest cannot be "
                    f"reloaded for current-input validation: {exc}")
                continue
        elif isinstance(record.get("contract"), dict):
            manifest = {"contract": record["contract"]}
        else:
            violations.append(
                f"record {record.get('episode_id')!r} has no input-bound contract")
            continue
        contract = manifest.get("contract")
        if not isinstance(contract, dict):
            violations.append(
                f"record {record.get('episode_id')!r} has no manifest contract")
            continue
        expected_task_hash = task_definition_hash(expected_task)
        if contract.get("task_definition_hash") != expected_task_hash:
            violations.append(
                f"record {record.get('episode_id')!r} task definition differs "
                "from the current frozen planning suite")
        expected_artifacts = expected_cfg.get("_e4_input_artifacts")
        if expected_artifacts is not None:
            if contract.get("construction_artifacts") != expected_artifacts:
                violations.append(
                    f"record {record.get('episode_id')!r} construction artifact "
                    "identity differs from current sealed inputs")
            if manifest_path and manifest.get("construction_artifacts") != expected_artifacts:
                violations.append(
                    f"record {record.get('episode_id')!r} top-level construction "
                    "artifact identity differs from current sealed inputs")
    return violations


def full_pilot_validation_spec(spec, states):
    """Project only validator coverage after the complete frozen bank is checked.

    The persisted configuration, reset definitions and episode manifests retain
    the original eighty-reset contract. No projected config is executable.
    """
    config = spec.raw
    full_ids = {state.reset_state_id for state in states}
    declared = config.get("contract", {}).get("reset_ids", [])
    pilot = config.get("pilot_reset_ids", [])
    if (config.get("cpu_reset_eligibility", {}).get("kind") != "automatic_full"
            or config.get("policy") != "pi05_droid_jointpos"
            or len(states) != 80 or len(full_ids) != 80
            or len(declared) != 80 or set(declared) != full_ids
            or len(pilot) != 20 or len(set(pilot)) != 20 or not set(pilot) <= full_ids):
        raise ValueError("fixed pilot reset IDs differ from full frozen bank")
    projected = copy.deepcopy(spec)
    projected.raw["contract"]["reset_ids"] = list(pilot)
    return projected


def run_matrix(config: dict, out_dir: str | Path, *, resume: bool = True,
               episodes_override: int | None = None, execution_stage: str | None = None) -> dict:
    spec = load_harness_spec(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = out_dir / "resolved_harness_config.json"
    if resolved_config_path.exists():
        try:
            saved_config = json.loads(resolved_config_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"cannot validate existing resolved harness config: {exc}"
            ) from exc
        if manifest_hash.canonical_hash(saved_config) != manifest_hash.canonical_hash(config):
            raise ValueError(
                "refusing resume because resolved harness config differs from "
                "the existing run"
            )
    else:
        _atomic_json(resolved_config_path, config)
    if not resume and (out_dir / LEDGER_NAME).exists():
        raise FileExistsError(
            "--no-resume requires a fresh output directory; an existing ledger "
            "would otherwise be duplicated"
        )
    scene_cfgs = config["scenes"]
    resolved_by_scene, scene_cfg_by_id, planning_tasks_by_scene = \
        _preflight_resolved_scenes(config, spec, out_dir)
    states = legacy._get_or_plan_reset_states(
        out_dir, _scene_configs_for_legacy_planner(scene_cfgs),
        config.get("seeds", [0]),
        episodes_override if episodes_override is not None else config.get("episodes", 1),
        config.get("task_filter", ""))
    strict_scene_ids = {
        scene_id for scene_id in resolved_by_scene
        if any(treatment.scene in E4_CONSTRUCTION_VARIANTS
               for treatment in spec.treatments.values())
    }
    _validate_planned_resets(states, planning_tasks_by_scene, strict_scene_ids)
    from robo.eval.e4_reset_eligibility import load_reset_eligibility
    reset_eligibility = load_reset_eligibility(config, spec, states, root=ROOT)
    planned_reset_ids = {state.reset_state_id for state in states}
    declared_reset_ids = config.get("contract", {}).get("reset_ids")
    if isinstance(declared_reset_ids, list) and {
        str(value) for value in declared_reset_ids
    } != planned_reset_ids:
        raise ValueError(
            "planned reset ids differ from contract.reset_ids before execution")
    full_policy = (isinstance(config.get("cpu_reset_eligibility"), dict)
                   and config["cpu_reset_eligibility"].get("kind") == "automatic_full")
    execution_states = states
    if execution_stage is not None:
        if execution_stage != "pilot" or not full_policy or config.get("policy") != "pi05_droid_jointpos":
            raise ValueError("execution subset requires the preregistered full real-policy pilot")
        full_pilot_validation_spec(spec, states)
        pilot_ids = config["pilot_reset_ids"]
        execution_states = [state for state in states if state.reset_state_id in set(pilot_ids)]
    elif full_policy and config.get("policy") == "pi05_droid_jointpos":
        from run.icra2027.e4_compact_policy import validate_real_pilot_before_full
        validate_real_pilot_before_full(out_dir, manifest_hash.git_snapshot(CODE_ROOT)["commit"])
    execution_reset_ids = {state.reset_state_id for state in execution_states}
    by_scene: dict[str, list] = {}
    for state in execution_states:
        by_scene.setdefault(state.scene_id, []).append(state)
    ledger_path = out_dir / LEDGER_NAME
    completed = _completed(ledger_path) if resume else {}
    git_snapshot = manifest_hash.git_snapshot(CODE_ROOT)
    controller_hash, camera_hash, action_convention, action_dim = \
        legacy._frozen_config_hashes()
    policy_id = config.get(
        "policy", config.get("contract", {}).get("policy", {}).get("id"))
    if not policy_id:
        raise ValueError("harness config is missing policy/contract.policy.id")
    policy_hash = legacy._policy_checkpoint_hash(
        policy_id,
        config.get("checkpoint_path", config.get("contract", {}).get(
            "policy", {}).get("checkpoint_path", "")))
    policy = None
    server_identity = None
    contract_policy = config.get("contract", {}).get("policy", {})
    has_e4_construction = any(
        treatment.scene in E4_CONSTRUCTION_VARIANTS
        for treatment in spec.treatments.values())
    bound_e4_real_policy = (
        has_e4_construction
        and isinstance(contract_policy, dict)
        and contract_policy.get("kind", "real") == "real")
    expected_server_identity = None
    menagerie_root = None
    automatic_prebuild = (
        isinstance(config.get("cpu_reset_eligibility"), dict)
        and config["cpu_reset_eligibility"].get("kind") in {"automatic_compact", "automatic_full"})
    if bound_e4_real_policy:
        # This happens outside the per-treatment build try/except.  Registry,
        # checkpoint, control-contract, OpenPI-worktree, or server-identity
        # failures therefore abort the run instead of becoming valid-looking
        # build_failure coverage rows.
        expected_server_identity = expected_server_identity_from_config(config)
        dependencies = config["contract"]["runtime_dependencies"]
        menagerie_root = dependencies["mujoco_menagerie"]["root"]
        if not automatic_prebuild:
            policy = legacy.get_policy(
                policy_id, pi05_env.rig.PANDA_HOME,
                host=config.get("host", "localhost"),
                port=int(config.get("port", 8000)),
                open_loop_horizon=int(config.get("open_loop_horizon", 15)),
                expected_server_identity=expected_server_identity)
            server_identity = policy.verified_server_identity

    if automatic_prebuild:
        # Authenticate declared checkpoint/runtime files above, but persist all
        # measured pre-policy rejections before connecting to a policy service.
        # The eligibility loader has already replayed every source CPU/camera
        # cell. These records claim neither invocation nor a simulator reset.
        for state in execution_states:
            for treatment in spec.treatments.values():
                key = (treatment.id, state.reset_state_id)
                evidence = reset_eligibility[key]
                if key in completed or evidence["passed"]:
                    continue
                scene_cfg = scene_cfg_by_id[state.scene_id]
                task = planning_tasks_by_scene[state.scene_id][state.task_id]
                resolved = resolved_by_scene[state.scene_id][treatment.id]
                contract = _runtime_contract(
                    config=config, state=state, controller_hash=controller_hash,
                    camera_hash=camera_hash, action_convention=action_convention,
                    action_dim=action_dim, policy_hash=policy_hash,
                    policy_not_invoked=True,
                    task_instruction=_planned_instruction(scene_cfg, state, config),
                    task=task, construction_artifacts=resolved.get("_e4_input_artifacts"))
                record = _failure_record(treatment, state,
                    evidence["failure_type"] + ": " + ", ".join(evidence["failed_checks"]),
                    contract, task)
                record["construction_validity_evidence"] = evidence
                _append_jsonl(ledger_path, record)
                completed[key] = record

    for scene_id, scene_states in by_scene.items():
        scene_cfg = scene_cfg_by_id.get(scene_id)
        planning_tasks = planning_tasks_by_scene.get(scene_id, {})
        if scene_cfg is None:
            for treatment in spec.treatments.values():
                for state in scene_states:
                    key = (treatment.id, state.reset_state_id)
                    if key in completed:
                        continue
                    contract = _runtime_contract(
                        config=config, state=state, controller_hash=controller_hash,
                        camera_hash=camera_hash, action_convention=action_convention,
                        action_dim=action_dim, policy_hash=policy_hash,
                        server_identity=server_identity,
                        task_instruction=_planned_instruction(None, state, config),
                        task=None)
                    record = _failure_record(
                        treatment, state, "missing scene config", contract)
                    _append_jsonl(ledger_path, record)
                    completed[key] = record
            continue
        for treatment in spec.treatments.values():
            pending = [state for state in scene_states
                       if (treatment.id, state.reset_state_id) not in completed]
            if not pending:
                continue
            resolved_cfg = copy.deepcopy(resolved_by_scene[scene_id][treatment.id])
            if menagerie_root is not None:
                resolved_cfg["menagerie_root"] = menagerie_root
            construction_artifacts = resolved_cfg.get("_e4_input_artifacts")
            eligible_pending = []
            for state in pending:
                evidence = reset_eligibility.get((treatment.id, state.reset_state_id))
                if evidence is None or evidence["passed"]:
                    eligible_pending.append(state)
                    continue
                planned_task = planning_tasks[state.task_id]
                contract = _runtime_contract(
                    config=config, state=state, controller_hash=controller_hash,
                    camera_hash=camera_hash, action_convention=action_convention,
                    action_dim=action_dim, policy_hash=policy_hash,
                    server_identity=server_identity,
                    task_instruction=_planned_instruction(scene_cfg, state, config),
                    task=planned_task, construction_artifacts=construction_artifacts)
                record = _failure_record(
                    treatment, state, "cpu_reset_validity_failure: " +
                    ", ".join(evidence["failed_checks"]), contract, planned_task)
                record["construction_validity_evidence"] = evidence
                _append_jsonl(ledger_path, record)
                completed[(treatment.id, state.reset_state_id)] = record
            pending = eligible_pending
            if not pending:
                continue  # Do not instantiate a rejected construction arm.
            if bound_e4_real_policy and policy is None:
                # A service/identity error is still fatal and never relabeled
                # as construction failure. Previously persisted certified
                # rejections remain valid, resumable denominator records.
                policy = legacy.get_policy(
                    policy_id, pi05_env.rig.PANDA_HOME,
                    host=config.get("host", "localhost"),
                    port=int(config.get("port", 8000)),
                    open_loop_horizon=int(config.get("open_loop_horizon", 15)),
                    expected_server_identity=expected_server_identity)
                server_identity = policy.verified_server_identity
            try:
                legacy_condition = "reference" if treatment.scene == "box_proxy" else "simany"
                env, suite, factory_dir = legacy.build_env(
                    resolved_cfg, legacy_condition, out_dir)
                source = build_source(
                    env, suite=suite, factory_dir=factory_dir,
                    options=treatment.options,
                    observation_variant=treatment.observation)
                pipeline = ObservationPipeline(
                    source, variant=treatment.observation, env=env,
                    enhancer=_enhancer_for(treatment),
                    image_keys=treatment.options.get("image_keys") or (
                        "observation/exterior_image_1_left",
                        "observation/wrist_image_left"),
                    restoration_options=treatment.options.get("restoration", {}))
                tasks = {task["task_id"]: task for task in suite["tasks"]}
                if policy is None:
                    policy = legacy.get_policy(
                        policy_id, pi05_env.rig.PANDA_HOME,
                        host=config.get("host", "localhost"),
                        port=int(config.get("port", 8000)),
                        open_loop_horizon=int(config.get("open_loop_horizon", 15)),
                        expected_server_identity=expected_server_identity)
                    server_identity = getattr(
                        policy, "verified_server_identity", None)
                if hasattr(policy, "warmup"):
                    policy.warmup(env.reset(settle_s=0.1), "warmup")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                for state in pending:
                    planned_task = planning_tasks.get(state.task_id)
                    contract = _runtime_contract(
                        config=config, state=state, controller_hash=controller_hash,
                        camera_hash=camera_hash, action_convention=action_convention,
                        action_dim=action_dim, policy_hash=policy_hash,
                        server_identity=server_identity,
                        task_instruction=_planned_instruction(scene_cfg, state, config),
                        task=planned_task,
                        construction_artifacts=construction_artifacts)
                    record = _failure_record(
                        treatment, state, error, contract, planned_task)
                    _append_jsonl(ledger_path, record)
                    completed[(treatment.id, state.reset_state_id)] = record
                continue
            for state in pending:
                task = tasks.get(state.task_id)
                if task is None:
                    planned_task = planning_tasks.get(state.task_id)
                    contract = _runtime_contract(
                        config=config, state=state, controller_hash=controller_hash,
                        camera_hash=camera_hash, action_convention=action_convention,
                        action_dim=action_dim, policy_hash=policy_hash,
                        server_identity=server_identity,
                        task_instruction=_planned_instruction(scene_cfg, state, config),
                        suite=suite, task=planned_task,
                        construction_artifacts=construction_artifacts)
                    record = _failure_record(
                        treatment, state, f"task {state.task_id!r} missing from suite",
                        contract, planned_task)
                    _append_jsonl(ledger_path, record)
                    completed[(treatment.id, state.reset_state_id)] = record
                    continue
                episode_id = _episode_id(treatment.id, state.reset_state_id)
                prompt = task["instructions"][config.get("variant", "default")]
                result, ticks, frames, reset_provenance = _run_episode(
                    env, task, policy, pipeline, episode_id=episode_id, prompt=prompt,
                    horizon_s=float(config.get(
                        "horizon_s", config.get("contract", {}).get(
                            "horizon_s", cc.FROZEN_CONTROL_CONTRACT.horizon_seconds))),
                    reset_seed=state.reset_seed,
                    jitter_xy=_episode_jitter(config, state),
                    policy_timeout_s=float(config.get("policy_timeout_s", 20.0)),
                    capture_video=bool(config.get("video", False)))
                if config.get("contract", {}).get("policy", {}).get("sampling") is not None:
                    result["policy_sampling_receipts"] = getattr(policy, "sampling_receipts", [])
                record = {
                    "episode_id": episode_id, "treatment_id": treatment.id,
                    "scene_id": state.scene_id, "task_id": state.task_id,
                    "task_family": _task_family(task),
                    "reset_state_id": state.reset_state_id,
                    "base_seed": state.base_seed, "reset_seed": state.reset_seed,
                    **result}
                if reset_provenance is not None:
                    record["reset_provenance_sha256"] = (
                        manifest_hash.canonical_hash(reset_provenance))
                manifest = _make_manifest(
                    treatment=treatment, state=state, suite=suite,
                    factory_dir=Path(factory_dir), config=config, result=result,
                    controller_hash=controller_hash, camera_hash=camera_hash,
                    action_convention=action_convention, action_dim=action_dim,
                    policy_hash=policy_hash, git_snapshot=git_snapshot,
                    server_identity=server_identity,
                    task_instruction=prompt, task=task,
                    construction_artifacts=construction_artifacts,
                    reset_provenance=reset_provenance)
                record = _write_episode_artifacts(out_dir, record, ticks, frames, manifest)
                evidence = reset_eligibility.get((treatment.id, state.reset_state_id))
                if evidence is not None:
                    record["construction_validity_evidence"] = evidence
                _append_jsonl(ledger_path, record)
                completed[(treatment.id, state.reset_state_id)] = record

    records = read_jsonl(ledger_path)
    current_resolved, _current_scenes, current_planning = \
        _preflight_resolved_scenes(config, spec, out_dir)
    validation_records = ([record for record in records if record["reset_state_id"] in execution_reset_ids]
                          if execution_stage == "pilot" else records)
    validation_ids = execution_reset_ids if execution_stage == "pilot" else planned_reset_ids
    if isinstance(config.get("contract"), dict):
        validation_spec = full_pilot_validation_spec(spec, states) if execution_stage == "pilot" else spec
        validation = validate_saved_treatment_records(
            validation_records, validation_spec, validation_ids, manifest_root=ROOT)
    else:
        validation = validate_records(validation_records, spec, validation_ids)
    validation["violations"].extend(
        _validate_record_inputs_against_current(
            records, current_resolved, current_planning)
    )
    validation["ok"] = not validation["violations"]
    if execution_stage == "pilot":
        _atomic_json(out_dir / "pilot_validation.json", validation)
        if not validation["ok"]:
            raise RuntimeError("fixed policy pilot validation failed: " + " | ".join(validation["violations"]))
        summary={"ledger":str(ledger_path),"validation":validation,"execution_stage":"pilot",
            "full_planned_episodes":len(planned_reset_ids)*len(spec.treatments),
            "pilot_planned_episodes":len(execution_reset_ids)*len(spec.treatments),
            "full_complete":False,"main_table":None,"paper_ready":False}
        _atomic_json(out_dir / "harness_pilot_summary.json", summary)
        return summary
    _atomic_json(out_dir / "validation.json", validation)
    _atomic_json(out_dir / "harness_validation.json", validation)
    if not validation["ok"]:
        raise RuntimeError("paired harness validation failed: "
                           + " | ".join(validation["violations"]))
    table = generate_main_table(config, ledger_path, out_dir / "paper_tables",
                                reuse_existing=resume)
    summary = {"ledger": str(ledger_path), "validation": validation,
               "main_table": table}
    _atomic_json(out_dir / "harness_summary.json", summary)
    return summary


def run_native_episode(adapter, policy, *, config: dict, reset_seed: int,
                       out_dir: Path, treatment_id: str,
                       execution_kind: str = "closed_loop_visual_policy") -> dict:
    """Native-adapter dispatch using this runner's canonical durable ledger writer.

    Native predicates remain authoritative; no DROID state or staged rubric.
    Fixed-action replay is typed separately from visual-policy execution.
    The caller owns the immutable reset/import, performed before this function.
    """
    from robo.roundtrip.adapters.robocasa import observation_hashes
    import imageio.v2 as imageio
    if execution_kind not in {"closed_loop_visual_policy", "fixed_action_replay"}:
        raise ValueError("unknown native execution kind")
    if execution_kind == "closed_loop_visual_policy" and not getattr(policy, "is_visual_policy", False):
        raise ValueError("replay/scripted actions cannot be labeled learned policy")
    schema2 = config.get("schema_version") == 2
    protocol = config.get("execution_protocol", "primary_native")
    if schema2:
        from robo.roundtrip.spec import validate_spec
        validate_spec(config)
        from robo.roundtrip.policy_engine import validate_engine
        engine_identity=policy.metadata if execution_kind=='closed_loop_visual_policy' else policy.metadata.get('source',{}).get('source_policy_identity',{})
        validate_engine(config,engine_identity)
        if treatment_id != config["controller_method"]:
            raise ValueError("treatment differs from resolved v2 method")
        if not getattr(adapter, "canonical_binding", None):
            raise ValueError("v2 native episode requires direct canonical XML/state binding")
        if execution_kind == "closed_loop_visual_policy":
            policy.bind_stream(canonical_instance_id=config["canonical_instance_id"],
                               reset_id=config["reset_id"], policy_rng_seed=config["policy_rng_seed"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    task = config["instance"]["task_id"]
    scene = f"layout{config['instance']['layout_id']}_style{config['instance']['style_id']}"
    reset_id = f"{scene}__{task}__seed{reset_seed}"
    if schema2:
        reset_id = config["reset_id"]
        key = {k: config[k] for k in ("cohort_id", "canonical_instance_id", "reset_id", "scope",
               "sensor_regime", "controller_method", "execution_protocol", "policy_rng_seed")}
        key["policy"] = config["policy"]
        key["policy_id"] = config.get("policy_id", manifest_hash.canonical_hash(config["policy"]))
        if config.get('policy_engine_protocol') is not None:
            key['policy_engine']=engine_identity['policy_engine']
        episode_id = "native-v2-" + manifest_hash.canonical_hash(key)
    else:
        episode_id = elog.episode_id_for(treatment_id + "__" + execution_kind, reset_id)
    ticks = []; actions = []; start = time.monotonic(); error = None
    outcome = elog.Outcome.TASK_FAILURE; success = False
    video = out_dir / "continuous.mp4"
    policy.reset(seed=config["policy_rng_seed"] if schema2 else reset_seed)
    first_success_step = None
    writer = imageio.get_writer(str(video), fps=adapter.native.control_freq,
                               codec="libx264", macro_block_size=1)
    initial = adapter.get_state()
    _atomic_json(out_dir / "initial_state.json", initial)
    try:
        for tick in range(config["horizon"]):
            obs = adapter.get_policy_observation()
            cameras = [obs[k] for k in config["video_image_keys"]]
            writer.append_data(np.concatenate(cameras, axis=1))
            t0 = time.monotonic()
            action = np.asarray(policy.infer(obs), dtype=np.float64)
            inference_s = time.monotonic() - t0
            _, _, done, _ = adapter.step_native_action(action)
            if not np.isfinite(adapter.native.sim.data.qpos).all() or not np.isfinite(adapter.native.sim.data.qvel).all():
                raise FloatingPointError("nonfinite native state")
            success = adapter.native_success()
            if success and first_success_step is None:
                first_success_step = tick + 1
            actions.append(action.tolist())
            ticks.append({"tick": tick, "simulation_time_s": float(adapter.native.sim.data.time),
                "action": action.tolist(), "observation_hashes": observation_hashes(obs),
                "objects": adapter.tracked_objects(), "native_predicates": adapter.native_stage_state(),
                "inference_s": inference_s, "qpos": adapter.native.sim.data.qpos.tolist(),
                "qvel": adapter.native.sim.data.qvel.tolist()})
            if getattr(adapter, "scope_contact_inventory", None) is not None:
                ticks[-1]["scope_contacts"] = adapter.last_scope_contact_steps
            if (execution_kind == "closed_loop_visual_policy" and
                    protocol == "primary_native" and (success or done)):
                break
        outcome = elog.Outcome.SUCCESS if success else elog.Outcome.TASK_FAILURE
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        outcome = elog.classify_exception(exc)
    finally:
        # Include final reached state; the film is one uninterrupted episode.
        video_error = None
        try:
            writer.append_data(np.concatenate([adapter.get_policy_observation()[k]
                                              for k in config["video_image_keys"]], axis=1))
            writer.close()
        except Exception as exc:
            video_error = f"{type(exc).__name__}: {exc}"
            try:
                writer.close()
            except Exception:
                pass
    _atomic_json(out_dir / "actions.json", actions)
    elog.write_timeseries(out_dir / "trace.json.gz", ticks)
    executed = len(ticks) > 0
    measured = outcome in {elog.Outcome.SUCCESS, elog.Outcome.TASK_FAILURE}
    record = {"schema_version": 3, "benchmark": "robocasa_native",
        "episode_id": episode_id, "scene_id": scene, "task_id": task,
        "reset_state_id": reset_id, "reset_seed": reset_seed,
        "treatment_id": treatment_id, "condition": treatment_id,
        "execution_kind": execution_kind, "outcome": outcome.value,
        "executed": executed, "success": bool(success) if measured else None,
        "native_predicates": adapter.native_stage_state() if measured else None,
        "ticks": len(ticks), "horizon": config["horizon"], "error": error,
        "replacement_scope": config["replacement_scope"],
        "oracle_context": config["oracle_context"], "sensor_regime": config["sensor_regime"],
        "config_sha256": manifest_hash.canonical_hash(config),
        "policy_identity": policy.metadata, "wall_s": time.monotonic()-start,
        "video_path": str(video), "timeseries_path": str(out_dir/"trace.json.gz"),
        "actions_path": str(out_dir/"actions.json"), "video_error": video_error}
    if schema2:
        import hashlib
        from robo.roundtrip.spec import paired_config_hash
        record.update({k: config[k] for k in ("cohort_id", "canonical_instance_id", "reset_id",
            "policy_rng_seed", "scope", "controller_method", "execution_protocol",
            "canonical_manifest_sha256", "reset_contract_sha256")})
        record.update(native_schema_version=2, layout_id=config["instance"]["layout_id"],
            style_id=config["instance"]["style_id"], renderer=config.get("renderer", "native"),
            policy_id=config.get("policy_id", manifest_hash.canonical_hash(policy.metadata)),
            policy_identity_sha256=manifest_hash.canonical_hash(policy.metadata),
            comparison_contract_sha256=paired_config_hash(config),
            canonical_reference=adapter.canonical_binding,
            initial_state_sha256=manifest_hash.canonical_hash(initial),
            imported_xml_sha256=hashlib.sha256(adapter.source_xml().encode()).hexdigest(),
            actions_sha256=hashlib.sha256((out_dir/"actions.json").read_bytes()).hexdigest(),
            first_success_step=first_success_step, success_ever=first_success_step is not None,
            success_at_horizon=bool(success) if measured and len(ticks)==config["horizon"] else None,
            full_horizon_completed=measured and len(ticks)==config["horizon"],
            post_success_rule="continue_same_policy_to_native_H" if protocol=="full_horizon_feedback_diagnostic" else "native_primary_termination",
            scorer_interpretation="raw_native_predicate; generated correspondence not established")
    if getattr(adapter, "scope_binding", None) is not None:
        record["scope_treatment"] = adapter.scope_binding
    if config.get('policy_engine_protocol') is not None:
        # Replays retain their source engine inside the canonical replay identity.
        record['policy_engine_protocol']=config['policy_engine_protocol']
        record['policy_engine']=policy.metadata.get('policy_engine') or policy.metadata.get('source',{}).get('source_policy_identity',{}).get('policy_engine')
    if getattr(adapter, "control_binding", None) is not None:
        record["construction_control"] = adapter.control_binding
    _append_jsonl(out_dir / LEDGER_NAME, record)
    _atomic_json(out_dir / "result.json", record)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    out_dir = args.out or config.get("out_dir") or (
        ROOT / "outputs" / "harness_runs" / Path(args.config).stem)
    summary = run_matrix(config, _resolve(out_dir),
                         resume=not args.no_resume,
                         episodes_override=args.episodes)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
