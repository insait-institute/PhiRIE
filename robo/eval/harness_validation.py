"""Mechanical, fail-closed validation of paired-harness contracts/outputs."""
from __future__ import annotations

import hashlib
import copy
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from robo.envs.pi05_env import (
    _settle_duration_tolerance,
    _settle_step_count,
)
from robo.eval import e3_factory_materializer as e3_materializer
from robo.eval.harness_spec import HarnessSpec
from robo.manifest.hash import canonical_hash, hash_checkpoint_path
from robo.policy.control_contract import FROZEN_CONTROL_CONTRACT
from robo.policy.registry import PolicyRegistry, PolicyRegistryError
from robo.policy.runtime_identity import (
    PolicyRuntimeIdentityError,
    expected_server_identity_from_config,
    validate_server_identity,
    verify_clean_git_checkout,
)

VALID_ROLLOUT_OUTCOMES = {"success", "task_failure"}
TERMINAL_ROLLOUT_OUTCOMES = VALID_ROLLOUT_OUTCOMES | {
    "build_failure", "policy_timeout", "safety_termination",
    "environment_crash", "enhancer_failure",
}
AXIS_OPTION_PREFIXES: dict[str, tuple[str, ...]] = {
    "scene": ("factory_dir", "tasks_json", "scene_xml"),
    "collision": ("collision_variants", "tasks_json", "scene_xml"),
    "observation": (
        "source_factory", "camera_names", "robot_roots", "enhancer",
        "restoration", "image_keys",
    ),
}
AXIS_ARTIFACT_PATHS: dict[str, tuple[str, ...]] = {
    "scene": (
        "construction_artifacts.accepted_slots",
        "construction_artifacts.materialization_manifest_sha256",
        "construction_artifacts.materialization_policy_id",
        "construction_artifacts.roster_sha256",
        "construction_artifacts.scene_xml_sha256",
    ),
    "collision": ("construction_artifacts.scene_xml_sha256",),
    "observation": (),
}
AXIS_RUNTIME_PATHS: dict[str, tuple[str, ...]] = {
    # Construction and collision interventions may change how free objects
    # settle before policy execution.  The nominal pre-jitter state, planned
    # jitter, settling protocol, robot state, fixtures, and controls remain
    # frozen and are still compared leaf-for-leaf.
    "scene": (
        "reset_provenance.post_settle_pre_policy_state.object_states",
        "reset_provenance.post_settle_pre_policy_state_sha256",
    ),
    "collision": (
        "reset_provenance.post_settle_pre_policy_state.object_states",
        "reset_provenance.post_settle_pre_policy_state_sha256",
    ),
    "observation": (),
}
RESET_PROVENANCE_KEYS = frozenset({
    "schema_version",
    "pre_jitter_nominal_state",
    "pre_jitter_nominal_state_sha256",
    "jitter",
    "jitter_sha256",
    "settle_protocol",
    "settle_protocol_sha256",
    "post_settle_pre_policy_state",
    "post_settle_pre_policy_state_sha256",
})
RESET_STATE_KEYS = frozenset({
    "time_s",
    "robot_arm_qpos_rad",
    "robot_arm_qvel_rad_s",
    "non_object_joint_states",
    "object_states",
    "actuator_controls",
})
RESET_JITTER_KEYS = frozenset({
    "algorithm",
    "reset_seed",
    "body",
    "max_abs_xy_m",
    "uniform_draw_0_1",
    "offset_xy_m",
    "applied",
})
RESET_SETTLE_KEYS = frozenset({
    "engine",
    "step_function",
    "requested_duration_s",
    "model_timestep_s",
    "step_count",
    "simulated_duration_s",
})

REQUIRED_CONTRACT_FIELDS = (
    "policy",
    "robot",
    "cameras",
    "action_convention",
    "controller",
    "control_rate_hz",
    "horizon_s",
    "task_instruction",
    "rubric",
    "reset_ids",
)

# Runtime manifests historically used hashes for the structured fields.  Keep
# those canonical names while accepting the newer explicit values as aliases.
MANIFEST_FROZEN_FIELDS: dict[str, tuple[str, ...]] = {
    "policy": ("contract.policy_id", "contract.policy.id"),
    "checkpoint": (
        "contract.policy_checkpoint_hash", "contract.policy.checkpoint_hash"),
    "robot": ("contract.robot", "contract.robot_config_hash"),
    "cameras": ("contract.cameras", "contract.camera_config_hash"),
    "action_convention": ("contract.action_convention",),
    "controller": ("contract.controller", "contract.controller_config_hash"),
    "control_rate": ("contract.control_rate_hz", "contract.control_hz"),
    "horizon": ("contract.horizon_s",),
    "task_instruction": ("contract.task_instruction",),
    "rubric": ("contract.rubric", "contract.rubric_version"),
    "reset_id": ("contract.reset_state_id", "reset_state_id"),
}
# New harness manifests bind the logical task and complete reset-plan tuple.
# They are optional for historical manifests, but if either side of a pair
# declares one it must be present and identical on the other side.
MANIFEST_OPTIONAL_PAIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "task_id": ("contract.task_id", "task_id"),
    "task_definition_hash": ("contract.task_definition_hash",),
    "reset_definition_hash": ("contract.reset_definition_hash",),
    "rollout_seed": ("contract.rollout_seed", "base_seed"),
    "reset_seed": ("contract.reset_seed", "reset_seed"),
}
CODE_ROOT = Path(__file__).resolve().parents[2]
if CODE_ROOT != e3_materializer.CODE_ROOT:
    raise RuntimeError("harness and E3 materializer code roots differ")
EVIDENCE_ROOT = e3_materializer.REPOSITORY_ROOT
# Backward-compatible name for callers which treated ROOT as the artifact
# checkout.  Git provenance must always use CODE_ROOT instead.
ROOT = EVIDENCE_ROOT
E4_CONSTRUCTION_VARIANTS = frozenset({"fixed_single_path", "agentic"})
_E4_VARIANT_PATH_FIELDS = ("factory_dir", "tasks_json", "scene_xml")


def is_placeholder_hash(value: Any) -> bool:
    """Return true unless *value* is a plausible 256-bit hexadecimal hash."""
    if not isinstance(value, str):
        return True
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "auto", "placeholder", "todo", "unknown"}:
        return True
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        return True
    return len(set(normalized)) == 1


def _two_finite_numbers(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float))
           for item in value):
        return None
    result = [float(item) for item in value]
    return result if all(math.isfinite(item) for item in result) else None


def validate_reset_provenance(
    value: Any, *, reset_definition: Any = None,
) -> list[str]:
    """Validate one realized reset and its persisted content hashes."""
    violations: list[str] = []
    if not isinstance(value, dict) or set(value) != RESET_PROVENANCE_KEYS:
        return ["reset_provenance schema differs"]
    if value.get("schema_version") != 1:
        violations.append("reset_provenance.schema_version differs")

    for state_name in (
        "pre_jitter_nominal_state", "post_settle_pre_policy_state"
    ):
        state = value.get(state_name)
        if not isinstance(state, dict) or set(state) != RESET_STATE_KEYS:
            violations.append(f"reset_provenance.{state_name} schema differs")
        try:
            actual_hash = canonical_hash(state)
        except (TypeError, ValueError):
            violations.append(
                f"reset_provenance.{state_name} is not canonical JSON")
        else:
            if value.get(f"{state_name}_sha256") != actual_hash:
                violations.append(
                    f"reset_provenance.{state_name}_sha256 differs")

    jitter = value.get("jitter")
    if not isinstance(jitter, dict) or set(jitter) != RESET_JITTER_KEYS:
        violations.append("reset_provenance.jitter schema differs")
    else:
        if jitter.get("algorithm") != (
            "numpy.random.RandomState.random_sample_then_affine"
        ):
            violations.append("reset_provenance.jitter.algorithm differs")
        seed = jitter.get("reset_seed")
        if (isinstance(seed, bool) or not isinstance(seed, int)
                or not 0 <= seed <= np.iinfo(np.uint32).max):
            violations.append("reset_provenance.jitter.reset_seed is invalid")
            seed = None
        draw = _two_finite_numbers(jitter.get("uniform_draw_0_1"))
        if draw is None or any(item < 0.0 or item >= 1.0 for item in draw):
            violations.append(
                "reset_provenance.jitter.uniform_draw_0_1 is invalid")
            draw = None
        offset = _two_finite_numbers(jitter.get("offset_xy_m"))
        if offset is None:
            violations.append("reset_provenance.jitter.offset_xy_m is invalid")
        max_abs = jitter.get("max_abs_xy_m")
        if (isinstance(max_abs, bool) or not isinstance(max_abs, (int, float))
                or not math.isfinite(float(max_abs)) or float(max_abs) < 0.0):
            violations.append("reset_provenance.jitter.max_abs_xy_m is invalid")
            max_abs = None
        applied = jitter.get("applied")
        body = jitter.get("body")
        if not isinstance(applied, bool):
            violations.append("reset_provenance.jitter.applied is invalid")
        elif applied != (isinstance(body, str) and bool(body) and
                         max_abs is not None and float(max_abs) > 0.0):
            violations.append("reset_provenance.jitter.applied differs")
        if body is not None and (not isinstance(body, str) or not body):
            violations.append("reset_provenance.jitter.body is invalid")
        if seed is not None and draw is not None:
            replayed = np.random.RandomState(seed).random_sample(2)
            if not np.array_equal(np.asarray(draw), replayed):
                violations.append(
                    "reset_provenance.jitter.uniform_draw_0_1 differs from seed")
        if draw is not None and offset is not None and max_abs is not None:
            expected = (2.0 * np.asarray(draw) - 1.0) * float(max_abs)
            if not np.array_equal(np.asarray(offset), expected):
                violations.append(
                    "reset_provenance.jitter.offset_xy_m differs from draw")
        try:
            jitter_hash = canonical_hash(jitter)
        except (TypeError, ValueError):
            violations.append("reset_provenance.jitter is not canonical JSON")
        else:
            if value.get("jitter_sha256") != jitter_hash:
                violations.append("reset_provenance.jitter_sha256 differs")

    settle = value.get("settle_protocol")
    if not isinstance(settle, dict) or set(settle) != RESET_SETTLE_KEYS:
        violations.append("reset_provenance.settle_protocol schema differs")
    else:
        if settle.get("engine") != "mujoco" or (
            settle.get("step_function") != "mujoco.mj_step"
        ):
            violations.append("reset_provenance.settle_protocol engine differs")
        requested = settle.get("requested_duration_s")
        timestep = settle.get("model_timestep_s")
        steps = settle.get("step_count")
        simulated = settle.get("simulated_duration_s")
        numeric = (requested, timestep, simulated)
        if any(isinstance(item, bool) or not isinstance(item, (int, float))
               or not math.isfinite(float(item)) for item in numeric):
            violations.append("reset_provenance.settle_protocol timing is invalid")
        elif (float(requested) < 0.0 or float(timestep) <= 0.0
              or float(simulated) < 0.0
              or isinstance(steps, bool) or not isinstance(steps, int)
              or steps < 0):
            violations.append("reset_provenance.settle_protocol timing is invalid")
        else:
            try:
                expected_steps = _settle_step_count(requested, timestep)
            except (TypeError, ValueError, OverflowError):
                violations.append(
                    "reset_provenance.settle_protocol timing is invalid")
                expected_steps = None
            if expected_steps is not None and steps != expected_steps:
                violations.append(
                    "reset_provenance.settle_protocol step_count differs")
            tolerance = _settle_duration_tolerance(requested, timestep)
            if not math.isclose(
                float(simulated),
                steps * float(timestep),
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                violations.append(
                    "reset_provenance.settle_protocol simulated_duration_s differs")
        try:
            settle_hash = canonical_hash(settle)
        except (TypeError, ValueError):
            violations.append(
                "reset_provenance.settle_protocol is not canonical JSON")
        else:
            if value.get("settle_protocol_sha256") != settle_hash:
                violations.append("reset_provenance.settle_protocol_sha256 differs")

    if isinstance(reset_definition, dict):
        if reset_definition.get("jitter") != jitter:
            violations.append(
                "reset_provenance.jitter differs from contract.reset_definition")
    else:
        violations.append("contract.reset_definition is missing for reset provenance")
    return violations


def _path_get(value: Any, path: str) -> tuple[bool, Any]:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _first_path(value: dict, paths: tuple[str, ...]) -> tuple[str, bool, Any]:
    for path in paths:
        present, found = _path_get(value, path)
        if present:
            return path, True, found
    return paths[0], False, None


def _validate_checkpoint_reference(owner: str, value: dict,
                                   violations: list[str]) -> None:
    declared_hash = value.get("checkpoint_hash")
    declared_kind = value.get("checkpoint_hash_kind")
    declared_path = value.get("checkpoint_path")
    if not declared_path:
        violations.append(f"{owner}.checkpoint_path is missing")
        return
    path = Path(str(declared_path)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve(strict=False)
    if not path.exists():
        violations.append(f"{owner}.checkpoint_path does not exist: {path}")
        return
    try:
        if declared_kind == "content_sha256":
            if not path.is_file():
                violations.append(
                    f"{owner}.checkpoint_hash_kind content_sha256 requires a file")
                return
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            actual_hash = hash_checkpoint_path(path)
    except OSError as exc:
        violations.append(f"{owner}.checkpoint_path cannot be fingerprinted: {exc}")
        return
    if declared_hash != actual_hash:
        violations.append(
            f"{owner}.checkpoint_hash does not match checkpoint_path fingerprint")


def _validate_e4_real_policy_contract(
    raw: dict, policy: dict, violations: list[str],
) -> None:
    """Bind E4's declaration to registry, clean code, and rig dependencies."""
    _validate_checkpoint_reference("contract.policy", policy, violations)
    if policy.get('sampling') is not None:
        from robo.policy.sampling_contract import validate_contract
        try:
            validate_contract(policy['sampling'], policy.get('training_config', ''))
        except (TypeError, ValueError) as exc:
            violations.append(f'policy sampling contract invalid: {exc}')
    if policy.get("checkpoint_hash_kind") != "tree_path_size_mtime_sha256":
        violations.append(
            "E4 real policy checkpoint_hash_kind must be "
            "tree_path_size_mtime_sha256")
    if not policy.get("training_config"):
        violations.append("contract.policy.training_config is missing")
    try:
        entry = PolicyRegistry.from_config_dir().get(str(policy.get("id", "")))
    except (OSError, ValueError, PolicyRegistryError) as exc:
        violations.append(f"contract.policy cannot resolve registry entry: {exc}")
    else:
        expected_fields = {
            "checkpoint_path": entry.checkpoint_path,
            "checkpoint_hash": entry.checkpoint_hash,
            "training_config": entry.training_config,
        }
        for field, expected in expected_fields.items():
            actual = policy.get(field)
            if field == "checkpoint_path" and actual and expected:
                actual = str(Path(str(actual)).expanduser().resolve(strict=False))
                expected = str(Path(str(expected)).expanduser().resolve(strict=False))
            if actual != expected:
                violations.append(
                    f"contract.policy.{field} differs from policy registry")
        if entry.client_kind != "pi05_server" or entry.status == "unavailable":
            violations.append("contract.policy registry entry is not runnable")

    top_checkpoint = raw.get("checkpoint_path")
    if top_checkpoint is not None:
        resolved_top = str(Path(str(top_checkpoint)).expanduser().resolve(strict=False))
        resolved_contract = str(Path(str(
            policy.get("checkpoint_path", ""))).expanduser().resolve(strict=False))
        if resolved_top != resolved_contract:
            violations.append(
                "top-level checkpoint_path differs from contract.policy.checkpoint_path")

    dependencies = raw.get("contract", {}).get("runtime_dependencies")
    if not isinstance(dependencies, dict):
        violations.append("contract.runtime_dependencies must be a mapping")
        return
    required = {"openpi", "mujoco_menagerie", "checkpoint_cache_root"}
    if set(dependencies) != required:
        violations.append(
            "contract.runtime_dependencies schema differs: "
            f"expected={sorted(required)}, current={sorted(dependencies)}")
    for owner in ("openpi", "mujoco_menagerie"):
        dependency = dependencies.get(owner)
        if not isinstance(dependency, dict) or set(dependency) != {"root", "commit"}:
            violations.append(
                f"contract.runtime_dependencies.{owner} must contain only root/commit")
            continue
        try:
            verify_clean_git_checkout(
                dependency.get("root", ""), dependency.get("commit", ""),
                owner=f"runtime_dependencies.{owner}")
        except PolicyRuntimeIdentityError as exc:
            violations.append(str(exc))
    menagerie = dependencies.get("mujoco_menagerie")
    if isinstance(menagerie, dict):
        root = Path(str(menagerie.get("root", ""))).expanduser().resolve(strict=False)
        for relative in (
            "franka_emika_panda/panda_nohand.xml",
            "robotiq_2f85/2f85.xml",
        ):
            if not (root / relative).is_file():
                violations.append(
                    f"runtime_dependencies.mujoco_menagerie lacks {relative}")
    cache_root = dependencies.get("checkpoint_cache_root")
    if not cache_root or not Path(str(cache_root)).expanduser().is_absolute():
        violations.append(
            "contract.runtime_dependencies.checkpoint_cache_root must be absolute")
    else:
        resolved_cache = Path(str(cache_root)).expanduser().resolve(strict=False)
        if resolved_cache == Path("/scratch") or not str(resolved_cache).startswith(
            "/scratch/"
        ):
            violations.append(
                "contract.runtime_dependencies.checkpoint_cache_root must be "
                "a scoped directory beneath /scratch")
    port = raw.get("port", 8000)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        violations.append("top-level port must be an integer in [1, 65535]")


def validate_harness_contract(spec: HarnessSpec) -> dict[str, Any]:
    """Validate the global paired-treatment config, including paper gates."""
    violations: list[str] = []
    raw = spec.raw
    paper_mode = bool(raw.get("paper_mode", False))
    contract = raw.get("contract")
    if not isinstance(contract, dict):
        return {
            "ok": False,
            "paper_mode": paper_mode,
            "violations": ["missing required mapping contract"],
        }
    for field in REQUIRED_CONTRACT_FIELDS:
        if field not in contract:
            violations.append(f"contract.{field} is missing")
        elif contract[field] is None or contract[field] == "" or contract[field] == {}:
            violations.append(f"contract.{field} is empty")
    reset_ids = contract.get("reset_ids")
    if not isinstance(reset_ids, list) or not reset_ids:
        violations.append("contract.reset_ids must be a non-empty list")
    elif len({str(value) for value in reset_ids}) != len(reset_ids):
        violations.append("contract.reset_ids contains duplicates")

    policy = contract.get("policy")
    if not isinstance(policy, dict):
        violations.append("contract.policy must be a mapping")
    else:
        policy_kind = str(policy.get("kind", "real"))
        if not policy.get("id"):
            violations.append("contract.policy.id is missing")
        if policy_kind not in {"real", "scripted_baseline", "scripted_smoke"}:
            violations.append(f"contract.policy.kind is unsupported: {policy_kind!r}")
        if policy_kind == "real":
            if is_placeholder_hash(policy.get("checkpoint_hash")):
                violations.append("contract.policy.checkpoint_hash is a placeholder")
            if policy.get("checkpoint_hash_kind") not in {
                "content_sha256", "tree_path_size_mtime_sha256",
                "path_size_mtime_sha256",
            }:
                violations.append("contract.policy.checkpoint_hash_kind is not a real hash kind")
            if paper_mode:
                _validate_checkpoint_reference("contract.policy", policy, violations)
        elif paper_mode and policy_kind.startswith("scripted_smoke"):
            violations.append("paper mode cannot use a scripted_smoke policy")

    has_e4_construction = any(
        treatment.scene in E4_CONSTRUCTION_VARIANTS
        for treatment in spec.treatments.values()
    )
    if has_e4_construction and isinstance(policy, dict):
        configured_policy = raw.get("policy")
        if configured_policy is None:
            violations.append(
                "E4 construction harness requires top-level policy to bind runtime selection"
            )
        elif configured_policy != policy.get("id"):
            violations.append(
                "top-level policy differs from contract.policy.id"
            )
        if contract.get("action_convention") != FROZEN_CONTROL_CONTRACT.env_action_convention:
            violations.append(
                "contract.action_convention differs from the frozen runtime contract"
            )
        if contract.get("control_rate_hz") != FROZEN_CONTROL_CONTRACT.rate_hz:
            violations.append(
                "contract.control_rate_hz differs from the frozen runtime contract"
            )
        if "horizon_s" in raw and float(raw["horizon_s"]) != float(
            contract.get("horizon_s", float("nan"))
        ):
            violations.append("top-level horizon_s differs from contract.horizon_s")
        if policy.get("kind", "real") == "real":
            _validate_e4_real_policy_contract(raw, policy, violations)

    for treatment in spec.treatments.values():
        if treatment.options.get("contract_overrides"):
            fields = sorted(treatment.options["contract_overrides"])
            violations.append(
                f"treatment {treatment.id!r} changes frozen field(s): {fields}")
        if paper_mode and treatment.observation == "harmonizer_c":
            enhancer = treatment.options.get("enhancer")
            if not isinstance(enhancer, dict):
                violations.append(
                    f"treatment {treatment.id!r} harmonizer_c requires options.enhancer")
                continue
            if not enhancer.get("backend"):
                violations.append(
                    f"treatment {treatment.id!r} enhancer.backend is missing")
            elif enhancer.get("backend") == "identity":
                violations.append(
                    f"treatment {treatment.id!r} uses identity enhancer in paper mode")
            if is_placeholder_hash(enhancer.get("checkpoint_hash")):
                violations.append(
                    f"treatment {treatment.id!r} enhancer.checkpoint_hash is a placeholder")
            if enhancer.get("checkpoint_hash_kind") not in {
                "content_sha256", "tree_path_size_mtime_sha256",
                "path_size_mtime_sha256",
            }:
                violations.append(
                    f"treatment {treatment.id!r} enhancer.checkpoint_hash_kind is not real")
            _validate_checkpoint_reference(
                f"treatment {treatment.id!r} enhancer", enhancer, violations)
    return {"ok": not violations, "paper_mode": paper_mode, "violations": violations}


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    path = Path(path)
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def _resolve_artifact_path(value: Any, root: Path) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else root / path).resolve(strict=False)


def _read_suite(path_value: Any, root: Path) -> tuple[Path, dict[str, Any]]:
    path = _resolve_artifact_path(path_value, root)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"task suite is not a mapping: {path}")
    return path, value


def _task_index(suite: dict[str, Any], owner: str,
                violations: list[str]) -> dict[str, dict[str, Any]]:
    tasks = suite.get("tasks")
    if not isinstance(tasks, list):
        violations.append(f"{owner} tasks must be a list")
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or not task.get("task_id"):
            violations.append(f"{owner} task[{index}] has no logical task_id")
            continue
        task_id = str(task["task_id"])
        if task_id in result:
            violations.append(f"{owner} has duplicate task_id {task_id!r}")
            continue
        result[task_id] = task
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resolved_scene_treatments(
    scene_cfg: dict[str, Any],
    resolved_by_treatment: dict[str, dict[str, Any]],
    spec: HarnessSpec,
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Preflight one scene before any environment or policy is constructed.

    E4 construction treatments resolve through per-scene A0/A4 entries.  This
    validator proves that the two entries are genuinely different assets while
    retaining one logical task/reset contract across every treatment.
    """
    root = Path(root)
    scene_id = str(scene_cfg.get("id", "?"))
    violations: list[str] = []
    e4_treatments = {
        treatment_id: treatment
        for treatment_id, treatment in spec.treatments.items()
        if treatment.scene in E4_CONSTRUCTION_VARIANTS
    }
    if not e4_treatments:
        # Historical simany/box_proxy matrices retain their original runtime
        # validation path; the stricter artifact/task contract is opt-in with
        # the explicit E3 construction-policy names.
        return {
            "ok": True,
            "scene_id": scene_id,
            "violations": [],
            "logical_task_ids": [],
            "treatment_artifacts": {},
        }

    variants = scene_cfg.get("construction_variants")
    if e4_treatments and not isinstance(variants, dict):
        violations.append(
            f"scene {scene_id!r} requires a construction_variants mapping")
        variants = {}
    for treatment_id, treatment in e4_treatments.items():
        entry = variants.get(treatment.scene) if isinstance(variants, dict) else None
        if not isinstance(entry, dict):
            violations.append(
                f"scene {scene_id!r} is missing construction variant "
                f"{treatment.scene!r} for treatment {treatment_id!r}")
            continue
        missing = [field for field in _E4_VARIANT_PATH_FIELDS if not entry.get(field)]
        if missing:
            violations.append(
                f"scene {scene_id!r} construction variant {treatment.scene!r} "
                f"is missing {missing}")

    task_freeze: dict[str, Any] | None = None
    task_freeze_manifest = scene_cfg.get("task_freeze_manifest")
    if not task_freeze_manifest:
        violations.append(
            f"scene {scene_id!r} requires task_freeze_manifest for paired "
            "E4 construction treatments"
        )
    else:
        try:
            from robo.eval.e4_task_freeze import validate_task_bundle

            task_freeze = validate_task_bundle(
                task_freeze_manifest,
                expected_scene_id=scene_id,
                repository_root=root,
            )
            expected_planning = _resolve_artifact_path(
                task_freeze["planning_tasks"], root
            )
            configured_planning = _resolve_artifact_path(
                scene_cfg.get("tasks_json", ""), root
            )
            if configured_planning != expected_planning:
                violations.append(
                    f"scene {scene_id!r} top-level tasks_json differs from its "
                    "sealed task-freeze planning suite"
                )
            for policy, variant_name in (
                ("A0", "fixed_single_path"), ("A4", "agentic")
            ):
                entry = variants.get(variant_name) if isinstance(variants, dict) else None
                if not isinstance(entry, dict):
                    continue
                expected_factory = _resolve_artifact_path(
                    task_freeze["factories"][policy], root
                )
                expected_tasks = _resolve_artifact_path(
                    task_freeze["variant_tasks"][policy], root
                )
                expected_xml = _resolve_artifact_path(
                    task_freeze["scene_xml"][policy], root
                )
                for field, expected in (
                    ("factory_dir", expected_factory),
                    ("tasks_json", expected_tasks),
                    ("scene_xml", expected_xml),
                ):
                    if _resolve_artifact_path(entry.get(field, ""), root) != expected:
                        violations.append(
                            f"scene {scene_id!r} construction variant "
                            f"{variant_name!r} {field} differs from its sealed "
                            "task-freeze binding"
                        )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            violations.append(
                f"scene {scene_id!r} has invalid paired task freeze: "
                f"{type(exc).__name__}: {exc}"
            )

    planning_suite: dict[str, Any] | None = None
    planning_tasks: dict[str, dict[str, Any]] = {}
    if not scene_cfg.get("tasks_json"):
        if e4_treatments:
            violations.append(
                f"scene {scene_id!r} requires top-level tasks_json for the "
                "variant-independent reset plan")
    else:
        try:
            _, planning_suite = _read_suite(scene_cfg["tasks_json"], root)
            planning_tasks = _task_index(
                planning_suite, f"scene {scene_id!r} planning suite", violations)
        except (OSError, TypeError, ValueError) as exc:
            violations.append(
                f"scene {scene_id!r} cannot load planning task suite: "
                f"{type(exc).__name__}: {exc}")

    loaded: dict[str, tuple[dict[str, Any], dict[str, dict[str, Any]], Path, Path]] = {}
    treatment_artifacts: dict[str, dict[str, Any]] = {}
    for treatment_id, cfg in resolved_by_treatment.items():
        treatment = spec.treatments[treatment_id]
        if not cfg.get("tasks_json"):
            violations.append(
                f"scene {scene_id!r} treatment {treatment_id!r} has no tasks_json")
            continue
        try:
            suite_path, suite = _read_suite(cfg["tasks_json"], root)
            tasks = _task_index(
                suite, f"scene {scene_id!r} treatment {treatment_id!r}", violations)
            factory_dir = _resolve_artifact_path(cfg.get("factory_dir", ""), root)
            xml_value = cfg.get("scene_xml") or suite.get("scene_xml")
            if not xml_value:
                raise KeyError("scene_xml")
            scene_xml = _resolve_artifact_path(xml_value, root)
            loaded[treatment_id] = (suite, tasks, factory_dir, scene_xml)
            if treatment.scene in E4_CONSTRUCTION_VARIANTS:
                if not factory_dir.is_dir():
                    violations.append(
                        f"scene {scene_id!r} treatment {treatment_id!r} "
                        f"factory_dir is not a directory: {factory_dir}")
                if not scene_xml.is_file():
                    violations.append(
                        f"scene {scene_id!r} treatment {treatment_id!r} "
                        f"scene_xml is not a file: {scene_xml}")
                if factory_dir.is_dir():
                    expected_policy = (
                        "A0" if treatment.scene == "fixed_single_path" else "A4"
                    )
                    try:
                        from robo.eval.e3_factory_materializer import (
                            validate_materialized_factory,
                        )

                        materialization = validate_materialized_factory(
                            factory_dir,
                            expected_scene_id=scene_id,
                            expected_policy_id=expected_policy,
                            repository_root=root,
                        )
                        treatment_artifacts[treatment_id] = {
                            "accepted_slots": materialization["roster"]["accepted_slots"],
                            "e3_claim_status": materialization["e3_claim_status"],
                            "e3_code_commit": materialization["e3_code_commit"],
                            "e3_freeze_id": materialization["e3_freeze_id"],
                            "e3_root": materialization["e3_root"],
                            "input_identities_sha256": materialization[
                                "input_identities_sha256"
                            ],
                            "materialization_manifest_sha256": materialization[
                                "manifest_sha256"
                            ],
                            "materialization_policy_id": expected_policy,
                            "object_slots": materialization["roster"]["object_slots"],
                            "roster_sha256": materialization["roster"]["sha256"],
                            "scene_xml_sha256": _file_sha256(scene_xml),
                            "task_freeze_manifest_sha256": (
                                task_freeze["bundle_manifest_sha256"]
                                if task_freeze is not None else None
                            ),
                            "task_contract_sha256": canonical_hash(
                                {
                                    key: value
                                    for key, value in suite.items()
                                    if key != "scene_xml"
                                }
                            ),
                            "source_scene_sha256": materialization[
                                "source_scene_sha256"
                            ],
                            "study_scope": materialization["study_scope"],
                        }
                        claim = materialization["e3_claim_status"]
                        if spec.raw.get("paper_mode") and not (
                            claim.get("paper_ready") is True
                            and claim.get("headline_eligible") is True
                        ):
                            violations.append(
                                f"scene {scene_id!r} treatment {treatment_id!r} "
                                "uses a non-paper E3 materialization in paper mode"
                            )
                    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
                        violations.append(
                            f"scene {scene_id!r} treatment {treatment_id!r} has "
                            f"invalid sealed E3 materialization: {type(exc).__name__}: {exc}")
                    variant_entry = variants.get(treatment.scene, {})
                    # Task suites live in the shared, sealed task-freeze
                    # bundle.  Only the executable scene XML must remain in
                    # the corresponding materialized factory.
                    try:
                        scene_xml.relative_to(factory_dir)
                    except ValueError:
                        violations.append(
                            f"scene {scene_id!r} treatment {treatment_id!r} "
                            f"scene_xml is outside its materialized factory: {scene_xml}")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            violations.append(
                f"scene {scene_id!r} cannot load treatment {treatment_id!r}: "
                f"{type(exc).__name__}: {exc}")

    if planning_suite is not None:
        frozen_suite_fields = ("scene", "robot", "table", "ext_cam", "exclude_objects")
        for treatment_id, (suite, tasks, _factory, _xml) in loaded.items():
            missing_ids = sorted(set(planning_tasks) - set(tasks))
            extra_ids = sorted(set(tasks) - set(planning_tasks))
            if missing_ids or extra_ids:
                violations.append(
                    f"scene {scene_id!r} treatment {treatment_id!r} logical task-id "
                    f"set mismatch: missing={missing_ids}, extra={extra_ids}")
            for task_id in sorted(set(planning_tasks) & set(tasks)):
                planning_hash = task_definition_hash(planning_tasks[task_id])
                treatment_hash = task_definition_hash(tasks[task_id])
                if planning_hash != treatment_hash:
                    violations.append(
                        f"scene {scene_id!r} treatment {treatment_id!r} task "
                        f"definition mismatch for {task_id!r}")
            for field in frozen_suite_fields:
                if suite.get(field) != planning_suite.get(field):
                    violations.append(
                        f"scene {scene_id!r} treatment {treatment_id!r} reset "
                        f"definition mismatch at suite.{field}")
            artifacts = treatment_artifacts.get(treatment_id)
            if artifacts is not None:
                accepted_slots = set(artifacts["accepted_slots"])
                for task_id, task in tasks.items():
                    target = task.get("target")
                    if target not in accepted_slots:
                        violations.append(
                            f"scene {scene_id!r} treatment {treatment_id!r} task "
                            f"{task_id!r} target {target!r} is not an accepted object"
                        )
                    receptacle = task.get("receptacle", task.get("receptacle_body"))
                    if receptacle is not None and receptacle not in accepted_slots:
                        violations.append(
                            f"scene {scene_id!r} treatment {treatment_id!r} task "
                            f"{task_id!r} receptacle {receptacle!r} is not accepted"
                        )

    # Asset identities must vary exactly along the declared intervention axis.
    # This is intentionally stricter than treatment labels: relabeling one XML
    # or swapping a factory directory must fail before environment creation.
    for comparison in spec.comparisons:
        ids = [value for value in comparison.treatments if value in loaded]
        for left_index, left_id in enumerate(ids):
            for right_id in ids[left_index + 1:]:
                left = loaded.get(left_id)
                right = loaded.get(right_id)
                if left is None or right is None:
                    continue
                left_factory, right_factory = left[2], right[2]
                left_xml, right_xml = left[3], right[3]
                same_xml_bytes = False
                if left_xml.is_file() and right_xml.is_file():
                    try:
                        same_xml_bytes = _file_sha256(left_xml) == _file_sha256(right_xml)
                    except OSError as exc:
                        violations.append(
                            f"scene {scene_id!r} cannot fingerprint treatment "
                            f"scene_xml assets: {exc}")
                if comparison.axis == "scene":
                    if left_factory == right_factory:
                        violations.append(
                            f"scene {scene_id!r} construction treatments {left_id!r} and "
                            f"{right_id!r} resolve to the same factory_dir: {left_factory}")
                    if left_xml == right_xml:
                        violations.append(
                            f"scene {scene_id!r} construction treatments {left_id!r} and "
                            f"{right_id!r} resolve to the same scene_xml: {left_xml}")
                    elif same_xml_bytes:
                        violations.append(
                            f"scene {scene_id!r} construction treatments "
                            f"{left_id!r} and {right_id!r} use byte-identical "
                            "scene_xml assets")
                    left_artifacts = treatment_artifacts.get(left_id, {})
                    right_artifacts = treatment_artifacts.get(right_id, {})
                    for field in (
                        "e3_claim_status",
                        "e3_code_commit",
                        "e3_freeze_id",
                        "e3_root",
                        "input_identities_sha256",
                        "object_slots",
                        "source_scene_sha256",
                        "study_scope",
                    ):
                        if left_artifacts.get(field) != right_artifacts.get(field):
                            violations.append(
                                f"scene {scene_id!r} construction treatments "
                                f"{left_id!r} and {right_id!r} differ at sealed {field}")
                elif comparison.axis == "collision":
                    if left_factory != right_factory:
                        violations.append(
                            f"scene {scene_id!r} collision treatments {left_id!r} and "
                            f"{right_id!r} change factory_dir")
                    if left_xml == right_xml or same_xml_bytes:
                        violations.append(
                            f"scene {scene_id!r} collision treatments {left_id!r} and "
                            f"{right_id!r} do not use distinct scene_xml assets")
                elif comparison.axis == "observation":
                    if left_factory != right_factory:
                        violations.append(
                            f"scene {scene_id!r} observation treatments {left_id!r} and "
                            f"{right_id!r} change factory_dir")
                    if left_xml != right_xml:
                        violations.append(
                            f"scene {scene_id!r} observation treatments {left_id!r} and "
                            f"{right_id!r} change scene_xml")

    return {
        "ok": not violations,
        "scene_id": scene_id,
        "violations": violations,
        "logical_task_ids": sorted(planning_tasks),
        "treatment_artifacts": treatment_artifacts,
    }


def task_definition_hash(task: dict[str, Any]) -> str:
    """Stable identity for a logical task shared by all construction arms."""
    return canonical_hash(task)


def validate_records(records: list[dict], spec: HarnessSpec,
                     planned_reset_ids: set[str]) -> dict:
    violations: list[str] = []
    if not planned_reset_ids:
        violations.append("planned reset id set is empty")
    keys = [(r.get("treatment_id"), r.get("reset_state_id")) for r in records]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        violations.append(f"duplicate treatment/reset records: {duplicates[:10]}")
    by_treatment = defaultdict(set)
    for record in records:
        treatment_id = record.get("treatment_id")
        if treatment_id not in spec.treatments:
            violations.append(f"unknown treatment in ledger: {treatment_id!r}")
            continue
        reset_id = record.get("reset_state_id")
        by_treatment[treatment_id].add(reset_id)
        if reset_id not in planned_reset_ids:
            violations.append(f"unplanned reset id {reset_id!r}")
        if record.get("outcome") not in TERMINAL_ROLLOUT_OUTCOMES:
            violations.append(f"unknown outcome {record.get('outcome')!r}")
    for treatment_id in spec.treatment_ids():
        missing = planned_reset_ids - by_treatment[treatment_id]
        if missing:
            violations.append(
                f"treatment {treatment_id!r} is missing {len(missing)} planned resets")
    for comparison in spec.comparisons:
        reset_sets = [by_treatment[t] for t in comparison.treatments]
        if reset_sets and any(current != reset_sets[0] for current in reset_sets[1:]):
            violations.append(f"comparison {comparison.id!r} has unmatched reset-state sets")
    expected = len(planned_reset_ids) * len(spec.treatments)
    terminal = sum(r.get("outcome") in TERMINAL_ROLLOUT_OUTCOMES for r in records)
    valid = sum(r.get("outcome") in VALID_ROLLOUT_OUTCOMES for r in records)
    return {
        "ok": not violations, "violations": violations,
        "planned_per_treatment": len(planned_reset_ids), "records": len(records),
        "valid_rollouts": valid,
        # Terminal failures are data rows, not missing coverage.  Keep the
        # evaluable subset separate for task metrics.
        "terminal_rows": terminal,
        "coverage": terminal / max(expected, 1),
        "evaluable_coverage": valid / max(expected, 1),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, path))
        return flattened
    return {prefix: value}


def _allowed_path(path: str, allowed_fields: set[str]) -> bool:
    for allowed in allowed_fields:
        if (
            path == allowed
            or path.startswith(allowed + ".")
            or path == f"treatment.{allowed}"
            or path.startswith(f"treatment.{allowed}.")
        ):
            return True
        for option_prefix in AXIS_OPTION_PREFIXES.get(allowed, ()):
            full_prefix = f"treatment.options.{option_prefix}"
            if path == full_prefix or path.startswith(full_prefix + "."):
                return True
        for artifact_path in AXIS_ARTIFACT_PATHS.get(allowed, ()):
            if path == artifact_path or path == f"contract.{artifact_path}":
                return True
        for runtime_path in AXIS_RUNTIME_PATHS.get(allowed, ()):
            if path == runtime_path or path.startswith(runtime_path + "."):
                return True
    return False


def validate_manifest_pair(a: dict, b: dict, allowed_fields: set[str]) -> list[str]:
    """Return exact leaf paths that drift outside a declared treatment axis."""
    ignored_prefixes = {
        "created_utc", "treatment_id", "scene_manifest_hash", "scene_id",
        "outcome", "artifacts", "factory_dir", "git",
        # Observed chunk counts may differ after early termination; each
        # receipt is validated against the same frozen seed and algorithm.
        "policy_sampling_receipts", "policy_sampling_ticks",
    }
    left, right = _flatten(a), _flatten(b)
    drift: list[str] = []
    for path in sorted(set(left) | set(right)):
        if any(path == prefix or path.startswith(prefix + ".")
               for prefix in ignored_prefixes):
            continue
        # Treatment identity/label are bookkeeping; semantic axis values are
        # allowed only when explicitly declared by the comparison.
        if path in {"treatment.id", "treatment.options.label"}:
            continue
        if _allowed_path(path, set(allowed_fields)):
            continue
        if left.get(path) != right.get(path):
            drift.append(path)
    return drift


def _frozen_manifest_drift(a: dict, b: dict) -> list[str]:
    drift: list[str] = []
    for field, aliases in MANIFEST_FROZEN_FIELDS.items():
        path_a, present_a, value_a = _first_path(a, aliases)
        path_b, present_b, value_b = _first_path(b, aliases)
        if not present_a or not present_b:
            missing = path_a if not present_a else path_b
            drift.append(f"{missing} (missing)")
        elif value_a != value_b:
            drift.append(path_a if path_a == path_b else f"{path_a}|{path_b}")
    for _field, aliases in MANIFEST_OPTIONAL_PAIRED_FIELDS.items():
        path_a, present_a, value_a = _first_path(a, aliases)
        path_b, present_b, value_b = _first_path(b, aliases)
        if not present_a and not present_b:
            continue
        if not present_a or not present_b:
            missing = path_a if not present_a else path_b
            drift.append(f"{missing} (missing)")
        elif value_a != value_b:
            drift.append(path_a if path_a == path_b else f"{path_a}|{path_b}")
    return drift


def _validate_e4_runtime_policy_manifest(
    manifest: dict, expected_identity: dict, expected_policy: dict,
    *, certified_prebuild: bool = False,
) -> list[str]:
    """Validate declared/current/server identity equality for one E4 row."""
    violations: list[str] = []
    treatment_id = manifest.get("treatment_id")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        return [f"treatment {treatment_id!r} contract is missing"]
    runtime_fingerprint = contract.get("runtime_policy_checkpoint_fingerprint")
    declared_fingerprint = contract.get("policy_checkpoint_hash")
    runtime_policy = contract.get("policy")
    if not isinstance(runtime_policy, dict):
        return [f"treatment {treatment_id!r} contract.policy is missing"]
    nested_declared = runtime_policy.get("checkpoint_hash")
    expected_fingerprint = expected_policy.get("checkpoint_hash")
    values = {
        "contract.policy_checkpoint_hash": declared_fingerprint,
        "contract.policy.checkpoint_hash": nested_declared,
        "contract.runtime_policy_checkpoint_fingerprint": runtime_fingerprint,
    }
    for path, value in values.items():
        if value != expected_fingerprint:
            violations.append(
                f"treatment {treatment_id!r} {path} differs from declared checkpoint")
    if contract.get("policy_execution") == "not_invoked_prebuild":
        if (not certified_prebuild or manifest.get("outcome") != "build_failure"
                or manifest.get("reset_provenance") is not None
                or runtime_policy.get("server_identity") is not None
                or runtime_policy != expected_policy):
            violations.append(
                f"treatment {treatment_id!r} uninvoked policy lacks exact certified prebuild contract")
        return violations
    sampling = expected_policy.get('sampling')
    if sampling is not None:
        if runtime_policy.get('sampling') != sampling:
            violations.append('runtime sampling contract differs')
        completed = manifest.get('outcome') in {'success', 'task_failure'}
        if completed or 'policy_sampling_receipts' in manifest:
            from robo.policy.sampling_contract import validate_episode_receipts
            try:
                validate_episode_receipts(manifest.get('policy_sampling_receipts'), sampling,
                    seed=contract.get('reset_seed'), ticks=manifest.get('policy_sampling_ticks'),
                    completed=completed, chunk_size=15)
            except (TypeError, ValueError) as exc:
                violations.append(f'policy sampling evidence invalid: {exc}')
    server_identity = runtime_policy.get("server_identity")
    try:
        validated = validate_server_identity(
            server_identity, expected=expected_identity,
            verify_runtime_checkpoint_now=False)
    except PolicyRuntimeIdentityError as exc:
        violations.append(
            f"treatment {treatment_id!r} server identity invalid: {exc}")
    else:
        server_policy = validated["policy"]
        server_fingerprint = server_policy["checkpoint"]["fingerprint"]
        if server_fingerprint != expected_fingerprint:
            violations.append(
                f"treatment {treatment_id!r} server checkpoint differs")
        if server_policy["training_config"] != expected_policy.get("training_config"):
            violations.append(
                f"treatment {treatment_id!r} server training config differs")
    return violations


def validate_treatment_manifests(
    manifests: list[dict],
    spec: HarnessSpec,
    planned_reset_ids: set[str],
    *, certified_prebuild_failures: frozenset[tuple[str, str]] = frozenset(),
) -> dict[str, Any]:
    """Validate matrix coverage and frozen-field equality for paired manifests."""
    records = [{
        "treatment_id": value.get("treatment_id"),
        "reset_state_id": value.get("reset_state_id", value.get("contract", {}).get(
            "reset_state_id")),
        "outcome": value.get("outcome"),
    } for value in manifests]
    base = validate_records(records, spec, planned_reset_ids)
    violations = list(base["violations"])
    declared_reset_ids = spec.raw.get("contract", {}).get("reset_ids")
    if isinstance(declared_reset_ids, list):
        declared = {str(value) for value in declared_reset_ids}
        if declared != planned_reset_ids:
            violations.append(
                "planned reset ids differ from contract.reset_ids: "
                f"planned={sorted(planned_reset_ids)}, declared={sorted(declared)}")
    by_key = {(row["treatment_id"], row["reset_state_id"]): manifest
              for row, manifest in zip(records, manifests)}
    contract_policy = spec.raw.get("contract", {}).get("policy", {})
    has_e4_construction = any(
        treatment.scene in E4_CONSTRUCTION_VARIANTS
        for treatment in spec.treatments.values())
    expected_server_identity = None
    if (has_e4_construction and contract_policy
            and contract_policy.get("kind", "real") == "real"):
        try:
            expected_server_identity = expected_server_identity_from_config(spec.raw)
        except (PolicyRuntimeIdentityError, TypeError, ValueError) as exc:
            violations.append(f"cannot resolve expected policy server identity: {exc}")
    if contract_policy and contract_policy.get("kind", "real") == "real":
        for manifest in manifests:
            path, present, checkpoint_hash = _first_path(
                manifest, MANIFEST_FROZEN_FIELDS["checkpoint"])
            if not present or is_placeholder_hash(checkpoint_hash):
                violations.append(
                    f"treatment {manifest.get('treatment_id')!r} {path} is a placeholder")
            if has_e4_construction and expected_server_identity is not None:
                violations.extend(_validate_e4_runtime_policy_manifest(
                    manifest, expected_server_identity, contract_policy,
                    certified_prebuild=(manifest.get("treatment_id"),
                        manifest.get("reset_state_id", manifest.get("contract", {}).get(
                            "reset_state_id"))) in certified_prebuild_failures))
    for manifest in manifests:
        treatment_id = manifest.get("treatment_id")
        contract = manifest.get("contract")
        contract = contract if isinstance(contract, dict) else {}
        reset_definition = contract.get("reset_definition")
        if isinstance(reset_definition, dict):
            try:
                actual_definition_hash = canonical_hash(reset_definition)
            except (TypeError, ValueError):
                violations.append(
                    f"treatment {treatment_id!r} contract.reset_definition "
                    "is not canonical JSON")
            else:
                if contract.get("reset_definition_hash") != actual_definition_hash:
                    violations.append(
                        f"treatment {treatment_id!r} "
                        "contract.reset_definition_hash differs")
        provenance = manifest.get("reset_provenance")
        treatment = manifest.get("treatment")
        e4_runtime = (
            isinstance(treatment, dict)
            and treatment.get("scene") in E4_CONSTRUCTION_VARIANTS
            and manifest.get("outcome") != "build_failure"
        )
        if provenance is None:
            if e4_runtime:
                violations.append(
                    f"treatment {treatment_id!r} reset_provenance is missing")
            continue
        for defect in validate_reset_provenance(
            provenance, reset_definition=reset_definition
        ):
            violations.append(f"treatment {treatment_id!r} {defect}")
    for comparison in spec.comparisons:
        baseline_id = comparison.baseline or comparison.treatments[0]
        for reset_id in sorted(planned_reset_ids):
            baseline = by_key.get((baseline_id, reset_id))
            if baseline is None:
                continue
            for treatment_id in comparison.treatments:
                current = by_key.get((treatment_id, reset_id))
                if current is None or treatment_id == baseline_id:
                    continue
                comparison_baseline, comparison_current = baseline, current
                if any(
                    (value.get("treatment_id"), reset_id) in certified_prebuild_failures
                    and value.get("outcome") == "build_failure"
                    and value.get("reset_provenance") is None
                    for value in (baseline, current)
                ):
                    # Both planned reset contracts remain fully compared.
                    # An authenticated prebuild failure has no executed state
                    # to pair with the individually validated other arm.
                    comparison_baseline = {k:v for k,v in baseline.items() if k != "reset_provenance"}
                    comparison_current = {k:v for k,v in current.items() if k != "reset_provenance"}
                    if any(
                        (value.get("treatment_id"), reset_id) in certified_prebuild_failures
                        and value.get("contract", {}).get("policy_execution") == "not_invoked_prebuild"
                        for value in (baseline, current)
                    ):
                        # Expected policy/checkpoint/runtime contracts remain
                        # fully paired. Only the separately authenticated live
                        # service observation is absent for an uninvoked arm.
                        comparison_baseline = copy.deepcopy(comparison_baseline)
                        comparison_current = copy.deepcopy(comparison_current)
                        for value in (comparison_baseline, comparison_current):
                            contract = value.get("contract", {})
                            contract.pop("policy_execution", None)
                            contract.get("policy", {}).pop("server_identity", None)
                for field in validate_manifest_pair(
                    comparison_baseline, comparison_current, {comparison.axis}
                ):
                    violations.append(
                        f"comparison {comparison.id!r} reset {reset_id!r} "
                        f"undeclared drift: {field}")
                for field in _frozen_manifest_drift(baseline, current):
                    message = (
                        f"comparison {comparison.id!r} reset {reset_id!r} "
                        f"undeclared drift: {field}")
                    if message not in violations:
                        violations.append(message)
    result = dict(base)
    result["violations"] = violations
    result["ok"] = not violations
    return result


def validate_saved_treatment_records(
    records: list[dict],
    spec: HarnessSpec,
    planned_reset_ids: set[str],
    *,
    manifest_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load saved success manifests and explicit failure contracts, then validate."""
    root = Path(manifest_root) if manifest_root is not None else Path.cwd()
    manifests: list[dict] = []
    load_violations: list[str] = []
    certified_prebuild_failures = set()
    for record in records:
        manifest_path = record.get("manifest_path")
        if manifest_path:
            path = Path(str(manifest_path))
            if not path.is_absolute():
                path = root / path
            try:
                value = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                load_violations.append(
                    f"cannot load manifest for {record.get('episode_id')!r}: {exc}")
                continue
            if not isinstance(value, dict):
                load_violations.append(
                    f"manifest for {record.get('episode_id')!r} is not a mapping")
                continue
            if value.get('contract', {}).get('policy', {}).get('sampling') is not None:
                if (record.get('policy_sampling_receipts') != value.get('policy_sampling_receipts')
                        or record.get('ticks') != value.get('policy_sampling_ticks')):
                    load_violations.append('ledger sampling receipts/ticks differ from manifest')
            manifests.append(value)
            continue
        if isinstance(record.get("contract"), dict):
            from robo.eval.e4_reset_eligibility import certify_prebuild_failure
            try:
                if certify_prebuild_failure(record, spec, root=root):
                    certified_prebuild_failures.add((record["treatment_id"], record["reset_state_id"]))
            except (KeyError, OSError, TypeError, ValueError) as exc:
                load_violations.append(f"invalid prebuild failure evidence: {exc}")
            # Build/environment failures may happen before artifact directories
            # exist.  Their ledger row is still a terminal, comparable contract.
            manifests.append({
                "schema_version": 1,
                "manifest_kind": "harness_rollout",
                "treatment_id": record.get("treatment_id"),
                "treatment": record.get("treatment"),
                "scene_id": record.get("scene_id"),
                "contract": record["contract"],
                "construction_artifacts": record["contract"].get(
                    "construction_artifacts"),
                "outcome": record.get("outcome"),
            })
        else:
            load_violations.append(
                f"record {record.get('episode_id')!r} has neither manifest_path nor contract")
    result = validate_treatment_manifests(manifests, spec, planned_reset_ids,
        certified_prebuild_failures=frozenset(certified_prebuild_failures))
    result["violations"].extend(load_violations)
    result["ok"] = not result["violations"]
    result["source_records"] = len(records)
    result["loaded_manifests"] = len(manifests)
    return result
