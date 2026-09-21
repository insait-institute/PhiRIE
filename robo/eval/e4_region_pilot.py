"""Fail-closed CPU preflight for the two-room E4 construction pilot.

This module deliberately stops before rendering, camera/oracle validation, or
policy execution.  It turns four sealed E3 materializations into full-room
MuJoCo exports, freezes one paired A0/A4 task bundle per scene, and exercises
the harness' resolution/reset-planning preflight without constructing an
environment.  A successful report therefore means *CPU prerequisites are
valid*, not that an E4 result is paper-ready or that a GPU rollout may start.

Every phase is deliberately fresh-only. If preparation or finalization stops
after creating generated/frozen inputs, no corresponding scene gate or final
preflight is published; recovery requires a new freeze ID rather than resuming
or blessing the partial tree.

The Slurm entry point is ``run/slurm/icra2027_e4_region_cpu.sbatch``.  Direct
CLI use is intentionally commit- and evidence-root-bound as well::

    python -m robo.eval.e4_region_pilot prepare-scene \
      --freeze-id <fresh-id> --scene-id b0a08200c9 \
      --expected-code-commit <40-char-sha>
    python -m robo.eval.e4_region_pilot finalize \
      --freeze-id <same-id> --expected-code-commit <40-char-sha>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
CODE_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EVIDENCE_ROOT = Path("/group/worldcept/PhiRIE/code/SimAny")
SCANNETPP_ROOT = Path("/data/ScanNetpp")
SPLATS_ROOT = Path("/data/ScanNetppv2_gsplat/splats")
FREEZE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OBJECT_SLOT_RE = re.compile(r"^obj_[0-9]+$")
SCENE_IDS = ("b0a08200c9", "825d228aec")
POLICIES = ("A0", "A4")
EXPECTED_TASK_SUFFIXES: dict[str, tuple[str, str]] = {
    "b0a08200c9": ("obj_00_to_region", "obj_03_to_region"),
    "825d228aec": ("obj_02_to_region", "obj_03_to_region"),
}
STABILITY_LIMIT_M = 0.03
PLANNING_SOURCE = "A4"
MAX_TASKS = 2
PILOT_EPISODES = 5
PILOT_SEEDS = (0,)
STUDY_SCOPE = "region_only_engineering_pilot"


class PilotGateError(RuntimeError):
    """One CPU prerequisite is missing, ambiguous, or scientifically invalid."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, root: Path) -> dict[str, Any]:
    path = _regular_file(path, root=root, label="artifact")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_recorded_identity(
    path: Path, *, root: Path, recorded: Any, label: str
) -> dict[str, Any]:
    current = _identity(path, root=root)
    if (
        not isinstance(recorded, Mapping)
        or set(recorded) != {"path", "sha256", "size_bytes"}
        or current != dict(recorded)
    ):
        raise PilotGateError(f"{label} identity changed")
    return current


def _absolute_regular_file(path: Path, *, label: str) -> Path:
    spelling = os.fspath(path)
    pure = PurePosixPath(spelling)
    if (
        not isinstance(spelling, str)
        or not spelling
        or "\\" in spelling
        or not pure.is_absolute()
        or spelling != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise PilotGateError(f"{label} is not a canonical absolute POSIX path")
    candidate = Path(spelling)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise PilotGateError(f"{label} contains a symlink component: {current}")
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or candidate.resolve(strict=True) != candidate
    ):
        raise PilotGateError(f"{label} is not a regular canonical file: {candidate}")
    return candidate


def _external_identity(path: Path, *, label: str) -> dict[str, Any]:
    source = _absolute_regular_file(path, label=label)
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
    }


def external_geometry_identities(scene_id: str) -> dict[str, dict[str, Any]]:
    if scene_id not in SCENE_IDS:
        raise PilotGateError(f"scene {scene_id!r} is outside the frozen pilot")
    scene = SCANNETPP_ROOT / "data" / scene_id
    return {
        name: _external_identity(scene / relative, label=f"{scene_id} {name}")
        for name, relative in (
            ("mesh", Path("scans/mesh_aligned_0.05.ply")),
            ("segments", Path("scans/segments.json")),
            ("segments_anno", Path("scans/segments_anno.json")),
            ("camera_colmap_images", Path("dslr/colmap/images.txt")),
            (
                "camera_transforms_undistorted",
                Path("dslr/nerfstudio/transforms_undistorted.json"),
            ),
        )
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _strict_absolute_directory(value: str | Path, *, label: str) -> Path:
    spelling = os.fspath(value)
    if (
        not isinstance(spelling, str)
        or not spelling
        or "\\" in spelling
        or any(part in {"", ".", ".."} for part in PurePosixPath(spelling).parts)
        or spelling != PurePosixPath(spelling).as_posix()
    ):
        raise PilotGateError(f"{label} is not a canonical POSIX path")
    raw = Path(spelling)
    if not raw.is_absolute():
        raise PilotGateError(f"{label} must be absolute")
    lexical = Path(os.path.abspath(raw))
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise PilotGateError(f"{label} contains a symlink component: {current}")
    if not lexical.is_dir() or lexical.resolve(strict=True) != lexical:
        raise PilotGateError(f"{label} is not a canonical directory: {lexical}")
    return lexical


def evidence_root() -> Path:
    raw = os.environ.get("SIMANY_EVIDENCE_ROOT")
    if raw is None:
        raise PilotGateError("SIMANY_EVIDENCE_ROOT is required")
    root = _strict_absolute_directory(raw, label="SIMANY_EVIDENCE_ROOT")
    if root != EXPECTED_EVIDENCE_ROOT:
        raise PilotGateError(
            "SIMANY_EVIDENCE_ROOT differs from the sealed artifact checkout: "
            f"expected={EXPECTED_EVIDENCE_ROOT}, observed={root}"
        )
    if root == CODE_ROOT:
        raise PilotGateError("code root and evidence root must be distinct")
    return root


def _git_snapshot(expected_commit: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise PilotGateError("expected code commit must be a full lowercase Git SHA")
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=CODE_ROOT, text=True
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=CODE_ROOT,
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotGateError("cannot obtain E4 code Git snapshot") from exc
    if Path(top).resolve(strict=True) != CODE_ROOT:
        raise PilotGateError("executing E4 module is not rooted at CODE_ROOT")
    if head != expected_commit:
        raise PilotGateError(
            f"E4 code commit differs: expected={expected_commit}, observed={head}"
        )
    if status:
        raise PilotGateError(f"E4 code worktree is dirty: {status!r}")
    return {"code_root": str(CODE_ROOT), "commit": head, "dirty": False}


def _validated_freeze_id(value: str) -> str:
    if FREEZE_ID_RE.fullmatch(value) is None:
        raise PilotGateError("freeze ID contains unsafe path characters")
    return value


def _experiment_root(root: Path, freeze_id: str) -> Path:
    return root / "outputs" / "icra2027" / _validated_freeze_id(freeze_id)


def _inside(path: Path, *, root: Path, label: str) -> Path:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise PilotGateError(f"{label} is not a canonical POSIX path")
    pure = PurePosixPath(raw)
    if raw != pure.as_posix() or any(part in {"", ".", ".."} for part in pure.parts):
        raise PilotGateError(f"{label} contains a dot segment or non-canonical spelling")
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise PilotGateError(f"{label} escapes the evidence root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PilotGateError(f"{label} contains a symlink component: {current}")
    return candidate


def _regular_file(path: Path, *, root: Path, label: str) -> Path:
    candidate = _inside(path, root=root, label=label)
    if not candidate.is_file() or candidate.is_symlink():
        raise PilotGateError(f"{label} is not a regular non-symlink file: {candidate}")
    if candidate.resolve(strict=True) != candidate:
        raise PilotGateError(f"{label} is not canonical: {candidate}")
    return candidate


def _regular_directory(path: Path, *, root: Path, label: str) -> Path:
    candidate = _inside(path, root=root, label=label)
    if not candidate.is_dir() or candidate.is_symlink():
        raise PilotGateError(
            f"{label} is not a regular non-symlink directory: {candidate}"
        )
    if candidate.resolve(strict=True) != candidate:
        raise PilotGateError(f"{label} is not canonical: {candidate}")
    return candidate


def _tree_inventory(
    directory: Path, *, factory: Path, root: Path, label: str
) -> list[dict[str, Any]]:
    """Hash the exact regular-file closure below one generated directory."""
    tree = _regular_directory(directory, root=root, label=label)
    factory = _regular_directory(factory, root=root, label=f"{label} factory")
    try:
        tree.relative_to(factory)
    except ValueError as exc:
        raise PilotGateError(f"{label} is outside its materialized factory") from exc

    inventory: list[dict[str, Any]] = []

    def visit(current: Path) -> None:
        try:
            children = sorted(current.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise PilotGateError(f"cannot enumerate {label}: {current}: {exc}") from exc
        for child in children:
            try:
                mode = child.lstat().st_mode
            except OSError as exc:
                raise PilotGateError(
                    f"cannot stat {label} member: {child}: {exc}"
                ) from exc
            if stat.S_ISLNK(mode):
                raise PilotGateError(f"{label} contains a symlink: {child}")
            if stat.S_ISDIR(mode):
                _inside(child, root=root, label=f"{label} directory")
                visit(child)
                continue
            if not stat.S_ISREG(mode):
                raise PilotGateError(f"{label} contains a non-regular member: {child}")
            source = _regular_file(child, root=root, label=f"{label} member")
            inventory.append(
                {
                    "path": source.relative_to(factory).as_posix(),
                    "sha256": _sha256(source),
                    "size_bytes": source.stat().st_size,
                }
            )

    visit(tree)
    if not inventory:
        raise PilotGateError(f"{label} contains no regular files")
    return inventory


def _read_json(path: Path, *, root: Path, label: str) -> Any:
    source = _regular_file(path, root=root, label=label)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotGateError(f"{label} is not valid JSON: {exc}") from exc


def _factory_dir(root: Path, freeze_id: str, policy: str, scene_id: str) -> Path:
    if policy not in POLICIES or scene_id not in SCENE_IDS:
        raise PilotGateError("unsupported policy/scene binding")
    return (
        _experiment_root(root, freeze_id)
        / "construction_variants"
        / policy
        / f"{scene_id}_factory"
    )


def _logical_task_id(scene_id: str, suffix: str) -> str:
    return f"{scene_id}__{suffix}"


def expected_task_ids(scene_id: str) -> tuple[str, str]:
    try:
        return tuple(
            _logical_task_id(scene_id, suffix)
            for suffix in EXPECTED_TASK_SUFFIXES[scene_id]
        )  # type: ignore[return-value]
    except KeyError as exc:
        raise PilotGateError(f"unsupported E4 pilot scene: {scene_id}") from exc


def _normalize_task_id(value: Any, *, scene_id: str) -> str:
    if not isinstance(value, str) or "__" not in value:
        raise PilotGateError("candidate task has an invalid task_id")
    prefix, suffix = value.split("__", 1)
    if prefix not in {scene_id, f"{scene_id}_factory"} or not suffix:
        raise PilotGateError(f"candidate task has a different scene binding: {value!r}")
    return _logical_task_id(scene_id, suffix)


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotGateError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PilotGateError(f"{label} is not finite and non-negative")
    return result


def _compile_scene_xml(path: Path, *, target_slots: Sequence[str]) -> dict[str, Any]:
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(path))
        target_body_ids = {slot: int(model.body(slot).id) for slot in target_slots}
    except Exception as exc:  # noqa: BLE001 - any compile/body lookup failure is fatal
        raise PilotGateError(f"MuJoCo XML compile/body check failed: {exc}") from exc
    return {
        "compiled": True,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "target_body_ids": target_body_ids,
    }


def validate_export_outputs(
    factory_dir: str | Path,
    *,
    scene_id: str,
    policy: str,
    root: str | Path,
    expected_object_slots: Sequence[str],
) -> dict[str, Any]:
    """Validate one full-room export and its exact pilot target candidates."""
    if policy not in POLICIES:
        raise PilotGateError(f"unsupported E4 pilot policy: {policy!r}")
    root = Path(root).resolve(strict=True)
    factory = _inside(Path(factory_dir), root=root, label=f"{policy} factory")
    if not factory.is_dir() or factory.is_symlink():
        raise PilotGateError(f"{policy} factory is not a regular directory")
    expected_ids = expected_task_ids(scene_id)
    accepted_slots = [str(slot) for slot in expected_object_slots]
    if (
        not accepted_slots
        or len(accepted_slots) != len(set(accepted_slots))
        or any(OBJECT_SLOT_RE.fullmatch(slot) is None for slot in accepted_slots)
    ):
        raise PilotGateError("materialization accepted-object roster is invalid")
    target_slots = tuple(task_id.split("__", 1)[1].rsplit("_to_region", 1)[0]
                         for task_id in expected_ids)
    export = factory / "sim_export"
    xml_path = _regular_file(export / "scene.xml", root=root, label="scene XML")
    collision_path = _regular_file(
        export / "room_collision_report.json", root=root, label="room collision report"
    )
    settle_path = _regular_file(
        export / "mujoco_settle.json", root=root, label="MuJoCo settle report"
    )
    tasks_path = _regular_file(
        export / "pi05_tasks.json", root=root, label="pi0.5 candidate tasks"
    )
    isaac_path = _regular_file(
        export / "isaac_manifest.json", root=root, label="Isaac object manifest"
    )

    try:
        parsed_xml = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise PilotGateError(f"scene XML is not well formed: {exc}") from exc
    if parsed_xml.tag != "mujoco":
        raise PilotGateError("scene XML root is not <mujoco>")
    compiler = parsed_xml.find("compiler")
    meshdir_raw = compiler.get("meshdir") if compiler is not None else None
    if not isinstance(meshdir_raw, str) or not meshdir_raw:
        raise PilotGateError("scene XML has no compiler.meshdir")
    meshdir = _inside(Path(meshdir_raw), root=root, label="scene XML meshdir")
    if meshdir != factory:
        raise PilotGateError(
            f"scene XML meshdir differs from its materialized factory: {meshdir}"
        )
    xml_body_names = [body.get("name") for body in parsed_xml.findall("./worldbody/body")]
    if (
        any(not isinstance(name, str) or OBJECT_SLOT_RE.fullmatch(name) is None
            for name in xml_body_names)
        or len(xml_body_names) != len(set(xml_body_names))
        or set(xml_body_names) != set(accepted_slots)
    ):
        raise PilotGateError(
            "scene XML object-body roster differs from materialization acceptance: "
            f"expected={accepted_slots}, observed={xml_body_names}"
        )

    referenced_meshes: dict[str, dict[str, Any]] = {}
    for mesh in parsed_xml.findall("./asset/mesh"):
        name = mesh.get("name")
        raw_file = mesh.get("file")
        if not isinstance(name, str) or not name or name in referenced_meshes:
            raise PilotGateError("scene XML contains a missing/duplicate mesh asset name")
        if not isinstance(raw_file, str) or not raw_file:
            raise PilotGateError(f"scene XML mesh {name!r} has no file")
        pure = PurePosixPath(raw_file)
        if (
            "\\" in raw_file
            or raw_file != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise PilotGateError(f"scene XML mesh {name!r} has a non-canonical file")
        mesh_path = Path(raw_file) if pure.is_absolute() else meshdir / Path(raw_file)
        mesh_path = _regular_file(mesh_path, root=root, label=f"scene XML mesh {name}")
        try:
            mesh_path.relative_to(factory)
        except ValueError as exc:
            raise PilotGateError(
                f"scene XML mesh {name!r} is outside its materialized factory"
            ) from exc
        referenced_meshes[name] = _identity(mesh_path, root=root)
    if not referenced_meshes:
        raise PilotGateError("scene XML has no referenced mesh assets")

    isaac = _read_json(isaac_path, root=root, label="Isaac object manifest")
    if not isinstance(isaac, Mapping) or not isinstance(isaac.get("objects"), list):
        raise PilotGateError("Isaac object manifest schema differs")
    isaac_names = [
        row.get("name") if isinstance(row, Mapping) else None for row in isaac["objects"]
    ]
    if (
        any(not isinstance(name, str) for name in isaac_names)
        or len(isaac_names) != len(set(isaac_names))
        or set(isaac_names) != set(accepted_slots)
    ):
        raise PilotGateError(
            "Isaac object roster differs from materialization acceptance: "
            f"expected={accepted_slots}, observed={isaac_names}"
        )

    collision = _read_json(collision_path, root=root, label="room collision report")
    if not isinstance(collision, Mapping) or collision.get("mode") != "room":
        raise PilotGateError("room collision report does not declare mode=room")
    benchmark = collision.get("benchmark")
    if not isinstance(benchmark, Mapping) or benchmark.get("finite") is not True:
        raise PilotGateError("room collision MuJoCo benchmark is absent or non-finite")
    steps = benchmark.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1000:
        raise PilotGateError("room collision benchmark did not execute 1000 steps")
    state_hash = benchmark.get("state_hash")
    if not isinstance(state_hash, str) or SHA256_RE.fullmatch(state_hash) is None:
        raise PilotGateError("room collision benchmark has no valid state hash")

    settle = _read_json(settle_path, root=root, label="MuJoCo settle report")
    if not isinstance(settle, Mapping) or not isinstance(settle.get("drift_m"), Mapping):
        raise PilotGateError("MuJoCo settle report schema differs")
    drift = {
        str(slot): _finite_nonnegative(value, label=f"settle drift for {slot}")
        for slot, value in settle["drift_m"].items()
    }
    if set(drift) != set(accepted_slots):
        raise PilotGateError(
            "MuJoCo settle body roster differs from materialization acceptance: "
            f"expected={accepted_slots}, observed={sorted(drift)}"
        )
    n = settle.get("n")
    stable = settle.get("stable_3cm")
    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n != len(drift)
        or isinstance(stable, bool)
        or not isinstance(stable, int)
        or stable != sum(value < STABILITY_LIMIT_M for value in drift.values())
    ):
        raise PilotGateError("MuJoCo settle count/stable_3cm summary differs")
    target_drift: dict[str, float] = {}
    for slot in target_slots:
        if slot not in drift:
            raise PilotGateError(f"pilot target {slot!r} is absent from settle report")
        target_drift[slot] = drift[slot]
        if drift[slot] >= STABILITY_LIMIT_M:
            raise PilotGateError(
                f"pilot target {slot!r} drift is not <30 mm: {drift[slot] * 1000:.3f} mm"
            )

    suite = _read_json(tasks_path, root=root, label="pi0.5 candidate tasks")
    if not isinstance(suite, Mapping) or not isinstance(suite.get("tasks"), list):
        raise PilotGateError("candidate task-suite schema differs")
    if suite.get("scene") not in {scene_id, f"{scene_id}_factory"}:
        raise PilotGateError("candidate task suite has a different scene binding")
    declared_xml = suite.get("scene_xml")
    if not isinstance(declared_xml, str):
        raise PilotGateError("candidate task suite has no canonical scene XML path")
    declared_xml_path = _regular_file(
        Path(declared_xml), root=root, label="candidate task-suite scene XML"
    )
    if declared_xml_path != xml_path:
        raise PilotGateError("candidate task suite points at a different scene XML")
    tasks: dict[str, Mapping[str, Any]] = {}
    for task in suite["tasks"]:
        if not isinstance(task, Mapping):
            raise PilotGateError("candidate task is not a mapping")
        logical_id = _normalize_task_id(task.get("task_id"), scene_id=scene_id)
        if logical_id in tasks:
            raise PilotGateError(f"duplicate logical task ID: {logical_id}")
        tasks[logical_id] = task
    missing = sorted(set(expected_ids) - set(tasks))
    if missing:
        raise PilotGateError(f"candidate suite is missing exact pilot tasks: {missing}")
    for task_id, target_slot in zip(expected_ids, target_slots):
        task = tasks[task_id]
        if task.get("target") != target_slot:
            raise PilotGateError(f"pilot task {task_id!r} has a different target")
        if task.get("receptacle") is not None or not isinstance(task.get("region"), Mapping):
            raise PilotGateError(f"pilot task {task_id!r} is not region-only")

    compile_report = _compile_scene_xml(xml_path, target_slots=target_slots)
    generated_trees = {
        name: _tree_inventory(
            factory / name,
            factory=factory,
            root=root,
            label=f"{scene_id}/{policy} generated {name} tree",
        )
        for name in ("sim", "sim_export")
    }
    return {
        "artifacts": {
            "scene_xml": _identity(xml_path, root=root),
            "room_collision_report": _identity(collision_path, root=root),
            "mujoco_settle": _identity(settle_path, root=root),
            "candidate_tasks": _identity(tasks_path, root=root),
            "isaac_manifest": _identity(isaac_path, root=root),
        },
        "benchmark": {
            "finite": True,
            "state_hash": state_hash,
            "steps": steps,
        },
        "compile_check": compile_report,
        "collision_mode": "room",
        "expected_task_ids": list(expected_ids),
        "generated_trees": generated_trees,
        "object_body_roster": accepted_slots,
        "policy_id": policy,
        "referenced_meshes": referenced_meshes,
        "scene_id": scene_id,
        "target_drift_m": target_drift,
    }


def validate_export_against_recorded(
    factory_dir: str | Path,
    *,
    scene_id: str,
    policy: str,
    root: str | Path,
    expected_object_slots: Sequence[str],
    recorded: Any,
) -> dict[str, Any]:
    current = validate_export_outputs(
        factory_dir,
        scene_id=scene_id,
        policy=policy,
        root=root,
        expected_object_slots=expected_object_slots,
    )
    if not isinstance(recorded, Mapping) or current != recorded:
        raise PilotGateError(f"{scene_id}/{policy} export changed after scene gate")
    return current


def _run_export(factory: Path, *, scene_id: str, root: Path) -> None:
    # Do not inherit hidden reconstruction/evaluation knobs from the login
    # shell or a parent experiment.  In particular, SIMANY_MESH_SRC,
    # SIMANY_AUTO/FULL, every legacy SIMF_* override, and CUDA visibility are
    # absent.  These CPU jobs use only the explicitly frozen roots below.
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(CODE_ROOT),
        "SIMANY_EVIDENCE_ROOT": str(root),
        "SIMANY_OUT": str(factory),
        "SIMANY_ROOT": str(CODE_ROOT),
        "SIMANY_SCANNETPP_ROOT": str(SCANNETPP_ROOT),
        "SIMANY_SPLATS_ROOT": str(SPLATS_ROOT),
        "SIMANY_SCENE": scene_id,
    }
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus is not None and cpus.isdigit() and int(cpus) > 0:
        environment["OMP_NUM_THREADS"] = cpus
    for name in ("TMPDIR", "XDG_CACHE_HOME"):
        value = os.environ.get(name)
        if value is None:
            continue
        cache_dir = _inside(Path(value), root=root, label=name)
        if not cache_dir.is_dir() or cache_dir.is_symlink():
            raise PilotGateError(f"{name} is not a repository-local directory")
        environment[name] = str(cache_dir)
    commands = (
        [sys.executable, "-m", "robo.sim.export_mjcf", "--test", "--collision-mode", "room"],
        [sys.executable, "-m", "robo.tasks.pi05_tasks", "--out-dir", str(factory)],
    )
    for command in commands:
        subprocess.run(command, cwd=CODE_ROOT, env=environment, check=True)


def _materialization_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "e3_claim_status": report["e3_claim_status"],
        "e3_code_commit": report["e3_code_commit"],
        "e3_freeze_id": report["e3_freeze_id"],
        "e3_root": report["e3_root"],
        "manifest_sha256": report["manifest_sha256"],
        "materializer_commit": report["materializer_commit"],
        "policy_id": report["policy_id"],
        "roster": report["roster"],
        "scene_id": report["scene_id"],
        "study_scope": report["study_scope"],
        "validator_commit": report["validator_commit"],
    }


def _atomic_create_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise PilotGateError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}"
    try:
        with temporary.open("xb") as handle:
            payload = _json_bytes(value)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


@contextmanager
def _atomic_directory(path: Path) -> Iterator[Path]:
    if path.exists() or path.is_symlink():
        raise PilotGateError(f"refusing to overwrite output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.tmp.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise PilotGateError(f"staging path already exists: {staging}")
    staging.mkdir()
    try:
        yield staging
        os.replace(staging, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def prepare_scene(*, freeze_id: str, scene_id: str, expected_commit: str) -> dict[str, Any]:
    """Run and seal both A0/A4 CPU exports for one scene."""
    if scene_id not in SCENE_IDS:
        raise PilotGateError(f"scene {scene_id!r} is outside the frozen pilot")
    snapshot = _git_snapshot(expected_commit)
    root = evidence_root()
    # Import only after SIMANY_EVIDENCE_ROOT has been checked: the materializer
    # binds its artifact root at module import time.
    from robo.eval.e3_factory_materializer import validate_materialized_factory

    factories = {
        policy: _factory_dir(root, freeze_id, policy, scene_id) for policy in POLICIES
    }
    materializations: dict[str, dict[str, Any]] = {}
    for policy, factory in factories.items():
        report = validate_materialized_factory(
            factory,
            expected_scene_id=scene_id,
            expected_policy_id=policy,
            repository_root=root,
        )
        if report.get("validator_commit") != expected_commit:
            raise PilotGateError(f"{policy} materialization validator commit differs")
        for generated in (factory / "sim", factory / "sim_export"):
            if generated.exists() or generated.is_symlink():
                raise PilotGateError(
                    f"refusing non-fresh {policy} export; path already exists: {generated}"
                )
        materializations[policy] = _materialization_summary(report)

    external_inputs = external_geometry_identities(scene_id)
    exports: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        _run_export(factories[policy], scene_id=scene_id, root=root)
        exports[policy] = validate_export_outputs(
            factories[policy],
            scene_id=scene_id,
            policy=policy,
            root=root,
            expected_object_slots=materializations[policy]["roster"]["accepted_slots"],
        )
    if external_geometry_identities(scene_id) != external_inputs:
        raise PilotGateError(f"{scene_id} external geometry changed during room export")

    report = {
        "code": snapshot,
        "created_utc": _utc_now(),
        "evidence_root": str(root),
        "expected_task_ids": list(expected_task_ids(scene_id)),
        "external_geometry_inputs": external_inputs,
        "exports": exports,
        "freeze_id": freeze_id,
        "manifest_kind": "e4_region_scene_cpu_gate",
        "materializations": materializations,
        "paper_ready": False,
        "scene_id": scene_id,
        "schema_version": SCHEMA_VERSION,
        "study_scope": STUDY_SCOPE,
    }
    gate_path = _experiment_root(root, freeze_id) / "scene_gates" / f"{scene_id}.json"
    _atomic_create_json(gate_path, report)
    return report


def _relative(path: str | Path, *, root: Path) -> str:
    candidate = Path(path).resolve(strict=True)
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise PilotGateError(f"artifact escapes evidence root: {candidate}") from exc


def _harness_config(
    bundle_reports: Mapping[str, Mapping[str, Any]], *, root: Path, freeze_id: str
) -> dict[str, Any]:
    task_ids = [
        task_id
        for scene_id in SCENE_IDS
        for task_id in expected_task_ids(scene_id)
    ]
    reset_ids = [
        f"{task_id}__seed{seed}__ep{episode}"
        for task_id in task_ids
        for seed in PILOT_SEEDS
        for episode in range(PILOT_EPISODES)
    ]
    scenes = []
    for scene_id in SCENE_IDS:
        report = bundle_reports[scene_id]
        # paired_runner historically resolves this top-level path against its
        # code checkout.  Pin the already-validated planning suite to the
        # distinct evidence checkout so this sealed config is safe to replay.
        planning_tasks = _regular_file(
            Path(report["planning_tasks"]),
            root=root,
            label=f"{scene_id} frozen planning tasks",
        )
        scenes.append(
            {
                "id": scene_id,
                "task_freeze_manifest": _relative(
                    _experiment_root(root, freeze_id)
                    / "task_freezes"
                    / scene_id
                    / "manifest.json",
                    root=root,
                ),
                "tasks_json": str(planning_tasks),
                "construction_variants": {
                    "fixed_single_path": {
                        "factory_dir": _relative(report["factories"]["A0"], root=root),
                        "tasks_json": _relative(report["variant_tasks"]["A0"], root=root),
                        "scene_xml": _relative(report["scene_xml"]["A0"], root=root),
                    },
                    "agentic": {
                        "factory_dir": _relative(report["factories"]["A4"], root=root),
                        "tasks_json": _relative(report["variant_tasks"]["A4"], root=root),
                        "scene_xml": _relative(report["scene_xml"]["A4"], root=root),
                    },
                },
            }
        )
    return {
        "schema_version": 1,
        "paper_mode": False,
        "dry_run_only": True,
        "study_scope": STUDY_SCOPE,
        "gpu_launch_allowed": False,
        "out_dir": (
            f"outputs/icra2027/{freeze_id}/manipulation-scripted-region-pilot"
        ),
        "policy": "scripted_sinusoid",
        "episodes": PILOT_EPISODES,
        "seeds": list(PILOT_SEEDS),
        "jitter": 0.01,
        "video": False,
        "variant": "default",
        "horizon_s": 32.0,
        "contract": {
            "policy": {
                "id": "scripted_sinusoid",
                "kind": "scripted_smoke",
                "checkpoint_hash": None,
                "checkpoint_hash_kind": "not_applicable",
            },
            "robot": {
                "arm": "panda_nohand",
                "gripper": "robotiq_2f85",
                "reset_pose_rad": [
                    0,
                    -0.6283185307,
                    0,
                    -2.5132741229,
                    0,
                    1.8849555922,
                    0,
                ],
            },
            "cameras": {
                "status": "task_suite_geometry_only_not_oracle_validated",
                "exterior": "suite_ext_cam",
                "wrist": "wrist_image_left",
                "resolution": [640, 360],
            },
            "action_convention": "absolute_joint_position",
            "controller": {
                "physics_dt": 0.0016666667,
                "substeps_per_tick": 40,
                "action_dim": 8,
                "gripper_binarize_threshold": 0.5,
                "per_tick_joint_delta_clamp_rad": 0.2,
            },
            "control_rate_hz": 15,
            "horizon_s": 32.0,
            "task_instruction": "use each sealed task's default instruction",
            "rubric": {
                "stages": ["grasp", "lift", "hover", "place"],
                "credit_per_stage": 0.25,
                "success_definition": "place held for 1.0s",
            },
            "reset_ids": reset_ids,
        },
        "scenes": scenes,
        "treatments": [
            {
                "id": "a0_full_room_raster",
                "scene": "fixed_single_path",
                "collision": "full_room",
                "observation": "raster",
                "options": {"label": "A0 fixed single path"},
            },
            {
                "id": "a4_full_room_raster",
                "scene": "agentic",
                "collision": "full_room",
                "observation": "raster",
                "options": {"label": "A4 agentic selection"},
            },
        ],
        "comparisons": [
            {
                "id": "construction_region_engineering_pilot",
                "axis": "scene",
                "baseline": "a0_full_room_raster",
                "treatments": ["a0_full_room_raster", "a4_full_room_raster"],
            }
        ],
    }


def _validate_recorded_scene_gate(
    *, root: Path, freeze_id: str, scene_id: str, expected_commit: str
) -> dict[str, Any]:
    gate_path = _experiment_root(root, freeze_id) / "scene_gates" / f"{scene_id}.json"
    recorded = _read_json(gate_path, root=root, label=f"{scene_id} scene gate")
    if not isinstance(recorded, Mapping):
        raise PilotGateError(f"{scene_id} scene gate is not a mapping")
    if set(recorded) != {
        "code",
        "created_utc",
        "evidence_root",
        "expected_task_ids",
        "external_geometry_inputs",
        "exports",
        "freeze_id",
        "manifest_kind",
        "materializations",
        "paper_ready",
        "scene_id",
        "schema_version",
        "study_scope",
    }:
        raise PilotGateError(f"{scene_id} scene gate schema differs")
    expected_bindings = {
        "evidence_root": str(root),
        "expected_task_ids": list(expected_task_ids(scene_id)),
        "freeze_id": freeze_id,
        "manifest_kind": "e4_region_scene_cpu_gate",
        "paper_ready": False,
        "scene_id": scene_id,
        "schema_version": SCHEMA_VERSION,
        "study_scope": STUDY_SCOPE,
    }
    for field, expected in expected_bindings.items():
        if recorded.get(field) != expected:
            raise PilotGateError(f"{scene_id} scene gate differs at {field}")
    if recorded.get("code") != {
        "code_root": str(CODE_ROOT),
        "commit": expected_commit,
        "dirty": False,
    }:
        raise PilotGateError(f"{scene_id} scene gate code binding differs")
    return dict(recorded)


def finalize(*, freeze_id: str, expected_commit: str) -> dict[str, Any]:
    """Freeze paired tasks and run a non-rendering harness contract dry-run."""
    snapshot = _git_snapshot(expected_commit)
    root = evidence_root()
    experiment = _experiment_root(root, freeze_id)
    from robo.eval.e3_factory_materializer import validate_materialized_factory
    from robo.eval.e4_task_freeze import freeze_task_bundle, validate_task_bundle

    current_exports: dict[str, dict[str, Any]] = {}
    materialization_summaries: dict[str, dict[str, Any]] = {}
    scene_gate_identities: dict[str, dict[str, Any]] = {}
    for scene_id in SCENE_IDS:
        scene_gate_path = (
            experiment / "scene_gates" / f"{scene_id}.json"
        )
        recorded = _validate_recorded_scene_gate(
            root=root,
            freeze_id=freeze_id,
            scene_id=scene_id,
            expected_commit=expected_commit,
        )
        scene_gate_identities[scene_id] = _identity(scene_gate_path, root=root)
        if recorded.get("external_geometry_inputs") != external_geometry_identities(scene_id):
            raise PilotGateError(f"{scene_id} external geometry changed after scene gate")
        current_exports[scene_id] = {}
        materialization_summaries[scene_id] = {}
        for policy in POLICIES:
            factory = _factory_dir(root, freeze_id, policy, scene_id)
            materialization = validate_materialized_factory(
                factory,
                expected_scene_id=scene_id,
                expected_policy_id=policy,
                repository_root=root,
            )
            summary = _materialization_summary(materialization)
            if summary != recorded.get("materializations", {}).get(policy):
                raise PilotGateError(
                    f"{scene_id}/{policy} materialization changed after scene gate"
                )
            export = validate_export_against_recorded(
                factory,
                scene_id=scene_id,
                policy=policy,
                root=root,
                expected_object_slots=summary["roster"]["accepted_slots"],
                recorded=recorded.get("exports", {}).get(policy),
            )
            materialization_summaries[scene_id][policy] = summary
            current_exports[scene_id][policy] = export

    provenance_fields = (
        "e3_claim_status",
        "e3_code_commit",
        "e3_freeze_id",
        "e3_root",
        "materializer_commit",
        "study_scope",
        "validator_commit",
    )
    reference = materialization_summaries[SCENE_IDS[0]]["A0"]
    for scene_id in SCENE_IDS:
        for policy in POLICIES:
            current = materialization_summaries[scene_id][policy]
            for field in provenance_fields:
                if current.get(field) != reference.get(field):
                    raise PilotGateError(
                        f"cross-scene E3/materializer provenance differs at {field}: "
                        f"{scene_id}/{policy}"
                    )

    bundle_dirs = {
        scene_id: experiment / "task_freezes" / scene_id for scene_id in SCENE_IDS
    }
    preflight_dir = experiment / "preflight"
    for output in (*bundle_dirs.values(), preflight_dir):
        if output.exists() or output.is_symlink():
            raise PilotGateError(f"refusing to reuse final-gate output: {output}")

    bundle_reports: dict[str, dict[str, Any]] = {}
    for scene_id in SCENE_IDS:
        freeze_task_bundle(
            scene_id=scene_id,
            a0_factory=_factory_dir(root, freeze_id, "A0", scene_id),
            a4_factory=_factory_dir(root, freeze_id, "A4", scene_id),
            a0_candidates=(
                _factory_dir(root, freeze_id, "A0", scene_id)
                / "sim_export"
                / "pi05_tasks.json"
            ),
            a4_candidates=(
                _factory_dir(root, freeze_id, "A4", scene_id)
                / "sim_export"
                / "pi05_tasks.json"
            ),
            planning_source=PLANNING_SOURCE,
            max_tasks=MAX_TASKS,
            out=bundle_dirs[scene_id],
            repository_root=root,
        )
        bundle = validate_task_bundle(
            bundle_dirs[scene_id] / "manifest.json",
            expected_scene_id=scene_id,
            repository_root=root,
        )
        if bundle.get("logical_task_ids") != list(expected_task_ids(scene_id)):
            raise PilotGateError(
                f"{scene_id} frozen task set differs: {bundle.get('logical_task_ids')!r}"
            )
        if bundle.get("planning_source") != PLANNING_SOURCE:
            raise PilotGateError(f"{scene_id} planning source is not A4")
        if bundle.get("max_tasks") != MAX_TASKS:
            raise PilotGateError(f"{scene_id} max_tasks is not 2")
        bundle_reports[scene_id] = bundle

    config = _harness_config(bundle_reports, root=root, freeze_id=freeze_id)
    # This is intentionally a resolution/reset-plan dry-run, not environment,
    # rendering, camera, oracle, or policy execution.
    from robo.eval import harness_runner
    from robo.eval.harness_spec import load_harness_spec
    from robo.eval.paired_runner import plan_reset_states

    spec = load_harness_spec(config)
    resolved, _scene_cfgs, planning_tasks = harness_runner._preflight_resolved_scenes(
        config, spec, experiment / "runtime" / "harness_resolution_dry_run"
    )
    states = plan_reset_states(
        config["scenes"], config["seeds"], config["episodes"], config.get("task_filter", "")
    )
    strict_scene_ids = set(SCENE_IDS)
    harness_runner._validate_planned_resets(states, planning_tasks, strict_scene_ids)
    planned_reset_ids = {state.reset_state_id for state in states}
    if planned_reset_ids != set(config["contract"]["reset_ids"]):
        raise PilotGateError("harness dry-run reset plan differs from frozen contract")
    if len(states) != len(SCENE_IDS) * MAX_TASKS * PILOT_EPISODES:
        raise PilotGateError("harness dry-run planned an unexpected reset count")
    if set(resolved) != set(SCENE_IDS):
        raise PilotGateError("harness dry-run did not resolve both scenes")

    # Rehash at publication time as well as immediately after validation. The
    # final sealed gate therefore authenticates the exact two scene-gate bytes
    # it consumed, and a concurrent/tardy edit fails closed.
    for scene_id in SCENE_IDS:
        _validate_recorded_identity(
            experiment / "scene_gates" / f"{scene_id}.json",
            root=root,
            recorded=scene_gate_identities[scene_id],
            label=f"{scene_id} scene gate",
        )

    gate = {
        "blocking_gates_before_gpu": [
            "camera_oracle_validation_not_run",
            "real_policy_checkpoint_not_bound",
            "real_policy_smoke_not_run",
        ],
        "bundle_manifest_sha256": {
            scene_id: bundle_reports[scene_id]["bundle_manifest_sha256"]
            for scene_id in SCENE_IDS
        },
        "code": snapshot,
        "cpu_prerequisites_ok": True,
        "created_utc": _utc_now(),
        "e3_claim_status": materialization_summaries[SCENE_IDS[0]]["A0"][
            "e3_claim_status"
        ],
        "evidence_root": str(root),
        "expected_task_ids": {
            scene_id: list(expected_task_ids(scene_id)) for scene_id in SCENE_IDS
        },
        "export_closure": current_exports,
        "external_geometry_inputs": {
            scene_id: external_geometry_identities(scene_id) for scene_id in SCENE_IDS
        },
        "freeze_id": freeze_id,
        "gpu_launch_allowed": False,
        "harness_dry_run": {
            "environment_constructed": False,
            "planned_reset_count": len(states),
            "rendering_executed": False,
            "resolved_scene_count": len(resolved),
            "treatment_count": len(spec.treatments),
        },
        "headline_eligible": False,
        "manifest_kind": "e4_region_cpu_preflight",
        "materializations": materialization_summaries,
        "paper_ready": False,
        "planning_source": PLANNING_SOURCE,
        "scene_gate_identities": scene_gate_identities,
        "schema_version": SCHEMA_VERSION,
        "study_scope": STUDY_SCOPE,
    }
    with _atomic_directory(preflight_dir) as staging:
        config_path = staging / "harness_config.json"
        config_path.write_bytes(_json_bytes(config))
        gate_path = staging / "gate.json"
        gate_path.write_bytes(_json_bytes(gate))
        seal = {
            "manifest_kind": "e4_region_cpu_preflight",
            "members": {
                "gate.json": {
                    "sha256": _sha256(gate_path),
                    "size_bytes": gate_path.stat().st_size,
                },
                "harness_config.json": {
                    "sha256": _sha256(config_path),
                    "size_bytes": config_path.stat().st_size,
                },
            },
            "schema_version": SCHEMA_VERSION,
        }
        (staging / "seal.json").write_bytes(_json_bytes(seal))
        for path in staging.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    return gate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-scene")
    prepare.add_argument("--freeze-id", required=True)
    prepare.add_argument("--scene-id", choices=SCENE_IDS, required=True)
    prepare.add_argument("--expected-code-commit", required=True)
    final = subparsers.add_parser("finalize")
    final.add_argument("--freeze-id", required=True)
    final.add_argument("--expected-code-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-scene":
            report = prepare_scene(
                freeze_id=args.freeze_id,
                scene_id=args.scene_id,
                expected_commit=args.expected_code_commit,
            )
        else:
            report = finalize(
                freeze_id=args.freeze_id,
                expected_commit=args.expected_code_commit,
            )
    except (FileNotFoundError, OSError, PilotGateError, subprocess.CalledProcessError,
            TypeError, ValueError) as exc:
        print(f"[e4-region-pilot] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
