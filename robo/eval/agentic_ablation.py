"""E3 agentic-construction ablation with a hard controller/GT boundary.

The driver is deliberately split into independently publishable phases:

``inventory``
    Resolve the frozen 50-scene/457-object conditional population and hash the
    raw TRELLIS/ReconViaGen proposal pool.  Legacy reports are read only here.
``observe``
    Render expected depth from the frozen source Scene Gaussian and unproject
    pixels selected by the frozen RGBA crop.  Evaluation geometry is not read.
``control``
    Register raw proposals to the observation, create one-convex-part physical
    artifacts, run an isolated PyBullet drop, and invoke the pure A0--A4
    controller.  Evaluation geometry is not read.
``evaluate``
    Verify the immutable controller-shard seal before opening the evaluation
    surface and computing F1/CD.
``aggregate``
    Verify every scene shard and publish the fixed-denominator tables.

All mutable artifacts and staging directories are below this repository.  The
only external files that may be opened are immutable, hash-pinned Scene
Gaussian/camera inputs declared by the E2 source-identity manifests.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from agents.orchestrator.artifact import (
    ProposalRecord,
    canonical_json,
    proposal_digest,
    sha256_file,
)
from agents.orchestrator.controller import plan_retry, run_policies
from agents.orchestrator.evidence import choose_best, evaluate_gate
from agents.orchestrator.job_graph import validate_ledger
from agents.orchestrator.policies import POLICY_IDS, validate_policy_config


SCHEMA_VERSION = 1
CODE_ROOT = Path(__file__).resolve().parents[2]


def _git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot authenticate Git checkout: {root}") from exc


def _validated_evidence_root(value: str | None) -> Path:
    """Allow one explicit sibling worktree's artifacts, never an arbitrary root."""
    if value is None:
        return CODE_ROOT
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError("SIMANY_EVIDENCE_ROOT must be absolute")
    # Inspect the lexical spelling before normalization, including dangling links.
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("SIMANY_EVIDENCE_ROOT contains a symlink")
    root = Path(os.path.abspath(raw))
    if not root.is_dir() or Path(_git_value(root, "rev-parse", "--show-toplevel")) != root:
        raise ValueError("SIMANY_EVIDENCE_ROOT must be an exact existing Git checkout root")
    code_common = _git_value(CODE_ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir")
    evidence_common = _git_value(root, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if code_common != evidence_common:
        raise ValueError("evidence and executing source must share the same Git common directory")
    return root


REPOSITORY_ROOT = _validated_evidence_root(os.environ.get("SIMANY_EVIDENCE_ROOT"))
SCENE_ID_RE = re.compile(r"^[0-9a-f]{10}$")
FREEZE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
OBJECT_SLOT_RE = re.compile(r"^obj_[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DROP_HZ = 240
MESH_SAMPLE_COUNT = 20_000
OBSERVATION_ALPHA_MIN = 0.60
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 8.0
RGBA_ALPHA_MIN = 128
PIXEL_CENTER_OFFSET = 0.5

OBSERVATION_PROTOCOL = {
    "render_mode": "RGB+ED",
    "rendered_rgb": "discarded",
    "rendered_channels_used": ["expected_depth", "gaussian_alpha"],
    "crop_channel_used": "rgba_alpha_only",
    "rgba_alpha_threshold_inclusive_uint8": RGBA_ALPHA_MIN,
    "gaussian_alpha_threshold_inclusive": OBSERVATION_ALPHA_MIN,
    "depth_range_m_inclusive": [DEPTH_MIN_M, DEPTH_MAX_M],
    "pixel_center_offset": PIXEL_CENTER_OFFSET,
    "camera_pose_input_convention": "colmap_world_to_camera",
    "camera_to_world_conversion": "matrix_inverse",
    "output_coordinate_frame": "world",
    "output_dtype": "float32",
    "output_unit": "metre",
    "forbidden_render_inputs": [
        "rendered_rgb",
        "evaluation_image",
        "evaluation_mask",
        "evaluation_geometry",
    ],
}

PROBE_PROTOCOL = {
    "mesh_sample_seed": 42,
    "mesh_sample_count": 20_000,
    "symmetric_clip_distance_m": 0.03,
    "support_band_m": 0.01,
    "initial_support_gap_m": 0.005,
    "collision_backend": "deterministic_single_convex_hull_v1",
    "collision_parts_generated": 1,
    "pybullet_hz": 240,
    "velocity_zero_settle_s": 2,
    "free_settle_s": 2,
    "sunk_when_aabb_min_z_m_less_than": -0.01,
    "settle_stable_drift_threshold_m": 0.03,
    "settle_stable_requires_not_sunk": True,
    "category_independent_physics": {
        "mass_kg": 0.3,
        "friction": 0.5,
        "restitution": 0.1,
    },
}

EVIDENCE_MANIFEST_FIELDS = (
    "producer",
    "producer_commit",
    "input_hashes",
    "raw_values",
    "missing_flags",
    "started_utc",
    "finished_utc",
    "wall_s",
)

RUNTIME_ARTIFACT_ROLES = {
    "registration.json": "registration",
    "evidence.json": "evidence",
    "physical/mesh_sim.obj": "mesh_sim",
    "physical/collision/part_00.obj": "collision",
    "physical/object.urdf": "urdf",
    "physical/physics.json": "physics",
    "physical/probe.json": "probe",
}

# These are forbidden only for the observe/control read boundary.  Inventory
# needs the two legacy audit inputs, and evaluation alone opens the GT surface.
CONTROLLER_FORBIDDEN_BASENAMES = frozenset(
    {
        "gt_points.ply",
        "report.json",
        "aligned.json",
        "aligned_all.json",
        "aligned_all_prehybrid.json",
        "rvg_eval.json",
        "hybrid.json",
        "hybrid_all.json",
        "eval_vs_gt.json",
    }
)
CONTROLLER_FORBIDDEN_FIELDS = frozenset(
    {
        "gt_object_id",
        "evaluation_ground_truth_geometry",
        "eval",
        "f1",
        "f1_20",
        "f1_40",
        "chamfer_med_m",
        "tier",
        "rejected",
        "winner",
        "canonical_source",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validated_observation_protocol(jobs: Mapping[str, Any]) -> dict[str, Any]:
    protocol = jobs.get("observation_protocol")
    if protocol != OBSERVATION_PROTOCOL:
        raise ValueError(
            "jobs.observation_protocol differs from the implemented frozen protocol"
        )
    return dict(OBSERVATION_PROTOCOL)


def _validate_probe_protocol(policies: Mapping[str, Any]) -> None:
    executor = policies.get("executor_contract")
    protocol = executor.get("probe") if isinstance(executor, Mapping) else None
    if protocol != PROBE_PROTOCOL:
        raise ValueError(
            "policies.executor_contract.probe differs from the implemented frozen probe"
        )
    evidence_contract = policies.get("construction_evidence")
    required_fields = (
        evidence_contract.get("evidence_manifest_required_fields")
        if isinstance(evidence_contract, Mapping)
        else None
    )
    if required_fields != list(EVIDENCE_MANIFEST_FIELDS):
        raise ValueError("frozen evidence-manifest fields differ from the executor")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, identity: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{identity} must be a lowercase SHA-256 digest")
    return value


def _require_scene_id(value: Any) -> str:
    value = str(value)
    if not SCENE_ID_RE.fullmatch(value):
        raise ValueError(f"invalid scene ID: {value!r}")
    return value


def _require_object_slot(value: Any) -> str:
    value = str(value)
    if not OBJECT_SLOT_RE.fullmatch(value) or Path(value).name != value:
        raise ValueError(f"invalid object slot: {value!r}")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _lexical_absolute(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = REPOSITORY_ROOT / value
    # abspath normalizes '..' without following symlinks.
    return Path(os.path.abspath(os.fspath(value)))


def _reject_symlink_components(path: Path, *, stop: Path | None = None) -> None:
    path = _lexical_absolute(path)
    stop = _lexical_absolute(stop or Path(path.anchor))
    if not _is_relative_to(path, stop):
        raise ValueError(f"path is outside checked root {stop}: {path}")
    current = stop
    if current.exists() and current.is_symlink():
        raise ValueError(f"symlink path component is forbidden: {current}")
    for component in path.relative_to(stop).parts:
        if component in {"", ".", ".."}:
            raise ValueError(f"unsafe path component in {path}")
        current = current / component
        if current.exists() and current.is_symlink():
            raise ValueError(f"symlink path component is forbidden: {current}")


def checked_repo_path(
    path: str | Path,
    identity: str,
    *,
    must_exist: bool = True,
    kind: str | None = None,
) -> Path:
    value = _lexical_absolute(path)
    root = _lexical_absolute(REPOSITORY_ROOT)
    if not _is_relative_to(value, root):
        raise ValueError(f"{identity} must stay below repository root: {value}")
    _reject_symlink_components(value, stop=root)
    if must_exist and not value.exists():
        raise FileNotFoundError(f"missing {identity}: {value}")
    if value.exists():
        if kind == "file" and not value.is_file():
            raise ValueError(f"{identity} is not a regular file: {value}")
        if kind == "dir" and not value.is_dir():
            raise ValueError(f"{identity} is not a directory: {value}")
    return value


def _checked_code_path(path: str | Path, identity: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = CODE_ROOT / value
    value = Path(os.path.abspath(value))
    if not _is_relative_to(value, CODE_ROOT):
        raise ValueError(f"{identity} must stay below executing code root: {value}")
    _reject_symlink_components(value, stop=CODE_ROOT)
    if not value.is_file():
        raise FileNotFoundError(f"missing {identity}: {value}")
    return value


def _load_yaml_code(path: str | Path, identity: str) -> dict[str, Any]:
    import yaml
    source = _checked_code_path(path, identity)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{identity} must be a YAML object")
    return value


def _checked_external_identity(identity: Mapping[str, Any], label: str) -> Path:
    """Resolve one E2-declared read-only source and verify size/hash."""
    path = Path(str(identity.get("path", "")))
    if not path.is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    path = _lexical_absolute(path)
    _reject_symlink_components(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    expected_size = identity.get("size_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise ValueError(f"{label}.size_bytes must be an integer")
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"{label} size drift: expected {expected_size}, found {path.stat().st_size}"
        )
    expected_hash = _require_sha256(identity.get("sha256"), f"{label}.sha256")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{label} hash drift: expected {expected_hash}, found {actual_hash}"
        )
    return path


def _display_path(path: str | Path) -> str:
    value = _lexical_absolute(path)
    if _is_relative_to(value, REPOSITORY_ROOT):
        return value.relative_to(REPOSITORY_ROOT).as_posix()
    return value.as_posix()


def _mkdir_repo_parents(path: Path) -> None:
    path = checked_repo_path(path, "output parent", must_exist=False)
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    _reject_symlink_components(current, stop=REPOSITORY_ROOT)
    if not current.is_dir():
        raise ValueError(f"output ancestor is not a directory: {current}")
    for directory in reversed(missing):
        directory.mkdir()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    destination = checked_repo_path(path, "output file", must_exist=False)
    _mkdir_repo_parents(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output: {destination}")
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    try:
        with staging.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if destination appeared.
        os.link(staging, destination)
        staging.unlink()
        _fsync_directory(destination.parent)
    finally:
        if staging.exists() and not staging.is_symlink():
            staging.unlink()
    return destination


def _atomic_write_json(path: str | Path, value: Any) -> Path:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    return _atomic_write_bytes(path, payload)


def _atomic_write_text(path: str | Path, value: str) -> Path:
    return _atomic_write_bytes(path, value.encode("utf-8"))


@contextlib.contextmanager
def _atomic_directory(destination: str | Path):
    final = checked_repo_path(destination, "phase output directory", must_exist=False)
    _mkdir_repo_parents(final.parent)
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite immutable output: {final}")
    staging = final.with_name(f".{final.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        yield staging
        _fsync_directory(staging)
        if final.exists() or final.is_symlink():
            raise FileExistsError(f"phase output appeared during publication: {final}")
        staging.rename(final)
        _fsync_directory(final.parent)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def _write_json_inside(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes_inside(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _load_json_repo(path: str | Path, identity: str) -> Any:
    source = checked_repo_path(path, identity, kind="file")
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml_repo(path: str | Path, identity: str) -> dict[str, Any]:
    import yaml

    source = checked_repo_path(path, identity, kind="file")
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{identity} must be a YAML mapping")
    return value


def _identity(path: Path) -> dict[str, Any]:
    path = checked_repo_path(path, "artifact", kind="file")
    return {
        "path": _display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_identity(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} identity must be an object")
    path = Path(str(value.get("path", "")))
    if not path.is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    _require_sha256(value.get("sha256"), f"{label}.sha256")
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{label}.size_bytes must be positive")
    # Inventory trusts the already-frozen E2 content digest and checks cheap
    # identity properties.  Observe re-hashes immediately before opening.
    _reject_symlink_components(path)
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"{label} is missing or has size drift: {path}")
    return {"path": path.as_posix(), "size_bytes": size, "sha256": value["sha256"]}


def _check_no_forbidden_fields(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in CONTROLLER_FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden controller field at {location}: {key}")
            _check_no_forbidden_fields(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_no_forbidden_fields(child, location=f"{location}[{index}]")


def _checked_controller_file(path: str | Path, identity: str) -> Path:
    value = checked_repo_path(path, identity, kind="file")
    if value.name in CONTROLLER_FORBIDDEN_BASENAMES or "evaluation_matching" in value.parts:
        raise ValueError(f"controller is forbidden from reading {value.name}: {value}")
    return value


def _load_controller_json(path: str | Path, identity: str) -> Any:
    value = _checked_controller_file(path, identity)
    with value.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _check_no_forbidden_fields(payload, location=identity)
    return payload


def _format_template(template: str, **values: Any) -> Path:
    try:
        formatted = template.format(**values)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid frozen path template {template!r}: {exc}") from exc
    return checked_repo_path(formatted, "templated repository input", kind="file")


def _validate_context(
    jobs_path: str | Path,
    policies_path: str | Path,
    contract_path: str | Path,
    freeze_id: str,
    out_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    if not FREEZE_ID_RE.fullmatch(freeze_id):
        raise ValueError(f"unsafe freeze ID: {freeze_id!r}")
    jobs_file = _checked_code_path(jobs_path, "agentic jobs config")
    policies_file = _checked_code_path(policies_path, "agentic policies config")
    contract_file = checked_repo_path(contract_path, "E0 contract", kind="file")
    jobs = _load_yaml_code(jobs_file, "agentic jobs config")
    policies = _load_yaml_code(policies_file, "agentic policies config")
    contract = _load_json_repo(contract_file, "E0 contract")
    if jobs.get("schema_version") != 1 or policies.get("schema_version") != 1:
        raise ValueError("E3 jobs and policies must use schema_version=1")
    if jobs.get("study_scope") != policies.get("study_scope"):
        raise ValueError("jobs and policies study_scope differ")
    validate_policy_config(policies)
    _validate_probe_protocol(policies)
    output = checked_repo_path(out_dir, "E3 output root", must_exist=False)
    declared = str(jobs.get("output_dir", "")).format(freeze_id=freeze_id)
    if output != checked_repo_path(declared, "declared E3 output", must_exist=False):
        raise ValueError(f"--out differs from jobs output_dir: {output} != {declared}")
    return jobs, policies, contract, output


def _scene_roster(jobs: Mapping[str, Any]) -> list[str]:
    population = jobs.get("population")
    if not isinstance(population, Mapping):
        raise ValueError("jobs.population must be an object")
    roster_path = _checked_code_path(
        population.get("scene_roster_config", ""), "scene roster"
    )
    roster = _load_yaml_code(roster_path, "scene roster")
    values = roster.get("population", {}).get("scene_ids")
    if not isinstance(values, list):
        raise ValueError("scene roster has no population.scene_ids list")
    scenes = [_require_scene_id(value) for value in values]
    expected = int(population.get("planned_scenes", -1))
    if len(scenes) != expected or len(set(scenes)) != expected:
        raise ValueError(f"scene roster must contain exactly {expected} unique scenes")
    expected_file_hash = population.get("scene_roster_config_file_sha256")
    if expected_file_hash and sha256_file(roster_path) != expected_file_hash:
        raise ValueError("scene roster config content hash mismatch")
    configured_hash = population.get("scene_roster_sha256")
    actual_roster_hash = hashlib.sha256(
        ("\n".join(scenes) + "\n").encode("utf-8")
    ).hexdigest()
    if configured_hash and configured_hash != actual_roster_hash:
        raise ValueError("normalized scene roster hash mismatch")
    return scenes


def _extract_e2_sources(
    jobs: Mapping[str, Any], scene_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = jobs.get("observation_source")
    if not isinstance(spec, Mapping):
        raise ValueError("jobs.observation_source must be an object")
    manifest = _format_template(str(spec.get("manifest_path_template", "")), scene_id=scene_id)
    payload = _load_json_repo(manifest, "E2 source identity manifest")
    if payload.get("scene_id") != scene_id:
        raise ValueError(f"E2 manifest scene mismatch for {scene_id}")
    gaussian_field = str(spec.get("manifest_fields", {}).get(
        "source_scene_gaussian", spec.get("manifest_field", "source_scene_gaussian")
    ))
    gaussian = _source_identity(payload.get(gaussian_field), f"{scene_id} Scene Gaussian")
    field_map = spec.get("manifest_fields", {})
    intrinsics_field = str(field_map.get("camera_intrinsics", "camera_artifacts.intrinsics"))
    poses_field = str(field_map.get("camera_poses", "camera_artifacts.poses"))

    def dotted(field: str) -> Any:
        value: Any = payload
        for component in field.split("."):
            if not isinstance(value, Mapping) or component not in value:
                raise ValueError(f"E2 manifest is missing {field}")
            value = value[component]
        return value

    cameras = {
        "intrinsics": _source_identity(dotted(intrinsics_field), f"{scene_id} intrinsics"),
        "poses": _source_identity(dotted(poses_field), f"{scene_id} poses"),
    }
    return {
        "manifest": _identity(manifest),
        "source_scene_gaussian": gaussian,
        "camera_artifacts": cameras,
    }, payload


def _accepted_report_rows(
    report: Mapping[str, Any], membership: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: Any = report
    for component in str(membership.get("object_rows_path", "objects")).split("."):
        if not isinstance(rows, Mapping) or component not in rows:
            raise ValueError(f"report is missing membership row path component {component}")
        rows = rows[component]
    if not isinstance(rows, list):
        raise ValueError("report membership rows must be a list")
    condition = membership.get("include_when", {})
    tiers = set(condition.get("tier_in", []))
    require_null = condition.get("rejected_is_null") is True
    accepted = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("report object row must be an object")
        index = row.get(str(membership.get("object_index_field", "index")))
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"invalid report object index: {index!r}")
        if index in seen:
            raise ValueError(f"duplicate report object index: {index}")
        seen.add(index)
        if row.get("tier") in tiers and (not require_null or row.get("rejected") is None):
            accepted.append(dict(row))
    return sorted(accepted, key=lambda row: row["index"])


def _manifest_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(
            rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def build_inventory_payloads(
    jobs: Mapping[str, Any],
    contract: Mapping[str, Any],
    freeze_id: str,
    *,
    expected_scenes: int | None = None,
    expected_jobs: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Resolve and validate the frozen pool without writing output.

    The optional expected counts make the resolver unit-testable on miniature
    fixtures.  Production callers always use the frozen values from YAML.
    """
    observation_protocol = _validated_observation_protocol(jobs)
    population = jobs["population"]
    scenes = _scene_roster(jobs)
    frozen_scene_count = int(population["planned_scenes"])
    if expected_scenes is None:
        expected_scenes = frozen_scene_count
    if expected_scenes != frozen_scene_count:
        raise ValueError("test expected_scenes may not override frozen production count")
    membership = population["membership"]
    pre_spec = population["prehybrid_cross_check"]
    proposal_spec = jobs["proposal_pool"]["proposals"]
    snapshots = {row["scene_id"]: row for row in population["scene_snapshots"]}
    if set(snapshots) != set(scenes):
        raise ValueError("scene_snapshots do not exactly cover the frozen scene roster")

    resolved_jobs: list[dict[str, Any]] = []
    resolved_proposals: list[dict[str, Any]] = []
    audit_scenes: list[dict[str, Any]] = []
    roster_rows: list[dict[str, Any]] = []
    mesh_manifests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    artifact_lines: list[tuple[str, str]] = []
    typed_unavailable: set[str] = set()
    source_rows: list[dict[str, Any]] = []
    evaluation_references: list[dict[str, Any]] = []
    all_pre_rows = 0

    for scene_id in scenes:
        snapshot = snapshots[scene_id]
        report_path = _format_template(membership["report_path_template"], scene_id=scene_id)
        report_identity = _identity(report_path)
        if report_identity["sha256"] != snapshot["report_sha256"]:
            raise ValueError(f"frozen report hash drift for {scene_id}")
        report = _load_json_repo(report_path, f"{scene_id} factory report")
        accepted = _accepted_report_rows(report, membership)
        if len(accepted) != int(snapshot["accepted_jobs"]):
            raise ValueError(f"accepted report count drift for {scene_id}")

        pre_path = _format_template(pre_spec["path_template"], scene_id=scene_id)
        pre_identity = _identity(pre_path)
        pre_rows = _load_json_repo(pre_path, f"{scene_id} pre-hybrid alignment audit")
        if not isinstance(pre_rows, list):
            raise ValueError(f"pre-hybrid cross-check for {scene_id} must be a list")
        all_pre_rows += len(pre_rows)
        pre_by_index: dict[int, Mapping[str, Any]] = {}
        for row in pre_rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("index"), int):
                raise ValueError(f"invalid pre-hybrid row for {scene_id}")
            if row["index"] in pre_by_index:
                raise ValueError(f"duplicate pre-hybrid object index for {scene_id}")
            pre_by_index[row["index"]] = row

        source, _ = _extract_e2_sources(jobs, scene_id)
        source_rows.append(
            {
                "scene_id": scene_id,
                "source_scene_gaussian": source["source_scene_gaussian"],
                "camera_intrinsics": source["camera_artifacts"]["intrinsics"],
                "camera_poses": source["camera_artifacts"]["poses"],
            }
        )
        scene_jobs = []
        scene_rvg_available = 0
        for report_row in accepted:
            index = int(report_row["index"])
            pre = pre_by_index.get(index)
            if pre is None:
                raise ValueError(f"accepted job absent from pre-hybrid audit: {scene_id}/{index}")
            if pre.get("tier") not in {"A", "B"} or pre.get("rejected") is not None:
                raise ValueError(f"report/pre-hybrid membership disagreement: {scene_id}/{index}")
            object_slot = f"obj_{index:02d}"
            job_id = f"{scene_id}/{object_slot}"
            meta_path = checked_repo_path(
                f"outputs/{scene_id}_factory/objects/{object_slot}/meta.json",
                "factory object metadata",
                kind="file",
            )
            rgba_path = checked_repo_path(
                f"outputs/{scene_id}_factory/objects/{object_slot}/rgba.png",
                "factory RGBA input",
                kind="file",
            )
            meta = _load_json_repo(meta_path, "factory object metadata")
            frame = meta.get("frame")
            bbox = meta.get("bbox_px")
            if not isinstance(frame, str) or Path(frame).name != frame:
                raise ValueError(f"invalid source frame for {job_id}")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(isinstance(v, bool) or not isinstance(v, int) for v in bbox)
                or not (bbox[0] < bbox[2] and bbox[1] < bbox[3])
            ):
                raise ValueError(f"invalid crop bbox for {job_id}")
            rgba_identity = _identity(rgba_path)
            meta_identity = _identity(meta_path)
            evaluation_path = checked_repo_path(
                f"outputs/{scene_id}_factory/objects/{object_slot}/gt_points.ply",
                "evaluation reference surface",
                kind="file",
            )
            evaluation_identity = _identity(evaluation_path)
            evaluation_references.append(
                {
                    "job_id": job_id,
                    "scene_id": scene_id,
                    "object_slot": object_slot,
                    **evaluation_identity,
                }
            )
            job = {
                "freeze_id": freeze_id,
                "job_id": job_id,
                "scene_id": scene_id,
                "object_slot": object_slot,
                "artifact_paths": {"rgba": rgba_identity["path"]},
                "artifact_hashes": {"rgba": rgba_identity["sha256"]},
                "construction_evidence": {
                    "source_frame": frame,
                    "bbox_px": bbox,
                    "legacy_mask_provenance": "opaque_fixed_input",
                },
            }
            _check_no_forbidden_fields(job, location=job_id)
            resolved_jobs.append(job)
            scene_jobs.append(job_id)

            rvg_complete = True
            for tool in ("trellis", "reconviagen"):
                tool_spec = proposal_spec[tool]
                mesh_candidate = checked_repo_path(
                    str(tool_spec["raw_mesh_path_template"]).format(
                        scene_id=scene_id, object_index=index
                    ),
                    "raw proposal mesh",
                    must_exist=False,
                )
                mesh_path = mesh_candidate if mesh_candidate.is_file() else None
                gs_formatted = str(tool_spec["raw_gaussian_path_template"]).format(
                    scene_id=scene_id, object_index=index
                )
                gs_candidate = checked_repo_path(
                    gs_formatted, "raw proposal Gaussian", must_exist=False
                )
                complete = mesh_path is not None and gs_candidate.is_file()
                proposal_id = f"{job_id}:{tool}"
                legacy_generation = jobs["proposal_pool"].get(
                    "legacy_generation_provenance", {}
                )
                if not isinstance(legacy_generation, Mapping):
                    raise ValueError("proposal_pool.legacy_generation_provenance must be an object")
                if (
                    legacy_generation.get("artifact_bytes_hash_frozen") is not True
                    or legacy_generation.get("generator_commit") is not None
                    or legacy_generation.get(
                        "current_controller_commit_may_backfill_generator_commit"
                    )
                    is not False
                ):
                    raise ValueError("legacy proposal provenance contract drift")
                proposal = {
                    "freeze_id": freeze_id,
                    "job_id": job_id,
                    "scene_id": scene_id,
                    "object_slot": object_slot,
                    "proposal_id": proposal_id,
                    "tool_id": tool,
                    "availability": "available" if complete else "typed_unavailable",
                    "artifact_paths": {},
                    "artifact_hashes": {},
                    "artifact_sizes": {},
                    "construction_evidence": {},
                    "raw_generator_provenance": {
                        "artifact_bytes_hash_frozen": True,
                        "generator_commit": None,
                        "checkpoint_identity_recorded": bool(
                            legacy_generation.get("original_checkpoint_identities_recorded")
                        ),
                        "runtime_manifest_recorded": bool(
                            legacy_generation.get("original_runtime_manifests_recorded")
                        ),
                        "claim_status": legacy_generation.get("claim_status"),
                    },
                }
                if complete:
                    mesh_identity = _identity(mesh_path)
                    gaussian_identity = _identity(gs_candidate)
                    proposal["artifact_paths"] = {
                        "raw_mesh": mesh_identity["path"],
                        "raw_gaussian": gaussian_identity["path"],
                    }
                    proposal["artifact_hashes"] = {
                        "raw_mesh": mesh_identity["sha256"],
                        "raw_gaussian": gaussian_identity["sha256"],
                    }
                    proposal["artifact_sizes"] = {
                        "raw_mesh": mesh_identity["size_bytes"],
                        "raw_gaussian": gaussian_identity["size_bytes"],
                    }
                    mesh_manifests[tool].append(
                        {
                            "job_id": job_id,
                            "tool": tool,
                            "path": mesh_identity["path"],
                            "size_bytes": mesh_identity["size_bytes"],
                            "sha256": mesh_identity["sha256"],
                        }
                    )
                    artifact_lines.extend(
                        [
                            (mesh_identity["sha256"], mesh_identity["path"]),
                            (gaussian_identity["sha256"], gaussian_identity["path"]),
                        ]
                    )
                    if tool == "reconviagen":
                        scene_rvg_available += 1
                else:
                    missing = []
                    if mesh_path is None:
                        missing.append("raw_mesh")
                    if not gs_candidate.is_file():
                        missing.append("raw_gaussian")
                    proposal["construction_evidence"] = {
                        "typed_failure": "required_raw_artifacts_missing",
                        "missing_artifacts": sorted(missing),
                    }
                    if tool == "trellis":
                        raise ValueError(f"required TRELLIS proposal is incomplete: {job_id}")
                    typed_unavailable.add(job_id)
                    rvg_complete = False
                _check_no_forbidden_fields(proposal, location=proposal_id)
                resolved_proposals.append(proposal)

            roster_rows.append(
                {
                    "scene_id": scene_id,
                    "object_index": index,
                    "gt_object_id": report_row.get("gt_object_id"),
                    "label": report_row.get("label"),
                    "rvg_complete": rvg_complete,
                }
            )
            artifact_lines.extend(
                [
                    (rgba_identity["sha256"], rgba_identity["path"]),
                    (meta_identity["sha256"], meta_identity["path"]),
                    (evaluation_identity["sha256"], evaluation_identity["path"]),
                ]
            )

        if scene_rvg_available != int(snapshot["reconviagen_available"]):
            raise ValueError(f"ReconViaGen count drift for {scene_id}")
        artifact_lines.extend(
            [
                (report_identity["sha256"], report_identity["path"]),
                (pre_identity["sha256"], pre_identity["path"]),
                (source["manifest"]["sha256"], source["manifest"]["path"]),
            ]
        )
        audit_scenes.append(
            {
                "scene_id": scene_id,
                "accepted_jobs": len(accepted),
                "reconviagen_available": scene_rvg_available,
                "report_identity": report_identity,
                "prehybrid_identity": pre_identity,
                "source_identity_manifest": source["manifest"],
                "source_scene_gaussian": source["source_scene_gaussian"],
                "camera_artifacts": source["camera_artifacts"],
                "job_ids": scene_jobs,
            }
        )

    expected_pre = int(pre_spec["expected_rows_all_scenes"])
    if all_pre_rows != expected_pre:
        raise ValueError(f"pre-hybrid row count drift: {all_pre_rows} != {expected_pre}")
    frozen_job_count = int(population["planned_jobs_per_policy"])
    if expected_jobs is None:
        expected_jobs = frozen_job_count
    if expected_jobs != frozen_job_count or len(resolved_jobs) != frozen_job_count:
        raise ValueError(
            f"resolved job count differs from frozen {frozen_job_count}: {len(resolved_jobs)}"
        )
    expected_typed = set(
        proposal_spec["reconviagen"].get("typed_unavailable_job_ids", [])
    )
    if typed_unavailable != expected_typed:
        raise ValueError(
            "typed ReconViaGen failures drifted; "
            f"missing={sorted(expected_typed - typed_unavailable)}, "
            f"extra={sorted(typed_unavailable - expected_typed)}"
        )
    roster_digest = _manifest_digest(roster_rows)
    if roster_digest != membership["resolved_roster_sha256"]:
        raise ValueError("resolved 457-object roster digest drift")

    for tool, rows in mesh_manifests.items():
        rows.sort(key=lambda row: row["job_id"])
        snapshot = jobs["proposal_pool"]["raw_mesh_snapshots"][tool]
        if len(rows) != int(snapshot["rows"]):
            raise ValueError(f"{tool} raw mesh row count drift")
        if sum(row["size_bytes"] for row in rows) != int(snapshot["total_size_bytes"]):
            raise ValueError(f"{tool} raw mesh total size drift")
        if _manifest_digest(rows) != snapshot["manifest_sha256"]:
            raise ValueError(f"{tool} raw mesh manifest digest drift")

    source_rows.sort(key=lambda row: row["scene_id"])
    combined_source_digest = _manifest_digest(source_rows)
    expected_source_digest = jobs["observation_source"].get(
        "canonical_identity_manifest_sha256"
    )
    if expected_source_digest and combined_source_digest != expected_source_digest:
        raise ValueError(
            "source Scene Gaussian/camera identity digest drift: "
            f"{combined_source_digest} != {expected_source_digest}"
        )

    config_identity = {
        "jobs_sha256": _canonical_digest(jobs),
        "jobs_hash_method": "canonical_structured_sha256",
        "contract_sha256": _canonical_digest(contract),
        "contract_hash_method": "canonical_json_sha256",
        "contract_freeze_id": contract.get("freeze_id"),
        "code_commit": contract.get("code", {}).get("commit", "unknown"),
    }
    resolved_jobs_payload = {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "study_scope": jobs["study_scope"],
        "counts": {
            "scenes": len(scenes),
            "jobs": len(resolved_jobs),
            "policy_object_rows": len(resolved_jobs) * len(POLICY_IDS),
        },
        "source_contract": config_identity,
        "observation_protocol": observation_protocol,
        "scenes": [
            {
                "scene_id": scene["scene_id"],
                "source_scene_gaussian": scene["source_scene_gaussian"],
                "camera_artifacts": scene["camera_artifacts"],
                "jobs": [job for job in resolved_jobs if job["scene_id"] == scene["scene_id"]],
            }
            for scene in audit_scenes
        ],
    }
    resolved_proposals_payload = {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "counts": {
            "jobs": len(resolved_jobs),
            "trellis_available": sum(
                p["availability"] == "available" and p["tool_id"] == "trellis"
                for p in resolved_proposals
            ),
            "reconviagen_available": sum(
                p["availability"] == "available" and p["tool_id"] == "reconviagen"
                for p in resolved_proposals
            ),
            "reconviagen_typed_unavailable": len(typed_unavailable),
        },
        "proposals": sorted(
            resolved_proposals, key=lambda row: (row["job_id"], row["tool_id"])
        ),
    }
    audit_payload = {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "created_utc": _utc_now(),
        "membership_authority": membership["authority"],
        "legacy_hybrid_winner_used": False,
        "counts": {
            "scenes": len(scenes),
            "report_objects_accepted": len(resolved_jobs),
            "prehybrid_objects_all": all_pre_rows,
            "trellis_available": resolved_proposals_payload["counts"]["trellis_available"],
            "reconviagen_available": resolved_proposals_payload["counts"][
                "reconviagen_available"
            ],
            "reconviagen_typed_unavailable": len(typed_unavailable),
        },
        "resolved_roster_sha256": roster_digest,
        "source_identity_manifest_sha256": combined_source_digest,
        # This list is deliberately absent from controller_members.  Only the
        # post-decision evaluation phase may open and revalidate these paths.
        "evaluation_references": sorted(
            evaluation_references, key=lambda row: row["job_id"]
        ),
        "scenes": audit_scenes,
    }
    artifact_text = "".join(
        f"{digest}  {path}\n" for digest, path in sorted(set(artifact_lines), key=lambda x: x[1])
    )
    return resolved_jobs_payload, resolved_proposals_payload, audit_payload, artifact_text


def run_inventory(
    jobs: Mapping[str, Any],
    contract: Mapping[str, Any],
    freeze_id: str,
    out_dir: Path,
) -> dict[str, Any]:
    if "automatic_sources" in jobs:
        from agents.orchestrator.automatic_inventory import build_payloads
        resolved_jobs, resolved_proposals, audit, hashes = build_payloads(jobs, contract, freeze_id)
    else:
        resolved_jobs, resolved_proposals, audit, hashes = build_inventory_payloads(
            jobs, contract, freeze_id
        )
    destination = out_dir / "input_inventory"
    with _atomic_directory(destination) as staging:
        jobs_path = staging / "resolved_jobs.json"
        proposals_path = staging / "resolved_proposals.json"
        audit_path = staging / "inventory_audit.json"
        hashes_path = staging / "artifact_hashes.sha256"
        _write_json_inside(jobs_path, resolved_jobs)
        _write_json_inside(proposals_path, resolved_proposals)
        _write_json_inside(audit_path, audit)
        _write_bytes_inside(hashes_path, hashes.encode("utf-8"))
        seal = {
            "schema_version": SCHEMA_VERSION,
            "members": {
                path.name: sha256_file(path)
                for path in (jobs_path, proposals_path, audit_path, hashes_path)
            },
            # Observe/control verify only these sanitized files.  They do not
            # even hash the audit/report provenance sidecars.
            "controller_members": {
                path.name: sha256_file(path) for path in (jobs_path, proposals_path)
            },
        }
        _write_json_inside(staging / "seal.json", seal)
    return audit


def _verify_sealed_directory(directory: Path, *, controller_safe: bool = False) -> dict[str, Any]:
    directory = checked_repo_path(directory, "sealed phase directory", kind="dir")
    loader = _load_controller_json if controller_safe else _load_json_repo
    seal = loader(directory / "seal.json", "phase seal")
    members = seal.get("controller_members" if controller_safe else "members")
    if not isinstance(members, Mapping) or not members:
        raise ValueError(f"invalid or empty phase seal: {directory}")
    for name, expected in members.items():
        if Path(name).name != name:
            raise ValueError(f"seal member must be a plain filename: {name}")
        path = _checked_controller_file(directory / name, "sealed member") if controller_safe \
            else checked_repo_path(directory / name, "sealed member", kind="file")
        if sha256_file(path) != _require_sha256(expected, f"seal digest for {name}"):
            raise ValueError(f"sealed member hash mismatch: {path}")
    return dict(seal)


def _exact_keys(value: Any, expected: set[str], identity: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"{identity} schema keys differ: expected={sorted(expected)}, "
            f"observed={observed}"
        )
    return value


def _unavailable_observation(job: Mapping[str, Any]) -> bool:
    return job["construction_evidence"].get("observation_status") == "unavailable"


def _missing_observation_record(job: Mapping[str, Any]) -> dict[str, Any]:
    """A typed absence, never an invented empty render or measured zero."""
    return {
        **{k: job[k] for k in ("freeze_id", "job_id", "scene_id", "object_slot")},
        "observation_status": "unavailable",
        "reason_code": job["construction_evidence"]["reason_code"],
        "observation_path": None, "observation_sha256": None,
        "registration_surface_hash": None, "source_frame": None,
        "rgba_sha256": None, "source_scene_gaussian_sha256": None,
        "observation_point_count": None, "observation_mask_fraction": None,
    }


def _validate_controller_inventory_schema(
    jobs: Mapping[str, Any], proposals: Mapping[str, Any]
) -> None:
    """Enforce the complete allowlist without reopening the GT-laden config."""
    _exact_keys(
        jobs,
        {
            "schema_version", "freeze_id", "study_scope", "counts",
            "source_contract", "observation_protocol", "scenes",
        },
        "sanitized job inventory",
    )
    _exact_keys(
        jobs["counts"], {"scenes", "jobs", "policy_object_rows"}, "job counts"
    )
    _exact_keys(
        jobs["source_contract"],
        {
            "jobs_sha256", "jobs_hash_method", "contract_sha256",
            "contract_hash_method", "contract_freeze_id", "code_commit",
        },
        "sanitized source contract",
    )
    if jobs.get("observation_protocol") != OBSERVATION_PROTOCOL:
        raise ValueError("sanitized observation protocol drift")
    version = jobs["schema_version"]
    if version not in (1, 2) or proposals.get("schema_version") != version:
        raise ValueError("unsupported or mixed sanitized inventory versions")
    job_ids: set[str] = set()
    unavailable_jobs: set[str] = set()
    job_bindings: dict[str, tuple[str, str]] = {}
    for scene in jobs["scenes"]:
        _exact_keys(
            scene,
            {"scene_id", "source_scene_gaussian", "camera_artifacts", "jobs"},
            "sanitized scene",
        )
        _exact_keys(
            scene["source_scene_gaussian"], {"path", "size_bytes", "sha256"},
            "sanitized Scene Gaussian identity",
        )
        _exact_keys(
            scene["camera_artifacts"], {"intrinsics", "poses"},
            "sanitized camera artifacts",
        )
        for identity in scene["camera_artifacts"].values():
            _exact_keys(
                identity, {"path", "size_bytes", "sha256"},
                "sanitized camera identity",
            )
        for job in scene["jobs"]:
            _exact_keys(
                job,
                {
                    "freeze_id", "job_id", "scene_id", "object_slot",
                    "artifact_paths", "artifact_hashes", "construction_evidence",
                },
                "sanitized object job",
            )
            evidence = job["construction_evidence"]
            if version == 1:
                expected_evidence = {"source_frame", "bbox_px", "legacy_mask_provenance"}
            else:
                expected_evidence = {"source_frame", "bbox_px", "mask_provenance",
                                     "observation_status", "reason_code"}
            _exact_keys(evidence, expected_evidence, "sanitized observation declaration")
            unavailable = version == 2 and _unavailable_observation(job)
            _exact_keys(job["artifact_paths"], set() if unavailable else {"rgba"}, "object artifact paths")
            _exact_keys(job["artifact_hashes"], set() if unavailable else {"rgba"}, "object artifact hashes")
            if version == 2:
                if unavailable:
                    if evidence["reason_code"] != "preparation_unavailable" or any(
                        evidence[k] is not None for k in ("source_frame", "bbox_px", "mask_provenance")
                    ):
                        raise ValueError("unavailable observation must contain typed absence only")
                    unavailable_jobs.add(job["job_id"])
                elif (evidence["observation_status"] != "available"
                      or evidence["reason_code"] is not None
                      or evidence["mask_provenance"] != "automatic_training_only"):
                    raise ValueError("automatic observation provenance differs")
            if job["scene_id"] != scene["scene_id"] or job["job_id"] in job_ids:
                raise ValueError("sanitized job scene binding or uniqueness failed")
            job_ids.add(job["job_id"])
            job_bindings[job["job_id"]] = (job["scene_id"], job["object_slot"])

    _exact_keys(
        proposals,
        {"schema_version", "freeze_id", "counts", "proposals"},
        "sanitized proposal inventory",
    )
    _exact_keys(
        proposals["counts"],
        {
            "jobs", "trellis_available", "reconviagen_available",
            "reconviagen_typed_unavailable",
        },
        "proposal counts",
    )
    seen_proposals: set[str] = set()
    proposal_jobs: Counter[str] = Counter()
    for proposal in proposals["proposals"]:
        _exact_keys(
            proposal,
            {
                "freeze_id", "job_id", "scene_id", "object_slot", "proposal_id",
                "tool_id", "availability", "artifact_paths", "artifact_hashes",
                "artifact_sizes", "construction_evidence", "raw_generator_provenance",
            },
            "sanitized proposal",
        )
        _exact_keys(
            proposal["raw_generator_provenance"],
            {
                "artifact_bytes_hash_frozen", "generator_commit",
                "checkpoint_identity_recorded", "runtime_manifest_recorded",
                "claim_status",
            },
            "raw generator provenance",
        )
        if proposal["availability"] == "available":
            _exact_keys(
                proposal["artifact_paths"], {"raw_mesh", "raw_gaussian"},
                "available proposal paths",
            )
            _exact_keys(
                proposal["artifact_hashes"], {"raw_mesh", "raw_gaussian"},
                "available proposal hashes",
            )
            _exact_keys(
                proposal["artifact_sizes"], {"raw_mesh", "raw_gaussian"},
                "available proposal sizes",
            )
            if any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in proposal["artifact_sizes"].values()
            ):
                raise ValueError("available proposal sizes must be positive integers")
            _exact_keys(proposal["construction_evidence"], set(), "proposal evidence")
        elif proposal["availability"] == "typed_unavailable":
            _exact_keys(proposal["artifact_paths"], set(), "unavailable proposal paths")
            _exact_keys(proposal["artifact_hashes"], set(), "unavailable proposal hashes")
            _exact_keys(proposal["artifact_sizes"], set(), "unavailable proposal sizes")
            _exact_keys(
                proposal["construction_evidence"],
                {"typed_failure", "missing_artifacts"},
                "unavailable proposal evidence",
            )
        else:
            raise ValueError(f"invalid proposal availability: {proposal['availability']!r}")
        if proposal["job_id"] in unavailable_jobs and proposal["availability"] != "typed_unavailable":
            raise ValueError("unavailable observation cannot have an available generation proposal")
        if proposal["job_id"] not in job_ids:
            raise ValueError("proposal refers to an unknown sanitized job")
        if proposal["proposal_id"] in seen_proposals:
            raise ValueError("duplicate sanitized proposal ID")
        seen_proposals.add(proposal["proposal_id"])
        proposal_jobs[proposal["job_id"]] += 1
    if set(proposal_jobs) != job_ids or any(count != 2 for count in proposal_jobs.values()):
        raise ValueError("sanitized proposal inventory must contain two rows per object job")
    if version == 2:
        if jobs["counts"] != {"scenes": len(jobs["scenes"]), "jobs": len(job_ids), "policy_object_rows": 5 * len(job_ids)}:
            raise ValueError("automatic planned-job denominator differs")
        pairs = Counter((row["job_id"], row["tool_id"]) for row in proposals["proposals"])
        if set(pairs) != {(job, tool) for job in job_ids for tool in ("trellis", "reconviagen")} or any(v != 1 for v in pairs.values()):
            raise ValueError("automatic initial tools must be paired exactly once per job")
        for row in proposals["proposals"]:
            if job_bindings[row["job_id"]] != (row["scene_id"], row["object_slot"]):
                raise ValueError("automatic proposal scene/object binding differs")
        expected_counts = {"jobs": len(job_ids),
            "trellis_available": sum(r["tool_id"] == "trellis" and r["availability"] == "available" for r in proposals["proposals"]),
            "reconviagen_available": sum(r["tool_id"] == "reconviagen" and r["availability"] == "available" for r in proposals["proposals"]),
            "reconviagen_typed_unavailable": sum(r["tool_id"] == "reconviagen" and r["availability"] == "typed_unavailable" for r in proposals["proposals"])}
        if proposals["counts"] != expected_counts:
            raise ValueError("automatic proposal availability counts differ")


def _load_inventory(out_dir: Path, *, controller_safe: bool = False):
    directory = out_dir / "input_inventory"
    _verify_sealed_directory(directory, controller_safe=controller_safe)
    loader = _load_controller_json if controller_safe else _load_json_repo
    jobs = loader(directory / "resolved_jobs.json", "resolved job inventory")
    proposals = loader(directory / "resolved_proposals.json", "resolved proposal inventory")
    if jobs.get("freeze_id") != proposals.get("freeze_id"):
        raise ValueError("resolved job/proposal inventories have different freeze IDs")
    if controller_safe:
        _validate_controller_inventory_schema(jobs, proposals)
    return jobs, proposals


def _scene_inventory(resolved_jobs: Mapping[str, Any], scene_id: str) -> Mapping[str, Any]:
    matches = [scene for scene in resolved_jobs.get("scenes", []) if scene.get("scene_id") == scene_id]
    if len(matches) != 1:
        raise ValueError(f"resolved inventory has {len(matches)} entries for scene {scene_id}")
    return matches[0]


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Write a byte-stable NPZ (NumPy's default embeds wall-clock ZIP times)."""
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(
        archive_bytes, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            npy = io.BytesIO()
            np.lib.format.write_array(npy, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return archive_bytes.getvalue()


def visible_observation_points(
    depth: np.ndarray,
    gaussian_alpha: np.ndarray,
    rgba_alpha: np.ndarray,
    bbox_px: Sequence[int],
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    rgba_alpha_min: int = RGBA_ALPHA_MIN,
    gaussian_alpha_min: float = OBSERVATION_ALPHA_MIN,
    depth_min_m: float = DEPTH_MIN_M,
    depth_max_m: float = DEPTH_MAX_M,
    pixel_center_offset: float = PIXEL_CENTER_OFFSET,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unproject a frozen crop mask against Scene-Gaussian expected depth."""
    depth = np.asarray(depth, dtype=np.float64)
    gaussian_alpha = np.asarray(gaussian_alpha, dtype=np.float64)
    rgba_alpha = np.asarray(rgba_alpha)
    if depth.ndim != 2 or gaussian_alpha.shape != depth.shape:
        raise ValueError("depth and Gaussian alpha must be equal HxW arrays")
    if rgba_alpha.ndim != 2:
        raise ValueError("RGBA alpha must be a 2-D crop")
    if len(bbox_px) != 4:
        raise ValueError("bbox_px must contain four integers")
    u0, v0, u1, v1 = map(int, bbox_px)
    if not (0 <= u0 < u1 <= depth.shape[1] and 0 <= v0 < v1 <= depth.shape[0]):
        raise ValueError("crop bbox is outside the rendered frame")
    if rgba_alpha.shape != (v1 - v0, u1 - u0):
        raise ValueError(
            f"RGBA alpha shape {rgba_alpha.shape} differs from bbox {(v1-v0, u1-u0)}"
        )
    crop_depth = depth[v0:v1, u0:u1]
    crop_gaussian_alpha = gaussian_alpha[v0:v1, u0:u1]
    if not isinstance(rgba_alpha_min, int) or not 0 <= rgba_alpha_min <= 255:
        raise ValueError("rgba_alpha_min must be an integer in [0,255]")
    if not math.isfinite(float(pixel_center_offset)):
        raise ValueError("pixel_center_offset must be finite")
    crop_mask = rgba_alpha >= rgba_alpha_min
    valid = (
        crop_mask
        & np.isfinite(crop_depth)
        & (crop_depth >= depth_min_m)
        & (crop_depth <= depth_max_m)
        & np.isfinite(crop_gaussian_alpha)
        & (crop_gaussian_alpha >= gaussian_alpha_min)
    )
    vv, uu = np.nonzero(valid)
    z = crop_depth[vv, uu]
    K = np.asarray(intrinsics, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if K.shape != (3, 3) or c2w.shape != (4, 4):
        raise ValueError("intrinsics/camera_to_world have invalid shape")
    u_full = uu.astype(np.float64) + u0 + pixel_center_offset
    v_full = vv.astype(np.float64) + v0 + pixel_center_offset
    x = (u_full - K[0, 2]) / K[0, 0] * z
    y = (v_full - K[1, 2]) / K[1, 1] * z
    camera_points = np.stack([x, y, z], axis=1)
    points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
    mask_count = int(crop_mask.sum())
    stats = {
        "observation_point_count": int(len(points)),
        "mask_pixel_count": mask_count,
        "crop_pixel_count": int(crop_mask.size),
        "observation_mask_fraction": float(len(points) / max(mask_count, 1)),
        "rgba_alpha_min_inclusive_uint8": int(rgba_alpha_min),
        "gaussian_alpha_min": float(gaussian_alpha_min),
        "depth_range_m": [float(depth_min_m), float(depth_max_m)],
        "pixel_center_offset": float(pixel_center_offset),
    }
    return points.astype(np.float32), stats


def run_observe(freeze_id: str, out_dir: Path, scene_id: str) -> dict[str, Any]:
    scene_id = _require_scene_id(scene_id)
    resolved_jobs, _ = _load_inventory(out_dir, controller_safe=True)
    if resolved_jobs.get("freeze_id") != freeze_id:
        raise ValueError("inventory freeze_id differs from requested freeze")
    protocol = resolved_jobs.get("observation_protocol")
    if protocol != OBSERVATION_PROTOCOL:
        raise ValueError("sealed observation protocol differs from the implementation")
    scene = _scene_inventory(resolved_jobs, scene_id)
    destination = out_dir / "observations" / scene_id
    with _atomic_directory(destination) as staging:
        records = [_missing_observation_record(job) for job in scene["jobs"]
                   if _unavailable_observation(job)]
        observed_jobs = [job for job in scene["jobs"] if not _unavailable_observation(job)]
        renderer_runtime: dict[str, Any] | None = None
        if observed_jobs:
            # Observe runs in the gsplat environment.  Heavy dependencies stay
            # phase-local so --smoke works in the same environment and control
            # stays in the main Open3D/PyBullet environment.
            from PIL import Image
            from agents.core import common as common
            from importlib import metadata
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("E3 observation requires exactly one CUDA device")
            renderer_runtime = {
                "python_executable": sys.executable,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "gsplat_version": metadata.version("gsplat"),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_compute_capability": ".".join(
                    str(value) for value in torch.cuda.get_device_capability(0)
                ),
            }

            gaussian_path = _checked_external_identity(
                scene["source_scene_gaussian"], f"{scene_id} Scene Gaussian"
            )
            intrinsics_path = _checked_external_identity(
                scene["camera_artifacts"]["intrinsics"], f"{scene_id} intrinsics"
            )
            poses_path = _checked_external_identity(
                scene["camera_artifacts"]["poses"], f"{scene_id} camera poses"
            )
            K, width, height, _ = common.load_intrinsics(intrinsics_path)
            poses = common.load_colmap_w2c(poses_path)
            gaussians = common.load_gaussians(gaussian_path, device="cuda")
            render_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for job in observed_jobs:
                job_id = job["job_id"]
                frame = job["construction_evidence"]["source_frame"]
                if frame not in poses:
                    raise ValueError(f"source frame {frame} is absent from frozen poses")
                if frame not in render_cache:
                    _, depth, alpha = common.render_view(
                        gaussians,
                        poses[frame],
                        K,
                        width,
                        height,
                        render_mode=protocol["render_mode"],
                        scale=1.0,
                    )
                    render_cache[frame] = (depth, alpha)
                rgba_path = _checked_controller_file(
                    job["artifact_paths"]["rgba"], "frozen RGBA observation input"
                )
                if sha256_file(rgba_path) != job["artifact_hashes"]["rgba"]:
                    raise ValueError(f"RGBA input hash drift for {job_id}")
                with Image.open(rgba_path) as image:
                    rgba = np.asarray(image.convert("RGBA"))
                depth, gaussian_alpha = render_cache[frame]
                points, stats = visible_observation_points(
                    depth,
                    gaussian_alpha,
                    rgba[..., 3],
                    job["construction_evidence"]["bbox_px"],
                    K,
                    np.linalg.inv(poses[frame]),
                    rgba_alpha_min=protocol["rgba_alpha_threshold_inclusive_uint8"],
                    gaussian_alpha_min=protocol["gaussian_alpha_threshold_inclusive"],
                    depth_min_m=protocol["depth_range_m_inclusive"][0],
                    depth_max_m=protocol["depth_range_m_inclusive"][1],
                    pixel_center_offset=protocol["pixel_center_offset"],
                )
                relative = Path("points") / f"{job['object_slot']}.npz"
                payload = _deterministic_npz(
                    {
                        "points": points,
                        "bbox_px": np.asarray(
                            job["construction_evidence"]["bbox_px"], dtype=np.int32
                        ),
                    }
                )
                path = staging / relative
                _write_bytes_inside(path, payload)
                records.append(
                    {
                        "freeze_id": freeze_id,
                        "job_id": job_id,
                        "scene_id": scene_id,
                        "object_slot": job["object_slot"],
                        "observation_path": (
                            destination / relative
                        ).relative_to(REPOSITORY_ROOT).as_posix(),
                        "observation_sha256": _sha256_bytes(payload),
                        "registration_surface_hash": _sha256_bytes(payload),
                        "source_frame": frame,
                        "rgba_sha256": job["artifact_hashes"]["rgba"],
                        "source_scene_gaussian_sha256": scene[
                            "source_scene_gaussian"
                        ]["sha256"],
                        **stats,
                    }
                )
        order = {job["job_id"]: i for i, job in enumerate(scene["jobs"])}
        records.sort(key=lambda row: order[row["job_id"]])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "freeze_id": freeze_id,
            "scene_id": scene_id,
            "created_utc": _utc_now(),
            "geometry_source": "source_scene_gaussian_expected_depth",
            "evaluation_geometry_read": False,
            "observation_protocol": protocol,
            "code_commit": resolved_jobs.get("source_contract", {}).get("code_commit"),
            "source_scene_gaussian": scene["source_scene_gaussian"],
            "camera_artifacts": scene["camera_artifacts"],
            "renderer_runtime": renderer_runtime,
            "records": records,
        }
        manifest_path = staging / "manifest.json"
        _write_json_inside(manifest_path, manifest)
        _write_json_inside(
            staging / "seal.json",
            {
                "schema_version": SCHEMA_VERSION,
                "members": {"manifest.json": sha256_file(manifest_path)},
                "controller_members": {
                    "manifest.json": sha256_file(manifest_path)
                },
                "observation_members": {
                    Path(row["observation_path"]).name: row["observation_sha256"]
                    for row in records if row.get("observation_status") != "unavailable"
                },
            },
        )
    return manifest


def _load_observation_scene(out_dir: Path, scene_id: str):
    directory = out_dir / "observations" / scene_id
    seal = _verify_sealed_directory(directory, controller_safe=True)
    manifest = _load_controller_json(directory / "manifest.json", "observation manifest")
    if manifest.get("scene_id") != scene_id or manifest.get("evaluation_geometry_read") is not False:
        raise ValueError(f"invalid observation manifest for {scene_id}")
    records = {row["job_id"]: row for row in manifest.get("records", [])}
    if len(records) != len(manifest.get("records", [])):
        raise ValueError("duplicate observation job IDs")
    declared_observations = seal.get("observation_members")
    rendered = [row for row in records.values() if row.get("observation_status") != "unavailable"]
    for row in records.values():
        if row.get("observation_status") == "unavailable":
            expected = _missing_observation_record({**row, "construction_evidence": {
                "reason_code": "preparation_unavailable"}})
            if row != expected:
                raise ValueError("unavailable observation contains fabricated measured evidence")
    actual_names = {Path(row["observation_path"]).name for row in rendered}
    if not isinstance(declared_observations, Mapping) or set(declared_observations) != actual_names:
        raise ValueError("observation seal does not exactly cover manifest point clouds")
    for row in rendered:
        path = _checked_controller_file(row["observation_path"], "observation point cloud")
        expected = declared_observations.get(path.name)
        if expected != row["observation_sha256"] or sha256_file(path) != expected:
            raise ValueError(f"observation hash mismatch: {path}")
    return directory, manifest, records


def _load_mesh(path: Path):
    import trimesh

    geometry = trimesh.load(path, process=False)
    if isinstance(geometry, trimesh.Scene):
        geometries = [g for g in geometry.geometry.values() if len(g.vertices)]
        if not geometries:
            raise ValueError(f"mesh scene contains no geometry: {path}")
        geometry = trimesh.util.concatenate(geometries)
    if not isinstance(geometry, trimesh.Trimesh) or len(geometry.vertices) < 4:
        raise ValueError(f"proposal is not a usable triangle mesh: {path}")
    if len(geometry.faces) < 4 or not np.isfinite(geometry.vertices).all():
        raise ValueError(f"proposal mesh is degenerate: {path}")
    return geometry


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _sample_mesh(mesh, count: int, seed: int) -> np.ndarray:
    import trimesh

    points, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError("deterministic mesh surface sampling failed")
    return points


def _recorded_generator_identity(proposal: Mapping[str, Any], final_scene: Path) -> dict[str, Any]:
    """Propagate only generator metadata authenticated by the raw inventory.

    Legacy direct callers without recorded generator provenance retain their
    explicit unknown status. The registration executor is never substituted.
    """
    provenance = proposal.get("raw_generator_provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("recorded generator provenance must be an object")
    if provenance is None or provenance.get("generator_commit") is None:
        return {"raw_generator_commit": "legacy-unrecorded"}
    commit = provenance["generator_commit"]
    if (not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit)
            or any(provenance.get(key) is not True for key in (
                "artifact_bytes_hash_frozen", "checkpoint_identity_recorded", "runtime_manifest_recorded"))):
        raise ValueError("recorded generator provenance is incomplete or malformed")
    # final_scene is the canonical out/control/scene destination. Resolve from
    # it instead of accepting a caller-selected alternative metadata source.
    if final_scene.parent.name != "control" or final_scene.name != proposal["scene_id"]:
        raise ValueError("recorded generator requires canonical scene destination")
    inventory_root = final_scene.parent.parent
    seal = _load_controller_json(
        inventory_root / "input_inventory" / "seal.json", "raw generator inventory seal"
    )
    # The generic phase verifier authenticates only the members it is given.
    # Attribution requires both sanitized files: omitting either must not turn
    # an edited, unsealed proposal or job binding into recorded provenance.
    members = seal.get("controller_members")
    if not isinstance(members, Mapping) or set(members) != {
        "resolved_jobs.json", "resolved_proposals.json"
    }:
        raise ValueError("raw generator seal must cover exactly both sanitized inventory members")
    jobs, inventory = _load_inventory(inventory_root, controller_safe=True)
    if jobs["freeze_id"] != proposal["freeze_id"]:
        raise ValueError("raw generator inventory freeze differs")
    matches = [row for row in inventory["proposals"]
               if row["proposal_id"] == proposal["proposal_id"]]
    if len(matches) != 1 or matches[0] != proposal:
        raise ValueError("candidate generator metadata differs from sealed raw source")
    return {
        "raw_generator_commit": commit,
        "raw_generator_provenance_sha256": _canonical_digest(provenance),
        "raw_generator_inventory_sha256": sha256_file(
            inventory_root / "input_inventory" / "resolved_proposals.json"
        ),
    }


def _process_candidate(
    proposal: Mapping[str, Any],
    observation_record: Mapping[str, Any],
    observation_points: np.ndarray,
    staging_scene: Path,
    final_scene: Path,
    executor_commit: str,
    *,
    retry: bool = False,
    parent_proposal_id: str | None = None,
) -> dict[str, Any]:
    generator_identity = _recorded_generator_identity(proposal, final_scene)
    # Heavy Open3D/PyBullet imports are intentionally control-local.  This is
    # what keeps --observe/--smoke importable in the lean gsplat environment.
    from agents.orchestrator.runtime import align_and_probe

    started = _utc_now()
    start = time.perf_counter()
    mesh_path = _checked_controller_file(
        proposal["artifact_paths"]["raw_mesh"], "raw proposal mesh"
    )
    if sha256_file(mesh_path) != proposal["artifact_hashes"]["raw_mesh"]:
        raise ValueError(f"raw proposal mesh hash drift: {mesh_path}")
    if mesh_path.stat().st_size != proposal["artifact_sizes"]["raw_mesh"]:
        raise ValueError(f"raw proposal mesh size drift: {mesh_path}")
    gaussian_path = _checked_controller_file(
        proposal["artifact_paths"]["raw_gaussian"], "raw proposal Gaussian"
    )
    if sha256_file(gaussian_path) != proposal["artifact_hashes"]["raw_gaussian"]:
        raise ValueError(f"raw proposal Gaussian hash drift: {gaussian_path}")
    if gaussian_path.stat().st_size != proposal["artifact_sizes"]["raw_gaussian"]:
        raise ValueError(f"raw proposal Gaussian size drift: {gaussian_path}")
    target = observation_points[:: max(len(observation_points) // 6000, 1)]
    if len(target) < 50:
        raise ValueError("visible observation contains fewer than 50 points")
    proposal_id = (
        f"{proposal['job_id']}:retry:signed-source-up"
        if retry
        else proposal["proposal_id"]
    )
    tool = "registration_retry" if retry else proposal["tool_id"]
    safe_name = proposal_id.replace("/", "_").replace(":", "_")
    relative = Path("proposals") / safe_name
    artifact_dir = staging_scene / relative
    try:
        runtime = align_and_probe(
            mesh_path,
            target,
            # The controller intentionally does not receive legacy GT labels.
            label="object",
            visible_fraction=float(observation_record["observation_mask_fraction"]),
            out_dir=artifact_dir,
            signed_source_up=retry,
            producer_commit=executor_commit,
            input_hashes={
                "raw_mesh": proposal["artifact_hashes"]["raw_mesh"],
                "raw_gaussian": proposal["artifact_hashes"]["raw_gaussian"],
                "visible_observation": observation_record["observation_sha256"],
            },
        )
        declared_runtime_files = set(runtime.get("artifact_files", []))
        if declared_runtime_files != set(RUNTIME_ARTIFACT_ROLES):
            raise ValueError("runtime artifact closure differs from the frozen contract")
        evidence_manifest = _load_json_repo(
            artifact_dir / "evidence.json", "proposal evidence manifest"
        )
        if set(evidence_manifest) != set(EVIDENCE_MANIFEST_FIELDS):
            raise ValueError("proposal evidence-manifest schema differs from frozen contract")
        expected_input_hashes = {
            "raw_mesh": proposal["artifact_hashes"]["raw_mesh"],
            "raw_gaussian": proposal["artifact_hashes"]["raw_gaussian"],
            "visible_observation": observation_record["observation_sha256"],
        }
        if (
            evidence_manifest["producer"]
            != "agents.orchestrator.runtime.align_and_probe"
            or evidence_manifest["producer_commit"] != executor_commit
            or evidence_manifest["input_hashes"] != expected_input_hashes
            or evidence_manifest["raw_values"] != runtime.get("evidence")
            or evidence_manifest["missing_flags"]
            != runtime.get("evidence", {}).get("missing_evidence")
            or evidence_manifest["wall_s"] != runtime.get("wall_s")
        ):
            raise ValueError("proposal evidence manifest differs from runtime result")
        transform = np.asarray(runtime["alignment"]["T"], dtype=np.float64)
        source_up = runtime["alignment"]["source_up_hypothesis"]
        transform_payload = io.BytesIO()
        np.lib.format.write_array(
            transform_payload, np.asarray(transform), allow_pickle=False
        )
        _write_bytes_inside(artifact_dir / "transform.npy", transform_payload.getvalue())
        evidence = dict(runtime["evidence"])
        if retry:
            _write_json_inside(
                artifact_dir / "retry.json",
                {
                    "proposal_id": proposal_id,
                    "parent_proposal_id": parent_proposal_id,
                    "action": "registration_signed_source_up_restart",
                    "source_up_hypothesis": source_up,
                },
            )
        artifact_paths = {
            "raw_mesh": _display_path(mesh_path),
            "raw_gaussian": _display_path(gaussian_path),
            "transform": _display_path(final_scene / relative / "transform.npy"),
        }
        artifact_paths.update(
            {
                role: _display_path(final_scene / relative / runtime_relative)
                for runtime_relative, role in RUNTIME_ARTIFACT_ROLES.items()
            }
        )
        if retry:
            artifact_paths["retry"] = _display_path(
                final_scene / relative / "retry.json"
            )
        artifact_hashes = {}
        artifact_sizes = {}
        for key, value in artifact_paths.items():
            if key in {"raw_mesh", "raw_gaussian"}:
                source = mesh_path if key == "raw_mesh" else gaussian_path
            else:
                final_artifact = _lexical_absolute(value)
                source = staging_scene / final_artifact.relative_to(final_scene)
            artifact_hashes[key] = sha256_file(source)
            artifact_sizes[key] = source.stat().st_size
        finished = _utc_now()
        record = ProposalRecord(
            freeze_id=proposal["freeze_id"],
            scene_id=proposal["scene_id"],
            object_id=proposal["object_slot"],
            job_id=proposal["job_id"],
            proposal_id=proposal_id,
            tool=tool,
            # Retry is a new registration action; its raw generator remains
            # the independently recorded parent generator.
            tool_commit=executor_commit if retry else generator_identity["raw_generator_commit"],
            input_hashes={
                "raw_mesh": proposal["artifact_hashes"]["raw_mesh"],
                "raw_gaussian": proposal["artifact_hashes"]["raw_gaussian"],
                "visible_observation": observation_record["observation_sha256"],
            },
            artifact_paths=artifact_paths,
            evidence=evidence,
            parent_proposal_ids=(parent_proposal_id,) if parent_proposal_id else (),
            started_utc=started,
            finished_utc=finished,
            wall_s=float(runtime["wall_s"]),
        )
        payload = record.as_dict()
        payload["artifact_hashes"] = artifact_hashes
        payload["artifact_sizes"] = artifact_sizes
        payload["source_up_hypothesis"] = source_up
        payload.update(generator_identity)
        payload["registration_executor_commit"] = executor_commit
        payload["proposal_digest"] = proposal_digest(record)
        return payload
    except Exception:
        if artifact_dir.exists() and not artifact_dir.is_symlink():
            shutil.rmtree(artifact_dir)
        proposals_dir = artifact_dir.parent
        if proposals_dir.is_dir() and not any(proposals_dir.iterdir()):
            proposals_dir.rmdir()
        raise


def _proposal_index(resolved_proposals: Mapping[str, Any], scene_id: str):
    rows = [row for row in resolved_proposals.get("proposals", []) if row["scene_id"] == scene_id]
    by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_job[row["job_id"]].append(row)
    for values in by_job.values():
        values.sort(key=lambda row: (0 if row["tool_id"] == "trellis" else 1, row["tool_id"]))
    return by_job


def _write_controller_shard(
    staging: Path,
    final: Path,
    shard: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
    selected: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(selected) != set(POLICY_IDS):
        raise ValueError("selected-asset map must exactly cover A0--A4")
    selected_hashes: dict[str, str] = {}
    for policy_id in POLICY_IDS:
        by_object = selected[policy_id]
        # Materialize empty policy directories as part of the handoff.  Some
        # frozen scenes have zero accepted object jobs, but aggregate/E4 still
        # require an explicit per-scene A0--A4 directory product.
        (staging / "selected_assets" / policy_id).mkdir(parents=True)
        for object_slot, payload in by_object.items():
            relative = Path("selected_assets") / policy_id / f"{object_slot}.json"
            path = staging / relative
            _write_json_inside(path, payload)
            selected_hashes[relative.as_posix()] = sha256_file(path)
    ledger_text = "".join(canonical_json(row) + "\n" for row in ledger_rows)
    ledger_path = staging / "job_ledger.jsonl"
    _write_bytes_inside(ledger_path, ledger_text.encode("utf-8"))
    shard["ledger_sha256"] = sha256_file(ledger_path)
    shard["selected_asset_hashes"] = selected_hashes
    shard_path = staging / "controller_shard.json"
    _write_json_inside(shard_path, shard)
    _write_json_inside(
        staging / "seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "members": {
                "controller_shard.json": sha256_file(shard_path),
                "job_ledger.jsonl": sha256_file(ledger_path),
            },
            "selected_asset_hashes": selected_hashes,
            "selected_asset_sizes": {
                relative: (staging / relative).stat().st_size
                for relative in sorted(selected_hashes)
            },
        },
    )


def run_control(
    policies: Mapping[str, Any],
    contract: Mapping[str, Any],
    freeze_id: str,
    out_dir: Path,
    scene_id: str,
) -> dict[str, Any]:
    scene_id = _require_scene_id(scene_id)
    validate_policy_config(policies)
    _validate_probe_protocol(policies)
    resolved_jobs, resolved_proposals = _load_inventory(out_dir, controller_safe=True)
    scene = _scene_inventory(resolved_jobs, scene_id)
    _, observation_manifest, observations = _load_observation_scene(out_dir, scene_id)
    if set(observations) != {job["job_id"] for job in scene["jobs"]}:
        raise ValueError("observation jobs differ from frozen scene jobs")
    proposal_index = _proposal_index(resolved_proposals, scene_id)
    destination = out_dir / "control" / scene_id
    code_commit = str(contract.get("code", {}).get("commit") or "unknown")
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("E0 contract lacks an exact code commit")

    with _atomic_directory(destination) as staging:
        proposal_records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        ledger: list[dict[str, Any]] = []
        selected: dict[str, dict[str, Any]] = {policy: {} for policy in POLICY_IDS}
        final_scene = destination
        for job in scene["jobs"]:
            job_id = job["job_id"]
            observation = observations[job_id]
            unavailable = _unavailable_observation(job)
            if unavailable != (observation.get("observation_status") == "unavailable"):
                raise ValueError("observation availability differs from frozen job")
            if unavailable:
                points = None  # No observation was produced or measured.
            else:
                observation_path = _checked_controller_file(
                    observation["observation_path"], "visible observation"
                )
                with np.load(observation_path, allow_pickle=False) as archive:
                    points = np.asarray(archive["points"], dtype=np.float64)
            initial: list[dict[str, Any]] = []
            frozen_proposals = proposal_index.get(job_id, [])
            for proposal in frozen_proposals:
                if proposal["availability"] != "available":
                    failures.append(
                        {
                            "job_id": job_id,
                            "proposal_id": proposal["proposal_id"],
                            "failure_type": "typed_unavailable",
                            "detail": proposal["construction_evidence"],
                        }
                    )
                    continue
                attempt_started_utc = _utc_now()
                attempt_started = time.perf_counter()
                try:
                    result = _process_candidate(
                        proposal,
                        observation,
                        points,
                        staging,
                        final_scene,
                        code_commit,
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "job_id": job_id,
                            "proposal_id": proposal["proposal_id"],
                            "tool": proposal["tool_id"],
                            "failure_type": "tool_crash",
                            "detail": f"{type(exc).__name__}: {exc}",
                            "started_utc": attempt_started_utc,
                            "finished_utc": _utc_now(),
                            "wall_s": float(time.perf_counter() - attempt_started),
                        }
                    )
                    continue
                initial.append(result)
                proposal_records.append(result)

            retry_candidate = None
            retry_required, retry_parent, retry_reasons = plan_retry(initial, policies)
            retry_attempted = bool(retry_required and retry_parent is not None)
            if retry_required and retry_parent is not None:
                retry_output = staging / "proposals" / (
                    f"{job_id}:retry:signed-source-up".replace("/", "_").replace(":", "_")
                )
                parent = next(row for row in initial if row["proposal_id"] == retry_parent)
                frozen_parent = next(
                    row for row in frozen_proposals if row["proposal_id"] == retry_parent
                )
                attempt_started_utc = _utc_now()
                attempt_started = time.perf_counter()
                try:
                    retry_candidate = _process_candidate(
                        frozen_parent,
                        observation,
                        points,
                        staging,
                        final_scene,
                        code_commit,
                        retry=True,
                        parent_proposal_id=retry_parent,
                    )
                    _require_sha256(
                        retry_candidate["artifact_hashes"].get("retry"),
                        "retry action artifact hash",
                    )
                    if retry_candidate.get("source_up_hypothesis") == "+z":
                        raise ValueError(
                            "signed-source-up retry selected the already-used +z hypothesis"
                        )
                    if retry_candidate["artifact_hashes"]["transform"] == parent[
                        "artifact_hashes"
                    ]["transform"]:
                        raise ValueError("retry transform artifact hash equals its parent")
                    if retry_candidate["proposal_digest"] == parent["proposal_digest"]:
                        raise ValueError("retry proposal digest equals its parent")
                    proposal_records.append(retry_candidate)
                except Exception as exc:
                    if retry_output.exists() and not retry_output.is_symlink():
                        shutil.rmtree(retry_output)
                    failures.append(
                        {
                            "job_id": job_id,
                            "proposal_id": f"{job_id}:retry:signed-source-up",
                            "tool": "registration_retry",
                            "failure_type": "retry_not_produced",
                            "detail": f"{type(exc).__name__}: {exc}",
                            "started_utc": attempt_started_utc,
                            "finished_utc": _utc_now(),
                            "wall_s": float(time.perf_counter() - attempt_started),
                            "attempted_artifact_hashes": (
                                dict(retry_candidate.get("artifact_hashes", {}))
                                if isinstance(retry_candidate, Mapping)
                                else {}
                            ),
                        }
                    )
                    retry_candidate = None

            controller = run_policies(
                initial, policies, retry_candidate=retry_candidate
            )
            by_id = {row["proposal_id"]: row for row in initial}
            if retry_candidate is not None:
                by_id[retry_candidate["proposal_id"]] = retry_candidate
            for outcome in controller.outcomes:
                selected_record = by_id.get(outcome.selected_proposal_id)
                terminal_proposal_id = outcome.selected_proposal_id or (
                    f"{job_id}:terminal:{outcome.policy_id}:{outcome.terminal_action}"
                )
                reason_codes = list(outcome.reason_codes)
                if (
                    retry_attempted
                    and retry_candidate is None
                    and outcome.policy_id in {"A3", "A4"}
                ):
                    reason_codes = [
                        value for value in reason_codes if value != "retry_not_triggered"
                    ]
                    if "bounded_retry_attempt_failed" not in reason_codes:
                        reason_codes.append("bounded_retry_attempt_failed")
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "freeze_id": freeze_id,
                    "scene_id": scene_id,
                    "object_slot": job["object_slot"],
                    "job_id": job_id,
                    "policy_id": outcome.policy_id,
                    "proposal_id": terminal_proposal_id,
                    "selected_proposal_id": outcome.selected_proposal_id,
                    "terminal_action": outcome.terminal_action,
                    "support_label": outcome.support_label,
                    "retry_attempted": bool(
                        retry_attempted and outcome.policy_id in {"A3", "A4"}
                    ),
                    "retry_produced": bool(
                        retry_candidate is not None and outcome.policy_id in {"A3", "A4"}
                    ),
                    "retry_invoked": bool(
                        retry_attempted and outcome.policy_id in {"A3", "A4"}
                    ),
                    "retry_required": controller.retry_required,
                    "retry_reason_codes": list(retry_reasons),
                    "reason_codes": reason_codes,
                }
                ledger.append(row)
                selected_payload = {
                    **row,
                    "selected_asset": (
                        {
                            "proposal_digest": selected_record["proposal_digest"],
                            "tool": selected_record["tool"],
                            "artifact_paths": selected_record["artifact_paths"],
                            "artifact_hashes": selected_record["artifact_hashes"],
                            "artifact_sizes": selected_record["artifact_sizes"],
                            "registration_surface_hash": observation[
                                "registration_surface_hash"
                            ],
                        }
                        if selected_record is not None
                        else None
                    ),
                }
                selected[outcome.policy_id][job["object_slot"]] = selected_payload

        validate_ledger(ledger)
        expected_rows = len(scene["jobs"]) * len(POLICY_IDS)
        if len(ledger) != expected_rows:
            raise ValueError(f"controller ledger has {len(ledger)} rows, expected {expected_rows}")
        shard = {
            "schema_version": SCHEMA_VERSION,
            "freeze_id": freeze_id,
            "scene_id": scene_id,
            "created_utc": _utc_now(),
            "policy_config_sha256": _canonical_digest(policies),
            "observation_manifest_sha256": sha256_file(
                out_dir / "observations" / scene_id / "manifest.json"
            ),
            "evaluation_geometry_read": False,
            "job_count": len(scene["jobs"]),
            "ledger_row_count": len(ledger),
            "proposals": proposal_records,
            "tool_failures": failures,
        }
        _write_controller_shard(staging, destination, shard, ledger, selected)
    return shard


def _load_control_scene(out_dir: Path, scene_id: str):
    directory = out_dir / "control" / scene_id
    seal = _verify_sealed_directory(directory, controller_safe=False)
    shard = _load_json_repo(directory / "controller_shard.json", "controller shard")
    if shard.get("scene_id") != scene_id or shard.get("evaluation_geometry_read") is not False:
        raise ValueError(f"invalid controller shard for {scene_id}")
    selected_root = directory / "selected_assets"
    if not selected_root.is_dir() or selected_root.is_symlink():
        raise ValueError(f"missing sealed selected-asset root for {scene_id}")
    if {path.name for path in selected_root.iterdir()} != set(POLICY_IDS) or any(
        not path.is_dir() or path.is_symlink() for path in selected_root.iterdir()
    ):
        raise ValueError(f"selected-asset policy directories differ from A0--A4: {scene_id}")
    actual_selected: set[str] = set()
    for current, directories, filenames in os.walk(selected_root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            child = current_path / name
            if child.is_symlink():
                raise ValueError(f"selected-asset tree contains a symlink: {child}")
            if current_path != selected_root:
                raise ValueError(f"selected-asset tree contains a nested directory: {child}")
        for name in filenames:
            child = current_path / name
            if child.is_symlink() or not child.is_file() or child.suffix != ".json":
                raise ValueError(f"selected-asset tree contains an invalid file: {child}")
            actual_selected.add(child.relative_to(directory).as_posix())
    declared_hashes = seal.get("selected_asset_hashes", {})
    declared_sizes = seal.get("selected_asset_sizes", {})
    if (
        not isinstance(declared_hashes, Mapping)
        or not isinstance(declared_sizes, Mapping)
        or set(declared_hashes) != actual_selected
        or set(declared_sizes) != actual_selected
    ):
        raise ValueError(f"selected-asset seal coverage mismatch: {scene_id}")
    for relative, expected in declared_hashes.items():
        path = checked_repo_path(directory / relative, "sealed selected asset", kind="file")
        if (
            sha256_file(path) != expected
            or path.stat().st_size != declared_sizes[relative]
        ):
            raise ValueError(f"selected asset hash mismatch: {path}")
    return directory, shard, seal


def run_evaluate(freeze_id: str, out_dir: Path, scene_id: str, *,
                 evaluation_manifest=None, evaluation_manifest_sha256=None) -> dict[str, Any]:
    scene_id = _require_scene_id(scene_id)
    # Seal and capture the GT-free controller result before opening the full
    # inventory, whose audit member contains the independent GT references.
    control_dir, shard, control_seal = _load_control_scene(out_dir, scene_id)
    if shard.get("freeze_id") != freeze_id:
        raise ValueError("controller shard freeze_id differs from requested freeze")
    controller_shard_hash = control_seal["members"]["controller_shard.json"]
    resolved_jobs, _ = _load_inventory(out_dir, controller_safe=False)
    scene = _scene_inventory(resolved_jobs, scene_id)
    external = evaluation_manifest is not None
    if external:
        from agents.eval.automatic_matching_manifest import load_references
        reference_rows, reference_manifest = load_references(
            evaluation_manifest, evaluation_manifest_sha256, out_dir, scene_id,
            controller_shard_hash, [j['job_id'] for j in scene['jobs']])
    else:
        audit = _load_json_repo(
            out_dir / "input_inventory" / "inventory_audit.json",
            "evaluation-only frozen reference inventory",
        )
        reference_rows = {
            row["job_id"]: row for row in audit.get("evaluation_references", [])
            if row.get("scene_id") == scene_id
        }
        if len(reference_rows) != len(scene["jobs"]):
            raise ValueError(
                f"evaluation reference inventory for {scene_id} is incomplete"
            )
    proposals = {row["proposal_id"]: row for row in shard["proposals"]}
    if len(proposals) != len(shard["proposals"]):
        raise ValueError("duplicate proposal IDs in sealed controller shard")
    ledger_rows = [
        json.loads(line)
        for line in (control_dir / "job_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_ledger(ledger_rows)

    from agents.assets.s5_align import apply_T
    from robo.eval.fidelity_metrics import geometry_metrics, load_surface

    metrics_by_proposal: dict[str, dict[str, Any]] = {}
    for proposal_id, proposal in proposals.items():
        object_slot = _require_object_slot(proposal["object_id"])
        reference_identity = reference_rows.get(proposal["job_id"])
        if not isinstance(reference_identity, Mapping):
            raise ValueError(f"missing frozen evaluation reference for {proposal['job_id']}")
        if external and reference_identity['status']=='unmatched':
            metrics_by_proposal[proposal_id] = {
                'f1_20':None, 'cd_cm':None, 'collapse':None,
                'geometry_evaluation_status':'unmatched',
                'settle_stable':proposal['evidence']['settle_stable'],
                'settle_drift_m':proposal['evidence']['settle_drift_m'],
            }
            continue
        if external:
            reference_identity = reference_identity['evaluation_surface']
            reference_path = checked_repo_path(reference_identity['path'], 'independent evaluation surface', kind='file')
        else:
            reference_path = checked_repo_path(
                reference_identity.get("path", ""),
                "evaluation reference surface",
                kind="file",
            )
            expected_reference = (
                REPOSITORY_ROOT
                / f"outputs/{scene_id}_factory/objects/{object_slot}/gt_points.ply"
            )
            if reference_path != expected_reference:
                raise ValueError("evaluation reference path differs from the frozen object job")
        reference_hash = sha256_file(reference_path)
        if (
            reference_hash != _require_sha256(
                reference_identity.get("sha256"), "evaluation reference SHA-256"
            )
            or reference_path.stat().st_size != reference_identity.get("size_bytes")
        ):
            raise ValueError(f"evaluation reference drift: {reference_path}")
        registration_hash = proposal["input_hashes"]["visible_observation"]
        if reference_hash == registration_hash:
            raise ValueError("registration and evaluation surface hashes are identical")
        raw_mesh_path = checked_repo_path(
            proposal["artifact_paths"]["raw_mesh"], "evaluated raw mesh", kind="file"
        )
        if sha256_file(raw_mesh_path) != proposal["artifact_hashes"]["raw_mesh"]:
            raise ValueError(f"raw mesh drift before evaluation: {raw_mesh_path}")
        transform_path = checked_repo_path(
            proposal["artifact_paths"]["transform"], "evaluated transform", kind="file"
        )
        if sha256_file(transform_path) != proposal["artifact_hashes"]["transform"]:
            raise ValueError(f"transform drift before evaluation: {transform_path}")
        mesh = _load_mesh(raw_mesh_path)
        predicted = _sample_mesh(
            mesh, MESH_SAMPLE_COUNT, _stable_seed(proposal_id, "evaluation")
        )
        transform = np.load(transform_path, allow_pickle=False)
        target = load_surface(reference_path)
        metrics = geometry_metrics(apply_T(transform, predicted), target, threshold_m=0.02)
        metrics_by_proposal[proposal_id] = {
            **metrics,
            "evaluation_surface_sha256": reference_hash,
            "registration_surface_sha256": registration_hash,
            "settle_stable": proposal["evidence"]["settle_stable"],
            "settle_drift_m": proposal["evidence"]["settle_drift_m"],
        }

    rows = []
    for row in ledger_rows:
        selected_id = row.get("selected_proposal_id")
        metrics = metrics_by_proposal.get(selected_id) if selected_id else None
        rows.append(
            {
                **row,
                "accepted": row["terminal_action"] == "accept",
                "metrics": metrics,
                "physical_metrics": ({key:proposals[selected_id]['evidence'][key]
                    for key in ('settle_stable','settle_drift_m')} if selected_id in proposals else None),
            }
        )
    if len(rows) != len(scene["jobs"]) * len(POLICY_IDS):
        raise ValueError("evaluation row count differs from frozen policy/object product")
    evaluation = {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "scene_id": scene_id,
        "created_utc": _utc_now(),
        "controller_shard_sha256_verified_before_gt_read": controller_shard_hash,
        "job_count": len(scene["jobs"]),
        "rows": rows,
        "proposal_metrics": metrics_by_proposal,
        "geometry_reference_jobs": sum(r.get('status','matched')=='matched' for r in reference_rows.values()),
        "geometry_unmatched_jobs": sum(r.get('status')=='unmatched' for r in reference_rows.values()),
        "external_evaluation_manifest_sha256": evaluation_manifest_sha256,
    }
    if external:
        evaluation['construction_freeze_id'] = freeze_id
        evaluation['evaluation_freeze_id'] = reference_manifest['freeze_id']
        evaluation['evaluation_code_commit'] = reference_manifest['code_commit']
        evaluation_root = REPOSITORY_ROOT / 'outputs/icra2027' / reference_manifest['freeze_id'] / 'agentic'
        if evaluation_root == out_dir:
            raise ValueError('external evaluation requires a new freeze separate from construction')
    else:
        evaluation_root = out_dir
    destination = evaluation_root / "evaluation" / scene_id
    with _atomic_directory(destination) as staging:
        shard_path = staging / "eval_shard.json"
        _write_json_inside(shard_path, evaluation)
        _write_json_inside(
            staging / "seal.json",
            {
                "schema_version": SCHEMA_VERSION,
                "members": {"eval_shard.json": sha256_file(shard_path)},
            },
        )
    return evaluation


def _load_eval_scene(out_dir: Path, scene_id: str, *, construction_root: Path | None = None) -> dict[str, Any]:
    directory = out_dir / "evaluation" / scene_id
    _verify_sealed_directory(directory, controller_safe=False)
    payload = _load_json_repo(directory / "eval_shard.json", "evaluation shard")
    if payload.get("scene_id") != scene_id:
        raise ValueError(f"evaluation shard scene mismatch: {scene_id}")
    control_seal = _load_json_repo(
        (construction_root or out_dir) / "control" / scene_id / "seal.json", "controller seal"
    )
    if payload.get("controller_shard_sha256_verified_before_gt_read") != control_seal[
        "members"
    ]["controller_shard.json"]:
        raise ValueError(f"evaluation/control seal mismatch: {scene_id}")
    return payload


def _csv_payload(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(materialized)) if materialized else None


def _policy_runtime_seconds(
    policy_id: str, job_id: str, control_shard: Mapping[str, Any], ledger_row: Mapping[str, Any]
) -> float:
    proposals = [p for p in control_shard["proposals"] if p["job_id"] == job_id]
    failed_attempts = [
        row
        for row in control_shard.get("tool_failures", [])
        if row.get("job_id") == job_id and isinstance(row.get("wall_s"), (int, float))
    ]
    if policy_id == "A0":
        invoked = [p for p in proposals if p["tool"] == "trellis"]
        failed_invoked = [row for row in failed_attempts if row.get("tool") == "trellis"]
    elif policy_id in {"A1", "A2"}:
        invoked = [p for p in proposals if p["tool"] in {"trellis", "reconviagen"}]
        failed_invoked = [
            row
            for row in failed_attempts
            if row.get("tool") in {"trellis", "reconviagen"}
        ]
    else:
        invoked = proposals
        failed_invoked = failed_attempts
    return float(
        sum(float(p["wall_s"]) for p in invoked)
        + sum(float(row["wall_s"]) for row in failed_invoked)
    )


def _aggregate_rows(
    all_rows: list[dict[str, Any]],
    control_by_scene: Mapping[str, Mapping[str, Any]],
    planned_jobs: int,
    planned_scenes: int,
) -> list[dict[str, Any]]:
    output = []
    for policy_id in POLICY_IDS:
        rows = [row for row in all_rows if row["policy_id"] == policy_id]
        if len(rows) != planned_jobs:
            raise ValueError(f"{policy_id} has {len(rows)} rows, expected {planned_jobs}")
        accepted = [row for row in rows if row["accepted"]]
        evaluated = [row for row in accepted if row.get("metrics") is not None
                     and row['metrics'].get('f1_20') is not None]
        physical = [row.get('physical_metrics', row.get('metrics')) for row in accepted]
        physical = [value for value in physical if value is not None and type(value.get('settle_stable')) is bool]
        runtime = sum(
            _policy_runtime_seconds(
                policy_id, row["job_id"], control_by_scene[row["scene_id"]], row
            )
            for row in rows
        )
        output.append(
            {
                "policy_id": policy_id,
                "planned_jobs": planned_jobs,
                "accepted_jobs": len(accepted),
                "build_coverage": len(accepted) / planned_jobs,
                "f1_20": _mean(row["metrics"]["f1_20"] for row in evaluated),
                "cd_cm": _mean(row["metrics"]["cd_cm"] for row in evaluated),
                "catastrophic_collapses": sum(
                    bool(row["metrics"]["collapse"]) for row in evaluated
                ),
                "stable_fraction": _mean(float(row['settle_stable']) for row in physical),
                "geometry_evaluated_jobs": len(evaluated),
                "physical_tested_jobs": len(physical),
                "physical_stable_jobs": sum(row['settle_stable'] for row in physical),
                "runtime_minutes_per_scene": runtime / 60.0 / planned_scenes,
                "retry_count": sum(bool(row["retry_invoked"]) for row in rows),
                "abstain_count": sum(row["terminal_action"] == "abstain" for row in rows),
            }
        )
    return output


def _coverage_sweep(
    policies: Mapping[str, Any],
    control_by_scene: Mapping[str, Mapping[str, Any]],
    eval_by_scene: Mapping[str, Mapping[str, Any]],
    planned_jobs: int,
) -> list[dict[str, Any]]:
    spec = policies.get("a4_registration_residual_sweep", {})
    thresholds = spec.get("values_m", [])
    if not thresholds:
        raise ValueError("A4 residual threshold sweep is empty")
    jobs: list[tuple[dict[str, Any] | None, Mapping[str, Any]]] = []
    for scene_id in sorted(control_by_scene):
        shard = control_by_scene[scene_id]
        evaluation = eval_by_scene[scene_id]
        metrics = evaluation["proposal_metrics"]
        by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for proposal in shard["proposals"]:
            by_job[proposal["job_id"]].append(proposal)
        job_ids = sorted({row["job_id"] for row in evaluation["rows"]})
        # Freeze the lexicographic choice once under the primary policy.  The
        # sweep varies acceptance only; it never reselects a candidate after
        # seeing a threshold, as required by selection_after_sweep_forbidden.
        jobs.extend(
            (choose_best(by_job.get(job_id, []), policies["gates"]), metrics)
            for job_id in job_ids
        )
    if len(jobs) != planned_jobs:
        raise ValueError("threshold-sweep job count differs from frozen denominator")
    output = []
    for threshold in thresholds:
        gates = dict(policies["gates"])
        gates["max_registration_residual_m"] = float(threshold)
        selected_metrics = []
        accepted_proposals = []
        for selected, metrics in jobs:
            if (
                selected is not None
                and evaluate_gate(selected["evidence"], gates).passed
            ):
                accepted_proposals.append(selected)
                metric=metrics.get(selected['proposal_id'])
                if metric is not None and metric.get('f1_20') is not None:
                    selected_metrics.append(metric)
        output.append(
            {
                "policy_id": "A4",
                "max_registration_residual_m": float(threshold),
                "planned_jobs": planned_jobs,
                "accepted_jobs": len(accepted_proposals),
                "build_coverage": len(accepted_proposals) / planned_jobs,
                "geometry_evaluated_jobs": len(selected_metrics),
                "physical_tested_jobs": len(accepted_proposals),
                "f1_20": _mean(row["f1_20"] for row in selected_metrics),
                "cd_cm": _mean(row["cd_cm"] for row in selected_metrics),
                "catastrophic_collapses": sum(row["collapse"] for row in selected_metrics),
                "stable_fraction": _mean(float(row["evidence"]["settle_stable"]) for row in accepted_proposals),
            }
        )
    return output


def _preliminary_retry_claim_status(genuine_retry_jobs: int, minimum: int) -> str:
    if genuine_retry_jobs < 0 or minimum < 1:
        raise ValueError("retry counts must be non-negative with a positive minimum")
    return (
        "retry_count_threshold_met_preliminary_only"
        if genuine_retry_jobs >= minimum
        else "case_study_only"
    )


def _publish_aggregate(out_dir: Path, files: Mapping[str, bytes], selected_tree: Path) -> None:
    destinations = [out_dir / name for name in files] + [out_dir / "selected_assets"]
    for destination in destinations:
        checked_repo_path(destination, "aggregate output", must_exist=False)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite aggregate output: {destination}")
    staging = out_dir / f".aggregate.staging-{uuid.uuid4().hex}"
    _mkdir_repo_parents(out_dir)
    staging.mkdir()
    published: list[Path] = []
    try:
        for name, payload in files.items():
            _write_bytes_inside(staging / name, payload)
        shutil.copytree(selected_tree, staging / "selected_assets", symlinks=False)
        selected_members: dict[str, dict[str, Any]] = {}
        selected_directories: list[str] = []
        selected_root = staging / "selected_assets"
        for current, directories, filenames in os.walk(selected_root, followlinks=False):
            current_path = Path(current)
            for name in sorted(directories):
                child = current_path / name
                if child.is_symlink():
                    raise ValueError(f"aggregate selected tree contains a symlink: {child}")
                selected_directories.append(
                    child.relative_to(selected_root).as_posix()
                )
            for name in sorted(filenames):
                child = current_path / name
                if child.is_symlink() or not child.is_file():
                    raise ValueError(f"aggregate selected tree has an invalid file: {child}")
                relative = child.relative_to(selected_root).as_posix()
                selected_members[relative] = {
                    "sha256": sha256_file(child),
                    "size_bytes": child.stat().st_size,
                }
        # Publish files without overwrite; completion marker is last.
        for name in files:
            destination = out_dir / name
            os.link(staging / name, destination)
            published.append(destination)
        (staging / "selected_assets").rename(out_dir / "selected_assets")
        published.append(out_dir / "selected_assets")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "members": {name: _sha256_bytes(payload) for name, payload in files.items()},
            "selected_asset_members": selected_members,
            "selected_asset_directories": sorted(selected_directories),
        }
        _atomic_write_json(out_dir / "aggregate_seal.json", manifest)
    except Exception:
        # Roll back only paths created by this invocation so a transient
        # publication failure remains safely retryable without overwriting.
        for path in reversed(published):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists() and not path.is_symlink():
                path.unlink()
        raise
    finally:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)


AGENTIC_UNCERTAINTY_PROTOCOL = {
    "schema_version": 1,
    "method": "paired_object_scene_cluster_percentile_bootstrap",
    "samples": 2000,
    "seed": 0,
    "confidence": 0.95,
    "contrasts": [["A0", "A1"], ["A1", "A2"], ["A2", "A3"], ["A0", "A4"]],
    "metrics": ["build_coverage", "f1_20", "cd_cm", "stable_fraction"],
}


def _paired_uncertainty(all_rows, jobs_by_scene, protocol):
    """Describe fixed contrasts using the existing scene-cluster bootstrap.

    Coverage retains every planned job. Geometry conditions on common accepted
    matched jobs; stability conditions on common accepted construction probes.
    No rollout, seed-level variation, or independent physical trial is inferred.
    """
    from robo.eval.fidelity_metrics import _cluster_bootstrap

    if protocol != AGENTIC_UNCERTAINTY_PROTOCOL:
        raise ValueError("agentic uncertainty protocol differs from predeclared specification")
    job_scene = {job: scene for scene, jobs in jobs_by_scene.items() for job in jobs}
    if len(job_scene) != sum(map(len, jobs_by_scene.values())):
        raise ValueError("uncertainty planned job identities are duplicated")
    expected = {(policy, job) for policy in POLICY_IDS for job in job_scene}
    rows = {(row['policy_id'], row['job_id']): row for row in all_rows}
    if (len(rows) != len(all_rows) or set(rows) != expected
            or any(row['scene_id'] != job_scene[row['job_id']] for row in all_rows)
            or any(type(row.get('accepted')) is not bool for row in all_rows)):
        raise ValueError("uncertainty requires the complete paired policy/job roster")

    def geometry(row):
        value = row.get('metrics')
        if not row['accepted'] or not isinstance(value, Mapping):
            return None
        if value.get('geometry_evaluation_status') == 'unmatched':
            return None
        if any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k])
               for k in ('f1_20', 'cd_cm')):
            return None
        if not 0 <= value['f1_20'] <= 1 or value['cd_cm'] < 0:
            raise ValueError("uncertainty geometry values violate metric domains")
        _require_sha256(value.get('evaluation_surface_sha256'), 'paired evaluation surface')
        return value

    def physical(row):
        value = row.get('physical_metrics')
        return (value if row['accepted'] and isinstance(value, Mapping)
                and type(value.get('settle_stable')) is bool else None)

    output = []
    planned_jobs = len(job_scene)
    for baseline, treatment in protocol['contrasts']:
        contrast = f'{treatment}-{baseline}'
        pairs = [(job, rows[(baseline, job)], rows[(treatment, job)])
                 for job in sorted(job_scene)]
        ga = {job: geometry(a) for job, a, _ in pairs}
        gb = {job: geometry(b) for job, _, b in pairs}
        pa = {job: physical(a) for job, a, _ in pairs}
        pb = {job: physical(b) for job, _, b in pairs}
        common_geometry = {job for job, _, _ in pairs if ga[job] is not None and gb[job] is not None}
        if any(ga[job]['evaluation_surface_sha256'] != gb[job]['evaluation_surface_sha256']
               for job in common_geometry):
            raise ValueError("paired geometry policies use different matched evaluation surfaces")
        common_physical = {job for job, _, _ in pairs if pa[job] is not None and pb[job] is not None}
        for metric in protocol['metrics']:
            if metric == 'build_coverage':
                eligible = set(job_scene)
                av = {job: float(a['accepted']) for job, a, _ in pairs}
                bv = {job: float(b['accepted']) for job, _, b in pairs}
                left_supported = right_supported = planned_jobs
                support = 'all_planned_jobs'
            elif metric in ('f1_20', 'cd_cm'):
                eligible = common_geometry
                av = {job: ga[job][metric] for job in eligible}
                bv = {job: gb[job][metric] for job in eligible}
                left_supported = sum(value is not None for value in ga.values())
                right_supported = sum(value is not None for value in gb.values())
                support = 'common_accepted_matched_geometry'
            else:
                eligible = common_physical
                av = {job: float(pa[job]['settle_stable']) for job in eligible}
                bv = {job: float(pb[job]['settle_stable']) for job in eligible}
                left_supported = sum(value is not None for value in pa.values())
                right_supported = sum(value is not None for value in pb.values())
                support = 'common_accepted_construction_probes'
            ids = sorted(eligible)
            observations = [{'scene': job_scene[job], 'value': bv[job] - av[job]} for job in ids]
            scenes = sorted({job_scene[job] for job in ids})
            ci = (_cluster_bootstrap(observations, group_key='scene', samples=protocol['samples'],
                                     seed=protocol['seed'])['ci95'] if len(scenes) >= 2 else [None, None])
            output.append(dict(
                contrast=contrast, baseline=baseline, treatment=treatment, metric=metric,
                support=support, planned_scenes=len(jobs_by_scene), planned_jobs=planned_jobs,
                baseline_accepted_jobs=sum(a['accepted'] for _, a, _ in pairs),
                treatment_accepted_jobs=sum(b['accepted'] for _, _, b in pairs),
                baseline_metric_eligible_jobs=left_supported,
                treatment_metric_eligible_jobs=right_supported,
                common_accepted_jobs=sum(a['accepted'] and b['accepted'] for _, a, b in pairs),
                paired_jobs=len(ids), paired_scenes=len(scenes), excluded_pair_jobs=planned_jobs-len(ids),
                unsupported_scenes=len(jobs_by_scene)-len(scenes),
                paired_job_ids_sha256=_canonical_digest(ids), paired_scene_ids=scenes,
                baseline_mean=_mean(av[job] for job in ids), treatment_mean=_mean(bv[job] for job in ids),
                delta=_mean(row['value'] for row in observations), ci95=ci,
                status='ESTIMATED' if len(scenes) >= 2 else 'NOT_ESTIMABLE',
                reason=None if len(scenes) >= 2 else 'fewer_than_two_supported_scenes',
                selection_conditioned=metric != 'build_coverage',
                independent_physical_validation=False,
            ))
    return dict(
        schema_version=1, protocol=protocol, rows=output,
        source_policy_rows_sha256=_canonical_digest(all_rows),
        planned_scenes=len(jobs_by_scene), planned_jobs=planned_jobs,
        interpretation='Object-pair-weighted point differences; whole supported scenes resampled. '
            'Conditional geometry/probe intervals do not measure population-wide quality gains. '
            'Intervals are pointwise descriptive and not multiplicity-adjusted; no seed-level replication.',
        headline_eligible=False, paper_ready=False, claim_gate='NOT_RUN',
    )


def run_aggregate(
    jobs: Mapping[str, Any], policies: Mapping[str, Any], freeze_id: str, out_dir: Path,
    *, evaluation_root: Path | None = None, uncertainty_protocol: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    resolved_jobs, _ = _load_inventory(out_dir, controller_safe=False)
    counts = resolved_jobs["counts"]
    population = jobs.get("population")
    if population is None and evaluation_root is not None:
        source = jobs.get("automatic_sources")
        if not isinstance(source, Mapping):
            raise ValueError("external automatic pilot requires a declared source population")
        source_scene = _require_scene_id(source["scene_id"])
        if [scene["scene_id"] for scene in resolved_jobs["scenes"]] != [source_scene]:
            raise ValueError("external automatic pilot scene differs from sealed inventory")
        population = {"planned_scenes": 1, "planned_jobs_per_policy": source["planned_jobs"],
                      "planned_policy_object_rows": len(POLICY_IDS) * source["planned_jobs"]}
    planned_scenes = int(population["planned_scenes"])
    planned_jobs = int(population["planned_jobs_per_policy"])
    planned_rows = int(population["planned_policy_object_rows"])
    if counts != {
        "scenes": planned_scenes,
        "jobs": planned_jobs,
        "policy_object_rows": planned_rows,
    }:
        raise ValueError("resolved inventory counts differ from frozen 50/457/2285")
    scene_ids = [scene["scene_id"] for scene in resolved_jobs["scenes"]]
    control_by_scene: dict[str, dict[str, Any]] = {}
    eval_by_scene: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    all_ledger: list[dict[str, Any]] = []
    publication_root = evaluation_root or out_dir
    selected_staging = publication_root / f".selected.staging-{uuid.uuid4().hex}"
    selected_staging.mkdir(parents=False)
    try:
        for scene_id in scene_ids:
            control_dir, control, _ = _load_control_scene(out_dir, scene_id)
            if (evaluation_root is not None
                    and control.get("policy_config_sha256") != _canonical_digest(policies)):
                raise ValueError("external aggregate policy differs from sealed controller")
            evaluation = _load_eval_scene(publication_root, scene_id, construction_root=out_dir)
            if evaluation_root is not None and (
                evaluation.get('evaluation_freeze_id') != publication_root.parent.name
                or evaluation.get('construction_freeze_id') != freeze_id
            ):
                raise ValueError('external evaluation aggregate freeze differs')
            control_by_scene[scene_id] = control
            eval_by_scene[scene_id] = evaluation
            all_rows.extend(evaluation["rows"])
            ledger = [
                json.loads(line)
                for line in (control_dir / "job_ledger.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            all_ledger.extend(ledger)
            for policy_id in POLICY_IDS:
                source = control_dir / "selected_assets" / policy_id
                destination = selected_staging / policy_id / scene_id
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination, symlinks=False)
        validate_ledger(all_ledger)
        if len(all_rows) != planned_rows or len(all_ledger) != planned_rows:
            raise ValueError(
                f"aggregate requires exactly {planned_rows} rows; "
                f"eval={len(all_rows)}, ledger={len(all_ledger)}"
            )
        main_rows = _aggregate_rows(
            all_rows, control_by_scene, planned_jobs, planned_scenes
        )
        runtime_accounting = None
        if "automatic_sources" in jobs:
            from robo.eval.agentic_runtime_accounting import account_runtime
            runtime_accounting = account_runtime(out_dir, jobs, control_by_scene, all_ledger)
            for row, timing in zip(main_rows, runtime_accounting["rows"]):
                canonical = timing["canonical_proposal_and_retry_wall_s"]
                row["canonical_runtime_minutes_per_scene"] = None if canonical is None else canonical / 60 / planned_scenes
                generation = timing["attributed_generation_process_wall_s"]
                total = timing["attributed_algorithm_phase_wall_s"]
                row["generation_process_minutes_per_scene"] = None if generation is None else generation / 60 / planned_scenes
                row["runtime_minutes_per_scene"] = None if total is None else total / 60 / planned_scenes
                row["runtime_accounting_status"] = timing["runtime_accounting_status"]
        sweep = _coverage_sweep(
            policies, control_by_scene, eval_by_scene, planned_jobs
        )
        retry_reasons = Counter(
            reason
            for row in all_ledger
            if row["policy_id"] == "A4" and row["retry_required"]
            for reason in row.get("retry_reason_codes", [])
        )
        retry_jobs = {
            row["job_id"]
            for row in all_ledger
            if row["policy_id"] == "A4" and row.get("retry_produced") is True
        }
        retry_rows = [
            {"reason_code": reason, "job_count": count}
            for reason, count in sorted(retry_reasons.items())
        ]
        transition_counts = Counter()
        for row in all_ledger:
            if row["policy_id"] != "A4" or not row["retry_required"]:
                continue
            source = "+".join(row.get("retry_reason_codes", [])) or "unspecified"
            transition_counts[(source, row["terminal_action"])] += 1
        transition_rows = [
            {"initial_failure": source, "terminal_action": terminal, "job_count": count}
            for (source, terminal), count in sorted(transition_counts.items())
        ]
        headline_min = int(
            policies["reporting_contract"]["minimum_genuine_retry_jobs_for_headline_claim"]
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "freeze_id": freeze_id,
            "created_utc": _utc_now(),
            "study_scope": jobs["study_scope"],
            "counts": {
                "scenes": planned_scenes,
                "jobs_per_policy": planned_jobs,
                "policy_object_rows": planned_rows,
                "genuine_retry_jobs": len(retry_jobs),
            },
            "rows": main_rows,
            # Schema v1 intentionally freezes legacy-generated proposal bytes
            # without generator/checkpoint/runtime provenance and both configs
            # are preliminary.  Meeting the retry-count floor is necessary but
            # cannot by itself make this output headline/paper eligible.
            "retry_claim_status": _preliminary_retry_claim_status(
                len(retry_jobs), headline_min
            ),
            "headline_eligible": False,
            "paper_ready": False,
            "policy_config_sha256": _canonical_digest(policies),
        }
        if evaluation_root is not None:
            payload['construction_freeze_id'] = freeze_id
            payload['freeze_id'] = publication_root.parent.name
        main_columns = list(policies["reporting_contract"]["main_columns"])
        main_columns += [key for key in ("geometry_evaluated_jobs", "physical_tested_jobs", "physical_stable_jobs") if key not in main_columns]
        if runtime_accounting is not None:
            payload["runtime_scope"] = runtime_accounting["scope"]
            main_columns += [key for key in ("canonical_runtime_minutes_per_scene", "generation_process_minutes_per_scene", "runtime_accounting_status") if key not in main_columns]
        files = {
            "job_ledger.jsonl": "".join(
                canonical_json(row) + "\n"
                for row in sorted(
                    all_ledger,
                    key=lambda row: (row["scene_id"], row["object_slot"], row["policy_id"]),
                )
            ).encode("utf-8"),
            "agentic_ablation.csv": _csv_payload(main_rows, main_columns).encode("utf-8"),
            "agentic_ablation.json": (
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8"),
            "coverage_fidelity_curve.csv": _csv_payload(
                sweep,
                [
                    "policy_id",
                    "max_registration_residual_m",
                    "geometry_evaluated_jobs",
                    "physical_tested_jobs",
                    "planned_jobs",
                    "accepted_jobs",
                    "build_coverage",
                    "f1_20",
                    "cd_cm",
                    "catastrophic_collapses",
                    "stable_fraction",
                ],
            ).encode("utf-8"),
            "retry_breakdown.csv": _csv_payload(
                retry_rows, ["reason_code", "job_count"]
            ).encode("utf-8"),
            "failure_transitions.csv": _csv_payload(
                transition_rows, ["initial_failure", "terminal_action", "job_count"]
            ).encode("utf-8"),
        }
        if runtime_accounting is not None:
            files["runtime_accounting.json"] = (json.dumps(runtime_accounting, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        if uncertainty_protocol is not None:
            uncertainty = _paired_uncertainty(all_rows,
                {scene['scene_id']: [job['job_id'] for job in scene['jobs']]
                 for scene in resolved_jobs['scenes']}, uncertainty_protocol)
            uncertainty.update(freeze_id=publication_root.parent.name, construction_freeze_id=freeze_id)
            files['agentic_paired_uncertainty.json'] = (
                json.dumps(uncertainty, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
            uncertainty_rows = [dict(row, ci95_low=row['ci95'][0], ci95_high=row['ci95'][1])
                                for row in uncertainty['rows']]
            columns = [key for key in uncertainty_rows[0] if key not in {'ci95', 'paired_scene_ids'}]
            files['agentic_paired_uncertainty.csv'] = _csv_payload(uncertainty_rows, columns).encode()
        _publish_aggregate(publication_root, files, selected_staging)
    finally:
        if selected_staging.exists() and not selected_staging.is_symlink():
            shutil.rmtree(selected_staging)
    return payload


def _smoke_evidence(*, residual: float, stable: bool = True) -> dict[str, Any]:
    return {
        "schema_frame_unit_valid": True,
        "symmetric_clipped_registration_residual_m": residual,
        "scale_ratio_vs_observation": 1.0,
        "observation_point_count": 400,
        "visible_fraction": 0.8,
        "support_gap_m": 0.005,
        "support_overlap_fraction": 1.0,
        "initial_penetration_m": 0.0,
        "collision_valid": True,
        "usable_convex_parts": 1,
        "settle_drift_m": 0.001 if stable else 0.08,
        "settle_sunk": False,
        "settle_stable": stable,
        "missing_evidence": [],
    }


def run_smoke(policies: Mapping[str, Any], freeze_id: str, out_dir: Path) -> dict[str, Any]:
    _validate_probe_protocol(policies)
    start = time.perf_counter()
    good_t = {"proposal_id": "smoke-1:t", "tool": "trellis", "evidence": _smoke_evidence(residual=0.01)}
    good_r = {"proposal_id": "smoke-1:r", "tool": "reconviagen", "evidence": _smoke_evidence(residual=0.008)}
    one = run_policies([good_t, good_r], policies)

    collapsed_r = {"proposal_id": "smoke-2:r", "tool": "reconviagen", "evidence": _smoke_evidence(residual=0.08, stable=False)}
    valid_t = {"proposal_id": "smoke-2:t", "tool": "trellis", "evidence": _smoke_evidence(residual=0.012)}
    two = run_policies([valid_t, collapsed_r], policies)

    bad_t = {"proposal_id": "smoke-3:t", "tool": "trellis", "evidence": _smoke_evidence(residual=0.08, stable=False)}
    bad_r = {"proposal_id": "smoke-3:r", "tool": "reconviagen", "evidence": _smoke_evidence(residual=0.07, stable=False)}
    planned = run_policies([bad_t, bad_r], policies)
    retry = {
        "proposal_id": "smoke-3:r:retry",
        "tool": "registration_retry",
        "parent_proposal_ids": [planned.retry_parent_proposal_id],
        "evidence": _smoke_evidence(residual=0.06, stable=False),
    }
    three = run_policies([bad_t, bad_r], policies, retry_candidate=retry)
    empty = run_policies([], policies)
    actions = {"retry"}
    results = []
    for scenario, result in (("both_initial_valid", one), ("multiview_collapse", two), ("retry_abstains", three), ("hard_export_impossible", empty)):
        outcomes = []
        for outcome in result.outcomes:
            actions.add(outcome.terminal_action)
            outcomes.append(
                {
                    "policy_id": outcome.policy_id,
                    "terminal_action": outcome.terminal_action,
                    "selected_proposal_id": outcome.selected_proposal_id,
                    "retry_invoked": outcome.retry_invoked,
                    "reason_codes": list(outcome.reason_codes),
                }
            )
        results.append({"scenario": scenario, "outcomes": outcomes})
    required = set(
        # Jobs config also declares this, but these four actions are the fixed
        # README smoke contract and are intentionally asserted in code.
        ("accept", "retry", "reject", "abstain")
    )
    if not required.issubset(actions):
        raise AssertionError(f"smoke did not exercise actions: {sorted(required - actions)}")
    elapsed = time.perf_counter() - start
    if elapsed >= 300:
        raise RuntimeError(f"smoke exceeded five-minute contract: {elapsed:.3f}s")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "freeze_id": freeze_id,
        "elapsed_s": elapsed,
        "actions_observed": sorted(actions),
        "scenarios": results,
    }
    _atomic_write_json(out_dir / "smoke" / "result.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--policies", required=True)
    parser.add_argument("--contract-manifest", required=True)
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene-id")
    parser.add_argument("--evaluation-root", help="new freeze agentic root, aggregate only")
    parser.add_argument("--uncertainty-config", help="E0-bound full matching config with paired uncertainty protocol; external aggregate only")
    parser.add_argument("--evaluation-manifest")
    parser.add_argument("--evaluation-manifest-sha256")
    phases = parser.add_mutually_exclusive_group(required=True)
    phases.add_argument("--inventory", action="store_true")
    phases.add_argument("--observe", action="store_true")
    phases.add_argument("--control", action="store_true")
    phases.add_argument("--evaluate", action="store_true")
    phases.add_argument("--aggregate", action="store_true")
    phases.add_argument("--smoke", action="store_true")
    return parser


def _validate_cli_execution(
    contract_path: str | Path, freeze_id: str, out_dir: Path, *,
    config_paths: Sequence[str | Path] = (), require_inventory: bool = False,
) -> dict[str, Any]:
    """Authenticate executing code independently from the artifact checkout.

    Observe/control never reopen the inventory-only jobs YAML. Inventory binds
    its bytes to E0 once; subsequent phases require that same clean source SHA
    and its sealed inventory. Control additionally checks its policy file.
    """
    from robo.manifest.hash import canonical_hash
    if Path(_git_value(CODE_ROOT, "rev-parse", "--show-toplevel")) != CODE_ROOT:
        raise ValueError("executing source is not its declared Git checkout")
    commit = _git_value(CODE_ROOT, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("executing source lacks a full Git commit")
    if _git_value(CODE_ROOT, "status", "--porcelain"):
        raise ValueError("executing source must be clean")
    contract = _load_controller_json(contract_path, "E0 execution contract")
    if (contract.get("freeze_id") != freeze_id
            or contract.get("code", {}).get("commit") != commit
            or contract.get("code", {}).get("dirty") is not False):
        raise ValueError("E0 execution code/freeze binding differs from executing source")
    digest = canonical_hash({key: value for key, value in contract.items()
                             if key not in {"created_utc", "environment", "contract_sha256"}})
    if contract.get("contract_sha256") != digest:
        raise ValueError("E0 execution contract digest differs")
    for config_path in config_paths:
        path = _checked_code_path(config_path, "executing config")
        matches = [row for row in contract.get("resource_inventory", [])
                   if row.get("resolved_path") == str(path)
                   and row.get("hash_method") == "content_sha256"]
        if not matches or any(row.get("sha256") != sha256_file(path) for row in matches):
            raise ValueError(f"E0 config bytes differ from executing source: {path}")
    if require_inventory:
        jobs, _ = _load_inventory(out_dir, controller_safe=True)
        source = jobs["source_contract"]
        if (jobs["freeze_id"] != freeze_id or source["code_commit"] != commit
                or source["contract_freeze_id"] != freeze_id
                or source["contract_sha256"] != digest):
            raise ValueError("sealed inventory differs from executing code/E0 contract")
    return contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.evaluation_manifest) != bool(args.evaluation_manifest_sha256):
        parser.error('evaluation manifest requires an explicit frozen SHA256')
    if args.evaluation_manifest and not args.evaluate:
        parser.error('GT reference manifest is allowed only in evaluation')
    if args.evaluation_root and not args.aggregate:
        parser.error('--evaluation-root is aggregate-only')
    if args.uncertainty_config and (not args.aggregate or not args.evaluation_root):
        parser.error('--uncertainty-config requires external aggregation')
    scene_phase = args.observe or args.control or args.evaluate
    if scene_phase != bool(args.scene_id):
        parser.error("--scene-id is required exactly for --observe/--control/--evaluate")
    if not FREEZE_ID_RE.fullmatch(args.freeze_id):
        parser.error(f"unsafe freeze ID: {args.freeze_id!r}")
    out_dir = checked_repo_path(args.out, "E3 output root", must_exist=False)
    expected_out = checked_repo_path(
        f"outputs/icra2027/{args.freeze_id}/agentic",
        "fixed E3 output root",
        must_exist=False,
    )
    if out_dir != expected_out:
        parser.error(f"--out must be exactly {expected_out}")

    if args.inventory or args.observe or args.control:
        config_paths = ([args.jobs, args.policies] if args.inventory else
                        [args.policies] if args.control else [])
        execution_contract = _validate_cli_execution(
            args.contract_manifest, args.freeze_id, out_dir,
            config_paths=config_paths, require_inventory=not args.inventory,
        )

    # Phase-aware loading is part of the GT isolation contract.  In
    # particular, observe/control/evaluate must not open the inventory-only
    # jobs YAML, which contains report membership fields.
    if args.inventory:
        jobs, policies, contract, out_dir = _validate_context(
            args.jobs,
            args.policies,
            args.contract_manifest,
            args.freeze_id,
            args.out,
        )
        result = run_inventory(jobs, contract, args.freeze_id, out_dir)
    elif args.observe:
        result = run_observe(args.freeze_id, out_dir, args.scene_id)
    elif args.control:
        policies = _load_yaml_code(args.policies, "agentic policies config")
        validate_policy_config(policies)
        _validate_probe_protocol(policies)
        contract = execution_contract
        # Defensive belt-and-suspenders guard for lazy imports in runtime.py.
        os.environ["SIMANY_NO_GT"] = "1"
        result = run_control(policies, contract, args.freeze_id, out_dir, args.scene_id)
    elif args.evaluate:
        result = run_evaluate(args.freeze_id, out_dir, args.scene_id,
                              evaluation_manifest=args.evaluation_manifest,
                              evaluation_manifest_sha256=args.evaluation_manifest_sha256)
    elif args.aggregate:
        jobs, policies, contract, out_dir = _validate_context(
            args.jobs,
            args.policies,
            args.contract_manifest,
            args.freeze_id,
            args.out,
        )
        evaluation_root = None
        uncertainty_protocol = None
        if "automatic_sources" in jobs and not args.evaluation_root:
            _validate_cli_execution(args.contract_manifest, args.freeze_id, out_dir,
                                    config_paths=[args.jobs, args.policies], require_inventory=True)
        if args.evaluation_root:
            evaluation_root = checked_repo_path(args.evaluation_root, 'evaluation aggregate root', kind='dir')
            evaluation_freeze = evaluation_root.parent.name
            if (evaluation_root != REPOSITORY_ROOT / 'outputs/icra2027' / evaluation_freeze / 'agentic'
                    or evaluation_root == out_dir):
                raise ValueError('external aggregation requires separate fixed freeze root')
            _validate_cli_execution(args.contract_manifest, evaluation_freeze, evaluation_root,
                                    config_paths=[args.jobs, args.policies] +
                                    ([args.uncertainty_config] if args.uncertainty_config else []))
        if args.uncertainty_config:
            uncertainty_config = _load_yaml_code(args.uncertainty_config, 'agentic uncertainty config')
            if (uncertainty_config.get('freeze_id') != evaluation_root.parent.name
                    or uncertainty_config.get('mode') != 'full'
                    or uncertainty_config.get('planned_scenes') != 50
                    or uncertainty_config.get('planned_jobs') != 1871):
                raise ValueError('uncertainty requires the complete independent evaluation configuration')
            uncertainty_protocol = uncertainty_config.get('agentic_uncertainty')
            if uncertainty_protocol != AGENTIC_UNCERTAINTY_PROTOCOL:
                raise ValueError('uncertainty protocol is missing or changed')
        result = run_aggregate(jobs, policies, args.freeze_id, out_dir,
                               evaluation_root=evaluation_root, uncertainty_protocol=uncertainty_protocol)
    else:
        policies = _load_yaml_code(args.policies, "agentic policies config")
        validate_policy_config(policies)
        _validate_probe_protocol(policies)
        result = run_smoke(policies, args.freeze_id, out_dir)
    summary = {
        "phase": next(
            name
            for name in ("inventory", "observe", "control", "evaluate", "aggregate", "smoke")
            if getattr(args, name)
        ),
        "freeze_id": args.freeze_id,
        "scene_id": args.scene_id,
        "result_sha256": _canonical_digest(result),
    }
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
