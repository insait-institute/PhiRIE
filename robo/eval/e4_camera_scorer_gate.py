"""One-GPU, fail-closed camera/workspace/scorer gate for the E4 pilot.

This gate consumes (but never mutates) the sealed two-room CPU prerequisite
freeze produced by :mod:`robo.eval.e4_region_pilot`.  It deliberately keeps
the older CPU producer commit distinct from the clean commit executing this
validator.  The CPU preflight seal is anchored by an expected SHA-256, then
the scene gates, paired task bundles, materializations, external geometry,
and complete ``sim``/``sim_export`` closures are revalidated before MuJoCo is
allowed to construct a renderer.

The runtime traverses exactly 2 scenes x 2 construction arms x 2 tasks x 5
resets.  Every cell receives real EGL RGB and segmentation renders from the
exterior and wrist cameras, reach-envelope diagnostics, language
disambiguation replay, and a direct-state replay of ``TaskScorer``.  Direct
state replay validates scorer predicates only; it is not an IK, motion
planning, policy, or task oracle.  Consequently, even a passing gate can only
authorize one *real-policy infrastructure smoke*.  It can never make the
experiment paper-ready or authorize a large rollout fleet.

The output directory is fresh-only and published by one atomic rename.  A
semantic gate failure (for example a <=30 mm qualifier margin) is still
published with all 40 diagnostic cells and the CLI exits non-zero.  An input,
build, or rendering exception removes the staging directory and publishes no
partial gate.
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from robo.eval import e4_region_pilot as cpu_pilot


SCHEMA_VERSION = 1
CODE_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EVIDENCE_ROOT = Path(os.environ.get("SIMANY_EXPECTED_EVIDENCE_ROOT", "/opt/phirie/evidence/SimAny"))
EXPECTED_SLURM_ACCOUNT = os.environ.get("SIMANY_EXPECTED_SLURM_ACCOUNT", "phirie")
EXPECTED_CPU_FREEZE_ID = (
    "icra2027-contract-v1-e4-0dda134578b2-region-cpu-prep1-20260904T135257Z"
)
EXPECTED_CPU_PRODUCER_COMMIT = "0dda134578b25cd12d1791193bb437c6d310776f"
EXPECTED_CPU_GATE_SHA256 = "b4bb242ace4ce641051e58df003d126a1383ae0ddd003e3fe4dce3e0493ef355"
EXPECTED_CPU_CONFIG_SHA256 = "fea0c39c9fd789122e75721d9283f1d1c254835e8640c67267271d785d9f5b26"
EXPECTED_MENAGERIE_ROOT = CODE_ROOT / "third_party" / "mujoco_menagerie"
MENAGERIE_COPY_SOURCE_ROOT = EXPECTED_EVIDENCE_ROOT / "third_party" / "mujoco_menagerie"
EXPECTED_MENAGERIE_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"
EXPECTED_MENAGERIE_CLOSURE_SHA256 = (
    "a275b73fb3d0b5fc52abbeb1cd58577c01a447c13ff5ba8d3f51199f27e7e46f"
)
EXPECTED_MENAGERIE_FILE_COUNT = 94
EXPECTED_MENAGERIE_SIZE_BYTES = 40_901_071
EXPECTED_OPENPI_ROOT = Path(os.environ.get("SIMANY_EXPECTED_OPENPI_ROOT", "/opt/phirie/evidence/openpi-wt/e4-policy-server"))
EXPECTED_OPENPI_COMMIT = "2f51088169d2e2b480ce54faa737be9b0279eed4"
EXPECTED_OPENPI_IMAGE_TOOLS_SHA256 = (
    "d48b4bd7f44e79fe6db8a8e07c9161144fa250be686e1245014a8b47e6171977"
)
OPENPI_IMAGE_TOOLS = (
    EXPECTED_OPENPI_ROOT
    / "packages"
    / "openpi-client"
    / "src"
    / "openpi_client"
    / "image_tools.py"
)
EXPECTED_GPU_NODE = "gcp-eu1-a100-80g-qrfh"
EXPECTED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
EXPECTED_GPU_COMPUTE_CAPABILITY = "8.0"
SCENE_IDS = cpu_pilot.SCENE_IDS
POLICIES = cpu_pilot.POLICIES
POLICY_TO_VARIANT = {"A0": "fixed_single_path", "A4": "agentic"}
POLICY_TO_TASK_FILE = {"A0": "a0_tasks.json", "A4": "a4_tasks.json"}
CAMERAS = ("exterior", "wrist")
EPISODES = 5
BASE_SEED = 0
JITTER_XY_M = 0.01
RENDER_WIDTH = 640
RENDER_HEIGHT = 360
REACH_MIN_M = 0.25
REACH_MAX_M = 0.80
FRONT_CONE_DEG = 60.0
QUALIFIER_MARGIN_M = 0.03
POLICY_INPUT_SIZE = 224
MIN_POLICY_OBJECT_PIXELS = 9
MIN_POLICY_BBOX_SIDE = 3
MAX_WRIST_ROBOT_FRACTION = 0.50
MIN_RGB_DYNAMIC_RANGE = 32
MIN_RGB_STD = 5.0
MIN_SAMPLED_UNIQUE_COLORS = 32
FREEZE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QUALIFIERS = ("leftmost", "rightmost", "nearest", "farthest")


class CameraScorerGateError(RuntimeError):
    """An upstream identity, runtime contract, or rendering operation failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, root: Path) -> dict[str, Any]:
    source = _regular_file(path, root=root, label="artifact")
    return {
        "path": source.relative_to(root).as_posix(),
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
    }


def _identity_without_path(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CameraScorerGateError(f"not a regular non-symlink file: {path}")
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _validate_identity(path: Path, recorded: Any, *, label: str) -> None:
    if not isinstance(recorded, Mapping) or set(recorded) != {"sha256", "size_bytes"}:
        raise CameraScorerGateError(f"{label} identity schema differs")
    digest = recorded.get("sha256")
    size = recorded.get("size_bytes")
    if (
        not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise CameraScorerGateError(f"{label} identity is invalid")
    if _identity_without_path(path) != dict(recorded):
        raise CameraScorerGateError(f"{label} bytes changed")


def _canonical_absolute_directory(value: str | Path, *, label: str) -> Path:
    try:
        spelling = os.fspath(value)
    except TypeError as exc:
        raise CameraScorerGateError(f"{label} must be a canonical absolute path") from exc
    pure = PurePosixPath(spelling)
    if (
        not isinstance(spelling, str)
        or not spelling
        or "\\" in spelling
        or not pure.is_absolute()
        or spelling != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise CameraScorerGateError(f"{label} must be a canonical absolute POSIX path")
    path = Path(spelling)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise CameraScorerGateError(f"{label} contains a symlink: {current}")
    if not path.is_dir() or path.resolve(strict=True) != path:
        raise CameraScorerGateError(f"{label} is not a canonical directory: {path}")
    return path


def _inside(path: Path, *, root: Path, label: str) -> Path:
    raw = os.fspath(path)
    pure = PurePosixPath(raw)
    if (
        not isinstance(raw, str)
        or not raw
        or "\\" in raw
        or raw != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise CameraScorerGateError(f"{label} is not a canonical POSIX path")
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise CameraScorerGateError(f"{label} escapes the evidence root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise CameraScorerGateError(f"{label} contains a symlink: {current}")
    return candidate


def _regular_file(path: Path, *, root: Path, label: str) -> Path:
    candidate = _inside(path, root=root, label=label)
    if (
        not candidate.is_file()
        or candidate.is_symlink()
        or candidate.resolve(strict=True) != candidate
    ):
        raise CameraScorerGateError(f"{label} is not a canonical regular file: {candidate}")
    return candidate


def _regular_directory(path: Path, *, root: Path, label: str) -> Path:
    candidate = _inside(path, root=root, label=label)
    if (
        not candidate.is_dir()
        or candidate.is_symlink()
        or candidate.resolve(strict=True) != candidate
    ):
        raise CameraScorerGateError(f"{label} is not a canonical directory: {candidate}")
    return candidate


def _read_json(path: Path, *, root: Path, label: str) -> Any:
    source = _regular_file(path, root=root, label=label)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraScorerGateError(f"{label} is not valid JSON: {exc}") from exc


def _git_snapshot(expected_commit: str) -> dict[str, Any]:
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise CameraScorerGateError("camera validator commit must be a full lowercase SHA")
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
        raise CameraScorerGateError("cannot obtain camera-validator Git snapshot") from exc
    if Path(top).resolve(strict=True) != CODE_ROOT:
        raise CameraScorerGateError("camera validator is not rooted at CODE_ROOT")
    if head != expected_commit:
        raise CameraScorerGateError(
            f"camera validator commit differs: expected={expected_commit}, observed={head}"
        )
    if status:
        raise CameraScorerGateError(f"camera validator worktree is dirty: {status!r}")
    return {"code_root": str(CODE_ROOT), "commit": head, "dirty": False}


def _tree_inventory(directory: Path, *, relative_to: Path) -> list[dict[str, Any]]:
    """Hash the exact regular-file closure and reject links/special files."""
    if not directory.is_dir() or directory.is_symlink():
        raise CameraScorerGateError(f"tree is not a regular directory: {directory}")
    inventory: list[dict[str, Any]] = []

    def visit(current: Path) -> None:
        for child in sorted(current.iterdir(), key=lambda item: item.name):
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise CameraScorerGateError(f"tree contains a symlink: {child}")
            if stat.S_ISDIR(mode):
                visit(child)
            elif stat.S_ISREG(mode):
                inventory.append(
                    {
                        "path": child.relative_to(relative_to).as_posix(),
                        "sha256": _sha256(child),
                        "size_bytes": child.stat().st_size,
                    }
                )
            else:
                raise CameraScorerGateError(f"tree contains a special file: {child}")

    visit(directory)
    if not inventory:
        raise CameraScorerGateError(f"tree contains no files: {directory}")
    return inventory


def _menagerie_snapshot(root: Path, expected_commit: str) -> dict[str, Any]:
    root = _canonical_absolute_directory(root, label="menagerie root")
    if root != EXPECTED_MENAGERIE_ROOT:
        raise CameraScorerGateError(
            f"menagerie root differs: expected={EXPECTED_MENAGERIE_ROOT}, observed={root}"
        )
    if expected_commit != EXPECTED_MENAGERIE_COMMIT:
        raise CameraScorerGateError("menagerie commit differs from the frozen gate contract")
    # The runtime tree is an ignored byte-for-byte copy inside the clean E4
    # validator worktree.  It is intentionally not a symlink and not a nested
    # Git checkout.  Its full closure digest, not ambient main-checkout state,
    # is the runtime authority.  The source path/commit below is provenance for
    # how that immutable copy was populated.
    required = {
        "franka_emika_panda": root / "franka_emika_panda",
        "robotiq_2f85": root / "robotiq_2f85",
    }
    files = [
        row
        for name in sorted(required)
        for row in _tree_inventory(required[name], relative_to=root)
    ]
    required_xml = {
        "franka_emika_panda/panda_nohand.xml",
        "robotiq_2f85/2f85.xml",
    }
    if not required_xml <= {row["path"] for row in files}:
        raise CameraScorerGateError("menagerie closure lacks a required rig XML")
    closure_sha256 = _canonical_hash(files)
    size_bytes = sum(row["size_bytes"] for row in files)
    if (
        len(files) != EXPECTED_MENAGERIE_FILE_COUNT
        or size_bytes != EXPECTED_MENAGERIE_SIZE_BYTES
        or closure_sha256 != EXPECTED_MENAGERIE_CLOSURE_SHA256
    ):
        raise CameraScorerGateError(
            "runtime menagerie copy differs from its frozen recursive closure"
        )
    return {
        "access_mode": "read_only_input; gate performs no writes",
        "closure_sha256": closure_sha256,
        "file_count": len(files),
        "files": files,
        "root": str(root),
        "size_bytes": size_bytes,
        "source_commit": expected_commit,
        "source_root": str(MENAGERIE_COPY_SOURCE_ROOT),
    }


def _openpi_snapshot() -> dict[str, Any]:
    root = _canonical_absolute_directory(EXPECTED_OPENPI_ROOT, label="OpenPI root")
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True
        ).strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CameraScorerGateError("cannot obtain OpenPI Git snapshot") from exc
    image_tools = OPENPI_IMAGE_TOOLS
    if (
        Path(top).resolve(strict=True) != root
        or head != EXPECTED_OPENPI_COMMIT
        or status
        or not image_tools.is_file()
        or image_tools.is_symlink()
        or image_tools.resolve(strict=True) != image_tools
        or _sha256(image_tools) != EXPECTED_OPENPI_IMAGE_TOOLS_SHA256
    ):
        raise CameraScorerGateError(
            f"OpenPI resize source differs: root={top!r}, commit={head!r}, status={status!r}"
        )
    return {
        "commit": head,
        "dirty": False,
        "image_tools": {
            "path": str(image_tools),
            "sha256": EXPECTED_OPENPI_IMAGE_TOOLS_SHA256,
            "size_bytes": image_tools.stat().st_size,
        },
        "root": str(root),
    }
def _evidence_root() -> Path:
    raw = os.environ.get("SIMANY_EVIDENCE_ROOT")
    if raw is None:
        raise CameraScorerGateError("SIMANY_EVIDENCE_ROOT is required")
    root = _canonical_absolute_directory(raw, label="SIMANY_EVIDENCE_ROOT")
    if root != EXPECTED_EVIDENCE_ROOT or root == CODE_ROOT:
        raise CameraScorerGateError("camera gate requires the distinct sealed evidence root")
    return root


def _validated_id(value: str, *, label: str) -> str:
    if FREEZE_ID_RE.fullmatch(value) is None:
        raise CameraScorerGateError(f"{label} contains unsafe path characters")
    return value


def _experiment_root(root: Path, freeze_id: str) -> Path:
    return root / "outputs" / "icra2027" / _validated_id(freeze_id, label="freeze ID")


@contextmanager
def _atomic_directory(path: Path, *, preserve_failed: bool = False) -> Iterator[Path]:
    if path.exists() or path.is_symlink():
        raise CameraScorerGateError(f"refusing to reuse output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.tmp.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise CameraScorerGateError(f"staging directory already exists: {staging}")
    staging.mkdir(mode=0o700)
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
            if preserve_failed:
                os.replace(staging, path.parent / f".{path.name}.failed.{os.getpid()}")
            else:
                shutil.rmtree(staging)
        raise


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            handle.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_path_identity(
    recorded: Any, *, root: Path, label: str
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(recorded, Mapping) or set(recorded) != {
        "path", "sha256", "size_bytes"
    }:
        raise CameraScorerGateError(f"{label} path identity schema differs")
    path_value = recorded.get("path")
    if not isinstance(path_value, str):
        raise CameraScorerGateError(f"{label} path is not text")
    path = _regular_file(Path(path_value), root=root, label=label)
    current = _identity(path, root=root)
    if current != dict(recorded):
        raise CameraScorerGateError(f"{label} path identity changed")
    return path, current


def _validate_materialization_chain(
    factory: Path,
    *,
    root: Path,
    scene_id: str,
    policy: str,
    recorded_summary: Any,
    cpu_commit: str,
) -> dict[str, Any]:
    if not isinstance(recorded_summary, Mapping):
        raise CameraScorerGateError(f"{scene_id}/{policy} materialization summary missing")
    factory = _regular_directory(
        factory, root=root, label=f"{scene_id}/{policy} materialized factory"
    )
    manifest_path = _regular_file(
        factory / "materialization_manifest.json",
        root=root,
        label=f"{scene_id}/{policy} materialization manifest",
    )
    seal_path = _regular_file(
        factory / "seal.json", root=root, label=f"{scene_id}/{policy} materialization seal"
    )
    seal = _read_json(
        seal_path, root=root, label=f"{scene_id}/{policy} materialization seal"
    )
    if (
        not isinstance(seal, Mapping)
        or set(seal) != {"manifest_kind", "members", "schema_version"}
        or seal.get("manifest_kind") != "e4_construction_variant_materialization"
        or seal.get("schema_version") != 1
        or not isinstance(seal.get("members"), Mapping)
        or set(seal["members"]) != {"materialization_manifest.json"}
    ):
        raise CameraScorerGateError(f"{scene_id}/{policy} materialization seal differs")
    _validate_identity(
        manifest_path,
        seal["members"]["materialization_manifest.json"],
        label=f"{scene_id}/{policy} materialization manifest",
    )
    manifest = _read_json(
        manifest_path, root=root, label=f"{scene_id}/{policy} materialization manifest"
    )
    if not isinstance(manifest, Mapping):
        raise CameraScorerGateError(f"{scene_id}/{policy} materialization is not a mapping")
    if (
        manifest.get("manifest_kind") != "e4_construction_variant_materialization"
        or manifest.get("schema_version") != 1
        or manifest.get("scene_id") != scene_id
        or manifest.get("policy_id") != policy
        or manifest.get("destination") != factory.relative_to(root).as_posix()
    ):
        raise CameraScorerGateError(f"{scene_id}/{policy} materialization binding differs")
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("code_root") != str(CODE_ROOT)
        or provenance.get("evidence_root") != str(root)
        or provenance.get("materializer_commit") != cpu_commit
        or provenance.get("validator_commit") != cpu_commit
        or provenance.get("materializer_dirty") is not False
        or provenance.get("validator_dirty") is not False
    ):
        raise CameraScorerGateError(f"{scene_id}/{policy} materialization provenance differs")
    manifest_sha = _sha256(manifest_path)
    expected_summary = {
        "e3_claim_status": manifest.get("e3_claim_status"),
        "e3_code_commit": manifest.get("e3_code_commit"),
        "e3_freeze_id": manifest.get("e3_freeze_id"),
        "e3_root": manifest.get("e3_root"),
        "manifest_sha256": manifest_sha,
        "materializer_commit": provenance.get("materializer_commit"),
        "policy_id": manifest.get("policy_id"),
        "roster": manifest.get("roster"),
        "scene_id": manifest.get("scene_id"),
        "study_scope": manifest.get("study_scope"),
        "validator_commit": provenance.get("validator_commit"),
    }
    if dict(recorded_summary) != expected_summary:
        raise CameraScorerGateError(
            f"{scene_id}/{policy} materialization differs from its scene gate"
        )

    input_identities = manifest.get("input_identities")
    if not isinstance(input_identities, Mapping) or not input_identities:
        raise CameraScorerGateError(f"{scene_id}/{policy} input identities missing")
    for name, identity in sorted(input_identities.items()):
        _validate_path_identity(
            identity, root=root, label=f"{scene_id}/{policy} materialization input {name}"
        )

    members = manifest.get("output_members")
    if not isinstance(members, Mapping) or not members:
        raise CameraScorerGateError(f"{scene_id}/{policy} output closure missing")
    member_paths: set[str] = set()
    for relative, identity in sorted(members.items()):
        if not isinstance(relative, str):
            raise CameraScorerGateError("materialization member path is not text")
        path = _regular_file(
            Path(relative), root=factory, label=f"{scene_id}/{policy} materialization member"
        )
        _validate_identity(
            path, identity, label=f"{scene_id}/{policy} materialization member {relative}"
        )
        member_paths.add(relative)

    # sim/ and sim_export/ were intentionally generated by the later CPU
    # scene job.  Everything else must still be the exact materializer
    # closure plus its two seal files -- no unsealed source-side additions.
    actual_source_files: set[str] = set()
    for current, directories, filenames in os.walk(factory, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(factory)
        if relative_dir.parts and relative_dir.parts[0] in {"sim", "sim_export"}:
            directories[:] = []
            continue
        for directory in list(directories):
            child = current_path / directory
            if child.is_symlink():
                raise CameraScorerGateError(f"materialization source has symlink: {child}")
        for filename in filenames:
            child = current_path / filename
            if child.is_symlink() or not child.is_file():
                raise CameraScorerGateError(f"materialization source has special file: {child}")
            relative = child.relative_to(factory).as_posix()
            if relative not in {"materialization_manifest.json", "seal.json"}:
                actual_source_files.add(relative)
    if actual_source_files != member_paths:
        raise CameraScorerGateError(
            f"{scene_id}/{policy} materialization source closure changed"
        )
    return {
        "manifest": _identity(manifest_path, root=root),
        "output_member_count": len(member_paths),
        "roster": expected_summary["roster"],
    }


def _validate_task_bundle_chain(
    bundle: Path,
    *,
    root: Path,
    scene_id: str,
    expected_manifest_sha256: str,
    cpu_commit: str,
    materializations: Mapping[str, Mapping[str, Any]],
    expected_task_ids: Sequence[str] | None = None,
    expected_max_tasks: int | None = 2,
) -> dict[str, Any]:
    task_ids = list(cpu_pilot.expected_task_ids(scene_id)) if expected_task_ids is None else list(expected_task_ids)
    bundle = _regular_directory(bundle, root=root, label=f"{scene_id} task bundle")
    actual_names = sorted(path.name for path in bundle.iterdir())
    if actual_names != [
        "a0_tasks.json",
        "a4_tasks.json",
        "manifest.json",
        "planning_tasks.json",
        "seal.json",
    ]:
        raise CameraScorerGateError(f"{scene_id} task bundle closure differs")
    manifest_path = _regular_file(
        bundle / "manifest.json", root=root, label=f"{scene_id} task manifest"
    )
    seal_path = _regular_file(bundle / "seal.json", root=root, label=f"{scene_id} task seal")
    seal = _read_json(seal_path, root=root, label=f"{scene_id} task seal")
    if (
        not isinstance(seal, Mapping)
        or set(seal) != {"manifest_kind", "members", "schema_version"}
        or seal.get("manifest_kind") != "e4_paired_task_freeze"
        or seal.get("schema_version") != 1
        or not isinstance(seal.get("members"), Mapping)
        or set(seal["members"]) != {"manifest.json"}
    ):
        raise CameraScorerGateError(f"{scene_id} task seal differs")
    _validate_identity(manifest_path, seal["members"]["manifest.json"], label="task manifest")
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise CameraScorerGateError(f"{scene_id} task manifest differs from CPU gate")
    manifest = _read_json(manifest_path, root=root, label=f"{scene_id} task manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("manifest_kind") != "e4_paired_task_freeze"
        or manifest.get("schema_version") != 1
        or manifest.get("scene_id") != scene_id
        or manifest.get("planning_source") != "A4"
        or manifest.get("max_tasks") != expected_max_tasks
        or manifest.get("logical_task_ids") != task_ids
    ):
        raise CameraScorerGateError(f"{scene_id} task manifest binding differs")
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("code_root") != str(CODE_ROOT)
        or provenance.get("evidence_root") != str(root)
        or provenance.get("freezer_commit") != cpu_commit
        or provenance.get("validator_commit") != cpu_commit
        or provenance.get("freezer_dirty") is not False
        or provenance.get("validator_dirty") is not False
    ):
        raise CameraScorerGateError(f"{scene_id} task provenance differs")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "planning_tasks.json", "a0_tasks.json", "a4_tasks.json"
    }:
        raise CameraScorerGateError(f"{scene_id} task file inventory differs")
    paths: dict[str, Path] = {}
    for name, identity in files.items():
        path = _regular_file(bundle / name, root=root, label=f"{scene_id} {name}")
        _validate_identity(path, identity, label=f"{scene_id} {name}")
        paths[name] = path

    candidate_suites = manifest.get("candidate_suites")
    if not isinstance(candidate_suites, Mapping) or set(candidate_suites) != set(POLICIES):
        raise CameraScorerGateError(f"{scene_id} candidate-suite inventory differs")
    for policy, identity in candidate_suites.items():
        _validate_path_identity(
            identity, root=root, label=f"{scene_id}/{policy} candidate task suite"
        )

    factories = manifest.get("factories")
    if not isinstance(factories, Mapping) or set(factories) != set(POLICIES):
        raise CameraScorerGateError(f"{scene_id} task factory inventory differs")
    suites = {
        "planning": _read_json(paths["planning_tasks.json"], root=root, label="planning suite"),
        "A0": _read_json(paths["a0_tasks.json"], root=root, label="A0 suite"),
        "A4": _read_json(paths["a4_tasks.json"], root=root, label="A4 suite"),
    }
    suite_keys = {
        "exclude_objects", "ext_cam", "robot", "scene", "scene_xml",
        "table", "tasks", "time_limit_s",
    }
    for name, suite in suites.items():
        if not isinstance(suite, Mapping) or set(suite) != suite_keys:
            raise CameraScorerGateError(f"{scene_id}/{name} task suite schema differs")
        if suite.get("scene") != scene_id:
            raise CameraScorerGateError(f"{scene_id}/{name} suite scene differs")
    if suites["planning"].get("scene_xml") is not None:
        raise CameraScorerGateError(f"{scene_id} planning suite scene_xml is not null")
    shared = {
        name: {key: value for key, value in suite.items() if key != "scene_xml"}
        for name, suite in suites.items()
    }
    if shared["planning"] != shared["A0"] or shared["planning"] != shared["A4"]:
        raise CameraScorerGateError(f"{scene_id} A0/A4 task semantics are not paired")
    logical_ids = [task.get("task_id") for task in suites["planning"]["tasks"]]
    if logical_ids != task_ids:
        raise CameraScorerGateError(f"{scene_id} task order differs")

    variant_paths = manifest.get("variant_task_paths")
    if not isinstance(variant_paths, Mapping) or set(variant_paths) != set(POLICIES):
        raise CameraScorerGateError(f"{scene_id} variant task paths differ")
    for policy in POLICIES:
        declared, _ = _validate_path_identity(
            {
                "path": variant_paths[policy],
                **files[POLICY_TO_TASK_FILE[policy]],
            },
            root=root,
            label=f"{scene_id}/{policy} frozen suite",
        )
        if declared != paths[POLICY_TO_TASK_FILE[policy]]:
            raise CameraScorerGateError(f"{scene_id}/{policy} frozen suite path differs")
        factory_row = factories.get(policy)
        if not isinstance(factory_row, Mapping):
            raise CameraScorerGateError(f"{scene_id}/{policy} task factory row missing")
        expected_factory = _regular_directory(
            Path(factory_row.get("factory_dir", "")),
            root=root,
            label=f"{scene_id}/{policy} task factory",
        )
        materialization = materializations[policy]
        if (
            factory_row.get("policy_id") != policy
            or factory_row.get("materialization_manifest_sha256")
            != materialization["manifest"]["sha256"]
        ):
            raise CameraScorerGateError(f"{scene_id}/{policy} task factory binding differs")
        scene_identity = factory_row.get("scene_xml")
        scene_xml, _ = _validate_path_identity(
            scene_identity, root=root, label=f"{scene_id}/{policy} scene XML"
        )
        if expected_factory not in scene_xml.parents:
            raise CameraScorerGateError(f"{scene_id}/{policy} scene XML escapes factory")
        if suites[policy]["scene_xml"] != str(scene_xml):
            raise CameraScorerGateError(f"{scene_id}/{policy} suite scene XML differs")
    return {
        "manifest": _identity(manifest_path, root=root),
        "suites": suites,
    }


def validate_cpu_chain(
    *,
    root: Path,
    cpu_freeze_id: str,
    cpu_producer_commit: str,
    expected_cpu_gate_sha256: str,
) -> dict[str, Any]:
    """Revalidate the anchored older CPU freeze using the current validator."""
    if cpu_freeze_id != EXPECTED_CPU_FREEZE_ID:
        raise CameraScorerGateError("CPU freeze differs from the pinned two-room pilot")
    if cpu_producer_commit != EXPECTED_CPU_PRODUCER_COMMIT:
        raise CameraScorerGateError("CPU producer commit differs from the pinned pilot")
    if expected_cpu_gate_sha256 != EXPECTED_CPU_GATE_SHA256:
        raise CameraScorerGateError("CPU gate SHA differs from the external anchor")
    cpu_root = _regular_directory(
        _experiment_root(root, cpu_freeze_id), root=root, label="CPU experiment root"
    )
    preflight = _regular_directory(
        cpu_root / "preflight", root=root, label="CPU preflight directory"
    )
    if sorted(path.name for path in preflight.iterdir()) != [
        "gate.json", "harness_config.json", "seal.json"
    ]:
        raise CameraScorerGateError("CPU preflight directory closure differs")
    gate_path = _regular_file(preflight / "gate.json", root=root, label="CPU gate")
    config_path = _regular_file(
        preflight / "harness_config.json", root=root, label="CPU harness config"
    )
    seal_path = _regular_file(preflight / "seal.json", root=root, label="CPU seal")
    if _sha256(gate_path) != expected_cpu_gate_sha256:
        raise CameraScorerGateError("CPU gate bytes differ from expected anchor")
    if _sha256(config_path) != EXPECTED_CPU_CONFIG_SHA256:
        raise CameraScorerGateError("CPU harness config bytes differ from expected anchor")
    seal = _read_json(seal_path, root=root, label="CPU preflight seal")
    if (
        not isinstance(seal, Mapping)
        or set(seal) != {"manifest_kind", "members", "schema_version"}
        or seal.get("manifest_kind") != "e4_region_cpu_preflight"
        or seal.get("schema_version") != 1
        or not isinstance(seal.get("members"), Mapping)
        or set(seal["members"]) != {"gate.json", "harness_config.json"}
    ):
        raise CameraScorerGateError("CPU preflight seal schema differs")
    _validate_identity(gate_path, seal["members"]["gate.json"], label="CPU gate")
    _validate_identity(
        config_path, seal["members"]["harness_config.json"], label="CPU harness config"
    )
    gate = _read_json(gate_path, root=root, label="CPU gate")
    config = _read_json(config_path, root=root, label="CPU harness config")
    if not isinstance(gate, Mapping) or not isinstance(config, Mapping):
        raise CameraScorerGateError("CPU preflight payload is not a mapping")
    cpu_code = gate.get("code")
    if (
        not isinstance(cpu_code, Mapping)
        or cpu_code.get("code_root") != str(CODE_ROOT)
        or cpu_code.get("commit") != cpu_producer_commit
        or cpu_code.get("dirty") is not False
        or gate.get("freeze_id") != cpu_freeze_id
        or gate.get("evidence_root") != str(root)
        or gate.get("cpu_prerequisites_ok") is not True
        or gate.get("gpu_launch_allowed") is not False
        or gate.get("paper_ready") is not False
        or gate.get("headline_eligible") is not False
        or gate.get("study_scope") != "region_only_engineering_pilot"
    ):
        raise CameraScorerGateError("CPU gate provenance or eligibility binding differs")
    if (
        config.get("schema_version") != 1
        or config.get("dry_run_only") is not True
        or config.get("gpu_launch_allowed") is not False
        or config.get("paper_mode") is not False
        or config.get("episodes") != EPISODES
        or config.get("seeds") != [BASE_SEED]
        or config.get("jitter") != JITTER_XY_M
        or config.get("study_scope") != "region_only_engineering_pilot"
    ):
        raise CameraScorerGateError("CPU harness top-level contract differs")
    contract = config.get("contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("cameras", {}).get("resolution")
        != [RENDER_WIDTH, RENDER_HEIGHT]
        or contract.get("rubric", {}).get("stages")
        != ["grasp", "lift", "hover", "place"]
        or contract.get("rubric", {}).get("credit_per_stage") != 0.25
        or contract.get("rubric", {}).get("success_definition") != "place held for 1.0s"
    ):
        raise CameraScorerGateError("CPU camera/rubric contract differs")
    scene_cfgs = config.get("scenes")
    if (
        not isinstance(scene_cfgs, list)
        or [row.get("id") for row in scene_cfgs if isinstance(row, Mapping)]
        != list(SCENE_IDS)
    ):
        raise CameraScorerGateError("CPU harness scene roster differs")

    all_suites: dict[str, dict[str, Mapping[str, Any]]] = {}
    all_factories: dict[str, dict[str, Path]] = {}
    materialization_identities: dict[str, dict[str, Any]] = {}
    task_manifest_identities: dict[str, dict[str, Any]] = {}
    current_exports: dict[str, dict[str, Any]] = {}
    for scene_id in SCENE_IDS:
        scene_gate_identity = gate.get("scene_gate_identities", {}).get(scene_id)
        scene_gate_path, _ = _validate_path_identity(
            scene_gate_identity, root=root, label=f"{scene_id} CPU scene gate"
        )
        scene_gate = _read_json(scene_gate_path, root=root, label=f"{scene_id} scene gate")
        if (
            not isinstance(scene_gate, Mapping)
            or scene_gate.get("freeze_id") != cpu_freeze_id
            or scene_gate.get("scene_id") != scene_id
            or scene_gate.get("evidence_root") != str(root)
            or scene_gate.get("code") != dict(cpu_code)
            or scene_gate.get("paper_ready") is not False
            or scene_gate.get("expected_task_ids")
            != list(cpu_pilot.expected_task_ids(scene_id))
        ):
            raise CameraScorerGateError(f"{scene_id} scene gate binding differs")
        current_external = cpu_pilot.external_geometry_identities(scene_id)
        if (
            scene_gate.get("external_geometry_inputs") != current_external
            or gate.get("external_geometry_inputs", {}).get(scene_id) != current_external
        ):
            raise CameraScorerGateError(f"{scene_id} external geometry changed")

        all_factories[scene_id] = {}
        materializations: dict[str, dict[str, Any]] = {}
        current_exports[scene_id] = {}
        for policy in POLICIES:
            factory = (
                cpu_root / "construction_variants" / policy / f"{scene_id}_factory"
            )
            all_factories[scene_id][policy] = factory
            materialization = _validate_materialization_chain(
                factory,
                root=root,
                scene_id=scene_id,
                policy=policy,
                recorded_summary=scene_gate.get("materializations", {}).get(policy),
                cpu_commit=cpu_producer_commit,
            )
            materializations[policy] = materialization
            materialization_identities[f"{scene_id}/{policy}"] = materialization["manifest"]
            roster = materialization.get("roster")
            if not isinstance(roster, Mapping) or not isinstance(
                roster.get("accepted_slots"), list
            ):
                raise CameraScorerGateError(f"{scene_id}/{policy} roster differs")
            current_exports[scene_id][policy] = cpu_pilot.validate_export_against_recorded(
                factory,
                scene_id=scene_id,
                policy=policy,
                root=root,
                expected_object_slots=roster["accepted_slots"],
                recorded=scene_gate.get("exports", {}).get(policy),
            )
            if (
                gate.get("export_closure", {}).get(scene_id, {}).get(policy)
                != current_exports[scene_id][policy]
            ):
                raise CameraScorerGateError(
                    f"{scene_id}/{policy} final CPU export closure differs"
                )
        bundle = _validate_task_bundle_chain(
            cpu_root / "task_freezes" / scene_id,
            root=root,
            scene_id=scene_id,
            expected_manifest_sha256=gate.get("bundle_manifest_sha256", {}).get(scene_id),
            cpu_commit=cpu_producer_commit,
            materializations=materializations,
        )
        task_manifest_identities[scene_id] = bundle["manifest"]
        all_suites[scene_id] = bundle["suites"]

    expected_reset_ids = [
        f"{task_id}__seed{BASE_SEED}__ep{episode}"
        for scene_id in SCENE_IDS
        for task_id in cpu_pilot.expected_task_ids(scene_id)
        for episode in range(EPISODES)
    ]
    if contract.get("reset_ids") != expected_reset_ids:
        raise CameraScorerGateError("CPU reset roster differs from exact 20-reset plan")
    if gate.get("expected_task_ids") != {
        scene_id: list(cpu_pilot.expected_task_ids(scene_id)) for scene_id in SCENE_IDS
    }:
        raise CameraScorerGateError("CPU expected task mapping differs")
    if gate.get("harness_dry_run") != {
        "environment_constructed": False,
        "planned_reset_count": 20,
        "rendering_executed": False,
        "resolved_scene_count": 2,
        "treatment_count": 2,
    }:
        raise CameraScorerGateError("CPU harness dry-run record differs")
    scene_by_id = {row["id"]: row for row in scene_cfgs}
    for scene_id in SCENE_IDS:
        row = scene_by_id[scene_id]
        if row.get("tasks_json") != str(
            cpu_root / "task_freezes" / scene_id / "planning_tasks.json"
        ):
            raise CameraScorerGateError(f"{scene_id} planning task path differs")
        variants = row.get("construction_variants")
        if not isinstance(variants, Mapping) or set(variants) != set(POLICY_TO_VARIANT.values()):
            raise CameraScorerGateError(f"{scene_id} construction variants differ")
        for policy, variant in POLICY_TO_VARIANT.items():
            spec = variants[variant]
            expected_factory = all_factories[scene_id][policy]
            if (
                spec.get("factory_dir") != expected_factory.relative_to(root).as_posix()
                or spec.get("tasks_json")
                != (cpu_root / "task_freezes" / scene_id / POLICY_TO_TASK_FILE[policy])
                .relative_to(root)
                .as_posix()
                or spec.get("scene_xml")
                != (expected_factory / "sim_export" / "scene.xml")
                .relative_to(root)
                .as_posix()
            ):
                raise CameraScorerGateError(f"{scene_id}/{policy} config binding differs")

    summary = {
        "cpu_config": _identity(config_path, root=root),
        "cpu_gate": _identity(gate_path, root=root),
        "cpu_producer_commit": cpu_producer_commit,
        "cpu_seal": _identity(seal_path, root=root),
        "export_closure_sha256": _canonical_hash(current_exports),
        "external_geometry_sha256": _canonical_hash(
            {scene_id: cpu_pilot.external_geometry_identities(scene_id) for scene_id in SCENE_IDS}
        ),
        "materialization_manifests": materialization_identities,
        "task_manifests": task_manifest_identities,
    }
    return {
        "factories": all_factories,
        "summary": summary,
        "suites": all_suites,
    }


def _gpu_runtime_attestation(profile: Mapping[str, str] | None = None) -> dict[str, Any]:
    profile = profile or {"node": EXPECTED_GPU_NODE, "name": EXPECTED_GPU_NAME,
                          "compute_capability": EXPECTED_GPU_COMPUTE_CAPABILITY,
                          "gres": "a100-80g:1"}
    required_environment = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "SLURMD_NODENAME": profile["node"],
        "SLURM_JOB_ACCOUNT": EXPECTED_SLURM_ACCOUNT,
        "SLURM_JOB_PARTITION": "batch",
        "SLURM_JOB_QOS": "normal",
        "SLURM_CPUS_PER_TASK": "4",
    }
    for name, expected in required_environment.items():
        if os.environ.get(name) != expected:
            raise CameraScorerGateError(
                f"GPU runtime differs at {name}: expected={expected!r}, "
                f"observed={os.environ.get(name)!r}"
            )
    job_id = os.environ.get("SLURM_JOB_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    qualifier_preflight_sha256 = os.environ.get(
        "E4_QUALIFIER_PREFLIGHT_GATE_SHA256"
    )
    if not job_id or not job_id.isdigit():
        raise CameraScorerGateError("camera gate requires one numeric Slurm job ID")
    if not visible or visible == "NoDevFiles" or "," in visible:
        raise CameraScorerGateError("camera gate requires exactly one visible GPU")
    if (
        not qualifier_preflight_sha256
        or SHA256_RE.fullmatch(qualifier_preflight_sha256) is None
    ):
        raise CameraScorerGateError(
            "camera gate requires the sealed qualifier-preflight SHA-256"
        )
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CameraScorerGateError("nvidia-smi GPU attestation failed") from exc
    rows = [row.strip() for row in output if row.strip()]
    if len(rows) != 1:
        raise CameraScorerGateError(f"expected one visible GPU, observed {rows!r}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise CameraScorerGateError("nvidia-smi attestation schema differs")
    name, uuid, compute_capability = fields
    if name != profile["name"] or compute_capability != profile["compute_capability"]:
        raise CameraScorerGateError(
            f"GPU hardware differs: name={name!r}, compute_capability={compute_capability!r}"
        )
    if not uuid.startswith("GPU-"):
        raise CameraScorerGateError("GPU UUID is invalid")
    return {
        "compute_capability": compute_capability,
        "cuda_visible_devices": visible,
        "gpu_name": name,
        "gpu_uuid": uuid,
        "job_id": job_id,
        "mujoco_gl": "egl",
        "node": profile["node"],
        "pyopengl_platform": "egl",
        "qualifier_preflight_gate_sha256": qualifier_preflight_sha256,
        "slurm_profile": {
            "account": EXPECTED_SLURM_ACCOUNT,
            "cpus_per_task": 4,
            "gres": profile["gres"],
            "mem": "32G",
            "nodes": 1,
            "ntasks": 1,
            "partition": "batch",
            "qos": "normal",
            "time": "01:00:00",
        },
    }


def _camera_pose(env: Any, camera_name: str) -> dict[str, Any]:
    camera_id = int(env.model.camera(camera_name).id)
    fovy = float(env.model.cam_fovy[camera_id])
    position = np.asarray(env.data.cam_xpos[camera_id], dtype=float)
    rotation = np.asarray(env.data.cam_xmat[camera_id], dtype=float).reshape(3, 3)
    if (
        not np.isfinite(position).all()
        or not np.isfinite(rotation).all()
        or not math.isfinite(fovy)
        or not 1.0 < fovy < 179.0
    ):
        raise CameraScorerGateError(f"camera {camera_name!r} has invalid geometry")
    return {
        "fovy_deg": fovy,
        "model_camera_id": camera_id,
        "name": camera_name,
        "position_world_m": position.tolist(),
        "rotation_camera_to_world": rotation.tolist(),
    }


def _body_descends_from(model: Any, body_id: int, ancestor_id: int) -> bool:
    current = int(body_id)
    while current > 0:
        if current == ancestor_id:
            return True
        current = int(model.body_parentid[current])
    return current == ancestor_id


def _named_body_mask(
    env: Any, segmentation: np.ndarray, body_names: Sequence[str]
) -> np.ndarray:
    """Return pixels belonging to geoms below any exact named body."""
    import mujoco

    array = np.asarray(segmentation)
    if array.shape != (RENDER_HEIGHT, RENDER_WIDTH, 2) or array.dtype.kind not in "iu":
        raise CameraScorerGateError(
            f"MuJoCo segmentation has wrong shape/dtype: {array.shape}, {array.dtype}"
        )
    object_id = array[:, :, 0]
    object_type = array[:, :, 1]
    allowed_types = {-1, int(mujoco.mjtObj.mjOBJ_GEOM)}
    if not set(int(value) for value in np.unique(object_type)) <= allowed_types:
        raise CameraScorerGateError("segmentation contains a non-geom object type")
    body_ids = [int(env.model.body(name).id) for name in body_names]
    geoms: list[int] = []
    for geom_id in range(env.model.ngeom):
        body_id = int(env.model.geom_bodyid[geom_id])
        if any(_body_descends_from(env.model, body_id, ancestor) for ancestor in body_ids):
            geoms.append(geom_id)
    if not geoms:
        raise CameraScorerGateError(f"bodies have no MuJoCo geoms: {list(body_names)!r}")
    return (
        (object_type == int(mujoco.mjtObj.mjOBJ_GEOM))
        & np.isin(object_id, geoms)
    )


def semantic_segmentation(
    env: Any,
    segmentation: np.ndarray,
    target: str,
    distractors: Sequence[str] = (),
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert MuJoCo pixels into semantic classes and exact object masks."""
    import mujoco

    array = np.asarray(segmentation)
    if array.shape != (RENDER_HEIGHT, RENDER_WIDTH, 2) or array.dtype.kind not in "iu":
        raise CameraScorerGateError(
            f"MuJoCo segmentation has wrong shape/dtype: {array.shape}, {array.dtype}"
        )
    object_id = array[:, :, 0]
    object_type = array[:, :, 1]
    allowed_types = {-1, int(mujoco.mjtObj.mjOBJ_GEOM)}
    if not set(int(value) for value in np.unique(object_type)) <= allowed_types:
        raise CameraScorerGateError("segmentation contains a non-geom object type")
    target_mask = _named_body_mask(env, array, [target])
    distractor_masks = {
        name: _named_body_mask(env, array, [name]) for name in distractors
    }
    robot_geoms: list[int] = []
    for geom_id in range(env.model.ngeom):
        body_id = int(env.model.geom_bodyid[geom_id])
        body_name = env.model.body(body_id).name or ""
        if body_name.startswith("robot/"):
            robot_geoms.append(geom_id)
    try:
        table_geom = int(env.model.geom("tabletop").id)
    except KeyError as exc:
        raise CameraScorerGateError("assembled environment lacks tabletop geom") from exc
    geom_pixels = object_type == int(mujoco.mjtObj.mjOBJ_GEOM)
    semantic = np.zeros(object_id.shape, dtype=np.uint8)
    semantic[geom_pixels] = 4  # other visible geometry
    semantic[geom_pixels & np.isin(object_id, robot_geoms)] = 2
    semantic[geom_pixels & (object_id == table_geom)] = 3
    for mask in distractor_masks.values():
        semantic[mask] = 5
    semantic[target_mask] = 1
    return semantic, {"target": target_mask, **distractor_masks}


def _semantic_rgb(semantic: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [0, 0, 0],       # background
            [255, 48, 48],   # target
            [48, 144, 255],  # robot
            [64, 220, 96],   # table
            [160, 160, 160], # other scene geometry
            [255, 176, 32],  # same-label distractor
        ],
        dtype=np.uint8,
    )
    if semantic.dtype != np.uint8 or np.any(semantic > 5):
        raise CameraScorerGateError("semantic segmentation contains an invalid class")
    return palette[semantic]


def _overlay(raw: np.ndarray, semantic_rgb: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    result = raw.astype(np.float32).copy()
    foreground = semantic > 0
    result[foreground] = (
        0.55 * result[foreground] + 0.45 * semantic_rgb[foreground].astype(np.float32)
    )
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _save_png(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    image.save(path, format="PNG", compress_level=6)


def _policy_resize_rgb(image: np.ndarray) -> np.ndarray:
    """Call the exact OpenPI client resize-with-pad used on policy requests."""
    from openpi_client import image_tools

    origin = Path(image_tools.__file__).resolve(strict=True)
    if origin != OPENPI_IMAGE_TOOLS:
        raise CameraScorerGateError(
            f"OpenPI resize import escaped clean pinned worktree: {origin}"
        )

    resized = np.asarray(
        image_tools.resize_with_pad(image, POLICY_INPUT_SIZE, POLICY_INPUT_SIZE)
    )
    if resized.shape != (POLICY_INPUT_SIZE, POLICY_INPUT_SIZE, 3) \
            or resized.dtype != np.uint8:
        raise CameraScorerGateError("OpenPI policy RGB resize contract differs")
    return resized


def _policy_resize_mask(mask: np.ndarray) -> np.ndarray:
    """Nearest-neighbor mask transform with OpenPI's exact pad geometry."""
    from PIL import Image

    source = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    width, height = source.size
    ratio = max(width / POLICY_INPUT_SIZE, height / POLICY_INPUT_SIZE)
    resized_height = int(height / ratio)
    resized_width = int(width / ratio)
    resized = source.resize((resized_width, resized_height), Image.Resampling.NEAREST)
    canvas = Image.new("L", (POLICY_INPUT_SIZE, POLICY_INPUT_SIZE), 0)
    canvas.paste(
        resized,
        (
            max(0, int((POLICY_INPUT_SIZE - resized_width) / 2)),
            max(0, int((POLICY_INPUT_SIZE - resized_height) / 2)),
        ),
    )
    return np.asarray(canvas, dtype=np.uint8) > 0


def _mask_measurement(mask: np.ndarray) -> dict[str, Any]:
    bbox = _bbox(mask)
    if bbox is None:
        width = height = 0
    else:
        width = bbox[2] - bbox[0] + 1
        height = bbox[3] - bbox[1] + 1
    pixels = int(np.count_nonzero(mask))
    return {
        "bbox_height_pixels": height,
        "bbox_width_pixels": width,
        "bbox_xyxy_inclusive": bbox,
        "checks": {
            "bbox_height_at_least_3": height >= MIN_POLICY_BBOX_SIDE,
            "bbox_width_at_least_3": width >= MIN_POLICY_BBOX_SIDE,
            "pixels_at_least_9": pixels >= MIN_POLICY_OBJECT_PIXELS,
        },
        "pixels": pixels,
    }


def _rotation_geodesic_rad(observed: np.ndarray, expected: np.ndarray) -> float:
    relative = expected.T @ observed
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def exterior_camera_replay(env: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Compare assembled ext_cam to the sealed suite world pose."""
    import mujoco

    if expected.get("mode") != "world":
        raise CameraScorerGateError("sealed exterior camera is not a world-pose camera")
    position = np.asarray(expected.get("pos"), dtype=float)
    quaternion = np.asarray(expected.get("quat_wxyz"), dtype=float)
    fovy = float(expected.get("fovy"))
    if position.shape != (3,) or quaternion.shape != (4,) \
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise CameraScorerGateError("sealed exterior camera pose is invalid")
    quaternion = quaternion / np.linalg.norm(quaternion)
    expected_rotation_flat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(expected_rotation_flat, quaternion)
    expected_rotation = expected_rotation_flat.reshape(3, 3)
    observed = _camera_pose(env, env.info["ext_cam"])
    observed_position = np.asarray(observed["position_world_m"], dtype=float)
    observed_rotation = np.asarray(observed["rotation_camera_to_world"], dtype=float)
    deltas = {
        "fovy_deg": abs(observed["fovy_deg"] - fovy),
        "position_m": float(np.linalg.norm(observed_position - position)),
        "rotation_geodesic_rad": _rotation_geodesic_rad(
            observed_rotation, expected_rotation
        ),
    }
    checks = {
        "fovy_delta_at_most_1e-6_deg": deltas["fovy_deg"] <= 1e-6,
        "position_delta_at_most_1e-6_m": deltas["position_m"] <= 1e-6,
        "rotation_delta_at_most_1e-6_rad": deltas["rotation_geodesic_rad"] <= 1e-6,
    }
    return {
        "checks": checks,
        "deltas": deltas,
        "expected": {
            "fovy_deg": fovy,
            "frame": expected.get("frame"),
            "position_world_m": position.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
        },
        "passed": all(checks.values()),
    }


def camera_metric(
    *,
    env: Any,
    raw: np.ndarray,
    segmentation: np.ndarray,
    camera_role: str,
    camera_name: str,
    target: str,
    distractors: Sequence[str],
    expected_ext_cam: Mapping[str, Any],
    output_dir: Path,
    artifact_root: Path,
    row_identity: Mapping[str, Any],
    singleton_task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    singleton = False
    if singleton_task is not None:
        singleton = (singleton_task.get("target") == target
                     and singleton_task.get("any_instance") is False
                     and _task_qualifier(singleton_task) is None
                     and _eligible_same_label_objects(env, singleton_task, env._task_rows) == [target]
                     and not distractors)
        if not singleton:
            raise CameraScorerGateError("singleton camera exemption lacks exact eligible target")
    raw = np.asarray(raw)
    if raw.shape != (RENDER_HEIGHT, RENDER_WIDTH, 3) or raw.dtype != np.uint8:
        raise CameraScorerGateError(
            f"{camera_role} RGB has wrong shape/dtype: {raw.shape}, {raw.dtype}"
        )
    semantic, object_masks = semantic_segmentation(
        env, segmentation, target, distractors=distractors
    )
    seg_rgb = _semantic_rgb(semantic)
    overlay = _overlay(raw, seg_rgb, semantic)
    raw_path = output_dir / f"{camera_role}_raw.png"
    seg_path = output_dir / f"{camera_role}_seg.png"
    overlay_path = output_dir / f"{camera_role}_overlay.png"
    policy_raw_path = output_dir / f"{camera_role}_policy224_raw.png"
    policy_seg_path = output_dir / f"{camera_role}_policy224_seg.png"
    policy_overlay_path = output_dir / f"{camera_role}_policy224_overlay.png"
    _save_png(raw_path, raw)
    _save_png(seg_path, seg_rgb)
    _save_png(overlay_path, overlay)
    policy_raw = _policy_resize_rgb(raw)
    policy_masks = {
        name: _policy_resize_mask(mask) for name, mask in object_masks.items()
    }
    policy_semantic = np.zeros((POLICY_INPUT_SIZE, POLICY_INPUT_SIZE), dtype=np.uint8)
    # Resize the pretty semantic image with nearest-neighbor class semantics.
    for semantic_class in (4, 3, 2, 5):
        policy_semantic[_policy_resize_mask(semantic == semantic_class)] = semantic_class
    for name in distractors:
        policy_semantic[policy_masks[name]] = 5
    policy_semantic[policy_masks["target"]] = 1
    policy_seg_rgb = _semantic_rgb(policy_semantic)
    policy_overlay = _overlay(policy_raw, policy_seg_rgb, policy_semantic)
    _save_png(policy_raw_path, policy_raw)
    _save_png(policy_seg_path, policy_seg_rgb)
    _save_png(policy_overlay_path, policy_overlay)
    sampled = raw[::4, ::4].reshape(-1, 3)
    unique_colors = int(np.unique(sampled, axis=0).shape[0])
    counts = {
        "background": int(np.count_nonzero(semantic == 0)),
        "target": int(np.count_nonzero(semantic == 1)),
        "robot": int(np.count_nonzero(semantic == 2)),
        "table": int(np.count_nonzero(semantic == 3)),
        "other_geometry": int(np.count_nonzero(semantic == 4)),
        "same_label_distractor": int(np.count_nonzero(semantic == 5)),
    }
    dynamic_range = int(raw.max()) - int(raw.min())
    rgb_std = float(raw.astype(np.float64).std())
    normalized_mean_luminance = float(
        np.mean(
            raw[:, :, 0].astype(np.float64) * 0.2126
            + raw[:, :, 1].astype(np.float64) * 0.7152
            + raw[:, :, 2].astype(np.float64) * 0.0722
        ) / 255.0
    )
    common_checks = {
        "normalized_mean_luminance_strictly_between_0.01_and_0.99": (
            0.01 < normalized_mean_luminance < 0.99
        ),
        "rgb_dynamic_range_at_least_32": dynamic_range >= MIN_RGB_DYNAMIC_RANGE,
        "rgb_std_at_least_5": rgb_std >= MIN_RGB_STD,
        "sampled_unique_colors_at_least_32": unique_colors >= MIN_SAMPLED_UNIQUE_COLORS,
    }
    policy_visibility = {
        name: _mask_measurement(mask) for name, mask in policy_masks.items()
    }
    if camera_role == "exterior":
        exterior_replay = exterior_camera_replay(env, expected_ext_cam)
        checks = {
            **common_checks,
            "exterior_camera_matches_sealed_pose": exterior_replay["passed"],
            "target_visible_at_policy_input": all(
                policy_visibility["target"]["checks"].values()
            ),
            "every_same_label_distractor_visible_at_policy_input": (singleton or bool(distractors))
            and all(
                all(policy_visibility[name]["checks"].values()) for name in distractors
            ),
        }
    elif camera_role == "wrist":
        exterior_replay = None
        total = semantic.size
        nonrobot_geometry = (
            counts["target"] + counts["same_label_distractor"]
            + counts["table"] + counts["other_geometry"]
        )
        checks = {
            **common_checks,
            "nonrobot_geometry_pixel_present": nonrobot_geometry >= 1,
            "robot_mask_fraction_below_50_percent": counts["robot"] / total < MAX_WRIST_ROBOT_FRACTION,
        }
    else:
        raise CameraScorerGateError(f"unknown camera role: {camera_role}")
    return {
        **dict(row_identity),
        "artifacts": {
            "overlay": _identity(overlay_path, root=artifact_root),
            "raw": _identity(raw_path, root=artifact_root),
            "segmentation": _identity(seg_path, root=artifact_root),
            "policy224_overlay": _identity(policy_overlay_path, root=artifact_root),
            "policy224_raw": _identity(policy_raw_path, root=artifact_root),
            "policy224_segmentation": _identity(policy_seg_path, root=artifact_root),
        },
        "camera": _camera_pose(env, camera_name),
        "camera_role": camera_role,
        "checks": checks,
        "exterior_camera_replay": exterior_replay,
        "passed": all(checks.values()),
        "pixel_counts": counts,
        "rgb": {
            "dynamic_range": dynamic_range,
            "max": int(raw.max()),
            "mean": float(raw.astype(np.float64).mean()),
            "normalized_mean_luminance": normalized_mean_luminance,
            "min": int(raw.min()),
            "sampled_unique_colors": unique_colors,
            "std": rgb_std,
        },
        "policy224_visibility": policy_visibility,
        "distractor_visibility_applicable": not singleton,
        "segmentation_format": (
            "semantic_rgb: target red, same-label distractor orange, robot blue, "
            "table green, other gray"
        ),
        "target_bbox_xyxy_inclusive": _bbox(semantic == 1),
    }


def reach_envelope(point_xy: Sequence[float], base_xy: Sequence[float], yaw: float) -> dict[str, Any]:
    point = np.asarray(point_xy, dtype=float)
    base = np.asarray(base_xy, dtype=float)
    if point.shape != (2,) or base.shape != (2,) or not np.isfinite(point).all() \
            or not np.isfinite(base).all() or not math.isfinite(float(yaw)):
        raise CameraScorerGateError("reach-envelope input is invalid")
    delta = point - base
    distance = float(np.linalg.norm(delta))
    bearing = (
        math.degrees(math.atan2(float(delta[1]), float(delta[0])) - float(yaw)) + 180.0
    ) % 360.0 - 180.0
    checks = {
        "outside_inner_radius": distance >= REACH_MIN_M,
        "inside_outer_radius": distance <= REACH_MAX_M,
        "inside_front_cone": abs(bearing) <= FRONT_CONE_DEG,
    }
    return {
        "bearing_off_heading_deg": bearing,
        "checks": checks,
        "distance_m": distance,
        "margins": {
            "front_cone_deg": FRONT_CONE_DEG - abs(bearing),
            "inner_radius_m": distance - REACH_MIN_M,
            "outer_radius_m": REACH_MAX_M - distance,
        },
        "passed": all(checks.values()),
        "point_xy_m": point.tolist(),
    }


def _project_with_depth(
    point_world: Sequence[float], camera: Mapping[str, Any], depth: np.ndarray
) -> dict[str, Any]:
    point = np.asarray(point_world, dtype=float)
    position = np.asarray(camera["position_world_m"], dtype=float)
    rotation = np.asarray(camera["rotation_camera_to_world"], dtype=float)
    if depth.shape != (RENDER_HEIGHT, RENDER_WIDTH) or not np.isfinite(depth).all():
        raise CameraScorerGateError("exterior depth render is invalid")
    camera_point = rotation.T @ (point - position)
    optical_depth = float(-camera_point[2])
    if optical_depth <= 0.0:
        return {
            "in_frame": False,
            "occlusion_test_passed": False,
            "optical_depth_m": optical_depth,
            "pixel_xy": None,
            "visible": False,
        }
    focal = RENDER_HEIGHT / (
        2.0 * math.tan(math.radians(float(camera["fovy_deg"])) / 2.0)
    )
    pixel_x = RENDER_WIDTH / 2.0 + focal * float(camera_point[0]) / optical_depth
    pixel_y = RENDER_HEIGHT / 2.0 - focal * float(camera_point[1]) / optical_depth
    in_frame = 0.0 <= pixel_x < RENDER_WIDTH and 0.0 <= pixel_y < RENDER_HEIGHT
    occlusion_pass = False
    observed_depth = None
    if in_frame:
        ix = int(np.clip(round(pixel_x), 0, RENDER_WIDTH - 1))
        iy = int(np.clip(round(pixel_y), 0, RENDER_HEIGHT - 1))
        neighborhood = depth[
            max(0, iy - 1):min(RENDER_HEIGHT, iy + 2),
            max(0, ix - 1):min(RENDER_WIDTH, ix + 2),
        ]
        observed_depth = float(np.max(neighborhood))
        occlusion_pass = optical_depth <= observed_depth + 0.02
    return {
        "depth_tolerance_m": 0.02,
        "in_frame": in_frame,
        "observed_max_depth_3x3_m": observed_depth,
        "occlusion_test_passed": occlusion_pass,
        "optical_depth_m": optical_depth,
        "pixel_xy": [pixel_x, pixel_y],
        "visible": in_frame and occlusion_pass,
    }


def workspace_metric(
    env: Any,
    task: Mapping[str, Any],
    suite: Mapping[str, Any],
    target_row: Mapping[str, Any],
    exterior_depth: np.ndarray,
    row_identity: Mapping[str, Any],
) -> dict[str, Any]:
    import mujoco

    from robo.eval.e4_candidate_screen import task_destination, CandidateScreenError
    try:
        destination_contract = task_destination(task, env)
    except CandidateScreenError as exc:
        raise CameraScorerGateError(str(exc)) from exc
    region = destination_contract["destination_region"]
    receptacle = destination_contract["destination_body"]
    required_region = ("cx", "cy", "hx", "hy", "zlo", "zhi")
    values = {name: float(region[name]) for name in required_region}
    if (
        not all(math.isfinite(value) for value in values.values())
        or values["hx"] <= 0
        or values["hy"] <= 0
        or values["zhi"] <= values["zlo"]
    ):
        raise CameraScorerGateError("task destination region is invalid")
    target_pose = env.body_pose(str(task["target"]))
    target_position = np.asarray(target_pose[0], dtype=float)
    target_quaternion = np.asarray(target_pose[1], dtype=float)
    if target_position.shape != (3,) or not np.isfinite(target_position).all():
        raise CameraScorerGateError("post-settle target position is invalid")
    robot = suite.get("robot")
    if not isinstance(robot, Mapping):
        raise CameraScorerGateError("task suite robot contract is missing")
    base_position = np.asarray(robot.get("base_pos"), dtype=float)
    yaw = float(robot.get("base_yaw"))
    target = reach_envelope(target_position[:2], base_position[:2], yaw)
    destination = reach_envelope([values["cx"], values["cy"]], base_position[:2], yaw)
    initially_in_region = bool(
        abs(target_position[0] - values["cx"]) <= values["hx"]
        and abs(target_position[1] - values["cy"]) <= values["hy"]
        and values["zlo"] <= target_position[2] <= values["zhi"]
    )
    dimensions = np.asarray(target_row.get("dims"), dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() \
            or np.any(dimensions <= 0.0):
        raise CameraScorerGateError("target dimensions are invalid")
    rotation_flat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation_flat, target_quaternion)
    rotation = rotation_flat.reshape(3, 3)
    world_xy_half_extents = np.abs(rotation[:2, :]) @ (dimensions / 2.0)
    target_xy_half_diagonal = float(np.linalg.norm(world_xy_half_extents))
    erosion_m = target_xy_half_diagonal + 0.01
    core_hx = values["hx"] - erosion_m
    core_hy = values["hy"] - erosion_m
    core_nonempty = core_hx > 0.0 and core_hy > 0.0
    sample_rows: list[dict[str, Any]] = []
    safe_reach_count = 0
    visible_count = 0
    camera = _camera_pose(env, env.info["ext_cam"])
    sample_z = float(np.clip(
        target_position[2], values["zlo"] + 0.001, values["zhi"] - 0.001
    ))
    if core_nonempty:
        xs = [values["cx"] - core_hx, values["cx"], values["cx"] + core_hx]
        ys = [values["cy"] - core_hy, values["cy"], values["cy"] + core_hy]
        for row_index, y in enumerate(ys):
            for column_index, x in enumerate(xs):
                envelope = reach_envelope([x, y], base_position[:2], yaw)
                margins = envelope["margins"]
                safe_reach = (
                    envelope["passed"]
                    and margins["inner_radius_m"] >= 0.02
                    and margins["outer_radius_m"] >= 0.02
                    and margins["front_cone_deg"] >= 5.0
                )
                projection = _project_with_depth([x, y, sample_z], camera, exterior_depth)
                safe_reach_count += int(safe_reach)
                visible_count += int(projection["visible"])
                sample_rows.append(
                    {
                        "column": column_index,
                        "point_world_m": [x, y, sample_z],
                        "projection": projection,
                        "reach_envelope": envelope,
                        "row": row_index,
                        "safe_reach_margin": safe_reach,
                    }
                )
    checks = {
        "destination_center_in_reach_envelope": destination["passed"],
        "eroded_goal_core_nonempty": core_nonempty,
        "exterior_visible_samples_at_least_5_of_9": visible_count >= 5,
        "safe_reach_sample_exists": safe_reach_count >= 1,
        "target_initially_outside_goal_region": not initially_in_region,
        "target_in_reach_envelope": target["passed"],
    }
    if receptacle is not None:
        from robo.eval.e4_candidate_screen import receptacle_goal_probe
        probe = receptacle_goal_probe(env, task)
        checks["receptacle_goal_collision_verified"] = probe["passed"]
        destination_contract = {**destination_contract,
            "goal_collision_status": "PASS" if probe["passed"] else "FAIL",
            "cavity_verified": probe["passed"], "goal_probe": probe}
    return {
        **dict(row_identity),
        **({"task_destination": destination_contract} if receptacle is not None else {}),
        "destination_region": values,
        "destination_region_center_envelope": destination,
        "eroded_goal_core": {
            "center_xy_m": [values["cx"], values["cy"]],
            "erosion_m": erosion_m,
            "half_extents_xy_m": [core_hx, core_hy],
            "nonempty": core_nonempty,
            "extent_method": (
                "abs(current_body_rotation_world[:2,:]) @ (sealed_world_dims/2), "
                "then XY half-diagonal norm"
            ),
            "target_local_dimensions_m": dimensions.tolist(),
            "target_world_xy_half_diagonal_m": target_xy_half_diagonal,
            "target_world_xy_half_extents_m": world_xy_half_extents.tolist(),
        },
        "checks": checks,
        "exterior_projection": {
            "sample_count": len(sample_rows),
            "samples": sample_rows,
            "visible_count": visible_count,
            "visible_required": 5,
        },
        "initially_in_goal_region": initially_in_region,
        "ik_or_collision_path_validated": False,
        "passed": all(checks.values()),
        "safe_reach_sample_count": safe_reach_count,
        "scope": (
            "eroded 2d radial-plus-front-cone reach envelope and exterior depth "
            "projection only; not IK, manipulability, collision-free motion planning, "
            "or an oracle"
        ),
        "target_post_settle_position_m": target_position.tolist(),
        "target_reach_envelope": target,
    }


def _task_qualifier(task: Mapping[str, Any]) -> str | None:
    instructions = task.get("instructions")
    if not isinstance(instructions, Mapping) or not isinstance(instructions.get("default"), str):
        raise CameraScorerGateError("task default instruction is missing")
    text = instructions["default"].lower()
    matches = [qualifier for qualifier in QUALIFIERS if qualifier in text.split()]
    if len(matches) > 1:
        raise CameraScorerGateError("task instruction contains multiple qualifiers")
    return matches[0] if matches else None


def _eligible_same_label_objects(
    env: Any, task: Mapping[str, Any], task_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    from robo.tasks import pi05_tasks

    label = str(task.get("target_label"))
    available = set(env.free_bodies)
    return sorted(
        {
            str(row["name"])
            for row in task_rows
            if row.get("label") == label
            and str(row.get("name")) in available
            and pi05_tasks._is_graspable(row)
        }
    )


def disambiguation_metric(
    env: Any,
    task: Mapping[str, Any],
    suite: Mapping[str, Any],
    task_rows: Sequence[Mapping[str, Any]],
    row_identity: Mapping[str, Any],
) -> dict[str, Any]:
    target = str(task.get("target"))
    label = str(task.get("target_label"))
    same_label = _eligible_same_label_objects(env, task, task_rows)
    if target not in same_label:
        raise CameraScorerGateError("task target is not an eligible same-label object")
    qualifier = _task_qualifier(task)
    any_instance = task.get("any_instance") is True
    result: dict[str, Any] = {
        **dict(row_identity),
        "any_instance": any_instance,
        "eligible_same_label_objects": same_label,
        "qualifier": qualifier,
        "strict_required_margin_m": QUALIFIER_MARGIN_M,
        "target_label": label,
    }
    if len(same_label) == 1:
        passed = qualifier is None and not any_instance
        result.update({"applicable": False, "observed_margin_m": None, "passed": passed})
        return result
    if qualifier is None:
        passed = any_instance
        result.update({"applicable": True, "observed_margin_m": None, "passed": passed})
        return result
    if any_instance:
        result.update({"applicable": True, "observed_margin_m": None, "passed": False})
        return result
    base = np.asarray(suite["robot"]["base_pos"][:2], dtype=float)
    yaw = float(suite["robot"]["base_yaw"])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    coordinates: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}
    for name in same_label:
        xy = np.asarray(env.body_pose(name)[0][:2], dtype=float)
        delta = xy - base
        forward = float(cosine * delta[0] + sine * delta[1])
        left = float(-sine * delta[0] + cosine * delta[1])
        coordinates[name] = {"forward_m": forward, "left_m": left}
        scores[name] = {
            "leftmost": left,
            "rightmost": -left,
            "nearest": -forward,
            "farthest": forward,
        }[qualifier]
    competitor = max((name for name in same_label if name != target), key=scores.__getitem__)
    margin = scores[target] - scores[competitor]
    result.update(
        {
            "applicable": True,
            "competitor": competitor,
            "coordinates_in_robot_base_frame": coordinates,
            "observed_margin_m": margin,
            "passed": margin > QUALIFIER_MARGIN_M,
            "qualifier_scores_m": scores,
        }
    )
    return result


def _settle_contract_checks(provenance: Mapping[str, Any]) -> dict[str, bool]:
    """Check the fixed 1.5 second, 900-step pi0.5 reset contract."""
    from robo.envs.pi05_env import (
        _settle_duration_tolerance,
        _settle_step_count,
    )
    from robo.rigs import pi05_rig

    settle = provenance.get("settle_protocol")
    if not isinstance(settle, Mapping):
        return {
            "settle_duration_matches_request": False,
            "settle_has_valid_timing": False,
            "settle_is_exactly_900_steps": False,
            "settle_timestep_matches_rig": False,
            "settle_uses_mujoco_step": False,
        }
    requested = settle.get("requested_duration_s")
    timestep = settle.get("model_timestep_s")
    steps = settle.get("step_count")
    simulated = settle.get("simulated_duration_s")
    timing_values = (requested, timestep, simulated)
    timing_valid = (
        all(
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating))
            and math.isfinite(float(value))
            for value in timing_values
        )
        and isinstance(steps, int)
        and not isinstance(steps, bool)
        and steps >= 0
    )
    expected_steps = None
    duration_matches = False
    timestep_matches = False
    if timing_valid:
        try:
            expected_steps = _settle_step_count(requested, timestep)
        except (TypeError, ValueError, OverflowError):
            timing_valid = False
        else:
            tolerance = _settle_duration_tolerance(requested, timestep)
            duration_matches = math.isclose(
                float(simulated),
                float(requested),
                rel_tol=0.0,
                abs_tol=tolerance,
            ) and math.isclose(
                float(simulated),
                int(steps) * float(timestep),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            timestep_matches = math.isclose(
                float(timestep),
                float(pi05_rig.PHYSICS_DT),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
    return {
        "settle_duration_matches_request": duration_matches,
        "settle_has_valid_timing": timing_valid,
        "settle_is_exactly_900_steps": (
            expected_steps == 900 and steps == expected_steps
        ),
        "settle_timestep_matches_rig": timestep_matches,
        "settle_uses_mujoco_step": (
            settle.get("engine") == "mujoco"
            and settle.get("step_function") == "mujoco.mj_step"
        ),
    }


def _build_headless_droid_env(
    *,
    scene_xml: str,
    base_pos: Sequence[float],
    base_yaw: float,
    table_box: Mapping[str, Any],
    ext_cam: Mapping[str, Any],
    exclude_objects: Sequence[str],
    menagerie_root: Path,
) -> Any:
    """Build the normal pi0.5 rig without constructing a MuJoCo renderer.

    The subclass inherits :meth:`DroidSimEnv.reset` unchanged.  Only
    ``get_obs`` is replaced because the normal method renders both cameras
    after the final settle step.  All model, actuator, and reset bookkeeping
    is initialized identically to ``DroidSimEnv.__init__`` except for the
    deliberately absent ``renderer`` member.
    """
    import mujoco
    from robo.envs.pi05_env import DroidSimEnv
    from robo.rigs import pi05_rig as rig

    class _HeadlessDroidSimEnv(DroidSimEnv):
        def get_obs(self) -> dict[str, Any]:
            return {}

    env = _HeadlessDroidSimEnv.__new__(_HeadlessDroidSimEnv)
    env.model, env.info = rig.build_scene_model(
        scene_xml,
        base_pos,
        base_yaw,
        table_box=table_box,
        ext_cam=ext_cam,
        exclude_objects=exclude_objects,
        menagerie_root=menagerie_root,
    )
    env.data = mujoco.MjData(env.model)
    model, info = env.model, env.info
    env._jadr = np.array(
        [model.joint(name).qposadr[0] for name in info["arm_joints"]]
    )
    env._jdadr = np.array(
        [model.joint(name).dofadr[0] for name in info["arm_joints"]]
    )
    env._aids = np.array(
        [model.actuator(name).id for name in info["arm_actuators"]]
    )
    env._grip_aid = model.actuator(info["gripper_actuator"]).id
    env._grip_jadr = model.joint(info["gripper_driver_joint"]).qposadr[0]
    env._ctrl_lo = model.actuator_ctrlrange[env._aids, 0]
    env._ctrl_hi = model.actuator_ctrlrange[env._aids, 1]
    env.free_bodies = [
        model.body(body_id).name
        for body_id in range(model.nbody)
        if model.body(body_id).jntnum[0] == 1
        and model.body(body_id).name.startswith("obj_")
    ]
    env.pad_geoms = {"left": [], "right": []}
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name or ""
        if "2f85" in name and "pad" in name:
            side = "left" if "left" in name else "right"
            env.pad_geoms[side].append(geom_id)
    if not env.pad_geoms["left"] or not env.pad_geoms["right"]:
        raise CameraScorerGateError("headless pi0.5 rig has no gripper pad geoms")
    env._body_geoms = {}
    for body_name in env.free_bodies:
        body_id = model.body(body_name).id
        address = model.body_geomadr[body_id]
        count = model.body_geomnum[body_id]
        env._body_geoms[body_name] = set(range(address, address + count))
    env.composite = None
    env.last_reset_provenance = None
    if hasattr(env, "renderer"):
        raise CameraScorerGateError("headless qualifier preflight constructed a renderer")
    return env


def scorer_replay(
    env: Any, task: Mapping[str, Any], row_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay scorer stages by direct object state mutation, never as an oracle."""
    import mujoco
    from robo.tasks.pi05_tasks import TaskScorer

    target = str(task["target"])
    body = env.model.body(target)
    if int(body.jntnum[0]) != 1:
        raise CameraScorerGateError("scorer replay target does not have one free joint")
    joint_id = int(body.jntadr[0])
    if int(env.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise CameraScorerGateError("scorer replay target joint is not free")
    qpos_address = int(env.model.jnt_qposadr[joint_id])
    qvel_address = int(env.model.jnt_dofadr[joint_id])
    baseline = {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "ctrl": env.data.ctrl.copy(),
        "time": float(env.data.time),
    }
    held_names: set[str] = set()
    original_grasped = env.grasped

    def fake_grasped(name: str) -> bool:
        return name in held_names

    def restore() -> None:
        env.data.qpos[:] = baseline["qpos"]
        env.data.qvel[:] = baseline["qvel"]
        env.data.ctrl[:] = baseline["ctrl"]
        env.data.time = baseline["time"]
        held_names.clear()
        mujoco.mj_forward(env.model, env.data)

    def move_target(x: float, y: float, z: float) -> None:
        env.data.qpos[qpos_address:qpos_address + 3] = [x, y, z]
        env.data.qvel[qvel_address:qvel_address + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)

    def snapshot(active_scorer: Any) -> dict[str, Any]:
        # TaskScorer.summary() intentionally exposes its stage mapping; deep
        # copy here so later replay ticks cannot mutate earlier evidence.
        return json.loads(json.dumps(active_scorer.summary()))

    initial_position = np.asarray(env.body_pose(target)[0], dtype=float)
    initial_z = float(env.start_pose[target][0][2])
    from robo.eval.e4_candidate_screen import task_destination, CandidateScreenError
    try:
        destination_contract = task_destination(task, env)
    except CandidateScreenError as exc:
        raise CameraScorerGateError(str(exc)) from exc
    region = destination_contract["destination_region"]
    destination_z = max(initial_z + 0.06, float(region["zlo"]) + 0.001)
    if destination_z > float(region["zhi"]) - 0.001:
        raise CameraScorerGateError("scorer replay has no lifted z inside destination")

    positive: dict[str, Any] = {}
    negative: dict[str, Any] = {}

    try:
        env.grasped = fake_grasped
        scorer = TaskScorer(env, task, hold_ticks=15)
        scorer.update()
        positive["initial"] = snapshot(scorer)
        held_names.add(target)
        scorer.update()
        positive["grasp"] = snapshot(scorer)
        move_target(float(initial_position[0]), float(initial_position[1]), initial_z + 0.06)
        scorer.update()
        positive["lift"] = snapshot(scorer)
        move_target(float(region["cx"]), float(region["cy"]), destination_z)
        scorer.update()
        positive["hover"] = snapshot(scorer)
        held_names.clear()
        for _ in range(14):
            scorer.update()
        positive["released_14_ticks"] = snapshot(scorer)
        scorer.update()
        positive["released_15_ticks"] = snapshot(scorer)

        # Negative 1: teleporting/moving to the goal without a grasp/lift must
        # never receive staged credit or success.
        restore()
        scorer = TaskScorer(env, task, hold_ticks=15)
        move_target(float(region["cx"]), float(region["cy"]), destination_z)
        for _ in range(20):
            scorer.update()
        negative["motion_without_grasp"] = snapshot(scorer)

        # Negative 2: a valid grasp/lift followed by release just outside the
        # strict region may reach hover (5 cm tolerance) but never place.
        restore()
        scorer = TaskScorer(env, task, hold_ticks=15)
        held_names.add(target)
        scorer.update()
        move_target(float(initial_position[0]), float(initial_position[1]), initial_z + 0.06)
        scorer.update()
        outside_x = float(region["cx"]) + float(region["hx"]) + 0.025
        move_target(outside_x, float(region["cy"]), destination_z)
        scorer.update()
        held_names.clear()
        for _ in range(20):
            scorer.update()
        negative["release_outside_strict_region"] = snapshot(scorer)

        # Negative 3: grasping a non-target must be diagnosed and cannot
        # advance the requested target's stages.
        restore()
        wrong_target = next((name for name in env.free_bodies if name != target), None)
        if wrong_target is None:
            raise CameraScorerGateError("scorer negative replay has no wrong target")
        scorer = TaskScorer(env, task, hold_ticks=15)
        held_names.add(wrong_target)
        scorer.update()
        negative["wrong_target_grasp"] = {
            **snapshot(scorer),
            "injected_wrong_target": wrong_target,
        }

        # Negative 4: a scene that starts with the target inside the region
        # cannot become successful without the ordered grasp/lift/hover path.
        restore()
        move_target(float(region["cx"]), float(region["cy"]), destination_z)
        scorer = TaskScorer(env, task, hold_ticks=15)
        for _ in range(20):
            scorer.update()
        negative["initially_in_region"] = snapshot(scorer)
    finally:
        restore()
        env.grasped = original_grasped

    checks = {
        "initial_score_zero": positive["initial"]["score"] == 0.0,
        "grasp_stage_reached": positive["grasp"]["stages"]["grasp"] is True,
        "lift_stage_reached": positive["lift"]["stages"]["lift"] is True,
        "hover_stage_reached": positive["hover"]["stages"]["hover"] is True,
        "place_not_early": positive["released_14_ticks"]["stages"]["place"] is False,
        "place_at_hold_tick_15": positive["released_15_ticks"]["stages"]["place"] is True,
        "success_and_score_one": (
            positive["released_15_ticks"]["success"] is True
            and positive["released_15_ticks"]["score"] == 1.0
        ),
        "motion_without_grasp_cannot_succeed": (
            negative["motion_without_grasp"]["success"] is False
            and negative["motion_without_grasp"]["score"] == 0.0
        ),
        "out_of_region_release_cannot_place": (
            negative["release_outside_strict_region"]["success"] is False
            and negative["release_outside_strict_region"]["stages"]["place"] is False
        ),
        "wrong_target_cannot_advance_requested_target": (
            negative["wrong_target_grasp"]["success"] is False
            and negative["wrong_target_grasp"]["score"] == 0.0
            and negative["wrong_target_grasp"]["injected_wrong_target"]
            in negative["wrong_target_grasp"]["wrong_grasps"]
        ),
        "initially_in_region_cannot_succeed": (
            negative["initially_in_region"]["success"] is False
            and negative["initially_in_region"]["score"] == 0.0
        ),
    }
    return {
        **dict(row_identity),
        "checks": checks,
        "hold_ticks": 15,
        **({"task_destination": destination_contract}
           if destination_contract["destination_body"] is not None else {}),
        "oracle_validated": False,
        "passed": all(checks.values()),
        "scope": (
            "direct-state TaskScorer predicate replay only; no IK, controller, policy, "
            "motion plan, grasp feasibility, or task oracle"
        ),
        "negative_replays": negative,
        "positive_replay": positive,
    }


def _render_segmentation(env: Any, camera_name: str) -> np.ndarray:
    env.renderer.enable_segmentation_rendering()
    try:
        env.renderer.update_scene(env.data, camera=camera_name)
        return np.asarray(env.renderer.render()).copy()
    finally:
        env.renderer.disable_segmentation_rendering()


def _render_depth(env: Any, camera_name: str) -> np.ndarray:
    env.renderer.enable_depth_rendering()
    try:
        env.renderer.update_scene(env.data, camera=camera_name)
        depth = np.asarray(env.renderer.render()).copy()
    finally:
        env.renderer.disable_depth_rendering()
    if depth.shape != (RENDER_HEIGHT, RENDER_WIDTH) or depth.dtype != np.float32 \
            or not np.isfinite(depth).all() or np.any(depth <= 0.0):
        raise CameraScorerGateError("MuJoCo exterior depth render is invalid")
    return depth


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _contact_sheets(
    camera_rows: Sequence[Mapping[str, Any]], *, artifact_root: Path
) -> dict[str, dict[str, Any]]:
    from PIL import Image, ImageDraw

    reports: dict[str, dict[str, Any]] = {}
    thumb_width, thumb_height, label_height = 192, 108, 18
    for scene_id in SCENE_IDS:
        for policy in POLICIES:
            for camera_role in CAMERAS:
                selected = [
                    row
                    for row in camera_rows
                    if row["scene_id"] == scene_id
                    and row["policy_id"] == policy
                    and row["camera_role"] == camera_role
                ]
                selected.sort(key=lambda row: (row["task_id"], row["episode"]))
                if len(selected) != 10:
                    raise CameraScorerGateError(
                        f"contact sheet {scene_id}/{policy}/{camera_role} has {len(selected)} rows"
                    )
                tile_width = thumb_width * 3
                tile_height = thumb_height + label_height
                sheet = Image.new("RGB", (tile_width * 2, tile_height * 5), "white")
                draw = ImageDraw.Draw(sheet)
                for index, row in enumerate(selected):
                    column, line = index % 2, index // 2
                    x0, y0 = column * tile_width, line * tile_height
                    for image_index, artifact_name in enumerate(
                        ("policy224_raw", "policy224_segmentation", "policy224_overlay")
                    ):
                        relative = row["artifacts"][artifact_name]["path"]
                        path = _regular_file(
                            Path(relative), root=artifact_root, label="contact-sheet input"
                        )
                        with Image.open(path) as source:
                            thumb = source.convert("RGB").resize(
                                (thumb_width, thumb_height), Image.Resampling.BILINEAR
                            )
                        sheet.paste(thumb, (x0 + image_index * thumb_width, y0 + label_height))
                    label = f"{row['task_id'].split('__', 1)[1]} ep{row['episode']} raw | seg | overlay"
                    draw.text((x0 + 3, y0 + 2), label, fill="black")
                out = artifact_root / "contact_sheets" / f"{scene_id}__{policy}__{camera_role}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                sheet.save(out, format="PNG", compress_level=6)
                reports[f"{scene_id}/{policy}/{camera_role}"] = _identity(
                    out, root=artifact_root
                )
    return reports


def _pairing_rows(
    observations: Mapping[tuple[str, str, int], Mapping[str, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(observations):
        scene_id, task_id, episode = key
        policies = observations[key]
        if set(policies) != set(POLICIES):
            raise CameraScorerGateError(f"paired reset lacks A0/A4 arms: {key}")
        a0, a4 = policies["A0"], policies["A4"]
        camera_deltas: dict[str, dict[str, Any]] = {}
        camera_checks: list[bool] = []
        for role in CAMERAS:
            pose0, pose4 = a0["cameras"][role], a4["cameras"][role]
            position_delta = float(
                np.max(
                    np.abs(
                        np.asarray(pose0["position_world_m"], dtype=float)
                        - np.asarray(pose4["position_world_m"], dtype=float)
                    )
                )
            )
            rotation_delta = float(
                np.max(
                    np.abs(
                        np.asarray(pose0["rotation_camera_to_world"], dtype=float)
                        - np.asarray(pose4["rotation_camera_to_world"], dtype=float)
                    )
                )
            )
            if role == "exterior":
                position_limit, rotation_limit = 1e-12, 1e-12
            else:
                # The wrist camera is physical robot geometry.  The two full
                # room collision constructions can induce sub-mm post-settle
                # joint differences, so bind a tight measured pose tolerance
                # instead of falsely demanding byte-identical dynamics.
                position_limit, rotation_limit = 0.002, 0.005
            passed = (
                position_delta <= position_limit
                and rotation_delta <= rotation_limit
                and pose0["fovy_deg"] == pose4["fovy_deg"]
            )
            camera_checks.append(passed)
            camera_deltas[role] = {
                "max_abs_position_delta_m": position_delta,
                "max_abs_rotation_matrix_delta": rotation_delta,
                "position_limit_m": position_limit,
                "rotation_matrix_limit": rotation_limit,
                "passed": passed,
            }
        checks = {
            "camera_pose_pairing": all(camera_checks),
            "jitter_replay_byte_equal": a0["jitter"] == a4["jitter"],
            "reset_seed_equal": a0["reset_seed"] == a4["reset_seed"],
            "reset_state_id_equal": a0["reset_state_id"] == a4["reset_state_id"],
        }
        rows.append(
            {
                "camera_deltas": camera_deltas,
                "checks": checks,
                "episode": episode,
                "passed": all(checks.values()),
                "reset_seed": a0["reset_seed"],
                "reset_state_id": a0["reset_state_id"],
                "scene_id": scene_id,
                "task_id": task_id,
            }
        )
    if len(rows) != 20:
        raise CameraScorerGateError(f"expected 20 paired resets, observed {len(rows)}")
    return rows


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for current, child_dirs, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.append(current_path)
        for directory in child_dirs:
            if (current_path / directory).is_symlink():
                raise CameraScorerGateError("output tree contains a symlink")
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink() or not path.is_file():
                raise CameraScorerGateError("output tree contains a special file")
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in reversed(directories):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _artifact_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CameraScorerGateError(f"artifact tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise CameraScorerGateError(f"artifact tree contains a special file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in {"manifest.json", "seal.json"}:
            continue
        inventory[relative] = _identity_without_path(path)
    return inventory


def run_qualifier_preflight(
    *,
    gate_id: str,
    cpu_freeze_id: str,
    cpu_producer_commit: str,
    expected_cpu_gate_sha256: str,
    expected_code_commit: str,
    menagerie_root: str | Path,
    expected_menagerie_commit: str,
) -> dict[str, Any]:
    """Run the 40-cell qualifier gate without EGL or a renderer.

    This is an independent scheduling preflight.  Passing it authorizes only
    submission of the camera gate; it never authorizes a policy run, fleet,
    paper claim, or headline result.  The later GPU gate recomputes the same
    qualifier rows instead of trusting this report as camera evidence.
    """
    gate_id = _validated_id(gate_id, label="camera gate ID")
    preflight_id = _validated_id(
        f"{gate_id}-qualifier-preflight", label="qualifier preflight ID"
    )
    code = _git_snapshot(expected_code_commit)
    root = _evidence_root()
    output_root = _experiment_root(root, preflight_id)
    if output_root == _experiment_root(root, cpu_freeze_id):
        raise CameraScorerGateError("qualifier preflight must be a fresh sibling")
    menagerie = _menagerie_snapshot(Path(menagerie_root), expected_menagerie_commit)
    upstream_before = validate_cpu_chain(
        root=root,
        cpu_freeze_id=cpu_freeze_id,
        cpu_producer_commit=cpu_producer_commit,
        expected_cpu_gate_sha256=expected_cpu_gate_sha256,
    )

    from robo.eval.episode_log import derive_reset_seed
    from robo.tasks import pi05_tasks

    final_gate: dict[str, Any]
    with _atomic_directory(output_root) as staging:
        _write_json(staging / "menagerie_inventory.json", menagerie)
        _write_json(staging / "upstream_revalidation.json", upstream_before["summary"])
        rows: list[dict[str, Any]] = []
        for scene_id in SCENE_IDS:
            for policy in POLICIES:
                suite = upstream_before["suites"][scene_id][policy]
                factory = upstream_before["factories"][scene_id][policy]
                task_rows = pi05_tasks._load_objects(factory)
                expected_tasks = list(cpu_pilot.expected_task_ids(scene_id))
                if [task.get("task_id") for task in suite["tasks"]] != expected_tasks:
                    raise CameraScorerGateError(
                        f"{scene_id}/{policy} qualifier task roster changed"
                    )
                env = _build_headless_droid_env(
                    scene_xml=str(suite["scene_xml"]),
                    base_pos=suite["robot"]["base_pos"],
                    base_yaw=suite["robot"]["base_yaw"],
                    table_box=suite["table"],
                    ext_cam=suite["ext_cam"],
                    exclude_objects=tuple(suite.get("exclude_objects", ())),
                    menagerie_root=Path(menagerie["root"]),
                )
                if hasattr(env, "renderer"):
                    raise CameraScorerGateError(
                        "CPU-only qualifier environment unexpectedly has a renderer"
                    )
                for task in suite["tasks"]:
                    task_id = str(task["task_id"])
                    for episode in range(EPISODES):
                        reset_seed = derive_reset_seed(BASE_SEED, task_id, episode)
                        reset_state_id = f"{task_id}__seed{BASE_SEED}__ep{episode}"
                        draw = np.random.RandomState(reset_seed).random_sample(2)
                        observation = env.reset(
                            settle_s=1.5,
                            jitter_body=task["target"],
                            jitter_xy=JITTER_XY_M,
                            jitter_uniform_draw=draw,
                            reset_seed=reset_seed,
                        )
                        provenance = env.last_reset_provenance
                        if not isinstance(provenance, Mapping):
                            raise CameraScorerGateError(
                                "headless environment did not publish reset provenance"
                            )
                        jitter = provenance.get("jitter")
                        settle = provenance.get("settle_protocol")
                        if not isinstance(jitter, Mapping) or not isinstance(settle, Mapping):
                            raise CameraScorerGateError(
                                "headless reset provenance schema differs"
                            )
                        settle_checks = _settle_contract_checks(provenance)
                        reset_checks = {
                            "headless_observation_has_no_images": observation == {},
                            "no_renderer_constructed": not hasattr(env, "renderer"),
                            "reset_seed_replayed": jitter.get("reset_seed") == reset_seed,
                            "target_jitter_replayed": (
                                jitter.get("body") == task["target"]
                                and jitter.get("max_abs_xy_m") == JITTER_XY_M
                                and jitter.get("uniform_draw_0_1") == draw.tolist()
                            ),
                        }
                        row_identity = {
                            "cell_id": f"{policy.lower()}__{reset_state_id}",
                            "episode": episode,
                            "policy_id": policy,
                            "reset_seed": reset_seed,
                            "reset_state_id": reset_state_id,
                            "scene_id": scene_id,
                            "task_id": task_id,
                            "target": task["target"],
                        }
                        qualifier = disambiguation_metric(
                            env, task, suite, task_rows, row_identity
                        )
                        checks = {
                            **reset_checks,
                            **settle_checks,
                            "qualifier_margin_passed": qualifier["passed"],
                        }
                        rows.append(
                            {
                                **row_identity,
                                "checks": checks,
                                "passed": all(checks.values()),
                                "qualifier": qualifier,
                                "reset_jitter": dict(jitter),
                                "settle_contract_checks": settle_checks,
                                "settle_protocol": dict(settle),
                            }
                        )

        if len(rows) != 40:
            raise CameraScorerGateError(
                f"qualifier preflight expected 40 cells, observed {len(rows)}"
            )
        _write_jsonl(staging / "qualifier_metrics.jsonl", rows)
        qualifier_failed_cells = [
            row for row in rows if not row["qualifier"]["passed"]
        ]
        settle_failed_cells = [
            row for row in rows
            if not all(row["settle_contract_checks"].values())
        ]
        reset_failed_cells = [
            row for row in rows
            if not all(
                value
                for name, value in row["checks"].items()
                if name not in row["settle_contract_checks"]
                and name != "qualifier_margin_passed"
            )
        ]
        blocking_issues = sorted(
            [
                f"qualifier_margin_failed:{row['cell_id']}"
                for row in qualifier_failed_cells
            ]
            + [
                f"settle_contract_failed:{row['cell_id']}"
                for row in settle_failed_cells
            ]
            + [
                f"headless_reset_contract_failed:{row['cell_id']}"
                for row in reset_failed_cells
            ]
        )
        allowed = not blocking_issues
        final_gate = {
            "blocking_issues": blocking_issues,
            "camera_job_submission_allowed": allowed,
            "camera_validator_code": code,
            "cell_coverage": {
                "cells": len(rows),
                "episodes_per_task": EPISODES,
                "policies": list(POLICIES),
                "scenes": list(SCENE_IDS),
                "tasks_per_scene": 2,
            },
            "created_utc": _utc_now(),
            "evidence_root": str(root),
            "gate_id": preflight_id,
            "gpu_work_executed": False,
            "headline_eligible": False,
            "large_rollout_launch_allowed": False,
            "manifest_kind": "e4_cpu_qualifier_preflight_gate",
            "menagerie": {
                key: value for key, value in menagerie.items() if key != "files"
            },
            "paper_ready": False,
            "preflight_cells_failed": sum(not bool(row["passed"]) for row in rows),
            "preflight_cells_passed": sum(bool(row["passed"]) for row in rows),
            "qualifier_margin_cells_failed": len(qualifier_failed_cells),
            "qualifier_margin_cells_passed": len(rows) - len(qualifier_failed_cells),
            "real_policy_infra_smoke_allowed": False,
            "renderer_constructed": False,
            "reset_contract_cells_failed": len(reset_failed_cells),
            "reset_contract_cells_passed": len(rows) - len(reset_failed_cells),
            "result_role": "independent_cpu_scheduling_preflight_only",
            "schema_version": SCHEMA_VERSION,
            "settle_contract_cells_failed": len(settle_failed_cells),
            "settle_contract_cells_passed": len(rows) - len(settle_failed_cells),
            "upstream": upstream_before["summary"],
        }
        _write_json(staging / "gate.json", final_gate)

        menagerie_after = _menagerie_snapshot(
            Path(menagerie["root"]), expected_menagerie_commit
        )
        if menagerie_after != menagerie:
            raise CameraScorerGateError(
                "menagerie closure changed during qualifier preflight"
            )
        upstream_after = validate_cpu_chain(
            root=root,
            cpu_freeze_id=cpu_freeze_id,
            cpu_producer_commit=cpu_producer_commit,
            expected_cpu_gate_sha256=expected_cpu_gate_sha256,
        )
        if upstream_after["summary"] != upstream_before["summary"]:
            raise CameraScorerGateError(
                "CPU evidence closure changed during qualifier preflight"
            )

        manifest = {
            "camera_validator_code": code,
            "cpu_gate_sha256": expected_cpu_gate_sha256,
            "files": _artifact_inventory(staging),
            "gate_id": preflight_id,
            "manifest_kind": "e4_cpu_qualifier_preflight_artifacts",
            "output_root": output_root.relative_to(root).as_posix(),
            "schema_version": SCHEMA_VERSION,
        }
        _write_json(staging / "manifest.json", manifest)
        seal = {
            "manifest_kind": "e4_cpu_qualifier_preflight_artifacts",
            "members": {
                "gate.json": _identity_without_path(staging / "gate.json"),
                "manifest.json": _identity_without_path(staging / "manifest.json"),
            },
            "schema_version": SCHEMA_VERSION,
        }
        _write_json(staging / "seal.json", seal)
        _fsync_tree(staging)
    return final_gate


def run_gate(
    *,
    gate_id: str,
    cpu_freeze_id: str,
    cpu_producer_commit: str,
    expected_cpu_gate_sha256: str,
    expected_code_commit: str,
    menagerie_root: str | Path,
    expected_menagerie_commit: str,
) -> dict[str, Any]:
    gate_id = _validated_id(gate_id, label="camera gate ID")
    code = _git_snapshot(expected_code_commit)
    root = _evidence_root()
    output_root = _experiment_root(root, gate_id)
    if output_root == _experiment_root(root, cpu_freeze_id):
        raise CameraScorerGateError("camera gate must be a fresh sibling of the CPU freeze")
    runtime = _gpu_runtime_attestation()
    menagerie = _menagerie_snapshot(Path(menagerie_root), expected_menagerie_commit)
    openpi = _openpi_snapshot()
    upstream_before = validate_cpu_chain(
        root=root,
        cpu_freeze_id=cpu_freeze_id,
        cpu_producer_commit=cpu_producer_commit,
        expected_cpu_gate_sha256=expected_cpu_gate_sha256,
    )

    import mujoco
    from robo.envs.pi05_env import DroidSimEnv
    from robo.eval.episode_log import derive_reset_seed
    from robo.tasks import pi05_tasks

    runtime["mujoco_version"] = mujoco.__version__
    final_gate: dict[str, Any]
    with _atomic_directory(output_root) as staging:
        _write_json(staging / "menagerie_inventory.json", menagerie)
        _write_json(staging / "openpi_resize_identity.json", openpi)
        _write_json(staging / "upstream_revalidation.json", upstream_before["summary"])
        camera_rows: list[dict[str, Any]] = []
        workspace_rows: list[dict[str, Any]] = []
        disambiguation_rows: list[dict[str, Any]] = []
        scorer_rows: list[dict[str, Any]] = []
        cell_rows: list[dict[str, Any]] = []
        pairing_observations: dict[
            tuple[str, str, int], dict[str, dict[str, Any]]
        ] = {}

        for scene_id in SCENE_IDS:
            for policy in POLICIES:
                suite = upstream_before["suites"][scene_id][policy]
                factory = upstream_before["factories"][scene_id][policy]
                task_rows = pi05_tasks._load_objects(factory)
                expected_tasks = list(cpu_pilot.expected_task_ids(scene_id))
                if [task.get("task_id") for task in suite["tasks"]] != expected_tasks:
                    raise CameraScorerGateError(f"{scene_id}/{policy} task roster changed")
                env = DroidSimEnv(
                    str(suite["scene_xml"]),
                    suite["robot"]["base_pos"],
                    suite["robot"]["base_yaw"],
                    table_box=suite["table"],
                    ext_cam=suite["ext_cam"],
                    exclude_objects=tuple(suite.get("exclude_objects", ())),
                    render_wh=(RENDER_WIDTH, RENDER_HEIGHT),
                    menagerie_root=Path(menagerie["root"]),
                )
                env._task_rows = task_rows
                try:
                    for task in suite["tasks"]:
                        task_id = str(task["task_id"])
                        target_matches = [
                            row for row in task_rows if row.get("name") == task.get("target")
                        ]
                        if len(target_matches) != 1:
                            raise CameraScorerGateError(
                                f"{scene_id}/{policy}/{task_id} target metadata differs"
                            )
                        target_row = target_matches[0]
                        for episode in range(EPISODES):
                            reset_seed = derive_reset_seed(BASE_SEED, task_id, episode)
                            reset_state_id = f"{task_id}__seed{BASE_SEED}__ep{episode}"
                            draw = np.random.RandomState(reset_seed).random_sample(2)
                            obs = env.reset(
                                settle_s=1.5,
                                jitter_body=task["target"],
                                jitter_xy=JITTER_XY_M,
                                jitter_uniform_draw=draw,
                                reset_seed=reset_seed,
                            )
                            reset_provenance = env.last_reset_provenance
                            if not isinstance(reset_provenance, Mapping):
                                raise CameraScorerGateError("environment did not publish reset provenance")
                            jitter = reset_provenance.get("jitter")
                            if (
                                not isinstance(jitter, Mapping)
                                or jitter.get("reset_seed") != reset_seed
                                or jitter.get("body") != task["target"]
                                or jitter.get("max_abs_xy_m") != JITTER_XY_M
                                or jitter.get("uniform_draw_0_1") != draw.tolist()
                            ):
                                raise CameraScorerGateError("reset jitter provenance differs")
                            settle_checks = _settle_contract_checks(reset_provenance)
                            settle_protocol = reset_provenance.get("settle_protocol")
                            row_identity = {
                                "cell_id": f"{policy.lower()}__{reset_state_id}",
                                "episode": episode,
                                "policy_id": policy,
                                "reset_seed": reset_seed,
                                "reset_state_id": reset_state_id,
                                "scene_id": scene_id,
                                "task_id": task_id,
                                "target": task["target"],
                            }
                            relative_cell = (
                                Path("cells")
                                / scene_id
                                / policy
                                / task_id.split("__", 1)[1]
                                / f"seed{BASE_SEED}_ep{episode}"
                            )
                            cell_dir = staging / relative_cell
                            eligible_same_label = _eligible_same_label_objects(
                                env, task, task_rows
                            )
                            distractors = [
                                name for name in eligible_same_label
                                if name != str(task["target"])
                            ]
                            if not distractors:
                                raise CameraScorerGateError(
                                    f"{scene_id}/{policy}/{task_id} has no language distractor"
                                )
                            exterior_depth = _render_depth(env, env.info["ext_cam"])
                            depth_path = cell_dir / "exterior_depth.npy"
                            _save_npy(depth_path, exterior_depth)
                            camera_by_role: dict[str, dict[str, Any]] = {}
                            camera_specs = {
                                "exterior": (
                                    env.info["ext_cam"],
                                    "observation/exterior_image_1_left",
                                ),
                                "wrist": (
                                    env.info["wrist_cam"],
                                    "observation/wrist_image_left",
                                ),
                            }
                            current_camera_rows: list[dict[str, Any]] = []
                            for camera_role in CAMERAS:
                                camera_name, observation_key = camera_specs[camera_role]
                                segmentation = _render_segmentation(env, camera_name)
                                metric = camera_metric(
                                    env=env,
                                    raw=np.asarray(obs[observation_key]),
                                    segmentation=segmentation,
                                    camera_role=camera_role,
                                    camera_name=camera_name,
                                    target=str(task["target"]),
                                    distractors=distractors,
                                    expected_ext_cam=suite["ext_cam"],
                                    output_dir=cell_dir,
                                    artifact_root=staging,
                                    row_identity=row_identity,
                                )
                                if camera_role == "exterior":
                                    metric["scan_frame"] = suite["ext_cam"].get("frame")
                                camera_rows.append(metric)
                                current_camera_rows.append(metric)
                                camera_by_role[camera_role] = metric["camera"]

                            workspace = workspace_metric(
                                env,
                                task,
                                suite,
                                target_row,
                                exterior_depth,
                                row_identity,
                            )
                            workspace["exterior_depth_artifact"] = _identity(
                                depth_path, root=staging
                            )
                            disambiguation = disambiguation_metric(
                                env, task, suite, task_rows, row_identity
                            )
                            scorer = scorer_replay(env, task, row_identity)
                            workspace_rows.append(workspace)
                            disambiguation_rows.append(disambiguation)
                            scorer_rows.append(scorer)
                            cell_checks = {
                                "both_camera_views_passed": all(
                                    row["passed"] for row in current_camera_rows
                                ),
                                "disambiguation_margin_passed": disambiguation["passed"],
                                "reset_settle_contract_passed": all(
                                    settle_checks.values()
                                ),
                                "scorer_predicate_replay_passed": scorer["passed"],
                                "workspace_reach_envelope_passed": workspace["passed"],
                            }
                            cell_rows.append(
                                {
                                    **row_identity,
                                    "checks": cell_checks,
                                    "passed": all(cell_checks.values()),
                                    "reset_jitter": dict(jitter),
                                    "settle_contract_checks": settle_checks,
                                    "settle_protocol": settle_protocol,
                                }
                            )
                            pairing_observations.setdefault(
                                (scene_id, task_id, episode), {}
                            )[policy] = {
                                "cameras": camera_by_role,
                                "jitter": dict(jitter),
                                "reset_seed": reset_seed,
                                "reset_state_id": reset_state_id,
                            }
                finally:
                    env.renderer.close()

        if len(cell_rows) != 40 or len(camera_rows) != 80:
            raise CameraScorerGateError(
                f"cell coverage differs: cells={len(cell_rows)}, cameras={len(camera_rows)}"
            )
        if len(workspace_rows) != 40 or len(disambiguation_rows) != 40 \
                or len(scorer_rows) != 40:
            raise CameraScorerGateError("per-cell diagnostic coverage differs")
        pairing_rows = _pairing_rows(pairing_observations)
        contact_sheets = _contact_sheets(camera_rows, artifact_root=staging)
        _write_jsonl(staging / "camera_metrics.jsonl", camera_rows)
        _write_jsonl(staging / "workspace_metrics.jsonl", workspace_rows)
        _write_jsonl(staging / "disambiguation_metrics.jsonl", disambiguation_rows)
        _write_jsonl(staging / "scorer_replay.jsonl", scorer_rows)
        _write_jsonl(staging / "pairing_metrics.jsonl", pairing_rows)
        _write_jsonl(staging / "cells.jsonl", cell_rows)

        blocking_issues: list[str] = []
        for row in camera_rows:
            if not row["passed"]:
                blocking_issues.append(
                    f"camera_validation_failed:{row['cell_id']}:{row['camera_role']}"
                )
        for row in cell_rows:
            if not all(row["settle_contract_checks"].values()):
                blocking_issues.append(
                    f"reset_settle_contract_failed:{row['cell_id']}"
                )
        for label, rows in (
            ("workspace_reach_envelope_failed", workspace_rows),
            ("task_disambiguation_margin_failed", disambiguation_rows),
            ("scorer_predicate_replay_failed", scorer_rows),
            ("paired_reset_or_camera_pose_failed", pairing_rows),
        ):
            for row in rows:
                if not row["passed"]:
                    blocking_issues.append(f"{label}:{row['reset_state_id']}")
        blocking_issues = sorted(set(blocking_issues))
        allowed = not blocking_issues
        final_gate = {
            "blocking_issues": blocking_issues,
            "camera_validator_code": code,
            "cell_coverage": {
                "camera_rows": len(camera_rows),
                "cells": len(cell_rows),
                "episodes_per_task": EPISODES,
                "paired_resets": len(pairing_rows),
                "policies": list(POLICIES),
                "scenes": list(SCENE_IDS),
                "tasks_per_scene": 2,
            },
            "contact_sheets": contact_sheets,
            "cpu_preflight": upstream_before["summary"],
            "created_utc": _utc_now(),
            "evidence_root": str(root),
            "gate_id": gate_id,
            "headline_eligible": False,
            "large_rollout_launch_allowed": False,
            "manifest_kind": "e4_camera_workspace_scorer_gate",
            "menagerie": {
                key: value for key, value in menagerie.items() if key != "files"
            },
            "paper_ready": False,
            "openpi_resize_source": openpi,
            "real_policy_infra_smoke_allowed": allowed,
            "runtime": runtime,
            "schema_version": SCHEMA_VERSION,
            "scope": {
                "camera_validation": "real MuJoCo EGL RGB plus geom segmentation",
                "ik_validated": False,
                "oracle_available": False,
                "scorer_validation": "direct-state predicate replay only",
                "workspace_validation": "2d radial-plus-front-cone reach envelope only",
            },
            "semantic_summary": {
                "camera_rows_passed": sum(bool(row["passed"]) for row in camera_rows),
                "cells_passed": sum(bool(row["passed"]) for row in cell_rows),
                "disambiguation_rows_passed": sum(
                    bool(row["passed"]) for row in disambiguation_rows
                ),
                "pairing_rows_passed": sum(bool(row["passed"]) for row in pairing_rows),
                "reset_settle_contract_rows_passed": sum(
                    all(row["settle_contract_checks"].values()) for row in cell_rows
                ),
                "scorer_rows_passed": sum(bool(row["passed"]) for row in scorer_rows),
                "workspace_rows_passed": sum(bool(row["passed"]) for row in workspace_rows),
            },
            "study_scope": "region_only_engineering_pilot",
        }
        _write_json(staging / "gate.json", final_gate)

        # Revalidate both immutable input closures after the last render and
        # before publication.  A concurrent edit cannot be hidden by the
        # identities recorded at startup.
        menagerie_after = _menagerie_snapshot(
            Path(menagerie["root"]), expected_menagerie_commit
        )
        if menagerie_after != menagerie:
            raise CameraScorerGateError("menagerie closure changed during camera gate")
        if _openpi_snapshot() != openpi:
            raise CameraScorerGateError("OpenPI resize source changed during camera gate")
        upstream_after = validate_cpu_chain(
            root=root,
            cpu_freeze_id=cpu_freeze_id,
            cpu_producer_commit=cpu_producer_commit,
            expected_cpu_gate_sha256=expected_cpu_gate_sha256,
        )
        if upstream_after["summary"] != upstream_before["summary"]:
            raise CameraScorerGateError("CPU evidence closure changed during camera gate")

        manifest = {
            "camera_validator_code": code,
            "cpu_gate_sha256": expected_cpu_gate_sha256,
            "files": _artifact_inventory(staging),
            "gate_id": gate_id,
            "manifest_kind": "e4_camera_workspace_scorer_artifacts",
            "output_root": output_root.relative_to(root).as_posix(),
            "schema_version": SCHEMA_VERSION,
        }
        _write_json(staging / "manifest.json", manifest)
        seal = {
            "manifest_kind": "e4_camera_workspace_scorer_artifacts",
            "members": {
                "gate.json": _identity_without_path(staging / "gate.json"),
                "manifest.json": _identity_without_path(staging / "manifest.json"),
            },
            "schema_version": SCHEMA_VERSION,
        }
        _write_json(staging / "seal.json", seal)
        _fsync_tree(staging)
    return final_gate


def validate_winner_chain(config: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Authenticate historical winner inputs without changing shared validators."""
    from robo.eval import e4_candidate_screen as screen
    from robo.eval import e4_reset_eligibility as eligibility
    from robo.eval.harness_spec import load_harness_spec
    from robo.eval.harness_validation import read_jsonl, validate_saved_treatment_records
    from robo.eval.paired_runner import plan_reset_states
    import yaml

    producer = config["producer_commit"]
    historical = producer == "3d3781712bdc36f2199426b766869ecda4b51c71"
    if not historical:
        if config.get("cpu_source_mode") != "fresh_canonical_cpu":
            raise CameraScorerGateError("winner CPU producer differs")
        _git_snapshot(producer)  # new CPU source must be this exact clean source

    if config["study_scope"] != "one_target_engineering_smoke" or config["paper_ready"] is not False:
        raise CameraScorerGateError("winner camera scope differs")
    inputs = {}
    for name, identity in config["inputs"].items():
        path, _ = _validate_path_identity(identity, root=root, label=name)
        inputs[name] = path
    bundle = screen._validate_bundle(inputs["cpu_gate"].parent, root=root,
        expected_kind="e4_one_target_winner_smoke_artifacts")
    gate = json.loads(inputs["cpu_gate"].read_text())
    if gate["code"] != {"code_root": str(CODE_ROOT), "commit": producer, "dirty": False}:
        raise CameraScorerGateError("winner CPU source identity differs")
    if gate["study_scope"] != config["study_scope"] or gate["planned_cells"] != 10:
        raise CameraScorerGateError("winner CPU scope or population differs")
    task_bundle = gate["task_bundle"]
    scene_id = "d755b3d9d8"
    task_ids = ["d755b3d9d8__obj_05_to_region"]
    factories = {p: Path(task_bundle["factories"][p]) for p in POLICIES}
    from run.icra2027 import e4_robust_floor_support_winners as winners
    common_static = root / "outputs/icra2027" / winners.EXTERNAL_SWEEP_ID / "variants" / winners.WINNER_VARIANT_ID / "common_static"
    static_identity = winners._validate_common_static(common_static, root=root,
        spec_sha256=winners.EXPECTED_WINNER_IDENTITY["variant_spec_sha256"])
    materializations = {}
    for policy, factory in factories.items():
        recorded = gate["rematerializations"][policy]
        keys = ("e3_claim_status", "e3_code_commit", "e3_freeze_id", "e3_root",
                "manifest_sha256", "materializer_commit", "policy_id", "roster",
                "scene_id", "study_scope", "validator_commit")
        materializations[policy] = _validate_materialization_chain(factory,
            root=root, scene_id=scene_id, policy=policy,
            recorded_summary={k: recorded[k] for k in keys}, cpu_commit=producer)
        # Capture-derived exports are pinned separately because their complete
        # tree was not included in the historical CPU bundle seal.
        observed = _tree_inventory(factory, relative_to=root)
        if _canonical_hash(observed) != config["factory_closure_sha256"][policy]:
            raise CameraScorerGateError(f"{policy} frozen factory/export closure differs")
        # Winner exports intentionally retain their original sealed meshdir;
        # never rewrite it to pretend the fresh rematerializer generated them.
        import xml.etree.ElementTree as ET
        xml = ET.parse(factory / "sim_export/scene.xml").getroot()
        meshdir = Path(xml.find("compiler").get("meshdir"))
        for mesh in xml.findall("./asset/mesh"):
            source_path = meshdir / mesh.get("file")
            if str(source_path) not in gate["source_identities"] and not source_path.is_relative_to(common_static):
                raise CameraScorerGateError("winner XML mesh lacks original producer identity")
    task = _validate_task_bundle_chain(inputs["task_manifest"].parent, root=root,
        scene_id=scene_id, expected_manifest_sha256=task_bundle["bundle_manifest_sha256"],
        cpu_commit=producer, materializations=materializations,
        expected_task_ids=task_ids, expected_max_tasks=None)
    for name, identity in gate["source_identities"].items():
        _validate_path_identity(identity, root=root, label=name)
    suites = {p: task["suites"][p] for p in POLICIES}
    contracts = screen._prepared_task_contracts({"gate": {**gate, "footprint_replays":
        json.loads((bundle["directory"] / "footprint_replays.json").read_text())},
        "suites": suites, "factories": factories, "task_bundle": task_bundle})
    rows = read_jsonl(bundle["directory"] / "metrics.jsonl")
    screen._replay_qualifier_metrics(rows, scene_id=scene_id, expected_task_contracts=contracts)
    harness = yaml.safe_load(inputs["harness_config"].read_text())
    spec = load_harness_spec(harness)
    states = plan_reset_states(harness["scenes"], [0], 5)
    index = eligibility.index_reset_eligibility(rows, states,
        {t.scene: t.id for t in spec.treatments.values()}, harness["cpu_reset_eligibility"])
    if not historical:
        eligibility.load_reset_eligibility(harness, spec, states, root=root)
        # A camera declaration correction is shared by both arms. All other
        # suite fields and the exact reset state reference remain frozen.
        for policy in POLICIES:
            reference = json.loads(inputs[f"reference_task_{policy}"].read_text())
            _assert_common_camera_correction(suites[policy], reference)
    records = read_jsonl(inputs["ledger"])
    reference_config = yaml.safe_load(inputs.get("reference_harness_config", inputs["harness_config"]).read_text())
    reference_spec = load_harness_spec(reference_config)
    validation = validate_saved_treatment_records(records, reference_spec,
        set(reference_config["contract"]["reset_ids"]), manifest_root=root)
    if not validation["ok"] or validation["records"] != 10:
        raise CameraScorerGateError("winner original ledger validation failed")
    for record in records:
        evidence = index[(record["treatment_id"], record["reset_state_id"])]
        if evidence["passed"] != (record["outcome"] != "build_failure"):
            raise CameraScorerGateError("winner original ledger eligibility differs")
    menagerie = _menagerie_snapshot(Path(config["menagerie_root"]), EXPECTED_MENAGERIE_COMMIT)
    if {k:v for k,v in menagerie.items() if k != "files"} != gate["menagerie"]:
        raise CameraScorerGateError("winner CPU/runtime menagerie differs")
    if _openpi_snapshot() != config["openpi_resize_identity"]:
        raise CameraScorerGateError("winner policy resize source differs")
    return {"suites": suites, "factories": factories, "rows": rows,
        "records": records, "menagerie": menagerie, "harness": harness,
        "summary": {"producer_commit": producer, "reset_reference_producer_commit": "3d3781712bdc36f2199426b766869ecda4b51c71",
        "camera_correction_shared_by_both_arms": not historical, "inputs": config["inputs"],
        "factory_closure_sha256": config["factory_closure_sha256"],
        "common_static": static_identity,
        "menagerie": gate["menagerie"], "cpu_paired_gate_pass": False,
        "planned": 10, "eligible": sum(r["passed"] for r in rows),
        "ledger_validation": validation}}


def _assert_common_camera_correction(current, reference):
    from robo.rigs.pi05_rig import lookat_quat
    if {k:v for k,v in reference.items() if k not in {"scene_xml", "ext_cam"}} != {k:v for k,v in current.items() if k not in {"scene_xml", "ext_cam"}}:
        raise CameraScorerGateError("camera correction changed another task/robot/table field")
    old_cam = reference["ext_cam"]
    expected = {**old_cam, "mode": "world", "frame": "construction_world_table_camera",
                "quat_wxyz": lookat_quat(old_cam["pos"], old_cam["target"]).tolist()}
    if current["ext_cam"] != expected:
        raise CameraScorerGateError("camera correction differs from original declared world position/target/FOV")


def write_winner_camera_config(*, cpu_bundle: str | Path, harness_config: str | Path,
                               freeze_id: str, out: str | Path, expected_code_commit: str):
    """Bind freshly frozen paired camera tasks to the immutable v18 reset reference."""
    from robo.eval import e4_candidate_screen as screen
    root = _evidence_root()
    code = _git_snapshot(expected_code_commit)
    config = json.loads((CODE_ROOT / "configs/experiments/icra2027/e4_winner_camera.json").read_text())
    original_task_manifest, _ = _validate_path_identity(config["inputs"]["task_manifest"], root=root, label="reference task manifest")
    original_manifest = json.loads(original_task_manifest.read_text())
    config["inputs"]["reference_task_manifest"] = config["inputs"]["task_manifest"]
    config["inputs"]["reference_harness_config"] = config["inputs"]["harness_config"]
    for policy in POLICIES:
        path = original_task_manifest.parent / POLICY_TO_TASK_FILE[policy]
        _validate_identity(path, original_manifest["files"][path.name], label="reference task suite")
        config["inputs"][f"reference_task_{policy}"] = _identity(path, root=root)
    bundle = screen._validate_bundle(Path(cpu_bundle), root=root,
        expected_kind="e4_one_target_winner_smoke_artifacts")
    gate_path = bundle["directory"] / "gate.json"
    cpu = json.loads(gate_path.read_text())
    if cpu["code"] != code or cpu["screen_id"] != freeze_id:
        raise CameraScorerGateError("fresh camera config CPU source or freeze differs")
    config.update(freeze_id=_validated_id(freeze_id, label="winner freeze"),
        producer_commit=code["commit"], cpu_source_mode="fresh_canonical_cpu",
        camera_change_scope="common correction of world-coordinate declaration in both construction arms; not a measured treatment effect")
    task_manifest = Path(cpu["task_bundle"]["planning_tasks"]).parent / "manifest.json"
    config["inputs"].update(cpu_gate=_identity(gate_path, root=root),
        task_manifest=_identity(task_manifest, root=root),
        harness_config=_identity(Path(harness_config), root=root))
    config["factory_closure_sha256"] = {p:_canonical_hash(_tree_inventory(
        Path(cpu["task_bundle"]["factories"][p]), relative_to=root)) for p in POLICIES}
    validate_winner_chain(config, root=root)
    with Path(out).open("xb") as stream:
        stream.write(_json_bytes(config))
    return config


def _winner_expected_world_camera(suite: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the frozen rig camera convention without changing the camera."""
    from robo.rigs.pi05_rig import lookat_quat
    camera = dict(suite["ext_cam"])
    if camera.get("mode") == "world":
        return camera
    base = np.asarray(suite["robot"]["base_pos"], dtype=float)
    yaw = float(suite["robot"]["base_yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    position = base + rotation @ np.asarray(camera["pos"], dtype=float)
    target = camera.get("target")
    target = base + rotation @ np.array([0.55, 0., 0.10]) if target is None else np.asarray(target, dtype=float)
    return {"mode": "world", "frame": "resolved_from_frozen_rig_base_offset_lookat",
            "pos": position.tolist(), "quat_wxyz": lookat_quat(position, target).tolist(),
            "fovy": camera.get("fovy", 68.0)}


def _winner_render_cell(env, task, suite, task_rows, row, original, staging):
    """Reuse canonical diagnostics on exactly one authenticated eligible reset."""
    if row["passed"] is not True:
        raise CameraScorerGateError("invalid CPU arm must not reach renderer")
    env._task_rows = task_rows
    jitter = row["reset_jitter"]
    obs = env.reset(settle_s=1.5, jitter_body=task["target"],
        jitter_xy=JITTER_XY_M, jitter_uniform_draw=np.asarray(jitter["uniform_draw_0_1"]),
        reset_seed=row["reset_seed"])
    if env.last_reset_provenance != original["reset_provenance"]:
        raise CameraScorerGateError("winner runtime reset differs from original v18 execution")
    return _diagnose_reset(env, task, suite, task_rows, row, obs, staging,
                           reset_check="exact_v18_reset_replay")


def _diagnose_reset(env, task, suite, task_rows, row, obs, staging, *, reset_check):
    """Shared camera/workspace/scorer producer after an authenticated reset."""
    identity = {k:row[k] for k in ("cell_id", "episode", "policy_id", "reset_seed",
                                  "scene_id", "task_id", "target")}
    identity["reset_state_id"] = f"{row['task_id']}__seed0__ep{row['episode']}"
    cell_dir = staging / "cells" / row["cell_id"]
    target_row = next(r for r in task_rows if r["name"] == task["target"])
    disambiguation = disambiguation_metric(env, task, suite, task_rows, identity)
    same_label = _eligible_same_label_objects(env, task, task_rows)
    distractors = [n for n in same_label if n != task["target"]]
    singleton = task if same_label == [task["target"]] and disambiguation["passed"] else None
    cameras = []
    for role, camera, key in (("exterior", env.info["ext_cam"], "observation/exterior_image_1_left"),
                              ("wrist", env.info["wrist_cam"], "observation/wrist_image_left")):
        cameras.append(camera_metric(env=env, raw=np.asarray(obs[key]),
            segmentation=_render_segmentation(env, camera), camera_role=role,
            camera_name=camera, target=task["target"], distractors=distractors,
            expected_ext_cam=_winner_expected_world_camera(suite), output_dir=cell_dir,
            artifact_root=staging, row_identity=identity, singleton_task=singleton))
    workspace = workspace_metric(env, task, suite, target_row,
        _render_depth(env, env.info["ext_cam"]), identity)
    scorer = scorer_replay(env, task, identity)
    checks = {"camera": all(r["passed"] for r in cameras), "workspace": workspace["passed"],
              "disambiguation": disambiguation["passed"], "scorer": scorer["passed"],
              reset_check: True}
    return {**identity, "executed": True, "outcome": "diagnostic_pass" if all(checks.values()) else "diagnostic_failure",
        "checks": checks, "passed": all(checks.values()), "camera_metrics": cameras,
        "frozen_camera_specification": suite["ext_cam"],
        "resolved_expected_camera": _winner_expected_world_camera(suite),
        "workspace_metrics": workspace, "disambiguation_metrics": disambiguation,
        "scorer_metrics": scorer, "reset_provenance": env.last_reset_provenance}


def winner_camera_pose_preflight(upstream: Mapping[str, Any]) -> dict[str, Any]:
    """CPU-only mj_forward audit of existing exterior/wrist camera semantics."""
    import mujoco
    reports = []
    eligible_policies = sorted({r["policy_id"] for r in upstream["rows"] if r["passed"]})
    for policy in eligible_policies:
        suite = upstream["suites"][policy]
        env = _build_headless_droid_env(scene_xml=suite["scene_xml"],
            base_pos=suite["robot"]["base_pos"], base_yaw=suite["robot"]["base_yaw"],
            table_box=suite["table"], ext_cam=suite["ext_cam"],
            exclude_objects=tuple(suite["exclude_objects"]),
            menagerie_root=Path(upstream["menagerie"]["root"]))
        mujoco.mj_forward(env.model, env.data)
        expected_ext = _winner_expected_world_camera(suite)
        exterior = exterior_camera_replay(env, expected_ext)
        wrist_id = env.model.camera(env.info["wrist_cam"]).id
        body_id = int(env.model.cam_bodyid[wrist_id])
        rotation_flat = np.empty(9)
        mujoco.mju_quat2Mat(rotation_flat, env.model.cam_quat[wrist_id])
        body_rotation = env.data.xmat[body_id].reshape(3, 3)
        expected_wrist_rotation = body_rotation @ rotation_flat.reshape(3, 3)
        expected_wrist_position = env.data.xpos[body_id] + body_rotation @ env.model.cam_pos[wrist_id]
        wrist = _camera_pose(env, env.info["wrist_cam"])
        wrist_checks = {
            "position_delta_at_most_1e-6_m": float(np.linalg.norm(np.asarray(wrist["position_world_m"])-expected_wrist_position)) <= 1e-6,
            "rotation_delta_at_most_1e-6_rad": _rotation_geodesic_rad(np.asarray(wrist["rotation_camera_to_world"]), expected_wrist_rotation) <= 1e-6,
            "fovy_matches_pinned_model": wrist["fovy_deg"] == float(env.model.cam_fovy[wrist_id])}
        reports.append({"policy_id": policy, "passed": exterior["passed"] and all(wrist_checks.values()),
            "original_camera_specification": suite["ext_cam"], "expected_exterior": expected_ext,
            "observed_exterior": _camera_pose(env, env.info["ext_cam"]), "exterior_replay": exterior,
            "expected_wrist": {"position_world_m": expected_wrist_position.tolist(),
                "rotation_camera_to_world": expected_wrist_rotation.tolist(), "attachment_body": env.model.body(body_id).name},
            "observed_wrist": wrist, "wrist_checks": wrist_checks,
            "renderer_constructed": False, "simulation_steps": 0,
            "state_scope": "canonical model initial state; CPU camera algebra only, not a new reset/episode"})
    return {"passed": bool(reports) and all(r["passed"] for r in reports), "reports": reports}


def run_winner_gate(*, config_path: str | Path, expected_code_commit: str,
                    preflight_only: bool = False) -> dict[str, Any]:
    """Ten planned cells; invalid arms remain unexecuted with null telemetry."""
    config_identity = _identity_without_path(Path(config_path))
    config = json.loads(Path(config_path).read_text())
    code = _git_snapshot(expected_code_commit)
    root = _evidence_root()
    before = validate_winner_chain(config, root=root)
    if preflight_only:
        camera_pose = winner_camera_pose_preflight(before)
        return {"camera_job_submission_allowed": before["summary"]["eligible"] > 0 and camera_pose["passed"],
                "paper_ready": False, "code": code, "upstream": before["summary"],
                "camera_pose_preflight": camera_pose}
    profile = config["gpu_profile"]
    if profile != {"node": "hala", "name": "NVIDIA RTX A6000", "compute_capability": "8.6", "gres": "a6000:1"}:
        raise CameraScorerGateError("winner GPU profile differs from frozen Hala allocation")
    runtime = _gpu_runtime_attestation(profile)
    if runtime["qualifier_preflight_gate_sha256"] != config["inputs"]["cpu_gate"]["sha256"]:
        raise CameraScorerGateError("winner runtime CPU gate anchor differs")
    from robo.envs.pi05_env import DroidSimEnv
    from robo.tasks import pi05_tasks
    output = root / "outputs/icra2027" / config["freeze_id"] / "harness/camera_scorer"
    records = {(r["treatment_id"], r["reset_state_id"]):r for r in before["records"]}
    with _atomic_directory(output, preserve_failed=True) as staging:
        cells = []
        for row in before["rows"]:
            reset_id = f"{row['task_id']}__seed0__ep{row['episode']}"
            record = records[(f"{row['policy_id'].lower()}_raster", reset_id)]
            if not row["passed"]:
                cells.append({"cell_id": row["cell_id"], "policy_id": row["policy_id"],
                    "reset_state_id": reset_id, "executed": False, "outcome": "build_failure",
                    "passed": False, "cpu_failed_checks": [k for k,v in row["checks"].items() if not v],
                    "camera_metrics": None, "workspace_metrics": None,
                    "disambiguation_metrics": None, "scorer_metrics": None, "reset_provenance": None})
                _write_json(staging / "cell_records" / f"{row['cell_id']}.json", cells[-1])
                continue
            suite = before["suites"][row["policy_id"]]
            factory = before["factories"][row["policy_id"]]
            task = next(t for t in suite["tasks"] if t["task_id"] == row["task_id"])
            manifest_path = _regular_file(Path(record["manifest_path"]), root=root, label="original rollout manifest")
            original = json.loads(manifest_path.read_text())
            env = DroidSimEnv(suite["scene_xml"], suite["robot"]["base_pos"], suite["robot"]["base_yaw"],
                table_box=suite["table"], ext_cam=suite["ext_cam"],
                exclude_objects=tuple(suite["exclude_objects"]), render_wh=(RENDER_WIDTH, RENDER_HEIGHT),
                menagerie_root=Path(config["menagerie_root"]))
            try:
                cell = _winner_render_cell(env, task, suite, pi05_tasks._load_objects(factory),
                    row, original, staging)
                _write_json(staging / "cell_records" / f"{row['cell_id']}.json", cell)
                cells.append(cell)
            finally:
                env.renderer.close()
        if len(cells) != 10:
            raise CameraScorerGateError("winner diagnostic denominator differs")
        after = validate_winner_chain(config, root=root)
        if _identity_without_path(Path(config_path)) != config_identity:
            raise CameraScorerGateError("winner config changed during diagnostics")
        if before["summary"] != after["summary"]:
            raise CameraScorerGateError("winner inputs changed during diagnostics")
        executed = [r for r in cells if r["executed"]]
        result = {"paper_ready": False, "headline_eligible": False,
            "study_scope": "one_target_engineering_smoke", "producer_commit": config["producer_commit"],
            "code": code, "upstream": before["summary"], "runtime": runtime,
            "planned_cells": 10, "executed_cells": len(executed),
            "build_failures": 10-len(executed), "diagnostic_passed_cells": sum(r["passed"] for r in executed),
            "paired_cpu_gate_pass": False, "large_rollout_launch_allowed": False,
            "real_policy_infra_smoke_allowed": bool(executed) and all(r["passed"] for r in executed),
            "failure_cells": [r["cell_id"] for r in executed if not r["passed"]],
            "scope": "camera and direct-state scorer predicates only; no policy action, IK, or success evidence"}
        _write_jsonl(staging / "cells.jsonl", cells)
        _write_jsonl(staging / "camera_metrics.jsonl", [c for r in executed for c in r["camera_metrics"]])
        for key in ("workspace_metrics", "disambiguation_metrics", "scorer_metrics"):
            _write_jsonl(staging / f"{key}.jsonl", [r[key] for r in executed])
        _write_json(staging / "gate.json", result)
        _write_json(staging / "manifest.json", {"code": code, "files": _artifact_inventory(staging),
            "config": config_identity, "producer_commit": config["producer_commit"]})
        _write_json(staging / "seal.json", {"manifest": _identity_without_path(staging / "manifest.json")})
        _fsync_tree(staging)
    return result


AUTOMATIC_SCOPE = "automatic_compact_camera_diagnostic"
FULL_AUTOMATIC_SCOPE = "automatic_full_cohort_camera_diagnostic"
FULL_PLANNED_CELLS = 2690


def validate_automatic_chain(config: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Authenticate the full automatic qualifier and fixed subset, without ranking."""
    if config.get("study_scope") == FULL_AUTOMATIC_SCOPE:
        return validate_full_automatic_chain(config, root=root)
    from run.icra2027 import e4_compact_harness as compact
    from robo.eval import e4_candidate_screen as screen
    if (config.get("schema_version") != 1 or config.get("study_scope") != AUTOMATIC_SCOPE
            or config.get("paper_ready") is not False or config.get("render_backend") != "osmesa"):
        raise CameraScorerGateError("automatic camera schema/scope/backend differs")
    _validated_id(config["freeze_id"], label="automatic camera freeze")
    producer = config["producer_commit"]
    _git_snapshot(producer)
    protocol = compact.checked_protocol(config["protocol"]["path"], config["protocol"]["sha256"])
    rows, sources = compact.qualifier_handoff(protocol,
        screen_id=config["qualifier_screen_id"], expected_commit=producer, root=root)
    menagerie = _menagerie_snapshot(Path(config["menagerie_root"]), EXPECTED_MENAGERIE_COMMIT)
    resize = _openpi_snapshot()
    if resize != config["openpi_resize_identity"]:
        raise CameraScorerGateError("automatic camera policy resize identity differs")
    scenes, task_identities = {}, {}
    for scene in compact.SCENES:
        prepared = screen._load_prepare(root=root, screen_id=config["qualifier_screen_id"],
            scene_id=scene, expected_commit=producer)
        task_bundle = prepared["task_bundle"]
        suites = {arm: json.loads(Path(task_bundle["variant_tasks"][arm]).read_text()) for arm in POLICIES}
        _automatic_paired_fields(suites)
        cpu_gate = json.loads(Path(sources[scene]["gate.json"]["path"]).read_text())
        if {k:v for k,v in menagerie.items() if k != "files"} != cpu_gate["menagerie"]:
            raise CameraScorerGateError("automatic CPU/render Menagerie closure differs")
        task_identities[scene] = task_bundle["bundle_manifest_sha256"]
        scenes[scene] = {"suites":suites, "factories":prepared["factories"], "task_bundle":task_bundle}
    return {"rows":rows, "scenes":scenes, "menagerie":menagerie,
        "summary":{"producer_commit":producer, "protocol":config["protocol"],
            "qualifier_sources":sources, "task_bundle_sha256":task_identities,
            "menagerie":{k:v for k,v in menagerie.items() if k != "files"},
            "openpi_resize_identity":resize,"planned":len(rows),"eligible":sum(r["passed"] for r in rows)}}


def validate_full_automatic_chain(config: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Authenticate all planned queries, including genuine planning-unavailable cells."""
    from run.icra2027 import e4_full_qualification as full
    from robo.eval import e4_candidate_screen as screen
    from robo.eval.episode_log import derive_reset_seed
    if (config.get("schema_version") != 1 or config.get("study_scope") != FULL_AUTOMATIC_SCOPE
            or config.get("paper_ready") is not False or config.get("render_backend") != "osmesa"
            or config.get("freeze_id") != config.get("qualifier_screen_id")):
        raise CameraScorerGateError("full automatic camera scope or qualification freeze differs")
    producer=config["producer_commit"];_git_snapshot(producer)
    freeze=_validated_id(config["freeze_id"],label="full automatic camera freeze")
    stage=screen._experiment_root(root,freeze)
    config_identity=full.shared.identity(config["qualification_config"]["path"])
    if config_identity != config["qualification_config"]:
        raise CameraScorerGateError("full qualification config identity differs")
    qualification,protocol=full.validate_stage(config_identity["path"],stage,producer)
    if (config["protocol"] != {k:qualification["source"]["protocol"][k] for k in ("path","sha256")}
            or config["menagerie_root"] != qualification["menagerie_root"]
            or config["openpi_resize_identity"] != qualification["openpi_resize_identity"]):
        raise CameraScorerGateError("full camera frozen protocol/model/resize differs")
    menagerie=_menagerie_snapshot(Path(config["menagerie_root"]),EXPECTED_MENAGERIE_COMMIT)
    resize=_openpi_snapshot()
    if ({k:v for k,v in menagerie.items() if k!="files"} != qualification["menagerie"]
            or resize != config["openpi_resize_identity"]):
        raise CameraScorerGateError("full camera runtime model/resize differs")
    rows=[];unavailable=[];sources={};scenes={};task_identities={}
    for source in protocol["source_populations"]:
        scene=source["scene_id"]
        tasks=sorted((q for q in protocol["qualification_tasks"] if q["scene_id"]==scene),key=lambda q:q["task_id"])
        ids=[q["task_id"] for q in tasks]
        directory=stage/"qualification_handoff"/scene
        bundle=screen._validate_bundle(directory,root=root,expected_kind=full.SCOPE)
        if (bundle["manifest"].get("code")!=screen._code_snapshot(producer)
                or bundle["manifest"].get("freeze_id")!=freeze or bundle["manifest"].get("scene_id")!=scene):
            raise CameraScorerGateError("full handoff manifest producer/freeze differs")
        result=screen._read_json_member(bundle["directory"],"result.json")
        expected=dict(schema_version=1,scope=full.SCOPE,scene_id=scene,source_commit=producer,
            freeze_id=freeze,config=config_identity,selected_task_ids=ids,planned_qualification_cells=10*len(ids),
            policy_executed=0,policy_success=None,camera=None,rollout_ledger=None,paper_ready=False)
        if any(result.get(k)!=v for k,v in expected.items()):
            raise CameraScorerGateError("full scene handoff identity or planned denominator differs")
        sources[scene]={"handoff":{name:full.shared.identity(directory/name)
            for name in ("manifest.json","seal.json","result.json")}}
        if not tasks:
            if (stage/"scene_prepares"/scene).exists() or (stage/"scene_qualifiers"/scene).exists():
                raise CameraScorerGateError("zero-query scene has unplanned qualification artifacts")
            if result.get("status")!="NOT_RUN" or result.get("reason")!="no_queries_in_original_protocol" or result.get("qualifier") is not None:
                raise CameraScorerGateError("zero-query source acquired qualification")
            continue
        common=dict(root=root,screen_id=freeze,scene_id=scene,expected_commit=producer)
        if result.get("status")=="PLANNING_UNQUALIFIED":
            if (result.get("reason")!=full.PLANNING_UNQUALIFIED or result.get("qualifier") is not None
                    or result.get("reset_definitions") is not None
                    or (stage/"scene_prepares"/scene).exists() or (stage/"scene_qualifiers"/scene).exists()):
                raise CameraScorerGateError("unqualified planning acquired executable tasks or measurements")
            try:screen._automatic_prepare_inputs(**common)
            except screen.CandidateScreenError as exc:
                if str(exc)!=full.PLANNING_UNQUALIFIED:raise
            else:raise CameraScorerGateError("planning-unqualified reason does not replay")
            for task in tasks:
                for arm in POLICIES:
                    for episode in range(EPISODES):
                        tid=task["task_id"]
                        unavailable.append(dict(cell_id=f"{arm.lower()}__{tid}__seed0__ep{episode}",
                            scene_id=scene,task_id=tid,policy_id=arm,episode=episode,
                            reset_seed=derive_reset_seed(0,tid,episode),target=task["target"],
                            status="NOT_RUN",reason="planning_unqualified",task_definition=None,
                            reset_provenance=None,camera_metrics=None,policy_success=None))
            continue
        if result.get("status")!="QUALIFICATION_COMPLETE":
            raise CameraScorerGateError("full qualification handoff is not terminal")
        checked=screen._validate_qualifier_output(**common)
        full.validate_selected_cells(checked,ids)
        qualifier=stage/"scene_qualifiers"/scene
        if (result.get("qualifier")!=full.shared.identity(qualifier/"manifest.json")
                or result.get("prerequisite_cells")!=checked["gate"]["cell_count"]
                or result.get("simulator_cells")!=checked["gate"]["exact_900_step_cells"]
                or result.get("strict_pass_task_ids")!=checked["gate"]["strict_pass_task_ids"]):
            raise CameraScorerGateError("full handoff differs from canonical qualifier")
        prepared=screen._load_prepare(**common)
        if "automatic_population" not in prepared["gate"]:
            raise CameraScorerGateError("full camera requires automatic source construction")
        task_bundle=prepared["task_bundle"]
        if sorted(task_bundle["logical_task_ids"])!=ids:
            raise CameraScorerGateError("full frozen task roster differs from selected protocol")
        suites={arm:json.loads(Path(task_bundle["variant_tasks"][arm]).read_text()) for arm in POLICIES}
        _automatic_paired_fields(suites)
        if checked["gate"]["menagerie"]!={k:v for k,v in menagerie.items() if k!="files"}:
            raise CameraScorerGateError("full CPU/camera Menagerie closure differs")
        sources[scene]["qualifier"]={name:full.shared.identity(qualifier/name)
            for name in ("gate.json","manifest.json","seal.json","metrics.jsonl")}
        task_identities[scene]=task_bundle["bundle_manifest_sha256"]
        scenes[scene]={"suites":suites,"factories":prepared["factories"],"task_bundle":task_bundle}
        rows.extend(checked["metric_rows"])
    def identity(row):
        return tuple(row[k] for k in ("scene_id","task_id","policy_id","episode","reset_seed","target","cell_id"))
    expected_cells={(q["scene_id"],q["task_id"],arm,ep,derive_reset_seed(0,q["task_id"],ep),q["target"],
                     f"{arm.lower()}__{q['task_id']}__seed0__ep{ep}")
        for q in protocol["qualification_tasks"] for arm in POLICIES for ep in range(EPISODES)}
    actual=[identity(row) for row in rows+unavailable]
    if len(expected_cells)!=FULL_PLANNED_CELLS or len(actual)!=len(expected_cells) or set(actual)!=expected_cells:
        raise CameraScorerGateError("full planned qualification/camera cell identities differ")
    return {"rows":rows,"planned_unavailable":unavailable,"scenes":scenes,"menagerie":menagerie,
        "summary":{"producer_commit":producer,"protocol":config["protocol"],"qualification_config":config_identity,
            "qualifier_sources":sources,"task_bundle_sha256":task_identities,
            "menagerie":{k:v for k,v in menagerie.items() if k!="files"},"openpi_resize_identity":resize,
            "planned_scenes":50,"input_semantic_queries":6155,"budget_exclusions":5886,
            "planned":FULL_PLANNED_CELLS,"prerequisite_cells":len(rows),"planning_unqualified_cells":len(unavailable),
            "eligible":sum(row["passed"] is True for row in rows)}}


def write_full_automatic_camera_config(*, qualification_config, expected_code_commit, out):
    """Derive camera inputs from the same E0-bound qualification stage, without edits."""
    from run.icra2027 import e4_full_qualification as full
    identity=full.shared.identity(qualification_config);qualification=full.shared.read(identity["path"])
    config={"schema_version":1,"study_scope":FULL_AUTOMATIC_SCOPE,"paper_ready":False,
        "freeze_id":qualification["freeze_id"],"qualifier_screen_id":qualification["freeze_id"],
        "producer_commit":expected_code_commit,"qualification_config":identity,
        "protocol":{k:qualification["source"]["protocol"][k] for k in ("path","sha256")},
        "menagerie_root":qualification["menagerie_root"],
        "openpi_resize_identity":qualification["openpi_resize_identity"],"render_backend":"osmesa"}
    validate_automatic_chain(config,root=_evidence_root())
    with Path(out).open("xb") as stream:stream.write(_json_bytes(config))
    return config


def _automatic_paired_fields(suites):
    """Task freezer owns complete validation; explicitly pin camera/reset inputs here."""
    if set(suites) != set(POLICIES):
        raise CameraScorerGateError("automatic camera requires A0 and A4")
    for key in ("robot", "table", "ext_cam", "exclude_objects", "tasks"):
        if key not in suites["A0"] or suites["A0"][key] != suites["A4"].get(key):
            raise CameraScorerGateError(f"automatic paired {key} differs")


def write_automatic_camera_config(*, protocol_path, protocol_sha256, qualifier_screen_id,
                                  freeze_id, menagerie_root, expected_code_commit, out):
    """Publish a fresh config only after real complete CPU qualification authenticates."""
    config = {"schema_version":1,"study_scope":AUTOMATIC_SCOPE,"paper_ready":False,
        "freeze_id":freeze_id,"qualifier_screen_id":qualifier_screen_id,
        "producer_commit":expected_code_commit,
        "protocol":{"path":str(Path(protocol_path).resolve(strict=True)),"sha256":protocol_sha256},
        "menagerie_root":str(Path(menagerie_root).resolve(strict=True)),
        "openpi_resize_identity":_openpi_snapshot(),"render_backend":"osmesa"}
    validate_automatic_chain(config,root=_evidence_root())
    with Path(out).open("xb") as stream:
        stream.write(_json_bytes(config))
    return config


def _automatic_render_cell(env, task, suite, task_rows, row, staging):
    """Replay the original CPU reset evidence, then run existing diagnostics."""
    from robo.eval import e4_candidate_screen as screen
    if row["passed"] is not True:
        raise CameraScorerGateError("rejected automatic CPU cell must not render")
    env._task_rows = task_rows
    obs = env.reset(settle_s=1.5,jitter_body=task["target"],jitter_xy=JITTER_XY_M,
        jitter_uniform_draw=np.asarray(row["reset_jitter"]["uniform_draw_0_1"]),reset_seed=row["reset_seed"])
    provenance = env.last_reset_provenance
    # Exact original comparisons, not relaxed numeric checks or a second camera search.
    replay = {"reset_jitter":provenance.get("jitter"),
        "settle_protocol":provenance.get("settle_protocol"),
        "settle_contract_checks":_settle_contract_checks(provenance),
        "stability":screen._runtime_stability(env=env,task=task,provenance=provenance),
        "workspace":screen._runtime_workspace(env=env,task=task,suite=suite,task_rows=task_rows)}
    if any(_canonical_hash(value) != _canonical_hash(row.get(key)) for key,value in replay.items()):
        raise CameraScorerGateError("automatic runtime reset differs from sealed CPU evidence")
    return _diagnose_reset(env,task,suite,task_rows,row,obs,staging,reset_check="exact_cpu_reset_replay")


def _automatic_camera_summary(cells, rows, *, planned_unavailable=None):
    expected = {r["cell_id"]:r for r in rows}
    observed = {r["cell_id"]:r for r in cells}
    count = 40 if planned_unavailable is None else FULL_PLANNED_CELLS-len(planned_unavailable)
    if len(rows) != count or len(expected) != count or len(cells) != count or set(observed) != set(expected):
        raise CameraScorerGateError("automatic camera planned-cell denominator differs")
    for cell_id,cell in observed.items():
        row = expected[cell_id]
        for key in ("scene_id","task_id","policy_id","episode","reset_seed","target"):
            if cell.get(key) != row[key]:
                raise CameraScorerGateError("automatic camera cell identity differs")
        if cell.get("reset_state_id") != f"{row['task_id']}__seed0__ep{row['episode']}":
            raise CameraScorerGateError("automatic camera reset ID differs")
        if row["passed"] is False:
            if (cell.get("executed") is not False or cell.get("passed") is not False
                    or cell.get("outcome") != "build_failure"
                    or cell.get("cpu_failed_checks") != sorted(k for k,v in row["checks"].items() if not v)
                    or any(cell.get(k) is not None for k in ("camera_metrics","workspace_metrics",
                        "disambiguation_metrics","scorer_metrics","reset_provenance"))):
                raise CameraScorerGateError("rejected CPU cell acquired diagnostic telemetry")
        elif cell.get("outcome") == "diagnostic_error":
            if (cell.get("passed") is not False or not cell.get("error")
                    or cell.get("executed") is not False or cell.get("checks") is not None
                    or any(cell.get(k) is not None for k in ("camera_metrics","workspace_metrics",
                        "disambiguation_metrics","scorer_metrics","reset_provenance"))):
                raise CameraScorerGateError("automatic diagnostic exception lost its failure")
        else:
            checks = cell.get("checks",{})
            if (set(checks) != {"camera","workspace","disambiguation","scorer","exact_cpu_reset_replay"}
                    or cell.get("executed") is not True or not checks["exact_cpu_reset_replay"]
                    or cell.get("passed") != all(checks.values())):
                raise CameraScorerGateError("automatic diagnostic check aggregate differs")
            cameras = cell.get("camera_metrics")
            components = [cell.get(k) for k in ("workspace_metrics","disambiguation_metrics","scorer_metrics")]
            provenance = cell.get("reset_provenance")
            if (not isinstance(cameras,list) or len(cameras)!=2
                    or any(not isinstance(m,dict) for m in cameras)
                    or {m.get("camera_role") for m in cameras}!={"exterior","wrist"}
                    or any(not isinstance(m,dict) or type(m.get("passed")) is not bool
                        or not m.get("checks") or m["passed"] != all(m["checks"].values())
                        for m in cameras+[components[0],components[2]])
                    or not isinstance(components[1],dict) or type(components[1].get("passed")) is not bool
                    or _canonical_hash({k:v for k,v in components[1].items() if k!="reset_state_id"})
                        != _canonical_hash(row.get("qualifier"))
                    or not isinstance(provenance,dict)
                    or provenance.get("jitter") != row["reset_jitter"]
                    or provenance.get("settle_protocol") != row["settle_protocol"]
                    or checks["camera"] != all(m["passed"] for m in cameras)
                    or any(checks[key]!=m["passed"] for key,m in zip(
                        ("workspace","disambiguation","scorer"),components))):
                raise CameraScorerGateError("automatic diagnostic telemetry/aggregate differs")
    eligible = [r for r in cells if expected[r["cell_id"]]["passed"]]
    extra = {} if planned_unavailable is None else {
        "planning_unqualified_cells":len(planned_unavailable), "prerequisite_cells":len(rows)}
    return {**extra,"planned_cells":40 if planned_unavailable is None else FULL_PLANNED_CELLS,"cpu_eligible_cells":len(eligible),
        "executed_cells":sum(r.get("executed") is True for r in cells),
        "build_failures":len(rows)-len(eligible),"diagnostic_passed_cells":sum(r["passed"] for r in eligible),
        "diagnostic_error_cells":sum(r.get("outcome")=="diagnostic_error" for r in eligible),
        "camera_scorer_pass":bool(eligible) and all(r["passed"] for r in eligible),
        "failure_cells":[r["cell_id"] for r in cells if not r["passed"]],
        "paper_ready":False,"headline_eligible":False,"real_policy_infra_smoke_allowed":False,
        "policy_launch_allowed":False,"large_rollout_launch_allowed":False}


def run_automatic_gate(*, config_path, expected_code_commit, preflight_only=False):
    """One sealed, explicitly scoped diagnostic; no policy actions or winner selection."""
    config_path = Path(config_path)
    identity = _identity_without_path(config_path)
    config = json.loads(config_path.read_text())
    code = _git_snapshot(expected_code_commit)
    root = _evidence_root()
    before = validate_automatic_chain(config,root=root)
    if config.get("study_scope")==FULL_AUTOMATIC_SCOPE:
        from run.icra2027.e4_full_qualification import cpu_guard
        cpu_guard()
    if preflight_only:
        poses = {}
        for scene,data in before["scenes"].items():
            poses[scene] = winner_camera_pose_preflight({**data,"menagerie":before["menagerie"],
                "rows":[r for r in before["rows"] if r["scene_id"]==scene]})
        return {"paper_ready":False,"upstream":before["summary"],"camera_pose_preflight":poses,
            "camera_job_submission_allowed":any(p["reports"] for p in poses.values())
                and all(r["passed"] for p in poses.values() for r in p["reports"])}
    if os.environ.get("MUJOCO_GL") != "osmesa" or os.environ.get("PYOPENGL_PLATFORM") != "osmesa":
        raise CameraScorerGateError("automatic diagnostic requires explicitly frozen OSMesa CPU backend")
    import mujoco
    from robo.envs.pi05_env import DroidSimEnv
    from robo.tasks import pi05_tasks
    output = root/"outputs/icra2027"/config["freeze_id"]/"harness/automatic_camera_scorer"
    with _atomic_directory(output,preserve_failed=True) as staging:
        cells=[]
        for row in before["rows"]:
            cell = {key:row[key] for key in ("cell_id","scene_id","task_id","policy_id","episode","reset_seed","target")}
            cell.update(reset_state_id=f"{row['task_id']}__seed0__ep{row['episode']}",
                executed=False,passed=False,outcome="build_failure",camera_metrics=None,
                workspace_metrics=None,disambiguation_metrics=None,scorer_metrics=None,reset_provenance=None)
            if row["passed"]:
                env=None
                try:
                    data=before["scenes"][row["scene_id"]];suite=data["suites"][row["policy_id"]]
                    task=next(t for t in suite["tasks"] if t["task_id"]==row["task_id"])
                    env=DroidSimEnv(suite["scene_xml"],suite["robot"]["base_pos"],suite["robot"]["base_yaw"],
                        table_box=suite["table"],ext_cam=suite["ext_cam"],exclude_objects=tuple(suite["exclude_objects"]),
                        render_wh=(RENDER_WIDTH,RENDER_HEIGHT),menagerie_root=Path(config["menagerie_root"]))
                    cell=_automatic_render_cell(env,task,suite,pi05_tasks._load_objects(data["factories"][row["policy_id"]]),row,staging)
                except Exception as exc:
                    cell.update(outcome="diagnostic_error",error=f"{type(exc).__name__}: {exc}")
                finally:
                    if env is not None:
                        env.renderer.close()
            else:
                cell["cpu_failed_checks"]=sorted(k for k,v in row["checks"].items() if not v)
            cells.append(cell)
            _write_json(staging/"cell_records"/f"{row['cell_id']}.json",cell)
        summary=_automatic_camera_summary(cells,before["rows"],planned_unavailable=before.get("planned_unavailable"))
        after=validate_automatic_chain(config,root=root)
        if before["summary"] != after["summary"] or identity != _identity_without_path(config_path):
            raise CameraScorerGateError("automatic camera inputs changed during diagnostics")
        result={**summary,"schema_version":1,"study_scope":config.get("study_scope",AUTOMATIC_SCOPE),"code":code,
            "upstream":before["summary"],"runtime":{"backend":"osmesa_cpu","mujoco":mujoco.__version__},
            "scope":"Camera and direct-state scorer only; no policy or manipulation success."}
        _write_jsonl(staging/"cells.jsonl",cells)
        if "planned_unavailable" in before:
            _write_jsonl(staging/"planned_unavailable_cells.jsonl",before["planned_unavailable"])
        _write_json(staging/"gate.json",result)
        _write_json(staging/"manifest.json",{"code":code,"files":_artifact_inventory(staging),
            "config":identity,"upstream":before["summary"]})
        _write_json(staging/"seal.json",{"manifest":_identity_without_path(staging/"manifest.json")})
        _fsync_tree(staging)
    return result


def validate_automatic_camera_output(*, config_path, output, expected_code_commit):
    """Authenticate complete sealed output and its current canonical upstream chain."""
    output=Path(output);config_path=Path(config_path)
    config=json.loads(config_path.read_text());code=_git_snapshot(expected_code_commit)
    if config.get("study_scope")==FULL_AUTOMATIC_SCOPE:
        expected=_evidence_root()/"outputs/icra2027"/config["freeze_id"]/"harness/automatic_camera_scorer"
        if _regular_directory(output,root=_evidence_root(),label="full camera output")!=expected:
            raise CameraScorerGateError("full camera output destination differs")
    upstream=validate_automatic_chain(config,root=_evidence_root())
    manifest=json.loads((output/"manifest.json").read_text())
    seal=json.loads((output/"seal.json").read_text())
    if ((output/"seal.json").read_bytes() != _json_bytes(seal)
            or seal.get("manifest") != _identity_without_path(output/"manifest.json")
            or manifest.get("code") != code or manifest.get("config") != _identity_without_path(config_path)
            or manifest.get("upstream") != upstream["summary"] or manifest.get("files") != _artifact_inventory(output)):
        raise CameraScorerGateError("automatic camera output seal/config/source/artifact drift")
    cells=[json.loads(line) for line in (output/"cells.jsonl").read_text().splitlines()]
    summary=_automatic_camera_summary(cells,upstream["rows"],planned_unavailable=upstream.get("planned_unavailable"))
    if "planned_unavailable" in upstream:
        unavailable=[json.loads(line) for line in (output/"planned_unavailable_cells.jsonl").read_text().splitlines()]
        if unavailable != upstream["planned_unavailable"]:
            raise CameraScorerGateError("full camera planned-unavailable population differs")
    gate=json.loads((output/"gate.json").read_text())
    if (gate.get("study_scope") != config.get("study_scope",AUTOMATIC_SCOPE) or gate.get("code") != code
            or gate.get("upstream") != upstream["summary"]
            or any(gate.get(k) != v for k,v in summary.items())):
        raise CameraScorerGateError("automatic camera gate differs from authenticated cells")
    return {"gate":gate,"cells":cells,"manifest_sha256":_sha256(output/"manifest.json")}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qualifier-preflight-only",
        action="store_true",
        help="run the independent CPU-only 40-cell qualifier scheduling gate",
    )
    parser.add_argument("--gate-id", required=True)
    parser.add_argument("--cpu-freeze-id", required=True)
    parser.add_argument("--cpu-producer-commit", required=True)
    parser.add_argument("--expected-cpu-gate-sha256", required=True)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--menagerie-root", required=True)
    parser.add_argument("--expected-menagerie-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    actual_args = list(sys.argv[1:] if argv is None else argv)
    if "--write-full-automatic-config" in actual_args:
        parser=argparse.ArgumentParser()
        parser.add_argument("--write-full-automatic-config",action="store_true")
        parser.add_argument("--qualification-config",required=True)
        parser.add_argument("--expected-code-commit",required=True)
        parser.add_argument("--out",required=True)
        args=parser.parse_args(actual_args)
        result=write_full_automatic_camera_config(qualification_config=args.qualification_config,
            expected_code_commit=args.expected_code_commit,out=args.out)
        print(json.dumps(result,indent=2));return 0
    if "--automatic-config" in actual_args:
        parser = argparse.ArgumentParser()
        parser.add_argument("--automatic-config", required=True)
        parser.add_argument("--expected-code-commit", required=True)
        parser.add_argument("--automatic-preflight-only", action="store_true")
        args = parser.parse_args(actual_args)
        result = run_automatic_gate(config_path=args.automatic_config,
            expected_code_commit=args.expected_code_commit,preflight_only=args.automatic_preflight_only)
        print(json.dumps(result,indent=2))
        return 0 if result.get("camera_job_submission_allowed",result.get("camera_scorer_pass")) else 3
    if "--write-winner-config" in actual_args:
        parser = argparse.ArgumentParser()
        parser.add_argument("--write-winner-config", action="store_true")
        parser.add_argument("--winner-cpu-bundle", required=True)
        parser.add_argument("--winner-harness-config", required=True)
        parser.add_argument("--winner-freeze-id", required=True)
        parser.add_argument("--out", required=True)
        parser.add_argument("--expected-code-commit", required=True)
        args = parser.parse_args(actual_args)
        config = write_winner_camera_config(cpu_bundle=args.winner_cpu_bundle,
            harness_config=args.winner_harness_config, freeze_id=args.winner_freeze_id,
            out=args.out, expected_code_commit=args.expected_code_commit)
        print(json.dumps({"freeze_id":config["freeze_id"], "config_path":args.out}))
        return 0
    if "--winner-config" in actual_args:
        parser = argparse.ArgumentParser()
        parser.add_argument("--winner-config", required=True)
        parser.add_argument("--expected-code-commit", required=True)
        parser.add_argument("--winner-preflight-only", action="store_true")
        args = parser.parse_args(actual_args)
        result = run_winner_gate(config_path=args.winner_config,
            expected_code_commit=args.expected_code_commit,
            preflight_only=args.winner_preflight_only)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("camera_job_submission_allowed", result.get("real_policy_infra_smoke_allowed")) else 3
    args = _parser().parse_args(actual_args)
    try:
        runner = run_qualifier_preflight if args.qualifier_preflight_only else run_gate
        gate = runner(
            gate_id=args.gate_id,
            cpu_freeze_id=args.cpu_freeze_id,
            cpu_producer_commit=args.cpu_producer_commit,
            expected_cpu_gate_sha256=args.expected_cpu_gate_sha256,
            expected_code_commit=args.expected_code_commit,
            menagerie_root=args.menagerie_root,
            expected_menagerie_commit=args.expected_menagerie_commit,
        )
    except (
        CameraScorerGateError,
        FileNotFoundError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"[e4-camera-scorer-gate] FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(gate, indent=2, sort_keys=True))
    if args.qualifier_preflight_only:
        if not gate["camera_job_submission_allowed"]:
            print(
                "[e4-camera-scorer-gate] CPU qualifier preflight failed; "
                "diagnostic output was sealed and no camera job is authorized",
                file=sys.stderr,
            )
            return 4
        return 0
    if not gate["real_policy_infra_smoke_allowed"]:
        print(
            "[e4-camera-scorer-gate] semantic gate failed; diagnostic output was sealed",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
