"""Materialize one E3 policy/scene selection as an immutable factory tree.

The E3 aggregate is the root of construction-selection provenance.  This
module verifies its sealed inventories and selected records, then copies only
the selected construction artifacts into a fresh repository-local directory.
It deliberately does not discover an E3 run implicitly and has no overwrite
mode.

Example::

    python -m robo.eval.e3_factory_materializer \
      --e3-root outputs/icra2027/<freeze>/agentic \
      --scene-id c50d2d1d42 --policy-id A4 \
      --out outputs/icra2027/<e4>/construction_variants/A4/c50d2d1d42_factory
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from agents.orchestrator.artifact import canonical_json, sha256_file
from robo.eval import agentic_ablation as e3


SCHEMA_VERSION = 1
ALLOWED_POLICIES = frozenset({"A0", "A4"})
SCENE_ID_RE = re.compile(r"^[0-9a-f]{10}$")
OBJECT_SLOT_RE = re.compile(r"^obj_([0-9]+)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
CODE_ROOT = Path(__file__).resolve().parents[2]


def _validated_evidence_root(value: str | None) -> Path:
    """Resolve the artifact checkout without weakening repository boundaries.

    E4 validation code runs from a clean, commit-pinned worktree while the
    sealed E3 inputs and newly published E4 products live in the long-lived
    SimAny artifact checkout.  The split is launcher-owned and explicit: an
    override must be absolute, existing, and contain no symlink component.
    """
    if value is None:
        return CODE_ROOT
    raw = Path(value)
    if not raw.is_absolute():
        raise RuntimeError("SIMANY_EVIDENCE_ROOT must be an absolute path")
    lexical = Path(os.path.abspath(raw))
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(
                f"SIMANY_EVIDENCE_ROOT contains a symlink component: {current}"
            )
    if not lexical.is_dir():
        raise RuntimeError(
            f"SIMANY_EVIDENCE_ROOT is not an existing directory: {lexical}"
        )
    return lexical


REPOSITORY_ROOT = _validated_evidence_root(os.environ.get("SIMANY_EVIDENCE_ROOT"))
# ``checked_repo_path`` and the atomic publisher are shared with E3.  Bind
# those helpers to the same evidence root selected above; Git provenance is
# still read exclusively from CODE_ROOT.
e3.REPOSITORY_ROOT = REPOSITORY_ROOT

AGGREGATE_MEMBERS = frozenset(
    {
        "agentic_ablation.csv",
        "agentic_ablation.json",
        "coverage_fidelity_curve.csv",
        "failure_transitions.csv",
        "job_ledger.jsonl",
        "retry_breakdown.csv",
    }
)
INVENTORY_MEMBERS = frozenset(
    {
        "artifact_hashes.sha256",
        "inventory_audit.json",
        "resolved_jobs.json",
        "resolved_proposals.json",
    }
)
SELECTED_RECORD_KEYS = frozenset(
    {
        "freeze_id",
        "job_id",
        "object_slot",
        "policy_id",
        "proposal_id",
        "reason_codes",
        "retry_attempted",
        "retry_invoked",
        "retry_produced",
        "retry_reason_codes",
        "retry_required",
        "scene_id",
        "schema_version",
        "selected_asset",
        "selected_proposal_id",
        "support_label",
        "terminal_action",
    }
)
SELECTED_ASSET_KEYS = frozenset(
    {
        "artifact_hashes",
        "artifact_paths",
        "artifact_sizes",
        "proposal_digest",
        "registration_surface_hash",
        "tool",
    }
)
REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "collision",
        "evidence",
        "mesh_sim",
        "physics",
        "probe",
        "raw_gaussian",
        "raw_mesh",
        "registration",
        "transform",
        "urdf",
    }
)
OPTIONAL_ARTIFACT_ROLES = frozenset({"retry"})
CAPTURED_TEXT_ROLES = frozenset(
    {"evidence", "physics", "probe", "registration", "retry", "urdf"}
)
ROLE_OUTPUTS = {
    "collision": "collision/part_00.obj",
    "evidence": "evidence.json",
    "mesh_sim": "mesh_sim.obj",
    "physics": "physics.json",
    "probe": "probe.json",
    "raw_gaussian": "trellis_gs.ply",
    "raw_mesh": "trellis_mesh.ply",
    "registration": "registration.json",
    "retry": "retry.json",
    "transform": "transform.npy",
    "urdf": "object.urdf",
}
RUNTIME_ROLE_SUFFIXES = {
    "collision": "physical/collision/part_00.obj",
    "evidence": "evidence.json",
    "mesh_sim": "physical/mesh_sim.obj",
    "physics": "physical/physics.json",
    "probe": "physical/probe.json",
    "registration": "registration.json",
    "retry": "retry.json",
    "transform": "transform.npy",
    "urdf": "physical/object.urdf",
}
MATERIALIZATION_MANIFEST_KEYS = frozenset(
    {
        "created_utc",
        "destination",
        "e3_claim_status",
        "e3_code_commit",
        "e3_freeze_id",
        "e3_root",
        "input_identities",
        "manifest_kind",
        "output_members",
        "policy_id",
        "provenance",
        "roster",
        "scene_id",
        "schema_version",
        "selected_records",
        "source_scene",
        "study_scope",
    }
)
MATERIALIZATION_PROVENANCE_KEYS = frozenset(
    {
        "code_root",
        "e3_producer_commit",
        "evidence_root",
        "materializer_commit",
        "materializer_dirty",
        "validator_commit",
        "validator_dirty",
    }
)
MATERIALIZATION_ROSTER_KEYS = frozenset(
    {
        "abstained_count",
        "abstained_slots",
        "accepted_count",
        "accepted_slots",
        "job_count",
        "object_slots",
        "sha256",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_snapshot() -> dict[str, Any]:
    """Return the exact Git identity of the executing materializer code."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=CODE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        top_level = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=CODE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=CODE_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("cannot obtain the materializer Git snapshot") from exc
    observed_root = Path(top_level).resolve(strict=True)
    if observed_root != CODE_ROOT:
        raise ValueError(
            "executing materializer is not rooted at its declared code root: "
            f"expected={CODE_ROOT}, observed={observed_root}"
        )
    if COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("materializer Git commit is not a full lowercase SHA")
    return {
        "code_root": str(observed_root),
        "commit": commit,
        "dirty": bool(status),
        "status": status,
    }


def _require_clean_code_snapshot() -> dict[str, Any]:
    snapshot = _git_snapshot()
    if snapshot.get("code_root") != str(CODE_ROOT):
        raise ValueError("materializer Git snapshot code root differs")
    commit = snapshot.get("commit")
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("materializer Git snapshot lacks a full lowercase commit")
    if snapshot.get("dirty") is not False or snapshot.get("status") != []:
        raise ValueError(
            "materializer requires a clean code worktree; Git status is "
            f"{snapshot.get('status')!r}"
        )
    return dict(snapshot)


def _provenance_record(
    *, e3_producer_commit: str, code_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    producer = _plain_string(e3_producer_commit, "E3 producer commit")
    if COMMIT_RE.fullmatch(producer) is None:
        raise ValueError("E3 producer commit must be a full lowercase Git SHA")
    commit = _plain_string(code_snapshot.get("commit"), "materializer commit")
    if COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("materializer commit must be a full lowercase Git SHA")
    if code_snapshot.get("dirty") is not False:
        raise ValueError("materializer Git snapshot is dirty")
    return {
        "code_root": str(CODE_ROOT),
        "e3_producer_commit": producer,
        "evidence_root": str(REPOSITORY_ROOT),
        "materializer_commit": commit,
        "materializer_dirty": False,
        # Materialization and standalone revalidation ship from this same
        # module, so creation records their shared code identity explicitly.
        "validator_commit": commit,
        "validator_dirty": False,
    }


def _validate_provenance_record(
    value: Any,
    *,
    e3_producer_commit: str,
    evidence_root: Path,
    validator_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = _exact_mapping(
        value, MATERIALIZATION_PROVENANCE_KEYS, "materialization provenance"
    )
    producer = _digest_commit(
        provenance["e3_producer_commit"], "materialization E3 producer commit"
    )
    expected_producer = _digest_commit(
        e3_producer_commit, "sealed E3 producer commit"
    )
    materializer_commit = _digest_commit(
        provenance["materializer_commit"], "materialization materializer commit"
    )
    recorded_validator = _digest_commit(
        provenance["validator_commit"], "materialization validator commit"
    )
    executing_validator = _digest_commit(
        validator_snapshot.get("commit"), "executing validator commit"
    )
    if producer != expected_producer:
        raise ValueError("materialization E3 producer provenance differs")
    if materializer_commit != recorded_validator:
        raise ValueError("materialization materializer/validator commits differ")
    if recorded_validator != executing_validator:
        raise ValueError("materialization validator commit differs from executing code")
    if provenance["materializer_dirty"] is not False:
        raise ValueError("materialization provenance records a dirty materializer")
    if provenance["validator_dirty"] is not False:
        raise ValueError("materialization provenance records a dirty validator")
    if provenance["code_root"] != str(CODE_ROOT):
        raise ValueError("materialization code root differs from executing code root")
    if provenance["evidence_root"] != str(evidence_root):
        raise ValueError("materialization evidence root differs from validation root")
    return dict(provenance)


def _exact_mapping(value: Any, keys: set[str] | frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        observed: Any = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(
            f"{label} schema keys differ: expected={sorted(keys)}, observed={observed}"
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _digest_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 40-character Git SHA")
    return value


def _plain_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _repo_relative_path(value: Any, label: str, *, kind: str = "file") -> Path:
    raw = _plain_string(value, f"{label}.path")
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or raw != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in raw
    ):
        raise ValueError(f"{label}.path must be a canonical repository-relative path")
    return e3.checked_repo_path(raw, label, kind=kind)


def _display(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _parse_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc


def _identity(path: Path) -> dict[str, Any]:
    path = e3.checked_repo_path(path, "identity artifact", kind="file")
    return {
        "path": _display(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _read_verified_identity(
    value: Any, label: str, *, parse_json: bool = False
) -> tuple[Path, bytes, Any | None, dict[str, Any]]:
    identity = _exact_mapping(value, {"path", "sha256", "size_bytes"}, label)
    path = _repo_relative_path(identity["path"], label)
    size = _positive_int(identity["size_bytes"], f"{label}.size_bytes")
    expected = _digest(identity["sha256"], f"{label}.sha256")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if len(payload) != size:
        raise ValueError(
            f"{label} size drift: expected {size}, found {len(payload)} at {path}"
        )
    if actual != expected:
        raise ValueError(
            f"{label} hash drift: expected {expected}, found {actual} at {path}"
        )
    parsed = _parse_json_bytes(payload, label) if parse_json else None
    return path, payload, parsed, {
        "path": _display(path),
        "size_bytes": size,
        "sha256": expected,
    }


def _read_json_path(path: Path, label: str) -> tuple[bytes, Any]:
    path = e3.checked_repo_path(path, label, kind="file")
    payload = path.read_bytes()
    return payload, _parse_json_bytes(payload, label)


def _verify_digest_member(path: Path, expected: Any, label: str) -> None:
    path = e3.checked_repo_path(path, label, kind="file")
    expected_digest = _digest(expected, f"{label} digest")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise ValueError(
            f"{label} hash drift: expected {expected_digest}, found {actual}"
        )


def _validate_external_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _exact_mapping(value, {"path", "sha256", "size_bytes"}, label)
    raw = _plain_string(identity["path"], f"{label}.path")
    if not Path(raw).is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    # External bytes are not inputs to this materialization.  Their immutable
    # identities must agree across the two independently sealed inventories,
    # but this repository-local operation deliberately does not open them.
    return {
        "path": raw,
        "size_bytes": _positive_int(identity["size_bytes"], f"{label}.size_bytes"),
        "sha256": _digest(identity["sha256"], f"{label}.sha256"),
    }


def _verify_aggregate(e3_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    seal_path = e3.checked_repo_path(
        e3_root / "aggregate_seal.json", "E3 aggregate seal", kind="file"
    )
    _, seal = _read_json_path(seal_path, "E3 aggregate seal")
    seal = _exact_mapping(
        seal,
        {"schema_version", "members", "selected_asset_directories", "selected_asset_members"},
        "E3 aggregate seal",
    )
    if seal["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported E3 aggregate seal schema version")
    members = _exact_mapping(seal["members"], AGGREGATE_MEMBERS, "aggregate members")
    for name, expected in members.items():
        if Path(name).name != name:
            raise ValueError(f"aggregate member is not a plain filename: {name!r}")
        _verify_digest_member(e3_root / name, expected, f"aggregate member {name}")

    selected_root = e3.checked_repo_path(
        e3_root / "selected_assets", "aggregate selected-assets tree", kind="dir"
    )
    declared_members = seal["selected_asset_members"]
    if not isinstance(declared_members, Mapping) or not declared_members:
        raise ValueError("aggregate selected_asset_members must be a non-empty object")
    declared_directories = seal["selected_asset_directories"]
    if (
        not isinstance(declared_directories, list)
        or any(not isinstance(item, str) for item in declared_directories)
        or len(declared_directories) != len(set(declared_directories))
        or declared_directories != sorted(declared_directories)
    ):
        raise ValueError("aggregate selected_asset_directories must be unique and sorted")

    actual_members: list[str] = []
    actual_directories: list[str] = []
    for current, directories, files in os.walk(selected_root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            child = current_path / name
            e3.checked_repo_path(child, "selected-assets directory", kind="dir")
            actual_directories.append(child.relative_to(selected_root).as_posix())
        for name in sorted(files):
            child = current_path / name
            e3.checked_repo_path(child, "selected-assets member", kind="file")
            actual_members.append(child.relative_to(selected_root).as_posix())
    actual_members.sort()
    actual_directories.sort()
    if actual_members != sorted(declared_members):
        raise ValueError("aggregate selected-assets files differ from its exact sealed inventory")
    if actual_directories != declared_directories:
        raise ValueError(
            "aggregate selected-assets directories differ from its exact sealed inventory"
        )

    selected_identities: dict[str, dict[str, Any]] = {}
    for relative in actual_members:
        parts = PurePosixPath(relative).parts
        if (
            len(parts) != 3
            or parts[0] not in {"A0", "A1", "A2", "A3", "A4"}
            or not SCENE_ID_RE.fullmatch(parts[1])
            or not parts[2].endswith(".json")
            or not OBJECT_SLOT_RE.fullmatch(parts[2][:-5])
        ):
            raise ValueError(f"invalid selected-assets member path: {relative!r}")
        declared = _exact_mapping(
            declared_members[relative], {"sha256", "size_bytes"}, f"seal entry {relative}"
        )
        size = _positive_int(declared["size_bytes"], f"seal entry {relative}.size_bytes")
        expected = _digest(declared["sha256"], f"seal entry {relative}.sha256")
        path = e3.checked_repo_path(selected_root / relative, relative, kind="file")
        if path.stat().st_size != size:
            raise ValueError(f"selected record size drift: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"selected record hash drift: {relative}")
        selected_identities[relative] = {
            "path": _display(path),
            "size_bytes": size,
            "sha256": expected,
        }
    _, aggregate_result = _read_json_path(
        e3_root / "agentic_ablation.json", "E3 aggregate result"
    )
    if not isinstance(aggregate_result, Mapping):
        raise ValueError("E3 aggregate result must be a JSON object")
    claim_status = _exact_mapping(
        {
            key: aggregate_result.get(key)
            for key in (
                "freeze_id",
                "headline_eligible",
                "paper_ready",
                "retry_claim_status",
                "study_scope",
            )
        },
        {
            "freeze_id",
            "headline_eligible",
            "paper_ready",
            "retry_claim_status",
            "study_scope",
        },
        "E3 aggregate claim status",
    )
    if not isinstance(claim_status["headline_eligible"], bool) or not isinstance(
        claim_status["paper_ready"], bool
    ):
        raise ValueError("E3 aggregate claim flags must be booleans")
    _plain_string(claim_status["freeze_id"], "E3 aggregate freeze_id")
    _plain_string(claim_status["retry_claim_status"], "E3 retry_claim_status")
    _plain_string(claim_status["study_scope"], "E3 aggregate study_scope")
    return dict(seal), {
        "aggregate_seal": _identity(seal_path),
        "claim_status": dict(claim_status),
        "selected_records": selected_identities,
    }


def _verify_inventory(
    e3_root: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    inventory_root = e3.checked_repo_path(
        e3_root / "input_inventory", "E3 input inventory", kind="dir"
    )
    actual = sorted(path.name for path in inventory_root.iterdir())
    expected_actual = sorted(INVENTORY_MEMBERS | {"seal.json"})
    if actual != expected_actual:
        raise ValueError(
            f"input inventory files differ: expected={expected_actual}, observed={actual}"
        )
    seal_path = e3.checked_repo_path(
        inventory_root / "seal.json", "input-inventory seal", kind="file"
    )
    _, seal = _read_json_path(seal_path, "input-inventory seal")
    seal = _exact_mapping(
        seal, {"schema_version", "members", "controller_members"}, "input-inventory seal"
    )
    if seal["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported input-inventory seal schema version")
    members = _exact_mapping(seal["members"], INVENTORY_MEMBERS, "inventory members")
    controller = _exact_mapping(
        seal["controller_members"],
        {"resolved_jobs.json", "resolved_proposals.json"},
        "controller inventory members",
    )
    for name, expected in members.items():
        _verify_digest_member(inventory_root / name, expected, f"inventory member {name}")
    if any(controller[name] != members[name] for name in controller):
        raise ValueError("controller member hashes differ from full inventory seal")

    _, jobs = _read_json_path(inventory_root / "resolved_jobs.json", "resolved jobs")
    _, proposals = _read_json_path(
        inventory_root / "resolved_proposals.json", "resolved proposals"
    )
    _, audit = _read_json_path(inventory_root / "inventory_audit.json", "inventory audit")
    if not isinstance(jobs, Mapping) or not isinstance(proposals, Mapping):
        raise ValueError("resolved E3 inventories must be JSON objects")
    e3._validate_controller_inventory_schema(jobs, proposals)
    if jobs.get("schema_version") not in (1, 2) or proposals.get("schema_version") != jobs.get("schema_version"):
        raise ValueError("unsupported resolved-inventory schema version")
    if jobs.get("freeze_id") != proposals.get("freeze_id"):
        raise ValueError("resolved job/proposal freeze IDs differ")
    if not isinstance(audit, Mapping):
        raise ValueError("inventory audit must be a JSON object")
    return jobs, proposals, audit, {
        "inventory_seal": _identity(seal_path),
        "resolved_jobs": _identity(inventory_root / "resolved_jobs.json"),
        "resolved_proposals": _identity(inventory_root / "resolved_proposals.json"),
        "inventory_audit": _identity(inventory_root / "inventory_audit.json"),
    }


def _indexed_rows(rows: Any, label: str) -> dict[int, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    result: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} entries must be objects")
        index = row.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{label} has invalid object index: {index!r}")
        if index in result:
            raise ValueError(f"{label} has duplicate object index: {index}")
        result[index] = row
    return result


def _validate_aabb(value: Any, label: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a 2x3 list")
    parsed: list[list[float]] = []
    for corner in value:
        if not isinstance(corner, list) or len(corner) != 3:
            raise ValueError(f"{label} must be a 2x3 list")
        values = []
        for item in corner:
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError(f"{label} coordinates must be finite numbers")
            values.append(float(item))
        parsed.append(values)
    if any(lo > hi for lo, hi in zip(parsed[0], parsed[1], strict=True)):
        raise ValueError(f"{label} lower corner exceeds upper corner")


def _identity_map(artifacts: Any, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(artifacts, list):
        raise ValueError(f"{label} must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        artifact = _exact_mapping(
            artifact, {"path", "sha256", "size_bytes"}, f"{label}[{index}]"
        )
        path = _plain_string(artifact["path"], f"{label}[{index}].path")
        if path in result:
            raise ValueError(f"{label} contains duplicate path: {path}")
        # Validate the declaration now; bytes are opened only for consumed
        # objects.json/meta.json members below.
        _repo_relative_path(path, f"{label}[{index}]")
        _digest(artifact["sha256"], f"{label}[{index}].sha256")
        _positive_int(artifact["size_bytes"], f"{label}[{index}].size_bytes")
        result[path] = artifact
    return result


def _scene_source_context(
    jobs: Mapping[str, Any], audit: Mapping[str, Any], scene_id: str
) -> dict[str, Any]:
    audit = _exact_mapping(
        audit,
        {
            "counts",
            "created_utc",
            "evaluation_references",
            "freeze_id",
            "legacy_hybrid_winner_used",
            "membership_authority",
            "resolved_roster_sha256",
            "scenes",
            "schema_version",
            "source_identity_manifest_sha256",
        },
        "inventory audit",
    )
    if audit["schema_version"] != SCHEMA_VERSION or audit["freeze_id"] != jobs["freeze_id"]:
        raise ValueError("inventory-audit schema/freeze differs from resolved jobs")
    if audit["legacy_hybrid_winner_used"] is not False:
        raise ValueError("legacy hybrid winner must not define E3 membership")
    if not isinstance(audit["scenes"], list):
        raise ValueError("inventory audit scenes must be a list")
    audit_matches = [row for row in audit["scenes"] if isinstance(row, Mapping) and row.get("scene_id") == scene_id]
    job_matches = [row for row in jobs["scenes"] if isinstance(row, Mapping) and row.get("scene_id") == scene_id]
    if len(audit_matches) != 1 or len(job_matches) != 1:
        raise ValueError(
            f"scene must occur exactly once in jobs and audit: {scene_id}"
        )
    audit_scene = _exact_mapping(
        audit_matches[0],
        {
            "accepted_jobs",
            "camera_artifacts",
            "job_ids",
            "prehybrid_identity",
            "reconviagen_available",
            "report_identity",
            "scene_id",
            "source_identity_manifest",
            "source_scene_gaussian",
        },
        "inventory-audit scene",
    )
    job_scene = job_matches[0]
    scene_jobs = job_scene["jobs"]
    if not isinstance(scene_jobs, list) or not scene_jobs:
        raise ValueError(f"resolved scene has no jobs: {scene_id}")
    job_ids = [job.get("job_id") for job in scene_jobs]
    if audit_scene["job_ids"] != job_ids or audit_scene["accepted_jobs"] != len(scene_jobs):
        raise ValueError("audit scene roster differs from resolved jobs")
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("resolved scene contains duplicate job IDs")

    _, _, report, report_identity = _read_verified_identity(
        audit_scene["report_identity"], "factory report", parse_json=True
    )
    _, _, prehybrid, prehybrid_identity = _read_verified_identity(
        audit_scene["prehybrid_identity"], "pre-hybrid audit", parse_json=True
    )
    _, _, source_manifest, source_manifest_identity = _read_verified_identity(
        audit_scene["source_identity_manifest"], "E2 source manifest", parse_json=True
    )
    if not isinstance(source_manifest, Mapping) or source_manifest.get("scene_id") != scene_id:
        raise ValueError("E2 source manifest scene binding differs")

    gaussian = _validate_external_identity(
        audit_scene["source_scene_gaussian"], "audit Scene Gaussian"
    )
    job_gaussian = _validate_external_identity(
        job_scene["source_scene_gaussian"], "jobs Scene Gaussian"
    )
    manifest_gaussian = _validate_external_identity(
        source_manifest.get("source_scene_gaussian"), "manifest Scene Gaussian"
    )
    if gaussian != job_gaussian or gaussian != manifest_gaussian:
        raise ValueError("Scene Gaussian provenance differs across sealed inventories")
    audit_cameras = _exact_mapping(
        audit_scene["camera_artifacts"], {"intrinsics", "poses"}, "audit cameras"
    )
    job_cameras = _exact_mapping(
        job_scene["camera_artifacts"], {"intrinsics", "poses"}, "job cameras"
    )
    manifest_cameras = _exact_mapping(
        source_manifest.get("camera_artifacts"),
        {"intrinsics", "poses"},
        "manifest cameras",
    )
    cameras: dict[str, dict[str, Any]] = {}
    for role in ("intrinsics", "poses"):
        cameras[role] = _validate_external_identity(audit_cameras[role], f"audit {role}")
        if cameras[role] != _validate_external_identity(job_cameras[role], f"jobs {role}"):
            raise ValueError(f"camera {role} provenance differs between audit and jobs")
        if cameras[role] != _validate_external_identity(
            manifest_cameras[role], f"manifest {role}"
        ):
            raise ValueError(f"camera {role} provenance differs from E2 manifest")

    construction_sources = source_manifest.get("construction_sources")
    if not isinstance(construction_sources, Mapping):
        raise ValueError("E2 source manifest lacks construction_sources")
    source = construction_sources.get("factorized_gt_discovery")
    if not isinstance(source, Mapping):
        raise ValueError("E2 source manifest lacks factorized_gt_discovery")
    expected_source = f"outputs/{scene_id}_factory"
    if source.get("source") != expected_source:
        raise ValueError("factory source path differs from the scene-bound expected path")
    accepted_ids = source.get("accepted_object_ids")
    indices: list[int] = []
    object_slots: list[str] = []
    for job in scene_jobs:
        slot = job.get("object_slot")
        match = OBJECT_SLOT_RE.fullmatch(slot) if isinstance(slot, str) else None
        if match is None:
            raise ValueError(f"invalid resolved object slot: {slot!r}")
        index = int(match.group(1))
        if slot != f"obj_{index:02d}" or job.get("scene_id") != scene_id:
            raise ValueError(f"non-canonical job binding for {scene_id}/{slot}")
        if job.get("job_id") != f"{scene_id}/{slot}":
            raise ValueError(f"non-canonical job ID for {scene_id}/{slot}")
        indices.append(index)
        object_slots.append(slot)
    if accepted_ids != indices:
        raise ValueError("E2 accepted object IDs differ from exact E3 scene roster")
    if len(indices) != len(set(indices)):
        raise ValueError("E3 scene roster contains duplicate object indices")
    canonical_order = sorted(range(len(indices)), key=indices.__getitem__)
    scene_jobs = [scene_jobs[index] for index in canonical_order]
    indices = [indices[index] for index in canonical_order]
    object_slots = [object_slots[index] for index in canonical_order]

    source_contract = jobs.get("source_contract")
    if not isinstance(source_contract, Mapping) or not re.fullmatch(
        r"[0-9a-f]{40}", str(source_contract.get("code_commit"))
    ):
        raise ValueError("E3 source contract code_commit must be a full Git SHA")

    artifacts = _identity_map(source.get("artifacts"), "E2 factory artifacts")
    objects_path = f"{expected_source}/objects/objects.json"
    if objects_path not in artifacts:
        raise ValueError("E2 source manifest does not authenticate factory objects.json")
    _, _, objects, objects_identity = _read_verified_identity(
        artifacts[objects_path], "factory objects metadata", parse_json=True
    )
    object_rows = _indexed_rows(objects, "factory objects metadata")
    report_rows = _indexed_rows(
        report.get("objects") if isinstance(report, Mapping) else None, "factory report objects"
    )
    prehybrid_rows = _indexed_rows(prehybrid, "pre-hybrid rows")
    accepted_report = sorted(
        index
        for index, row in report_rows.items()
        if row.get("tier") in {"A", "B"} and row.get("rejected") is None
    )
    if accepted_report != sorted(indices):
        raise ValueError("authenticated factory report membership differs from E3 roster")

    output_rows: list[dict[str, Any]] = []
    report_tiers: dict[str, str] = {}
    meta_identities: dict[str, dict[str, Any]] = {}
    rgba_identities: dict[str, dict[str, Any]] = {}
    for job, index, slot in zip(scene_jobs, indices, object_slots, strict=True):
        if index not in object_rows or index not in report_rows or index not in prehybrid_rows:
            raise ValueError(f"roster object is absent from source metadata: {scene_id}/{slot}")
        object_row = object_rows[index]
        report_row = report_rows[index]
        pre_row = prehybrid_rows[index]
        label = object_row.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"source object label is invalid: {scene_id}/{slot}")
        _validate_aabb(object_row.get("aabb"), f"{scene_id}/{slot} aabb")
        if (
            report_row.get("label") != label
            or pre_row.get("label") != label
            or report_row.get("gt_object_id") != object_row.get("gt_object_id")
            or pre_row.get("tier") not in {"A", "B"}
            or pre_row.get("rejected") is not None
            or pre_row.get("tier") != report_row.get("tier")
        ):
            raise ValueError(f"source report/pre-hybrid/object metadata disagree: {scene_id}/{slot}")

        meta_path = f"{expected_source}/objects/{slot}/meta.json"
        if meta_path not in artifacts:
            raise ValueError(f"E2 source manifest does not authenticate {meta_path}")
        _, _, meta, meta_identity = _read_verified_identity(
            artifacts[meta_path], f"{scene_id}/{slot} metadata", parse_json=True
        )
        if not isinstance(meta, Mapping):
            raise ValueError(f"{scene_id}/{slot} metadata must be an object")
        evidence = job.get("construction_evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"{scene_id}/{slot} construction evidence must be an object")
        if (
            meta.get("frame") != evidence.get("source_frame")
            or meta.get("bbox_px") != evidence.get("bbox_px")
            or object_row.get("frame") != evidence.get("source_frame")
            or object_row.get("bbox_px") != evidence.get("bbox_px")
        ):
            raise ValueError(f"source frame/crop provenance differs: {scene_id}/{slot}")

        paths = _exact_mapping(job.get("artifact_paths"), {"rgba"}, "job artifact paths")
        hashes = _exact_mapping(job.get("artifact_hashes"), {"rgba"}, "job artifact hashes")
        expected_rgba_path = f"{expected_source}/objects/{slot}/rgba.png"
        if paths["rgba"] != expected_rgba_path:
            raise ValueError(f"RGBA path differs from scene/object binding: {scene_id}/{slot}")
        rgba_path = _repo_relative_path(paths["rgba"], f"{scene_id}/{slot} RGBA")
        rgba_digest = _digest(hashes["rgba"], f"{scene_id}/{slot} RGBA hash")
        actual_rgba_digest = sha256_file(rgba_path)
        if actual_rgba_digest != rgba_digest:
            raise ValueError(f"RGBA hash drift: {scene_id}/{slot}")
        rgba_identities[slot] = {
            "path": _display(rgba_path),
            "size_bytes": rgba_path.stat().st_size,
            "sha256": rgba_digest,
        }
        meta_identities[slot] = meta_identity
        report_tiers[slot] = report_row["tier"]
        output_rows.append(dict(object_row))

    return {
        "audit_scene": dict(audit_scene),
        "jobs": list(scene_jobs),
        "object_slots": object_slots,
        "object_rows": output_rows,
        "report_tiers": report_tiers,
        "source_factory": expected_source,
        "source_objects": objects_identity,
        "report": report_identity,
        "prehybrid": prehybrid_identity,
        "source_manifest": source_manifest_identity,
        "source_scene_gaussian": gaussian,
        "camera_artifacts": cameras,
        "meta_identities": meta_identities,
        "rgba_identities": rgba_identities,
    }


def _string_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{label} must be a unique list of non-empty strings")
    return list(value)


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _validate_registration(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "T",
        "mesh_sample_count",
        "mesh_sample_seed",
        "observation_dims_m",
        "registration_initial_objective_m",
        "registration_median_m",
        "residual_tilt_deg",
        "scale",
        "scale_ratio_vs_observation",
        "source_up_hypothesis",
        "symmetric_clipped_registration_residual_m",
        "world_dims_m",
    }
    value = _exact_mapping(value, expected, label)
    transform = value["T"]
    if not isinstance(transform, list) or len(transform) != 4:
        raise ValueError(f"{label}.T must be a 4x4 matrix")
    parsed: list[list[float]] = []
    for row in transform:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{label}.T must be a 4x4 matrix")
        parsed.append([_finite_number(item, f"{label}.T") for item in row])
    if any(abs(a - b) > 1e-12 for a, b in zip(parsed[3], [0.0, 0.0, 0.0, 1.0], strict=True)):
        raise ValueError(f"{label}.T has an invalid homogeneous final row")
    dims = value["world_dims_m"]
    if not isinstance(dims, list) or len(dims) != 3:
        raise ValueError(f"{label}.world_dims_m must have three values")
    world_dims = [_finite_number(item, f"{label}.world_dims_m", positive=True) for item in dims]
    scale = _finite_number(value["scale"], f"{label}.scale", positive=True)
    registration_median = _finite_number(
        value["registration_median_m"], f"{label}.registration_median_m"
    )
    symmetric_residual = _finite_number(
        value["symmetric_clipped_registration_residual_m"],
        f"{label}.symmetric_clipped_registration_residual_m",
    )
    if registration_median < 0 or symmetric_residual < 0:
        raise ValueError(f"{label} registration residuals must be non-negative")
    return {
        "T": parsed,
        "registration_initial_objective_m": _finite_number(
            value["registration_initial_objective_m"],
            f"{label}.registration_initial_objective_m",
        ),
        "registration_median_m": registration_median,
        "residual_tilt_deg": _finite_number(
            value["residual_tilt_deg"], f"{label}.residual_tilt_deg"
        ),
        "scale": scale,
        "scale_ratio_vs_observation": _finite_number(
            value["scale_ratio_vs_observation"],
            f"{label}.scale_ratio_vs_observation",
            positive=True,
        ),
        "source_up_hypothesis": _plain_string(
            value["source_up_hypothesis"], f"{label}.source_up_hypothesis"
        ),
        "symmetric_clipped_registration_residual_m": symmetric_residual,
        "world_dims": world_dims,
    }


def _validate_physics(value: Any, label: str) -> None:
    value = _exact_mapping(value, {"friction", "mass_kg", "restitution", "source"}, label)
    _finite_number(value["mass_kg"], f"{label}.mass_kg", positive=True)
    if _finite_number(value["friction"], f"{label}.friction") < 0:
        raise ValueError(f"{label}.friction must be non-negative")
    if _finite_number(value["restitution"], f"{label}.restitution") < 0:
        raise ValueError(f"{label}.restitution must be non-negative")
    _plain_string(value["source"], f"{label}.source")


def _validate_probe(value: Any, label: str) -> None:
    value = _exact_mapping(
        value,
        {
            "collision_backend",
            "collision_valid",
            "contact_dynamics",
            "final_aabb_min_z_m",
            "initial_penetration_m",
            "settle_drift_m",
            "settle_stable",
            "settle_sunk",
            "support_gap_m",
            "support_overlap_fraction",
            "usable_convex_parts",
        },
        label,
    )
    if value["collision_valid"] is not True:
        raise ValueError(f"{label} has no valid collision")
    if _positive_int(value["usable_convex_parts"], f"{label}.usable_convex_parts") < 1:
        raise ValueError(f"{label} has no usable collision part")


def _validate_urdf(payload: bytes, label: str) -> None:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError(f"{label} is not valid XML") from exc
    if root.tag != "robot":
        raise ValueError(f"{label} root must be <robot>")
    filenames = []
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not isinstance(filename, str):
            raise ValueError(f"{label} mesh lacks filename")
        pure = PurePosixPath(filename)
        if pure.is_absolute() or filename != pure.as_posix() or ".." in pure.parts:
            raise ValueError(f"{label} contains a non-local mesh reference: {filename!r}")
        filenames.append(filename)
    expected = {"mesh_sim.obj", "collision/part_00.obj"}
    if set(filenames) != expected:
        raise ValueError(
            f"{label} mesh references differ: expected={sorted(expected)}, observed={sorted(set(filenames))}"
        )


def _expected_artifact_path(
    e3_root: Path, scene_id: str, slot: str, proposal_id: str, role: str
) -> str | None:
    if role in {"raw_mesh", "raw_gaussian"}:
        return None
    safe_name = proposal_id.replace("/", "_").replace(":", "_")
    return _display(e3_root / "control" / scene_id / "proposals" / safe_name / RUNTIME_ROLE_SUFFIXES[role])


def _prepare_selected_records(
    *,
    e3_root: Path,
    scene_id: str,
    policy_id: str,
    freeze_id: str,
    proposals: Mapping[str, Any],
    source: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected_root = e3_root / "selected_assets"
    expected_names = sorted(f"{slot}.json" for slot in source["object_slots"])
    scene_dir = e3.checked_repo_path(
        (Path(source["selected_scene_root"]) / policy_id if source.get("automatic")
         else selected_root / policy_id / scene_id), "selected scene directory", kind="dir"
    )
    observed_names = sorted(path.name for path in scene_dir.iterdir())
    if observed_names != expected_names:
        raise ValueError(
            f"selected scene roster differs: expected={expected_names}, observed={observed_names}"
        )

    proposal_rows = proposals.get("proposals")
    if not isinstance(proposal_rows, list):
        raise ValueError("resolved proposals has no proposal list")
    proposal_index: dict[str, Mapping[str, Any]] = {}
    for proposal in proposal_rows:
        if not isinstance(proposal, Mapping):
            raise ValueError("resolved proposal row must be an object")
        proposal_id = proposal.get("proposal_id")
        if not isinstance(proposal_id, str) or proposal_id in proposal_index:
            raise ValueError(f"invalid or duplicate resolved proposal ID: {proposal_id!r}")
        proposal_index[proposal_id] = proposal

    prepared: list[dict[str, Any]] = []
    for job, slot, object_row in zip(
        source["jobs"], source["object_slots"], source["object_rows"], strict=True
    ):
        relative = f"{policy_id}/{scene_id}/{slot}.json"
        selected_identity = aggregate["selected_records"].get(relative)
        if selected_identity is None:
            raise ValueError(f"selected record is absent from aggregate seal: {relative}")
        path = e3.checked_repo_path(
            selected_identity["path"] if source.get("automatic") else selected_root / relative,
            relative, kind="file")
        payload = path.read_bytes()
        if (
            len(payload) != selected_identity["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != selected_identity["sha256"]
        ):
            raise ValueError(f"selected record drifted after aggregate verification: {relative}")
        record = _parse_json_bytes(payload, relative)
        record = _exact_mapping(record, SELECTED_RECORD_KEYS, f"selected record {relative}")
        if (
            record["schema_version"] != SCHEMA_VERSION
            or record["freeze_id"] != freeze_id
            or record["scene_id"] != scene_id
            or record["policy_id"] != policy_id
            or record["object_slot"] != slot
            or record["job_id"] != job["job_id"]
        ):
            raise ValueError(f"selected record binding differs: {relative}")
        _string_list(record["reason_codes"], f"{relative}.reason_codes")
        _string_list(record["retry_reason_codes"], f"{relative}.retry_reason_codes")
        for field in (
            "retry_attempted",
            "retry_invoked",
            "retry_produced",
            "retry_required",
        ):
            if not isinstance(record[field], bool):
                raise ValueError(f"{relative}.{field} must be boolean")

        action = record["terminal_action"]
        if action == "abstain" or (source.get("automatic") and action == "reject"):
            if action == "abstain" and policy_id != "A4":
                raise ValueError("A0 must not contain an abstention")
            if action == "reject" and policy_id != "A0":
                raise ValueError("automatic A4 must preserve its abstention action")
            if (
                record["selected_asset"] is not None
                or record["selected_proposal_id"] is not None
                or record["support_label"] != "unsupported"
                or not record["reason_codes"]
                or record["proposal_id"] != f"{job['job_id']}:terminal:{policy_id}:{action}"
            ):
                raise ValueError(f"invalid A4 abstention record: {relative}")
            prepared.append(
                {
                    "action": action,
                    "job": job,
                    "object_row": object_row,
                    "record": dict(record),
                    "record_bytes": payload,
                    "record_identity": selected_identity,
                    "artifacts": {},
                    "alignment": None,
                }
            )
            continue
        if action != "accept":
            raise ValueError(f"unsupported terminal action in {relative}: {action!r}")
        expected_support_label = "not_evaluated" if policy_id == "A0" else "supported"
        if (
            record["support_label"] != expected_support_label
            or not isinstance(record["selected_proposal_id"], str)
            or record["selected_proposal_id"] != record["proposal_id"]
        ):
            raise ValueError(f"invalid accepted selection binding: {relative}")
        asset = _exact_mapping(record["selected_asset"], SELECTED_ASSET_KEYS, f"{relative}.selected_asset")
        _digest(asset["proposal_digest"], f"{relative}.proposal_digest")
        _digest(asset["registration_surface_hash"], f"{relative}.registration_surface_hash")
        tool = asset["tool"]
        if tool not in {"trellis", "reconviagen", "registration_retry"}:
            raise ValueError(f"invalid selected tool in {relative}: {tool!r}")
        paths = asset["artifact_paths"]
        hashes = asset["artifact_hashes"]
        sizes = asset["artifact_sizes"]
        if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or not isinstance(sizes, Mapping):
            raise ValueError(f"accepted artifact maps must be objects: {relative}")
        roles = set(paths)
        if roles != set(hashes) or roles != set(sizes):
            raise ValueError(f"accepted artifact path/hash/size roles differ: {relative}")
        allowed = set(REQUIRED_ARTIFACT_ROLES)
        if tool == "registration_retry":
            allowed.add("retry")
        if roles != allowed:
            raise ValueError(
                f"accepted artifact roles differ: expected={sorted(allowed)}, observed={sorted(roles)}"
            )

        artifact_specs: dict[str, dict[str, Any]] = {}
        captured_values: dict[str, Any] = {}
        raw_prefix = f"outputs/{scene_id}_factory/objects/{slot}/"
        raw_variants: dict[str, str] = {}
        for role in sorted(roles):
            role_path = _repo_relative_path(paths[role], f"{relative}.{role}")
            expected_size = _positive_int(sizes[role], f"{relative}.{role}.size")
            expected_hash = _digest(hashes[role], f"{relative}.{role}.hash")
            if role_path.stat().st_size != expected_size:
                raise ValueError(f"selected artifact size drift: {relative}.{role}")
            expected_path = _expected_artifact_path(
                e3_root, scene_id, slot, record["selected_proposal_id"], role
            )
            if expected_path is not None and paths[role] != expected_path:
                raise ValueError(f"selected runtime artifact path binding differs: {relative}.{role}")
            if role in {"raw_mesh", "raw_gaussian"}:
                if source.get("automatic"):
                    # Automatic generation uses a separate prepared-index namespace.
                    # Exact path/hash/size and stable job ownership are checked below
                    # against the sealed proposal, never a guessed GT factory path.
                    name = Path(paths[role]).name
                    prefix = "trellis" if name.startswith("trellis_") else "rvg"
                    suffix = f"{prefix}/{name}"
                else:
                    if not paths[role].startswith(raw_prefix):
                        raise ValueError(f"raw artifact escapes source object: {relative}.{role}")
                    suffix = paths[role][len(raw_prefix) :]
                allowed_suffixes = {
                    "raw_mesh": {"trellis/trellis_mesh.ply", "rvg/rvg_mesh.ply"},
                    "raw_gaussian": {"trellis/trellis_gs.ply", "rvg/rvg_gs.ply"},
                }
                if suffix not in allowed_suffixes[role]:
                    raise ValueError(f"unexpected raw artifact path: {relative}.{role}")
                raw_variants[role] = suffix.split("/", 1)[0]
            spec = {
                "path": role_path,
                "path_display": paths[role],
                "size_bytes": expected_size,
                "sha256": expected_hash,
                "captured": None,
            }
            if role in CAPTURED_TEXT_ROLES:
                role_payload = role_path.read_bytes()
                if len(role_payload) != expected_size or hashlib.sha256(role_payload).hexdigest() != expected_hash:
                    raise ValueError(f"selected artifact hash/size drift: {relative}.{role}")
                spec["captured"] = role_payload
                if role == "urdf":
                    _validate_urdf(role_payload, f"{relative}.urdf")
                else:
                    captured_values[role] = _parse_json_bytes(
                        role_payload, f"{relative}.{role}"
                    )
            artifact_specs[role] = spec
        if raw_variants.get("raw_mesh") != raw_variants.get("raw_gaussian"):
            raise ValueError(f"raw mesh/Gaussian variants differ: {relative}")
        raw_variant = raw_variants["raw_mesh"]
        proposal_id = record["selected_proposal_id"]
        if tool == "trellis" and (
            raw_variant != "trellis" or not proposal_id.endswith(":trellis")
        ):
            raise ValueError(f"selected TRELLIS tool/proposal/raw variant differ: {relative}")
        if tool == "reconviagen" and (
            raw_variant != "rvg" or not proposal_id.endswith(":reconviagen")
        ):
            raise ValueError(
                f"selected ReconViaGen tool/proposal/raw variant differ: {relative}"
            )
        if tool == "registration_retry":
            if not proposal_id.endswith(":retry:signed-source-up"):
                raise ValueError(f"selected retry proposal identity differs: {relative}")
            retry_value = captured_values.get("retry")
            if not isinstance(retry_value, Mapping):
                raise ValueError(f"selected retry manifest is absent: {relative}")
            parent_id = retry_value.get("parent_proposal_id")
            expected_parent_suffix = (
                ":trellis" if raw_variant == "trellis" else ":reconviagen"
            )
            if (
                retry_value.get("proposal_id") != proposal_id
                or not isinstance(parent_id, str)
                or not parent_id.endswith(expected_parent_suffix)
            ):
                raise ValueError(f"selected retry parent/raw variant differ: {relative}")
            resolved_id = parent_id
            expected_tool = (
                "trellis" if raw_variant == "trellis" else "reconviagen"
            )
        else:
            resolved_id = proposal_id
            expected_tool = tool
        resolved = proposal_index.get(resolved_id)
        if (
            not isinstance(resolved, Mapping)
            or resolved.get("availability") != "available"
            or resolved.get("tool_id") != expected_tool
            or resolved.get("scene_id") != scene_id
            or resolved.get("object_slot") != slot
            or resolved.get("job_id") != job["job_id"]
        ):
            raise ValueError(
                f"selected proposal is not an available sealed source proposal: {relative}"
            )
        for field, selected_map in (
            ("artifact_paths", paths),
            ("artifact_hashes", hashes),
            ("artifact_sizes", sizes),
        ):
            resolved_map = resolved.get(field)
            if not isinstance(resolved_map, Mapping) or {
                role: selected_map[role] for role in ("raw_mesh", "raw_gaussian")
            } != dict(resolved_map):
                raise ValueError(
                    f"selected raw artifact identity differs from resolved proposal: "
                    f"{relative}.{field}"
                )
        alignment = _validate_registration(
            captured_values["registration"], f"{relative}.registration"
        )
        _validate_physics(captured_values["physics"], f"{relative}.physics")
        _validate_probe(captured_values["probe"], f"{relative}.probe")
        if not isinstance(captured_values["evidence"], Mapping):
            raise ValueError(f"{relative}.evidence must be an object")
        if "retry" in roles and not isinstance(captured_values["retry"], Mapping):
            raise ValueError(f"{relative}.retry must be an object")
        prepared.append(
            {
                "action": "accept",
                "job": job,
                "object_row": object_row,
                "record": dict(record),
                "record_bytes": payload,
                "record_identity": selected_identity,
                "artifacts": artifact_specs,
                "alignment": alignment,
            }
        )
    return prepared


def _write_inside(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _copy_verified_inside(source: Path, destination: Path, size: int, digest: str) -> dict[str, Any]:
    source = e3.checked_repo_path(source, "selected artifact at copy", kind="file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    total = 0
    with source.open("rb") as src, destination.open("xb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            total += len(chunk)
            hasher.update(chunk)
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    actual = hasher.hexdigest()
    if total != size or actual != digest:
        raise ValueError(
            f"selected artifact drifted during copy: {source} "
            f"expected=({size},{digest}) found=({total},{actual})"
        )
    return {"size_bytes": total, "sha256": actual}


def _record_member(
    members: dict[str, dict[str, Any]], root: Path, path: Path, identity: Mapping[str, Any]
) -> None:
    relative = path.relative_to(root).as_posix()
    if relative in members:
        raise ValueError(f"duplicate output member: {relative}")
    members[relative] = {
        "size_bytes": identity["size_bytes"],
        "sha256": identity["sha256"],
    }


def _fsync_tree(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        e3._fsync_directory(path)
    e3._fsync_directory(root)


def _validated_local_path(
    value: str | Path,
    *,
    repository_root: Path,
    label: str,
    kind: str,
) -> Path:
    """Resolve one path beneath *repository_root* without following symlinks.

    The materializer itself is fixed to this checkout.  The explicit root here
    additionally lets the harness exercise the same validator in isolated test
    repositories instead of weakening production path checks.
    """
    repository_root = repository_root.resolve(strict=True)
    raw_text = os.fspath(value)
    if not isinstance(raw_text, str) or not raw_text:
        raise ValueError(f"{label} must be a non-empty path")
    raw = Path(raw_text)
    if "\\" in raw_text or any(part in {".", ".."} for part in raw.parts):
        raise ValueError(f"{label} must not contain dot segments or backslashes")
    # Path() removes explicit './' and duplicate separators.  Reject those
    # non-canonical spellings as well so a manifest cannot authenticate one
    # lexical path while the OS opens another.
    canonical_spelling = raw.as_posix()
    if raw_text != canonical_spelling:
        raise ValueError(f"{label} must use a canonical POSIX path spelling")
    candidate = raw if raw.is_absolute() else repository_root / raw
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(repository_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository: {candidate}") from exc
    current = repository_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")
    if kind == "file" and not candidate.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {candidate}")
    if kind == "dir" and not candidate.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ValueError(f"{label} resolves through a non-canonical path: {candidate}")
    return candidate


def validate_materialized_factory(
    factory_dir: str | Path,
    *,
    expected_scene_id: str | None = None,
    expected_policy_id: str | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Revalidate a published E3->E4 factory before any rollout uses it.

    ``sim_export`` is intentionally allowed to be added after materialization,
    but the complete ``objects`` subtree must remain byte-for-byte equal to the
    sealed materializer output.  This prevents swapped A0/A4 labels, arbitrary
    look-alike directories, and post-materialization object mutation.
    """
    validator_snapshot = _require_clean_code_snapshot()
    root = Path(REPOSITORY_ROOT if repository_root is None else repository_root).resolve(
        strict=True
    )
    if root != REPOSITORY_ROOT:
        raise ValueError(
            "validation root differs from SIMANY_EVIDENCE_ROOT: "
            f"expected={REPOSITORY_ROOT}, observed={root}"
        )
    factory = _validated_local_path(
        factory_dir, repository_root=root, label="materialized factory", kind="dir"
    )
    manifest_path = _validated_local_path(
        factory / "materialization_manifest.json",
        repository_root=root,
        label="materialization manifest",
        kind="file",
    )
    seal_path = _validated_local_path(
        factory / "seal.json",
        repository_root=root,
        label="materialization seal",
        kind="file",
    )
    seal = _parse_json_bytes(seal_path.read_bytes(), "materialization seal")
    seal = _exact_mapping(
        seal,
        {"manifest_kind", "members", "schema_version"},
        "materialization seal",
    )
    if (
        seal["schema_version"] != SCHEMA_VERSION
        or seal["manifest_kind"] != "e4_construction_variant_materialization"
    ):
        raise ValueError("unsupported materialization seal identity")
    members = _exact_mapping(
        seal["members"], {"materialization_manifest.json"}, "materialization seal members"
    )
    manifest_identity = _exact_mapping(
        members["materialization_manifest.json"],
        {"sha256", "size_bytes"},
        "materialization manifest identity",
    )
    manifest_bytes = manifest_path.read_bytes()
    expected_size = _positive_int(
        manifest_identity["size_bytes"], "materialization manifest size_bytes"
    )
    expected_digest = _digest(
        manifest_identity["sha256"], "materialization manifest sha256"
    )
    if len(manifest_bytes) != expected_size:
        raise ValueError("materialization manifest size drift")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_digest:
        raise ValueError("materialization manifest hash drift")

    manifest = _parse_json_bytes(manifest_bytes, "materialization manifest")
    if isinstance(manifest, Mapping) and manifest.get("schema_version") == 2:
        return _validate_automatic_factory(factory, manifest, validator_snapshot,
                                           expected_scene_id, expected_policy_id)
    manifest = _exact_mapping(
        manifest, MATERIALIZATION_MANIFEST_KEYS, "materialization manifest"
    )
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["manifest_kind"] != "e4_construction_variant_materialization"
    ):
        raise ValueError("unsupported materialization manifest identity")
    scene_id = _plain_string(manifest["scene_id"], "materialization scene_id")
    policy_id = _plain_string(manifest["policy_id"], "materialization policy_id")
    if not SCENE_ID_RE.fullmatch(scene_id):
        raise ValueError(f"invalid materialization scene_id: {scene_id!r}")
    if policy_id not in ALLOWED_POLICIES:
        raise ValueError(f"invalid materialization policy_id: {policy_id!r}")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError(
            f"materialization scene binding differs: expected {expected_scene_id}, found {scene_id}"
        )
    if expected_policy_id is not None and policy_id != expected_policy_id:
        raise ValueError(
            f"materialization policy binding differs: expected {expected_policy_id}, found {policy_id}"
        )
    destination_raw = _plain_string(
        manifest["destination"], "materialization destination"
    )
    destination = PurePosixPath(destination_raw)
    if (
        destination.is_absolute()
        or destination.as_posix() != destination_raw
        or any(part in {"", ".", ".."} for part in destination.parts)
    ):
        raise ValueError("materialization destination must be canonical repository-relative")
    if _validated_local_path(
        root / Path(destination_raw),
        repository_root=root,
        label="materialization destination",
        kind="dir",
    ) != factory:
        raise ValueError("materialization destination does not name the validated factory")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest["e3_code_commit"])):
        raise ValueError("materialization e3_code_commit is not a full Git SHA")
    _plain_string(manifest["e3_freeze_id"], "materialization e3_freeze_id")

    e3_root_raw = _plain_string(manifest["e3_root"], "materialization e3_root")
    e3_root_pure = PurePosixPath(e3_root_raw)
    if (
        e3_root_pure.is_absolute()
        or e3_root_pure.as_posix() != e3_root_raw
        or any(part in {"", ".", ".."} for part in e3_root_pure.parts)
    ):
        raise ValueError("materialization e3_root must be canonical repository-relative")
    e3_root_path = _validated_local_path(
        root / Path(e3_root_raw),
        repository_root=root,
        label="materialization E3 root",
        kind="dir",
    )
    _, aggregate = _verify_aggregate(e3_root_path)
    jobs, _proposals, audit, inventory = _verify_inventory(e3_root_path)
    if (
        aggregate["claim_status"]["freeze_id"] != jobs.get("freeze_id")
        or aggregate["claim_status"]["study_scope"] != jobs.get("study_scope")
    ):
        raise ValueError("E3 aggregate claim status differs from the sealed inventory")
    if (
        jobs.get("freeze_id") != manifest["e3_freeze_id"]
        or jobs.get("source_contract", {}).get("code_commit")
        != manifest["e3_code_commit"]
    ):
        raise ValueError("materialization E3 freeze/code binding differs")
    if (
        aggregate["claim_status"]["freeze_id"] != manifest["e3_freeze_id"]
        or aggregate["claim_status"]["study_scope"] != manifest["study_scope"]
    ):
        raise ValueError("materialization E3 aggregate claim binding differs")
    if manifest["e3_claim_status"] != aggregate["claim_status"]:
        raise ValueError("materialization E3 claim status differs from sealed aggregate")
    if manifest["study_scope"] != jobs.get("study_scope"):
        raise ValueError("materialization study scope differs from sealed E3 inventory")
    provenance = _validate_provenance_record(
        manifest["provenance"],
        e3_producer_commit=jobs["source_contract"]["code_commit"],
        evidence_root=root,
        validator_snapshot=validator_snapshot,
    )
    expected_input_identities = {
        **inventory,
        "aggregate_seal": aggregate["aggregate_seal"],
    }
    if manifest["input_identities"] != expected_input_identities:
        raise ValueError("materialization input identities differ from sealed E3 inputs")

    source = _scene_source_context(jobs, audit, scene_id)
    prepared = _prepare_selected_records(
        e3_root=e3_root_path,
        scene_id=scene_id,
        policy_id=policy_id,
        freeze_id=manifest["e3_freeze_id"],
        proposals=_proposals,
        source=source,
        aggregate=aggregate,
    )
    expected_source_scene = {
        "camera_artifacts": source["camera_artifacts"],
        "external_source_bytes_opened": False,
        "factory_objects": source["source_objects"],
        "factory_report": source["report"],
        "object_metadata": source["meta_identities"],
        "prehybrid_audit": source["prehybrid"],
        "rgba_inputs": source["rgba_identities"],
        "source_factory": source["source_factory"],
        "source_identity_manifest": source["source_manifest"],
        "source_scene_gaussian": source["source_scene_gaussian"],
    }
    if manifest["source_scene"] != expected_source_scene:
        raise ValueError("materialization source-scene provenance differs from sealed E3 inputs")

    roster = _exact_mapping(
        manifest["roster"], MATERIALIZATION_ROSTER_KEYS, "materialization roster"
    )
    slots = _string_list(roster["object_slots"], "materialization object_slots")
    accepted = _string_list(roster["accepted_slots"], "materialization accepted_slots")
    abstained = _string_list(roster["abstained_slots"], "materialization abstained_slots")
    if any(not OBJECT_SLOT_RE.fullmatch(slot) for slot in slots):
        raise ValueError("materialization roster contains an invalid object slot")
    slot_indices = [int(OBJECT_SLOT_RE.fullmatch(slot).group(1)) for slot in slots]
    if slot_indices != sorted(slot_indices) or len(slots) != len(set(slots)):
        raise ValueError(
            "materialization object_slots must be unique and numerically sorted"
        )
    if set(accepted) | set(abstained) != set(slots) or set(accepted) & set(abstained):
        raise ValueError("materialization accepted/abstained roster is not an exact partition")
    counts = {
        "job_count": len(slots),
        "accepted_count": len(accepted),
        "abstained_count": len(abstained),
    }
    for field, actual in counts.items():
        if _nonnegative_int(roster[field], f"materialization {field}") != actual:
            raise ValueError(f"materialization {field} does not match its roster")
    if policy_id == "A0" and abstained:
        raise ValueError("A0 materialization contains abstentions")
    expected_accepted = [
        row["record"]["object_slot"] for row in prepared if row["action"] == "accept"
    ]
    expected_abstained = [
        row["record"]["object_slot"] for row in prepared if row["action"] == "abstain"
    ]
    expected_roster_rows = [
        {
            "job_id": row["record"]["job_id"],
            "object_slot": row["record"]["object_slot"],
            "terminal_action": row["action"],
        }
        for row in prepared
    ]
    expected_roster_digest = hashlib.sha256(
        canonical_json(expected_roster_rows).encode("utf-8")
    ).hexdigest()
    if (
        slots != list(source["object_slots"])
        or accepted != expected_accepted
        or abstained != expected_abstained
        or _digest(roster["sha256"], "materialization roster sha256")
        != expected_roster_digest
    ):
        raise ValueError("materialization roster differs from sealed E3 selections")

    output_members = manifest["output_members"]
    if not isinstance(output_members, Mapping) or not output_members:
        raise ValueError("materialization output_members must be non-empty")
    declared_object_members: list[str] = []
    for relative, identity in sorted(output_members.items()):
        if not isinstance(relative, str):
            raise ValueError("materialization output member names must be strings")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or relative != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or not relative.startswith("objects/")
        ):
            raise ValueError(f"invalid materialization output member path: {relative!r}")
        identity = _exact_mapping(
            identity, {"sha256", "size_bytes"}, f"materialization output {relative}"
        )
        member = _validated_local_path(
            factory / relative,
            repository_root=root,
            label=f"materialization output {relative}",
            kind="file",
        )
        size = _positive_int(identity["size_bytes"], f"materialization output {relative} size")
        digest = _digest(identity["sha256"], f"materialization output {relative} sha256")
        if member.stat().st_size != size or sha256_file(member) != digest:
            raise ValueError(f"materialization output drift: {relative}")
        declared_object_members.append(relative)
    objects_root = _validated_local_path(
        factory / "objects", repository_root=root, label="materialized objects", kind="dir"
    )
    actual_object_members = sorted(
        path.relative_to(factory).as_posix()
        for path in objects_root.rglob("*")
        if path.is_file()
    )
    if actual_object_members != sorted(declared_object_members):
        raise ValueError("materialized objects tree differs from its exact sealed inventory")

    selected_records = manifest["selected_records"]
    expected_selected_records = [
        {
            "job_id": row["record"]["job_id"],
            "object_slot": row["record"]["object_slot"],
            "selected_proposal_id": row["record"]["selected_proposal_id"],
            "selected_record": dict(row["record_identity"]),
            "terminal_action": row["action"],
        }
        for row in prepared
    ]
    if selected_records != expected_selected_records:
        raise ValueError("materialization selected_records differ from sealed E3 selections")

    expected_member_identities: dict[str, tuple[int, str]] = {}
    expected_objects_bytes = _json_bytes(source["object_rows"])
    expected_member_identities["objects/objects.json"] = (
        len(expected_objects_bytes),
        hashlib.sha256(expected_objects_bytes).hexdigest(),
    )
    for row in prepared:
        slot = row["record"]["object_slot"]
        expected_member_identities[f"objects/{slot}/selected_asset.json"] = (
            row["record_identity"]["size_bytes"],
            row["record_identity"]["sha256"],
        )
        if row["action"] == "abstain":
            aligned = {
                "index": row["object_row"]["index"],
                "job_id": row["record"]["job_id"],
                "label": row["object_row"]["label"],
                "policy_id": policy_id,
                "reason_codes": row["record"]["reason_codes"],
                "rejected": "e3_policy_abstention",
                "schema_version": SCHEMA_VERSION,
                "terminal_action": "abstain",
            }
        else:
            alignment = row["alignment"]
            aligned = {
                "T": alignment["T"],
                "index": row["object_row"]["index"],
                "job_id": row["record"]["job_id"],
                "label": row["object_row"]["label"],
                "policy_id": policy_id,
                "proposal_id": row["record"]["selected_proposal_id"],
                "registration_initial_objective_m": alignment[
                    "registration_initial_objective_m"
                ],
                "registration_median_m": alignment["registration_median_m"],
                "rejected": None,
                "residual_tilt_deg": alignment["residual_tilt_deg"],
                "scale": alignment["scale"],
                "scale_ratio_vs_observation": alignment[
                    "scale_ratio_vs_observation"
                ],
                "schema_version": SCHEMA_VERSION,
                "source_up_hypothesis": alignment["source_up_hypothesis"],
                "symmetric_clipped_registration_residual_m": alignment[
                    "symmetric_clipped_registration_residual_m"
                ],
                "terminal_action": "accept",
                "tier": source["report_tiers"][slot],
                "world_dims": alignment["world_dims"],
            }
            for role, spec in row["artifacts"].items():
                expected_member_identities[
                    f"objects/{slot}/{ROLE_OUTPUTS[role]}"
                ] = (spec["size_bytes"], spec["sha256"])
        aligned_bytes = _json_bytes(aligned)
        expected_member_identities[f"objects/{slot}/aligned.json"] = (
            len(aligned_bytes),
            hashlib.sha256(aligned_bytes).hexdigest(),
        )
    observed_member_identities = {
        relative: (identity["size_bytes"], identity["sha256"])
        for relative, identity in output_members.items()
    }
    if observed_member_identities != expected_member_identities:
        raise ValueError("materialization output inventory differs from sealed E3 selections")
    return {
        "code_root": str(CODE_ROOT),
        "e3_claim_status": dict(aggregate["claim_status"]),
        "e3_code_commit": manifest["e3_code_commit"],
        "e3_producer_commit": provenance["e3_producer_commit"],
        "e3_freeze_id": manifest["e3_freeze_id"],
        "e3_root": e3_root_raw,
        "evidence_root": str(root),
        "factory_dir": str(factory),
        "input_identities_sha256": hashlib.sha256(
            canonical_json(manifest["input_identities"]).encode("utf-8")
        ).hexdigest(),
        "manifest_sha256": expected_digest,
        "materializer_commit": provenance["materializer_commit"],
        "policy_id": policy_id,
        "provenance": provenance,
        "roster": dict(roster),
        "scene_id": scene_id,
        "source_scene_sha256": hashlib.sha256(
            canonical_json(manifest["source_scene"]).encode("utf-8")
        ).hexdigest(),
        "study_scope": manifest["study_scope"],
        "validator_commit": validator_snapshot["commit"],
    }


def materialize_factory_variant(
    *, e3_root: str | Path, scene_id: str, policy_id: str, out: str | Path,
    automatic_scene_contract: str | Path | None = None,
) -> dict[str, Any]:
    """Verify and atomically materialize one exact E3 policy/scene roster."""
    code_snapshot = _require_clean_code_snapshot()
    if not isinstance(scene_id, str) or not SCENE_ID_RE.fullmatch(scene_id):
        raise ValueError(f"invalid scene ID: {scene_id!r}")
    if policy_id not in ALLOWED_POLICIES:
        raise ValueError(f"policy must be one of {sorted(ALLOWED_POLICIES)}")
    e3_root_path = e3.checked_repo_path(e3_root, "E3 aggregate root", kind="dir")
    destination = e3.checked_repo_path(out, "materialized factory output", must_exist=False)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite materialized factory output: {destination}")
    try:
        destination.relative_to(e3_root_path)
    except ValueError:
        pass
    else:
        raise ValueError("materialized output must not be nested inside its E3 input")
    try:
        e3_root_path.relative_to(destination)
    except ValueError:
        pass
    else:
        raise ValueError("materialized output must not contain its E3 input")

    if automatic_scene_contract is not None:
        return _materialize_automatic_variant(e3_root_path, scene_id, policy_id, destination,
                                              Path(automatic_scene_contract), code_snapshot)

    _, aggregate = _verify_aggregate(e3_root_path)
    jobs, _proposals, audit, inventory = _verify_inventory(e3_root_path)
    if (
        aggregate["claim_status"]["freeze_id"] != jobs.get("freeze_id")
        or aggregate["claim_status"]["study_scope"] != jobs.get("study_scope")
    ):
        raise ValueError("E3 aggregate claim status differs from the sealed inventory")
    source = _scene_source_context(jobs, audit, scene_id)
    prepared = _prepare_selected_records(
        e3_root=e3_root_path,
        scene_id=scene_id,
        policy_id=policy_id,
        freeze_id=jobs["freeze_id"],
        proposals=_proposals,
        source=source,
        aggregate=aggregate,
    )
    accepted = [row for row in prepared if row["action"] == "accept"]
    abstained = [row for row in prepared if row["action"] == "abstain"]
    if policy_id == "A0" and abstained:
        raise ValueError("A0 materialization cannot contain abstentions")

    with e3._atomic_directory(destination) as staging:
        members: dict[str, dict[str, Any]] = {}
        objects_path = staging / "objects" / "objects.json"
        identity = _write_inside(objects_path, _json_bytes(source["object_rows"]))
        _record_member(members, staging, objects_path, identity)

        selected_manifest_rows: list[dict[str, Any]] = []
        for row in prepared:
            slot = row["record"]["object_slot"]
            object_dir = staging / "objects" / slot
            selected_out = object_dir / "selected_asset.json"
            selected_identity = _write_inside(selected_out, row["record_bytes"])
            if (
                selected_identity["size_bytes"] != row["record_identity"]["size_bytes"]
                or selected_identity["sha256"] != row["record_identity"]["sha256"]
            ):
                raise ValueError(f"selected record changed while materializing: {slot}")
            _record_member(members, staging, selected_out, selected_identity)

            if row["action"] == "abstain":
                aligned = {
                    "index": row["object_row"]["index"],
                    "job_id": row["record"]["job_id"],
                    "label": row["object_row"]["label"],
                    "policy_id": policy_id,
                    "reason_codes": row["record"]["reason_codes"],
                    "rejected": "e3_policy_abstention",
                    "schema_version": SCHEMA_VERSION,
                    "terminal_action": "abstain",
                }
            else:
                alignment = row["alignment"]
                aligned = {
                    "T": alignment["T"],
                    "index": row["object_row"]["index"],
                    "job_id": row["record"]["job_id"],
                    "label": row["object_row"]["label"],
                    "policy_id": policy_id,
                    "proposal_id": row["record"]["selected_proposal_id"],
                    "registration_initial_objective_m": alignment[
                        "registration_initial_objective_m"
                    ],
                    "registration_median_m": alignment["registration_median_m"],
                    "rejected": None,
                    "residual_tilt_deg": alignment["residual_tilt_deg"],
                    "scale": alignment["scale"],
                    "scale_ratio_vs_observation": alignment[
                        "scale_ratio_vs_observation"
                    ],
                    "schema_version": SCHEMA_VERSION,
                    "source_up_hypothesis": alignment["source_up_hypothesis"],
                    "symmetric_clipped_registration_residual_m": alignment[
                        "symmetric_clipped_registration_residual_m"
                    ],
                    "terminal_action": "accept",
                    "tier": source["report_tiers"][slot],
                    "world_dims": alignment["world_dims"],
                }
            aligned_path = object_dir / "aligned.json"
            aligned_identity = _write_inside(aligned_path, _json_bytes(aligned))
            _record_member(members, staging, aligned_path, aligned_identity)

            for role, spec in sorted(row["artifacts"].items()):
                artifact_out = object_dir / ROLE_OUTPUTS[role]
                if spec["captured"] is not None:
                    artifact_identity = _write_inside(artifact_out, spec["captured"])
                    if (
                        artifact_identity["size_bytes"] != spec["size_bytes"]
                        or artifact_identity["sha256"] != spec["sha256"]
                    ):
                        raise ValueError(f"captured selected artifact changed: {slot}/{role}")
                else:
                    artifact_identity = _copy_verified_inside(
                        spec["path"], artifact_out, spec["size_bytes"], spec["sha256"]
                    )
                _record_member(members, staging, artifact_out, artifact_identity)

            selected_manifest_rows.append(
                {
                    "job_id": row["record"]["job_id"],
                    "object_slot": slot,
                    "selected_proposal_id": row["record"]["selected_proposal_id"],
                    "selected_record": dict(row["record_identity"]),
                    "terminal_action": row["action"],
                }
            )

        roster_rows = [
            {
                "job_id": row["record"]["job_id"],
                "object_slot": row["record"]["object_slot"],
                "terminal_action": row["action"],
            }
            for row in prepared
        ]
        manifest = {
            "created_utc": _utc_now(),
            "destination": _display(destination),
            "e3_claim_status": aggregate["claim_status"],
            "e3_code_commit": jobs["source_contract"]["code_commit"],
            "e3_freeze_id": jobs["freeze_id"],
            "e3_root": _display(e3_root_path),
            "input_identities": {
                **inventory,
                "aggregate_seal": aggregate["aggregate_seal"],
            },
            "manifest_kind": "e4_construction_variant_materialization",
            "output_members": dict(sorted(members.items())),
            "policy_id": policy_id,
            "provenance": _provenance_record(
                e3_producer_commit=jobs["source_contract"]["code_commit"],
                code_snapshot=code_snapshot,
            ),
            "roster": {
                "abstained_count": len(abstained),
                "abstained_slots": [row["record"]["object_slot"] for row in abstained],
                "accepted_count": len(accepted),
                "accepted_slots": [row["record"]["object_slot"] for row in accepted],
                "job_count": len(prepared),
                "object_slots": list(source["object_slots"]),
                "sha256": hashlib.sha256(canonical_json(roster_rows).encode("utf-8")).hexdigest(),
            },
            "scene_id": scene_id,
            "schema_version": SCHEMA_VERSION,
            "selected_records": selected_manifest_rows,
            "source_scene": {
                "camera_artifacts": source["camera_artifacts"],
                "external_source_bytes_opened": False,
                "factory_objects": source["source_objects"],
                "factory_report": source["report"],
                "object_metadata": source["meta_identities"],
                "prehybrid_audit": source["prehybrid"],
                "rgba_inputs": source["rgba_identities"],
                "source_factory": source["source_factory"],
                "source_identity_manifest": source["source_manifest"],
                "source_scene_gaussian": source["source_scene_gaussian"],
            },
            "study_scope": jobs["study_scope"],
        }
        manifest_path = staging / "materialization_manifest.json"
        manifest_identity = _write_inside(manifest_path, _json_bytes(manifest))
        seal = {
            "manifest_kind": "e4_construction_variant_materialization",
            "members": {"materialization_manifest.json": manifest_identity},
            "schema_version": SCHEMA_VERSION,
        }
        _write_inside(staging / "seal.json", _json_bytes(seal))
        _fsync_tree(staging)
    return manifest


def _automatic_scene_audit(jobs, audit, scene_id):
    """Select one scene without losing the sealed complete-population boundary."""
    scenes = jobs.get("scenes", [])
    scene_ids = [scene.get("scene_id") for scene in scenes]
    if scene_id not in scene_ids:
        raise ValueError("automatic requested scene is absent from sealed inventory")
    if "scene_audits" not in audit:
        if len(scenes) != 1:
            raise ValueError("automatic multi-scene inventory lacks per-scene audits")
        return audit
    scene_audits = audit["scene_audits"]
    expected_counts = {"scenes": len(scenes),
                       "jobs": sum(len(scene["jobs"]) for scene in scenes),
                       "policy_object_rows": 5 * sum(len(scene["jobs"]) for scene in scenes)}
    if (audit.get("schema_version") != 2 or audit.get("freeze_id") != jobs.get("freeze_id")
            or audit.get("paper_ready") is not False
            or audit.get("scene_roster") != scene_ids
            or audit.get("counts") != expected_counts or jobs.get("counts") != expected_counts
            or not isinstance(scene_audits, Mapping) or set(scene_audits) != set(scene_ids)
            or "source_discovery" in audit or "jobs" in audit):
        raise ValueError("automatic per-scene audit population binding differs")
    provenances = set()
    for scene in scenes:
        entry = scene_audits[scene["scene_id"]]
        if (not isinstance(entry, Mapping) or entry.get("schema_version") != 2
                or entry.get("freeze_id") != jobs.get("freeze_id")
                or entry.get("paper_ready") is not False
                or not isinstance(entry.get("jobs"), list)
                or any(not isinstance(row, Mapping) for row in entry["jobs"])
                or [row.get("job_id") for row in entry["jobs"]] != [job["job_id"] for job in scene["jobs"]]):
            raise ValueError("automatic per-scene audit job roster differs")
        for mapping, job in zip(entry["jobs"], scene["jobs"], strict=True):
            automatic_id = int(job["object_slot"][4:])
            if (mapping.get("automatic_instance_id") != automatic_id
                    or mapping.get("source_job_id") != f"{scene['scene_id']}:auto:{automatic_id}"):
                raise ValueError("automatic per-scene audit object identity differs")
        provenance = entry.get("source_gaussian_training_provenance")
        if provenance not in {"UNKNOWN", "FRESH_OFFICIAL_TRAIN_ONLY"}:
            raise ValueError("automatic per-scene source provenance is invalid")
        provenances.add(provenance)
    expected_provenance = next(iter(provenances)) if len(provenances) == 1 else "MIXED"
    if audit.get("source_gaussian_training_provenance") != expected_provenance:
        raise ValueError("automatic population provenance differs from scene audits")
    return scene_audits[scene_id]


def _automatic_source_context(jobs, audit, scene_id, descriptor_path):
    """Authenticate automatic metadata separately from controller-safe inputs."""
    from agents.core import common

    audit = _automatic_scene_audit(jobs, audit, scene_id)
    descriptor_path = e3.checked_repo_path(descriptor_path, "automatic scene descriptor", kind="file")
    _, descriptor = _read_json_path(descriptor_path, "automatic scene descriptor")
    _exact_mapping(descriptor, {"schema_version", "scene_id", "discovery_directory", "discovery_hashes"}, "automatic scene descriptor")
    if (descriptor["schema_version"] != 1 or descriptor["scene_id"] != scene_id
            or jobs.get("schema_version") != 2
            or jobs.get("study_scope") != "automatic_training_only_engineering"
            or audit.get("paper_ready") is not False
            or audit.get("source_discovery") != descriptor["discovery_hashes"]):
        raise ValueError("automatic source scope/scene/hash binding differs")
    directory = e3.checked_repo_path(descriptor["discovery_directory"], "automatic discovery", kind="dir")
    names = {"pilot_summary.json", "input_manifest.json", "output_hashes.json", "postrun_audit.json"}
    if "all_jobs_manifest.json" in descriptor["discovery_hashes"]:
        names.add("all_jobs_manifest.json")
    if set(descriptor["discovery_hashes"]) != names:
        raise ValueError("automatic discovery closure incomplete")
    for name,digest in descriptor["discovery_hashes"].items():
        _verify_digest_member(directory/name,digest,"automatic discovery member")
    summary = json.loads((directory/'pilot_summary.json').read_text())
    manifest = json.loads((directory/'input_manifest.json').read_text())
    output_index = json.loads((directory/'output_hashes.json').read_text())
    postrun = json.loads((directory/'postrun_audit.json').read_text())
    provenance = manifest.get('source_gaussian_training_provenance', 'UNKNOWN')
    if (provenance != summary.get('source_gaussian_training_provenance', 'UNKNOWN')
            or provenance != audit.get('source_gaussian_training_provenance')
            or provenance not in {'UNKNOWN', 'FRESH_OFFICIAL_TRAIN_ONLY'}):
        raise ValueError("automatic Gaussian provenance differs across source closures")
    if (postrun['summary_sha256'] != sha256_file(directory/'pilot_summary.json')
            or postrun['output_hashes_sha256'] != sha256_file(directory/'output_hashes.json')):
        raise ValueError("automatic discovery run is not complete")
    if provenance == 'FRESH_OFFICIAL_TRAIN_ONLY':
        if 'all_jobs_manifest.json' not in names or manifest.get('scene_id') != scene_id:
            raise ValueError("fresh automatic source requires its complete all-jobs scene anchor")
        # Reuse the canonical fresh producer validator: all-jobs/postrun seals,
        # full segmentation population, stage roster and allowed training crops.
        from run.icra2027.e3_trellis_generation_pilot import source_jobs
        complete_jobs = source_jobs(directory)
        if any(row['job_id'] != f"{scene_id}:auto:{row['automatic_instance_id']}" for row in complete_jobs):
            raise ValueError("fresh automatic source job scene differs")
    elif 'all_jobs_manifest.json' in names or any(stage.get('exit_code') != 0 for stage in summary['stages']):
        raise ValueError("legacy automatic source has unexpected completion schema")
    scene = e3._scene_inventory(jobs,scene_id)
    expected_sources = {'gaussian':scene['source_scene_gaussian'], **scene['camera_artifacts']}
    observed_sources = {'gaussian':manifest['gaussian'],
        'intrinsics':manifest['metadata']['nerfstudio/transforms_undistorted.json'],
        'poses':manifest['metadata']['colmap/images.txt']}
    for role,identity in expected_sources.items():
        original = observed_sources[role]
        if (identity['path'],identity['sha256'],identity['size_bytes']) != (original['path'],original['sha256'],original['bytes']):
            raise ValueError("automatic scene sources differ from controller inventory")
    source_files = {}
    for name in ('derived_mesh.ply','auto_instances.npz','objects/objects.json'):
        identity = output_index[name]
        path = e3.checked_repo_path(directory/'construction'/name, name, kind='file')
        if (str(path) != identity['path'] or path.stat().st_size != identity['bytes']
                or sha256_file(path) != identity['sha256']):
            raise ValueError("automatic derived mesh/segmentation/objects identity differs")
        source_files[name] = _identity(path)
    # Explicit automatic loader never dispatches to load_gt_instances.
    instances = common.load_auto_instances(
        instances_path=directory/'construction/auto_instances.npz',
        mesh_path=directory/'construction/derived_mesh.ply')
    by_id = {int(row['object_id']):row for row in instances}
    job_rows = sorted(scene['jobs'],key=lambda row:int(row['object_slot'][4:]))
    expected_ids = {int(row['object_slot'][4:]) for row in job_rows}
    if len(by_id) != len(instances) or set(by_id) != expected_ids:
        raise ValueError("automatic instance and planned-job denominators differ")
    summary_rows = {row['automatic_instance_id']:row for row in summary['rows']}
    if len(summary_rows) != len(summary['rows']) or set(summary_rows) != expected_ids:
        raise ValueError("automatic summary lost or duplicated jobs")
    raw_objects = _indexed_rows(json.loads((directory/'construction/objects/objects.json').read_text()),'automatic prepared objects')
    mappings = {row['job_id']:row for row in audit['jobs']}
    if len(mappings) != len(audit['jobs']) or set(mappings) != {row['job_id'] for row in job_rows}:
        raise ValueError("automatic identity audit denominator differs")
    prepared_indices=[];object_rows=[]
    for job in job_rows:
        slot=job['object_slot'];index=int(slot[4:]);original=by_id[index];mapping=mappings[job['job_id']]
        if (slot != f'obj_{index}' or job['job_id'] != f'{scene_id}/{slot}'
                or mapping['automatic_instance_id'] != index
                or mapping['source_job_id'] != f'{scene_id}:auto:{index}'):
            raise ValueError("automatic stable object identity differs")
        prepared = not e3._unavailable_observation(job)
        source_row=summary_rows[index]
        if prepared != source_row['prepared'] or mapping['prepared_output_index'] != source_row.get('output_index'):
            raise ValueError("automatic prepared-index mapping differs")
        if prepared:
            output_index_value=mapping['prepared_output_index'];prepared_indices.append(output_index_value)
            meta=raw_objects[output_index_value]
            # This old producer key is an automatic namespace alias, not a GT lookup.
            if meta.get('gt_object_id') != index or meta['label'] != original['label']:
                raise ValueError("automatic prepared object identity/label differs")
            rgba=directory/f'construction/objects/obj_{output_index_value:02d}/rgba.png'
            if e3.checked_repo_path(job['artifact_paths']['rgba'],'automatic RGBA',kind='file') != rgba or sha256_file(rgba) != job['artifact_hashes']['rgba']:
                raise ValueError("automatic crop no longer matches prepared mapping")
        aabb=original['aabb'].tolist();_validate_aabb(aabb,'automatic AABB')
        object_rows.append({'index':index,'automatic_instance_id':index,'instance_namespace':'automatic',
            'prepared_output_index':mapping['prepared_output_index'],'label':original['label'],
            'aabb':aabb,'centroid':original['centroid'].tolist(),
            'observation_status':job['construction_evidence']['observation_status']})
    if len(prepared_indices) != len(set(prepared_indices)) or set(prepared_indices) != set(raw_objects):
        raise ValueError("automatic prepared-index mapping is not a bijection")
    return {'automatic':True,'jobs':job_rows,'object_slots':[row['object_slot'] for row in job_rows],
        'object_rows':object_rows,'source_files':source_files,'descriptor':_identity(descriptor_path),
        'source_descriptor':descriptor,'source_gaussian_training_provenance':audit['source_gaussian_training_provenance']}


def _automatic_products(e3_root, scene_id, policy_id, descriptor_path):
    """Reuse canonical selected-record validation without evaluation aggregation."""
    if policy_id not in ALLOWED_POLICIES:
        raise ValueError("automatic materialization requires A0 or A4")
    jobs,proposals,audit,inventory=_verify_inventory(e3_root)
    source=_automatic_source_context(jobs,audit,scene_id,descriptor_path)
    control,shard,seal=e3._load_control_scene(e3_root,scene_id)
    _, observation_manifest, observations=e3._load_observation_scene(e3_root,scene_id)
    expected_ids={row['job_id'] for row in source['jobs']}
    scene = e3._scene_inventory(jobs, scene_id)
    expected_observation_identity = {
        'freeze_id': jobs['freeze_id'],
        'code_commit': jobs['source_contract']['code_commit'],
        'source_scene_gaussian': scene['source_scene_gaussian'],
        'camera_artifacts': scene['camera_artifacts'],
        'observation_protocol': jobs['observation_protocol'],
    }
    if any(observation_manifest.get(key) != value for key, value in expected_observation_identity.items()):
        raise ValueError("automatic observation source identity differs from inventory")
    if (shard['freeze_id'] != jobs['freeze_id'] or shard['job_count'] != len(expected_ids)
            or shard['ledger_row_count'] != 5*len(expected_ids)
            or set(observations) != expected_ids
            or shard['observation_manifest_sha256'] != sha256_file(e3_root/'observations'/scene_id/'manifest.json')):
        raise ValueError("automatic control/observation denominator or freeze differs")
    if (shard.get('selected_asset_hashes') != seal.get('selected_asset_hashes')
            or shard.get('ledger_sha256') != sha256_file(control/'job_ledger.jsonl')):
        raise ValueError("automatic controller shard differs from selected/ledger closure")
    ledger=[json.loads(line) for line in (control/'job_ledger.jsonl').read_text().splitlines()]
    e3.validate_ledger(ledger)
    if {(row['job_id'],row['policy_id']) for row in ledger} != {(job,policy) for job in expected_ids for policy in e3.POLICY_IDS} or len(ledger)!=5*len(expected_ids):
        raise ValueError("automatic terminal ledger denominator differs")
    ledger_by_cell = {(row['job_id'], row['policy_id']): row for row in ledger}
    selected={}
    for policy in e3.POLICY_IDS:
        folder=control/'selected_assets'/policy
        if {p.name for p in folder.iterdir()} != {row['object_slot']+'.json' for row in source['jobs']}:
            raise ValueError("automatic selected records do not cover all planned jobs")
        for path in folder.iterdir():
            record = json.loads(path.read_text())
            terminal = {key: value for key, value in record.items() if key != 'selected_asset'}
            if terminal != ledger_by_cell.get((record.get('job_id'), policy)):
                raise ValueError("automatic selected decision differs from terminal ledger")
            selected[f'{policy}/{scene_id}/{path.name}']=_identity(path)
    source['selected_scene_root']=str(control/'selected_assets')
    prepared=_prepare_selected_records(e3_root=e3_root,scene_id=scene_id,policy_id=policy_id,
        freeze_id=jobs['freeze_id'],proposals=proposals,source=source,aggregate={'selected_records':selected})
    outputs={'objects/objects.json':{'captured':_json_bytes(source['object_rows'])}}
    for row in prepared:
        slot=row['record']['object_slot'];outputs[f'objects/{slot}/selected_asset.json']={'captured':row['record_bytes']}
        aligned={'schema_version':2,'index':row['object_row']['index'],'job_id':row['record']['job_id'],
                 'label':row['object_row']['label'],'policy_id':policy_id,'terminal_action':row['action'],
                 'construction_eligible':row['action']=='accept','tier':None}
        if row['action']=='accept':
            aligned.update(row['alignment']);aligned.update(rejected=None,proposal_id=row['record']['selected_proposal_id'])
        else:
            aligned.update(rejected='e3_policy_'+row['action'],reason_codes=row['record']['reason_codes'])
        outputs[f'objects/{slot}/aligned.json']={'captured':_json_bytes(aligned)}
        for role,spec in row['artifacts'].items():outputs[f'objects/{slot}/{ROLE_OUTPUTS[role]}']=dict(spec)
    for name in ('derived_mesh.ply','auto_instances.npz'):
        identity=source['source_files'][name]
        outputs[name]={'captured':None,**identity}
    for spec in outputs.values():
        if spec['captured'] is not None:
            spec.update(size_bytes=len(spec['captured']),sha256=hashlib.sha256(spec['captured']).hexdigest())
    partitions={action:[row['record']['object_slot'] for row in prepared if row['action']==action] for action in ('accept','reject','abstain')}
    roster={'job_count':len(prepared),'object_slots':source['object_slots'],
        **{key:value for action,slots in partitions.items() for key,value in ((action+'ed_slots' if action!='abstain' else 'abstained_slots',slots),
        (action+'ed_count' if action!='abstain' else 'abstained_count',len(slots)))}}
    core={'schema_version':2,'manifest_kind':'e4_automatic_construction_variant_materialization',
        'e3_root':_display(e3_root),'e3_freeze_id':jobs['freeze_id'],'e3_code_commit':jobs['source_contract']['code_commit'],
        'scene_id':scene_id,'policy_id':policy_id,'study_scope':jobs['study_scope'],'paper_ready':False,
        'source_scene':{'automatic_scene_descriptor':source['descriptor'], 'discovery_sources':source['source_files'],
                        'source_gaussian_training_provenance':source['source_gaussian_training_provenance']},
        'input_identities':{**inventory,'control_seal':_identity(control/'seal.json'),
                           'observation_seal':_identity(e3_root/'observations'/scene_id/'seal.json')},
        'roster':roster,'selected_records':[{'job_id':r['record']['job_id'],'object_slot':r['record']['object_slot'],
            'terminal_action':r['action'],'selected_record':r['record_identity']} for r in prepared],
        'output_members':{name:{'size_bytes':spec['size_bytes'],'sha256':spec['sha256']} for name,spec in sorted(outputs.items())}}
    return core,outputs


def _materialize_automatic_variant(e3_root,scene_id,policy_id,destination,descriptor_path,snapshot):
    core,outputs=_automatic_products(e3_root,scene_id,policy_id,descriptor_path)
    manifest={**core,'destination':_display(destination),'created_utc':_utc_now(),
              'provenance':_provenance_record(e3_producer_commit=core['e3_code_commit'],code_snapshot=snapshot)}
    with e3._atomic_directory(destination) as staging:
        for name,spec in outputs.items():
            if spec['captured'] is not None:_write_inside(staging/name,spec['captured'])
            else:_copy_verified_inside(e3.checked_repo_path(spec['path'],'automatic source artifact',kind='file'),staging/name,spec['size_bytes'],spec['sha256'])
        identity=_write_inside(staging/'materialization_manifest.json',_json_bytes(manifest))
        _write_inside(staging/'seal.json',_json_bytes({'schema_version':1,'manifest_kind':'e4_construction_variant_materialization',
            'members':{'materialization_manifest.json':identity}}))
        _fsync_tree(staging)
    return manifest


def _validate_automatic_factory(factory,manifest,snapshot,expected_scene,expected_policy):
    if expected_scene is not None and manifest['scene_id'] != expected_scene:raise ValueError('automatic factory scene differs')
    if expected_policy is not None and manifest['policy_id'] != expected_policy:raise ValueError('automatic factory policy differs')
    descriptor=manifest['source_scene']['automatic_scene_descriptor']
    path=e3.checked_repo_path(descriptor['path'],'automatic descriptor',kind='file')
    if _identity(path) != descriptor:raise ValueError('automatic descriptor changed')
    core,_=_automatic_products(e3.checked_repo_path(manifest['e3_root'],'automatic E3 source',kind='dir'),
        manifest['scene_id'],manifest['policy_id'],path)
    if {k:v for k,v in manifest.items() if k not in {'created_utc','provenance','destination'}} != core:
        raise ValueError('automatic materialization provenance or outputs differ')
    if manifest['destination'] != _display(factory):raise ValueError('automatic output destination differs')
    provenance=_validate_provenance_record(manifest['provenance'],e3_producer_commit=core['e3_code_commit'],
        evidence_root=REPOSITORY_ROOT,validator_snapshot=snapshot)
    actual={str(path.relative_to(factory)) for path in (factory/'objects').rglob('*') if path.is_file()}
    expected={name for name in core['output_members'] if name.startswith('objects/')}
    if actual != expected:raise ValueError('automatic objects subtree differs')
    for name,identity in core['output_members'].items():
        path=e3.checked_repo_path(factory/name,'automatic output member',kind='file')
        if path.stat().st_size!=identity['size_bytes'] or sha256_file(path)!=identity['sha256']:
            raise ValueError('automatic materialized member changed')
    return {'scene_id':core['scene_id'],'policy_id':core['policy_id'],'roster':core['roster'],
            'manifest_sha256':sha256_file(factory/'materialization_manifest.json'),
            'e3_producer_commit':core['e3_code_commit'],'code_root':str(CODE_ROOT),
            'materializer_commit':provenance['materializer_commit'],'validator_commit':snapshot['commit'],
            'automatic_scene_descriptor':descriptor,'paper_ready':False}



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize one sealed E3 A0/A4 selection into a fresh factory tree."
    )
    parser.add_argument("--e3-root", required=True, help="repository-local E3 agentic aggregate")
    parser.add_argument("--automatic-scene-contract", help="hash-bound automatic discovery descriptor")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--policy-id", required=True, choices=sorted(ALLOWED_POLICIES))
    parser.add_argument("--out", required=True, help="fresh repository-local output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = materialize_factory_variant(
            e3_root=args.e3_root,
            scene_id=args.scene_id,
            policy_id=args.policy_id,
            out=args.out,
            automatic_scene_contract=args.automatic_scene_contract,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(f"materialization failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "out": manifest["destination"],
                "policy_id": manifest["policy_id"],
                "roster": manifest["roster"],
                "scene_id": manifest["scene_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
