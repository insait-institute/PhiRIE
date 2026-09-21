"""Build leakage-free replacement object bundles for the E2 fidelity export.

The legacy object builds are immutable inputs.  For one remediation bundle this
tool re-runs the existing factory prepare -> mask refinement -> TRELLIS ->
alignment path in a fresh staging directory, while restricting factory_prepare
to the official DSLR *train* roster.  A bundle is published with one atomic
rename only after every pinned object was recovered by ``gt_object_id``, kept
its legacy output index, and received an accepted regenerated alignment.

The fixed five-bundle/eight-object remediation contract lives inside the E2
fidelity manifest so it is covered by the normal E0 config freeze.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SCENE_RE = re.compile(r"[0-9a-f]{10}")
FREEZE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
EXPECTED_OUTPUT_ROOT = "outputs/icra2027/{freeze_id}/fidelity/replacements"

# Exact reconstruction policy recorded in every replacement manifest and
# required again on resume.  These values deliberately mirror the production
# factory stages rather than being inferred from whatever code happens to be
# installed when an old bundle is validated.
EXPECTED_STAGE_PARAMETERS: dict[str, Any] = {
    "factory_prepare": {
        "min_bbox_px": 48,
        "visibility_tolerance_m": 0.02,
        "min_mask_px": 400,
        "full_vocabulary": False,
        "gt_surface_sample_seed": 42,
    },
    "trellis": {
        "attention_backend": "xformers",
        "sparse_convolution_algorithm": "native",
    },
    "factory_align": {
        "policy": "signed_source_up_v1",
        "source_up_hypotheses_order": ["+z", "-z", "+x", "-x", "+y", "-y"],
        "yaw_step_deg": 10,
        "cross_hypothesis_tie_epsilon_m": 1e-9,
        "icp_distance_m": 0.03,
        "tilt_snap_deg": 15.0,
        "alignment_mesh_sample_seed": 42,
        "size_ratio_range": [0.4, 2.5],
        "tier_a_f1_20": 0.40,
        "tier_b_f1_40": 0.20,
    },
}

# Duplicating these eight identifiers here is deliberate: config drift must not
# silently change which known leaked single-view inputs are remediated.
EXPECTED_BUNDLES: dict[str, dict[str, Any]] = {
    "0d2ee665be-auto": {
        "scene_id": "0d2ee665be",
        "mode": "auto",
        "legacy_source": "outputs/0d2ee665be_auto",
        "targets": (
            (6, 1007, "bottle", "DSC00152.JPG"),
            (26, 1028, "bottle", "DSC00155.JPG"),
        ),
    },
    "0d2ee665be-factory": {
        "scene_id": "0d2ee665be",
        "mode": "factory",
        "legacy_source": "outputs/0d2ee665be_factory",
        "targets": ((4, 16, "mug", "DSC00154.JPG"),),
    },
    "286b55a2bf-auto": {
        "scene_id": "286b55a2bf",
        "mode": "auto",
        "legacy_source": "outputs/286b55a2bf_auto",
        "targets": (
            (2, 1002, "bottle", "DSC02539.JPG"),
            (10, 1010, "bottle", "DSC02538.JPG"),
            (27, 1027, "bottle", "DSC02542.JPG"),
        ),
    },
    "286b55a2bf-factory": {
        "scene_id": "286b55a2bf",
        "mode": "factory",
        "legacy_source": "outputs/286b55a2bf_factory",
        "targets": ((9, 24, "bottle", "DSC02539.JPG"),),
    },
    "3f15a9266d-factory": {
        "scene_id": "3f15a9266d",
        "mode": "factory",
        "legacy_source": "outputs/3f15a9266d_factory",
        "targets": ((193, 232, "book", "DSC07489.JPG"),),
    },
}

StageRunner = Callable[[str, list[str], dict[str, str], Path, Path], None]
Sam3RuntimeProbe = Callable[[str, dict[str, str], Path], dict[str, Any]]


class ReplacementError(RuntimeError):
    """The replacement cannot be built without weakening the E2 contract."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplacementError(f"cannot read valid {label} JSON: {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _repo_path(
    value: str | os.PathLike[str],
    *,
    root: Path,
    label: str,
    must_exist: bool,
) -> Path:
    """Resolve a lexical repository path while rejecting every symlink hop."""
    root = Path(os.path.abspath(root))
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ReplacementError(f"{label} must stay inside {root}: {candidate}") from exc
    current = root
    if current.is_symlink():
        raise ReplacementError(f"repository root is a symlink: {current}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ReplacementError(f"{label} uses a symlink component: {current}")
    if must_exist and not candidate.exists():
        raise ReplacementError(f"missing {label}: {candidate}")
    return candidate


def _regular_file(path: Path, label: str, *, nonempty: bool = True) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReplacementError(f"{label} must be a regular non-symlink file: {path}")
    if nonempty and path.stat().st_size <= 0:
        raise ReplacementError(f"{label} is empty: {path}")
    return path


def _plain_frame_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ReplacementError(f"{label} must be a non-empty list")
    frames: list[str] = []
    for frame in value:
        if (
            not isinstance(frame, str)
            or not frame
            or frame in {".", ".."}
            or Path(frame).name != frame
            or "/" in frame
            or "\\" in frame
        ):
            raise ReplacementError(f"{label} contains a non-plain frame name: {frame!r}")
        frames.append(frame)
    if len(frames) != len(set(frames)):
        raise ReplacementError(f"{label} contains duplicate frame names")
    return frames


def _artifact(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    _regular_file(path, "artifact")
    if root is not None:
        try:
            display = str(path.relative_to(root))
        except ValueError:
            display = str(path)
    else:
        display = str(path)
    return {
        "path": display,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_model_input_schema(remediation: dict[str, Any]) -> dict[str, Any]:
    model_inputs = remediation.get("model_inputs")
    expected_names = {
        "trellis_source", "trellis_snapshot", "dinov2_source",
        "dinov2_checkpoint", "sam3_source", "sam3_checkpoint",
    }
    if not isinstance(model_inputs, dict) or set(model_inputs) != expected_names:
        raise ReplacementError(
            "leakage_remediation.model_inputs must pin TRELLIS, DINOv2 source/weight, and SAM3"
        )
    for name in (
        "trellis_source", "trellis_snapshot", "dinov2_source", "sam3_source",
    ):
        record = model_inputs.get(name)
        if not isinstance(record, dict):
            raise ReplacementError(f"model_inputs.{name} must be an object")
        path = record.get("path")
        count = record.get("file_count")
        digest = str(record.get("tree_sha256", "")).lower()
        if (
            not isinstance(path, str) or not path or Path(path).is_absolute()
            or ".." in Path(path).parts
            or isinstance(count, bool) or not isinstance(count, int) or count <= 0
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise ReplacementError(f"model_inputs.{name} tree contract is invalid")
        if name in {"trellis_source", "sam3_source"}:
            if (
                not isinstance(record.get("upstream_repository"), str)
                or not record["upstream_repository"]
                or COMMIT_RE.fullmatch(
                    str(record.get("upstream_commit", "")).lower()
                ) is None
            ):
                raise ReplacementError(
                    f"model_inputs.{name} upstream provenance is invalid"
                )
            if name == "trellis_source" and COMMIT_RE.fullmatch(
                str(record.get("flexicubes_commit", "")).lower()
            ) is None:
                raise ReplacementError(
                    "model_inputs.trellis_source FlexiCubes provenance is invalid"
                )
            if name == "sam3_source" and (
                not isinstance(record.get("package_version"), str)
                or not record["package_version"]
            ):
                raise ReplacementError(
                    "model_inputs.sam3_source package version is invalid"
                )
        elif name == "dinov2_source":
            upstream_commit = record.get("upstream_commit")
            if upstream_commit is not None and (
                not isinstance(upstream_commit, str)
                or COMMIT_RE.fullmatch(upstream_commit.lower()) is None
            ):
                raise ReplacementError("model_inputs.dinov2_source upstream_commit is invalid")
            upstream_repository = record.get("upstream_repository")
            if upstream_repository is not None and (
                not isinstance(upstream_repository, str) or not upstream_repository
            ):
                raise ReplacementError("model_inputs.dinov2_source upstream_repository is invalid")
            if upstream_commit is None and (
                not isinstance(record.get("upstream_commit_reason"), str)
                or not record["upstream_commit_reason"]
            ):
                raise ReplacementError(
                    "model_inputs.dinov2_source must explain its unavailable commit"
                )
        elif COMMIT_RE.fullmatch(str(record.get("upstream_revision", "")).lower()) is None:
            raise ReplacementError("model_inputs.trellis_snapshot upstream_revision is invalid")
    for name in ("dinov2_checkpoint", "sam3_checkpoint"):
        record = model_inputs.get(name)
        if not isinstance(record, dict):
            raise ReplacementError(f"model_inputs.{name} must be an object")
        path = record.get("path")
        size = record.get("size_bytes")
        digest = str(record.get("sha256", "")).lower()
        if (
            not isinstance(path, str) or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or isinstance(size, bool) or not isinstance(size, int) or size <= 0
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise ReplacementError(f"model_inputs.{name} file contract is invalid")
        if name == "sam3_checkpoint" and COMMIT_RE.fullmatch(
            str(record.get("upstream_revision", "")).lower()
        ) is None:
            raise ReplacementError("model_inputs.sam3_checkpoint upstream_revision is invalid")
    return json.loads(json.dumps(model_inputs, allow_nan=False))


def _tree_inventory(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise ReplacementError(f"{label} must be a regular non-symlink directory: {path}")
    files: list[tuple[str, Path]] = []
    for current_text, directories, names in os.walk(path, followlinks=False):
        current = Path(current_text)
        for directory in directories:
            child = current / directory
            if child.is_symlink() or not child.is_dir():
                raise ReplacementError(f"{label} contains an unsafe directory: {child}")
        for name in names:
            child = current / name
            if child.is_symlink() or not child.is_file():
                raise ReplacementError(f"{label} contains a non-regular file: {child}")
            files.append((child.relative_to(path).as_posix(), child))
    files.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    records = []
    for relative, child in files:
        size = child.stat().st_size
        file_sha = _sha256(child)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
        records.append({
            "relative_path": relative,
            "size_bytes": size,
            "sha256": file_sha,
        })
    return {
        "file_count": len(records),
        "tree_sha256": digest.hexdigest(),
        "files": records,
    }


def _validate_model_inputs(
    config: dict[str, Any], *, root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    declared = _validate_model_input_schema(config["leakage_remediation"])
    observed: dict[str, Any] = {}
    resolved: dict[str, Path] = {}
    for name in (
        "trellis_source", "trellis_snapshot", "dinov2_source", "sam3_source",
    ):
        record = declared[name]
        path = _repo_path(
            record["path"], root=root, label=f"{name} model input", must_exist=True
        )
        tree = _tree_inventory(path, label=name)
        if tree["file_count"] != record["file_count"]:
            raise ReplacementError(f"{name} file count differs from the frozen config")
        if tree["tree_sha256"] != record["tree_sha256"]:
            raise ReplacementError(f"{name} tree hash differs from the frozen config")
        observed_record = {
            "path": record["path"],
            **tree,
            "source_identity": f"tree-sha256:{tree['tree_sha256']}",
        }
        if name == "trellis_source":
            observed_record.update({
                "upstream_repository": record["upstream_repository"],
                "upstream_commit": record["upstream_commit"],
                "flexicubes_commit": record["flexicubes_commit"],
            })
        elif name == "sam3_source":
            observed_record.update({
                "upstream_repository": record["upstream_repository"],
                "upstream_commit": record["upstream_commit"],
                "package_version": record["package_version"],
            })
        elif name == "trellis_snapshot":
            observed_record["upstream_revision"] = record["upstream_revision"]
        else:
            observed_record.update({
                "upstream_repository": record.get("upstream_repository"),
                "upstream_commit": record.get("upstream_commit"),
                "upstream_commit_reason": record.get("upstream_commit_reason"),
                "upstream_commit_provenance": (
                "pinned" if record.get("upstream_commit")
                else "unavailable_from_gitless_torch_hub_snapshot"
                ),
            })
        observed[name] = observed_record
        resolved[name] = path
    for name in ("dinov2_checkpoint", "sam3_checkpoint"):
        record = declared[name]
        path = _repo_path(
            record["path"], root=root, label=f"{name} model input", must_exist=True
        )
        _regular_file(path, name)
        size = path.stat().st_size
        digest = _sha256(path)
        if size != record["size_bytes"] or digest != record["sha256"]:
            raise ReplacementError(f"{name} size/hash differs from the frozen config")
        observed[name] = {
            "path": record["path"], "size_bytes": size, "sha256": digest,
            **(
                {"upstream_revision": record.get("upstream_revision")}
                if name == "sam3_checkpoint" else {}
            ),
        }
        resolved[name] = path
    return {"declared": declared, "observed": observed}, resolved


def _probe_sam3_runtime(
    python: str, environment: dict[str, str], source_root: Path,
) -> dict[str, Any]:
    """Resolve SAM3 as the stage interpreter will, without importing its models."""
    program = """
import ast
import importlib.util
import json
from pathlib import Path
spec = importlib.util.find_spec("sam3")
if spec is None or spec.origin is None:
    raise SystemExit("sam3 package cannot be resolved")
source = Path(spec.origin).resolve()
tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
version = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__version__":
                version = ast.literal_eval(node.value)
print(json.dumps({"module_file": str(source), "package_version": version}))
"""
    try:
        process = subprocess.run(
            [python, "-c", program], cwd=source_root, env=environment,
            check=True, capture_output=True, text=True,
        )
        payload = json.loads(process.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ReplacementError(f"cannot resolve pinned SAM3 runtime: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplacementError("SAM3 runtime probe returned a non-object")
    return payload


def _validate_sam3_runtime(
    payload: Any,
    *,
    source_root: Path,
    expected_version: str,
    root: Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ReplacementError("SAM3 runtime resolution must be an object")
    module_value = payload.get("module_file")
    if not isinstance(module_value, str) or not module_value:
        raise ReplacementError("SAM3 runtime resolution lacks module_file")
    module_path = Path(os.path.abspath(module_value))
    try:
        module_path.relative_to(source_root)
    except ValueError as exc:
        raise ReplacementError(
            f"SAM3 resolved outside the frozen source tree: {module_path}"
        ) from exc
    _regular_file(module_path, "resolved SAM3 module")
    if payload.get("package_version") != expected_version:
        raise ReplacementError("resolved SAM3 package version differs from config")
    return {
        "package_version": expected_version,
        "module_file": str(module_path.relative_to(root)),
        "module_size_bytes": module_path.stat().st_size,
        "module_sha256": _sha256(module_path),
    }


def _git_snapshot(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip().lower()
        status_lines = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReplacementError(f"cannot inspect Git snapshot for {root}: {exc}") from exc
    if COMMIT_RE.fullmatch(commit) is None:
        raise ReplacementError(f"Git did not return a full commit SHA: {commit!r}")
    return {"commit": commit, "dirty": bool(status_lines), "status": status_lines}


def _validate_target(raw: Any, *, bundle_id: str, position: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ReplacementError(f"{bundle_id}.targets[{position}] must be an object")
    required = ("legacy_index", "gt_object_id", "label", "old_frame")
    if set(raw) != set(required):
        raise ReplacementError(
            f"{bundle_id}.targets[{position}] fields must be exactly {list(required)}"
        )
    index = raw["legacy_index"]
    object_id = raw["gt_object_id"]
    if (
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        or isinstance(object_id, bool) or not isinstance(object_id, int) or object_id < 0
    ):
        raise ReplacementError(f"{bundle_id} target IDs must be non-negative integers")
    label = raw["label"]
    if not isinstance(label, str) or not label.strip() or label != label.strip():
        raise ReplacementError(f"{bundle_id} target label must be canonical text")
    old_frame = _plain_frame_list([raw["old_frame"]], "old_frame")[0]
    return {
        "legacy_index": index,
        "gt_object_id": object_id,
        "label": label,
        "old_frame": old_frame,
    }


def _load_config(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _read_json(config_path, "E2 fidelity config")
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ReplacementError("E2 fidelity config must have schema_version 2")
    remediation = payload.get("leakage_remediation")
    if not isinstance(remediation, dict) or remediation.get("schema_version") != 1:
        raise ReplacementError("leakage_remediation must have schema_version 1")
    fixed = {
        "official_split_artifact": "dslr/train_test_lists.json",
        "output_root": EXPECTED_OUTPUT_ROOT,
        "require_train_only_generation": True,
        "require_accepted_alignment": True,
    }
    for key, expected in fixed.items():
        if remediation.get(key) != expected:
            raise ReplacementError(f"leakage_remediation.{key} must be {expected!r}")
    _validate_model_input_schema(remediation)
    raw_bundles = remediation.get("bundles")
    if not isinstance(raw_bundles, list):
        raise ReplacementError("leakage_remediation.bundles must be a list")
    bundles: dict[str, dict[str, Any]] = {}
    seen_indices: set[tuple[str, str, int]] = set()
    seen_object_ids: set[tuple[str, str, int]] = set()
    for position, raw in enumerate(raw_bundles):
        if not isinstance(raw, dict):
            raise ReplacementError(f"remediation bundle {position} must be an object")
        if set(raw) != {"bundle_id", "scene_id", "mode", "legacy_source", "targets"}:
            raise ReplacementError(f"remediation bundle {position} has unexpected fields")
        bundle_id = raw.get("bundle_id")
        scene_id = raw.get("scene_id")
        mode = raw.get("mode")
        if not isinstance(bundle_id, str) or bundle_id in bundles:
            raise ReplacementError(f"invalid or duplicate remediation bundle_id: {bundle_id!r}")
        if not isinstance(scene_id, str) or SCENE_RE.fullmatch(scene_id) is None:
            raise ReplacementError(f"invalid remediation scene_id: {scene_id!r}")
        if mode not in {"factory", "auto"} or bundle_id != f"{scene_id}-{mode}":
            raise ReplacementError(f"bundle identity/mode mismatch: {bundle_id!r}")
        canonical_source = f"outputs/{scene_id}_{mode}"
        if raw.get("legacy_source") != canonical_source:
            raise ReplacementError(
                f"{bundle_id}.legacy_source must be {canonical_source!r}"
            )
        targets_raw = raw.get("targets")
        if not isinstance(targets_raw, list) or not targets_raw:
            raise ReplacementError(f"{bundle_id}.targets must be non-empty")
        targets = [
            _validate_target(target, bundle_id=bundle_id, position=index)
            for index, target in enumerate(targets_raw)
        ]
        for target in targets:
            index_key = (scene_id, mode, target["legacy_index"])
            object_key = (scene_id, mode, target["gt_object_id"])
            if index_key in seen_indices or object_key in seen_object_ids:
                raise ReplacementError(f"duplicate remediation target in {bundle_id}")
            seen_indices.add(index_key)
            seen_object_ids.add(object_key)
        bundles[bundle_id] = {
            "bundle_id": bundle_id,
            "scene_id": scene_id,
            "mode": mode,
            "legacy_source": canonical_source,
            "targets": targets,
        }

    if set(bundles) != set(EXPECTED_BUNDLES):
        raise ReplacementError("remediation config must pin the exact five known bundles")
    for bundle_id, expected in EXPECTED_BUNDLES.items():
        observed = bundles[bundle_id]
        observed_targets = tuple(
            (
                target["legacy_index"], target["gt_object_id"],
                target["label"], target["old_frame"],
            )
            for target in observed["targets"]
        )
        if (
            observed["scene_id"] != expected["scene_id"]
            or observed["mode"] != expected["mode"]
            or observed["legacy_source"] != expected["legacy_source"]
            or observed_targets != expected["targets"]
        ):
            raise ReplacementError(f"known remediation target contract drifted: {bundle_id}")
    population = payload.get("population", {})
    scene_ids = population.get("scene_ids") if isinstance(population, dict) else None
    if not isinstance(scene_ids, list) or not {
        bundle["scene_id"] for bundle in bundles.values()
    } <= set(scene_ids):
        raise ReplacementError("remediation scenes are absent from the E2 population")
    return payload, bundles


def _validate_contract(
    contract_path: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    freeze_id: str,
    git: dict[str, Any],
) -> dict[str, Any]:
    contract = _read_json(contract_path, "E0 contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise ReplacementError("E0 contract must have schema_version 1")
    if contract.get("mode") not in {"smoke", "paper"}:
        raise ReplacementError("E0 contract mode must be smoke or paper")
    code = contract.get("code")
    if not isinstance(code, dict) or code.get("dirty") is not False:
        raise ReplacementError("E0 contract must record a clean code snapshot")
    contract_commit = str(code.get("commit", "")).lower()
    if COMMIT_RE.fullmatch(contract_commit) is None or contract_commit != git["commit"]:
        raise ReplacementError("E0 contract commit differs from the replacement code")
    contract_freeze = contract.get("freeze_id")
    if not isinstance(contract_freeze, str) or not contract_freeze \
            or not freeze_id.startswith(contract_freeze):
        raise ReplacementError("replacement freeze_id does not inherit the E0 freeze_id")
    declared = str(contract.get("contract_sha256", "")).lower()
    observed = _canonical_hash({
        key: value for key, value in contract.items()
        if key not in {"created_utc", "environment", "contract_sha256"}
    })
    if SHA256_RE.fullmatch(declared) is None or declared != observed:
        raise ReplacementError("E0 contract_sha256 is invalid")
    records = contract.get("configs")
    matches = [
        record for record in records if isinstance(record, dict)
        and record.get("field") == "fidelity_manifest"
    ] if isinstance(records, list) else []
    if len(matches) != 1:
        raise ReplacementError("E0 contract must freeze one fidelity_manifest")
    record = matches[0]
    if record.get("source_content_sha256") != _sha256(config_path):
        raise ReplacementError("fidelity config bytes differ from the E0 contract")
    if record.get("sha256") != _canonical_hash(config):
        raise ReplacementError("fidelity config semantics differ from the E0 contract")
    return {
        "path": str(contract_path),
        "sha256": _sha256(contract_path),
        "contract_sha256": declared,
        "freeze_id": contract_freeze,
        "mode": contract["mode"],
    }


def _load_split(split_path: Path) -> tuple[list[str], list[str]]:
    _regular_file(split_path, "official train/test split")
    split = _read_json(split_path, "official train/test split")
    if not isinstance(split, dict):
        raise ReplacementError("official train/test split must be an object")
    train = _plain_frame_list(split.get("train"), "official train frames")
    test = _plain_frame_list(split.get("test"), "official test frames")
    overlap = sorted(set(train) & set(test))
    if overlap:
        raise ReplacementError(f"official train/test frames overlap: {overlap[:5]}")
    return train, test


def _legacy_provenance(
    bundle: dict[str, Any], *, root: Path, test_frames: set[str]
) -> tuple[Path, list[dict[str, Any]], list[Path]]:
    source = _repo_path(
        bundle["legacy_source"], root=root, label="legacy source", must_exist=True
    )
    if source.is_symlink() or not source.is_dir():
        raise ReplacementError(f"legacy source must be a non-symlink directory: {source}")
    objects_path = _regular_file(source / "objects" / "objects.json", "legacy objects")
    objects = _read_json(objects_path, "legacy objects")
    if not isinstance(objects, list):
        raise ReplacementError("legacy objects.json must be a list")
    by_index = {
        item.get("index"): item for item in objects
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    artifacts = [_artifact(objects_path, root=root)]
    records: list[dict[str, Any]] = []
    used_paths = [objects_path]
    for target in bundle["targets"]:
        index = target["legacy_index"]
        object_meta = by_index.get(index)
        odir = source / "objects" / f"obj_{index:02d}"
        meta_path = _regular_file(odir / "meta.json", "legacy target meta")
        aligned_path = _regular_file(odir / "aligned.json", "legacy target alignment")
        rgba_path = _regular_file(odir / "rgba.png", "legacy generation image")
        meta = _read_json(meta_path, "legacy target meta")
        aligned = _read_json(aligned_path, "legacy target alignment")
        expected_fields = {
            "index": index,
            "gt_object_id": target["gt_object_id"],
            "label": target["label"],
            "frame": target["old_frame"],
        }
        for field, expected in expected_fields.items():
            if not isinstance(meta, dict) or meta.get(field) != expected:
                raise ReplacementError(
                    f"legacy meta mismatch for {bundle['bundle_id']} index {index}: {field}"
                )
            if not isinstance(object_meta, dict) or object_meta.get(field) != expected:
                raise ReplacementError(
                    f"legacy objects.json mismatch for {bundle['bundle_id']} index {index}: {field}"
                )
        if target["old_frame"] not in test_frames:
            raise ReplacementError(
                f"pinned leaked frame is not official test: {target['old_frame']}"
            )
        if (
            not isinstance(aligned, dict)
            or aligned.get("index") != index
            or aligned.get("rejected")
            or aligned.get("tier") not in {"A", "B"}
        ):
            raise ReplacementError(f"legacy target alignment is not accepted: index {index}")
        target_artifacts = [
            _artifact(path, root=root) for path in (meta_path, aligned_path, rgba_path)
        ]
        artifacts.extend(target_artifacts)
        used_paths.extend((meta_path, aligned_path, rgba_path))
        records.append({
            **target,
            "legacy_artifacts": target_artifacts,
            "legacy_alignment_tier": aligned["tier"],
        })
    return source, records, used_paths


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
    _regular_file(source, "copy source")
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if _sha256(source) != _sha256(destination):
        raise ReplacementError(f"copy hash mismatch: {source} -> {destination}")
    return _artifact(destination)


def _run_subprocess_stage(
    stage: str,
    command: list[str],
    environment: dict[str, str],
    staging: Path,
    repo_root: Path,
) -> None:
    del staging
    print(f"[replacement] stage={stage} module={command[2]}", flush=True)
    try:
        subprocess.run(command, cwd=repo_root, env=environment, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReplacementError(f"replacement stage failed: {stage}: {exc}") from exc


def _prepared_records(
    staging: Path,
    bundle: dict[str, Any],
    *,
    train_frames: set[str],
    test_frames: set[str],
    require_trellis: bool,
    require_alignment: bool,
) -> list[dict[str, Any]]:
    objects_path = _regular_file(staging / "objects" / "objects.json", "new objects")
    objects = _read_json(objects_path, "new objects")
    if not isinstance(objects, list) or len(objects) != len(bundle["targets"]):
        raise ReplacementError("new objects.json does not contain the exact target population")
    by_object_id: dict[int, dict[str, Any]] = {}
    for meta in objects:
        if not isinstance(meta, dict) or isinstance(meta.get("gt_object_id"), bool) \
                or not isinstance(meta.get("gt_object_id"), int):
            raise ReplacementError("new objects.json contains invalid metadata")
        if meta["gt_object_id"] in by_object_id:
            raise ReplacementError("new objects.json repeats a gt_object_id")
        by_object_id[meta["gt_object_id"]] = meta
    expected_dirs = {f"obj_{target['legacy_index']:02d}" for target in bundle["targets"]}
    actual_dirs = {
        path.name for path in (staging / "objects").iterdir()
        if path.is_dir() and path.name.startswith("obj_")
    }
    if actual_dirs != expected_dirs:
        raise ReplacementError("new object directories do not preserve the sparse legacy indices")

    records = []
    for target in bundle["targets"]:
        object_id = target["gt_object_id"]
        meta = by_object_id.get(object_id)
        if meta is None:
            raise ReplacementError(f"train-only preparation did not recover gt_object_id {object_id}")
        expected_fields = {
            "index": target["legacy_index"],
            "gt_object_id": object_id,
            "label": target["label"],
            "gt_surface_sample_seed": EXPECTED_STAGE_PARAMETERS[
                "factory_prepare"
            ]["gt_surface_sample_seed"],
        }
        for field, expected in expected_fields.items():
            if meta.get(field) != expected:
                raise ReplacementError(f"new metadata mismatch for {object_id}: {field}")
        frame = _plain_frame_list([meta.get("frame")], "new optimization frame")[0]
        if frame not in train_frames or frame in test_frames:
            raise ReplacementError(
                f"new generation frame is not train-only for {object_id}: {frame}"
            )
        odir = staging / "objects" / f"obj_{target['legacy_index']:02d}"
        per_meta = _read_json(
            _regular_file(odir / "meta.json", "new target meta"), "new target meta"
        )
        for field in (*expected_fields, "frame"):
            if not isinstance(per_meta, dict) or per_meta.get(field) != meta.get(field):
                raise ReplacementError(f"per-object metadata mismatch for {object_id}: {field}")
        _regular_file(odir / "rgba.png", "new target RGBA")
        _regular_file(odir / "gt_points.ply", "new target GT points")
        if require_trellis:
            for name in (
                "trellis_mesh.ply", "mesh_sim.ply", "mesh_sim.obj", "trellis_gs.ply",
            ):
                _regular_file(odir / name, f"new target {name}")
        alignment = None
        if require_alignment:
            alignment_path = _regular_file(odir / "aligned.json", "new target alignment")
            alignment = _read_json(alignment_path, "new target alignment")
            align_policy = EXPECTED_STAGE_PARAMETERS["factory_align"]
            if (
                not isinstance(alignment, dict)
                or alignment.get("index") != target["legacy_index"]
                or alignment.get("rejected")
                or alignment.get("tier") not in {"A", "B"}
                or alignment.get("source_up_hypothesis")
                not in align_policy["source_up_hypotheses_order"]
                or alignment.get("alignment_mesh_sample_seed")
                != align_policy["alignment_mesh_sample_seed"]
            ):
                raise ReplacementError(
                    "regenerated alignment rejected or lacks exact policy "
                    f"provenance for gt_object_id {object_id}"
                )
        records.append({
            "legacy_index": target["legacy_index"],
            "gt_object_id": object_id,
            "label": target["label"],
            "old_test_frame": target["old_frame"],
            "new_train_frame": frame,
            "optimization_input_frames": [frame],
            "alignment_tier": alignment.get("tier") if alignment else None,
        })
    if require_alignment:
        aligned_all_path = _regular_file(
            staging / "objects" / "aligned_all.json", "new aligned population"
        )
        aligned_all = _read_json(aligned_all_path, "new aligned population")
        if not isinstance(aligned_all, list) or {
            row.get("index") for row in aligned_all if isinstance(row, dict)
        } != {target["legacy_index"] for target in bundle["targets"]} \
                or len(aligned_all) != len(bundle["targets"]):
            raise ReplacementError("aligned_all.json does not contain the exact target population")
        if any(
            not isinstance(row, dict) or row.get("rejected")
            or row.get("tier") not in {"A", "B"}
            for row in aligned_all
        ):
            raise ReplacementError("aligned_all.json contains a rejected replacement")
    return records


def _assert_regular_tree(root: Path) -> None:
    for current_text, directories, files in os.walk(root, followlinks=False):
        current = Path(current_text)
        for name in directories:
            path = current / name
            if path.is_symlink() or not path.is_dir():
                raise ReplacementError(f"replacement tree contains unsafe directory: {path}")
        for name in files:
            path = current / name
            mode = path.lstat().st_mode
            if path.is_symlink() or not stat.S_ISREG(mode):
                raise ReplacementError(f"replacement tree contains a non-regular file: {path}")


def _generated_artifacts(staging: Path, target: dict[str, Any]) -> list[dict[str, Any]]:
    odir = staging / "objects" / f"obj_{target['legacy_index']:02d}"
    artifacts = []
    for path in sorted(odir.iterdir(), key=lambda item: item.name):
        if path.is_file() and not path.is_symlink():
            item = _artifact(path)
            item["path"] = str(path.relative_to(staging))
            artifacts.append(item)
    if not artifacts:
        raise ReplacementError(f"replacement target has no artifacts: {odir}")
    return artifacts


def _fsync_tree(root: Path) -> None:
    for current_text, directories, files in os.walk(root, topdown=False):
        current = Path(current_text)
        for name in files:
            descriptor = os.open(current / name, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for name in directories:
            descriptor = os.open(current / name, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_replacement_bundle(
    *,
    config_path: str | os.PathLike[str],
    contract_manifest: str | os.PathLike[str],
    bundle_id: str,
    freeze_id: str,
    out: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str] | None = None,
    python: str | os.PathLike[str] = sys.executable,
    sam3_python: str | os.PathLike[str] | None = None,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    allow_dirty: bool = False,
    _stage_runner: StageRunner | None = None,
    _sam3_runtime_probe: Sam3RuntimeProbe | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one pinned replacement bundle."""
    root = Path(os.path.abspath(repo_root))
    config_file = _repo_path(
        config_path, root=root, label="E2 fidelity config", must_exist=True
    )
    contract_file = _repo_path(
        contract_manifest, root=root, label="E0 contract", must_exist=True
    )
    _regular_file(config_file, "E2 fidelity config")
    _regular_file(contract_file, "E0 contract")
    if not isinstance(freeze_id, str) or FREEZE_RE.fullmatch(freeze_id) is None:
        raise ReplacementError("freeze_id contains unsafe path characters")
    config, bundles = _load_config(config_file)
    if bundle_id not in bundles:
        raise ReplacementError(f"unknown remediation bundle: {bundle_id}")
    bundle = bundles[bundle_id]
    expected_out = _repo_path(
        f"outputs/icra2027/{freeze_id}/fidelity/replacements/{bundle_id}",
        root=root, label="replacement output", must_exist=False,
    )
    output = _repo_path(out, root=root, label="replacement output", must_exist=False)
    if output != expected_out:
        raise ReplacementError(f"replacement output must be exactly {expected_out}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite replacement bundle: {output}")

    git = _git_snapshot(root)
    if git["dirty"] and not allow_dirty:
        raise ReplacementError("replacement build requires a clean Git worktree")
    contract = _validate_contract(
        contract_file, config_path=config_file, config=config,
        freeze_id=freeze_id, git=git,
    )
    model_inputs, resolved_model_inputs = _validate_model_inputs(config, root=root)

    data_root = Path(os.path.abspath(
        dataset_root
        if dataset_root is not None
        else os.environ.get("SIMANY_SCANNETPP_ROOT", "/data/ScanNetpp")
    ))
    scene_root = data_root / "data" / bundle["scene_id"]
    split_path = scene_root / "dslr" / "train_test_lists.json"
    images_dir = scene_root / "dslr" / "resized_undistorted_images"
    train, test = _load_split(split_path)
    if images_dir.is_symlink() or not images_dir.is_dir():
        raise ReplacementError(f"missing non-symlink DSLR image directory: {images_dir}")
    dataset_paths = {
        "split": split_path,
        "intrinsics": scene_root / "dslr/nerfstudio/transforms_undistorted.json",
        "poses": scene_root / "dslr/colmap/images.txt",
        "mesh": scene_root / "scans/mesh_aligned_0.05.ply",
        "segments": scene_root / "scans/segments.json",
        "segments_anno": scene_root / "scans/segments_anno.json",
    }
    dataset_static = {
        name: _artifact(_regular_file(path, f"dataset {name}"))
        for name, path in dataset_paths.items()
    }
    source, legacy_records, legacy_used_paths = _legacy_provenance(
        bundle, root=root, test_frames=set(test)
    )
    if output == source or source in output.parents or output in source.parents:
        raise ReplacementError("replacement output must be disjoint from the legacy source")
    legacy_hashes_before = {str(path): _sha256(path) for path in legacy_used_paths}

    main_python = str(Path(python))
    mask_python = str(Path(sam3_python) if sam3_python is not None else Path(python))
    runner = _stage_runner or _run_subprocess_stage
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise ReplacementError(f"replacement parent is a symlink: {output.parent}")
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    if staging.exists() or staging.is_symlink():
        raise ReplacementError(f"unexpected replacement staging path: {staging}")

    try:
        (staging / "objects").mkdir(parents=True)
        (staging / "provenance").mkdir()
        allowlist_path = staging / "provenance" / "official_train_frames.json"
        index_map_path = staging / "provenance" / "preserved_index_by_gt_object_id.json"
        read_frames_path = staging / "provenance" / "factory_read_frames.json"
        _write_json(allowlist_path, {"frames": train})
        _write_json(index_map_path, {
            str(target["gt_object_id"]): target["legacy_index"]
            for target in bundle["targets"]
        })

        copied_auto = None
        if bundle["mode"] == "auto":
            auto_source = _regular_file(source / "auto_instances.npz", "legacy auto instances")
            legacy_hashes_before[str(auto_source)] = _sha256(auto_source)
            copied_auto = _copy_file(auto_source, staging / "auto_instances.npz")
            copied_auto["path"] = "auto_instances.npz"
            copied_auto["source_path"] = str(auto_source.relative_to(root))
            copied_auto["source_sha256"] = legacy_hashes_before[str(auto_source)]

        environment = dict(os.environ)
        environment.update({
            "SIMANY_ROOT": str(root),
            "SIMANY_SCENE": bundle["scene_id"],
            "SIMANY_OUT": str(staging),
            "SIMANY_SCANNETPP_ROOT": str(data_root),
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "SIMANY_TRELLIS_DIR": str(resolved_model_inputs["trellis_source"]),
            "SIMANY_TRELLIS_MODEL": str(resolved_model_inputs["trellis_snapshot"]),
            "SIMANY_DINOV2_REPO": str(resolved_model_inputs["dinov2_source"]),
            "SIMANY_SAM3_CKPT": str(resolved_model_inputs["sam3_checkpoint"]),
            "SIMANY_MIN_BBOX_PX": "48",
            "SIMANY_VIS_TOL_M": "0.02",
            "SIMANY_MIN_MASK_PX": "400",
            "TORCH_HOME": str(resolved_model_inputs["dinov2_source"].parent.parent),
            "ATTN_BACKEND": "xformers",
            "SPCONV_ALGO": "native",
        })
        sam3_source = str(resolved_model_inputs["sam3_source"])
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONOPTIMIZE", None)
        environment.pop("SIMF_OUT", None)
        environment.pop("SIMF_SCENE", None)
        environment.pop("SIMANY_MESH_SRC", None)
        environment.pop("SIMF_MESH_SRC", None)
        environment.pop("SIMANY_FULL", None)
        environment.pop("SIMF_FULL", None)
        for legacy_threshold in (
            "SIMF_MIN_BBOX_PX", "SIMF_VIS_TOL_M", "SIMF_MIN_MASK_PX",
        ):
            environment.pop(legacy_threshold, None)
        if bundle["mode"] == "auto":
            environment["SIMANY_AUTO"] = "1"
        else:
            environment.pop("SIMANY_AUTO", None)
            environment.pop("SIMF_AUTO", None)

        sam3_environment = dict(environment)
        sam3_environment["PYTHONPATH"] = sam3_source
        probe = _sam3_runtime_probe or _probe_sam3_runtime
        sam3_runtime = _validate_sam3_runtime(
            probe(mask_python, sam3_environment, resolved_model_inputs["sam3_source"]),
            source_root=resolved_model_inputs["sam3_source"],
            expected_version=config["leakage_remediation"]["model_inputs"][
                "sam3_source"
            ]["package_version"],
            root=root,
        )
        model_inputs["runtime"] = {"sam3": sam3_runtime}

        commands = [
            (
                "factory_prepare",
                [main_python, "-m", "agents.discover.factory_prepare",
                 "--frame-allowlist", str(allowlist_path),
                 "--output-index-map", str(index_map_path),
                 "--read-frames-out", str(read_frames_path)],
            ),
            (
                "factory_refine_masks",
                [mask_python, "-m", "agents.discover.factory_refine_masks",
                 "--images-dir", str(images_dir), "--out-dir", str(staging)],
            ),
            ("trellis", [main_python, "-m", "models.s4_trellis"]),
            ("factory_align", [main_python, "-m", "agents.assets.factory_align"]),
        ]
        stage_records = []
        prepared: list[dict[str, Any]] = []
        dataset_inputs: dict[str, Any] | None = None
        for stage, command in commands:
            stage_environment = (
                sam3_environment if stage == "factory_refine_masks" else environment
            )
            runner(stage, command, stage_environment, staging, root)
            if stage == "factory_prepare":
                read_payload = _read_json(read_frames_path, "factory read frames")
                read_frames = _plain_frame_list(
                    read_payload.get("frames") if isinstance(read_payload, dict) else None,
                    "factory read frames",
                )
                if not set(read_frames) <= set(train):
                    raise ReplacementError(
                        "factory_prepare read-frame set is not train-only"
                    )
                read_images = []
                for frame in sorted(read_frames):
                    item = _artifact(
                        _regular_file(images_dir / frame, f"read DSLR frame {frame}")
                    )
                    item["frame"] = frame
                    read_images.append(item)
                read_control = _artifact(read_frames_path)
                read_control["path"] = str(read_frames_path.relative_to(staging))
                dataset_inputs = {
                    **dataset_static,
                    "read_frames_control": read_control,
                    "read_images": read_images,
                }
                prepared = _prepared_records(
                    staging, bundle, train_frames=set(train), test_frames=set(test),
                    require_trellis=False, require_alignment=False,
                )
            elif stage == "factory_refine_masks":
                prepared = _prepared_records(
                    staging, bundle, train_frames=set(train), test_frames=set(test),
                    require_trellis=False, require_alignment=False,
                )
            elif stage == "trellis":
                prepared = _prepared_records(
                    staging, bundle, train_frames=set(train), test_frames=set(test),
                    require_trellis=True, require_alignment=False,
                )
            else:
                prepared = _prepared_records(
                    staging, bundle, train_frames=set(train), test_frames=set(test),
                    require_trellis=True, require_alignment=True,
                )
            relative_args = []
            for argument in command[3:]:
                relative_args.append(
                    str(Path(argument).relative_to(staging))
                    if str(argument).startswith(f"{staging}{os.sep}") else argument
                )
            stage_records.append({
                "name": stage,
                "module": command[2],
                "python": str(Path(command[0]).resolve(strict=False)),
                "arguments": relative_args,
                "status": "passed",
            })

        if not prepared:
            raise ReplacementError("replacement preparation produced no targets")
        if dataset_inputs is None:
            raise ReplacementError("factory dataset read closure was not recorded")
        optimization_frames = sorted({
            record["new_train_frame"] for record in prepared
        })
        if set(optimization_frames) & set(test):
            raise ReplacementError("replacement optimization frames overlap official test")

        legacy_hashes_after = {path: _sha256(Path(path)) for path in legacy_hashes_before}
        if legacy_hashes_after != legacy_hashes_before:
            raise ReplacementError("a legacy source artifact changed during replacement build")
        for prepared_record, legacy_record in zip(prepared, legacy_records, strict=True):
            if prepared_record["gt_object_id"] != legacy_record["gt_object_id"]:
                raise ReplacementError("legacy/new target ordering drifted")
            prepared_record["legacy_artifacts"] = legacy_record["legacy_artifacts"]
            prepared_record["generated_artifacts"] = _generated_artifacts(
                staging, prepared_record
            )
        for name, path in dataset_paths.items():
            if dataset_inputs[name] != _artifact(path):
                raise ReplacementError(f"dataset {name} changed during replacement build")
        for record in dataset_inputs["read_images"]:
            frame_path = images_dir / record["frame"]
            observed = _artifact(frame_path)
            observed["frame"] = record["frame"]
            if observed != record:
                raise ReplacementError(
                    f"dataset read image changed during replacement build: {record['frame']}"
                )

        _assert_regular_tree(staging)
        controls = []
        for path in (
            allowlist_path, index_map_path,
            read_frames_path,
            staging / "objects" / "objects.json",
            staging / "objects" / "aligned_all.json",
        ):
            item = _artifact(path)
            item["path"] = str(path.relative_to(staging))
            controls.append(item)
        manifest = {
            "schema_version": 1,
            "task": "E2 leakage remediation",
            "freeze_id": freeze_id,
            "bundle_id": bundle_id,
            "dataset_id": "scannetpp_v2",
            "split_id": "nvs_sem_val",
            "scene_id": bundle["scene_id"],
            "mode": bundle["mode"],
            "created_utc": _now_utc(),
            "git": git,
            "config_artifact": {
                **_artifact(config_file, root=root),
                "canonical_sha256": _canonical_hash(config),
            },
            "contract_artifact": contract,
            "model_inputs": model_inputs,
            "stage_parameters": json.loads(json.dumps(EXPECTED_STAGE_PARAMETERS)),
            "official_split_artifact": _artifact(split_path),
            "dataset_inputs": dataset_inputs,
            "official_train_frame_count": len(train),
            "official_test_frame_count": len(test),
            "official_train_frames_sha256": _canonical_hash(train),
            "official_test_frames_sha256": _canonical_hash(test),
            "legacy_source": str(source.relative_to(root)),
            "legacy_source_immutable": True,
            "auto_instances": copied_auto,
            "optimization_input_frames": optimization_frames,
            "targets": prepared,
            "stages": stage_records,
            "bundle_artifacts": controls,
            "validation": {
                "fixed_target_population": True,
                "target_count": len(prepared),
                "mapped_by_gt_object_id": True,
                "preserved_legacy_indices": True,
                "generation_frames_official_train_only": True,
                "generation_eval_overlap": [],
                "all_regenerated_alignments_accepted": True,
                "legacy_sources_unchanged": True,
                "atomic_fresh_publish": True,
            },
            "paper_ready": False,
            "paper_ready_reason": (
                "Replacement assets are validated inputs; the complete held-out "
                "room export and fidelity metrics must still be regenerated."
            ),
        }
        _write_json(staging / "replacement_manifest.json", manifest)
        _assert_regular_tree(staging)
        _fsync_tree(staging)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"replacement output appeared during build: {output}")
        staging.rename(output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def _verify_artifact_record(
    record: Any,
    *,
    base: Path,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(record, dict):
        raise ReplacementError(f"{label} artifact record must be an object")
    declared_path = record.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise ReplacementError(f"{label} artifact has no path")
    if expected_path is None:
        relative = Path(declared_path)
        if (
            relative.is_absolute()
            or relative in {Path("."), Path("..")}
            or ".." in relative.parts
        ):
            raise ReplacementError(f"{label} artifact path is unsafe: {declared_path}")
        path = Path(os.path.abspath(base / relative))
        try:
            path.relative_to(base)
        except ValueError as exc:
            raise ReplacementError(f"{label} artifact escapes its bundle") from exc
    else:
        path = expected_path
        if declared_path not in {str(path), str(path.relative_to(base))}:
            raise ReplacementError(f"{label} artifact path mismatch: {declared_path}")
    _regular_file(path, label)
    if record.get("size_bytes") != path.stat().st_size:
        raise ReplacementError(f"{label} artifact size mismatch: {path}")
    declared_sha = str(record.get("sha256", "")).lower()
    if SHA256_RE.fullmatch(declared_sha) is None or declared_sha != _sha256(path):
        raise ReplacementError(f"{label} artifact hash mismatch: {path}")
    return path


def validate_replacement_bundle(
    *,
    config_path: str | os.PathLike[str],
    contract_manifest: str | os.PathLike[str],
    bundle_id: str,
    freeze_id: str,
    out: str | os.PathLike[str],
    dataset_root: str | os.PathLike[str] | None = None,
    repo_root: str | os.PathLike[str] = REPO_ROOT,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Validate a closed, already-published bundle for safe job resume."""
    root = Path(os.path.abspath(repo_root))
    config_file = _repo_path(
        config_path, root=root, label="E2 fidelity config", must_exist=True
    )
    contract_file = _repo_path(
        contract_manifest, root=root, label="E0 contract", must_exist=True
    )
    _regular_file(config_file, "E2 fidelity config")
    _regular_file(contract_file, "E0 contract")
    if not isinstance(freeze_id, str) or FREEZE_RE.fullmatch(freeze_id) is None:
        raise ReplacementError("freeze_id contains unsafe path characters")
    config, bundles = _load_config(config_file)
    if bundle_id not in bundles:
        raise ReplacementError(f"unknown remediation bundle: {bundle_id}")
    bundle = bundles[bundle_id]
    expected_out = _repo_path(
        f"outputs/icra2027/{freeze_id}/fidelity/replacements/{bundle_id}",
        root=root, label="replacement output", must_exist=True,
    )
    output = _repo_path(out, root=root, label="replacement output", must_exist=True)
    if output != expected_out or output.is_symlink() or not output.is_dir():
        raise ReplacementError(f"replacement output must be the canonical directory: {expected_out}")
    _assert_regular_tree(output)

    git = _git_snapshot(root)
    if git["dirty"] and not allow_dirty:
        raise ReplacementError("replacement validation requires a clean Git worktree")
    contract = _validate_contract(
        contract_file, config_path=config_file, config=config,
        freeze_id=freeze_id, git=git,
    )
    model_inputs, _resolved_model_inputs = _validate_model_inputs(config, root=root)
    data_root = Path(os.path.abspath(
        dataset_root
        if dataset_root is not None
        else os.environ.get("SIMANY_SCANNETPP_ROOT", "/data/ScanNetpp")
    ))
    scene_root = data_root / "data" / bundle["scene_id"]
    split_path = scene_root / "dslr/train_test_lists.json"
    images_dir = scene_root / "dslr/resized_undistorted_images"
    train, test = _load_split(split_path)
    source, legacy_records, _legacy_paths = _legacy_provenance(
        bundle, root=root, test_frames=set(test)
    )

    manifest_path = _regular_file(
        output / "replacement_manifest.json", "replacement manifest"
    )
    manifest = _read_json(manifest_path, "replacement manifest")
    if not isinstance(manifest, dict):
        raise ReplacementError("replacement manifest must be an object")
    fixed = {
        "schema_version": 1,
        "task": "E2 leakage remediation",
        "freeze_id": freeze_id,
        "bundle_id": bundle_id,
        "dataset_id": "scannetpp_v2",
        "split_id": "nvs_sem_val",
        "scene_id": bundle["scene_id"],
        "mode": bundle["mode"],
        "legacy_source": bundle["legacy_source"],
        "legacy_source_immutable": True,
        "paper_ready": False,
    }
    for field, expected in fixed.items():
        if manifest.get(field) != expected:
            raise ReplacementError(f"replacement manifest {field} mismatch")
    manifest_git = manifest.get("git")
    if not isinstance(manifest_git, dict) or manifest_git.get("commit") != git["commit"]:
        raise ReplacementError("replacement manifest Git commit mismatch")
    if bool(manifest_git.get("dirty")) and not allow_dirty:
        raise ReplacementError("replacement manifest records a dirty build")

    expected_config_artifact = {
        **_artifact(config_file, root=root),
        "canonical_sha256": _canonical_hash(config),
    }
    if manifest.get("config_artifact") != expected_config_artifact:
        raise ReplacementError("replacement config provenance mismatch")
    if manifest.get("contract_artifact") != contract:
        raise ReplacementError("replacement E0 contract provenance mismatch")
    manifest_model_inputs = manifest.get("model_inputs")
    if not isinstance(manifest_model_inputs, dict) or {
        key: manifest_model_inputs.get(key) for key in ("declared", "observed")
    } != model_inputs:
        raise ReplacementError("replacement model input provenance mismatch")
    runtime_inputs = manifest_model_inputs.get("runtime")
    sam3_record = (
        runtime_inputs.get("sam3") if isinstance(runtime_inputs, dict) else None
    )
    if not isinstance(sam3_record, dict):
        raise ReplacementError("replacement SAM3 runtime provenance missing")
    module_path = _repo_path(
        sam3_record.get("module_file", ""), root=root,
        label="recorded SAM3 runtime module", must_exist=True,
    )
    expected_sam3_runtime = _validate_sam3_runtime(
        {
            "module_file": str(module_path),
            "package_version": sam3_record.get("package_version"),
        },
        source_root=_resolved_model_inputs["sam3_source"],
        expected_version=config["leakage_remediation"]["model_inputs"][
            "sam3_source"
        ]["package_version"],
        root=root,
    )
    if sam3_record != expected_sam3_runtime:
        raise ReplacementError("replacement SAM3 runtime provenance mismatch")
    if manifest.get("stage_parameters") != EXPECTED_STAGE_PARAMETERS:
        raise ReplacementError("replacement stage parameters mismatch")
    if manifest.get("official_split_artifact") != _artifact(split_path):
        raise ReplacementError("replacement official split provenance mismatch")
    if (
        manifest.get("official_train_frame_count") != len(train)
        or manifest.get("official_test_frame_count") != len(test)
        or manifest.get("official_train_frames_sha256") != _canonical_hash(train)
        or manifest.get("official_test_frames_sha256") != _canonical_hash(test)
    ):
        raise ReplacementError("replacement official split roster provenance mismatch")
    dataset_paths = {
        "split": split_path,
        "intrinsics": scene_root / "dslr/nerfstudio/transforms_undistorted.json",
        "poses": scene_root / "dslr/colmap/images.txt",
        "mesh": scene_root / "scans/mesh_aligned_0.05.ply",
        "segments": scene_root / "scans/segments.json",
        "segments_anno": scene_root / "scans/segments_anno.json",
    }
    dataset_inputs = manifest.get("dataset_inputs")
    if not isinstance(dataset_inputs, dict):
        raise ReplacementError("replacement dataset input provenance missing")
    for name, path in dataset_paths.items():
        if dataset_inputs.get(name) != _artifact(_regular_file(path, f"dataset {name}")):
            raise ReplacementError(f"replacement dataset {name} provenance mismatch")
    read_control = dataset_inputs.get("read_frames_control")
    read_control_path = _verify_artifact_record(
        read_control, base=output, label="factory read-frame control",
        expected_path=output / "provenance/factory_read_frames.json",
    )
    read_payload = _read_json(read_control_path, "factory read frames")
    read_frames = _plain_frame_list(
        read_payload.get("frames") if isinstance(read_payload, dict) else None,
        "factory read frames",
    )
    if not set(read_frames) <= set(train):
        raise ReplacementError("recorded factory reads are not official-train-only")
    read_images = dataset_inputs.get("read_images")
    if not isinstance(read_images, list) or len(read_images) != len(read_frames):
        raise ReplacementError("replacement read-image population mismatch")
    read_by_frame = {
        row.get("frame"): row for row in read_images if isinstance(row, dict)
    }
    if set(read_by_frame) != set(read_frames) or len(read_by_frame) != len(read_images):
        raise ReplacementError("replacement read-image frame set mismatch")
    for frame in read_frames:
        record = read_by_frame[frame]
        _verify_artifact_record(
            record, base=(images_dir / frame).parent, label=f"read DSLR frame {frame}",
            expected_path=images_dir / frame,
        )

    prepared = _prepared_records(
        output, bundle, train_frames=set(train), test_frames=set(test),
        require_trellis=True, require_alignment=True,
    )
    manifest_targets = manifest.get("targets")
    if not isinstance(manifest_targets, list) \
            or len(manifest_targets) != len(prepared):
        raise ReplacementError("replacement manifest target population mismatch")
    declared_output_paths: set[str] = {"replacement_manifest.json"}
    for observed, declared, legacy in zip(
        prepared, manifest_targets, legacy_records, strict=True
    ):
        if not isinstance(declared, dict):
            raise ReplacementError("replacement target record must be an object")
        for field in (
            "legacy_index", "gt_object_id", "label", "old_test_frame",
            "new_train_frame", "optimization_input_frames", "alignment_tier",
        ):
            if declared.get(field) != observed.get(field):
                raise ReplacementError(f"replacement target {field} mismatch")
        if declared.get("legacy_artifacts") != legacy.get("legacy_artifacts"):
            raise ReplacementError("replacement legacy target provenance mismatch")
        for legacy_artifact in declared["legacy_artifacts"]:
            legacy_path = _repo_path(
                legacy_artifact["path"], root=root,
                label="legacy target artifact", must_exist=True,
            )
            _verify_artifact_record(
                legacy_artifact, base=root, label="legacy target",
                expected_path=legacy_path,
            )
        generated = declared.get("generated_artifacts")
        if not isinstance(generated, list) or not generated:
            raise ReplacementError("replacement target lacks generated artifacts")
        for artifact in generated:
            path = _verify_artifact_record(
                artifact, base=output, label="generated target"
            )
            relative = str(path.relative_to(output))
            if relative in declared_output_paths:
                raise ReplacementError("replacement artifact is declared more than once")
            declared_output_paths.add(relative)

    expected_optimization = sorted({row["new_train_frame"] for row in prepared})
    if manifest.get("optimization_input_frames") != expected_optimization \
            or set(expected_optimization) & set(test):
        raise ReplacementError("replacement optimization frame union is invalid")
    controls = manifest.get("bundle_artifacts")
    if not isinstance(controls, list) or len(controls) != 5:
        raise ReplacementError("replacement control artifact population mismatch")
    if {row.get("path") for row in controls if isinstance(row, dict)} != {
        "provenance/official_train_frames.json",
        "provenance/preserved_index_by_gt_object_id.json",
        "provenance/factory_read_frames.json",
        "objects/objects.json",
        "objects/aligned_all.json",
    }:
        raise ReplacementError("replacement control artifact set mismatch")
    for artifact in controls:
        path = _verify_artifact_record(artifact, base=output, label="bundle control")
        relative = str(path.relative_to(output))
        if relative in declared_output_paths:
            raise ReplacementError("replacement artifact is declared more than once")
        declared_output_paths.add(relative)

    auto_record = manifest.get("auto_instances")
    if bundle["mode"] == "auto":
        auto_path = _verify_artifact_record(
            auto_record, base=output, label="copied auto instances"
        )
        if str(auto_path.relative_to(output)) != "auto_instances.npz":
            raise ReplacementError("copied auto instance path mismatch")
        auto_source = _regular_file(source / "auto_instances.npz", "legacy auto instances")
        if (
            auto_record.get("source_path") != str(auto_source.relative_to(root))
            or auto_record.get("source_sha256") != _sha256(auto_source)
        ):
            raise ReplacementError("copied auto instance source provenance mismatch")
        declared_output_paths.add("auto_instances.npz")
    elif auto_record is not None:
        raise ReplacementError("factory replacement must not carry auto instances")

    stages = manifest.get("stages")
    expected_stages = [
        "factory_prepare", "factory_refine_masks", "trellis", "factory_align",
    ]
    if not isinstance(stages, list) or [row.get("name") for row in stages
                                       if isinstance(row, dict)] != expected_stages \
            or any(row.get("status") != "passed" for row in stages):
        raise ReplacementError("replacement stage provenance is incomplete")
    validation = manifest.get("validation")
    required_validation = {
        "fixed_target_population": True,
        "target_count": len(prepared),
        "mapped_by_gt_object_id": True,
        "preserved_legacy_indices": True,
        "generation_frames_official_train_only": True,
        "generation_eval_overlap": [],
        "all_regenerated_alignments_accepted": True,
        "legacy_sources_unchanged": True,
        "atomic_fresh_publish": True,
    }
    if validation != required_validation:
        raise ReplacementError("replacement validation declaration mismatch")

    actual_output_paths = {
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    }
    if actual_output_paths != declared_output_paths:
        missing = sorted(actual_output_paths - declared_output_paths)
        stale = sorted(declared_output_paths - actual_output_paths)
        raise ReplacementError(
            f"replacement artifact closure mismatch: undeclared={missing}, missing={stale}"
        )
    actual_directories = {
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_dir()
    }
    expected_directories = {"objects", "provenance"} | {
        f"objects/obj_{target['legacy_index']:02d}" for target in bundle["targets"]
    }
    if actual_directories != expected_directories:
        raise ReplacementError("replacement directory closure mismatch")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="frozen E2 fidelity_manifest.json")
    parser.add_argument("--contract-manifest", required=True, help="fresh E0 freeze_manifest.json")
    parser.add_argument("--bundle-id", required=True, choices=tuple(EXPECTED_BUNDLES))
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--out", required=True, help="fresh canonical replacement directory")
    parser.add_argument("--dataset-root", help="read-only ScanNet++ root")
    parser.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    parser.add_argument("--sam3-python")
    parser.add_argument(
        "--validate-existing", action="store_true",
        help="strictly validate and reuse --out without running generation stages",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        kwargs = {
            "config_path": args.config,
            "contract_manifest": args.contract_manifest,
            "bundle_id": args.bundle_id,
            "freeze_id": args.freeze_id,
            "out": args.out,
            "dataset_root": args.dataset_root,
        }
        if args.validate_existing:
            manifest = validate_replacement_bundle(**kwargs)
        else:
            if not args.sam3_python:
                raise ReplacementError("--sam3-python is required for a new build")
            manifest = build_replacement_bundle(
                **kwargs, python=args.python, sam3_python=args.sam3_python,
            )
    except (FileExistsError, ReplacementError) as exc:
        print(f"E2_REPLACEMENT=FAIL: {exc}", file=sys.stderr)
        return 2
    print("E2_REPLACEMENT=REUSED" if args.validate_existing else "E2_REPLACEMENT=PASS")
    print(f"bundle_id={manifest['bundle_id']}")
    print(f"output={args.out}")
    print(f"targets={len(manifest['targets'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
