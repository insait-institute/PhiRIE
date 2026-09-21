"""Build the canonical E2 fidelity source manifest without mutating sources.

The inventory has two deliberately separate responsibilities:

* validate freshly exported, paired room renders under ``room_runs``; and
* audit legacy metric-only artifacts as comparison anchors.

Legacy ``render_metrics_v2.json`` files never become evaluator inputs.  They
do not archive their rendered/target images or enough run provenance to be
recomputed.  Likewise, reviewed object aggregates are exposed only when the
currently enumerable per-scene source snapshot reproduces the reviewed
population and values.  Otherwise their canonical values remain ``null``.

The destination is published as one repository-local, no-overwrite bundle.
No source artifact is changed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from PIL import Image

from robo.eval.fidelity_replacements import (
    EXPECTED_BUNDLES,
    EXPECTED_STAGE_PARAMETERS,
    ReplacementError,
    _validate_model_input_schema,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/icra2027/fidelity_manifest.json"
DEFAULT_CONSTRUCTION_CONFIG = (
    REPO_ROOT / "configs/experiments/icra2027/construction_regimes.yaml"
)
DEFAULT_OBJECT_REVIEW = REPO_ROOT / "outputs/review_recompute_hybrid.json"

SCENE_RE = re.compile(r"^[0-9a-f]{10}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
FREEZE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

ROOM_METHODS = {
    "Input scene Gaussian, reconstruction ceiling": "input_scene_gaussian",
    "Factorized composite, GT discovery": "factorized_gt_discovery",
    "Factorized composite, automatic discovery": "factorized_auto_discovery",
    "Composite + Harmonizer Option C": "harmonizer_option_c",
}
AVAILABLE_ROOM_METHOD_IDS = tuple(list(ROOM_METHODS.values())[:3])
OBJECT_METHODS = {
    "TRELLIS, best single view": "trellis_best_single_view",
    "ReconViaGen, multi-view": "reconviagen_multi_view",
    "Evidence-selected proposal": "evidence_selected_proposal",
    "Evaluation-only oracle candidate": "evaluation_only_oracle_candidate",
}
EXPECTED_DATASET = "scannetpp_v2"
EXPECTED_POPULATION_SPLIT = "nvs_sem_val"
EXPECTED_EXPORT_SPLIT = "nvs_sem_val:official-test"
EXPECTED_VIEWS_PER_SCENE = 8

KNOWN_ROOM_ANCHORS = {
    "input_scene_gaussian": {"psnr": 28.89, "ssim": 0.909},
    "factorized_gt_discovery": {"psnr": 28.32, "ssim": 0.906},
    "factorized_auto_discovery": {"psnr": 27.52},
}
METHOD_TO_REMEDIATION_MODE = {
    "factorized_gt_discovery": "factory",
    "factorized_auto_discovery": "auto",
}


class InventoryError(ValueError):
    """Raised for an unsafe or internally contradictory inventory request."""


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(
    value: str | os.PathLike[str], *, label: str, must_exist: bool = False,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as exc:
        raise InventoryError(f"cannot resolve {label}: {path}: {exc}") from exc
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise InventoryError(
            f"{label} must stay inside the SimAny repository: {resolved}"
        ) from exc
    return resolved


def _relative(path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise InventoryError(f"path is outside the SimAny repository: {path}") from exc


def _first_symlink(path: Path) -> Path | None:
    """Return the first symlink component without following the final path."""
    absolute = Path(os.path.abspath(path))
    root = Path(os.path.abspath(REPO_ROOT))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise InventoryError(f"path is outside the SimAny repository: {path}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _safe_file(path: Path, *, role: str, issues: list[str]) -> bool:
    try:
        symlink = _first_symlink(path)
    except InventoryError:
        _append_once(issues, f"{role}_outside_repository")
        return False
    if symlink is not None:
        _append_once(issues, f"{role}_uses_symlink_path")
        return False
    if not path.is_file():
        _append_once(issues, f"missing_{role}")
        return False
    return True


def _artifact(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": _relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _external_artifact(
    path: Path, *, role: str, issues: list[str],
) -> dict[str, Any] | None:
    """Hash an absolute read-only input while rejecting every symlink hop."""
    if not path.is_absolute():
        _append_once(issues, f"{role}_path_not_absolute")
        return None
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            _append_once(issues, f"{role}_uses_symlink_path")
            return None
    if not absolute.is_file():
        _append_once(issues, f"missing_{role}")
        return None
    return {
        "role": role,
        "path": str(absolute),
        "size_bytes": absolute.stat().st_size,
        "sha256": _sha256_file(absolute),
    }


def _load_json(
    path: Path,
    *,
    role: str,
    issues: list[str],
    artifacts: list[dict[str, Any]],
) -> Any | None:
    if not _safe_file(path, role=role, issues=issues):
        return None
    artifacts.append(_artifact(path, role=role))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _append_once(issues, f"malformed_{role}:{type(exc).__name__}")
        return None


def _normalize_scenes(values: Iterable[Any], *, label: str) -> list[str]:
    scenes = [str(value).strip() for value in values if str(value).strip()]
    duplicates = sorted(scene for scene, count in Counter(scenes).items() if count > 1)
    if duplicates:
        raise InventoryError(f"duplicate {label}: {duplicates}")
    invalid = sorted(scene for scene in scenes if SCENE_RE.fullmatch(scene) is None)
    if invalid:
        raise InventoryError(f"invalid {label}: {invalid}")
    return sorted(scenes)


def _config_population(payload: Any, *, label: str) -> list[str]:
    if not isinstance(payload, dict):
        raise InventoryError(f"{label} must be a JSON/YAML object")
    population = payload.get("population")
    if not isinstance(population, dict):
        raise InventoryError(f"{label} lacks population object")
    if population.get("dataset") != EXPECTED_DATASET:
        raise InventoryError(f"{label} population dataset must be {EXPECTED_DATASET}")
    if population.get("split") != EXPECTED_POPULATION_SPLIT:
        raise InventoryError(
            f"{label} population split must be {EXPECTED_POPULATION_SPLIT}"
        )
    if population.get("preserve_failed_scenes") is not True:
        raise InventoryError(f"{label} must preserve failed scenes")
    count = population.get("planned_scenes")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise InventoryError(f"{label} planned_scenes must be a positive integer")
    scenes = population.get("scene_ids")
    if not isinstance(scenes, list):
        raise InventoryError(f"{label} population.scene_ids must be a list")
    normalized = _normalize_scenes(scenes, label=f"{label} scene IDs")
    if len(normalized) != count:
        raise InventoryError(
            f"{label} declares {count} scenes but contains {len(normalized)} IDs"
        )
    return normalized


def _load_yaml(path: Path) -> Any:
    # PyYAML is already a repository dependency, but keeping the import local
    # avoids making JSON-only users import it at module import time.
    import yaml

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise InventoryError(f"cannot parse construction config {path}: {exc}") from exc


def _validate_config(config: Any, expected_scenes: int) -> list[str]:
    if not isinstance(config, dict) or config.get("schema_version") != 2:
        raise InventoryError("fidelity config must use schema_version 2")
    scenes = _config_population(config, label="fidelity config")
    if len(scenes) != expected_scenes:
        raise InventoryError(
            f"expected exactly {expected_scenes} configured scenes, found {len(scenes)}"
        )
    smoke_scenes = config["population"].get("smoke_scene_ids")
    if (
        not isinstance(smoke_scenes, list)
        or len(smoke_scenes) != 2
        or len(set(smoke_scenes)) != 2
        or not set(smoke_scenes) <= set(scenes)
    ):
        raise InventoryError(
            "fidelity population must pin two unique in-roster smoke_scene_ids"
        )
    room_split = config.get("room_split")
    if room_split != {
        "name": "official_dslr_test",
        "unit": "held_out_view",
        "views_per_scene": EXPECTED_VIEWS_PER_SCENE,
        "require_official_test": True,
        "require_paired_filenames": True,
    }:
        raise InventoryError("fidelity room_split differs from the fixed E2 contract")
    for key, expected in (
        ("room_methods", ROOM_METHODS), ("object_methods", OBJECT_METHODS),
    ):
        methods = config.get(key)
        if not isinstance(methods, dict) or set(methods) != set(expected):
            raise InventoryError(f"{key} labels differ from the fixed E2 contract")
        for label, source_id in expected.items():
            spec = methods[label]
            if not isinstance(spec, dict) or spec.get("source_id") != source_id:
                raise InventoryError(f"{key}.{label} has the wrong source_id")
            if spec.get("records") != []:
                raise InventoryError(f"checked-in {key}.{label}.records must start empty")
    coverage = config.get("coverage")
    if not isinstance(coverage, dict):
        raise InventoryError("fidelity config lacks coverage contract")
    expected_scene_floors = {
        method: expected_scenes for method in AVAILABLE_ROOM_METHOD_IDS
    } | {"harmonizer_option_c": 0}
    expected_view_floors = {
        method: expected_scenes * EXPECTED_VIEWS_PER_SCENE
        for method in AVAILABLE_ROOM_METHOD_IDS
    } | {"harmonizer_option_c": 0}
    if coverage.get("minimum_room_scenes_by_method") != expected_scene_floors:
        raise InventoryError("minimum room scene coverage differs from the fixed roster")
    if coverage.get("minimum_room_views_by_method") != expected_view_floors:
        raise InventoryError("minimum room view coverage differs from the fixed roster")
    render_export = config.get("render_export")
    if not isinstance(render_export, dict):
        raise InventoryError("fidelity config lacks render_export contract")
    if render_export.get("allow_legacy_metrics_as_measurements") is not False:
        raise InventoryError("legacy metrics must not be enabled as E2 measurements")
    remediation = _parse_leakage_remediation(config)
    if expected_scenes == 50 and not {
        scene_id for scene_id, _mode in remediation
    } <= set(scenes):
        raise InventoryError("leakage remediation scenes are outside the E2 population")
    return scenes


def _parse_leakage_remediation(
    config: Any,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Parse and pin the five known scene/mode replacement contracts."""
    remediation = config.get("leakage_remediation") if isinstance(config, dict) else None
    if not isinstance(remediation, dict) or remediation.get("schema_version") != 1:
        raise InventoryError("fidelity config lacks leakage_remediation schema 1")
    fixed = {
        "official_split_artifact": "dslr/train_test_lists.json",
        "output_root": "outputs/icra2027/{freeze_id}/fidelity/replacements",
        "require_train_only_generation": True,
        "require_accepted_alignment": True,
    }
    for field, expected in fixed.items():
        if remediation.get(field) != expected:
            raise InventoryError(f"leakage_remediation.{field} contract drifted")
    try:
        model_inputs = _validate_model_input_schema(remediation)
    except ReplacementError as exc:
        raise InventoryError(str(exc)) from exc
    bundles = remediation.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != len(EXPECTED_BUNDLES):
        raise InventoryError("leakage remediation must contain the exact five bundles")
    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in bundles:
        if not isinstance(raw, dict):
            raise InventoryError("leakage remediation bundle must be an object")
        bundle_id = raw.get("bundle_id")
        if bundle_id not in EXPECTED_BUNDLES:
            raise InventoryError(f"unknown leakage remediation bundle: {bundle_id!r}")
        expected = EXPECTED_BUNDLES[bundle_id]
        targets = raw.get("targets")
        normalized_targets = tuple(
            (
                target.get("legacy_index"), target.get("gt_object_id"),
                target.get("label"), target.get("old_frame"),
            )
            for target in targets
            if isinstance(target, dict)
        ) if isinstance(targets, list) else ()
        if (
            raw.get("scene_id") != expected["scene_id"]
            or raw.get("mode") != expected["mode"]
            or raw.get("legacy_source") != expected["legacy_source"]
            or normalized_targets != expected["targets"]
        ):
            raise InventoryError(f"leakage remediation target contract drifted: {bundle_id}")
        key = (expected["scene_id"], expected["mode"])
        if key in parsed:
            raise InventoryError(f"duplicate leakage remediation bundle: {bundle_id}")
        parsed[key] = {
            "bundle_id": bundle_id,
            "scene_id": expected["scene_id"],
            "mode": expected["mode"],
            "legacy_source": expected["legacy_source"],
            "targets": [
                {
                    "legacy_index": index,
                    "gt_object_id": object_id,
                    "label": label,
                    "old_test_frame": old_frame,
                }
                for index, object_id, label, old_frame in expected["targets"]
            ],
            "model_inputs": copy.deepcopy(model_inputs),
        }
    if {item["bundle_id"] for item in parsed.values()} != set(EXPECTED_BUNDLES):
        raise InventoryError("leakage remediation does not pin all known bundles")
    return parsed


def _git_snapshot() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": True, "status": ["git_unavailable"]}
    return {"commit": commit, "dirty": bool(status), "status": status}


def _audit_contract(
    contract_path: Path,
    *,
    freeze_id: str,
    smoke: bool,
    inventory_git: dict[str, Any],
    config_file: Path,
    construction_file: Path,
) -> dict[str, Any]:
    issues: list[str] = []
    artifacts: list[dict[str, Any]] = []
    payload = _load_json(
        contract_path,
        role="e0_contract_manifest",
        issues=issues,
        artifacts=artifacts,
    )
    checks: dict[str, bool] = {}
    if isinstance(payload, dict):
        declared_contract_hash = str(payload.get("contract_sha256", "")).lower()
        observed_contract_hash = _canonical_hash({
            key: value for key, value in payload.items()
            if key not in {"created_utc", "environment", "contract_sha256"}
        })
        code = payload.get("code") if isinstance(payload.get("code"), dict) else {}
        contract_commit = str(code.get("commit", "")).strip().lower()
        inventory_commit = str(inventory_git.get("commit", "")).strip().lower()
        config_records = payload.get("configs")
        by_field = {
            str(item.get("field")): item
            for item in config_records
            if isinstance(item, dict) and item.get("field")
        } if isinstance(config_records, list) else {}
        fidelity = by_field.get("fidelity_manifest", {})
        construction = by_field.get("construction_config", {})
        checks = {
            "schema_version": payload.get("schema_version") == 1,
            # A clean E0 smoke contract is sufficient to bind evidence
            # generation.  It does not make the resulting claim paper-ready;
            # that status remains separately fail-closed below.
            "mode": payload.get("mode") in {"smoke", "paper"},
            "clean_contract_code": code.get("dirty") is False,
            "contract_commit_matches_inventory": bool(
                COMMIT_RE.fullmatch(contract_commit)
                and COMMIT_RE.fullmatch(inventory_commit)
                and inventory_commit == contract_commit
            ),
            "freeze_id_inherits_contract": bool(
                str(payload.get("freeze_id", ""))
                and str(freeze_id).startswith(str(payload.get("freeze_id")))
            ),
            "declared_contract_hash_valid": bool(
                SHA256_RE.fullmatch(declared_contract_hash)
                and declared_contract_hash == observed_contract_hash
            ),
            "fidelity_config_hash_matches": (
                fidelity.get("source_content_sha256") == _sha256_file(config_file)
            ),
            "construction_config_hash_matches": (
                construction.get("source_content_sha256")
                == _sha256_file(construction_file)
            ),
        }
        reason_by_check = {
            "schema_version": "e0_contract_schema_invalid",
            "mode": "e0_contract_mode_invalid_for_inventory_mode",
            "clean_contract_code": "e0_contract_records_dirty_code",
            "contract_commit_matches_inventory": "inventory_commit_differs_from_e0_contract",
            "freeze_id_inherits_contract": "e2_freeze_id_does_not_inherit_e0_contract",
            "declared_contract_hash_valid": "e0_contract_declared_hash_mismatch",
            "fidelity_config_hash_matches": "fidelity_config_differs_from_e0_contract",
            "construction_config_hash_matches": "construction_config_differs_from_e0_contract",
        }
        issues.extend(
            reason_by_check[name] for name, passed in checks.items() if not passed
        )
        details = {
            "freeze_id": payload.get("freeze_id"),
            "mode": payload.get("mode"),
            "code": code,
            "declared_contract_sha256": declared_contract_hash,
            "observed_contract_sha256": observed_contract_hash,
            "fidelity_config_source_sha256": fidelity.get("source_content_sha256"),
            "construction_config_source_sha256": construction.get(
                "source_content_sha256"
            ),
        }
    else:
        details = {}
        _append_once(issues, "e0_contract_manifest_invalid")
    return {
        "valid": bool(isinstance(payload, dict) and checks and all(checks.values())),
        "path": _relative(contract_path),
        "artifact": artifacts[0] if artifacts else None,
        "checks": checks,
        "validity_reasons": sorted(set(issues)),
        **details,
    }


def _finite_metric(value: Any, *, metric: str, issues: list[str]) -> float | None:
    if isinstance(value, bool):
        _append_once(issues, f"invalid_{metric}")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        _append_once(issues, f"invalid_{metric}")
        return None
    if not math.isfinite(number):
        _append_once(issues, f"nonfinite_{metric}")
        return None
    if metric == "psnr" and number < 0:
        _append_once(issues, "invalid_psnr")
        return None
    if metric == "ssim" and not -1.0 <= number <= 1.0:
        _append_once(issues, "invalid_ssim")
        return None
    if metric == "lpips" and number < 0:
        _append_once(issues, "invalid_lpips")
        return None
    return number


def _declared_hash(value: Any, *, role: str, issues: list[str]) -> str | None:
    text = str(value).lower() if value is not None else ""
    if SHA256_RE.fullmatch(text) is None:
        _append_once(issues, f"invalid_{role}_sha256")
        return None
    return text


def _safe_relative_output(value: Any, *, issues: list[str]) -> str | None:
    if not isinstance(value, str) or not value:
        _append_once(issues, "invalid_output_artifact_path")
        return None
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        _append_once(issues, "unsafe_output_artifact_path")
        return None
    normalized = str(pure)
    if normalized != value or "\\" in value:
        _append_once(issues, "noncanonical_output_artifact_path")
        return None
    return normalized


def _plain_unique_frame_names(
    value: Any, *, role: str, issues: list[str], require_nonempty: bool = True,
) -> list[str]:
    """Validate an auditable frame-name set without accepting path aliases."""
    if not isinstance(value, list) or (require_nonempty and not value):
        _append_once(issues, f"{role}_missing_or_invalid")
        return []
    if any(
        not isinstance(frame, str)
        or not frame
        or frame in {".", ".."}
        or Path(frame).name != frame
        or "/" in frame
        or "\\" in frame
        for frame in value
    ):
        _append_once(issues, f"{role}_contains_invalid_frame_name")
        return []
    if len(value) != len(set(value)):
        _append_once(issues, f"{role}_contains_duplicates")
        return []
    return list(value)


def _validate_png(
    path: Path, *, width: int, height: int, role: str, issues: list[str],
) -> dict[str, Any] | None:
    if not _safe_file(path, role=role, issues=issues):
        return None
    try:
        with Image.open(path) as image:
            image.load()
            actual_format = image.format
            actual_size = image.size
            actual_mode = image.mode
    except (OSError, ValueError) as exc:
        _append_once(issues, f"malformed_{role}:{type(exc).__name__}")
        return None
    if actual_format != "PNG":
        _append_once(issues, f"{role}_is_not_png")
    if actual_size != (width, height):
        _append_once(issues, f"{role}_resolution_mismatch")
    if actual_mode != "RGB":
        _append_once(issues, f"{role}_is_not_rgb")
    return _artifact(path, role=role)


def _validate_construction_source(
    payload: Any,
    *,
    expected_source: Path,
    role: str,
    issues: list[str],
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        _append_once(issues, f"invalid_{role}_construction_source")
        return artifacts
    if payload.get("source") != _relative(expected_source):
        _append_once(issues, f"{role}_construction_source_path_mismatch")
    ids = payload.get("accepted_object_ids")
    count = payload.get("n_objects_composited")
    if (
        not isinstance(ids, list)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in ids)
        or len(ids) != len(set(ids))
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count != len(ids)
    ):
        _append_once(issues, f"{role}_accepted_object_population_invalid")
    declared = payload.get("artifacts")
    if not isinstance(declared, list) or not declared:
        _append_once(issues, f"{role}_construction_artifacts_missing")
        return artifacts
    seen: set[str] = set()
    for index, item in enumerate(declared):
        item_role = f"{role}_construction_artifact_{index}"
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            _append_once(issues, f"invalid_{item_role}")
            continue
        try:
            source_path = _repo_path(item["path"], label=item_role, must_exist=True)
        except InventoryError:
            _append_once(issues, f"{item_role}_outside_repository")
            continue
        source_rel = _relative(source_path)
        if source_rel in seen:
            _append_once(issues, f"duplicate_{role}_construction_artifact")
            continue
        seen.add(source_rel)
        if not _safe_file(source_path, role=item_role, issues=issues):
            continue
        observed = _artifact(source_path, role=item_role)
        artifacts.append(observed)
        if item.get("size_bytes") != observed["size_bytes"]:
            _append_once(issues, f"{item_role}_size_mismatch")
        declared_sha = _declared_hash(item.get("sha256"), role=item_role, issues=issues)
        if declared_sha is not None and declared_sha != observed["sha256"]:
            _append_once(issues, f"{item_role}_hash_mismatch")
    return artifacts


def _validate_replacement_artifact(
    payload: Any,
    *,
    bundle_root: Path,
    role: str,
    issues: list[str],
    source_artifacts: list[dict[str, Any]],
) -> str | None:
    if not isinstance(payload, dict):
        _append_once(issues, f"invalid_{role}_replacement_artifact")
        return None
    relative = _safe_relative_output(payload.get("path"), issues=issues)
    if relative is None:
        _append_once(issues, f"invalid_{role}_replacement_artifact_path")
        return None
    path = bundle_root / relative
    if not _safe_file(path, role=role, issues=issues):
        return relative
    observed = _artifact(path, role=role)
    source_artifacts.append(observed)
    if payload.get("size_bytes") != observed["size_bytes"]:
        _append_once(issues, f"{role}_replacement_artifact_size_mismatch")
    declared_sha = _declared_hash(
        payload.get("sha256"), role=f"{role}_replacement_artifact", issues=issues,
    )
    if declared_sha is not None and declared_sha != observed["sha256"]:
        _append_once(issues, f"{role}_replacement_artifact_hash_mismatch")
    return relative


def _validate_legacy_replacement_artifact(
    payload: Any,
    *,
    role: str,
    issues: list[str],
    source_artifacts: list[dict[str, Any]],
) -> None:
    if not isinstance(payload, dict) or not isinstance(payload.get("path"), str):
        _append_once(issues, f"invalid_{role}_legacy_replacement_artifact")
        return
    try:
        path = _repo_path(payload["path"], label=role, must_exist=True)
    except InventoryError:
        _append_once(issues, f"{role}_legacy_replacement_artifact_outside_repository")
        return
    if not _safe_file(path, role=role, issues=issues):
        return
    observed = _artifact(path, role=role)
    source_artifacts.append(observed)
    if payload.get("size_bytes") != observed["size_bytes"]:
        _append_once(issues, f"{role}_legacy_replacement_artifact_size_mismatch")
    declared_sha = _declared_hash(
        payload.get("sha256"), role=f"{role}_legacy_replacement_artifact", issues=issues,
    )
    if declared_sha is not None and declared_sha != observed["sha256"]:
        _append_once(issues, f"{role}_legacy_replacement_artifact_hash_mismatch")


def _declared_tree_closure(
    files: Any, *, role: str, issues: list[str],
) -> tuple[int, str] | None:
    """Recompute the frozen tree identity from a manifest's complete file list."""
    if not isinstance(files, list) or not files:
        _append_once(issues, f"{role}_model_tree_files_missing")
        return None
    rows: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            _append_once(issues, f"{role}_model_tree_file_{index}_invalid")
            continue
        relative = item.get("relative_path")
        size = item.get("size_bytes")
        digest = str(item.get("sha256", "")).lower()
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or not relative
            or pure.is_absolute()
            or relative in {".", ".."}
            or ".." in pure.parts
            or "\\" in relative
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or SHA256_RE.fullmatch(digest) is None
        ):
            _append_once(issues, f"{role}_model_tree_file_{index}_invalid")
            continue
        normalized = pure.as_posix()
        if normalized in seen:
            _append_once(issues, f"{role}_model_tree_file_duplicate")
            continue
        seen.add(normalized)
        rows.append((normalized, size, digest))
    if len(rows) != len(files):
        return None
    digest = hashlib.sha256()
    for relative, size, file_sha in sorted(rows):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\n")
    return len(rows), digest.hexdigest()


def _validate_replacement_model_inputs(
    payload: Any,
    *,
    expected: dict[str, Any],
    role: str,
    issues: list[str],
) -> None:
    """Bind replacement model provenance to config and its internal hash closure."""
    if not isinstance(payload, dict):
        _append_once(issues, f"{role}_replacement_model_inputs_missing")
        return
    declared = payload.get("declared")
    observed = payload.get("observed")
    if declared != expected:
        _append_once(issues, f"{role}_replacement_model_inputs_config_mismatch")
    if not isinstance(observed, dict) or set(observed) != set(expected):
        _append_once(issues, f"{role}_replacement_model_inputs_observed_set_mismatch")
        return
    tree_names = {
        "trellis_source", "trellis_snapshot", "dinov2_source", "sam3_source",
    }
    for name, frozen in expected.items():
        record = observed.get(name)
        item_role = f"{role}_replacement_{name}"
        if not isinstance(record, dict):
            _append_once(issues, f"{item_role}_model_input_invalid")
            continue
        if record.get("path") != frozen.get("path"):
            _append_once(issues, f"{item_role}_path_mismatch")
        if name in tree_names:
            closure = _declared_tree_closure(
                record.get("files"), role=item_role, issues=issues,
            )
            if closure is None:
                continue
            count, tree_sha = closure
            if (
                record.get("file_count") != frozen.get("file_count")
                or count != frozen.get("file_count")
            ):
                _append_once(issues, f"{item_role}_file_count_mismatch")
            if (
                record.get("tree_sha256") != frozen.get("tree_sha256")
                or tree_sha != frozen.get("tree_sha256")
                or record.get("source_identity")
                != f"tree-sha256:{frozen.get('tree_sha256')}"
            ):
                _append_once(issues, f"{item_role}_tree_hash_mismatch")
            if name == "trellis_source":
                for field in (
                    "upstream_repository", "upstream_commit", "flexicubes_commit",
                ):
                    if record.get(field) != frozen.get(field):
                        _append_once(issues, f"{item_role}_{field}_mismatch")
            elif name == "sam3_source":
                for field in (
                    "upstream_repository", "upstream_commit", "package_version",
                ):
                    if record.get(field) != frozen.get(field):
                        _append_once(issues, f"{item_role}_{field}_mismatch")
            elif name == "trellis_snapshot":
                if record.get("upstream_revision") != frozen.get("upstream_revision"):
                    _append_once(issues, f"{item_role}_upstream_revision_mismatch")
            else:
                for field in (
                    "upstream_repository", "upstream_commit", "upstream_commit_reason",
                ):
                    if record.get(field) != frozen.get(field):
                        _append_once(issues, f"{item_role}_{field}_mismatch")
                expected_provenance = (
                    "pinned" if frozen.get("upstream_commit")
                    else "unavailable_from_gitless_torch_hub_snapshot"
                )
                if record.get("upstream_commit_provenance") != expected_provenance:
                    _append_once(issues, f"{item_role}_commit_provenance_mismatch")
        else:
            for field in ("size_bytes", "sha256"):
                if record.get(field) != frozen.get(field):
                    _append_once(issues, f"{item_role}_{field}_mismatch")
            if name == "sam3_checkpoint" and (
                record.get("upstream_revision") != frozen.get("upstream_revision")
            ):
                _append_once(issues, f"{item_role}_upstream_revision_mismatch")

    runtime = payload.get("runtime")
    sam3_runtime = runtime.get("sam3") if isinstance(runtime, dict) else None
    sam3_source = expected.get("sam3_source", {})
    sam3_observed = observed.get("sam3_source", {})
    if not isinstance(sam3_runtime, dict):
        _append_once(issues, f"{role}_replacement_sam3_runtime_missing")
        return
    module_file = sam3_runtime.get("module_file")
    source_prefix = str(sam3_source.get("path", "")).rstrip("/") + "/"
    if (
        sam3_runtime.get("package_version") != sam3_source.get("package_version")
        or not isinstance(module_file, str)
        or not module_file.startswith(source_prefix)
    ):
        _append_once(issues, f"{role}_replacement_sam3_runtime_resolution_mismatch")
        return
    module_relative = module_file[len(source_prefix):]
    source_files = {
        item.get("relative_path"): item
        for item in sam3_observed.get("files", [])
        if isinstance(item, dict)
    } if isinstance(sam3_observed, dict) else {}
    source_record = source_files.get(module_relative)
    if (
        not isinstance(source_record, dict)
        or sam3_runtime.get("module_size_bytes") != source_record.get("size_bytes")
        or sam3_runtime.get("module_sha256") != source_record.get("sha256")
    ):
        _append_once(issues, f"{role}_replacement_sam3_runtime_hash_mismatch")


def _validate_replacement_dataset_inputs(
    payload: Any,
    *,
    bundle_root: Path,
    official_train_frames: set[str],
    room_split_sha256: str | None,
    room_camera_artifacts: Any,
    scene_id: str,
    role: str,
    issues: list[str],
    source_artifacts: list[dict[str, Any]],
) -> list[str]:
    """Re-open every dataset artifact consumed by replacement preparation."""
    expected_keys = {
        "split", "intrinsics", "poses", "mesh", "segments", "segments_anno",
        "read_frames_control", "read_images",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        _append_once(issues, f"{role}_replacement_dataset_inputs_invalid")
        return []
    observed_static: dict[str, dict[str, Any]] = {}
    declared_paths: dict[str, Path] = {}
    for name in ("split", "intrinsics", "poses", "mesh", "segments", "segments_anno"):
        record = payload.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            _append_once(issues, f"{role}_replacement_dataset_{name}_invalid")
            continue
        path = Path(record["path"])
        declared_paths[name] = Path(os.path.abspath(path))
        observed = _external_artifact(
            path, role=f"{role}_replacement_dataset_{name}", issues=issues,
        )
        if observed is None:
            _append_once(issues, f"{role}_replacement_dataset_{name}_path_invalid")
            continue
        source_artifacts.append(observed)
        observed_static[name] = observed
        if (
            record.get("size_bytes") != observed["size_bytes"]
            or record.get("sha256") != observed["sha256"]
        ):
            _append_once(issues, f"{role}_replacement_dataset_{name}_hash_mismatch")
    split_path = declared_paths.get("split")
    dataset_scene_root = split_path.parents[1] if split_path is not None else None
    expected_relative_paths = {
        "split": "dslr/train_test_lists.json",
        "intrinsics": "dslr/nerfstudio/transforms_undistorted.json",
        "poses": "dslr/colmap/images.txt",
        "mesh": "scans/mesh_aligned_0.05.ply",
        "segments": "scans/segments.json",
        "segments_anno": "scans/segments_anno.json",
    }
    if dataset_scene_root is None or dataset_scene_root.name != scene_id:
        _append_once(issues, f"{role}_replacement_dataset_scene_root_mismatch")
    else:
        for name, relative in expected_relative_paths.items():
            if declared_paths.get(name) != dataset_scene_root / relative:
                _append_once(issues, f"{role}_replacement_dataset_{name}_path_mismatch")
    split_record = payload.get("split")
    if not isinstance(split_record, dict) or split_record.get("sha256") != room_split_sha256:
        _append_once(issues, f"{role}_replacement_dataset_split_room_mismatch")
    room_cameras = room_camera_artifacts if isinstance(room_camera_artifacts, dict) else {}
    for dataset_name, room_name in (("intrinsics", "intrinsics"), ("poses", "poses")):
        room_record = room_cameras.get(room_name)
        dataset_record = payload.get(dataset_name)
        if (
            not isinstance(room_record, dict)
            or not isinstance(dataset_record, dict)
            or dataset_record.get("sha256") != room_record.get("sha256")
        ):
            _append_once(
                issues, f"{role}_replacement_dataset_{dataset_name}_room_mismatch",
            )

    control = payload.get("read_frames_control")
    relative = _validate_replacement_artifact(
        control,
        bundle_root=bundle_root,
        role=f"{role}_replacement_read_frames_control",
        issues=issues,
        source_artifacts=source_artifacts,
    )
    if relative != "provenance/factory_read_frames.json":
        _append_once(issues, f"{role}_replacement_read_frames_control_path_mismatch")
        read_frames: list[str] = []
    else:
        payload_frames = _load_json(
            bundle_root / relative,
            role=f"{role}_replacement_read_frames",
            issues=issues,
            artifacts=[],
        )
        read_frames = _plain_unique_frame_names(
            payload_frames.get("frames") if isinstance(payload_frames, dict) else None,
            role=f"{role}_replacement_read_frames",
            issues=issues,
        )
    if not set(read_frames) <= official_train_frames:
        _append_once(issues, f"{role}_replacement_dataset_reads_not_official_train")

    image_rows = payload.get("read_images")
    if not isinstance(image_rows, list) or len(image_rows) != len(read_frames):
        _append_once(issues, f"{role}_replacement_read_image_population_mismatch")
        image_rows = image_rows if isinstance(image_rows, list) else []
    seen_frames: set[str] = set()
    for index, record in enumerate(image_rows):
        frame = record.get("frame") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or frame not in read_frames
            or frame in seen_frames
            or not isinstance(record.get("path"), str)
        ):
            _append_once(issues, f"{role}_replacement_read_image_{index}_invalid")
            continue
        seen_frames.add(frame)
        path = Path(record["path"])
        if (
            dataset_scene_root is not None
            and Path(os.path.abspath(path))
            != dataset_scene_root / "dslr/resized_undistorted_images" / frame
        ):
            _append_once(issues, f"{role}_replacement_read_image_{index}_path_mismatch")
        observed = _external_artifact(
            path, role=f"{role}_replacement_read_image_{index}", issues=issues,
        )
        if observed is None:
            _append_once(issues, f"{role}_replacement_read_image_{index}_path_invalid")
            continue
        source_artifacts.append(observed)
        if (
            record.get("size_bytes") != observed["size_bytes"]
            or record.get("sha256") != observed["sha256"]
        ):
            _append_once(issues, f"{role}_replacement_read_image_{index}_hash_mismatch")
    if seen_frames != set(read_frames):
        _append_once(issues, f"{role}_replacement_read_image_set_mismatch")
    return read_frames


def _validate_replacement_declaration(
    declaration: Any,
    *,
    expected: dict[str, Any] | None,
    outputs_root: Path,
    freeze_id: str,
    role: str,
    official_train_frames: set[str],
    room_split_sha256: str | None,
    room_camera_artifacts: Any,
    room_export_commit: str | None,
    config_file: Path,
    config_payload: dict[str, Any],
    contract_file: Path,
    issues: list[str],
) -> dict[str, Any]:
    """Validate one replacement declaration and its complete published tree."""
    result: dict[str, Any] = {
        "new_frames": [], "old_frames": [], "targets": [], "artifacts": [],
    }
    if expected is None:
        if declaration is not None:
            _append_once(issues, f"unexpected_{role}_replacement_bundle")
        return result
    if not isinstance(declaration, dict):
        _append_once(issues, f"missing_{role}_replacement_bundle")
        return result
    bundle_id = expected["bundle_id"]
    expected_manifest = (
        outputs_root / "icra2027" / freeze_id / "fidelity" / "replacements"
        / bundle_id / "replacement_manifest.json"
    )
    manifest_path_value = declaration.get("manifest_path")
    if not isinstance(manifest_path_value, str):
        _append_once(issues, f"{role}_replacement_manifest_path_missing")
        return result
    try:
        manifest_path = _repo_path(
            manifest_path_value,
            label=f"{role}_replacement_manifest",
            must_exist=True,
        )
    except InventoryError:
        _append_once(issues, f"{role}_replacement_manifest_outside_repository")
        return result
    if manifest_path != expected_manifest.resolve(strict=False):
        _append_once(issues, f"{role}_replacement_manifest_path_mismatch")
    manifest_artifacts: list[dict[str, Any]] = []
    replacement = _load_json(
        manifest_path,
        role=f"{role}_replacement_manifest",
        issues=issues,
        artifacts=manifest_artifacts,
    )
    result["artifacts"].extend(manifest_artifacts)
    if not isinstance(replacement, dict):
        return result
    observed_manifest_sha = (
        manifest_artifacts[0]["sha256"] if manifest_artifacts else None
    )
    declared_manifest_sha = _declared_hash(
        declaration.get("manifest_sha256"),
        role=f"{role}_replacement_manifest",
        issues=issues,
    )
    if declared_manifest_sha is not None and declared_manifest_sha != observed_manifest_sha:
        _append_once(issues, f"{role}_replacement_manifest_hash_mismatch")
    if declaration.get("bundle_id") != bundle_id:
        _append_once(issues, f"{role}_replacement_bundle_id_mismatch")
    fixed_fields = {
        "schema_version": 1,
        "task": "E2 leakage remediation",
        "freeze_id": freeze_id,
        "bundle_id": bundle_id,
        "dataset_id": EXPECTED_DATASET,
        "split_id": EXPECTED_POPULATION_SPLIT,
        "scene_id": expected["scene_id"],
        "mode": expected["mode"],
        "legacy_source": expected["legacy_source"],
        "legacy_source_immutable": True,
        "paper_ready": False,
    }
    for field, value in fixed_fields.items():
        if replacement.get(field) != value:
            _append_once(issues, f"{role}_replacement_{field}_mismatch")
    replacement_git = replacement.get("git")
    if not isinstance(replacement_git, dict):
        _append_once(issues, f"{role}_replacement_git_missing")
    else:
        replacement_commit = str(replacement_git.get("commit", "")).lower()
        if (
            COMMIT_RE.fullmatch(replacement_commit) is None
            or replacement_commit != room_export_commit
        ):
            _append_once(issues, f"{role}_replacement_git_commit_mismatch")
        if replacement_git.get("dirty") is not False:
            _append_once(issues, f"{role}_replacement_git_dirty")

    expected_config = {
        "path": _relative(config_file),
        "size_bytes": config_file.stat().st_size,
        "sha256": _sha256_file(config_file),
        "canonical_sha256": _canonical_hash(config_payload),
    }
    if replacement.get("config_artifact") != expected_config:
        _append_once(issues, f"{role}_replacement_config_provenance_mismatch")
    _validate_replacement_model_inputs(
        replacement.get("model_inputs"),
        expected=expected["model_inputs"],
        role=role,
        issues=issues,
    )
    if replacement.get("stage_parameters") != EXPECTED_STAGE_PARAMETERS:
        _append_once(issues, f"{role}_replacement_stage_parameters_mismatch")
    contract_payload = _load_json(
        contract_file,
        role=f"{role}_replacement_contract_source",
        issues=issues,
        artifacts=result["artifacts"],
    )
    expected_contract_hash = (
        contract_payload.get("contract_sha256")
        if isinstance(contract_payload, dict) else None
    )
    contract_record = replacement.get("contract_artifact")
    if not isinstance(contract_record, dict):
        _append_once(issues, f"{role}_replacement_contract_provenance_missing")
    else:
        try:
            declared_contract_path = Path(str(contract_record.get("path", ""))).resolve()
        except OSError:
            declared_contract_path = Path("/")
        if (
            declared_contract_path != contract_file.resolve()
            or contract_record.get("sha256") != _sha256_file(contract_file)
            or contract_record.get("contract_sha256") != expected_contract_hash
        ):
            _append_once(issues, f"{role}_replacement_contract_provenance_mismatch")
    replacement_split = replacement.get("official_split_artifact")
    if (
        not isinstance(replacement_split, dict)
        or replacement_split.get("sha256") != room_split_sha256
    ):
        _append_once(issues, f"{role}_replacement_split_hash_mismatch")

    bundle_root = manifest_path.parent
    dataset_read_frames = _validate_replacement_dataset_inputs(
        replacement.get("dataset_inputs"),
        bundle_root=bundle_root,
        official_train_frames=official_train_frames,
        room_split_sha256=room_split_sha256,
        room_camera_artifacts=room_camera_artifacts,
        scene_id=expected["scene_id"],
        role=role,
        issues=issues,
        source_artifacts=result["artifacts"],
    )

    manifest_targets = replacement.get("targets")
    declared_targets = declaration.get("targets")
    if not isinstance(manifest_targets, list) or declared_targets != manifest_targets:
        _append_once(issues, f"{role}_replacement_embedded_targets_mismatch")
        manifest_targets = manifest_targets if isinstance(manifest_targets, list) else []
    if len(manifest_targets) != len(expected["targets"]):
        _append_once(issues, f"{role}_replacement_target_population_mismatch")
    declared_output_paths: set[str] = {"replacement_manifest.json"}
    expected_by_object = {
        target["gt_object_id"]: target for target in expected["targets"]
    }
    seen_objects: set[int] = set()
    new_frames: list[str] = []
    old_frames: list[str] = []
    for position, target in enumerate(manifest_targets):
        target_role = f"{role}_replacement_target_{position}"
        if not isinstance(target, dict):
            _append_once(issues, f"invalid_{target_role}")
            continue
        object_id = target.get("gt_object_id")
        expected_target = expected_by_object.get(object_id)
        if expected_target is None or object_id in seen_objects:
            _append_once(issues, f"{target_role}_identity_mismatch")
            continue
        seen_objects.add(object_id)
        for field in ("legacy_index", "gt_object_id", "label", "old_test_frame"):
            if target.get(field) != expected_target[field]:
                _append_once(issues, f"{target_role}_{field}_mismatch")
        new_frame_list = _plain_unique_frame_names(
            target.get("optimization_input_frames"),
            role=f"{target_role}_optimization_input_frames",
            issues=issues,
        )
        new_frame = target.get("new_train_frame")
        if new_frame_list != [new_frame]:
            _append_once(issues, f"{target_role}_new_frame_list_mismatch")
        if new_frame not in official_train_frames:
            _append_once(issues, f"{target_role}_new_frame_not_official_train")
        if target.get("old_test_frame") == new_frame:
            _append_once(issues, f"{target_role}_old_frame_reused")
        if target.get("alignment_tier") not in {"A", "B"}:
            _append_once(issues, f"{target_role}_alignment_not_accepted")
        if isinstance(new_frame, str):
            new_frames.append(new_frame)
        if isinstance(target.get("old_test_frame"), str):
            old_frames.append(target["old_test_frame"])
        legacy_artifacts = target.get("legacy_artifacts")
        if not isinstance(legacy_artifacts, list) or not legacy_artifacts:
            _append_once(issues, f"{target_role}_legacy_artifacts_missing")
        else:
            for artifact_index, artifact in enumerate(legacy_artifacts):
                _validate_legacy_replacement_artifact(
                    artifact,
                    role=f"{target_role}_legacy_artifact_{artifact_index}",
                    issues=issues,
                    source_artifacts=result["artifacts"],
                )
        generated = target.get("generated_artifacts")
        if not isinstance(generated, list) or not generated:
            _append_once(issues, f"{target_role}_generated_artifacts_missing")
        else:
            expected_prefix = f"objects/obj_{expected_target['legacy_index']:02d}/"
            observed_target_paths: set[str] = set()
            for artifact_index, artifact in enumerate(generated):
                relative = _validate_replacement_artifact(
                    artifact,
                    bundle_root=bundle_root,
                    role=f"{target_role}_generated_artifact_{artifact_index}",
                    issues=issues,
                    source_artifacts=result["artifacts"],
                )
                if relative is not None:
                    if not relative.startswith(expected_prefix):
                        _append_once(issues, f"{target_role}_artifact_wrong_object_dir")
                    if relative in declared_output_paths:
                        _append_once(issues, f"{role}_replacement_artifact_duplicate")
                    declared_output_paths.add(relative)
                    observed_target_paths.add(relative)
            expected_target_paths = {
                expected_prefix + name
                for name in (
                    "aligned.json", "gt_points.ply", "mesh_sim.obj", "mesh_sim.ply",
                    "meta.json", "rgba.png", "trellis_gs.ply", "trellis_mesh.ply",
                )
            }
            if observed_target_paths != expected_target_paths:
                _append_once(issues, f"{target_role}_artifact_set_mismatch")

    if seen_objects != set(expected_by_object):
        _append_once(issues, f"{role}_replacement_exact_target_set_mismatch")
    optimization_union = _plain_unique_frame_names(
        replacement.get("optimization_input_frames"),
        role=f"{role}_replacement_optimization_input_frames",
        issues=issues,
    )
    if optimization_union != sorted(set(new_frames)):
        _append_once(issues, f"{role}_replacement_optimization_union_mismatch")
    if not set(optimization_union) <= official_train_frames:
        _append_once(issues, f"{role}_replacement_optimization_not_official_train")
    if not set(optimization_union) <= set(dataset_read_frames):
        _append_once(issues, f"{role}_replacement_selected_frames_not_recorded_as_read")

    controls = replacement.get("bundle_artifacts")
    if not isinstance(controls, list) or len(controls) != 5:
        _append_once(issues, f"{role}_replacement_control_artifact_population_mismatch")
        controls = controls if isinstance(controls, list) else []
    train_allowlist_path: Path | None = None
    for index, artifact in enumerate(controls):
        relative = _validate_replacement_artifact(
            artifact,
            bundle_root=bundle_root,
            role=f"{role}_replacement_control_artifact_{index}",
            issues=issues,
            source_artifacts=result["artifacts"],
        )
        if relative is not None:
            if relative in declared_output_paths:
                _append_once(issues, f"{role}_replacement_artifact_duplicate")
            declared_output_paths.add(relative)
            if relative == "provenance/official_train_frames.json":
                train_allowlist_path = bundle_root / relative
    if {
        artifact.get("path") for artifact in controls if isinstance(artifact, dict)
    } != {
        "provenance/official_train_frames.json",
        "provenance/preserved_index_by_gt_object_id.json",
        "provenance/factory_read_frames.json",
        "objects/objects.json",
        "objects/aligned_all.json",
    }:
        _append_once(issues, f"{role}_replacement_control_artifact_set_mismatch")
    if train_allowlist_path is None:
        _append_once(issues, f"{role}_replacement_train_allowlist_missing")
    else:
        allowlist_artifacts: list[dict[str, Any]] = []
        allowlist = _load_json(
            train_allowlist_path,
            role=f"{role}_replacement_train_allowlist",
            issues=issues,
            artifacts=allowlist_artifacts,
        )
        if not isinstance(allowlist, dict):
            _append_once(issues, f"{role}_replacement_train_allowlist_invalid")
        else:
            stored_train = _plain_unique_frame_names(
                allowlist.get("frames"),
                role=f"{role}_replacement_stored_train_frames",
                issues=issues,
            )
            if stored_train != sorted(official_train_frames) \
                    and set(stored_train) != official_train_frames:
                _append_once(issues, f"{role}_replacement_train_allowlist_mismatch")
            if replacement.get("official_train_frame_count") != len(stored_train):
                _append_once(issues, f"{role}_replacement_train_count_mismatch")
            if replacement.get("official_train_frames_sha256") != _canonical_hash(stored_train):
                _append_once(issues, f"{role}_replacement_train_hash_mismatch")

    auto_record = replacement.get("auto_instances")
    if expected["mode"] == "auto":
        relative = _validate_replacement_artifact(
            auto_record,
            bundle_root=bundle_root,
            role=f"{role}_replacement_auto_instances",
            issues=issues,
            source_artifacts=result["artifacts"],
        )
        if relative != "auto_instances.npz":
            _append_once(issues, f"{role}_replacement_auto_instances_path_mismatch")
        if relative is not None:
            declared_output_paths.add(relative)
        source_auto = outputs_root / f"{expected['scene_id']}_auto" / "auto_instances.npz"
        if (
            not _safe_file(
                source_auto,
                role=f"{role}_replacement_auto_instances_source",
                issues=issues,
            )
            or not isinstance(auto_record, dict)
            or auto_record.get("source_path") != _relative(source_auto)
            or auto_record.get("source_sha256") != _sha256_file(source_auto)
        ):
            _append_once(issues, f"{role}_replacement_auto_instances_source_mismatch")
    elif auto_record is not None:
        _append_once(issues, f"unexpected_{role}_replacement_auto_instances")

    actual_output_paths: set[str] = set()
    if bundle_root.is_dir():
        for path in bundle_root.rglob("*"):
            if path.is_symlink():
                _append_once(issues, f"{role}_replacement_tree_contains_symlink")
            elif path.is_file():
                actual_output_paths.add(str(path.relative_to(bundle_root)))
    if actual_output_paths != declared_output_paths:
        _append_once(issues, f"{role}_replacement_artifact_closure_mismatch")
    validation = replacement.get("validation")
    if not isinstance(validation, dict) or not all(
        validation.get(field) is True
        for field in (
            "fixed_target_population", "mapped_by_gt_object_id",
            "preserved_legacy_indices", "generation_frames_official_train_only",
            "all_regenerated_alignments_accepted", "legacy_sources_unchanged",
            "atomic_fresh_publish",
        )
    ) or validation.get("generation_eval_overlap") != [] \
            or validation.get("target_count") != len(expected["targets"]):
        _append_once(issues, f"{role}_replacement_validation_declaration_invalid")
    stages = replacement.get("stages")
    if not isinstance(stages, list) or [
        item.get("name") for item in stages if isinstance(item, dict)
    ] != ["factory_prepare", "factory_refine_masks", "trellis", "factory_align"] \
            or any(not isinstance(item, dict) or item.get("status") != "passed"
                   for item in stages):
        _append_once(issues, f"{role}_replacement_stage_provenance_invalid")
    result.update(
        new_frames=sorted(set(new_frames)),
        old_frames=sorted(set(old_frames)),
        targets=manifest_targets,
    )
    return result


def _scan_room_scene(
    room_runs_root: Path,
    outputs_root: Path,
    scene_id: str,
    freeze_id: str,
    *,
    remediation: dict[tuple[str, str], dict[str, Any]],
    config_file: Path,
    config_payload: dict[str, Any],
    contract_file: Path,
) -> dict[str, Any]:
    scene_root = room_runs_root / scene_id
    manifest_path = scene_root / "manifest.json"
    issues: list[str] = []
    source_artifacts: list[dict[str, Any]] = []
    payload = _load_json(
        manifest_path,
        role="room_export_manifest",
        issues=issues,
        artifacts=source_artifacts,
    )
    records: dict[str, list[dict[str, Any]]] = {
        method: [] for method in AVAILABLE_ROOM_METHOD_IDS
    }
    if not isinstance(payload, dict):
        return {
            "scene_id": scene_id,
            "valid": False,
            "issues": sorted(issues),
            "records": records,
            "source_artifacts": source_artifacts,
            "manifest": None,
        }

    fixed_fields = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "dataset_id": EXPECTED_DATASET,
        "split_id": EXPECTED_EXPORT_SPLIT,
        "scene_id": scene_id,
        "evaluation_unit": "held_out_view",
        "n_eval_frames": EXPECTED_VIEWS_PER_SCENE,
        "methods": list(AVAILABLE_ROOM_METHOD_IDS),
    }
    for field, expected in fixed_fields.items():
        if payload.get(field) != expected:
            _append_once(issues, f"room_export_{field}_mismatch")

    frames = payload.get("eval_frames")
    if (
        not isinstance(frames, list)
        or len(frames) != EXPECTED_VIEWS_PER_SCENE
        or len(set(map(str, frames))) != EXPECTED_VIEWS_PER_SCENE
        or any(
            not isinstance(frame, str)
            or Path(frame).name != frame
            or frame in {".", ".."}
            for frame in frames
        )
    ):
        _append_once(issues, "room_export_eval_frames_invalid")
        frames = []
    stems = [Path(frame).stem for frame in frames]
    if len(set(stems)) != len(stems):
        _append_once(issues, "room_export_frame_stems_collide")

    global_optimization_frames: list[str] | None = None
    if "optimization_input_frames" in payload:
        global_optimization_frames = _plain_unique_frame_names(
            payload.get("optimization_input_frames"),
            role="optimization_input_frames",
            issues=issues,
        )

    width = payload.get("image_width")
    height = payload.get("image_height")
    if (
        not isinstance(width, int) or isinstance(width, bool) or width <= 0
        or not isinstance(height, int) or isinstance(height, bool) or height <= 0
    ):
        _append_once(issues, "room_export_image_dimensions_invalid")
        width = height = 0
    if not isinstance(payload.get("color_space"), str) or not payload["color_space"]:
        _append_once(issues, "room_export_color_space_missing")

    git = payload.get("git")
    if not isinstance(git, dict):
        _append_once(issues, "room_export_git_provenance_missing")
    else:
        if COMMIT_RE.fullmatch(str(git.get("commit", "")).lower()) is None:
            _append_once(issues, "room_export_git_commit_invalid")
        if git.get("dirty") is not False:
            _append_once(issues, "room_export_git_snapshot_dirty")

    split = payload.get("split_artifact")
    split_sha: str | None = None
    if not isinstance(split, dict):
        _append_once(issues, "room_export_split_artifact_missing")
    else:
        split_sha = _declared_hash(
            split.get("sha256"), role="split_artifact", issues=issues,
        )
        if not isinstance(split.get("path"), str) or not split["path"]:
            _append_once(issues, "split_artifact_path_missing")
    official_train_frames = _plain_unique_frame_names(
        payload.get("official_train_frames"),
        role="official_train_frames",
        issues=issues,
    )
    if set(official_train_frames) & set(frames):
        _append_once(issues, "official_train_and_evaluation_frames_overlap")

    camera_artifacts = payload.get("camera_artifacts")
    if not isinstance(camera_artifacts, dict):
        _append_once(issues, "room_export_camera_artifacts_missing")
    else:
        for camera_role in ("intrinsics", "poses"):
            camera = camera_artifacts.get(camera_role)
            if not isinstance(camera, dict):
                _append_once(issues, f"room_export_{camera_role}_artifact_missing")
                continue
            _declared_hash(
                camera.get("sha256"), role=f"camera_{camera_role}", issues=issues,
            )
            if not isinstance(camera.get("path"), str) or not camera["path"]:
                _append_once(issues, f"camera_{camera_role}_path_missing")

    gaussian = payload.get("source_scene_gaussian")
    if not isinstance(gaussian, dict):
        _append_once(issues, "room_export_scene_gaussian_provenance_missing")
    else:
        _declared_hash(gaussian.get("sha256"), role="scene_gaussian", issues=issues)
        if (
            not isinstance(gaussian.get("size_bytes"), int)
            or isinstance(gaussian.get("size_bytes"), bool)
            or gaussian["size_bytes"] <= 0
        ):
            _append_once(issues, "scene_gaussian_size_invalid")
        if not isinstance(gaussian.get("path"), str) or not gaussian["path"]:
            _append_once(issues, "scene_gaussian_path_missing")

    source_images = payload.get("source_images")
    image_by_frame: dict[str, dict[str, Any]] = {}
    if not isinstance(source_images, list) or len(source_images) != len(frames):
        _append_once(issues, "room_export_source_image_population_mismatch")
    else:
        for entry in source_images:
            if not isinstance(entry, dict) or not isinstance(entry.get("frame"), str):
                _append_once(issues, "room_export_source_image_record_invalid")
                continue
            frame = entry["frame"]
            if frame in image_by_frame:
                _append_once(issues, "room_export_source_image_duplicate")
            image_by_frame[frame] = entry
            _declared_hash(
                entry.get("sha256"), role="source_image", issues=issues,
            )
            if not isinstance(entry.get("path"), str) or not entry["path"]:
                _append_once(issues, "source_image_path_missing")
            if entry.get("exported_png") != f"{Path(frame).stem}.png":
                _append_once(issues, "source_image_export_name_mismatch")
        if set(image_by_frame) != set(frames):
            _append_once(issues, "room_export_source_frames_mismatch")

    declared_outputs = payload.get("output_artifacts")
    declared_by_path: dict[str, str | None] = {}
    if not isinstance(declared_outputs, list):
        _append_once(issues, "room_export_output_artifacts_missing")
    else:
        for item in declared_outputs:
            if not isinstance(item, dict):
                _append_once(issues, "room_export_output_artifact_invalid")
                continue
            relative = _safe_relative_output(item.get("relative_path"), issues=issues)
            if relative is None:
                continue
            if relative in declared_by_path:
                _append_once(issues, "room_export_output_artifact_duplicate")
            declared_by_path[relative] = _declared_hash(
                item.get("sha256"), role="output_artifact", issues=issues,
            )

    expected_paths = {
        f"{directory}/{stem}.png"
        for directory in ("gt", *AVAILABLE_ROOM_METHOD_IDS)
        for stem in stems
    }
    if set(declared_by_path) != expected_paths:
        _append_once(issues, "room_export_output_artifact_set_mismatch")
    actual_pngs: set[str] = set()
    if scene_root.is_dir() and _first_symlink(scene_root) is None:
        for candidate in scene_root.rglob("*.png"):
            try:
                actual_pngs.add(str(candidate.relative_to(scene_root)))
            except ValueError:
                _append_once(issues, "room_export_png_outside_scene_root")
    if actual_pngs != expected_paths:
        _append_once(issues, "room_export_png_set_mismatch")

    png_artifacts: dict[str, dict[str, Any]] = {}
    if width and height:
        for relative in sorted(expected_paths):
            observed = _validate_png(
                scene_root / relative,
                width=width,
                height=height,
                role="room_export_png",
                issues=issues,
            )
            if observed is None:
                continue
            png_artifacts[relative] = observed
            declared_sha = declared_by_path.get(relative)
            if declared_sha is not None and declared_sha != observed["sha256"]:
                _append_once(issues, "room_export_png_hash_mismatch")
            source_artifacts.append(observed)

    construction = payload.get("construction_sources")
    per_source_optimization: dict[str, list[str]] = {}
    replacement_by_method: dict[str, dict[str, Any]] = {}
    if not isinstance(construction, dict):
        _append_once(issues, "room_export_construction_sources_missing")
    else:
        source_artifacts.extend(_validate_construction_source(
            construction.get("factorized_gt_discovery"),
            expected_source=outputs_root / f"{scene_id}_factory",
            role="factory",
            issues=issues,
        ))
        source_artifacts.extend(_validate_construction_source(
            construction.get("factorized_auto_discovery"),
            expected_source=outputs_root / f"{scene_id}_auto",
            role="auto",
            issues=issues,
        ))
        for method in ("factorized_gt_discovery", "factorized_auto_discovery"):
            source = construction.get(method)
            if isinstance(source, dict) and "optimization_input_frames" in source:
                per_source_optimization[method] = _plain_unique_frame_names(
                    source.get("optimization_input_frames"),
                    role=f"{method}_optimization_input_frames",
                    issues=issues,
                )
            mode = METHOD_TO_REMEDIATION_MODE[method]
            replacement = _validate_replacement_declaration(
                source.get("replacement_bundle") if isinstance(source, dict) else None,
                expected=remediation.get((scene_id, mode)),
                outputs_root=outputs_root,
                freeze_id=freeze_id,
                role=mode,
                official_train_frames=set(official_train_frames),
                room_split_sha256=split_sha,
                room_camera_artifacts=camera_artifacts,
                room_export_commit=(
                    str(git.get("commit", "")).lower()
                    if isinstance(git, dict) else None
                ),
                config_file=config_file,
                config_payload=config_payload,
                contract_file=contract_file,
                issues=issues,
            )
            source_artifacts.extend(replacement["artifacts"])
            replacement_by_method[method] = replacement
            source_frames = set(per_source_optimization.get(method, []))
            if not set(replacement["new_frames"]) <= source_frames:
                _append_once(issues, f"{mode}_replacement_new_frames_missing_from_source")
            if set(replacement["old_frames"]) & source_frames:
                _append_once(issues, f"{mode}_leaked_old_frames_remain_in_source")
            expected_replacement = remediation.get((scene_id, mode))
            if expected_replacement is not None and isinstance(source, dict):
                accepted_ids = source.get("accepted_object_ids")
                expected_indices = {
                    target["legacy_index"]
                    for target in expected_replacement["targets"]
                }
                if (
                    not isinstance(accepted_ids, list)
                    or any(not isinstance(value, int) or isinstance(value, bool)
                           for value in accepted_ids)
                    or not expected_indices <= set(accepted_ids)
                ):
                    _append_once(issues, f"{mode}_replacement_targets_not_composited")

    if global_optimization_frames is None:
        if set(per_source_optimization) != {
            "factorized_gt_discovery", "factorized_auto_discovery",
        }:
            _append_once(issues, "optimization_input_frames_missing")
        optimization_input_frames = sorted({
            frame
            for source_frames in per_source_optimization.values()
            for frame in source_frames
        })
    else:
        optimization_input_frames = sorted(global_optimization_frames)
        per_source_union = {
            frame
            for source_frames in per_source_optimization.values()
            for frame in source_frames
        }
        if not per_source_union <= set(global_optimization_frames):
            _append_once(
                issues,
                "per_source_optimization_frames_absent_from_global_union",
            )
    if not optimization_input_frames:
        _append_once(issues, "optimization_input_frames_empty")
    leaked_frames = sorted(set(optimization_input_frames) & set(frames))
    if leaked_frames:
        _append_once(issues, "optimization_and_evaluation_frames_overlap")
    remediated_old_frames = {
        frame
        for replacement in replacement_by_method.values()
        for frame in replacement["old_frames"]
    }
    if remediated_old_frames & set(optimization_input_frames):
        _append_once(issues, "remediated_old_frames_remain_in_global_optimization_inputs")

    # Only publish canonical records when the whole scene is paired and
    # coherent; partial per-method inclusion would change the denominator.
    if not issues:
        manifest_artifact = source_artifacts[0]
        split_sha = str(split["sha256"])
        gaussian_sha = str(gaussian["sha256"])
        export_commit = str(git["commit"])
        for method in AVAILABLE_ROOM_METHOD_IDS:
            views: list[dict[str, Any]] = []
            for frame, stem in zip(frames, stems):
                gt_relative = f"gt/{stem}.png"
                gt_artifact = png_artifacts[gt_relative]
                render_relative = f"{method}/{stem}.png"
                render_artifact = png_artifacts[render_relative]
                views.append({
                    "view_id": frame,
                    "render_path": _relative(scene_root / render_relative),
                    "gt_path": _relative(scene_root / gt_relative),
                    "render_sha256": render_artifact["sha256"],
                    "gt_sha256": gt_artifact["sha256"],
                    "source_image_sha256": image_by_frame[frame]["sha256"],
                    "exact_duplicate_to_gt": (
                        render_artifact["sha256"] == gt_artifact["sha256"]
                    ),
                })
            if method == "factorized_gt_discovery":
                source_build = _relative(outputs_root / f"{scene_id}_factory")
            elif method == "factorized_auto_discovery":
                source_build = _relative(outputs_root / f"{scene_id}_auto")
            else:
                source_build = str(gaussian["path"])
            records[method].append({
                "freeze_id": freeze_id,
                "dataset_id": EXPECTED_DATASET,
                "split_id": EXPECTED_EXPORT_SPLIT,
                "unit": "held_out_view",
                "scene_id": scene_id,
                "source_build": source_build,
                "render_dir": _relative(scene_root / method),
                "gt_dir": _relative(scene_root / "gt"),
                "mask_dir": None,
                "color_space": payload["color_space"],
                "crop_policy": "full_frame",
                "mask_policy": "none",
                "coverage": {"render_only": [], "gt_only": []},
                "n_views": len(views),
                "evaluation_frames": [f"{stem}.png" for stem in stems],
                "source_eval_frames": list(frames),
                "optimization_input_frames": optimization_input_frames,
                "views": views,
                "source_scene_gaussian_sha256": gaussian_sha,
                "split_artifact_sha256": split_sha,
                "source_scene_manifest_path": manifest_artifact["path"],
                "source_scene_manifest_sha256": manifest_artifact["sha256"],
                "source_build_commit": export_commit,
                "replacement_bundle": (
                    {
                        "manifest_path": construction[method]["replacement_bundle"][
                            "manifest_path"
                        ],
                        "manifest_sha256": construction[method]["replacement_bundle"][
                            "manifest_sha256"
                        ],
                        "bundle_id": construction[method]["replacement_bundle"][
                            "bundle_id"
                        ],
                    }
                    if method in METHOD_TO_REMEDIATION_MODE
                    and remediation.get(
                        (scene_id, METHOD_TO_REMEDIATION_MODE[method])
                    ) is not None
                    else None
                ),
            })

    return {
        "scene_id": scene_id,
        "valid": not issues,
        "issues": sorted(issues),
        "records": records,
        "source_artifacts": sorted(
            source_artifacts, key=lambda item: (item["path"], item["role"])
        ),
        "manifest": {
            "path": _relative(manifest_path),
            "git": payload.get("git"),
            "split_artifact": payload.get("split_artifact"),
            "camera_artifacts": payload.get("camera_artifacts"),
            "source_scene_gaussian": payload.get("source_scene_gaussian"),
            "official_train_frames": payload.get("official_train_frames"),
            "replacement_bundles": {
                method: (
                    construction.get(method, {}).get("replacement_bundle")
                    if isinstance(construction, dict)
                    and isinstance(construction.get(method), dict)
                    else None
                )
                for method in METHOD_TO_REMEDIATION_MODE
            },
        },
    }


def _metric_block(payload: Any, key: str, issues: list[str]) -> dict[str, float] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get(key), dict):
        _append_once(issues, f"legacy_missing_{key}_metric_block")
        return None
    result: dict[str, float] = {}
    for metric in ("psnr", "ssim", "lpips"):
        value = _finite_metric(payload[key].get(metric), metric=metric, issues=issues)
        if value is None:
            return None
        result[metric] = value
    return result


def _scan_legacy_metric_file(path: Path, *, role: str) -> dict[str, Any]:
    issues: list[str] = []
    artifacts: list[dict[str, Any]] = []
    payload = _load_json(path, role=role, issues=issues, artifacts=artifacts)
    result = {
        "path": _relative(path),
        "artifact": artifacts[0] if artifacts else None,
        "valid": False,
        "issues": issues,
        "frames": [],
        "n_objects_composited": None,
    }
    if not isinstance(payload, dict):
        return result
    if payload.get("split") != "official-test":
        _append_once(issues, "legacy_metric_not_official_test")
    names = payload.get("eval_frames")
    rows = payload.get("frames")
    if (
        not isinstance(names, list)
        or not isinstance(rows, list)
        or len(names) != EXPECTED_VIEWS_PER_SCENE
        or len(rows) != EXPECTED_VIEWS_PER_SCENE
        or len(set(map(str, names))) != EXPECTED_VIEWS_PER_SCENE
    ):
        _append_once(issues, "legacy_metric_frame_population_invalid")
        return result
    parsed: list[dict[str, Any]] = []
    for index, (name, row) in enumerate(zip(names, rows)):
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(row, dict)
            or row.get("frame") != name
        ):
            _append_once(issues, f"legacy_metric_frame_{index}_identity_mismatch")
            continue
        bg = _metric_block(row, "bg", issues)
        twin = _metric_block(row, "twin", issues)
        if bg is not None and twin is not None:
            parsed.append({"frame": name, "bg": bg, "twin": twin})
    mean = payload.get("mean")
    for variant in ("bg", "twin"):
        declared = _metric_block(mean, variant, issues)
        if declared is None or len(parsed) != EXPECTED_VIEWS_PER_SCENE:
            continue
        for metric in ("psnr", "ssim", "lpips"):
            observed = math.fsum(row[variant][metric] for row in parsed) / len(parsed)
            if not math.isclose(observed, declared[metric], rel_tol=0.0, abs_tol=1e-12):
                _append_once(issues, f"legacy_{variant}_{metric}_mean_mismatch")
    n_objects = payload.get("n_objects_composited")
    if not isinstance(n_objects, int) or isinstance(n_objects, bool) or n_objects < 0:
        _append_once(issues, "legacy_n_objects_composited_invalid")
    result.update(
        valid=not issues,
        issues=sorted(issues),
        frames=parsed,
        n_objects_composited=n_objects,
    )
    return result


def _audit_room_anchors(
    outputs_root: Path,
    scene_ids: Sequence[str],
    room_scenes: Sequence[dict[str, Any]],
    *,
    full_population: bool,
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    issues: list[str] = [
        "legacy_metric_renders_and_targets_not_archived",
        "legacy_metric_run_commit_unrecorded",
        "legacy_split_source_hash_unrecorded",
    ]
    audit_only_reasons: list[str] = []
    values: dict[str, list[float]] = {
        f"{method}:{metric}": []
        for method in AVAILABLE_ROOM_METHOD_IDS
        for metric in ("psnr", "ssim", "lpips")
    }
    total_objects = {"factorized_gt_discovery": 0, "factorized_auto_discovery": 0}
    fresh_frames = {
        scene["scene_id"]: {
            frame
            for record in scene["records"]["input_scene_gaussian"]
            for frame in record["source_eval_frames"]
        }
        for scene in room_scenes
        if scene["valid"]
    }
    valid_scenes = 0
    fresh_protocol_frames_match = True
    per_scene: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        factory = _scan_legacy_metric_file(
            outputs_root / f"{scene_id}_factory" / "render_metrics_v2.json",
            role="legacy_factory_render_metrics",
        )
        automatic = _scan_legacy_metric_file(
            outputs_root / f"{scene_id}_auto" / "render_metrics_v2.json",
            role="legacy_auto_render_metrics",
        )
        for record in (factory, automatic):
            if record["artifact"] is not None:
                sources.append(record["artifact"])
        scene_issues = list(factory["issues"]) + list(automatic["issues"])
        scene_audit_only: list[str] = []
        factory_frames = factory["frames"]
        auto_frames = automatic["frames"]
        if factory["valid"] and automatic["valid"]:
            factory_names = [row["frame"] for row in factory_frames]
            auto_names = [row["frame"] for row in auto_frames]
            if factory_names != auto_names:
                _append_once(scene_issues, "legacy_factory_auto_frame_pairing_mismatch")
            for left, right in zip(factory_frames, auto_frames):
                for metric in ("psnr", "ssim", "lpips"):
                    if not math.isclose(
                        left["bg"][metric], right["bg"][metric],
                        rel_tol=0.0, abs_tol=1e-12,
                    ):
                        _append_once(scene_issues, "legacy_factory_auto_background_mismatch")
            if scene_id in fresh_frames and set(factory_names) != fresh_frames[scene_id]:
                _append_once(
                    scene_audit_only, "legacy_and_fresh_export_frame_mismatch",
                )
                _append_once(
                    audit_only_reasons, "legacy_and_fresh_export_frame_mismatch",
                )
                fresh_protocol_frames_match = False
        if not scene_issues:
            valid_scenes += 1
            total_objects["factorized_gt_discovery"] += int(
                factory["n_objects_composited"]
            )
            total_objects["factorized_auto_discovery"] += int(
                automatic["n_objects_composited"]
            )
            for row in factory_frames:
                for metric in ("psnr", "ssim", "lpips"):
                    values[f"input_scene_gaussian:{metric}"].append(row["bg"][metric])
                    values[f"factorized_gt_discovery:{metric}"].append(row["twin"][metric])
            for row in auto_frames:
                for metric in ("psnr", "ssim", "lpips"):
                    values[f"factorized_auto_discovery:{metric}"].append(row["twin"][metric])
        else:
            for issue in scene_issues:
                _append_once(issues, issue)
        per_scene.append({
            "scene_id": scene_id,
            "valid": not scene_issues,
            "issues": sorted(set(scene_issues)),
            "audit_only_reasons": sorted(set(scene_audit_only)),
            "factory_path": factory["path"],
            "auto_path": automatic["path"],
        })

    aggregates: dict[str, Any] = {}
    for method in AVAILABLE_ROOM_METHOD_IDS:
        aggregate = {
            metric: (
                math.fsum(values[f"{method}:{metric}"])
                / len(values[f"{method}:{metric}"])
                if values[f"{method}:{metric}"] else None
            )
            for metric in ("psnr", "ssim", "lpips")
        }
        aggregate["n_views"] = len(values[f"{method}:psnr"])
        if method in total_objects:
            aggregate["n_objects_composited"] = total_objects[method]
        aggregates[method] = aggregate

    comparisons: dict[str, Any] = {}
    source_population_complete = full_population and valid_scenes == len(scene_ids)
    metric_values_reproduced = source_population_complete
    for method, expected in KNOWN_ROOM_ANCHORS.items():
        actual = aggregates[method]
        metric_checks = {
            metric: (
                actual[metric] is not None
                and round(actual[metric], 2 if metric == "psnr" else 3) == target
            )
            for metric, target in expected.items()
        }
        comparisons[method] = {
            "expected_rounded": expected,
            "observed": {metric: actual[metric] for metric in expected},
            "checks": metric_checks,
            "metric_values_match": bool(
                source_population_complete and all(metric_checks.values())
            ),
            "reproduced": bool(
                source_population_complete
                and fresh_protocol_frames_match
                and all(metric_checks.values())
            ),
        }
        metric_values_reproduced = (
            metric_values_reproduced and comparisons[method]["metric_values_match"]
        )
    all_reproduced = metric_values_reproduced and fresh_protocol_frames_match
    for reason in audit_only_reasons:
        _append_once(issues, reason)

    return {
        "status": (
            "reproduced_from_metric_only_legacy_sources"
            if all_reproduced
            else (
                "legacy_metric_values_match_but_fresh_view_protocol_not_reproducible"
                if metric_values_reproduced else "not_reproduced"
            )
        ),
        "valid_scene_count": valid_scenes,
        "expected_scene_count": len(scene_ids),
        "all_reproduced": all_reproduced,
        "legacy_metric_values_reproduced": metric_values_reproduced,
        "fresh_protocol_frames_match": fresh_protocol_frames_match,
        "canonical_measurement_input": False,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "validity_reasons": sorted(set(issues)),
        "audit_only_reasons": sorted(set(audit_only_reasons)),
        "source_artifacts": sorted(sources, key=lambda item: item["path"]),
        "per_scene": per_scene,
    }


def _review_number(payload: Any, path: Sequence[str]) -> float | None:
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, bool):
        return None
    try:
        number = float(current)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _audit_object_anchors(
    outputs_root: Path,
    review_path: Path,
    scene_ids: Sequence[str],
    *,
    full_population: bool,
) -> dict[str, Any]:
    issues: list[str] = [
        "held_out_object_renders_and_masks_unavailable",
        "independent_registration_and_evaluation_surfaces_unavailable",
        "gt_points_used_for_candidate_registration_selection_and_evaluation",
        "reviewed_aggregate_does_not_pin_source_hashes",
    ]
    artifacts: list[dict[str, Any]] = []
    review_issues: list[str] = []
    review = _load_json(
        review_path,
        role="reviewed_object_aggregate",
        issues=review_issues,
        artifacts=artifacts,
    )
    issues.extend(review_issues)

    rows: list[dict[str, Any]] = []
    source_scene_count = 0
    source_details: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        path = outputs_root / f"{scene_id}_factory" / "objects" / "hybrid_all.json"
        source_issues: list[str] = []
        payload = _load_json(
            path,
            role="object_hybrid_source",
            issues=source_issues,
            artifacts=artifacts,
        )
        scene_rows: list[Any] = []
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            scene_rows = payload["rows"]
            if payload.get("n_objects") != len(scene_rows):
                _append_once(source_issues, "object_hybrid_n_objects_mismatch")
            for index, row in enumerate(scene_rows):
                if not isinstance(row, dict):
                    _append_once(source_issues, f"object_hybrid_row_{index}_invalid")
                    continue
                criterion = str(row.get("criterion", ""))
                if "gt_points.ply" not in criterion:
                    _append_once(source_issues, "object_hybrid_selection_criterion_changed")
                enriched = dict(row)
                enriched["_scene_id"] = scene_id
                rows.append(enriched)
        elif payload is not None:
            _append_once(source_issues, "object_hybrid_schema_invalid")
        if not source_issues:
            source_scene_count += 1
        else:
            issues.extend(source_issues)
        source_details.append({
            "scene_id": scene_id,
            "path": _relative(path),
            "n_rows": len(scene_rows),
            "valid": not source_issues,
            "issues": sorted(set(source_issues)),
        })

    trellis: list[float] = []
    rvg: list[float] = []
    selected: list[float] = []
    oracle: list[float] = []
    for row in rows:
        trellis_payload = row.get("trellis")
        trellis_value = _review_number(trellis_payload, ("f1_20",))
        if trellis_value is None or not 0.0 <= trellis_value <= 1.0:
            _append_once(issues, "object_trellis_f1_invalid")
            continue
        trellis.append(trellis_value)
        rvg_payload = row.get("rvg")
        rvg_value = _review_number(rvg_payload, ("f1_20",))
        if rvg_value is not None:
            if not 0.0 <= rvg_value <= 1.0:
                _append_once(issues, "object_rvg_f1_invalid")
                rvg_value = None
            else:
                rvg.append(rvg_value)
        winner = row.get("winner")
        if winner == "trellis":
            selected.append(trellis_value)
        elif winner == "rvg" and rvg_value is not None:
            selected.append(rvg_value)
        else:
            _append_once(issues, "object_selected_winner_invalid_or_unscored")
        nonrejected_rvg = (
            rvg_value
            if isinstance(rvg_payload, dict) and not rvg_payload.get("rejected")
            else None
        )
        oracle.append(max(
            value for value in (trellis_value, nonrejected_rvg) if value is not None
        ))

    observed = {
        "trellis_best_single_view": {
            "f1_20": math.fsum(trellis) / len(trellis) if trellis else None,
            "n_objects": len(trellis),
            "catastrophic_collapses": sum(value < 0.1 for value in trellis),
        },
        "reconviagen_multi_view": {
            "f1_20": math.fsum(rvg) / len(rvg) if rvg else None,
            "n_objects": len(rvg),
            "catastrophic_collapses": sum(value < 0.1 for value in rvg),
        },
        "evidence_selected_proposal": {
            "f1_20": math.fsum(selected) / len(selected) if selected else None,
            "n_objects": len(selected),
            "catastrophic_collapses": sum(value < 0.1 for value in selected),
        },
        "evaluation_only_oracle_candidate": {
            "f1_20": math.fsum(oracle) / len(oracle) if oracle else None,
            "n_objects": len(oracle),
            "catastrophic_collapses": sum(value < 0.1 for value in oracle),
        },
    }
    reviewed = {
        "trellis_best_single_view": {
            "f1_20": _review_number(
                review, ("dataset_wide", "pooled_mean_f1_20_trellis_only"),
            ),
            "n_objects": _review_number(review, ("provenance", "n_objects_total")),
            "catastrophic_collapses": None,
        },
        "reconviagen_multi_view": {
            "f1_20": _review_number(
                review, ("dataset_wide", "pooled_mean_f1_20_rvg_all_scored"),
            ),
            "n_objects": _review_number(
                review, ("dataset_wide", "gate_vs_f1_oracle", "n_objects_both_scored"),
            ),
            "catastrophic_collapses": _review_number(
                review, ("dataset_wide", "rvg_collapses_f1_lt_0.1", "n_collapses"),
            ),
        },
        "evidence_selected_proposal": {
            "f1_20": _review_number(
                review, ("dataset_wide", "pooled_mean_f1_20_hybrid"),
            ),
            "n_objects": _review_number(review, ("provenance", "n_objects_total")),
            "catastrophic_collapses": 0.0,
        },
        "evaluation_only_oracle_candidate": {
            "f1_20": _review_number(
                review,
                ("oracle_hybrid_upper_bound", "pooled_mean_f1_20_oracle_nonrejected_candidates"),
            ),
            "n_objects": _review_number(review, ("provenance", "n_objects_total")),
            "catastrophic_collapses": None,
        },
    }

    comparisons: dict[str, Any] = {}
    coherent = bool(
        full_population
        and isinstance(review, dict)
        and source_scene_count == len(scene_ids)
        and not any(
            reason.startswith("object_") or reason.startswith("missing_object")
            for reason in issues
        )
    )
    for method in OBJECT_METHODS.values():
        checks: dict[str, bool] = {}
        for field in ("f1_20", "n_objects", "catastrophic_collapses"):
            expected = reviewed[method][field]
            if expected is None:
                continue
            actual = observed[method][field]
            tolerance = 5e-5 if field == "f1_20" else 0.0
            checks[field] = bool(
                actual is not None
                and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
            )
        comparisons[method] = {
            "reviewed": reviewed[method],
            "current_enumerated_sources": observed[method],
            "checks": checks,
            "matches_reviewed_snapshot": bool(checks and all(checks.values())),
        }
        coherent = coherent and comparisons[method]["matches_reviewed_snapshot"]
    reviewed_scene_count = _review_number(review, ("provenance", "n_scenes"))
    if reviewed_scene_count != float(len(scene_ids)):
        coherent = False
        _append_once(issues, "reviewed_object_scene_count_mismatch")
    if not coherent:
        _append_once(issues, "reviewed_object_snapshot_not_reproducible_from_enumerated_sources")

    canonical = {
        method: (
            {**observed[method], "status": "legacy_leaky_geometry_diagnostic"}
            if coherent
            else {
                "f1_20": None,
                "n_objects": None,
                "catastrophic_collapses": None,
                "status": "unavailable",
                "reason": "reviewed source snapshot cannot be enumerated and reproduced",
            }
        )
        for method in OBJECT_METHODS.values()
    }
    return {
        "status": (
            "reviewed_snapshot_reproduced_but_leakage_invalid"
            if coherent else "unavailable_provenance_or_snapshot_mismatch"
        ),
        "source_snapshot_coherent": coherent,
        "canonical_legacy_diagnostics": canonical,
        "comparisons": comparisons,
        "validity_reasons": sorted(set(issues)),
        "reviewed_aggregate": (
            artifacts[0] if artifacts and artifacts[0]["role"] == "reviewed_object_aggregate"
            else None
        ),
        "source_artifacts": sorted(artifacts, key=lambda item: item["path"]),
        "per_scene": source_details,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_inventory(
    outputs_root: str | os.PathLike[str],
    room_runs_root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    freeze_id: str,
    *,
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG,
    construction_config_path: str | os.PathLike[str] = DEFAULT_CONSTRUCTION_CONFIG,
    reviewed_object_aggregate_path: str | os.PathLike[str] = DEFAULT_OBJECT_REVIEW,
    contract_manifest_path: str | os.PathLike[str] | None = None,
    expected_scenes: int = 50,
    selected_scene_ids: Sequence[str] | None = None,
    smoke: bool = False,
    smoke_limit: int = 2,
    _inject_failure_after: str | None = None,
) -> dict[str, Any]:
    """Validate source evidence and atomically publish the E2 inventory."""
    outputs = _repo_path(outputs_root, label="outputs_root", must_exist=True)
    room_runs = _repo_path(room_runs_root, label="room_runs_root", must_exist=True)
    destination = _repo_path(out_dir, label="out_dir")
    config_file = _repo_path(config_path, label="config", must_exist=True)
    construction_file = _repo_path(
        construction_config_path, label="construction_config", must_exist=True,
    )
    review_file = _repo_path(
        reviewed_object_aggregate_path,
        label="reviewed_object_aggregate",
        must_exist=True,
    )
    if contract_manifest_path is None:
        raise InventoryError("contract_manifest is required")
    contract_file = _repo_path(
        contract_manifest_path, label="contract_manifest", must_exist=True,
    )
    if not outputs.is_dir():
        raise InventoryError(f"outputs_root is not a directory: {outputs}")
    if not room_runs.is_dir():
        raise InventoryError(f"room_runs_root is not a directory: {room_runs}")
    if destination.exists() or destination.is_symlink():
        raise InventoryError(f"refusing to overwrite existing inventory: {destination}")
    if FREEZE_RE.fullmatch(str(freeze_id).strip()) is None:
        raise InventoryError("freeze_id contains unsafe or unsupported characters")
    if expected_scenes <= 0 or isinstance(expected_scenes, bool):
        raise InventoryError("expected_scenes must be a positive integer")
    if smoke_limit <= 0:
        raise InventoryError("smoke_limit must be positive")

    config_bytes = config_file.read_bytes()
    try:
        config = json.loads(config_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot parse fidelity config: {exc}") from exc
    planned = _validate_config(config, expected_scenes)
    remediation = _parse_leakage_remediation(config)
    construction = _load_yaml(construction_file)
    construction_scenes = _config_population(
        construction, label="construction config",
    )
    if planned != construction_scenes:
        raise InventoryError("E2 fidelity roster differs from the pinned E1 construction roster")

    if selected_scene_ids is None:
        selected = planned[:smoke_limit] if smoke else list(planned)
    else:
        selected = _normalize_scenes(selected_scene_ids, label="selected scene IDs")
    outside = sorted(set(selected) - set(planned))
    if outside:
        raise InventoryError(f"selected scenes are outside the planned population: {outside}")
    if not selected:
        raise InventoryError("scene selection is empty")
    partial = selected != planned
    if partial and not smoke:
        raise InventoryError("partial scene selection requires smoke=True")

    inventory_git = _git_snapshot()
    contract_report = _audit_contract(
        contract_file,
        freeze_id=freeze_id,
        smoke=smoke,
        inventory_git=inventory_git,
        config_file=config_file,
        construction_file=construction_file,
    )
    room_scenes = [
        _scan_room_scene(
            room_runs,
            outputs,
            scene_id,
            freeze_id,
            remediation=remediation,
            config_file=config_file,
            config_payload=config,
            contract_file=contract_file,
        )
        for scene_id in selected
    ]
    full_population = not smoke and not partial
    room_anchor_report = _audit_room_anchors(
        outputs, selected, room_scenes, full_population=full_population,
    )
    object_anchor_report = _audit_object_anchors(
        outputs, review_file, selected, full_population=full_population,
    )

    output_manifest = copy.deepcopy(config)
    output_manifest["freeze_id"] = freeze_id
    output_manifest["source_config"] = {
        "path": _relative(config_file),
        "sha256": _sha256_bytes(config_bytes),
    }
    output_manifest["construction_roster_source"] = {
        "path": _relative(construction_file),
        "sha256": _sha256_file(construction_file),
    }
    output_manifest["upstream_e0_contract"] = contract_report
    output_manifest["selection"] = {
        "mode": "smoke" if smoke else "full",
        "planned_scene_ids": planned,
        "selected_scene_ids": selected,
        "full_population_preserved": full_population,
    }
    if smoke:
        # The checked-in config pins the full 50-scene floors. A smoke keeps
        # that planned roster in `selection`, but its executable minima must
        # match the explicitly selected subset so the strict evaluator can
        # exercise the identical code path without pretending it is full.
        output_manifest["coverage"]["minimum_room_scenes_by_method"] = {
            method: len(selected) for method in AVAILABLE_ROOM_METHOD_IDS
        } | {"harmonizer_option_c": 0}
        output_manifest["coverage"]["minimum_room_views_by_method"] = {
            method: len(selected) * EXPECTED_VIEWS_PER_SCENE
            for method in AVAILABLE_ROOM_METHOD_IDS
        } | {"harmonizer_option_c": 0}
    all_records: dict[str, list[dict[str, Any]]] = {
        method: [] for method in ROOM_METHODS.values()
    }
    for scene in room_scenes:
        for method in AVAILABLE_ROOM_METHOD_IDS:
            all_records[method].extend(scene["records"][method])
    for label, method in ROOM_METHODS.items():
        output_manifest["room_methods"][label]["records"] = all_records[method]
    for label in OBJECT_METHODS:
        output_manifest["object_methods"][label]["records"] = []

    observed_scenes = {
        method: len({record["scene_id"] for record in all_records[method]})
        for method in ROOM_METHODS.values()
    }
    observed_views = {
        method: sum(int(record["n_views"]) for record in all_records[method])
        for method in ROOM_METHODS.values()
    }
    output_manifest["coverage"]["observed_room_scenes_by_method"] = observed_scenes
    output_manifest["coverage"]["observed_room_views_by_method"] = observed_views
    output_manifest["legacy_anchor_audit"] = {
        "room": {
            key: room_anchor_report[key]
            for key in (
                "status", "all_reproduced", "aggregates", "comparisons",
                "validity_reasons", "audit_only_reasons",
                "legacy_metric_values_reproduced", "fresh_protocol_frames_match",
            )
        },
        "object": {
            key: object_anchor_report[key]
            for key in (
                "status", "source_snapshot_coherent", "canonical_legacy_diagnostics",
                "comparisons", "validity_reasons",
            )
        },
    }

    global_reasons: list[str] = []
    if smoke:
        global_reasons.append("smoke_subset_not_paper_population")
    if inventory_git["dirty"]:
        global_reasons.append("inventory_generated_from_dirty_worktree")
    global_reasons.extend(contract_report["validity_reasons"])
    invalid_room_scenes = [scene["scene_id"] for scene in room_scenes if not scene["valid"]]
    if invalid_room_scenes:
        global_reasons.append("room_export_population_incomplete_or_invalid")
    if not room_anchor_report["all_reproduced"]:
        global_reasons.append("known_room_anchors_not_reproduced")
    if not object_anchor_report["source_snapshot_coherent"]:
        global_reasons.append("object_anchor_source_snapshot_not_reproducible")
    export_commits = sorted({
        str(scene["manifest"]["git"].get("commit"))
        for scene in room_scenes
        if scene["manifest"] is not None
        and isinstance(scene["manifest"].get("git"), dict)
        and COMMIT_RE.fullmatch(
            str(scene["manifest"]["git"].get("commit", "")).lower()
        )
    })
    if len(export_commits) != 1:
        global_reasons.append("room_exports_do_not_share_one_build_commit")
    elif export_commits[0] != inventory_git["commit"]:
        global_reasons.append("room_export_commit_differs_from_inventory_commit")
    global_reasons.extend(object_anchor_report["validity_reasons"])
    output_manifest["paper_ready"] = False
    output_manifest["paper_ready_reasons"] = sorted(set(global_reasons))

    report = {
        "schema_version": 1,
        "freeze_id": freeze_id,
        "mode": "smoke" if smoke else "full",
        "valid": bool(
            room_scenes
            and not invalid_room_scenes
            and contract_report["valid"]
        ),
        "paper_ready": False,
        "population": {
            "planned_scene_count": len(planned),
            "planned_scene_ids": planned,
            "planned_scene_ids_sha256": _sha256_bytes(
                ("\n".join(planned) + "\n").encode("ascii")
            ),
            "selected_scene_count": len(selected),
            "selected_scene_ids": selected,
            "full_population_preserved": full_population,
        },
        "inventory_git": inventory_git,
        "config": output_manifest["source_config"],
        "construction_roster_source": output_manifest["construction_roster_source"],
        "upstream_e0_contract": contract_report,
        "coverage": {
            "expected_room_scenes_by_method": {
                method: len(selected) for method in AVAILABLE_ROOM_METHOD_IDS
            } | {"harmonizer_option_c": 0},
            "observed_room_scenes_by_method": observed_scenes,
            "expected_room_views_by_method": {
                method: len(selected) * EXPECTED_VIEWS_PER_SCENE
                for method in AVAILABLE_ROOM_METHOD_IDS
            } | {"harmonizer_option_c": 0},
            "observed_room_views_by_method": observed_views,
        },
        "room_exports": {
            "valid_scene_count": len(room_scenes) - len(invalid_room_scenes),
            "invalid_scene_count": len(invalid_room_scenes),
            "invalid_scene_ids": invalid_room_scenes,
            "expected_views_per_available_method": len(selected) * EXPECTED_VIEWS_PER_SCENE,
            "observed_scenes_by_method": observed_scenes,
            "observed_views_by_method": observed_views,
            "records": [
                {
                    "scene_id": scene["scene_id"],
                    "valid": scene["valid"],
                    "issues": scene["issues"],
                    "source_artifacts": scene["source_artifacts"],
                    "manifest": scene["manifest"],
                }
                for scene in room_scenes
            ],
            "export_commits": export_commits,
        },
        "room_anchor_reproduction": room_anchor_report,
        "object_anchor_reproduction": object_anchor_report,
        "missing_artifact_counts": dict(sorted(Counter(
            issue for scene in room_scenes for issue in scene["issues"]
        ).items())),
        "global_validity_reasons": sorted(set(global_reasons)),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        manifest_path = staging / "manifests/fidelity_manifest.json"
        _write_json(manifest_path, output_manifest)
        if _inject_failure_after == "manifest":
            raise RuntimeError("injected fidelity inventory failure after manifest")
        report["manifest_path"] = _relative(
            destination / "manifests/fidelity_manifest.json"
        )
        report["manifest_sha256"] = _sha256_file(manifest_path)
        _write_json(staging / "validation_report.json", report)
        if _inject_failure_after == "report":
            raise RuntimeError("injected fidelity inventory failure after report")
        if destination.exists() or destination.is_symlink():
            raise InventoryError(
                f"inventory destination appeared during scan: {destination}"
            )
        directories = [path for path in staging.rglob("*") if path.is_dir()]
        for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            _fsync_directory(directory)
        _fsync_directory(staging)
        staging.rename(destination)
        _fsync_directory(destination.parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        key: value
        for key, value in report.items()
        if key not in {"room_anchor_reproduction", "object_anchor_reproduction"}
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS))
    parser.add_argument("--room-runs-root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--scene-list", dest="construction_config", default=str(DEFAULT_CONSTRUCTION_CONFIG),
        help="pinned E1 construction YAML; must exactly match the E2 roster",
    )
    parser.add_argument(
        "--reviewed-object-aggregate", default=str(DEFAULT_OBJECT_REVIEW),
    )
    parser.add_argument("--contract-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--expected-scenes", type=int, default=50)
    parser.add_argument("--scene-id", action="append", dest="selected_scene_ids")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = build_inventory(
            args.outputs_root,
            args.room_runs_root,
            args.out,
            args.freeze_id,
            config_path=args.config,
            construction_config_path=args.construction_config,
            reviewed_object_aggregate_path=args.reviewed_object_aggregate,
            contract_manifest_path=args.contract_manifest,
            expected_scenes=args.expected_scenes,
            selected_scene_ids=args.selected_scene_ids,
            smoke=args.smoke,
            smoke_limit=args.smoke_limit,
        )
    except (InventoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"fidelity inventory failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
