"""Inventory existing E1 construction evidence without running constructors.

The scanner publishes one record for every planned ``(regime, scene)`` pair,
including failed attempts.  Values are derived from the artifact that defines
their denominator (prepared objects, accepted reports, matched GT evaluations,
and attempted drop tests); disagreement between those populations makes a row
preliminary instead of being silently averaged away.

All outputs are built in an adjacent repository-local staging directory and
renamed into place as a bundle.  The command never overwrites an earlier audit.
"""
from __future__ import annotations

import argparse
import csv
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
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from robo.eval.construction_metrics import REQUIRED_COLUMNS, REQUIRED_REGIMES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
SCENE_RE = re.compile(r"^[0-9a-f]{10}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
FREEZE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LEGACY_COMMIT = "legacy-unrecorded"

REGIME_SPECS = {
    REQUIRED_REGIMES[0]: {
        "id": "gt_segments_scan_mesh", "suffix": "factory", "automatic": False,
    },
    REQUIRED_REGIMES[1]: {
        "id": "auto_discovery_scan_mesh", "suffix": "auto", "automatic": True,
    },
    REQUIRED_REGIMES[2]: {
        "id": "gt_segments_splat_fused_mesh", "suffix": "rowC", "automatic": False,
    },
    REQUIRED_REGIMES[3]: {
        "id": "auto_discovery_splat_fused_mesh", "suffix": "rowC2", "automatic": True,
    },
    REQUIRED_REGIMES[4]: {
        "id": "single_rgb_metric_depth", "suffix": None, "automatic": True,
    },
}

AUDIT_COLUMNS = (
    "source_artifacts",
    "f1_source",
    "stability_source",
    "runtime_source",
    "denominator_status",
)
CSV_FIELDS = tuple(REQUIRED_COLUMNS) + AUDIT_COLUMNS
SOURCE_MANIFEST_NAMES = (
    "build_manifest.json", "run_manifest.json", "provenance.json",
)


class InventoryError(ValueError):
    """Raised when an inventory request itself is unsafe or ambiguous."""


def _relative(path: Path, root: Path = REPO_ROOT) -> str:
    lexical_path = Path(os.path.abspath(path))
    lexical_root = Path(os.path.abspath(root))
    try:
        return str(lexical_path.relative_to(lexical_root))
    except ValueError as exc:
        raise InventoryError(f"path is outside the SimAny repository: {path}") from exc


def _repo_path(value: str | os.PathLike[str], *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise InventoryError(f"{label} must stay inside the SimAny repository: {path}") from exc
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _symlink_component(path: Path) -> Path | None:
    """Return the first symlink in a lexical repository-local path."""
    relative = Path(_relative(path))
    current = REPO_ROOT
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _safe_regular_file(path: Path, role: str, issues: list[str]) -> bool:
    """Check a file without following any symlink component."""
    try:
        component = _symlink_component(path)
    except InventoryError:
        _append_once(issues, f"{role}_outside_repository")
        return False
    if component is not None:
        _append_once(issues, f"{role}_uses_symlink_path")
        return False
    return path.is_file()


def _file_info(path: Path, role: str) -> dict[str, Any]:
    """Fingerprint a metadata file, never following a symlink out of the repo."""
    result: dict[str, Any] = {"role": role, "path": _relative(path)}
    if path.is_symlink():
        target = os.readlink(path)
        data = ("symlink:" + target).encode("utf-8")
        result.update(
            kind="symlink", target=target, target_exists="not_followed",
            size_bytes=len(data), sha256=_sha256(data),
        )
        return result
    data = path.read_bytes()
    result.update(kind="file", size_bytes=len(data), sha256=_sha256(data))
    return result


def _source_hash(artifacts: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    unique = {(item["path"], item["sha256"]) for item in artifacts}
    for path, sha in sorted(unique):
        digest.update(f"{sha}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _load_json(
    path: Path,
    role: str,
    artifacts: list[dict[str, Any]],
    issues: list[str],
) -> Any | None:
    if path.is_symlink():
        artifacts.append(_file_info(path, role))
        _append_once(issues, f"{role}_uses_symlink_path")
        return None
    if not _safe_regular_file(path, role, issues):
        return None
    artifacts.append(_file_info(path, role))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _append_once(issues, f"malformed_{role}:{type(exc).__name__}")
        return None


def _as_count(value: Any, *, name: str, issues: list[str]) -> int | None:
    if value is None or isinstance(value, bool):
        _append_once(issues, f"missing_or_invalid_{name}")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        _append_once(issues, f"missing_or_invalid_{name}")
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        _append_once(issues, f"missing_or_invalid_{name}")
        return None
    return int(number)


def _as_score(value: Any, *, name: str, issues: list[str]) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        _append_once(issues, f"invalid_{name}")
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        _append_once(issues, f"invalid_{name}")
        return None
    return score


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _report_values(payload: Any, issues: list[str]) -> dict[str, Any]:
    result = {"reported_inputs": None, "accepted": None, "f1": None, "f1_weight": 0}
    if payload is None:
        return result
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
        _append_once(issues, "report_schema_invalid")
        return result
    inputs = _as_count(payload.get("n_instances"), name="report_n_instances", issues=issues)
    tier_a = _as_count(payload.get("tier_A"), name="report_tier_A", issues=issues)
    tier_b = _as_count(payload.get("tier_B"), name="report_tier_B", issues=issues)
    tier_c = _as_count(
        payload.get("tier_C_rejected"), name="report_tier_C_rejected", issues=issues,
    )
    objects = payload["objects"]
    if inputs is not None and inputs != len(objects):
        _append_once(issues, "report_objects_denominator_mismatch")
    if None not in (inputs, tier_a, tier_b, tier_c):
        assert inputs is not None and tier_a is not None and tier_b is not None and tier_c is not None
        if tier_a + tier_b + tier_c != inputs:
            _append_once(issues, "report_tier_denominator_mismatch")
        result["accepted"] = tier_a + tier_b
    result["reported_inputs"] = inputs

    observed_tiers = Counter(
        str(item.get("tier")) for item in objects if isinstance(item, dict)
    )
    if tier_a is not None and observed_tiers["A"] != tier_a:
        _append_once(issues, "report_object_tier_A_mismatch")
    if tier_b is not None and observed_tiers["B"] != tier_b:
        _append_once(issues, "report_object_tier_B_mismatch")
    if tier_c is not None and observed_tiers["C"] != tier_c:
        _append_once(issues, "report_object_tier_C_mismatch")

    f1_values: list[float] = []
    accepted_objects = [
        item for item in objects
        if isinstance(item, dict) and item.get("tier") in {"A", "B"}
    ]
    for index, item in enumerate(accepted_objects):
        value = _as_score(item.get("f1_20"), name=f"report_object_{index}_f1_20", issues=issues)
        if value is not None:
            f1_values.append(value)
    if result["accepted"] is not None and len(accepted_objects) != result["accepted"]:
        _append_once(issues, "report_accepted_population_mismatch")
    if len(f1_values) != len(accepted_objects):
        _append_once(issues, "report_f1_population_incomplete")
    result.update(f1=_mean(f1_values), f1_weight=len(f1_values))
    return result


def _eval_values(payload: Any, expected_inputs: int, accepted: int, issues: list[str]) -> tuple[float | None, int]:
    if payload is None:
        return None, 0
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
        _append_once(issues, "gt_eval_schema_invalid")
        return None, 0
    objects = payload["objects"]
    declared = _as_count(payload.get("n_objects"), name="gt_eval_n_objects", issues=issues)
    if declared is not None and declared != len(objects):
        _append_once(issues, "gt_eval_objects_denominator_mismatch")
    if len(objects) != expected_inputs:
        _append_once(issues, "gt_eval_and_prepared_population_mismatch")
    accepted_objects = [
        item for item in objects
        if isinstance(item, dict) and item.get("tier") in {"A", "B"}
    ]
    if len(accepted_objects) != accepted:
        _append_once(issues, "gt_eval_and_accepted_population_mismatch")
    values: list[float] = []
    for index, item in enumerate(accepted_objects):
        value = _as_score(item.get("f1_20_gt"), name=f"gt_eval_object_{index}_f1_20", issues=issues)
        if value is not None:
            values.append(value)
    return _mean(values), len(values)


def _drop_values(payload: Any, accepted: int, issues: list[str]) -> tuple[int | None, int | None]:
    if payload is None:
        return None, None
    if not isinstance(payload, dict):
        _append_once(issues, "drop_schema_invalid")
        return None, None
    tested = _as_count(payload.get("n_tested"), name="drop_n_tested", issues=issues)
    stable = _as_count(payload.get("n_stable"), name="drop_n_stable", issues=issues)
    if tested is None or stable is None:
        return None, None
    if stable > tested:
        _append_once(issues, "drop_stable_exceeds_tested")
        return None, None
    if tested > accepted:
        _append_once(issues, "drop_tested_exceeds_accepted")
        return None, None
    objects = payload.get("objects")
    if isinstance(objects, list):
        if len(objects) != tested:
            _append_once(issues, "drop_objects_denominator_mismatch")
        stable_objects = sum(
            item.get("stable") is True for item in objects if isinstance(item, dict)
        )
        if stable_objects != stable:
            _append_once(issues, "drop_stable_count_mismatch")
    return stable, tested


def _timing_minutes(path: Path, artifacts: list[dict[str, Any]], issues: list[str]) -> float | None:
    if path.is_symlink():
        artifacts.append(_file_info(path, "runtime"))
        _append_once(issues, "runtime_uses_symlink_path")
        return None
    if not _safe_regular_file(path, "runtime", issues):
        return None
    artifacts.append(_file_info(path, "runtime"))
    values: list[float] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            seconds = float(line.rsplit(maxsplit=1)[-1])
        except (IndexError, ValueError):
            _append_once(issues, f"malformed_runtime_line_{line_number}")
            continue
        if not math.isfinite(seconds) or seconds < 0:
            _append_once(issues, f"invalid_runtime_line_{line_number}")
            continue
        values.append(seconds)
    if not values:
        _append_once(issues, "runtime_has_no_stage_durations")
        return None
    return math.fsum(values) / 60.0


def _extract_commit(payload: Any, issues: list[str]) -> str:
    if not isinstance(payload, dict):
        _append_once(issues, "build_manifest_schema_invalid")
        return LEGACY_COMMIT
    candidates = []
    for value in (
        payload.get("build_commit"), payload.get("code_commit"),
        payload.get("code", {}).get("commit") if isinstance(payload.get("code"), dict) else None,
        payload.get("git", {}).get("commit") if isinstance(payload.get("git"), dict) else None,
    ):
        if value not in {None, ""}:
            candidates.append(str(value).strip().lower())
    if len(set(candidates)) > 1:
        _append_once(issues, "build_manifest_commit_fields_disagree")
        return LEGACY_COMMIT
    if not candidates or not COMMIT_RE.fullmatch(candidates[0]):
        _append_once(issues, "build_commit_unrecorded")
        return LEGACY_COMMIT
    return candidates[0]


def _build_provenance(scene_dir: Path, artifacts: list[dict[str, Any]], issues: list[str]) -> tuple[str, str | None]:
    candidates = [
        scene_dir / name
        for name in SOURCE_MANIFEST_NAMES
        if _safe_regular_file(scene_dir / name, "source_build_manifest", issues)
    ]
    if len(candidates) > 1:
        _append_once(issues, "multiple_source_build_manifests")
    if not candidates:
        _append_once(issues, "build_manifest_unrecorded")
        _append_once(issues, "build_commit_unrecorded")
        return LEGACY_COMMIT, None
    source = candidates[0]
    payload = _load_json(source, "source_build_manifest", artifacts, issues)
    return _extract_commit(payload, issues), _relative(source)


def _artifact_state(path: Path) -> dict[str, Any]:
    state: dict[str, Any] = {
        "path": _relative(path),
        "present": False,
        "is_symlink": False,
    }
    component = _symlink_component(path)
    if component is not None:
        state["is_symlink"] = True
        state["symlink_component"] = _relative(component)
    if path.is_symlink():
        state["target"] = os.readlink(path)
        state["target_exists"] = "not_followed"
    elif component is None and path.is_file():
        state["present"] = True
        state["size_bytes"] = path.stat().st_size
    return state


def _object_population(objects: Any, scene_dir: Path, issues: list[str]) -> tuple[int, int, int, bool]:
    if not isinstance(objects, list):
        return 0, 0, 0, False
    expected_names: set[str] = set()
    for position, item in enumerate(objects):
        raw_index = item.get("index", position) if isinstance(item, dict) else position
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            _append_once(issues, "prepared_object_index_invalid")
            continue
        expected_names.add(f"obj_{index:02d}")
    objects_root = scene_dir / "objects"
    if _symlink_component(objects_root) is not None:
        _append_once(issues, "objects_directory_uses_symlink_path")
        return len(objects), 0, 0, False
    object_dirs = {
        path.name
        for path in objects_root.glob("obj_*")
        if not path.is_symlink() and path.is_dir()
    }
    if any(path.is_symlink() for path in objects_root.glob("obj_*")):
        _append_once(issues, "object_directory_uses_symlink_path")
    mesh_names = {
        name for name in object_dirs
        if _safe_regular_file(
            scene_dir / "objects" / name / "trellis_mesh.ply",
            "candidate_mesh",
            issues,
        )
    }
    if object_dirs - expected_names:
        _append_once(issues, "stale_extra_object_directories")
    complete = mesh_names == expected_names
    return len(objects), len(object_dirs), len(mesh_names), complete


def _base_row(freeze_id: str, regime: str, scene_id: str, manifest_path: str) -> dict[str, Any]:
    return {
        "freeze_id": freeze_id,
        "regime": regime,
        "scene_id": scene_id,
        "scene_status": "unavailable",
        "input_instances": 0,
        "accepted_instances": 0,
        "f1_20": None,
        "f1_weight": 0,
        "stable_instances": None,
        "tested_instances": None,
        "runtime_minutes": None,
        "build_commit": LEGACY_COMMIT,
        "build_manifest_path": manifest_path,
        "failure_reason": "",
        "source_artifact_hash": _source_hash([]),
        "record_valid": False,
        "validity_reasons": [],
        "geometry_reference_status": "unavailable",
        "source_artifacts": [],
        "f1_source": "unavailable",
        "stability_source": "unavailable",
        "runtime_source": "unavailable",
        "denominator_status": "unknown",
    }


def _known_log(outputs_root: Path, suffix: str | None, scene_id: str) -> Path | None:
    names = {
        "factory": f"fleet_factory_{scene_id}.log",
        "auto": f"fleet_auto_{scene_id}.log",
        "rowC": f"ablation_rowC_{scene_id}.log",
        "rowC2": f"ablation_rowC2_{scene_id}.log",
        None: f"ablation_rowD_{scene_id}.log",
    }
    path = outputs_root / names[suffix]
    return path if not path.is_symlink() and path.is_file() else None


def _scan_report_regime(
    outputs_root: Path,
    freeze_id: str,
    regime: str,
    scene_id: str,
    manifest_path: str,
) -> dict[str, Any]:
    spec = REGIME_SPECS[regime]
    suffix = str(spec["suffix"])
    scene_dir = outputs_root / f"{scene_id}_{suffix}"
    issues: list[str] = []
    artifacts: list[dict[str, Any]] = []

    objects_path = scene_dir / "objects" / "objects.json"
    aligned_path = scene_dir / "objects" / "aligned_all.json"
    report_path = scene_dir / "report.json"
    eval_path = scene_dir / "eval_vs_gt.json"
    drop_path = scene_dir / "drop_v2.json"
    timing_path = scene_dir / "timings.txt"
    derived_path = scene_dir / "derived_mesh.ply"
    auto_path = scene_dir / "auto_instances.npz"

    objects = _load_json(objects_path, "prepared_objects", artifacts, issues)
    aligned = _load_json(aligned_path, "registration", artifacts, issues)
    report = _load_json(report_path, "acceptance_report", artifacts, issues)
    gt_eval = _load_json(eval_path, "independent_gt_eval", artifacts, issues)
    drop = _load_json(drop_path, "corrected_drop_test", artifacts, issues)
    report_values = _report_values(report, issues)
    prepared, object_dirs, mesh_count, candidates_complete = _object_population(
        objects, scene_dir, issues,
    )

    # Prepared objects define the asset-stage denominator.  A legacy report is
    # only a fallback when the prepared record itself is unavailable.
    if isinstance(objects, list):
        inputs = prepared
        denominator_status = "prepared_objects_json"
        if report_values["reported_inputs"] is not None and report_values["reported_inputs"] != inputs:
            _append_once(issues, "prepared_and_report_input_denominators_disagree")
    elif report_values["reported_inputs"] is not None:
        inputs = int(report_values["reported_inputs"])
        denominator_status = "report_fallback_prepared_record_missing"
        _append_once(issues, "input_denominator_not_traced_to_prepared_objects")
    else:
        inputs = 0
        denominator_status = "unavailable_encoded_zero_pre_asset_failure"
        _append_once(issues, "input_denominator_unavailable")

    accepted = report_values["accepted"]
    if accepted is None:
        accepted = 0
    if accepted > inputs:
        _append_once(issues, "accepted_exceeds_prepared_inputs")

    if isinstance(aligned, list) and len(aligned) != inputs:
        _append_once(issues, "registration_and_prepared_populations_disagree")
    elif aligned is not None and not isinstance(aligned, list):
        _append_once(issues, "registration_schema_invalid")

    if gt_eval is not None:
        f1, f1_weight = _eval_values(gt_eval, inputs, accepted, issues)
        f1_source = "eval_vs_gt.json legacy matched-GT diagnostic; protocol unverified"
        geometry_status = "legacy_gt_evaluation_protocol_unverified"
        _append_once(issues, "gt_eval_protocol_manifest_missing")
        _append_once(issues, "gt_eval_deterministic_one_to_one_matching_unproven")
    else:
        f1, f1_weight = report_values["f1"], int(report_values["f1_weight"])
        f1_source = "report.json registration-reference diagnostic"
        geometry_status = "non_independent_registration_reference"
        _append_once(issues, "independent_gt_geometry_evaluation_missing")

    observed_metrics = {
        "input_instances": inputs,
        "accepted_instances": accepted,
        "f1_20": f1,
        "f1_weight": f1_weight,
        "stable_instances": None,
        "tested_instances": None,
        "runtime_minutes": None,
    }

    if suffix == "rowC":
        geometry_status = "invalid_splat_ablation_gt_scan_registration_leakage"
        _append_once(issues, "gt_vertex_index_geometry_leakage")
        _append_once(issues, "rowC_did_not_use_splat_mesh_for_registration")
        _append_once(issues, "gt_geometry_leakage_in_registration_and_evaluation")
        f1, f1_weight = None, 0
        f1_source = "withheld: GT scan geometry leaked into registration/evaluation"
    elif suffix == "rowC2":
        # The surviving eval files and current acceptance reports come from
        # different fleet passes.  Keep their existence in the audit, but do
        # not join them into one strict scene record.
        f1, f1_weight = None, 0
        f1_source = "withheld: eval_vs_gt.json and report.json are incoherent snapshots"
        geometry_status = "incoherent_asset_eval_snapshots"
        _append_once(issues, "incoherent_asset_eval_snapshots")
        if report_path.is_file() and eval_path.is_file() and eval_path.stat().st_mtime_ns < report_path.stat().st_mtime_ns:
            _append_once(issues, "rowC2_gt_eval_older_than_current_report")
        _append_once(issues, "rowC2_current_population_differs_from_paper_59pct_population")

    if accepted > inputs:
        # Preserve the impossible legacy value above as a diagnostic, but do
        # not publish a scene record that the strict table producer must reject.
        accepted = 0
        f1, f1_weight = None, 0
        f1_source = "withheld: accepted population exceeds prepared inputs"
        geometry_status = "invalid_population_counts"

    stable, tested = _drop_values(drop, accepted, issues)
    runtime = _timing_minutes(timing_path, artifacts, issues)
    observed_metrics.update(
        stable_instances=stable,
        tested_instances=tested,
        runtime_minutes=runtime,
    )
    if suffix in {"rowC", "rowC2"}:
        stable, tested, runtime = None, None, None
        _append_once(issues, f"{suffix}_stability_snapshot_unavailable")
        _append_once(issues, f"{suffix}_runtime_snapshot_unavailable")
    build_commit, source_manifest = _build_provenance(scene_dir, artifacts, issues)

    discovery_present = isinstance(objects, list)
    if bool(spec["automatic"]):
        discovery_present = discovery_present and _safe_regular_file(
            auto_path, "automatic_instances", issues
        )
    derived_required = suffix in {"rowC", "rowC2"}
    derived_present = (
        _safe_regular_file(derived_path, "derived_mesh", issues)
        if derived_required
        else True
    )
    if derived_required and not derived_present:
        _append_once(issues, "derived_mesh_missing_or_broken")
    presence = {
        "discovery_output_present": discovery_present,
        "candidate_generation_complete": candidates_complete,
        "registration_result_present": isinstance(aligned, list),
        "accepted_status_present": report_values["accepted"] is not None,
        "physics_drop_result_present": stable is not None and tested is not None,
        "runtime_source_present": runtime is not None,
        "gt_matching_result_present": gt_eval is not None,
    }
    for key, present in presence.items():
        if not present:
            _append_once(issues, "missing_" + key.removesuffix("_present"))

    log = _known_log(outputs_root, spec["suffix"], scene_id)
    if log is not None:
        # Logs are often large and are not numeric inputs; record a stat-only
        # pointer while hashes remain limited to consumed metadata.
        log_state = _artifact_state(log)
    else:
        log_state = None

    row = _base_row(freeze_id, regime, scene_id, manifest_path)
    required_completion = {
        "discovery": discovery_present,
        "candidate_generation": candidates_complete,
        "registration": isinstance(aligned, list),
        "accepted_status": report_values["accepted"] is not None,
        "derived_mesh": derived_present,
    }
    incoherent_population = any(
        "denominator" in reason or "population" in reason for reason in issues
    )
    if report_values["accepted"] is not None and inputs == 0:
        status = "empty"
        failure_reason = "no prepared input instances"
    elif report_values["accepted"] is not None and all(required_completion.values()) and not incoherent_population:
        status = "success"
        failure_reason = ""
    elif report_values["accepted"] is not None:
        status = "invalid"
        failed_stages = sorted(
            name for name, present in required_completion.items() if not present
        )
        failure_reason = "incomplete or incoherent legacy result"
        if failed_stages:
            failure_reason += ": " + ", ".join(failed_stages)
    elif os.path.lexists(scene_dir) or log is not None:
        status = "failed"
        failure_reason = "acceptance report missing after partial attempt"
    else:
        status = "unavailable"
        failure_reason = "no output directory or attempt log found"
    if any("denominator" in reason or "population" in reason for reason in issues):
        denominator_status += ";incoherent"

    row.update(
        scene_status=status,
        input_instances=inputs,
        accepted_instances=accepted,
        f1_20=f1,
        f1_weight=f1_weight,
        stable_instances=stable,
        tested_instances=tested,
        runtime_minutes=runtime,
        build_commit=build_commit,
        failure_reason=failure_reason,
        source_artifact_hash=_source_hash(artifacts),
        record_valid=(not issues and status in {"success", "empty"}),
        validity_reasons=sorted(issues),
        geometry_reference_status=geometry_status,
        source_artifacts=sorted({item["path"] for item in artifacts}),
        f1_source=f1_source,
        stability_source="drop_v2.json corrected attempted-body counts" if tested is not None else "unavailable",
        runtime_source="timings.txt summed stage durations" if runtime is not None else "unavailable",
        denominator_status=denominator_status,
    )
    return {
        "row": row,
        "source_manifest": source_manifest,
        "source_artifacts": artifacts,
        "observed_metrics": observed_metrics,
        "inventory": {
            "scene_id": scene_id,
            "regime": regime,
            "scene_dir": _relative(scene_dir),
            "scene_status": status,
            "record_valid": row["record_valid"],
            **presence,
            "missing": sorted(key for key, present in presence.items() if not present),
            "coherence_issues": sorted(
                reason for reason in issues if "denominator" in reason or "population" in reason
            ),
            "validity_reasons": sorted(issues),
            "prepared_objects": prepared if isinstance(objects, list) else None,
            "object_directories": object_dirs,
            "candidate_meshes": mesh_count,
            "derived_mesh": _artifact_state(derived_path) if derived_required else None,
            "automatic_instances": _artifact_state(auto_path) if bool(spec["automatic"]) else None,
            "attempt_log": log_state,
            "observed_legacy_metrics": observed_metrics,
        },
    }


def _rowd_values(aligned: Any, issues: list[str]) -> tuple[int, float | None, int]:
    if aligned is None:
        return 0, None, 0
    if not isinstance(aligned, list):
        _append_once(issues, "registration_schema_invalid")
        return 0, None, 0
    accepted_values: list[float] = []
    for index, item in enumerate(aligned):
        if not isinstance(item, dict) or item.get("rejected"):
            continue
        evaluation = item.get("eval")
        if not isinstance(evaluation, dict):
            continue
        try:
            f20_raw = evaluation["f1@20mm"]["f1"]
            f40_raw = evaluation["f1@40mm"]["f1"]
        except (KeyError, TypeError):
            _append_once(issues, f"rowD_object_{index}_eval_schema_invalid")
            continue
        f20 = _as_score(f20_raw, name=f"rowD_object_{index}_f1_20", issues=issues)
        f40 = _as_score(f40_raw, name=f"rowD_object_{index}_f1_40", issues=issues)
        if f20 is None or f40 is None:
            continue
        if f20 >= 0.40 or f40 >= 0.20:
            accepted_values.append(f20)
    return len(accepted_values), _mean(accepted_values), len(accepted_values)


def _scan_rowd(
    outputs_root: Path,
    freeze_id: str,
    scene_id: str,
    manifest_path: str,
    audit: Any,
    audit_path: Path,
) -> dict[str, Any]:
    regime = REQUIRED_REGIMES[4]
    scene_dir = outputs_root / scene_id
    issues: list[str] = [
        "mixed_legacy_code_states",
        "gt_oracle_frame_selection",
        "gt_centroid_frame_selection_leakage",
        "geometry_independence_not_proven",
        "rowD_scored_population_excludes_generation_failures",
        "missing_corrected_drop_test",
        "missing_runtime",
    ]
    artifacts: list[dict[str, Any]] = []
    if _safe_regular_file(audit_path, "rowD_population_audit", issues):
        artifacts.append(_file_info(audit_path, "rowD_population_audit"))

    objects_path = scene_dir / "objects" / "objects.json"
    aligned_path = scene_dir / "objects" / "aligned_all.json"
    objects = _load_json(objects_path, "prepared_objects", artifacts, issues)
    aligned = _load_json(aligned_path, "registration_and_gt_eval", artifacts, issues)
    prepared, object_dirs, mesh_count, candidates_complete = _object_population(objects, scene_dir, issues)
    entry = audit.get("per_scene", {}).get(scene_id) if isinstance(audit, dict) else None

    if isinstance(objects, list):
        inputs = prepared
        denominator_status = "prepared_objects_json;scored_population_from_aligned_all"
    else:
        inputs = 0
        denominator_status = "zero_before_asset_stage_or_unavailable"
        if entry is None:
            _append_once(issues, "rowD_audit_entry_missing")

    scored = len(aligned) if isinstance(aligned, list) else 0
    accepted, f1, f1_weight = _rowd_values(aligned, issues)
    observed_metrics = {
        "input_instances": inputs,
        "accepted_instances": accepted,
        "f1_20": f1,
        "f1_weight": f1_weight,
        "stable_instances": None,
        "tested_instances": None,
        "runtime_minutes": None,
    }
    if accepted > inputs:
        _append_once(issues, "accepted_exceeds_prepared_inputs")
        accepted = 0

    audit_scored = None
    expected_accepted = None
    status_text = "row-D audit entry missing"
    if isinstance(entry, dict):
        status_text = str(entry.get("status", ""))
        if entry.get("n_instances") is not None:
            audit_scored = _as_count(entry.get("n_instances"), name="rowD_audit_scored_instances", issues=issues)
        tier_a = _as_count(entry.get("n_tier_A", 0), name="rowD_audit_tier_A", issues=issues)
        tier_b = _as_count(entry.get("n_tier_B", 0), name="rowD_audit_tier_B", issues=issues)
        if tier_a is not None and tier_b is not None:
            expected_accepted = tier_a + tier_b
    if audit_scored is not None and audit_scored != scored:
        _append_once(issues, "rowD_audit_and_scored_population_disagree")
    if expected_accepted is not None and expected_accepted != accepted:
        _append_once(issues, "rowD_audit_and_accepted_population_disagree")

    if "completed (DONE)" in status_text and inputs > 0:
        status = "success"
        failure_reason = ""
    elif "zero instances" in status_text:
        status = "empty"
        failure_reason = status_text
    elif inputs > 0 and not isinstance(aligned, list):
        status = "failed"
        failure_reason = status_text or "generation failed before registration"
    else:
        status = "failed" if entry is not None else "unavailable"
        failure_reason = status_text or "row-D evidence unavailable"

    build_commit, source_manifest = _build_provenance(scene_dir, artifacts, issues)
    presence = {
        "discovery_output_present": isinstance(objects, list),
        "candidate_generation_complete": candidates_complete,
        "registration_result_present": isinstance(aligned, list),
        "accepted_status_present": entry is not None and expected_accepted is not None,
        "physics_drop_result_present": False,
        "runtime_source_present": False,
        "gt_matching_result_present": isinstance(aligned, list),
    }
    for key, present in presence.items():
        if not present:
            _append_once(issues, "missing_" + key.removesuffix("_present"))

    if status == "success" and not all(
        presence[key]
        for key in (
            "discovery_output_present",
            "candidate_generation_complete",
            "registration_result_present",
            "accepted_status_present",
        )
    ):
        status = "invalid"
        failure_reason = "incomplete legacy construction stages"

    log = _known_log(outputs_root, None, scene_id)
    row = _base_row(freeze_id, regime, scene_id, manifest_path)
    row.update(
        scene_status=status,
        input_instances=inputs,
        accepted_instances=accepted,
        f1_20=None,
        f1_weight=0,
        build_commit=build_commit,
        failure_reason=failure_reason,
        source_artifact_hash=_source_hash(artifacts),
        record_valid=False,
        validity_reasons=sorted(issues),
        geometry_reference_status="legacy_gt_matching_with_selection_leakage",
        source_artifacts=sorted({item["path"] for item in artifacts}),
        f1_source=("withheld: aligned_all.json uses GT-oracle selection and mixed legacy states"
                   if f1 is not None else "unavailable"),
        denominator_status=denominator_status,
    )
    return {
        "row": row,
        "source_manifest": source_manifest,
        "source_artifacts": artifacts,
        "observed_metrics": observed_metrics,
        "inventory": {
            "scene_id": scene_id,
            "regime": regime,
            "scene_dir": _relative(scene_dir),
            "scene_status": status,
            "record_valid": False,
            **presence,
            "missing": sorted(key for key, present in presence.items() if not present),
            "coherence_issues": sorted(
                reason for reason in issues if "denominator" in reason or "population" in reason
            ),
            "validity_reasons": sorted(issues),
            "prepared_inputs": inputs,
            "scored_instances": scored,
            "audit_scored_instances": audit_scored,
            "accepted_instances": accepted,
            "object_directories": object_dirs,
            "candidate_meshes": mesh_count,
            "attempt_log": _artifact_state(log) if log is not None else None,
            "observed_legacy_metrics": observed_metrics,
        },
    }


def _normalize_scene_ids(values: Iterable[str], *, label: str) -> list[str]:
    scenes = [str(value).strip() for value in values if str(value).strip()]
    duplicates = sorted(scene for scene, count in Counter(scenes).items() if count > 1)
    if duplicates:
        raise InventoryError(f"duplicate {label}: {duplicates}")
    invalid = sorted(scene for scene in scenes if not SCENE_RE.fullmatch(scene))
    if invalid:
        raise InventoryError(f"invalid {label}: {invalid}")
    return sorted(scenes)


def _resolve_regimes(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return list(REQUIRED_REGIMES)
    by_id = {str(spec["id"]): label for label, spec in REGIME_SPECS.items()}
    resolved = [by_id.get(value, value) for value in values]
    invalid = [value for value in resolved if value not in REQUIRED_REGIMES]
    if invalid:
        raise InventoryError(f"unknown construction regimes: {invalid}")
    if len(set(resolved)) != len(resolved):
        raise InventoryError("duplicate construction regime selection")
    return [regime for regime in REQUIRED_REGIMES if regime in resolved]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_snapshot() -> dict[str, Any]:
    commit = _git_commit()
    try:
        status_lines = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        status_lines = ["git_status_unavailable"]
    return {
        "commit": commit,
        "dirty": bool(status_lines),
        "status": status_lines,
    }


def _upstream_contract(
    path_value: str | os.PathLike[str] | None,
    *,
    freeze_id: str,
    inventory_git: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """Audit the E0 manifest that this legacy import claims to inherit."""
    reasons: list[str] = []
    if path_value is None:
        reasons.append("e0_contract_manifest_not_supplied")
        return {"provided": False}, reasons, None

    path = _repo_path(path_value, label="contract_manifest")
    artifacts: list[dict[str, Any]] = []
    issues: list[str] = []
    payload = _load_json(path, "e0_contract_manifest", artifacts, issues)
    if not isinstance(payload, dict):
        reasons.extend(issues or ["e0_contract_manifest_invalid"])
        return {
            "provided": True,
            "path": _relative(path),
            "valid_json": False,
        }, reasons, artifacts[0] if artifacts else None

    artifact = artifacts[0]
    code = payload.get("code") if isinstance(payload.get("code"), dict) else {}
    contract_commit = str(code.get("commit", "")).strip().lower()
    inventory_commit = str(inventory_git["commit"]).strip().lower()
    configs = payload.get("configs") if isinstance(payload.get("configs"), list) else []
    construction = next(
        (
            item
            for item in configs
            if isinstance(item, dict) and item.get("field") == "construction_config"
        ),
        None,
    )
    config_path = REPO_ROOT / "configs/experiments/icra2027/construction_regimes.yaml"
    current_config_hash = _sha256(config_path.read_bytes())
    contract_config_hash = (
        construction.get("source_content_sha256")
        if isinstance(construction, dict)
        else None
    )
    checks = {
        "paper_mode": payload.get("mode") == "paper",
        "clean_contract_code": code.get("dirty") is False,
        "inventory_commit_matches_contract": (
            COMMIT_RE.fullmatch(inventory_commit) is not None
            and re.fullmatch(r"[0-9a-f]{7,40}", contract_commit) is not None
            and inventory_commit.startswith(contract_commit)
        ),
        "construction_config_matches_contract": (
            contract_config_hash == current_config_hash
        ),
        "freeze_id_inherits_contract": str(freeze_id).startswith(
            str(payload.get("freeze_id", ""))
        ),
    }
    reason_by_check = {
        "paper_mode": "e0_contract_is_smoke_not_paper",
        "clean_contract_code": "e0_contract_records_dirty_code",
        "inventory_commit_matches_contract": "inventory_commit_differs_from_e0_contract",
        "construction_config_matches_contract": "construction_config_differs_from_e0_contract",
        "freeze_id_inherits_contract": "e1_freeze_id_does_not_inherit_e0_freeze_id",
    }
    reasons.extend(
        reason_by_check[name] for name, passed in checks.items() if not passed
    )
    reasons.extend(issues)
    return {
        "provided": True,
        "path": _relative(path),
        "manifest_file_sha256": artifact["sha256"],
        "declared_contract_sha256": payload.get("contract_sha256"),
        "freeze_id": payload.get("freeze_id"),
        "mode": payload.get("mode"),
        "code": code,
        "construction_config_source_sha256": contract_config_hash,
        "current_construction_config_source_sha256": current_config_hash,
        "checks": checks,
    }, reasons, artifact


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_records(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for source in rows:
            row = {key: source.get(key) for key in CSV_FIELDS}
            row["record_valid"] = "true" if source["record_valid"] else "false"
            row["validity_reasons"] = json.dumps(source["validity_reasons"], separators=(",", ":"))
            row["source_artifacts"] = json.dumps(source["source_artifacts"], separators=(",", ":"))
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_scene_list(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".yaml", ".yml"}:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
        if isinstance(payload, dict):
            population = payload.get("population")
            if isinstance(population, dict):
                if population.get("dataset") != "scannetpp_v2":
                    raise InventoryError(
                        "construction population dataset must be scannetpp_v2"
                    )
                if population.get("split") != "nvs_sem_val":
                    raise InventoryError(
                        "construction population split must be nvs_sem_val"
                    )
                if population.get("preserve_failed_scenes") is not True:
                    raise InventoryError(
                        "construction population must preserve failed scenes"
                    )
                declared_count = population.get("planned_scenes")
                if not isinstance(declared_count, int):
                    raise InventoryError(
                        "construction population planned_scenes must be an integer"
                    )
                regimes = payload.get("regimes")
                labels = (
                    [str(item.get("paper_label", "")).strip() for item in regimes]
                    if isinstance(regimes, list)
                    and all(isinstance(item, dict) for item in regimes)
                    else []
                )
                if labels != list(REQUIRED_REGIMES):
                    raise InventoryError(
                        "construction population regime labels differ from the fixed contract"
                    )
                payload = population.get("scene_ids", population.get("scenes"))
                if isinstance(payload, list) and len(payload) != declared_count:
                    raise InventoryError(
                        "construction population scene_ids length differs from planned_scenes"
                    )
            else:
                payload = payload.get("scene_ids", payload.get("scenes"))
        if not isinstance(payload, list):
            raise InventoryError(
                "structured scene list must be a list or contain "
                "scene_ids/scenes (optionally below population)"
            )
        return _normalize_scene_ids(payload, label="planned scene IDs")
    return _normalize_scene_ids(
        (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()),
        label="planned scene IDs",
    )


def _validate_generated_rows(rows: Sequence[dict[str, Any]]) -> None:
    """Reject impossible normalized records before publishing the bundle."""
    for row in rows:
        record = f"{row['regime']}/{row['scene_id']}"
        inputs = row["input_instances"]
        accepted = row["accepted_instances"]
        f1_weight = row["f1_weight"]
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in (inputs, accepted, f1_weight)
        ):
            raise InventoryError(f"{record} has non-integer or negative counts")
        if accepted > inputs:
            raise InventoryError(f"{record} has accepted_instances > input_instances")
        if f1_weight > accepted:
            raise InventoryError(f"{record} has f1_weight > accepted_instances")
        if (row["f1_20"] is None) != (f1_weight == 0):
            raise InventoryError(f"{record} has incoherent F1 value/weight")
        stable = row["stable_instances"]
        tested = row["tested_instances"]
        if (stable is None) != (tested is None):
            raise InventoryError(f"{record} has only one stability count")
        if stable is not None:
            if not all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in (stable, tested)
            ):
                raise InventoryError(f"{record} has invalid stability counts")
            if stable > tested or tested > accepted:
                raise InventoryError(f"{record} has impossible stability counts")
        if row["scene_status"] != "success" and not row["failure_reason"]:
            raise InventoryError(f"{record} lacks a terminal failure_reason")
        if not row["record_valid"] and not row["validity_reasons"]:
            raise InventoryError(f"{record} is preliminary without validity reasons")


def build_inventory(
    outputs_root: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    freeze_id: str,
    *,
    expected_scenes: int = 50,
    planned_scene_ids: Sequence[str] | None = None,
    selected_scene_ids: Sequence[str] | None = None,
    selected_regimes: Sequence[str] | None = None,
    smoke: bool = False,
    smoke_limit: int = 2,
    roster_source: str | None = None,
    contract_manifest_path: str | os.PathLike[str] | None = None,
    _inject_failure_after: str | None = None,
) -> dict[str, Any]:
    """Scan existing artifacts and atomically publish records plus audit manifests."""
    outputs = _repo_path(outputs_root, label="outputs_root")
    destination = _repo_path(out_dir, label="out_dir")
    if not outputs.is_dir():
        raise InventoryError(f"outputs_root is not a directory: {outputs}")
    if destination.exists():
        raise InventoryError(f"refusing to overwrite existing inventory: {destination}")
    if FREEZE_ID_RE.fullmatch(str(freeze_id).strip()) is None:
        raise InventoryError("freeze_id contains unsafe or unsupported characters")
    if expected_scenes <= 0:
        raise InventoryError("expected_scenes must be positive")

    derived_roster = planned_scene_ids is None
    if planned_scene_ids is None:
        planned = _normalize_scene_ids(
            (path.name[:-8] for path in outputs.glob("*_factory") if path.is_dir()),
            label="factory-derived planned scene IDs",
        )
        roster_source = roster_source or "existing_factory_directories"
    else:
        planned = _normalize_scene_ids(planned_scene_ids, label="planned scene IDs")
        roster_source = roster_source or "explicit_scene_list"
    if len(planned) != expected_scenes:
        raise InventoryError(
            f"expected exactly {expected_scenes} planned scenes, found {len(planned)}"
        )

    regimes = _resolve_regimes(selected_regimes)
    if selected_scene_ids is None:
        selected = planned[:smoke_limit] if smoke else list(planned)
    else:
        selected = _normalize_scene_ids(selected_scene_ids, label="selected scene IDs")
    outside = sorted(set(selected) - set(planned))
    if outside:
        raise InventoryError(f"selected scenes are outside the planned population: {outside}")
    if not selected:
        raise InventoryError("scene selection is empty")
    partial = selected != planned or regimes != list(REQUIRED_REGIMES)
    if partial and not smoke:
        raise InventoryError("partial scene/regime selection requires smoke=True")

    inventory_git = _git_snapshot()
    upstream_contract, contract_reasons, contract_artifact = _upstream_contract(
        contract_manifest_path,
        freeze_id=freeze_id,
        inventory_git=inventory_git,
    )

    audit_path = outputs / "review_recompute_rowD.json"
    audit_issues: list[str] = []
    audit_artifacts: list[dict[str, Any]] = []
    rowd_audit = _load_json(audit_path, "rowD_population_audit", audit_artifacts, audit_issues)
    if not isinstance(rowd_audit, dict):
        rowd_audit = {}

    bundles: list[dict[str, Any]] = []
    for regime in regimes:
        regime_id = str(REGIME_SPECS[regime]["id"])
        for scene_id in selected:
            manifest_path = _relative(destination / "manifests" / regime_id / f"{scene_id}.json")
            if regime == REQUIRED_REGIMES[4]:
                bundle = _scan_rowd(
                    outputs, freeze_id, scene_id, manifest_path, rowd_audit, audit_path,
                )
            else:
                bundle = _scan_report_regime(
                    outputs, freeze_id, regime, scene_id, manifest_path,
                )
            bundles.append(bundle)

    global_reasons: list[str] = [
        "legacy_evidence_import_not_frozen_at_build_time",
        "dataset_split_source_hash_unrecorded",
        *contract_reasons,
    ]
    if inventory_git["dirty"]:
        global_reasons.append("inventory_generated_from_dirty_worktree")
    if derived_roster:
        global_reasons.append("planned_scene_roster_derived_from_existing_outputs")
    if smoke:
        global_reasons.append("smoke_subset_not_paper_population")
    known_commits = sorted({
        bundle["row"]["build_commit"] for bundle in bundles
        if COMMIT_RE.fullmatch(bundle["row"]["build_commit"])
    })
    if len(known_commits) > 1:
        global_reasons.append("mixed_build_commits_across_construction_records")
    if audit_issues:
        global_reasons.extend(audit_issues)
    for bundle in bundles:
        row = bundle["row"]
        if contract_artifact is not None:
            bundle["source_artifacts"].append(contract_artifact)
            row["source_artifacts"] = sorted(
                {item["path"] for item in bundle["source_artifacts"]}
            )
            row["source_artifact_hash"] = _source_hash(bundle["source_artifacts"])
        for reason in global_reasons:
            _append_once(row["validity_reasons"], reason)
            _append_once(bundle["inventory"]["validity_reasons"], reason)
        row["validity_reasons"] = sorted(row["validity_reasons"])
        bundle["inventory"]["validity_reasons"] = sorted(bundle["inventory"]["validity_reasons"])
        row["record_valid"] = bool(row["record_valid"] and not global_reasons)
        bundle["inventory"]["record_valid"] = row["record_valid"]

    rows = sorted(
        (bundle["row"] for bundle in bundles),
        key=lambda row: (REQUIRED_REGIMES.index(row["regime"]), row["scene_id"]),
    )
    inventory_records = sorted(
        (bundle["inventory"] for bundle in bundles),
        key=lambda row: (REQUIRED_REGIMES.index(row["regime"]), row["scene_id"]),
    )
    if len({(row["regime"], row["scene_id"]) for row in rows}) != len(rows):
        raise InventoryError("duplicate regime/scene records generated")
    _validate_generated_rows(rows)

    denominator_issues = sorted({
        issue for record in inventory_records for issue in record["coherence_issues"]
    })
    by_regime: dict[str, Any] = {}
    for regime in regimes:
        regime_rows = [row for row in rows if row["regime"] == regime]
        regime_inventory = [row for row in inventory_records if row["regime"] == regime]
        reason_counts = Counter(
            reason for row in regime_rows for reason in row["validity_reasons"]
        )
        missing_counts = Counter(
            missing for row in regime_inventory for missing in row["missing"]
        )
        by_regime[regime] = {
            "planned_scenes": len(planned),
            "selected_records": len(regime_rows),
            "success": sum(row["scene_status"] == "success" for row in regime_rows),
            "completed_scenes": sum(
                row["scene_status"] in {"success", "empty"} for row in regime_rows
            ),
            "contributing_scenes": sum(
                int(row["input_instances"]) > 0 for row in regime_rows
            ),
            "failed_or_unavailable": sum(
                row["scene_status"] in {"failed", "unavailable", "invalid"}
                for row in regime_rows
            ),
            "empty": sum(row["scene_status"] == "empty" for row in regime_rows),
            "valid": sum(bool(row["record_valid"]) for row in regime_rows),
            "input_instances": sum(int(row["input_instances"]) for row in regime_rows),
            "accepted_instances": sum(int(row["accepted_instances"]) for row in regime_rows),
            "f1_weight": sum(int(row["f1_weight"]) for row in regime_rows),
            "missing_artifact_counts": dict(sorted(missing_counts.items())),
            "validity_reason_counts": dict(sorted(reason_counts.items())),
            "build_commits": sorted({row["build_commit"] for row in regime_rows}),
        }

    summary = {
        "schema_version": 2,
        "freeze_id": freeze_id,
        "mode": "smoke" if smoke else "full",
        "paper_ready": bool(not smoke and not partial and rows and all(row["record_valid"] for row in rows)),
        "denominator_coherent": not denominator_issues,
        "denominator_issues": denominator_issues,
        "population": {
            "roster_source": roster_source,
            "planned_scene_count": len(planned),
            "planned_scene_ids": planned,
            "planned_scene_ids_sha256": _sha256(
                ("\n".join(planned) + "\n").encode("ascii")
            ),
            "planned_record_count": len(planned) * len(REQUIRED_REGIMES),
            "selected_scene_count": len(selected),
            "selected_scene_ids": selected,
            "selected_regimes": regimes,
            "selected_record_count": len(rows),
            "full_population_preserved": not partial,
        },
        "records_csv": _relative(destination / "scene_records.csv"),
        "inventory_commit": inventory_git["commit"],
        "inventory_git": inventory_git,
        "upstream_e0_contract": upstream_contract,
        "global_validity_reasons": sorted(set(global_reasons)),
        "by_regime": by_regime,
        "records": inventory_records,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging-{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        for bundle in bundles:
            row = bundle["row"]
            regime_id = str(REGIME_SPECS[row["regime"]]["id"])
            manifest = {
                "schema_version": 1,
                "import_kind": "legacy_evidence_import",
                "freeze_id": freeze_id,
                "inventory_commit": summary["inventory_commit"],
                "regime": row["regime"],
                "scene_id": row["scene_id"],
                "build_commit": row["build_commit"],
                "source_build_manifest": bundle["source_manifest"],
                "source_artifact_hash": row["source_artifact_hash"],
                "source_artifacts": bundle["source_artifacts"],
                "observed_legacy_metrics": bundle["observed_metrics"],
                "record_valid": row["record_valid"],
                "validity_reasons": row["validity_reasons"],
                "record": {key: row[key] for key in REQUIRED_COLUMNS},
            }
            _write_json(staging / "manifests" / regime_id / f"{row['scene_id']}.json", manifest)
        if _inject_failure_after == "manifests":
            raise RuntimeError("injected construction inventory failure after manifests")
        _write_records(staging / "scene_records.csv", rows)
        summary["records_sha256"] = _sha256((staging / "scene_records.csv").read_bytes())
        _write_json(staging / "missing_artifacts.json", summary)
        if _inject_failure_after == "report":
            raise RuntimeError("injected construction inventory failure after report")
        if destination.exists():
            raise InventoryError(f"inventory destination appeared during scan: {destination}")
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

    return {key: value for key, value in summary.items() if key != "records"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--expected-scenes", type=int, default=50)
    parser.add_argument(
        "--scene-list",
        help=(
            "repository-local JSON/YAML population config or one-scene-per-line roster"
        ),
    )
    parser.add_argument(
        "--contract-manifest",
        help="repository-local E0 freeze_manifest.json inherited by this audit",
    )
    parser.add_argument("--scene-id", action="append", dest="selected_scene_ids")
    parser.add_argument(
        "--regime", action="append", dest="selected_regimes",
        help="fixed paper label or regime ID; partial selections require --smoke",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        planned = None
        roster_source = None
        if args.scene_list:
            scene_list = _repo_path(args.scene_list, label="scene_list")
            if not scene_list.is_file():
                raise InventoryError(f"scene_list is not a file: {scene_list}")
            planned = _read_scene_list(scene_list)
            roster_source = _relative(scene_list)
        result = build_inventory(
            args.outputs_root,
            args.out,
            args.freeze_id,
            expected_scenes=args.expected_scenes,
            planned_scene_ids=planned,
            selected_scene_ids=args.selected_scene_ids,
            selected_regimes=args.selected_regimes,
            smoke=args.smoke,
            smoke_limit=args.smoke_limit,
            roster_source=roster_source,
            contract_manifest_path=args.contract_manifest,
        )
    except (InventoryError, OSError, RuntimeError, ValueError) as exc:
        print(f"construction inventory failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
