"""Strict aggregation for the ICRA 2027 room-construction table.

The input is one record per ``(regime, scene_id)``.  Publication denominators
come only from explicit count columns: normalized per-scene values are never
used as fallbacks.  Failed scenes remain part of the planned-scene population,
and all three output files are staged beside their destinations before being
atomically replaced.

The default path requires the complete five-regime population.  ``--smoke``
permits a canonical regime subset for fast contract checks, but its artifacts
are always marked as preliminary and not valid for the paper.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


LATEX_NEWLINE = r"\\"
SCHEMA_VERSION = 2
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONSTRUCTION_CONFIG = (
    REPOSITORY_ROOT / "configs/experiments/icra2027/construction_regimes.yaml"
)
REQUIRED_REGIMES = (
    "GT segments + scan mesh",
    "Auto discovery + scan mesh",
    "GT segments + splat-fused mesh",
    "Auto discovery + splat-fused mesh",
    "Single RGB + metric depth",
)
REQUIRED_COLUMNS = (
    "freeze_id",
    "regime",
    "scene_id",
    "scene_status",
    "input_instances",
    "accepted_instances",
    "f1_20",
    "f1_weight",
    "stable_instances",
    "tested_instances",
    "runtime_minutes",
    "build_commit",
    "build_manifest_path",
    "failure_reason",
    "source_artifact_hash",
    "record_valid",
    "validity_reasons",
    "geometry_reference_status",
)
# Explicit prospective E1 scope. Legacy records do not opt into this schema.
SCOPED_FIELDS = ("measurement_scope", "controller_accepted_instances", "runtime_components_seconds")
RUNTIME_COMPONENTS = ("input_reconstruction", "discovery_preparation", "initial_generation",
                      "registration_decisions_retry", "collision_simulator_export", "construction_drop_verification")
MEASUREMENT_SCOPE = {
    "schema_version": 1,
    "constructor_protocol": "simany_A4_FULL_shared_initial_pool_bounded_registration_retry_v1",
    "acceptance": "verified_simulator_export_bodies",
    "geometry": "independent_gt_matches_of_verified_exported_bodies",
    "physical_protocol": "canonical_factory_report_drop_v1",
    "collision_representation": "selected_frozen_E3_single_convex_hull_v1",
    "runtime": "sum_declared_stage_seconds_not_parallel_wall_latency_v1",
}


def _structured(value, field):
    if isinstance(value, str):
        try: value = json.loads(value)
        except json.JSONDecodeError as exc: raise ValueError(f"invalid {field} JSON") from exc
    if not isinstance(value, dict): raise ValueError(f"{field} must be an explicit mapping")
    return value


def _scoped_fields(raw, *, record, accepted, inputs, runtime):
    scope = _structured(raw["measurement_scope"], "measurement_scope")
    if scope != MEASUREMENT_SCOPE:
        raise ValueError(f"{record} mixed or unsupported construction measurement scope/protocol")
    controller = _count(raw, "controller_accepted_instances", record=record, nullable=True)
    if inputs is None and controller is not None:
        raise ValueError(f"{record} controller count lacks input denominator")
    if controller is not None and controller > inputs:
        raise ValueError(f"{record} controller accepted exceeds inputs")
    if accepted is not None and (controller is None or accepted > controller):
        raise ValueError(f"{record} verified exports exceed controller accepted count")
    components = _structured(raw["runtime_components_seconds"], "runtime_components_seconds")
    if set(components) != set(RUNTIME_COMPONENTS):
        raise ValueError(f"{record} runtime component roster differs")
    components = {k: _number(components, k, record=record, nullable=True) for k in RUNTIME_COMPONENTS}
    if any(v is not None and v < 0 for v in components.values()):
        raise ValueError(f"{record} runtime component is negative")
    complete = all(v is not None for v in components.values())
    if complete:
        expected = math.fsum(components.values()) / 60
        if runtime is None or not math.isclose(runtime, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{record} runtime differs from declared complete components")
    elif runtime is not None:
        raise ValueError(f"{record} incomplete runtime components cannot become total runtime")
    return {"measurement_scope": scope, "controller_accepted_instances": controller,
            "runtime_components_seconds": components}


SCENE_STATUSES = frozenset(
    {"success", "failed", "empty", "unavailable", "invalid"}
)
FAILURE_STATUSES = frozenset({"failed", "unavailable", "invalid"})
_MISSING_STRINGS = frozenset({"", "--", "null", "none"})
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_COMMIT = "legacy-unrecorded"


def _load_paper_scene_ids() -> tuple[str, ...]:
    """Load the checked-in benchmark roster and reject a malformed contract."""
    payload = yaml.safe_load(CONSTRUCTION_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("population"), dict):
        raise RuntimeError(f"invalid construction population config: {CONSTRUCTION_CONFIG}")
    population = payload["population"]
    raw_scenes = population.get("scene_ids")
    if not isinstance(raw_scenes, list):
        raise RuntimeError("construction population config is missing scene_ids")
    scene_ids = tuple(str(value).strip() for value in raw_scenes)
    if (
        population.get("planned_scenes") != 50
        or len(scene_ids) != 50
        or len(set(scene_ids)) != 50
        or any(not scene_id for scene_id in scene_ids)
    ):
        raise RuntimeError("construction population must pin exactly 50 unique scenes")
    regimes = payload.get("regimes")
    labels = (
        tuple(str(item.get("paper_label", "")).strip() for item in regimes)
        if isinstance(regimes, list) and all(isinstance(item, dict) for item in regimes)
        else ()
    )
    if labels != REQUIRED_REGIMES:
        raise RuntimeError("construction config regime labels differ from the paper contract")
    return scene_ids


PAPER_SCENE_IDS = _load_paper_scene_ids()


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("rows", "records", "scenes"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"could not find construction rows in {path}")


def _missing(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().lower() in _MISSING_STRINGS
    )


def _required_text(row: dict[str, Any], key: str, *, record: str) -> str:
    value = row[key]
    if _missing(value):
        raise ValueError(f"{record} has an empty {key}")
    return str(value).strip()


def _boolean(value: Any, *, field: str, record: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{record} requires explicit boolean {field}")


def _reason_list(value: Any) -> list[str]:
    if _missing(value):
        return []
    if isinstance(value, list):
        return [str(reason).strip() for reason in value if str(reason).strip()]
    text = str(value).strip()
    if text.startswith("["):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return [
                str(reason).strip() for reason in decoded if str(reason).strip()
            ]
    return [reason.strip() for reason in text.split(";") if reason.strip()]


def _resolve_build_manifest(value: str, *, record: str, scoped: bool = False) -> Path:
    """Keep legacy manifests local; scoped records may bind a sibling evidence checkout."""
    root = Path(os.path.abspath(REPOSITORY_ROOT))
    if scoped and os.environ.get("SIMANY_EVIDENCE_ROOT") is not None:
        from robo.eval.agentic_ablation import _validated_evidence_root
        root = _validated_evidence_root(os.environ["SIMANY_EVIDENCE_ROOT"])
    declared = Path(value)
    candidate = declared if declared.is_absolute() else root / declared
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{record} build_manifest_path resolves outside the repository: {value!r}"
        ) from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{record} build_manifest_path must not use symlinks")
    if not lexical.is_file():
        raise ValueError(f"{record} build_manifest_path is not a file: {value!r}")
    return lexical


def _validate_build_manifest(
    path: Path, normalized: dict[str, Any], *, record: str
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{record} build manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("record"), dict):
        raise ValueError(f"{record} build manifest lacks a normalized record object")

    for key in (
        "freeze_id",
        "regime",
        "scene_id",
        "build_commit",
        "source_artifact_hash",
        "record_valid",
        "validity_reasons",
    ):
        if payload.get(key) != normalized[key]:
            raise ValueError(
                f"{record} build manifest {key} disagrees with scene_records"
            )
    manifest_record = payload["record"]
    for key in REQUIRED_COLUMNS + (SCOPED_FIELDS if "measurement_scope" in normalized else ()):
        if manifest_record.get(key) != normalized[key]:
            raise ValueError(
                f"{record} build manifest record.{key} disagrees with scene_records"
            )


def _number(
    row: dict[str, Any], key: str, *, record: str, nullable: bool
) -> float | None:
    value = row[key]
    if _missing(value):
        if nullable:
            return None
        raise ValueError(f"{record} has an empty {key}")
    if isinstance(value, bool):
        raise ValueError(f"{record} has a non-numeric {key}: {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{record} has a non-numeric {key}: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{record} has a non-finite {key}: {value!r}")
    return result


def _count(
    row: dict[str, Any], key: str, *, record: str, nullable: bool = False
) -> int | None:
    value = _number(row, key, record=record, nullable=nullable)
    if value is None:
        return None
    if value < 0 or not value.is_integer():
        raise ValueError(f"{record} requires non-negative integer {key}")
    return int(value)


def _ordered_difference(values: set[str], expected: Iterable[str]) -> list[str]:
    return [value for value in expected if value not in values]


def _combined_source_hash(source_hashes: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for source_hash in sorted(source_hashes):
        digest.update(source_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_rows(
    rows: list[dict[str, Any]], *, allow_preliminary: bool = False,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Validate and normalize the complete scene-record contract."""
    if not rows:
        raise ValueError("construction input is empty")

    scoped = any("measurement_scope" in row for row in rows)
    if scoped and not all(all(k in row for k in SCOPED_FIELDS) for row in rows):
        raise ValueError("cannot mix legacy and explicitly scoped construction records")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    observed_regimes: set[str] = set()
    freeze_ids: set[str] = set()

    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"construction row {index} is not an object")
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw]
        if missing_columns:
            raise ValueError(
                f"construction row {index} missing required columns: "
                + ", ".join(missing_columns)
            )

        regime = _required_text(raw, "regime", record=f"row {index}")
        scene_id = _required_text(raw, "scene_id", record=f"row {index}")
        freeze_id = _required_text(raw, "freeze_id", record=f"row {index}")
        record = f"{regime}/{scene_id}"
        if regime not in REQUIRED_REGIMES:
            raise ValueError(f"unexpected construction regime: {regime!r}")

        pair = (regime, scene_id)
        if pair in seen:
            raise ValueError(f"duplicate construction row: {regime}/{scene_id}")
        seen.add(pair)
        observed_regimes.add(regime)
        freeze_ids.add(freeze_id)

        scene_status = _required_text(raw, "scene_status", record=record).lower()
        if scene_status not in SCENE_STATUSES:
            raise ValueError(
                f"{record} has invalid scene_status {scene_status!r}; "
                f"expected one of {sorted(SCENE_STATUSES)}"
            )

        build_commit = _required_text(raw, "build_commit", record=record)
        build_manifest_path = _required_text(
            raw, "build_manifest_path", record=record
        )
        resolved_manifest = _resolve_build_manifest(
            build_manifest_path, record=record, scoped=scoped
        )
        failure_reason = (
            "" if _missing(raw["failure_reason"]) else str(raw["failure_reason"]).strip()
        )
        if scene_status != "success" and not failure_reason:
            raise ValueError(
                f"{record} has scene_status={scene_status!r} but no failure_reason"
            )

        input_instances = _count(raw, "input_instances", record=record, nullable=scoped)
        accepted_instances = _count(raw, "accepted_instances", record=record, nullable=scoped)
        if accepted_instances is not None and input_instances is None:
            raise ValueError(f"{record} verified exports lack input denominator")
        if accepted_instances is not None and accepted_instances > input_instances:
            raise ValueError(f"{record} has accepted_instances > input_instances")

        f1_20 = _number(raw, "f1_20", record=record, nullable=True)
        f1_weight = _count(raw, "f1_weight", record=record)
        assert f1_weight is not None
        if f1_20 is None and f1_weight != 0:
            raise ValueError(f"{record} requires f1_weight=0 when f1_20 is missing")
        if f1_20 is not None:
            if not 0 <= f1_20 <= 1:
                raise ValueError(f"{record} has f1_20 outside [0, 1]")
            if f1_weight <= 0:
                raise ValueError(
                    f"{record} requires a positive explicit f1_weight when f1_20 is set"
                )
        if f1_weight > 0 and (accepted_instances is None or f1_weight > accepted_instances):
            raise ValueError(f"{record} has f1_weight > accepted_instances")

        stable_is_missing = _missing(raw["stable_instances"])
        tested_is_missing = _missing(raw["tested_instances"])
        if stable_is_missing != tested_is_missing:
            raise ValueError(
                f"{record} must provide both stability counts or leave both missing"
            )
        stable_instances = tested_instances = None
        if not stable_is_missing:
            stable_instances = _count(raw, "stable_instances", record=record)
            tested_instances = _count(raw, "tested_instances", record=record)
            assert stable_instances is not None and tested_instances is not None
            if stable_instances > tested_instances:
                raise ValueError(
                    f"{record} has stable_instances > tested_instances"
                )
            if accepted_instances is None or tested_instances > accepted_instances:
                raise ValueError(
                    f"{record} has tested_instances > accepted_instances"
                )

        runtime_minutes = _number(
            raw, "runtime_minutes", record=record, nullable=True
        )
        if runtime_minutes is not None and runtime_minutes < 0:
            raise ValueError(f"{record} has negative runtime_minutes")

        record_valid = _boolean(
            raw["record_valid"], field="record_valid", record=record
        )
        validity_reasons = _reason_list(raw["validity_reasons"])
        source_artifact_hash = _required_text(
            raw, "source_artifact_hash", record=record
        )
        if _SHA256_RE.fullmatch(source_artifact_hash) is None:
            raise ValueError(
                f"{record} requires source_artifact_hash to be a lowercase SHA-256"
            )
        geometry_reference_status = _required_text(
            raw, "geometry_reference_status", record=record
        )
        if record_valid and validity_reasons:
            raise ValueError(
                f"{record} is valid but has validity_reasons: {validity_reasons}"
            )
        if not record_valid and not validity_reasons:
            raise ValueError(f"{record} is invalid but has no validity_reasons")
        if scoped and record_valid and (input_instances is None or accepted_instances is None):
            raise ValueError(f"{record} valid scoped record lacks measured counts")
        # Independent match availability is distinct from construction yield.
        # A measured empty match set is explicit; unexecuted evaluation cannot
        # acquire that meaning merely by supplying a null score.
        no_matches = scoped and geometry_reference_status == "independent_gt_evaluation_no_matches"
        if no_matches and (f1_20 is not None or f1_weight != 0):
            raise ValueError(f"{record} independent no-matches evidence requires null F1 and zero weight")
        if record_valid and accepted_instances is not None and accepted_instances > 0 and not no_matches:
            if f1_20 is None or f1_weight <= 0:
                raise ValueError(
                    f"{record} is valid with accepted assets but lacks independent F1 evidence"
                )
            if geometry_reference_status != "independent_gt_evaluation":
                raise ValueError(
                    f"{record} valid accepted assets require "
                    "geometry_reference_status='independent_gt_evaluation'"
                )
        commit_is_git_sha = _GIT_COMMIT_RE.fullmatch(build_commit) is not None
        if record_valid and not commit_is_git_sha:
            raise ValueError(
                f"{record} requires build_commit to be a lowercase 7-40 "
                "character Git SHA"
            )
        if not record_valid and not (
            commit_is_git_sha or build_commit == _LEGACY_COMMIT
        ):
            raise ValueError(
                f"{record} invalid/preliminary build_commit must be a Git SHA "
                f"or {_LEGACY_COMMIT!r}"
            )
        if not record_valid and not allow_preliminary:
            raise ValueError(
                f"preliminary/invalid construction row refused for {record}: "
                + "; ".join(validity_reasons)
            )

        scope_values = (_scoped_fields(raw, record=record, accepted=accepted_instances,
                                      inputs=input_instances, runtime=runtime_minutes) if scoped else {})
        normalized_row = {
                **raw,
                **scope_values,
                "freeze_id": freeze_id,
                "regime": regime,
                "scene_id": scene_id,
                "scene_status": scene_status,
                "input_instances": input_instances,
                "accepted_instances": accepted_instances,
                "f1_20": f1_20,
                "f1_weight": f1_weight,
                "stable_instances": stable_instances,
                "tested_instances": tested_instances,
                "runtime_minutes": runtime_minutes,
                "build_commit": build_commit,
                "build_manifest_path": build_manifest_path,
                "failure_reason": failure_reason,
                "source_artifact_hash": source_artifact_hash,
                "record_valid": record_valid,
                "validity_reasons": validity_reasons,
                "geometry_reference_status": geometry_reference_status,
            }
        _validate_build_manifest(
            resolved_manifest, normalized_row, record=record
        )
        normalized.append(normalized_row)

    expected_regimes = set(REQUIRED_REGIMES)
    if not smoke and observed_regimes != expected_regimes:
        missing = _ordered_difference(observed_regimes, REQUIRED_REGIMES)
        extra = sorted(observed_regimes - expected_regimes)
        raise ValueError(
            "construction rows must contain exactly the five fixed regimes; "
            f"missing={missing}, extra={extra}"
        )
    if len(freeze_ids) != 1:
        raise ValueError(
            "all construction rows must share one freeze_id; "
            f"got {sorted(freeze_ids)}"
        )
    active_regimes = tuple(
        regime for regime in REQUIRED_REGIMES if regime in observed_regimes
    )
    scene_ids_by_regime = {
        regime: {
            row["scene_id"] for row in normalized if row["regime"] == regime
        }
        for regime in active_regimes
    }
    reference_regime = active_regimes[0]
    reference_scene_ids = scene_ids_by_regime[reference_regime]
    for regime in active_regimes[1:]:
        scene_ids = scene_ids_by_regime[regime]
        if scene_ids != reference_scene_ids:
            raise ValueError(
                "all five construction regimes must cover identical planned "
                f"scene IDs; {regime!r} differs from {reference_regime!r}: "
                f"missing={sorted(reference_scene_ids - scene_ids)}, "
                f"extra={sorted(scene_ids - reference_scene_ids)}"
            )
    if not smoke and reference_scene_ids != set(PAPER_SCENE_IDS):
        raise ValueError(
            "paper construction rows must cover the exact pinned 50-scene roster; "
            f"missing={sorted(set(PAPER_SCENE_IDS) - reference_scene_ids)}, "
            f"extra={sorted(reference_scene_ids - set(PAPER_SCENE_IDS))}"
        )
    return normalized


def aggregate(
    rows: list[dict[str, Any]], *, allow_preliminary: bool = False,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Aggregate validated rows in the paper's fixed regime order."""
    values = validate_rows(
        rows, allow_preliminary=allow_preliminary, smoke=smoke
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        grouped[row["regime"]].append(row)

    output: list[dict[str, Any]] = []
    active_regimes = tuple(
        regime for regime in REQUIRED_REGIMES if regime in grouped
    )
    for regime in active_regimes:
        records = sorted(grouped[regime], key=lambda row: row["scene_id"])
        scoped = "measurement_scope" in records[0]
        complete_inputs = all(row["input_instances"] is not None for row in records)
        complete_exports = all(row["accepted_instances"] is not None for row in records)
        input_instances = sum(row["input_instances"] for row in records) if complete_inputs else None
        accepted_instances = sum(row["accepted_instances"] for row in records) if complete_exports else None

        f1_records = [row for row in records if row["f1_weight"] > 0]
        f1_weight = math.fsum(row["f1_weight"] for row in f1_records)
        f1_20 = (
            math.fsum(row["f1_20"] * row["f1_weight"] for row in f1_records)
            / f1_weight
            if f1_weight
            else None
        )

        stability_records = [
            row for row in records if row["tested_instances"] is not None
        ]
        stable_instances = sum(
            row["stable_instances"] for row in stability_records
        )
        tested_instances = sum(
            row["tested_instances"] for row in stability_records
        )
        runtimes = [
            row["runtime_minutes"]
            for row in records
            if row["runtime_minutes"] is not None
        ]

        successful_scenes = sum(
            row["scene_status"] == "success" for row in records
        )
        completed_scenes = sum(
            row["scene_status"] in {"success", "empty"} for row in records
        )
        contributing_scenes = sum(
            row["input_instances"] is not None and row["input_instances"] > 0 for row in records
        )
        validity_reasons = sorted(
            {
                reason
                for row in records
                for reason in row["validity_reasons"]
            }
        )
        scope_totals = {}
        if scoped:
            controllers = [r["controller_accepted_instances"] for r in records]
            controller_total = sum(controllers) if all(v is not None for v in controllers) else None
            scope_totals = {"measurement_scope": records[0]["measurement_scope"],
                "controller_accepted_instances": controller_total,
                "controller_coverage": controller_total / input_instances if input_instances and controller_total is not None else None,
                "verified_export_count_observed": sum(r["accepted_instances"] or 0 for r in records),
                "verified_export_scene_count": sum(r["accepted_instances"] is not None for r in records),
                "runtime_complete_scenes": len(runtimes),
                "runtime_components_seconds": {key: math.fsum(r["runtime_components_seconds"][key] for r in records)
                    if all(r["runtime_components_seconds"][key] is not None for r in records) else None for key in RUNTIME_COMPONENTS}}
        output.append(
            {
                **scope_totals,
                "freeze_id": records[0]["freeze_id"],
                "regime": regime,
                "scenes": len(records),
                "planned_scenes": len(records),
                "successful_scenes": successful_scenes,
                "completed_scenes": completed_scenes,
                "contributing_scenes": contributing_scenes,
                "failed_scenes": sum(
                    row["scene_status"] in FAILURE_STATUSES for row in records
                ),
                "empty_scenes": sum(
                    row["scene_status"] == "empty" for row in records
                ),
                "instances": input_instances,
                "accepted_instances": accepted_instances,
                "yield": (
                    accepted_instances / input_instances
                    if input_instances and accepted_instances is not None
                    else None
                ),
                "f1_20": f1_20,
                "f1_weight": f1_weight,
                "stable_instances": (
                    stable_instances if stability_records else None
                ),
                "tested_instances": (
                    tested_instances if stability_records else None
                ),
                "stability": (
                    stable_instances / tested_instances
                    if tested_instances
                    else None
                ),
                "runtime_minutes": (
                    math.fsum(runtimes) / len(runtimes) if runtimes and (not scoped or len(runtimes)==len(records)) else None
                ),
                "runtime_scene_count": len(runtimes),
                "valid_for_paper": all(row["record_valid"] for row in records) and (not scoped or
                    (complete_inputs and complete_exports and len(runtimes)==len(records) and
                     all(r["accepted_instances"]==0 or (r["tested_instances"] is not None and
                         r["tested_instances"]==r["accepted_instances"]) for r in records))),
                "invalid_record_count": sum(
                    not row["record_valid"] for row in records
                ),
                "validity_reasons": "; ".join(validity_reasons),
                "source_artifact_hash": _combined_source_hash(
                    row["source_artifact_hash"] for row in records
                ),
                "geometry_reference_statuses": "; ".join(
                    sorted(
                        {
                            row["geometry_reference_status"]
                            for row in records
                        }
                    )
                ),
            }
        )
    return output


def _fmt(value: Any, *, percent: bool = False, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    if percent:
        return f"{100.0 * float(value):.1f}\\%"
    return f"{float(value):.{digits}f}"


def render_latex(
    rows: list[dict[str, Any]], *, paper_ready: bool = True,
    smoke: bool = False,
) -> str:
    if smoke:
        warning = [
            "% SMOKE/PRELIMINARY OUTPUT - NOT VALID FOR PAPER",
            r"\textbf{SMOKE/PRELIMINARY OUTPUT---NOT VALID FOR PAPER}",
        ]
    elif not paper_ready:
        warning = [
            "% PRELIMINARY AUDIT OUTPUT - NOT VALID FOR PAPER",
            r"\textbf{PRELIMINARY AUDIT OUTPUT---NOT VALID FOR PAPER}",
        ]
    else:
        warning = []
    lines = [
        r"\begin{table}[t]",
        r"\caption{\textbf{Automatic room construction.} Build yield uses summed instance counts, geometry uses explicitly weighted independent matches, and stability uses only attempted drop tests. Failed scenes remain in the planned-scene counts in the JSON and CSV artifacts.}",
        r"\label{tab:construction}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        (
            r"Input regime & inst./planned scenes & yield $\uparrow$ & "
            r"F1@20 $\uparrow$ & stable $\uparrow$ & min/scene $\downarrow$ "
            + LATEX_NEWLINE
        ),
        r"\midrule",
    ]
    scoped = bool(rows and "measurement_scope" in rows[0])
    if scoped:
        lines[1] = (r"\caption{\textbf{Construction scope audit.} Yield counts verified simulator-export bodies over all discovered inputs; controller acceptance is reported separately in JSON/CSV. Geometry uses independent matched exported bodies. Stability uses the canonical factory-report drop protocol. Runtime requires every declared component and is a sum of stage durations, not parallel wall latency. Missing counts and incomplete runtime remain unavailable.}")
    lines[1:1] = warning
    for row in rows:
        lines.append(
            f"{row['regime']} & {row['instances'] if row['instances'] is not None else '--'}/{row['planned_scenes']} & "
            f"{_fmt(row['yield'], percent=True)} & {_fmt(row['f1_20'])} & "
            f"{_fmt(row['stability'], percent=True)} & "
            f"{_fmt(row['runtime_minutes'], digits=1)} {LATEX_NEWLINE}"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def _csv_text(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    fieldnames = list(rows[0]) if rows else []
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows({k: json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v
                      for k,v in row.items()} for row in rows)
    return stream.getvalue()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bundle(out_dir: Path, outputs: dict[str, str]) -> None:
    """Publish a complete new directory atomically and never overwrite it."""
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite construction table: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(f".{out_dir.name}.staging-{uuid.uuid4().hex}")
    try:
        staging.mkdir()
        for name, contents in outputs.items():
            destination = staging / name
            with destination.open("x", encoding="utf-8", newline="") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        if out_dir.exists() or out_dir.is_symlink():
            raise FileExistsError(
                f"construction table destination appeared during generation: {out_dir}"
            )
        staging.rename(out_dir)
        _fsync_directory(out_dir.parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def generate(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    allow_preliminary: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_path)
    raw_rows = _read_rows(input_path)
    rows = aggregate(
        raw_rows,
        allow_preliminary=allow_preliminary,
        smoke=smoke,
    )
    paper_ready = not smoke and not allow_preliminary and all(
        row["valid_for_paper"] for row in rows
    )
    manifest_entries = []
    for raw in raw_rows:
        record = f"{raw.get('regime', '<missing>')}/{raw.get('scene_id', '<missing>')}"
        declared = str(raw["build_manifest_path"]).strip()
        path = _resolve_build_manifest(declared, record=record, scoped="measurement_scope" in raw)
        manifest_entries.append(
            {
                "path": declared,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest_digest = hashlib.sha256()
    for entry in sorted(manifest_entries, key=lambda item: (item["path"], item["sha256"])):
        manifest_digest.update(
            f"{entry['sha256']}  {entry['path']}\n".encode("utf-8")
        )
    payload = {
        "schema_version": 3 if rows and "measurement_scope" in rows[0] else SCHEMA_VERSION,
        "paper_ready": paper_ready,
        "preliminary_override": bool(allow_preliminary),
        "smoke": bool(smoke),
        "provenance": {
            "scene_records_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "build_manifest_count": len(manifest_entries),
            "build_manifest_set_sha256": manifest_digest.hexdigest(),
            "construction_config_source_sha256": hashlib.sha256(
                CONSTRUCTION_CONFIG.read_bytes()
            ).hexdigest(),
        },
        "rows": rows,
    }
    out_dir = Path(out_dir)
    outputs = {
        "construction_table.json": (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ),
        "construction_table.csv": _csv_text(rows),
        "construction_table.tex": render_latex(
            rows, paper_ready=paper_ready, smoke=smoke
        ),
    }
    _atomic_write_bundle(out_dir, outputs)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "accept a nonempty canonical regime subset for smoke validation; "
            "output is always marked paper_ready=false"
        ),
    )
    parser.add_argument(
        "--allow-preliminary",
        action="store_true",
        help=(
            "render invalid records for artifact audit only; output is always "
            "marked paper_ready=false"
        ),
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            generate(
                args.input,
                args.out,
                allow_preliminary=args.allow_preliminary,
                smoke=args.smoke,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
