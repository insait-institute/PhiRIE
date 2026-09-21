"""Build E6 task-support features, then join separate evaluation labels.

``--smoke`` produces synthetic fixtures, never the 72-row paper population.
``--stage features`` consumes only hashed public construction bundles.
``--stage join`` verifies the feature seal before opening evaluation inputs.
Feature extraction is published unlabeled first; labels are produced by a
separate stage and joined one-to-one only after the unlabeled CSV is closed.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

from robo.eval.audit_loso import SIGNAL_GROUP_NAMES, signal_groups

ROOT = Path(__file__).resolve().parents[2]
SMOKE_FREEZE_ID = "e6-fast-smoke-v1"
SMOKE_SCENES = ("synthetic_scene_00", "synthetic_scene_01")
JOIN_KEY = ("freeze_id", "scene_id", "task_id", "condition_id")

IDENTITY_FIELDS = (
    "freeze_id",
    "scene_id",
    "task_id",
    "task_family",
    "condition_id",
    "graph_path",
    "build_manifest_path",
)
FEATURE_FIELDS = (
    "global_psnr_db",
    "global_f1_score",
    "scene_build_completeness",
    "scene_collision_stability",
    "visual_target_iou",
    "visual_target_iou_missing",
    "geometry_translation_error_cm",
    "geometry_scale_error_pct",
    "support_contact_confidence",
    "support_contact_missing",
    "robot_alignment_error_cm",
    "robot_workspace_visibility",
)
LABEL_FIELDS = (*JOIN_KEY, "invalid_label", "invalid_reasons")
JOINED_FIELDS = (
    "freeze_id",
    "scene_id",
    "task_id",
    "task_family",
    "condition_id",
    "invalid_label",
    "invalid_reasons",
    "graph_path",
    "build_manifest_path",
    *FEATURE_FIELDS,
)

# Feature specifications contain no validity labels or hidden-GT reason codes.
_QUERY_SPECS = (
    {
        "task_id": "move_mug_to_region",
        "task_family": "object_to_region",
        "condition_id": "clean",
        "target": "mug",
        "destination": "left_table_region",
    },
    {
        "task_id": "put_bottle_in_tray",
        "task_family": "object_to_receptacle",
        "condition_id": "mild",
        "target": "bottle",
        "destination": "tray",
    },
    {
        "task_id": "move_book_to_region",
        "task_family": "object_to_region",
        "condition_id": "severe",
        "target": "book",
        "destination": "right_table_region",
    },
)

# Synthetic labels are deliberately isolated from the feature profiles above.
_SMOKE_LABELS = {
    ("synthetic_scene_00", "move_mug_to_region", "clean"): (0, ""),
    ("synthetic_scene_00", "put_bottle_in_tray", "mild"): (0, ""),
    ("synthetic_scene_00", "move_book_to_region", "severe"): (
        1, "required_object_not_recovered;required_visible_failed"),
    ("synthetic_scene_01", "move_mug_to_region", "clean"): (0, ""),
    ("synthetic_scene_01", "put_bottle_in_tray", "mild"): (
        1, "support_correct_failed"),
    ("synthetic_scene_01", "move_book_to_region", "severe"): (
        1, "collision_valid_failed;translation_cm_exceeds_threshold"),
}


def _feature_values(scene_index: int, condition: str) -> dict[str, Any]:
    if condition == "clean":
        return {
            "global_psnr_db": 31.0 + 0.4 * scene_index,
            "global_f1_score": 0.94 - 0.01 * scene_index,
            "scene_build_completeness": 0.98,
            "scene_collision_stability": 0.97,
            "visual_target_iou": 0.95 - 0.01 * scene_index,
            "visual_target_iou_missing": 0,
            "geometry_translation_error_cm": 0.8 + 0.2 * scene_index,
            "geometry_scale_error_pct": 2.0 + scene_index,
            "support_contact_confidence": 0.96,
            "support_contact_missing": 0,
            "robot_alignment_error_cm": 0.4 + 0.1 * scene_index,
            "robot_workspace_visibility": 0.97,
        }
    if condition == "mild":
        return {
            "global_psnr_db": 27.4 - 0.3 * scene_index,
            "global_f1_score": 0.83 - 0.03 * scene_index,
            "scene_build_completeness": 0.86 - 0.04 * scene_index,
            "scene_collision_stability": 0.84 - 0.14 * scene_index,
            "visual_target_iou": 0.82 - 0.14 * scene_index,
            "visual_target_iou_missing": 0,
            "geometry_translation_error_cm": 3.2 + 2.5 * scene_index,
            "geometry_scale_error_pct": 9.0 + 5.0 * scene_index,
            "support_contact_confidence": 0.78 - 0.34 * scene_index,
            "support_contact_missing": 0,
            "robot_alignment_error_cm": 1.5 + 1.2 * scene_index,
            "robot_workspace_visibility": 0.82 - 0.12 * scene_index,
        }
    if condition == "severe":
        return {
            "global_psnr_db": 20.5 - 0.5 * scene_index,
            "global_f1_score": 0.47 - 0.05 * scene_index,
            "scene_build_completeness": 0.48 - 0.08 * scene_index,
            "scene_collision_stability": 0.34 - 0.08 * scene_index,
            "visual_target_iou": 0.31 if scene_index == 0 else "",
            "visual_target_iou_missing": scene_index,
            "geometry_translation_error_cm": 11.0 + 2.0 * scene_index,
            "geometry_scale_error_pct": 27.0 + 4.0 * scene_index,
            "support_contact_confidence": "" if scene_index else 0.25,
            "support_contact_missing": int(scene_index == 1),
            "robot_alignment_error_cm": 6.0 + scene_index,
            "robot_workspace_visibility": 0.42 - 0.08 * scene_index,
        }
    raise ValueError(f"unknown smoke condition {condition!r}")


def _checked_output(value: str | Path) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else ROOT / raw
    if candidate.is_symlink():
        raise ValueError(f"smoke output cannot be a symlink: {candidate}")
    output = candidate.resolve(strict=False)
    root = ROOT.resolve()
    if os.environ.get("SIMANY_EVIDENCE_ROOT"):
        from robo.eval.e3_factory_materializer import _validated_evidence_root
        root = _validated_evidence_root(os.environ["SIMANY_EVIDENCE_ROOT"])
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"smoke output must remain inside repository: {output}") from exc
    if output == root:
        raise ValueError("smoke output cannot be the repository root")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite E6 smoke output: {output}")
    return output


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def _atomic_output(value: str | Path) -> Iterator[tuple[Path, Path]]:
    output = _checked_output(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        yield staging, output
        _fsync_directory(staging)
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"E6 smoke output appeared during publication: {output}")
        staging.rename(output)
        _fsync_directory(output.parent)
    except Exception:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def _write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_new(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text_new(path, text)


def _write_csv_new(path: Path, rows: list[dict[str, Any]],
                   fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _graph(scene_id: str, query: dict[str, str]) -> dict[str, Any]:
    condition = query["condition_id"]
    unresolved = ["manipulated_object"] if condition == "severe" else []
    return {
        "condition_id": condition,
        "edges": [
            {"relation": "observes", "source": "policy_camera", "target": "workspace"},
            {"relation": "manipulates", "source": "robot", "target": "target"},
            {"relation": "places_into", "source": "target", "target": "destination"},
        ],
        "mode": "deterministic_synthetic_smoke",
        "nodes": [
            {"id": "robot", "kind": "robot", "provenance": "synthetic_fixture"},
            {"id": "policy_camera", "kind": "camera", "provenance": "synthetic_fixture"},
            {"id": "target", "kind": "object", "label": query["target"],
             "provenance": "synthetic_feature_builder"},
            {"id": "destination", "kind": "destination",
             "label": query["destination"], "provenance": "synthetic_feature_builder"},
        ],
        "paper_ready": False,
        "scene_id": scene_id,
        "task_family": query["task_family"],
        "task_id": query["task_id"],
        "unresolved_references": unresolved,
    }


def _build_manifest(scene_id: str, query: dict[str, str]) -> dict[str, Any]:
    status = {"clean": "complete", "mild": "degraded", "severe": "failed"}[
        query["condition_id"]]
    return {
        "build_status": status,
        "condition_id": query["condition_id"],
        "manifest_kind": "e6_synthetic_smoke_build",
        "paper_ready": False,
        "scene_id": scene_id,
        "task_id": query["task_id"],
    }


def build_smoke_features(staging: Path) -> list[dict[str, Any]]:
    """Write graphs/manifests and the label-free feature table."""
    rows: list[dict[str, Any]] = []
    for scene_index, scene_id in enumerate(SMOKE_SCENES):
        for query in _QUERY_SPECS:
            relative_graph = Path("graphs") / scene_id / query["task_id"] / (
                f"{query['condition_id']}.json")
            relative_manifest = Path("build_manifests") / scene_id / (
                f"{query['task_id']}__{query['condition_id']}.json")
            _write_json_new(staging / relative_graph, _graph(scene_id, query))
            _write_json_new(
                staging / relative_manifest, _build_manifest(scene_id, query))
            rows.append({
                "freeze_id": SMOKE_FREEZE_ID,
                "scene_id": scene_id,
                "task_id": query["task_id"],
                "task_family": query["task_family"],
                "condition_id": query["condition_id"],
                "graph_path": relative_graph.as_posix(),
                "build_manifest_path": relative_manifest.as_posix(),
                **_feature_values(scene_index, query["condition_id"]),
            })
    _write_csv_new(
        staging / "features_unlabeled.csv", rows, (*IDENTITY_FIELDS, *FEATURE_FIELDS))
    return rows


def build_smoke_labels(staging: Path) -> list[dict[str, Any]]:
    """Write synthetic labels without receiving or reading feature values."""
    rows: list[dict[str, Any]] = []
    for scene_id in SMOKE_SCENES:
        for query in _QUERY_SPECS:
            lookup = (scene_id, query["task_id"], query["condition_id"])
            invalid_label, reasons = _SMOKE_LABELS[lookup]
            rows.append({
                "freeze_id": SMOKE_FREEZE_ID,
                "scene_id": scene_id,
                "task_id": query["task_id"],
                "condition_id": query["condition_id"],
                "invalid_label": invalid_label,
                "invalid_reasons": reasons,
            })
    _write_csv_new(staging / "labels.csv", rows, LABEL_FIELDS)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def join_labels(features_path: Path, labels_path: Path, output_path: Path) -> list[dict]:
    """Join labels one-to-one onto an already-persisted unlabeled feature CSV."""
    features = _read_csv(features_path)
    labels = _read_csv(labels_path)

    def key(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(row[field] for field in JOIN_KEY)

    feature_keys = [key(row) for row in features]
    label_keys = [key(row) for row in labels]
    if len(set(feature_keys)) != len(feature_keys):
        raise ValueError("unlabeled feature join keys are not unique")
    if len(set(label_keys)) != len(label_keys):
        raise ValueError("label join keys are not unique")
    if set(feature_keys) != set(label_keys):
        missing = sorted(set(feature_keys) - set(label_keys))
        extra = sorted(set(label_keys) - set(feature_keys))
        raise ValueError(f"label join is not one-to-one: missing={missing}, extra={extra}")
    labels_by_key = {key(row): row for row in labels}
    joined = []
    for feature in features:
        label = labels_by_key[key(feature)]
        joined.append({
            **feature,
            "invalid_label": label["invalid_label"],
            "invalid_reasons": label["invalid_reasons"],
        })
    fields = [*IDENTITY_FIELDS[:5], "invalid_label", "invalid_reasons",
              *[name for name in features[0] if name not in IDENTITY_FIELDS[:5]]]
    _write_csv_new(output_path, joined, fields)
    return joined


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_smoke(out: str | Path) -> dict[str, Any]:
    groups = signal_groups(list(FEATURE_FIELDS))
    if tuple(groups) != SIGNAL_GROUP_NAMES:
        raise AssertionError("E6 signal groups differ from the fixed six-row contract")
    with _atomic_output(out) as (staging, output):
        config = {
            "conditions": ["clean", "mild", "severe"],
            "freeze_id": SMOKE_FREEZE_ID,
            "mode": "deterministic_synthetic_fast_smoke",
            "paper_ready": False,
            "queries_per_scene": 3,
            "scene_ids": list(SMOKE_SCENES),
            "synthetic_only": True,
        }
        _write_text_new(
            staging / "config_resolved.yaml",
            yaml.safe_dump(config, sort_keys=True))

        features = build_smoke_features(staging)
        unlabeled_hash = _sha256(staging / "features_unlabeled.csv")
        labels = build_smoke_labels(staging)
        joined = join_labels(
            staging / "features_unlabeled.csv",
            staging / "labels.csv",
            staging / "task_local_features_and_labels.csv",
        )
        members = {}
        for path in sorted(value for value in staging.rglob("*") if value.is_file()):
            relative = path.relative_to(staging).as_posix()
            members[relative] = {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        manifest = {
            "feature_generation_read_labels": False,
            "features_unlabeled_sha256_before_label_join": unlabeled_hash,
            "label_join_key": list(JOIN_KEY),
            "manifest_kind": "e6_task_support_fast_smoke",
            "members": members,
            "mode": "deterministic_synthetic_fast_smoke",
            "paper_ready": False,
            "query_rows": len(joined),
            "scenes": list(SMOKE_SCENES),
            "schema_version": 1,
            "signal_groups": groups,
            "stage_order": ["features_unlabeled", "labels", "label_join"],
            "synthetic_only": True,
        }
        if len(features) != 6 or len(labels) != 6 or len(joined) != 6:
            raise AssertionError("E6 smoke must contain exactly six query rows")
        _write_json_new(staging / "smoke_manifest.json", manifest)
    return {
        "mode": manifest["mode"],
        "out": str(output),
        "paper_ready": False,
        "query_rows": 6,
        "signals": list(groups),
    }


def _public_payload(value: Any) -> None:
    """Reject known evaluation channels before invoking construction tools."""
    if isinstance(value, dict):
        for key, child in value.items():
            name = str(key).lower()
            if (name.startswith(("gt_", "global_f1", "global_psnr", "invalid_"))
                    or any(token in name for token in ("ground_truth", "held_out", "threshold"))
                    or name in {"success", "reward", "label_definition", "role_refs"}):
                raise ValueError(f"forbidden evaluation field in public construction input: {key}")
            _public_payload(child)
    elif isinstance(value, list):
        for child in value:
            _public_payload(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite public evidence")


def _public_read(reference: dict, roots: list[Path]) -> tuple[dict, dict]:
    path = Path(reference["path"]).resolve(strict=True)
    if (not any(path.is_relative_to(root) for root in roots)
            or any("vault" in part.lower() or part.lower() in {"gt", "ground_truth"}
                   for part in path.parts)):
        raise ValueError(f"public artifact escapes approved roots or enters vault: {path}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != reference["sha256"]:
        raise ValueError(f"public artifact hash mismatch: {path}")
    payload = yaml.safe_load(data)
    _public_payload(payload)
    return payload, {"path": str(path), "sha256": digest}


def _population(config: dict) -> set[tuple[str, str, str, str]]:
    scenes = config["scene_ids"]
    queries = config["task_ids_by_scene"]
    conditions = config["conditions"]
    if not scenes or set(queries) != set(scenes) or any(not queries[s] for s in scenes):
        raise ValueError("missing predeclared task roster; cannot infer queries from successful builds")
    if (len(set(scenes)) != len(scenes) or len(set(conditions)) != len(conditions)
            or any(len(set(queries[s])) != len(queries[s]) for s in scenes)):
        raise ValueError("duplicate planned population identity")
    if config["tier"] == "full" and (len(scenes) != 6 or set(conditions) != {"clean", "mild", "severe"}
                                    or any(len(queries[s]) != 4 for s in scenes)):
        raise ValueError("full E6 requires six scenes, four queries each, three conditions (72 rows)")
    return {(config["freeze_id"], scene, task, condition)
            for scene in scenes for task in queries[scene] for condition in conditions}


def _construction_features(bundle: dict) -> tuple[dict, dict]:
    from robo.certification.task_graph import build_graph
    from robo.certification.features.visual import extract_visual_features
    from robo.certification.features.geometry import extract_geometry_features
    from robo.certification.features.support_contact import extract_support_contact_features
    from robo.certification.features.robot_control import extract_robot_control_features

    failed = bundle["build_status"] != "complete"
    graph = ({"scene_id": bundle["scene_id"], "task_id": bundle["task_id"], "nodes": [],
              "edges": [], "role_resolutions": {}, "unresolved_references": ["build_failed"]}
             if failed else build_graph(bundle["scene"], bundle["task"]))
    audit = {"objects": []} if failed else bundle.get("audit", {"objects": []})
    # Normalize absent status evidence to missing before using existing producers.
    checks = ("ghost_object", "penetration", "support_overlap", "drop_response")
    for obj in audit.get("objects", []):
        for name in checks:
            check = obj.setdefault("checks", {}).setdefault(name, {})
            check.setdefault("status", "not_applicable")
    manipulated = [h["object_id"] for h in graph.get("role_resolutions", {}).get(
        "manipulated_object", {}).get("hypotheses", [])]
    local = sorted({node["node_id"] for node in graph["nodes"]
                    if node.get("kind") == "object" and node.get("roles")})
    graph["feature_scopes"] = {"manipulated_object": manipulated, "task_graph": local}

    def vg(ids):
        visual = extract_visual_features(audit, ids)
        # These are scene-level held-out/alignment values, not local evidence.
        visual = {k: v for k, v in visual.items()
                  if not k.startswith(("visual_held_out_", "visual_camera_reprojection_"))}
        geometry = {k.replace("geom_", "geometry_", 1): v for k, v in
                    extract_geometry_features(audit, ids, task_graph=graph).items()
                    if not k.startswith("geom_scale_ci95_")}
        return {**visual, **geometry}

    unresolved = [role for role in graph.get("roles_requested", [])
                  if not graph.get("role_resolutions", {}).get(role, {}).get("hypotheses")]
    graph["unresolved_references"] = sorted(set(graph.get("unresolved_references", [])) | set(unresolved))
    missing = graph.get("missing_evidence", [])
    values = {"scene_build_failed": float(failed),
              "geometry_required_role_missing": float(bool(unresolved) or failed),
              "robot_frame_missing": float("robot_frame" in missing or failed),
              "geometry_policy_cameras_missing": float("policy_cameras" in missing or failed)}
    values.update({f"scene_{k}": v for k, v in vg(None).items()})
    values.update({f"object_{k}": v for k, v in vg(manipulated).items()})
    values.update(vg(local))
    values.update(extract_support_contact_features(audit, local, task_graph=graph))
    robot = extract_robot_control_features(audit, task_graph=graph if not failed else None)
    values.update({k: v for k, v in robot.items() if not k.startswith("robot_open_loop_")})
    for key, value in values.items():
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"nonfinite constructed feature: {key}")
    return graph, values


def generate_features(config_path: Path, out: str | Path) -> dict:
    """Seal public features. This stage never opens the label protocol or vault."""
    config = yaml.safe_load(config_path.read_text())
    if config.get("label_definition") != "construction_validity":
        raise ValueError("E6 label_definition must be construction_validity")
    if config.get("freeze_id") in {None, "inherit"}:
        raise ValueError("resolve freeze_id before feature construction")
    if config.get("tier") not in {"pilot", "full", "fixture"}:
        raise ValueError("tier must be pilot, full, or fixture")
    if config.get("source_kind") not in {"real", "synthetic_fixture"}:
        raise ValueError("declare real or synthetic_fixture source_kind")
    if config["tier"] != "fixture" and config["source_kind"] != "real":
        raise ValueError("synthetic fixtures cannot be pilot/full experiments")
    expected = _population(config)
    protocol_hash = config.get("label_protocol_sha256", "")
    if len(protocol_hash) != 64:
        raise ValueError("pin label protocol hash before feature construction")
    roots = [Path(root).resolve(strict=True) for root in config["public_roots"]]
    manifest, manifest_ref = _public_read(config["public_manifest"], roots)
    if manifest.get("freeze_id") != config["freeze_id"]:
        raise ValueError("mixed freeze in public manifest")
    planned = manifest["rows"]
    keys = [tuple(row[field] for field in JOIN_KEY) for row in planned]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("public manifest does not match exact predeclared population")
    source_refs, rows = [manifest_ref], []
    public_inputs = [_public_read(item["build_manifest"], roots) for item in planned]
    references = []
    public_count = 0
    for bundle, _ in public_inputs:
        if bundle.get("public_grounding") is not None:
            public_count += 1
            if bundle["public_grounding"] not in references:
                references.append(bundle["public_grounding"])
    if public_count and public_count != len(planned):
        raise ValueError("cannot mix authenticated grounding and unbound public bundles")
    from robo.certification.public_feature_bridge import validate_references
    public_grounding_cache = validate_references(references, tier=config["tier"]) if references else {}
    with _atomic_output(out) as (staging, output):
        for index, item in enumerate(planned):
            bundle, source_ref = public_inputs[index]
            if any(bundle.get(k) != item[k] for k in JOIN_KEY):
                raise ValueError("build manifest identity/freeze mismatch")
            if bundle.get("evidence_source") != "construction_observation":
                raise ValueError("build evidence must declare construction_observation provenance")
            if bundle.get("source_kind") != config["source_kind"]:
                raise ValueError("mixed real and synthetic construction inputs")
            if bundle.get("build_status") not in {"complete", "failed", "rejected", "abstained"}:
                raise ValueError("missing terminal build status; cannot silently drop planned jobs")
            if bundle["build_status"] == "complete" and (
                    bundle["scene"].get("scene_id") != item["scene_id"]
                    or bundle["task"].get("task_id") != item["task_id"]):
                raise ValueError("scene/task content does not match planned row")
            from robo.certification.public_feature_bridge import authenticate_bundle
            authenticate_bundle(bundle, public_grounding_cache)
            graph, features = _construction_features(bundle)
            graph.setdefault("unresolved_references", [role for role in graph.get("roles_requested", [])
                              if not graph.get("role_resolutions", {}).get(role, {}).get("hypotheses")])
            graph["input_provenance"] = source_ref
            for member in [*graph["nodes"], *graph["edges"]]:
                member["construction_manifest_sha256"] = source_ref["sha256"]
            graph_name = f"graphs/{index:04d}.json"
            _write_json_new(staging / graph_name, graph)
            source_refs.append(source_ref)
            rows.append({**{k: item[k] for k in JOIN_KEY}, "task_family": item["task_family"],
                         "graph_path": str(output / graph_name), "build_manifest_path": source_ref["path"],
                         **features})
        fields = [*IDENTITY_FIELDS, *sorted(set(rows[0]) - set(IDENTITY_FIELDS))]
        _write_csv_new(staging / "features_unlabeled.csv", rows, fields)
        _write_json_new(staging / "config_resolved.json", config)
        members = {str(path.relative_to(staging)): _sha256(path)
                   for path in sorted(staging.rglob("*")) if path.is_file()}
        seal = {"schema_version": 1, "freeze_id": config["freeze_id"], "query_rows": len(rows),
                "source_kind": config["source_kind"], "paper_ready": False,
                "feature_generation_read_labels": False, "label_protocol_sha256": protocol_hash,
                "config_sha256": _sha256(config_path), "source_refs": source_refs, "members": members,
                "evaluation_baseline_columns": ["global_psnr_db", "global_f1_score"],
                "driver_sha256": _sha256(Path(__file__))}
        _write_json_new(staging / "feature_seal.json", seal)
    return {"out": str(output), "query_rows": len(rows), "paper_ready": False,
            "stage": "features", "source_kind": config["source_kind"]}


def join_evaluation(features_dir: Path, protocol_path: Path, measurements_path: Path,
                    measurements_sha256: str, out: str | Path) -> dict:
    """Evaluation-only, no permission changes; verify every sealed feature first."""
    from robo.eval import audit_labels
    seal_path = features_dir / "feature_seal.json"
    seal = json.loads(seal_path.read_text())
    for name, digest in seal["members"].items():
        path = (features_dir / name).resolve(strict=True)
        if not path.is_relative_to(features_dir.resolve()) or _sha256(path) != digest:
            raise ValueError(f"feature seal mismatch: {name}")
    if "features_unlabeled.csv" not in seal["members"]:
        raise ValueError("feature seal omits unlabeled table")
    for reference in seal["source_refs"]:
        if _sha256(Path(reference["path"])) != reference["sha256"]:
            raise ValueError(f"construction input changed after feature seal: {reference['path']}")
    if _sha256(protocol_path) != seal["label_protocol_sha256"]:
        raise ValueError("label protocol was not pinned before features")
    protocol = yaml.safe_load(protocol_path.read_text())
    if protocol.get("label_definition") != "construction_validity":
        raise ValueError("wrong evaluation label definition")
    if protocol.get("label_producer_sha256") != _sha256(Path(audit_labels.__file__)):
        raise ValueError("label producer changed since protocol declaration")
    thresholds = audit_labels.ValidityThresholds(**protocol["thresholds"])
    if any(not math.isfinite(v) or v < 0 for v in thresholds.__dict__.values()):
        raise ValueError("invalid pinned validity thresholds")
    if _sha256(measurements_path) != measurements_sha256:
        raise ValueError("evaluation measurement hash mismatch")
    measurements = yaml.safe_load(measurements_path.read_text())
    if (measurements.get("freeze_id") != seal["freeze_id"] or
            measurements.get("feature_seal_sha256") != _sha256(seal_path)):
        raise ValueError("evaluation measurements refer to a different freeze/feature seal")
    features = _read_csv(features_dir / "features_unlabeled.csv")
    by_key = {tuple(row[k] for k in JOIN_KEY): row for row in features}
    records = measurements["rows"]
    keys = [tuple(row[k] for k in JOIN_KEY) for row in records]
    if len(set(keys)) != len(keys) or set(keys) != set(by_key) or len(by_key) != len(features):
        raise ValueError("evaluation join requires exact unique planned key coverage")
    labels, baselines = [], {}
    for record, key in zip(records, keys):
        if any(k in record for k in ("success", "manipulation_success", "invalid_label")):
            raise ValueError("validity must be derived from invariants, never rollout outcomes")
        if record.get("measurement_scope") != "all_required_task_invariants":
            raise ValueError("measurement scope must include every required task invariant")
        for name in thresholds.__dict__:
            value = record.get(name)
            if value not in {None, "", "--"} and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"invalid numeric validity measurement: {name}")
        result = audit_labels.label_record(record, thresholds)
        reasons = list(result["invalid_reasons"])
        if float(by_key[key]["scene_build_failed"]):
            reasons.append("construction_failed")
        # Explicitly absent construction frames cannot support an alignment or
        # visibility measurement. This is the same terminal-input constraint as
        # construction_failed, not a label derived from role confidence/features.
        if float(by_key[key].get("robot_frame_missing", 0)):
            reasons.append("construction_robot_frame_missing")
        if float(by_key[key].get("geometry_policy_cameras_missing", 0)):
            reasons.append("construction_policy_cameras_missing")
        if audit_labels._bool(record.get("robot_alignment_correct")) is not True:
            reasons.append("robot_alignment_failed_or_missing")
        labels.append({**dict(zip(JOIN_KEY, key)), "invalid_label": int(bool(reasons)),
                       "invalid_reasons": ";".join(reasons)})
        baseline = {}
        for name in seal["evaluation_baseline_columns"]:
            value = record.get(name)
            if value not in {None, ""} and not math.isfinite(float(value)):
                raise ValueError(f"nonfinite evaluation baseline: {name}")
            baseline[name] = value
        baselines[key] = baseline
    with _atomic_output(out) as (staging, output):
        _write_csv_new(staging / "labels.csv", labels, LABEL_FIELDS)
        joined = join_labels(features_dir / "features_unlabeled.csv", staging / "labels.csv",
                             staging / "features_with_validity.csv")
        for row in joined:
            row.update(baselines[tuple(row[k] for k in JOIN_KEY)])
        _write_csv_new(staging / "task_local_features_and_labels.csv", joined, list(joined[0]))
        groups = signal_groups([k for k in joined[0]
                                if k not in {*IDENTITY_FIELDS, "invalid_label", "invalid_reasons"}])
        scenes = sorted({r["scene_id"] for r in joined})
        fold_health = [{"heldout_scene": scene,
                        "training_has_both_labels": {int(r["invalid_label"]) for r in joined
                                                     if r["scene_id"] != scene} == {0, 1},
                        "heldout_has_both_labels": {int(r["invalid_label"]) for r in joined
                                                    if r["scene_id"] == scene} == {0, 1}}
                       for scene in scenes]
        admissible = len(scenes) >= 2 and all(f["training_has_both_labels"] and
                                            f["heldout_has_both_labels"] for f in fold_health)
        _write_json_new(staging / "label_join_manifest.json", {
            "freeze_id": seal["freeze_id"], "source_kind": seal["source_kind"], "paper_ready": False,
            "feature_seal_sha256": _sha256(seal_path), "protocol_sha256": _sha256(protocol_path),
            "measurement_sha256": measurements_sha256, "query_rows": len(joined),
            "signal_groups": groups, "fold_health": fold_health,
            "loso_admissible": admissible,
            "members": {p.name: _sha256(p) for p in staging.iterdir() if p.is_file()}})
    if not admissible:
        raise ValueError(f"LOSO blocked: every training/test fold requires both labels; "
                         f"all rows and fold diagnostics preserved at {output}")
    return {"stage": "join", "out": str(output), "query_rows": len(joined), "paper_ready": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stage", choices=("features", "join"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--label-protocol", type=Path)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--measurements-sha256")
    args = parser.parse_args(argv)
    if args.smoke == bool(args.stage):
        parser.error("choose --smoke or --stage features|join")
    try:
        if args.smoke:
            result = generate_smoke(args.out)
        elif args.stage == "features":
            if args.config is None:
                parser.error("--stage features requires --config")
            result = generate_features(args.config, args.out)
        else:
            if not all((args.features_dir, args.label_protocol, args.measurements, args.measurements_sha256)):
                parser.error("--stage join requires --features-dir, --label-protocol, --measurements, --measurements-sha256")
            result = join_evaluation(args.features_dir, args.label_protocol, args.measurements,
                                     args.measurements_sha256, args.out)
    except (FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
