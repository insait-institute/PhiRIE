#!/usr/bin/env python3
"""Sealed CPU diagnostics for the E4 common-background collision repair.

This is an orchestration script, not a paper evaluator.  It deliberately keeps
the repaired full-room export and the legacy private-shim diagnostic in
different materialized factory roots because ``export_mjcf`` writes in place to
``SIMANY_OUT/sim_export``.  The final command publishes a gate and returns
non-zero when the repair is not strong enough to release the already-enqueued
five-scene candidate screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EVIDENCE_ROOT = Path(os.environ.get("SIMANY_EXPECTED_EVIDENCE_ROOT", "/opt/phirie/evidence/SimAny"))
SCANNETPP_ROOT = Path("/data/ScanNetpp")
SPLATS_ROOT = Path("/data/ScanNetppv2_gsplat/splats")
PILOT_SCENES = ("3db0a1c8f3", "d755b3d9d8")
PILOT_SCENE_SELECTION = {
    "3db0a1c8f3": "largest accepted-object roster and nonzero pre-repair stable coverage",
    "d755b3d9d8": "second-largest accepted-object roster and nonzero paired-policy stable coverage",
}
POLICIES = ("A0", "A4")
MODES = ("room", "shim")
STABLE_DRIFT_LIMIT_M = 0.03
INITIAL_STATIC_PENETRATION_LIMIT_M = 0.005
CARVE_MARGIN_M = 0.02
LOWER_SUPPORT_MARGIN_M = 0.0
EXPORT_STEPS = 1000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SLOT_RE = re.compile(r"^obj_[0-9]+$")


class RepairPilotError(RuntimeError):
    """A provenance, artifact, or scientific repair contract is invalid."""


def _candidate():
    from robo.eval import e4_candidate_screen as candidate

    return candidate


def _validated_id(value: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise RepairPilotError("repair ID is unsafe")
    return value


def _validated_scene(value: str) -> str:
    if value not in PILOT_SCENES:
        raise RepairPilotError(f"scene is outside the repair pilot: {value}")
    return value


def _validated_policy(value: str) -> str:
    if value not in POLICIES:
        raise RepairPilotError(f"invalid policy: {value}")
    return value


def _validated_mode(value: str) -> str:
    if value not in MODES:
        raise RepairPilotError(f"invalid collision mode: {value}")
    return value


def _code_snapshot(expected_commit: str) -> dict[str, Any]:
    try:
        result = _candidate()._code_snapshot(expected_commit)
    except Exception as exc:  # candidate converts all snapshot failures to its own type
        raise RepairPilotError(str(exc)) from exc
    if result != {
        "code_root": str(CODE_ROOT),
        "commit": expected_commit,
        "dirty": False,
    }:
        raise RepairPilotError("repair-pilot code snapshot differs")
    return result


def _evidence_root() -> Path:
    try:
        root = _candidate().evidence_root()
    except Exception as exc:
        raise RepairPilotError(str(exc)) from exc
    if root != EXPECTED_EVIDENCE_ROOT or root == CODE_ROOT:
        raise RepairPilotError("repair-pilot evidence/code root binding differs")
    return root


def _experiment_root(root: Path, repair_id: str) -> Path:
    return root / "outputs" / "icra2027" / _validated_id(repair_id)


def _mode_screen_id(repair_id: str, mode: str) -> str:
    return f"{_validated_id(repair_id)}-{_validated_mode(mode)}"


def _factory(root: Path, repair_id: str, mode: str, policy: str, scene: str) -> Path:
    return (
        root
        / "outputs"
        / "icra2027"
        / _mode_screen_id(repair_id, mode)
        / "construction_variants"
        / _validated_policy(policy)
        / f"{_validated_scene(scene)}_factory"
    )


def _read_json(path: Path) -> Any:
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise RepairPilotError(f"not a regular canonical JSON file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepairPilotError(f"invalid JSON {path}: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _same_replayed_structure(left: Any, right: Any) -> bool:
    """Exact structure/provenance, with 1e-12 only for finite float leaves."""
    return bool(_candidate()._same_replay_structure(left, right))


def _identity(path: Path, *, root: Path) -> dict[str, Any]:
    try:
        return _candidate()._identity(path, root=root)
    except Exception as exc:
        raise RepairPilotError(str(exc)) from exc


def _publish(
    destination: Path,
    *,
    kind: str,
    gate: Mapping[str, Any],
    code: Mapping[str, Any],
    repair_id: str,
) -> dict[str, Any]:
    try:
        return _candidate()._publish_bundle(
            destination,
            manifest_kind=kind,
            payloads={"gate.json": _candidate()._json_bytes(dict(gate))},
            manifest_fields={
                "code": dict(code),
                "repair_id": repair_id,
                "study_scope": "e4_collision_repair_cpu_gate",
            },
        )
    except Exception as exc:
        raise RepairPilotError(str(exc)) from exc


def _validated_bundle_gate(
    directory: Path,
    *,
    root: Path,
    expected_kind: str,
    repair_id: str,
    scene: str | None = None,
    policy: str | None = None,
    mode: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        bundle = _candidate()._validate_bundle(
            directory, root=root, expected_kind=expected_kind
        )
    except Exception as exc:
        raise RepairPilotError(str(exc)) from exc
    gate = _read_json(directory / "gate.json")
    if not isinstance(gate, Mapping) or gate.get("repair_id") != repair_id:
        raise RepairPilotError(f"{expected_kind} gate repair binding differs")
    for key, expected in (("scene_id", scene), ("policy_id", policy), ("collision_mode", mode)):
        if expected is not None and gate.get(key) != expected:
            raise RepairPilotError(f"{expected_kind} gate {key} differs")
    return dict(gate), bundle


def _validate_materialization(
    factory: Path, *, scene: str, policy: str, root: Path, expected_commit: str
) -> dict[str, Any]:
    from robo.eval.e3_factory_materializer import validate_materialized_factory

    try:
        report = validate_materialized_factory(
            factory,
            expected_scene_id=scene,
            expected_policy_id=policy,
            repository_root=root,
        )
    except Exception as exc:
        raise RepairPilotError(f"materialization validation failed: {exc}") from exc
    if report.get("validator_commit") != expected_commit:
        raise RepairPilotError("materialization validator commit differs")
    roster = report.get("roster")
    if not isinstance(roster, Mapping):
        raise RepairPilotError("materialization roster is absent")
    accepted = roster.get("accepted_slots")
    discovered = roster.get("object_slots")
    for label, slots in (("accepted", accepted), ("discovered", discovered)):
        if (
            not isinstance(slots, list)
            or len(slots) != len(set(slots))
            or any(not isinstance(slot, str) or SLOT_RE.fullmatch(slot) is None for slot in slots)
        ):
            raise RepairPilotError(f"materialization {label} roster is invalid")
    if not set(accepted).issubset(discovered):
        raise RepairPilotError("accepted roster is not a subset of discovered roster")
    return report


def _source_binding(report: Mapping[str, Any]) -> dict[str, Any]:
    """Destination-independent binding used to compare room/shim materializations."""
    return {
        key: report[key]
        for key in (
            "e3_claim_status",
            "e3_code_commit",
            "e3_producer_commit",
            "e3_freeze_id",
            "e3_root",
            "input_identities_sha256",
            "materializer_commit",
            "policy_id",
            "provenance",
            "roster",
            "scene_id",
            "source_scene_sha256",
            "study_scope",
            "validator_commit",
        )
    }


def export_variant(
    *, repair_id: str, scene: str, policy: str, mode: str, expected_commit: str
) -> dict[str, Any]:
    repair_id = _validated_id(repair_id)
    scene, policy, mode = _validated_scene(scene), _validated_policy(policy), _validated_mode(mode)
    code = _code_snapshot(expected_commit)
    root = _evidence_root()
    factory = _factory(root, repair_id, mode, policy, scene)
    materialization = _validate_materialization(
        factory, scene=scene, policy=policy, root=root, expected_commit=expected_commit
    )
    for generated in (factory / "sim", factory / "sim_export"):
        if generated.exists() or generated.is_symlink():
            raise RepairPilotError(f"refusing non-fresh in-place export: {generated}")

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
        "SIMANY_SCENE": scene,
        "OMP_NUM_THREADS": os.environ.get("SLURM_CPUS_PER_TASK", "1"),
    }
    for name in ("TMPDIR", "XDG_CACHE_HOME"):
        raw = os.environ.get(name)
        if raw:
            candidate_path = Path(raw)
            try:
                candidate_path.resolve(strict=True).relative_to(root)
            except (FileNotFoundError, ValueError) as exc:
                raise RepairPilotError(f"{name} escapes the evidence root") from exc
            if not candidate_path.is_dir() or candidate_path.is_symlink():
                raise RepairPilotError(f"{name} is not a regular directory")
            environment[name] = str(candidate_path)
    exporter_argv = [
        sys.executable,
        "-m",
        "robo.sim.export_mjcf",
        "--test",
        "--collision-mode",
        mode,
    ]
    if mode == "room":
        # The repaired exporter needs both policies even though this worker
        # writes only one factory.  The scheduler therefore waits for the two
        # paired materializations before starting either room export.
        for paired_policy in POLICIES:
            exporter_argv.extend(
                [
                    "--background-carve-factory",
                    str(_factory(root, repair_id, mode, paired_policy, scene)),
                ]
            )
    subprocess.run(
        exporter_argv,
        cwd=CODE_ROOT,
        env=environment,
        check=True,
    )
    export_dir = factory / "sim_export"
    artifacts = {
        name: _identity(export_dir / name, root=root)
        for name in (
            "isaac_manifest.json",
            "mujoco_settle.json",
            "room_collision_report.json",
            "scene.xml",
        )
    }
    collision = _read_json(export_dir / "room_collision_report.json")
    if not isinstance(collision, Mapping) or collision.get("mode") != mode:
        raise RepairPilotError("export collision mode differs")
    from robo.eval import e4_region_pilot as sealed_cpu

    generated_trees = {
        name: sealed_cpu._tree_inventory(
            factory / name,
            factory=factory,
            root=root,
            label=f"repair export {scene}/{policy}/{mode}/{name}",
        )
        for name in ("sim", "sim_export")
        if (factory / name).is_dir()
    }
    if "sim_export" not in generated_trees or (mode == "room" and "sim" not in generated_trees):
        raise RepairPilotError("export generated-tree closure is incomplete")
    gate = {
        "artifacts": artifacts,
        "code": code,
        "collision_mode": mode,
        "duplicate_materialization_reason": (
            "export_mjcf_writes_in_place_to_SIMANY_OUT_sim_export; independent room/shim "
            "factory roots prevent overwrite and preserve both diagnostics"
        ),
        "factory": factory.relative_to(root).as_posix(),
        "generated_trees": generated_trees,
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_collision_repair_export_receipt_gate",
        "paper_ready": False,
        "policy_id": policy,
        "repair_id": repair_id,
        "scene_id": scene,
        "source_binding": _source_binding(materialization),
        "status": "exported",
    }
    bundle = _publish(
        _experiment_root(root, repair_id) / "export_receipts" / mode / policy / scene,
        kind="e4_collision_repair_export_receipt_artifacts",
        gate=gate,
        code=code,
        repair_id=repair_id,
    )
    return {**gate, "bundle_manifest_sha256": bundle["manifest_sha256"]}


def _initial_contact_diagnostics(xml_path: Path, accepted_slots: Sequence[str]) -> dict[str, Any]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    accepted = set(accepted_slots)
    max_any = 0.0
    max_static = 0.0
    static_by_geom_pair: dict[tuple[str, str, str, str], float] = {}
    for index in range(data.ncon):
        contact = data.contact[index]
        penetration = max(0.0, -float(contact.dist))
        max_any = max(max_any, penetration)
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world"
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world"
        geom_name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or f"geom_{geom1}"
        geom_name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or f"geom_{geom2}"
        if (body1 == 0 and name2 in accepted) or (body2 == 0 and name1 in accepted):
            max_static = max(max_static, penetration)
            if penetration > 0.0:
                key = (name1, geom_name1, name2, geom_name2)
                static_by_geom_pair[key] = max(
                    penetration, static_by_geom_pair.get(key, 0.0)
                )
    static_pairs = [
        {
            "body_a": key[0],
            "geom_a": key[1],
            "body_b": key[2],
            "geom_b": key[3],
            "penetration_m": static_by_geom_pair[key],
        }
        for key in sorted(static_by_geom_pair)
    ]
    return {
        "contact_count": int(data.ncon),
        "max_any_penetration_m": max_any,
        "max_static_penetration_m": max_static,
        "static_penetrations": static_pairs,
    }


def _validate_background_contract(
    collision: Mapping[str, Any], *, discovered_slots: Sequence[str], accepted_slots: Sequence[str]
) -> tuple[dict[str, bool], dict[str, Any]]:
    carve = collision.get("background_carve")
    exclusion = collision.get("collision_exclusion")
    carve_keys = {
        "background_sha256",
        "carved_slots",
        "discovered_slots",
        "geometry_source",
        "hull_count",
        "lower_support_margin_m",
        "margin_m",
        "mode",
        "policy_ids",
        "roster_sha256",
        "schema_version",
        "source_mesh_sha256",
        "specification_sha256",
        "support_clip_source",
    }
    exclusion_keys = {
        "coacd_candidate_parts",
        "coacd_parts_rejected_intrusion",
        "emitted_coacd_intrusion_count",
        "hull_count",
        "primitive_intrusions",
        "schema_version",
        "unresolved_intrusion_count",
    }
    carve_mapping = isinstance(carve, Mapping)
    exclusion_mapping = isinstance(exclusion, Mapping)
    discovered = list(discovered_slots)
    accepted = list(accepted_slots)
    checks = {
        "background_carve_exact_schema": carve_mapping and set(carve) == carve_keys,
        "background_uses_paired_policy_union": carve_mapping
        and carve.get("mode") == "paired_policy_union"
        and carve.get("policy_ids") == ["A0", "A4"],
        "background_uses_transformed_convex_hulls": carve_mapping
        and carve.get("geometry_source") == "transformed_convex_collision_hulls",
        "background_support_clip_uses_discovered_scan_bottom": carve_mapping
        and carve.get("support_clip_source")
        == "discovered_scan_aabb_bottom_plus_5mm",
        "background_margin_is_20mm_side_top_and_zero_below": carve_mapping
        and math.isclose(float(carve.get("margin_m", math.nan)), CARVE_MARGIN_M, abs_tol=1e-12)
        and math.isclose(
            float(carve.get("lower_support_margin_m", math.nan)),
            LOWER_SUPPORT_MARGIN_M,
            abs_tol=1e-12,
        ),
        "background_discovered_roster_matches_materialization": carve_mapping
        and carve.get("discovered_slots") == discovered,
        "background_carves_every_discovered_slot": carve_mapping
        and carve.get("carved_slots") == sorted(discovered),
        "accepted_roster_is_subset_of_common_carve": set(accepted).issubset(discovered),
        "background_hashes_are_sealed": carve_mapping
        and all(
            isinstance(carve.get(key), str) and SHA256_RE.fullmatch(carve[key]) is not None
            for key in (
                "roster_sha256",
                "source_mesh_sha256",
                "specification_sha256",
                "background_sha256",
            )
        ),
        "background_hull_count_is_positive": carve_mapping
        and isinstance(carve.get("hull_count"), int)
        and not isinstance(carve.get("hull_count"), bool)
        and carve["hull_count"] > 0,
        "collision_exclusion_exact_schema": exclusion_mapping
        and set(exclusion) == exclusion_keys,
        "collision_exclusion_hull_count_matches_carve": carve_mapping
        and exclusion_mapping
        and exclusion.get("hull_count") == carve.get("hull_count"),
        "no_emitted_coacd_intrusion": exclusion_mapping
        and exclusion.get("emitted_coacd_intrusion_count") == 0,
        "no_primitive_intrusion": exclusion_mapping
        and exclusion.get("primitive_intrusions") == [],
        "no_unresolved_intrusion": exclusion_mapping
        and exclusion.get("unresolved_intrusion_count") == 0,
        "collision_exclusion_counts_are_nonnegative": exclusion_mapping
        and all(
            isinstance(exclusion.get(key), int)
            and not isinstance(exclusion.get(key), bool)
            and exclusion[key] >= 0
            for key in (
                "hull_count",
                "coacd_candidate_parts",
                "coacd_parts_rejected_intrusion",
                "emitted_coacd_intrusion_count",
                "unresolved_intrusion_count",
            )
        ),
    }
    summary = {
        "background_sha256": carve.get("background_sha256") if carve_mapping else None,
        "carved_slots": carve.get("carved_slots") if carve_mapping else None,
        "discovered_slots": carve.get("discovered_slots") if carve_mapping else None,
        "hull_count": carve.get("hull_count") if carve_mapping else None,
        "roster_sha256": carve.get("roster_sha256") if carve_mapping else None,
        "specification_sha256": carve.get("specification_sha256") if carve_mapping else None,
        "source_mesh_sha256": carve.get("source_mesh_sha256") if carve_mapping else None,
        "coacd_candidate_parts": (
            exclusion.get("coacd_candidate_parts") if exclusion_mapping else None
        ),
        "coacd_parts_rejected_intrusion": (
            exclusion.get("coacd_parts_rejected_intrusion") if exclusion_mapping else None
        ),
    }
    return checks, summary


def _build_export_validation(
    *, repair_id: str, scene: str, policy: str, mode: str, expected_commit: str
) -> dict[str, Any]:
    code = _code_snapshot(expected_commit)
    root = _evidence_root()
    factory = _factory(root, repair_id, mode, policy, scene)
    receipt_gate, receipt_bundle = _validated_bundle_gate(
        _experiment_root(root, repair_id) / "export_receipts" / mode / policy / scene,
        root=root,
        expected_kind="e4_collision_repair_export_receipt_artifacts",
        repair_id=repair_id,
        scene=scene,
        policy=policy,
        mode=mode,
    )
    materialization = _validate_materialization(
        factory, scene=scene, policy=policy, root=root, expected_commit=expected_commit
    )
    source_binding = _source_binding(materialization)
    if receipt_gate.get("source_binding") != source_binding:
        raise RepairPilotError("export receipt source binding differs")
    roster = materialization["roster"]
    accepted = list(roster["accepted_slots"])
    discovered = list(roster["object_slots"])
    export_dir = factory / "sim_export"
    xml_path = export_dir / "scene.xml"
    collision = _read_json(export_dir / "room_collision_report.json")
    settle = _read_json(export_dir / "mujoco_settle.json")
    isaac = _read_json(export_dir / "isaac_manifest.json")
    if not all(isinstance(item, Mapping) for item in (collision, settle, isaac)):
        raise RepairPilotError("export JSON roots must be objects")
    artifacts = {
        name: _identity(export_dir / name, root=root)
        for name in (
            "isaac_manifest.json",
            "mujoco_settle.json",
            "room_collision_report.json",
            "scene.xml",
        )
    }
    if receipt_gate.get("artifacts") != artifacts:
        raise RepairPilotError("export artifacts changed after receipt")

    from robo.eval import e4_region_pilot as sealed_cpu

    generated_trees = {
        name: sealed_cpu._tree_inventory(
            factory / name,
            factory=factory,
            root=root,
            label=f"repair validation {scene}/{policy}/{mode}/{name}",
        )
        for name in ("sim", "sim_export")
        if (factory / name).is_dir()
    }
    if receipt_gate.get("generated_trees") != generated_trees:
        raise RepairPilotError("generated export tree changed after sealed receipt")

    try:
        xml = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise RepairPilotError(f"scene XML is invalid: {exc}") from exc
    compiler = xml.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    bodies = [body.get("name") for body in xml.findall("./worldbody/body")]
    floor = xml.find("./worldbody/geom[@name='floor']")
    floor_pos = floor.get("pos", "0 0 0").split() if floor is not None else []
    floor_z_xml = float(floor_pos[2]) if len(floor_pos) == 3 else math.nan
    drift_raw = settle.get("drift_m")
    drift = (
        {str(key): float(value) for key, value in drift_raw.items()}
        if isinstance(drift_raw, Mapping)
        else {}
    )
    stable_slots = sorted(slot for slot, value in drift.items() if value < STABLE_DRIFT_LIMIT_M)
    benchmark = collision.get("benchmark")
    benchmark_mapping = isinstance(benchmark, Mapping)
    isaac_rows = isaac.get("objects")
    isaac_names = (
        [row.get("name") if isinstance(row, Mapping) else None for row in isaac_rows]
        if isinstance(isaac_rows, list)
        else []
    )
    contacts = _initial_contact_diagnostics(xml_path, accepted)
    checks: dict[str, bool] = {
        "receipt_is_sealed": bool(receipt_bundle.get("manifest_sha256")),
        "collision_mode_matches": collision.get("mode") == mode,
        "xml_meshdir_matches_factory": meshdir == str(factory),
        "xml_body_roster_matches_accepted": len(bodies) == len(set(bodies))
        and set(bodies) == set(accepted),
        "isaac_roster_matches_accepted": len(isaac_names) == len(set(isaac_names))
        and set(isaac_names) == set(accepted),
        "settle_roster_matches_accepted": set(drift) == set(accepted)
        and all(math.isfinite(value) and value >= 0.0 for value in drift.values()),
        "settle_summary_replays": settle.get("n") == len(drift)
        and settle.get("stable_3cm") == len(stable_slots),
        "benchmark_is_finite_1000_steps": benchmark_mapping
        and benchmark.get("finite") is True
        and benchmark.get("steps") == EXPORT_STEPS
        and isinstance(benchmark.get("state_hash"), str)
        and SHA256_RE.fullmatch(benchmark["state_hash"]) is not None,
    }
    background_summary = None
    if mode == "room":
        # Reuse the production candidate validator as an independent closure
        # check. It recompiles XML, inventories every generated member, and
        # verifies every XML-referenced collision mesh.
        production_validation = _candidate().validate_full_room_export(
            factory,
            scene_id=scene,
            policy=policy,
            root=root,
            expected_object_slots=accepted,
            expected_discovered_slots=discovered,
        )
        if production_validation.get("generated_trees") != generated_trees:
            raise RepairPilotError("production validator tree closure differs")
        floor_patch = collision.get("floor_patch_selection")
        floor_patch_mapping = isinstance(floor_patch, Mapping)
        background_checks, background_summary = _validate_background_contract(
            collision, discovered_slots=discovered, accepted_slots=accepted
        )
        checks.update(background_checks)
        checks.update(
            {
                "xml_floor_is_scannetpp_z0": math.isclose(
                    floor_z_xml, 0.0, abs_tol=1e-12
                ),
                "report_floor_is_scannetpp_z0": collision.get("floor_z_m") == 0.0
                and collision.get("floor_source") == "scannetpp_world_frame_z0",
                "floor_patch_selection_schema_is_valid": floor_patch_mapping
                and set(floor_patch)
                == {"detected_z_m", "expected_z_m", "status", "tolerance_m"}
                and floor_patch.get("expected_z_m") == 0.0
                and floor_patch.get("tolerance_m") == 0.05
                and floor_patch.get("status") in {"matched", "not_detected"},
                "floor_patch_is_near_z0_or_absent": floor_patch_mapping
                and (
                    (
                        floor_patch.get("status") == "matched"
                        and isinstance(floor_patch.get("detected_z_m"), (int, float))
                        and not isinstance(floor_patch.get("detected_z_m"), bool)
                        and math.isfinite(float(floor_patch["detected_z_m"]))
                        and abs(float(floor_patch["detected_z_m"])) <= 0.05
                    )
                    or (
                        floor_patch.get("status") == "not_detected"
                        and floor_patch.get("detected_z_m") is None
                    )
                ),
            }
        )
        checks["initial_static_penetration_below_5mm"] = (
            contacts["max_static_penetration_m"] <= INITIAL_STATIC_PENETRATION_LIMIT_M
        )
    else:
        floor_report = collision.get("floor_z_m")
        checks.update(
            {
                "shim_count_matches_accepted": collision.get("n_shims") == len(accepted),
                "shim_has_no_common_background_claim": "background_carve" not in collision
                and "collision_exclusion" not in collision,
                "shim_preserves_legacy_floor_contract": isinstance(
                    floor_report, (int, float)
                )
                and not isinstance(floor_report, bool)
                and math.isfinite(float(floor_report))
                and collision.get("floor_source") == "legacy_below_lowest_object"
                and math.isclose(floor_z_xml, float(floor_report), abs_tol=5.1e-5),
            }
        )
    return {
        "accepted_slots": accepted,
        "artifacts": artifacts,
        "background": background_summary,
        "checks": checks,
        "code": code,
        "collision_mode": mode,
        "discovered_slots": discovered,
        "drift_m": drift,
        "generated_trees": generated_trees,
        "gpu_launch_allowed": False,
        "initial_contacts": contacts,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_collision_repair_export_validation_gate",
        "paper_ready": False,
        "policy_id": policy,
        "repair_id": repair_id,
        "scene_id": scene,
        "source_binding": source_binding,
        "stable_slot_count": len(stable_slots),
        "stable_slots": stable_slots,
        "status": "validated" if all(checks.values()) else "scientific_gate_failed",
        "validation_passed": all(checks.values()),
    }


def validate_export(
    *, repair_id: str, scene: str, policy: str, mode: str, expected_commit: str
) -> dict[str, Any]:
    repair_id = _validated_id(repair_id)
    scene, policy, mode = _validated_scene(scene), _validated_policy(policy), _validated_mode(mode)
    code = _code_snapshot(expected_commit)
    try:
        gate = _build_export_validation(
            repair_id=repair_id,
            scene=scene,
            policy=policy,
            mode=mode,
            expected_commit=expected_commit,
        )
    except Exception as exc:
        gate = {
            "checks": {},
            "code": code,
            "collision_mode": mode,
            "error": f"{type(exc).__name__}:{exc}",
            "gpu_launch_allowed": False,
            "large_rollout_launch_allowed": False,
            "manifest_kind": "e4_collision_repair_export_validation_gate",
            "paper_ready": False,
            "policy_id": policy,
            "repair_id": repair_id,
            "scene_id": scene,
            "status": "invalid",
            "validation_passed": False,
        }
    root = _evidence_root()
    bundle = _publish(
        _experiment_root(root, repair_id) / "export_validations" / mode / policy / scene,
        kind="e4_collision_repair_export_validation_artifacts",
        gate=gate,
        code=code,
        repair_id=repair_id,
    )
    return {**gate, "bundle_manifest_sha256": bundle["manifest_sha256"]}


def _revalidate_export_closure(
    gate: Mapping[str, Any],
    *,
    root: Path,
    repair_id: str,
    scene: str,
    policy: str,
    mode: str,
    expected_commit: str,
) -> None:
    """Recompute the entire validation gate from raw files and compare exactly."""
    del root  # the replay obtains and revalidates the exact evidence root itself
    replayed = _build_export_validation(
        repair_id=repair_id,
        scene=scene,
        policy=policy,
        mode=mode,
        expected_commit=expected_commit,
    )
    if not _same_replayed_structure(dict(gate), replayed):
        raise RepairPilotError(
            f"{scene}/{policy}/{mode} validation summary differs from raw replay"
        )


def build_mode_comparison_gate(
    *,
    repair_id: str,
    scene: str,
    policy: str,
    code: Mapping[str, Any],
    validations: Mapping[str, Mapping[str, Any]],
    validation_bundle_sha256: Mapping[str, str],
    errors: Mapping[str, str],
) -> dict[str, Any]:
    """Derive every comparison field only from replayed export validations."""
    room = validations.get("room", {})
    shim = validations.get("shim", {})
    checks = {
        "room_validation_present_and_passed": room.get("validation_passed") is True,
        "shim_validation_present_and_passed": shim.get("validation_passed") is True,
        "room_shim_source_binding_matches": bool(room)
        and room.get("source_binding") == shim.get("source_binding"),
        "room_shim_accepted_roster_matches": bool(room)
        and room.get("accepted_slots") == shim.get("accepted_slots"),
        "room_shim_discovered_roster_matches": bool(room)
        and room.get("discovered_slots") == shim.get("discovered_slots"),
    }
    comparison_passed = all(checks.values())
    room_stable = (
        room.get("stable_slots", []) if isinstance(room.get("stable_slots"), list) else []
    )
    shim_stable = (
        shim.get("stable_slots", []) if isinstance(shim.get("stable_slots"), list) else []
    )
    return {
        "background": room.get("background"),
        "checks": checks,
        "code": dict(code),
        "collision_mode": "room_vs_shim",
        "comparison_passed": comparison_passed,
        "errors": dict(errors),
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_collision_repair_mode_comparison_gate",
        "paper_ready": False,
        "policy_id": policy,
        "repair_id": repair_id,
        "room_initial_static_penetration_m": room.get("initial_contacts", {}).get(
            "max_static_penetration_m"
        ),
        "room_stable_slot_count": len(room_stable),
        "room_stable_slots": room_stable,
        "scene_id": scene,
        "shim_initial_static_penetration_m": shim.get("initial_contacts", {}).get(
            "max_static_penetration_m"
        ),
        "shim_stable_slot_count": len(shim_stable),
        "shim_stable_slots": shim_stable,
        "stable_count_delta_room_minus_shim": len(room_stable) - len(shim_stable),
        "status": "validated" if comparison_passed else "invalid_or_failed",
        "validation_bundle_manifest_sha256": dict(validation_bundle_sha256),
    }


def compare_modes(
    *, repair_id: str, scene: str, policy: str, expected_commit: str
) -> dict[str, Any]:
    repair_id = _validated_id(repair_id)
    scene, policy = _validated_scene(scene), _validated_policy(policy)
    code = _code_snapshot(expected_commit)
    root = _evidence_root()
    validations: dict[str, dict[str, Any]] = {}
    validation_bundle_sha256: dict[str, str] = {}
    errors: dict[str, str] = {}
    for mode in MODES:
        try:
            gate, bundle = _validated_bundle_gate(
                _experiment_root(root, repair_id)
                / "export_validations"
                / mode
                / policy
                / scene,
                root=root,
                expected_kind="e4_collision_repair_export_validation_artifacts",
                repair_id=repair_id,
                scene=scene,
                policy=policy,
                mode=mode,
            )
            _revalidate_export_closure(
                gate,
                root=root,
                repair_id=repair_id,
                scene=scene,
                policy=policy,
                mode=mode,
                expected_commit=expected_commit,
            )
            validations[mode] = gate
            validation_bundle_sha256[mode] = bundle["manifest_sha256"]
        except Exception as exc:
            errors[mode] = f"{type(exc).__name__}:{exc}"
    gate = build_mode_comparison_gate(
        repair_id=repair_id,
        scene=scene,
        policy=policy,
        code=code,
        validations=validations,
        validation_bundle_sha256=validation_bundle_sha256,
        errors=errors,
    )
    bundle = _publish(
        _experiment_root(root, repair_id) / "mode_comparisons" / policy / scene,
        kind="e4_collision_repair_mode_comparison_artifacts",
        gate=gate,
        code=code,
        repair_id=repair_id,
    )
    return {**gate, "bundle_manifest_sha256": bundle["manifest_sha256"]}


def build_aggregate_gate(
    *, repair_id: str, code: Mapping[str, Any], comparisons: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    scene_rows = []
    all_comparisons_pass = True
    all_backgrounds_consistent = True
    all_scenes_have_two_paired_stable = True
    for scene in PILOT_SCENES:
        per_policy = {policy: comparisons.get(f"{scene}/{policy}", {}) for policy in POLICIES}
        comparison_pass = all(
            row.get("comparison_passed") is True for row in per_policy.values()
        )
        all_comparisons_pass &= comparison_pass
        a0_background = per_policy["A0"].get("background")
        a4_background = per_policy["A4"].get("background")
        backgrounds_consistent = (
            isinstance(a0_background, Mapping)
            and isinstance(a4_background, Mapping)
            and a0_background == a4_background
        )
        all_backgrounds_consistent &= backgrounds_consistent
        stable_sets = []
        for policy in POLICIES:
            slots = per_policy[policy].get("room_stable_slots")
            stable_sets.append(set(slots) if isinstance(slots, list) else set())
        paired_stable = sorted(stable_sets[0] & stable_sets[1])
        has_two = len(paired_stable) >= 2
        all_scenes_have_two_paired_stable &= has_two
        scene_rows.append(
            {
                "backgrounds_consistent_across_A0_A4": backgrounds_consistent,
                "comparison_passed_for_both_policies": comparison_pass,
                "paired_room_stable_slot_count": len(paired_stable),
                "paired_room_stable_slots": paired_stable,
                "scene_id": scene,
            }
        )
    repair_gate_passed = (
        all_comparisons_pass
        and all_backgrounds_consistent
        and all_scenes_have_two_paired_stable
    )
    return {
        "checks": {
            "all_four_room_vs_shim_comparisons_passed": all_comparisons_pass,
            "common_background_identical_across_A0_A4_per_scene": (
                all_backgrounds_consistent
            ),
            "each_pilot_scene_has_at_least_two_paired_stable_room_slots": (
                all_scenes_have_two_paired_stable
            ),
        },
        "code": dict(code),
        "comparison_count": len(comparisons),
        "downstream_candidate_screen_allowed": repair_gate_passed,
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_collision_repair_aggregate_gate",
        "paper_ready": False,
        "repair_gate_passed": repair_gate_passed,
        "repair_id": repair_id,
        "pilot_scene_selection": dict(PILOT_SCENE_SELECTION),
        "scene_rows": scene_rows,
        "study_scope": "e4_collision_repair_cpu_gate",
    }


def aggregate(*, repair_id: str, expected_commit: str) -> dict[str, Any]:
    repair_id = _validated_id(repair_id)
    code = _code_snapshot(expected_commit)
    root = _evidence_root()
    comparisons: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for scene in PILOT_SCENES:
        for policy in POLICIES:
            key = f"{scene}/{policy}"
            try:
                gate, _ = _validated_bundle_gate(
                    _experiment_root(root, repair_id) / "mode_comparisons" / policy / scene,
                    root=root,
                    expected_kind="e4_collision_repair_mode_comparison_artifacts",
                    repair_id=repair_id,
                    scene=scene,
                    policy=policy,
                    mode="room_vs_shim",
                )
                # Do not trust the comparison's copied metrics. Revalidate
                # both underlying mode bundles and their full file trees again
                # immediately before deciding whether to release downstream.
                replay_bundle_sha256 = {}
                replay_validations = {}
                for mode in MODES:
                    export_gate, export_bundle = _validated_bundle_gate(
                        _experiment_root(root, repair_id)
                        / "export_validations"
                        / mode
                        / policy
                        / scene,
                        root=root,
                        expected_kind="e4_collision_repair_export_validation_artifacts",
                        repair_id=repair_id,
                        scene=scene,
                        policy=policy,
                        mode=mode,
                    )
                    _revalidate_export_closure(
                        export_gate,
                        root=root,
                        repair_id=repair_id,
                        scene=scene,
                        policy=policy,
                        mode=mode,
                        expected_commit=expected_commit,
                    )
                    replay_validations[mode] = export_gate
                    replay_bundle_sha256[mode] = export_bundle["manifest_sha256"]
                replayed_comparison = build_mode_comparison_gate(
                    repair_id=repair_id,
                    scene=scene,
                    policy=policy,
                    code=code,
                    validations=replay_validations,
                    validation_bundle_sha256=replay_bundle_sha256,
                    errors={},
                )
                if not _same_replayed_structure(gate, replayed_comparison):
                    raise RepairPilotError("comparison summary differs from raw replay")
                comparisons[key] = gate
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}:{exc}"
                comparisons[key] = {
                    "comparison_passed": False,
                    "policy_id": policy,
                    "scene_id": scene,
                    "status": "missing_or_invalid",
                }
    gate = build_aggregate_gate(repair_id=repair_id, code=code, comparisons=comparisons)
    gate["errors"] = errors
    bundle = _publish(
        _experiment_root(root, repair_id) / "aggregate",
        kind="e4_collision_repair_aggregate_artifacts",
        gate=gate,
        code=code,
        repair_id=repair_id,
    )
    return {**gate, "bundle_manifest_sha256": bundle["manifest_sha256"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repair-id", required=True)
    common.add_argument("--expected-code-commit", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", parents=[common])
    validate = subparsers.add_parser("validate-export", parents=[common])
    compare = subparsers.add_parser("compare", parents=[common])
    for subparser in (export, validate):
        subparser.add_argument("--scene-id", choices=PILOT_SCENES, required=True)
        subparser.add_argument("--policy-id", choices=POLICIES, required=True)
        subparser.add_argument("--collision-mode", choices=MODES, required=True)
    compare.add_argument("--scene-id", choices=PILOT_SCENES, required=True)
    compare.add_argument("--policy-id", choices=POLICIES, required=True)
    subparsers.add_parser("aggregate", parents=[common])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            report = export_variant(
                repair_id=args.repair_id,
                scene=args.scene_id,
                policy=args.policy_id,
                mode=args.collision_mode,
                expected_commit=args.expected_code_commit,
            )
            passed = True
        elif args.command == "validate-export":
            report = validate_export(
                repair_id=args.repair_id,
                scene=args.scene_id,
                policy=args.policy_id,
                mode=args.collision_mode,
                expected_commit=args.expected_code_commit,
            )
            passed = report["validation_passed"] is True
        elif args.command == "compare":
            report = compare_modes(
                repair_id=args.repair_id,
                scene=args.scene_id,
                policy=args.policy_id,
                expected_commit=args.expected_code_commit,
            )
            passed = report["comparison_passed"] is True
        else:
            report = aggregate(
                repair_id=args.repair_id,
                expected_commit=args.expected_code_commit,
            )
            passed = report["repair_gate_passed"] is True
    except (RepairPilotError, FileNotFoundError, OSError, subprocess.CalledProcessError,
            TypeError, ValueError) as exc:
        print(f"[e4-collision-repair] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
