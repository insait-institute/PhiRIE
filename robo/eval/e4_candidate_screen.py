"""Sealed CPU-only discovery fleet for E4 paired manipulation scenes.

This module is deliberately separate from the paper harness and from
``TaskScorer``.  It screens a fixed five-scene roster before any policy or GPU
work is allowed:

* materialized A0/A4 construction arms are exported as full-room MuJoCo;
* a shared, paired-aware robot base and region-only task suite are planned;
* every candidate is replayed through the exact 1.5 s / 900 ``mj_step`` reset;
* a final after-any aggregate ranks only scenes with at least two strict tasks.

All published directories are fresh-only and atomically renamed.  A screening
result is diagnostic evidence only: every output keeps GPU, rollout, headline,
and paper authorization false.
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
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robo.eval import e4_region_pilot as sealed_cpu


SCHEMA_VERSION = 1
CODE_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_EVIDENCE_ROOT = Path("/group/worldcept/PhiRIE/code/SimAny")
SCANNETPP_ROOT = Path("/data/ScanNetpp")
SPLATS_ROOT = Path("/data/ScanNetppv2_gsplat/splats")
EXPECTED_E3_ROOT = Path(
    "outputs/icra2027/"
    "icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
)
EXPECTED_MENAGERIE_ROOT = CODE_ROOT / "third_party" / "mujoco_menagerie"
EXPECTED_MENAGERIE_COMMIT = "71f066ad0be9cd271f7ed58c030243ef157af9f4"

SCENE_IDS = (
    "3db0a1c8f3",
    "27dd4da69e",
    "d755b3d9d8",
    "acd95847c5",
    "40aec5fffa",
)
EXCLUDED_SCENE_IDS = (
    "825d228aec",
    "9071e139d9",
    "cc5237fd77",
)
REFERENCE_ONLY_SCENE_IDS = ("b0a08200c9",)
POLICIES = ("A0", "A4")
PLANNING_SOURCE = "A4"
STUDY_SCOPE = "e4_cpu_candidate_scene_discovery"

EPISODES = 5
BASE_SEED = 0
JITTER_XY_M = 0.01
EXPORT_SETTLE_S = 2.0
EXPORT_SETTLE_STEPS = 1000
RESET_SETTLE_S = 1.5
EXPECTED_RESET_STEPS = 900
QUALIFIER_MARGIN_M = 0.03
STABILITY_DRIFT_LIMIT_M = 0.03
AT_REST_LINEAR_SPEED_M_S = 0.05

REACH_MIN_M = 0.25
REACH_MAX_M = 0.80
FRONT_CONE_DEG = 60.0
MIN_RADIAL_MARGIN_M = 0.02
MIN_CONE_MARGIN_DEG = 5.0
MIN_TASK_DISPLACEMENT_M = 0.20
MIN_OBSTACLE_CLEARANCE_M = 0.02
ROBOT_BASE_RADIUS_M = 0.18
MIN_BASE_BODY_CLEARANCE_M = 0.02
TARGET_GOAL_PADDING_M = 0.01
TABLE_EDGE_CLEARANCE_M = 0.02
BASE_GRID_STEP_M = 0.05
FLOAT_REPLAY_ABS_TOLERANCE = 1e-12
REGION_GRID_STEP_M = 0.05
MIN_REGION_HALF_EXTENT_M = 0.10

SCREEN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OBJECT_SLOT_RE = re.compile(r"^obj_[0-9]+$")
QUALIFIERS = ("leftmost", "rightmost", "nearest", "farthest")
CELL_CHECK_KEYS = (
    "exactly_900_mujoco_steps",
    "headless_observation_empty",
    "jitter_draw_replayed",
    "no_renderer_constructed",
    "qualifier_strictly_passed",
    "settle_contract_passed",
    "stability_passed",
    "workspace_passed",
)
SETTLE_PROTOCOL_KEYS = (
    "engine",
    "model_timestep_s",
    "requested_duration_s",
    "simulated_duration_s",
    "step_count",
    "step_function",
)
SETTLE_CHECK_KEYS = (
    "settle_duration_matches_request",
    "settle_has_valid_timing",
    "settle_is_exactly_900_steps",
    "settle_timestep_matches_rig",
    "settle_uses_mujoco_step",
)
STABILITY_CHECK_KEYS = (
    "all_body_linear_speed_below_50mm_s",
    "all_body_settle_drift_below_30mm",
    "target_settle_drift_below_30mm",
)
WORKSPACE_CHECK_KEYS = (
    "base_clear_of_all_accepted_bodies",
    "destination_center_has_reach_margin",
    "eroded_goal_core_has_5_safe_samples",
    "eroded_goal_core_nonempty",
    "goal_clear_of_all_non_target_bodies",
    "target_center_displacement_at_least_200mm",
    "target_footprint_clear_of_goal",
    "target_has_reach_margin",
)


class CandidateScreenError(RuntimeError):
    """An input identity, runtime contract, or sealed output is invalid."""


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
    source = sealed_cpu._regular_file(path, root=root, label="candidate artifact")
    return {
        "path": source.relative_to(root).as_posix(),
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
    }


def _identity_without_path(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CandidateScreenError(f"not a regular file: {path}")
    return {"sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _validated_screen_id(value: str) -> str:
    if not isinstance(value, str) or SCREEN_ID_RE.fullmatch(value) is None:
        raise CandidateScreenError("screen ID is unsafe")
    return value


def _validated_scene_id(value: str) -> str:
    if value not in SCENE_IDS:
        raise CandidateScreenError(f"scene is outside the frozen candidate roster: {value}")
    return value


def _experiment_root(root: Path, screen_id: str) -> Path:
    return root / "outputs" / "icra2027" / _validated_screen_id(screen_id)


def _factory_dir(root: Path, screen_id: str, policy: str, scene_id: str) -> Path:
    _validated_scene_id(scene_id)
    if policy not in POLICIES:
        raise CandidateScreenError(f"invalid construction policy: {policy}")
    return (
        _experiment_root(root, screen_id)
        / "construction_variants"
        / policy
        / f"{scene_id}_factory"
    )


def _code_snapshot(expected_commit: str) -> dict[str, Any]:
    if not isinstance(expected_commit, str) or COMMIT_RE.fullmatch(expected_commit) is None:
        raise CandidateScreenError("expected code commit must be a full lowercase SHA")
    try:
        snapshot = sealed_cpu._git_snapshot(expected_commit)
    except (OSError, ValueError, sealed_cpu.PilotGateError) as exc:
        raise CandidateScreenError(str(exc)) from exc
    if snapshot != {
        "code_root": str(CODE_ROOT),
        "commit": expected_commit,
        "dirty": False,
    }:
        raise CandidateScreenError("candidate code snapshot differs")
    return snapshot


def evidence_root() -> Path:
    try:
        root = sealed_cpu.evidence_root()
    except (OSError, ValueError, sealed_cpu.PilotGateError) as exc:
        raise CandidateScreenError(str(exc)) from exc
    if root != EXPECTED_EVIDENCE_ROOT or root == CODE_ROOT:
        raise CandidateScreenError("candidate evidence/code root binding differs")
    return root


def materialize(
    *, screen_id: str, scene_id: str, policy_id: str,
    e3_root: str | Path, expected_commit: str,
) -> dict[str, Any]:
    """Materialize one exact E3 arm into its fixed fresh candidate path."""
    _validated_screen_id(screen_id)
    _validated_scene_id(scene_id)
    if policy_id not in POLICIES:
        raise CandidateScreenError(f"invalid construction policy: {policy_id}")
    source = Path(e3_root)
    if source.is_absolute() or source != EXPECTED_E3_ROOT:
        raise CandidateScreenError("E3 root differs from the frozen candidate source")
    _code_snapshot(expected_commit)
    root = evidence_root()
    from robo.eval.e3_factory_materializer import materialize_factory_variant

    manifest = materialize_factory_variant(
        e3_root=source,
        scene_id=scene_id,
        policy_id=policy_id,
        out=_factory_dir(root, screen_id, policy_id, scene_id),
    )
    if manifest.get("scene_id") != scene_id or manifest.get("policy_id") != policy_id:
        raise CandidateScreenError("materializer returned a different scene/policy binding")
    return manifest


def external_geometry_identities(scene_id: str) -> dict[str, dict[str, Any]]:
    _validated_scene_id(scene_id)
    scene = SCANNETPP_ROOT / "data" / scene_id
    try:
        return {
            name: sealed_cpu._external_identity(scene / relative, label=f"{scene_id} {name}")
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
    except (OSError, ValueError, sealed_cpu.PilotGateError) as exc:
        raise CandidateScreenError(str(exc)) from exc


def _materialization_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "e3_claim_status", "e3_code_commit", "e3_freeze_id", "e3_root",
        "manifest_sha256", "materializer_commit", "policy_id", "roster",
        "scene_id", "study_scope", "validator_commit",
    )
    try:
        return {key: report[key] for key in keys}
    except KeyError as exc:
        raise CandidateScreenError(f"materialization report lacks {exc.args[0]}") from exc


def _automatic_static_spec(scene_id):
    from robo.sim import room_collision as rc
    return rc.validate_room_diagnostic_spec({
        'schema_version': 1, 'variant_id': 'automatic-fixed-default', 'scene_id': scene_id,
        'hull_bottom': 'raw', 'intrusive_primitive': 'fail', 'support_z': 'extent',
        'plane_residual_tol_m': rc.PLANE_RESIDUAL_TOL, 'min_support_area_m2': rc.MIN_SUPPORT_AREA,
        'carve_side_top_margin_m': rc.CARVE_MARGIN_M, 'lower_carve_margin_m': 0.0,
        'support_clip_offset_m': 0.005}, expected_scene_id=scene_id)


def _automatic_static_layout(factory, *, root):
    parent = Path(factory).parent
    package = sealed_cpu._inside(parent/'shared_room_static', root=root, label='shared static package')
    spec = sealed_cpu._inside(parent/'shared_room_static_spec.json', root=root, label='shared static spec')
    return package, spec


def _load_automatic_static_package(factory, factories, *, scene_id, root):
    from robo.sim import export_mjcf as exporter
    from run.icra2027 import e4_automatic_export_reuse as reuse_api
    reuse = reuse_api.resolve(factory, factories, scene_id=scene_id, root=root)
    package_path, spec_path = _automatic_static_layout(factory, root=root)
    if reuse:
        package_path = Path(reuse['trees']['static']['path'])
        spec_path = Path(reuse['identities']['static_spec']['path'])
        factories = {p:Path(reuse['factories'][p]) for p in POLICIES}
    if spec_path.is_symlink() or spec_path.read_bytes() != _json_bytes(_automatic_static_spec(scene_id)):
        raise CandidateScreenError('shared static specification differs from frozen default recipe')
    package = exporter.load_common_room_static_package(package_path, _automatic_static_spec(scene_id), _sha256(spec_path), expected_scene_id=scene_id)
    if package['source_manifest_sha256'] != {p:_sha256(Path(factories[p])/'materialization_manifest.json') for p in POLICIES}:
        raise CandidateScreenError('shared static package belongs to different construction sources')
    return package


def _validate_static_package_xml(parsed, package):
    expected_meshes = {ET.fromstring(line).get('name'): dict(ET.fromstring(line).attrib)
                       for line in package['asset_xml_lines']}
    actual_meshes = {node.get('name'): dict(node.attrib) for node in parsed.findall('./asset/mesh')}
    if any(actual_meshes.get(name) != attrs for name,attrs in expected_meshes.items()):
        raise CandidateScreenError('scene static mesh declaration differs from shared package')
    expected_geoms = sorted([dict(ET.fromstring(line).attrib) for line in package['geom_xml_lines']], key=lambda row:row['name'])
    actual_geoms = sorted([dict(node.attrib) for node in parsed.findall('./worldbody/geom') if node.get('name') != 'floor'], key=lambda row:row['name'])
    if actual_geoms != expected_geoms:
        raise CandidateScreenError('scene static collision geoms differ from shared package')
    floor = parsed.find('./worldbody/geom[@name="floor"]')
    if floor is None or dict(floor.attrib) != {
            'name':'floor','type':'plane','pos':'0 0 0.0000','size':'20 20 1','friction':'0.8 0.005 0.0001'}:
        raise CandidateScreenError('scene floor differs from shared ScanNet++ world contract')
    return expected_meshes


def _static_collision_identity(parsed, meshes):
    geoms = [dict(node.attrib) for node in parsed.findall('./worldbody/geom')]
    names = [row.get('name') for row in geoms]
    if any(not isinstance(name,str) for name in names) or len(names) != len(set(names)):
        raise CandidateScreenError('static collision geom identities are invalid')
    mesh_names = sorted({row['mesh'] for row in geoms if 'mesh' in row})
    if any(name not in meshes for name in mesh_names):
        raise CandidateScreenError('static collision references an unknown mesh')
    bound = {'geoms': sorted(geoms, key=lambda row: row['name']),
             'meshes': {name: {key: meshes[name][key] for key in ('sha256','size_bytes')}
                        for name in mesh_names}}
    return {**bound, 'sha256': _canonical_hash(bound)}


def validate_full_room_export(
    factory_dir: str | Path,
    *, scene_id: str, policy: str, root: str | Path,
    expected_object_slots: Sequence[str], expected_discovered_slots: Sequence[str],
    automatic_factories: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Validate a generic full-room export without imposing a task count."""
    if policy not in POLICIES:
        raise CandidateScreenError("invalid export policy")
    automatic = (_automatic_export_context(automatic_factories, scene_id=scene_id, root=Path(root))
                 if automatic_factories is not None else None)
    if automatic is None:
        _validated_scene_id(scene_id)
    elif (Path(factory_dir).resolve() != Path(automatic_factories[policy]).resolve()
          or list(expected_object_slots) != automatic["rosters"][policy]["accepted_slots"]
          or list(expected_discovered_slots) != automatic["object_slots"]):
        raise CandidateScreenError("automatic export expected roster differs")
    root = Path(root).resolve(strict=True)
    try:
        factory = sealed_cpu._inside(Path(factory_dir), root=root, label="candidate factory")
        if not factory.is_dir() or factory.is_symlink():
            raise CandidateScreenError("candidate factory is not a regular directory")
        accepted = [str(slot) for slot in expected_object_slots]
        if (
            len(accepted) != len(set(accepted))
            or any(OBJECT_SLOT_RE.fullmatch(slot) is None for slot in accepted)
        ):
            raise CandidateScreenError("accepted-object roster is invalid")
        from run.icra2027 import e4_automatic_export_reuse as reuse_api
        reuse = reuse_api.resolve(factory, automatic_factories, scene_id=scene_id, root=root) if automatic else None
        export = factory / "sim_export"
        xml_path = sealed_cpu._regular_file(export / "scene.xml", root=root, label="scene XML")
        collision_path = sealed_cpu._regular_file(
            export / "room_collision_report.json", root=root,
            label="room collision report",
        )
        settle_path = sealed_cpu._regular_file(
            export / "mujoco_settle.json", root=root, label="MuJoCo settle report"
        )
        isaac_path = sealed_cpu._regular_file(
            export / "isaac_manifest.json", root=root, label="Isaac manifest"
        )
        collision_header = sealed_cpu._read_json(collision_path, root=root, label='collision')
        package_identity = collision_header.get('raw_diagnostics', {}).get('static_package_identity')
        package = None
        if automatic is not None and package_identity is not None:
            package = _load_automatic_static_package(factory, automatic_factories, scene_id=scene_id, root=root)
            if package_identity != package['static_package_identity']:
                raise CandidateScreenError('exported shared static identity differs')
        parsed = ET.parse(xml_path).getroot()
        package_meshes = _validate_static_package_xml(parsed, package) if package else {}
        if parsed.tag != "mujoco":
            raise CandidateScreenError("scene XML root is not mujoco")
        compiler = parsed.find("compiler")
        meshdir_raw = compiler.get("meshdir") if compiler is not None else None
        if not isinstance(meshdir_raw, str):
            raise CandidateScreenError("scene XML has no meshdir")
        meshdir = sealed_cpu._inside(Path(meshdir_raw), root=root, label="scene meshdir")
        expected_meshdir = Path(reuse['factories']['A0']) if reuse and policy == 'A0' else factory
        if meshdir != expected_meshdir:
            raise CandidateScreenError("scene XML meshdir differs from factory")
        if reuse and policy == 'A0':
            reuse_api.verify_a0(factory.parent, reuse)
        body_names = [body.get("name") for body in parsed.findall("./worldbody/body")]
        if (
            any(not isinstance(name, str) or OBJECT_SLOT_RE.fullmatch(name) is None
                for name in body_names)
            or len(body_names) != len(set(body_names))
            or set(body_names) != set(accepted)
        ):
            raise CandidateScreenError("scene XML body roster differs from materialization")
        meshes: dict[str, dict[str, Any]] = {}
        for mesh in parsed.findall("./asset/mesh"):
            name, raw_file = mesh.get("name"), mesh.get("file")
            if not isinstance(name, str) or not name or name in meshes \
                    or not isinstance(raw_file, str) or not raw_file:
                raise CandidateScreenError("scene XML mesh declaration is invalid")
            mesh_path = Path(raw_file)
            if not mesh_path.is_absolute():
                mesh_path = meshdir / mesh_path
            mesh_path = sealed_cpu._regular_file(mesh_path, root=root, label=f"mesh {name}")
            try:
                mesh_path.relative_to(expected_meshdir)
            except ValueError as exc:
                if package is None or name not in package_meshes or str(mesh_path) != package_meshes[name].get('file'):
                    raise CandidateScreenError("scene mesh escapes its factory or verified shared static package") from exc
            if reuse and policy == 'A0' and name not in package_meshes:
                relative = str(mesh_path.relative_to(expected_meshdir))
                if reuse['identities'].get('A0:'+relative) != reuse_api.api.identity(mesh_path):
                    raise CandidateScreenError('reused XML mesh lacks exact original identity')
            meshes[name] = _identity(mesh_path, root=root)
        if accepted and not meshes:
            raise CandidateScreenError("non-empty export has no mesh assets")

        collision = sealed_cpu._read_json(collision_path, root=root, label="collision")
        if not isinstance(collision, Mapping):
            raise CandidateScreenError("room collision report schema differs")
        benchmark = collision.get("benchmark")
        if collision.get("mode") != "room" or not isinstance(benchmark, Mapping) \
                or benchmark.get("finite") is not True:
            raise CandidateScreenError("full-room finite collision benchmark is absent")
        floor_z = collision.get("floor_z_m")
        if isinstance(floor_z, bool) or not isinstance(floor_z, (int, float)) \
                or float(floor_z) != 0.0 \
                or collision.get("floor_source") != "scannetpp_world_frame_z0":
            raise CandidateScreenError("ScanNet++ floor is not fixed at world z=0")
        floor_patch = collision.get("floor_patch_selection")
        if not isinstance(floor_patch, Mapping) \
                or floor_patch.get("expected_z_m") != 0.0 \
                or floor_patch.get("tolerance_m") != 0.05 \
                or floor_patch.get("status") not in {"matched", "not_detected"}:
            raise CandidateScreenError("ScanNet++ floor-patch selection contract differs")
        detected_floor = floor_patch.get("detected_z_m")
        if floor_patch.get("status") == "matched":
            if isinstance(detected_floor, bool) \
                    or not isinstance(detected_floor, (int, float)) \
                    or not math.isfinite(float(detected_floor)) \
                    or abs(float(detected_floor)) > 0.05:
                raise CandidateScreenError("high horizontal patch was misclassified as floor")
        elif detected_floor is not None:
            raise CandidateScreenError("absent floor patch has a detected height")
        discovered = [str(slot) for slot in expected_discovered_slots]
        if len(discovered) != len(set(discovered)) or any(
            OBJECT_SLOT_RE.fullmatch(slot) is None for slot in discovered
        ):
            raise CandidateScreenError("discovered-object roster is invalid")
        carve = collision.get("background_carve")
        if not isinstance(carve, Mapping) or carve.get("schema_version") != 1 \
                or carve.get("mode") != "paired_policy_union" \
                or carve.get("geometry_source") != "transformed_convex_collision_hulls" \
                or carve.get("margin_m") != 0.02 \
                or carve.get("lower_support_margin_m") != 0.0 \
                or carve.get("support_clip_source") \
                != ("automatic_instance_aabb_bottom_plus_5mm" if automatic else "discovered_scan_aabb_bottom_plus_5mm") \
                or carve.get("discovered_slots") != discovered \
                or carve.get("carved_slots") != (automatic["accepted_union"] if automatic else sorted(discovered)) \
                or carve.get("policy_ids") != list(POLICIES):
            raise CandidateScreenError("paired common background-carve contract differs")
        hull_count = carve.get("hull_count")
        carve_hashes = {
            key: carve.get(key)
            for key in ("roster_sha256", "source_mesh_sha256",
                        "specification_sha256", "background_sha256")
        }
        if isinstance(hull_count, bool) or not isinstance(hull_count, int) \
                or (hull_count < 0 if automatic else hull_count <= 0) \
                or (automatic is not None and bool(hull_count) != bool(automatic["accepted_union"])) or any(
                    not isinstance(value, str) or SHA256_RE.fullmatch(value) is None
                    for value in carve_hashes.values()
                ):
            raise CandidateScreenError("paired background-carve identities are invalid")
        if automatic is not None and carve_hashes["source_mesh_sha256"] != automatic["source_mesh_sha256"]:
            raise CandidateScreenError("automatic collision source is not the sealed derived mesh")
        exclusion = collision.get("collision_exclusion")
        if not isinstance(exclusion, Mapping) or exclusion.get("schema_version") != 1 \
                or exclusion.get("hull_count") != hull_count \
                or exclusion.get("emitted_coacd_intrusion_count") != 0 \
                or exclusion.get("primitive_intrusions") != [] \
                or exclusion.get("unresolved_intrusion_count") != 0:
            raise CandidateScreenError("room collision re-enters the protected object carve")
        for key in ("coacd_candidate_parts", "coacd_parts_rejected_intrusion"):
            value = exclusion.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CandidateScreenError("collision-exclusion count is invalid")
        steps = benchmark.get("steps")
        state_hash = benchmark.get("state_hash")
        if isinstance(steps, bool) or not isinstance(steps, int) \
                or steps != EXPORT_SETTLE_STEPS \
                or not isinstance(state_hash, str) or SHA256_RE.fullmatch(state_hash) is None:
            raise CandidateScreenError("full-room benchmark contract differs")

        settle = sealed_cpu._read_json(settle_path, root=root, label="settle")
        drift_raw = settle.get("drift_m") if isinstance(settle, Mapping) else None
        if not isinstance(drift_raw, Mapping):
            raise CandidateScreenError("settle report schema differs")
        drift = {str(slot): float(value) for slot, value in drift_raw.items()}
        if set(drift) != set(accepted) or not all(
            math.isfinite(value) and value >= 0.0 for value in drift.values()
        ):
            raise CandidateScreenError("settle body roster/value differs")
        if settle.get("n") != len(drift) or settle.get("stable_3cm") != sum(
            value < STABILITY_DRIFT_LIMIT_M for value in drift.values()
        ):
            raise CandidateScreenError("settle summary differs")

        isaac = sealed_cpu._read_json(isaac_path, root=root, label="Isaac manifest")
        isaac_rows = isaac.get("objects") if isinstance(isaac, Mapping) else None
        if not isinstance(isaac_rows, list):
            raise CandidateScreenError("Isaac manifest schema differs")
        isaac_names = [row.get("name") if isinstance(row, Mapping) else None for row in isaac_rows]
        if len(isaac_names) != len(set(isaac_names)) or set(isaac_names) != set(accepted):
            raise CandidateScreenError("Isaac roster differs from materialization")
        compile_report = sealed_cpu._compile_scene_xml(xml_path, target_slots=())
        trees = {
            name: sealed_cpu._tree_inventory(
                factory / name, factory=factory, root=root,
                label=f"{scene_id}/{policy}/{name}",
            )
            for name in ("sim", "sim_export")
        }
    except (ET.ParseError, OSError, ValueError, TypeError, sealed_cpu.PilotGateError) as exc:
        if isinstance(exc, CandidateScreenError):
            raise
        raise CandidateScreenError(str(exc)) from exc
    return {
        **({'reuse': {'declaration': _identity(factory.parent/'export_reuse/source.json', root=root),
            'static_producer_commit': reuse['code_commit'],
            'export_producer_commit': reuse['code_commit'] if policy == 'A0' else reuse_api.api.read(factory.parent/'export_reuse/source.json')['code']['commit'],
            'measured_settle_reused': policy == 'A0'}} if reuse else {}),
        "artifacts": {
            "isaac_manifest": _identity(isaac_path, root=root),
            "mujoco_settle": _identity(settle_path, root=root),
            "room_collision_report": _identity(collision_path, root=root),
            "scene_xml": _identity(xml_path, root=root),
        },
        "benchmark": {"finite": True, "state_hash": state_hash, "steps": steps},
        "collision_mode": "room",
        "background_carve": {
            **carve_hashes,
            "hull_count": hull_count,
        },
        "collision_exclusion": dict(exclusion),
        "floor_z_m": 0.0,
        "compile_check": compile_report,
        "drift_m": drift,
        "generated_trees": trees,
        "referenced_meshes": meshes,
        "static_collision": _static_collision_identity(parsed, meshes),
        **({'shared_static_package': {'path':package['package_dir'], **package['static_package_identity']}}
           if package else {}),
    }


EXPORT_SUBPROCESS_ENVIRONMENT_FIELDS = frozenset({
    "NPY_DISABLE_CPU_FEATURES", "LP_NUM_THREADS", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MUJOCO_GL",
    "PYOPENGL_PLATFORM", "LD_LIBRARY_PATH",
})


def _run_full_room_export(
    factory: Path, *, scene_id: str, root: Path,
    common_carve_factories: Sequence[Path],
    automatic: bool = False,
    subprocess_environment: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Run the exporter, optionally with explicitly bound numerical runtime fields.

    Existing callers retain their restricted environment. The opt-in map cannot
    override source, data, executable search or construction-mode identities.
    """
    if subprocess_environment is not None:
        if (not isinstance(subprocess_environment, Mapping)
                or not set(subprocess_environment) <= EXPORT_SUBPROCESS_ENVIRONMENT_FIELDS
                or any(not isinstance(v, str) or not v or "\x00" in v
                       for v in subprocess_environment.values())):
            raise CandidateScreenError("invalid explicit exporter subprocess environment")
    calls = []
    def run_export(command, environment):
        subprocess.run(command, cwd=CODE_ROOT, env=environment, check=True)
        calls.append({"argv": list(command), "cwd": str(CODE_ROOT),
                      "environment": dict(environment), "returncode": 0})
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
    if automatic:
        _automatic_export_context(dict(zip(POLICIES, common_carve_factories, strict=True)),
                                  scene_id=scene_id, root=root)
        environment.update(SIMANY_AUTO="1", SIMANY_NO_GT="1", SIMANY_MESH_SRC="derived")
    reuse = None
    if automatic:
        from run.icra2027 import e4_automatic_export_reuse as reuse_api
        reuse = reuse_api.resolve(factory, dict(zip(POLICIES, common_carve_factories, strict=True)), scene_id=scene_id, root=root)
        if reuse:
            common_carve_factories = [Path(reuse['factories'][p]) for p in POLICIES]
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus and cpus.isdigit() and int(cpus) > 0:
        environment["OMP_NUM_THREADS"] = cpus
    if subprocess_environment is not None:
        environment.update(subprocess_environment)
    for name in ("TMPDIR", "XDG_CACHE_HOME"):
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            path = sealed_cpu._inside(Path(raw), root=root, label=name)
        except sealed_cpu.PilotGateError as exc:
            raise CandidateScreenError(str(exc)) from exc
        if not path.is_dir() or path.is_symlink():
            raise CandidateScreenError(f"{name} is not a repository-local directory")
        environment[name] = str(path)
    carve_args = []
    for carve_factory in common_carve_factories:
        try:
            bound = sealed_cpu._inside(
                Path(carve_factory), root=root, label="common carve factory"
            )
        except sealed_cpu.PilotGateError as exc:
            raise CandidateScreenError(str(exc)) from exc
        if not bound.is_dir() or bound.is_symlink():
            raise CandidateScreenError("common carve factory is not a regular directory")
        carve_args.extend(["--background-carve-factory", str(bound)])
    static_args = []
    if automatic:
        package_path, spec_path = _automatic_static_layout(factory, root=root)
        if reuse:
            package_path = Path(reuse['trees']['static']['path'])
            spec_path = Path(reuse['identities']['static_spec']['path'])
        spec_bytes = _json_bytes(_automatic_static_spec(scene_id))
        if spec_path.exists():
            if spec_path.is_symlink() or spec_path.read_bytes() != spec_bytes:
                raise CandidateScreenError('existing shared static specification changed')
        else:
            with spec_path.open('xb') as stream: stream.write(spec_bytes)
        if not package_path.exists():
            run_export([sys.executable, '-m', 'robo.sim.export_mjcf', '--collision-mode', 'room',
                *carve_args, '--room-diagnostic-spec', str(spec_path), '--room-static-package-out', str(package_path)],
                environment)
        static_args = ['--room-diagnostic-spec', str(spec_path), '--room-static-package', str(package_path)]
    run_export(
        [
            sys.executable,
            "-m",
            "robo.sim.export_mjcf",
            "--test",
            "--collision-mode",
            "room",
            *carve_args,
            *static_args,
        ],
        environment,
    )

    return {"calls": calls, "requested_environment": dict(subprocess_environment)} if subprocess_environment is not None else None


def _world_xy_rectangle(position: np.ndarray, quaternion: np.ndarray, dimensions: np.ndarray) -> dict[str, Any]:
    import mujoco

    rotation_flat = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation_flat, quaternion)
    half = np.abs(rotation_flat.reshape(3, 3)[:2, :]) @ (dimensions / 2.0)
    rectangle = _rectangle(position[:2], half)
    return {
        **rectangle,
        "center_xy_m": position[:2].tolist(),
        "half_extents_xy_m": half.tolist(),
        "position_world_m": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
    }


def snapshot_export_footprints(
    factory: Path, *, expected_slots: Sequence[str], steps: int = EXPORT_SETTLE_STEPS
) -> dict[str, Any]:
    """Capture every accepted body's nominal and post-export settled footprint."""
    import mujoco

    if isinstance(steps, bool) or not isinstance(steps, int) or steps != EXPORT_SETTLE_STEPS:
        raise CandidateScreenError("export footprint replay must use exactly 1000 steps")
    model = mujoco.MjModel.from_xml_path(str(factory / "sim_export" / "scene.xml"))
    if not math.isclose(
        float(steps * model.opt.timestep), EXPORT_SETTLE_S, rel_tol=0.0, abs_tol=1e-12
    ):
        raise CandidateScreenError("export footprint replay is not exactly 2.0 seconds")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dimensions: dict[str, np.ndarray] = {}
    for slot in expected_slots:
        aligned = json.loads((factory / "objects" / slot / "aligned.json").read_text())
        dims = np.asarray(aligned.get("world_dims"), dtype=float)
        if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0.0):
            raise CandidateScreenError(f"invalid dimensions for {slot}")
        dimensions[slot] = dims

    def capture() -> dict[str, dict[str, Any]]:
        result = {}
        for slot in expected_slots:
            body = model.body(slot)
            result[slot] = _world_xy_rectangle(
                np.asarray(data.xpos[body.id], dtype=float),
                np.asarray(data.xquat[body.id], dtype=float),
                dimensions[slot],
            )
        return result

    nominal = capture()
    for _ in range(steps):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise CandidateScreenError("non-finite post-export footprint replay")
    settled = capture()
    footprints = {
        slot: {"nominal": nominal[slot], "settled": settled[slot]}
        for slot in expected_slots
    }
    drift = {
        slot: float(np.linalg.norm(
            np.asarray(settled[slot]["position_world_m"])
            - np.asarray(nominal[slot]["position_world_m"])
        ))
        for slot in expected_slots
    }
    return {
        "engine": "mujoco",
        "footprints": footprints,
        "model_timestep_s": float(model.opt.timestep),
        "settle_drift_m": drift,
        "simulated_duration_s": float(steps * model.opt.timestep),
        "step_count": steps,
        "step_function": "mujoco.mj_step",
    }


def _same_replay_structure(left: Any, right: Any) -> bool:
    """Compare a replay with exact structure and trillionth-unit float tolerance.

    MuJoCo state replays are bit-stable within one host, but the final floating
    point bit can differ across the heterogeneous sof1 nodes.  Non-floating
    provenance remains exact; finite floats get only the same 1e-12 absolute
    tolerance already used by the independent export-drift cross-check.
    """
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) \
                or set(left) != set(right):
            return False
        return all(_same_replay_structure(left[key], right[key]) for key in left)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) \
                or len(left) != len(right):
            return False
        return all(_same_replay_structure(a, b) for a, b in zip(left, right))
    if type(left) is not type(right):
        return False
    if isinstance(left, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(
            left,
            right,
            rel_tol=0.0,
            abs_tol=FLOAT_REPLAY_ABS_TOLERANCE,
        )
    return left == right


def _validate_export_drift_binding(
    *, rows_by_policy: Mapping[str, Mapping[str, Mapping[str, Any]]],
    footprint_replays: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Bind exporter task rows to the independent 1000-step footprint replay."""
    if set(rows_by_policy) != set(POLICIES) \
            or set(footprint_replays) != set(POLICIES):
        raise CandidateScreenError("paired export drift policy roster differs")
    bound: dict[str, dict[str, float]] = {}
    for policy in POLICIES:
        rows = rows_by_policy[policy]
        replay = footprint_replays[policy]
        footprints = replay.get("footprints") if isinstance(replay, Mapping) else None
        replay_drifts = replay.get("settle_drift_m") if isinstance(replay, Mapping) else None
        if not isinstance(rows, Mapping) or not isinstance(footprints, Mapping) \
                or not isinstance(replay_drifts, Mapping) \
                or set(rows) != set(footprints) \
                or set(rows) != set(replay_drifts):
            raise CandidateScreenError(f"{policy} export drift body roster differs")
        bound[policy] = {}
        for slot, row in rows.items():
            states = footprints[slot]
            nominal = states.get("nominal") if isinstance(states, Mapping) else None
            settled = states.get("settled") if isinstance(states, Mapping) else None
            nominal_position = nominal.get("position_world_m") \
                if isinstance(nominal, Mapping) else None
            settled_position = settled.get("position_world_m") \
                if isinstance(settled, Mapping) else None
            nominal_array = np.asarray(nominal_position, dtype=float)
            settled_array = np.asarray(settled_position, dtype=float)
            row_drift_raw = row.get("drift") if isinstance(row, Mapping) else None
            replay_drift_raw = replay_drifts[slot]
            numeric_types = (int, float, np.integer, np.floating)
            if nominal_array.shape != (3,) or settled_array.shape != (3,) \
                    or not np.isfinite(nominal_array).all() \
                    or not np.isfinite(settled_array).all() \
                    or isinstance(row_drift_raw, (bool, np.bool_)) \
                    or not isinstance(row_drift_raw, numeric_types) \
                    or isinstance(replay_drift_raw, (bool, np.bool_)) \
                    or not isinstance(replay_drift_raw, numeric_types):
                raise CandidateScreenError(f"{policy}/{slot} export drift value differs")
            measured = float(np.linalg.norm(settled_array - nominal_array))
            row_drift = float(row_drift_raw)
            replay_drift = float(replay_drift_raw)
            if not math.isfinite(row_drift) or row_drift < 0.0 \
                    or not math.isfinite(replay_drift) or replay_drift < 0.0 \
                    or not math.isclose(
                        replay_drift, measured, rel_tol=0.0, abs_tol=1e-12
                    ) \
                    or not math.isclose(
                        row_drift, measured, rel_tol=0.0, abs_tol=1e-12
                    ):
                raise CandidateScreenError(
                    f"{policy}/{slot} exporter drift differs from 1000-step replay"
                )
            bound[policy][slot] = measured
    return bound


def reach_envelope(point_xy: Sequence[float], base_xy: Sequence[float], yaw: float) -> dict[str, Any]:
    point = np.asarray(point_xy, dtype=float)
    base = np.asarray(base_xy, dtype=float)
    if point.shape != (2,) or base.shape != (2,) or not np.isfinite(point).all() \
            or not np.isfinite(base).all() or not math.isfinite(float(yaw)):
        raise CandidateScreenError("reach-envelope input is invalid")
    delta = point - base
    distance = float(np.linalg.norm(delta))
    bearing = (
        math.degrees(math.atan2(float(delta[1]), float(delta[0])) - float(yaw)) + 180.0
    ) % 360.0 - 180.0
    margins = {
        "front_cone_deg": FRONT_CONE_DEG - abs(bearing),
        "inner_radius_m": distance - REACH_MIN_M,
        "outer_radius_m": REACH_MAX_M - distance,
    }
    return {
        "bearing_off_axis_deg": bearing,
        "distance_m": distance,
        "margins": margins,
        "passed": all(value >= 0.0 for value in margins.values()),
        "safe": (
            margins["inner_radius_m"] >= MIN_RADIAL_MARGIN_M
            and margins["outer_radius_m"] >= MIN_RADIAL_MARGIN_M
            and margins["front_cone_deg"] >= MIN_CONE_MARGIN_DEG
        ),
    }


def _signed_point_rectangle_distance(point_xy: Sequence[float], rectangle: Mapping[str, Any]) -> float:
    """Euclidean separation outside an AABB; negative depth when inside."""
    point = np.asarray(point_xy, dtype=float)
    lo = np.asarray(rectangle["lo_xy_m"], dtype=float)
    hi = np.asarray(rectangle["hi_xy_m"], dtype=float)
    outside = np.maximum(np.maximum(lo - point, point - hi), 0.0)
    if np.any(outside > 0.0):
        return float(np.linalg.norm(outside))
    return -float(np.min(np.concatenate((point - lo, hi - point))))


def _signed_rectangle_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Euclidean separation for disjoint AABBs; negative overlap depth."""
    alo, ahi = np.asarray(a["lo_xy_m"], dtype=float), np.asarray(a["hi_xy_m"], dtype=float)
    blo, bhi = np.asarray(b["lo_xy_m"], dtype=float), np.asarray(b["hi_xy_m"], dtype=float)
    gap = np.maximum(np.maximum(blo - ahi, alo - bhi), 0.0)
    if np.any(gap > 0.0):
        return float(np.linalg.norm(gap))
    overlap = np.minimum(ahi, bhi) - np.maximum(alo, blo)
    return -float(np.min(overlap))


def _rectangle(center_xy: Sequence[float], half_xy: Sequence[float]) -> dict[str, list[float]]:
    center = np.asarray(center_xy, dtype=float)
    half = np.asarray(half_xy, dtype=float)
    if center.shape != (2,) or half.shape != (2,) or not np.isfinite(center).all() \
            or not np.isfinite(half).all() or np.any(half <= 0.0):
        raise CandidateScreenError("rectangle input is invalid")
    return {
        "lo_xy_m": (center - half).tolist(),
        "hi_xy_m": (center + half).tolist(),
    }


def base_clearance(
    base_xy: Sequence[float], footprints: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for slot, states in sorted(footprints.get(policy, {}).items()):
            for state in ("nominal", "settled"):
                if state not in states:
                    raise CandidateScreenError(f"missing {policy}/{slot}/{state} footprint")
                edge_distance = _signed_point_rectangle_distance(base_xy, states[state])
                clearance = edge_distance - ROBOT_BASE_RADIUS_M
                rows.append({
                    "clearance_m": clearance,
                    "policy_id": policy,
                    "slot": slot,
                    "state": state,
                })
    minimum = min((row["clearance_m"] for row in rows), default=None)
    blockers = [
        row for row in rows if row["clearance_m"] < MIN_BASE_BODY_CLEARANCE_M
    ]
    return {
        "minimum_clearance_m": minimum,
        "passed": bool(rows) and not blockers,
        "required_clearance_m": MIN_BASE_BODY_CLEARANCE_M,
        "robot_base_radius_m": ROBOT_BASE_RADIUS_M,
        "blocking_footprints": blockers,
    }


def region_obstacle_clearance(
    region: Mapping[str, Any],
    *,
    target_slot: str,
    footprints: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    region_rect = _rectangle(
        [float(region["cx"]), float(region["cy"])],
        [float(region["hx"]), float(region["hy"])],
    )
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        for slot, states in sorted(footprints.get(policy, {}).items()):
            if slot == target_slot:
                continue
            for state in ("nominal", "settled"):
                clearance = _signed_rectangle_distance(region_rect, states[state])
                rows.append({
                    "clearance_m": clearance,
                    "policy_id": policy,
                    "slot": slot,
                    "state": state,
                })
    minimum = min((row["clearance_m"] for row in rows), default=None)
    blockers = [row for row in rows if row["clearance_m"] < MIN_OBSTACLE_CLEARANCE_M]
    return {
        "minimum_clearance_m": minimum,
        "passed": not blockers,
        "required_clearance_m": MIN_OBSTACLE_CLEARANCE_M,
        "blocking_footprints": blockers,
    }


def _eligibility_reasons(row: Mapping[str, Any] | None) -> list[str]:
    if row is None:
        return ["not_accepted_in_construction_arm"]
    from robo.tasks import pi05_tasks

    reasons = []
    label = str(row.get("label", "")).lower()
    dims = np.asarray(row.get("dims"), dtype=float)
    if label not in pi05_tasks.GRASP_LABELS:
        reasons.append("label_not_graspable")
    if row.get("construction_eligible") is not True and row.get("tier") not in ("A", "B"):
        reasons.append("tier_not_A_or_B")
    if dims.shape != (3,) or not np.isfinite(dims).all():
        reasons.append("dimensions_invalid")
    else:
        if float(np.max(dims)) > 0.28:
            reasons.append("dimension_above_280mm")
        if float(np.min(dims)) < 0.005:
            reasons.append("dimension_below_5mm")
    mass = row.get("mass")
    if isinstance(mass, bool) or not isinstance(mass, (int, float)) \
            or not math.isfinite(float(mass)) or float(mass) > 1.0:
        reasons.append("mass_above_1kg_or_invalid")
    drift = row.get("drift")
    if isinstance(drift, bool) or not isinstance(drift, (int, float)) \
            or not math.isfinite(float(drift)) \
            or float(drift) >= STABILITY_DRIFT_LIMIT_M:
        reasons.append("export_target_drift_not_below_30mm")
    return reasons


def task_destination(task: Mapping[str, Any], env: Any) -> dict[str, Any]:
    """Resolve the existing TaskScorer rubric, without certifying a cavity."""
    from types import SimpleNamespace
    from robo.tasks.pi05_tasks import TaskScorer

    receptacle = task.get("receptacle")
    family = "object_to_receptacle" if receptacle is not None else "object_to_region"
    if task.get("task_family", family) != family:
        raise CandidateScreenError("declared task family differs from destination")
    if receptacle is None:
        if not isinstance(task.get("region"), Mapping):
            raise CandidateScreenError("region task lacks its declared destination")
        return {"task_family": family, "destination_body": None,
                "destination_region": dict(task["region"])}
    if not isinstance(receptacle, str) or not receptacle or receptacle == task.get("target"):
        raise CandidateScreenError("receptacle must be a distinct named body")
    if task.get("region") is not None:
        raise CandidateScreenError("receptacle task cannot also declare a region")
    dimensions = np.asarray(task.get("receptacle_dims"), dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise CandidateScreenError("receptacle dimensions are invalid")
    try:
        position, quaternion = env.body_pose(receptacle)
    except (KeyError, ValueError) as exc:
        raise CandidateScreenError("receptacle is absent from runtime") from exc
    if np.asarray(position).shape != (3,) or not np.isfinite(position).all():
        raise CandidateScreenError("receptacle runtime position is invalid")
    region = TaskScorer._resolve_region(SimpleNamespace(env=env), task)
    return {"task_family": family, "destination_body": receptacle,
            "destination_region": region,
            "receptacle_dims_m": dimensions.tolist(),
            "receptacle_position_world_m": np.asarray(position).tolist(),
            "goal_collision_status": "NOT_RUN", "cavity_verified": False,
            "scope": "canonical TaskScorer region only; no container cavity or goal collision certification"}


def paired_receptacle_evidence(
    task: Mapping[str, Any], rows_by_policy: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Check a declared pair using existing role constraints; never select tasks."""
    from robo.tasks import pi05_tasks

    if (task.get("receptacle") is None or set(rows_by_policy) != set(POLICIES)
            or task.get("task_family", "object_to_receptacle") != "object_to_receptacle"):
        raise CandidateScreenError("paired receptacle declaration requires A0 and A4")
    target, destination = task.get("target"), task["receptacle"]
    if not isinstance(target, str) or not isinstance(destination, str) or target == destination:
        raise CandidateScreenError("paired target and receptacle must be distinct")
    evidence = {}
    for policy in POLICIES:
        rows = rows_by_policy[policy]
        target_row, destination_row = rows.get(target), rows.get(destination)
        reasons = list(_eligibility_reasons(target_row))
        if destination_row is None:
            reasons.append("receptacle_not_accepted_in_construction_arm")
        else:
            dimensions = np.asarray(destination_row.get("dims"), dtype=float)
            if (dimensions.shape != (3,) or not np.isfinite(dimensions).all()
                    or not math.isfinite(float(destination_row.get("drift", float("nan"))))):
                reasons.append("receptacle_geometry_invalid")
            elif not pi05_tasks._is_receptacle({**destination_row, "dims": dimensions}):
                reasons.append("receptacle_fails_canonical_role_constraints")
        evidence[policy] = {"reasons": reasons, "role_constraints_passed": not reasons}
    return {"task_family": "object_to_receptacle", "target": target,
            "destination_body": destination, "by_policy": evidence,
            "role_constraints_passed": all(x["role_constraints_passed"] for x in evidence.values()),
            "goal_collision_status": "NOT_RUN", "cavity_verified": False,
            "qualification_passed": False}


def _goal_probe_checks(probe: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, bool]:
    """Replay fixed-witness evidence; these are geometry checks, not success."""
    expected_keys = {"schema_version", "task_family", "target", "receptacle", "placement_rule",
                     "destination_region", "original_positions_m", "original_quaternions_wxyz",
                     "witness_position_m", "step_count", "timestep_s", "duration_s",
                     "state_before_sha256", "scope", "probe_start_positions_m",
                     "probe_start_quaternions_wxyz", "initial_target_contacts", "settled_positions_m",
                     "settled_target_contacts", "settled_linear_velocity_m_s", "state_after_sha256"}
    if set(probe) - {"checks", "passed"} != expected_keys:
        raise CandidateScreenError("goal probe evidence schema differs")
    before, after = probe["probe_start_positions_m"], probe["settled_positions_m"]
    speeds = probe["settled_linear_velocity_m_s"]
    if set(before) != set(after) or set(before) != set(speeds):
        raise CandidateScreenError("goal probe body denominator differs")
    drift = [float(np.linalg.norm(np.asarray(after[k])-before[k])) for k in before]
    speed = [float(np.linalg.norm(speeds[k])) for k in before]
    arrays = [*before.values(), *after.values(), *speeds.values()]
    if not arrays or any(np.asarray(x).shape != (3,) or not np.isfinite(x).all() for x in arrays):
        raise CandidateScreenError("goal probe measured states are invalid")
    contacts = [*probe["initial_target_contacts"], *probe["settled_target_contacts"]]
    for contact in contacts:
        if (not isinstance(contact, Mapping) or set(contact) != {"bodies", "geom_ids", "distance_m"}
                or not isinstance(contact["bodies"], list) or len(contact["bodies"]) != 2
                or task["target"] not in contact["bodies"]
                or any(not isinstance(x, str) for x in contact["bodies"])
                or not isinstance(contact["geom_ids"], list) or len(contact["geom_ids"]) != 2
                or any(type(x) is not int or x < 0 for x in contact["geom_ids"])
                or isinstance(contact["distance_m"], bool)
                or not isinstance(contact["distance_m"], (int, float))
                or not math.isfinite(contact["distance_m"])):
            raise CandidateScreenError("goal probe contact evidence is invalid")
    region = probe["destination_region"]
    position = after[task["target"]]
    return {
        "initial_witness_has_no_penetration": all(c["distance_m"] >= 0 for c in probe["initial_target_contacts"]),
        "target_contacts_receptacle_after_settle": any(task["receptacle"] in c["bodies"] for c in probe["settled_target_contacts"]),
        # Same 10mm penetration bound as the frozen E3 physical-verification policy.
        "settled_contact_penetration_at_most_10mm": all(c["distance_m"] >= -0.01 for c in probe["settled_target_contacts"]),
        "all_body_settle_drift_below_30mm": max(drift) < STABILITY_DRIFT_LIMIT_M,
        "all_body_linear_speed_below_50mm_s": max(speed) < AT_REST_LINEAR_SPEED_M_S,
        "settled_target_in_canonical_region": bool(abs(position[0]-region["cx"]) <= region["hx"] and abs(position[1]-region["cy"]) <= region["hy"] and region["zlo"] <= position[2] <= region["zhi"]),
        "exactly_900_mujoco_steps": probe["step_count"] == EXPECTED_RESET_STEPS and probe["duration_s"] == RESET_SETTLE_S and math.isclose(probe["timestep_s"] * probe["step_count"], RESET_SETTLE_S, rel_tol=0, abs_tol=1e-12),
        "complete_integration_state_restored": probe["state_before_sha256"] == probe["state_after_sha256"],
    }


def receptacle_goal_probe(env: Any, task: Mapping[str, Any]) -> dict[str, Any]:
    """One fixed interior witness in the actual model, restoring all MjData."""
    import mujoco
    from robo.envs.pi05_env import _settle_step_count

    destination = task_destination(task, env)
    if destination["destination_body"] is None:
        raise CandidateScreenError("goal probe requires a receptacle task")
    target, receptacle = task["target"], task["receptacle"]
    slots = sorted(set(env.free_bodies) | {receptacle})
    if target not in slots:
        raise CandidateScreenError("goal probe target is absent")
    body = env.model.body(target)
    if int(body.jntnum[0]) != 1 or int(body.geomnum[0]) < 1:
        raise CandidateScreenError("goal probe target needs one free joint and collision geometry")
    joint = int(body.jntadr[0])
    if env.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise CandidateScreenError("goal probe target is not free")
    for name in (target, receptacle):
        b = env.model.body(name)
        ids = range(int(b.geomadr[0]), int(b.geomadr[0])+int(b.geomnum[0]))
        if not any(env.model.geom_contype[g] or env.model.geom_conaffinity[g] for g in ids):
            raise CandidateScreenError("goal probe body lacks enabled collision geometry")
    count = _settle_step_count(RESET_SETTLE_S, float(env.model.opt.timestep))
    if count != EXPECTED_RESET_STEPS:
        raise CandidateScreenError("goal probe requires frozen 600Hz model")
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    def state_hash():
        state = np.empty(mujoco.mj_stateSize(env.model, state_spec))
        mujoco.mj_getState(env.model, env.data, state, state_spec)
        if not np.isfinite(state).all():
            raise CandidateScreenError("goal probe integration state is nonfinite")
        return hashlib.sha256(state.tobytes()).hexdigest()
    def positions():
        return {name: np.asarray(env.body_pose(name)[0]).tolist() for name in slots}
    def contacts():
        result = []
        for contact in env.data.contact[:env.data.ncon]:
            bodies = [env.model.body(int(env.model.geom_bodyid[g])).name for g in (contact.geom1,contact.geom2)]
            if target in bodies:
                result.append({"bodies": bodies, "geom_ids": [int(contact.geom1),int(contact.geom2)], "distance_m": float(contact.dist)})
        return result
    original_positions = positions()
    original_quaternions = {name: np.asarray(env.body_pose(name)[1]).tolist() for name in slots}
    before_hash = state_hash()
    backup = mujoco.MjData(env.model)
    mujoco.mj_copyData(backup, env.model, env.data)
    witness = np.asarray(env.body_pose(receptacle)[0], dtype=float)
    region = destination["destination_region"]
    probe = {"schema_version":1, "task_family":"object_to_receptacle", "target":target,
             "receptacle":receptacle, "placement_rule":"receptacle_body_center_preserve_target_quaternion",
             "destination_region":region, "original_positions_m":original_positions,
             "original_quaternions_wxyz":original_quaternions,
             "witness_position_m":witness.tolist(), "step_count":count,
             "timestep_s":float(env.model.opt.timestep), "duration_s":RESET_SETTLE_S,
             "state_before_sha256":before_hash,
             "scope":"one fixed interior collision/settle witness only; not all scorer positions, motion planning or policy success"}
    try:
        qadr, vadr = int(env.model.jnt_qposadr[joint]), int(env.model.jnt_dofadr[joint])
        env.data.qpos[qadr:qadr+3] = witness
        env.data.qvel[vadr:vadr+6] = 0
        mujoco.mj_forward(env.model, env.data)
        probe["probe_start_positions_m"] = positions()
        probe["probe_start_quaternions_wxyz"] = {name: np.asarray(env.body_pose(name)[1]).tolist() for name in slots}
        probe["initial_target_contacts"] = contacts()
        for _ in range(count):
            mujoco.mj_step(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        probe["settled_positions_m"] = positions()
        probe["settled_target_contacts"] = contacts()
        velocities = {}
        for name in slots:
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, env.model.body(name).id, velocity, 0)
            velocities[name] = velocity[3:].tolist()
        probe["settled_linear_velocity_m_s"] = velocities
    finally:
        mujoco.mj_copyData(env.data, env.model, backup)
    probe["state_after_sha256"] = state_hash()
    if probe["state_after_sha256"] != before_hash or positions() != original_positions:
        raise CandidateScreenError("goal probe failed to restore original state")
    probe["checks"] = _goal_probe_checks(probe, task)
    probe["passed"] = all(probe["checks"].values())
    return probe


def _validate_goal_probe(probe, *, task, current, destination):
    if (not isinstance(probe, Mapping) or probe.get("schema_version") != 1
            or probe.get("task_family") != "object_to_receptacle"
            or probe.get("target") != task["target"] or probe.get("receptacle") != task["receptacle"]
            or probe.get("placement_rule") != "receptacle_body_center_preserve_target_quaternion"
            or probe.get("destination_region") != destination["destination_region"]):
        raise CandidateScreenError("goal probe task binding differs")
    original = {name: row["position_world_m"] for name,row in current.items()}
    quaternions = {name: row["quaternion_wxyz"] for name,row in current.items()}
    if (probe.get("original_quaternions_wxyz") != quaternions
            or probe.get("probe_start_quaternions_wxyz") != quaternions):
        raise CandidateScreenError("goal probe changed the frozen object orientation")
    if probe.get("original_positions_m") != original:
        raise CandidateScreenError("goal probe original state differs from reset footprints")
    witness = original[task["receptacle"]]
    expected_start = {**original, task["target"]: witness}
    if probe.get("witness_position_m") != witness or probe.get("probe_start_positions_m") != expected_start:
        raise CandidateScreenError("goal probe placement rule differs")
    for name in ("state_before_sha256", "state_after_sha256"):
        if SHA256_RE.fullmatch(str(probe.get(name))) is None:
            raise CandidateScreenError("goal probe state hash is invalid")
    checks = _goal_probe_checks(probe, task)
    if probe.get("checks") != checks or type(probe.get("passed")) is not bool or probe["passed"] != all(checks.values()):
        raise CandidateScreenError("goal probe derived checks differ")
    return probe["passed"]



def _cluster_targets(rows: Sequence[Mapping[str, Any]], tolerance_m: float = 0.05) -> list[list[str]]:
    ordered = sorted((float(row["bottom_z"]), str(row["name"])) for row in rows)
    groups: list[list[tuple[float, str]]] = []
    for z, slot in ordered:
        if groups and z - groups[-1][-1][0] < tolerance_m:
            groups[-1].append((z, slot))
        else:
            groups.append([(z, slot)])
    return [[slot for _z, slot in group] for group in groups]


def _axis_values(lo: float, hi: float, step: float) -> list[float]:
    if hi < lo:
        return []
    count = int(math.floor((hi - lo) / step + 1e-9))
    values = [lo + index * step for index in range(count + 1)]
    if not values or hi - values[-1] > 1e-9:
        values.append(hi)
    return sorted({round(value, 9) for value in values})


def _table_for_cluster(
    cluster_slots: Sequence[str], rows_by_policy: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    target_rows = [rows_by_policy[PLANNING_SOURCE][slot] for slot in cluster_slots]
    level = float(np.mean([float(row["bottom_z"]) for row in target_rows]))
    members: dict[tuple[str, str], Mapping[str, Any]] = {}
    for policy in POLICIES:
        for slot, row in rows_by_policy[policy].items():
            if abs(float(row["bottom_z"]) - level) < 0.05:
                members[(policy, slot)] = row
    if not members:
        raise CandidateScreenError("support cluster has no accepted members")
    lo = np.min([np.asarray(row["aabb"], dtype=float)[0, :2] for row in members.values()], axis=0) - 0.30
    hi = np.max([np.asarray(row["aabb"], dtype=float)[1, :2] for row in members.values()], axis=0) + 0.30
    return {
        "cx": float((lo[0] + hi[0]) / 2.0),
        "cy": float((lo[1] + hi[1]) / 2.0),
        "hx": float((hi[0] - lo[0]) / 2.0),
        "hy": float((hi[1] - lo[1]) / 2.0),
        "top_z": float(min(float(row["bottom_z"]) for row in members.values()) - 0.02),
        # Match the exported 20 mm support slabs. A tabletop must not fill
        # the entire under-table volume occupied by preserved floor objects.
        "thickness_m": 0.02,
        "inference": "union of both-arm accepted source AABBs within 50mm support level, padded 300mm",
        "member_slots": sorted({slot for _policy, slot in members}),
        "policy_slot_members": [
            {"policy_id": policy, "slot": slot}
            for policy, slot in sorted(members)
        ],
        "support_level_m": level,
    }


def _base_candidates(table: Mapping[str, Any], target_rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    table_xlo = float(table["cx"]) - float(table["hx"]) + 0.10
    table_xhi = float(table["cx"]) + float(table["hx"]) - 0.10
    table_ylo = float(table["cy"]) - float(table["hy"]) + 0.10
    table_yhi = float(table["cy"]) + float(table["hy"]) - 0.10
    target_xy = np.asarray([np.asarray(row["center"], dtype=float)[:2] for row in target_rows])
    # Bound the Cartesian search to the intersection relevant to these targets;
    # same-height objects elsewhere in a room must not explode the grid.
    xlo = max(table_xlo, float(np.min(target_xy[:, 0])) - 0.80)
    xhi = min(table_xhi, float(np.max(target_xy[:, 0])) + 0.80)
    ylo = max(table_ylo, float(np.min(target_xy[:, 1])) - 0.80)
    yhi = min(table_yhi, float(np.max(target_xy[:, 1])) + 0.80)
    # A full table grid is deterministic and is augmented with target-centric
    # polar samples so narrow two-target reach intersections are not missed.
    values = [
        [x, y]
        for x in _axis_values(xlo, xhi, BASE_GRID_STEP_M)
        for y in _axis_values(ylo, yhi, BASE_GRID_STEP_M)
    ]
    centers = [np.asarray(row["center"], dtype=float)[:2] for row in target_rows]
    anchors = [*centers, np.mean(centers, axis=0)]
    for anchor in anchors:
        for radius in (0.35, 0.45, 0.55, 0.65, 0.75):
            for degrees in range(0, 360, 10):
                point = anchor + radius * np.array([
                    math.cos(math.radians(degrees)),
                    math.sin(math.radians(degrees)),
                ])
                if xlo <= point[0] <= xhi and ylo <= point[1] <= yhi:
                    values.append(point.tolist())
    return [list(point) for point in sorted({(round(p[0], 9), round(p[1], 9)) for p in values})]


def _yaw_candidates(base_xy: Sequence[float], target_rows: Sequence[Mapping[str, Any]]) -> list[float]:
    base = np.asarray(base_xy, dtype=float)
    centers = [np.asarray(row["center"], dtype=float)[:2] for row in target_rows]
    anchors = [*centers, np.mean(centers, axis=0)]
    values = [math.atan2(float(point[1] - base[1]), float(point[0] - base[0])) for point in anchors]
    return sorted({round(value, 12) for value in values})


def _base_frame_coordinates(point_xy: Sequence[float], base_xy: Sequence[float], yaw: float) -> tuple[float, float]:
    delta = np.asarray(point_xy, dtype=float) - np.asarray(base_xy, dtype=float)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (
        float(cosine * delta[0] + sine * delta[1]),
        float(-sine * delta[0] + cosine * delta[1]),
    )


def shared_qualifier(
    target_slot: str,
    *, rows_by_policy: Mapping[str, Mapping[str, Mapping[str, Any]]],
    footprints: Mapping[str, Mapping[str, Mapping[str, Any]]],
    base_xy: Sequence[float], base_yaw: float,
) -> dict[str, Any]:
    """Find one strict language qualifier valid in both construction arms."""
    target = rows_by_policy[PLANNING_SOURCE][target_slot]
    label = str(target["label"])
    per_policy: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        same = [
            row for row in rows_by_policy[policy].values()
            if str(row["label"]) == label and not _eligibility_reasons(row)
        ]
        coordinates = {
            str(row["name"]): _base_frame_coordinates(
                footprints[policy][str(row["name"])]["settled"]["center_xy_m"],
                base_xy,
                base_yaw,
            )
            for row in same
        }
        if target_slot not in coordinates:
            return {
                "passed": False,
                "reason": f"target_not_eligible_in_{policy}",
                "target_label": label,
            }
        per_policy[policy] = {
            "eligible_same_label_slots": sorted(coordinates),
            "coordinates_forward_left_m": {
                slot: [values[0], values[1]] for slot, values in sorted(coordinates.items())
            },
        }
    cardinalities = {policy: len(per_policy[policy]["eligible_same_label_slots"]) for policy in POLICIES}
    if all(value == 1 for value in cardinalities.values()):
        return {
            "any_instance": False,
            "margin_m": None,
            "passed": True,
            "per_policy": per_policy,
            "qualifier": None,
            "target_label": label,
        }
    if any(value == 1 for value in cardinalities.values()):
        return {
            "passed": False,
            "per_policy": per_policy,
            "reason": "same_label_cardinality_differs_between_arms",
            "target_label": label,
        }
    candidates = []
    for qualifier in QUALIFIERS:
        margins = {}
        for policy in POLICIES:
            coordinates = per_policy[policy]["coordinates_forward_left_m"]
            index, sign = {
                "leftmost": (1, 1.0),
                "rightmost": (1, -1.0),
                "nearest": (0, -1.0),
                "farthest": (0, 1.0),
            }[qualifier]
            target_score = sign * float(coordinates[target_slot][index])
            competitor = max(
                sign * float(values[index])
                for slot, values in coordinates.items()
                if slot != target_slot
            )
            margins[policy] = target_score - competitor
        minimum = min(margins.values())
        if minimum > QUALIFIER_MARGIN_M:
            candidates.append((minimum, -QUALIFIERS.index(qualifier), qualifier, margins))
    if not candidates:
        return {
            "passed": False,
            "per_policy": per_policy,
            "reason": "no_shared_qualifier_strictly_above_30mm",
            "target_label": label,
        }
    minimum, _order, qualifier, margins = max(candidates)
    return {
        "any_instance": False,
        "margin_m": minimum,
        "margins_by_policy_m": margins,
        "passed": True,
        "per_policy": per_policy,
        "qualifier": qualifier,
        "strict_required_margin_m": QUALIFIER_MARGIN_M,
        "target_label": label,
    }


def _target_max_half_extents(
    target_slot: str, footprints: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> list[float]:
    values = []
    for policy in POLICIES:
        for state in ("nominal", "settled"):
            half = np.asarray(
                footprints[policy][target_slot][state]["half_extents_xy_m"], dtype=float
            )
            if half.shape != (2,) or not np.isfinite(half).all() or np.any(half <= 0.0):
                raise CandidateScreenError("target XY half extents are invalid")
            values.append(half)
    return np.max(values, axis=0).astype(float).tolist()


def _region_candidates(
    target_slot: str,
    *, table: Mapping[str, Any],
    footprints: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    target_half_xy = _target_max_half_extents(target_slot, footprints)
    half_x = max(
        min(0.30, float(table["hx"]) / 3.0),
        target_half_xy[0] + TARGET_GOAL_PADDING_M,
    )
    half_y = max(
        min(0.30, float(table["hy"]) / 3.0),
        target_half_xy[1] + TARGET_GOAL_PADDING_M,
    )
    table_xlo = float(table["cx"]) - float(table["hx"]) + half_x + TABLE_EDGE_CLEARANCE_M
    table_xhi = float(table["cx"]) + float(table["hx"]) - half_x - TABLE_EDGE_CLEARANCE_M
    table_ylo = float(table["cy"]) - float(table["hy"]) + half_y + TABLE_EDGE_CLEARANCE_M
    table_yhi = float(table["cy"]) + float(table["hy"]) - half_y - TABLE_EDGE_CLEARANCE_M
    target_centers = np.asarray([
        footprints[policy][target_slot]["settled"]["center_xy_m"] for policy in POLICIES
    ])
    xlo = max(table_xlo, float(np.min(target_centers[:, 0])) - 0.75)
    xhi = min(table_xhi, float(np.max(target_centers[:, 0])) + 0.75)
    ylo = max(table_ylo, float(np.min(target_centers[:, 1])) - 0.75)
    yhi = min(table_yhi, float(np.max(target_centers[:, 1])) + 0.75)
    result = []
    for x in _axis_values(xlo, xhi, REGION_GRID_STEP_M):
        for y in _axis_values(ylo, yhi, REGION_GRID_STEP_M):
            region = {
                "cx": x,
                "cy": y,
                "hx": half_x,
                "hy": half_y,
                "zlo": float(table["top_z"]) - 0.02,
                "zhi": float(table["top_z"]) + 0.30,
            }
            obstacle = region_obstacle_clearance(
                region, target_slot=target_slot, footprints=footprints
            )
            region_rect = _rectangle([x, y], [half_x, half_y])
            target_clearance_rows = []
            displacement_rows = []
            for policy in POLICIES:
                for state in ("nominal", "settled"):
                    target_rect = footprints[policy][target_slot][state]
                    target_clearance_rows.append({
                        "clearance_m": _signed_rectangle_distance(region_rect, target_rect),
                        "policy_id": policy,
                        "state": state,
                    })
                center = footprints[policy][target_slot]["settled"]["center_xy_m"]
                displacement_rows.append({
                    "displacement_m": float(np.linalg.norm(np.asarray([x, y]) - np.asarray(center))),
                    "policy_id": policy,
                })
            target_clearance = min(row["clearance_m"] for row in target_clearance_rows)
            displacement = min(row["displacement_m"] for row in displacement_rows)
            reasons = []
            if not obstacle["passed"]:
                reasons.append("goal_obstacle_clearance_below_20mm")
            if target_clearance < MIN_OBSTACLE_CLEARANCE_M:
                reasons.append("target_initial_footprint_not_clear_of_goal")
            if displacement < MIN_TASK_DISPLACEMENT_M:
                reasons.append("goal_center_displacement_below_200mm")
            result.append({
                "goal_obstacle_clearance": obstacle,
                "minimum_center_displacement_m": displacement,
                "minimum_target_goal_clearance_m": target_clearance,
                "region": region,
                "rejection_reasons": reasons,
                "target_goal_clearance_rows": target_clearance_rows,
                "target_max_half_extents_xy_m": target_half_xy,
            })
    return result


def _region_for_base(
    options: Sequence[Mapping[str, Any]], *, base_xy: Sequence[float], base_yaw: float
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    rejected: dict[str, int] = {}
    passing = []
    for option in options:
        reasons = list(option["rejection_reasons"])
        region = option["region"]
        center = reach_envelope([region["cx"], region["cy"]], base_xy, base_yaw)
        target_half_xy = option["target_max_half_extents_xy_m"]
        core_hx = float(region["hx"]) - float(target_half_xy[0]) \
            - TARGET_GOAL_PADDING_M
        core_hy = float(region["hy"]) - float(target_half_xy[1]) \
            - TARGET_GOAL_PADDING_M
        safe_samples = []
        if core_hx > 0.0 and core_hy > 0.0:
            for x in (region["cx"] - core_hx, region["cx"], region["cx"] + core_hx):
                for y in (region["cy"] - core_hy, region["cy"], region["cy"] + core_hy):
                    safe_samples.append(reach_envelope([x, y], base_xy, base_yaw))
        safe_count = sum(bool(row["safe"]) for row in safe_samples)
        if not center["safe"]:
            reasons.append("goal_center_lacks_reach_margin")
        if safe_count < 5:
            reasons.append("fewer_than_5_of_9_eroded_core_samples_reachable")
        for reason in set(reasons):
            rejected[reason] = rejected.get(reason, 0) + 1
        if reasons:
            continue
        reach_margin = min(center["margins"].values())
        passing.append((
            min(
                float(option["goal_obstacle_clearance"]["minimum_clearance_m"])
                if option["goal_obstacle_clearance"]["minimum_clearance_m"] is not None
                else 1.0,
                1.0,
            ),
            reach_margin,
            float(option["minimum_center_displacement_m"]),
            -float(region["cx"]),
            -float(region["cy"]),
            {
                **dict(option),
                "destination_center_reach": center,
                "eroded_core_half_extents_m": [core_hx, core_hy],
                "eroded_core_safe_count": safe_count,
                "eroded_core_samples": safe_samples,
            },
        ))
    if not passing:
        return None, dict(sorted(rejected.items()))
    return max(passing)[-1], dict(sorted(rejected.items()))


def plan_paired_region_tasks(
    *, scene_id: str,
    rows_by_policy: Mapping[str, Mapping[str, Mapping[str, Any]]],
    footprints: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Jointly select one base and every strict, obstacle-clear region task."""
    _validated_scene_id(scene_id)
    all_slots = sorted(set().union(*(set(rows_by_policy[p]) for p in POLICIES)))
    object_audit: dict[str, dict[str, Any]] = {}
    eligible = []
    for slot in all_slots:
        reasons_by_policy = {
            policy: _eligibility_reasons(rows_by_policy[policy].get(slot))
            for policy in POLICIES
        }
        reasons = sorted({reason for values in reasons_by_policy.values() for reason in values})
        object_audit[slot] = {
            "eligibility_reasons_by_policy": reasons_by_policy,
            "paired_eligible": not reasons,
            "rejection_reasons": reasons,
        }
        if not reasons:
            eligible.append(rows_by_policy[PLANNING_SOURCE][slot])
    clusters = _cluster_targets(eligible)
    selected = None
    clear_base_pose_count = 0
    for cluster_slots in clusters:
        table = _table_for_cluster(cluster_slots, rows_by_policy)
        target_rows = [rows_by_policy[PLANNING_SOURCE][slot] for slot in cluster_slots]
        region_options = {
            slot: _region_candidates(slot, table=table, footprints=footprints)
            for slot in cluster_slots
        }
        for base_xy in _base_candidates(table, target_rows):
            clearance = base_clearance(base_xy, footprints)
            if not clearance["passed"]:
                continue
            for yaw in _yaw_candidates(base_xy, target_rows):
                clear_base_pose_count += 1
                tasks = []
                per_target = {}
                for slot in cluster_slots:
                    target_reach = {
                        policy: reach_envelope(
                            footprints[policy][slot]["settled"]["center_xy_m"],
                            base_xy,
                            yaw,
                        )
                        for policy in POLICIES
                    }
                    qualifier = shared_qualifier(
                        slot,
                        rows_by_policy=rows_by_policy,
                        footprints=footprints,
                        base_xy=base_xy,
                        base_yaw=yaw,
                    )
                    reasons = []
                    if not all(value["safe"] for value in target_reach.values()):
                        reasons.append("target_lacks_reach_margin_in_both_arms")
                    if not qualifier.get("passed"):
                        reasons.append(str(qualifier.get("reason", "qualifier_failed")))
                    region = None
                    region_rejections = {}
                    if not reasons:
                        region, region_rejections = _region_for_base(
                            region_options[slot], base_xy=base_xy, base_yaw=yaw
                        )
                        if region is None:
                            reasons.append("no_reachable_obstacle_clear_region")
                    per_target[slot] = {
                        "qualifier": qualifier,
                        "region_rejection_counts": region_rejections,
                        "rejection_reasons": reasons,
                        "target_reach": target_reach,
                    }
                    if reasons:
                        continue
                    label = str(rows_by_policy[PLANNING_SOURCE][slot]["label"])
                    tag = qualifier["qualifier"]
                    phrase = f"the {tag} {label}" if tag else f"the {label}"
                    task = {
                        "any_instance": False,
                        "instructions": {
                            "default": f"move {phrase} to the designated region",
                            "specific": f"pick up {phrase} and place it in the designated region",
                            "vague": f"move {phrase}",
                        },
                        "receptacle": None,
                        "region": region["region"],
                        "target": slot,
                        "target_label": label,
                        "task_id": f"{scene_id}__{slot}_to_region",
                    }
                    tasks.append({
                        "planning_metrics": {
                            "goal": region,
                            "qualifier": qualifier,
                            "target_reach": target_reach,
                        },
                        "task": task,
                    })
                task_minimum = min(
                    (
                        min(
                            min(value["margins"]["inner_radius_m"],
                                value["margins"]["outer_radius_m"])
                            for value in row["planning_metrics"]["target_reach"].values()
                        )
                        for row in tasks
                    ),
                    default=-1.0,
                )
                candidate = {
                    "base_clearance": clearance,
                    "base_pos": [float(base_xy[0]), float(base_xy[1]), float(table["top_z"])],
                    "base_yaw": float(yaw),
                    "cluster_slots": list(cluster_slots),
                    "per_target": per_target,
                    "score": [len(tasks), task_minimum, clearance["minimum_clearance_m"]],
                    "table": table,
                    "tasks": tasks,
                }
                if selected is None or (
                    candidate["score"][0], candidate["score"][1], candidate["score"][2],
                    -candidate["base_pos"][0], -candidate["base_pos"][1], -candidate["base_yaw"],
                ) > (
                    selected["score"][0], selected["score"][1], selected["score"][2],
                    -selected["base_pos"][0], -selected["base_pos"][1], -selected["base_yaw"],
                ):
                    selected = candidate
    if selected is None:
        selected = {
            "base_clearance": None,
            "base_pos": None,
            "base_yaw": None,
            "cluster_slots": [],
            "per_target": {},
            "score": [0, -1.0, -1.0],
            "table": None,
            "tasks": [],
        }
    selected_slots = {row["task"]["target"] for row in selected["tasks"]}
    for slot in all_slots:
        if slot in selected_slots:
            object_audit[slot]["selected"] = True
            continue
        object_audit[slot]["selected"] = False
        if object_audit[slot]["paired_eligible"]:
            reasons = selected.get("per_target", {}).get(slot, {}).get(
                "rejection_reasons", ["not_on_selected_support_cluster"]
            )
            object_audit[slot]["rejection_reasons"] = list(reasons)
    return {
        "all_object_candidates": object_audit,
        "base_search": {
            "clear_base_pose_count": clear_base_pose_count,
            "hard_base_clearance_m": MIN_BASE_BODY_CLEARANCE_M,
            "robot_base_radius_m": ROBOT_BASE_RADIUS_M,
            "support_clusters": clusters,
        },
        "candidate_count": len(selected["tasks"]),
        "planning_source": PLANNING_SOURCE,
        "selected": selected,
    }


def _write_and_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_bundle(
    destination: Path,
    *, manifest_kind: str,
    payloads: Mapping[str, bytes],
    manifest_fields: Mapping[str, Any],
) -> dict[str, Any]:
    if not payloads or any(
        not name or "/" in name or name in {"manifest.json", "seal.json"}
        for name in payloads
    ):
        raise CandidateScreenError("atomic bundle payload names are invalid")
    try:
        with sealed_cpu._atomic_directory(destination) as staging:
            files = {}
            for name, payload in sorted(payloads.items()):
                path = staging / name
                _write_and_fsync(path, payload)
                files[name] = _identity_without_path(path)
            manifest = {
                **dict(manifest_fields),
                "files": files,
                "manifest_kind": manifest_kind,
                "schema_version": SCHEMA_VERSION,
            }
            manifest_path = staging / "manifest.json"
            _write_and_fsync(manifest_path, _json_bytes(manifest))
            members = {"manifest.json": _identity_without_path(manifest_path)}
            if "gate.json" in payloads:
                members["gate.json"] = files["gate.json"]
            seal = {
                "manifest_kind": manifest_kind,
                "members": members,
                "schema_version": SCHEMA_VERSION,
            }
            _write_and_fsync(staging / "seal.json", _json_bytes(seal))
            for path in staging.iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
    except sealed_cpu.PilotGateError as exc:
        raise CandidateScreenError(str(exc)) from exc
    return {
        "directory": str(destination),
        "manifest": manifest,
        "manifest_sha256": _sha256(destination / "manifest.json"),
        "seal_sha256": _sha256(destination / "seal.json"),
    }


def _validate_bundle(
    directory: Path, *, root: Path, expected_kind: str,
) -> dict[str, Any]:
    try:
        bundle = sealed_cpu._regular_directory(directory, root=root, label=expected_kind)
        manifest_path = sealed_cpu._regular_file(
            bundle / "manifest.json", root=root, label=f"{expected_kind} manifest"
        )
        seal_path = sealed_cpu._regular_file(
            bundle / "seal.json", root=root, label=f"{expected_kind} seal"
        )
        manifest = sealed_cpu._read_json(manifest_path, root=root, label="bundle manifest")
        seal = sealed_cpu._read_json(seal_path, root=root, label="bundle seal")
    except sealed_cpu.PilotGateError as exc:
        raise CandidateScreenError(str(exc)) from exc
    if not isinstance(manifest, Mapping) or manifest.get("manifest_kind") != expected_kind \
            or manifest.get("schema_version") != SCHEMA_VERSION:
        raise CandidateScreenError(f"{expected_kind} manifest binding differs")
    if not isinstance(seal, Mapping) or seal != {
        "manifest_kind": expected_kind,
        "members": seal.get("members"),
        "schema_version": SCHEMA_VERSION,
    } or not isinstance(seal.get("members"), Mapping):
        raise CandidateScreenError(f"{expected_kind} seal schema differs")
    expected_members = {"manifest.json": _identity_without_path(manifest_path)}
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise CandidateScreenError(f"{expected_kind} file inventory is empty")
    if "gate.json" in files:
        expected_members["gate.json"] = files["gate.json"]
    if seal["members"] != expected_members:
        raise CandidateScreenError(f"{expected_kind} seal members differ")
    expected_names = sorted([*files, "manifest.json", "seal.json"])
    if sorted(path.name for path in bundle.iterdir()) != expected_names:
        raise CandidateScreenError(f"{expected_kind} directory inventory differs")
    for name, identity in files.items():
        path = bundle / name
        if not path.is_file() or path.is_symlink() or _identity_without_path(path) != identity:
            raise CandidateScreenError(f"{expected_kind} member changed: {name}")
    return {
        "directory": bundle,
        "manifest": dict(manifest),
        "manifest_sha256": _sha256(manifest_path),
        "seal_sha256": _sha256(seal_path),
    }


def _rows_for_factory(factory: Path) -> dict[str, Mapping[str, Any]]:
    from robo.tasks import pi05_tasks

    result = {}
    for row in pi05_tasks._load_objects(factory):
        slot = str(row.get("name"))
        if OBJECT_SLOT_RE.fullmatch(slot) is None or slot in result:
            raise CandidateScreenError("task-planning object roster is invalid")
        result[slot] = row
    return result


def _suite_for_plan(
    *, scene_id: str, scene_xml: Path, plan: Mapping[str, Any],
    rows_by_policy: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    selected = plan["selected"]
    table = selected.get("table") or {
        "cx": 0.0, "cy": 0.0, "hx": 0.10, "hy": 0.10, "top_z": 0.0,
        "inference": "empty_candidate_placeholder",
        "member_slots": [], "support_level_m": 0.0,
    }
    base_pos = selected.get("base_pos") or [0.0, 0.0, float(table["top_z"])]
    base_yaw = selected.get("base_yaw") or 0.0
    excluded = sorted({
        slot
        for policy in POLICIES
        for slot, row in rows_by_policy[policy].items()
        if float(row.get("drift", 0.0)) > 0.25
    })
    tasks = [row["task"] for row in selected["tasks"]]
    # Table-derived coordinates are already in world space.  Declare that
    # convention explicitly so the rig does not apply the robot base twice.
    from robo.rigs.pi05_rig import lookat_quat
    camera_position = [float(table["cx"]), float(table["cy"]) + 1.0, float(table["top_z"]) + 0.75]
    camera_target = [float(table["cx"]), float(table["cy"]), float(table["top_z"]) + 0.12]
    return {
        "exclude_objects": excluded,
        "ext_cam": {
            "mode": "world",
            "frame": "construction_world_table_camera",
            "pos": camera_position,
            "target": camera_target,
            "quat_wxyz": lookat_quat(camera_position, camera_target).tolist(),
            "fovy": 68.0,
        },
        "robot": {"base_pos": list(base_pos), "base_yaw": float(base_yaw)},
        "scene": scene_id,
        "scene_xml": str(scene_xml),
        "table": {
            key: table[key] for key in ("cx", "cy", "hx", "hy", "top_z", "thickness_m")
            if key in table
        },
        "tasks": tasks,
        "time_limit_s": 16.0,
    }


def _automatic_export_context(factories, *, scene_id, root):
    """Bind automatic carve expectations to real canonical paired materializations."""
    from robo.eval import e3_factory_materializer as materializer
    if set(factories) != set(POLICIES):
        raise CandidateScreenError("automatic export requires exactly A0/A4")
    manifests = {}; reports = {}
    for policy in POLICIES:
        reports[policy] = materializer.validate_materialized_factory(
            factories[policy], expected_scene_id=scene_id,
            expected_policy_id=policy, repository_root=root)
        manifest = json.loads((Path(factories[policy])/'materialization_manifest.json').read_text())
        if manifest.get('manifest_kind') != 'e4_automatic_construction_variant_materialization':
            raise CandidateScreenError("automatic candidate cannot consume a GT materialization")
        manifests[policy] = manifest
    a, b = (manifests[p] for p in POLICIES)
    if any(a[key] != b[key] for key in ('scene_id', 'e3_root', 'e3_freeze_id', 'e3_code_commit', 'source_scene', 'input_identities')) \
            or a['roster']['object_slots'] != b['roster']['object_slots']:
        raise CandidateScreenError("automatic paired source closure differs")
    return {'rosters': {p: manifests[p]['roster'] for p in POLICIES},
            'reports': reports, 'object_slots': a['roster']['object_slots'],
            'accepted_union': sorted(set().union(*(set(manifests[p]['roster']['accepted_slots']) for p in POLICIES))),
            'source_mesh_sha256': a['source_scene']['discovery_sources']['derived_mesh.ply']['sha256'],
            'source_scene': a['source_scene']}


def automatic_candidate_population(*, e3_root, scene_id, automatic_scene_descriptor):
    """Declare every semantic pair from discovery, before reading arm decisions.

    This roster is a denominator, not a robot task or qualification result.
    Construction, reach, support, cavity, reset and camera gates remain separate.
    """
    from robo.eval import e3_factory_materializer as materializer
    from robo.tasks.pi05_tasks import GRASP_LABELS, RECEPTACLE_LABELS
    e3_root = materializer.e3.checked_repo_path(e3_root, 'automatic candidate E3 source', kind='dir')
    jobs, _, audit, inventory = materializer._verify_inventory(e3_root)
    source = materializer._automatic_source_context(jobs, audit, scene_id, automatic_scene_descriptor)
    objects = {f"obj_{row['automatic_instance_id']}": row for row in source['object_rows']}
    targets = sorted(slot for slot, row in objects.items() if row['label'] in GRASP_LABELS)
    receptacles = sorted(slot for slot, row in objects.items() if row['label'] in RECEPTACLE_LABELS)
    pairs = []
    for target in targets:
        for destination in [None, *(slot for slot in receptacles if slot != target)]:
            family = 'object_to_region' if destination is None else 'object_to_receptacle'
            pairs.append({'candidate_id': f'{scene_id}/{target}/'+(destination or 'region'),
                          'task_family': family, 'target': target, 'receptacle': destination,
                          'target_label': objects[target]['label'],
                          'receptacle_label': objects[destination]['label'] if destination else None,
                          'qualification_state': 'NOT_RUN', 'qualified': None})
    return {'schema_version': 1, 'scope': 'automatic_candidate_population', 'paper_ready': False,
            'scene_id': scene_id, 'e3_root': str(Path(e3_root).resolve()),
            'e3_freeze_id': jobs['freeze_id'], 'e3_producer_commit': jobs['source_contract']['code_commit'],
            'inventory_identities': inventory, 'descriptor': source['descriptor'],
            'source_files': source['source_files'],
            'source_gaussian_training_provenance': source['source_gaussian_training_provenance'],
            'objects': objects, 'object_slots': source['object_slots'], 'pairs': pairs,
            'counts': {'planned_objects': len(objects), 'semantic_pairs': len(pairs),
                       'region_pairs': sum(p['receptacle'] is None for p in pairs),
                       'receptacle_pairs': sum(p['receptacle'] is not None for p in pairs)}}


def qualification_selection(protocol_path, population, *, root):
    """Authenticate complete discovery first, then apply the frozen CPU budget.

    The population remains complete; this receipt names measured and excluded
    task IDs without reading construction quality or policy outcomes.
    """
    import yaml
    from run.icra2027 import e4_full_protocol as full
    from robo.eval import e3_factory_materializer as materializer
    path=sealed_cpu._regular_file(Path(protocol_path),root=root,label='qualification protocol')
    protocol=yaml.safe_load(path.read_text())
    def source(ref):
        value=Path(ref['path']).absolute()
        if value.resolve()!=value or not value.is_file() or value.is_symlink():
            raise CandidateScreenError('protocol source is not a canonical regular file')
        if (set(ref)!={'path','bytes','mtime_ns','sha256'} or _sha256(value)!=ref['sha256']
                or value.stat().st_size!=ref['bytes'] or value.stat().st_mtime_ns!=ref['mtime_ns']):
            raise CandidateScreenError('protocol source identity differs')
        return value
    refs=protocol['input_references']
    if set(refs)!={'source_cohort','role_definition'}:raise CandidateScreenError('protocol source reference roster differs')
    cohort=yaml.safe_load(source(refs['source_cohort']).read_text())
    if _sha256(source(refs['role_definition']))!=_sha256(CODE_ROOT/'robo/tasks/pi05_tasks.py'):
        raise CandidateScreenError('task role vocabulary differs from preregistration')
    rebuilt=[]
    for row in protocol['source_populations']:
        if set(row)!={'scene_id','planned_objects','all_jobs_manifest','discovery_audit','queries'}:
            raise CandidateScreenError('protocol population row schema differs')
        report=json.loads(source(row['all_jobs_manifest']).read_text())
        audit=json.loads(source(row['discovery_audit']).read_text())
        if (report['scene_id']!=row['scene_id'] or report['planned_jobs']!=len(report['rows'])
                or report['planned_jobs']!=row['planned_objects']
                or audit['all_jobs_manifest_sha256']!=row['all_jobs_manifest']['sha256']
                or report['source_gaussian_training_provenance']!=full.original.canonical.FRESH):
            raise CandidateScreenError('protocol discovery population integrity differs')
        rebuilt.append({**row,'queries':full.original.semantic_pairs(row['scene_id'],report['rows'])})
    cohort_scenes = [full.original.canonical.e3._require_scene_id(value)
                     for value in cohort['population']['scene_ids']]
    if [r['scene_id'] for r in rebuilt] != cohort_scenes:
        raise CandidateScreenError('complete protocol scene roster differs')
    if protocol!=full.protocol_from_population(rebuilt,references=refs):
        raise CandidateScreenError('frozen qualification budget selection differs')
    jobs,_,inventory_audit,_=materializer._verify_inventory(Path(population['e3_root']))
    if (jobs['schema_version']!=2 or jobs['freeze_id']!=population['e3_freeze_id']
            or jobs['source_contract']['code_commit']!=population['e3_producer_commit']
            or [x['scene_id'] for x in jobs['scenes']]!=[x['scene_id'] for x in rebuilt]
            or [len(x['jobs']) for x in jobs['scenes']]!=[x['planned_objects'] for x in rebuilt]):
        raise CandidateScreenError('budget protocol requires the complete original E3 inventory')
    counts={'scenes':len(rebuilt),'jobs':sum(row['planned_objects'] for row in rebuilt),
            'policy_object_rows':5*sum(row['planned_objects'] for row in rebuilt)}
    if (jobs.get('counts')!=counts or inventory_audit.get('counts')!=counts
            or inventory_audit.get('scene_roster')!=[row['scene_id'] for row in rebuilt]
            or set(inventory_audit.get('scene_audits',{}))!={row['scene_id'] for row in rebuilt}):
        raise CandidateScreenError('complete E3 source audit differs')
    for row in rebuilt:
        discovery=inventory_audit['scene_audits'][row['scene_id']]['source_discovery']
        if (discovery.get('all_jobs_manifest.json')!=row['all_jobs_manifest']['sha256']
                or discovery.get('postrun_audit.json')!=row['discovery_audit']['sha256']):
            raise CandidateScreenError('E3 inventory binds different discovery inputs')
    matching=[row for row in rebuilt if row['scene_id']==population['scene_id']]
    if len(matching)!=1 or matching[0]['planned_objects']!=population['counts']['planned_objects']:
        raise CandidateScreenError('budget scene population differs')
    pairs=[dict(scene_id=population['scene_id'],task_id=f"{population['scene_id']}__{q['target']}_to_{q['receptacle'] or 'region'}",
                target=q['target'],receptacle=q['receptacle'],task_family=q['task_family']) for q in population['pairs']]
    if matching[0]['queries']!=pairs:raise CandidateScreenError('budget source semantic pair roster differs')
    chosen=sorted(q['task_id'] for q in protocol['qualification_tasks'] if q['scene_id']==population['scene_id'])
    all_ids=sorted(q['task_id'] for q in pairs)
    return dict(schema_version=1,scope='preregistered_full_discovery_qualification_budget',
        protocol=_identity(path,root=root),scene_id=population['scene_id'],
        input_semantic_queries=len(all_ids),selected_task_ids=chosen,
        excluded_task_ids=sorted(set(all_ids)-set(chosen)),
        qualification_queries=len(chosen),qualification_cells=10*len(chosen),
        full_input_semantic_queries=protocol['input_semantic_queries'],
        full_qualification_queries=protocol['planned_semantic_queries'],
        full_budget_exclusions=protocol['unselected_input_semantic_queries'],
        exclusion_reason='predeclared_qualification_budget',paper_ready=False)


def plan_automatic_population_tasks(population, *, qualification_task_ids=None):
    """Plan every source-declared pair before arm geometry or outcomes are read.

    Prepared discovery AABBs anchor the existing table/base heuristic. This is
    planning evidence only: no mass, stability or construction tier is invented.
    All declared pairs remain, including unavailable observations/endpoints.
    """
    from robo.tasks import pi05_tasks
    objects = population['objects']
    rows = {}
    for slot, item in objects.items():
        aabb = np.asarray(item['aabb'], dtype=float)
        if aabb.shape != (2, 3) or not np.isfinite(aabb).all() or np.any(aabb[1] <= aabb[0]):
            raise CandidateScreenError('automatic planning source AABB is invalid')
        rows[slot] = {'name': slot, 'label': item['label'], 'aabb': aabb,
                      'dims': aabb[1]-aabb[0], 'center': aabb.mean(axis=0),
                      'bottom_z': float(aabb[0,2]),
                      'prepared': item['prepared_output_index'] is not None}
    def planning_grasp(row):
        return (row['prepared'] and row['label'] in pi05_tasks.GRASP_LABELS
                and row['dims'].max() <= .28 and row['dims'].min() >= .005)
    prepared_rows = [row for row in rows.values() if row['prepared']]
    if not any(planning_grasp(row) for row in prepared_rows):
        raise CandidateScreenError('automatic task planning has no prepared size-admissible target; all declared pairs remain unqualified')
    try:
        level, members = pi05_tasks._pick_table(prepared_rows, planning_predicate=planning_grasp)
        placement = pi05_tasks._place_robot(members, planning_predicate=planning_grasp)
    except SystemExit as exc:
        raise CandidateScreenError(str(exc)) from exc
    placement['table']['thickness_m'] = .020
    tasks = []
    for pair in population['pairs']:
        target = rows[pair['target']]
        same = [row for row in rows.values() if row['label'] == target['label']]
        qualifier = pi05_tasks._qualifier(target, same, placement['base_pos'], placement['base_yaw'])
        name = f"the {qualifier} {target['label']}" if qualifier else f"the {target['label']}"
        task = {'task_id': f"{population['scene_id']}__{pair['target']}_to_"+(pair['receptacle'] or 'region'),
                'target': pair['target'], 'target_label': target['label'],
                'any_instance': len(same)>1 and not qualifier, 'receptacle': pair['receptacle'],
                'task_family': pair['task_family']}
        if pair['receptacle'] is not None:
            destination = rows[pair['receptacle']]
            task.update(receptacle_dims=destination['dims'].tolist(), instructions={
                'default': f"put {name} in the {destination['label']}",
                'vague': f"put {name} away",
                'specific': f"pick up {name} and place it inside the {destination['label']}"})
        else:
            table = placement['table']
            side = -1.0 if target['center'][1] > table['cy'] else 1.0
            word = 'left' if side*np.cos(placement['base_yaw']) >= 0 else 'right'
            task.update(region={'cx':table['cx'], 'cy':table['cy']+side*table['hy']/2,
                'hx':min(table['hx'],.30), 'hy':table['hy']/3,
                'zlo':table['top_z']-.02, 'zhi':table['top_z']+.30}, instructions={
                'default':f"move {name} to the {word} side of the table", 'vague':f"move {name}",
                'specific':f"pick up {name} and set it down on the {word} side of the table"})
        tasks.append({'candidate_id':pair['candidate_id'], 'task':task})
    budget={}
    if qualification_task_ids is not None:
        selected=list(qualification_task_ids);all_ids=[row['task']['task_id'] for row in tasks]
        if len(selected)!=len(set(selected)) or not set(selected)<=set(all_ids):
            raise CandidateScreenError('budget task IDs duplicate or outside semantic population')
        tasks=[row for row in tasks if row['task']['task_id'] in set(selected)]
        budget={'input_semantic_queries':len(all_ids),'budget_excluded_task_ids':sorted(set(all_ids)-set(selected))}
    return {'scope':'automatic_source_geometry_task_planning', 'paper_ready':False, **budget,
            'candidate_count':len(tasks), 'planned_pair_arm_rows':2*len(tasks),
            'placement_rule':'existing_pi05_prepared_source_aabb_size_only',
            'source_geometry_sha256':_canonical_hash(objects),
            'prepared_slots':sorted(row['name'] for row in prepared_rows),
            'support_level_m':level, 'member_slots':sorted(row['name'] for row in members),
            'selected':{**placement,'tasks':tasks}, 'construction_eligibility_checked':False}


def _automatic_suites_for_plan(*, scene_id, factories, plan):
    # Empty rows prevent the legacy outcome-based object exclusion heuristic.
    # No constructed object is removed from an automatic treatment scene.
    return {policy:_suite_for_plan(scene_id=scene_id,
        scene_xml=factories[policy]/'sim_export/scene.xml', plan=plan,
        rows_by_policy={p:{} for p in POLICIES}) for policy in POLICIES}


def prepare_automatic_candidates(*, screen_id, scene_id, e3_root,
                                 automatic_scene_descriptor, expected_commit, export=False, export_reuse=None,
                                 qualification_protocol=None):
    """Materialize the complete automatic population and retain every failed pair.

    ``export`` invokes the existing full-room collision producer. This phase
    never supplies fabricated robot placement or turns a semantic pair into a
    passed task; canonical workspace/camera/scorer qualification follows later.
    """
    from robo.eval import e3_factory_materializer as materializer
    screen_id = _validated_screen_id(screen_id)
    code = _code_snapshot(expected_commit); root = evidence_root()
    before = automatic_candidate_population(e3_root=e3_root, scene_id=scene_id,
                                           automatic_scene_descriptor=automatic_scene_descriptor)
    directory = sealed_cpu._inside(_experiment_root(root, screen_id)/'automatic_candidates'/scene_id,
                                   root=root, label='automatic candidate output')
    if directory.is_symlink() or (directory/'population').exists():
        raise FileExistsError(directory)
    declaration = directory/'declaration'
    selection=qualification_selection(qualification_protocol,before,root=root) if qualification_protocol is not None else None
    declared = {**before, 'code': code, 'screen_id': screen_id}
    if selection is not None:declared['qualification_selection']=selection
    if export_reuse is not None:
        declared['export_reuse'] = export_reuse
    if declaration.exists():
        _validate_bundle(declaration, root=root, expected_kind='e4_automatic_candidate_declaration')
        if _read_json_member(declaration, 'population.json') != declared:
            raise CandidateScreenError('automatic candidate declaration belongs to another frozen source')
    else:
        if directory.exists():
            raise CandidateScreenError('partial automatic output lacks its source declaration')
        _publish_bundle(declaration, manifest_kind='e4_automatic_candidate_declaration',
            payloads={'population.json': _json_bytes(declared)},
            manifest_fields={'code': code, 'scene_id': scene_id, 'screen_id': screen_id})
    # The sealed all-pairs denominator precedes every expensive or fallible
    # construction/export action; an interrupted phase cannot erase failures.
    factories = {policy: directory/'materialized'/policy for policy in POLICIES}
    for policy in POLICIES:
        if factories[policy].exists():
            materializer.validate_materialized_factory(factories[policy],
                expected_scene_id=scene_id, expected_policy_id=policy, repository_root=root)
            manifest = json.loads((factories[policy]/'materialization_manifest.json').read_text())
            if (materializer.e3.checked_repo_path(manifest['e3_root'], 'resumed E3 source', kind='dir')
                    != materializer.e3.checked_repo_path(e3_root, 'configured E3 source', kind='dir')
                    or manifest['e3_freeze_id'] != before['e3_freeze_id']
                    or manifest['e3_code_commit'] != before['e3_producer_commit']
                    or manifest['source_scene']['automatic_scene_descriptor'] != before['descriptor']):
                raise CandidateScreenError('partial automatic materialization belongs to another frozen source')
        else:
            materializer.materialize_factory_variant(e3_root=e3_root, scene_id=scene_id,
                policy_id=policy, out=factories[policy], automatic_scene_contract=automatic_scene_descriptor)
    context = _automatic_export_context(factories, scene_id=scene_id, root=root)
    if export_reuse is not None:
        from run.icra2027 import e4_automatic_export_reuse as reuse_api
        reuse_api.publish(directory/'materialized', export_reuse, code=code)
        reuse_api.resolve(factories['A0'], factories, scene_id=scene_id, root=root)
        reuse_api.copy_a0(directory/'materialized', export_reuse)
    exports = {}
    if export:
        for policy in POLICIES:
            if not (factories[policy]/'sim_export').exists() and not (factories[policy]/'sim').exists():
                _run_full_room_export(factories[policy], scene_id=scene_id, root=root,
                                     common_carve_factories=[factories[p] for p in POLICIES], automatic=True)
            exports[policy] = validate_full_room_export(factories[policy], scene_id=scene_id,
                policy=policy, root=root, expected_object_slots=context['rosters'][policy]['accepted_slots'],
                expected_discovered_slots=before['object_slots'], automatic_factories=factories)
        if exports['A0']['background_carve'] != exports['A4']['background_carve']:
            raise CandidateScreenError('automatic paired backgrounds differ')
        if exports['A0'].get('shared_static_package') != exports['A4'].get('shared_static_package'):
            raise CandidateScreenError('automatic paired static package identity differs')
        if exports['A0']['static_collision'] != exports['A4']['static_collision']:
            raise CandidateScreenError('automatic paired static collision bytes differ')
    pairs = []
    for original in before['pairs']:
        pair = dict(original); pair['construction_by_policy'] = {}
        required = [pair['target']] + ([pair['receptacle']] if pair['receptacle'] else [])
        for policy in POLICIES:
            roster = context['rosters'][policy]
            missing = [slot for slot in required if slot not in roster['accepted_slots']]
            pair['construction_by_policy'][policy] = {
                'state': 'NOT_RUN' if not missing else 'failed_build',
                'required_object_slots': required,
                'failed_objects': [{'object_slot': slot,
                    'terminal_action': 'reject' if slot in roster['rejected_slots'] else 'abstain'} for slot in missing],
                'workspace': None, 'camera': None, 'scorer': None, 'manipulation_success': None}
        pairs.append(pair)
    if automatic_candidate_population(e3_root=e3_root, scene_id=scene_id,
            automatic_scene_descriptor=automatic_scene_descriptor) != before:
        raise CandidateScreenError('automatic source changed during candidate preparation')
    after = _automatic_export_context(factories, scene_id=scene_id, root=root)
    if after != context:
        raise CandidateScreenError('automatic materialization changed during candidate preparation')
    gate = {**before, 'pairs': pairs, 'code': code, 'screen_id': screen_id,
            'materializations': context['reports'], 'exports': exports,
            'planned_pair_arm_rows': len(pairs)*len(POLICIES),
            'static_export_state': 'PASS' if export else 'NOT_RUN',
            'qualified_tasks': None, 'gpu_launch_allowed': False,
            'large_rollout_launch_allowed': False, 'headline_eligible': False}
    if selection is not None:
        if qualification_selection(qualification_protocol,before,root=root)!=selection:
            raise CandidateScreenError('qualification protocol changed during preparation')
        gate['qualification_selection']=selection
    _publish_bundle(directory/'population', manifest_kind='e4_automatic_candidate_population',
        payloads={'gate.json': _json_bytes(gate), 'pairs.json': _json_bytes(pairs)},
        manifest_fields={'code': code, 'scene_id': scene_id, 'screen_id': screen_id,
                         'planned_pair_arm_rows': len(pairs)*len(POLICIES)})
    return gate


def _automatic_prepare_inputs(*, root, screen_id, scene_id, expected_commit):
    from robo.eval import e4_task_freeze as task_freeze
    directory = _experiment_root(root, screen_id)/'automatic_candidates'/scene_id
    bundle = _validate_bundle(directory/'population', root=root,
                              expected_kind='e4_automatic_candidate_population')
    population = _read_json_member(bundle['directory'], 'gate.json')
    expected_code = {'code_root':str(CODE_ROOT), 'commit':expected_commit, 'dirty':False}
    if population.get('code') != expected_code or population.get('screen_id') != screen_id:
        raise CandidateScreenError('automatic source/config freeze differs')
    factories = {p:directory/'materialized'/p for p in POLICIES}
    reports = {p:task_freeze._factory_report(factories[p], scene_id=scene_id,
                policy=p, root=root) for p in POLICIES}
    automatic_tasks=task_freeze._automatic_population_tasks(directory/'population/gate.json', root=root,
        scene_id=scene_id, factories=factories, reports=reports)
    footprints = {p:snapshot_export_footprints(factories[p],
                    expected_slots=reports[p]['roster']['accepted_slots']) for p in POLICIES}
    rows = {p:_rows_for_factory(factories[p]) for p in POLICIES}
    _validate_export_drift_binding(rows_by_policy=rows, footprint_replays=footprints)
    plan = (plan_automatic_population_tasks(population,qualification_task_ids=automatic_tasks)
            if 'qualification_selection' in population else plan_automatic_population_tasks(population))
    return {'population':population, 'population_bundle':bundle, 'factories':factories,
            'materializations':reports, 'footprint_replays':footprints, 'plan':plan,
            'suites':_automatic_suites_for_plan(scene_id=scene_id, factories=factories, plan=plan)}


def prepare_automatic_task_suites(*, screen_id, scene_id, expected_commit):
    """Bridge authenticated automatic exports into the canonical CPU qualifier."""
    from robo.eval.e4_task_freeze import freeze_task_bundle, validate_task_bundle
    screen_id = _validated_screen_id(screen_id)
    code = _code_snapshot(expected_commit); root = evidence_root()
    inputs = _automatic_prepare_inputs(root=root, screen_id=screen_id,
        scene_id=scene_id, expected_commit=expected_commit)
    experiment = _experiment_root(root, screen_id)
    candidate_dir = experiment/'candidate_suites'/scene_id
    if (experiment/'scene_prepares'/scene_id).exists():
        raise CandidateScreenError('refusing to overwrite completed automatic prepare')
    payloads={**{f'{p.lower()}_candidates.json':_json_bytes(inputs['suites'][p]) for p in POLICIES},
              'planning.json':_json_bytes(inputs['plan'])}
    fields={'code':code,'scene_id':scene_id,'screen_id':screen_id}
    if candidate_dir.exists():
        candidate=_validate_bundle(candidate_dir,root=root,expected_kind='e4_candidate_task_suites')
        if (any(candidate['manifest'].get(k)!=v for k,v in fields.items())
                or any((candidate_dir/name).read_bytes()!=data for name,data in payloads.items())):
            raise CandidateScreenError('partial candidate bundle differs from frozen source planning')
    else:
        candidate = _publish_bundle(candidate_dir, manifest_kind='e4_candidate_task_suites',
            payloads=payloads,manifest_fields=fields)
    task_dir = experiment/'task_freezes'/scene_id
    population_path = inputs['population_bundle']['directory']/'gate.json'
    if not task_dir.exists():
        freeze_task_bundle(scene_id=scene_id, a0_factory=inputs['factories']['A0'],
            a4_factory=inputs['factories']['A4'], a0_candidates=candidate_dir/'a0_candidates.json',
            a4_candidates=candidate_dir/'a4_candidates.json', planning_source=PLANNING_SOURCE,
            out=task_dir, repository_root=root, automatic_population=population_path)
    task_bundle = validate_task_bundle(task_dir/'manifest.json', expected_scene_id=scene_id,
                                      repository_root=root)
    task_manifest=json.loads((task_dir/'manifest.json').read_text())
    if (task_manifest['candidate_suites']!={p:_identity(candidate_dir/f'{p.lower()}_candidates.json',root=root) for p in POLICIES}
            or task_bundle.get('automatic_population')!=_identity(population_path,root=root)
            or task_bundle['planning_source']!=PLANNING_SOURCE or task_bundle['max_tasks'] is not None):
        raise CandidateScreenError('partial task freeze differs from current candidate/source binding')
    gate = {'manifest_kind':'e4_candidate_scene_prepare_gate', 'code':code,
        'created_utc':_utc_now(), 'evidence_root':str(root), 'scene_id':scene_id,
        'screen_id':screen_id, 'study_scope':STUDY_SCOPE,
        'automatic_population':_identity(population_path,root=root),
        'candidate_count':inputs['plan']['candidate_count'],
        'planned_pair_arm_rows':inputs['plan']['planned_pair_arm_rows'],
        'candidate_suite_manifest_sha256':candidate['manifest_sha256'],
        'planning_sha256':_canonical_hash(inputs['plan']),
        'task_bundle_manifest_sha256':task_bundle['bundle_manifest_sha256'],
        'materializations':inputs['materializations'], 'exports':inputs['population']['exports'],
        'footprint_replays':inputs['footprint_replays'],
        'gpu_launch_allowed':False, 'headline_eligible':False,
        'large_rollout_launch_allowed':False, 'paper_ready':False}
    after = _automatic_prepare_inputs(root=root, screen_id=screen_id,
        scene_id=scene_id, expected_commit=expected_commit)
    if not _same_replay_structure(inputs, after):
        raise CandidateScreenError('automatic construction closure changed during task freeze')
    published = _publish_bundle(experiment/'scene_prepares'/scene_id,
        manifest_kind='e4_candidate_scene_prepare_artifacts', payloads={'gate.json':_json_bytes(gate)},
        manifest_fields={'code':code,'scene_id':scene_id,'screen_id':screen_id})
    return {**gate,'prepare_manifest_sha256':published['manifest_sha256'],
            'prepare_seal_sha256':published['seal_sha256']}


def _load_automatic_prepare(*, root, screen_id, scene_id, expected_commit,
                            prepared, gate, candidate, plan, suites):
    from robo.eval.e4_task_freeze import validate_task_bundle
    inputs = _automatic_prepare_inputs(root=root, screen_id=screen_id,
        scene_id=scene_id, expected_commit=expected_commit)
    if gate['automatic_population'] != _identity(inputs['population_bundle']['directory']/'gate.json',root=root):
        raise CandidateScreenError('automatic prepare population identity changed')
    if plan != inputs['plan'] or suites != inputs['suites']:
        raise CandidateScreenError('automatic task planning differs from frozen discovery geometry')
    for key,value in {'materializations':inputs['materializations'],
                      'exports':inputs['population']['exports'],
                      'footprint_replays':inputs['footprint_replays']}.items():
        if not _same_replay_structure(gate.get(key),value):
            raise CandidateScreenError(f'automatic prepared {key} changed')
    task_bundle = validate_task_bundle(_experiment_root(root,screen_id)/'task_freezes'/scene_id/'manifest.json',
        expected_scene_id=scene_id, repository_root=root)
    if task_bundle['bundle_manifest_sha256'] != gate['task_bundle_manifest_sha256'] \
            or len(task_bundle['logical_task_ids']) != gate['candidate_count']:
        raise CandidateScreenError('automatic paired task bundle changed after prepare')
    # The frozen task freezer must have copied these exact shared definitions.
    for policy in POLICIES:
        actual = json.loads(Path(task_bundle['variant_tasks'][policy]).read_text())
        expected = dict(suites[policy])
        expected['tasks'] = sorted(expected['tasks'],key=lambda row:row['task_id'])
        if actual != expected:
            raise CandidateScreenError('automatic frozen suite differs from source-derived plan')
    return {'candidate_bundle':candidate,'factories':inputs['factories'],'gate':dict(gate),
            'plan':dict(plan),'prepare_bundle':prepared,'suites':suites,'task_bundle':task_bundle}


def prepare_scene(*, screen_id: str, scene_id: str, expected_commit: str) -> dict[str, Any]:
    """Export both arms and publish a sealed 0/1/2+ paired candidate plan."""
    screen_id = _validated_screen_id(screen_id)
    scene_id = _validated_scene_id(scene_id)
    code = _code_snapshot(expected_commit)
    root = evidence_root()
    experiment = _experiment_root(root, screen_id)
    from robo.eval.e3_factory_materializer import validate_materialized_factory

    factories = {
        policy: _factory_dir(root, screen_id, policy, scene_id) for policy in POLICIES
    }
    materializations = {}
    for policy, factory in factories.items():
        report = validate_materialized_factory(
            factory,
            expected_scene_id=scene_id,
            expected_policy_id=policy,
            repository_root=root,
        )
        if report.get("validator_commit") != expected_commit:
            raise CandidateScreenError("materialization validator commit differs")
        for generated in (factory / "sim", factory / "sim_export"):
            if generated.exists() or generated.is_symlink():
                raise CandidateScreenError(f"refusing non-fresh export: {generated}")
        materializations[policy] = _materialization_summary(report)

    external_before = external_geometry_identities(scene_id)
    exports = {}
    footprint_replays = {}
    rows_by_policy = {}
    footprints = {}
    for policy in POLICIES:
        factory = factories[policy]
        _run_full_room_export(
            factory, scene_id=scene_id, root=root,
            common_carve_factories=[factories[item] for item in POLICIES],
        )
        accepted = materializations[policy]["roster"]["accepted_slots"]
        exports[policy] = validate_full_room_export(
            factory,
            scene_id=scene_id,
            policy=policy,
            root=root,
            expected_object_slots=accepted,
            expected_discovered_slots=materializations[policy]["roster"]["object_slots"],
        )
        footprint_replays[policy] = snapshot_export_footprints(
            factory, expected_slots=accepted
        )
        footprints[policy] = footprint_replays[policy]["footprints"]
        rows_by_policy[policy] = _rows_for_factory(factory)
        if set(rows_by_policy[policy]) != set(accepted):
            raise CandidateScreenError(f"{policy} planning rows differ from accepted roster")
    common_carves = [exports[policy]["background_carve"] for policy in POLICIES]
    if common_carves[0] != common_carves[1]:
        raise CandidateScreenError("A0/A4 full-room backgrounds are not identical")
    _validate_export_drift_binding(
        rows_by_policy=rows_by_policy,
        footprint_replays=footprint_replays,
    )
    if external_geometry_identities(scene_id) != external_before:
        raise CandidateScreenError("external geometry changed during prepare")

    plan = plan_paired_region_tasks(
        scene_id=scene_id,
        rows_by_policy=rows_by_policy,
        footprints=footprints,
    )
    candidate_dir = experiment / "candidate_suites" / scene_id
    suites = {
        policy: _suite_for_plan(
            scene_id=scene_id,
            scene_xml=factories[policy] / "sim_export" / "scene.xml",
            plan=plan,
            rows_by_policy=rows_by_policy,
        )
        for policy in POLICIES
    }
    candidate_bundle = _publish_bundle(
        candidate_dir,
        manifest_kind="e4_candidate_task_suites",
        payloads={
            "a0_candidates.json": _json_bytes(suites["A0"]),
            "a4_candidates.json": _json_bytes(suites["A4"]),
            "planning.json": _json_bytes(plan),
        },
        manifest_fields={
            "candidate_count": plan["candidate_count"],
            "code": code,
            "created_utc": _utc_now(),
            "evidence_root": str(root),
            "scene_id": scene_id,
            "screen_id": screen_id,
        },
    )

    task_bundle = None
    if plan["candidate_count"]:
        from robo.eval.e4_task_freeze import freeze_task_bundle, validate_task_bundle

        task_dir = experiment / "task_freezes" / scene_id
        freeze_task_bundle(
            scene_id=scene_id,
            a0_factory=factories["A0"],
            a4_factory=factories["A4"],
            a0_candidates=candidate_dir / "a0_candidates.json",
            a4_candidates=candidate_dir / "a4_candidates.json",
            planning_source=PLANNING_SOURCE,
            max_tasks=None,
            out=task_dir,
            repository_root=root,
        )
        task_bundle = validate_task_bundle(
            task_dir / "manifest.json",
            expected_scene_id=scene_id,
            repository_root=root,
        )
        if len(task_bundle["logical_task_ids"]) != plan["candidate_count"]:
            raise CandidateScreenError("task-freeze candidate count differs")

    gate = {
        "candidate_count": plan["candidate_count"],
        "candidate_suite_manifest_sha256": candidate_bundle["manifest_sha256"],
        "code": code,
        "created_utc": _utc_now(),
        "evidence_root": str(root),
        "exports": exports,
        "external_geometry_inputs": external_before,
        "footprint_replays": footprint_replays,
        "gpu_launch_allowed": False,
        "headline_eligible": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_candidate_scene_prepare_gate",
        "materializations": materializations,
        "paper_ready": False,
        "planning_sha256": _canonical_hash(plan),
        "scene_id": scene_id,
        "screen_id": screen_id,
        "study_scope": STUDY_SCOPE,
        "task_bundle_manifest_sha256": (
            task_bundle["bundle_manifest_sha256"] if task_bundle else None
        ),
    }
    prepare_bundle = _publish_bundle(
        experiment / "scene_prepares" / scene_id,
        manifest_kind="e4_candidate_scene_prepare_artifacts",
        payloads={"gate.json": _json_bytes(gate)},
        manifest_fields={
            "candidate_suite_manifest_sha256": candidate_bundle["manifest_sha256"],
            "code": code,
            "created_utc": _utc_now(),
            "scene_id": scene_id,
            "screen_id": screen_id,
        },
    )
    return {
        **gate,
        "prepare_manifest_sha256": prepare_bundle["manifest_sha256"],
        "prepare_seal_sha256": prepare_bundle["seal_sha256"],
    }


def _read_json_member(bundle: Path, name: str) -> Any:
    path = bundle / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateScreenError(f"invalid JSON bundle member {path}: {exc}") from exc


def _load_prepare(
    *, root: Path, screen_id: str, scene_id: str, expected_commit: str,
) -> dict[str, Any]:
    experiment = _experiment_root(root, screen_id)
    prepared = _validate_bundle(
        experiment / "scene_prepares" / scene_id,
        root=root,
        expected_kind="e4_candidate_scene_prepare_artifacts",
    )
    gate = _read_json_member(prepared["directory"], "gate.json")
    if not isinstance(gate, Mapping) or gate.get("manifest_kind") != "e4_candidate_scene_prepare_gate" \
            or gate.get("scene_id") != scene_id or gate.get("screen_id") != screen_id \
            or gate.get("study_scope") != STUDY_SCOPE:
        raise CandidateScreenError("prepare gate binding differs")
    if gate.get("code") != {
        "code_root": str(CODE_ROOT), "commit": expected_commit, "dirty": False
    }:
        raise CandidateScreenError("prepare gate code binding differs")
    for flag in (
        "gpu_launch_allowed", "headline_eligible", "large_rollout_launch_allowed", "paper_ready"
    ):
        if gate.get(flag) is not False:
            raise CandidateScreenError(f"prepare gate illegally authorizes {flag}")

    candidate = _validate_bundle(
        experiment / "candidate_suites" / scene_id,
        root=root,
        expected_kind="e4_candidate_task_suites",
    )
    if candidate["manifest_sha256"] != gate.get("candidate_suite_manifest_sha256"):
        raise CandidateScreenError("candidate-suite manifest changed after prepare")
    plan = _read_json_member(candidate["directory"], "planning.json")
    suites = {
        "A0": _read_json_member(candidate["directory"], "a0_candidates.json"),
        "A4": _read_json_member(candidate["directory"], "a4_candidates.json"),
    }
    if not isinstance(plan, Mapping) or _canonical_hash(plan) != gate.get("planning_sha256"):
        raise CandidateScreenError("candidate planning bytes differ from prepare")
    candidate_count = gate.get("candidate_count")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) \
            or candidate_count < 0 or plan.get("candidate_count") != candidate_count:
        raise CandidateScreenError("prepare candidate count is invalid")
    for policy in POLICIES:
        suite = suites[policy]
        if not isinstance(suite, Mapping) or suite.get("scene") != scene_id \
                or not isinstance(suite.get("tasks"), list) \
                or len(suite["tasks"]) != candidate_count:
            raise CandidateScreenError(f"{policy} candidate suite binding differs")
    shared_a0 = {key: value for key, value in suites["A0"].items() if key != "scene_xml"}
    shared_a4 = {key: value for key, value in suites["A4"].items() if key != "scene_xml"}
    if shared_a0 != shared_a4:
        raise CandidateScreenError("candidate suites differ outside scene XML")

    if 'automatic_population' in gate:
        return _load_automatic_prepare(root=root, screen_id=screen_id, scene_id=scene_id,
            expected_commit=expected_commit, prepared=prepared, gate=gate,
            candidate=candidate, plan=plan, suites=suites)

    from robo.eval.e3_factory_materializer import validate_materialized_factory

    factories = {policy: _factory_dir(root, screen_id, policy, scene_id) for policy in POLICIES}
    current_materializations = {}
    current_exports = {}
    current_footprints = {}
    current_rows = {}
    for policy in POLICIES:
        materialization = validate_materialized_factory(
            factories[policy],
            expected_scene_id=scene_id,
            expected_policy_id=policy,
            repository_root=root,
        )
        current_materializations[policy] = _materialization_summary(materialization)
        if current_materializations[policy] != gate.get("materializations", {}).get(policy):
            raise CandidateScreenError(f"{policy} materialization changed after prepare")
        accepted = current_materializations[policy]["roster"]["accepted_slots"]
        current_exports[policy] = validate_full_room_export(
            factories[policy],
            scene_id=scene_id,
            policy=policy,
            root=root,
            expected_object_slots=accepted,
            expected_discovered_slots=current_materializations[policy]["roster"][
                "object_slots"
            ],
        )
        if current_exports[policy] != gate.get("exports", {}).get(policy):
            raise CandidateScreenError(f"{policy} export changed after prepare")
        current_footprints[policy] = snapshot_export_footprints(
            factories[policy], expected_slots=accepted
        )
        if not _same_replay_structure(
            current_footprints[policy], gate.get("footprint_replays", {}).get(policy)
        ):
            raise CandidateScreenError(f"{policy} footprint replay changed after prepare")
        current_rows[policy] = _rows_for_factory(factories[policy])
        if set(current_rows[policy]) != set(accepted):
            raise CandidateScreenError(f"{policy} planning rows changed after prepare")
        declared_xml = Path(str(suites[policy].get("scene_xml")))
        if declared_xml != factories[policy] / "sim_export" / "scene.xml":
            raise CandidateScreenError(f"{policy} candidate scene XML differs")
    _validate_export_drift_binding(
        rows_by_policy=current_rows,
        footprint_replays=current_footprints,
    )
    if gate.get("external_geometry_inputs") != external_geometry_identities(scene_id):
        raise CandidateScreenError("external geometry changed after prepare")

    task_bundle = None
    if candidate_count:
        from robo.eval.e4_task_freeze import validate_task_bundle

        task_bundle = validate_task_bundle(
            experiment / "task_freezes" / scene_id / "manifest.json",
            expected_scene_id=scene_id,
            repository_root=root,
        )
        if task_bundle.get("bundle_manifest_sha256") != gate.get("task_bundle_manifest_sha256") \
                or len(task_bundle.get("logical_task_ids", [])) != candidate_count:
            raise CandidateScreenError("paired task bundle changed after prepare")
    elif gate.get("task_bundle_manifest_sha256") is not None:
        raise CandidateScreenError("zero-candidate prepare unexpectedly binds a task bundle")
    return {
        "candidate_bundle": candidate,
        "factories": factories,
        "gate": dict(gate),
        "plan": dict(plan),
        "prepare_bundle": prepared,
        "suites": suites,
        "task_bundle": task_bundle,
    }


def _runtime_footprints(env: Any, task_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_slot = {str(row["name"]): row for row in task_rows}
    result = {}
    for slot in sorted(env.free_bodies):
        if slot not in by_slot:
            raise CandidateScreenError(f"runtime body lacks task row: {slot}")
        position, quaternion = env.body_pose(slot)
        dimensions = np.asarray(by_slot[slot]["dims"], dtype=float)
        result[slot] = _world_xy_rectangle(
            np.asarray(position, dtype=float),
            np.asarray(quaternion, dtype=float),
            dimensions,
        )
    return result


def _runtime_base_clearance(
    base_xy: Sequence[float], current: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    rows = []
    for slot, rectangle in sorted(current.items()):
        clearance = _signed_point_rectangle_distance(base_xy, rectangle) - ROBOT_BASE_RADIUS_M
        rows.append({"clearance_m": clearance, "slot": slot})
    minimum = min((row["clearance_m"] for row in rows), default=None)
    blockers = [row for row in rows if row["clearance_m"] < MIN_BASE_BODY_CLEARANCE_M]
    return {
        "blocking_footprints": blockers,
        "minimum_clearance_m": minimum,
        "passed": bool(rows) and not blockers,
        "required_clearance_m": MIN_BASE_BODY_CLEARANCE_M,
        "robot_base_radius_m": ROBOT_BASE_RADIUS_M,
    }


def _workspace_from_footprints(
    *, current: Mapping[str, Mapping[str, Any]],
    task: Mapping[str, Any], suite: Mapping[str, Any], goal_probe=None,
) -> dict[str, Any]:
    target = str(task["target"])
    from types import SimpleNamespace
    poses = SimpleNamespace(body_pose=lambda name: (
        current[name]["position_world_m"], current[name]["quaternion_wxyz"]))
    destination = task_destination(task, poses)
    region = destination["destination_region"]
    receptacle = destination["destination_body"]
    if target not in current:
        raise CandidateScreenError("candidate target is absent from runtime")
    base_xy = suite["robot"]["base_pos"][:2]
    yaw = float(suite["robot"]["base_yaw"])
    target_center = current[target]["center_xy_m"]
    target_reach = reach_envelope(target_center, base_xy, yaw)
    destination_reach = reach_envelope([region["cx"], region["cy"]], base_xy, yaw)
    target_half_xy = np.asarray(current[target]["half_extents_xy_m"], dtype=float)
    if target_half_xy.shape != (2,) or not np.isfinite(target_half_xy).all() \
            or np.any(target_half_xy <= 0.0):
        raise CandidateScreenError("runtime target XY half extents are invalid")
    core_hx = float(region["hx"]) - float(target_half_xy[0]) - TARGET_GOAL_PADDING_M
    core_hy = float(region["hy"]) - float(target_half_xy[1]) - TARGET_GOAL_PADDING_M
    samples = []
    if core_hx > 0.0 and core_hy > 0.0:
        for x in (region["cx"] - core_hx, region["cx"], region["cx"] + core_hx):
            for y in (region["cy"] - core_hy, region["cy"], region["cy"] + core_hy):
                samples.append(reach_envelope([x, y], base_xy, yaw))
    safe_count = sum(row["safe"] for row in samples)
    region_rect = _rectangle([region["cx"], region["cy"]], [region["hx"], region["hy"]])
    obstacle_rows = [
        {"clearance_m": _signed_rectangle_distance(region_rect, rectangle), "slot": slot}
        for slot, rectangle in sorted(current.items()) if slot not in {target, receptacle}
    ]
    obstacle_min = min((row["clearance_m"] for row in obstacle_rows), default=None)
    obstacle_blockers = [
        row for row in obstacle_rows if row["clearance_m"] < MIN_OBSTACLE_CLEARANCE_M
    ]
    target_goal_clearance = _signed_rectangle_distance(region_rect, current[target])
    displacement = float(np.linalg.norm(
        np.asarray([region["cx"], region["cy"]], dtype=float)
        - np.asarray(target_center, dtype=float)
    ))
    base_clear = _runtime_base_clearance(base_xy, current)
    checks = {
        "base_clear_of_all_accepted_bodies": base_clear["passed"],
        "destination_center_has_reach_margin": destination_reach["safe"],
        "eroded_goal_core_nonempty": core_hx > 0.0 and core_hy > 0.0,
        "eroded_goal_core_has_5_safe_samples": safe_count >= 5,
        "goal_clear_of_all_non_target_bodies": not obstacle_blockers,
        "target_center_displacement_at_least_200mm": displacement >= MIN_TASK_DISPLACEMENT_M,
        "target_footprint_clear_of_goal": target_goal_clearance >= MIN_OBSTACLE_CLEARANCE_M,
        "target_has_reach_margin": target_reach["safe"],
    }
    if receptacle is not None:
        # A scorer bounding region is not evidence of a physically usable cavity.
        verified = False if goal_probe is None else _validate_goal_probe(
            goal_probe, task=task, current=current, destination=destination)
        checks["receptacle_goal_collision_verified"] = verified
        if goal_probe is not None:
            destination = {**destination, "goal_collision_status": "PASS" if verified else "FAIL",
                           "cavity_verified": verified, "goal_probe": dict(goal_probe)}
        # Preserve the existing pi05 task producer's strict maximum separation.
        checks["target_receptacle_separation_below_700mm"] = displacement < 0.70
    return {
        **({"task_destination": destination} if receptacle is not None else {}),
        "base_clearance": base_clear,
        "checks": checks,
        "destination_center_reach": destination_reach,
        "eroded_core_half_extents_m": [core_hx, core_hy],
        "eroded_core_safe_count": safe_count,
        "eroded_core_samples": samples,
        "minimum_center_displacement_m": displacement,
        "obstacle_clearance": {
            "blocking_footprints": obstacle_blockers,
            "minimum_clearance_m": obstacle_min,
            "required_clearance_m": MIN_OBSTACLE_CLEARANCE_M,
        },
        "passed": all(checks.values()),
        "runtime_footprints": dict(current),
        "target_goal_clearance_m": target_goal_clearance,
        "target_reach": target_reach,
    }


def _runtime_workspace(
    *, env: Any, task: Mapping[str, Any], suite: Mapping[str, Any],
    task_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _workspace_from_footprints(
        current=_runtime_footprints(env, task_rows),
        task=task,
        suite=suite,
        goal_probe=receptacle_goal_probe(env, task) if task.get("receptacle") is not None else None,
    )


def _runtime_stability(
    *, env: Any, task: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, Any]:
    nominal = provenance.get("pre_jitter_nominal_state", {}).get("object_states")
    settled = provenance.get("post_settle_pre_policy_state", {}).get("object_states")
    jitter = provenance.get("jitter")
    if not isinstance(nominal, Mapping) or not isinstance(settled, Mapping) \
            or not isinstance(jitter, Mapping) or set(nominal) != set(settled):
        raise CandidateScreenError("reset provenance object-state schema differs")
    target = str(task["target"])
    offset = np.asarray(jitter.get("offset_xy_m"), dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise CandidateScreenError("reset jitter offset is invalid")
    drift_rows = []
    speed_rows = []
    for slot in sorted(nominal):
        start = np.asarray(nominal[slot]["world_position_m"], dtype=float)
        expected = start.copy()
        if slot == target:
            expected[:2] += offset
        end = np.asarray(settled[slot]["world_position_m"], dtype=float)
        drift_rows.append({
            "drift_m": float(np.linalg.norm(end - expected)),
            "expected_post_jitter_position_m": expected.tolist(),
            "observed_post_settle_position_m": end.tolist(),
            "slot": slot,
        })
        linear_velocity = np.asarray(env.body_vel(slot)[3:], dtype=float)
        speed_rows.append({
            "linear_speed_m_s": float(np.linalg.norm(linear_velocity)),
            "linear_velocity_xyz_m_s": linear_velocity.tolist(),
            "slot": slot,
        })
    target_drift = next(row["drift_m"] for row in drift_rows if row["slot"] == target)
    max_drift = max((row["drift_m"] for row in drift_rows), default=0.0)
    max_speed = max((row["linear_speed_m_s"] for row in speed_rows), default=0.0)
    checks = {
        "all_body_settle_drift_below_30mm": max_drift < STABILITY_DRIFT_LIMIT_M,
        "all_body_linear_speed_below_50mm_s": max_speed < AT_REST_LINEAR_SPEED_M_S,
        "target_settle_drift_below_30mm": target_drift < STABILITY_DRIFT_LIMIT_M,
    }
    return {
        "checks": checks,
        "drift_rows": drift_rows,
        "maximum_room_drift_m": max_drift,
        "maximum_room_linear_speed_m_s": max_speed,
        "passed": all(checks.values()),
        "speed_rows": speed_rows,
        "target_drift_m": target_drift,
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        for row in rows
    )


def _prepared_task_contracts(prepared: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    gate = prepared.get("gate")
    suites = prepared.get("suites")
    factories = prepared.get("factories")
    if not isinstance(gate, Mapping) or not isinstance(suites, Mapping) \
            or not isinstance(factories, Mapping):
        raise CandidateScreenError("prepared task contract is absent")
    candidate_count = gate.get("candidate_count")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) \
            or candidate_count < 0:
        raise CandidateScreenError("prepared candidate count is invalid")
    reference_by_id: dict[str, Mapping[str, Any]] | None = None
    for policy in POLICIES:
        suite = suites.get(policy)
        tasks = suite.get("tasks") if isinstance(suite, Mapping) else None
        if not isinstance(tasks, list) or len(tasks) != candidate_count:
            raise CandidateScreenError(f"{policy} prepared task roster differs")
        rows_by_id: dict[str, Mapping[str, Any]] = {}
        for task in tasks:
            task_id = task.get("task_id") if isinstance(task, Mapping) else None
            target = task.get("target") if isinstance(task, Mapping) else None
            if not isinstance(task_id, str) or not task_id \
                    or not isinstance(target, str) \
                    or OBJECT_SLOT_RE.fullmatch(target) is None:
                raise CandidateScreenError(f"{policy} prepared task identity is invalid")
            if task_id in rows_by_id:
                raise CandidateScreenError(f"{policy} prepared task IDs are not unique")
            rows_by_id[task_id] = dict(task)
        if reference_by_id is None:
            reference_by_id = rows_by_id
        elif rows_by_id != reference_by_id:
            raise CandidateScreenError("prepared A0/A4 task identities differ")
    reference_by_id = reference_by_id or {}
    task_bundle = prepared.get("task_bundle")
    if candidate_count:
        logical_ids = task_bundle.get("logical_task_ids") \
            if isinstance(task_bundle, Mapping) else None
        if not isinstance(logical_ids, list) \
                or any(not isinstance(task_id, str) or not task_id for task_id in logical_ids) \
                or len(logical_ids) != len(set(logical_ids)) \
                or set(logical_ids) != set(reference_by_id):
            raise CandidateScreenError("prepared task bundle ID roster differs")
        reference = [reference_by_id[task_id] for task_id in logical_ids]
    elif task_bundle is not None:
        raise CandidateScreenError("zero-candidate prepare has a task bundle")
    else:
        reference = []
    from robo.eval import e4_camera_scorer_gate as camera_gate
    from robo.tasks import pi05_tasks

    rows_by_policy = {}
    excluded_by_policy = {}
    runtime_slots_by_policy = {}
    runtime_dimensions_by_policy = {}
    nominal_positions_by_policy = {}
    for policy in POLICIES:
        factory = factories.get(policy)
        if not isinstance(factory, Path):
            raise CandidateScreenError(f"{policy} prepared factory binding differs")
        rows_by_policy[policy] = _rows_for_factory(factory)
        excluded = suites[policy].get("exclude_objects")
        if not isinstance(excluded, list) \
                or any(not isinstance(slot, str) for slot in excluded) \
                or len(excluded) != len(set(excluded)):
            raise CandidateScreenError(f"{policy} prepared exclusion roster differs")
        excluded_by_policy[policy] = set(excluded)
        runtime_slots = sorted(set(rows_by_policy[policy]) - excluded_by_policy[policy])
        runtime_slots_by_policy[policy] = runtime_slots
        runtime_dimensions_by_policy[policy] = {}
        nominal_positions_by_policy[policy] = {}
        replay = gate.get("footprint_replays", {}).get(policy)
        replayed_footprints = replay.get("footprints") if isinstance(replay, Mapping) else None
        if not isinstance(replayed_footprints, Mapping) \
                or set(replayed_footprints) != set(rows_by_policy[policy]):
            raise CandidateScreenError(f"{policy} prepared footprint roster differs")
        robot = suites[policy].get("robot")
        if not isinstance(robot, Mapping):
            raise CandidateScreenError(f"{policy} prepared robot contract differs")
        for slot in runtime_slots:
            dimensions = np.asarray(rows_by_policy[policy][slot].get("dims"), dtype=float)
            nominal = replayed_footprints[slot].get("nominal") \
                if isinstance(replayed_footprints[slot], Mapping) else None
            position = nominal.get("position_world_m") if isinstance(nominal, Mapping) else None
            position_array = np.asarray(position, dtype=float)
            if dimensions.shape != (3,) or not np.isfinite(dimensions).all() \
                    or np.any(dimensions <= 0.0) \
                    or position_array.shape != (3,) \
                    or not np.isfinite(position_array).all():
                raise CandidateScreenError(f"{policy}/{slot} prepared geometry differs")
            runtime_dimensions_by_policy[policy][slot] = dimensions.tolist()
            nominal_positions_by_policy[policy][slot] = position_array.tolist()

    contracts = {}
    for task in reference:
        task_id = str(task["task_id"])
        target = str(task["target"])
        target_label = task.get("target_label")
        any_instance = task.get("any_instance")
        if not isinstance(target_label, str) or not target_label \
                or type(any_instance) is not bool:
            raise CandidateScreenError("prepared task language contract differs")
        try:
            qualifier = camera_gate._task_qualifier(task)
        except camera_gate.CameraScorerGateError as exc:
            raise CandidateScreenError(str(exc)) from exc
        construction_failures = {p: _automatic_construction_failures(task, task_bundle, p)
                                 for p in POLICIES}
        requirement_failures = {p: _automatic_task_requirement_failures(task, task_bundle, p, rows_by_policy)
                                for p in POLICIES}
        eligible_by_policy = {}
        for policy in POLICIES:
            eligible = sorted(
                slot
                for slot, row in rows_by_policy[policy].items()
                if row.get("label") == target_label
                and slot not in excluded_by_policy[policy]
                and pi05_tasks._is_graspable(row)
            )
            if target not in eligible and not construction_failures[policy] and not requirement_failures[policy]:
                raise CandidateScreenError(
                    f"{policy} prepared target is absent from expected same-label roster"
                )
            eligible_by_policy[policy] = eligible
        family_evidence = None
        if task.get("receptacle") is not None:
            family_evidence = paired_receptacle_evidence(task, rows_by_policy)
            if 'automatic_population' not in task_bundle and (not family_evidence["role_constraints_passed"] or any(
                task["receptacle"] not in runtime_slots_by_policy[policy] for policy in POLICIES
            )):
                raise CandidateScreenError("paired receptacle role/runtime contract differs")
        contracts[task_id] = {
            **({'construction_failures_by_policy': construction_failures,
                'task_requirement_failures_by_policy': requirement_failures}
               if 'automatic_population' in task_bundle else {}),
            **({"task_family_evidence": family_evidence} if family_evidence is not None else {}),
            "any_instance": any_instance,
            "eligible_same_label_by_policy": eligible_by_policy,
            "qualifier": qualifier,
            "runtime_body_slots_by_policy": runtime_slots_by_policy,
            "runtime_dimensions_by_policy": runtime_dimensions_by_policy,
            "runtime_nominal_positions_by_policy": nominal_positions_by_policy,
            "task": dict(task),
            "target": target,
            "target_label": target_label,
            "workspace_suite_by_policy": {
                policy: {"robot": dict(suites[policy]["robot"])}
                for policy in POLICIES
            },
        }
    return contracts


def _finite_metric_number(value: Any, *, label: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise CandidateScreenError(f"{label} is not a finite number")
    return float(value)


def _boolean_checks(
    value: Any, *, label: str, expected_keys: Sequence[str] | None = None,
) -> bool:
    if not isinstance(value, Mapping) or not value:
        raise CandidateScreenError(f"{label} checks are absent")
    if expected_keys is not None and set(value) != set(expected_keys):
        raise CandidateScreenError(f"{label} check roster differs")
    if any(type(flag) is not bool for flag in value.values()):
        raise CandidateScreenError(f"{label} checks are not boolean")
    return all(value.values())


def _same_metric_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _replay_reach_record(value: Any, *, label: str) -> bool:
    if not isinstance(value, Mapping):
        raise CandidateScreenError(f"{label} reach record is absent")
    distance = _finite_metric_number(value.get("distance_m"), label=f"{label} distance")
    bearing = _finite_metric_number(
        value.get("bearing_off_axis_deg"), label=f"{label} bearing"
    )
    margins = value.get("margins")
    expected_margin_keys = {"front_cone_deg", "inner_radius_m", "outer_radius_m"}
    if not isinstance(margins, Mapping) or set(margins) != expected_margin_keys:
        raise CandidateScreenError(f"{label} reach margin roster differs")
    observed_margins = {
        key: _finite_metric_number(margins[key], label=f"{label} {key}")
        for key in expected_margin_keys
    }
    expected_margins = {
        "front_cone_deg": FRONT_CONE_DEG - abs(bearing),
        "inner_radius_m": distance - REACH_MIN_M,
        "outer_radius_m": REACH_MAX_M - distance,
    }
    if any(
        not _same_metric_number(observed_margins[key], expected_margins[key])
        for key in expected_margin_keys
    ):
        raise CandidateScreenError(f"{label} reach margins differ from distance/bearing")
    expected_passed = all(value >= 0.0 for value in expected_margins.values())
    expected_safe = (
        expected_margins["inner_radius_m"] >= MIN_RADIAL_MARGIN_M
        and expected_margins["outer_radius_m"] >= MIN_RADIAL_MARGIN_M
        and expected_margins["front_cone_deg"] >= MIN_CONE_MARGIN_DEG
    )
    if type(value.get("passed")) is not bool or value["passed"] != expected_passed \
            or type(value.get("safe")) is not bool or value["safe"] != expected_safe:
        raise CandidateScreenError(f"{label} reach pass flags differ")
    return expected_safe


def _replay_settle_record(row: Mapping[str, Any], *, label: str) -> bool:
    settle = row.get("settle_protocol")
    if not isinstance(settle, Mapping) or set(settle) != set(SETTLE_PROTOCOL_KEYS):
        raise CandidateScreenError(f"{label} settle protocol roster differs")
    for key in (
        "model_timestep_s", "requested_duration_s", "simulated_duration_s"
    ):
        _finite_metric_number(settle.get(key), label=f"{label} settle {key}")
    if type(settle.get("step_count")) is not int or settle["step_count"] < 0 \
            or not isinstance(settle.get("engine"), str) \
            or not isinstance(settle.get("step_function"), str):
        raise CandidateScreenError(f"{label} settle protocol values differ")
    from robo.eval import e4_camera_scorer_gate as camera_gate

    expected_checks = camera_gate._settle_contract_checks({"settle_protocol": settle})
    recorded_checks = row.get("settle_contract_checks")
    if not isinstance(recorded_checks, Mapping) \
            or set(recorded_checks) != set(SETTLE_CHECK_KEYS) \
            or any(type(flag) is not bool for flag in recorded_checks.values()) \
            or dict(recorded_checks) != expected_checks:
        raise CandidateScreenError(f"{label} settle checks differ from raw protocol")
    return all(expected_checks.values())


def _replay_qualifier_record(
    value: Any, *, identity: Mapping[str, Any], contract: Mapping[str, Any],
    policy: str, runtime_footprints: Mapping[str, Mapping[str, Any]], label: str,
) -> tuple[bool, float | None]:
    if not isinstance(value, Mapping):
        raise CandidateScreenError(f"{label} qualifier record is absent")
    if any(value.get(key) != expected for key, expected in identity.items()):
        raise CandidateScreenError(f"{label} qualifier binding differs")
    same_label = contract.get("eligible_same_label_by_policy", {}).get(policy)
    target = identity["target"]
    if not isinstance(same_label, list) or not same_label or target not in same_label \
            or value.get("eligible_same_label_objects") != same_label:
        raise CandidateScreenError(f"{label} qualifier differs from expected factory roster")
    any_instance = contract.get("any_instance")
    qualifier = contract.get("qualifier")
    target_label = contract.get("target_label")
    if type(any_instance) is not bool or qualifier not in (*QUALIFIERS, None) \
            or not isinstance(target_label, str) or not target_label \
            or value.get("any_instance") is not any_instance \
            or value.get("qualifier") != qualifier \
            or value.get("target_label") != target_label \
            or value.get("strict_required_margin_m") != QUALIFIER_MARGIN_M \
            or value.get("eligible_same_label_objects") != same_label:
        raise CandidateScreenError(f"{label} qualifier frozen contract differs")
    applicable = value.get("applicable")
    margin = _finite_metric_number(
        value.get("observed_margin_m"), label=f"{label} qualifier margin", allow_none=True
    )
    base_keys = set(identity) | {
        "any_instance",
        "applicable",
        "eligible_same_label_objects",
        "observed_margin_m",
        "passed",
        "qualifier",
        "strict_required_margin_m",
        "target_label",
    }
    scored = False
    if len(same_label) == 1:
        expected_passed = qualifier is None and not any_instance
        if applicable is not False or margin is not None:
            raise CandidateScreenError(f"{label} unique-target qualifier semantics differ")
    elif qualifier is None:
        expected_passed = any_instance
        if applicable is not True or margin is not None:
            raise CandidateScreenError(f"{label} any-instance qualifier semantics differ")
    elif any_instance:
        expected_passed = False
        if applicable is not True or margin is not None:
            raise CandidateScreenError(f"{label} qualified-any semantics differ")
    else:
        scored = True
        if applicable is not True or margin is None:
            raise CandidateScreenError(f"{label} multi-target qualifier semantics differ")
        coordinates = value.get("coordinates_in_robot_base_frame")
        scores = value.get("qualifier_scores_m")
        if not isinstance(coordinates, Mapping) or set(coordinates) != set(same_label) \
                or not isinstance(scores, Mapping) or set(scores) != set(same_label):
            raise CandidateScreenError(f"{label} qualifier score roster differs")
        suite = contract.get("workspace_suite_by_policy", {}).get(policy)
        robot = suite.get("robot") if isinstance(suite, Mapping) else None
        if not isinstance(robot, Mapping):
            raise CandidateScreenError(f"{label} qualifier robot contract is absent")
        base_position = np.asarray(robot.get("base_pos"), dtype=float)
        base_yaw = robot.get("base_yaw")
        if base_position.shape != (3,) or not np.isfinite(base_position).all() \
                or isinstance(base_yaw, bool) \
                or not isinstance(base_yaw, (int, float)) \
                or not math.isfinite(float(base_yaw)):
            raise CandidateScreenError(f"{label} qualifier robot contract differs")
        replayed_scores = {}
        for slot in same_label:
            footprint = runtime_footprints.get(slot)
            if not isinstance(footprint, Mapping):
                raise CandidateScreenError(f"{label} qualifier runtime body is absent")
            replayed_forward, replayed_left = _base_frame_coordinates(
                footprint["center_xy_m"], base_position[:2], float(base_yaw)
            )
            coordinates_row = coordinates[slot]
            if not isinstance(coordinates_row, Mapping) \
                    or set(coordinates_row) != {"forward_m", "left_m"}:
                raise CandidateScreenError(f"{label} qualifier coordinates differ")
            forward = _finite_metric_number(
                coordinates_row["forward_m"], label=f"{label} qualifier forward"
            )
            left = _finite_metric_number(
                coordinates_row["left_m"], label=f"{label} qualifier left"
            )
            if not _same_metric_number(forward, replayed_forward) \
                    or not _same_metric_number(left, replayed_left):
                raise CandidateScreenError(
                    f"{label} qualifier coordinates differ from runtime footprint/base"
                )
            replayed_scores[slot] = {
                "leftmost": replayed_left,
                "rightmost": -replayed_left,
                "nearest": -replayed_forward,
                "farthest": replayed_forward,
            }[qualifier]
            recorded_score = _finite_metric_number(
                scores[slot], label=f"{label} qualifier score"
            )
            if not _same_metric_number(recorded_score, replayed_scores[slot]):
                raise CandidateScreenError(f"{label} qualifier score differs from coordinates")
        competitor = max(
            (slot for slot in same_label if slot != target),
            key=replayed_scores.__getitem__,
        )
        replayed_margin = replayed_scores[target] - replayed_scores[competitor]
        if value.get("competitor") != competitor \
                or not _same_metric_number(margin, replayed_margin):
            raise CandidateScreenError(f"{label} qualifier margin differs from raw scores")
        expected_passed = replayed_margin > QUALIFIER_MARGIN_M
    expected_keys = base_keys | (
        {"competitor", "coordinates_in_robot_base_frame", "qualifier_scores_m"}
        if scored else set()
    )
    if set(value) != expected_keys:
        raise CandidateScreenError(f"{label} qualifier conditional schema differs")
    if type(value.get("passed")) is not bool or value["passed"] != expected_passed:
        raise CandidateScreenError(f"{label} qualifier pass differs from raw semantics")
    return expected_passed, margin


def _replay_stability_record(
    value: Any, *, target: str, expected_slots: Sequence[str],
    expected_nominal_positions: Mapping[str, Sequence[float]],
    jitter_offset_xy: Sequence[float],
    runtime_footprints: Mapping[str, Mapping[str, Any]], label: str,
) -> tuple[bool, float]:
    expected_keys = {
        "checks",
        "drift_rows",
        "maximum_room_drift_m",
        "maximum_room_linear_speed_m_s",
        "passed",
        "speed_rows",
        "target_drift_m",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise CandidateScreenError(f"{label} stability record is absent")
    drift_rows = value.get("drift_rows")
    speed_rows = value.get("speed_rows")
    if not isinstance(drift_rows, list) or not isinstance(speed_rows, list) \
            or len(drift_rows) != len(expected_slots) \
            or len(speed_rows) != len(expected_slots) \
            or not isinstance(expected_nominal_positions, Mapping) \
            or set(expected_nominal_positions) != set(expected_slots) \
            or set(runtime_footprints) != set(expected_slots) \
            or target not in expected_slots:
        raise CandidateScreenError(f"{label} stability body roster differs")
    offset = np.asarray(jitter_offset_xy, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise CandidateScreenError(f"{label} stability jitter differs")
    drifts = {}
    for raw in drift_rows:
        row_keys = {
            "drift_m",
            "expected_post_jitter_position_m",
            "observed_post_settle_position_m",
            "slot",
        }
        if not isinstance(raw, Mapping) or set(raw) != row_keys:
            raise CandidateScreenError(f"{label} drift row schema differs")
        slot = raw["slot"]
        if slot not in expected_slots or slot in drifts:
            raise CandidateScreenError(f"{label} drift row roster differs")
        nominal = np.asarray(expected_nominal_positions[slot], dtype=float)
        expected_position = nominal.copy()
        if slot == target:
            expected_position[:2] += offset
        recorded_expected = np.asarray(raw["expected_post_jitter_position_m"], dtype=float)
        observed = np.asarray(raw["observed_post_settle_position_m"], dtype=float)
        footprint_position = np.asarray(
            runtime_footprints[slot]["position_world_m"], dtype=float
        )
        if nominal.shape != (3,) or not np.isfinite(nominal).all() \
                or recorded_expected.shape != (3,) \
                or observed.shape != (3,) \
                or not np.isfinite(recorded_expected).all() \
                or not np.isfinite(observed).all() \
                or not np.allclose(
                    recorded_expected, expected_position, rtol=0.0, atol=1e-12
                ) \
                or not np.allclose(
                    observed, footprint_position, rtol=0.0, atol=1e-12
                ):
            raise CandidateScreenError(f"{label} drift positions differ from reset/workspace")
        replayed_drift = float(np.linalg.norm(observed - expected_position))
        recorded_drift = _finite_metric_number(raw["drift_m"], label=f"{label} drift")
        if recorded_drift < 0.0 or not _same_metric_number(recorded_drift, replayed_drift):
            raise CandidateScreenError(f"{label} drift differs from raw positions")
        drifts[slot] = replayed_drift
    speeds = {}
    for raw in speed_rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "linear_speed_m_s", "linear_velocity_xyz_m_s", "slot"
        }:
            raise CandidateScreenError(f"{label} speed row schema differs")
        slot = raw["slot"]
        if slot not in expected_slots or slot in speeds:
            raise CandidateScreenError(f"{label} speed row roster differs")
        velocity = np.asarray(raw["linear_velocity_xyz_m_s"], dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise CandidateScreenError(f"{label} raw velocity differs")
        replayed_speed = float(np.linalg.norm(velocity))
        recorded_speed = _finite_metric_number(
            raw["linear_speed_m_s"], label=f"{label} linear speed"
        )
        if recorded_speed < 0.0 or not _same_metric_number(recorded_speed, replayed_speed):
            raise CandidateScreenError(f"{label} speed differs from raw velocity")
        speeds[slot] = replayed_speed
    if set(drifts) != set(expected_slots) or set(speeds) != set(expected_slots):
        raise CandidateScreenError(f"{label} stability exact body roster differs")
    maximum_drift = max(drifts.values())
    maximum_speed = max(speeds.values())
    target_drift = drifts[target]
    for key, replayed in (
        ("maximum_room_drift_m", maximum_drift),
        ("maximum_room_linear_speed_m_s", maximum_speed),
        ("target_drift_m", target_drift),
    ):
        recorded = _finite_metric_number(value.get(key), label=f"{label} {key}")
        if not _same_metric_number(recorded, replayed):
            raise CandidateScreenError(f"{label} stability summary differs from raw rows")
    expected_checks = {
        "all_body_settle_drift_below_30mm": maximum_drift < STABILITY_DRIFT_LIMIT_M,
        "all_body_linear_speed_below_50mm_s": maximum_speed < AT_REST_LINEAR_SPEED_M_S,
        "target_settle_drift_below_30mm": target_drift < STABILITY_DRIFT_LIMIT_M,
    }
    recorded_checks = value.get("checks")
    if not isinstance(recorded_checks, Mapping) \
            or set(recorded_checks) != set(STABILITY_CHECK_KEYS) \
            or dict(recorded_checks) != expected_checks:
        raise CandidateScreenError(f"{label} stability checks differ from raw rows")
    expected_passed = all(expected_checks.values())
    if type(value.get("passed")) is not bool or value["passed"] != expected_passed:
        raise CandidateScreenError(f"{label} stability pass differs from raw rows")
    return expected_passed, maximum_drift


def _clearance_blockers(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, list):
        raise CandidateScreenError(f"{label} blockers are not a list")
    result = {}
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != {"clearance_m", "slot"}:
            raise CandidateScreenError(f"{label} blocker schema differs")
        slot = raw["slot"]
        if not isinstance(slot, str) or OBJECT_SLOT_RE.fullmatch(slot) is None \
                or slot in result:
            raise CandidateScreenError(f"{label} blocker roster differs")
        clearance = _finite_metric_number(
            raw["clearance_m"], label=f"{label} blocker clearance"
        )
        if clearance >= MIN_OBSTACLE_CLEARANCE_M:
            raise CandidateScreenError(f"{label} includes a non-blocking footprint")
        result[slot] = clearance
    return result


def _replay_workspace_record(
    value: Any, *, contract: Mapping[str, Any], policy: str, label: str,
) -> tuple[bool, float, float | None, dict[str, dict[str, Any]]]:
    if not isinstance(value, Mapping):
        raise CandidateScreenError(f"{label} workspace record is absent")
    raw_footprints = value.get("runtime_footprints")
    expected_slots = contract.get("runtime_body_slots_by_policy", {}).get(policy)
    expected_dimensions = contract.get("runtime_dimensions_by_policy", {}).get(policy)
    if not isinstance(raw_footprints, Mapping) or not isinstance(expected_slots, list) \
            or not isinstance(expected_dimensions, Mapping) \
            or set(raw_footprints) != set(expected_slots) \
            or set(expected_dimensions) != set(expected_slots):
        raise CandidateScreenError(f"{label} runtime footprint roster differs")
    footprint_keys = {
        "center_xy_m",
        "half_extents_xy_m",
        "hi_xy_m",
        "lo_xy_m",
        "position_world_m",
        "quaternion_wxyz",
    }
    footprints = {}
    for slot in expected_slots:
        raw = raw_footprints[slot]
        if not isinstance(raw, Mapping) or set(raw) != footprint_keys:
            raise CandidateScreenError(f"{label} runtime footprint schema differs")
        position = np.asarray(raw["position_world_m"], dtype=float)
        quaternion = np.asarray(raw["quaternion_wxyz"], dtype=float)
        dimensions = np.asarray(expected_dimensions[slot], dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,) \
                or dimensions.shape != (3,) \
                or not np.isfinite(position).all() \
                or not np.isfinite(quaternion).all() \
                or not np.isfinite(dimensions).all() \
                or np.any(dimensions <= 0.0) \
                or not math.isclose(
                    float(np.linalg.norm(quaternion)), 1.0, rel_tol=0.0, abs_tol=1e-6
                ):
            raise CandidateScreenError(f"{label} runtime footprint values differ")
        replayed_footprint = _world_xy_rectangle(position, quaternion, dimensions)
        if any(
            not np.allclose(
                np.asarray(raw[key], dtype=float),
                np.asarray(replayed_footprint[key], dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
            for key in footprint_keys
        ):
            raise CandidateScreenError(
                f"{label} runtime footprint differs from factory dimensions and raw pose"
            )
        # Pose/dimension consistency was checked above at the fixed 1e-12
        # tolerance. Preserve serialized values for exact derived-record
        # replay: normalizing a saved unit quaternion again can change a
        # half-extent by one ULP without any physical difference.
        footprints[slot] = dict(raw)
    task = contract.get("task")
    suite = contract.get("workspace_suite_by_policy", {}).get(policy)
    if not isinstance(task, Mapping) or not isinstance(suite, Mapping):
        raise CandidateScreenError(f"{label} frozen workspace contract is absent")
    replayed = _workspace_from_footprints(
        current=footprints,
        task=task,
        suite=suite,
        goal_probe=value.get("task_destination", {}).get("goal_probe"),
    )
    if dict(value) != replayed:
        raise CandidateScreenError(f"{label} workspace differs from raw footprints")
    return (
        replayed["passed"],
        replayed["base_clearance"]["minimum_clearance_m"],
        replayed["obstacle_clearance"]["minimum_clearance_m"],
        footprints,
    )


def _replay_qualifier_metrics(
    metric_rows: Sequence[Mapping[str, Any]],
    *, scene_id: str, expected_task_contracts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the exact cell lattice and every aggregate-facing summary."""
    if any(
        not isinstance(task_id, str) or not task_id
        or not isinstance(contract, Mapping)
        for task_id, contract in expected_task_contracts.items()
    ) \
            or any(
                not isinstance(contract.get("target"), str)
                or OBJECT_SLOT_RE.fullmatch(contract["target"]) is None
                for contract in expected_task_contracts.values()
            ):
        raise CandidateScreenError("expected qualifier task identities are invalid")
    from robo.eval.episode_log import derive_reset_seed

    expected_cells = {
        (task_id, policy, episode)
        for task_id in expected_task_contracts
        for policy in POLICIES
        for episode in range(EPISODES)
    }
    observed_cells: set[tuple[str, str, int]] = set()
    validated_rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(metric_rows):
        if not isinstance(raw_row, Mapping):
            raise CandidateScreenError(f"qualifier metric row {index} is not an object")
        row = dict(raw_row)
        task_id = row.get("task_id")
        policy = row.get("policy_id")
        episode = row.get("episode")
        if task_id not in expected_task_contracts or policy not in POLICIES \
                or isinstance(episode, bool) or not isinstance(episode, int) \
                or episode not in range(EPISODES):
            raise CandidateScreenError(f"qualifier metric row {index} identity differs")
        cell = (task_id, policy, episode)
        if cell in observed_cells:
            raise CandidateScreenError("qualifier metric cell is duplicated")
        observed_cells.add(cell)
        contract = expected_task_contracts[task_id]
        reset_seed = derive_reset_seed(BASE_SEED, task_id, episode)
        cell_id = f"{policy.lower()}__{task_id}__seed{BASE_SEED}__ep{episode}"
        identity = {
            "cell_id": cell_id,
            "episode": episode,
            "policy_id": policy,
            "reset_seed": reset_seed,
            "scene_id": scene_id,
            "task_id": task_id,
            "target": contract["target"],
        }
        if any(row.get(key) != value for key, value in identity.items()):
            raise CandidateScreenError(f"qualifier metric row {index} binding differs")

        failures = contract.get('construction_failures_by_policy', {}).get(policy)
        if failures:
            expected_failure = {**identity, 'failure_type': 'construction_endpoint_unavailable',
                'construction_failures': failures,
                'checks': {'construction_endpoints_accepted': False}, 'passed': False,
                **{key: None for key in ('qualifier', 'reset_jitter', 'settle_contract_checks',
                    'settle_protocol', 'stability', 'workspace')}}
            if row != expected_failure:
                raise CandidateScreenError('unexecuted automatic construction failure differs from its source')
            validated_rows.append({**row, **{key: None for key in (
                '_summary_base_clearance_m', '_summary_maximum_drift_m',
                '_summary_obstacle_clearance_m', '_summary_qualifier_margin_m')}})
            continue
        requirement_failures = contract.get('task_requirement_failures_by_policy', {}).get(policy)
        if requirement_failures:
            expected_failure = {**identity, 'failure_type': 'task_construction_requirement_failed',
                'task_requirement_failures': requirement_failures,
                'checks': {'task_construction_requirements_passed': False}, 'passed': False,
                **{key: None for key in ('qualifier', 'reset_jitter', 'settle_contract_checks',
                    'settle_protocol', 'stability', 'workspace')}}
            if row != expected_failure:
                raise CandidateScreenError('unexecuted automatic task precondition differs from its source')
            validated_rows.append({**row, **{key: None for key in (
                '_summary_base_clearance_m', '_summary_maximum_drift_m',
                '_summary_obstacle_clearance_m', '_summary_qualifier_margin_m')}})
            continue
        checks = row.get("checks")
        checks_passed = _boolean_checks(
            checks, label=f"qualifier metric row {index}", expected_keys=CELL_CHECK_KEYS
        )
        if type(row.get("passed")) is not bool or row["passed"] != checks_passed:
            raise CandidateScreenError(f"qualifier metric row {index} passed/checks differ")

        row_label = f"qualifier metric row {index}"
        exact_900 = _replay_settle_record(row, label=row_label)
        if checks["exactly_900_mujoco_steps"] != exact_900:
            raise CandidateScreenError(f"qualifier metric row {index} exact-step check differs")
        if checks["settle_contract_passed"] != exact_900:
            raise CandidateScreenError(f"qualifier metric row {index} settle check differs")

        jitter = row.get("reset_jitter")
        draw = np.random.RandomState(reset_seed).random_sample(2)
        expected_offset = ((draw * 2.0 - 1.0) * JITTER_XY_M).tolist()
        jitter_valid = (
            isinstance(jitter, Mapping)
            and jitter.get("algorithm") == "numpy.random.RandomState.random_sample_then_affine"
            and jitter.get("applied") is True
            and jitter.get("body") == contract["target"]
            and jitter.get("max_abs_xy_m") == JITTER_XY_M
            and jitter.get("offset_xy_m") == expected_offset
            and jitter.get("reset_seed") == reset_seed
            and jitter.get("uniform_draw_0_1") == draw.tolist()
        )
        if checks["jitter_draw_replayed"] != jitter_valid:
            raise CandidateScreenError(f"qualifier metric row {index} jitter check differs")

        workspace_passed, base_clearance, obstacle_clearance, runtime_footprints = (
            _replay_workspace_record(
                row.get("workspace"), contract=contract, policy=policy, label=row_label
            )
        )
        if checks["workspace_passed"] != workspace_passed:
            raise CandidateScreenError(f"qualifier metric row {index} workspace check differs")
        qualifier_passed, margin = _replay_qualifier_record(
            row.get("qualifier"),
            identity=identity,
            contract=contract,
            policy=policy,
            runtime_footprints=runtime_footprints,
            label=row_label,
        )
        if checks["qualifier_strictly_passed"] != qualifier_passed:
            raise CandidateScreenError(f"qualifier metric row {index} qualifier check differs")
        stability_passed, maximum_drift = _replay_stability_record(
            row.get("stability"),
            target=identity["target"],
            expected_slots=contract["runtime_body_slots_by_policy"][policy],
            expected_nominal_positions=(
                contract["runtime_nominal_positions_by_policy"][policy]
            ),
            jitter_offset_xy=expected_offset,
            runtime_footprints=runtime_footprints,
            label=row_label,
        )
        if checks["stability_passed"] != stability_passed:
            raise CandidateScreenError(f"qualifier metric row {index} stability check differs")
        validated_rows.append({
            **row,
            "_summary_base_clearance_m": base_clearance,
            "_summary_maximum_drift_m": maximum_drift,
            "_summary_obstacle_clearance_m": obstacle_clearance,
            "_summary_qualifier_margin_m": margin,
        })
    if observed_cells != expected_cells:
        raise CandidateScreenError("qualifier metric exact task/policy/episode coverage differs")

    task_summaries = []
    for task_id in sorted(expected_task_contracts):
        cells = [row for row in validated_rows if row["task_id"] == task_id]
        obstacle_values = [
            row["_summary_obstacle_clearance_m"] for row in cells
            if row["_summary_obstacle_clearance_m"] is not None
        ]
        qualifier_values = [
            row["_summary_qualifier_margin_m"] for row in cells
            if row["_summary_qualifier_margin_m"] is not None
        ]
        task_summaries.append({
            "cell_count": len(cells),
            "cells_failed": sum(not row["passed"] for row in cells),
            "cells_passed": sum(row["passed"] for row in cells),
            "maximum_room_drift_m": max(
                (row["_summary_maximum_drift_m"] for row in cells
                 if row["_summary_maximum_drift_m"] is not None), default=None
            ),
            "minimum_base_clearance_m": min(
                (row["_summary_base_clearance_m"] for row in cells
                 if row["_summary_base_clearance_m"] is not None), default=None
            ),
            "minimum_goal_obstacle_clearance_m": min(obstacle_values, default=None),
            "minimum_qualifier_margin_m": min(qualifier_values, default=None),
            "strict_pass": all(row["passed"] for row in cells),
            "task_id": task_id,
        })
    strict_ids = [row["task_id"] for row in task_summaries if row["strict_pass"]]
    return {
        "cell_count": len(validated_rows),
        "exact_900_step_cells": sum(
            row["checks"].get("exactly_900_mujoco_steps", False) for row in validated_rows
        ),
        "semantic_blockers": sorted(
            f"cell_failed:{row['cell_id']}" for row in validated_rows if not row["passed"]
        ),
        "strict_pass_task_count": len(strict_ids),
        "strict_pass_task_ids": strict_ids,
        "task_summaries": task_summaries,
    }


def _frozen_menagerie_summary() -> dict[str, Any]:
    from robo.eval import e4_camera_scorer_gate as camera_gate

    try:
        snapshot = camera_gate._menagerie_snapshot(
            EXPECTED_MENAGERIE_ROOT, EXPECTED_MENAGERIE_COMMIT
        )
    except camera_gate.CameraScorerGateError as exc:
        raise CandidateScreenError(str(exc)) from exc
    return {key: value for key, value in snapshot.items() if key != "files"}


def _automatic_construction_failures(task, task_bundle, policy):
    """Read endpoint availability only from an authenticated all-planned task bundle."""
    if 'automatic_population' not in task_bundle:
        return None
    rosters = task_bundle.get('construction_rosters')
    if not isinstance(rosters, Mapping) or set(rosters) != set(POLICIES):
        raise CandidateScreenError('automatic CPU task bundle lacks paired construction rosters')
    roster = rosters[policy]
    planned = roster.get('object_slots')
    partitions = {action: roster.get(action+'ed_slots' if action != 'abstain' else 'abstained_slots')
                  for action in ('accept','reject','abstain')}
    if not isinstance(planned, list) or len(planned) != len(set(planned)) \
            or any(not isinstance(slots,list) or len(slots) != len(set(slots)) for slots in partitions.values()) \
            or sorted(slot for slots in partitions.values() for slot in slots) != sorted(planned):
        raise CandidateScreenError('automatic construction partitions differ from planned objects')
    required = [task['target']] + ([task['receptacle']] if task.get('receptacle') else [])
    if any(slot not in planned for slot in required):
        raise CandidateScreenError('CPU task refers to an undeclared automatic endpoint')
    return [{'object_slot':slot,'terminal_action':action}
            for slot in required for action in ('reject','abstain') if slot in partitions[action]]


def _automatic_task_requirement_failures(task, task_bundle, policy, rows_by_policy):
    if 'automatic_population' not in task_bundle:
        return None
    if _automatic_construction_failures(task, task_bundle, policy):
        return []
    required = [task['target']] + ([task['receptacle']] if task.get('receptacle') else [])
    if any(slot not in rows_by_policy[policy] for slot in required):
        raise CandidateScreenError('accepted task endpoint is missing its validated export row')
    if task.get('receptacle'):
        return paired_receptacle_evidence(task, rows_by_policy)['by_policy'][policy]['reasons']
    return _eligibility_reasons(rows_by_policy[policy][task['target']])


def _qualify_task_suites(
    *, scene_id: str, task_bundle: Mapping[str, Any] | None,
    factories: Mapping[str, Path], menagerie_root: Path,
) -> list[dict[str, Any]]:
    """Reuse the exact CPU reset/workspace/qualifier checks for a frozen suite."""
    from robo.eval import e4_camera_scorer_gate as camera_gate
    from robo.eval.episode_log import derive_reset_seed
    from robo.tasks import pi05_tasks

    rows: list[dict[str, Any]] = []
    if task_bundle is not None:
        suites = {
            policy: json.loads(Path(task_bundle["variant_tasks"][policy]).read_text())
            for policy in POLICIES
        }
        expected_ids = list(task_bundle["logical_task_ids"])
        construction_rows = {p: _rows_for_factory(factories[p]) for p in POLICIES} if 'automatic_population' in task_bundle else None
        for policy in POLICIES:
            suite = suites[policy]
            if [task.get("task_id") for task in suite["tasks"]] != expected_ids:
                raise CandidateScreenError(f"{policy} frozen task roster differs")
            factory = factories[policy]
            task_rows = pi05_tasks._load_objects(factory)
            env = None
            for task in suite["tasks"]:
                task_id = str(task["task_id"])
                construction_failures = _automatic_construction_failures(task, task_bundle, policy)
                if construction_failures:
                    for episode in range(EPISODES):
                        rows.append({
                            'cell_id': f'{policy.lower()}__{task_id}__seed{BASE_SEED}__ep{episode}',
                            'episode': episode, 'policy_id': policy, 'scene_id': scene_id,
                            'task_id': task_id, 'target': task['target'],
                            'reset_seed': derive_reset_seed(BASE_SEED, task_id, episode),
                            'failure_type': 'construction_endpoint_unavailable',
                            'construction_failures': construction_failures,
                            'checks': {'construction_endpoints_accepted': False}, 'passed': False,
                            **{key: None for key in ('qualifier', 'reset_jitter', 'settle_contract_checks',
                                'settle_protocol', 'stability', 'workspace')}})
                    continue
                requirement_failures = _automatic_task_requirement_failures(task, task_bundle, policy, construction_rows)
                if requirement_failures:
                    for episode in range(EPISODES):
                        rows.append({
                            'cell_id': f'{policy.lower()}__{task_id}__seed{BASE_SEED}__ep{episode}',
                            'episode': episode, 'policy_id': policy, 'scene_id': scene_id,
                            'task_id': task_id, 'target': task['target'],
                            'reset_seed': derive_reset_seed(BASE_SEED, task_id, episode),
                            'failure_type': 'task_construction_requirement_failed',
                            'task_requirement_failures': requirement_failures,
                            'checks': {'task_construction_requirements_passed': False}, 'passed': False,
                            **{key: None for key in ('qualifier', 'reset_jitter', 'settle_contract_checks',
                                'settle_protocol', 'stability', 'workspace')}})
                    continue
                if env is None:
                    env = camera_gate._build_headless_droid_env(
                        scene_xml=str(suite["scene_xml"]),
                        base_pos=suite["robot"]["base_pos"],
                        base_yaw=suite["robot"]["base_yaw"],
                        table_box=suite["table"],
                        ext_cam=suite["ext_cam"],
                        exclude_objects=tuple(suite.get("exclude_objects", ())),
                        menagerie_root=menagerie_root,
                    )
                    if hasattr(env, "renderer"):
                        raise CandidateScreenError("headless candidate screen constructed a renderer")
                for episode in range(EPISODES):
                    reset_seed = derive_reset_seed(BASE_SEED, task_id, episode)
                    draw = np.random.RandomState(reset_seed).random_sample(2)
                    observation = env.reset(
                        settle_s=RESET_SETTLE_S,
                        jitter_body=task["target"],
                        jitter_xy=JITTER_XY_M,
                        jitter_uniform_draw=draw,
                        reset_seed=reset_seed,
                    )
                    provenance = env.last_reset_provenance
                    if not isinstance(provenance, Mapping):
                        raise CandidateScreenError("headless reset lacks provenance")
                    settle_checks = camera_gate._settle_contract_checks(provenance)
                    settle = provenance.get("settle_protocol")
                    if not isinstance(settle, Mapping):
                        raise CandidateScreenError("headless settle protocol is absent")
                    exact_900 = (
                        settle.get("step_count") == EXPECTED_RESET_STEPS
                        and math.isclose(
                            float(settle.get("requested_duration_s")),
                            RESET_SETTLE_S,
                            rel_tol=0.0,
                            abs_tol=0.0,
                        )
                    )
                    row_identity = {
                        "cell_id": f"{policy.lower()}__{task_id}__seed{BASE_SEED}__ep{episode}",
                        "episode": episode,
                        "policy_id": policy,
                        "reset_seed": reset_seed,
                        "scene_id": scene_id,
                        "task_id": task_id,
                        "target": task["target"],
                    }
                    qualifier = camera_gate.disambiguation_metric(
                        env, task, suite, task_rows, row_identity
                    )
                    stability = _runtime_stability(
                        env=env, task=task, provenance=provenance
                    )
                    workspace = _runtime_workspace(
                        env=env,
                        task=task,
                        suite=suite,
                        task_rows=task_rows,
                    )
                    jitter = provenance.get("jitter")
                    reset_checks = {
                        "exactly_900_mujoco_steps": exact_900,
                        "headless_observation_empty": observation == {},
                        "jitter_draw_replayed": (
                            isinstance(jitter, Mapping)
                            and jitter.get("reset_seed") == reset_seed
                            and jitter.get("body") == task["target"]
                            and jitter.get("max_abs_xy_m") == JITTER_XY_M
                            and jitter.get("uniform_draw_0_1") == draw.tolist()
                        ),
                        "no_renderer_constructed": not hasattr(env, "renderer"),
                        "settle_contract_passed": all(settle_checks.values()),
                    }
                    checks = {
                        **reset_checks,
                        "qualifier_strictly_passed": bool(qualifier["passed"]),
                        "stability_passed": bool(stability["passed"]),
                        "workspace_passed": bool(workspace["passed"]),
                    }
                    rows.append({
                        **row_identity,
                        "checks": checks,
                        "passed": all(checks.values()),
                        "qualifier": qualifier,
                        "reset_jitter": dict(jitter) if isinstance(jitter, Mapping) else jitter,
                        "settle_contract_checks": settle_checks,
                        "settle_protocol": dict(settle),
                        "stability": stability,
                        "workspace": workspace,
                    })
    return rows


def qualify_scene(
    *, screen_id: str, scene_id: str, expected_commit: str,
    menagerie_root: str | Path, expected_menagerie_commit: str,
    automatic_population: bool = False,
) -> dict[str, Any]:
    """Run the exact headless 900-step screen and seal all semantic failures."""
    screen_id = _validated_screen_id(screen_id)
    if automatic_population:
        from robo.eval.e3_factory_materializer import SCENE_ID_RE
        if not isinstance(scene_id,str) or SCENE_ID_RE.fullmatch(scene_id) is None:
            raise CandidateScreenError("automatic scene ID is unsafe")
    else:
        scene_id = _validated_scene_id(scene_id)
    code = _code_snapshot(expected_commit)
    root = evidence_root()
    from robo.eval import e4_camera_scorer_gate as camera_gate
    from robo.eval.episode_log import derive_reset_seed
    from robo.tasks import pi05_tasks

    try:
        menagerie_before = camera_gate._menagerie_snapshot(
            Path(menagerie_root), expected_menagerie_commit
        )
    except camera_gate.CameraScorerGateError as exc:
        raise CandidateScreenError(str(exc)) from exc
    prepared_before = _load_prepare(
        root=root,
        screen_id=screen_id,
        scene_id=scene_id,
        expected_commit=expected_commit,
    )
    if automatic_population != ("automatic_population" in prepared_before["gate"]):
        raise CandidateScreenError("automatic qualification scope does not match sealed prepare")
    candidate_count = int(prepared_before["gate"]["candidate_count"])
    expected_task_contracts = _prepared_task_contracts(prepared_before)
    if len(expected_task_contracts) != candidate_count:
        raise CandidateScreenError("prepared candidate task count differs")
    rows = _qualify_task_suites(
        scene_id=scene_id,
        task_bundle=prepared_before["task_bundle"] if candidate_count else None,
        factories=prepared_before["factories"],
        menagerie_root=Path(menagerie_before["root"]),
    )
    replay = _replay_qualifier_metrics(
        rows, scene_id=scene_id, expected_task_contracts=expected_task_contracts
    )
    gate = {
        "candidate_count": candidate_count,
        "cell_count": replay["cell_count"],
        "code": code,
        "created_utc": _utc_now(),
        "evidence_root": str(root),
        "exact_900_step_cells": replay["exact_900_step_cells"],
        "gpu_launch_allowed": False,
        "headline_eligible": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_candidate_scene_qualifier_gate",
        "menagerie": {key: value for key, value in menagerie_before.items() if key != "files"},
        "paper_ready": False,
        "prepare_manifest_sha256": prepared_before["prepare_bundle"]["manifest_sha256"],
        "renderer_constructed": False,
        "scene_id": scene_id,
        "screen_id": screen_id,
        "semantic_blockers": replay["semantic_blockers"],
        "strict_pass_task_count": replay["strict_pass_task_count"],
        "strict_pass_task_ids": replay["strict_pass_task_ids"],
        "study_scope": STUDY_SCOPE,
        "task_summaries": replay["task_summaries"],
    }
    prepared_after = _load_prepare(
        root=root,
        screen_id=screen_id,
        scene_id=scene_id,
        expected_commit=expected_commit,
    )
    if prepared_after["gate"] != prepared_before["gate"] \
            or prepared_after["prepare_bundle"]["manifest_sha256"] != prepared_before["prepare_bundle"]["manifest_sha256"]:
        raise CandidateScreenError("prepare closure changed during qualification")
    try:
        menagerie_after = camera_gate._menagerie_snapshot(
            Path(menagerie_root), expected_menagerie_commit
        )
    except camera_gate.CameraScorerGateError as exc:
        raise CandidateScreenError(str(exc)) from exc
    if menagerie_after != menagerie_before:
        raise CandidateScreenError("menagerie closure changed during qualification")
    output = _publish_bundle(
        _experiment_root(root, screen_id) / "scene_qualifiers" / scene_id,
        manifest_kind="e4_candidate_scene_qualifier_artifacts",
        payloads={
            "gate.json": _json_bytes(gate),
            "metrics.jsonl": _jsonl_bytes(rows),
        },
        manifest_fields={
            "code": code,
            "created_utc": _utc_now(),
            "prepare_manifest_sha256": prepared_before["prepare_bundle"]["manifest_sha256"],
            "scene_id": scene_id,
            "screen_id": screen_id,
        },
    )
    return {
        **gate,
        "qualifier_manifest_sha256": output["manifest_sha256"],
        "qualifier_seal_sha256": output["seal_sha256"],
    }


def _validate_qualifier_output(
    *, root: Path, screen_id: str, scene_id: str, expected_commit: str,
) -> dict[str, Any]:
    bundle = _validate_bundle(
        _experiment_root(root, screen_id) / "scene_qualifiers" / scene_id,
        root=root,
        expected_kind="e4_candidate_scene_qualifier_artifacts",
    )
    gate = _read_json_member(bundle["directory"], "gate.json")
    if not isinstance(gate, Mapping) or gate.get("manifest_kind") != "e4_candidate_scene_qualifier_gate" \
            or gate.get("scene_id") != scene_id or gate.get("screen_id") != screen_id \
            or gate.get("study_scope") != STUDY_SCOPE:
        raise CandidateScreenError("qualifier gate binding differs")
    expected_code = {
        "code_root": str(CODE_ROOT), "commit": expected_commit, "dirty": False
    }
    if gate.get("code") != expected_code or gate.get("evidence_root") != str(root):
        raise CandidateScreenError("qualifier code binding differs")
    for flag in (
        "gpu_launch_allowed", "headline_eligible", "large_rollout_launch_allowed", "paper_ready"
    ):
        if gate.get(flag) is not False:
            raise CandidateScreenError(f"qualifier illegally authorizes {flag}")
    if gate.get("renderer_constructed") is not False:
        raise CandidateScreenError("qualifier claims a renderer")
    prepared = _load_prepare(
        root=root,
        screen_id=screen_id,
        scene_id=scene_id,
        expected_commit=expected_commit,
    )
    expected_task_contracts = _prepared_task_contracts(prepared)
    candidate_count = gate.get("candidate_count")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) \
            or candidate_count != len(expected_task_contracts) \
            or candidate_count != prepared["gate"].get("candidate_count"):
        raise CandidateScreenError("qualifier candidate count differs from prepare")
    prepare_sha256 = prepared["prepare_bundle"]["manifest_sha256"]
    if gate.get("prepare_manifest_sha256") != prepare_sha256:
        raise CandidateScreenError("prepare changed after qualifier publication")
    manifest = bundle["manifest"]
    if manifest.get("code") != expected_code \
            or manifest.get("scene_id") != scene_id \
            or manifest.get("screen_id") != screen_id \
            or manifest.get("prepare_manifest_sha256") != prepare_sha256:
        raise CandidateScreenError("qualifier artifact manifest binding differs")
    if gate.get("menagerie") != _frozen_menagerie_summary():
        raise CandidateScreenError("qualifier menagerie closure differs")

    metrics_path = bundle["directory"] / "metrics.jsonl"
    metric_rows = []
    for line_number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CandidateScreenError(f"invalid metrics JSONL line {line_number}") from exc
        if not isinstance(row, Mapping):
            raise CandidateScreenError("qualifier metric row is not an object")
        metric_rows.append(row)
    replay = _replay_qualifier_metrics(
        metric_rows,
        scene_id=scene_id,
        expected_task_contracts=expected_task_contracts,
    )
    for field in (
        "cell_count",
        "exact_900_step_cells",
        "semantic_blockers",
        "strict_pass_task_count",
        "strict_pass_task_ids",
        "task_summaries",
    ):
        if gate.get(field) != replay[field]:
            raise CandidateScreenError(f"qualifier {field} differs from sealed metrics replay")
    expected_cell_count = candidate_count * len(POLICIES) * EPISODES
    # A construction failure is an explicit terminal cell with null telemetry.
    # Its identity and reason have already been replayed against the sealed
    # task contract above; it cannot also count as 900 executed MuJoCo steps.
    expected_unexecuted_cells = EPISODES * sum(
        bool(contract.get("construction_failures_by_policy", {}).get(policy)
             or contract.get("task_requirement_failures_by_policy", {}).get(policy))
        for contract in expected_task_contracts.values() for policy in POLICIES
    )
    if replay["cell_count"] != expected_cell_count \
            or replay["exact_900_step_cells"] != expected_cell_count - expected_unexecuted_cells:
        raise CandidateScreenError("qualifier executed/prebuild-failure coverage differs")
    return {
        "bundle": bundle,
        "gate": dict(gate),
        "metric_rows": metric_rows,
    }


def build_aggregate_gate(
    *, screen_id: str, code: Mapping[str, Any], scene_results: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pure deterministic aggregate; only >=2 strict-task scenes are selected."""
    scene_rows = []
    eligible = []
    for scene_id in SCENE_IDS:
        result = scene_results[scene_id]
        status = str(result["status"])
        gate = result.get("gate") if status == "validated" else None
        row: dict[str, Any] = {
            "candidate_count": int(gate["candidate_count"]) if gate else None,
            "qualifier_manifest_sha256": result.get("qualifier_manifest_sha256"),
            "reason": result.get("reason"),
            "scene_id": scene_id,
            "selected": False,
            "status": status,
            "strict_pass_task_count": int(gate["strict_pass_task_count"]) if gate else 0,
            "strict_pass_task_ids": list(gate["strict_pass_task_ids"]) if gate else [],
        }
        if gate and row["strict_pass_task_count"] >= 2:
            strict = [summary for summary in gate["task_summaries"] if summary["strict_pass"]]
            base_values = [summary["minimum_base_clearance_m"] for summary in strict]
            obstacle_values = [
                summary["minimum_goal_obstacle_clearance_m"]
                for summary in strict
                if summary["minimum_goal_obstacle_clearance_m"] is not None
            ]
            qualifier_values = [
                summary["minimum_qualifier_margin_m"]
                for summary in strict
                if summary["minimum_qualifier_margin_m"] is not None
            ]
            row.update({
                "maximum_room_drift_m": max(summary["maximum_room_drift_m"] for summary in strict),
                "minimum_base_clearance_m": min(base_values),
                "minimum_goal_obstacle_clearance_m": min(obstacle_values, default=None),
                "minimum_qualifier_margin_m": min(qualifier_values, default=None),
            })
            eligible.append(row)
        elif gate:
            row["reason"] = "fewer_than_two_strict_pass_tasks"
        scene_rows.append(row)
    eligible.sort(
        key=lambda row: (
            -row["strict_pass_task_count"],
            -float(row["minimum_base_clearance_m"]),
            -float(
                row["minimum_goal_obstacle_clearance_m"]
                if row["minimum_goal_obstacle_clearance_m"] is not None else 1.0
            ),
            float(row["maximum_room_drift_m"]),
            -float(
                row["minimum_qualifier_margin_m"]
                if row["minimum_qualifier_margin_m"] is not None else 1.0
            ),
            row["scene_id"],
        )
    )
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
        row["selected"] = True
    gate = {
        "code": dict(code),
        "created_utc": _utc_now(),
        "gpu_launch_allowed": False,
        "headline_eligible": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_candidate_screen_aggregate_gate",
        "paper_ready": False,
        "roster": list(SCENE_IDS),
        "scene_count": len(SCENE_IDS),
        "screen_id": screen_id,
        "selected_scene_count": len(eligible),
        "selected_scenes": [
            {
                "rank": row["rank"],
                "scene_id": row["scene_id"],
                "strict_pass_task_count": row["strict_pass_task_count"],
                "strict_pass_task_ids": row["strict_pass_task_ids"],
            }
            for row in eligible
        ],
        "study_scope": STUDY_SCOPE,
    }
    return gate, scene_rows


def _ranking_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    columns = (
        "scene_id", "status", "candidate_count", "strict_pass_task_count",
        "selected", "rank", "minimum_base_clearance_m",
        "minimum_goal_obstacle_clearance_m", "minimum_qualifier_margin_m",
        "maximum_room_drift_m", "reason", "qualifier_manifest_sha256",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return handle.getvalue().encode()


def aggregate(*, screen_id: str, expected_commit: str) -> dict[str, Any]:
    """Publish an after-any aggregate even when individual scene jobs are absent."""
    screen_id = _validated_screen_id(screen_id)
    code = _code_snapshot(expected_commit)
    root = evidence_root()
    scene_results = {}
    for scene_id in SCENE_IDS:
        output = _experiment_root(root, screen_id) / "scene_qualifiers" / scene_id
        if not output.exists() and not output.is_symlink():
            scene_results[scene_id] = {
                "reason": "qualifier_output_missing_after_dependency_chain",
                "status": "missing",
            }
            continue
        try:
            validated = _validate_qualifier_output(
                root=root,
                screen_id=screen_id,
                scene_id=scene_id,
                expected_commit=expected_commit,
            )
        except (CandidateScreenError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
            scene_results[scene_id] = {
                "reason": f"invalid_qualifier_output:{type(exc).__name__}:{exc}",
                "status": "invalid",
            }
        else:
            scene_results[scene_id] = {
                "gate": validated["gate"],
                "qualifier_manifest_sha256": validated["bundle"]["manifest_sha256"],
                "status": "validated",
            }
    gate, ranking = build_aggregate_gate(
        screen_id=screen_id,
        code=code,
        scene_results=scene_results,
    )
    gate["evidence_root"] = str(root)
    gate["scene_status_counts"] = {
        status: sum(row["status"] == status for row in ranking)
        for status in ("validated", "missing", "invalid")
    }
    output = _publish_bundle(
        _experiment_root(root, screen_id) / "aggregate",
        manifest_kind="e4_candidate_screen_aggregate_artifacts",
        payloads={
            "gate.json": _json_bytes(gate),
            "ranking.csv": _ranking_csv(ranking),
            "ranking.json": _json_bytes(ranking),
        },
        manifest_fields={
            "code": code,
            "created_utc": _utc_now(),
            "screen_id": screen_id,
        },
    )
    return {
        **gate,
        "aggregate_manifest_sha256": output["manifest_sha256"],
        "aggregate_seal_sha256": output["seal_sha256"],
    }


def _copy_verified_winner_member(
    source: Path, destination: Path, expected: Mapping[str, Any], *, root: Path,
) -> dict[str, Any]:
    """Copy cached bytes only after source identity validation, without overwrite."""
    identity = _identity(source, root=root)
    if identity["sha256"] != expected["sha256"] \
            or identity["size_bytes"] != expected["size_bytes"]:
        raise CandidateScreenError(f"winner export member drift: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    if _identity_without_path(destination) != {
        "sha256": expected["sha256"], "size_bytes": expected["size_bytes"]
    }:
        raise CandidateScreenError(f"winner copy changed: {destination}")
    return identity


def prepare_winner_smoke(
    *, screen_id: str, expected_commit: str, contract_manifest: str | Path,
) -> dict[str, Any]:
    """Prepare the one-target d755 smoke from authenticated existing exports.

    This does not regenerate assets, change eligibility, or replace the original
    pilot population. Failure is published with the complete candidate audit.
    A passing CPU report permits camera diagnostics only, never policy results.
    """
    from run.icra2027 import e4_robust_floor_support_winners as winners
    from robo.eval import e4_camera_scorer_gate as camera_gate
    from robo.eval.e3_factory_materializer import (
        _verify_digest_member, materialize_factory_variant,
        validate_materialized_factory,
    )
    from robo.eval.e4_task_freeze import freeze_task_bundle, validate_task_bundle

    screen_id = _validated_screen_id(screen_id)
    code = _code_snapshot(expected_commit)
    root = evidence_root()
    destination = _experiment_root(root, screen_id) / "harness" / "winner_smoke"
    if destination.exists() or destination.is_symlink():
        raise CandidateScreenError("refusing to overwrite winner smoke")
    contract_path = sealed_cpu._regular_file(
        Path(contract_manifest), root=root, label="winner smoke E0 contract"
    )
    contract = json.loads(contract_path.read_text())
    if contract.get("code", {}).get("commit") != expected_commit \
            or contract.get("code", {}).get("dirty") is not False:
        raise CandidateScreenError("winner smoke needs a clean exact-commit E0 contract")
    digest_input = {
        k: v for k, v in contract.items()
        if k not in {"created_utc", "environment", "contract_sha256"}
    }
    if _canonical_hash(digest_input) != contract.get("contract_sha256"):
        raise CandidateScreenError("winner smoke E0 digest differs")
    registry = winners.read_validated_winner_registry(root)
    scene_id = winners.WINNER_SCENE_ID
    source = root / "outputs/icra2027" / winners.EXTERNAL_SWEEP_ID \
        / "variants" / winners.WINNER_VARIANT_ID
    source_factories = {
        policy: source / "construction_variants" / policy / f"{scene_id}_factory"
        for policy in POLICIES
    }
    factories = {
        policy: _factory_dir(root, screen_id, policy, scene_id)
        for policy in POLICIES
    }
    source_identities = {}
    rows_by_policy = {}
    footprint_replays = {}
    materializations = {}
    for policy, factory in factories.items():
        original_factory = source_factories[policy]
        original_manifest_path = original_factory / "materialization_manifest.json"
        _verify_digest_member(
            original_manifest_path,
            winners.EXPECTED_WINNER_IDENTITY["source_manifest_sha256"][policy],
            "authenticated winner original materialization",
        )
        original_manifest = json.loads(original_manifest_path.read_text())
        # The canonical materializer intentionally binds its validator to its
        # own source commit. Recreate only this cheap artifact contract from
        # the same sealed selections; do not rewrite old producer provenance.
        fresh_manifest = materialize_factory_variant(
            e3_root=EXPECTED_E3_ROOT, scene_id=scene_id, policy_id=policy,
            out=factory,
        )
        for field in ("input_identities", "source_scene", "selected_records",
                      "roster", "output_members"):
            if fresh_manifest[field] != original_manifest[field]:
                raise CandidateScreenError(f"winner rematerialization differs at {field}")
        source_identities[str(original_manifest_path)] = _identity(
            original_manifest_path, root=root
        )
        for relative, expected in original_manifest["output_members"].items():
            path = original_factory / relative
            actual = _identity(path, root=root)
            if {k: actual[k] for k in ("sha256", "size_bytes")} != expected:
                raise CandidateScreenError(f"winner original object member drift: {path}")
            source_identities[str(path)] = actual
        materializations[policy] = validate_materialized_factory(
            factory, expected_scene_id=scene_id, expected_policy_id=policy,
            repository_root=root,
        )
        accepted = materializations[policy]["roster"]["accepted_slots"]
        # Authenticate every generated mesh/XML and object-record member read by
        # the existing planner; source exports remain read-only.
        bundle = source / "policy_evals" / policy
        _validate_bundle(bundle, root=root, expected_kind=
                         "e4_robust_floor_support_policy_eval_artifacts")
        published = json.loads((bundle / "gate.json").read_text())
        for tree, members in published["generated_trees"].items():
            for member in members:
                path = original_factory / member["path"]
                identity = _copy_verified_winner_member(
                    path, factory / member["path"], member, root=root,
                )
                source_identities[str(path)] = identity
        rows_by_policy[policy] = _rows_for_factory(factory)
        if set(rows_by_policy[policy]) != set(accepted):
            raise CandidateScreenError("winner task rows differ from accepted roster")
        footprint_replays[policy] = snapshot_export_footprints(
            factory, expected_slots=accepted
        )
    _validate_export_drift_binding(
        rows_by_policy=rows_by_policy, footprint_replays=footprint_replays,
    )
    plan = plan_paired_region_tasks(
        scene_id=scene_id, rows_by_policy=rows_by_policy,
        footprints={p: r["footprints"] for p, r in footprint_replays.items()},
    )
    selected = [row["task"]["target"] for row in plan["selected"]["tasks"]]
    if selected not in ([], ["obj_05"]):
        raise CandidateScreenError("one-target smoke scope drift")
    suites = {
        policy: _suite_for_plan(
            scene_id=scene_id, scene_xml=factory / "sim_export/scene.xml",
            plan=plan, rows_by_policy=rows_by_policy,
        ) for policy, factory in factories.items()
    }
    candidate_dir = destination.parent / "winner_candidates"
    _publish_bundle(
        candidate_dir, manifest_kind="e4_winner_smoke_candidate_suites",
        payloads={f"{p.lower()}_candidates.json": _json_bytes(s)
                  for p, s in suites.items()},
        manifest_fields={"code": code, "screen_id": screen_id,
                         "study_scope": "one_target_engineering_smoke"},
    )
    task_bundle = None
    metrics = []
    menagerie = None
    if selected:
        task_dir = destination.parent / "winner_task_freeze"
        freeze_task_bundle(
            scene_id=scene_id, a0_factory=factories["A0"],
            a4_factory=factories["A4"],
            a0_candidates=candidate_dir / "a0_candidates.json",
            a4_candidates=candidate_dir / "a4_candidates.json",
            planning_source=PLANNING_SOURCE, max_tasks=None,
            out=task_dir, repository_root=root,
        )
        task_bundle = validate_task_bundle(
            task_dir / "manifest.json", expected_scene_id=scene_id,
            repository_root=root,
        )
        menagerie = camera_gate._menagerie_snapshot(
            EXPECTED_MENAGERIE_ROOT, EXPECTED_MENAGERIE_COMMIT
        )
        metrics = _qualify_task_suites(
            scene_id=scene_id, task_bundle=task_bundle, factories=factories,
            menagerie_root=EXPECTED_MENAGERIE_ROOT,
        )
    for path, expected in source_identities.items():
        if _identity(Path(path), root=root) != expected:
            raise CandidateScreenError(f"winner source changed during smoke: {path}")
    if winners.read_validated_winner_registry(root) != registry:
        raise CandidateScreenError("winner registry changed during smoke")
    passed = len(metrics) == len(POLICIES) * EPISODES and all(r["passed"] for r in metrics)
    gate = {
        "manifest_kind": "e4_one_target_winner_smoke_gate", "code": code,
        "created_utc": _utc_now(), "screen_id": screen_id,
        "study_scope": "one_target_engineering_smoke", "paper_ready": False,
        "gpu_launch_allowed": False, "large_rollout_launch_allowed": False,
        "camera_smoke_prerequisites_pass": passed,
        "candidate_count": len(selected), "planned_cells": 10,
        "checked_cells": len(metrics), "passed_cells": sum(r["passed"] for r in metrics),
        "blocker": (None if passed else "no_valid_region_task" if not selected
                    else "cpu_reset_workspace_qualifier_gate_failed"),
        "source_registry": registry, "contract": _identity(contract_path, root=root),
        "source_identities": source_identities,
        "rematerializations": materializations,
        "task_bundle": task_bundle,
        "menagerie": {k: v for k, v in (menagerie or {}).items() if k != "files"},
    }
    _publish_bundle(
        destination, manifest_kind="e4_one_target_winner_smoke_artifacts",
        payloads={"gate.json": _json_bytes(gate), "planning.json": _json_bytes(plan),
                  "metrics.jsonl": _jsonl_bytes(metrics),
                  "footprint_replays.json": _json_bytes(footprint_replays)},
        manifest_fields={"code": code, "screen_id": screen_id},
    )
    return gate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--screen-id", required=True)
    common.add_argument("--expected-code-commit", required=True)
    winner = subparsers.add_parser("prepare-winner-smoke", parents=[common])
    winner.add_argument("--contract-manifest", required=True)
    materialize_parser = subparsers.add_parser("materialize", parents=[common])
    materialize_parser.add_argument("--scene-id", choices=SCENE_IDS, required=True)
    materialize_parser.add_argument("--policy-id", choices=POLICIES, required=True)
    materialize_parser.add_argument("--e3-root", required=True)
    automatic = subparsers.add_parser("prepare-automatic-candidates", parents=[common])
    automatic.add_argument("--scene-id", required=True)
    automatic.add_argument("--e3-root", required=True)
    automatic.add_argument("--automatic-scene-descriptor", required=True)
    automatic.add_argument("--export", action="store_true")
    automatic.add_argument("--qualification-protocol", help="authenticated complete-discovery CPU budget; preserve full population")
    auto_tasks = subparsers.add_parser("prepare-automatic-tasks", parents=[common])
    auto_tasks.add_argument("--scene-id", required=True)
    prepare = subparsers.add_parser("prepare-scene", parents=[common])
    prepare.add_argument("--scene-id", choices=SCENE_IDS, required=True)
    qualify = subparsers.add_parser("qualify-scene", parents=[common])
    qualify.add_argument("--scene-id", required=True)
    qualify.add_argument("--automatic-population", action="store_true")
    qualify.add_argument("--menagerie-root", required=True)
    qualify.add_argument("--expected-menagerie-commit", required=True)
    subparsers.add_parser("aggregate", parents=[common])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-winner-smoke":
            report = prepare_winner_smoke(
                screen_id=args.screen_id, expected_commit=args.expected_code_commit,
                contract_manifest=args.contract_manifest,
            )
        elif args.command == "materialize":
            report = materialize(
                screen_id=args.screen_id,
                scene_id=args.scene_id,
                policy_id=args.policy_id,
                e3_root=args.e3_root,
                expected_commit=args.expected_code_commit,
            )
        elif args.command == "prepare-automatic-candidates":
            report = prepare_automatic_candidates(screen_id=args.screen_id, scene_id=args.scene_id,
                e3_root=args.e3_root, automatic_scene_descriptor=args.automatic_scene_descriptor,
                expected_commit=args.expected_code_commit, export=args.export,
                qualification_protocol=args.qualification_protocol)
        elif args.command == "prepare-automatic-tasks":
            report = prepare_automatic_task_suites(screen_id=args.screen_id, scene_id=args.scene_id,
                expected_commit=args.expected_code_commit)
        elif args.command == "prepare-scene":
            report = prepare_scene(
                screen_id=args.screen_id,
                scene_id=args.scene_id,
                expected_commit=args.expected_code_commit,
            )
        elif args.command == "qualify-scene":
            report = qualify_scene(
                screen_id=args.screen_id,
                scene_id=args.scene_id,
                expected_commit=args.expected_code_commit,
                menagerie_root=args.menagerie_root,
                expected_menagerie_commit=args.expected_menagerie_commit,
                automatic_population=args.automatic_population,
            )
        else:
            report = aggregate(
                screen_id=args.screen_id,
                expected_commit=args.expected_code_commit,
            )
    except (CandidateScreenError, FileNotFoundError, OSError, subprocess.CalledProcessError,
            TypeError, ValueError) as exc:
        print(f"[e4-candidate-screen] FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
