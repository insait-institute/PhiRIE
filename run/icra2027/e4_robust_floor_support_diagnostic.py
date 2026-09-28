#!/usr/bin/env python3
"""Sealed CPU-only E4 robust-floor/support diagnostic.

Four frozen cells are evaluated: robust floor alone (F) and robust floor plus
object-associated robust supports (FS), for scenes 3db0a1c8f3 and d755b3d9d8.
The diagnostic is intentionally incapable of authorizing GPU, large-rollout,
or paper use.  The completed 65-cell sweep is read only as authenticated
control evidence; none of its generated factories is reused.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[2]
BASE_RUNNER = CODE_ROOT / "run/icra2027/e4_collision_diagnostic_sweep.py"
EXPECTED_EVIDENCE_ROOT = Path(os.environ.get("SIMANY_EXPECTED_EVIDENCE_ROOT", "/opt/phirie/evidence/SimAny"))
EXPECTED_E3_ROOT = Path(
    "outputs/icra2027/"
    "icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic"
)
POLICIES = ("A0", "A4")
CONTROL_SWEEP_ID = (
    "icra2027-contract-v1-e4-46bab70bc69c-collision-diagnostic-"
    "20260904T181737Z"
)
CONTROL_CODE_COMMIT = "46bab70bc69cb9be68c7a2eca0e17ce2ee82adfb"
CONTROL_VARIANTS = {
    "3db0a1c8f3": "3db-hclip-pfail",
    "d755b3d9d8": "d755-r20-zmean-hclip",
}


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "e4_collision_diagnostic_sweep_base", BASE_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sealed E4 diagnostic base runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base()
DiagnosticSweepError = _BASE.DiagnosticSweepError
_LEGACY_BUILD_POLICY_GATE = _BASE._build_policy_gate


def _variant(
    variant_id: str, scene_id: str, room_surface_policy: str
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "variant_id": variant_id,
        "scene_id": scene_id,
        "hull_bottom": "clip_scan_aabb_bottom",
        "intrusive_primitive": "fail",
        "plane_residual_tol_m": 0.02,
        "support_z": "fitted_mean_20mm",
        "min_support_area_m2": 0.05,
        "carve_side_top_margin_m": 0.02,
        "lower_carve_margin_m": 0.0,
        "support_clip_offset_m": 0.005,
        "room_surface_policy": room_surface_policy,
    }


VARIANTS = {
    "3db-f": _variant("3db-f", "3db0a1c8f3", "robust_floor_only"),
    "3db-fs": _variant(
        "3db-fs", "3db0a1c8f3", "robust_floor_plus_support"
    ),
    "d755-f": _variant("d755-f", "d755b3d9d8", "robust_floor_only"),
    "d755-fs": _variant(
        "d755-fs", "d755b3d9d8", "robust_floor_plus_support"
    ),
}
VARIANT_IDS = tuple(VARIANTS)

# The base runner provides the already-reviewed materialization, dual-root,
# package, hashing, inventory, and replay helpers.  Its functions resolve
# these globals at call time, so bind them to this smaller frozen matrix.
_BASE.VARIANTS = VARIANTS
_BASE.VARIANT_IDS = VARIANT_IDS

EXPECTED_ROBUST_GUARDS = {
    "upward_normal_z_min_exclusive": 0.65,
    "floor_min_area_m2": 1.0,
    "floor_min_span_xy_m": [1.0, 1.0],
    "floor_max_weighted_p95_residual_m": 0.02,
    "floor_dominance_min_area_ratio": 4.0,
    "support_min_area_m2": 0.01,
    "support_max_weighted_p95_residual_m": 0.02,
    "support_min_projected_aabb_fill": 0.25,
    "support_min_overlap_xy_m2": 0.002,
    "support_min_object_footprint_fraction": 0.1,
    "support_bottom_minus_surface_range_m": [-0.005, 0.03],
    "support_intrusion_tolerance_m": 0.005,
    "support_box_min_half_extent_xy_m": 0.01,
    "support_projected_overlap_method": (
        "exact_triangle_union_clipped_to_object_aabb"
    ),
    "support_box_false_overlap_policy": "reject_component",
}
PROJECTED_OVERLAP_METHOD = "exact_triangle_union_clipped_to_object_aabb"
SUPPORT_ASSOCIATION_KEYS = frozenset(
    {
        "slot",
        "overlap_xy_m2",
        "overlap_object_fraction",
        "scan_bottom_minus_z50_m",
    }
)
BOX_FALSE_OVERLAP_REJECTION_KEYS = frozenset(
    {
        "slot",
        "emitted_box_overlap_xy_m2",
        "emitted_box_overlap_object_fraction",
        "projected_overlap_xy_m2",
        "projected_overlap_object_fraction",
        "scan_bottom_minus_z50_m",
    }
)

POLICY_KIND = "e4_robust_floor_support_policy_eval_artifacts"
COMPARISON_KIND = "e4_robust_floor_support_pair_comparison_artifacts"
AGGREGATE_KIND = "e4_robust_floor_support_aggregate_artifacts"


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _support_rejection_contract(
    *,
    collision: Mapping[str, Any],
    raw: Mapping[str, Any],
    feature: Mapping[str, Any],
    supports: Mapping[str, Any],
    components: Mapping[int, Mapping[str, Any]],
    policy: str,
) -> bool:
    """Validate lossless FS rejection and final emitted-support accounting."""
    room_collision = raw.get("room_collision")
    exclusion = collision.get("collision_exclusion")
    if not isinstance(room_collision, Mapping) or not isinstance(exclusion, Mapping):
        return False
    rejections = room_collision.get("support_rejections")
    demotion = room_collision.get("robust_support_demotion")
    if not isinstance(rejections, list) or not isinstance(demotion, Mapping):
        return False
    if set(demotion) != {
        "rejected_support_count",
        "demoted_source_face_count",
        "residual_rebuild_face_count",
        "face_inventory_validated",
    }:
        return False

    selected_ids = supports["selected_component_ids"]
    rejected_ids = supports["rejected_component_ids"]
    emitted_ids = supports["emitted_component_ids"]
    rejected_set, emitted_set = set(rejected_ids), set(emitted_ids)
    if (
        rejected_set & emitted_set
        or rejected_set | emitted_set != set(selected_ids)
        or [cid for cid in selected_ids if cid in rejected_set] != rejected_ids
        or [cid for cid in selected_ids if cid in emitted_set] != emitted_ids
        or (policy == "robust_floor_only" and rejected_ids)
        or any(components[cid]["selected_as"] != "residual" for cid in rejected_ids)
        or any(components[cid]["selected_as"] != "support" for cid in emitted_ids)
        or {
            cid
            for cid, row in components.items()
            if row["selected_as"] == "support"
        }
        != emitted_set
    ):
        return False

    def associated_slots(component_ids: Sequence[int]) -> list[str]:
        return sorted(
            {
                association["slot"]
                for component_id in component_ids
                for association in components[component_id]["support_associations"]
            }
        )

    if (
        supports["associated_slots"] != associated_slots(selected_ids)
        or supports["emitted_associated_slots"] != associated_slots(emitted_ids)
        or collision.get("n_supports") != len(emitted_ids)
        or collision.get("support_boxes") != len(emitted_ids)
        or supports["selected_count"] != len(selected_ids)
        or supports["rejected_count"] != len(rejected_ids)
        or supports["emitted_count"] != len(emitted_ids)
        or supports["selected_count"]
        != supports["rejected_count"] + supports["emitted_count"]
    ):
        return False

    expected_geoms = [
        f"room_support_{selected_ids.index(component_id)}"
        for component_id in rejected_ids
    ]
    demoted_faces = 0
    for rejection, component_id, expected_geom in zip(
        rejections, rejected_ids, expected_geoms
    ):
        if not isinstance(rejection, Mapping) or set(rejection) != {
            "geom",
            "reason",
            "protected_hull_ids",
            "penetration_depths_m",
            "max_penetration_depth_m",
            "source_face_count",
            "source_area_m2",
        }:
            return False
        protected = rejection["protected_hull_ids"]
        depths = rejection["penetration_depths_m"]
        if (
            rejection["geom"] != expected_geom
            or rejection["reason"] != "penetration_depth_gt_0.005_m"
            or not isinstance(protected, list)
            or not protected
            or any(not isinstance(value, str) or not value for value in protected)
            or protected != sorted(set(protected))
            or not isinstance(depths, list)
            or len(depths) != len(protected)
        ):
            return False
        observed_depths: list[float] = []
        for depth, protected_hull_id in zip(depths, protected):
            if (
                not isinstance(depth, Mapping)
                or set(depth) != {"protected_hull_id", "depth_m"}
                or depth["protected_hull_id"] != protected_hull_id
                or not _is_number(depth["depth_m"])
                or float(depth["depth_m"]) <= 0.005
            ):
                return False
            observed_depths.append(float(depth["depth_m"]))
        component = components[component_id]
        if (
            not _is_number(rejection["max_penetration_depth_m"])
            or float(rejection["max_penetration_depth_m"]) > 0.020
            or float(rejection["max_penetration_depth_m"])
            != max(observed_depths)
            or not _int(rejection["source_face_count"], minimum=1)
            or not _is_number(rejection["source_area_m2"])
            or float(rejection["source_area_m2"]) <= 0.0
            or rejection["source_face_count"] != component["face_count"]
            or rejection["source_area_m2"] != component["area_m2"]
        ):
            return False
        demoted_faces += rejection["source_face_count"]
    residual_before = feature.get("residual_triangle_count")
    if (
        len(rejections) != len(rejected_ids)
        or not _int(demotion.get("rejected_support_count"))
        or demotion.get("rejected_support_count") != len(rejections)
        or not _int(demotion.get("demoted_source_face_count"))
        or demotion.get("demoted_source_face_count") != demoted_faces
        or not _int(residual_before)
        or not _int(demotion.get("residual_rebuild_face_count"))
        or demotion.get("residual_rebuild_face_count")
        != residual_before + demoted_faces
        or demotion.get("face_inventory_validated") is not True
    ):
        return False
    robust_emission = room_collision.get("robust_emission")
    if robust_emission != {
        "floor_representation": "global_plane_only",
        "primitive_intrusion_tolerance_m": 0.005,
        "room_surface_policy": policy,
    }:
        return False
    before_policy = room_collision.get("primitive_intrusions_before_policy")
    expected_before_policy = [
        {
            "geom": rejection["geom"],
            "exclusions": rejection["protected_hull_ids"],
        }
        for rejection in rejections
    ]
    if (
        not isinstance(before_policy, list)
        or before_policy != expected_before_policy
        or exclusion.get("unresolved_intrusion_count") != 0
        or exclusion.get("primitive_intrusions") != []
        or exclusion.get("emitted_coacd_intrusion_count") != 0
    ):
        return False
    return True


def _robust_surface_contract(
    collision: Mapping[str, Any], spec: Mapping[str, Any], discovered: Sequence[str]
) -> bool:
    """Fail-closed validation of the frozen core robust-surface report."""
    try:
        raw = collision["raw_diagnostics"]
        robust = raw["feature_extraction"]["robust_surfaces"]
        feature = raw["feature_extraction"]
        floor = robust["floor"]
        supports = robust["supports"]
        components = robust["components"]
    except (KeyError, TypeError):
        return False
    if not all(
        isinstance(row, Mapping)
        for row in (raw, feature, robust, floor, supports)
    ):
        return False
    if (
        set(robust) != {
            "schema_version",
            "policy",
            "guards",
            "floor",
            "supports",
            "components",
        }
        or robust.get("schema_version") != 1
        or robust.get("policy") != spec["room_surface_policy"]
        or robust.get("guards") != EXPECTED_ROBUST_GUARDS
        or collision.get("floor_source") != "robust_area_weighted_component"
        or collision.get("n_floor") != 1
        or not _is_number(collision.get("floor_z_m"))
    ):
        return False
    required_floor = {
        "candidate_count",
        "eligible_count",
        "selected_component_id",
        "area_m2",
        "z50_m",
        "weighted_p95_residual_m",
        "span_xy_m",
        "runner_up_eligible_area_m2",
        "area_ratio_to_next_eligible",
        "dominance_passed",
        "bounds_xy_m",
    }
    if set(floor) != required_floor:
        return False
    if (
        not _int(floor["candidate_count"], minimum=1)
        or not _int(floor["eligible_count"], minimum=1)
        or not _int(floor["selected_component_id"])
        or not _is_number(floor["area_m2"])
        or float(floor["area_m2"]) < 1.0
        or not _is_number(floor["z50_m"])
        or float(collision["floor_z_m"]) != float(floor["z50_m"])
        or not _is_number(floor["weighted_p95_residual_m"])
        or float(floor["weighted_p95_residual_m"]) > 0.02
        or floor["dominance_passed"] is not True
    ):
        return False
    span = floor["span_xy_m"]
    bounds_xy = floor["bounds_xy_m"]
    if (
        not isinstance(span, list)
        or len(span) != 2
        or any(not _is_number(value) or float(value) < 1.0 for value in span)
        or not isinstance(bounds_xy, list)
        or len(bounds_xy) != 2
        or any(
            not isinstance(point, list)
            or len(point) != 2
            or any(not _is_number(value) for value in point)
            for point in bounds_xy
        )
        or any(
            float(bounds_xy[1][axis]) <= float(bounds_xy[0][axis])
            for axis in range(2)
        )
    ):
        return False
    runner_up = floor["runner_up_eligible_area_m2"]
    ratio = floor["area_ratio_to_next_eligible"]
    if (runner_up is None) != (ratio is None):
        return False
    if runner_up is not None and (
        not _is_number(runner_up)
        or not _is_number(ratio)
        or float(ratio) < 4.0
    ):
        return False
    expected_mode = (
        "legacy_max_residual"
        if spec["room_surface_policy"] == "robust_floor_only"
        else "robust_associated_weighted_p95"
    )
    required_supports = {
        "selection_mode",
        "candidate_count",
        "selected_count",
        "selected_component_ids",
        "rejected_count",
        "emitted_count",
        "rejected_component_ids",
        "emitted_component_ids",
        "associated_slots",
        "emitted_associated_slots",
    }
    if set(supports) != required_supports:
        return False
    selected_ids = supports["selected_component_ids"]
    rejected_ids = supports["rejected_component_ids"]
    emitted_ids = supports["emitted_component_ids"]
    associated = supports["associated_slots"]
    emitted_associated = supports["emitted_associated_slots"]
    if (
        supports["selection_mode"] != expected_mode
        or not _int(supports["candidate_count"])
        or not _int(supports["selected_count"])
        or not _int(supports["rejected_count"])
        or not _int(supports["emitted_count"])
        or not isinstance(selected_ids, list)
        or len(selected_ids) != supports["selected_count"]
        or any(not _int(value) for value in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
        or not isinstance(rejected_ids, list)
        or any(not _int(value) for value in rejected_ids)
        or len(set(rejected_ids)) != len(rejected_ids)
        or not isinstance(emitted_ids, list)
        or any(not _int(value) for value in emitted_ids)
        or len(set(emitted_ids)) != len(emitted_ids)
        or not isinstance(associated, list)
        or any(not isinstance(slot, str) or slot not in discovered for slot in associated)
        or len(set(associated)) != len(associated)
        or associated != sorted(associated)
        or not isinstance(emitted_associated, list)
        or any(
            not isinstance(slot, str) or slot not in discovered
            for slot in emitted_associated
        )
        or len(set(emitted_associated)) != len(emitted_associated)
        or emitted_associated != sorted(emitted_associated)
        or any(slot not in discovered for slot in associated)
        or not isinstance(components, list)
    ):
        return False
    by_id: dict[int, Mapping[str, Any]] = {}
    component_keys = {
        "component_id",
        "face_count",
        "area_m2",
        "z50_m",
        "weighted_p95_residual_m",
        "max_vertex_residual_m",
        "span_xy_m",
        "projected_aabb_fill",
        "projected_overlap_method",
        "box_footprint_safe",
        "box_false_overlap_rejections",
        "eligible_floor",
        "support_associations",
        "selected_as",
        "bounds_xy_m",
    }
    for row in components:
        if not isinstance(row, Mapping) or set(row) != component_keys:
            return False
        cid = row["component_id"]
        if not _int(cid) or cid in by_id or not _int(row["face_count"], minimum=1):
            return False
        if any(
            not _is_number(row[key])
            for key in (
                "area_m2",
                "z50_m",
                "weighted_p95_residual_m",
                "max_vertex_residual_m",
                "projected_aabb_fill",
            )
        ):
            return False
        if (
            float(row["area_m2"]) <= 0.0
            or float(row["weighted_p95_residual_m"]) < 0.0
            or float(row["max_vertex_residual_m"]) < 0.0
            or float(row["projected_aabb_fill"]) < 0.0
        ):
            return False
        if row["selected_as"] not in {"floor", "support", "residual"}:
            return False
        if (
            not isinstance(row["eligible_floor"], bool)
            or row["projected_overlap_method"] != PROJECTED_OVERLAP_METHOD
            or not isinstance(row["box_footprint_safe"], bool)
        ):
            return False
        row_span = row["span_xy_m"]
        row_bounds = row["bounds_xy_m"]
        associations = row["support_associations"]
        box_rejections = row["box_false_overlap_rejections"]
        if (
            not isinstance(row_span, list)
            or len(row_span) != 2
            or any(not _is_number(value) for value in row_span)
            or any(float(value) <= 0.0 for value in row_span)
            or not isinstance(row_bounds, list)
            or len(row_bounds) != 2
            or any(
                not isinstance(point, list)
                or len(point) != 2
                or any(not _is_number(value) for value in point)
                for point in row_bounds
            )
            or any(
                float(row_bounds[1][axis]) <= float(row_bounds[0][axis])
                for axis in range(2)
            )
            or not isinstance(associations, list)
            or not isinstance(box_rejections, list)
        ):
            return False
        association_slots = []
        for association in associations:
            if (
                not isinstance(association, Mapping)
                or set(association) != SUPPORT_ASSOCIATION_KEYS
                or association["slot"] not in discovered
                or any(
                    not _is_number(association[key])
                    for key in (
                        "overlap_xy_m2",
                        "overlap_object_fraction",
                        "scan_bottom_minus_z50_m",
                    )
                )
                or float(association["overlap_xy_m2"]) < 0.002
                or not 0.1
                <= float(association["overlap_object_fraction"])
                <= 1.0
                or not -0.005
                <= float(association["scan_bottom_minus_z50_m"])
                <= 0.03
            ):
                return False
            association_slots.append(association["slot"])
        rejection_slots = []
        for rejection in box_rejections:
            if (
                not isinstance(rejection, Mapping)
                or set(rejection) != BOX_FALSE_OVERLAP_REJECTION_KEYS
                or rejection["slot"] not in discovered
                or any(
                    not _is_number(rejection[key])
                    for key in BOX_FALSE_OVERLAP_REJECTION_KEYS - {"slot"}
                )
                or float(rejection["emitted_box_overlap_xy_m2"]) < 0.002
                or not 0.1
                <= float(rejection["emitted_box_overlap_object_fraction"])
                <= 1.0
                or float(rejection["projected_overlap_xy_m2"]) < 0.0
                or not 0.0
                <= float(rejection["projected_overlap_object_fraction"])
                <= 1.0
                or not -0.005
                <= float(rejection["scan_bottom_minus_z50_m"])
                <= 0.03
                or (
                    float(rejection["projected_overlap_xy_m2"]) >= 0.002
                    and float(rejection["projected_overlap_object_fraction"])
                    >= 0.1
                )
            ):
                return False
            rejection_slots.append(rejection["slot"])
        if (
            association_slots != sorted(set(association_slots))
            or rejection_slots != sorted(set(rejection_slots))
            or set(association_slots) & set(rejection_slots)
            or row["box_footprint_safe"] is not (not box_rejections)
        ):
            return False
        expected_floor_eligible = bool(
            float(row["area_m2"]) >= 1.0
            and float(row_span[0]) >= 1.0
            and float(row_span[1]) >= 1.0
            and float(row["weighted_p95_residual_m"]) <= 0.02
        )
        if row["eligible_floor"] is not expected_floor_eligible:
            return False
        by_id[cid] = row
    floor_row = by_id.get(floor["selected_component_id"])
    eligible_rows = sorted(
        (row for row in by_id.values() if row["eligible_floor"]),
        key=lambda row: (-float(row["area_m2"]), row["component_id"]),
    )
    runner_up_row = eligible_rows[1] if len(eligible_rows) > 1 else None
    expected_runner_up = (
        None if runner_up_row is None else runner_up_row["area_m2"]
    )
    expected_ratio = (
        None
        if runner_up_row is None or floor_row is None
        else float(floor_row["area_m2"]) / float(runner_up_row["area_m2"])
    )
    if (
        len(by_id) != floor["candidate_count"]
        or len(eligible_rows) != floor["eligible_count"]
        or not eligible_rows
        or eligible_rows[0] is not floor_row
        or supports["candidate_count"] != max(len(by_id) - 1, 0)
        or floor_row is None
        or floor_row["selected_as"] != "floor"
        or floor_row["eligible_floor"] is not True
        or floor_row["area_m2"] != floor["area_m2"]
        or floor_row["z50_m"] != floor["z50_m"]
        or floor_row["weighted_p95_residual_m"]
        != floor["weighted_p95_residual_m"]
        or floor_row["span_xy_m"] != floor["span_xy_m"]
        or floor_row["bounds_xy_m"] != floor["bounds_xy_m"]
        or floor["runner_up_eligible_area_m2"] != expected_runner_up
        or floor["area_ratio_to_next_eligible"] != expected_ratio
    ):
        return False
    legacy_min_support_area = spec.get("min_support_area_m2")
    legacy_max_support_residual = spec.get("plane_residual_tol_m")
    if (
        not _is_number(legacy_min_support_area)
        or not _is_number(legacy_max_support_residual)
    ):
        return False
    expected_selected_ids = []
    for row in components:
        if row is floor_row:
            continue
        if spec["room_surface_policy"] == "robust_floor_only":
            selected = (
                float(row["area_m2"]) >= float(legacy_min_support_area)
                and float(row["max_vertex_residual_m"])
                <= float(legacy_max_support_residual)
                and row["box_footprint_safe"] is True
            )
        else:
            selected = (
                float(row["area_m2"]) >= 0.01
                and float(row["weighted_p95_residual_m"]) <= 0.02
                and float(row["projected_aabb_fill"]) >= 0.25
                and bool(row["support_associations"])
                and row["box_footprint_safe"] is True
            )
        if selected:
            expected_selected_ids.append(row["component_id"])
    if selected_ids != expected_selected_ids:
        return False
    if (
        any(cid not in by_id for cid in selected_ids + rejected_ids + emitted_ids)
        or not _support_rejection_contract(
            collision=collision,
            raw=raw,
            feature=feature,
            supports=supports,
            components=by_id,
            policy=spec["room_surface_policy"],
        )
    ):
        return False
    emitted_id_set = set(emitted_ids)
    if any(
        row["selected_as"]
        != (
            "floor"
            if row is floor_row
            else "support"
            if row["component_id"] in emitted_id_set
            else "residual"
        )
        for row in components
    ):
        return False
    selection = collision.get("floor_patch_selection")
    if not isinstance(selection, Mapping) or selection != {
        "expected_z_m": None,
        "tolerance_m": 0.02,
        "detected_z_m": floor["z50_m"],
        "status": "robust_matched",
        "selection_policy": spec["room_surface_policy"],
        "selected_component_id": floor["selected_component_id"],
    }:
        return False
    common_clips = raw.get("floor_aware_common_hulls")
    dynamic_hulls = raw.get("dynamic_object_hulls")
    if (
        not isinstance(common_clips, list)
        or not common_clips
        or not isinstance(dynamic_hulls, list)
        or not dynamic_hulls
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("floor_aware_clip"), Mapping)
            for row in dynamic_hulls
        )
    ):
        return False
    return True


def _publish(
    destination: Path,
    *,
    kind: str,
    gate: Mapping[str, Any],
    code: Mapping[str, Any],
    sweep_id: str,
) -> dict[str, Any]:
    try:
        return _BASE._candidate()._publish_bundle(
            destination,
            manifest_kind=kind,
            payloads={"gate.json": _BASE._json_bytes(dict(gate))},
            manifest_fields={
                "code": dict(code),
                "study_scope": "e4_robust_floor_support_diagnostic_cpu_only",
                "sweep_id": sweep_id,
            },
        )
    except Exception as exc:
        raise DiagnosticSweepError(str(exc)) from exc


def build_common(
    *, sweep_id: str, variant_id: str, e3_root: str | Path, expected_commit: str
) -> dict[str, Any]:
    report = _BASE.build_common(
        sweep_id=sweep_id,
        variant_id=variant_id,
        e3_root=e3_root,
        expected_commit=expected_commit,
    )
    return {
        **report,
        "manifest_kind": "e4_robust_floor_support_common_build",
    }


def _build_policy_gate(
    *, sweep_id: str, variant_id: str, policy_id: str, expected_commit: str
) -> dict[str, Any]:
    gate = _LEGACY_BUILD_POLICY_GATE(
        sweep_id=sweep_id,
        variant_id=variant_id,
        policy_id=policy_id,
        expected_commit=expected_commit,
    )
    root = _BASE._evidence_root()
    spec, _ = _BASE._spec_and_hash(root, sweep_id, variant_id)
    factory = _BASE._factory(root, sweep_id, variant_id, policy_id)
    collision = _BASE._read_json(factory / "sim_export/room_collision_report.json")
    checks = dict(gate["checks"])
    checks.pop("floor_is_scannetpp_world_z0", None)
    checks["robust_surface_contract_matches"] = (
        isinstance(collision, Mapping)
        and _robust_surface_contract(collision, spec, gate["discovered_slots"])
    )
    scientific_pass = all(checks.values())
    return {
        **gate,
        "checks": checks,
        "manifest_kind": "e4_robust_floor_support_policy_eval_gate",
        "scientific_pass": scientific_pass,
        "status": "pass" if scientific_pass else "diagnostic_fail",
    }


def evaluate_policy(
    *, sweep_id: str, variant_id: str, policy_id: str, expected_commit: str
) -> dict[str, Any]:
    sweep_id = _BASE._validated_id(sweep_id)
    variant_id = _BASE._validated_variant(variant_id)
    policy_id = _BASE._validated_policy(policy_id)
    code = _BASE._code_snapshot(expected_commit)
    root = _BASE._evidence_root()
    spec, spec_sha256 = _BASE._spec_and_hash(root, sweep_id, variant_id)
    factories = [
        _BASE._factory(root, sweep_id, variant_id, policy) for policy in POLICIES
    ]
    factory = _BASE._factory(root, sweep_id, variant_id, policy_id)
    _BASE._materialization(
        factory,
        scene_id=spec["scene_id"],
        policy_id=policy_id,
        root=root,
        expected_commit=expected_commit,
    )
    for generated in (factory / "sim", factory / "sim_export"):
        if generated.exists() or generated.is_symlink():
            raise DiagnosticSweepError(f"refusing non-fresh policy export: {generated}")
    package_dir = _BASE._package(root, sweep_id, variant_id)
    _BASE._load_package(package_dir, spec=spec, spec_sha256=spec_sha256)
    subprocess.run(
        _BASE._exporter_argv(
            factories=factories,
            spec_path=_BASE._spec_path(root, sweep_id, variant_id),
            package_in=package_dir,
            test=True,
        ),
        cwd=CODE_ROOT,
        env=_BASE._export_environment(
            factory, root=root, scene_id=spec["scene_id"]
        ),
        check=True,
    )
    gate = _build_policy_gate(
        sweep_id=sweep_id,
        variant_id=variant_id,
        policy_id=policy_id,
        expected_commit=expected_commit,
    )
    bundle = _publish(
        _BASE._policy_bundle(root, sweep_id, variant_id, policy_id),
        kind=POLICY_KIND,
        gate=gate,
        code=code,
        sweep_id=sweep_id,
    )
    return {**gate, "bundle_manifest_sha256": bundle["manifest_sha256"]}


def _build_comparison_gate(
    *, sweep_id: str, variant_id: str, expected_commit: str
) -> dict[str, Any]:
    sweep_id = _BASE._validated_id(sweep_id)
    variant_id = _BASE._validated_variant(variant_id)
    code = _BASE._code_snapshot(expected_commit)
    root = _BASE._evidence_root()
    spec, spec_sha256 = _BASE._spec_and_hash(root, sweep_id, variant_id)
    policies: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for policy in POLICIES:
        try:
            recomputed = _build_policy_gate(
                sweep_id=sweep_id,
                variant_id=variant_id,
                policy_id=policy,
                expected_commit=expected_commit,
            )
            published, bundle = _BASE._validated_bundle_gate(
                _BASE._policy_bundle(root, sweep_id, variant_id, policy),
                root=root,
                expected_kind=POLICY_KIND,
                sweep_id=sweep_id,
                variant_id=variant_id,
                policy_id=policy,
            )
            if not _BASE._candidate()._same_replay_structure(published, recomputed):
                raise DiagnosticSweepError("published policy gate does not replay")
            policies[policy] = {
                **recomputed,
                "bundle_manifest_sha256": bundle["manifest_sha256"],
            }
        except Exception as exc:
            errors[policy] = f"{type(exc).__name__}: {exc}"
    identities = {
        policy: gate["static_package_identity"] for policy, gate in policies.items()
    }
    coverages = {
        policy: gate.get("collision_coverage") for policy, gate in policies.items()
    }
    static_equal = set(identities) == set(POLICIES) and identities["A0"] == identities["A4"]
    coverage_equal = (
        set(coverages) == set(POLICIES)
        and coverages["A0"] is not None
        and coverages["A0"] == coverages["A4"]
    )
    roster_equal = (
        set(policies) == set(POLICIES)
        and policies["A0"]["discovered_slots"]
        == policies["A4"]["discovered_slots"]
    )
    paired = (
        sorted(
            set(policies["A0"]["stable_slots"])
            & set(policies["A4"]["stable_slots"])
        )
        if set(policies) == set(POLICIES)
        else []
    )
    policy_pass = {
        policy: bool(gate["scientific_pass"]) for policy, gate in policies.items()
    }
    comparison_pass = (
        not errors
        and static_equal
        and coverage_equal
        and roster_equal
        and set(policy_pass) == set(POLICIES)
        and all(policy_pass.values())
        and len(paired) >= 2
    )
    return {
        "code": code,
        "collision_coverage_by_policy": coverages,
        "collision_coverage_equal": coverage_equal,
        "conservative_collision_coverage": _BASE._conservative_pair_coverage(policies),
        "comparison_pass": comparison_pass,
        "diagnostic_variant": {"spec": spec, "spec_file_sha256": spec_sha256},
        "discovered_roster_equal": roster_equal,
        "errors": errors,
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_robust_floor_support_pair_comparison_gate",
        "paired_stable_at_least_two": len(paired) >= 2,
        "paired_stable_count": len(paired),
        "paired_stable_slots": paired,
        "paper_ready": False,
        "policies": policies,
        "policy_scientific_pass": policy_pass,
        "scene_id": spec["scene_id"],
        "static_package_identity_by_policy": identities,
        "static_package_identity_equal": static_equal,
        "status": "pass" if comparison_pass else "diagnostic_fail",
        "sweep_id": sweep_id,
        "variant_id": variant_id,
    }


def compare_pair(
    *, sweep_id: str, variant_id: str, expected_commit: str
) -> dict[str, Any]:
    sweep_id = _BASE._validated_id(sweep_id)
    variant_id = _BASE._validated_variant(variant_id)
    code = _BASE._code_snapshot(expected_commit)
    root = _BASE._evidence_root()
    gate = _build_comparison_gate(
        sweep_id=sweep_id,
        variant_id=variant_id,
        expected_commit=expected_commit,
    )
    bundle = _publish(
        _BASE._comparison_bundle(root, sweep_id, variant_id),
        kind=COMPARISON_KIND,
        gate=gate,
        code=code,
        sweep_id=sweep_id,
    )
    return {**gate, "bundle_manifest_sha256": bundle["manifest_sha256"]}


def _control_reference(root: Path, scene_id: str) -> dict[str, Any]:
    variant_id = CONTROL_VARIANTS[scene_id]
    control_root = root / "outputs/icra2027" / CONTROL_SWEEP_ID
    comparison_dir = control_root / "variants" / variant_id / "comparison"
    comparison_bundle = _BASE._candidate()._validate_bundle(
        comparison_dir,
        root=root,
        expected_kind="e4_collision_diagnostic_pair_comparison_artifacts",
    )
    comparison = _BASE._read_json(comparison_dir / "gate.json")
    aggregate_dir = control_root / "aggregate"
    aggregate_bundle = _BASE._candidate()._validate_bundle(
        aggregate_dir,
        root=root,
        expected_kind="e4_collision_diagnostic_sweep_aggregate_artifacts",
    )
    aggregate = _BASE._read_json(aggregate_dir / "gate.json")
    if (
        comparison.get("sweep_id") != CONTROL_SWEEP_ID
        or comparison.get("variant_id") != variant_id
        or comparison.get("scene_id") != scene_id
        or comparison.get("code", {}).get("commit") != CONTROL_CODE_COMMIT
        or aggregate.get("sweep_id") != CONTROL_SWEEP_ID
        or aggregate.get("code", {}).get("commit") != CONTROL_CODE_COMMIT
        or aggregate.get("status") != "complete"
        or any(
            comparison.get(key) is not False
            for key in (
                "gpu_launch_allowed",
                "large_rollout_launch_allowed",
                "paper_ready",
            )
        )
    ):
        raise DiagnosticSweepError("sealed 65-run control binding differs")
    recorded = aggregate.get("variants", {}).get(variant_id)
    if not isinstance(recorded, Mapping):
        raise DiagnosticSweepError("control aggregate lacks comparison cell")
    replay = dict(recorded)
    if replay.pop("bundle_manifest_sha256", None) != comparison_bundle["manifest_sha256"]:
        raise DiagnosticSweepError("control comparison manifest binding differs")
    if not _BASE._candidate()._same_replay_structure(replay, comparison):
        raise DiagnosticSweepError("control comparison does not replay from aggregate")
    return {
        "aggregate_bundle_manifest_sha256": aggregate_bundle["manifest_sha256"],
        "code_commit": CONTROL_CODE_COMMIT,
        "comparison_bundle_manifest_sha256": comparison_bundle["manifest_sha256"],
        "comparison_pass": bool(comparison["comparison_pass"]),
        "paired_stable_count": int(comparison["paired_stable_count"]),
        "paired_stable_slots": list(comparison["paired_stable_slots"]),
        "scene_id": scene_id,
        "sweep_id": CONTROL_SWEEP_ID,
        "variant_id": variant_id,
    }


def _choose_scene_winner(
    scene_id: str, variants: Mapping[str, Mapping[str, Any]]
) -> tuple[str | None, str]:
    prefix = "3db" if scene_id == "3db0a1c8f3" else "d755"
    floor_id, support_id = f"{prefix}-f", f"{prefix}-fs"
    floor = variants.get(floor_id)
    support = variants.get(support_id)
    floor_pass = bool(floor and floor.get("comparison_pass"))
    support_pass = bool(support and support.get("comparison_pass"))
    if floor_pass:
        if support_pass and int(support["paired_stable_count"]) > int(
            floor["paired_stable_count"]
        ):
            return support_id, "fs_strictly_increases_paired_stable_over_passing_f"
        return floor_id, "f_passes_and_fs_does_not_strictly_increase_paired_stable"
    if support_pass:
        return support_id, "f_fails_and_fs_passes"
    return None, "neither_f_nor_fs_passes"


def aggregate(*, sweep_id: str, expected_commit: str) -> dict[str, Any]:
    sweep_id = _BASE._validated_id(sweep_id)
    code = _BASE._code_snapshot(expected_commit)
    root = _BASE._evidence_root()
    variants: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for variant_id in VARIANT_IDS:
        try:
            recomputed = _build_comparison_gate(
                sweep_id=sweep_id,
                variant_id=variant_id,
                expected_commit=expected_commit,
            )
            published, bundle = _BASE._validated_bundle_gate(
                _BASE._comparison_bundle(root, sweep_id, variant_id),
                root=root,
                expected_kind=COMPARISON_KIND,
                sweep_id=sweep_id,
                variant_id=variant_id,
            )
            if not _BASE._candidate()._same_replay_structure(published, recomputed):
                raise DiagnosticSweepError("published comparison does not replay")
            variants[variant_id] = {
                **recomputed,
                "bundle_manifest_sha256": bundle["manifest_sha256"],
            }
        except Exception as exc:
            errors[variant_id] = f"{type(exc).__name__}: {exc}"
    controls: dict[str, dict[str, Any]] = {}
    for scene_id in CONTROL_VARIANTS:
        try:
            controls[scene_id] = _control_reference(root, scene_id)
        except Exception as exc:
            errors[f"control:{scene_id}"] = f"{type(exc).__name__}: {exc}"
    comparison_error_variants = sorted(
        variant_id
        for variant_id, comparison in variants.items()
        if not isinstance(comparison.get("errors"), Mapping)
        or bool(comparison["errors"])
    )
    complete = (
        not errors
        and set(variants) == set(VARIANT_IDS)
        and set(controls) == set(CONTROL_VARIANTS)
        and not comparison_error_variants
    )
    winners: dict[str, str | None] = {}
    decisions: dict[str, dict[str, Any]] = {}
    for scene_id in CONTROL_VARIANTS:
        winner, reason = _choose_scene_winner(scene_id, variants)
        if not complete:
            winner = None
            reason = (
                "comparison_artifact_or_runtime_error"
                if comparison_error_variants
                else "aggregate_or_control_incomplete"
            )
        winners[scene_id] = winner
        prefix = "3db" if scene_id == "3db0a1c8f3" else "d755"
        decisions[scene_id] = {
            "control": controls.get(scene_id),
            "f_paired_stable_count": variants.get(f"{prefix}-f", {}).get(
                "paired_stable_count"
            ),
            "fs_paired_stable_count": variants.get(f"{prefix}-fs", {}).get(
                "paired_stable_count"
            ),
            "reason": reason,
            "winner": winner,
        }
    gate = {
        "code": code,
        "comparison_pass_count": sum(
            bool(row["comparison_pass"]) for row in variants.values()
        ),
        "comparison_error_variants": comparison_error_variants,
        "complete_variant_count": len(variants),
        "control_references": controls,
        "errors": errors,
        "gpu_launch_allowed": False,
        "large_rollout_launch_allowed": False,
        "manifest_kind": "e4_robust_floor_support_aggregate_gate",
        "paper_ready": False,
        "scene_decisions": decisions,
        "scene_winners": winners,
        "selection_rule": [
            "eligible_requires_pair_comparison_pass_and_at_least_two_paired_stable_slots",
            "choose_f_when_f_passes_unless_fs_also_passes_and_strictly_increases_paired_stable",
            "choose_fs_when_f_fails_and_fs_passes",
            "artifact_or_runtime_error_in_any_comparison_forces_incomplete_and_null_winners",
            "sealed_65_run_controls_are_authenticated_references_only",
            "never_release_gpu_large_rollout_or_paper_from_this_diagnostic",
        ],
        "status": "complete" if complete else "incomplete",
        "sweep_id": sweep_id,
        "variant_count": len(VARIANT_IDS),
        "variant_ids": list(VARIANT_IDS),
        "variants": variants,
    }
    bundle = _publish(
        _BASE._aggregate_bundle(root, sweep_id),
        kind=AGGREGATE_KIND,
        gate=gate,
        code=code,
        sweep_id=sweep_id,
    )
    return {**gate, "bundle_manifest_sha256": bundle["manifest_sha256"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sweep-id", required=True)
    common.add_argument("--expected-code-commit", required=True)
    build = subparsers.add_parser("build-common", parents=[common])
    build.add_argument("--variant-id", choices=VARIANT_IDS, required=True)
    build.add_argument("--e3-root", required=True)
    evaluate = subparsers.add_parser("eval-policy", parents=[common])
    evaluate.add_argument("--variant-id", choices=VARIANT_IDS, required=True)
    evaluate.add_argument("--policy-id", choices=POLICIES, required=True)
    compare = subparsers.add_parser("compare-pair", parents=[common])
    compare.add_argument("--variant-id", choices=VARIANT_IDS, required=True)
    subparsers.add_parser("aggregate", parents=[common])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build-common":
            report = build_common(
                sweep_id=args.sweep_id,
                variant_id=args.variant_id,
                e3_root=args.e3_root,
                expected_commit=args.expected_code_commit,
            )
        elif args.command == "eval-policy":
            report = evaluate_policy(
                sweep_id=args.sweep_id,
                variant_id=args.variant_id,
                policy_id=args.policy_id,
                expected_commit=args.expected_code_commit,
            )
        elif args.command == "compare-pair":
            report = compare_pair(
                sweep_id=args.sweep_id,
                variant_id=args.variant_id,
                expected_commit=args.expected_code_commit,
            )
        else:
            report = aggregate(
                sweep_id=args.sweep_id,
                expected_commit=args.expected_code_commit,
            )
    except (
        DiagnosticSweepError,
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        print(
            f"[e4-robust-floor-support] FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
