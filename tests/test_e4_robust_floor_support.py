"""Focused contract tests for the sealed E4 robust-floor diagnostic core."""

import json

import numpy as np
import pytest
import trimesh

from robo.sim import export_mjcf
from robo.sim import room_collision as rc


def _v1_spec():
    return {
        "schema_version": 1,
        "variant_id": "legacy-v1",
        "scene_id": export_mjcf.C.SCENE_ID,
        "hull_bottom": "raw",
        "intrusive_primitive": "fail",
        "plane_residual_tol_m": 0.025,
        "support_z": "extent",
        "min_support_area_m2": 0.05,
        "carve_side_top_margin_m": 0.02,
        "lower_carve_margin_m": 0.0,
        "support_clip_offset_m": 0.005,
    }


def _v2_spec(policy="robust_floor_only", **updates):
    spec = {
        "schema_version": 2,
        "variant_id": "robust-f",
        "scene_id": export_mjcf.C.SCENE_ID,
        "hull_bottom": "clip_scan_aabb_bottom",
        "intrusive_primitive": "fail",
        "plane_residual_tol_m": 0.02,
        "support_z": "fitted_mean_20mm",
        "min_support_area_m2": 0.05,
        "carve_side_top_margin_m": 0.02,
        "lower_carve_margin_m": 0.0,
        "support_clip_offset_m": 0.005,
        "room_surface_policy": policy,
    }
    spec.update(updates)
    return spec


def _grid(x0, x1, y0, y1, z, cells=1, raised=None):
    """Open, upward triangulated grid; ``raised=(i,j,dz)`` is one outlier."""
    xs = np.linspace(x0, x1, cells + 1)
    ys = np.linspace(y0, y1, cells + 1)
    vertices = np.asarray(
        [[x, y, z] for y in ys for x in xs], dtype=np.float64
    )
    if raised is not None:
        i, j, dz = raised
        vertices[j * (cells + 1) + i, 2] += dz
    faces = []
    for j in range(cells):
        for i in range(cells):
            a = j * (cells + 1) + i
            b = a + 1
            c = a + cells + 1
            d = c + 1
            faces.extend(((a, b, d), (a, d, c)))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def test_v1_is_unchanged_and_v2_is_an_exact_frozen_extension():
    v1 = _v1_spec()
    assert rc.validate_room_diagnostic_spec(v1) == v1
    assert json.dumps(rc.validate_room_diagnostic_spec(v1), sort_keys=True) \
        == json.dumps(v1, sort_keys=True)

    v2 = _v2_spec()
    assert rc.validate_room_diagnostic_spec(v2) == v2
    assert set(v2) == set(v1) | {"room_surface_policy"}
    with pytest.raises(ValueError, match="keys differ"):
        rc.validate_room_diagnostic_spec({**v2, "unsealed": True})
    with pytest.raises(ValueError, match="legacy policies differ"):
        rc.validate_room_diagnostic_spec({**v2, "hull_bottom": "raw"})
    with pytest.raises(ValueError, match="outside the freeze"):
        rc.validate_room_diagnostic_spec({**v2, "plane_residual_tol_m": 0.025})
    with pytest.raises(ValueError, match="must be finite"):
        rc.validate_room_diagnostic_spec({**v2, "support_clip_offset_m": np.nan})


def test_area_weighted_quantiles_ignore_a_small_high_outlier():
    values = [0.0, 0.001, 0.002, 0.100]
    weights = [0.60, 0.30, 0.09, 0.01]
    z50 = rc.area_weighted_quantile(values, weights, 0.50)
    r95 = rc.area_weighted_quantile(
        np.abs(np.asarray(values) - z50), weights, 0.95
    )
    assert z50 == 0.0
    assert r95 == pytest.approx(0.002)
    with pytest.raises(ValueError, match="finite positive"):
        rc.area_weighted_quantile([0.0, np.nan], [1.0, 1.0], 0.5)


def test_projected_triangle_overlap_is_a_union_not_a_sum():
    triangle = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # A repeated projection must not double-count past either the true union or
    # the object footprint.  Folded/repeated scan triangles occur in practice.
    overlap = rc.projected_triangle_aabb_overlap_m2(
        np.asarray([triangle, triangle]), [0.0, 0.0], [1.0, 1.0]
    )
    assert overlap == pytest.approx(0.5)


def _robust_mesh_and_objects():
    floor = _grid(-2.0, 2.0, -2.0, 2.0, 0.0)
    # One 30 mm vertex outlier: max residual fails F, while <5% of the
    # triangulated area is affected so weighted-p95 remains within 20 mm.
    support = _grid(3.0, 4.0, 0.0, 1.0, 0.40, cells=10,
                    raised=(5, 5, 0.030))
    fragment = _grid(5.0, 5.2, 0.0, 0.2, 0.60)
    mesh = trimesh.util.concatenate((floor, support, fragment))
    objects = [{
        "index": 12,
        "aabb": [[3.40, 0.40, 0.405], [3.60, 0.60, 0.70]],
    }]
    return mesh, objects


def test_robust_floor_and_f_fs_support_rules_are_distinct_and_auditable():
    mesh, objects = _robust_mesh_and_objects()
    f_features, f_diag = rc.extract_robust_room_features(
        mesh, objects, "robust_floor_only"
    )
    fs_features, fs_diag = rc.extract_robust_room_features(
        mesh, objects, "robust_floor_plus_support"
    )

    assert f_features.floor.z == pytest.approx(0.0)
    assert fs_features.floor.z == pytest.approx(0.0)
    assert f_diag["floor"]["area_m2"] == pytest.approx(16.0)
    assert f_diag["floor"]["dominance_passed"] is True
    assert f_diag["supports"]["selection_mode"] == "legacy_max_residual"
    assert f_diag["supports"]["selected_count"] == 0
    assert fs_diag["supports"]["selection_mode"] \
        == "robust_associated_weighted_p95"
    assert fs_diag["supports"]["selected_count"] == 1
    assert fs_diag["supports"]["associated_slots"] == ["obj_12"]
    assert len(fs_features.supports) == 1
    selected = next(
        row for row in fs_diag["components"] if row["selected_as"] == "support"
    )
    assert selected["max_vertex_residual_m"] > 0.020
    assert selected["weighted_p95_residual_m"] <= 0.020
    assert selected["projected_aabb_fill"] >= 0.25
    assert selected["support_associations"][0]["slot"] == "obj_12"
    assert all(
        row["selected_as"] == "residual"
        for row in fs_diag["components"]
        if row["bounds_xy_m"][0][0] >= 5.0
    )


def _rectangular_ring(outer, inner, z_top, z_bottom):
    """Small watertight rectangular ring with one genuine central void."""
    outer = np.asarray(outer, dtype=np.float64)
    inner = np.asarray(inner, dtype=np.float64)
    outer_top = np.column_stack((outer, np.full(4, z_top)))
    inner_top = np.column_stack((inner, np.full(4, z_top)))
    outer_bottom = np.column_stack((outer, np.full(4, z_bottom)))
    inner_bottom = np.column_stack((inner, np.full(4, z_bottom)))
    vertices = np.vstack((outer_top, inner_top, outer_bottom, inner_bottom))
    faces = []
    for index in range(4):
        following = (index + 1) % 4
        # Upward top ring and downward bottom ring.
        faces.extend((
            (index, following, 4 + following),
            (index, 4 + following, 4 + index),
            (8 + index, 12 + following, 8 + following),
            (8 + index, 12 + index, 12 + following),
            # Outer wall and inner wall (the latter points into the void).
            (index, 8 + following, following),
            (index, 8 + index, 8 + following),
            (4 + index, 4 + following, 12 + following),
            (4 + index, 12 + following, 12 + index),
        ))
    return trimesh.Trimesh(
        vertices=vertices, faces=np.asarray(faces), process=False
    )


def test_annulus_hole_is_not_falsely_associated_or_filled_by_support_box(
        tmp_path):
    floor = _grid(-2.0, 2.0, -2.0, 2.0, 0.0)
    outer = np.asarray([
        [3.0, 0.0], [4.0, 0.0], [4.0, 1.0], [3.0, 1.0],
    ])
    inner = np.asarray([
        [3.25, 0.25], [3.75, 0.25], [3.75, 0.75], [3.25, 0.75],
    ])
    annulus = _rectangular_ring(outer, inner, 0.40, 0.38)
    assert annulus.is_watertight
    assert len(annulus.faces) < rc.COACD_MIN_FACES
    source = trimesh.util.concatenate((floor, annulus))
    # The object is wholly inside the empty 0.5 x 0.5 m hole.  Component-AABB
    # overlap is 0.04 m2, but true projected-triangle overlap is exactly zero.
    objects = [{
        "index": 7,
        "aabb": [[3.40, 0.40, 0.396], [3.60, 0.60, 0.70]],
    }]
    features, diagnostics = rc.extract_robust_room_features(
        source, objects, "robust_floor_plus_support"
    )
    assert features.supports == []
    candidate = next(
        row for row in diagnostics["components"]
        if row["bounds_xy_m"][0][0] == pytest.approx(3.0)
    )
    assert candidate["projected_overlap_method"] \
        == "exact_triangle_union_clipped_to_object_aabb"
    assert candidate["support_associations"] == []
    assert candidate["box_footprint_safe"] is False
    rejected = candidate["box_false_overlap_rejections"]
    assert [row["slot"] for row in rejected] == ["obj_07"]
    assert rejected[0]["emitted_box_overlap_xy_m2"] == pytest.approx(0.04)
    assert rejected[0]["projected_overlap_xy_m2"] == pytest.approx(0.0)
    assert candidate["selected_as"] == "residual"

    # The rejected surface remains in the lossless residual inventory.  Its
    # <300-face convex-hull fast path would bridge the hole, so the matching
    # protected hull must reject that candidate before publication.
    protected = trimesh.creation.box(extents=(0.20, 0.20, 0.10))
    protected.apply_translation((3.50, 0.50, 0.446))
    exclusion = rc.make_convex_exclusion(
        protected.vertices,
        name="A0/obj_07/part_00.obj",
        slot="obj_07",
        policy_id="A0",
        margin_m=0.0,
    )
    _assets, geoms, manifest, collision_mesh = rc.emit_room_mjcf(
        features,
        tmp_path / "annulus",
        exclusions=[exclusion],
        support_z="fitted_mean_20mm",
        collect_diagnostics=True,
        emit_floor_primitive=False,
        primitive_intrusion_tolerance_m=0.005,
        room_surface_policy="robust_floor_plus_support",
        primitive_exclusions=[exclusion],
    )
    assert manifest["support_boxes"] == 0
    assert all('name="room_support_' not in line for line in geoms)
    assert manifest["coacd_candidate_parts"] == 1
    assert manifest["coacd_parts_rejected_exclusion"] == 1
    assert manifest["coacd_parts"] == 0
    assert manifest["coacd_emitted_intrusions"] == []
    decomposition = manifest["raw_diagnostics"]["decomposition"]
    assert decomposition["convex_hull_fastpath_component_count"] == 1
    assert decomposition["intrusion_rejected_candidate_count"] == 1
    assert decomposition["intrusion_rejected_candidates"][0]["exclusions"] \
        == ["A0/obj_07/part_00.obj"]
    assert len(collision_mesh.faces) == 0


def test_robust_floor_fails_closed_on_missing_or_ambiguous_component():
    too_small = _grid(0.0, 0.9, 0.0, 0.9, 0.0)
    with pytest.raises(ValueError, match="no eligible"):
        rc.extract_robust_room_features(
            too_small, [], "robust_floor_only"
        )

    largest = _grid(0.0, 2.0, 0.0, 2.0, 0.0)
    runner_up = _grid(3.0, 5.0, 0.0, 1.0, 0.4)
    ambiguous = trimesh.util.concatenate((largest, runner_up))
    with pytest.raises(ValueError, match="ambiguous"):
        rc.extract_robust_room_features(
            ambiguous, [], "robust_floor_only"
        )


def _emission_features():
    floor = _grid(-2.0, 2.0, -2.0, 2.0, 0.0)
    support = trimesh.creation.box(extents=(1.0, 1.0, 0.02))
    support.apply_translation((0.0, 0.0, 0.39))
    source = trimesh.util.concatenate((floor, support))
    objects = [{
        "index": 12,
        "aabb": [[-0.1, -0.1, 0.405], [0.1, 0.1, 0.70]],
    }]
    features, _diagnostics = rc.extract_robust_room_features(
        source, objects, "robust_floor_plus_support"
    )
    assert len(features.supports) == 1
    return features


def _exclusion_with_bottom(bottom):
    obj = trimesh.creation.box(extents=(0.2, 0.2, 0.10))
    obj.apply_translation((0.0, 0.0, bottom + 0.05))
    return rc.make_convex_exclusion(
        obj.vertices, name="A0/obj_12/part_00.obj", slot="obj_12",
        policy_id="A0", margin_m=0.0,
    )


def test_robust_emit_has_no_floor_box_and_uses_five_mm_intrusion_guard(tmp_path):
    tolerated_exclusion = _exclusion_with_bottom(0.396)
    assert rc.support_intrusion_depth_m(
        [-0.5, -0.5], [0.5, 0.5], 0.40, tolerated_exclusion
    ) == pytest.approx(0.004, abs=2e-7)
    _assets, geoms, manifest, _mesh = rc.emit_room_mjcf(
        _emission_features(),
        tmp_path / "tolerated",
        exclusions=[tolerated_exclusion],
        support_z="fitted_mean_20mm",
        collect_diagnostics=True,
        emit_floor_primitive=False,
        primitive_intrusion_tolerance_m=0.005,
        room_surface_policy="robust_floor_plus_support",
        primitive_exclusions=[tolerated_exclusion],
    )
    assert all('name="room_floor"' not in line for line in geoms)
    assert sum('name="room_support_0"' in line for line in geoms) == 1
    assert manifest["primitive_intrusions"] == []
    patch = manifest["raw_diagnostics"]["classified_patches"][0]
    assert patch["tolerated_exclusions"] == ["A0/obj_12/part_00.obj"]
    assert manifest["raw_diagnostics"]["robust_emission"] == {
        "floor_representation": "global_plane_only",
        "primitive_intrusion_tolerance_m": 0.005,
        "room_surface_policy": "robust_floor_plus_support",
    }
    assert manifest["raw_diagnostics"]["support_rejections"] == []

    severe_exclusion = _exclusion_with_bottom(0.394)
    _assets, severe_geoms, severe, _mesh = rc.emit_room_mjcf(
        _emission_features(),
        tmp_path / "severe",
        exclusions=[severe_exclusion],
        support_z="fitted_mean_20mm",
        collect_diagnostics=True,
        emit_floor_primitive=False,
        primitive_intrusion_tolerance_m=0.005,
        room_surface_policy="robust_floor_plus_support",
        primitive_exclusions=[severe_exclusion],
    )
    assert all('name="room_support_0"' not in line for line in severe_geoms)
    assert severe["primitive_intrusions"] == []
    rejection = severe["raw_diagnostics"]["support_rejections"]
    assert len(rejection) == 1
    assert rejection[0]["reason"] == "penetration_depth_gt_0.005_m"
    assert rejection[0]["protected_hull_ids"] \
        == ["A0/obj_12/part_00.obj"]
    assert rejection[0]["max_penetration_depth_m"] \
        == pytest.approx(0.006, abs=2e-7)
    assert rejection[0]["source_face_count"] == 2
    demotion = severe["raw_diagnostics"]["robust_support_demotion"]
    assert demotion["rejected_support_count"] == 1
    assert demotion["demoted_source_face_count"] == 2
    assert demotion["residual_rebuild_face_count"] == 12
    assert demotion["face_inventory_validated"] is True


def test_robust_emit_freezes_the_two_cm_support_thickness(tmp_path):
    with pytest.raises(ValueError, match="robust emission contract differs"):
        rc.emit_room_mjcf(
            _emission_features(),
            tmp_path / "wrong-thickness",
            floor_thickness=0.04,
            support_z="fitted_mean_20mm",
            collect_diagnostics=True,
            emit_floor_primitive=False,
            primitive_intrusion_tolerance_m=0.005,
            room_surface_policy="robust_floor_plus_support",
        )


def test_intrusion_guard_uses_min_half_expanded_emitted_box_footprint(tmp_path):
    floor = _grid(-2.0, 2.0, -2.0, 2.0, 0.0)
    narrow_support = _grid(0.0, 0.005, 0.0, 2.0, 0.40)
    residual = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    residual.apply_translation((10.0, 0.0, 0.50))
    source = trimesh.util.concatenate((floor, narrow_support, residual))
    objects = [{
        "index": 0,
        "aabb": [[0.0005, 0.50, 0.405], [0.0045, 1.10, 0.70]],
    }]
    features, diagnostics = rc.extract_robust_room_features(
        source, objects, "robust_floor_plus_support"
    )
    assert diagnostics["supports"]["selected_count"] == 1

    # Raw patch x is [0, 5] mm, but the emitted box has the frozen minimum
    # 10 mm half-extent: x=[-7.5, 12.5] mm.  This hull is therefore outside the
    # patch yet penetrates the actual emitted primitive by 6 mm.
    hull = trimesh.creation.box(extents=(0.004, 0.20, 0.10))
    hull.apply_translation((0.010, 0.70, 0.444))
    exclusion = rc.make_convex_exclusion(
        hull.vertices,
        name="A0/obj_00/part_00.obj",
        slot="obj_00",
        policy_id="A0",
        margin_m=0.0,
    )
    support = features.supports[0]
    assert rc.support_intrusion_depth_m(
        support.lo[:2], support.hi[:2], support.z, exclusion
    ) == 0.0

    _assets, geoms, manifest, _mesh = rc.emit_room_mjcf(
        features,
        tmp_path / "narrow",
        exclusions=[exclusion],
        support_z="fitted_mean_20mm",
        collect_diagnostics=True,
        emit_floor_primitive=False,
        primitive_intrusion_tolerance_m=0.005,
        room_surface_policy="robust_floor_plus_support",
        primitive_exclusions=[exclusion],
    )
    assert all('name="room_support_0"' not in line for line in geoms)
    rejection = manifest["raw_diagnostics"]["support_rejections"]
    assert len(rejection) == 1
    assert rejection[0]["max_penetration_depth_m"] \
        == pytest.approx(0.006, abs=2e-7)
    assert manifest["raw_diagnostics"]["robust_support_demotion"] == {
        "rejected_support_count": 1,
        "demoted_source_face_count": 2,
        "residual_rebuild_face_count": 14,
        "face_inventory_validated": True,
    }


def test_false_overlap_guard_uses_xml_quantized_box_footprint():
    floor = _grid(-2.0, 2.0, -2.0, 2.0, 0.0)
    # center.x and half.x both round upward, moving the parser-visible maximum
    # from 0.020102 m to 0.0202 m.  The very long object makes that sub-mm
    # sliver cross both frozen overlap thresholds despite having zero overlap
    # with the actual source triangles.
    support = _grid(0.0, 0.020102, 0.0, 30.0, 0.40)
    source = trimesh.util.concatenate((floor, support))
    objects = [{
        "index": 3,
        "aabb": [[0.020103, 0.0, 0.405], [0.020603, 30.0, 0.70]],
    }]
    features, diagnostics = rc.extract_robust_room_features(
        source, objects, "robust_floor_only"
    )
    assert features.supports == []
    candidate = next(
        row for row in diagnostics["components"]
        if row["bounds_xy_m"][0][0] == pytest.approx(0.0)
        and row["bounds_xy_m"][1][0] == pytest.approx(0.020102)
    )
    rejected = candidate["box_false_overlap_rejections"]
    assert [row["slot"] for row in rejected] == ["obj_03"]
    assert rejected[0]["emitted_box_overlap_xy_m2"] \
        == pytest.approx((0.0202 - 0.020103) * 30.0)
    assert rejected[0]["emitted_box_overlap_object_fraction"] \
        == pytest.approx((0.0202 - 0.020103) / 0.0005)
    assert rejected[0]["projected_overlap_xy_m2"] == pytest.approx(0.0)
    assert candidate["box_footprint_safe"] is False
    assert candidate["selected_as"] == "residual"


def test_five_mm_gate_uses_xml_quantized_support_top(tmp_path):
    features = _emission_features()
    support = features.supports[0]
    support.z = 0.400051
    lo = np.asarray(support.lo, dtype=np.float64).copy()
    hi = np.asarray(support.hi, dtype=np.float64).copy()
    lo[2] = hi[2] = support.z
    exclusion = _exclusion_with_bottom(0.395071)

    raw_mesh = rc._box_trimesh(
        lo, hi, hang_below=0.02, min_half=0.01
    )
    xml_mesh = rc._box_trimesh(
        lo, hi, hang_below=0.02, min_half=0.01, xml_quantized=True
    )
    raw_depth = rc.support_intrusion_depth_m(
        raw_mesh.bounds[0, :2], raw_mesh.bounds[1, :2],
        float(raw_mesh.bounds[1, 2]), exclusion,
    )
    xml_depth = rc.support_intrusion_depth_m(
        xml_mesh.bounds[0, :2], xml_mesh.bounds[1, :2],
        float(xml_mesh.bounds[1, 2]), exclusion,
    )
    assert raw_mesh.bounds[1, 2] == pytest.approx(0.400051)
    assert xml_mesh.bounds[1, 2] == pytest.approx(0.4001)
    assert raw_depth < 0.005
    assert xml_depth > 0.005

    _assets, geoms, manifest, _mesh = rc.emit_room_mjcf(
        features,
        tmp_path / "xml-rounded-gate",
        exclusions=[exclusion],
        support_z="fitted_mean_20mm",
        collect_diagnostics=True,
        emit_floor_primitive=False,
        primitive_intrusion_tolerance_m=0.005,
        room_surface_policy="robust_floor_plus_support",
        primitive_exclusions=[exclusion],
    )
    assert all('name="room_support_0"' not in line for line in geoms)
    rejection = manifest["raw_diagnostics"]["support_rejections"]
    assert len(rejection) == 1
    assert rejection[0]["max_penetration_depth_m"] == pytest.approx(xml_depth)
    patch = manifest["raw_diagnostics"]["classified_patches"][0]
    assert patch["emitted_bounds_m"][1][2] == pytest.approx(0.4001)


def test_floor_aware_hull_clip_uses_overlap_and_fails_on_empty_volume():
    binding = {
        "floor_z_m": 0.0,
        "bounds_xy_m": np.asarray([[-2.0, -2.0], [2.0, 2.0]]),
        "selected_component_id": 7,
    }
    cube = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    objects = [{"index": 12, "aabb": [[-0.1, -0.1, -0.05],
                                        [0.1, 0.1, 0.2]]}]
    hulls = [{
        "name": "A0/obj_12/part_00.obj",
        "slot": "obj_12",
        "policy_id": "A0",
        "vertices": np.asarray(cube.vertices),
        "support_clip_z_m": -0.045,
    }]
    selected, diagnostics = export_mjcf._clip_common_hulls_to_robust_floor(
        hulls, objects, binding
    )
    assert np.asarray(selected[0]["vertices"])[:, 2].min() == pytest.approx(0.0)
    assert diagnostics[0]["selected_clip_z_m"] == pytest.approx(0.0)
    assert diagnostics[0]["floor_xy_overlap_m2"] > 0.0
    assert diagnostics[0]["clip"]["retained_volume_fraction"] \
        == pytest.approx(0.5)

    below = trimesh.creation.box(extents=(0.2, 0.2, 0.1))
    below.apply_translation((0.0, 0.0, -0.15))
    hulls[0]["vertices"] = np.asarray(below.vertices)
    with pytest.raises(ValueError, match="removed common hull"):
        export_mjcf._clip_common_hulls_to_robust_floor(
            hulls, objects, binding
        )

    detected = 0.36054566696696705
    assert float(export_mjcf._global_floor_z_text(detected, robust=True)) == detected
    assert export_mjcf._global_floor_z_text(0.0, robust=False) == "0.0000"


def test_robust_floor_report_binding_is_strict():
    policy = "robust_floor_plus_support"
    report = {
        "floor_z_m": 0.79,
        "floor_source": "robust_area_weighted_component",
        "n_floor": 1,
        "n_supports": 2,
        "raw_diagnostics": {"feature_extraction": {"robust_surfaces": {
            "schema_version": 1,
            "policy": policy,
            "floor": {
                "selected_component_id": 3,
                "z50_m": 0.79,
                "bounds_xy_m": [[-1.0, -1.0], [1.0, 1.0]],
                "dominance_passed": True,
            },
            "supports": {"selected_count": 2, "emitted_count": 2},
        }}},
    }
    binding = export_mjcf._robust_floor_binding(report, _v2_spec(policy))
    assert binding["floor_z_m"] == pytest.approx(0.79)
    broken = json.loads(json.dumps(report))
    broken["floor_z_m"] = 0.0
    with pytest.raises(ValueError, match="binding differs"):
        export_mjcf._robust_floor_binding(broken, _v2_spec(policy))


def test_static_report_seals_final_support_rejection_accounting(tmp_path, monkeypatch):
    from robo.sim import s7_sim

    source = _emission_features().source_mesh
    objects = [
        {"index": 12, "aabb": [[-0.1, -0.1, 0.405], [0.1, 0.1, 0.70]]},
        {"index": 29, "aabb": [[-0.1, -0.1, 0.20], [0.1, 0.1, 0.60]]},
    ]
    intrusive = trimesh.creation.box(extents=(0.2, 0.2, 0.10))
    intrusive.apply_translation((0.0, 0.0, 0.397))
    hulls = [{
        "name": "A0/obj_29/part_00.obj",
        "slot": "obj_29",
        "policy_id": "A0",
        "vertices": np.asarray(intrusive.vertices),
        "support_clip_z_m": 0.205,
    }]
    sources = {
        "policy_ids": ["A0", "A4"],
        "source_manifest_sha256": {"A0": "0" * 64, "A4": "4" * 64},
        "raw_hull_diagnostics": {"A0": [], "A4": []},
    }
    monkeypatch.setattr(
        export_mjcf, "load_common_carve_inputs",
        lambda _factories, _spec: (objects, hulls, sources),
    )
    monkeypatch.setattr(export_mjcf.C, "load_instances", lambda: [])
    monkeypatch.setattr(export_mjcf.C, "OUT", tmp_path / "scratch")

    def fake_background(_objects, _aligned, _gts, **_kwargs):
        sim = export_mjcf.C.OUT / "sim"
        sim.mkdir(parents=True)
        path = sim / "background.obj"
        source.export(path)
        path.with_name("background_carve.json").write_text("{}\n")
        return path, {
            "schema_version": 1,
            "mode": "paired_policy_union",
            "geometry_source": "transformed_convex_collision_hulls",
            "margin_m": 0.02,
            "lower_support_margin_m": 0.0,
            "support_clip_source": "discovered_scan_aabb_bottom_plus_5mm",
            "background_sha256": "b" * 64,
            "source_mesh_sha256": "s" * 64,
            "specification_sha256": "c" * 64,
            "roster_sha256": "r" * 64,
            "policy_ids": ["A0", "A4"],
            "discovered_slots": ["obj_12", "obj_29"],
            "carved_slots": ["obj_12", "obj_29"],
            "hull_count": 1,
        }

    monkeypatch.setattr(s7_sim, "build_background", fake_background)
    result = export_mjcf._room_static_report(
        [tmp_path / "A0", tmp_path / "A4"],
        _v2_spec("robust_floor_plus_support"),
        tmp_path / "collision",
    )
    report = result["room_report"]
    robust = report["raw_diagnostics"]["feature_extraction"]["robust_surfaces"]
    emission = report["raw_diagnostics"]["room_collision"]
    assert report["floor_source"] == "robust_area_weighted_component"
    assert report["floor_z_m"] == pytest.approx(0.0)
    assert report["n_floor"] == 1
    assert report["n_supports"] == report["support_boxes"] == 0
    assert robust["supports"]["selected_count"] == 1
    assert robust["supports"]["rejected_count"] == 1
    assert robust["supports"]["emitted_count"] == 0
    rejected_id = robust["supports"]["rejected_component_ids"][0]
    rejected = next(row for row in robust["components"]
                    if row["component_id"] == rejected_id)
    assert rejected["selected_as"] == "residual"
    assert emission["support_rejections"][0]["protected_hull_ids"] \
        == ["A0/obj_29/part_00.obj"]
    assert emission["robust_support_demotion"]["face_inventory_validated"] is True
    assert report["collision_exclusion"]["unresolved_intrusion_count"] == 0
    assert all('name="room_floor"' not in line for line in result["geom_xml_lines"])
    assert all('name="room_support_0"' not in line
               for line in result["geom_xml_lines"])
