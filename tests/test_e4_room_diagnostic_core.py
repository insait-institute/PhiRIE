"""Focused geometry/sealing tests for the E4 room diagnostic variants."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from robo.sim import export_mjcf
from robo.sim import room_collision as rc


def _spec(**updates):
    spec = {
        "schema_version": 1,
        "variant_id": "test-hclip-pdemote",
        "scene_id": export_mjcf.C.SCENE_ID,
        "hull_bottom": "clip_scan_aabb_bottom",
        "intrusive_primitive": "demote_to_residual",
        "plane_residual_tol_m": 0.02,
        "support_z": "extent",
        "min_support_area_m2": 0.05,
        "carve_side_top_margin_m": 0.02,
        "lower_carve_margin_m": 0.0,
        "support_clip_offset_m": 0.005,
    }
    spec.update(updates)
    return spec


def test_room_diagnostic_spec_is_exact_and_symlinks_are_rejected(tmp_path):
    spec = _spec()
    assert rc.validate_room_diagnostic_spec(
        spec, expected_scene_id=export_mjcf.C.SCENE_ID
    ) == spec

    with pytest.raises(ValueError, match="keys differ"):
        rc.validate_room_diagnostic_spec({**spec, "unsealed_extension": True})
    with pytest.raises(ValueError, match="outside the freeze"):
        rc.validate_room_diagnostic_spec({**spec, "plane_residual_tol_m": 0.021})
    with pytest.raises(ValueError, match="scene binding differs"):
        rc.validate_room_diagnostic_spec(spec, expected_scene_id="0000000000")

    path = tmp_path / "variant.json"
    raw = json.dumps(spec, sort_keys=True).encode()
    path.write_bytes(raw)
    loaded, digest = export_mjcf.load_room_diagnostic_spec(
        path, export_mjcf.C.SCENE_ID
    )
    assert loaded == spec
    assert digest == hashlib.sha256(raw).hexdigest()
    link = tmp_path / "variant-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="not a regular file"):
        export_mjcf.load_room_diagnostic_spec(link, export_mjcf.C.SCENE_ID)


def test_clipped_convex_hull_has_exact_cap_and_positive_half_volume():
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    clipped, diagnostic = rc.clip_convex_hull_at_z(cube.vertices, 0.0)

    assert clipped.is_watertight
    assert clipped.volume == pytest.approx(4.0)
    assert clipped.bounds[0, 2] == pytest.approx(0.0)
    assert diagnostic["raw_hull_volume_m3"] == pytest.approx(8.0)
    assert diagnostic["retained_volume_fraction"] == pytest.approx(0.5)
    assert diagnostic["cap_vertex_count"] == 4

    unchanged, no_op = rc.clip_convex_hull_at_z(cube.vertices, -1.1)
    assert unchanged.is_watertight
    assert unchanged.volume == pytest.approx(8.0)
    assert no_op["clipped"] is False
    with pytest.raises(ValueError, match="removes all positive volume"):
        rc.clip_convex_hull_at_z(cube.vertices, 1.0)


def _intrusive_support_features():
    source = trimesh.creation.box(extents=(1.0, 1.0, 0.02))
    source.apply_translation((0.0, 0.0, 0.39))
    top = np.flatnonzero(
        np.all(np.isclose(np.asarray(source.vertices)[source.faces, 2], 0.40), axis=1)
    )
    other = np.setdiff1d(np.arange(len(source.faces)), top)
    patch = rc.PlanePatch(
        kind="support",
        z=0.40,
        lo=np.asarray([-0.5, -0.5, 0.40]),
        hi=np.asarray([0.5, 0.5, 0.40]),
        area=1.0,
        residual_m=0.0,
        face_indices=top,
    )
    return rc.RoomFeatures(
        floor=None,
        supports=[patch],
        walls=[],
        residual_mesh=source.submesh([other], append=True),
        source_tris=len(source.faces),
        classified_tris=len(top),
        source_mesh=source,
    )


def _object_exclusion():
    obj = trimesh.creation.box(extents=(0.2, 0.2, 0.22))
    obj.apply_translation((0.0, 0.0, 0.50))
    return rc.make_convex_exclusion(
        obj.vertices,
        name="A0/obj_00/part_00.obj",
        slot="obj_00",
        policy_id="A0",
        margin_m=0.0,
    )


def test_intrusive_primitive_demotion_restores_exact_source_faces(tmp_path):
    features = _intrusive_support_features()
    _assets, geoms, manifest, _collision = rc.emit_room_mjcf(
        features,
        tmp_path,
        exclusions=[_object_exclusion()],
        intrusive_primitive="demote_to_residual",
        collect_diagnostics=True,
    )

    assert all('name="room_support_0"' not in line for line in geoms)
    assert manifest["primitive_intrusions"] == []
    raw = manifest["raw_diagnostics"]
    assert raw["primitive_intrusions_before_policy"] == [{
        "geom": "room_support_0",
        "exclusions": ["A0/obj_00/part_00.obj"],
    }]
    assert raw["demoted_source_face_count"] == 2
    assert raw["residual_face_count_after_demotion"] \
        == raw["residual_face_count_before_demotion"] + 2
    assert raw["primitive_demotions"][0]["source_face_count"] == 2


def test_fail_policy_reports_intrusion_and_support_z_controls_top(tmp_path):
    features = _intrusive_support_features()
    _assets, geoms, manifest, _collision = rc.emit_room_mjcf(
        features,
        tmp_path / "fail",
        exclusions=[_object_exclusion()],
        intrusive_primitive="fail",
        collect_diagnostics=True,
    )
    assert any('name="room_support_0"' in line for line in geoms)
    assert manifest["primitive_intrusions"] == [{
        "geom": "room_support_0",
        "exclusions": ["A0/obj_00/part_00.obj"],
    }]

    patch = rc.PlanePatch(
        kind="support", z=0.40,
        lo=np.asarray([-0.5, -0.5, 0.38]),
        hi=np.asarray([0.5, 0.5, 0.42]),
        area=1.0, residual_m=0.02,
    )
    empty = trimesh.Trimesh()
    simple = rc.RoomFeatures(None, [patch], [], empty, 2, 2)
    _a, _g, _m, extent_mesh = rc.emit_room_mjcf(
        simple, tmp_path / "extent", support_z="extent"
    )
    _a, _g, _m, mean_mesh = rc.emit_room_mjcf(
        simple, tmp_path / "mean", support_z="fitted_mean_20mm"
    )
    assert extent_mesh.bounds[1, 2] == pytest.approx(0.42)
    assert mean_mesh.bounds[1, 2] == pytest.approx(0.40)
    assert mean_mesh.bounds[0, 2] == pytest.approx(0.38)


def test_static_package_round_trip_and_tamper_rejection(tmp_path, monkeypatch):
    spec = _spec()
    spec_sha256 = "a" * 64

    def fake_static(_factories, _spec_value, collision_out_dir):
        assert export_mjcf.C.OUT != original_out
        part = collision_out_dir / "room_part_00.obj"
        part.write_bytes(b"sealed convex mesh")
        sim = export_mjcf.C.OUT / "sim"
        sim.mkdir(parents=True)
        background = sim / "background.obj"
        carve = sim / "background_carve.json"
        background.write_bytes(b"sealed background")
        carve.write_text("{}\n")
        return {
            "asset_xml_lines": [
                f'    <mesh name="room_obstacle_0" file="{part.resolve()}"/>'
            ],
            "geom_xml_lines": [
                '    <geom name="room_obstacle_0" type="mesh" '
                'mesh="room_obstacle_0"/>'
            ],
            "room_report": {"mode": "room", "raw_diagnostics": {}},
            "background_path": background,
            "background_carve_path": carve,
        }

    original_out = export_mjcf.C.OUT
    monkeypatch.setattr(export_mjcf, "_room_static_report", fake_static)
    monkeypatch.setattr(
        export_mjcf,
        "_factory_manifest_sources",
        lambda _factories: {"A0": "0" * 64, "A4": "4" * 64},
    )
    package_dir = tmp_path / "common-static"
    built = export_mjcf.build_common_room_static_package(
        [tmp_path / "A0", tmp_path / "A4"], package_dir, spec, spec_sha256
    )
    assert export_mjcf.C.OUT == original_out
    assert set(built) == {
        "asset_xml_lines", "geom_xml_lines", "room_report",
        "source_manifest_sha256", "static_package_identity", "package_dir",
    }
    assert str(package_dir.resolve()) in built["asset_xml_lines"][0]
    assert set(built["static_package_identity"]) == {
        "manifest_sha256", "content_sha256", "static_xml_sha256",
        "background_sha256",
    }
    replay = export_mjcf.load_common_room_static_package(
        package_dir, spec, spec_sha256
    )
    assert replay["static_package_identity"] == built["static_package_identity"]

    (package_dir / "room_collision" / "room_part_00.obj").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="member identity differs"):
        export_mjcf.load_common_room_static_package(
            package_dir, spec, spec_sha256
        )
