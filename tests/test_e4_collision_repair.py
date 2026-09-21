import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from robo.sim import export_mjcf
from robo.sim import room_collision as rc
from robo.sim import s7_sim
from robo.eval import e4_candidate_screen as screen


def _box(center=(0.0, 0.0, 0.5), extents=(0.2, 0.2, 0.2)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def test_convex_exclusion_expands_sides_but_preserves_support_boundary():
    exclusion = rc.make_convex_exclusion(
        _box().vertices,
        name="A0/obj_00/part_00.obj",
        slot="obj_00",
        policy_id="A0",
        margin_m=0.02,
        lower_support_margin_m=0.0,
    )
    points = np.asarray([
        [0.115, 0.0, 0.50],  # inside the 20 mm side safety margin
        [0.0, 0.0, 0.401],   # positive-volume object interior
        [0.0, 0.0, 0.400],   # exact real-support contact boundary
        [0.0, 0.0, 0.390],   # below the object: real support is preserved
    ])
    assert rc.points_inside_exclusion(points, exclusion).tolist() == [
        True, True, False, False
    ]


def test_convex_intrusion_is_strict_and_allows_boundary_contact():
    exclusion = rc.make_convex_exclusion(
        _box().vertices,
        name="A0/obj_00/part_00.obj",
        slot="obj_00",
        policy_id="A0",
        margin_m=0.0,
    )
    penetrating = _box(center=(0.0, 0.0, 0.39), extents=(0.4, 0.4, 0.04))
    touching = _box(center=(0.0, 0.0, 0.39), extents=(0.4, 0.4, 0.02))
    assert rc.convex_mesh_intrudes_exclusion(penetrating, exclusion)
    assert not rc.convex_mesh_intrudes_exclusion(touching, exclusion)


def test_support_clip_preserves_table_but_unclipped_hull_audit_detects_bulge():
    hull = _box(center=(0.0, 0.0, 0.45), extents=(0.2, 0.2, 0.30))
    clipped = rc.make_convex_exclusion(
        hull.vertices,
        name="A0/obj_00/part_00.obj",
        slot="obj_00",
        policy_id="A0",
        margin_m=0.02,
        support_clip_z_m=0.40,
    )
    actual = rc.make_convex_exclusion(
        hull.vertices,
        name="A0/obj_00/part_00.obj",
        slot="obj_00",
        policy_id="A0",
        margin_m=0.02,
    )
    table = _box(center=(0.0, 0.0, 0.39), extents=(0.5, 0.5, 0.02))

    assert rc.points_inside_exclusion(
        np.asarray([[0.0, 0.0, 0.35], [0.0, 0.0, 0.401]]), clipped
    ).tolist() == [False, True]
    assert not rc.convex_mesh_intrudes_exclusion(table, clipped)
    assert rc.convex_mesh_intrudes_exclusion(table, actual)


def test_scannetpp_floor_anchor_keeps_high_tabletop_as_support():
    table = _box(center=(0.0, 0.0, 0.79), extents=(1.0, 0.8, 0.02))
    features = rc.extract_room_features(
        table,
        expected_floor_z=0.0,
        floor_z_tol=0.05,
    )

    assert features.floor is None
    assert any(patch.z == pytest.approx(0.80) for patch in features.supports)


def test_room_emitter_reports_positive_volume_support_intrusion(tmp_path):
    empty = trimesh.Trimesh()
    support = rc.PlanePatch(
        kind="support",
        z=0.40,
        lo=np.asarray([-0.5, -0.5, 0.40]),
        hi=np.asarray([0.5, 0.5, 0.40]),
        area=1.0,
        residual_m=0.0,
    )
    features = rc.RoomFeatures(
        floor=None,
        supports=[support],
        walls=[],
        residual_mesh=empty,
        source_tris=2,
        classified_tris=2,
    )
    exclusion = rc.make_convex_exclusion(
        _box(center=(0.0, 0.0, 0.50), extents=(0.2, 0.2, 0.22)).vertices,
        name="A0/obj_00/part_00.obj",
        slot="obj_00",
        policy_id="A0",
        margin_m=0.02,
    )

    _assets, _geoms, manifest, _mesh = rc.emit_room_mjcf(
        features,
        tmp_path,
        exclusions=[exclusion],
    )

    assert manifest["primitive_intrusions"] == [{
        "geom": "room_support_0",
        "exclusions": ["A0/obj_00/part_00.obj"],
    }]


def test_residual_convex_parts_cannot_bridge_back_into_carve(tmp_path):
    protected = _box(center=(0.0, 0.0, 0.5))
    exclusion = rc.make_convex_exclusion(
        protected.vertices,
        name="A0/obj_00/part_00.obj",
        slot="obj_00",
        policy_id="A0",
        margin_m=0.02,
    )
    overlapping_room_debris = _box(center=(0.08, 0.0, 0.5))
    valid_room_obstacle = _box(center=(1.0, 0.0, 0.5))
    residual = trimesh.util.concatenate([overlapping_room_debris, valid_room_obstacle])

    files, parts, stats = rc.coacd_decompose(
        residual,
        tmp_path,
        "room",
        exclusions=[exclusion],
        return_stats=True,
    )

    assert stats == {"candidate_parts": 2, "rejected_intrusions": 1}
    assert len(files) == len(parts) == 1
    assert files[0].name == "room_part_00.obj"
    assert not rc.convex_mesh_intrudes_exclusion(parts[0], exclusion)


def _write_factory(root: Path, policy: str, *, reject_second: bool):
    root.mkdir(parents=True)
    objects = [
        {"index": 0, "label": "mug", "aabb": [[0, 0, 0.4], [0.2, 0.2, 0.6]]},
        {"index": 1, "label": "book", "aabb": [[1, 0, 0.4], [1.2, 0.2, 0.6]]},
    ]
    slots = ["obj_00", "obj_01"]
    accepted = ["obj_00"] if reject_second else slots
    (root / "objects").mkdir()
    (root / "objects" / "objects.json").write_text(json.dumps(objects))
    (root / "materialization_manifest.json").write_text(json.dumps({
        "scene_id": export_mjcf.C.SCENE_ID,
        "policy_id": policy,
        "roster": {"object_slots": slots, "accepted_slots": accepted},
    }))
    for index, slot in enumerate(slots):
        object_dir = root / "objects" / slot
        object_dir.mkdir()
        rejected = reject_second and index == 1
        transform = np.eye(4)
        transform[0, 3] = float(index)
        (object_dir / "aligned.json").write_text(json.dumps({
            "rejected": "e3_policy_abstention" if rejected else None,
            "T": None if rejected else transform.tolist(),
        }))
        if not rejected:
            collision = object_dir / "collision"
            collision.mkdir()
            _box(center=(0.0, 0.0, 0.5)).export(collision / "part_00.obj")
    return objects


def test_common_carve_is_full_discovered_roster_union_of_both_policy_hulls(tmp_path):
    a0 = tmp_path / "A0"
    a4 = tmp_path / "A4"
    objects = _write_factory(a0, "A0", reject_second=False)
    _write_factory(a4, "A4", reject_second=True)

    discovered, hulls, provenance = export_mjcf.load_common_carve_inputs([a0, a4])

    assert discovered == objects
    assert provenance["policy_ids"] == ["A0", "A4"]
    assert {row["slot"] for row in hulls} == {"obj_00", "obj_01"}
    assert {row["policy_id"] for row in hulls} == {"A0", "A4"}
    assert len(hulls) == 3  # A0: both discovered objects; A4: accepted obj_00 only


def test_common_carve_fails_closed_when_policy_discovery_metadata_drift(tmp_path):
    a0 = tmp_path / "A0"
    a4 = tmp_path / "A4"
    _write_factory(a0, "A0", reject_second=False)
    objects = _write_factory(a4, "A4", reject_second=True)
    objects[0]["label"] = "cup"
    (a4 / "objects" / "objects.json").write_text(json.dumps(objects))

    with pytest.raises(ValueError, match="metadata differ"):
        export_mjcf.load_common_carve_inputs([a0, a4])


def test_carve_contract_binds_source_mesh_content_and_is_hull_order_stable(tmp_path):
    mesh = tmp_path / "mesh.ply"
    mesh.write_bytes(b"real mesh identity")
    objects = [
        {"index": 0, "label": "mug", "aabb": [[0, 0, 0], [1, 1, 1]]},
        {"index": 1, "label": "book", "aabb": [[1, 0, 0], [2, 1, 1]]},
    ]
    hulls = [
        {"name": "A0/obj_00/p0", "policy_id": "A0", "slot": "obj_00",
         "support_clip_z_m": 0.405, "vertices": _box().vertices},
        {"name": "A4/obj_01/p0", "policy_id": "A4", "slot": "obj_01",
         "support_clip_z_m": 0.405,
         "vertices": _box(center=(1, 0, 0.5)).vertices},
    ]

    first = s7_sim._carve_contract(
        objects, hulls, margin_m=0.02, lower_support_margin_m=0.0,
        mesh_path=mesh,
    )
    second = s7_sim._carve_contract(
        objects, list(reversed(hulls)), margin_m=0.02,
        lower_support_margin_m=0.0, mesh_path=mesh,
    )

    assert first == second
    assert first["mode"] == "paired_policy_union"
    assert first["geometry_source"] == "transformed_convex_collision_hulls"
    assert first["carved_slots"] == ["obj_00", "obj_01"]
    assert first["source_mesh_sha256"] == (
        "f2c290349e940d1e56c50a49b21b642296909c550a4faac6a76849e4a48ac026"
    )


def _write_minimal_repaired_export(root: Path):
    factory = root / "factory"
    export = factory / "sim_export"
    sim = factory / "sim"
    export.mkdir(parents=True)
    sim.mkdir()
    (sim / "background.obj").write_text("# sealed synthetic background\n")
    (export / "scene.xml").write_text(f"""<mujoco model="repair_contract">
  <compiler meshdir="{factory}"/>
  <worldbody><geom name="floor" type="plane" size="1 1 1"/></worldbody>
</mujoco>
""")
    digest = "a" * 64
    collision = {
        "mode": "room",
        "benchmark": {"finite": True, "steps": 1000, "state_hash": digest},
        "floor_z_m": 0.0,
        "floor_source": "scannetpp_world_frame_z0",
        "floor_patch_selection": {
            "expected_z_m": 0.0,
            "tolerance_m": 0.05,
            "detected_z_m": None,
            "status": "not_detected",
        },
        "background_carve": {
            "schema_version": 1,
            "mode": "paired_policy_union",
            "geometry_source": "transformed_convex_collision_hulls",
            "margin_m": 0.02,
            "lower_support_margin_m": 0.0,
            "support_clip_source": "discovered_scan_aabb_bottom_plus_5mm",
            "discovered_slots": ["obj_00"],
            "carved_slots": ["obj_00"],
            "policy_ids": ["A0", "A4"],
            "hull_count": 2,
            "roster_sha256": digest,
            "source_mesh_sha256": digest,
            "specification_sha256": digest,
            "background_sha256": digest,
        },
        "collision_exclusion": {
            "schema_version": 1,
            "hull_count": 2,
            "coacd_candidate_parts": 3,
            "coacd_parts_rejected_intrusion": 1,
            "emitted_coacd_intrusion_count": 0,
            "primitive_intrusions": [],
            "unresolved_intrusion_count": 0,
        },
    }
    (export / "room_collision_report.json").write_text(json.dumps(collision))
    (export / "mujoco_settle.json").write_text(json.dumps({
        "drift_m": {}, "n": 0, "stable_3cm": 0,
    }))
    (export / "isaac_manifest.json").write_text(json.dumps({"objects": []}))
    return factory, collision


def test_candidate_validator_hard_fails_positive_volume_primitive_intrusion(tmp_path):
    factory, collision = _write_minimal_repaired_export(tmp_path)
    validated = screen.validate_full_room_export(
        factory,
        scene_id=screen.SCENE_IDS[0],
        policy="A0",
        root=tmp_path,
        expected_object_slots=[],
        expected_discovered_slots=["obj_00"],
    )
    assert validated["collision_exclusion"]["unresolved_intrusion_count"] == 0

    collision["collision_exclusion"]["primitive_intrusions"] = [
        {"geom": "room_support_0", "exclusions": ["A0/obj_00/part_00.obj"]}
    ]
    collision["collision_exclusion"]["unresolved_intrusion_count"] = 1
    (factory / "sim_export" / "room_collision_report.json").write_text(
        json.dumps(collision)
    )
    with pytest.raises(screen.CandidateScreenError, match="re-enters"):
        screen.validate_full_room_export(
            factory,
            scene_id=screen.SCENE_IDS[0],
            policy="A0",
            root=tmp_path,
            expected_object_slots=[],
            expected_discovered_slots=["obj_00"],
        )
