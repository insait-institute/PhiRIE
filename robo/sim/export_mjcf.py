"""automated_image2sim: export the reconstructed scene to MuJoCo (MJCF) and
an Isaac-Lab-ready manifest, then smoke-test the MJCF headlessly.

MuJoCo notes:
- object collision = the CoACD convex parts (MuJoCo convexifies each mesh
  geom, and the parts are already convex, so this is exact)
- the scan background historically could NOT be a MuJoCo collision mesh (a
  single concave geom gets convexified into a room-sized hull -- verified
  empirically while building robo/sim/room_collision.py: a dropped ball
  rested on the hull bridging a concave channel's two walls instead of
  settling inside it). `--collision-mode room` (default) fixes this the
  same way s6_physics.py already fixes it for objects: convex-decompose the
  carved background mesh (via robo.sim.room_collision, reusing
  s7_sim.build_background's carving) into a floor + validated
  support/wall box primitives + CoACD parts for the residual (obstacles and
  anything not confidently planar), all sharing bit 0 of contype/conaffinity
  with objects/robot/floor -- no more private per-object shims.
  `--collision-mode shim` keeps the ORIGINAL v0 scheme as an explicit
  ablation (plan Task 06/19): background visual-only + one PRIVATE static
  support shim per object on its own greedy-coloured collision channel, so
  it can never touch another object, the robot, or the room. See
  robo/sim/room_collision.py and plan/06_FULL_ROOM_COLLISION_EXPORT.md.
Isaac: per-object URDFs already exist (s6); the manifest lists them with
world poses for Isaac Lab's UrdfConverter / spawner. USD/Isaac export
proper is out of scope per docs/ICRA_RESEARCH_CONTRACT.md -- see
robo/sim/export_usd.py.

Usage: export_mjcf.py [--test] [--collision-mode room|shim]
       (env: SIMANY_SCENE/SIMANY_OUT)
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from agents.core import common as C
from robo.sim import room_collision as rc

SLAB = 0.02      # m, support slab thickness (shim mode)
PAD = 0.10       # m, shim overhang past the object footprint, per side (shim mode)
SAFE = 0.05      # m, clearance two objects need before they may share a channel (shim mode)
SUPPORT_CLIP_OFFSET_M = 0.005
TIMESTEP = 0.002  # s, physics timestep -- shared by the XML <option> and the
                  # --test/benchmark step-count so they can never drift apart
# (cluster_heights() deleted: dead since it was written -- an exact duplicate of
#  pi05_tasks._cluster, and its single-link chaining is the wrong tool here. On
#  c50d2d1d42 tol=0.06 chains a 76 mm spread of object bottoms into one "level",
#  so a per-level box drops objects by up to 74 mm and takes stable_3cm from
#  7/16 to 3/16. The bottoms genuinely span 73 mm on ONE desktop -- that is
#  per-object alignment error, not a clustering tolerance.)

ROOM_STATIC_PACKAGE_SCHEMA_VERSION = 1
ROOM_STATIC_PACKAGE_KIND = "simany_room_static_package"
ROOM_STATIC_PACKAGE_PLACEHOLDER = "__ROOM_STATIC_PACKAGE__"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_slot(meta):
    index = meta.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("background-carve object index must be a non-negative integer")
    return f"obj_{index:02d}"


def load_room_diagnostic_spec(path, expected_scene_id):
    """Load an exact, regular-file diagnostic spec and retain its byte hash."""
    unresolved = Path(path)
    if unresolved.is_symlink():
        raise ValueError(f"room diagnostic spec is not a regular file: {unresolved}")
    path = unresolved.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"room diagnostic spec is not a regular file: {path}")
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"room diagnostic spec is not valid JSON: {path}") from exc
    spec = rc.validate_room_diagnostic_spec(raw, expected_scene_id=expected_scene_id)
    return spec, hashlib.sha256(raw_bytes).hexdigest()


def _factory_carve_inputs(factory, diagnostic_spec=None):
    """Load policy identity + actual transformed collision hulls from a factory."""
    import trimesh

    factory = Path(factory).resolve(strict=True)
    if not factory.is_dir():
        raise ValueError(f"background-carve factory is not a directory: {factory}")
    manifest_path = factory / "materialization_manifest.json"
    objects_path = factory / "objects" / "objects.json"
    if not manifest_path.is_file() or manifest_path.is_symlink() \
            or not objects_path.is_file() or objects_path.is_symlink():
        raise ValueError(f"background-carve factory lacks sealed metadata: {factory}")
    manifest = json.loads(manifest_path.read_text())
    policy_id = manifest.get("policy_id")
    if manifest.get("scene_id") != C.SCENE_ID or policy_id not in {"A0", "A4"}:
        raise ValueError("background-carve factory scene/policy binding differs")
    automatic = manifest.get("schema_version") == 2
    if automatic:
        validate_automatic_export_context(factory, auto_mode=C.env("AUTO"), pipeline_mesh=factory / "derived_mesh.ply")
    objects = json.loads(objects_path.read_text())
    if not isinstance(objects, list):
        raise ValueError("background-carve objects metadata must be a list")
    slots = [_object_slot(meta) for meta in objects]
    roster = manifest.get("roster")
    if not isinstance(roster, dict) or roster.get("object_slots") != slots:
        raise ValueError("background-carve discovered roster differs from materialization")

    hulls = []
    hull_diagnostics = []
    accepted_slots = []
    for meta in objects:
        slot = _object_slot(meta)
        aabb = np.asarray(meta.get("aabb"), dtype=np.float64)
        if aabb.shape != (2, 3) or not np.isfinite(aabb).all() \
                or np.any(aabb[1] <= aabb[0]):
            raise ValueError(f"background-carve scan AABB is invalid: {policy_id}/{slot}")
        support_clip_offset_m = (SUPPORT_CLIP_OFFSET_M if diagnostic_spec is None
                                 else diagnostic_spec["support_clip_offset_m"])
        support_clip_z_m = float(aabb[0, 2] + support_clip_offset_m)
        object_dir = factory / "objects" / slot
        aligned_path = object_dir / "aligned.json"
        if not aligned_path.is_file() or aligned_path.is_symlink():
            raise ValueError(f"background-carve alignment is absent: {policy_id}/{slot}")
        aligned = json.loads(aligned_path.read_text())
        if aligned.get("rejected") is not None:
            continue
        transform = np.asarray(aligned.get("T"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() \
                or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
            raise ValueError(f"background-carve transform is invalid: {policy_id}/{slot}")
        parts = sorted((object_dir / "collision").glob("part_*.obj"))
        if not parts:
            raise ValueError(f"accepted object has no collision hull: {policy_id}/{slot}")
        part_count = 0
        for part in parts:
            if not part.is_file() or part.is_symlink():
                raise ValueError(f"background-carve collision part is invalid: {part}")
            mesh = trimesh.load(part, process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            if len(vertices) < 4 or not np.isfinite(vertices).all():
                continue
            world_raw = vertices @ transform[:3, :3].T + transform[:3, 3]
            world = world_raw
            hull_diagnostic = {
                "name": f"{policy_id}/{slot}/{part.name}",
                "raw_bounds_m": [world_raw.min(axis=0).tolist(),
                                 world_raw.max(axis=0).tolist()],
                "scan_aabb_m": aabb.tolist(),
                "raw_bottom_minus_scan_bottom_m": float(
                    world_raw[:, 2].min() - aabb[0, 2]
                ),
                "selected_bottom_policy": "raw",
            }
            if diagnostic_spec is not None \
                    and diagnostic_spec["hull_bottom"] == "clip_scan_aabb_bottom":
                try:
                    clipped, clip_diagnostic = rc.clip_convex_hull_at_z(
                        world_raw, float(aabb[0, 2]),
                        name=f"{policy_id}/{slot}/{part.name}",
                    )
                except ValueError as exc:
                    hull_diagnostic["status"] = "removed_all"
                    hull_diagnostic["error"] = str(exc)
                    hull_diagnostics.append(hull_diagnostic)
                    if _robust_surface_policy(diagnostic_spec) is not None:
                        raise ValueError(
                            f"diagnostic v2 hull-bottom policy removed "
                            f"{policy_id}/{slot}/{part.name}: {exc}"
                        ) from exc
                    continue
                world = np.asarray(clipped.vertices, dtype=np.float64)
                hull_diagnostic["selected_bottom_policy"] = \
                    "clip_scan_aabb_bottom"
                hull_diagnostic["clip"] = clip_diagnostic
                retained = clip_diagnostic.get("retained_volume_fraction")
                if _robust_surface_policy(diagnostic_spec) is not None \
                        and (isinstance(retained, bool)
                             or not isinstance(retained, (int, float))
                             or not np.isfinite(retained) or retained <= 0.0):
                    raise ValueError(
                        f"diagnostic v2 hull retained no volume: "
                        f"{policy_id}/{slot}/{part.name}"
                    )
            hull_diagnostic["selected_bounds_m"] = [
                world.min(axis=0).tolist(), world.max(axis=0).tolist()
            ]
            hull_diagnostic["status"] = "selected"
            # make_convex_exclusion performs the full-dimensional qhull check.
            try:
                rc.make_convex_exclusion(
                    world, name=f"{policy_id}/{slot}/{part.name}", slot=slot,
                    policy_id=policy_id,
                )
            except ValueError as exc:
                hull_diagnostic["status"] = "rejected_degenerate"
                hull_diagnostic["error"] = str(exc)
                hull_diagnostics.append(hull_diagnostic)
                continue
            hulls.append({
                "name": f"{policy_id}/{slot}/{part.name}",
                "policy_id": policy_id,
                "slot": slot,
                "support_clip_z_m": support_clip_z_m,
                "vertices": world,
                "raw_diagnostic": hull_diagnostic,
            })
            hull_diagnostics.append(hull_diagnostic)
            part_count += 1
        if part_count == 0:
            raise ValueError(f"accepted object has no usable collision hull: {policy_id}/{slot}")
        accepted_slots.append(slot)
    if accepted_slots != roster.get("accepted_slots"):
        raise ValueError("background-carve accepted roster differs from materialization")
    return {
        "factory": factory,
        **({"automatic_scene_sources": {name: manifest["output_members"][name]
            for name in ("derived_mesh.ply", "auto_instances.npz")},
            "accepted_slots": accepted_slots} if automatic else {}),
        "manifest_sha256": _sha256_file(manifest_path),
        "objects": objects,
        "policy_id": policy_id,
        "hulls": hulls,
        "hull_diagnostics": hull_diagnostics,
    }


def load_common_carve_inputs(factories, diagnostic_spec=None):
    """Build one fail-closed discovered-roster/hull union for both E4 arms."""
    rows = [_factory_carve_inputs(factory, diagnostic_spec) for factory in factories]
    if not rows:
        raise ValueError("at least one background-carve factory is required")
    policies = [row["policy_id"] for row in rows]
    if len(policies) != len(set(policies)):
        raise ValueError("background-carve policy factory is duplicated")
    if len(rows) > 1 and set(policies) != {"A0", "A4"}:
        raise ValueError("paired background carve must bind exactly A0 and A4")
    canonical = json.dumps(rows[0]["objects"], sort_keys=True, separators=(",", ":"))
    for row in rows[1:]:
        if json.dumps(row["objects"], sort_keys=True, separators=(",", ":")) != canonical:
            raise ValueError("A0/A4 discovered object metadata differ")
    hulls = [hull for row in rows for hull in row["hulls"]]
    discovered = [_object_slot(meta) for meta in rows[0]["objects"]]
    carved = {hull["slot"] for hull in hulls}
    automatic = any("automatic_scene_sources" in row for row in rows)
    if automatic:
        if any(row.get("automatic_scene_sources") != rows[0].get("automatic_scene_sources") for row in rows):
            raise ValueError("paired automatic static scene sources differ")
        expected_carved = {slot for row in rows for slot in row["accepted_slots"]}
    else:
        expected_carved = set(discovered)
    if carved != expected_carved:
        missing = sorted(expected_carved - carved)
        raise ValueError(f"common hull carve does not cover construction roster: {missing}")
    sources = {
        "policy_ids": sorted(policies),
        "source_manifest_sha256": {
            row["policy_id"]: row["manifest_sha256"] for row in rows
        },
    }
    if automatic:
        sources["automatic_scene_sources"] = rows[0]["automatic_scene_sources"]
    if diagnostic_spec is not None:
        sources["raw_hull_diagnostics"] = {
            row["policy_id"]: row["hull_diagnostics"] for row in rows
        }
    return rows[0]["objects"], hulls, sources


def _canonical_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _member_identity(path):
    path = Path(path)
    return {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _static_xml_sha256(asset_xml_lines, geom_xml_lines):
    return hashlib.sha256(_canonical_json_bytes({
        "asset_xml_lines": list(asset_xml_lines),
        "geom_xml_lines": list(geom_xml_lines),
    })).hexdigest()


def _is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _robust_surface_policy(diagnostic_spec):
    if diagnostic_spec is None \
            or diagnostic_spec.get("schema_version") \
            != rc.ROOM_DIAGNOSTIC_SPEC_SCHEMA_VERSION_V2:
        return None
    return diagnostic_spec["room_surface_policy"]


def _robust_floor_binding(room_report, diagnostic_spec):
    """Fail-closed extraction of the floor contract from a sealed V2 report."""
    policy = _robust_surface_policy(diagnostic_spec)
    if policy is None:
        return None
    if not isinstance(room_report, dict):
        raise ValueError("robust room report is invalid")
    raw = room_report.get("raw_diagnostics")
    feature = raw.get("feature_extraction") if isinstance(raw, dict) else None
    robust = feature.get("robust_surfaces") if isinstance(feature, dict) else None
    floor = robust.get("floor") if isinstance(robust, dict) else None
    supports = robust.get("supports") if isinstance(robust, dict) else None
    if not isinstance(floor, dict) or not isinstance(supports, dict) \
            or robust.get("schema_version") != 1 or robust.get("policy") != policy:
        raise ValueError("robust surface report binding differs")
    floor_z = room_report.get("floor_z_m")
    selected_z = floor.get("z50_m")
    selected_component_id = floor.get("selected_component_id")
    selected_support_count = supports.get("selected_count")
    if isinstance(floor_z, bool) or not isinstance(floor_z, (int, float)) \
            or not np.isfinite(floor_z) or isinstance(selected_z, bool) \
            or not isinstance(selected_z, (int, float)) \
            or not np.isfinite(selected_z) or float(floor_z) != float(selected_z) \
            or room_report.get("floor_source") != "robust_area_weighted_component" \
            or room_report.get("n_floor") != 1 \
            or isinstance(selected_component_id, bool) \
            or not isinstance(selected_component_id, int) \
            or selected_component_id < 0 \
            or isinstance(selected_support_count, bool) \
            or not isinstance(selected_support_count, int) \
            or selected_support_count < 0 \
            or room_report.get("n_supports") != supports.get("emitted_count") \
            or floor.get("dominance_passed") is not True:
        raise ValueError("robust floor report binding differs")
    bounds = np.asarray(floor.get("bounds_xy_m"), dtype=np.float64)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() \
            or np.any(bounds[1] <= bounds[0]):
        raise ValueError("robust floor footprint is invalid")
    return {
        "policy": policy,
        "floor_z_m": float(floor_z),
        "bounds_xy_m": bounds,
        "selected_component_id": selected_component_id,
    }


def _floor_aware_clip_height(aabb, floor_binding):
    """Return frozen clip height and positive XY overlap with robust floor."""
    aabb = np.asarray(aabb, dtype=np.float64)
    if aabb.shape != (2, 3) or not np.isfinite(aabb).all() \
            or np.any(aabb[1] <= aabb[0]):
        raise ValueError("floor-aware object AABB is invalid")
    floor_bounds = np.asarray(floor_binding["bounds_xy_m"], dtype=np.float64)
    overlap_extent = np.maximum(
        np.minimum(aabb[1, :2], floor_bounds[1])
        - np.maximum(aabb[0, :2], floor_bounds[0]),
        0.0,
    )
    overlap = float(overlap_extent[0] * overlap_extent[1])
    clip_z = float(aabb[0, 2])
    if overlap > 0.0:
        clip_z = max(clip_z, float(floor_binding["floor_z_m"]))
    return clip_z, overlap


def _global_floor_z_text(floor_z_m, *, robust=False):
    """Preserve legacy four decimals, but round-trip the detected V2 height."""
    if isinstance(floor_z_m, bool) or not isinstance(floor_z_m, (int, float)) \
            or not np.isfinite(floor_z_m) or not isinstance(robust, bool):
        raise ValueError("global floor height binding is invalid")
    return repr(float(floor_z_m)) if robust else f"{float(floor_z_m):.4f}"


def _clip_common_hulls_to_robust_floor(hulls, objects, floor_binding):
    """Apply the same floor-aware positive-volume guard to common exclusions."""
    object_rows = {_object_slot(meta): meta for meta in objects}
    if len(object_rows) != len(objects):
        raise ValueError("floor-aware object roster is duplicated")
    selected = []
    diagnostics = []
    for row in hulls:
        slot = row["slot"]
        if slot not in object_rows:
            raise ValueError(f"floor-aware hull slot is undiscovered: {slot}")
        aabb = np.asarray(object_rows[slot].get("aabb"), dtype=np.float64)
        clip_z, overlap = _floor_aware_clip_height(aabb, floor_binding)
        try:
            clipped, clip = rc.clip_convex_hull_at_z(
                row["vertices"], clip_z, name=str(row["name"])
            )
        except ValueError as exc:
            raise ValueError(
                f"floor-aware clipping removed common hull {row['name']}: {exc}"
            ) from exc
        retained = clip.get("retained_volume_fraction")
        if isinstance(retained, bool) or not isinstance(retained, (int, float)) \
                or not np.isfinite(retained) or retained <= 0.0:
            raise ValueError(f"floor-aware common hull retained no volume: {row['name']}")
        updated = dict(row)
        updated["vertices"] = np.asarray(clipped.vertices, dtype=np.float64)
        updated["support_clip_z_m"] = max(
            float(row["support_clip_z_m"]),
            float(floor_binding["floor_z_m"]) if overlap > 0.0 else clip_z,
        )
        selected.append(updated)
        diagnostics.append({
            "name": str(row["name"]),
            "slot": str(slot),
            "floor_component_id": int(floor_binding["selected_component_id"]),
            "floor_xy_overlap_m2": overlap,
            "scan_bottom_m": float(aabb[0, 2]),
            "selected_clip_z_m": clip_z,
            "clip": clip,
        })
    return selected, diagnostics


def _factory_manifest_sources(factories):
    """Read only the immutable factory identities needed by a consumer."""
    result = {}
    for factory in factories:
        unresolved = Path(factory)
        if unresolved.is_symlink():
            raise ValueError(f"background-carve factory is a symlink: {factory}")
        factory = unresolved.resolve(strict=True)
        manifest_path = factory / "materialization_manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"background-carve factory lacks sealed metadata: {factory}")
        manifest = json.loads(manifest_path.read_text())
        policy_id = manifest.get("policy_id")
        if manifest.get("scene_id") != C.SCENE_ID or policy_id not in {"A0", "A4"}:
            raise ValueError("background-carve factory scene/policy binding differs")
        if policy_id in result:
            raise ValueError("background-carve policy factory is duplicated")
        result[policy_id] = _sha256_file(manifest_path)
    if set(result) != {"A0", "A4"}:
        raise ValueError("paired background carve must bind exactly A0 and A4")
    return result


def _room_static_report(factories, diagnostic_spec, collision_out_dir, *, source_factory=None):
    """Build the shared static geometry once, returning its exact XML/report."""
    from robo.sim.s7_sim import build_background
    import trimesh

    background_objects, carve_hulls, carve_sources = load_common_carve_inputs(
        factories, diagnostic_spec
    )
    if "automatic_scene_sources" in carve_sources:
        source = C.OUT if source_factory is None else Path(source_factory)
        factory_roots = {Path(factory).resolve() for factory in factories}
        if source.absolute() != source.resolve() or source.resolve() not in factory_roots:
            raise ValueError("automatic static producer must use a paired factory as its active source")
        if validate_automatic_export_context(
                source, auto_mode=C.env("AUTO"), pipeline_mesh=C.PIPELINE_MESH_PLY) is None:
            raise ValueError("automatic static producer lacks its active source closure")
        if source_factory is not None:
            scratch = C.OUT.absolute()
            if scratch != scratch.resolve() or any(
                    scratch == factory or factory in scratch.parents or scratch in factory.parents
                    for factory in factory_roots):
                raise ValueError("automatic static scratch must be outside source factories without symlinks")
        instances = C.load_auto_instances(instances_path=source / "auto_instances.npz",
                                          mesh_path=source / "derived_mesh.ply")
    else:
        if source_factory is not None:
            raise ValueError("explicit static source requires automatic source closure")
        instances = C.load_instances()
    gts = {g["object_id"]: g for g in instances}
    bg_path, background_carve = build_background(
        background_objects,
        [{} for _ in background_objects],
        gts,
        carve_hulls=carve_hulls,
        carve_margin_m=diagnostic_spec["carve_side_top_margin_m"],
        lower_support_margin_m=diagnostic_spec["lower_carve_margin_m"],
        paired_policy_ids=carve_sources["policy_ids"],
        return_report=True,
    )
    if background_carve.get("mode") != "paired_policy_union":
        raise ValueError("explicit E4 background carve did not bind a paired union")
    carve_contract_keys = (
        "background_sha256", "carved_slots", "discovered_slots",
        "geometry_source", "hull_count", "lower_support_margin_m",
        "margin_m", "mode", "policy_ids", "roster_sha256",
        "schema_version", "source_mesh_sha256", "specification_sha256",
        "support_clip_source",
    )
    background_carve_contract = {
        key: background_carve[key] for key in carve_contract_keys
    }
    background_carve_diagnostics = {
        key: value for key, value in background_carve.items()
        if key not in carve_contract_keys
    }
    room_mesh = trimesh.load(bg_path, process=False)
    decimated, decim_stats = rc.decimate_mesh(room_mesh)
    robust_policy = _robust_surface_policy(diagnostic_spec)
    robust_surfaces = None
    floor_aware_hull_diagnostics = []
    exclusion_hulls = carve_hulls
    if robust_policy is None:
        feats = rc.extract_room_features(
            decimated,
            plane_residual_tol=diagnostic_spec["plane_residual_tol_m"],
            min_support_area=diagnostic_spec["min_support_area_m2"],
            expected_floor_z=rc.SCANNETPP_FLOOR_Z_M,
            floor_z_tol=rc.SCANNETPP_FLOOR_TOL_M,
        )
        if feats.floor is not None \
                and abs(feats.floor.z - rc.SCANNETPP_FLOOR_Z_M) \
                > rc.SCANNETPP_FLOOR_TOL_M:
            raise ValueError("room feature extractor misclassified a high support as floor")
        floor_z = 0.0
        floor_source = "scannetpp_world_frame_z0"
        floor_patch_selection = {
            "expected_z_m": rc.SCANNETPP_FLOOR_Z_M,
            "tolerance_m": rc.SCANNETPP_FLOOR_TOL_M,
            "detected_z_m": None if feats.floor is None else float(feats.floor.z),
            "status": "not_detected" if feats.floor is None else "matched",
        }
    else:
        feats, robust_surfaces = rc.extract_robust_room_features(
            decimated,
            background_objects,
            robust_policy,
            legacy_support_residual_tol=diagnostic_spec["plane_residual_tol_m"],
            legacy_min_support_area=diagnostic_spec["min_support_area_m2"],
        )
        floor_z = float(feats.floor.z)
        floor_source = "robust_area_weighted_component"
        robust_floor_binding = {
            "floor_z_m": floor_z,
            "bounds_xy_m": np.asarray(
                robust_surfaces["floor"]["bounds_xy_m"], dtype=np.float64
            ),
            "selected_component_id": robust_surfaces["floor"][
                "selected_component_id"
            ],
        }
        exclusion_hulls, floor_aware_hull_diagnostics = \
            _clip_common_hulls_to_robust_floor(
                carve_hulls, background_objects, robust_floor_binding
            )
        floor_patch_selection = {
            "expected_z_m": None,
            "tolerance_m": rc.ROBUST_SURFACE_P95_MAX_M,
            "detected_z_m": floor_z,
            "status": "robust_matched",
            "selection_policy": robust_policy,
            "selected_component_id": robust_surfaces["floor"][
                "selected_component_id"
            ],
        }
    exclusions = tuple(
        rc.make_convex_exclusion(
            row["vertices"], name=str(row["name"]), slot=str(row["slot"]),
            policy_id=str(row["policy_id"]),
            margin_m=diagnostic_spec["carve_side_top_margin_m"],
            lower_support_margin_m=diagnostic_spec["lower_carve_margin_m"],
        )
        for row in exclusion_hulls
    )
    primitive_exclusions = (tuple(
        rc.make_convex_exclusion(
            row["vertices"], name=str(row["name"]), slot=str(row["slot"]),
            policy_id=str(row["policy_id"]), margin_m=0.0,
            lower_support_margin_m=0.0,
        )
        for row in exclusion_hulls
    ) if robust_policy is not None else None)
    room_assets, static_geoms, room_manifest, collision_mesh = rc.emit_room_mjcf(
        feats,
        out_dir=collision_out_dir,
        exclusions=exclusions,
        intrusive_primitive=diagnostic_spec["intrusive_primitive"],
        support_z=diagnostic_spec["support_z"],
        collect_diagnostics=True,
        emit_floor_primitive=robust_policy is None,
        primitive_intrusion_tolerance_m=(
            rc.ROBUST_SUPPORT_INTRUSION_TOL_M if robust_policy is not None else 0.0
        ),
        room_surface_policy=robust_policy,
        primitive_exclusions=primitive_exclusions,
    )
    if robust_policy is not None:
        support_raw = room_manifest.get("raw_diagnostics", {})
        support_rejections = support_raw.get("support_rejections")
        support_demotion = support_raw.get("robust_support_demotion")
        selected_ids = robust_surfaces["supports"]["selected_component_ids"]
        if not isinstance(support_rejections, list) \
                or not isinstance(support_demotion, dict) \
                or len(selected_ids) != robust_surfaces["supports"]["selected_count"]:
            raise ValueError("robust support rejection diagnostics differ")
        rejected_indices = []
        for rejection in support_rejections:
            geom = rejection.get("geom") if isinstance(rejection, dict) else None
            prefix = "room_support_"
            if not isinstance(geom, str) or not geom.startswith(prefix) \
                    or not geom[len(prefix):].isdigit():
                raise ValueError("robust support rejection identity is invalid")
            index = int(geom[len(prefix):])
            if index >= len(selected_ids) or index in rejected_indices:
                raise ValueError("robust support rejection index differs")
            rejected_indices.append(index)
        rejected_indices.sort()
        rejected_ids = [selected_ids[index] for index in rejected_indices]
        emitted_ids = [component_id for index, component_id in enumerate(selected_ids)
                       if index not in set(rejected_indices)]
        component_rows = {
            row["component_id"]: row for row in robust_surfaces["components"]
        }
        if len(component_rows) != len(robust_surfaces["components"]) \
                or any(component_id not in component_rows
                       for component_id in selected_ids):
            raise ValueError("robust support component inventory differs")
        for component_id in rejected_ids:
            if component_rows[component_id].get("selected_as") != "support":
                raise ValueError("rejected robust support was not selected")
            component_rows[component_id]["selected_as"] = "residual"
        for component_id in emitted_ids:
            if component_rows[component_id].get("selected_as") != "support":
                raise ValueError("emitted robust support selection differs")
        emitted_associated_slots = sorted({
            association["slot"]
            for component_id in emitted_ids
            for association in component_rows[component_id]["support_associations"]
        })
        robust_surfaces["supports"].update({
            "rejected_count": int(len(rejected_ids)),
            "emitted_count": int(len(emitted_ids)),
            "rejected_component_ids": rejected_ids,
            "emitted_component_ids": emitted_ids,
            "emitted_associated_slots": emitted_associated_slots,
        })
        if room_manifest["support_boxes"] != len(emitted_ids) \
                or support_demotion.get("rejected_support_count") \
                != len(rejected_ids) \
                or support_demotion.get("demoted_source_face_count") \
                != sum(rejection["source_face_count"]
                       for rejection in support_rejections) \
                or support_demotion.get("residual_rebuild_face_count") \
                != support_raw.get("residual_face_count_before_demotion", 0) \
                + support_demotion.get("demoted_source_face_count", -1) \
                or support_demotion.get("face_inventory_validated") is not True \
                or (robust_policy == "robust_floor_only" and rejected_ids):
            raise ValueError("robust support emission accounting differs")
        unresolved = (
            len(room_manifest["primitive_intrusions"])
            + len(room_manifest["coacd_emitted_intrusions"])
        )
        if unresolved:
            raise ValueError(
                f"robust room collision has {unresolved} unresolved intrusion(s)"
            )
        if any('name="room_floor"' in line for line in static_geoms):
            raise ValueError("robust floor was emitted as a coincident room box")
    coverage = rc.collision_coverage_metrics(room_mesh, collision_mesh)
    primitive_intrusions = room_manifest["primitive_intrusions"]
    emitted_coacd_intrusions = room_manifest["coacd_emitted_intrusions"]
    collision_exclusion = {
        "schema_version": 1,
        "hull_count": len(exclusions),
        "coacd_candidate_parts": room_manifest["coacd_candidate_parts"],
        "coacd_parts_rejected_intrusion": room_manifest[
            "coacd_parts_rejected_exclusion"
        ],
        "emitted_coacd_intrusion_count": len(emitted_coacd_intrusions),
        "primitive_intrusions": primitive_intrusions,
        "unresolved_intrusion_count": (
            len(primitive_intrusions) + len(emitted_coacd_intrusions)
        ),
    }
    room_report = {
        "mode": "room", **room_manifest, **decim_stats,
        "coverage": coverage,
        "n_floor": 1 if feats.floor else 0,
        "n_supports": (room_manifest["support_boxes"]
                       if robust_policy is not None else len(feats.supports)),
        "n_walls": len(feats.walls),
        "floor_z_m": floor_z,
        "floor_source": floor_source,
        "floor_patch_selection": floor_patch_selection,
        "background_carve": background_carve_contract,
        "background_carve_diagnostics": background_carve_diagnostics,
        "background_carve_sources": carve_sources,
        "collision_exclusion": collision_exclusion,
        "raw_diagnostics": {
            "common_carve_hulls": carve_sources.get("raw_hull_diagnostics", {}),
            **({"floor_aware_common_hulls": floor_aware_hull_diagnostics}
               if robust_policy is not None else {}),
            "room_collision": room_manifest.get("raw_diagnostics", {}),
            "feature_extraction": {
                "plane_residual_tol_m": diagnostic_spec["plane_residual_tol_m"],
                "min_support_area_m2": diagnostic_spec["min_support_area_m2"],
                "source_triangle_count": int(feats.source_tris),
                "classified_triangle_count": int(feats.classified_tris),
                "residual_triangle_count": int(len(feats.residual_mesh.faces)),
                **({"robust_surfaces": robust_surfaces}
                   if robust_policy is not None else {}),
            },
        },
    }
    return {
        "asset_xml_lines": room_assets,
        "geom_xml_lines": static_geoms,
        "room_report": room_report,
        "background_path": Path(bg_path),
        "background_carve_path": Path(bg_path).with_name("background_carve.json"),
    }


def build_common_room_static_package(
        factories, package_dir, diagnostic_spec, diagnostic_spec_sha256):
    """Build and atomically seal one static package shared by A0 and A4."""
    diagnostic_spec = rc.validate_room_diagnostic_spec(
        diagnostic_spec, expected_scene_id=C.SCENE_ID
    )
    if not _is_sha256(diagnostic_spec_sha256):
        raise ValueError("room diagnostic spec file hash is invalid")
    factories = tuple(factories)
    if len(factories) != 2:
        raise ValueError("common static package requires exactly two factories")
    package_dir = Path(package_dir)
    if package_dir.exists() or package_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite static package: {package_dir}")
    package_dir = package_dir.absolute()
    if package_dir.resolve() != package_dir:
        raise ValueError("room static package path cannot contain symlinks")
    for factory in factories:
        factory = Path(factory).absolute()
        if package_dir == factory or factory in package_dir.parents:
            raise ValueError("room static package must be outside source factories")
    package_dir.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_sha256 = _factory_manifest_sources(factories)

    with tempfile.TemporaryDirectory(
            prefix=f".{package_dir.name}.build-", dir=package_dir.parent) as tmp_name:
        tmp = Path(tmp_name)
        stage = tmp / "package"
        scratch = tmp / "scratch"
        collision_dir = stage / "room_collision"
        collision_dir.mkdir(parents=True)
        scratch.mkdir()
        previous_out = C.OUT
        automatic = any(json.loads((Path(factory) / "materialization_manifest.json").read_text())
                        .get("schema_version") == 2 for factory in factories
                        if (Path(factory) / "materialization_manifest.json").is_file())
        try:
            # s7_sim.build_background has a legacy C.OUT output contract.  A
            # private scratch root keeps the shared builder strictly read-only
            # with respect to both materialized policy factories.
            C.OUT = scratch
            if automatic:
                built = _room_static_report(factories, diagnostic_spec, collision_dir,
                                            source_factory=previous_out)
            else:
                built = _room_static_report(factories, diagnostic_spec, collision_dir)
        finally:
            C.OUT = previous_out

        shutil.copyfile(built["background_path"], stage / "background.obj")
        shutil.copyfile(
            built["background_carve_path"], stage / "background_carve.json"
        )
        stage_prefix = str(collision_dir.resolve())
        logical_assets = [
            line.replace(
                stage_prefix,
                f"{ROOM_STATIC_PACKAGE_PLACEHOLDER}/room_collision",
            )
            for line in built["asset_xml_lines"]
        ]
        if any(stage_prefix in line for line in logical_assets):
            raise ValueError("static asset XML retained a transient build path")
        diagnostic_variant = {
            "spec": diagnostic_spec,
            "spec_file_sha256": diagnostic_spec_sha256,
        }
        room_report = dict(built["room_report"])
        room_report["diagnostic_variant"] = diagnostic_variant
        payload = {
            "schema_version": ROOM_STATIC_PACKAGE_SCHEMA_VERSION,
            "scene_id": C.SCENE_ID,
            "diagnostic_variant": diagnostic_variant,
            "asset_xml_lines": logical_assets,
            "geom_xml_lines": list(built["geom_xml_lines"]),
            "room_report": room_report,
        }
        (stage / "payload.json").write_bytes(_canonical_json_bytes(payload) + b"\n")

        member_paths = sorted(path for path in stage.rglob("*") if path.is_file())
        members = {
            path.relative_to(stage).as_posix(): _member_identity(path)
            for path in member_paths
        }
        content_sha256 = hashlib.sha256(_canonical_json_bytes(members)).hexdigest()
        replay_sources = _factory_manifest_sources(factories)
        if replay_sources != source_manifest_sha256:
            raise ValueError("source factory manifests changed during static build")
        manifest = {
            "schema_version": ROOM_STATIC_PACKAGE_SCHEMA_VERSION,
            "kind": ROOM_STATIC_PACKAGE_KIND,
            "scene_id": C.SCENE_ID,
            "diagnostic_variant": diagnostic_variant,
            "source_manifest_sha256": source_manifest_sha256,
            "members": members,
            "content_sha256": content_sha256,
            "static_xml_sha256": _static_xml_sha256(
                logical_assets, built["geom_xml_lines"]
            ),
            "background_sha256": members["background.obj"]["sha256"],
        }
        manifest_bytes = _canonical_json_bytes(manifest) + b"\n"
        (stage / "manifest.json").write_bytes(manifest_bytes)
        seal = {
            "schema_version": ROOM_STATIC_PACKAGE_SCHEMA_VERSION,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        (stage / "seal.json").write_bytes(_canonical_json_bytes(seal) + b"\n")
        # Validate before publication so an internal serialization error can
        # never leave a seemingly sealed but unusable target behind.
        load_common_room_static_package(
            stage, diagnostic_spec, diagnostic_spec_sha256
        )
        os.rename(stage, package_dir)

    return load_common_room_static_package(
        package_dir, diagnostic_spec, diagnostic_spec_sha256
    )


def _load_exact_json(path, label):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file")
    try:
        return json.loads(path.read_bytes())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{label} is invalid JSON") from exc


def load_common_room_static_package(
        package_dir, diagnostic_spec, diagnostic_spec_sha256, *, expected_scene_id=None):
    """Verify every sealed byte and return path-rendered shared static XML."""
    scene_id = C.SCENE_ID if expected_scene_id is None else expected_scene_id
    diagnostic_spec = rc.validate_room_diagnostic_spec(
        diagnostic_spec, expected_scene_id=scene_id
    )
    if not _is_sha256(diagnostic_spec_sha256):
        raise ValueError("room diagnostic spec file hash is invalid")
    unresolved = Path(package_dir)
    if unresolved.is_symlink():
        raise ValueError("room static package is a symlink")
    package_dir = unresolved.resolve(strict=True)
    if not package_dir.is_dir():
        raise ValueError("room static package is not a directory")

    manifest_path = package_dir / "manifest.json"
    seal_path = package_dir / "seal.json"
    manifest = _load_exact_json(manifest_path, "room static package manifest")
    seal = _load_exact_json(seal_path, "room static package seal")
    expected_variant = {
        "spec": diagnostic_spec,
        "spec_file_sha256": diagnostic_spec_sha256,
    }
    manifest_keys = {
        "schema_version", "kind", "scene_id", "diagnostic_variant",
        "source_manifest_sha256", "members", "content_sha256",
        "static_xml_sha256", "background_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != manifest_keys:
        raise ValueError("room static package manifest schema differs")
    if manifest["schema_version"] != ROOM_STATIC_PACKAGE_SCHEMA_VERSION \
            or manifest["kind"] != ROOM_STATIC_PACKAGE_KIND \
            or manifest["scene_id"] != scene_id \
            or manifest["diagnostic_variant"] != expected_variant:
        raise ValueError("room static package binding differs")
    source_manifest_sha256 = manifest["source_manifest_sha256"]
    if not isinstance(source_manifest_sha256, dict) \
            or set(source_manifest_sha256) != {"A0", "A4"} \
            or any(not _is_sha256(value)
                   for value in source_manifest_sha256.values()):
        raise ValueError("room static package source identity schema differs")
    if not isinstance(seal, dict) \
            or set(seal) != {"schema_version", "manifest_sha256"} \
            or seal["schema_version"] != ROOM_STATIC_PACKAGE_SCHEMA_VERSION:
        raise ValueError("room static package seal schema differs")
    manifest_sha256 = _sha256_file(manifest_path)
    if seal["manifest_sha256"] != manifest_sha256:
        raise ValueError("room static package manifest seal differs")

    members = manifest["members"]
    if not isinstance(members, dict) or not members:
        raise ValueError("room static package member inventory is invalid")
    actual_members = set()
    actual_dirs = set()
    for path in package_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"room static package contains a symlink: {path}")
        if path.is_dir():
            actual_dirs.add(path.relative_to(package_dir).as_posix())
            continue
        if path.is_file():
            rel = path.relative_to(package_dir).as_posix()
            if rel not in {"manifest.json", "seal.json"}:
                actual_members.add(rel)
            continue
        raise ValueError(f"room static package contains a non-regular entry: {path}")
    if actual_dirs != {"room_collision"}:
        raise ValueError("room static package directory inventory differs")
    if actual_members != set(members):
        raise ValueError("room static package member inventory differs")
    for rel, identity in members.items():
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts \
                or rel_path.as_posix() != rel:
            raise ValueError("room static package member path is unsafe")
        if not isinstance(identity, dict) \
                or set(identity) != {"sha256", "size_bytes"}:
            raise ValueError("room static package member identity schema differs")
        path = package_dir / rel_path
        if not path.is_file() or path.is_symlink() \
                or identity["sha256"] != _sha256_file(path) \
                or identity["size_bytes"] != path.stat().st_size:
            raise ValueError(f"room static package member identity differs: {rel}")
    content_sha256 = hashlib.sha256(_canonical_json_bytes(members)).hexdigest()
    if manifest["content_sha256"] != content_sha256:
        raise ValueError("room static package content identity differs")
    if manifest["background_sha256"] \
            != members.get("background.obj", {}).get("sha256"):
        raise ValueError("room static package background identity differs")

    payload = _load_exact_json(package_dir / "payload.json", "room static payload")
    payload_keys = {
        "schema_version", "scene_id", "diagnostic_variant",
        "asset_xml_lines", "geom_xml_lines", "room_report",
    }
    if not isinstance(payload, dict) or set(payload) != payload_keys \
            or payload["schema_version"] != ROOM_STATIC_PACKAGE_SCHEMA_VERSION \
            or payload["scene_id"] != scene_id \
            or payload["diagnostic_variant"] != expected_variant \
            or not isinstance(payload["room_report"], dict):
        raise ValueError("room static package payload binding differs")
    assets = payload["asset_xml_lines"]
    geoms = payload["geom_xml_lines"]
    if not isinstance(assets, list) or not isinstance(geoms, list) \
            or any(not isinstance(line, str) for line in assets + geoms):
        raise ValueError("room static package XML payload is invalid")
    if _static_xml_sha256(assets, geoms) != manifest["static_xml_sha256"]:
        raise ValueError("room static package XML identity differs")
    rendered_assets = [
        line.replace(ROOM_STATIC_PACKAGE_PLACEHOLDER, str(package_dir))
        for line in assets
    ]
    if any(ROOM_STATIC_PACKAGE_PLACEHOLDER in line for line in rendered_assets):
        raise ValueError("room static package XML placeholder was not rendered")
    identity = {
        "manifest_sha256": manifest_sha256,
        "content_sha256": content_sha256,
        "static_xml_sha256": manifest["static_xml_sha256"],
        "background_sha256": manifest["background_sha256"],
    }
    return {
        "asset_xml_lines": rendered_assets,
        "geom_xml_lines": list(geoms),
        "room_report": payload["room_report"],
        "source_manifest_sha256": dict(source_manifest_sha256),
        "static_package_identity": identity,
        "package_dir": str(package_dir),
    }


def validate_automatic_export_context(factory_dir, *, auto_mode, pipeline_mesh):
    """Fail before any instance/scan read when an automatic factory is misrouted."""
    factory_dir = Path(factory_dir)
    manifest_path = factory_dir / "materialization_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 2:
        return None
    if manifest.get("manifest_kind") != "e4_automatic_construction_variant_materialization":
        raise ValueError("unknown automatic materialization schema")
    if auto_mode != "1" or Path(pipeline_mesh).resolve() != (factory_dir / "derived_mesh.ply").resolve():
        raise ValueError("automatic export requires AUTO=1 and its exact derived mesh")
    from robo.eval.e3_factory_materializer import validate_materialized_factory
    report = validate_materialized_factory(factory_dir)
    for name in ("derived_mesh.ply", "auto_instances.npz"):
        path = factory_dir / name
        identity = manifest["output_members"][name]
        if path.stat().st_size != identity["size_bytes"] or _sha256_file(path) != identity["sha256"]:
            raise ValueError("automatic export scene-source bytes changed")
    objects = json.loads((factory_dir / "objects/objects.json").read_text())
    if any(row.get("instance_namespace") != "automatic" or "gt_object_id" in row
           or row.get("automatic_instance_id") != row.get("index") for row in objects):
        raise ValueError("automatic factory contains a GT or ambiguous object namespace")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--collision-mode", choices=["room", "shim"], default="room",
                    help="'room' (default): full room/table/wall/obstacle static "
                         "collision via robo.sim.room_collision -- no private "
                         "per-object support shims (acceptance criterion for plan "
                         "Task 06). 'shim': legacy v0 ablation, private per-object "
                         "support shims only, no table/wall/obstacle collision "
                         "at all (plan Task 19).")
    ap.add_argument(
        "--background-carve-factory", action="append", default=[],
        help="repeat once per paired E4 construction factory (A0 and A4). "
             "Room mode then uses their common discovered roster and union "
             "of actual transformed collision hulls for both backgrounds",
    )
    ap.add_argument(
        "--room-diagnostic-spec",
        help="exact sealed E4 room-diagnostic variant JSON (room mode only)",
    )
    ap.add_argument(
        "--room-static-package-out",
        help="build a sealed common static package at this path and exit",
    )
    ap.add_argument(
        "--room-static-package",
        help="consume and verify a previously sealed common static package",
    )
    args = ap.parse_args()
    validate_automatic_export_context(C.OUT, auto_mode=C.env("AUTO"), pipeline_mesh=C.PIPELINE_MESH_PLY)
    if args.collision_mode != "room" and args.background_carve_factory:
        ap.error("--background-carve-factory is valid only with --collision-mode room")
    if args.collision_mode != "room" and (args.room_diagnostic_spec
                                           or args.room_static_package_out
                                           or args.room_static_package):
        ap.error("room diagnostic/static-package options require --collision-mode room")
    if bool(args.room_static_package_out) and bool(args.room_static_package):
        ap.error("--room-static-package-out and --room-static-package are exclusive")
    if (args.room_static_package_out or args.room_static_package) \
            and not args.room_diagnostic_spec:
        ap.error("room static packages require --room-diagnostic-spec")
    if args.room_diagnostic_spec \
            and not (args.room_static_package_out or args.room_static_package):
        ap.error("--room-diagnostic-spec requires a static package direction")
    if (args.room_static_package_out or args.room_static_package) \
            and len(args.background_carve_factory) != 2:
        ap.error("room static packages require exactly two background-carve factories")

    diagnostic_spec = diagnostic_spec_sha256 = None
    if args.room_diagnostic_spec:
        diagnostic_spec, diagnostic_spec_sha256 = load_room_diagnostic_spec(
            args.room_diagnostic_spec, C.SCENE_ID
        )

    # The common builder must be a read-only consumer of both freshly
    # materialized policy factories.  Branch before even creating C.OUT/sim or
    # C.OUT/sim_export; the builder redirects s7's legacy output into scratch.
    if args.room_static_package_out:
        package = build_common_room_static_package(
            args.background_carve_factory,
            args.room_static_package_out,
            diagnostic_spec,
            diagnostic_spec_sha256,
        )
        identity = package["static_package_identity"]
        print(f"[mx] sealed common room-static package: "
              f"{package['package_dir']} manifest={identity['manifest_sha256'][:12]} "
              f"content={identity['content_sha256'][:12]}")
        return

    out = C.OUT
    exp = out / "sim_export"
    exp.mkdir(exist_ok=True)
    static_package = None
    robust_floor_binding = None
    if args.room_static_package:
        static_package = load_common_room_static_package(
            args.room_static_package, diagnostic_spec, diagnostic_spec_sha256
        )
        source_hashes = _factory_manifest_sources(args.background_carve_factory)
        if static_package["source_manifest_sha256"] != source_hashes:
            raise ValueError("room static package source factory binding differs")
        sim_dir = out / "sim"
        sim_dir.mkdir(exist_ok=True)
        shutil.copyfile(
            Path(static_package["package_dir"]) / "background.obj",
            sim_dir / "background.obj",
        )
        shutil.copyfile(
            Path(static_package["package_dir"]) / "background_carve.json",
            sim_dir / "background_carve.json",
        )
        robust_floor_binding = _robust_floor_binding(
            static_package["room_report"], diagnostic_spec
        )
    objects = json.loads((out / "objects" / "objects.json").read_text())

    bodies, manifest = [], []
    assets = []
    supports = []  # (z_top, xy_lo, xy_hi)
    local_carve_hulls = []
    object_hull_diagnostics = []
    objs_meta, aligneds_kept = [], []  # kept (non-rejected) raw metadata, for
                                        # room-mode background carving below
    for m in objects:
        odir = out / "objects" / f"obj_{m['index']:02d}"
        al = json.loads((odir / "aligned.json").read_text())
        if al.get("rejected") or not (odir / "object.urdf").exists():
            continue
        objs_meta.append(m)
        aligneds_kept.append(al)
        ph = json.loads((odir / "physics.json").read_text())
        s, R, t = C.decompose_similarity(np.array(al["T"]))
        q = C.rot_to_quat_wxyz(R)
        # CoACD occasionally emits a degenerate near-zero-volume sliver hull
        # that MuJoCo's mesh compiler rejects outright; drop those, keep the
        # rest -- one dropped sliver out of ~16 hulls doesn't meaningfully
        # change the collision volume. TWO distinct rejections to dodge:
        #   "at least 4 vertices required"  -> vertex count
        #   "mesh volume is too small"      -> near-zero volume even with >=4
        #      verts (a flat/collinear sliver). Hit on 7b6477cb95 obj_12_p12,
        #      which a vertex-count-only filter let through.
        # Volume is taken from the convex hull so it is well defined even when
        # the raw part mesh is not watertight.
        import trimesh as _tm
        MIN_HULL_VOL = 1e-9  # m^3, i.e. a 1 mm cube

        def _usable(pth):
            try:
                t = _tm.load(pth, process=False)
                if len(t.vertices) < 4:
                    return False
                return float(t.convex_hull.volume) >= MIN_HULL_VOL
            except Exception:  # noqa: BLE001  unreadable/degenerate -> drop
                return False

        all_parts = sorted((odir / "collision").glob("part_*.obj"))
        parts = [pth for pth in all_parts if _usable(pth)]
        if len(parts) < len(all_parts):
            print(f"[mx] {m['label']} (obj_{m['index']:02d}): dropped "
                  f"{len(all_parts) - len(parts)} degenerate collision "
                  f"part(s) (<4 verts or hull volume < {MIN_HULL_VOL:g} m^3)")
        name = f"obj_{m['index']:02d}"
        if diagnostic_spec is not None:
            aabb = np.asarray(m.get("aabb"), dtype=np.float64)
            if aabb.shape != (2, 3) or not np.isfinite(aabb).all() \
                    or np.any(aabb[1] <= aabb[0]):
                raise ValueError(f"diagnostic scan AABB is invalid: {name}")
            selected_parts = []
            clipped_dir = exp / "diagnostic_object_collision" / name
            for part_index, pth in enumerate(parts):
                mesh = _tm.load(pth, process=False)
                local_vertices = np.asarray(mesh.vertices, dtype=np.float64)
                world_raw = local_vertices * s @ R.T + t
                row = {
                    "name": f"{name}/{pth.name}",
                    "raw_bounds_m": [world_raw.min(axis=0).tolist(),
                                     world_raw.max(axis=0).tolist()],
                    "scan_aabb_m": aabb.tolist(),
                    "raw_bottom_minus_scan_bottom_m": float(
                        world_raw[:, 2].min() - aabb[0, 2]
                    ),
                    "selected_bottom_policy": diagnostic_spec["hull_bottom"],
                }
                if diagnostic_spec["hull_bottom"] == "clip_scan_aabb_bottom":
                    clip_z = float(aabb[0, 2])
                    if robust_floor_binding is not None:
                        clip_z, floor_overlap = _floor_aware_clip_height(
                            aabb, robust_floor_binding
                        )
                        row["floor_aware_clip"] = {
                            "floor_component_id": robust_floor_binding[
                                "selected_component_id"
                            ],
                            "floor_z_m": robust_floor_binding["floor_z_m"],
                            "floor_xy_overlap_m2": floor_overlap,
                            "selected_clip_z_m": clip_z,
                        }
                    try:
                        clipped_world, clip_diagnostic = rc.clip_convex_hull_at_z(
                            world_raw, clip_z, name=row["name"]
                        )
                    except ValueError as exc:
                        row["status"] = "removed_all"
                        row["error"] = str(exc)
                        object_hull_diagnostics.append(row)
                        if robust_floor_binding is not None:
                            raise ValueError(
                                f"floor-aware diagnostic hull clipping removed "
                                f"{row['name']}: {exc}"
                            ) from exc
                        continue
                    row["clip"] = clip_diagnostic
                    retained = clip_diagnostic.get("retained_volume_fraction")
                    if robust_floor_binding is not None \
                            and (isinstance(retained, bool)
                                 or not isinstance(retained, (int, float))
                                 or not np.isfinite(retained) or retained <= 0.0):
                        raise ValueError(
                            f"floor-aware diagnostic hull retained no volume: "
                            f"{row['name']}"
                        )
                    selected_world = np.asarray(clipped_world.vertices, dtype=np.float64)
                    if clip_diagnostic["clipped"]:
                        clipped_dir.mkdir(parents=True, exist_ok=True)
                        # Body pose and mesh-scale remain unchanged.  Map the
                        # capped world hull back to the canonical mesh frame.
                        selected_local = (selected_world - t) @ R / s
                        selected_mesh = _tm.Trimesh(
                            vertices=selected_local,
                            faces=np.asarray(clipped_world.faces), process=False,
                        )
                        selected_path = clipped_dir / f"part_{part_index:02d}.obj"
                        selected_mesh.export(selected_path)
                    else:
                        selected_path = pth
                else:
                    selected_world = world_raw
                    selected_path = pth
                row["status"] = "selected"
                row["selected_bounds_m"] = [selected_world.min(axis=0).tolist(),
                                             selected_world.max(axis=0).tolist()]
                row["selected_path"] = str(selected_path.relative_to(out))
                object_hull_diagnostics.append(row)
                selected_parts.append(selected_path)
            parts = selected_parts
            if not parts:
                raise ValueError(f"diagnostic hull-bottom policy removed {name}")
        dims = np.array(al["world_dims"])
        mass = ph["mass_kg"]
        dx, dy, dz = np.maximum(dims, 1e-3)
        diag = (mass / 12 * (dy**2 + dz**2), mass / 12 * (dx**2 + dz**2),
                mass / 12 * (dx**2 + dy**2))
        # {m} stays literal here and is substituted once the collision channels
        # are known (an interpolated f-string value is not re-scanned for braces)
        geoms = "\n".join(
            f'      <geom type="mesh" mesh="{name}_p{k}" '
            f'friction="{ph["friction"]:.2f} 0.005 0.0001" group="3" '
            f'contype="{{m}}" conaffinity="{{m}}"/>'
            for k in range(len(parts)))
        for k, pth in enumerate(parts):
            assets.append(
                f'    <mesh name="{name}_p{k}" '
                f'file="{pth.relative_to(out)}" scale="{s:.6f} {s:.6f} {s:.6f}"/>')
        assets.append(
            f'    <mesh name="{name}_vis" '
            f'file="objects/{name}/mesh_sim.obj" scale="{s:.6f} {s:.6f} {s:.6f}"/>')
        bodies.append(f"""    <body name="{name}" pos="{t[0]:.5f} {t[1]:.5f} {t[2]:.5f}"
          quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{mass:.4f}"
                diaginertia="{diag[0]:.12e} {diag[1]:.12e} {diag[2]:.12e}"/>
      <geom type="mesh" mesh="{name}_vis" contype="0" conaffinity="0"
            group="2" rgba="0.8 0.8 0.8 1"/>
{geoms}
    </body>""")
        manifest.append({
            "name": name, "label": m["label"],
            "urdf": str(odir / "object.urdf"),
            "pos": t.tolist(), "quat_wxyz": q.tolist(),
            "mass_kg": mass, "friction": ph["friction"],
            "tier": al.get("tier", "A")})
        # Support height AND footprint come from the CoACD collision hulls at
        # FULL vertex resolution -- exactly the geometry MuJoCo will collide.
        # This replaces THREE errors, two of them heights with different causes:
        #  - the height was read from a ~500-point strided subsample
        #    (max(len//500, 1)) of the ~20k-vertex VISUAL mesh: 0.1-6.5 mm bias
        #    on c50d2d1d42 (obj_05 +6.5, obj_04 +3.7, obj_08 +3.2 mm);
        #  - the visual mesh is NOT what collides. CoACD inflates hulls past the
        #    input surface: 7 of obj_06's parts sit 76-93 mm BELOW its visual
        #    bottom, so its shim was 94 mm above the true collision bottom --
        #    93 mm of that is hull bulge, only 1 mm the subsample. Using all
        #    20k VISUAL verts would still leave obj_06 92 mm out; hull verts
        #    zero all 7 own-slab penetrations by construction;
        #  - the footprint came from the detected scan AABB, a different
        #    geometry source than the height, so a shim could sit offset from
        #    the object it supports.
        if parts:
            hv = np.concatenate([np.asarray(_tm.load(p, process=False).vertices)
                                 for p in parts])
        else:  # every hull was degenerate -> no collision geoms at all; fall
               # back to the FULL visual mesh (still never a subsample)
            hv = np.asarray(
                _tm.load(odir / "mesh_sim.ply", process=False).vertices)
        hull_w = hv * s @ R.T + t
        supports.append((float(hull_w[:, 2].min()),
                         hull_w[:, :2].min(axis=0), hull_w[:, :2].max(axis=0)))
        for part in parts:
            vertices = np.asarray(_tm.load(part, process=False).vertices)
            local_carve_hulls.append({
                "name": f"local/{name}/{part.name}",
                "policy_id": "local",
                "slot": name,
                "support_clip_z_m": float(
                    np.asarray(m["aabb"], dtype=np.float64)[0, 2]
                    + (SUPPORT_CLIP_OFFSET_M if diagnostic_spec is None else
                       diagnostic_spec["support_clip_offset_m"])
                ),
                "vertices": vertices * s @ R.T + t,
            })

    # Collision group scheme, shared by both modes: MuJoCo pairs geoms when
    # (contype1 & conaffinity2) || (contype2 & conaffinity1). Objects, the
    # floor, the robot, and (room mode) the room itself all carry bit 0, so
    # object<->object, <->floor, <->robot and <->room all collide normally.
    if args.collision_mode == "shim":
        # The shim ablation has no scan-derived room contract.  Preserve its
        # historical safe-below-lowest-object fallback exactly; it must not
        # inherit any repair available only to the main full-room condition.
        floor_z = (min(z for z, _, _ in supports) - 0.05) if supports else 0.0
        floor_source = "legacy_below_lowest_object"
        # LEGACY ABLATION (plan Task 19): one static micro-shim per object at
        # that object's own collision-hull bottom, PRIVATE to that object via
        # a greedy-coloured channel bit ABOVE bit 0 -- it can never touch
        # another object, the floor, or the robot. No table/wall/obstacle
        # collision exists at all in this mode; see room_collision.py's
        # build_shim_geoms docstring for the exact history (impaled
        # neighbours, why colouring instead of one-bit-per-object, etc.) --
        # ported here verbatim, this call reproduces that logic exactly.
        masks, static_geoms = rc.build_shim_geoms(supports, pad=PAD, slab=SLAB,
                                                  safe_margin=SAFE)
        bodies = [b.replace("{m}", str(1 | masks[i])) for i, b in enumerate(bodies)]
        n_channels = len(set(masks)) if masks else 0
        room_report = {"mode": "shim", "n_shims": len(static_geoms),
                       "n_channels": n_channels, "floor_z_m": floor_z,
                       "floor_source": floor_source}
        print(f"[mx] collision-mode=shim: {len(static_geoms)} private support "
              f"shims on {n_channels} collision channels (legacy ablation -- "
              f"no table/wall/obstacle collision)")
    else:
        # ROOM MODE (default): carve the reconstructed scan mesh the same way
        # s7_sim's PyBullet reference does (objects subtracted out, cropped,
        # decimated), then classify it into floor/support/wall primitives +
        # CoACD parts for the residual. Objects need no private shim at all;
        # bit 0 alone lets them rest on/collide with the real room.
        bodies = [b.replace("{m}", "1") for b in bodies]
        # Every room-mode source is the ScanNet++ mesh/world frame loaded by
        # s7_sim.scene_mesh_arrays.  Its floor contract is z=0; deriving this
        # plane from whichever policy happened to accept the lowest object
        # made the A0/A4 environments physically different.
        floor_z = (robust_floor_binding["floor_z_m"]
                   if robust_floor_binding is not None else 0.0)
        floor_source = ("robust_area_weighted_component"
                        if robust_floor_binding is not None
                        else "scannetpp_world_frame_z0")
        if static_package is not None:
            assets += static_package["asset_xml_lines"]
            static_geoms = static_package["geom_xml_lines"]
            # Copy via JSON so later benchmark augmentation cannot mutate the
            # loader's verified object through a shared nested reference.
            room_report = json.loads(json.dumps(static_package["room_report"]))
            raw_diagnostics = room_report.setdefault("raw_diagnostics", {})
            raw_diagnostics["dynamic_object_hulls"] = object_hull_diagnostics
            raw_diagnostics["static_package_identity"] = static_package[
                "static_package_identity"
            ]
            if robust_floor_binding is not None:
                if room_report.get("floor_z_m") != floor_z \
                        or room_report.get("floor_source") != floor_source:
                    raise ValueError("robust global floor differs from static package")
            print(f"[mx] collision-mode=room: consumed common static package "
                  f"{static_package['static_package_identity']['manifest_sha256'][:12]} "
                  f"floor={room_report['n_floor']} supports={room_report['n_supports']} "
                  f"walls={room_report['n_walls']} "
                  f"coacd_parts={room_report['coacd_parts']}")
        else:
            from robo.sim.s7_sim import build_background
            gts = {g["object_id"]: g for g in C.load_instances()}
            carve_sources = None
            if args.background_carve_factory:
                background_objects, carve_hulls, carve_sources = \
                    load_common_carve_inputs(args.background_carve_factory)
                background_aligneds = [{} for _ in background_objects]
            else:
                background_objects = objs_meta
                background_aligneds = aligneds_kept
                carve_hulls = local_carve_hulls
            bg_path, background_carve = build_background(
                background_objects,
                background_aligneds,
                gts,
                carve_hulls=carve_hulls,
                carve_margin_m=rc.CARVE_MARGIN_M,
                lower_support_margin_m=0.0,
                paired_policy_ids=carve_sources["policy_ids"] if carve_sources is not None else None,
                return_report=True,
            )
            if args.background_carve_factory \
                    and background_carve.get("mode") != "paired_policy_union":
                raise ValueError("explicit E4 background carve did not bind a paired union")
            carve_contract_keys = (
                "background_sha256", "carved_slots", "discovered_slots",
                "geometry_source", "hull_count", "lower_support_margin_m",
                "margin_m", "mode", "policy_ids", "roster_sha256",
                "schema_version", "source_mesh_sha256", "specification_sha256",
                "support_clip_source",
            )
            background_carve_contract = {
                key: background_carve[key] for key in carve_contract_keys
            }
            background_carve_diagnostics = {
                key: value for key, value in background_carve.items()
                if key not in carve_contract_keys
            }
            exclusions = tuple(
                rc.make_convex_exclusion(
                    row["vertices"], name=str(row["name"]), slot=str(row["slot"]),
                    policy_id=str(row["policy_id"]), margin_m=rc.CARVE_MARGIN_M,
                    lower_support_margin_m=0.0,
                )
                for row in carve_hulls
            )
            import trimesh
            room_mesh = trimesh.load(bg_path, process=False)
            decimated, decim_stats = rc.decimate_mesh(room_mesh)
            feats = rc.extract_room_features(
                decimated,
                expected_floor_z=rc.SCANNETPP_FLOOR_Z_M,
                floor_z_tol=rc.SCANNETPP_FLOOR_TOL_M,
            )
            if feats.floor is not None \
                    and abs(feats.floor.z - rc.SCANNETPP_FLOOR_Z_M) \
                    > rc.SCANNETPP_FLOOR_TOL_M:
                raise ValueError("room feature extractor misclassified a high support as floor")
            floor_patch_selection = {
                "expected_z_m": rc.SCANNETPP_FLOOR_Z_M,
                "tolerance_m": rc.SCANNETPP_FLOOR_TOL_M,
                "detected_z_m": None if feats.floor is None else float(feats.floor.z),
                "status": "not_detected" if feats.floor is None else "matched",
            }
            room_assets, static_geoms, room_manifest, collision_mesh = rc.emit_room_mjcf(
                feats, out_dir=exp / "room_collision", exclusions=exclusions)
            assets += room_assets
            coverage = rc.collision_coverage_metrics(room_mesh, collision_mesh)
            primitive_intrusions = room_manifest["primitive_intrusions"]
            emitted_coacd_intrusions = room_manifest["coacd_emitted_intrusions"]
            collision_exclusion = {
                "schema_version": 1,
                "hull_count": len(exclusions),
                "coacd_candidate_parts": room_manifest["coacd_candidate_parts"],
                "coacd_parts_rejected_intrusion": room_manifest[
                    "coacd_parts_rejected_exclusion"
                ],
                "emitted_coacd_intrusion_count": len(emitted_coacd_intrusions),
                "primitive_intrusions": primitive_intrusions,
                "unresolved_intrusion_count": (
                    len(primitive_intrusions) + len(emitted_coacd_intrusions)
                ),
            }
            room_report = {"mode": "room", **room_manifest, **decim_stats,
                           "coverage": coverage, "n_floor": 1 if feats.floor else 0,
                           "n_supports": len(feats.supports),
                           "n_walls": len(feats.walls),
                           "floor_z_m": floor_z, "floor_source": floor_source,
                           "floor_patch_selection": floor_patch_selection,
                           "background_carve": background_carve_contract,
                           "background_carve_diagnostics": background_carve_diagnostics,
                           "collision_exclusion": collision_exclusion}
            if carve_sources is not None:
                room_report["background_carve_sources"] = carve_sources
            print(f"[mx] collision-mode=room: floor={room_report['n_floor']} "
                  f"supports={room_report['n_supports']} "
                  f"walls={room_report['n_walls']} "
                  f"coacd_parts={room_manifest['coacd_parts']} "
                  f"hausdorff_p95={decim_stats['hausdorff_p95_m'] * 1000:.1f}mm "
                  f"occupancy_agreement={coverage.get('occupancy_agreement')}")

    floor_z_xml = _global_floor_z_text(
        floor_z, robust=robust_floor_binding is not None
    )
    xml = f"""<mujoco model="phiroom_{C.SCENE_ID}">
  <compiler meshdir="{out}" angle="radian" autolimits="true"
            balanceinertia="true"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast"/>
  <visual><global offwidth="1920" offheight="1080"/></visual>
  <asset>
{chr(10).join(assets)}
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="0 0 {floor_z_xml}" size="20 20 1"
          friction="0.8 0.005 0.0001"/>
{chr(10).join(static_geoms)}
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""
    (exp / "scene.xml").write_text(xml)
    C.save_json(exp / "isaac_manifest.json", {
        "scene": C.SCENE_ID, "world_frame": "scannetpp mesh frame (z-up, m)",
        "note": "load each urdf at pos/quat_wxyz; URDF meshes are canonical "
                "with scale baked into <mesh scale>",
        "objects": manifest})
    C.save_json(exp / "room_collision_report.json", room_report)
    print(f"[mx] wrote {exp / 'scene.xml'} ({len(manifest)} bodies, "
          f"collision-mode={args.collision_mode}) + isaac_manifest.json + "
          f"room_collision_report.json")

    # cheap compile self-check (rounding once flipped valid thin-object
    # inertias to invalid; catch that class loudly instead of downstream)
    try:
        import mujoco
        try:
            mujoco.MjModel.from_xml_path(str(exp / "scene.xml"))
            print("[mx] scene.xml compile self-check: PASS")
        except Exception as e:
            print(f"[mx] scene.xml compile self-check: FAIL ({e})")
    except ImportError:
        pass

    if args.test:
        # rc.benchmark_model() replaces the old bespoke settle-and-drift loop
        # with one call that ALSO reports compile time, physics speed,
        # memory, and a contact trajectory (plan Task 06 step 6) -- and
        # `settle_drift_m` is computed identically (final xpos - initial
        # xpos, per body), so mujoco_settle.json's schema stays byte-for-byte
        # unchanged: pi05_tasks.py reads drift_m/stable_3cm/n from it and
        # must keep working without modification.
        report = rc.benchmark_model(exp / "scene.xml", steps=int(2.0 / TIMESTEP),
                                    body_names=[mrow["name"] for mrow in manifest])
        assert report["finite"], "NaN in qpos/qvel - unstable model"
        drifts = report["settle_drift_m"]
        C.save_json(exp / "mujoco_settle.json",
                    {"drift_m": drifts,
                     "stable_3cm": sum(d < 0.03 for d in drifts.values()),
                     "n": len(drifts)})
        med = float(np.median(list(drifts.values()))) if drifts else 0.0
        print(f"[mx] MuJoCo 2s settle: {sum(d < 0.03 for d in drifts.values())}"
              f"/{len(drifts)} stable (<3cm), median drift {med * 1000:.1f}mm, "
              f"no NaN")

        room_report["benchmark"] = {k: v for k, v in report.items()
                                    if k not in ("settle_drift_m", "final_body_pos")}
        C.save_json(exp / "room_collision_report.json", room_report)
        print(f"[mx] benchmark: compile={report['compile_time_s'] * 1000:.1f}ms "
              f"speed={report['steps_per_sec']:.0f} steps/s mem={report['mem_mb']:.0f}MB "
              f"contact_pairs={len(report['contact_pairs'])} state_hash="
              f"{report['state_hash'][:12]}")


if __name__ == "__main__":
    main()
