"""Construction-only registration and isolated physical evidence for E3.

This module receives a raw proposal mesh and a visible observation point
cloud.  It deliberately has no evaluation-reference input.  Evaluation is a
separate post-decision phase in :mod:`robo.eval.agentic_ablation`.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from agents.assets.s5_align import (
    align_object,
    align_object_with_alternative_source_up,
    apply_T,
    sym_score,
)
from agents.assets.s6_physics import write_urdf
from agents.core import common as C
from robo.sim.s7_sim import link_pose


MESH_SAMPLE_SEED = 42
MESH_SAMPLE_COUNT = 20_000
SUPPORT_BAND_M = 0.01
INITIAL_SUPPORT_GAP_M = 0.005
DROP_HZ = 240
SETTLE_SECONDS = 2
FREE_SECONDS = 2
STABLE_DRIFT_M = 0.03
SUNK_AABB_MIN_Z_M = -0.01
EVIDENCE_MANIFEST_FIELDS = (
    "producer",
    "producer_commit",
    "input_hashes",
    "raw_values",
    "missing_flags",
    "started_utc",
    "finished_utc",
    "wall_s",
)


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _support_overlap(vertices_world: np.ndarray) -> float | None:
    """Bottom-band footprint area divided by the complete xy footprint."""
    from scipy.spatial import ConvexHull, QhullError

    if len(vertices_world) < 4:
        return None
    z_min = float(vertices_world[:, 2].min())
    bottom = vertices_world[vertices_world[:, 2] <= z_min + SUPPORT_BAND_M]
    if len(bottom) < 3:
        return 0.0
    try:
        all_area = float(ConvexHull(vertices_world[:, :2]).volume)
        bottom_area = float(ConvexHull(bottom[:, :2]).volume)
    except QhullError:
        return 0.0
    if not math.isfinite(all_area) or all_area <= 1e-12:
        return 0.0
    return float(np.clip(bottom_area / all_area, 0.0, 1.0))


def _frozen_physics() -> dict[str, Any]:
    """Return the category-independent E3 probe parameters.

    The controller contract forbids label/category-specific rules.  Keeping a
    single declared mass/friction pair also prevents the legacy semantic label
    from crossing the sanitized inventory boundary.
    """
    return {
        "mass_kg": 0.3,
        "friction": 0.5,
        "restitution": 0.1,
        "source": "e3_category_independent_constant_v1",
    }


def _apply_contact_dynamics(
    pybullet: Any,
    body_id: int,
    *,
    role: str,
    physics: dict[str, Any],
) -> dict[str, Any]:
    """Apply and read back the frozen Bullet contact parameters."""
    lateral_friction = float(physics["friction"])
    restitution = float(physics["restitution"])
    pybullet.changeDynamics(
        body_id,
        -1,
        lateralFriction=lateral_friction,
        restitution=restitution,
    )
    dynamics = pybullet.getDynamicsInfo(body_id, -1)
    reported_friction = float(dynamics[1])
    reported_restitution = float(dynamics[5])
    if not math.isclose(reported_friction, lateral_friction, abs_tol=1e-12):
        raise RuntimeError(
            f"Bullet {role} friction mismatch: requested {lateral_friction}, "
            f"reported {reported_friction}"
        )
    if not math.isclose(reported_restitution, restitution, abs_tol=1e-12):
        raise RuntimeError(
            f"Bullet {role} restitution mismatch: requested {restitution}, "
            f"reported {reported_restitution}"
        )
    return {
        "application": "pybullet.changeDynamics_getDynamicsInfo_v1",
        "link_index": -1,
        "lateral_friction": reported_friction,
        "restitution": reported_restitution,
        "role": role,
    }


def isolated_probe(mesh, T: np.ndarray, label: str, out_dir: Path) -> dict[str, Any]:
    """One deterministic convex-hull collision and corrected link-frame drop."""
    import pybullet as p
    import trimesh
    from scipy.spatial.transform import Rotation

    out_dir.mkdir(parents=True, exist_ok=False)
    collision_dir = out_dir / "collision"
    collision_dir.mkdir()
    hull = mesh.convex_hull
    vertices = np.asarray(hull.vertices, dtype=np.float64)
    faces = np.asarray(hull.faces, dtype=np.int64)
    collision_valid = bool(
        len(vertices) >= 4
        and len(faces) >= 4
        and hull.is_watertight
        and math.isfinite(float(hull.volume))
        and float(hull.volume) > 1e-12
    )
    hull.export(out_dir / "mesh_sim.obj")
    hull.export(collision_dir / "part_00.obj")

    scale, rotation, _ = C.decompose_similarity(np.asarray(T, dtype=np.float64))
    transformed = vertices * scale @ rotation.T
    support_overlap = _support_overlap(transformed)
    physics = _frozen_physics()
    write_urdf(
        out_dir,
        "e3_candidate",
        scale,
        scale * (vertices.max(axis=0) - vertices.min(axis=0)),
        physics,
        ["part_00.obj"],
        vertices.mean(axis=0) * scale,
    )
    _json(out_dir / "physics.json", physics)

    result: dict[str, Any] = {
        "collision_backend": "deterministic_single_convex_hull_v1",
        "collision_valid": collision_valid,
        "usable_convex_parts": 1 if collision_valid else 0,
        "support_gap_m": INITIAL_SUPPORT_GAP_M,
        "support_overlap_fraction": support_overlap,
        "initial_penetration_m": 0.0,
        "contact_dynamics": None,
        "settle_sunk": None,
        "settle_stable": None,
        "settle_drift_m": None,
    }
    if not collision_valid:
        _json(out_dir / "probe.json", result)
        return result

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / DROP_HZ)
        p.setPhysicsEngineParameter(numSolverIterations=100)
        plane = p.createCollisionShape(p.GEOM_PLANE)
        plane_body = p.createMultiBody(0, plane)
        plane_dynamics = _apply_contact_dynamics(
            p,
            plane_body,
            role="plane",
            physics=physics,
        )
        quaternion = Rotation.from_matrix(rotation).as_quat().tolist()
        initial_z = -float(transformed[:, 2].min()) + INITIAL_SUPPORT_GAP_M
        body = p.loadURDF(
            str((out_dir / "object.urdf").resolve()),
            basePosition=[0.0, 0.0, initial_z],
            baseOrientation=quaternion,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        object_dynamics = _apply_contact_dynamics(
            p,
            body,
            role="object",
            physics=physics,
        )
        result["contact_dynamics"] = {
            "object": object_dynamics,
            "plane": plane_dynamics,
        }
        # URDF inertial-origin handling can offset the link from the supplied
        # base position.  Correct from Bullet's actual collision AABB before
        # measuring the declared 5 mm initial support gap.
        position, orientation = p.getBasePositionAndOrientation(body)
        aabb_min_z = float(p.getAABB(body)[0][2])
        corrected = [position[0], position[1], position[2] + INITIAL_SUPPORT_GAP_M - aabb_min_z]
        p.resetBasePositionAndOrientation(body, corrected, orientation)
        result["initial_penetration_m"] = max(0.0, -float(p.getAABB(body)[0][2]))

        for _ in range(SETTLE_SECONDS * DROP_HZ):
            p.stepSimulation()
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        before, _ = link_pose(p, body)
        for _ in range(FREE_SECONDS * DROP_HZ):
            p.stepSimulation()
        after, _ = link_pose(p, body)
        drift = float(np.linalg.norm(np.asarray(after) - np.asarray(before)))
        final_min_z = float(p.getAABB(body)[0][2])
        sunk = bool(final_min_z < SUNK_AABB_MIN_Z_M)
        result.update({
            "settle_sunk": sunk,
            "settle_stable": bool(drift < STABLE_DRIFT_M and not sunk),
            "settle_drift_m": drift,
            "final_aabb_min_z_m": final_min_z,
        })
    except Exception as exc:
        result.update({
            "collision_valid": False,
            "usable_convex_parts": 0,
            "probe_error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        p.disconnect(client)
    _json(out_dir / "probe.json", result)
    return result


def align_and_probe(
    mesh_path: str | Path,
    observation_points: np.ndarray,
    *,
    label: str,
    visible_fraction: float,
    out_dir: str | Path,
    signed_source_up: bool,
    producer_commit: str,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Register one raw proposal, create physical evidence, and persist it."""
    import trimesh

    started_utc = _utc_now()
    started = time.monotonic()
    mesh_path = Path(mesh_path)
    output = Path(out_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite proposal evidence: {output}")
    output.mkdir(parents=True)
    mesh = trimesh.load(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise ValueError(f"proposal mesh is not a usable triangle mesh: {mesh_path}")
    points, _ = trimesh.sample.sample_surface(
        mesh, MESH_SAMPLE_COUNT, seed=MESH_SAMPLE_SEED
    )
    mesh_points = np.asarray(points, dtype=np.float64)
    observation = np.asarray(observation_points, dtype=np.float64)
    if observation.ndim != 2 or observation.shape[1] != 3 or len(observation) < 3:
        raise ValueError("observation_points must have shape [N,3]")
    if not np.isfinite(observation).all():
        raise ValueError("observation_points contain non-finite values")

    if signed_source_up:
        T, median, initial, tilt, source_up = align_object_with_alternative_source_up(
            mesh_points, observation
        )
    else:
        T, median, initial, tilt = align_object(mesh_points, observation)
        source_up = "+z"
    scale, _, _ = C.decompose_similarity(T)
    registered = apply_T(T, mesh_points)
    residual = float(sym_score(registered, observation))
    observed_extent = np.percentile(observation, 98, axis=0) - np.percentile(
        observation, 2, axis=0
    )
    world_dims = scale * (
        np.asarray(mesh.vertices).max(axis=0) - np.asarray(mesh.vertices).min(axis=0)
    )
    ratio = float(world_dims.max()) / max(float(observed_extent.max()), 1e-9)
    alignment = {
        "T": np.asarray(T).tolist(),
        "scale": float(scale),
        "source_up_hypothesis": source_up,
        "mesh_sample_seed": MESH_SAMPLE_SEED,
        "mesh_sample_count": MESH_SAMPLE_COUNT,
        "registration_median_m": float(median),
        "registration_initial_objective_m": float(initial),
        "symmetric_clipped_registration_residual_m": residual,
        "residual_tilt_deg": float(tilt),
        "world_dims_m": world_dims.tolist(),
        "observation_dims_m": observed_extent.tolist(),
        "scale_ratio_vs_observation": ratio,
    }
    _json(output / "registration.json", alignment)
    physical = isolated_probe(mesh, T, label, output / "physical")
    evidence = {
        "schema_frame_unit_valid": True,
        "symmetric_clipped_registration_residual_m": residual,
        "scale_ratio_vs_observation": ratio,
        "observation_point_count": int(len(observation)),
        "visible_fraction": float(visible_fraction),
        "support_gap_m": physical.get("support_gap_m"),
        "support_overlap_fraction": physical.get("support_overlap_fraction"),
        "initial_penetration_m": physical.get("initial_penetration_m"),
        "collision_valid": physical.get("collision_valid"),
        "usable_convex_parts": physical.get("usable_convex_parts"),
        "settle_drift_m": physical.get("settle_drift_m"),
        "settle_sunk": physical.get("settle_sunk"),
        "settle_stable": physical.get("settle_stable"),
        "missing_evidence": [],
    }
    evidence["missing_evidence"] = sorted(
        key for key, value in evidence.items()
        if key != "missing_evidence" and value is None
    )
    wall_s = float(time.monotonic() - started)
    evidence_manifest = {
        "producer": "agents.orchestrator.runtime.align_and_probe",
        "producer_commit": producer_commit,
        "input_hashes": dict(sorted(input_hashes.items())),
        "raw_values": evidence,
        "missing_flags": list(evidence["missing_evidence"]),
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "wall_s": wall_s,
    }
    if set(evidence_manifest) != set(EVIDENCE_MANIFEST_FIELDS):
        raise RuntimeError("internal evidence-manifest schema drift")
    _json(output / "evidence.json", evidence_manifest)
    return {
        "alignment": alignment,
        "evidence": evidence,
        "wall_s": wall_s,
        "artifact_files": [
            "registration.json",
            "evidence.json",
            "physical/mesh_sim.obj",
            "physical/collision/part_00.obj",
            "physical/object.urdf",
            "physical/physics.json",
            "physical/probe.json",
        ],
    }
