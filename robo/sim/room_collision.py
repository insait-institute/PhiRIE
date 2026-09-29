"""Full-room MuJoCo collision export.

Replaces the legacy "one PRIVATE static support shim per object" scheme in
`export_mjcf.py` with real static collision derived from the reconstructed
room mesh: a floor plane, validated support/wall box primitives, and a CoACD
convex decomposition of everything else (obstacles + unclassified geometry).
Objects, the robot, and the room all then share ONE collision channel (bit 0
of MuJoCo's `contype`/`conaffinity`), so contacts that used to be structurally
impossible (object vs. room away from its own shim, object vs. object at a
shared support, robot vs. table/obstacle) are now physically present.

Why decomposition instead of one big `<geom type="mesh">`: verified
empirically (not just by inherited comment) that this MuJoCo build (3.10.0)
convexifies mesh geoms for collision -- a concave "U" channel mesh let a
dropped ball rest on the hull bridging its two walls instead of settling
inside the channel. So a single room-mesh collision geom silently becomes a
room-spanning convex hull; CoACD parts are already convex, so MuJoCo's
convexification of each part is a no-op and collision is exact (same trick
`agents/assets/s6_physics.py:coacd_parts` uses per-object).

Two collision modes, both built from the same room-feature extraction:
- "room"  (new, default): floor + validated support/wall primitives + CoACD
  parts of the residual mesh, all on the shared bit-0 channel. No private
  shims.
- "shim"  (legacy, explicit ablation): the original per-object private
  micro-slab on its own greedy-coloured channel bit, ported here verbatim
  from `export_mjcf.py` as `build_shim_geoms` so the ablation stays byte-for-
  byte comparable. Objects still get bit 0 from the shared floor plane, but
  nothing represents the table, walls, or obstacles.

`export_mjcf.py` imports `build_shim_geoms` / room-mode helpers and exposes
`--collision-mode {room,shim}`. This module also ships a self-contained CLI
(`python -m robo.sim.room_collision --scene <dir>`) that builds and
smoke-tests either mode against a synthetic fixture scene (a `scene.json`
spec, see the "synthetic fixture" section at the bottom of this file)
without touching any real scanned scene.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import resource
import time
from pathlib import Path

import numpy as np

UP = np.array([0.0, 0.0, 1.0])

# -------------------------------------------------------------- defaults ---
FLOOR_NORMAL_TOL = 0.35     # accept normal.z > 1 - tol as "horizontal, up"
WALL_NORMAL_TOL = 0.35      # accept |normal.z| < tol as "vertical"
PLANE_RESIDUAL_TOL = 0.02   # m, max vertex-to-fitted-plane distance to validate
MIN_SUPPORT_AREA = 0.05     # m^2, below this a horizontal patch is not a "surface"
MIN_WALL_AREA = 0.15        # m^2
DECIMATE_MAX_TRIS = 40_000
COACD_THRESHOLD = 0.05
COACD_MAX_PARTS = 64
# Below this face count, run CoACD's MCTS search at all is not worth its own
# fixed per-call overhead (~0.1-1s of preprocessing regardless of size) --
# verified on a real scene (38d58a7a31): the residual mesh split into 1602
# connected components, almost all tiny debris fragments, and calling CoACD
# on each one individually made room-mode export take 15+ minutes. A plain
# convex hull is a fine (if slightly coarser) static-collision approximation
# for small, simple debris; CoACD's real value is on the few large/complex
# components (a room shell, a big non-convex obstacle) that are actually
# worth decomposing into several convex pieces.
COACD_MIN_FACES = 300
CARVE_MARGIN_M = 0.02
CARVE_INTERIOR_TOL_M = 1e-7
SCANNETPP_FLOOR_Z_M = 0.0
SCANNETPP_FLOOR_TOL_M = 0.05

# Keep the public v1 names stable: the first diagnostic sweep imports these
# constants and its sealed spec bytes must continue to validate verbatim.
ROOM_DIAGNOSTIC_SPEC_SCHEMA_VERSION = 1
ROOM_DIAGNOSTIC_SPEC_KEYS = frozenset({
    "schema_version",
    "variant_id",
    "scene_id",
    "hull_bottom",
    "intrusive_primitive",
    "plane_residual_tol_m",
    "support_z",
    "min_support_area_m2",
    "carve_side_top_margin_m",
    "lower_carve_margin_m",
    "support_clip_offset_m",
})
ROOM_DIAGNOSTIC_SPEC_SCHEMA_VERSION_V2 = 2
ROOM_DIAGNOSTIC_SPEC_V2_KEYS = frozenset({
    *ROOM_DIAGNOSTIC_SPEC_KEYS,
    "room_surface_policy",
})

# Frozen robust-surface guards.  They are intentionally code constants rather
# than sweep knobs: the sealed v2 spec selects only F versus FS, while the
# geometric decision rule remains identical in every scene and policy arm.
ROBUST_UPWARD_NORMAL_Z_MIN = 0.65
ROBUST_FLOOR_MIN_AREA_M2 = 1.0
ROBUST_FLOOR_MIN_SPAN_XY_M = (1.0, 1.0)
ROBUST_SURFACE_P95_MAX_M = 0.020
ROBUST_FLOOR_DOMINANCE_RATIO = 4.0
ROBUST_SUPPORT_MIN_AREA_M2 = 0.010
ROBUST_SUPPORT_MIN_PROJECTED_FILL = 0.25
ROBUST_SUPPORT_MIN_OVERLAP_M2 = 0.002
ROBUST_SUPPORT_MIN_OBJECT_FRACTION = 0.10
ROBUST_SUPPORT_BOTTOM_RANGE_M = (-0.005, 0.030)
ROBUST_SUPPORT_INTRUSION_TOL_M = 0.005
ROBUST_SUPPORT_THICKNESS_M = 0.020
ROBUST_SUPPORT_BOX_MIN_HALF_XY_M = ROBUST_SUPPORT_THICKNESS_M / 2.0
ROBUST_PROJECTED_OVERLAP_METHOD = "exact_triangle_union_clipped_to_object_aabb"

# legacy shim geometry constants, unchanged from export_mjcf.py
SHIM_SLAB = 0.02   # m, support slab thickness
SHIM_PAD = 0.10    # m, shim overhang past the object footprint, per side
SHIM_SAFE = 0.05   # m, clearance two objects need before they may share a channel


# ============================================================ extraction ==

@dataclasses.dataclass
class PlanePatch:
    kind: str            # "floor" | "support" | "wall"
    z: float             # fitted height (floor/support) or mean coord (wall)
    lo: np.ndarray        # box min corner, world xyz (3,)
    hi: np.ndarray        # box max corner, world xyz (3,)
    area: float
    residual_m: float     # planarity validation residual
    # Exact source-face membership is retained so a diagnostic variant can
    # demote an unsafe fitted primitive back into the residual decomposition
    # without silently losing the scan triangles represented by that patch.
    face_indices: "np.ndarray | None" = None


@dataclasses.dataclass
class RoomFeatures:
    floor: "PlanePatch | None"
    supports: list
    walls: list
    residual_mesh: object   # trimesh.Trimesh, may be empty (0 faces)
    source_tris: int
    classified_tris: int
    source_mesh: object = None


def validate_room_diagnostic_spec(raw, *, expected_scene_id=None):
    """Validate and canonicalize the sealed E4 room-diagnostic spec v1/v2.

    Both versions deliberately accept no undeclared keys: the file hash plus an
    exact schema makes every sweep cell independently replayable.  V2 is the
    exact V1 object plus ``room_surface_policy``; V1 and production/default
    exports retain their old validation and output bytes.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            "room diagnostic spec keys differ "
            f"(missing={sorted(ROOM_DIAGNOSTIC_SPEC_KEYS)}, extra=[])"
        )
    version = raw.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("unsupported room diagnostic spec schema")
    if version == ROOM_DIAGNOSTIC_SPEC_SCHEMA_VERSION:
        expected_keys = ROOM_DIAGNOSTIC_SPEC_KEYS
    elif version == ROOM_DIAGNOSTIC_SPEC_SCHEMA_VERSION_V2:
        expected_keys = ROOM_DIAGNOSTIC_SPEC_V2_KEYS
    else:
        raise ValueError("unsupported room diagnostic spec schema")
    if set(raw) != expected_keys:
        missing = sorted(expected_keys - set(raw))
        extra = sorted(set(raw) - expected_keys)
        raise ValueError(
            f"room diagnostic spec keys differ (missing={missing}, extra={extra})"
        )
    variant_id = raw["variant_id"]
    scene_id = raw["scene_id"]
    if not isinstance(variant_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", variant_id):
        raise ValueError("room diagnostic variant_id is invalid")
    if not isinstance(scene_id, str) or not re.fullmatch(r"[0-9a-f]{10}", scene_id):
        raise ValueError("room diagnostic scene_id is invalid")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError("room diagnostic spec scene binding differs")
    if raw["hull_bottom"] not in {"raw", "clip_scan_aabb_bottom"}:
        raise ValueError("room diagnostic hull_bottom is invalid")
    if raw["intrusive_primitive"] not in {"fail", "demote_to_residual"}:
        raise ValueError("room diagnostic intrusive_primitive is invalid")
    if raw["support_z"] not in {"extent", "fitted_mean_20mm"}:
        raise ValueError("room diagnostic support_z is invalid")
    if version == ROOM_DIAGNOSTIC_SPEC_SCHEMA_VERSION_V2:
        if raw["room_surface_policy"] not in {
                "robust_floor_only", "robust_floor_plus_support"}:
            raise ValueError("room diagnostic room_surface_policy is invalid")
        # These legacy fields stay in the exact extension so package consumers
        # can share one interface with V1.  V2 freezes them to their strictest
        # values; robust surface thresholds themselves are the constants above.
        if raw["hull_bottom"] != "clip_scan_aabb_bottom" \
                or raw["intrusive_primitive"] != "fail" \
                or raw["support_z"] != "fitted_mean_20mm":
            raise ValueError("room diagnostic v2 legacy policies differ from the freeze")

    def _finite_number(key):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not np.isfinite(value):
            raise ValueError(f"room diagnostic {key} must be finite")
        return float(value)

    residual = _finite_number("plane_residual_tol_m")
    allowed_residuals = ({0.02} if version == ROOM_DIAGNOSTIC_SPEC_SCHEMA_VERSION_V2
                         else {0.02, 0.025, 0.03})
    if residual not in allowed_residuals:
        raise ValueError("room diagnostic plane_residual_tol_m is outside the freeze")
    frozen_numbers = {
        "min_support_area_m2": 0.05,
        "carve_side_top_margin_m": 0.02,
        "lower_carve_margin_m": 0.0,
        "support_clip_offset_m": 0.005,
    }
    for key, expected in frozen_numbers.items():
        if _finite_number(key) != expected:
            raise ValueError(f"room diagnostic {key} differs from the freeze")
    # JSON-native canonical object (rather than a dataclass) keeps runner and
    # report serialization straightforward and makes equality exact.
    result = {
        "schema_version": version,
        "variant_id": variant_id,
        "scene_id": scene_id,
        "hull_bottom": raw["hull_bottom"],
        "intrusive_primitive": raw["intrusive_primitive"],
        "plane_residual_tol_m": residual,
        "support_z": raw["support_z"],
        "min_support_area_m2": float(raw["min_support_area_m2"]),
        "carve_side_top_margin_m": float(raw["carve_side_top_margin_m"]),
        "lower_carve_margin_m": float(raw["lower_carve_margin_m"]),
        "support_clip_offset_m": float(raw["support_clip_offset_m"]),
    }
    if version == ROOM_DIAGNOSTIC_SPEC_SCHEMA_VERSION_V2:
        result["room_surface_policy"] = raw["room_surface_policy"]
    return result


def clip_convex_hull_at_z(vertices, clip_z_m, *, name="convex hull"):
    """Intersect a convex hull with ``z >= clip_z_m`` and close the cut.

    Merely deleting below-plane vertices leaves an open/non-convex collision
    mesh.  We instead intersect every hull edge with the plane and convex-hull
    the kept + intersection points; the latter creates an explicit planar cap.
    The returned diagnostics retain the raw geometry needed to audit whether a
    sweep win came from removing a large collision-hull bulge.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 \
            or not np.isfinite(points).all():
        raise ValueError(f"invalid vertices for {name}")
    if isinstance(clip_z_m, bool) or not isinstance(clip_z_m, (int, float)) \
            or not np.isfinite(clip_z_m):
        raise ValueError(f"invalid clip plane for {name}")
    clip_z_m = float(clip_z_m)
    from scipy.spatial import ConvexHull
    import trimesh

    try:
        raw_hull = ConvexHull(points)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot construct raw {name}: {exc}") from exc
    raw_points = points[np.asarray(raw_hull.vertices, dtype=np.int64)]
    raw_min = float(raw_points[:, 2].min())
    raw_max = float(raw_points[:, 2].max())
    raw_volume = float(raw_hull.volume)
    tol = 1e-10
    if raw_max <= clip_z_m + tol:
        raise ValueError(f"clip plane removes all positive volume from {name}")

    if raw_min >= clip_z_m - tol:
        # scipy's ``simplices`` do not carry a consistent outward winding;
        # using them directly creates a watertight-looking mesh with zero
        # signed volume.  trimesh's hull wrapper fixes face orientation.
        mesh = trimesh.convex.convex_hull(points)
        return mesh, {
            "raw_min_z_m": raw_min,
            "raw_max_z_m": raw_max,
            "clip_z_m": clip_z_m,
            "clipped": False,
            "raw_hull_volume_m3": raw_volume,
            "clipped_hull_volume_m3": raw_volume,
            "retained_volume_fraction": 1.0,
            "cap_vertex_count": 0,
            "output_min_z_m": raw_min,
            "output_max_z_m": raw_max,
        }

    edges = set()
    for tri in np.asarray(raw_hull.simplices, dtype=np.int64):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.add(tuple(sorted((int(a), int(b)))))
    kept = [p.copy() for p in points if p[2] >= clip_z_m - tol]
    intersections = []
    for ia, ib in sorted(edges):
        a, b = points[ia], points[ib]
        da, db = float(a[2] - clip_z_m), float(b[2] - clip_z_m)
        if (da < -tol and db > tol) or (db < -tol and da > tol):
            alpha = (clip_z_m - a[2]) / (b[2] - a[2])
            p = a + alpha * (b - a)
            p[2] = clip_z_m
            intersections.append(p)
        elif abs(da) <= tol:
            p = a.copy(); p[2] = clip_z_m; intersections.append(p)
        elif abs(db) <= tol:
            p = b.copy(); p[2] = clip_z_m; intersections.append(p)
    candidate = np.asarray(kept + intersections, dtype=np.float64)
    if len(candidate):
        # Stable geometric de-duplication; inputs are metric world coordinates.
        candidate = np.unique(np.round(candidate, decimals=12), axis=0)
    if len(candidate) < 4:
        raise ValueError(f"clipped {name} has fewer than four vertices")
    try:
        clipped_hull = ConvexHull(candidate)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot cap clipped {name}: {exc}") from exc
    mesh = trimesh.convex.convex_hull(candidate)
    cap_count = int(np.isclose(
        np.asarray(mesh.vertices)[:, 2], clip_z_m, atol=1e-10, rtol=0.0
    ).sum())
    if cap_count < 3 or float(candidate[:, 2].min()) < clip_z_m - 1e-9:
        raise ValueError(f"clipped {name} lacks a closed planar cap")
    clipped_volume = float(clipped_hull.volume)
    return mesh, {
        "raw_min_z_m": raw_min,
        "raw_max_z_m": raw_max,
        "clip_z_m": clip_z_m,
        "clipped": True,
        "raw_hull_volume_m3": raw_volume,
        "clipped_hull_volume_m3": clipped_volume,
        "retained_volume_fraction": clipped_volume / raw_volume,
        "cap_vertex_count": cap_count,
        "output_min_z_m": float(candidate[:, 2].min()),
        "output_max_z_m": float(candidate[:, 2].max()),
    }


@dataclasses.dataclass(frozen=True)
class ConvexExclusion:
    """One transformed object-collision hull used to protect a carve.

    ``equations`` follows :class:`scipy.spatial.ConvexHull`: every interior
    point satisfies ``normal @ point + offset <= 0``.  The side and upper
    planes are expanded by the requested safety margin.  Downward-facing
    planes use ``lower_support_margin_m`` (normally zero), because expanding
    an object carve *through* the real tabletop would erase the very shared
    support that full-room collision is intended to measure.
    """

    name: str
    slot: str
    policy_id: str
    equations: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    margin_m: float
    lower_support_margin_m: float
    support_clip_z_m: "float | None"


def make_convex_exclusion(
    vertices,
    *,
    name: str,
    slot: str,
    policy_id: str,
    margin_m: float = CARVE_MARGIN_M,
    lower_support_margin_m: float = 0.0,
    support_clip_z_m: float | None = None,
) -> ConvexExclusion:
    """Build a finite, full-dimensional exclusion from world-space vertices."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 \
            or not np.isfinite(points).all():
        raise ValueError(f"invalid convex-exclusion vertices for {name}")
    if not isinstance(name, str) or not name or not isinstance(slot, str) or not slot \
            or not isinstance(policy_id, str) or not policy_id:
        raise ValueError("convex-exclusion identity must be non-empty strings")
    if not np.isfinite(margin_m) or margin_m < 0.0 \
            or not np.isfinite(lower_support_margin_m) or lower_support_margin_m < 0.0:
        raise ValueError("convex-exclusion margins must be finite and non-negative")
    if support_clip_z_m is not None and not np.isfinite(support_clip_z_m):
        raise ValueError("convex-exclusion support clip must be finite")

    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(points)
    except Exception as exc:  # noqa: BLE001 - qhull errors need object identity
        raise ValueError(f"cannot construct convex exclusion for {name}: {exc}") from exc
    equations = np.asarray(hull.equations, dtype=np.float64).copy()
    normals = equations[:, :3]
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0.0) or not np.isfinite(equations).all():
        raise ValueError(f"invalid convex-exclusion halfspaces for {name}")
    equations /= lengths[:, None]
    downward = equations[:, 2] < -0.5
    expansion = np.where(downward, lower_support_margin_m, margin_m)
    equations[:, 3] -= expansion
    if support_clip_z_m is not None:
        # Protect scan-derived support below the observed object bottom.  This
        # does not pardon a collision hull that bulges into that support: the
        # exporter separately audits the un-clipped hull and hard-fails it.
        equations = np.vstack([
            equations,
            np.array([0.0, 0.0, -1.0, float(support_clip_z_m)]),
        ])

    # Broad-phase bounds only.  Side/top expansion is conservatively bounded
    # by margin_m; the protected support side gets its independent margin.
    lo = points.min(axis=0) - np.array(
        [margin_m, margin_m, lower_support_margin_m], dtype=np.float64
    )
    if support_clip_z_m is not None:
        lo[2] = max(lo[2], float(support_clip_z_m))
    hi = points.max(axis=0) + margin_m
    return ConvexExclusion(
        name=name,
        slot=slot,
        policy_id=policy_id,
        equations=equations,
        lo=lo,
        hi=hi,
        margin_m=float(margin_m),
        lower_support_margin_m=float(lower_support_margin_m),
        support_clip_z_m=(None if support_clip_z_m is None
                          else float(support_clip_z_m)),
    )


def points_inside_exclusion(points, exclusion: ConvexExclusion, *,
                            interior_tol_m: float = CARVE_INTERIOR_TOL_M):
    """Strict-interior mask; contact on the unexpanded support side is kept."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("exclusion query points must have shape (N, 3)")
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    broad = ((values >= exclusion.lo) & (values <= exclusion.hi)).all(axis=1)
    result = np.zeros(len(values), dtype=bool)
    idx = np.flatnonzero(broad)
    if len(idx):
        signed = (values[idx] @ exclusion.equations[:, :3].T
                  + exclusion.equations[:, 3])
        result[idx] = (signed < -interior_tol_m).all(axis=1)
    return result


def convex_mesh_intrudes_exclusion(mesh, exclusion: ConvexExclusion, *,
                                   interior_tol_m: float = CARVE_INTERIOR_TOL_M) -> bool:
    """Whether two convex volumes have a positive-interior intersection.

    Vertex-only tests miss the common crossing-polytopes case.  Feasibility of
    the combined halfspaces is exact for these convex inputs; strict inward
    tolerances allow legitimate object/table boundary contact.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) < 4:
        return False
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    if np.any(hi <= exclusion.lo + interior_tol_m) \
            or np.any(exclusion.hi <= lo + interior_tol_m):
        return False
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull

    try:
        equations = np.asarray(ConvexHull(vertices).equations, dtype=np.float64)
    except Exception:
        return False
    a = np.vstack([equations[:, :3], exclusion.equations[:, :3]])
    b = np.concatenate([
        -equations[:, 3] - interior_tol_m,
        -exclusion.equations[:, 3] - interior_tol_m,
    ])
    result = linprog(
        np.zeros(3, dtype=np.float64), A_ub=a, b_ub=b,
        bounds=[(None, None)] * 3, method="highs",
    )
    return bool(result.success)


def support_intrusion_depth_m(lo_xy, hi_xy, surface_z_m, exclusion,
                              *, slab_thickness_m=SHIM_SLAB,
                              interior_tol_m=CARVE_INTERIOR_TOL_M):
    """Maximum vertical depth of a protected convex hull below a support top.

    The linear program minimizes z inside the exact hull while constraining x/y
    to the support footprint and z to lie strictly below the top.  A depth of
    zero therefore means disjoint or boundary-only contact; callers compare the
    finite result against the frozen 5 mm tolerance.
    """
    lo_xy = np.asarray(lo_xy, dtype=np.float64)
    hi_xy = np.asarray(hi_xy, dtype=np.float64)
    if lo_xy.shape != (2,) or hi_xy.shape != (2,) \
            or not np.isfinite(lo_xy).all() or not np.isfinite(hi_xy).all() \
            or np.any(hi_xy <= lo_xy):
        raise ValueError("support intrusion footprint is invalid")
    if isinstance(surface_z_m, bool) or not isinstance(surface_z_m, (int, float)) \
            or not np.isfinite(surface_z_m):
        raise ValueError("support intrusion surface height is invalid")
    if isinstance(interior_tol_m, bool) \
            or not isinstance(interior_tol_m, (int, float)) \
            or not np.isfinite(interior_tol_m) or interior_tol_m < 0.0:
        raise ValueError("support intrusion tolerance is invalid")
    if isinstance(slab_thickness_m, bool) \
            or not isinstance(slab_thickness_m, (int, float)) \
            or not np.isfinite(slab_thickness_m) or slab_thickness_m <= 0.0:
        raise ValueError("support intrusion slab thickness is invalid")
    surface_z_m = float(surface_z_m)
    slab_thickness_m = float(slab_thickness_m)
    interior_tol_m = float(interior_tol_m)
    xy_lo = np.maximum(lo_xy, np.asarray(exclusion.lo[:2], dtype=np.float64))
    xy_hi = np.minimum(hi_xy, np.asarray(exclusion.hi[:2], dtype=np.float64))
    z_lo = max(float(exclusion.lo[2]), surface_z_m - slab_thickness_m)
    z_hi = min(float(exclusion.hi[2]), surface_z_m - interior_tol_m)
    if np.any(xy_hi <= xy_lo + 2.0 * interior_tol_m) \
            or z_hi <= z_lo + interior_tol_m:
        return 0.0
    from scipy.optimize import linprog

    equations = np.asarray(exclusion.equations, dtype=np.float64)
    result = linprog(
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        A_ub=equations[:, :3],
        b_ub=-equations[:, 3] - interior_tol_m,
        bounds=[
            (float(xy_lo[0] + interior_tol_m),
             float(xy_hi[0] - interior_tol_m)),
            (float(xy_lo[1] + interior_tol_m),
             float(xy_hi[1] - interior_tol_m)),
            (z_lo + interior_tol_m, z_hi),
        ],
        method="highs",
    )
    if not result.success:
        return 0.0
    minimum_z = float(result.x[2])
    depth = surface_z_m - minimum_z
    if not np.isfinite(depth) or depth < 0.0:
        raise ValueError("support intrusion depth is invalid")
    return depth


def _face_clusters(mesh, face_mask):
    """Union-find connected components of `mesh.faces[face_mask]`, using
    edge-adjacency restricted to the masked subset. Returns a list of
    face-index arrays (each a cluster), all indices into the FULL mesh."""
    idx = np.nonzero(face_mask)[0]
    if len(idx) == 0:
        return []
    pos = {f: i for i, f in enumerate(idx)}
    parent = list(range(len(idx)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    adj = mesh.face_adjacency  # (M,2) face index pairs sharing an edge
    if len(adj):
        m = face_mask[adj[:, 0]] & face_mask[adj[:, 1]]
        for f0, f1 in adj[m]:
            union(pos[f0], pos[f1])

    groups: dict = {}
    for i, f in enumerate(idx):
        groups.setdefault(find(i), []).append(f)
    return [np.asarray(g, dtype=np.int64) for g in groups.values()]


def _patch_from_cluster(mesh, face_idx, kind):
    verts = mesh.vertices[np.unique(mesh.faces[face_idx].reshape(-1))]
    area = float(mesh.area_faces[face_idx].sum())
    if kind == "wall":
        # fit the plane's own normal-axis coordinate (mean over cluster faces).
        # Guard against a "ring" artifact: a connected loop of faces wrapping
        # a corner (e.g. a floor slab's own perimeter skirt, or two walls
        # meeting at 90 deg) is one face-adjacency cluster but is NOT one
        # plane, and its raw normals cancel toward zero on averaging -- which
        # would make every vertex alias to ~0 on that near-null axis and the
        # residual check spuriously pass. Reject (residual=inf) whenever the
        # cluster's unit normals disagree too much for a single-plane fit to
        # be meaningful; it falls through to exact CoACD decomposition
        # instead of a wrong/oversized box.
        raw = mesh.face_normals[face_idx].mean(axis=0)
        consistency = float(np.linalg.norm(raw))
        if consistency < 0.8:
            return PlanePatch(kind=kind, z=0.0, lo=verts.min(axis=0) if len(verts) else
                              np.zeros(3), hi=verts.max(axis=0) if len(verts) else
                              np.zeros(3), area=area, residual_m=float("inf"),
                              face_indices=np.asarray(face_idx, dtype=np.int64).copy())
        n = raw / consistency
        coord = verts @ n
        z = float(coord.mean())
        residual = float(np.max(np.abs(coord - z))) if len(coord) else 0.0
    else:
        z = float(verts[:, 2].mean())
        residual = float(np.max(np.abs(verts[:, 2] - z))) if len(verts) else 0.0
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    return PlanePatch(kind=kind, z=z, lo=lo, hi=hi, area=area, residual_m=residual,
                      face_indices=np.asarray(face_idx, dtype=np.int64).copy())


def area_weighted_quantile(values, weights, quantile):
    """Deterministic lower area-weighted quantile for finite 1-D samples."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape \
            or len(values) == 0 or not np.isfinite(values).all() \
            or not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("area-weighted quantile inputs must be finite positive vectors")
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)) \
            or not np.isfinite(quantile) or not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("area-weighted quantile must be finite and in [0, 1]")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("area-weighted quantile total weight must be finite and positive")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    target = float(quantile) * total
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _clip_polygon_to_aabb_xy(polygon, lo_xy, hi_xy):
    """Clip one ordered convex polygon to an axis-aligned XY rectangle."""
    polygon = np.asarray(polygon, dtype=np.float64)
    lo_xy = np.asarray(lo_xy, dtype=np.float64)
    hi_xy = np.asarray(hi_xy, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2 \
            or lo_xy.shape != (2,) or hi_xy.shape != (2,) \
            or not np.isfinite(polygon).all() or not np.isfinite(lo_xy).all() \
            or not np.isfinite(hi_xy).all() or np.any(hi_xy <= lo_xy):
        raise ValueError("projected-overlap clipping inputs are invalid")

    result = polygon.copy()
    for axis, bound, keep_above in (
            (0, lo_xy[0], True), (0, hi_xy[0], False),
            (1, lo_xy[1], True), (1, hi_xy[1], False)):
        if len(result) == 0:
            break
        clipped = []
        previous = result[-1]
        previous_inside = (previous[axis] >= bound if keep_above
                           else previous[axis] <= bound)
        for current in result:
            current_inside = (current[axis] >= bound if keep_above
                              else current[axis] <= bound)
            if current_inside != previous_inside:
                denominator = current[axis] - previous[axis]
                if denominator == 0.0:
                    raise ValueError("projected-overlap clipping edge is invalid")
                alpha = (bound - previous[axis]) / denominator
                crossing = previous + alpha * (current - previous)
                crossing[axis] = bound
                clipped.append(crossing)
            if current_inside:
                clipped.append(current.copy())
            previous = current
            previous_inside = current_inside
        result = np.asarray(clipped, dtype=np.float64).reshape((-1, 2))
    return result


def _polygon_area_xy(polygon):
    polygon = np.asarray(polygon, dtype=np.float64)
    if len(polygon) < 3:
        return 0.0
    return 0.5 * abs(float(
        np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
        - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
    ))


def _proper_segment_crossing_x(a, b, c, d):
    """X of an interior 2-D segment crossing, else ``None``.

    Polygon vertices are already scan-line breakpoints.  Only proper interior
    crossings can change edge ordering inside an open vertical slab; collinear
    overlaps and endpoint touches therefore need no additional breakpoint.
    """
    r = b - a
    s = d - c
    denominator = float(r[0] * s[1] - r[1] * s[0])
    scale = max(float(np.linalg.norm(r)), float(np.linalg.norm(s)), 1.0)
    if abs(denominator) <= np.finfo(np.float64).eps * scale * scale * 16.0:
        return None
    delta = c - a
    t = float((delta[0] * s[1] - delta[1] * s[0]) / denominator)
    u = float((delta[0] * r[1] - delta[1] * r[0]) / denominator)
    epsilon = 1e-12
    if epsilon < t < 1.0 - epsilon and epsilon < u < 1.0 - epsilon:
        return float(a[0] + t * r[0])
    return None


def _polygon_vertical_interval(polygon, x):
    values = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        low, high = sorted((float(a[0]), float(b[0])))
        if x < low or x > high:
            continue
        dx = float(b[0] - a[0])
        if dx == 0.0:
            if x == float(a[0]):
                values.extend((float(a[1]), float(b[1])))
            continue
        alpha = (x - float(a[0])) / dx
        if 0.0 <= alpha <= 1.0:
            values.append(float(a[1] + alpha * (b[1] - a[1])))
    if not values:
        return None
    return min(values), max(values)


def _projected_polygon_union_area(polygons):
    """Exact (up to float arithmetic) union area of clipped convex polygons.

    A vertical sweep integrates the union of Y intervals.  Polygon vertex X
    coordinates and every proper edge crossing partition the plane into slabs
    on which interval ordering is fixed, hence union length is linear and the
    trapezoidal integral is exact.  This avoids double-counting repeated or
    folded projected triangles and remains dependency-free.
    """
    polygons = [np.asarray(polygon, dtype=np.float64) for polygon in polygons
                if len(polygon) >= 3 and _polygon_area_xy(polygon) > 0.0]
    if not polygons:
        return 0.0
    edges = [
        (a, b)
        for polygon in polygons
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0))
        if not np.array_equal(a, b)
    ]
    critical = [float(point[0]) for polygon in polygons for point in polygon]
    for index, (a, b) in enumerate(edges):
        for c, d in edges[index + 1:]:
            crossing = _proper_segment_crossing_x(a, b, c, d)
            if crossing is not None:
                critical.append(crossing)
    critical.sort()
    scale = max(max(abs(value) for value in critical), 1.0)
    merge_tol = np.finfo(np.float64).eps * scale * 64.0
    xs = []
    for value in critical:
        if not xs or value - xs[-1] > merge_tol:
            xs.append(value)
        else:
            xs[-1] = (xs[-1] + value) / 2.0

    def union_length(x):
        intervals = sorted(
            interval for interval in (
                _polygon_vertical_interval(polygon, x) for polygon in polygons
            ) if interval is not None and interval[1] > interval[0]
        )
        if not intervals:
            return 0.0
        total = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                total += end - start
                start, end = next_start, next_end
        return total + end - start

    area = 0.0
    for left, right in zip(xs, xs[1:]):
        width = right - left
        if width <= merge_tol:
            continue
        # Midpoint subdivision makes the calculation robust to a numerically
        # coincident crossing while retaining exact integration of linear spans.
        middle = (left + right) / 2.0
        area += (middle - left) * (
            union_length(left) + union_length(middle)
        ) / 2.0
        area += (right - middle) * (
            union_length(middle) + union_length(right)
        ) / 2.0
    if not np.isfinite(area) or area < 0.0:
        raise ValueError("projected triangle union area is invalid")
    return float(area)


def projected_triangle_aabb_overlap_m2(triangles_xy, lo_xy, hi_xy):
    """Union area of projected source triangles inside an object XY AABB."""
    triangles_xy = np.asarray(triangles_xy, dtype=np.float64)
    lo_xy = np.asarray(lo_xy, dtype=np.float64)
    hi_xy = np.asarray(hi_xy, dtype=np.float64)
    if triangles_xy.ndim != 3 or triangles_xy.shape[1:] != (3, 2) \
            or len(triangles_xy) == 0 or not np.isfinite(triangles_xy).all() \
            or lo_xy.shape != (2,) or hi_xy.shape != (2,) \
            or not np.isfinite(lo_xy).all() or not np.isfinite(hi_xy).all() \
            or np.any(hi_xy <= lo_xy):
        raise ValueError("projected triangle/AABB overlap inputs are invalid")
    polygons = [
        clipped for clipped in (
            _clip_polygon_to_aabb_xy(triangle, lo_xy, hi_xy)
            for triangle in triangles_xy
        ) if len(clipped) >= 3 and _polygon_area_xy(clipped) > 0.0
    ]
    area = _projected_polygon_union_area(polygons)
    rectangle_area = float(np.prod(hi_xy - lo_xy))
    tolerance = max(1e-12, rectangle_area * 1e-9)
    if area > rectangle_area + tolerance:
        raise ValueError("projected triangle union exceeds object footprint")
    return min(area, rectangle_area)


def _robust_discovered_objects(objects):
    """Validate the discovered roster used by the frozen support association."""
    if not isinstance(objects, list):
        raise ValueError("robust surface discovered objects must be a list")
    rows = []
    seen = set()
    for raw in objects:
        if not isinstance(raw, dict):
            raise ValueError("robust surface discovered object is invalid")
        index = raw.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("robust surface discovered object index is invalid")
        slot = f"obj_{index:02d}"
        if slot in seen:
            raise ValueError("robust surface discovered object roster is duplicated")
        seen.add(slot)
        aabb = np.asarray(raw.get("aabb"), dtype=np.float64)
        if aabb.shape != (2, 3) or not np.isfinite(aabb).all() \
                or np.any(aabb[1] <= aabb[0]):
            raise ValueError(f"robust surface scan AABB is invalid: {slot}")
        rows.append((slot, aabb))
    return rows


def _robust_horizontal_component(mesh, face_idx, component_id, objects):
    """Compute all frozen metrics for one shared-edge upward component."""
    face_idx = np.asarray(face_idx, dtype=np.int64)
    face_areas = np.asarray(mesh.area_faces[face_idx], dtype=np.float64)
    centroids_z = np.asarray(mesh.triangles_center[face_idx, 2], dtype=np.float64)
    normals_z = np.asarray(mesh.face_normals[face_idx, 2], dtype=np.float64)
    if len(face_idx) == 0 or not np.isfinite(face_areas).all() \
            or np.any(face_areas <= 0.0) or not np.isfinite(centroids_z).all() \
            or not np.isfinite(normals_z).all() \
            or np.any(normals_z <= ROBUST_UPWARD_NORMAL_Z_MIN):
        raise ValueError("robust upward component contains invalid face metrics")
    vertices = np.asarray(
        mesh.vertices[np.unique(mesh.faces[face_idx].reshape(-1))], dtype=np.float64
    )
    if len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError("robust upward component contains invalid vertices")
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    spans = hi[:2] - lo[:2]
    area = float(face_areas.sum())
    z50 = area_weighted_quantile(centroids_z, face_areas, 0.50)
    r95 = area_weighted_quantile(np.abs(centroids_z - z50), face_areas, 0.95)
    max_residual = float(np.max(np.abs(vertices[:, 2] - z50)))
    projected_area = float(np.sum(face_areas * normals_z))
    footprint_area = float(spans[0] * spans[1])
    projected_fill = projected_area / footprint_area if footprint_area > 0.0 else 0.0
    metrics = (area, z50, r95, max_residual, projected_area, footprint_area,
               projected_fill)
    if not all(np.isfinite(value) for value in metrics) or area <= 0.0 \
            or projected_area <= 0.0 or projected_fill < 0.0:
        raise ValueError("robust upward component metrics are non-finite")

    triangles_xy = np.asarray(mesh.triangles[face_idx, :, :2], dtype=np.float64)
    # Use the parser-visible box here too: a raw component edge can move by
    # almost 0.1 mm when both center and half extent are serialized to four
    # decimals.  The false-overlap safeguard must cover exactly what MuJoCo
    # will receive, including that near-boundary expansion.
    box_center, box_half = _xml_quantized_box_center_half(
        lo,
        hi,
        hang_below=ROBUST_SUPPORT_THICKNESS_M,
        min_half=ROBUST_SUPPORT_BOX_MIN_HALF_XY_M,
    )
    box_center_xy = box_center[:2]
    box_half_xy = box_half[:2]
    emitted_box_lo_xy = box_center_xy - box_half_xy
    emitted_box_hi_xy = box_center_xy + box_half_xy
    associations = []
    box_false_overlap_rejections = []
    for slot, aabb in objects:
        overlap_lo = np.maximum(emitted_box_lo_xy, aabb[0, :2])
        overlap_hi = np.minimum(emitted_box_hi_xy, aabb[1, :2])
        box_overlap_extent = np.maximum(overlap_hi - overlap_lo, 0.0)
        box_overlap = float(box_overlap_extent[0] * box_overlap_extent[1])
        object_area = float(np.prod(aabb[1, :2] - aabb[0, :2]))
        box_overlap_fraction = box_overlap / object_area
        bottom_delta = float(aabb[0, 2] - z50)
        height_matches = (
            ROBUST_SUPPORT_BOTTOM_RANGE_M[0] <= bottom_delta
            <= ROBUST_SUPPORT_BOTTOM_RANGE_M[1]
        )
        box_spatially_matches = (
            box_overlap >= ROBUST_SUPPORT_MIN_OVERLAP_M2
            and box_overlap_fraction >= ROBUST_SUPPORT_MIN_OBJECT_FRACTION
        )
        # The source triangles are bounded by the (possibly min-half-expanded)
        # emitted box, so a spatially insufficient box proves the exact overlap
        # insufficient too.  This broad phase avoids an expensive polygon-union
        # sweep for the overwhelming majority of component/object pairs.
        projected_overlap = (
            projected_triangle_aabb_overlap_m2(
                triangles_xy, aabb[0, :2], aabb[1, :2]
            )
            if height_matches and box_spatially_matches else 0.0
        )
        projected_overlap_fraction = projected_overlap / object_area
        projected_matches = (
            projected_overlap >= ROBUST_SUPPORT_MIN_OVERLAP_M2
            and projected_overlap_fraction >= ROBUST_SUPPORT_MIN_OBJECT_FRACTION
            and height_matches
        )
        box_matches = box_spatially_matches and height_matches
        if projected_matches:
            associations.append({
                "slot": slot,
                "overlap_xy_m2": projected_overlap,
                "overlap_object_fraction": projected_overlap_fraction,
                "scan_bottom_minus_z50_m": bottom_delta,
            })
        if box_matches and not projected_matches:
            box_false_overlap_rejections.append({
                "slot": slot,
                "emitted_box_overlap_xy_m2": box_overlap,
                "emitted_box_overlap_object_fraction": box_overlap_fraction,
                "projected_overlap_xy_m2": projected_overlap,
                "projected_overlap_object_fraction": projected_overlap_fraction,
                "scan_bottom_minus_z50_m": bottom_delta,
            })
    associations.sort(key=lambda row: row["slot"])
    box_false_overlap_rejections.sort(key=lambda row: row["slot"])
    eligible_floor = bool(
        area >= ROBUST_FLOOR_MIN_AREA_M2
        and spans[0] >= ROBUST_FLOOR_MIN_SPAN_XY_M[0]
        and spans[1] >= ROBUST_FLOOR_MIN_SPAN_XY_M[1]
        and r95 <= ROBUST_SURFACE_P95_MAX_M
    )
    diagnostic = {
        "component_id": int(component_id),
        "face_count": int(len(face_idx)),
        "area_m2": area,
        "z50_m": z50,
        "weighted_p95_residual_m": r95,
        "max_vertex_residual_m": max_residual,
        "span_xy_m": [float(spans[0]), float(spans[1])],
        "bounds_xy_m": [lo[:2].tolist(), hi[:2].tolist()],
        "projected_aabb_fill": projected_fill,
        "projected_overlap_method": ROBUST_PROJECTED_OVERLAP_METHOD,
        "box_footprint_safe": not box_false_overlap_rejections,
        "box_false_overlap_rejections": box_false_overlap_rejections,
        "eligible_floor": eligible_floor,
        "support_associations": associations,
        "selected_as": "residual",
    }
    patch = PlanePatch(
        kind="support",
        z=z50,
        lo=lo.copy(),
        hi=hi.copy(),
        area=area,
        residual_m=r95,
        face_indices=face_idx.copy(),
    )
    return {"face_indices": face_idx, "patch": patch, "diagnostic": diagnostic}


def extract_robust_room_features(
    mesh,
    objects,
    room_surface_policy,
    *,
    legacy_support_residual_tol=PLANE_RESIDUAL_TOL,
    legacy_min_support_area=MIN_SUPPORT_AREA,
    min_wall_area=MIN_WALL_AREA,
):
    """Extract frozen F/FS surfaces from one decimated paired-union mesh.

    Floor selection is robust to sparse outlier triangles because both height
    and residual use triangle-area weights.  F keeps the conservative legacy
    max-residual rule for non-floor supports; FS alone admits the smaller,
    associated weighted-p95 supports.  Every selected face is consumed exactly
    once, and all unselected geometry remains in the residual decomposition.
    """
    if room_surface_policy not in {
            "robust_floor_only", "robust_floor_plus_support"}:
        raise ValueError("robust room surface policy is invalid")
    for label, value in {
        "legacy support residual tolerance": legacy_support_residual_tol,
        "legacy minimum support area": legacy_min_support_area,
        "minimum wall area": min_wall_area,
    }.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be finite and non-negative")
    if len(mesh.faces) == 0:
        raise ValueError("robust floor cannot be selected from an empty mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() \
            or faces.ndim != 2 or faces.shape[1] != 3 \
            or normals.shape != (len(faces), 3) or not np.isfinite(normals).all() \
            or areas.shape != (len(faces),) or not np.isfinite(areas).all() \
            or np.any(areas < 0.0):
        raise ValueError("robust floor source mesh contains invalid geometry")
    discovered = _robust_discovered_objects(objects)
    upward = (normals[:, 2] > ROBUST_UPWARD_NORMAL_Z_MIN) & (areas > 0.0)
    components = [
        _robust_horizontal_component(mesh, face_idx, index, discovered)
        for index, face_idx in enumerate(_face_clusters(mesh, upward))
    ]
    eligible = sorted(
        (row for row in components if row["diagnostic"]["eligible_floor"]),
        key=lambda row: (-row["patch"].area, row["diagnostic"]["component_id"]),
    )
    if not eligible:
        raise ValueError("robust floor has no eligible upward connected component")
    floor_row = eligible[0]
    runner_up = eligible[1] if len(eligible) > 1 else None
    ratio = (floor_row["patch"].area / runner_up["patch"].area
             if runner_up is not None else None)
    if ratio is not None and ratio < ROBUST_FLOOR_DOMINANCE_RATIO:
        raise ValueError(
            "robust floor is ambiguous: largest eligible component is less "
            f"than {ROBUST_FLOOR_DOMINANCE_RATIO:g}x the runner-up"
        )

    used = np.zeros(len(faces), dtype=bool)
    floor_patch = floor_row["patch"]
    floor_patch.kind = "floor"
    used[floor_row["face_indices"]] = True
    floor_row["diagnostic"]["selected_as"] = "floor"

    supports = []
    selected_support_ids = []
    associated_slots = set()
    for row in components:
        if row is floor_row:
            continue
        diagnostic = row["diagnostic"]
        if room_surface_policy == "robust_floor_only":
            selected = (
                diagnostic["area_m2"] >= float(legacy_min_support_area)
                and diagnostic["max_vertex_residual_m"]
                <= float(legacy_support_residual_tol)
                and diagnostic["box_footprint_safe"]
            )
        else:
            selected = (
                diagnostic["area_m2"] >= ROBUST_SUPPORT_MIN_AREA_M2
                and diagnostic["weighted_p95_residual_m"]
                <= ROBUST_SURFACE_P95_MAX_M
                and diagnostic["projected_aabb_fill"]
                >= ROBUST_SUPPORT_MIN_PROJECTED_FILL
                and len(diagnostic["support_associations"]) >= 1
                and diagnostic["box_footprint_safe"]
            )
        if not selected:
            continue
        patch = row["patch"]
        patch.kind = "support"
        patch.residual_m = (diagnostic["max_vertex_residual_m"]
                            if room_surface_policy == "robust_floor_only"
                            else diagnostic["weighted_p95_residual_m"])
        supports.append(patch)
        used[row["face_indices"]] = True
        diagnostic["selected_as"] = "support"
        selected_support_ids.append(diagnostic["component_id"])
        associated_slots.update(
            association["slot"] for association in diagnostic["support_associations"]
        )

    # Preserve the established wall rule.  Robust policies alter only upward
    # floor/support classification, so vertical default bytes remain untouched.
    walls = []
    vertical = np.abs(normals[:, 2]) < WALL_NORMAL_TOL
    vert_idx = np.nonzero(vertical & ~used)[0]
    if len(vert_idx):
        vn = normals[vert_idx]
        axis = np.argmax(np.abs(vn[:, :2]), axis=1)
        sign = np.sign(vn[np.arange(len(vn)), axis])
        bucket_id = axis * 2 + (sign > 0)
        buckets = [np.zeros(len(faces), dtype=bool) for _ in range(4)]
        for bucket, face_index in zip(bucket_id, vert_idx):
            buckets[bucket][face_index] = True
    else:
        buckets = []
    for bucket_mask in buckets:
        for face_idx in _face_clusters(mesh, bucket_mask):
            patch = _patch_from_cluster(mesh, face_idx, "wall")
            if patch.area >= min_wall_area \
                    and patch.residual_m <= legacy_support_residual_tol:
                walls.append(patch)
                used[face_idx] = True

    residual_idx = np.flatnonzero(~used)
    residual_mesh = (mesh.submesh([residual_idx], append=True) if len(residual_idx)
                     else mesh.submesh([[]], append=True))
    diagnostics = {
        "schema_version": 1,
        "policy": room_surface_policy,
        "guards": {
            "upward_normal_z_min_exclusive": ROBUST_UPWARD_NORMAL_Z_MIN,
            "floor_min_area_m2": ROBUST_FLOOR_MIN_AREA_M2,
            "floor_min_span_xy_m": list(ROBUST_FLOOR_MIN_SPAN_XY_M),
            "floor_max_weighted_p95_residual_m": ROBUST_SURFACE_P95_MAX_M,
            "floor_dominance_min_area_ratio": ROBUST_FLOOR_DOMINANCE_RATIO,
            "support_min_area_m2": ROBUST_SUPPORT_MIN_AREA_M2,
            "support_max_weighted_p95_residual_m": ROBUST_SURFACE_P95_MAX_M,
            "support_min_projected_aabb_fill": ROBUST_SUPPORT_MIN_PROJECTED_FILL,
            "support_min_overlap_xy_m2": ROBUST_SUPPORT_MIN_OVERLAP_M2,
            "support_min_object_footprint_fraction":
                ROBUST_SUPPORT_MIN_OBJECT_FRACTION,
            "support_bottom_minus_surface_range_m":
                list(ROBUST_SUPPORT_BOTTOM_RANGE_M),
            "support_intrusion_tolerance_m": ROBUST_SUPPORT_INTRUSION_TOL_M,
            "support_box_min_half_extent_xy_m":
                ROBUST_SUPPORT_BOX_MIN_HALF_XY_M,
            "support_projected_overlap_method": ROBUST_PROJECTED_OVERLAP_METHOD,
            "support_box_false_overlap_policy": "reject_component",
        },
        "floor": {
            "candidate_count": int(len(components)),
            "eligible_count": int(len(eligible)),
            "selected_component_id": floor_row["diagnostic"]["component_id"],
            "area_m2": float(floor_patch.area),
            "z50_m": float(floor_patch.z),
            "weighted_p95_residual_m": floor_row["diagnostic"][
                "weighted_p95_residual_m"
            ],
            "span_xy_m": list(floor_row["diagnostic"]["span_xy_m"]),
            "bounds_xy_m": [list(point) for point in floor_row["diagnostic"][
                "bounds_xy_m"
            ]],
            "runner_up_eligible_area_m2": (
                None if runner_up is None else float(runner_up["patch"].area)
            ),
            "area_ratio_to_next_eligible": None if ratio is None else float(ratio),
            "dominance_passed": True,
        },
        "supports": {
            "selection_mode": (
                "legacy_max_residual" if room_surface_policy == "robust_floor_only"
                else "robust_associated_weighted_p95"
            ),
            "candidate_count": int(max(len(components) - 1, 0)),
            "selected_count": int(len(supports)),
            "selected_component_ids": selected_support_ids,
            "rejected_count": 0,
            "emitted_count": int(len(supports)),
            "rejected_component_ids": [],
            "emitted_component_ids": list(selected_support_ids),
            "associated_slots": sorted(associated_slots),
            "emitted_associated_slots": sorted(associated_slots),
        },
        "components": [row["diagnostic"] for row in components],
    }
    features = RoomFeatures(
        floor_patch,
        supports,
        walls,
        residual_mesh,
        source_tris=len(faces),
        classified_tris=int(used.sum()),
        source_mesh=mesh,
    )
    return features, diagnostics


def extract_room_features(
    mesh,
    floor_normal_tol=FLOOR_NORMAL_TOL,
    wall_normal_tol=WALL_NORMAL_TOL,
    plane_residual_tol=PLANE_RESIDUAL_TOL,
    min_support_area=MIN_SUPPORT_AREA,
    min_wall_area=MIN_WALL_AREA,
    expected_floor_z=None,
    floor_z_tol=SCANNETPP_FLOOR_TOL_M,
) -> RoomFeatures:
    """Classify a room/background mesh's faces into floor / support-surface /
    wall primitives (validated: planar within `plane_residual_tol`, big
    enough to matter) plus a residual mesh of everything else -- irregular
    obstacles, small clutter-shaped surfaces, anything that failed planarity.
    The residual is NOT dropped; callers convex-decompose it (see
    `coacd_decompose` below) so no collision volume is silently lost, per the
    plan's "primitives only for validated major support planes" -- floor and
    tabletops become clean box primitives, unvalidated or under-area regions
    just fall through to exact (if slower) convex decomposition instead.
    """
    if len(mesh.faces) == 0:
        return RoomFeatures(None, [], [], mesh.copy(), 0, 0, source_mesh=mesh)

    normals = mesh.face_normals
    up_dot = normals @ UP
    horiz_up = up_dot > (1.0 - floor_normal_tol)
    vertical = np.abs(up_dot) < wall_normal_tol

    used = np.zeros(len(mesh.faces), dtype=bool)
    supports = []
    floor_patch = None
    for face_idx in _face_clusters(mesh, horiz_up):
        patch = _patch_from_cluster(mesh, face_idx, "support")
        if patch.area >= min_support_area and patch.residual_m <= plane_residual_tol:
            supports.append(patch)
            used[face_idx] = True
        # else: leave unclassified -> residual mesh (too small / not planar)

    if supports and expected_floor_z is None:
        # Generic/fixture mode retains the legacy area-aware heuristic.
        area_max = max(p.area for p in supports)
        floor_candidates = [p for p in supports if p.area >= 0.3 * area_max]
        floor_patch = min(floor_candidates, key=lambda p: p.z)
        floor_patch.kind = "floor"
        supports = [p for p in supports if p is not floor_patch]
    elif supports:
        if not np.isfinite(expected_floor_z) or not np.isfinite(floor_z_tol) \
                or floor_z_tol < 0.0:
            raise ValueError("expected floor z/tolerance must be finite")
        # ScanNet++ has a calibrated z=0 floor.  If the cropped mesh contains
        # only tabletops, *none* is a floor; keep them all as supports instead
        # of relabelling the lowest high patch and deleting it from support.
        floor_candidates = [
            p for p in supports if abs(p.z - expected_floor_z) <= floor_z_tol
        ]
        if floor_candidates:
            floor_patch = max(
                floor_candidates,
                key=lambda p: (p.area, -abs(p.z - expected_floor_z)),
            )
            floor_patch.kind = "floor"
            supports = [p for p in supports if p is not floor_patch]

    # Bucket vertical faces by dominant horizontal normal axis+sign BEFORE
    # connectivity clustering. Without this, a thin double-sided wall/panel
    # mesh (both faces present, joined by its thin side edges) face-adjacency
    # clusters into one ring spanning two opposite-facing planes, which
    # `_patch_from_cluster`'s consistency guard then has to reject outright.
    # Splitting by direction first means each cluster is genuinely one plane,
    # so real opposite-facing wall surfaces (e.g. two rooms sharing a wall)
    # are still each captured as their own validated primitive.
    walls = []
    vert_idx = np.nonzero(vertical & ~used)[0]
    if len(vert_idx):
        vn = normals[vert_idx]
        axis = np.argmax(np.abs(vn[:, :2]), axis=1)          # 0=x, 1=y dominant
        sign = np.sign(vn[np.arange(len(vn)), axis])
        bucket_id = axis * 2 + (sign > 0)                    # 0..3
        buckets = [np.zeros(len(mesh.faces), dtype=bool) for _ in range(4)]
        for b, f in zip(bucket_id, vert_idx):
            buckets[b][f] = True
    else:
        buckets = []
    for bucket_mask in buckets:
        for face_idx in _face_clusters(mesh, bucket_mask):
            patch = _patch_from_cluster(mesh, face_idx, "wall")
            if patch.area >= min_wall_area and patch.residual_m <= plane_residual_tol:
                walls.append(patch)
                used[face_idx] = True

    residual_idx = np.nonzero(~used)[0]
    residual_mesh = mesh.submesh([residual_idx], append=True) if len(residual_idx) \
        else mesh.submesh([[]], append=True)
    return RoomFeatures(floor_patch, supports, walls, residual_mesh,
                        source_tris=len(mesh.faces),
                        classified_tris=int(used.sum()), source_mesh=mesh)


# ============================================================ decimation ==

def decimate_mesh(mesh, voxel_size=0.02, max_tris=DECIMATE_MAX_TRIS):
    """Vertex-clustering decimation (same trick as `s7_sim.build_background`)
    with a bounded two-sided Hausdorff-ish error estimate between the
    decimated mesh and the source, computed by nearest-surface-point
    sampling in both directions (robust p95 + true max)."""
    if len(mesh.faces) == 0:
        return mesh.copy(), {"hausdorff_p95_m": 0.0, "hausdorff_max_m": 0.0,
                             "tris_before": 0, "tris_after": 0}
    if len(mesh.faces) <= max_tris:
        # under budget -> an exact copy, not an approximation; skip the
        # Monte-Carlo distance estimate below (comparing two independent
        # random samples of the SAME surface would report nonzero "error"
        # that is really just sampling-density noise, not geometric drift)
        return mesh.copy(), {"hausdorff_p95_m": 0.0, "hausdorff_max_m": 0.0,
                             "tris_before": len(mesh.faces), "tris_after": len(mesh.faces)}
    else:
        from agents.assets.s5_align import open3d_registration_backend
        import trimesh

        o3d = open3d_registration_backend()
        decimated = mesh
        # Vertex-clustering reduction depends on voxel_size vs. mesh scale,
        # not a triangle target -- on a real room mesh, one pass at the
        # default 2 cm voxel barely dents a dense scan (180k -> 165k
        # triangles, verified on scene 38d58a7a31.faces). Double voxel_size
        # until under budget (or give up after a bounded number of
        # doublings and keep the best effort -- a coarser-than-requested
        # residual still gets convex-decomposed correctly downstream, it is
        # just less finely diced).
        vs = voxel_size
        for _ in range(6):
            o3 = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh.vertices),
                o3d.utility.Vector3iVector(mesh.faces))
            o3 = o3.simplify_vertex_clustering(voxel_size=vs)
            o3.remove_degenerate_triangles()
            o3.remove_unreferenced_vertices()
            cand = trimesh.Trimesh(
                np.asarray(o3.vertices), np.asarray(o3.triangles), process=False)
            if len(cand.faces) == 0:
                break  # degenerate collapse -> stop, keep the last good candidate
            decimated = cand
            if len(decimated.faces) <= max_tris:
                break
            vs *= 2.0
        if len(decimated.faces) == 0:  # every attempt degenerate -> keep source
            decimated = mesh.copy()

    # Sample POINTS at a fixed target DENSITY (per m^2), not a fixed count:
    # a fixed n=4000 badly understates error on a full-room mesh (tens to
    # hundreds of m^2) -- on a real 100+ m^2 scan, 4000 points gives ~10 cm
    # average nearest-neighbour spacing even between a mesh and an EXACT
    # copy of itself, which would misreport pure sampling sparsity as
    # decimation error. Both meshes have >=1 face here: source was already
    # checked non-empty above, and `decimated` falls back to a copy of
    # `mesh` if it collapsed.
    TARGET_DENSITY = 2500.0  # points / m^2 (~2 cm avg NN spacing)
    n = int(np.clip(max(mesh.area, decimated.area) * TARGET_DENSITY, 4000, 150_000))
    import trimesh
    pa = trimesh.sample.sample_surface(mesh, n)[0]
    pb = trimesh.sample.sample_surface(decimated, n)[0]
    from scipy.spatial import cKDTree
    d_ab = cKDTree(pb).query(pa)[0] if len(pb) else np.array([np.inf])
    d_ba = cKDTree(pa).query(pb)[0] if len(pa) else np.array([np.inf])
    d = np.concatenate([d_ab, d_ba])
    return decimated, {
        "hausdorff_p95_m": float(np.percentile(d, 95)),
        "hausdorff_max_m": float(d.max()),
        "tris_before": int(len(mesh.faces)),
        "tris_after": int(len(decimated.faces)),
    }


# ==================================================== convex decomposition =

def coacd_decompose(mesh, out_dir, prefix, threshold=COACD_THRESHOLD,
                    max_parts=COACD_MAX_PARTS, *, exclusions=(),
                    return_stats=False, collect_diagnostics=False):
    """CoACD convex decomposition of an arbitrary (possibly multi-component,
    non-watertight) mesh -> (collision/<prefix>_part_*.obj paths, meshes),
    same order. Mirrors
    `agents/assets/s6_physics.py:coacd_parts`'s exact import order
    (torch BEFORE coacd) -- importing torch after coacd runs segfaults on
    some builds (bundled libgomp clash), verified independently
    while building this module.

    Decomposed PER CONNECTED COMPONENT rather than as one CoACD call over
    the whole residual: the residual mesh here is deliberately a grab-bag
    (real obstacles + carved-out debris ribbons left over from the
    floor/support/wall classification), and one big CoACD call would
    remesh/merge all of it into a single watertight blob, smearing real
    obstacle geometry together with unrelated debris and making per-obstacle
    collision unpredictable. Splitting first means a real closed obstacle
    mesh decomposes cleanly on its own, and a CoACD failure on one
    (typically degenerate) component falls back to just that component's
    convex hull instead of discarding the whole batch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(mesh.faces) == 0:
        empty = ([], [], {"candidate_parts": 0, "rejected_intrusions": 0})
        return empty if return_stats else empty[:2]
    import trimesh
    try:
        import torch  # noqa: F401  MUST precede `import coacd` (see docstring)
        import coacd
        have_coacd = True
    except ImportError:
        have_coacd = False

    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        components = [mesh]
    n_hull_fastpath = n_dropped = 0
    dropped_faces = 0
    dropped_area_m2 = 0.0
    meshes = []
    for comp in components:
        if len(comp.faces) == 0:
            continue
        got = []
        if have_coacd and len(comp.faces) >= COACD_MIN_FACES:
            try:
                cmesh = coacd.Mesh(np.asarray(comp.vertices), np.asarray(comp.faces))
                parts = coacd.run_coacd(cmesh, threshold=threshold, max_convex_hull=max_parts)
                got = [trimesh.Trimesh(pv, pf, process=False) for pv, pf in parts]
            except Exception as e:  # noqa: BLE001  this component -> hull fallback
                print(f"[room_collision] coacd_decompose: component fallback "
                      f"to convex hull ({prefix}): {e}")
        if not got:
            try:
                # a pathologically degenerate component (e.g. <4 non-coplanar
                # points after triangle-soup dedup -- verified on a real
                # scene, 38d58a7a31) makes qhull itself raise, not just
                # return a zero-volume hull; that case is dropped here
                # rather than crashing the whole export, same spirit as the
                # zero-volume filter below.
                got = [comp.convex_hull]
                n_hull_fastpath += 1
            except Exception as e:  # noqa: BLE001  qhull failure -> drop
                print(f"[room_collision] coacd_decompose: dropping a "
                      f"degenerate residual component ({prefix}, "
                      f"{len(comp.faces)} faces): {e}")
                n_dropped += 1
                dropped_faces += int(len(comp.faces))
                dropped_area_m2 += float(comp.area)
                continue
        meshes.extend(got)
    if len(components) > 20:
        n_coacd = len(components) - n_hull_fastpath - n_dropped
        print(f"[room_collision] coacd_decompose({prefix}): {len(components)} "
              f"residual components, {n_hull_fastpath} took the plain-convex-"
              f"hull fast path (< {COACD_MIN_FACES} faces or CoACD unavailable/"
              f"failed), {n_coacd} ran full CoACD, {n_dropped} dropped "
              f"(degenerate, qhull could not even build a hull)")

    files, kept_meshes = [], []
    rejected_intrusions = 0
    rejected_intrusion_details = []
    zero_volume_candidates = 0
    for candidate_index, m in enumerate(meshes):
        vol = float(m.convex_hull.volume) if len(m.vertices) >= 4 else 0.0
        # `not (vol >= 1e-9)` rather than `vol < 1e-9`: a fully flat/coplanar
        # residual sliver (e.g. an isolated single quad left over from
        # classification) makes trimesh's mass-properties divide by zero and
        # return NaN, and `nan < 1e-9` is False in plain float comparison --
        # the mirror-image bug of export_mjcf.py's object filter would let a
        # NaN-volume mesh part slip through uncaught.
        if len(m.vertices) < 4 or not np.isfinite(vol) or vol < 1e-9:
            zero_volume_candidates += 1
            continue  # degenerate sliver, mirrors export_mjcf.py's object filter
        if collect_diagnostics:
            overlaps = [
                exclusion.name for exclusion in exclusions
                if convex_mesh_intrudes_exclusion(m, exclusion)
            ]
        else:
            overlaps = ["<unrecorded>"] if any(
                convex_mesh_intrudes_exclusion(m, exclusion)
                for exclusion in exclusions
            ) else []
        if overlaps:
            # Convex decomposition can bridge a cavity even when the source
            # triangles were correctly carved.  Publishing such a part would
            # put static room collision back inside the reconstructed object.
            rejected_intrusions += 1
            if collect_diagnostics:
                bounds = np.asarray(m.bounds, dtype=np.float64)
                rejected_intrusion_details.append({
                    "candidate_index": int(candidate_index),
                    "face_count": int(len(m.faces)),
                    "volume_m3": vol,
                    "bounds_m": bounds.tolist(),
                    "exclusions": sorted(overlaps),
                })
            continue
        k = len(files)
        f = out_dir / f"{prefix}_part_{k:02d}.obj"
        m.export(f)
        files.append(f)
        kept_meshes.append(m)
    stats = {
        "candidate_parts": len(meshes),
        "rejected_intrusions": rejected_intrusions,
    }
    if collect_diagnostics:
        n_coacd = len(components) - n_hull_fastpath - n_dropped
        stats["raw_diagnostics"] = {
            "residual_component_count": int(len(components)),
            "convex_hull_fastpath_component_count": int(n_hull_fastpath),
            "coacd_component_count": int(n_coacd),
            "degenerate_component_count": int(n_dropped),
            "degenerate_component_face_count": int(dropped_faces),
            "degenerate_component_area_m2": float(dropped_area_m2),
            "candidate_part_count_before_filters": int(len(meshes)),
            "zero_volume_candidate_count": int(zero_volume_candidates),
            "intrusion_rejected_candidate_count": int(rejected_intrusions),
            "intrusion_rejected_candidates": rejected_intrusion_details,
            "emitted_part_count": int(len(files)),
        }
    result = (files, kept_meshes, stats)
    return result if return_stats else result[:2]


# ============================================================== MJCF emit =

def _box_center_half(lo, hi, hang_below=None, min_half=0.01):
    """Shared centre/half-extent math for a static room box, from a
    world-space AABB (lo, hi) -- floored per-axis at `min_half` so a
    degenerate (single-face) cluster still gets a real collision volume
    instead of a zero-thickness box.

    `hang_below`: for floor/support patches the cluster is the (zero-
    thickness) TOP face only, and we know solid material is strictly below
    it -- so instead of the symmetric min-half padding, extrude the box
    downward by `hang_below` from the true surface height (lo.z == hi.z
    already) rather than straddling it. Walls get the symmetric case: we do
    not know which side is "solid" from a single detected face, so padding
    both ways is the safe default (a small conservative overshoot into open
    space beats a gap the arm can poke through)."""
    lo = np.asarray(lo, dtype=np.float64).copy()
    hi = np.asarray(hi, dtype=np.float64).copy()
    if hang_below is not None:
        lo[2] -= hang_below
    c = (lo + hi) / 2
    h = np.maximum((hi - lo) / 2, min_half)
    return c, h


def _xml_quantized_box_center_half(lo, hi, hang_below=None, min_half=0.01):
    """Box vectors after the exact four-decimal `_box_geom` serialization."""
    c, h = _box_center_half(lo, hi, hang_below, min_half)
    quantize = lambda values: np.asarray(  # noqa: E731 - local exact transform
        [float(f"{value:.4f}") for value in values], dtype=np.float64
    )
    return quantize(c), quantize(h)


def _box_geom(name, lo, hi, mask, friction, rgba, hang_below=None, min_half=0.01):
    """Static MJCF box geom string from a world-space AABB -- see
    `_box_center_half` for the placement rules."""
    c, h = _box_center_half(lo, hi, hang_below, min_half)
    cx, cy, cz = c
    hx, hy, hz = h
    return (f'    <geom name="{name}" type="box" pos="{cx:.4f} {cy:.4f} {cz:.4f}" '
            f'size="{hx:.4f} {hy:.4f} {hz:.4f}" contype="{mask}" conaffinity="{mask}" '
            f'friction="{friction:.2f} 0.005 0.0001" rgba="{rgba}"/>')


def _box_trimesh(lo, hi, hang_below=None, min_half=0.01, *, xml_quantized=False):
    """The same box as `_box_geom`, as a trimesh.Trimesh -- used to build the
    true-collision-representation mesh for coverage/penetration metrics."""
    import trimesh
    if not isinstance(xml_quantized, bool):
        raise ValueError("box XML quantization flag is invalid")
    if xml_quantized:
        # `_box_geom` serializes both vectors with exactly four decimals.  The
        # robust 5 mm gate must audit those parser-visible numbers, not their
        # full-precision precursors (which can differ by roughly 0.1 mm after
        # center and half-extent errors combine at one boundary).
        c, h = _xml_quantized_box_center_half(
            lo, hi, hang_below=hang_below, min_half=min_half
        )
    else:
        c, h = _box_center_half(lo, hi, hang_below, min_half)
    box = trimesh.creation.box(extents=2 * h)
    box.apply_translation(c)
    return box


def emit_room_mjcf(features: RoomFeatures, out_dir: Path,
                   mask=1, friction=0.7, floor_thickness=0.02, wall_thickness=0.05,
                   coacd_threshold=COACD_THRESHOLD, coacd_max_parts=COACD_MAX_PARTS,
                   *, exclusions=(), intrusive_primitive="fail",
                   support_z="extent", collect_diagnostics=False,
                   emit_floor_primitive=True,
                   primitive_intrusion_tolerance_m=0.0,
                   room_surface_policy=None,
                   primitive_exclusions=None):
    """features -> (asset_xml_lines, geom_xml_lines, manifest, collision_mesh).
    Geoms are static (no <body>, no <freejoint>) and sit directly in
    <worldbody>, all on `mask` (default MuJoCo channel bit 0, same
    convention the floor plane and object bit-0 already use) -- one shared
    room collision group rather than a parallel numbering scheme.

    `collision_mesh` (a trimesh.Trimesh, may be empty) is the union of
    every emitted primitive box + CoACD part, in the SAME world frame as
    `features` -- i.e. exactly what MuJoCo will actually collide against.
    Callers should compare THIS (not the pre-classification decimated mesh)
    against the true source mesh for coverage/penetration metrics, since
    the box-primitive placement (hang_below/min_half padding) is itself a
    source of geometric deviation the metric should catch."""
    if intrusive_primitive not in {"fail", "demote_to_residual"}:
        raise ValueError("intrusive primitive policy is invalid")
    if support_z not in {"extent", "fitted_mean_20mm"}:
        raise ValueError("support z policy is invalid")
    if isinstance(floor_thickness, bool) \
            or not isinstance(floor_thickness, (int, float)) \
            or not np.isfinite(floor_thickness) or floor_thickness <= 0.0:
        raise ValueError("floor thickness is invalid")
    floor_thickness = float(floor_thickness)
    if not isinstance(emit_floor_primitive, bool):
        raise ValueError("floor primitive emission flag is invalid")
    if isinstance(primitive_intrusion_tolerance_m, bool) \
            or not isinstance(primitive_intrusion_tolerance_m, (int, float)) \
            or not np.isfinite(primitive_intrusion_tolerance_m) \
            or primitive_intrusion_tolerance_m < 0.0 \
            or primitive_intrusion_tolerance_m >= floor_thickness:
        raise ValueError("primitive intrusion tolerance is invalid")
    primitive_intrusion_tolerance_m = float(primitive_intrusion_tolerance_m)
    exclusions = tuple(exclusions)
    primitive_exclusions = (exclusions if primitive_exclusions is None
                            else tuple(primitive_exclusions))
    if room_surface_policy is not None:
        if room_surface_policy not in {
                "robust_floor_only", "robust_floor_plus_support"}:
            raise ValueError("robust emission surface policy is invalid")
        if emit_floor_primitive or support_z != "fitted_mean_20mm" \
                or intrusive_primitive != "fail" \
                or not collect_diagnostics \
                or floor_thickness != ROBUST_SUPPORT_THICKNESS_M \
                or primitive_intrusion_tolerance_m \
                != ROBUST_SUPPORT_INTRUSION_TOL_M:
            raise ValueError("robust emission contract differs from the freeze")
    assets, geoms, mesh_parts = [], [], []
    manifest = {"support_boxes": 0, "wall_boxes": 0, "coacd_parts": 0,
                "coacd_source_tris": len(features.residual_mesh.faces)
                if features.residual_mesh is not None else 0,
                "exclusion_hulls": len(exclusions),
                "coacd_candidate_parts": 0,
                "coacd_parts_rejected_exclusion": 0,
                "coacd_emitted_intrusions": [],
                "primitive_intrusions": []}

    raw_primitive_intrusions = []
    primitive_demotions = []
    support_rejections = []
    patch_diagnostics = []
    demoted_faces = []
    robust_demoted_faces = []

    def _patch_bounds(p, kind):
        lo = np.asarray(p.lo, dtype=np.float64).copy()
        hi = np.asarray(p.hi, dtype=np.float64).copy()
        if kind in {"floor", "support"} and support_z == "fitted_mean_20mm":
            lo[2] = hi[2] = float(p.z)
        return lo, hi

    def _record_primitive(name, mesh, patch, kind, intrusion_depths=None):
        if intrusion_depths is None:
            all_overlaps = [
                exclusion.name for exclusion in exclusions
                if convex_mesh_intrudes_exclusion(mesh, exclusion)
            ]
            overlaps = all_overlaps
            tolerated = []
        else:
            all_overlaps = [row["protected_hull_id"] for row in intrusion_depths
                            if row["depth_m"] > 0.0]
            severe_rows = [
                row for row in intrusion_depths
                if row["depth_m"] > primitive_intrusion_tolerance_m
            ]
            overlaps = [row["protected_hull_id"] for row in severe_rows]
            tolerated = sorted(set(all_overlaps) - set(overlaps))
        if collect_diagnostics:
            row = {
                "geom": name,
                "kind": kind,
                "area_m2": float(patch.area),
                "fitted_z_m": float(patch.z),
                "raw_bounds_m": [
                    np.asarray(patch.lo, dtype=np.float64).tolist(),
                    np.asarray(patch.hi, dtype=np.float64).tolist(),
                ],
                "residual_m": float(patch.residual_m),
                "emitted_bounds_m": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
                "intruding_exclusions": sorted(overlaps),
            }
            if intrusion_depths is not None:
                row["tolerated_exclusions"] = tolerated
                row["penetration_depths_m"] = intrusion_depths
            patch_diagnostics.append(row)
        if overlaps:
            row = {"geom": name, "exclusions": sorted(overlaps)}
            raw_primitive_intrusions.append(row)
            if kind == "support" \
                    and room_surface_policy == "robust_floor_plus_support":
                if features.source_mesh is None or patch.face_indices is None:
                    raise ValueError(
                        f"cannot losslessly reject robust support {name}: "
                        "source faces are absent"
                    )
                ids = np.asarray(patch.face_indices, dtype=np.int64)
                if ids.ndim != 1 or len(ids) == 0 \
                        or len(np.unique(ids)) != len(ids) \
                        or np.any(ids < 0) \
                        or np.any(ids >= len(features.source_mesh.faces)):
                    raise ValueError(
                        f"cannot losslessly reject robust support {name}: "
                        "source face provenance is invalid"
                    )
                robust_demoted_faces.extend(ids.tolist())
                severe_rows = sorted(
                    (depth for depth in intrusion_depths
                     if depth["protected_hull_id"] in set(overlaps)),
                    key=lambda depth: depth["protected_hull_id"],
                )
                support_rejections.append({
                    "geom": name,
                    "reason": "penetration_depth_gt_0.005_m",
                    "protected_hull_ids": sorted(overlaps),
                    "penetration_depths_m": severe_rows,
                    "max_penetration_depth_m": max(
                        depth["depth_m"] for depth in severe_rows
                    ),
                    "source_face_count": int(len(ids)),
                    "source_area_m2": float(patch.area),
                })
                return False
            if intrusive_primitive == "demote_to_residual":
                if features.source_mesh is None or patch.face_indices is None:
                    raise ValueError(
                        f"cannot losslessly demote {name}: source faces are absent"
                    )
                ids = np.asarray(patch.face_indices, dtype=np.int64)
                demoted_faces.extend(ids.tolist())
                primitive_demotions.append({
                    **row,
                    "source_face_count": int(len(ids)),
                    "source_area_m2": float(patch.area),
                })
                return False
            manifest["primitive_intrusions"].append(row)
        mesh_parts.append(mesh)
        return True

    if features.floor is not None and emit_floor_primitive:
        p = features.floor
        lo, hi = _patch_bounds(p, "floor")
        mesh = _box_trimesh(lo, hi, hang_below=floor_thickness,
                            min_half=floor_thickness / 2)
        if _record_primitive("room_floor", mesh, p, "floor"):
            geoms.append(_box_geom("room_floor", lo, hi, mask, friction,
                                   "0.55 0.55 0.55 1", hang_below=floor_thickness,
                                   min_half=floor_thickness / 2))
    for i, p in enumerate(features.supports):
        lo, hi = _patch_bounds(p, "support")
        name = f"room_support_{i}"
        mesh = _box_trimesh(lo, hi, hang_below=floor_thickness,
                            min_half=floor_thickness / 2,
                            xml_quantized=room_surface_policy is not None)
        intrusion_depths = None
        if primitive_intrusion_tolerance_m > 0.0:
            emitted_bounds = np.asarray(mesh.bounds, dtype=np.float64)
            intrusion_depths = sorted(({
                "protected_hull_id": exclusion.name,
                "depth_m": support_intrusion_depth_m(
                    emitted_bounds[0, :2], emitted_bounds[1, :2],
                    float(emitted_bounds[1, 2]), exclusion,
                    slab_thickness_m=floor_thickness,
                ),
            } for exclusion in primitive_exclusions),
                key=lambda row: row["protected_hull_id"])
        if _record_primitive(name, mesh, p, "support", intrusion_depths):
            geoms.append(_box_geom(name, lo, hi, mask, friction,
                                   "0.75 0.68 0.55 1", hang_below=floor_thickness,
                                   min_half=floor_thickness / 2))
            manifest["support_boxes"] += 1
    for i, p in enumerate(features.walls):
        # wall box: lo/hi are already the true world AABB of the detected
        # face (degenerate along the face's own normal axis); pad that one
        # axis out to wall_thickness/2 on each side (see _box_geom docstring
        # for why walls get symmetric padding instead of hang_below).
        name = f"room_wall_{i}"
        mesh = _box_trimesh(p.lo, p.hi, min_half=wall_thickness / 2)
        if _record_primitive(name, mesh, p, "wall"):
            geoms.append(_box_geom(name, p.lo, p.hi, mask, friction,
                                   "0.6 0.6 0.65 1", min_half=wall_thickness / 2))
            manifest["wall_boxes"] += 1

    residual = features.residual_mesh
    residual_faces_before_demotion = int(len(residual.faces)) \
        if residual is not None else 0
    robust_rebuild_face_count = residual_faces_before_demotion
    robust_face_inventory_validated = False
    robust_original_residual_ids = None
    robust_classified = []
    if room_surface_policy is not None:
        if features.source_mesh is None:
            raise ValueError("robust emission lacks the source mesh")
        source_face_count = int(len(features.source_mesh.faces))
        patches = ([features.floor] if features.floor is not None else []) \
            + list(features.supports) + list(features.walls)
        for patch in patches:
            if patch.face_indices is None:
                raise ValueError("robust emission lacks classified face provenance")
            ids = np.asarray(patch.face_indices, dtype=np.int64)
            if ids.ndim != 1 or len(ids) == 0 \
                    or len(np.unique(ids)) != len(ids) \
                    or np.any(ids < 0) or np.any(ids >= source_face_count):
                raise ValueError("robust emission face provenance is invalid")
            robust_classified.extend(ids.tolist())
        if len(robust_classified) != len(set(robust_classified)):
            raise ValueError("robust classified face provenance overlaps")
        robust_original_residual_ids = np.setdiff1d(
            np.arange(source_face_count, dtype=np.int64),
            np.asarray(robust_classified, dtype=np.int64),
            assume_unique=False,
        )
        if len(robust_original_residual_ids) != residual_faces_before_demotion:
            raise ValueError("robust residual source-face inventory differs")
        robust_face_inventory_validated = True
    if robust_demoted_faces:
        demoted_ids = np.unique(np.asarray(robust_demoted_faces, dtype=np.int64))
        if len(demoted_ids) != len(robust_demoted_faces) \
                or not set(demoted_ids.tolist()).issubset(set(robust_classified)):
            raise ValueError("robust support demotion face inventory differs")
        residual_ids = np.sort(np.concatenate([
            robust_original_residual_ids, demoted_ids
        ]))
        if len(np.unique(residual_ids)) != len(residual_ids):
            raise ValueError("robust rebuilt residual face inventory overlaps")
        residual = features.source_mesh.submesh([residual_ids], append=True)
        robust_rebuild_face_count = int(len(residual.faces))
        if robust_rebuild_face_count \
                != residual_faces_before_demotion + len(demoted_ids):
            raise ValueError("robust residual reconstruction lost source faces")
        manifest["coacd_source_tris"] = robust_rebuild_face_count
    if demoted_faces:
        import trimesh
        ids = np.unique(np.asarray(demoted_faces, dtype=np.int64))
        demoted_mesh = features.source_mesh.submesh([ids], append=True)
        residual = trimesh.util.concatenate(
            [part for part in (residual, demoted_mesh)
            if part is not None and len(part.faces)]
        )
        manifest["coacd_source_tris"] = int(len(residual.faces))
    if residual is not None and len(residual.faces) > 0:
        parts, part_meshes, decomposition = coacd_decompose(
            residual, out_dir, "room", threshold=coacd_threshold,
            max_parts=coacd_max_parts, exclusions=exclusions, return_stats=True,
            collect_diagnostics=collect_diagnostics)
        for k, f in enumerate(parts):
            mesh_name = f"room_obstacle_{k}"
            # ALWAYS absolute: export_mjcf.py's <compiler meshdir="{out}"/>
            # prepends meshdir to any relative `file` -- if `out_dir` (and
            # hence `f`) was itself given relative to cwd (the common case;
            # docs/ROBOT.md's own examples set SIMANY_OUT to a relative
            # path), a relative `f` here gets meshdir prepended TWICE,
            # producing a doubled, nonexistent path. Verified on a real
            # scene (38d58a7a31): pytest's tmp_path fixtures are always
            # absolute, so the synthetic-fixture tests never hit this --
            # only a real SIMANY_OUT-driven export did.
            assets.append(f'    <mesh name="{mesh_name}" file="{f.resolve()}"/>')
            geoms.append(
                f'    <geom name="{mesh_name}" type="mesh" mesh="{mesh_name}" '
                f'contype="{mask}" conaffinity="{mask}" group="3" '
                f'friction="{friction:.2f} 0.005 0.0001" rgba="0.6 0.4 0.4 1"/>')
        mesh_parts.extend(part_meshes)
        manifest["coacd_parts"] = len(parts)
        manifest["coacd_candidate_parts"] = decomposition["candidate_parts"]
        manifest["coacd_parts_rejected_exclusion"] = decomposition[
            "rejected_intrusions"
        ]
        emitted_intrusions = []
        for index, part in enumerate(part_meshes):
            overlaps = [
                exclusion.name for exclusion in exclusions
                if convex_mesh_intrudes_exclusion(part, exclusion)
            ]
            if overlaps:
                emitted_intrusions.append({
                    "geom": f"room_obstacle_{index}",
                    "exclusions": sorted(overlaps),
                })
        manifest["coacd_emitted_intrusions"] = emitted_intrusions
        if robust_demoted_faces \
                and decomposition.get("candidate_parts", 0) <= 0:
            raise ValueError("robust demoted residual decomposition produced no candidates")

    if collect_diagnostics:
        manifest["raw_diagnostics"] = {
            "intrusive_primitive_policy": intrusive_primitive,
            "support_z_policy": support_z,
            "classified_patches": patch_diagnostics,
            "primitive_intrusions_before_policy": raw_primitive_intrusions,
            "primitive_demotions": primitive_demotions,
            "demoted_source_face_count": int(len(set(demoted_faces))),
            "residual_face_count_before_demotion": residual_faces_before_demotion,
            "residual_face_count_after_demotion": int(len(residual.faces))
                if residual is not None else 0,
            "decomposition": decomposition.get("raw_diagnostics", {})
                if residual is not None and len(residual.faces) > 0 else {},
        }
        if not emit_floor_primitive or primitive_intrusion_tolerance_m > 0.0:
            demoted_unique_count = int(len(set(robust_demoted_faces)))
            manifest["raw_diagnostics"]["support_rejections"] = support_rejections
            manifest["raw_diagnostics"]["robust_support_demotion"] = {
                "rejected_support_count": int(len(support_rejections)),
                "demoted_source_face_count": demoted_unique_count,
                "residual_rebuild_face_count": int(robust_rebuild_face_count),
                "face_inventory_validated": bool(robust_face_inventory_validated),
            }
            manifest["raw_diagnostics"]["robust_emission"] = {
                "floor_representation": (
                    "room_floor_box" if emit_floor_primitive else "global_plane_only"
                ),
                "primitive_intrusion_tolerance_m":
                    primitive_intrusion_tolerance_m,
                "room_surface_policy": room_surface_policy,
            }

    import trimesh
    collision_mesh = (trimesh.util.concatenate(mesh_parts) if mesh_parts
                      else trimesh.Trimesh())
    return assets, geoms, manifest, collision_mesh


# ============================================================ shim ablation =

def greedy_channel_color(footprints, safe_margin=SHIM_SAFE):
    """footprints: list of (lo_xy, hi_xy) PADDED boxes -> {index: channel}.
    Verbatim port of export_mjcf.py's greedy adjacency colouring: two
    footprints may share a channel only if they are >= safe_margin apart, so
    a private shim on that channel can never reach the neighbour."""
    n = len(footprints)
    adj = [set() for _ in range(n)]
    for i, (lo_i, hi_i) in enumerate(footprints):
        for j, (lo_j, hi_j) in enumerate(footprints):
            if i < j and (np.minimum(hi_i + safe_margin, hi_j)
                          > np.maximum(lo_i - safe_margin, lo_j)).all():
                adj[i].add(j)
                adj[j].add(i)
    chan = {}
    for i in sorted(range(n), key=lambda k: -len(adj[k])):
        used = {chan[j] for j in adj[i] if j in chan}
        chan[i] = next(c for c in range(n + 1) if c not in used)
    assert not chan or max(chan.values()) < 30, \
        f"{max(chan.values()) + 1} support channels exceeds the 31-bit mask"
    return chan


def build_shim_geoms(supports, pad=SHIM_PAD, slab=SHIM_SLAB, safe_margin=SHIM_SAFE):
    """supports: list of (z_bottom, lo_xy, hi_xy) per object (its own
    collision-hull bottom + footprint, in world xy) -> (masks, slab_xml_lines).

    `masks[i]` is ONLY the object's private greedy-coloured CHANNEL bit
    (matching the original `export_mjcf.py`) -- the slab geom carries
    exactly this value, so it is invisible to every other object, the
    floor, and the robot. The CALLER is responsible for giving the object's
    OWN collision geoms `1 | masks[i]` (bit 0, shared with the floor/other
    objects/robot, PLUS this private channel bit so it also matches its own
    slab). This is the legacy ablation path, unchanged from the original
    `export_mjcf.py`."""
    if not supports:
        return [], []
    foot = [(lo - pad, hi + pad) for _, lo, hi in supports]
    chan = greedy_channel_color(foot, safe_margin)
    masks = [1 << (1 + chan[i]) for i in range(len(supports))]
    slabs = []
    for i, (z, _, _) in enumerate(supports):
        lo, hi = foot[i]
        cx, cy = (lo + hi) / 2
        hx, hy = (hi - lo) / 2
        slabs.append(
            f'    <geom name="support_{i}" type="box" '
            f'pos="{cx:.4f} {cy:.4f} {z - 0.001 - slab / 2:.4f}" '
            f'size="{hx:.4f} {hy:.4f} {slab / 2:.4f}" '
            f'contype="{masks[i]}" conaffinity="{masks[i]}" '
            f'friction="0.6 0.005 0.0001" rgba="0.9 0.85 0.7 0.4"/>')
    return masks, slabs


# ==================================================== coverage / penetration

def collision_coverage_metrics(source_mesh, collision_mesh_union, n_rays=1500,
                               n_points=3000, seed=0):
    """Ray and point coverage of a static collision representation against
    the source mesh it was built from.

    - ray coverage: cast rays from outside the combined bounding box toward
      sampled source-surface points; a ray "agrees" if the collision union
      is also hit and the two hit distances match within a tolerance. This
      is a real proxy for "does the collision geometry occlude/support the
      same surface the visual mesh shows", not a hardcoded pass.
    - point occupancy: random points in the shared bounding box; agreement
      rate between `source_mesh.contains` and `collision_mesh_union.contains`
      is an IoU-style occupancy match, and disagreements where the
      collision union claims "inside" but the source does not are counted
      as penetration volume proxies (the collision hull bulging past the
      true surface -- CoACD is known to do this, see export_mjcf.py's own
      comments on hull bulge).
    """
    rng = np.random.default_rng(seed)
    out = {"n_rays": 0, "ray_hit_agreement": None, "ray_dist_p95_m": None,
          "n_points": 0, "occupancy_agreement": None, "penetration_frac": None}
    if len(source_mesh.faces) == 0:
        return out

    import trimesh
    # cast from a point offset along the outward normal back toward the
    # surface it came from -- if the collision representation matches, it
    # is hit at (about) the same distance.
    surf_pts, face_ids = trimesh.sample.sample_surface(
        source_mesh, min(n_rays, 20000), seed=seed
    )
    normals = source_mesh.face_normals[face_ids]
    origins = surf_pts + normals * 0.5
    directions = -normals

    ray_ok = True
    try:
        if collision_mesh_union is not None and len(collision_mesh_union.faces):
            col_loc, col_idx, _ = collision_mesh_union.ray.intersects_location(
                origins, directions, multiple_hits=False)
        else:
            col_loc, col_idx = np.zeros((0, 3)), np.zeros((0,), dtype=np.int64)
        src_loc, src_idx, _ = source_mesh.ray.intersects_location(
            origins, directions, multiple_hits=False)
    except ImportError as e:
        # trimesh's tree-based ray intersector needs `rtree` (or pyembree);
        # coverage is a diagnostic extra, not core collision, so degrade to
        # "not computed" rather than fail the whole export/benchmark.
        print(f"[room_collision] ray-based coverage unavailable ({e}); "
              f"skipping ray_hit_agreement/ray_dist_p95_m")
        ray_ok = False
        src_loc = src_idx = col_loc = col_idx = np.zeros((0,), dtype=np.int64)

    n = len(origins)
    out["n_rays"] = n
    if ray_ok:
        src_dist_map = {}
        for loc, i in zip(src_loc, src_idx):
            d = float(np.linalg.norm(loc - origins[i]))
            if i not in src_dist_map or d < src_dist_map[i]:
                src_dist_map[i] = d
        col_dist_map = {}
        for loc, i in zip(col_loc, col_idx):
            d = float(np.linalg.norm(loc - origins[i]))
            if i not in col_dist_map or d < col_dist_map[i]:
                col_dist_map[i] = d

        both = sorted(set(src_dist_map) & set(col_dist_map))
        agree = len(both) / max(len(src_dist_map), 1)
        diffs = [abs(src_dist_map[i] - col_dist_map[i]) for i in both]
        out["ray_hit_agreement"] = float(agree)
        out["ray_dist_p95_m"] = float(np.percentile(diffs, 95)) if diffs else None

    lo = np.minimum(source_mesh.bounds[0],
                    collision_mesh_union.bounds[0] if collision_mesh_union is not None
                    and len(collision_mesh_union.faces) else source_mesh.bounds[0])
    hi = np.maximum(source_mesh.bounds[1],
                    collision_mesh_union.bounds[1] if collision_mesh_union is not None
                    and len(collision_mesh_union.faces) else source_mesh.bounds[1])
    pts = rng.uniform(lo, hi, size=(n_points, 3))
    src_in = source_mesh.contains(pts)
    if collision_mesh_union is not None and len(collision_mesh_union.faces):
        col_in = collision_mesh_union.contains(pts)
    else:
        col_in = np.zeros(n_points, dtype=bool)
    out["n_points"] = n_points
    out["occupancy_agreement"] = float((src_in == col_in).mean())
    pen = col_in & ~src_in
    out["penetration_frac"] = float(pen.mean())
    return out


# ===================================================== benchmark / smoke ===

def state_hash(data, ndigits=6):
    """Deterministic fingerprint of a MuJoCo qpos/qvel state, rounded to
    dodge harmless ULP-level float noise while still catching any real
    divergence."""
    arr = np.concatenate([np.round(data.qpos, ndigits), np.round(data.qvel, ndigits)])
    return hashlib.sha256(arr.tobytes()).hexdigest()


def benchmark_model(model_xml_path, steps=1000, body_names=None,
                    push=None, probe_traj=None, seed=0):
    """Compile + smoke-run a scene.xml, reporting compile time, physics
    speed, memory, per-body settle drift, a sparse contact trajectory, and a
    deterministic final-state hash.

    push: optional {"body": name, "force": (fx,fy,fz), "steps": k} applied
    via xfrc_applied for the first k steps (models a scripted shove).
    probe_traj: optional {"body": name, "waypoints": [(t_frac, xyz), ...]}
    driving a robot end-effector proxy across the rollout via linear
    interpolation between waypoints, by directly overwriting that body's
    freejoint qpos (and zeroing its qvel) every step BEFORE mj_step.

    This is deliberately NOT a MuJoCo `mocap` body: verified empirically
    that MuJoCo excludes collision between a mocap body and any worldbody
    (static) geom -- both are "no-DOF" from the solver's point of view, so a
    mocap probe sitting inside a table box registers zero contacts even
    though it visibly overlaps. A regular freejoint body with its pose
    force-set every step has real DOF and collides normally (exactly like
    the dynamic objects already do), while still following an exact
    scripted trajectory since we overwrite its qpos before every step.
    """
    import mujoco

    t0 = time.time()
    model = mujoco.MjModel.from_xml_path(str(model_xml_path))
    compile_time_s = time.time() - t0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_names = body_names or []
    start = {n: data.xpos[model.body(n).id].copy() for n in body_names}

    push_body_id = None
    if push:
        push_body_id = model.body(push["body"]).id
    probe_qpos_adr = probe_dof_adr = None
    if probe_traj:
        pbid = model.body(probe_traj["body"]).id
        jadr = model.body_jntadr[pbid]
        probe_qpos_adr = model.jnt_qposadr[jadr]
        probe_dof_adr = model.jnt_dofadr[jadr]

    def _probe_pos_at(frac):
        wps = probe_traj["waypoints"]
        for k in range(len(wps) - 1):
            t0f, p0 = wps[k]
            t1f, p1 = wps[k + 1]
            if t0f <= frac <= t1f:
                a = (frac - t0f) / max(t1f - t0f, 1e-9)
                return (1 - a) * np.asarray(p0) + a * np.asarray(p1)
        return np.asarray(wps[-1][1])

    contacts_seen = {}
    t_run0 = time.time()
    for step in range(steps):
        if push_body_id is not None and step < push["steps"]:
            data.xfrc_applied[push_body_id, :3] = push["force"]
        elif push_body_id is not None:
            data.xfrc_applied[push_body_id, :3] = 0.0
        if probe_qpos_adr is not None:
            frac = step / max(steps - 1, 1)
            data.qpos[probe_qpos_adr:probe_qpos_adr + 3] = _probe_pos_at(frac)
            data.qpos[probe_qpos_adr + 3:probe_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[probe_dof_adr:probe_dof_adr + 6] = 0.0
        mujoco.mj_step(model, data)
        for c in data.contact[:data.ncon]:
            g1, g2 = model.geom(c.geom1).name, model.geom(c.geom2).name
            key = tuple(sorted((g1, g2)))
            contacts_seen[key] = contacts_seen.get(key, 0) + 1
    run_time_s = time.time() - t_run0

    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    drift = {n: float(np.linalg.norm(data.xpos[model.body(n).id] - start[n]))
             for n in body_names}
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # linux: KB->MB

    return {
        "compile_time_s": compile_time_s,
        "steps": steps,
        "run_time_s": run_time_s,
        "steps_per_sec": steps / run_time_s if run_time_s > 0 else float("inf"),
        "mem_mb": mem_mb,
        "finite": finite,
        "settle_drift_m": drift,
        "contact_pairs": {f"{a}|{b}": n for (a, b), n in sorted(contacts_seen.items())},
        "final_body_pos": {n: data.xpos[model.body(n).id].tolist() for n in body_names},
        "state_hash": state_hash(data),
    }


# ======================================================= synthetic fixture =
# A fixture `scene.json` describes a floor + table
# + wall + two obstacle boxes (the "room", built into one mesh below exactly
# like a carved real scan mesh would be) plus three dynamic objects, a mocap
# robot-end-effector sweep, and two scripted pushes. Everything here is
# ordinary geometry code operating on that spec -- no hand-authored MJCF.

def _box_lo_hi(center, size):
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(size, dtype=np.float64) / 2
    return c - h, c + h


def build_fixture_room_mesh(spec):
    """room sub-spec -> one concatenated trimesh (floor + table + wall +
    obstacles), standing in for a carved reconstructed background mesh."""
    import trimesh

    room = spec["room"]
    parts = []

    f = room["floor"]
    m = trimesh.creation.box(extents=(*f["size_xy"], f["thickness"]))
    m.apply_translation((*f["center_xy"], f["top_z"] - f["thickness"] / 2))
    parts.append(m)

    t = room["table"]
    m = trimesh.creation.box(extents=(*t["size_xy"], t["thickness"]))
    m.apply_translation((*t["center_xy"], t["top_z"] - t["thickness"] / 2))
    parts.append(m)

    w = room["wall"]
    y0, y1 = w["y_range"]
    m = trimesh.creation.box(extents=(w["thickness"], y1 - y0, w["height"]))
    m.apply_translation((w["center_x"], (y0 + y1) / 2, w["base_z"] + w["height"] / 2))
    parts.append(m)

    for o in room["obstacles"]:
        m = trimesh.creation.box(extents=tuple(o["size"]))
        m.apply_translation(tuple(o["center"]))
        parts.append(m)

    return trimesh.util.concatenate(parts)


def load_scene_spec(scene_dir):
    return json.loads((Path(scene_dir) / "scene.json").read_text())


def assemble_fixture_mjcf(spec, collision_mode, work_dir):
    """spec + mode -> (scene.xml text, info dict with the body/mocap/push
    names benchmark_model() needs). `work_dir` is where room-mode CoACD
    parts get written (mesh files referenced by the XML via relative-to-cwd
    absolute paths, same convention export_mjcf.py uses)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    assets, static_geoms = [], []
    manifest = {"mode": collision_mode}

    # -- room / floor --------------------------------------------------
    floor = spec["room"]["floor"]
    static_geoms.append(
        f'    <geom name="floor" type="plane" '
        f'pos="0 0 {floor["top_z"]:.4f}" size="20 20 1" friction="0.8 0.005 0.0001"/>')

    room_mesh = build_fixture_room_mesh(spec)
    if collision_mode == "room":
        feats = extract_room_features(room_mesh)
        decimated, decim_stats = decimate_mesh(room_mesh)
        # re-run extraction on the decimated mesh so the emitted geometry is
        # the same one whose triangle budget/error we just measured (a real
        # scan mesh is decimated BEFORE classification for tractability;
        # this fixture is small enough that decimated == source, but the
        # call path matches what export_mjcf.py does on a real scene).
        feats = extract_room_features(decimated)
        room_assets, room_geoms, room_manifest, collision_mesh = emit_room_mjcf(
            feats, out_dir=work_dir / "room_collision")
        assets += room_assets
        static_geoms += room_geoms
        coverage = collision_coverage_metrics(room_mesh, collision_mesh)
        manifest["room"] = {**room_manifest, **decim_stats, "coverage": coverage,
                            "n_floor": 1 if feats.floor else 0,
                            "n_supports": len(feats.supports), "n_walls": len(feats.walls)}
        shim_masks = {}
    else:  # shim: legacy private per-object slabs, no room/table/wall geoms
        supports = []
        for o in spec["objects"]:
            lo, hi = _box_lo_hi(o["center"], o["size"])
            supports.append((lo[2], lo[:2], hi[:2]))
        masks, slab_geoms = build_shim_geoms(supports)
        static_geoms += slab_geoms
        shim_masks = {o["name"]: m for o, m in zip(spec["objects"], masks)}
        manifest["shim"] = {"n_shims": len(slab_geoms),
                            "n_channels": len(set(shim_masks.values()))}

    # -- dynamic objects -------------------------------------------------
    bodies = []
    body_names = []
    for o in spec["objects"]:
        cx, cy, cz = o["center"]
        sx, sy, sz = o["size"]
        mask = 1 | shim_masks.get(o["name"], 0)  # bit 0 (shared) | private channel bit
        mass = o["mass"]
        ixx = mass / 12 * (sy ** 2 + sz ** 2)
        iyy = mass / 12 * (sx ** 2 + sz ** 2)
        izz = mass / 12 * (sx ** 2 + sy ** 2)
        bodies.append(f"""    <body name="{o['name']}" pos="{cx:.4f} {cy:.4f} {cz:.4f}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{mass:.4f}" diaginertia="{ixx:.3e} {iyy:.3e} {izz:.3e}"/>
      <geom name="{o['name']}_geom" type="box" size="{sx / 2:.4f} {sy / 2:.4f} {sz / 2:.4f}" contype="{mask}" conaffinity="{mask}"
            friction="{o['friction']:.2f} 0.005 0.0001" rgba="0.3 0.5 0.8 1"/>
    </body>""")
        body_names.append(o["name"])

    # -- robot end-effector probe (sweep proxy) --------------------------
    # A regular freejoint body, NOT a MuJoCo `mocap` body: mocap bodies are
    # excluded from collision against worldbody-static geoms (verified
    # empirically -- see benchmark_model's docstring), which is exactly what
    # the room/table/wall/obstacle geoms are. benchmark_model() drives this
    # body's pose by overwriting its freejoint qpos every step instead.
    r = spec["probe_sweep"]["radius"]
    p0 = spec["probe_sweep"]["waypoints"][0][1]
    probe = (f'    <body name="robot_probe" pos="{p0[0]:.4f} {p0[1]:.4f} {p0[2]:.4f}">\n'
            f'      <freejoint/>\n'
            f'      <inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/>\n'
            f'      <geom name="robot_probe_geom" type="sphere" size="{r:.4f}" contype="1" conaffinity="1" '
            f'rgba="0.9 0.2 0.2 1"/>\n    </body>')

    xml = f"""<mujoco model="collision_fixture_{collision_mode}">
  <compiler angle="radian" autolimits="true" balanceinertia="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <asset>
{chr(10).join(assets) if assets else ''}
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 4" dir="0 0 -1"/>
{chr(10).join(static_geoms)}
{chr(10).join(bodies)}
{probe}
  </worldbody>
</mujoco>
"""
    info = {"body_names": body_names + ["robot_probe"], "manifest": manifest,
            "probe_traj": {"body": "robot_probe", "waypoints": spec["probe_sweep"]["waypoints"]}}
    return xml, info


def run_fixture(scene_dir, collision_mode="room", smoke_steps=1000, out_dir=None, seed=0):
    """Build + smoke-test the synthetic collision fixture in one mode.
    Returns the full report dict (room/shim manifest + benchmark_model()
    output + push/cross-push results), and also writes it to
    `<out_dir>/room_collision_report_<mode>.json` if `out_dir` is given."""
    spec = load_scene_spec(scene_dir)
    work_dir = Path(out_dir) if out_dir else Path(scene_dir) / f"_room_collision_{collision_mode}"
    work_dir.mkdir(parents=True, exist_ok=True)
    xml, info = assemble_fixture_mjcf(spec, collision_mode, work_dir)
    xml_path = work_dir / "scene.xml"
    xml_path.write_text(xml)

    push = spec["push_off_support"]
    cross = spec["cross_object_push"]

    # benchmark_model only supports ONE scripted push; run the smoke rollout
    # twice from a fresh model/data each time (push_off_support, then
    # cross_object_push) so both scripted probes are exercised with a clean
    # deterministic start, and merge into one report.
    report = {"collision_mode": collision_mode, "scene_xml": str(xml_path), **info}
    report["push_off_support"] = benchmark_model(
        xml_path, steps=smoke_steps, body_names=info["body_names"],
        push={"body": push["object"], "force": push["force"], "steps": push["force_steps"]},
        probe_traj=info["probe_traj"], seed=seed)
    report["cross_object_push"] = benchmark_model(
        xml_path, steps=smoke_steps, body_names=info["body_names"],
        push={"body": cross["object"], "force": cross["force"], "steps": cross["force_steps"]},
        probe_traj=info["probe_traj"], seed=seed)
    # top-level convenience aliases for the common case (compile/speed/mem
    # are ~identical between the two runs; report the push_off_support one)
    for k in ("compile_time_s", "steps_per_sec", "mem_mb", "finite", "state_hash"):
        report[k] = report["push_off_support"][k]

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    (work_dir / f"room_collision_report_{collision_mode}.json").write_text(
        json.dumps(report, indent=1, default=_default))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True, help="fixture scene dir (has scene.json)")
    ap.add_argument("--collision-mode", choices=["room", "shim"], default="room")
    ap.add_argument("--smoke-steps", type=int, default=1000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    report = run_fixture(args.scene, collision_mode=args.collision_mode,
                         smoke_steps=args.smoke_steps, out_dir=args.out_dir, seed=args.seed)
    m = report.get("manifest", {})
    print(f"[room_collision] mode={args.collision_mode} "
          f"compile={report['compile_time_s'] * 1000:.1f}ms "
          f"speed={report['steps_per_sec']:.0f} steps/s mem={report['mem_mb']:.0f}MB "
          f"finite={report['finite']}")
    if "room" in m:
        print(f"[room_collision] room: floor={m['room']['n_floor']} "
              f"supports={m['room']['n_supports']} walls={m['room']['n_walls']} "
              f"coacd_parts={m['room']['coacd_parts']} "
              f"hausdorff_p95={m['room']['hausdorff_p95_m'] * 1000:.1f}mm "
              f"coverage={m['room']['coverage']}")
    elif "shim" in m:
        print(f"[room_collision] shim: {m['shim']}")
    print(f"[room_collision] push_off_support contacts: "
          f"{report['push_off_support']['contact_pairs']}")
    print(f"[room_collision] cross_object_push contacts: "
          f"{report['cross_object_push']['contact_pairs']}")


if __name__ == "__main__":
    main()
