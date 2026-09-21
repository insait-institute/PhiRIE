"""robo.eval.reference_scene: the hand-built "official/reference" scene
builder for the mujoco_paired protocol (Task 09; the design decision this
module embodies is documented in full in docs/MUJOCO_PAIRED_PROTOCOL.md --
read that first).

docs/ICRA_RESEARCH_CONTRACT.md Decision 1 defers PolaRiS/Isaac Lab, so the
`mujoco_paired` protocol has no existing hand-built reference environment
to pair the SimAny reconstruction against; contract G2 requires one
("one hand-built reference task variant runs in MuJoCo with an identical
robot/camera/control/rubric contract to the SimAny reconstruction of the
same task family"). This module IS that reference-variant builder.

Design summary (docs/MUJOCO_PAIRED_PROTOCOL.md has the full reasoning):
given a SimAny factory build's `objects/*/aligned.json` +
`objects/*/physics.json` -- the SAME files robo/sim/export_mjcf.py reads to
build the scanned-mesh scene.xml -- and its `sim_export/pi05_tasks.json`
task suite, build a SECOND scene.xml for the identical scene that:

  - keeps every frozen field byte-identical: robot base_pos/base_yaw, the
    table_box, the ext_cam pose, exclude_objects, task defs, rubric,
    control contract. None of that lives in this module -- it is passed
    straight through, unchanged, by robo.eval.paired_runner to the SAME
    robo.rigs.pi05_rig.build_scene_model / robo.envs.pi05_env.DroidSimEnv
    code path already used for the real reconstruction;
  - keeps every object's initial WORLD POSE exactly as measured by the
    reconstruction pipeline (`aligned.json`'s `T`, decomposed into
    (pos, quat) -- the exact position/orientation export_mjcf.py itself
    places that object's real mesh at), so "reset states are byte-
    identical across conditions" (Task 09 step 1) holds at the geometry
    level too, not just at the RNG-seed level;
  - replaces the SimAny reconstruction's actual asset (a CoACD-decomposed
    scanned mesh) AND its full room-mode background carving (floor +
    supports + walls extracted from the scan, robo/sim/room_collision.py)
    with the plainest primitive that preserves the same physical
    footprint: one box per object, sized to that object's own
    `world_dims` (mass/friction unchanged, from the SAME physics.json),
    plus a flat floor plane -- no scanned room geometry at all. The table
    itself is already a primitive box in BOTH conditions (added by
    pi05_rig.build_scene_model from the shared `table_box` field, which
    this module never touches), so nothing extra is needed for it here.

Given the scope narrowing in docs/ICRA_RESEARCH_CONTRACT.md Decision 1 (no
PolaRiS/Isaac Lab oracle available for this protocol), this is the
simplest defensible "official" stand-in: every DECLARED scene-construction
fact (asset identity, collision representation) differs between
conditions; every frozen control/camera/robot/rubric fact and every
object's initial pose does not. A manifest diff between a "reference" and
"simany" run of the same task_id/seed is therefore expected to show ONLY
scene_manifest_hash (and downstream outcome fields) differing -- see
robo.manifest.io.diff_manifests(..., frozen_only=True), which must return
`{}` for such a pair.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robo.rigs.pi05_rig import rot_to_quat_wxyz

#: Safety margin (m) below the lowest reference object's conservative
#: bounding-sphere bottom -- never a frozen field (the table_box the arm
#: actually works against is identical in both conditions), just insurance
#: against anything ever falling off the table.
FLOOR_MARGIN_M = 0.05


def _decompose_similarity(T):
    """4x4 with isotropic scale folded into R -> (s, R, t).

    Deliberately duplicated (4 lines) rather than imported from
    agents.core.common.decompose_similarity: that module reads
    SIMANY_SCENE/SIMANY_OUT environment variables as a side effect of
    OTHER functions its callers rely on (see robo/tasks/pi05_tasks.py's
    `_pick_scan_camera`, which sets those env vars before importing it),
    and this module -- pure linear algebra over an already-loaded
    `aligned.json` -- has no business inheriting that global-state
    coupling for four lines of math.
    """
    T = np.asarray(T, dtype=np.float64)
    A = T[:3, :3]
    s = float(np.cbrt(max(np.linalg.det(A), 1e-12)))
    return s, A / s, T[:3, 3].copy()


def _load_reference_objects(factory_dir):
    """Load the SAME object set robo/sim/export_mjcf.py's real-mesh
    scene.xml would contain: identical inclusion test (`not rejected` and
    `object.urdf` exists) applied to the SAME `objects.json` /
    `aligned.json` / `physics.json` files, so the reference and
    reconstructed scene.xml declare an IDENTICAL body-name superset
    *before* either one applies a task suite's `exclude_objects` -- that
    identity is what keeps paired reset poses and exclusions aligned
    across conditions without robo.eval.paired_runner needing to know
    anything about how either scene.xml was actually built.

    Returns a list of dicts: name, label, tier, pos (3,), quat_wxyz (4,),
    dims (3,) (full box extents, meters), mass_kg, friction.
    """
    factory_dir = Path(factory_dir)
    objects = json.loads((factory_dir / "objects" / "objects.json").read_text())
    rows = []
    for m in objects:
        name = f"obj_{m['index']:02d}"
        odir = factory_dir / "objects" / name
        af = odir / "aligned.json"
        if not af.exists():
            continue
        al = json.loads(af.read_text())
        if al.get("rejected") or not (odir / "object.urdf").exists():
            continue
        pf = odir / "physics.json"
        ph = json.loads(pf.read_text()) if pf.exists() else {}
        s, R, t = _decompose_similarity(al["T"])
        dims = np.maximum(np.asarray(al["world_dims"], dtype=float), 1e-3)
        rows.append({
            "name": name,
            "label": m.get("label", "object"),
            "tier": al.get("tier", "A"),
            "pos": t,
            "quat_wxyz": rot_to_quat_wxyz(R),
            "dims": dims,
            "mass_kg": float(ph.get("mass_kg", 0.1)),
            "friction": float(ph.get("friction", 0.5)),
        })
    return rows


def _body_xml(row):
    name = row["name"]
    t, q, dims = row["pos"], row["quat_wxyz"], row["dims"]
    hx, hy, hz = (dims / 2.0).tolist()
    mass = row["mass_kg"]
    dx, dy, dz = dims.tolist()
    # Uniform-density box inertia -- the same approximation
    # robo/sim/export_mjcf.py uses for its (much less box-shaped) real
    # mesh bodies, exact here since the reference geometry IS a box.
    diag = (mass / 12 * (dy ** 2 + dz ** 2),
            mass / 12 * (dx ** 2 + dz ** 2),
            mass / 12 * (dx ** 2 + dy ** 2))
    return f"""    <body name="{name}" pos="{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}"
        quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{mass:.4f}"
                diaginertia="{diag[0]:.12e} {diag[1]:.12e} {diag[2]:.12e}"/>
      <geom type="box" size="{hx:.6f} {hy:.6f} {hz:.6f}" contype="1" conaffinity="1"
            friction="{row['friction']:.3f} 0.005 0.0001"
            rgba="0.72 0.72 0.78 1"/>
    </body>"""


def build_reference_scene_xml(factory_dir, suite, out_path):
    """Write a primitive-only reference scene.xml for `suite` (a loaded
    sim_export/pi05_tasks.json dict) alongside the real reconstruction's
    own scene.xml. Returns (out_path, [kept object names]).

    Raises `ValueError` (never a silent empty scene) if no structurally-
    valid object survives the inclusion test -- robo.eval.paired_runner
    wraps this in `robo.eval.episode_log.BuildFailureError` so an
    uninstantiable reference scene is a logged coverage failure, not a
    skip (Task 09 step 2).

    `suite["exclude_objects"]` is intentionally NOT applied here: both
    this module and export_mjcf.py write every structurally-valid object
    into scene.xml, and exclusion happens later, identically, inside
    robo.rigs.pi05_rig.build_scene_model (the same `exclude_objects=`
    kwarg robo.envs.pi05_env.DroidSimEnv forwards for both conditions) --
    duplicating that filter here would just be a second place it could
    drift out of sync.
    """
    rows = _load_reference_objects(factory_dir)
    if not rows:
        raise ValueError(
            f"no structurally-valid objects found under {factory_dir}/objects "
            f"-- cannot build a reference scene for {suite.get('scene')!r}")
    bodies = [_body_xml(r) for r in rows]
    # Conservative safety floor: each object's bounding-sphere-safe lower
    # z (not its true box-corner bottom -- cheap and safely low is all
    # this needs, since the table_box is what the arm actually rests
    # objects on/against in both conditions).
    lowest = min(r["pos"][2] - 0.5 * float(np.linalg.norm(r["dims"])) for r in rows)
    floor_z = lowest - FLOOR_MARGIN_M

    xml = f"""<mujoco model="reference_{suite.get('scene', 'scene')}">
  <compiler angle="radian" autolimits="true" balanceinertia="true"/>
  <option timestep="0.0016666667" integrator="implicitfast"/>
  <visual><global offwidth="1920" offheight="1080"/></visual>
  <worldbody>
    <light directional="true" pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" pos="0 0 {floor_z:.4f}" size="20 20 1"
          friction="0.8 0.005 0.0001"/>
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml)
    return out_path, [r["name"] for r in rows]


def instance_inventory(factory_dir):
    """[{object_id, label, tier, asset_hash}], matching
    robo.manifest.schema.InstanceInventoryItem's shape, for building a
    SceneBuildManifest of the reference condition. `asset_hash` is always
    None here: a primitive box has no source asset to hash -- the
    schema's documented legal "not applicable" value, not a missing
    measurement.
    """
    return [{"object_id": r["name"], "label": r["label"], "tier": r["tier"],
             "asset_hash": None} for r in _load_reference_objects(factory_dir)]


__all__ = ["build_reference_scene_xml", "instance_inventory"]
