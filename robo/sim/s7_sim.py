"""Stage 7: compose the digital twin in PyBullet, settle, run a dynamics demo.

- Background collision mesh = GT scan mesh with the extracted objects carved
  out (their GT instance faces + fitted AABBs), cropped around the action and
  decimated. (The paper reconstructs an inpainted background splat instead;
  carving the scan is the ScanNet++ shortcut for the same idea.)
- Settle with the paper's trick: step the sim while force-zeroing velocities
  every step until poses converge; cache settled poses.
- Dynamics demo from the settled state: drop a duplicated bottle ("object
  cousin" spawn) + push a mug; record 30 fps trajectories for s8 rendering.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from agents.core import common as C

SETTLE_STEPS = 2400
SIM_SECONDS = 5.0
HZ = 240
REC_EVERY = 8  # -> 30 fps
BG_TRIS = 300_000
CARVE_SCHEMA_VERSION = 1


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slot(meta):
    index = meta.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("background-carve object index must be a non-negative integer")
    return f"obj_{index:02d}"


def _carve_contract(objects, carve_hulls, *, margin_m, lower_support_margin_m,
                    mesh_path, paired_policy_ids=None):
    discovered_slots = [_slot(meta) for meta in objects]
    if len(discovered_slots) != len(set(discovered_slots)):
        raise ValueError("background-carve discovered roster contains duplicate slots")
    hull_rows = []
    for row in carve_hulls:
        vertices = np.asarray(row["vertices"], dtype="<f8")
        hull_rows.append({
            "name": str(row["name"]),
            "policy_id": str(row["policy_id"]),
            "sha256": hashlib.sha256(vertices.tobytes(order="C")).hexdigest(),
            "slot": str(row["slot"]),
            "support_clip_z_m": float(row["support_clip_z_m"]),
            "vertex_count": len(vertices),
        })
    hull_rows.sort(key=lambda row: (row["policy_id"], row["slot"], row["name"],
                                    row["sha256"]))
    policy_ids = sorted({row["policy_id"] for row in hull_rows})
    if paired_policy_ids is not None:
        if sorted(paired_policy_ids) != ["A0", "A4"]:
            raise ValueError("paired background must declare exactly A0 and A4")
        policy_ids = list(sorted(paired_policy_ids))
    carved_slots = sorted({row["slot"] for row in hull_rows})
    roster_sha256 = hashlib.sha256(
        _canonical_json(discovered_slots).encode("utf-8")
    ).hexdigest()
    mesh_stat = Path(mesh_path).stat()
    specification = {
        "schema_version": CARVE_SCHEMA_VERSION,
        "mode": "paired_policy_union" if len(policy_ids) > 1 else "single_policy",
        "geometry_source": "transformed_convex_collision_hulls",
        "margin_m": float(margin_m),
        "lower_support_margin_m": float(lower_support_margin_m),
        "support_clip_source": ("automatic_instance_aabb_bottom_plus_5mm"
                                if objects and all(meta.get("instance_namespace") == "automatic" for meta in objects)
                                else "discovered_scan_aabb_bottom_plus_5mm"),
        "discovered_slots": discovered_slots,
        "carved_slots": carved_slots,
        "policy_ids": policy_ids,
        "hulls": hull_rows,
        "roster_sha256": roster_sha256,
        "objects_sha256": hashlib.sha256(
            _canonical_json(objects).encode("utf-8")
        ).hexdigest(),
        "source_mesh": {
            "size_bytes": mesh_stat.st_size,
            "sha256": _sha256_file(mesh_path),
        },
    }
    specification_sha256 = hashlib.sha256(
        _canonical_json(specification).encode("utf-8")
    ).hexdigest()
    public = {
        key: specification[key]
        for key in (
            "schema_version", "mode", "geometry_source", "margin_m",
            "lower_support_margin_m", "discovered_slots", "carved_slots",
            "policy_ids", "roster_sha256", "support_clip_source",
        )
    }
    public.update({
        "hull_count": len(hull_rows),
        "source_mesh_sha256": specification["source_mesh"]["sha256"],
        "specification_sha256": specification_sha256,
    })
    return public


def build_background(
    objects,
    aligneds,
    gts,
    *,
    carve_hulls=(),
    carve_margin_m=0.02,
    lower_support_margin_m=0.0,
    return_report=False,
    paired_policy_ids=None,
):
    from agents.assets.s5_align import (
        open3d_registration_backend,
        scene_mesh_arrays,
    )

    if any(meta.get("automatic_instance_id") is not None for meta in objects):
        if C.env("AUTO") != "1" or any(meta.get("instance_namespace") != "automatic" or "gt_object_id" in meta for meta in objects):
            raise ValueError("automatic background requires explicit automatic namespace and AUTO=1")

    # The public Open3D package eagerly imports its optional Dash/Flask UI
    # stack.  Slurm execution deliberately disables user-site leakage, so use
    # the repository-validated CPU pybind backend directly for geometry/I/O.
    o3d = open3d_registration_backend()

    if len(objects) != len(aligneds):
        raise ValueError("background object/alignment rosters differ")
    from robo.sim import room_collision as rc

    carve_hulls = tuple(carve_hulls)
    exclusions = tuple(
        rc.make_convex_exclusion(
            row["vertices"], name=str(row["name"]), slot=str(row["slot"]),
            policy_id=str(row["policy_id"]), margin_m=carve_margin_m,
            lower_support_margin_m=lower_support_margin_m,
            support_clip_z_m=float(row["support_clip_z_m"]),
        )
        for row in carve_hulls
    )
    mesh_path = getattr(C, "PIPELINE_MESH_PLY", C.MESH_PLY)
    contract = _carve_contract(
        objects, carve_hulls, margin_m=carve_margin_m,
        lower_support_margin_m=lower_support_margin_m, mesh_path=mesh_path,
        paired_policy_ids=paired_policy_ids,
    )

    out = C.OUT / "sim"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "background.obj"
    report_path = out / "background_carve.json"
    if path.is_file() and report_path.is_file():
        try:
            cached = json.loads(report_path.read_text())
        except (OSError, ValueError):
            cached = None
        if isinstance(cached, dict) \
                and cached.get("specification_sha256") == contract["specification_sha256"] \
                and cached.get("background_sha256") == _sha256_file(path):
            report = cached
        else:
            report = None
    else:
        report = None
    if report is not None:
        print(f"[s7] background: cached {path}")
        return (path, report) if return_report else path

    verts, faces = scene_mesh_arrays(mesh_sha256=contract["source_mesh_sha256"])
    remove = np.zeros(len(verts), bool)
    carved_slots = {row["slot"] for row in carve_hulls}
    for meta in objects:
        automatic_id = meta.get("automatic_instance_id")
        # Failed automatic jobs remain part of the scene and denominator.
        # Carve only the union actually replaced by either construction arm.
        if automatic_id is not None and _slot(meta) not in carved_slots:
            continue
        if automatic_id is not None:
            if (C.env("AUTO") != "1" or meta.get("instance_namespace") != "automatic"
                    or "gt_object_id" in meta or automatic_id != meta.get("index")):
                raise ValueError("automatic object cannot enter GT instance carving")
            if automatic_id not in gts:
                raise ValueError("automatic instance is absent from construction segmentation")
            remove[gts[automatic_id]["vert_idx"]] = True
        elif meta.get("gt_object_id") is not None:
            remove[gts[meta["gt_object_id"]]["vert_idx"]] = True
        lo, hi = np.array(meta["aabb"])
        box = ((verts > lo - 0.02) & (verts < hi + 0.02)).all(axis=1)
        box &= verts[:, 2] > lo[2] + 0.005  # protect the desk surface below
        remove |= box

    # crop to the action region (objects + 3 m margin in xy, floor kept).
    # Zero-discovery episodes (SAM3 found nothing graspable at all) have no
    # aabb to crop around -- keep the full scene mesh rather than crashing
    # on an empty-array reduction; the background stays visual-only, which
    # is already the AUTO-mode default, so this is a correct degenerate case,
    # not a silent corruption.
    if objects:
        los = np.array([m["aabb"][0] for m in objects]).min(axis=0)
        his = np.array([m["aabb"][1] for m in objects]).max(axis=0)
        keep_region = ((verts[:, 0] > los[0] - 3) & (verts[:, 0] < his[0] + 3) &
                       (verts[:, 1] > los[1] - 3) & (verts[:, 1] < his[1] + 3) &
                       (verts[:, 2] < his[2] + 2))
    else:
        print("[s7] background: 0 objects discovered, keeping full scene mesh uncropped")
        keep_region = np.ones(len(verts), bool)
    # The scan AABB above removes the observed instance.  The additional
    # transformed-hull union removes space occupied by either construction
    # arm.  This is what makes A0/A4 share one physical background rather
    # than silently changing the room together with the policy.
    hull_vertex_removals = 0
    if exclusions:
        from scipy.spatial import cKDTree

        vertex_tree = cKDTree(verts)
        for exclusion in exclusions:
            center = (exclusion.lo + exclusion.hi) / 2.0
            radius = float(np.linalg.norm((exclusion.hi - exclusion.lo) / 2.0))
            candidates = np.asarray(vertex_tree.query_ball_point(center, radius),
                                    dtype=np.int64)
            if len(candidates):
                inside = rc.points_inside_exclusion(verts[candidates], exclusion)
                fresh = candidates[inside & ~remove[candidates]]
                hull_vertex_removals += len(fresh)
                remove[candidates[inside]] = True

    keep_v = keep_region & ~remove
    fmask = keep_v[faces].all(axis=1)

    # A large scan triangle can cross a hull even when all three vertices are
    # outside it.  Centroid rejection catches that failure mode before convex
    # decomposition; the post-decomposition exclusion in room_collision is
    # the final fail-safe against any cavity bridging that remains.
    hull_face_removals = 0
    if exclusions and np.any(fmask):
        candidate_faces = np.flatnonzero(fmask)
        centers = verts[faces[candidate_faces]].mean(axis=1)
        from scipy.spatial import cKDTree

        center_tree = cKDTree(centers)
        rejected = np.zeros(len(candidate_faces), dtype=bool)
        for exclusion in exclusions:
            center = (exclusion.lo + exclusion.hi) / 2.0
            radius = float(np.linalg.norm((exclusion.hi - exclusion.lo) / 2.0))
            candidates = np.asarray(center_tree.query_ball_point(center, radius),
                                    dtype=np.int64)
            if len(candidates):
                rejected[candidates] |= rc.points_inside_exclusion(
                    centers[candidates], exclusion
                )
        fmask[candidate_faces[rejected]] = False
        hull_face_removals = int(rejected.sum())

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces[fmask]))
    m.remove_unreferenced_vertices()
    if len(m.triangles) > BG_TRIS:
        # vertex clustering: seconds instead of the ~1h quadric decimation
        m = m.simplify_vertex_clustering(voxel_size=0.025)
        m.remove_degenerate_triangles()
    o3d.io.write_triangle_mesh(str(path), m)
    report = {
        **contract,
        "background_sha256": _sha256_file(path),
        "source_vertex_count": int(len(verts)),
        "source_face_count": int(len(faces)),
        "kept_face_count": int(len(m.triangles)),
        "hull_vertex_removals": int(hull_vertex_removals),
        "hull_centroid_face_removals": int(hull_face_removals),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[s7] carve: {contract['mode']} hulls={contract['hull_count']} "
          f"slots={len(contract['carved_slots'])} margin={carve_margin_m:.3f}m")
    print(f"[s7] background: {len(m.triangles)} tris -> {path}")
    return (path, report) if return_report else path


def wxyz_to_xyzw(q):
    return [q[1], q[2], q[3], q[0]]


def link_pose(p, b):
    """pybullet reports the inertial/COM frame; convert to the URDF link
    (canonical) frame that loadURDF and the s8 gaussian compositing use."""
    pos, q = p.getBasePositionAndOrientation(b)
    ip, iq = p.getDynamicsInfo(b, -1)[3:5]
    lp, lq = p.multiplyTransforms(pos, q, *p.invertTransform(ip, iq))
    return list(lp), list(lq)


def com_pose(p, b, lp, lq):
    """Inverse of link_pose: link-frame pose -> COM pose for resetBase*."""
    ip, iq = p.getDynamicsInfo(b, -1)[3:5]
    return p.multiplyTransforms(lp, lq, ip, iq)


def main():
    import pybullet as p

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    aligneds = [json.loads(
        (C.OUT / "objects" / f"obj_{m['index']:02d}" / "aligned.json").read_text())
        for m in objects]
    kept = [(m, a) for m, a in zip(objects, aligneds) if not a.get("rejected")]
    for m, a in zip(objects, aligneds):
        if a.get("rejected"):
            print(f"[s7] obj_{m['index']:02d}: skip ({a['rejected']})")
    objects = [m for m, _ in kept]
    aligneds = [a for _, a in kept]
    if not objects:
        print("[s7] no objects survived; nothing to simulate")
        return
    physes = [json.loads(
        (C.OUT / "objects" / f"obj_{m['index']:02d}" / "physics.json").read_text())
        for m in objects]
    gts = {g["object_id"]: g for g in C.load_instances()}
    bg_path = build_background(objects, aligneds, gts)

    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / HZ)
    col = p.createCollisionShape(p.GEOM_MESH, fileName=str(bg_path),
                                 flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=str(bg_path),
                              rgbaColor=[0.75, 0.73, 0.70, 1.0])
    bg = p.createMultiBody(0, col, vis)
    p.changeDynamics(bg, -1, lateralFriction=0.6, restitution=0.05)

    bodies, names = [], []
    rng = np.random.RandomState(0)
    for meta, al, ph in zip(objects, aligneds, physes):
        odir = C.OUT / "objects" / f"obj_{meta['index']:02d}"
        s, R, t = C.decompose_similarity(np.array(al["T"]))
        q = wxyz_to_xyzw(C.rot_to_quat_wxyz(R))
        bid = p.loadURDF(str(odir / "object.urdf"), basePosition=t,
                         baseOrientation=q, flags=p.URDF_USE_INERTIA_FROM_FILE)
        p.changeDynamics(bid, -1, lateralFriction=ph["friction"],
                         restitution=ph["restitution"],
                         linearDamping=0.04, angularDamping=0.04)
        p.changeVisualShape(bid, -1, rgbaColor=list(rng.uniform(0.2, 0.9, 3)) + [1])
        bodies.append(bid)
        names.append(f"obj_{meta['index']:02d}")

    # --- settle: velocities force-zeroed every step (paper App. E.4) -------
    prev = {b: p.getBasePositionAndOrientation(b) for b in bodies}
    settled_at = SETTLE_STEPS
    for i in range(SETTLE_STEPS):
        p.stepSimulation()
        for b in bodies:
            p.resetBaseVelocity(b, [0, 0, 0], [0, 0, 0])
        if (i + 1) % 100 == 0:
            cur = {b: p.getBasePositionAndOrientation(b) for b in bodies}
            dmax = max(np.linalg.norm(np.array(cur[b][0]) - np.array(prev[b][0]))
                       for b in bodies)
            prev = cur
            if (i + 1) % 400 == 0:
                print(f"[s7] settle step {i + 1}/{SETTLE_STEPS} dmax={dmax:.5f}",
                      flush=True)
            if dmax < 2e-4:
                settled_at = i + 1
                break
    settled = {}
    for b, name, meta, al in zip(bodies, names, objects, aligneds):
        pos, q = link_pose(p, b)
        s, R0, t0 = C.decompose_similarity(np.array(al["T"]))
        settled[name] = {"pos": pos, "quat_xyzw": q, "scale": s,
                         "drift_m": float(np.linalg.norm(np.array(pos) - t0))}
    C.save_json(C.OUT / "sim" / "settled.json",
                {"settled_at_step": settled_at, "bodies": settled})
    drifts = [v["drift_m"] for v in settled.values()]
    print(f"[s7] settled at step {settled_at}; drift median "
          f"{np.median(drifts) * 1000:.1f}mm max {max(drifts) * 1000:.1f}mm")

    # --- dynamics demo ------------------------------------------------------
    def find(label_sub):
        for k, meta in enumerate(objects):
            if label_sub in meta["label"]:
                return k
        return None

    events = []
    kb = find("bottle")
    if kb is not None:
        src = C.OUT / "objects" / f"obj_{objects[kb]['index']:02d}" / "object.urdf"
        pos0 = np.array(settled[names[kb]]["pos"]) + [0.05, 0.04, 0.25]
        dup = p.loadURDF(str(src), basePosition=pos0,
                         baseOrientation=settled[names[kb]]["quat_xyzw"],
                         flags=p.URDF_USE_INERTIA_FROM_FILE)
        p.changeDynamics(dup, -1, lateralFriction=0.5, restitution=0.1)
        bodies.append(dup)
        names.append(f"{names[kb]}_cousin")
        settled[names[-1]] = {"scale": settled[names[kb]]["scale"]}
        events.append(f"dropped a duplicate of {objects[kb]['label']} from 25cm")
    km = find("mug")
    push_body = bodies[km] if km is not None else None
    if km is not None:
        events.append(f"pushed the {objects[km]['label']} laterally for 0.2s")

    n_steps = int(SIM_SECONDS * HZ)
    frames = []
    for i in range(n_steps):
        if push_body is not None and i < 0.2 * HZ:
            com, _ = p.getBasePositionAndOrientation(push_body)
            p.applyExternalForce(push_body, -1, [1.2, 0.6, 0], list(com),
                                 p.WORLD_FRAME)
        p.stepSimulation()
        if i % REC_EVERY == 0:
            fr = {}
            for b, name in zip(bodies, names):
                pos, q = link_pose(p, b)
                fr[name] = [pos, q]
            frames.append(fr)
    C.save_json(C.OUT / "sim" / "dynamics.json",
                {"fps": HZ / REC_EVERY, "events": events,
                 "spawn_map": {n: n.replace("_cousin", "") for n in names},
                 "frames": frames})
    print(f"[s7] dynamics: {len(frames)} frames, events: {events}")

    # --- debug video (tiny renderer) ---------------------------------------
    rep = json.loads((C.OUT / "frame" / "rep_frame.json").read_text())
    c2w = np.linalg.inv(np.array(rep["w2c"]))
    eye = c2w[:3, 3]
    target = eye + c2w[:3, :3] @ [0, 0, 1]
    up = -(c2w[:3, :3] @ [0, 1, 0])
    K, W, H, _ = C.load_intrinsics()
    view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(
        np.rad2deg(2 * np.arctan(H / (2 * K[1, 1]))), W / H, 0.05, 20)

    import imageio.v2 as imageio
    dbg = []
    for k, fr in enumerate(frames):
        for b, name in zip(bodies, names):
            if name in fr:
                p.resetBasePositionAndOrientation(
                    b, *com_pose(p, b, fr[name][0], fr[name][1]))
        if k % 2 == 0:  # 15 fps debug video is fine
            _, _, rgb, _, _ = p.getCameraImage(
                640, 426, view, proj, renderer=p.ER_TINY_RENDERER)
            dbg.append(np.asarray(rgb, dtype=np.uint8).reshape(426, 640, 4)[..., :3])
    imageio.mimwrite(C.OUT / "sim" / "debug.mp4", dbg, fps=15,
                     macro_block_size=None)
    print(f"[s7] debug video: {len(dbg)} frames")
    p.disconnect()


if __name__ == "__main__":
    main()
