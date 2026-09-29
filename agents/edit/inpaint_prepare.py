"""Inpainting stage 1 (venv, CPU): per placed object, prepare everything the
removal+fill pipeline needs.

For each non-rejected object of a factory scene:
  - removal set: scene-splat gaussians within RADIUS of the GT instance
    vertices (these are the "original object" gaussians to delete)
  - support plane: least-squares plane through the scan-mesh ring around the
    object footprint (the desk the object stood on) + footprint hull in
    plane (u,v) coords
  - related views: top-K frames by the same visibility scoring used for the
    best-view crop, plus the PROJECTED object mask per view (occlusion-aware
    raycast) for the SAM3 refinement stage
Writes OUT/inpaint/{removal_mask.npz, obj_XX/{plane.json, views.json,
proj_masks.npz}}.
"""
import json
import shutil

import numpy as np
from scipy.spatial import cKDTree

from agents.core import common as C

RADIUS = 0.03       # m, gaussian-removal distance from instance surface
TOP_K_VIEWS = 3
RING = 0.10         # m, xy ring around footprint for plane fitting


def surface_removal_indices(splat_tree, points, radius=RADIUS):
    """Canonical radius-based observed/asset surface removal primitive."""
    hit=splat_tree.query_ball_point(points,radius)
    parts=[np.asarray(h,dtype=np.int64) for h in hit if h]
    return np.unique(np.concatenate(parts or [np.array([],dtype=np.int64)]))


def fit_plane(pts):
    """SVD plane with 2 rounds of MAD trimming (nearby-object contamination
    sits 0-30mm above the desk and would bias/tilt a plain LSQ fit).
    Also returns trim_ok: False means outliers were detected on the first
    pass but trimming would leave <50 pts, so the result is the untrimmed
    one-shot fit (possibly biased by contamination)."""
    keep = pts
    trim_ok = False
    for _ in range(3):
        o = keep.mean(axis=0)
        _, _, vt = np.linalg.svd(keep - o, full_matrices=False)
        n = vt[2]
        res = (keep - o) @ n
        mad = np.median(np.abs(res - np.median(res))) + 1e-6
        sel = np.abs(res - np.median(res)) < 2.5 * mad
        if sel.all():       # fit is MAD-consistent: converged
            trim_ok = True
            break
        if sel.sum() < 50:  # can't trim without losing support: keep fit as-is
            break
        keep = keep[sel]    # trim BEFORE any break so the refit sees it
        trim_ok = True
    if n[2] < 0:
        n = -n
    u = np.cross(n, [1.0, 0, 0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return o, n, u, v, trim_ok


def _sample_asset(object_dir, alignment):
    import trimesh
    from agents.assets.s5_align import apply_T
    mesh = trimesh.load(object_dir/'mesh_sim.ply', process=False)
    points, _ = trimesh.sample.sample_surface(mesh, 5000)
    transform = np.asarray(alignment['T'], dtype=float)
    return apply_T(transform, np.asarray(points))  # canonical raw mesh; exactly once


def _prepare():
    import open3d as o3d
    from plyfile import PlyData
    from agents.assets.s5_align import scene_mesh_arrays

    factory = C.OUT
    K, W, H, _ = C.load_intrinsics()
    w2c_all = sorted(C.load_colmap_w2c().items())
    verts, faces = scene_mesh_arrays()
    scene_full = C.make_raycast_scene()
    instances = C.load_instances()
    objects = json.loads((factory/'objects/objects.json').read_text())
    gts = {g['object_id']: g for g in instances}

    splat_xyz = np.stack([np.asarray(PlyData.read(str(C.SPLAT_PLY))["vertex"][a],
                                     dtype=np.float64) for a in "xyz"], axis=1)
    splat_tree = cKDTree(splat_xyz)

    inp = factory / "inpaint"
    inp.mkdir(exist_ok=True)
    union = np.zeros(len(splat_xyz), bool)
    rng = np.random.RandomState(0)

    seen_gt = set()
    asset_rm_failed = []
    for m in objects:
        name = f"obj_{m['index']:02d}"
        # stale per-object outputs poison downstream existence-guards on rerun
        shutil.rmtree(inp / name, ignore_errors=True)
        al = json.loads((factory / "objects" / name / "aligned.json").read_text())
        if al.get("rejected"):
            continue
        if m.get('gt_object_id') is None or m['gt_object_id'] in seen_gt:
            print(f"[ip] {name}: no/duplicate GT id, skip")
            continue
        seen_gt.add(m['gt_object_id'])
        odir = inp / name
        odir.mkdir(exist_ok=True)
        gt = gts[m['gt_object_id']]
        gv = verts[gt["vert_idx"]]

        # --- removal set ---------------------------------------------------
        # (a) near the GT instance surface; (b) near the ALIGNED generated
        # asset surface - scans miss transparent parts (bottle bodies), so
        # their splat gaussians sit far from any GT vertex, but the generated
        # asset covers the full extent
        idx_parts = [surface_removal_indices(splat_tree,gv,RADIUS)]
        try:
            apts = _sample_asset(factory/'objects'/name, al)
            idx_parts.append(surface_removal_indices(splat_tree,apts,RADIUS * 0.8))
        except Exception as e:
            print(f"[ip] {name}: asset-volume removal skipped ({e})")
            asset_rm_failed.append(name)
        idx = np.unique(np.concatenate(
            idx_parts or [np.array([], dtype=np.int64)]))
        mask = np.zeros(len(splat_xyz), bool)
        mask[idx] = True
        union |= mask
        np.save(odir / "removal_idx.npy", idx)

        # --- support plane + footprint ------------------------------------
        zmin = gv[:, 2].min()
        cen = gv.mean(axis=0)
        d_xy = np.linalg.norm(verts[:, :2] - cen[:2], axis=1)
        r_obj = np.linalg.norm(gv[:, :2] - cen[:2], axis=1).max()
        ring = ((d_xy > r_obj) & (d_xy < r_obj + RING) &
                (np.abs(verts[:, 2] - zmin) < 0.03))
        inset = np.zeros(len(verts), bool)
        inset[gt["vert_idx"]] = True
        ring &= ~inset
        if ring.sum() < 100:
            print(f"[ip] {name}: no support ring ({ring.sum()} pts) - "
                  "floor-standing or shelved object; skip plane fill")
            plane = None
        else:
            o, n, u, v, trim_ok = fit_plane(verts[ring])
            if not trim_ok:
                print(f"[ip] {name}: WARNING plane MAD-trim fell back to "
                      f"one-shot fit ({int(ring.sum())} ring pts, "
                      "contamination kept)")
            fp = gv - o
            uv = np.stack([fp @ u, fp @ v], axis=1)  # hull computed in fill
            plane = {"origin": o, "normal": n, "u": u, "v": v,
                     "trim_ok": trim_ok,
                     "footprint_uv": uv[rng.choice(len(uv),
                                                   min(len(uv), 800),
                                                   replace=False)]}
            C.save_json(odir / "plane.json", plane)

        # --- related views + projected masks -------------------------------
        sample = gv[rng.choice(len(gv), min(120, len(gv)), replace=False)]
        cand = []
        for fname, w2c in w2c_all:
            pc = sample @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            if (z < 0.25).mean() > 0.05:
                continue
            upx = pc[:, 0] / z * K[0, 0] + K[0, 2]
            vpx = pc[:, 1] / z * K[1, 1] + K[1, 2]
            inb = (z > 0.25) & (upx >= 0) & (upx < W) & (vpx >= 0) & (vpx < H)
            if inb.mean() < 0.7:
                continue
            cam = np.linalg.inv(w2c)[:3, 3]
            dirs = sample[inb] - cam
            dist = np.linalg.norm(dirs, axis=1)
            rays = o3d.core.Tensor(np.concatenate(
                [np.broadcast_to(cam, dirs.shape), dirs / dist[:, None]],
                axis=1).astype(np.float32))
            t_hit = scene_full.cast_rays(rays)["t_hit"].numpy()
            vis = float((np.abs(t_hit - dist) < 0.02).mean()) * inb.mean()
            if vis < 0.5:
                continue
            cand.append((vis * np.sqrt(np.ptp(upx[inb]) * np.ptp(vpx[inb])),
                         fname, w2c))
        cand.sort(key=lambda c: -c[0])
        views = cand[:TOP_K_VIEWS]

        # occlusion-aware projected mask per view (instance submesh raycast)
        fmask = inset[faces].all(axis=1)
        sub = o3d.t.geometry.RaycastingScene()
        sub.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts),
                o3d.utility.Vector3iVector(faces[fmask]))))
        pm, vnames = [], []
        for _, fname, w2c in views:
            c2w = np.linalg.inv(w2c)
            uu, vv = np.meshgrid(np.arange(W, dtype=np.float64) + 0.5,
                                 np.arange(H, dtype=np.float64) + 0.5)
            d_cam = np.stack([(uu - K[0, 2]) / K[0, 0],
                              (vv - K[1, 2]) / K[1, 1],
                              np.ones_like(uu)], axis=-1)
            d_world = (d_cam @ c2w[:3, :3].T).reshape(-1, 3).astype(np.float32)
            o_world = np.broadcast_to(c2w[:3, 3], d_world.shape).astype(np.float32)
            rays = o3d.core.Tensor(np.concatenate([o_world, d_world], axis=1))
            t_i = sub.cast_rays(rays)["t_hit"].numpy()
            t_f = scene_full.cast_rays(rays)["t_hit"].numpy()
            pm.append((np.isfinite(t_i) & (t_i < t_f + 0.005))
                      .reshape(H, W))
            vnames.append(fname)
        np.savez_compressed(odir / "proj_masks.npz",
                            masks=np.stack(pm) if pm else np.zeros((0, H, W), bool),
                            frames=np.array(vnames))
        C.save_json(odir / "views.json",
                    [{"frame": f, "w2c": w.tolist()} for _, f, w in views])
        print(f"[ip] {name} {m['label']}: remove {len(idx)} gaussians, "
              f"plane={'ok' if plane else 'none'}, views={vnames}")

    np.save(inp / "removal_union_idx.npy", np.nonzero(union)[0])
    C.save_json(inp / "prepare_meta.json",
                {"asset_removal_failures": len(asset_rm_failed),
                 "asset_removal_failed": asset_rm_failed})
    print(f"[ip] union removal: {int(union.sum())} / {len(splat_xyz)} gaussians")
    print(f"[ip] asset-volume removal failures: {len(asset_rm_failed)}"
          + (f" -> {asset_rm_failed}" if asset_rm_failed else ""))


def main(argv=None):
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    return _prepare()


if __name__ == "__main__":
    main()
