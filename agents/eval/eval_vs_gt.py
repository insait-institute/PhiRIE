"""Honest re-evaluation: F1 of registered assets vs INDEPENDENT ScanNet++ GT.

Under SIMANY_AUTO the factory writes each object's gt_points.ply by sampling the
pipeline's OWN discovered submesh, so factory_align's reported F1 is "vs own
extraction". Here every registered asset is re-scored against the true GT
instance submesh instead:
  1. match each discovered object to a GT instance: vertex IoU when discovery
     raycasted the GT mesh itself (the *_auto fleet), else mutual 2cm
     nearest-neighbor point overlap (derived-mesh rungs); threshold 0.25
  2. sample 20k points from the TRUE GT submesh (factory_prepare convention)
  3. F1@20/40mm of aligned.json's T applied to the same 20k-sample asset
     surface factory_align uses (f1_eval/apply_T imported for bit-identity)
Writes eval_vs_gt.json into each scene output dir.

Usage: eval_vs_gt.py [out_dir ...]      default: all outputs/*_auto
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from agents.core import common as C
from agents.discover.factory_prepare import MIN_DIM, MAX_DIM, WHITELIST
from agents.assets.s5_align import apply_T, f1_eval

MATCH_THR = 0.25
OVERLAP_TOL_M = 0.02
N_GT_SAMPLES = 20_000
N_MESH_SAMPLES = 20_000
N_OVERLAP_PTS = 4000

DATA = C.SCENE_DIR.parent


def load_scene_gt(scene_id, *, dataset_root=None):
    """load_gt_instances(), parameterized by scene (env-free, multi-scene)."""
    from plyfile import PlyData

    data = Path(dataset_root) / "data" if dataset_root is not None else DATA
    sd = data / scene_id / "scans"
    seg_idx = np.asarray(json.loads((sd / "segments.json").read_text())["segIndices"])
    anno = json.loads((sd / "segments_anno.json").read_text())
    ply = PlyData.read(str(sd / "mesh_aligned_0.05.ply"))
    v = ply["vertex"]
    verts = np.stack([np.asarray(v[a], dtype=np.float64) for a in "xyz"], axis=1)
    faces = np.vstack(ply["face"]["vertex_indices"])
    gts = []
    for g in anno["segGroups"]:
        idx = np.nonzero(np.isin(seg_idx, np.asarray(g["segments"])))[0]
        if len(idx) < 10:
            continue
        gts.append({"object_id": g["objectId"], "label": g["label"],
                    "vert_idx": idx})
    return verts, faces, gts


def submesh_points(verts, faces, vidx, n=N_GT_SAMPLES, *, seed=None):
    """Same sampling convention as factory_prepare's gt_points.ply."""
    import trimesh
    inset = np.zeros(len(verts), bool)
    inset[vidx] = True
    fmask = inset[faces].all(axis=1)
    if fmask.sum() < 4:
        return verts[vidx]
    sub = trimesh.Trimesh(verts, faces[fmask], process=False)
    kwargs = {} if seed is None else {"seed": seed}
    pts, _ = trimesh.sample.sample_surface(sub, n, **kwargs)
    return np.asarray(pts, dtype=np.float64)


def overlap_score(a_pts, g_pts):
    """Mutual nearest-neighbor coverage at 2cm (works across meshes)."""
    from scipy.spatial import cKDTree
    a = a_pts[:: max(len(a_pts) // N_OVERLAP_PTS, 1)]
    g = g_pts[:: max(len(g_pts) // N_OVERLAP_PTS, 1)]
    cov_a = float((cKDTree(g).query(a, k=1)[0] < OVERLAP_TOL_M).mean())
    cov_g = float((cKDTree(a).query(g, k=1)[0] < OVERLAP_TOL_M).mean())
    return min(cov_a, cov_g)


def best_match(a_idx, a_pts, gts, gt_verts, same_mesh):
    """Best GT instance for one discovered instance, or (None, score)."""
    a_set = np.unique(a_idx)
    lo, hi = a_pts.min(axis=0) - 0.1, a_pts.max(axis=0) + 0.1
    best, best_s = None, 0.0
    for g in gts:
        if (g["hi"] < lo).any() or (g["lo"] > hi).any():
            continue
        if same_mesh:  # vertex IoU on the shared mesh
            inter = np.intersect1d(a_set, g["vert_idx"], assume_unique=True).size
            s = inter / (len(a_set) + len(g["vert_idx"]) - inter)
        else:
            s = overlap_score(a_pts, gt_verts[g["vert_idx"]])
        if s > best_s:
            best, best_s = g, s
    return (best, best_s) if best_s >= MATCH_THR else (None, best_s)


def eval_scene(out):
    import trimesh

    scene_id = out.name.split("_")[0]
    verts, faces, gts = load_scene_gt(scene_id)
    gts = [g for g in gts
           if g["label"].strip().lower() not in C.STRUCTURAL_EXCLUDE]
    for g in gts:
        gp = verts[g["vert_idx"]]
        g["lo"], g["hi"] = gp.min(axis=0), gp.max(axis=0)
    by_id = {g["object_id"]: g for g in gts}
    # the population factory_prepare's GT mode would enumerate (missed count)
    wl_ids = [g["object_id"] for g in gts
              if g["label"].strip().lower() in WHITELIST
              and MIN_DIM <= (g["hi"] - g["lo"]).max()
              <= C.VOCAB.get(g["label"].strip().lower(), MAX_DIM)]

    # discovered geometry lives on whatever mesh auto_segment raycasted
    derived = out / "derived_mesh.ply"
    same_mesh = not derived.exists()
    if same_mesh:
        pverts = verts
    else:
        from plyfile import PlyData
        v = PlyData.read(str(derived))["vertex"]
        pverts = np.stack([np.asarray(v[a], dtype=np.float64) for a in "xyz"],
                          axis=1)
    npz = out / "auto_instances.npz"
    auto = np.load(npz) if npz.exists() else None

    objects = json.loads((out / "objects" / "objects.json").read_text())
    recs, matched_ids = [], set()
    for meta in objects:
        name = f"obj_{meta['index']:02d}"
        rec = {"name": name, "label": meta["label"], "matched_gt_id": None,
               "matched_gt_label": None, "match_iou": 0.0,
               "f1_20_gt": None, "f1_40_gt": None, "tier": None}
        oid = meta.get("gt_object_id")
        if oid is None:
            g, iou = None, 0.0
        elif oid < 1000:  # GT-driven mode: the instance IS a GT instance
            g, iou = by_id.get(oid), 1.0
        else:
            a_idx = auto[f"vert_idx_{oid - 1000}"]
            g, iou = best_match(a_idx, pverts[a_idx], gts, verts, same_mesh)
        if g is not None:
            rec["matched_gt_id"] = int(g["object_id"])
            rec["matched_gt_label"] = g["label"]
            rec["match_iou"] = float(iou)
            matched_ids.add(g["object_id"])

        odir = out / "objects" / name
        al = None
        if (odir / "aligned.json").exists():
            al = json.loads((odir / "aligned.json").read_text())
            rec["tier"] = al.get("tier")
        if g is not None and al is not None \
                and (odir / "trellis_mesh.ply").exists():
            gt_pts = submesh_points(verts, faces, g["vert_idx"])
            mesh = trimesh.load(odir / "trellis_mesh.ply", process=False)
            mesh_pts, _ = trimesh.sample.sample_surface(mesh, N_MESH_SAMPLES)
            ev = f1_eval(apply_T(np.asarray(al["T"], dtype=np.float64),
                                 np.asarray(mesh_pts, dtype=np.float64)),
                         gt_pts)
            rec["f1_20_gt"] = ev["f1@20mm"]["f1"]
            rec["f1_40_gt"] = ev["f1@40mm"]["f1"]
        recs.append(rec)
        f20 = rec["f1_20_gt"]
        print(f"[ev] {out.name} {name} {meta['label']}: "
              f"gt={rec['matched_gt_id']} ({rec['matched_gt_label']}) "
              f"iou={iou:.2f} tier={rec['tier']} "
              + (f"F1@20={f20:.3f}" if f20 is not None else "F1@20=-"))

    scored = [r["f1_20_gt"] for r in recs
              if r["f1_20_gt"] is not None and r["tier"] in ("A", "B")]
    res = {"scene": scene_id, "out_dir": str(out),
           "n_objects": len(recs),
           "n_matched": sum(r["matched_gt_id"] is not None for r in recs),
           "n_unmatched": sum(r["matched_gt_id"] is None for r in recs),
           "n_gt_missed": sum(i not in matched_ids for i in wl_ids),
           "n_gt_whitelist": len(wl_ids),
           "mean_f1_20_gt_AB": float(np.mean(scored)) if scored else None,
           "objects": recs}
    C.save_json(out / "eval_vs_gt.json", res)
    print(f"[ev] {out.name}: {res['n_matched']}/{res['n_objects']} matched, "
          f"{res['n_gt_missed']}/{res['n_gt_whitelist']} GT missed, "
          f"mean F1@20 vs GT (A/B) = {res['mean_f1_20_gt_AB']}")
    return res


def main():
    outs = [Path(a) for a in sys.argv[1:]] \
        or sorted((C.ROOT / "outputs").glob("*_auto"))
    pooled20, pooled40, means, fails = [], [], [], []
    n_obj = n_match = n_missed = n_wl = 0
    for out in outs:
        try:
            res = eval_scene(out)
        except Exception as e:
            traceback.print_exc()
            fails.append((out.name, repr(e)))
            continue
        n_obj += res["n_objects"]
        n_match += res["n_matched"]
        n_missed += res["n_gt_missed"]
        n_wl += res["n_gt_whitelist"]
        pooled20 += [r["f1_20_gt"] for r in res["objects"]
                     if r["f1_20_gt"] is not None and r["tier"] in ("A", "B")]
        pooled40 += [r["f1_40_gt"] for r in res["objects"]
                     if r["f1_40_gt"] is not None and r["tier"] in ("A", "B")]
        if res["mean_f1_20_gt_AB"] is not None:
            means.append(res["mean_f1_20_gt_AB"])
    print(f"\n[ev] scenes ok {len(outs) - len(fails)}/{len(outs)}; "
          f"objects={n_obj} matched={n_match} "
          f"({n_match / max(n_obj, 1):.1%}) gt_missed={n_missed}/{n_wl}")
    if pooled20:
        print(f"[ev] pooled F1 vs GT (matched, tier A/B, n={len(pooled20)}): "
              f"@20mm={np.mean(pooled20):.4f} @40mm={np.mean(pooled40):.4f}")
    if means:
        print(f"[ev] unweighted per-scene mean F1@20mm vs GT "
              f"({len(means)} scenes): {np.mean(means):.4f}")
    for name, err in fails:
        print(f"[ev] FAILED {name}: {err}")


if __name__ == "__main__":
    main()
