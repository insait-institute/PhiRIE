"""Stage 0: pick the representative frame.

SimFoundry [arXiv:2606.28276] defaults to frame 0 of a video that starts on a clear whole-scene
view; ScanNet++ DSLR captures walk the room, so instead we auto-pick the frame
that sees the most target objects (GT centroids projected + mesh occlusion
check). GT is used ONLY to choose the frame; perception never sees it.

GT-FREE FALLBACK (BEHAVIOR/oracle/DROID `recon_scenes` layouts have no
`scans/segments.json` + `scans/segments_anno.json`, so `C.load_gt_instances()`
cannot resolve there; also reachable by an explicit `--no-gt` / SIMANY_NO_GT=1
override on any scene): frame choice switches to `select_frame_gt_free()`,
which needs no object-level GT at all --
  1. ORB keypoint density per candidate frame (cheap, deterministic, image-
     only): a proxy for "how much graspable clutter is in view", since a
     cluttered desk/counter produces far more distinct corners than a blank
     wall or floor.
  2. IF a non-GT reconstruction mesh is available (`C.PIPELINE_MESH_PLY` --
     never a path under a `gt/` dir or shaped like `*segments*`/`*anno*`,
     see `is_gt_path()`): a raycast-based room-footprint-coverage score --
     how much of the room's overall XY extent this one view's ray hits
     span, as a "widest FOV / most of the room visible" signal. The room's
     own reconstructed geometry is fair game (it is not the hidden
     interactive/simulator object-level state `oracle/gt_export.py` vaults);
     skipped gracefully (feature density alone decides) when no such mesh
     exists, which is the common case for `recon_scenes` scenes that haven't
     run a mesh-derivation stage.
The two signals are summed (equal weight, both normalized to ~[0,1]); ties
broken by frame filename so the same inputs always pick the same frame.
See `agents/recon/droid_static_select.py` for the sibling GT-free "good
frame" heuristic built for the DROID track -- it solves a different
sub-problem (viewpoint-diverse multi-frame capping for splat training, not
single-frame selection) so there is no shared code, but the same "no GT,
cheap/CPU, deterministic" design contract applies here.

Outputs OUT/frame/rep_frame.json + copies the chosen image.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np

from agents.core import common as C

GT_FREE_MAX_FEATURES = 1500   # ORB cap per frame -- cheap even over hundreds of frames
GT_FREE_COVERAGE_STRIDE = 24  # raycast grid stride (px) for the room-coverage signal


def gt_instances_available() -> bool:
    """Cheap existence check for `load_gt_instances()`'s two hard file
    dependencies, WITHOUT ever reading/parsing either of them."""
    return C.SEGMENTS_JSON.exists() and C.SEGMENTS_ANNO_JSON.exists()


def no_gt_requested(no_gt_flag: bool = False) -> bool:
    """--no-gt CLI flag OR SIMANY_NO_GT=1 (C.env() also honours the legacy
    SIMF_NO_GT prefix, same as every other SIMANY_* setting)."""
    return bool(no_gt_flag) or C.env("NO_GT") == "1"


def is_gt_path(p) -> bool:
    """Hard safety gate: True if a path lives under a `gt/` directory or is
    named like an instance-GT annotation file (segments*/*anno*) -- the two
    shapes of ground truth `load_gt_instances()` (ScanNet++) and
    `oracle/gt_export.py` (BEHAVIOR) guard. `C.PIPELINE_MESH_PLY` never
    resolves under `gt/` by construction (see agents/core/common.py), but
    the GT-free path checks this explicitly anyway before touching it --
    belt and suspenders, and it is exactly the property
    tests/test_gt_free_frame_selection.py verifies."""
    p = Path(p)
    parts = {part.lower() for part in p.parts}
    name = p.name.lower()
    return "gt" in parts or "segments" in name or "anno" in name


def select_frame_gt(K, W, H, w2c_all):
    """Original GT-driven heuristic (unchanged): highest weighted count of
    GT target-object centroids that project in-frame and are unoccluded."""
    gts = C.load_gt_instances()
    scene = C.make_raycast_scene()

    gt_labels_flat = {l for ls in C.TARGET_PROMPTS.values() for l in ls}
    targets = [g for g in gts if g["label"] in gt_labels_flat]
    # weight actually-graspable things higher than keyboards/boxes
    weights = {"plastic bottle": 2.0, "glass bottle": 2.0, "bottle": 2.0,
               "mug": 2.0, "cup": 2.0, "mouse": 1.5}
    cents = np.stack([g["centroid"] for g in targets])

    import open3d as o3d
    best = None
    for name, w2c in sorted(w2c_all.items()):
        c2w = np.linalg.inv(w2c)
        cam = c2w[:3, 3]
        pc = cents @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        uv = pc[:, :2] / np.clip(z[:, None], 1e-6, None)
        u = uv[:, 0] * K[0, 0] + K[0, 2]
        v = uv[:, 1] * K[1, 1] + K[1, 2]
        m = 0.08  # keep centroids away from image border (8%)
        inb = ((z > 0.7) & (z < 3.5) &
               (u > W * m) & (u < W * (1 - m)) & (v > H * m) & (v < H * (1 - m)))
        idx = np.nonzero(inb)[0]
        if len(idx) == 0:
            continue
        # occlusion: first mesh hit along the ray to the centroid must be near it
        dirs = cents[idx] - cam
        dist = np.linalg.norm(dirs, axis=1)
        rays = o3d.core.Tensor(np.concatenate(
            [np.broadcast_to(cam, dirs.shape), dirs / dist[:, None]],
            axis=1).astype(np.float32))
        t_hit = scene.cast_rays(rays)["t_hit"].numpy()
        vis = idx[np.abs(t_hit - dist) < 0.20]
        score = sum(weights.get(targets[i]["label"], 1.0) for i in vis)
        if best is None or score > best["score"]:
            best = {"frame": name, "score": float(score),
                    "visible": [{"object_id": targets[i]["object_id"],
                                 "label": targets[i]["label"],
                                 "centroid": targets[i]["centroid"].tolist()}
                                for i in vis],
                    "w2c": w2c.tolist()}

    assert best is not None, "no frame sees any target object"
    return best


def feature_density(img_path, max_features: int = GT_FREE_MAX_FEATURES):
    """ORB keypoint count / max_features -> ~[0,1] density proxy. Returns
    None if the image can't be read (missing/corrupt frame)."""
    import cv2

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    orb = cv2.ORB_create(nfeatures=max_features)
    kps = orb.detect(img, None)
    return len(kps) / max_features


def room_coverage_score(mesh, scene, K, w2c, W, H,
                         stride: int = GT_FREE_COVERAGE_STRIDE):
    """Fraction of the room's overall XY footprint this single view's ray
    hits span -- a GT-free "widest FOV / most of the room visible" signal.
    `mesh`/`scene` come from the reconstruction's own mesh
    (`C.PIPELINE_MESH_PLY`): the room's geometry, not per-object GT."""
    lo_room = np.asarray(mesh.get_min_bound())[:2]
    hi_room = np.asarray(mesh.get_max_bound())[:2]
    room_area = float(np.prod(np.maximum(hi_room - lo_room, 1e-6)))

    depth = C.mesh_zdepth(scene, K, w2c, W, H, stride=stride)
    vv_idx, uu_idx = np.nonzero(np.isfinite(depth))
    if len(vv_idx) == 0:
        return 0.0
    us = np.arange(0, W, stride, dtype=np.float64) + 0.5
    vs = np.arange(0, H, stride, dtype=np.float64) + 0.5
    u, v, z = us[uu_idx], vs[vv_idx], depth[vv_idx, uu_idx]
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z], axis=1)
    c2w = np.linalg.inv(w2c)
    pts = pts_cam @ c2w[:3, :3].T + c2w[:3, 3]
    lo_hit, hi_hit = pts[:, :2].min(axis=0), pts[:, :2].max(axis=0)
    hit_area = float(np.prod(np.maximum(hi_hit - lo_hit, 1e-6)))
    return float(np.clip(hit_area / max(room_area, 1e-6), 0.0, 1.0))


def select_frame_gt_free(K, W, H, w2c_all):
    """GT-free frame choice: ORB feature density (+ room-coverage raycast
    when a non-GT mesh is available). See module docstring."""
    mesh, scene = None, None
    mesh_path = C.PIPELINE_MESH_PLY
    if mesh_path.exists() and not is_gt_path(mesh_path):
        import open3d as o3d
        try:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        except Exception as exc:
            print(f"[s0] GT-free: mesh at {mesh_path} unusable for the "
                  f"coverage signal ({type(exc).__name__}: {exc}); "
                  "falling back to feature density only")
            mesh, scene = None, None

    scored = {}
    for name, w2c in sorted(w2c_all.items()):
        fd = feature_density(C.IMAGES_DIR / name)
        if fd is None:
            continue
        cov = room_coverage_score(mesh, scene, K, w2c, W, H) \
            if mesh is not None else None
        combined = fd + (cov if cov is not None else 0.0)
        scored[name] = {"feature_density": fd, "coverage_score": cov,
                         "combined_score": combined}

    assert scored, ("GT-free frame selection found no readable frames "
                     f"under {C.IMAGES_DIR}")
    best_name = max(scored, key=lambda n: (scored[n]["combined_score"], n))
    s = scored[best_name]
    return {"frame": best_name, "score": s["combined_score"],
            "mode": "gt_free_heuristic",
            "feature_density": s["feature_density"],
            "coverage_score": s["coverage_score"],
            "visible": [], "w2c": w2c_all[best_name].tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gt", action="store_true",
                     help="force the GT-free fallback heuristic even when "
                          "GT instance files ARE present (SIMANY_NO_GT=1 has "
                          "the same effect); the default already falls back "
                          "automatically whenever they're absent")
    args = ap.parse_args(argv)

    K, W, H, _ = C.load_intrinsics()
    w2c_all = C.load_colmap_w2c()

    if gt_instances_available() and not no_gt_requested(args.no_gt):
        best = select_frame_gt(K, W, H, w2c_all)
    else:
        best = select_frame_gt_free(K, W, H, w2c_all)

    out = C.OUT / "frame"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(C.IMAGES_DIR / best["frame"], out / best["frame"])
    C.save_json(out / "rep_frame.json", best)
    if best.get("mode") == "gt_free_heuristic":
        print(f"[s0] GT-free: chose {best['frame']}  "
              f"feature_density={best['feature_density']:.3f}  "
              f"coverage={best['coverage_score']}  score={best['score']:.3f}")
    else:
        print(f"[s0] chose {best['frame']}  score={best['score']:.1f}  "
              f"visible={len(best['visible'])}: "
              f"{sorted(set(v['label'] for v in best['visible']))}")


if __name__ == "__main__":
    main()
