"""SHARP-variant lift stage (mirrors s3_lift.py's per-instance gates/outputs
exactly), but sourcing depth from the SHARP splat's own gsplat render
(sharp_render_depth.py) instead of DA3 monodepth + scan-mesh scale bridge,
and with NO ground truth anywhere (single feedforward splat from one photo,
no ScanNet++ instance annotations to match against).

Per object obj_XX/: rgba.png (TRELLIS input), points.ply (alignment target
for s5_align.py), meta.json (gt fields hard-set to None/null).

points.ply frame fix (2nd pass): points.ply is written in an approximate
z-up WORLD frame (pts @ R_ZUP.T, see sharp_ply_meta.R_ZUP), not the raw
SHARP camera frame -- s5_align.py's align_object structurally assumes a
z-up world (rz(yaw) rotates about world-Z; scale is matched along the
longest-observed axis against the TRELLIS canonical mesh's z axis), and the
SHARP splat's own frame is OpenCV camera convention (y-down, z-forward),
NOT z-up. Rejecting most flat/thin real objects at the size-sanity gate in
the first pass traced back to exactly this mismatch. meta.json's
centroid/extent/aabb/bbox_px are DELIBERATELY LEFT in the original camera
frame (unrotated) -- sharp_render_object_views.py needs the original-frame
centroid as its lookat_point for gsplat rendering, and its depth-band
masking operates in that same original camera frame. `extent`'s magnitudes
are unaffected by R_ZUP either way (pure permutation + single sign flip).
points.ply is therefore the ONLY file whose frame changes between passes.

.venv, no GPU needed (pure numpy/open3d/cv2). No CLI args; reads
$SIMANY_OUT/masks/masks.npz + $SIMANY_OUT/depth/depth.npz, and the fixed
original-photo path below. Idempotent/rerunnable; skips rewriting rgba.png
if it already exists so s4_trellis.py's mtime-based cache stays valid
across reruns of this stage alone.
"""
import cv2
import numpy as np
from PIL import Image

from agents.core import common as C
from agents.single_image.sharp_ply_meta import R_ZUP

ORIG_JPG = "/group/worldcept/PhiRIE/code/sharp/inputs/DSC08561/DSC08561.JPG"

MIN_PTS = 200
MAX_EXTENT = 0.9   # metres; larger detections (monitors, desks) are skipped
MIN_EXTENT = 0.025


def main():
    d = np.load(C.OUT / "depth" / "depth.npz")
    K = np.asarray(d["K"], dtype=np.float64)
    W, H = int(d["W"]), int(d["H"])
    depth = d["depth_corr"].astype(np.float64)
    c2w = np.eye(4)  # splat frame == original photo's identity camera pose

    md = np.load(C.OUT / "masks" / "masks.npz", allow_pickle=True)
    masks, labels, scores = md["masks"], md["labels"], md["scores"]
    rgb = np.asarray(Image.open(ORIG_JPG).convert("RGB"))
    assert rgb.shape[:2] == (H, W), \
        f"orig photo {rgb.shape[:2]} != ply camera (H,W)=({H},{W})"

    import open3d as o3d

    objects = []
    for i, (m, label, score) in enumerate(zip(masks, labels, scores)):
        label = str(label)
        core = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        pts = C.unproject(K, depth, core, c2w)
        if len(pts) < MIN_PTS:
            print(f"[sl] {i} {label}: only {len(pts)} pts, skip")
            continue
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pts = np.asarray(pc.points)
        if len(pts) < MIN_PTS:
            print(f"[sl] {i} {label}: only {len(pts)} pts after outlier removal, skip")
            continue
        lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
        ext = hi - lo
        if ext.max() > MAX_EXTENT or ext.max() < MIN_EXTENT:
            print(f"[sl] {i} {label}: extent {np.round(ext, 3)} out of range, skip")
            continue

        cen = pts.mean(axis=0)  # original camera frame, for meta.json only

        odir = C.OUT / "objects" / f"obj_{i:02d}"
        odir.mkdir(parents=True, exist_ok=True)
        # points.ply: rotated into the approximate z-up world frame that
        # s5_align.py's registration assumes. meta.json below stays in the
        # original (unrotated) camera frame -- see module docstring.
        pts_zup = pts @ R_ZUP.T
        pc_zup = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_zup))
        o3d.io.write_point_cloud(str(odir / "points.ply"), pc_zup)

        vv, uu = np.nonzero(m)
        pad = int(0.15 * max(vv.ptp(), uu.ptp()) + 8)
        v0, v1 = max(vv.min() - pad, 0), min(vv.max() + pad, H)
        u0, u1 = max(uu.min() - pad, 0), min(uu.max() + pad, W)
        rgba_path = odir / "rgba.png"
        if not rgba_path.exists():
            rgba = np.dstack([rgb[v0:v1, u0:u1],
                              (m[v0:v1, u0:u1] * 255).astype(np.uint8)])
            Image.fromarray(rgba).save(rgba_path)

        meta = {"index": int(i), "label": label, "score": float(score),
                "n_pts": len(pts), "centroid": cen, "extent": ext,
                "aabb": np.stack([lo, hi]), "bbox_px": [int(u0), int(v0), int(u1), int(v1)],
                "gt_object_id": None, "gt_centroid_dist": None}
        C.save_json(odir / "meta.json", meta)
        objects.append(meta)
        print(f"[sl] {odir.name} {label} score={score:.2f} pts={len(pts)} "
              f"ext={np.round(ext, 3)}")

    C.save_json(C.OUT / "objects" / "objects.json", objects)
    print(f"[sl] lifted {len(objects)} objects")


if __name__ == "__main__":
    main()
