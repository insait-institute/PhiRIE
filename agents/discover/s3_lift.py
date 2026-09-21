"""Stage 3: lift each SAM3 instance to a world-frame point cloud + RGBA crop.

Per object obj_XX/: rgba.png (input for TRELLIS), points.ply (partial cloud,
alignment target for s5), meta.json. Also matches each instance to a GT
object (evaluation + background carving only).
"""
import json

import cv2
import numpy as np
from PIL import Image

from agents.core import common as C

MIN_PTS = 200
MAX_EXTENT = 0.9   # metres; larger detections (monitors, desks) are skipped
MIN_EXTENT = 0.025


def main():
    K, W, H, _ = C.load_intrinsics()
    rep = json.loads((C.OUT / "frame" / "rep_frame.json").read_text())
    w2c = np.array(rep["w2c"])
    c2w = np.linalg.inv(w2c)
    depth = np.load(C.OUT / "depth" / "depth.npz")["depth_corr"].astype(np.float64)
    md = np.load(C.OUT / "masks" / "masks.npz", allow_pickle=True)
    masks, labels, scores = md["masks"], md["labels"], md["scores"]
    rgb = np.asarray(Image.open(C.OUT / "frame" / rep["frame"]).convert("RGB"))

    gts = C.load_gt_instances()
    import open3d as o3d

    objects = []
    for i, (m, label, score) in enumerate(zip(masks, labels, scores)):
        label = str(label)
        core = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        pts = C.unproject(K, depth, core, c2w)
        if len(pts) < MIN_PTS:
            print(f"[s3] {i} {label}: only {len(pts)} pts, skip")
            continue
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pts = np.asarray(pc.points)
        if len(pts) < MIN_PTS:
            continue
        lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
        ext = hi - lo
        if ext.max() > MAX_EXTENT or ext.max() < MIN_EXTENT:
            print(f"[s3] {i} {label}: extent {np.round(ext, 3)} out of range, skip")
            continue

        # GT match for eval/carving: nearest compatible-label GT centroid
        cen = pts.mean(axis=0)
        gt_labels = C.TARGET_PROMPTS.get(label, [])
        cands = [g for g in gts if g["label"] in gt_labels]
        gt_id, gt_dist = None, None
        if cands:
            d = [np.linalg.norm(g["centroid"] - cen) for g in cands]
            j = int(np.argmin(d))
            if d[j] < 0.35:
                gt_id, gt_dist = int(cands[j]["object_id"]), float(d[j])

        odir = C.OUT / "objects" / f"obj_{i:02d}"
        odir.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(odir / "points.ply"), pc)

        vv, uu = np.nonzero(m)
        pad = int(0.15 * max(vv.ptp(), uu.ptp()) + 8)
        v0, v1 = max(vv.min() - pad, 0), min(vv.max() + pad, H)
        u0, u1 = max(uu.min() - pad, 0), min(uu.max() + pad, W)
        rgba = np.dstack([rgb[v0:v1, u0:u1],
                          (m[v0:v1, u0:u1] * 255).astype(np.uint8)])
        Image.fromarray(rgba).save(odir / "rgba.png")

        meta = {"index": int(i), "label": label, "score": float(score),
                "n_pts": len(pts), "centroid": cen, "extent": ext,
                "aabb": np.stack([lo, hi]), "bbox_px": [int(u0), int(v0), int(u1), int(v1)],
                "gt_object_id": gt_id, "gt_centroid_dist": gt_dist}
        C.save_json(odir / "meta.json", meta)
        objects.append(meta)
        print(f"[s3] {odir.name} {label} score={score:.2f} pts={len(pts)} "
              f"ext={np.round(ext, 3)} gt={gt_id}")

    C.save_json(C.OUT / "objects" / "objects.json", objects)
    print(f"[s3] lifted {len(objects)} objects")


if __name__ == "__main__":
    main()
