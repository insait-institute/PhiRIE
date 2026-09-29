"""automated_image2sim stage 0 (sam3 env, GPU): discover 3D object instances
WITHOUT GT semantic annotations.

Inputs: RGB frames + camera poses + scene mesh (+ nothing else).
  1. sample K frames uniformly over the trajectory
  2. SAM3 text-prompted segmentation with an open vocabulary of manipulable
     classes on every sampled frame
  3. lift each 2D instance mask to 3D by ray-casting masked pixels onto the
     mesh (first hit), keeping the dominant depth cluster
  4. merge per-frame proposals across frames by voxel IoU (union-find);
     instances confirmed in >=2 frames survive
Emits SIMANY_OUT/auto_instances.npz: per-instance mesh-vertex index sets +
label + score - the same contract load_gt_instances() provides, so the rest
of the factory runs unchanged.

Usage: auto_segment.py --scene-dir /data/ScanNetpp/data/<id> --out-dir OUT
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

from agents.core import common as _C

CKPT = _C.resolve_sam3_ckpt()

PROMPTS = ["bottle", "mug", "computer mouse", "keyboard", "headphones",
           "telephone", "box", "book", "laptop", "backpack", "shoe",
           "bowl", "plate", "kettle", "plant pot", "speaker",
           # furniture tier (SIMANY_FULL=1)
           ] + (["table", "chair", "office chair", "sofa", "monitor",
                 "shelf", "cabinet", "trash can", "lamp", "whiteboard",
                 "printer", "fan", "pillow", "suitcase"]
                if _C.env("FULL") == "1" else [])
FRAME_STRIDE = 12
SCORE_MIN = 0.45
PIX_STRIDE = 4
VOXEL = 0.02
MERGE_IOU = 0.25
MIN_FRAMES_SEEN = 2
MIN_VOXELS = 25
MAX_EXTENT = 0.9


def parse_colmap(images_txt):
    w2c = {}
    lines = [ln.strip() for ln in open(images_txt) if ln.strip()
             and not ln.startswith("#")]
    for ln in lines[::2]:
        p = ln.split()
        qw, qx, qy, qz = map(float, p[1:5])
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = list(map(float, p[5:8]))
        w2c[p[9]] = M
    return w2c


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frame-stride", type=int, default=FRAME_STRIDE)
    ap.add_argument("--mesh-path", default="",
                    help="override the scan mesh raycasted against (ablation: "
                         "a self-derived mesh instead of the dataset's own). "
                         "Emitted vert_idx then indexes THIS mesh's vertex "
                         "array - common.load_auto_instances() must read the "
                         "same file (PIPELINE_MESH_PLY / SIMANY_MESH_SRC).")
    args = ap.parse_args()
    sd, out = Path(args.scene_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh_path = Path(args.mesh_path) if args.mesh_path else \
        sd / "scans" / "mesh_aligned_0.05.ply"

    import open3d as o3d
    meta = json.loads((sd / "dslr" / "nerfstudio" /
                       "transforms_undistorted.json").read_text())
    K = np.array([[meta["fl_x"], 0, meta["cx"]],
                  [0, meta["fl_y"], meta["cy"]], [0, 0, 1]])
    W, H = int(meta["w"]), int(meta["h"])
    w2c_all = sorted(parse_colmap(sd / "dslr" / "colmap" / "images.txt").items())
    frames = w2c_all[::args.frame_stride]

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    verts = np.asarray(mesh.vertices)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    from scipy.spatial import cKDTree
    vtree = cKDTree(verts)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(device="cuda", checkpoint_path=CKPT,
                                   load_from_HF=False, compile=False)
    proc = Sam3Processor(model, resolution=1008, device="cuda",
                         confidence_threshold=SCORE_MIN)

    # ---- per-frame proposals lifted to mesh voxels -------------------------
    props = []  # {label, score, voxels(set), pts}
    for fi, (fname, w2c) in enumerate(frames):
        img = Image.open(sd / "dslr" / "resized_undistorted_images" / fname
                         ).convert("RGB")
        c2w = np.linalg.inv(w2c)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(img)
            for prompt in PROMPTS:
                proc.reset_all_prompts(state)
                state = proc.set_text_prompt(prompt=prompt, state=state)
                masks = state["masks"].squeeze(1).cpu().numpy()
                scores = state["scores"].float().cpu().numpy()
                for m, s in zip(masks, scores):
                    area = int(m.sum())
                    max_frac = 0.15 if _C.VOCAB.get(prompt, 1) <= 1.2 else 0.6
                    if not (500 <= area <= max_frac * W * H):
                        continue
                    vv, uu = np.nonzero(m[::PIX_STRIDE, ::PIX_STRIDE])
                    vv = vv * PIX_STRIDE + 0.5
                    uu = uu * PIX_STRIDE + 0.5
                    d_cam = np.stack([(uu - K[0, 2]) / K[0, 0],
                                      (vv - K[1, 2]) / K[1, 1],
                                      np.ones_like(uu)], axis=-1)
                    d_world = d_cam @ c2w[:3, :3].T
                    o_world = np.broadcast_to(c2w[:3, 3], d_world.shape)
                    rays = o3d.core.Tensor(np.concatenate(
                        [o_world, d_world], axis=1).astype(np.float32))
                    t_hit = scene.cast_rays(rays)["t_hit"].numpy()
                    ok = np.isfinite(t_hit)
                    if ok.sum() < 30:
                        continue
                    pts = o_world[ok] + d_world[ok] * t_hit[ok, None]
                    # dominant depth cluster: reject silhouette-spill hits
                    med = np.median(pts, axis=0)
                    rad = max(0.5, 0.7 * _C.VOCAB.get(prompt, MAX_EXTENT))
                    keep = np.linalg.norm(pts - med, axis=1) < rad
                    pts = pts[keep]
                    if len(pts) < 30:  # before max(): empty pts crashes it
                        continue
                    ext = pts.max(0) - pts.min(0)
                    if ext.max() > _C.VOCAB.get(prompt, MAX_EXTENT):
                        continue
                    vox = set(map(tuple, np.floor(pts / VOXEL).astype(np.int64)))
                    if len(vox) < MIN_VOXELS // 2:
                        continue
                    props.append({"label": prompt, "score": float(s),
                                  "vox": vox, "pts": pts, "frame": fname})
        print(f"[as] frame {fi + 1}/{len(frames)} {fname}: "
              f"{len(props)} proposals so far", flush=True)

    # ---- merge across frames ----------------------------------------------
    uf = UnionFind(len(props))
    for i in range(len(props)):
        for j in range(i + 1, len(props)):
            a, b = props[i], props[j]
            inter = len(a["vox"] & b["vox"])
            if inter == 0:
                continue
            iou = inter / len(a["vox"] | b["vox"])
            if iou > MERGE_IOU:
                uf.union(i, j)
    groups = {}
    for i in range(len(props)):
        groups.setdefault(uf.find(i), []).append(i)

    instances = []
    for gid, idxs in groups.items():
        frames_seen = {props[i]["frame"] for i in idxs}
        if len(frames_seen) < MIN_FRAMES_SEEN:
            continue
        pts = np.concatenate([props[i]["pts"] for i in idxs])
        vox = set().union(*[props[i]["vox"] for i in idxs])
        if len(vox) < MIN_VOXELS:
            continue
        labels = {}
        for i in idxs:
            labels[props[i]["label"]] = labels.get(props[i]["label"], 0) \
                + props[i]["score"]
        label = max(labels, key=labels.get)
        score = float(np.mean([props[i]["score"] for i in idxs]))
        # map to mesh vertices (the contract the factory consumes)
        d, vi = vtree.query(pts[:: max(len(pts) // 20000, 1)], k=1)
        vidx = np.unique(vi[d < 0.03])
        if len(vidx) < 10:
            continue
        gpts = verts[vidx]
        ext = gpts.max(0) - gpts.min(0)
        if ext.max() > _C.VOCAB.get(label, MAX_EXTENT) or ext.max() < 0.03:
            continue
        instances.append({"label": label, "score": score, "vert_idx": vidx,
                          "n_frames": len(frames_seen)})

    # cross-label dedupe (same physical object found via two prompts)
    instances.sort(key=lambda x: -x["score"])
    final = []
    for inst in instances:
        s = set(inst["vert_idx"])
        dup = False
        for f in final:
            fs = set(f["vert_idx"])
            if len(s & fs) / max(min(len(s), len(fs)), 1) > 0.5:
                dup = True
                break
        if not dup:
            final.append(inst)

    np.savez_compressed(
        out / "auto_instances.npz",
        labels=np.array([x["label"] for x in final]),
        scores=np.array([x["score"] for x in final]),
        n_frames=np.array([x["n_frames"] for x in final]),
        **{f"vert_idx_{k}": x["vert_idx"] for k, x in enumerate(final)})
    for k, x in enumerate(final):
        print(f"[as] inst_{k:02d} {x['label']} score={x['score']:.2f} "
              f"frames={x['n_frames']} verts={len(x['vert_idx'])}")
    print(f"[as] {len(final)} instances (from {len(props)} proposals, "
          f"{len(frames)} frames, no GT used)")


if __name__ == "__main__":
    main()
