"""Baseline: MaskClustering (CVPR 2024, PKU-EPIC) on ScanNet++ DSLR.

MaskClustering discovers class-agnostic 3D instances by merging per-frame 2D
masks (Cropformer) via multi-view mask-graph clustering, given posed RGB-D +
a point cloud. This adapter runs it on the SAME protocol as our
auto_segment.py (image + pose + mesh only, no GT), so results are comparable.

What MaskClustering needs per scene (their ScanNetPPDataset layout, all paths
relative to third_party/MaskClustering/data/scannetpp/):
  data/<seq>/iphone/rgb/frame_%06d.jpg        RGB (undistorted DSLR here)
  data/<seq>/iphone/render_depth/frame_%06d.png  MESH-RENDERED z-depth, uint16 mm
  data/<seq>/iphone/colmap/{cameras,images}.txt  COLMAP PINHOLE + w2c poses
  data/<seq>/output/mask/frame_%06d.png        Cropformer 2D instance masks (GPU)
  pcld_0.25/<seq>.pth                          scene point cloud {'sampled_coords'}
NB their scannetpp pipeline uses *rendered* depth (render_depth), not sensor
depth -- so ScanNet++ DSLR (no depth at all) is fully supported: we ray-cast
the scan mesh (common.mesh_zdepth) exactly as they render iPhone depth.

This adapter does three things:
  prepare      : build one scene's inputs in the layout above (CPU: mesh raycast).
  gpu-commands : print the exact GPU commands to run (Cropformer + clustering);
                 it does NOT execute them (run them on a GPU machine yourself).
  convert      : map MaskClustering's class-agnostic output (masks over the
                 downsampled scene point cloud) back to mesh-vertex index sets and
                 write auto_instances.npz -- the exact contract load_auto_instances
                 consumes (labels / scores / vert_idx_<k>).

Usage:
  python -m agents.baselines.maskclustering prepare      --scene-dir /path/to/ScanNetpp/data/<scene>
  python -m agents.baselines.maskclustering gpu-commands --seq <scene>
  python -m agents.baselines.maskclustering convert      --seq <scene> --out-dir OUT
  python -m agents.baselines.maskclustering smoke        --scene-dir /path/to/ScanNetpp/data/<scene>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MC_ROOT = ROOT / "third_party" / "MaskClustering"
MC_DATA = MC_ROOT / "data" / "scannetpp"          # their code hardcodes ./data/scannetpp
CONFIG_NAME = "scannetpp_dslr"
CROPFORMER_CKPT = MC_ROOT / "ckpt" / "Mask2Former_hornet_3x_576d0b.pth"
CROPFORMER_DEMO = (MC_ROOT / "third_party" / "detectron2" / "projects" /
                   "CropFormer" / "demo_cropformer" / "mask_predict.py")
CROPFORMER_CFG = (MC_ROOT / "third_party" / "detectron2" / "projects" /
                  "CropFormer" / "configs" / "entityv2" / "entity_segmentation" /
                  "mask2former_hornet_3x.yaml")

sys.path.insert(0, str(ROOT))
from agents.core import common as C  # noqa: E402  (mesh_zdepth, load_intrinsics)

DEPTH_SCALE = 1000.0        # MaskClustering ScanNetPPDataset.depth_scale (mm)
DEPTH_FAR = 20.0            # their DEPTH_TRUNC; matches render.yml far plane
DEFAULT_VOXEL = 0.025       # scene-point cloud downsample (m)
DEFAULT_STRIDE = 2          # keep every Nth DSLR frame


# --------------------------------------------------------------- helpers --

def _load_colmap_records(images_txt: Path):
    """Ordered list of {name, qvec(wxyz), tvec} from colmap images.txt."""
    lines = [ln.rstrip("\n") for ln in images_txt.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    recs = []
    for ln in lines[::2]:                     # meta line, then 2D-points line
        p = ln.split()
        recs.append({
            "qvec": tuple(map(float, p[1:5])),   # qw qx qy qz
            "tvec": tuple(map(float, p[5:8])),   # tx ty tz  (w2c)
            "name": p[9],
        })
    return recs


def _raycast_scene(mesh_ply: Path):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_ply))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene, np.asarray(mesh.vertices)


def _seq_dir(seq: str) -> Path:
    return MC_DATA / "data" / seq


# --------------------------------------------------------------- prepare --

def prepare(scene_dir: Path, seq: str | None = None, stride: int = DEFAULT_STRIDE,
            max_frames: int | None = None, voxel: float = DEFAULT_VOXEL):
    import cv2
    import open3d as o3d

    scene_dir = Path(scene_dir)
    seq = seq or scene_dir.name
    transforms = scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    images_txt = scene_dir / "dslr" / "colmap" / "images.txt"
    img_dir = scene_dir / "dslr" / "resized_undistorted_images"
    mesh_ply = scene_dir / "scans" / "mesh_aligned_0.05.ply"

    K, W, H, _ = C.load_intrinsics(transforms)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    recs = _load_colmap_records(images_txt)
    recs = recs[::stride]
    if max_frames is not None:
        recs = recs[:max_frames]

    sd = _seq_dir(seq)
    rgb_dir = sd / "iphone" / "rgb"
    depth_dir = sd / "iphone" / "render_depth"
    colmap_dir = sd / "iphone" / "colmap"
    for d in (rgb_dir, depth_dir, colmap_dir, sd / "output" / "mask"):
        d.mkdir(parents=True, exist_ok=True)
    (MC_DATA / "pcld_0.25").mkdir(parents=True, exist_ok=True)

    # cameras.txt : single PINHOLE camera (undistorted DSLR)
    (colmap_dir / "cameras.txt").write_text(
        "# Camera list\n"
        f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    # raycast scene once, render z-depth per frame
    scene, mesh_verts = _raycast_scene(mesh_ply)

    img_lines, frame_map = [], {}
    for i, r in enumerate(recs):
        name6 = f"frame_{i:06d}"
        # rgb: symlink the undistorted DSLR image
        dst = rgb_dir / f"{name6}.jpg"
        src = img_dir / r["name"]
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError:
            import shutil
            shutil.copy(src, dst)

        # depth: ray-cast mesh -> z-depth (m) -> uint16 mm
        w2c = np.eye(4)
        w2c[:3, :3] = C.quat_to_rot_wxyz(np.array(r["qvec"]))
        w2c[:3, 3] = r["tvec"]
        z = C.mesh_zdepth(scene, K, w2c, W, H, stride=1)   # (H,W) m, inf=no hit
        z = np.where(np.isfinite(z) & (z < DEPTH_FAR), z, 0.0)
        depth_mm = np.clip(np.round(z * DEPTH_SCALE), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(depth_dir / f"{name6}.png"), depth_mm)

        qw, qx, qy, qz = r["qvec"]
        tx, ty, tz = r["tvec"]
        img_lines.append(
            f"{i + 1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {name6}.jpg")
        img_lines.append("0.0 0.0 -1")            # dummy 2D-points line (unused)
        frame_map[name6] = r["name"]

    (colmap_dir / "images.txt").write_text(
        "# Image list\n" + "\n".join(img_lines) + "\n")

    # scene point cloud: voxel-downsample the mesh vertices they/we score on
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh_verts)
    coords = np.asarray(pcd.voxel_down_sample(voxel).points)
    import torch
    torch.save({"sampled_coords": coords},
               MC_DATA / "pcld_0.25" / f"{seq}.pth")

    meta = {"seq": seq, "scene_dir": str(scene_dir), "mesh_ply": str(mesh_ply),
            "W": W, "H": H, "n_frames": len(recs), "stride": stride,
            "voxel": voxel, "n_scene_points": int(len(coords)),
            "n_mesh_verts": int(len(mesh_verts)), "frame_map": frame_map}
    (sd / "mc_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[mc-prep] {seq}: {len(recs)} frames, {len(coords)} scene points "
          f"({len(mesh_verts)} mesh verts), img {W}x{H} -> {sd}")
    return meta


# ---------------------------------------------------------- gpu-commands --

def gpu_commands(seq: str):
    lines = [
        "# ---- MaskClustering baseline: GPU steps (run from the MaskClustering repo) ----",
        f"cd {MC_ROOT}",
        "",
        "# Step 1 - 2D instance masks with Cropformer.  BLOCKER: Cropformer is a",
        "#   separate source build (detectron2 + Entity CropFormer + MSDeformAttn",
        "#   CUDA op via ops/make.sh + mmcv) and needs the Mask2Former_hornet_3x",
        "#   checkpoint (HuggingFace).  Neither is installed; build in its own env",
        "#   on the GPU node, then point the python below at that env.  Masks land",
        "#   in data/scannetpp/data/<seq>/output/mask/frame_%06d.png (uint16 id map).",
        "#   Fallback: any 2D segmenter (e.g. our SAM3) can write those same PNGs.",
        f"<cropformer_env>/bin/python {CROPFORMER_DEMO} \\",
        f"    --config-file {CROPFORMER_CFG} \\",
        f"    --root data/scannetpp/data --image_path_pattern 'iphone/rgb/*.jpg' \\",
        f"    --dataset scannetpp --seq_name_list {seq} \\",
        f"    --opts MODEL.WEIGHTS {CROPFORMER_CKPT}",
        "",
        "# Step 2 - mask-graph clustering -> class-agnostic 3D instances.",
        f"{ROOT}/.venv-mc/bin/python main.py --config {CONFIG_NAME} --seq_name_list {seq}",
        "#   writes data/prediction/%s_class_agnostic/%s.npz" % (CONFIG_NAME, seq),
        "#      and data/scannetpp/data/%s/output/object/%s/object_dict.npy" % (seq, CONFIG_NAME),
        "",
        "# Step 3 - convert to our format (CPU):",
        f"cd {ROOT} && .venv-mc/bin/python -m agents.baselines.maskclustering "
        f"convert --seq {seq} --out-dir <SIMANY_OUT>",
    ]
    print("\n".join(lines))


# --------------------------------------------------------------- convert --

def convert(seq: str, out_dir: Path, radius: float | None = None):
    """MaskClustering class-agnostic masks -> auto_instances.npz (our format)."""
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    import torch

    meta = json.loads((_seq_dir(seq) / "mc_meta.json").read_text())
    radius = radius if radius is not None else meta["voxel"] * 1.5

    pred_npz = (MC_ROOT / "data" / "prediction" /
                f"{CONFIG_NAME}_class_agnostic" / f"{seq}.npz")
    d = np.load(pred_npz)
    pred_masks = d["pred_masks"]                      # (n_scene_points, n_inst) bool
    pred_score = d["pred_score"]
    n_inst = pred_masks.shape[1]

    coords = torch.load(MC_DATA / "pcld_0.25" / f"{seq}.pth")["sampled_coords"]
    coords = np.asarray(coords)

    ply = PlyData.read(meta["mesh_ply"])
    v = ply["vertex"]
    verts = np.stack([np.asarray(v[a], dtype=np.float64) for a in "xyz"], axis=1)
    vtree = cKDTree(verts)

    # optional per-instance frame counts / labels from object_dict.npy
    obj_dict = {}
    od_path = (_seq_dir(seq) / "output" / "object" / CONFIG_NAME / "object_dict.npy")
    if od_path.exists():
        obj_dict = np.load(od_path, allow_pickle=True).item()

    labels, scores, n_frames, vert_arrays = [], [], [], []
    for i in range(n_inst):
        scene_idx = np.nonzero(pred_masks[:, i])[0]
        if len(scene_idx) == 0:
            continue
        # expand each downsampled scene point back to nearby mesh vertices
        nbrs = vtree.query_ball_point(coords[scene_idx], r=radius)
        vidx = np.unique(np.concatenate([np.asarray(n, dtype=np.int64)
                                         for n in nbrs if len(n)])) \
            if any(len(n) for n in nbrs) else np.array([], dtype=np.int64)
        if len(vidx) < 10:
            continue
        nf = 0
        if i in obj_dict:
            nf = len({fid for (fid, _m, _c) in obj_dict[i]["mask_list"]})
        labels.append("object")                      # class-agnostic baseline
        scores.append(float(pred_score[i]) if i < len(pred_score) else 1.0)
        n_frames.append(int(nf))
        vert_arrays.append(vidx)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "auto_instances.npz",
        labels=np.array(labels),
        scores=np.array(scores, dtype=np.float64),
        n_frames=np.array(n_frames, dtype=np.int64),
        **{f"vert_idx_{k}": a for k, a in enumerate(vert_arrays)})
    print(f"[mc-convert] {seq}: {len(vert_arrays)} instances "
          f"(from {n_inst} clusters) -> {out_dir / 'auto_instances.npz'}")
    return len(vert_arrays)


# ----------------------------------------------------------------- smoke --

def smoke(scene_dir: Path):
    """CPU-only end-to-end check: prep a few frames, then fake a MaskClustering
    output and run the conversion, validating the auto_instances.npz contract."""
    import tempfile

    scene_dir = Path(scene_dir)
    seq = scene_dir.name
    print("=== smoke: prepare (3 frames) ===")
    meta = prepare(scene_dir, seq, stride=1, max_frames=3)

    # sanity: the prepared inputs load through MaskClustering's own dataset class
    print("=== smoke: MaskClustering ScanNetPPDataset loads the prepared scene ===")
    sys.path.insert(0, str(MC_ROOT))
    cwd = os.getcwd()
    os.chdir(MC_ROOT)
    try:
        from dataset.scannetpp import ScanNetPPDataset
        ds = ScanNetPPDataset(seq)
        frames = ds.get_frame_list(1)
        assert ds.image_size == (meta["W"], meta["H"]), ds.image_size
        depth = ds.get_depth(frames[0])
        pts = ds.get_scene_points()
        print(f"    image_size={ds.image_size} frames={len(frames)} "
              f"depth={depth.shape} max_depth_m={float(depth.max()):.2f} "
              f"scene_points={pts.shape}")
        assert depth.shape == (meta["H"], meta["W"])
        assert pts.shape[0] == meta["n_scene_points"]
        # ball_query shim import path used by clustering
        from utils.mask_backprojection import ball_query  # noqa: F401
        print("    ball_query import OK (shim or pytorch3d)")
    finally:
        os.chdir(cwd)

    # synthesize a class-agnostic prediction over the scene point cloud
    print("=== smoke: synthesize MaskClustering output + convert ===")
    coords = pts
    center = coords.mean(0)
    within = np.linalg.norm(coords - center, axis=1) < 0.3
    pred_masks = np.stack([within, ~within], axis=1)      # 2 fake instances
    pred_dir = MC_ROOT / "data" / "prediction" / f"{CONFIG_NAME}_class_agnostic"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(pred_dir / f"{seq}.npz", pred_masks=pred_masks,
             pred_score=np.ones(2), pred_classes=np.zeros(2, np.int32))

    with tempfile.TemporaryDirectory() as td:
        n = convert(seq, Path(td))
        d = np.load(Path(td) / "auto_instances.npz")
        assert "labels" in d and "scores" in d
        keys = [k for k in d.files if k.startswith("vert_idx_")]
        assert len(keys) == n, (len(keys), n)
        for k in keys:
            assert d[k].dtype.kind in "iu" and d[k].ndim == 1
        print(f"    auto_instances.npz OK: {n} instances, "
              f"labels={list(d['labels'])}, "
              f"verts_per_inst={[int(d[k].size) for k in keys]}")
    print("=== smoke PASSED ===")


# ------------------------------------------------------------------ main --

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--scene-dir", required=True)
    p.add_argument("--seq", default=None)
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--voxel", type=float, default=DEFAULT_VOXEL)

    s = sub.add_parser("gpu-commands")
    s.add_argument("--seq", required=True)

    c = sub.add_parser("convert")
    c.add_argument("--seq", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--radius", type=float, default=None)

    k = sub.add_parser("smoke")
    k.add_argument("--scene-dir", required=True)

    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.scene_dir, a.seq, a.stride, a.max_frames, a.voxel)
    elif a.cmd == "gpu-commands":
        gpu_commands(a.seq)
    elif a.cmd == "convert":
        convert(a.seq, a.out_dir, a.radius)
    elif a.cmd == "smoke":
        smoke(a.scene_dir)


if __name__ == "__main__":
    main()
