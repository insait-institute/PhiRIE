"""V_mesh slot alternative: ReconViaGen (multi-view reconstruction-consistent
generation, arXiv 2510.23306) instead of single-view TRELLIS.

Per object: collect up to N_VIEWS occlusion-aware masked crops across the
DSLR trajectory, run TrellisVGGTTo3DPipeline (VGGT-conditioned TRELLIS),
save rvg/ mesh+gaussians, align with the SAME registration and evaluate the
SAME GT F1 as the TRELLIS asset for a head-to-head comparison.

GPU. Usage: s4_reconviagen.py --objects 0,3,6,7,10,13 [--n-views 12]
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
sys.modules.setdefault("rembg", types.ModuleType("rembg"))

import numpy as np
from PIL import Image, ImageFilter

# xformers 0.0.28.post1 dropped the fmha-level re-export (same patch as s4)
import xformers.ops.fmha as _fmha
from xformers.ops.fmha.attn_bias import BlockDiagonalMask as _BDM

if not hasattr(_fmha, "BlockDiagonalMask"):
    _fmha.BlockDiagonalMask = _BDM

from agents.core import common as C

RVG_DIR = Path(os.environ.get("SIMANY_RVG_DIR", C.ROOT / "third_party" / "ReconViaGen"))
sys.path.insert(0, str(RVG_DIR))
# vendored VGGT uses absolute self-imports (from vggt.models...)
sys.path.insert(0, str(RVG_DIR / "wheels" / "vggt"))

N_VIEWS = 12
MIN_FRAME_GAP = 4  # enforce viewpoint diversity


def collect_views(gt, K, W, H, w2c_all, scene_full, verts, faces, n_views):
    """Occlusion-aware ranked views + full-frame instance masks."""
    import open3d as o3d
    rng = np.random.RandomState(0)
    gv = verts[gt["vert_idx"]]
    sample = gv[rng.choice(len(gv), min(120, len(gv)), replace=False)]
    cand = []
    for fi, (fname, w2c) in enumerate(w2c_all):
        pc = sample @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        if (z < 0.25).mean() > 0.05:
            continue
        u = pc[:, 0] / z * K[0, 0] + K[0, 2]
        v = pc[:, 1] / z * K[1, 1] + K[1, 2]
        inb = (z > 0.25) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
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
        cand.append((vis * np.sqrt(u[inb].ptp() * v[inb].ptp()), fi, fname, w2c))
    cand.sort(key=lambda c: -c[0])
    picked = []
    for score, fi, fname, w2c in cand:
        if all(abs(fi - p[1]) >= MIN_FRAME_GAP for p in picked):
            picked.append((score, fi, fname, w2c))
        if len(picked) == n_views:
            break

    inset = np.zeros(len(verts), bool)
    inset[gt["vert_idx"]] = True
    fmask = inset[faces].all(axis=1)
    sub = o3d.t.geometry.RaycastingScene()
    sub.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                  o3d.utility.Vector3iVector(faces[fmask]))))
    views = []
    for _, _, fname, w2c in picked:
        pc = gv @ w2c[:3, :3].T + w2c[:3, 3]
        u = pc[:, 0] / pc[:, 2] * K[0, 0] + K[0, 2]
        v = pc[:, 1] / pc[:, 2] * K[1, 1] + K[1, 2]
        pad = int(0.15 * max(u.ptp(), v.ptp()) + 8)
        u0, u1 = int(max(u.min() - pad, 0)), int(min(u.max() + pad, W))
        v0, v1 = int(max(v.min() - pad, 0)), int(min(v.max() + pad, H))
        c2w = np.linalg.inv(w2c)
        uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
        d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                          np.ones_like(uu)], axis=-1)
        d_world = (d_cam @ c2w[:3, :3].T).reshape(-1, 3).astype(np.float32)
        o_world = np.broadcast_to(c2w[:3, 3], d_world.shape).astype(np.float32)
        rays = o3d.core.Tensor(np.concatenate([o_world, d_world], axis=1))
        t_i = sub.cast_rays(rays)["t_hit"].numpy()
        t_f = scene_full.cast_rays(rays)["t_hit"].numpy()
        m = (np.isfinite(t_i) & (t_i < t_f + 0.005)).reshape(uu.shape)
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                       .filter(ImageFilter.MaxFilter(7))) > 127
        if m.sum() < 400:
            continue
        views.append((fname, (u0, v0, u1, v1), m))
    return views


def main():
    import torch
    import trimesh
    from trellis.pipelines import TrellisVGGTTo3DPipeline

    from agents.assets.factory_align import TIER_A_F1_20  # noqa: F401 (env parity)
    from agents.assets.s5_align import align_object, apply_T, f1_eval, gt_submesh_points, \
        scene_mesh_arrays

    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True)
    ap.add_argument("--n-views", type=int, default=N_VIEWS)
    args = ap.parse_args()
    picks = [int(x) for x in args.objects.split(",")]

    K, W, H, _ = C.load_intrinsics()
    w2c_all = sorted(C.load_colmap_w2c().items())
    verts, faces = scene_mesh_arrays()
    scene_full = C.make_raycast_scene()
    gts = {g["object_id"]: g for g in C.load_instances()}
    objects = {m["index"]: m
               for m in json.loads((C.OUT / "objects" / "objects.json").read_text())}

    pipe = TrellisVGGTTo3DPipeline.from_pretrained("Stable-X/trellis-vggt-v0-2")
    # their low_vram path forgets to move the slat conditioning modules;
    # A6000 fits the whole no-refinement pipeline, so move everything
    pipe.low_vram = False
    if hasattr(pipe, "cuda"):
        pipe.cuda()
    for mod in pipe.models.values():
        mod.to("cuda")
    for attr in vars(pipe).values():
        if isinstance(attr, torch.nn.Module):
            attr.to("cuda")

    results = []
    for idx in picks:
        m = objects[idx]
        odir = C.OUT / "objects" / f"obj_{idx:02d}"
        rdir = odir / "rvg"
        rdir.mkdir(exist_ok=True)
        gt = gts[m["gt_object_id"]]
        views = collect_views(gt, K, W, H, w2c_all, scene_full, verts, faces,
                              args.n_views)
        imgs = []
        for k, (fname, (u0, v0, u1, v1), mask) in enumerate(views):
            rgb = np.asarray(Image.open(C.IMAGES_DIR / fname).convert("RGB"))
            rgba = np.dstack([rgb[v0:v1, u0:u1],
                              (mask * 255).astype(np.uint8)])
            pil = Image.fromarray(rgba)
            pil.save(rdir / f"view_{k:02d}.png")
            imgs.append(pipe.preprocess_image(pil))
        print(f"[rv] obj_{idx:02d} {m['label']}: {len(imgs)} views")

        out, _, _ = pipe.run(image=imgs, seed=42,
                             formats=["mesh", "gaussian"],
                             preprocess_image=False)
        mesh = out["mesh"][0]
        v = mesh.vertices.cpu().numpy().astype(np.float64)
        f = mesh.faces.cpu().numpy()
        cols = None
        if getattr(mesh, "vertex_attrs", None) is not None \
                and mesh.vertex_attrs.shape[-1] >= 3:
            cols = (mesh.vertex_attrs[:, :3].clamp(0, 1).cpu().numpy()
                    * 255).astype(np.uint8)
        trimesh.Trimesh(v, f, vertex_colors=cols, process=False).export(
            rdir / "rvg_mesh.ply")
        try:
            out["gaussian"][0].save_ply(str(rdir / "rvg_gs.ply"), transform=None)
        except Exception as e:
            print(f"[rv] gaussian save failed: {e}")
        del out
        torch.cuda.empty_cache()

        # identical registration + eval as TRELLIS
        mesh_tm = trimesh.load(rdir / "rvg_mesh.ply", process=False)
        pts, _ = trimesh.sample.sample_surface(mesh_tm, 20000)
        pts = np.asarray(pts, dtype=np.float64)
        import open3d as o3d
        tgt = np.asarray(o3d.io.read_point_cloud(
            str(odir / "gt_points.ply")).points)
        T, chamfer, _, _ = align_object(pts, tgt[:: max(len(tgt) // 6000, 1)])
        ev = f1_eval(apply_T(T, pts), gt_submesh_points(gt))
        old = json.loads((odir / "aligned.json").read_text())
        rec = {"index": idx, "label": m["label"], "n_views": len(imgs),
               "rvg_f1_20": ev["f1@20mm"]["f1"],
               "rvg_f1_40": ev["f1@40mm"]["f1"],
               "rvg_chamfer_mm": chamfer * 1000,
               "trellis_f1_20": old["eval"]["f1@20mm"]["f1"],
               "trellis_f1_40": old["eval"]["f1@40mm"]["f1"]}
        C.save_json(rdir / "rvg_eval.json", rec)
        results.append(rec)
        print(f"[rv] obj_{idx:02d} {m['label']}: RVG F1@20 "
              f"{rec['rvg_f1_20']:.3f} vs TRELLIS {rec['trellis_f1_20']:.3f}"
              f"  (@40: {rec['rvg_f1_40']:.3f} vs {rec['trellis_f1_40']:.3f})")

    C.save_json(C.OUT / "rvg_comparison.json", results)
    mean = lambda k: float(np.mean([r[k] for r in results]))
    print(f"[rv] MEAN F1@20: RVG {mean('rvg_f1_20'):.3f} vs "
          f"TRELLIS {mean('trellis_f1_20'):.3f}  "
          f"(@40: {mean('rvg_f1_40'):.3f} vs {mean('trellis_f1_40'):.3f})")


if __name__ == "__main__":
    main()
