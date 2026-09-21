"""Photoreal Gaussian rendering driven by a physics simulator's poses.

Consumes a pose log ({name: [pos, quat_wxyz]} per frame - MuJoCo xpos/xquat,
Isaac rigid-body states, or PyBullet link poses all fit) and renders the
scene as: background splat (inpainted clean_background.ply when available)
+ per-object canonical asset gaussians transformed to each frame's pose.
This is the "Gaussian renderer for MuJoCo/Isaac": the simulator does
physics, gsplat does the pixels.

Run under the mini-viewer env on a GPU node.
Usage: gsplat_sim_render.py --poses P.json --out video.mp4 [--camera FRAME]
"""
import argparse
import json
import time

import numpy as np

from agents.core import common as C


def main():
    import imageio.v2 as imageio
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default="",
                    help="DSLR frame name for the camera; default = best view")
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    K, W, H, _ = C.load_intrinsics()
    w2c_all = C.load_colmap_w2c()
    log = json.loads(open(args.poses).read())

    clean = C.OUT / "inpaint" / "clean_background.ply"
    bg_src = clean if clean.exists() else C.SPLAT_PLY
    bg = C.load_gaussians(bg_src)
    print(f"[gr] background: {bg_src.name} ({bg['means'].shape[0]} gaussians)")

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    canon, scales = {}, {}
    for m in objects:
        name = f"obj_{m['index']:02d}"
        odir = C.OUT / "objects" / name
        al = json.loads((odir / "aligned.json").read_text())
        if al.get("rejected") or not (odir / "trellis_gs.ply").exists():
            continue
        canon[name] = C.pad_sh(C.load_gaussians(odir / "trellis_gs.ply"),
                               bg["sh_degree"])
        scales[name] = C.decompose_similarity(np.array(al["T"]))[0]

    if args.camera:
        w2c = w2c_all[args.camera]
    else:  # camera that sees the most objects (reuse centroids)
        cents = np.array([m["centroid"] for m in objects])
        best, w2c = -1, None
        for name, cand in sorted(w2c_all.items()):
            pc = cents @ cand[:3, :3].T + cand[:3, 3]
            z = np.clip(pc[:, 2], 1e-6, None)
            u = pc[:, 0] / z * K[0, 0] + K[0, 2]
            v = pc[:, 1] / z * K[1, 1] + K[1, 2]
            n = int(((z > 0.3) & (u > 0) & (u < W) & (v > 0) & (v < H)).sum())
            if n > best:
                best, w2c = n, cand

    def pose_T(s, pos, quat_wxyz):
        T = np.eye(4)
        T[:3, :3] = s * C.quat_to_rot_wxyz(quat_wxyz)
        T[:3, 3] = pos
        return T

    cams = log.get("cams")  # optional per-frame w2c (scripted camera path)
    writer = imageio.get_writer(args.out, fps=int(log["fps"]),
                                macro_block_size=None, quality=8)
    t0, n = time.time(), 0
    for fi, fr in enumerate(log["frames"]):
        if cams is not None:
            w2c = np.array(cams[fi])
        parts = [bg]
        for name, (pos, quat) in fr.items():
            if name in canon:
                parts.append(C.transform_gaussians(
                    canon[name], pose_T(scales[name], pos, quat)))
        gs = C.cat_gaussians(parts)
        rgb, _, _ = C.render_view(gs, w2c, K, W, H, scale=args.scale)
        writer.append_data((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        del gs
        n += 1
        if n % 30 == 0:
            torch.cuda.empty_cache()
            print(f"[gr] frame {n}/{len(log['frames'])} "
                  f"({n / (time.time() - t0):.1f} fps)", flush=True)
    writer.close()
    print(f"[gr] wrote {args.out}: {n} frames, "
          f"avg {n / (time.time() - t0):.1f} fps at scale {args.scale}")


if __name__ == "__main__":
    main()
