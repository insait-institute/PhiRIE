"""Stage 2 (SHARP variant): render the SHARP splat from the ORIGINAL photo's
own camera pose (identity, per predict_image()'s unproject_gaussians call)
to recover an RGB+depth+alpha view exactly matching the input photo.

Mini-viewer env only (gsplat rasterization); reads/writes are otherwise
plain common.OUT-relative -- no ScanNet++ dependency.

Usage: sharp_render_depth.py --ply /path/to/sharp_output.ply
(needs SIMANY_OUT set; no --out-dir, everything goes to $SIMANY_OUT/depth/)
"""
import argparse

import numpy as np

from agents.core import common as C
from agents.single_image.sharp_ply_meta import read_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="SHARP-produced gaussian .ply")
    args = ap.parse_args()

    f_px, W, H = read_camera(args.ply)
    K = np.array([[f_px, 0.0, W / 2.0],
                  [0.0, f_px, H / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    print(f"[sd] camera from ply: f_px={f_px} W={W} H={H}")

    gs = C.load_gaussians(ply_path=args.ply, device="cuda")
    print(f"[sd] loaded {gs['means'].shape[0]} gaussians, sh_degree={gs['sh_degree']}")

    # The splat's own coordinate origin IS the input photo's camera
    # (predict_image() unprojects with extrinsics=eye(4)).
    w2c = np.eye(4)

    rgb, depth, alpha = C.render_view(gs, w2c, K, W, H, render_mode="RGB+ED")

    valid = alpha > 0.5
    if valid.sum() == 0:
        raise SystemExit("[sd] FATAL: no pixels with alpha>0.5 -- degenerate render")
    d_valid = depth[valid]
    dmin, dmed, dmax = float(d_valid.min()), float(np.median(d_valid)), float(d_valid.max())
    print(f"[sd] depth[alpha>0.5]: min={dmin:.3f} median={dmed:.3f} max={dmax:.3f} "
          f"(n={int(valid.sum())}/{valid.size})")
    print(f"[sd] alpha: min={alpha.min():.3f} mean={alpha.mean():.3f} max={alpha.max():.3f}")

    # Sanity gate: an office-desk photo should have plausible metric depth.
    if not np.isfinite(dmin) or not np.isfinite(dmax):
        raise SystemExit("[sd] FATAL: non-finite depth in valid region")
    if dmax > 50.0 or dmed < 0.05:
        raise SystemExit(
            f"[sd] FATAL: depth stats implausible (min={dmin:.3f} med={dmed:.3f} "
            f"max={dmax:.3f}) -- likely a K/w2c convention mismatch, stop and debug")
    if not (0.1 <= dmed <= 10.0):
        print(f"[sd] WARNING: median depth {dmed:.3f}m outside the expected "
              "~0.3-4m office-desk range -- inspect depth_vis.png before proceeding")

    out = C.OUT / "depth"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "depth.npz",
        depth_corr=depth.astype(np.float32),
        alpha=alpha.astype(np.float32),
        K=K, W=W, H=H, f_px=f_px)
    print(f"[sd] wrote {out / 'depth.npz'}")

    # quick visual sanity artifact (no matplotlib dependency assumed present)
    try:
        from PIL import Image
        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_u8).save(out / "render_rgb.png")
        d_vis = np.clip(depth, 0, max(dmax, 1e-6)) / max(dmax, 1e-6)
        d_vis = (d_vis * 255).astype(np.uint8)
        d_vis[~valid] = 0
        Image.fromarray(d_vis).save(out / "render_depth_vis.png")
        print(f"[sd] wrote render_rgb.png / render_depth_vis.png for visual check")
    except Exception as e:
        print(f"[sd] (non-fatal) could not write visual sanity images: {e}")


if __name__ == "__main__":
    main()
