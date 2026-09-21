"""Metricize a VGGT reconstruction: metric scale + z-up + floor at z=0.

Brings models/vggt_scene.py output into the pipeline's world convention
(metric, z-up, floor near z=0 - the frame every downstream stage assumes):

1. Scale: DA3METRIC-LARGE (same model/loading as models/s2_depth) on evenly
   spaced frames; scale = median over pixels+frames of metric/vggt depth
   ratios after per-frame percentile trimming.
2. Up-axis: the mean camera-image-up direction is the prior (3.7 deg from
   GT up on the a29cccc784 pilot); the floor is the lowest strong height
   mode along it, its slab refined by trimmed LSQ (see find_floor - plane
   RANSAC is deliberately NOT used; walls/tables out-vote the floor).
   Plane normal -> +z, floor -> 0. Aborts if the median camera ends up
   outside 0.3-3.5 m above the floor or >10% of the cloud below it.

Same npz schema in and out (plus metric_scale / T_align metadata keys).

GPU, main .venv. Usage:
    python -m agents.recon.metricize --recon recon.npz --images-dir D \
        --out recon_metric.npz
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np

N_CAL_FRAMES = 7        # >=5 per contract; odd so the median is a real frame
DA3_PROC_RES = 1008     # matches models/s2_depth's validated recipe
DA3_CANON_FOCAL = 300.0 # DA3METRIC canonical camera focal (px)
TRIM_LO, TRIM_HI = 20, 80  # per-frame ratio percentile trim: kills sky /
                           # reflections / VGGT edge bleed before the median
RANSAC_POINTS = 80_000  # floor-fit subsample; more adds nothing but time
# The camera-up prior was 3.7 deg from GT up on the a29cccc784 pilot; 30 deg
# leaves room for tilted phone captures while still excluding every wall.
UP_CONE_DEG = 30.0
FLOOR_MIN_FRAC = 0.01   # a visible floor is >=1% of the fused cloud
# Sparse backends (COLMAP tracks) leave too few init points for gsplat's
# densifier to grow from quickly; below this the init cloud is rebuilt from
# the DA3 metric depths (already computed for scaling, so it costs nothing).
DENSE_MIN_POINTS = 800_000
DENSE_KEEP_FRAC = 0.35  # per-frame pixel thinning before backprojection
DENSE_MAX_POINTS = 2_000_000


def da3_metric_depth(model, img_path, K, W):
    """One frame -> metric depth at DA3's processing resolution.

    DA3METRIC outputs canonical depth for a focal-300px camera:
    metric = canonical * (f_processed / 300)  (see models/s2_depth)."""
    pred = model.inference([str(img_path)], process_res=DA3_PROC_RES,
                           process_res_method="upper_bound_resize")
    canonical = pred.depth[0].astype(np.float64)
    s_res = canonical.shape[1] / W
    f_proc = 0.5 * (K[0, 0] + K[1, 1]) * s_res
    return canonical * (f_proc / DA3_CANON_FOCAL)


def estimate_scale(rec, images_dir, *, checkpoint="depth-anything/DA3METRIC-LARGE"):
    import cv2
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(str(checkpoint))
    model = model.to("cuda").eval()

    names = [str(n) for n in rec["names"]]
    K = rec["K"]
    W = int(rec["frame_wh"][0])
    depth = rec["depth"].astype(np.float32)
    hd, wd = depth.shape[1:]
    idx = np.unique(np.linspace(0, len(names) - 1, N_CAL_FRAMES).astype(int))

    ratios = []
    da3_cache = {}
    for i in idx:
        metric = da3_metric_depth(model, Path(images_dir) / names[i], K, W)
        da3_cache[int(i)] = metric.astype(np.float32)
        metric = cv2.resize(metric, (wd, hd), interpolation=cv2.INTER_LINEAR)
        dv = depth[i].astype(np.float64)
        valid = np.isfinite(metric) & np.isfinite(dv) & (metric > 0.1) & \
            (dv > 1e-4)
        # 60, not 500: the COLMAP backend provides SPARSE depth (a few
        # hundred to a few thousand track projections per frame)
        if valid.sum() < 60:
            continue
        r = metric[valid] / dv[valid]
        lo, hi = np.percentile(r, [TRIM_LO, TRIM_HI])
        r = r[(r >= lo) & (r <= hi)]
        ratios.append(r)
        print(f"[metricize] {names[i]}: frame-median ratio {np.median(r):.4f}"
              f" ({len(r)} px)")
    if not ratios:
        raise SystemExit("[metricize] no valid depth overlap for scaling")
    s = float(np.median(np.concatenate(ratios)))
    spread = float(np.std([np.median(r) for r in ratios]))
    print(f"[metricize] scale s={s:.4f} (per-frame median spread {spread:.4f})")
    return s, da3_cache


def find_floor(pts, up_prior):
    """Floor plane -> (unit up normal, offset d with n.x + d = 0, slab pts).

    Deterministic histogram + slab-refine, NOT plane-RANSAC. Measured on the
    a29cccc784 pilot cloud: the floor is only ~3% of the fused points while
    table/ceiling planes carry ~20x its support, so any largest-plane or
    lowest-plane RANSAC rule lands on a wall (85.9 deg off), a tilted
    pseudo-plane (19 deg), or a table. But the camera-image-up prior is
    ~4 deg accurate, and the floor slab itself is 1 cm RMS planar - so:

      1. height = pts . up; coarse 10 cm histogram of the low region; floor
         level = lowest bin holding >= FLOOR_MIN_FRAC of the cloud (coarse
         bins absorb the height smear a tilted prior causes: 4 deg over an
         8 m room spreads the floor +-0.3 m).
      2. trimmed LSQ plane on the slab around that level -> refined normal
         (recovers the floor's own 1-deg-accurate orientation).
      3. repeat once with the refined normal and fine bins/slab.

    Aborts if the final slab is thick (not a floor) or the refined normal
    strays far from the prior (fit grabbed something else).
    """
    rng = np.random.RandomState(0)
    sub = pts
    if len(sub) > RANSAC_POINTS:
        sub = sub[rng.choice(len(sub), RANSAC_POINTS, replace=False)]

    def slab_fit(up, bin_m, slab_m):
        h = sub @ up
        lo, hi = np.percentile(h, [0.5, 99.5])
        nbins = max(4, int((hi - lo) / bin_m))
        hist, edges = np.histogram(h, bins=nbins, range=(lo, hi))
        need = int(FLOOR_MIN_FRAC * len(sub))
        idx = next((i for i in range(nbins) if hist[i] >= need), None)
        if idx is None:
            raise SystemExit(
                f"[metricize] no height bin holds {FLOOR_MIN_FRAC:.0%} of the "
                f"cloud - floor not visible enough in the capture")
        level = (edges[idx] + edges[idx + 1]) / 2
        slab = sub[np.abs(h - level) < slab_m]
        # trimmed LSQ: one pass to drop wall-bottoms/objects in the slab
        for _ in range(2):
            c = slab.mean(0)
            _, _, Vt = np.linalg.svd(slab - c, full_matrices=False)
            n = Vt[2]
            if n @ up < 0:
                n = -n
            dist = (slab - c) @ n
            keep = np.abs(dist) < max(0.02, 2.0 * dist.std())
            if keep.all():
                break
            slab = slab[keep]
        return n, -n @ slab.mean(0), slab

    n, d, slab = slab_fit(up_prior, bin_m=0.10, slab_m=0.07)
    n, d, slab = slab_fit(n, bin_m=0.03, slab_m=0.04)

    thick = float(np.abs(slab @ n + d).std())
    dev = np.degrees(np.arccos(np.clip(n @ up_prior, -1, 1)))
    print(f"[metricize] floor slab: {len(slab)}/{len(sub)} pts, "
          f"normal {np.round(n, 3)}, {dev:.1f} deg from camera-up prior, "
          f"thickness rms {thick * 100:.1f} cm")
    if thick > 0.03:
        raise SystemExit("[metricize] ABORT: floor slab rms "
                         f"{thick * 100:.1f} cm - not a planar floor")
    if dev > UP_CONE_DEG:
        raise SystemExit(f"[metricize] ABORT: refined floor normal {dev:.1f} "
                         f"deg from camera-up prior - fit grabbed a non-floor "
                         f"structure")
    return n, d, slab


def rot_between(a, b):
    """Minimal rotation taking unit vector a onto unit vector b (Rodrigues)."""
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recon", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", default="depth-anything/DA3METRIC-LARGE")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f'refusing to overwrite metric reconstruction: {args.out}')

    rec = dict(np.load(args.recon, allow_pickle=False))
    s, da3_cache = estimate_scale(rec, args.images_dir, checkpoint=args.checkpoint)

    w2c = rec["w2c"].astype(np.float64)
    w2c[:, :3, 3] *= s                           # camera centers scale with s
    pts = rec["points"].astype(np.float64) * s
    depth = rec["depth"].astype(np.float32) * s

    c2w_R = np.stack([np.linalg.inv(m)[:3, :3] for m in w2c])
    up_prior = -c2w_R[:, :, 1].mean(0)           # OpenCV cam +y = image down
    up_prior /= np.linalg.norm(up_prior)

    n, d, inl = find_floor(pts, up_prior)

    Ra = rot_between(n, np.array([0.0, 0.0, 1.0]))
    z_floor = float(np.median((inl @ Ra.T)[:, 2]))
    ta = np.array([0.0, 0.0, -z_floor])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Ra, ta

    pts_new = pts @ Ra.T + ta
    w2c_new = np.stack([m @ np.linalg.inv(T) for m in w2c])

    cams = np.stack([np.linalg.inv(m)[:3, 3] for m in w2c_new])
    below = float((pts_new[:, 2] < -0.2).mean())
    z_med = float(np.median(cams[:, 2]))
    print(f"[metricize] camera heights z: {cams[:, 2].min():.2f}.."
          f"{cams[:, 2].max():.2f} m (median {z_med:.2f}); "
          f"cloud below z=-0.2: {below:.3f}")
    # handheld/DSLR captures sit 0.3-3.5 m above the floor, and almost no
    # scene mass can be below it - either failing means the plane we chose
    # was not the floor, and guessing further would poison every downstream
    # metric silently
    if not (0.3 <= z_med <= 3.5) or below > 0.10:
        raise SystemExit("[metricize] ABORT: floor sanity failed "
                         f"(median camera z {z_med:.2f} m, cloud-below-floor "
                         f"{below:.2%}) - chosen plane is not the floor.")

    pts_out, rgb_out = pts_new, rec["points_rgb"]
    if len(pts_out) < DENSE_MIN_POINTS:
        # sparse backend (COLMAP): densify the splat-init cloud from the DA3
        # METRIC depths already computed for scaling - they need no further
        # alignment, only the world transform
        import cv2
        W, H = int(rec["frame_wh"][0]), int(rec["frame_wh"][1])
        names = [str(x) for x in rec["names"]]
        dense_p, dense_c = [pts_new.astype(np.float32)], [rec["points_rgb"]]
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        for i, dm in da3_cache.items():
            z = cv2.resize(dm, (W, H), interpolation=cv2.INTER_LINEAR)
            m = np.isfinite(z) & (z > 0.2) & (z < 12.0)
            m &= (np.random.RandomState(i).rand(H, W) <
                  DENSE_KEEP_FRAC)                      # thin per frame
            zs = z[m]
            x = (uu[m] - rec["K"][0, 2]) / rec["K"][0, 0] * zs
            y = (vv[m] - rec["K"][1, 2]) / rec["K"][1, 1] * zs
            c2w = np.linalg.inv(w2c_new[i])
            dense_p.append((np.stack([x, y, zs], 1) @ c2w[:3, :3].T
                            + c2w[:3, 3]).astype(np.float32))
            bgr = cv2.imread(str(Path(args.images_dir) / names[i]))
            dense_c.append(bgr[..., ::-1][m])
        pts_out = np.concatenate(dense_p)
        rgb_out = np.concatenate(dense_c)
        if len(pts_out) > DENSE_MAX_POINTS:
            keep = np.random.RandomState(0).choice(
                len(pts_out), DENSE_MAX_POINTS, replace=False)
            pts_out, rgb_out = pts_out[keep], rgb_out[keep]
        print(f"[metricize] densified init cloud "
              f"{len(pts_new)} -> {len(pts_out)} pts from "
              f"{len(da3_cache)} DA3 metric depths")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, w2c=w2c_new, K=rec["K"], K_depth=rec["K_depth"],
        names=rec["names"], depth=depth.astype(np.float16),
        points=pts_out.astype(np.float32), points_rgb=rgb_out,
        frame_wh=rec["frame_wh"], metric_scale=np.float64(s), T_align=T)
    print(f"[metricize] wrote {args.out}")


if __name__ == "__main__":
    main()
