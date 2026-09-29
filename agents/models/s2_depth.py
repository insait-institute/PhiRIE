"""Stage 2: metric depth on the representative frame (paper's V_im2depth = DA3).

DA3METRIC-LARGE outputs canonical depth for a focal-300px camera:
metric = canonical * (f_processed / 300). We then compute a single robust
scale s* against the scan-mesh z-depth (stand-in for the paper's Umeyama
background-alignment "rigid bridge"; without it, DA3's global scale error
would misplace objects relative to the background collision mesh).

Writes OUT/depth/depth.npz {depth_raw, depth_corr (full res), s_star, stats}.
"""
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np

from agents.core import common as C


def main():
    K, W, H, _ = C.load_intrinsics()
    rep = json.loads((C.OUT / "frame" / "rep_frame.json").read_text())
    img_path = str(C.OUT / "frame" / rep["frame"])
    w2c = np.array(rep["w2c"])

    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
    model = model.to("cuda").eval()
    pred = model.inference([img_path], process_res=1008,
                           process_res_method="upper_bound_resize")
    canonical = pred.depth[0].astype(np.float64)  # (Hp, Wp)
    Hp, Wp = canonical.shape
    s_res = Wp / W
    f_proc = 0.5 * (K[0, 0] + K[1, 1]) * s_res
    metric = canonical * (f_proc / 300.0)
    depth_raw = cv2.resize(metric, (W, H), interpolation=cv2.INTER_LINEAR)

    scene = C.make_raycast_scene()
    dmesh = C.mesh_zdepth(scene, K, w2c, W, H, stride=4)
    draw4 = depth_raw[::4, ::4][: dmesh.shape[0], : dmesh.shape[1]]
    valid = np.isfinite(dmesh) & (dmesh > 0.3) & (dmesh < 8.0) & (draw4 > 0.1)
    ratio = dmesh[valid] / draw4[valid]
    s_star = float(np.median(ratio))
    depth_corr = depth_raw * s_star

    rel_raw = float(np.median(np.abs(draw4[valid] - dmesh[valid]) / dmesh[valid]))
    rel_corr = float(np.median(
        np.abs(draw4[valid] * s_star - dmesh[valid]) / dmesh[valid]))
    stats = {"s_star": s_star, "median_rel_err_raw": rel_raw,
             "median_rel_err_corrected": rel_corr, "n_valid": int(valid.sum())}
    print(f"[s2] DA3 vs mesh: raw rel err {rel_raw:.3f}, "
          f"scale s*={s_star:.3f}, corrected rel err {rel_corr:.3f}")

    out = C.OUT / "depth"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "depth.npz",
                        depth_raw=depth_raw.astype(np.float32),
                        depth_corr=depth_corr.astype(np.float32),
                        s_star=s_star)
    C.save_json(out / "stats.json", stats)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (d, title) in zip(axes, [
            (depth_corr, "DA3 metric (scale-corrected)"),
            (np.where(np.isfinite(dmesh), dmesh, np.nan), "mesh z-depth"),
            (np.abs(cv2.resize(np.where(np.isfinite(dmesh), dmesh, np.nan),
                               (W, H), interpolation=cv2.INTER_NEAREST)
                    - depth_corr), "abs err")]):
        im = ax.imshow(d, cmap="turbo", vmin=0, vmax=6 if "err" not in title else 0.5)
        ax.set_title(title); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(out / "depth_vis.png", dpi=110)


if __name__ == "__main__":
    main()
