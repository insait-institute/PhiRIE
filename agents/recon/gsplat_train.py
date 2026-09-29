"""Train an Inria-format 3DGS splat for an emulated scene dir.

gsplat 1.5.3 rasterization + DefaultStrategy densification, initialized from
the VGGT fused cloud (init_points.ply). Reads images/K/w2c through the SAME
agents.core.common parsers the pipeline uses, and writes a ply whose fields
byte-for-byte match what common.load_gaussians parses (same writer layout as
agents/edit/inpaint_fill.py), plus train_report.json next to it.

GPU, mini-viewer env (run_gs). Usage:
    python -m agents.recon.gsplat_train --scene-dir <emulated scene dir> \
        --init-ply <scene dir>/init_points.ply \
        --out $SIMANY_SPLATS_ROOT/<scene>.ply [--iters 15000] \
        [--holdout-every 10]
"""
import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from agents.core import common as C

SH_DEGREE = 3          # matches the ScanNet++ reference splats the whole
                       # pipeline (and load_gaussians) was validated on
SH_STEP_EVERY = 1000   # classic 3DGS: unlock one SH band per 1k iters
ITERS = 15000
HOLDOUT_EVERY = 10     # internal diagnostic; initialization may use these frames
SSIM_WEIGHT = 0.2      # classic 3DGS loss mix: 0.8 L1 + 0.2 (1-SSIM)
INIT_OPACITY = 0.1
LR_MEANS = 1.6e-4      # * scene_scale, exp-decayed to 1% (3DGS defaults)
LR_SCALES, LR_QUATS, LR_OPAC, LR_SH0 = 5e-3, 1e-3, 5e-2, 2.5e-3
LR_SHN_DIV = 20.0      # rest bands train 20x slower than DC (3DGS default)
REFINE_STOP_FRAC = 0.5 # densify for the first half only, then pure refine
NEAR, FAR = 0.01, 100.0  # must match common.render_view for eval parity


def build_ssim(device):
    """11x11 gaussian SSIM on (3,H,W) pairs; local impl because the
    mini-viewer env ships neither fused_ssim nor pytorch_msssim."""
    import torch
    import torch.nn.functional as F

    sigma, ksize = 1.5, 11
    x = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    win = (g[:, None] @ g[None, :]).expand(3, 1, ksize, ksize).contiguous()
    C1, C2 = 0.01 ** 2, 0.03 ** 2

    def ssim(a, b):
        a, b = a[None], b[None]  # (1,3,H,W)
        mu_a = F.conv2d(a, win, padding=ksize // 2, groups=3)
        mu_b = F.conv2d(b, win, padding=ksize // 2, groups=3)
        var_a = F.conv2d(a * a, win, padding=ksize // 2, groups=3) - mu_a ** 2
        var_b = F.conv2d(b * b, win, padding=ksize // 2, groups=3) - mu_b ** 2
        cov = F.conv2d(a * b, win, padding=ksize // 2, groups=3) - mu_a * mu_b
        s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / \
            ((mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2))
        return s.mean()

    return ssim


def load_init(init_ply, device):
    """init_points.ply -> raw (pre-activation) gaussian parameters."""
    import torch
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    v = PlyData.read(str(init_ply))["vertex"]
    pts = np.stack([np.asarray(v[a], np.float32) for a in "xyz"], 1)
    rgb = np.stack([np.asarray(v[c], np.float32)
                    for c in ("red", "green", "blue")], 1) / 255.0
    n = len(pts)
    # scale init: mean distance to the 3 nearest neighbors (3DGS recipe);
    # clamped so duplicate points cannot produce log(0)
    d, _ = cKDTree(pts).query(pts, k=4)
    mean_d = np.clip(d[:, 1:].mean(1), 1e-4, None)
    C0 = 0.2820948
    k_rest = (SH_DEGREE + 1) ** 2 - 1

    t = lambda a: torch.tensor(np.asarray(a, np.float32), device=device)
    quats = np.zeros((n, 4), np.float32)
    quats[:, 0] = 1.0
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(t(pts)),
        "scales": torch.nn.Parameter(t(np.log(mean_d))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(t(quats)),
        "opacities": torch.nn.Parameter(
            t(np.full(n, math.log(INIT_OPACITY / (1 - INIT_OPACITY)),
                      np.float32))),
        "sh0": torch.nn.Parameter(t((rgb - 0.5) / C0)[:, None, :]),
        "shN": torch.nn.Parameter(
            torch.zeros(n, k_rest, 3, device=device))}).to(device)
    return params


def write_inria_ply(params, out_path):
    """Raw params -> Inria ply, same field layout inpaint_fill.py writes and
    common.load_gaussians reads (f_rest grouped channel-major)."""
    from plyfile import PlyData, PlyElement

    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    n = len(p["means"])
    sh = np.concatenate([p["sh0"], p["shN"]], axis=1)  # (N,K,3)
    k_rest = sh.shape[1] - 1
    names = (["x", "y", "z", "nx", "ny", "nz"] +
             [f"f_dc_{i}" for i in range(3)] +
             [f"f_rest_{i}" for i in range(3 * k_rest)] +
             ["opacity"] + [f"scale_{i}" for i in range(3)] +
             [f"rot_{i}" for i in range(4)])
    arr = np.zeros(n, dtype=[(nm, "f4") for nm in names])
    arr["x"], arr["y"], arr["z"] = p["means"].T
    for i in range(3):
        arr[f"f_dc_{i}"] = sh[:, 0, i]
    rest = sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)  # inria layout
    for i in range(3 * k_rest):
        arr[f"f_rest_{i}"] = rest[:, i]
    arr["opacity"] = p["opacities"].reshape(-1)          # raw logit
    for i in range(3):
        arr[f"scale_{i}"] = p["scales"][:, i]            # raw log
    q = p["quats"] / (np.linalg.norm(p["quats"], axis=1, keepdims=True)
                      + 1e-9)
    for i in range(4):
        arr[f"rot_{i}"] = q[:, i]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(out_path))


def main(argv=None):
    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-dir", required=True, type=Path)
    ap.add_argument("--init-ply", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--holdout-every", type=int, default=HOLDOUT_EVERY)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.iters < 1 or args.holdout_every < 2 or args.seed < 0:
        raise ValueError('invalid iteration, internal diagnostic split, or seed')
    report_paths = [args.out.with_name('train_report.json'),
                    args.out.with_name(args.out.stem + '_train_report.json')]
    if any(p.exists() or p.is_symlink() for p in [args.out, *report_paths]):
        raise ValueError('refusing to overwrite Gaussian/report outputs')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Retain the exclusive reservation on failure; no retry can mix artifacts.
    with args.out.with_suffix(args.out.suffix + '.lock').open('x') as lock:
        lock.write(str(os.getpid()) + '\n')
    t0 = time.time()
    device = "cuda"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    K, W, H, _ = C.load_intrinsics(
        args.scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json")
    w2c_map = C.load_colmap_w2c(
        args.scene_dir / "dslr" / "colmap" / "images.txt")
    img_dir = args.scene_dir / "dslr" / "resized_undistorted_images"
    items = sorted(w2c_map.items())
    views = []
    for nm, m in items:
        img = np.asarray(Image.open(img_dir / nm).convert("RGB"))
        assert img.shape[:2] == (H, W), f"{nm}: {img.shape} vs K for {W}x{H}"
        views.append((nm, m, img))  # uint8 on CPU; moved to GPU per iter
    hold = list(range(0, len(views), args.holdout_every))
    train_idx = [i for i in range(len(views)) if i not in set(hold)]
    if not train_idx or not hold:
        raise ValueError('training and internal diagnostic subsets must be nonempty')
    print(f"[gs] {len(train_idx)} train / {len(hold)} holdout views, "
          f"{W}x{H}")

    params = load_init(args.init_ply, device)
    cams = np.stack([np.linalg.inv(m)[:3, 3] for _, m, _ in views])
    scene_scale = 1.1 * float(
        np.linalg.norm(cams - cams.mean(0), axis=1).max())

    optimizers = {
        "means": torch.optim.Adam([params["means"]],
                                  lr=LR_MEANS * scene_scale, eps=1e-15),
        "scales": torch.optim.Adam([params["scales"]], lr=LR_SCALES,
                                   eps=1e-15),
        "quats": torch.optim.Adam([params["quats"]], lr=LR_QUATS, eps=1e-15),
        "opacities": torch.optim.Adam([params["opacities"]], lr=LR_OPAC,
                                      eps=1e-15),
        "sh0": torch.optim.Adam([params["sh0"]], lr=LR_SH0, eps=1e-15),
        "shN": torch.optim.Adam([params["shN"]], lr=LR_SH0 / LR_SHN_DIV,
                                eps=1e-15),
    }
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / args.iters))
    strategy = DefaultStrategy(
        refine_stop_iter=int(args.iters * REFINE_STOP_FRAC), verbose=False)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    ssim = build_ssim(device)

    Kt = torch.tensor(K, dtype=torch.float32, device=device)[None]
    rng = np.random.RandomState(args.seed)

    def render(view_i, sh_degree):
        _, m, img = views[view_i]
        viewmat = torch.tensor(m, dtype=torch.float32, device=device)[None]
        colors, alphas, info = rasterization(
            means=params["means"], quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]).reshape(-1),
            colors=torch.cat([params["sh0"], params["shN"]], dim=1),
            viewmats=viewmat, Ks=Kt, width=W, height=H,
            sh_degree=sh_degree, render_mode="RGB", packed=False,
            near_plane=NEAR, far_plane=FAR)
        gt = torch.tensor(img, dtype=torch.float32, device=device) / 255.0
        return colors[0], gt, info

    for step in range(args.iters):
        sh_deg = min(step // SH_STEP_EVERY, SH_DEGREE)
        vi = train_idx[int(rng.randint(len(train_idx)))]
        pred, gt, info = render(vi, sh_deg)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        l1 = (pred - gt).abs().mean()
        loss = (1 - SSIM_WEIGHT) * l1 + SSIM_WEIGHT * (
            1 - ssim(pred.permute(2, 0, 1).clamp(0, 1),
                     gt.permute(2, 0, 1)))
        loss.backward()
        strategy.step_post_backward(params, optimizers, state, step, info,
                                    packed=False)
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 500 == 0:
            print(f"[gs] iter {step}: loss {float(loss):.4f}, "
                  f"{len(params['means'])} gaussians")

    with torch.no_grad():
        psnrs = []
        for i in hold:
            pred, gt, _ = render(i, SH_DEGREE)
            psnrs.append(C.psnr(pred.clamp(0, 1).cpu().numpy(),
                                gt.cpu().numpy()))
    psnr_hold = float(np.mean(psnrs))

    if not all(torch.isfinite(p).all().item() for p in params.values()) or not math.isfinite(psnr_hold):
        raise ValueError('nonfinite Gaussian parameters or internal diagnostic PSNR')
    write_inria_ply(params, args.out)
    report = {"psnr_internal_diagnostic": psnr_hold,
              "psnr_internal_diagnostic_per_view": [float(p) for p in psnrs],
              "independent_heldout_evaluation": False,
              "diagnostic_caveat": "Every diagnostic RGB may contribute to initialization; official TEST evaluation is separate.",
              "seed": args.seed, "python": sys.executable,
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "gpu_name": torch.cuda.get_device_name(),
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
              "bitwise_determinism_claimed": False,
              "gradient_train_frames": [views[i][0] for i in train_idx],
              "internal_diagnostic_frames": [views[i][0] for i in hold],
              "n_gaussians": int(len(params["means"])),
              "iters": args.iters, "wall_s": time.time() - t0,
              "n_train": len(train_idx), "n_holdout": len(hold),
              "scene_scale": scene_scale}
    # contract-literal name next to the ply, plus a per-scene copy so
    # several scenes sharing one splats/ dir don't overwrite each other
    C.save_json(Path(args.out).with_name("train_report.json"), report)
    C.save_json(Path(args.out).with_name(
        Path(args.out).stem + "_train_report.json"), report)
    print(f"[gs] internal diagnostic PSNR {psnr_hold:.2f} dB (not independent), "
          f"{report['n_gaussians']} gaussians, "
          f"{report['wall_s'] / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
