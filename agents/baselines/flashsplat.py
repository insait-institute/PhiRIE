"""FlashSplat baseline (ECCV 2024) for gaussian-selection object removal.

Shen, Yang, Wang: "FlashSplat: 2D to 3D Gaussian Splatting Segmentation
Solved Optimally" (arXiv 2409.08270, github.com/florinshen/FlashSplat,
cloned in third_party/FlashSplat). Their method: for every view, accumulate
each gaussian's alpha-blending weight (alpha * T at each pixel during
front-to-back compositing) separately for pixels inside vs outside the 2D
object mask -> counts[2, N]; the globally-optimal binary label assignment of
the resulting linear objective is a per-gaussian argmax with an optional
background bias ("slackness"; their object-removal script
edit_object_inpaint.py hardcodes -0.4 to dilate the removal set).

Why not their rasterizer: the counts come from a custom CUDA fork
(submodules/flashsplat-rasterization, forward.cu:
`atomicAdd(&used_count[obj_id * P + gid], alpha * T)`). It has no prebuilt
wheels (PyPI: nothing for flashsplat-rasterization / diff-gaussian-
rasterization) and needs a source CUDA build - blocked here (system gcc 14,
no nvcc, no g++ <= 13 on GPU nodes).

Equivalent reimplementation on gsplat (this file): rendering is LINEAR in
per-gaussian colors, render(pix) = sum_g w_{g,pix} c_g + T_final bg. So with
a dummy 2-channel color tensor c (requires_grad) and

    L = sum(render[..., 0] * mask) + sum(render[..., 1] * (1 - mask)),

autograd gives dL/dc_g = (sum_{pix in mask} w_{g,pix},
sum_{pix not in mask} w_{g,pix}) - exactly FlashSplat's used_count rows,
computed by gsplat's rasterizer. Verified to 1e-14 against direct
accumulation in a float64 reference renderer (run `--selftest`, CPU-only).

Data contract (ours): scene splat via common.load_gaussians (Inria ply),
per-object full-frame binary masks outputs/<scene>_factory/inpaint/obj_XX/
mask_<k>.png with views.json ({frame, w2c} OpenCV, world = mesh frame),
PINHOLE intrinsics via common.load_intrinsics. Outputs (this script):
  inpaint/obj_XX/flashsplat_removal_idx.npy    (sorted int64, our format)
  inpaint/flashsplat_removal_union_idx.npy
  inpaint/flashsplat_report.json               (timings + agreement vs ours)

Run on a GPU node (login node: only --selftest):
  srun --partition=debug --gpus=a6000:1 --mem=48G --time=00:30:00 \
    bash -c '\
      SIMANY_SCENE=c50d2d1d42 \
      SIMANY_OUT=${SIMANY_ROOT}/outputs/c50d2d1d42_factory \
      .venv/bin/python -m agents.baselines.flashsplat --slackness -0.4'
"""
import argparse
import json
import time

import numpy as np

from agents.core import common as C


# ------------------------------------------------- FlashSplat label solver --

def flashsplat_assign(counts, slackness=0.0):
    """Faithful port of FlashSplat's multi_instance_opt (objremoval.py) for
    the binary (obj_num=1) case used by their object-removal pipeline.

    counts: torch [2, N], accumulated alpha*T weights summed over views
            (row 0 = pixels outside the mask, row 1 = inside).
    slackness: background bias; <0 dilates the object set (their removal
            uses -0.4), >0 shrinks it (their segmentation uses 0.4..0.8).
    Returns bool tensor [N]: True = labeled object (remove).
    """
    import torch

    # FlashSplat assumes scene-covering views where every gaussian gets
    # background evidence; with a few object-centric views, gaussians with
    # ZERO evidence in both channels must default to background, else a
    # negative slackness assigns the whole room to the object
    no_evidence = counts.abs().sum(dim=0) == 0
    counts = torch.nn.functional.normalize(counts, dim=0)
    total = counts.sum(dim=0)
    obj = counts[1]
    pair = torch.stack([total - obj, obj], dim=0)
    if slackness != 0:
        pair = torch.nn.functional.normalize(pair, dim=0)
        pair[0, :] += slackness
    label = pair.max(dim=0)[1].bool()
    label[no_evidence] = False
    return label


# ------------------------------------------- per-view counts (gsplat, GPU) --

def view_counts(gs, w2c, K, width, height, mask, scale=1.0):
    """used_count[2, N] for one view via the gradient identity (see module
    docstring): gsplat render of a dummy 2-channel color, backward of the
    masked sums. mask: bool [H, W] at full intrinsics resolution."""
    import torch
    from gsplat import rasterization

    device = gs["means"].device
    K = np.asarray(K, dtype=np.float64).copy()
    if scale != 1.0:
        K[:2] *= scale
        width, height = int(round(width * scale)), int(round(height * scale))
        m = torch.from_numpy(np.ascontiguousarray(mask)).to(device)[None, None]
        m = torch.nn.functional.interpolate(m.float(), (height, width),
                                            mode="nearest")[0, 0] > 0.5
    else:
        m = torch.from_numpy(np.ascontiguousarray(mask)).to(device)
    m = m.float()

    N = gs["means"].shape[0]
    colors = torch.zeros(N, 2, device=device, requires_grad=True)
    viewmat = torch.from_numpy(np.asarray(w2c)).float().to(device)[None]
    Kt = torch.from_numpy(K).float().to(device)[None]
    with torch.enable_grad():
        img, _, _ = rasterization(
            means=gs["means"], quats=gs["quats"], scales=gs["scales"],
            opacities=gs["opacities"], colors=colors,
            viewmats=viewmat, Ks=Kt, width=width, height=height,
            sh_degree=None, render_mode="RGB", packed=False,
            near_plane=0.01, far_plane=100.0)
        loss = (img[0, ..., 0] * m).sum() + (img[0, ..., 1] * (1.0 - m)).sum()
        (grad,) = torch.autograd.grad(loss, colors)
    return torch.stack([grad[:, 1], grad[:, 0]], dim=0)  # [out, in]


def solve_object(gs, odir, K, width, height, slackness, scale=1.0):
    """Accumulate counts over the object's views and solve. Returns
    (removal_idx int64 sorted, stats dict) or None if <1 usable mask."""
    import torch
    from PIL import Image

    views = json.loads((odir / "views.json").read_text())
    pairs = []
    for k, vw in enumerate(views):
        mp = odir / f"mask_{k}.png"
        if mp.exists():
            pairs.append((np.asarray(Image.open(mp)) > 127,
                          np.asarray(vw["w2c"], dtype=np.float64)))
    if not pairs:
        return None

    t0 = time.time()
    counts = torch.zeros(2, gs["means"].shape[0], device=gs["means"].device)
    for mask, w2c in pairs:
        counts += view_counts(gs, w2c, K, width, height, mask, scale)
    t_counts = time.time() - t0

    t0 = time.time()
    label = flashsplat_assign(counts, slackness)
    t_solve = time.time() - t0

    idx = torch.nonzero(label).flatten().cpu().numpy().astype(np.int64)
    idx.sort()
    stats = {"n_views": len(pairs), "n_selected": int(idx.size),
             "t_counts_s": round(t_counts, 3), "t_solve_s": round(t_solve, 4)}
    return idx, stats


def agreement(pred, ref):
    """IoU / precision / recall of index sets (ref = our mask-vote removal;
    a comparison reference, not ground truth)."""
    p, r = set(pred.tolist()), set(ref.tolist())
    inter = len(p & r)
    return {"iou": round(inter / max(len(p | r), 1), 4),
            "precision": round(inter / max(len(p), 1), 4),
            "recall": round(inter / max(len(r), 1), 4),
            "n_pred": len(p), "n_ref": len(r)}


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slackness", type=float, default=-0.4,
                    help="background bias; FlashSplat removal default -0.4")
    ap.add_argument("--objects", type=int, nargs="*", default=None,
                    help="subset of object indices (default: all)")
    ap.add_argument("--render-scale", type=float, default=1.0,
                    help="render at scaled resolution (masks resized nearest)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU-only synthetic validation, no scene data needed")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    K, W, H, _ = C.load_intrinsics()
    print(f"[fs] scene {C.SCENE_ID}: loading splat {C.SPLAT_PLY}")
    gs = C.load_gaussians(C.SPLAT_PLY, device=args.device)
    N = gs["means"].shape[0]
    print(f"[fs] {N} gaussians, slackness {args.slackness}, "
          f"scale {args.render_scale}")

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    report = {"scene": C.SCENE_ID, "n_gaussians": N,
              "slackness": args.slackness, "render_scale": args.render_scale,
              "objects": {}}
    union = []
    t_total = time.time()
    for m in objects:
        oi = m["index"]
        if args.objects is not None and oi not in args.objects:
            continue
        odir = C.OUT / "inpaint" / f"obj_{oi:02d}"
        if not (odir / "views.json").exists():
            continue
        out = solve_object(gs, odir, K, W, H, args.slackness,
                           args.render_scale)
        if out is None:
            continue
        idx, stats = out
        np.save(odir / "flashsplat_removal_idx.npy", idx)
        union.append(idx)
        ref_p = odir / "removal_idx.npy"
        if ref_p.exists():
            stats["vs_ours"] = agreement(idx, np.load(ref_p))
        report["objects"][oi] = {"label": m.get("label"), **stats}
        agr = stats.get("vs_ours", {})
        print(f"[fs] obj_{oi:02d} ({m.get('label')}): {stats['n_selected']} "
              f"gaussians from {stats['n_views']} views in "
              f"{stats['t_counts_s'] + stats['t_solve_s']:.2f}s"
              + (f", IoU vs ours {agr['iou']:.3f}" if agr else ""))

    if union:
        u = np.unique(np.concatenate(union))
        np.save(C.OUT / "inpaint" / "flashsplat_removal_union_idx.npy", u)
        # reference = the pipeline's FINAL removal set (fill_gaussians.npz
        # includes the multi-view mask-vote gaussians); removal_union_idx.npy
        # is a stale pre-vote snapshot missing exactly the transparency set.
        fg_p = C.OUT / "inpaint" / "fill_gaussians.npz"
        ref_p = C.OUT / "inpaint" / "removal_union_idx.npy"
        report["union"] = {"n_selected": int(u.size)}
        if fg_p.exists():
            ref = np.load(fg_p)["removal_idx"]
            report["union"]["vs_ours"] = agreement(u, ref)
            report["union"]["ref_source"] = "fill_gaussians.npz"
        elif ref_p.exists():
            report["union"]["vs_ours"] = agreement(u, np.load(ref_p))
            report["union"]["ref_source"] = "removal_union_idx.npy (pre-vote)"
        print(f"[fs] union: {u.size} gaussians"
              + (f", IoU vs ours {report['union']['vs_ours']['iou']:.3f}"
                 if "vs_ours" in report["union"] else ""))
    report["t_total_s"] = round(time.time() - t_total, 2)
    C.save_json(C.OUT / "inpaint" / "flashsplat_report.json", report)
    print(f"[fs] done in {report['t_total_s']}s -> "
          f"{C.OUT / 'inpaint' / 'flashsplat_report.json'}")


# -------------------------------------------- CPU self-test (no GPU/scene) --

def selftest():
    """Validate the two claims this baseline rests on, without a GPU:
    1. gradient identity: backward of the masked render sums == FlashSplat's
       used_count (direct alpha*T accumulation), on a float64 reference
       renderer with Inria/gsplat compositing rules;
    2. the solver port selects the right gaussians and slackness behaves as
       the paper's background bias (negative dilates, positive shrinks)."""
    import torch

    torch.set_default_dtype(torch.float64)

    def quat_to_rotmat(q):
        w, x, y, z = q.unbind(-1)
        return torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ], dim=-1).reshape(-1, 3, 3)

    def render_counts(means, quats, scales, opac, colors, K, w2c, W, H, mask):
        """Naive front-to-back compositing (EWA projection, +0.3 dilation,
        alpha clamp 0.99, alpha >= 1/255, stop at T < 1e-4)."""
        R, t = w2c[:3, :3], w2c[:3, 3]
        pc = means @ R.T + t
        z = pc[:, 2]
        Rq = quat_to_rotmat(quats)
        S = torch.diag_embed(scales)
        cov_cam = R @ (Rq @ S @ S @ Rq.transpose(1, 2)) @ R.T
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x, y = pc[:, 0], pc[:, 1]
        J = torch.zeros(len(z), 2, 3)
        J[:, 0, 0], J[:, 0, 2] = fx / z, -fx * x / z ** 2
        J[:, 1, 1], J[:, 1, 2] = fy / z, -fy * y / z ** 2
        cov2d = J @ cov_cam @ J.transpose(1, 2)
        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3
        cinv = torch.linalg.inv(cov2d)
        uv = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=1)
        order = torch.argsort(z)
        img = torch.zeros(H, W, colors.shape[1])
        counts = torch.zeros(2, means.shape[0])
        for py in range(H):
            for px in range(W):
                T = 1.0
                for gi in order:
                    g = int(gi)
                    if z[g] <= 0.01:
                        continue
                    d = torch.tensor([uv[g, 0] - px, uv[g, 1] - py])
                    power = -0.5 * (d @ cinv[g] @ d)
                    if power > 0:
                        continue
                    alpha = min(0.99, float(opac[g] * torch.exp(power)))
                    if alpha < 1.0 / 255.0:
                        continue
                    if T * (1 - alpha) < 1e-4:
                        break
                    img[py, px] += alpha * T * colors[g]
                    counts[int(mask[py, px]), g] += alpha * T
                    T *= 1 - alpha
        return img, counts

    def make_view(deg, radius=3.0):
        a = torch.tensor(deg * torch.pi / 180.0)
        cpos = torch.tensor([radius * torch.sin(a), 0.0, -radius * torch.cos(a)])
        fwd = -cpos / cpos.norm()
        up = torch.tensor([0.0, -1.0, 0.0])
        right = torch.linalg.cross(up, fwd)
        right = right / right.norm()
        Rc2w = torch.stack([right, torch.linalg.cross(fwd, right), fwd], dim=1)
        w2c = torch.eye(4)
        w2c[:3, :3] = Rc2w.T
        w2c[:3, 3] = -Rc2w.T @ cpos
        return w2c

    # 0,1 solid object; 2 low-opacity glass halo; 3 wall behind; 4,5 distractors
    W = H = 40
    K = torch.tensor([[40.0, 0, W / 2], [0, 40.0, H / 2], [0, 0, 1.0]])
    means = torch.tensor([
        [0.00, 0.00, 0.00], [0.12, 0.05, 0.05], [-0.05, -0.10, -0.08],
        [0.00, 0.00, 1.50], [1.20, 0.00, 0.00], [-1.20, 0.30, 0.20]])
    scales = torch.tensor([
        [.10, .10, .10], [.08, .08, .08], [.18, .18, .18],
        [.90, .90, .05], [.15, .15, .15], [.15, .15, .15]])
    quats = torch.zeros(6, 4)
    quats[:, 0] = 1.0
    opac = torch.tensor([0.95, 0.9, 0.15, 0.95, 0.9, 0.9])
    views = [make_view(0.0), make_view(35.0)]

    total = torch.zeros(2, 6)
    for w2c in views:
        sel = torch.tensor([0, 1, 2])  # GT mask: render object-only alpha
        mimg, _ = render_counts(means[sel], quats[sel], scales[sel], opac[sel],
                                torch.ones(3, 1), K, w2c, W, H,
                                torch.zeros(H, W, dtype=torch.long))
        mask = (mimg[..., 0] > 0.5).long()
        colors = torch.zeros(6, 2, requires_grad=True)
        img, counts = render_counts(means, quats, scales, opac, colors,
                                    K, w2c, W, H, mask)
        loss = (img[..., 0] * mask).sum() + (img[..., 1] * (1 - mask)).sum()
        (grad,) = torch.autograd.grad(loss, colors)
        err = (counts - torch.stack([grad[:, 1], grad[:, 0]])).abs().max()
        print(f"[selftest] gradient identity max|err| = {err:.2e}")
        assert err < 1e-9, "gradient trick != direct used_count accumulation"
        total += counts

    lab0 = flashsplat_assign(total, 0.0)
    labm = flashsplat_assign(total, -0.75)
    labp = flashsplat_assign(total, 0.4)
    print(f"[selftest] slack  0.0  -> {lab0.nonzero().flatten().tolist()}")
    print(f"[selftest] slack -0.75 -> {labm.nonzero().flatten().tolist()}")
    print(f"[selftest] slack +0.4  -> {labp.nonzero().flatten().tolist()}")
    assert set(lab0.nonzero().flatten().tolist()) == {0, 1}
    assert set(labm.nonzero().flatten().tolist()) == {0, 1, 2}  # + glass halo
    assert bool((labm | lab0).eq(labm).all())  # negative slackness dilates
    assert bool((lab0 | labp).eq(lab0).all())  # positive slackness shrinks
    print("[selftest] OK: formulation + solver validated (CPU, float64)")


if __name__ == "__main__":
    main()
