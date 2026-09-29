"""Inpainting stage 4 (mini-viewer env, GPU): carve + normal-guided fill +
photometric refinement against the Qwen-inpainted views.

- carve: drop the removal-set gaussians from the scene splat (the hole)
- fill: new thin gaussians on the fitted support plane inside the object
  footprint (normal-oriented disks), colors initialized from surviving
  neighbor gaussians
- refine: optimize the NEW gaussians only (position/scale/opacity/color,
  orientation frozen to the plane) with L1 loss inside the removal masks
  against the inpainted images, differentiable gsplat rendering
Outputs: inpaint/fill_gaussians.npz, clean_background.ply (carved+filled,
Inria format), before/hole/filled comparison renders + hole-region PSNR.
"""
import json
from pathlib import Path

import numpy as np

from agents.core import common as C

GRID = 0.005          # m, fill-point spacing on the plane
HULL_OFFSET = 0.035   # m, metric outward offset: must cover the carve RADIUS
ITERS = 1500
RENDER_SCALE = 0.5
DISK = (0.004, 0.004, 0.0008)  # m, fill gaussian scales (thin axis = normal)


def quat_from_frame(u, v, n):
    R = np.stack([u, v, n], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1
    return C.rot_to_quat_wxyz(R)


def _fill():
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from PIL import Image
    from scipy.spatial import ConvexHull, Delaunay, cKDTree

    factory = C.OUT
    preparation = C.OUT/'inpaint'
    output = C.OUT/'inpaint'
    mask_path = lambda slot, k: preparation/slot/f'mask_{k}.png'
    image_path = lambda slot, k: preparation/slot/f'inpainted_{k}.png'
    K, W, H, _ = C.load_intrinsics()
    gs = C.load_gaussians(C.SPLAT_PLY)  # activated params, cuda
    N = gs["means"].shape[0]
    rm_idx = np.load(preparation / "removal_union_idx.npy", allow_pickle=False)

    # multi-view mask-vote removal: transparent objects are modeled by the
    # splat as diffuse low-opacity gaussian clouds whose centers can sit far
    # from the physical surface (behind glass, above the desk), so geometric
    # proximity misses them. A gaussian near the object's AABB whose center
    # projects inside the object's 2D mask in >=2 related views belongs to it.
    means_np = gs["means"].cpu().numpy().astype(np.float64)
    extra = []
    objects0 = json.loads((factory / "objects" / "objects.json").read_text())
    from PIL import Image as PImage
    for m in objects0:
        odir = preparation / f"obj_{m['index']:02d}"
        if not (odir / "views.json").exists():
            continue
        vws = json.loads((odir / "views.json").read_text())
        masks = []
        for k, vw in enumerate(vws):
            mp = mask_path(odir.name, k)
            if mp.exists():
                masks.append((np.asarray(PImage.open(mp)) > 127,
                              np.array(vw["w2c"])))
        if len(masks) < 2:
            continue
        lo, hi = np.array(m["aabb"])
        cand = np.nonzero(
            (means_np[:, 0] > lo[0] - 0.15) & (means_np[:, 0] < hi[0] + 0.15) &
            (means_np[:, 1] > lo[1] - 0.15) & (means_np[:, 1] < hi[1] + 0.15) &
            (means_np[:, 2] > lo[2] - 0.05) & (means_np[:, 2] < hi[2] + 0.15))[0]
        if len(cand) == 0:
            continue
        votes = np.zeros(len(cand), dtype=np.int32)
        for msk, w2c in masks:
            pc = means_np[cand] @ w2c[:3, :3].T + w2c[:3, 3]
            z = np.clip(pc[:, 2], 1e-6, None)
            u = (pc[:, 0] / z * K[0, 0] + K[0, 2]).astype(int)
            v = (pc[:, 1] / z * K[1, 1] + K[1, 2]).astype(int)
            ok = (pc[:, 2] > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            votes[ok] += msk[v[ok], u[ok]]
        extra.append(cand[votes >= 2])
    if extra:
        extra = np.unique(np.concatenate(extra))
        n0 = len(rm_idx)
        rm_idx = np.unique(np.concatenate([rm_idx, extra]))
        print(f"[if] mask-vote removal: +{len(rm_idx) - n0} gaussians "
              f"(total {len(rm_idx)})")

    keep = torch.ones(N, dtype=torch.bool, device="cuda")
    keep[torch.from_numpy(rm_idx).long().cuda()] = False
    kept = {k: (v[keep] if k != "sh_degree" else v) for k, v in gs.items()}
    kept_tree = cKDTree(kept["means"].cpu().numpy())

    objects = json.loads((factory / "objects" / "objects.json").read_text())
    new_means, new_quats, new_scales, views_all = [], [], [], {}
    slices, cursor = {}, 0
    for m in objects:
        odir = preparation / f"obj_{m['index']:02d}"
        if not (odir / "plane.json").exists():
            continue
        pl = json.loads((odir / "plane.json").read_text())
        o, n = np.array(pl["origin"]), np.array(pl["normal"])
        u, v = np.array(pl["u"]), np.array(pl["v"])
        uv = np.array(pl["footprint_uv"])
        from scipy.spatial import QhullError
        try:
            hull = ConvexHull(uv)
        except QhullError:
            print(f"[if] obj_{m['index']:02d}: degenerate footprint, skip")
            continue
        poly = uv[hull.vertices]
        # metric outward offset so the fill covers the whole carved annulus
        c = poly.mean(0)
        r = np.linalg.norm(poly - c, axis=1)
        poly = c + (poly - c) * ((r + HULL_OFFSET) / np.maximum(r, 1e-6))[:, None]
        tri = Delaunay(poly)
        g0, g1 = poly.min(0), poly.max(0)
        gu, gv_ = np.meshgrid(np.arange(g0[0], g1[0], GRID),
                              np.arange(g0[1], g1[1], GRID))
        pts_uv = np.stack([gu.ravel(), gv_.ravel()], axis=1)
        pts_uv = pts_uv[tri.find_simplex(pts_uv) >= 0]
        if len(pts_uv) == 0:
            continue
        pts = o + pts_uv[:, :1] * u + pts_uv[:, 1:2] * v + 5e-4 * n
        q = quat_from_frame(u, v, n)
        new_means.append(pts)
        new_quats.append(np.tile(q, (len(pts), 1)))
        new_scales.append(np.tile(list(DISK), (len(pts), 1)))
        slices[f"obj_{m['index']:02d}"] = (cursor, cursor + len(pts))
        cursor += len(pts)
        # (mask, inpaint) PAIRS per frame: each inpainted_k erased only its
        # own object, so the target must be composited per frame from every
        # object's pixels inside its own mask (last-writer-wins would bake
        # neighbor objects' photos into the fill)
        vws = json.loads((odir / "views.json").read_text())
        for k, view in enumerate(vws):
            key = view["frame"]
            views_all.setdefault(key, {"w2c": np.array(view["w2c"]),
                                       "pairs": []})
            ip = image_path(odir.name, k)
            mk = mask_path(odir.name, k)
            if ip.exists() and mk.exists():
                views_all[key]["pairs"].append((str(mk), str(ip)))

    if not new_means:
        print("[if] nothing to fill")
        return
    new_means = np.concatenate(new_means)
    new_quats = np.concatenate(new_quats)
    new_scales = np.concatenate(new_scales)
    M = len(new_means)
    print(f"[if] {M} fill gaussians across {len(slices)} objects, "
          f"{len(views_all)} target views")

    # color init: project each fill point into its object's best inpainted
    # view and take that pixel directly; kNN neighbor-gaussian color is only
    # the fallback (neighbor colors made the fill start far off-target and
    # 500 iters of inconsistent multi-view supervision left blotches)
    _, nn = kept_tree.query(new_means, k=5)
    sh0_np = kept["sh"][:, 0][torch.from_numpy(nn).long().cuda()] \
        .mean(dim=1).cpu().numpy()
    C0 = 0.2820948
    from PIL import Image as PImage
    for oname, (a, b) in slices.items():
        odir = preparation / oname
        ip = image_path(oname, 0)
        if not (odir / "views.json").exists() or not ip.exists():
            continue
        vws = json.loads((odir / "views.json").read_text())
        img = np.asarray(PImage.open(ip).convert("RGB"),
                         dtype=np.float32) / 255.0
        w2c = np.array(vws[0]["w2c"])
        pc = new_means[a:b] @ w2c[:3, :3].T + w2c[:3, 3]
        z = np.clip(pc[:, 2], 1e-6, None)
        upx = (pc[:, 0] / z * K[0, 0] + K[0, 2]).astype(int)
        vpx = (pc[:, 1] / z * K[1, 1] + K[1, 2]).astype(int)
        ok = (pc[:, 2] > 0.05) & (upx >= 0) & (upx < W) & (vpx >= 0) & (vpx < H)
        idxs = np.arange(a, b)[ok]
        sh0_np[idxs] = (img[vpx[ok], upx[ok]] - 0.5) / C0
    sh0_init = torch.tensor(sh0_np, dtype=torch.float32, device="cuda")

    dev = "cuda"
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    p_off = torch.zeros(M, 3, device=dev, requires_grad=True)
    p_lsc = torch.log(t(new_scales)).clone().requires_grad_(True)
    p_opa = torch.full((M,), 2.2, device=dev, requires_grad=True)  # sig~0.9
    p_sh0 = sh0_init.clone().requires_grad_(True)
    base_means = t(new_means)
    base_quats = t(new_quats)
    Kdeg = kept["sh"].shape[1]

    def composite_target(d):
        """One target per frame: original photo with each object's inpainted
        pixels pasted inside its own mask; loss mask = union of the masks.
        Where two objects' dilated masks overlap, the FIRST-pasted object
        (stable priority = object index order) wins: each single-object
        inpaint still depicts the OTHER, un-removed object there, so the
        patches are mutually inconsistent and last-writer-wins would bake
        the neighbor's photo back into the overlap band."""
        img = np.asarray(Image.open(C.IMAGES_DIR / d['frame']).convert("RGB"),
                         dtype=np.float32) / 255.0
        mask = np.zeros((H, W), bool)
        overlap = 0
        for mk, ip in d["pairs"]:
            mm = np.asarray(Image.open(mk)) > 127
            overlap += int((mm & mask).sum())
            mm &= ~mask
            img[mm] = np.asarray(Image.open(ip).convert("RGB"),
                                 dtype=np.float32)[mm] / 255.0
            mask |= mm
        if overlap:
            print(f"[if] {d['frame']}: {overlap} px object-mask overlap "
                  "(first object kept)")
        return img, mask

    # supervise ONLY on primary (view-0) frames: each Qwen inpaint is
    # internally consistent but different frames invent different desk
    # texture - mixing them trains the fill toward a blotchy average
    primary = set()
    for oname in slices:
        vj = preparation / oname / "views.json"
        vws = json.loads(vj.read_text()) if vj.exists() else []
        if vws:
            primary.add(vws[0]["frame"])

    targets = []
    for frame, d in views_all.items():
        if not d["pairs"] or frame not in primary:
            continue
        d["frame"] = frame
        img, mask = composite_target(d)
        s = RENDER_SCALE
        h2, w2 = int(round(H * s)), int(round(W * s))
        img_t = torch.from_numpy(img).to(dev)
        img_t = F.interpolate(img_t.permute(2, 0, 1)[None], (h2, w2),
                              mode="bilinear")[0].permute(1, 2, 0)
        m_t = F.interpolate(torch.from_numpy(mask)[None, None].float(),
                            (h2, w2))[0, 0].to(dev) > 0.5
        Ks = np.array(K)
        Ks[:2] *= s
        targets.append({"w2c": t(d["w2c"]), "K": t(Ks), "img": img_t,
                        "mask": m_t, "hw": (h2, w2)})
    assert targets, "no inpainted targets found - run inpaint_qwen first"

    opt = torch.optim.Adam([
        {"params": [p_off], "lr": 1e-3},
        {"params": [p_lsc], "lr": 5e-3},
        {"params": [p_opa], "lr": 5e-2},
        {"params": [p_sh0], "lr": 2.5e-2},
    ])
    for it in range(ITERS):
        if it == int(ITERS * 0.55):
            for grp in opt.param_groups:
                grp["lr"] *= 0.3
        tg = targets[it % len(targets)]
        sh_new = torch.cat([p_sh0[:, None, :],
                            torch.zeros(M, Kdeg - 1, 3, device=dev)], dim=1)
        means = torch.cat([kept["means"], base_means + p_off])
        quats = torch.cat([kept["quats"], base_quats])
        scales = torch.cat([kept["scales"], torch.exp(p_lsc)])
        opac = torch.cat([kept["opacities"], torch.sigmoid(p_opa)])
        sh = torch.cat([kept["sh"], sh_new])
        img, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac,
            colors=sh, viewmats=tg["w2c"][None], Ks=tg["K"][None],
            width=tg["hw"][1], height=tg["hw"][0],
            sh_degree=gs["sh_degree"], render_mode="RGB", packed=False)
        loss = (img[0] - tg["img"]).abs()[tg["mask"]].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0:
            print(f"[if] iter {it}: L1 {float(loss):.4f}")

    # ---- save fill + clean background ply ----------------------------------
    with torch.no_grad():
        fill = {
            "means": (base_means + p_off).cpu().numpy(),
            "quats": base_quats.cpu().numpy(),
            "scales": torch.exp(p_lsc).cpu().numpy(),
            "opacities": torch.sigmoid(p_opa).cpu().numpy(),
            "sh0": p_sh0.cpu().numpy(),
        }
    np.savez_compressed(output / "fill_gaussians.npz",
                        removal_idx=rm_idx,
                        slices=json.dumps({k: list(vv) for k, vv in slices.items()}),
                        **fill)

    from plyfile import PlyData, PlyElement
    km = {k: v.cpu().numpy() for k, v in kept.items() if k != "sh_degree"}
    n_all = len(km["means"]) + M
    k_rest = Kdeg - 1
    names = (["x", "y", "z", "nx", "ny", "nz"] +
             [f"f_dc_{i}" for i in range(3)] +
             [f"f_rest_{i}" for i in range(3 * k_rest)] +
             ["opacity"] + [f"scale_{i}" for i in range(3)] +
             [f"rot_{i}" for i in range(4)])
    arr = np.zeros(n_all, dtype=[(nm, "f4") for nm in names])
    means_all = np.concatenate([km["means"], fill["means"]])
    sh_all = np.concatenate(
        [km["sh"], np.concatenate([fill["sh0"][:, None, :],
                                   np.zeros((M, k_rest, 3), np.float32)], axis=1)])
    opac_all = np.concatenate([km["opacities"], fill["opacities"]])
    scal_all = np.concatenate([km["scales"], fill["scales"]])
    quat_all = np.concatenate([km["quats"], fill["quats"]])
    arr["x"], arr["y"], arr["z"] = means_all.T
    for i in range(3):
        arr[f"f_dc_{i}"] = sh_all[:, 0, i]
    rest = sh_all[:, 1:, :].transpose(0, 2, 1).reshape(n_all, -1)  # inria layout
    for i in range(3 * k_rest):
        arr[f"f_rest_{i}"] = rest[:, i]
    eps = 1e-6
    ol = np.clip(opac_all, eps, 1 - eps)
    arr["opacity"] = np.log(ol / (1 - ol))
    for i in range(3):
        arr[f"scale_{i}"] = np.log(np.clip(scal_all[:, i], eps, None))
    for i in range(4):
        arr[f"rot_{i}"] = quat_all[:, i]
    PlyData([PlyElement.describe(arr, "vertex")]).write(
        str(output / "clean_background.ply"))

    # ---- comparison renders + hole PSNR ------------------------------------
    carved = {k: (v if k == "sh_degree" else v) for k, v in kept.items()}
    carved["sh_degree"] = gs["sh_degree"]
    filled = C.cat_gaussians([carved, {
        "means": torch.tensor(fill["means"], device=dev),
        "quats": torch.tensor(fill["quats"], device=dev),
        "scales": torch.tensor(fill["scales"], device=dev),
        "opacities": torch.tensor(fill["opacities"], device=dev),
        "sh": torch.cat([torch.tensor(fill["sh0"], device=dev)[:, None, :],
                         torch.zeros(M, k_rest, 3, device=dev)], dim=1),
        "sh_degree": gs["sh_degree"]}])
    # backend per "obj_XX/k" as logged by inpaint_qwen (possibly sharded)
    ip_meta = {}
    for mj in sorted(preparation.glob("inpaint_meta*.json")):
        ip_meta.update(json.loads(mj.read_text()))
    stats = []
    for frame, d in views_all.items():
        if not d["pairs"]:
            continue
        w2c = d["w2c"]
        d["frame"] = frame
        tgt, mask = composite_target(d)
        r_orig, _, _ = C.render_view(gs, w2c, K, W, H)
        r_hole, _, _ = C.render_view(carved, w2c, K, W, H)
        r_fill, _, _ = C.render_view(filled, w2c, K, W, H)
        p = C.psnr(r_fill[mask], tgt[mask])
        objs, bks = [], set()
        for _, ip in d["pairs"]:
            oname, stem = Path(ip).parent.name, Path(ip).stem
            objs.append(oname)
            bks.add(ip_meta.get(f"{oname}/{stem.split('_')[-1]}"))
        bks.discard(None)
        stats.append({"frame": frame, "hole_psnr_vs_inpainted": p,
                      "held": frame not in primary,
                      "backend": (bks.pop() if len(bks) == 1
                                  else "mixed" if bks else None),
                      "objects": sorted(set(objs))})
        side = np.concatenate([r_orig, r_hole, r_fill], axis=1)
        Image.fromarray((np.clip(side, 0, 1) * 255).astype(np.uint8)).save(
            output / f"compare_{frame}.jpg", quality=88)
        print(f"[if] {frame}: hole PSNR vs inpainted {p:.2f} dB "
              f"({'held' if frame not in primary else 'trained'})")
    C.save_json(output / "fill_stats.json", stats)
    # aggregates split by supervision: the primary (view-0) frames are
    # directly L1-optimized, so only mean_hole_psnr_held measures
    # generalization - cite held-only in the paper
    mean = lambda v: float(np.mean(v)) if v else None
    agg = {"mean_hole_psnr_trained": mean(
               [s["hole_psnr_vs_inpainted"] for s in stats if not s["held"]]),
           "mean_hole_psnr_held": mean(
               [s["hole_psnr_vs_inpainted"] for s in stats if s["held"]]),
           "n_trained": sum(not s["held"] for s in stats),
           "n_held": sum(s["held"] for s in stats)}
    C.save_json(output / "fill_stats_summary.json", agg)
    print(f"[if] hole PSNR mean: trained {agg['mean_hole_psnr_trained']} "
          f"(n={agg['n_trained']}), held {agg['mean_hole_psnr_held']} "
          f"(n={agg['n_held']})")
    print("[if] done -> clean_background.ply")
    return dict(source_gaussians=N, removal_gaussians=len(rm_idx), fill_gaussians=M,
                filled_objects=len(slices), primary_frames=sorted(primary), diagnostic_frames=list(views_all),
                iterations=ITERS)


def main(argv=None):
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    _fill()


if __name__=='__main__':
    main()
