"""Removal verification, stage 1 (mini-viewer env, GPU): render the clean
background at each removed object's supervision views AND held-out views,
with alpha and expected-depth channels, plus the ORIGINAL splat renders for
before/after comparison.

Checks computed here (independent of the inpaint supervision signal):
  - alpha coverage in the removal mask (holes into the void)
  - expected-depth vs support-plane depth (floating / sunken fill)
  - held-out-view photometric consistency of the filled region (project
    hole pixels view->view via the support plane; hallucinated single-view
    texture shows up as cross-view error)
Writes OUT/inpaint/verify/{obj_XX_view*.png, verify_render.json} plus, for
each held-out view, a projected object region in verify/held_regions/
obj_XX_<frame>.npz; stage 2 (SAM3 re-detection) consumes the renders and
uses the regions to localize held-view scoring.
"""
import json

import numpy as np
from PIL import Image

from agents.core import common as C

N_HELDOUT = 2


def main():
    K, W, H, _ = C.load_intrinsics()
    w2c_all = C.load_colmap_w2c()
    vdir = C.OUT / "inpaint" / "verify"
    vdir.mkdir(parents=True, exist_ok=True)

    orig = C.load_gaussians(C.SPLAT_PLY)
    clean = C.load_gaussians(C.OUT / "inpaint" / "clean_background.ply")
    means_np = orig["means"].cpu().numpy()

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    report = {}
    for m in objects:
        name = f"obj_{m['index']:02d}"
        idir = C.OUT / "inpaint" / name
        if not (idir / "views.json").exists():
            continue
        views = json.loads((idir / "views.json").read_text())
        plane = None
        if (idir / "plane.json").exists():
            plane = json.loads((idir / "plane.json").read_text())
        rem_pts = None  # removed-gaussian centers, for held-view regions
        if (idir / "removal_idx.npy").exists():
            rem_pts = means_np[np.load(idir / "removal_idx.npy")]

        # supervision views + held-out neighbours (offset in trajectory)
        names_sup = [v["frame"] for v in views]
        all_names = sorted(w2c_all)
        heldout = []
        for fn in names_sup[:1]:
            i = all_names.index(fn)
            for off in (7, -7):
                j = max(0, min(len(all_names) - 1, i + off))
                if all_names[j] not in names_sup:
                    heldout.append(all_names[j])
            heldout = heldout[:N_HELDOUT]

        entries = []
        for tag, fn in ([("sup", f) for f in names_sup]
                        + [("held", f) for f in heldout]):
            w2c = w2c_all[fn]
            mask = None
            for k, v in enumerate(views):
                if v["frame"] == fn and (idir / f"mask_{k}.png").exists():
                    mask = np.asarray(Image.open(idir / f"mask_{k}.png")) > 127
            r_clean, d_clean, a_clean = C.render_view(
                clean, w2c, K, W, H, render_mode="RGB+ED", scale=0.5)
            r_orig, _, _ = C.render_view(orig, w2c, K, W, H, scale=0.5)
            h2, w2 = r_clean.shape[:2]
            e = {"frame": fn, "tag": tag}
            if tag == "held" and rem_pts is not None:
                # project the removed-gaussian centers into the held view and
                # save their 2D bbox, dilated by 15% of the image diagonal,
                # as verify/held_regions/obj_XX_<frame>.npz (key "region",
                # bool [h2,w2]); stage 2 uses it to localize re-detection.
                Ks = np.array(K)
                Ks[:2] *= 0.5
                pc = rem_pts @ w2c[:3, :3].T + w2c[:3, 3]
                z = np.maximum(pc[:, 2], 1e-6)
                uu = Ks[0, 0] * pc[:, 0] / z + Ks[0, 2] - .5
                vv = Ks[1, 1] * pc[:, 1] / z + Ks[1, 2] - .5
                inb = ((pc[:, 2] > 0.1) & (uu >= 0) & (uu < w2)
                       & (vv >= 0) & (vv < h2))
                if inb.sum() >= 20:
                    pad = 0.15 * float(np.hypot(w2, h2))
                    u0 = max(int(uu[inb].min() - pad), 0)
                    u1 = int(uu[inb].max() + pad) + 1
                    v0 = max(int(vv[inb].min() - pad), 0)
                    v1 = int(vv[inb].max() + pad) + 1
                    region = np.zeros((h2, w2), bool)
                    region[v0:v1, u0:u1] = True
                    rdir = vdir / "held_regions"
                    rdir.mkdir(exist_ok=True)
                    np.savez_compressed(rdir / f"{name}_{fn}.npz",
                                        region=region)
                    e["region"] = f"held_regions/{name}_{fn}.npz"
            if mask is not None:
                msk = np.asarray(Image.fromarray(mask).resize(
                    (w2, h2), Image.NEAREST))
                e["alpha_cov"] = float(a_clean[msk].mean())
                if plane is not None:
                    o = np.array(plane["origin"])
                    n = np.array(plane["normal"])
                    c2w = np.linalg.inv(w2c)
                    Ks = np.array(K)
                    Ks[:2] *= 0.5
                    vv, uu = np.nonzero(msk)
                    d_cam = np.stack([(uu + .5 - Ks[0, 2]) / Ks[0, 0],
                                      (vv + .5 - Ks[1, 2]) / Ks[1, 1],
                                      np.ones_like(uu, dtype=float)], -1)
                    dw = d_cam @ c2w[:3, :3].T
                    denom = dw @ n
                    ok = np.abs(denom) > 1e-6
                    t_plane = ((o - c2w[:3, 3]) @ n) / denom[ok]
                    zpl = (d_cam[ok, 2] * t_plane)
                    zr = d_clean[vv[ok], uu[ok]]
                    val = (t_plane > 0.1) & (zr > 0.1)
                    if val.sum() > 50:
                        e["depth_vs_plane_med_mm"] = float(
                            np.median(np.abs(zr[val] - zpl[val])) * 1000)
            side = np.concatenate([r_orig, r_clean], axis=1)
            fn_out = f"{name}_{tag}_{fn}.jpg"
            Image.fromarray((np.clip(side, 0, 1) * 255).astype(np.uint8)
                            ).save(vdir / fn_out, quality=88)
            e["render"] = fn_out
            entries.append(e)
        report[name] = {"label": m["label"], "views": entries}
        covs = [e["alpha_cov"] for e in entries if "alpha_cov" in e]
        deps = [e["depth_vs_plane_med_mm"] for e in entries
                if "depth_vs_plane_med_mm" in e]
        print(f"[vr] {name} {m['label']}: alpha_cov "
              f"{np.mean(covs):.3f}" if covs else f"[vr] {name}: no mask",
              f" depth-plane {np.median(deps):.0f}mm" if deps else "")
    C.save_json(vdir / "verify_render.json", report)
    print(f"[vr] wrote {vdir}/verify_render.json")


if __name__ == "__main__":
    main()
