"""Factory stage 4: novel-view render metrics (mini-viewer env, GPU).

On up to 8 held-out DSLR frames -- the scene's official
dslr/train_test_lists.json "test" split intersected with COLMAP-registered
frames (legacy linspace over ALL registered frames only if that file is
missing) -- compare against the GT photo:
  - background scene splat alone  (= the SceneSplat/GaussianWorld baseline)
  - composite digital twin        (splat + tier-A/B assets at aligned poses)
  - if inpaint/clean_background.ply exists, additionally: clean background
    alone ("clean_bg") and assets on clean background ("ours_twin_clean")
PSNR / SSIM / LPIPS(alex). Writes render_metrics_v2.json + one comparison jpg.
"""
import json

import numpy as np
from PIL import Image

from agents.core import common as C

N_FRAMES = 8


def main():
    import lpips
    import torch
    from skimage.metrics import structural_similarity as ssim_fn

    K, W, H, _ = C.load_intrinsics()
    w2c_all = C.load_colmap_w2c()
    tt_json = C.SCENE_DIR / "dslr" / "train_test_lists.json"
    test = [n for n in json.loads(tt_json.read_text())["test"]
            if n in w2c_all] if tt_json.exists() else []
    if test:
        split = "official-test"
        idx = np.linspace(0, len(test) - 1, min(N_FRAMES, len(test))).astype(int)
        names = [test[i] for i in idx]
    else:
        split = "legacy-linspace"
        print("[fe] WARNING: no usable dslr/train_test_lists.json for "
              f"{C.SCENE_ID} -- falling back to linspace over ALL registered "
              "frames (these overlap the splat's TRAINING views)")
        reg = sorted(w2c_all)
        idx = np.linspace(0, len(reg) - 1, N_FRAMES).astype(int)
        names = [reg[i] for i in idx]
    frames = [(n, w2c_all[n]) for n in names]

    bg = C.load_gaussians(C.SPLAT_PLY)
    parts = [bg]
    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    n_used = 0
    for m in objects:
        odir = C.OUT / "objects" / f"obj_{m['index']:02d}"
        al = json.loads((odir / "aligned.json").read_text())
        if al.get("rejected"):
            continue
        gs = C.pad_sh(C.load_gaussians(odir / "trellis_gs.ply"), bg["sh_degree"])
        parts.append(C.transform_gaussians(gs, np.array(al["T"])))
        n_used += 1
    twin = C.cat_gaussians(parts) if n_used else bg

    variants = {"bg": bg, "twin": twin}
    clean_ply = C.OUT / "inpaint" / "clean_background.ply"
    if clean_ply.exists():
        clean = C.load_gaussians(clean_ply)
        variants["clean_bg"] = clean
        variants["ours_twin_clean"] = (C.cat_gaussians([clean] + parts[1:])
                                       if n_used else clean)

    loss_fn = lpips.LPIPS(net="alex", verbose=False).cuda()

    def metrics(render, photo):
        p = float(C.psnr(render, photo))
        s = float(ssim_fn(photo, render, channel_axis=2, data_range=1.0))
        with torch.no_grad():
            a = torch.from_numpy(render).permute(2, 0, 1)[None].float().cuda() * 2 - 1
            b = torch.from_numpy(photo).permute(2, 0, 1)[None].float().cuda() * 2 - 1
            l = float(loss_fn(a, b).item())
        return p, s, l

    rows = []
    for name, w2c in frames:
        photo = np.asarray(Image.open(C.IMAGES_DIR / name).convert("RGB"),
                           dtype=np.float32) / 255.0
        row = {"frame": name}
        for key, gs in variants.items():
            r, _, _ = C.render_view(gs, w2c, K, W, H)
            row[key] = dict(zip(("psnr", "ssim", "lpips"), metrics(r, photo)))
            if key == "bg":
                r_bg = r
            elif key == "twin":
                r_tw = r
        rows.append(row)
        print(f"[fe] {name}: " + "  ".join(
            f"{k} {row[k]['psnr']:.2f}/{row[k]['ssim']:.3f}/{row[k]['lpips']:.3f}"
            for k in variants))

    mean = lambda k1, k2: float(np.mean([r[k1][k2] for r in rows]))
    summary = {"n_objects_composited": n_used, "split": split,
               "eval_frames": names, "frames": rows,
               "mean": {v: {k: mean(v, k) for k in ("psnr", "ssim", "lpips")}
                        for v in variants}}
    C.save_json(C.OUT / "render_metrics_v2.json", summary)
    side = np.concatenate([photo, r_bg, r_tw], axis=1)
    Image.fromarray((np.clip(side, 0, 1) * 255).astype(np.uint8)).save(
        C.OUT / "eval_photo_bg_twin.jpg", quality=88)
    m = summary["mean"]
    print(f"[fe] split={split} MEAN " + " | ".join(
        f"{v} PSNR {m[v]['psnr']:.2f} SSIM {m[v]['ssim']:.3f} "
        f"LPIPS {m[v]['lpips']:.3f}" for v in variants))


if __name__ == "__main__":
    main()
