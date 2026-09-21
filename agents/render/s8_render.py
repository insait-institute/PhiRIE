"""Stage 8: photoreal renders of the digital twin with gsplat.

- background = GaussianWorld scene splat (still contains the original objects;
  the paper would use the inpainted background splat instead - known artifact)
- objects = TRELLIS gaussians, transformed by settled poses / per-frame
  simulated poses -> physics video composited into the real scene splat.
Outputs in OUT/render/: photo_vs_twin.png, physics.mp4, asset_gallery.png,
per-object turntable.png, stats.json.
"""
import json

import numpy as np
from PIL import Image

from agents.core import common as C


def look_at_w2c(eye, target, up=(0, 0, 1)):
    """OpenCV w2c from eye/target (world z-up)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([right, down, fwd])
    w2c[:3, 3] = -w2c[:3, :3] @ eye
    return w2c


def pose_T(scale, pos, quat_xyzw):
    q = np.asarray(quat_xyzw, float)
    R = C.quat_to_rot_wxyz([q[3], q[0], q[1], q[2]])
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = pos
    return T


def main():
    import imageio.v2 as imageio
    import torch

    K, W, H, _ = C.load_intrinsics()
    rep = json.loads((C.OUT / "frame" / "rep_frame.json").read_text())
    w2c = np.array(rep["w2c"])
    out = C.OUT / "render"
    out.mkdir(parents=True, exist_ok=True)

    bg = C.load_gaussians(C.SPLAT_PLY)
    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    if not objects:
        print("[s8] no objects lifted; nothing to render")
        return
    settled = json.loads((C.OUT / "sim" / "settled.json").read_text())["bodies"]
    canon = {}
    for meta in objects:
        name = f"obj_{meta['index']:02d}"
        aligned = json.loads(
            (C.OUT / "objects" / name / "aligned.json").read_text())
        if aligned.get("rejected"):
            print(f"[s8] {name}: skip ({aligned['rejected']})")
            continue
        canon[name] = C.pad_sh(C.load_gaussians(
            C.OUT / "objects" / name / "trellis_gs.ply"), bg["sh_degree"])

    # --- background sanity + settled twin composite -------------------------
    photo = np.asarray(Image.open(C.OUT / "frame" / rep["frame"])).astype(np.float32) / 255
    rgb_bg, _, _ = C.render_view(bg, w2c, K, W, H)
    stats = {"psnr_background_vs_photo": C.psnr(rgb_bg, photo)}
    print(f"[s8] background splat PSNR vs photo: {stats['psnr_background_vs_photo']:.2f} dB")

    def compose(frame_poses):
        parts = [bg]
        for name, (pos, quat) in frame_poses.items():
            base = name.replace("_cousin", "")
            if base not in canon:
                continue
            s = settled[base]["scale"] if base in settled else 1.0
            parts.append(C.transform_gaussians(canon[base], pose_T(s, pos, quat)))
        return C.cat_gaussians(parts)

    twin_poses = {n: (v["pos"], v["quat_xyzw"]) for n, v in settled.items()
                  if "pos" in v}
    gs_twin = compose(twin_poses)
    rgb_twin, _, _ = C.render_view(gs_twin, w2c, K, W, H)
    side = np.concatenate([photo, rgb_twin], axis=1)
    Image.fromarray((side * 255).astype(np.uint8)).save(out / "photo_vs_twin.png")
    stats["psnr_twin_vs_photo"] = C.psnr(rgb_twin, photo)
    del gs_twin
    torch.cuda.empty_cache()

    # --- per-object canonical turntables ------------------------------------
    Ko = np.array([[550.0, 0, 256], [0, 550.0, 256], [0, 0, 1]])
    tiles = []
    for name, gs in canon.items():
        views = []
        for az in (0, 90, 180, 270):
            a = np.deg2rad(az)
            eye = np.array([np.cos(a), np.sin(a), 0.55]) * 1.1
            rgb, _, _ = C.render_view(gs, look_at_w2c(eye, [0, 0, 0]), Ko,
                                      512, 512, background=(1, 1, 1))
            views.append(rgb)
        row = np.concatenate(views, axis=1)
        Image.fromarray((row * 255).astype(np.uint8)).save(
            C.OUT / "objects" / name / "turntable.png")
        tiles.append(row)
    gallery = np.concatenate(tiles, axis=0)
    Image.fromarray((gallery * 255).astype(np.uint8)).save(out / "asset_gallery.png")

    # --- physics video -------------------------------------------------------
    dyn = json.loads((C.OUT / "sim" / "dynamics.json").read_text())
    writer = imageio.get_writer(out / "physics.mp4", fps=int(dyn["fps"]),
                                macro_block_size=None, quality=8)
    for k, fr in enumerate(dyn["frames"]):
        gs = compose(fr)
        rgb, _, _ = C.render_view(gs, w2c, K, W, H, scale=0.5)
        writer.append_data((rgb * 255).astype(np.uint8))
        if k == 0:
            Image.fromarray((rgb * 255).astype(np.uint8)).save(out / "physics_first.png")
        del gs
        if k % 20 == 0:
            torch.cuda.empty_cache()
            print(f"[s8] physics frame {k}/{len(dyn['frames'])}")
    writer.close()
    Image.fromarray((rgb * 255).astype(np.uint8)).save(out / "physics_last.png")

    C.save_json(out / "stats.json", stats)
    print(f"[s8] done. twin PSNR {stats['psnr_twin_vs_photo']:.2f} dB, "
          f"video {len(dyn['frames'])} frames")


if __name__ == "__main__":
    main()
