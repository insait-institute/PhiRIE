"""Mini-viewer env, one-shot: render N_VIEWS small-disparity synthetic views
of the SHARP splat per object, for ReconViaGen's multi-view conditioning.

Camera math is SHARP's own validated near-field demo trajectory
(sharp/utils/camera.py: compute_max_offset + create_eye_trajectory_rotate +
PinholeCameraModel.compute, all defaults) PORTED INLINE as plain numpy --
the `sharp` package is not installed in this env (mini-viewer venv has no
`click`/`sharp`), so rather than fight that import we hand-port the small
amount of math actually needed, verified against the source at
${SHARP_ROOT}/repo/src/sharp/utils/camera.py:53-71,155-176,233-345.
This is a deliberate, documented deviation from "import sharp.utils.camera
directly" -- the formulas are copied faithfully, not reinvented.

Per object: 12 eye positions on a small ellipse (max_disparity=0.08,
max_zoom=0.15, SHARP's own library defaults), each looking at that object's
own centroid (lookat_mode="point" semantics: look_at = centroid, since
origin=zeros(3) in that mode). A pixel is "object" in a given view if it
falls in the object's projected+padded 2D bbox AND alpha>0.5 AND rendered
depth lies within the object's own (per-view) depth band +- its AABB
extent as margin -- a depth-band approximation standing in for
factory_prepare.py's real occlusion raycast (no separate instance mesh
exists here to raycast against; this is noted as an approximation, not a
bug, per spec).

Writes obj_XX/rvg/view_NN.png (RGBA crops), obj_XX/rvg/_debug_views.png
(contact sheet), obj_XX/rvg/.skip_rvg marker if <2 usable views survive.

NOTE (2nd pass): sharp_s2_lift.py now writes obj_XX/points.ply rotated into
an approximate z-up world frame (sharp_ply_meta.R_ZUP) for s5_align.py's
benefit. This script's own camera math (create_camera_w2c/render_view/K)
all operates in the ORIGINAL SHARP camera frame (same as meta.json's
centroid, deliberately left unrotated), so points.ply is un-rotated back
via `@ R_ZUP` (R_ZUP is orthogonal, so right-multiplying by R_ZUP is its
own frame-inverse of the `@ R_ZUP.T` used to build it) immediately after
loading, before any projection -- otherwise the object's own point cloud
would be misaligned with this script's camera poses/masking.
"""
import os

import numpy as np
from PIL import Image, ImageDraw

from agents.core import common as C
from agents.single_image.sharp_ply_meta import R_ZUP, read_camera

N_VIEWS = 12
MAX_DISPARITY = 0.08
MAX_ZOOM = 0.15
MIN_MASK_PX = 400
MIN_PROJ_PTS = 3


def create_camera_w2c(eye_pos, look_at, world_up=np.array([0.0, -1.0, 0.0])):
    """Port of sharp.utils.camera.create_camera_matrix(..., inverse=True)
    composed with PinholeCameraModel.compute's lookat_mode="point" (origin
    = zeros(3)) -- OpenCV w2c convention (x right, y down, z forward)."""
    front = look_at - eye_pos
    front = front / np.linalg.norm(front)
    right = np.cross(front, world_up)
    right = right / np.linalg.norm(right)
    down = np.cross(front, right)
    R_c2w = np.stack([right, down, front], axis=1)  # columns = camera axes in world
    R_w2c = R_c2w.T
    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = -R_w2c @ eye_pos
    return w2c


def compute_max_offset(means_np, W, H, f_px, max_disparity=MAX_DISPARITY,
                       max_zoom=MAX_ZOOM):
    """Port of sharp.utils.camera.compute_max_offset (extrinsics=eye(4),
    q_near=0.001 quantile of the WHOLE scene's own depth distribution)."""
    z = means_np[:, 2]
    z = z[z > 0]
    min_depth = float(np.quantile(z, 0.001))
    diagonal = np.sqrt((W / f_px) ** 2 + (H / f_px) ** 2)
    max_lateral = max_disparity * diagonal * min_depth
    max_medial = max_zoom * min_depth
    return np.array([max_lateral, max_lateral, max_medial]), min_depth


def eye_trajectory_rotate(offset_xyz, distance_m=0.0, num_steps=N_VIEWS, num_repeats=1):
    """Port of sharp.utils.camera.create_eye_trajectory_rotate."""
    ox, oy, _ = offset_xyz
    ts = np.linspace(0, num_repeats, num_steps * num_repeats, endpoint=False)
    return np.stack([ox * np.sin(2 * np.pi * ts),
                     oy * np.cos(2 * np.pi * ts),
                     np.full_like(ts, distance_m)], axis=1)


def project(pts, K, w2c):
    pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    u = pc[:, 0] / np.clip(z, 1e-9, None) * K[0, 0] + K[0, 2]
    v = pc[:, 1] / np.clip(z, 1e-9, None) * K[1, 1] + K[1, 2]
    return u, v, z


def read_points_ply(path):
    """Plain xyz point cloud reader via plyfile -- the mini-viewer env has no
    open3d (deliberately, per derive_mesh_from_splat.py's docstring: gsplat
    and open3d don't reliably coexist in one env on this cluster), so read
    the points.ply written by open3d elsewhere (.venv) with plyfile instead."""
    from plyfile import PlyData

    v = PlyData.read(str(path))["vertex"]
    return np.stack([np.asarray(v[c], dtype=np.float64) for c in ("x", "y", "z")],
                    axis=1)


def read_points_ply_cam_frame(path):
    """points.ply is stored z-up-rotated (R_ZUP) by sharp_s2_lift.py; undo
    that here since this script's projections need the original camera
    frame (see module docstring)."""
    return read_points_ply(path) @ R_ZUP


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=None,
                    help="comma-separated obj indices to (re)generate views "
                         "for; default: all objects in objects.json")
    args = ap.parse_args()
    picks = None if args.objects is None \
        else {int(x) for x in args.objects.split(",")}

    ply_path = os.path.join(os.environ.get("SHARP_ROOT", "third_party/sharp"),
                            "outputs/DSC08561/DSC08561.ply")
    f_px, W, H = read_camera(ply_path)
    K = np.array([[f_px, 0.0, W / 2.0], [0.0, f_px, H / 2.0], [0.0, 0.0, 1.0]])

    gs = C.load_gaussians(ply_path=ply_path, device="cuda")
    means_np = gs["means"].detach().cpu().numpy().astype(np.float64)

    offset_xyz, min_depth = compute_max_offset(means_np, W, H, f_px)
    eyes = eye_trajectory_rotate(offset_xyz, distance_m=0.0, num_steps=N_VIEWS)
    print(f"[rv] scene min_depth(q=0.001)={min_depth:.3f}m offset_xyz="
          f"{np.round(offset_xyz, 4).tolist()} ({N_VIEWS} eye positions)")

    import json
    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    if picks is not None:
        objects = [m for m in objects if m["index"] in picks]

    summary = []
    for meta in objects:
        idx = meta["index"]
        odir = C.OUT / "objects" / f"obj_{idx:02d}"
        rdir = odir / "rvg"
        rdir.mkdir(parents=True, exist_ok=True)

        centroid = np.asarray(meta["centroid"], dtype=np.float64)
        margin = float(np.max(meta["extent"]))
        obj_pts = read_points_ply_cam_frame(odir / "points.ply")

        n_ok = 0
        crops = []
        for k, eye in enumerate(eyes):
            w2c_k = create_camera_w2c(eye, centroid)
            rgb, depth, alpha = C.render_view(gs, w2c_k, K, W, H, render_mode="RGB+ED")

            u, v, z = project(obj_pts, K, w2c_k)
            inb = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if inb.sum() < MIN_PROJ_PTS:
                print(f"[rv] obj_{idx:02d} view {k:02d}: only {int(inb.sum())} "
                      "object points project in-frame, skip")
                continue
            uu, vv, zz = u[inb], v[inb], z[inb]
            pad = int(0.15 * max(uu.max() - uu.min(), vv.max() - vv.min()) + 8)
            u0, u1 = int(max(uu.min() - pad, 0)), int(min(uu.max() + pad, W))
            v0, v1 = int(max(vv.min() - pad, 0)), int(min(vv.max() + pad, H))
            if u1 <= u0 or v1 <= v0:
                continue

            zlo, zhi = zz.min() - margin, zz.max() + margin
            crop_alpha = alpha[v0:v1, u0:u1]
            crop_depth = depth[v0:v1, u0:u1]
            mask = (crop_alpha > 0.5) & (crop_depth > zlo) & (crop_depth < zhi)
            if mask.sum() < MIN_MASK_PX:
                print(f"[rv] obj_{idx:02d} view {k:02d}: mask {int(mask.sum())}px "
                      f"< {MIN_MASK_PX}, skip")
                continue

            rgb_crop = (np.clip(rgb[v0:v1, u0:u1], 0, 1) * 255).astype(np.uint8)
            rgba = np.dstack([rgb_crop, (mask * 255).astype(np.uint8)])
            out_path = rdir / f"view_{n_ok:02d}.png"
            Image.fromarray(rgba).save(out_path)
            crops.append(rgba)
            n_ok += 1

        print(f"[rv] obj_{idx:02d} {meta['label']}: {n_ok}/{N_VIEWS} usable views")
        summary.append({"index": idx, "label": meta["label"], "n_views": n_ok})

        if crops:
            cell = 160
            cols = 4
            rows = (len(crops) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * cell, rows * cell), (60, 60, 60))
            draw = ImageDraw.Draw(sheet)
            for k, c in enumerate(crops):
                im = Image.fromarray(c).convert("RGBA")
                bg = Image.new("RGB", im.size, (30, 30, 30))
                bg.paste(im, mask=im.split()[3])
                bg.thumbnail((cell - 10, cell - 10))
                r, cidx = divmod(k, cols)
                x, y = cidx * cell, r * cell
                sheet.paste(bg, (x + 5, y + 5))
                draw.text((x + 5, y + cell - 12), f"v{k:02d}", fill=(255, 255, 0))
            sheet.save(rdir / "_debug_views.png")

        skip_marker = rdir / ".skip_rvg"
        if n_ok < 2:
            skip_marker.write_text(
                f"only {n_ok} usable views (<2) -- RVG skipped, TRELLIS auto-wins\n")
            print(f"[rv] obj_{idx:02d}: <2 usable views, wrote .skip_rvg marker")
        elif skip_marker.exists():
            # a rerun (e.g. after the frame fix) now has enough views --
            # clear a stale marker from an earlier pass so sharp_hybrid.py
            # doesn't wrongly skip this object.
            skip_marker.unlink()
            print(f"[rv] obj_{idx:02d}: cleared stale .skip_rvg marker")

    # merge with any prior summary so a --objects subset rerun doesn't lose
    # the other objects' entries
    summary_path = C.OUT / "objects" / "rvg_views_summary.json"
    if picks is not None and summary_path.exists():
        prior = {r["index"]: r for r in json.loads(summary_path.read_text())}
        for r in summary:
            prior[r["index"]] = r
        summary = sorted(prior.values(), key=lambda r: r["index"])
    C.save_json(summary_path, summary)
    n_skip = sum(1 for s in summary if s["n_views"] < 2)
    print(f"[rv] DONE: {len(summary)} objects total, {n_skip} with <2 usable views")


if __name__ == "__main__":
    main()
