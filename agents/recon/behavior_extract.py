"""PointWorld-BEHAVIOR episodes -> emulated ScanNet++-style scene dir.

Collects the posed initial RGB frame of every clip (all requested cameras,
across episodes of one task) into the shared recon-scene contract that the
rest of the pipeline already understands:

  <root>/data/<scene>/dslr/{resized_undistorted_images/, nerfstudio/
      transforms_undistorted.json, colmap/images.txt}
  <root>/data/<scene>/init_points.ply          xyz+rgb from backprojected GT depth
  <root>/data/<scene>/gt/depth/<frame>.png     uint16 mm GT depth per kept frame
  <root>/data/<scene>/gt/<mesh>.ply            per-mesh LOCAL-frame points (<=50k)
  <root>/data/<scene>/gt/gt_objects.json       per-mesh first pose + counts

Pose conventions (verified on task-0000/episode_00000010 against the
released scene_mesh points, overlap test in the world frame):
  * per-clip 'extrinsic' is w2c with OpenCV camera axes, expressed in the
    clip's ROBOT frame (base_pose is identity at clip start);
  * 'world_to_robot' maps global world -> robot frame, so the global w2c
    written to colmap/images.txt is  extrinsic @ world_to_robot;
  * the global world frame is already metric, z-up, floor at z=0
    (1st-percentile z of fused depth = 0.000) - no metricize step needed.
  * scene_mesh_trajectories poses (xyz qx qy qz qw) live in the same
    per-clip robot frame; static furniture mapped to the global frame agrees
    across clips AND episodes to <1e-4 m, so episodes of one task are the
    same scene (dynamic objects DO start at different poses per episode -
    keep --episodes 1 if ghosting of the task object hurts the splat).

Env: none of the three SimAny envs ships h5py; run with any python that has
h5py + numpy + cv2 (verified: ${SIMANY_H5_PY}),
from the repo root so `agents` resolves:

  ${SIMANY_H5_PY} -m agents.recon.behavior_extract \
      --task task-0000 --episodes 3 --static-only \
      --scene-name behavior_task0000 \
      --root ${SIMANY_ROOT}/data/recon_scenes
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from agents.core.common import quat_to_rot_wxyz, save_json

REPO_ROOT = Path(__file__).resolve().parents[2]
FLOWS_ROOT = Path(os.environ.get(
    "SIMANY_BEHAVIOR_FLOWS_ROOT",
    REPO_ROOT / "data/pointworld_behavior_restored/behavior/flows"))

# camera_head has fx=90.67 while left/right share fx=136; the scene contract
# carries ONE PINHOLE camera (transforms_undistorted.json), so the default
# excludes head rather than silently mixing intrinsics.
DEFAULT_CAMERAS = "left,right"

# Depth gates for init_points backprojection: sub-5cm is robot self-hits /
# sensor noise, far tail is unreliable at 320x180.
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 8.0

INIT_POINTS_CAP = 2_000_000   # mirrors the vggt_scene.py fused-cloud cap
GT_MESH_PTS_CAP = 50_000      # per-mesh cap for the generation-gap eval
MAX_FRAMES = 240              # uniform subsample above this (gsplat train cost)

# Static furniture reproduces across clips to ~0; anything drifting more
# than this in the global frame is flagged non-static in the manifest.
STATIC_DRIFT_M = 0.02


def need(grp, key: str, path: str):
    """Schema-drift guard: fail loudly listing what WAS found."""
    if key not in grp:
        raise SystemExit(f"[behavior_extract] missing '{key}' under {path}; "
                         f"found: {sorted(grp.keys())}")
    return grp[key]


def pose7_to_mat(p7):
    """BEHAVIOR pose vector [x y z qx qy qz qw] -> 4x4."""
    p7 = np.asarray(p7, dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = quat_to_rot_wxyz(p7[[6, 3, 4, 5]])
    M[:3, 3] = p7[:3]
    return M


def rot_to_quat_wxyz_np(R):
    """R -> quat wxyz without utils3d (not present in the h5py env)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12)) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def jpeg_bytes(ds):
    """initial_rgb is a (1,) object dataset of encoded JPEG bytes; tolerate
    raw uint8 image arrays too (re-encode) for schema drift."""
    import cv2
    a = ds[()]
    if getattr(a, "dtype", None) == np.uint8 and getattr(a, "ndim", 0) == 3:
        ok, buf = cv2.imencode(".jpg", a[..., ::-1],
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buf.tobytes()
    raw = a[0] if getattr(a, "shape", None) == (1,) else a
    return raw if isinstance(raw, bytes) else np.asarray(raw).tobytes()


def write_ply_xyzrgb(path: Path, pts, rgb):
    """Binary little-endian PLY, xyz float32 + rgb uchar (matches the
    init_points.ply the gsplat trainer and open3d both read)."""
    pts = np.asarray(pts, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    el = np.empty(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("red", "u1"), ("green", "u1"),
                                   ("blue", "u1")])
    el["x"], el["y"], el["z"] = pts.T
    el["red"], el["green"], el["blue"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(el.tobytes())


def mesh_short(key: str) -> str:
    """'World__scene_0__coffee_table_koagbh_0__base_link__visuals' ->
    'coffee_table_koagbh_0'; fall back to a sanitized full key."""
    parts = key.split("__")
    return parts[2] if len(parts) >= 3 else key.replace("/", "_")


def fuse_gt_mesh(scene_dir: Path):
    """TSDF-fuse the extracted GT depth pngs into gt/mesh_gt.ply.

    Reads only the emulated scene dir (no HDF5/h5py), so it can run in the
    main .venv where open3d lives: poses from dslr/colmap/images.txt via the
    pipeline's own parser, native K from gt/intrinsics_native.json, color
    from the (possibly upscaled) pipeline jpgs resized back to native."""
    import cv2
    import open3d as o3d
    from agents.core.common import load_colmap_w2c

    meta = json.loads((scene_dir / "gt" / "intrinsics_native.json").read_text())
    K = np.asarray(meta["K"], np.float64)
    w, h = int(meta["w"]), int(meta["h"])
    w2c_by_name = load_colmap_w2c(scene_dir / "dslr" / "colmap" / "images.txt")
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01, sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intr = o3d.camera.PinholeCameraIntrinsic(
        w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    n = 0
    for name, w2c in sorted(w2c_by_name.items()):
        dp = scene_dir / "gt" / "depth" / (Path(name).stem + ".png")
        ip = scene_dir / "dslr" / "resized_undistorted_images" / name
        if not dp.exists() or not ip.exists():
            continue
        depth_mm = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        bgr = cv2.resize(cv2.imread(str(ip)), (w, h),
                         interpolation=cv2.INTER_AREA)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(bgr[..., ::-1])),
            o3d.geometry.Image(depth_mm), depth_scale=1000.0,
            depth_trunc=DEPTH_MAX_M, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.asarray(w2c, np.float64))
        n += 1
    mesh = vol.extract_triangle_mesh()
    out = scene_dir / "gt" / "mesh_gt.ply"
    o3d.io.write_triangle_mesh(str(out), mesh)
    print(f"[behavior_extract] GT-depth TSDF mesh: {len(mesh.vertices)} "
          f"verts from {n} views -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="e.g. task-0000")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--static-only", action="store_true",
                    help="skip clips with any_object_moving=True")
    ap.add_argument("--scene-name", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--flows-root", default=str(FLOWS_ROOT))
    ap.add_argument("--cameras", default=DEFAULT_CAMERAS,
                    help="comma list of head/left/right (must share K)")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--fuse-only", action="store_true",
                    help="only TSDF-fuse gt/mesh_gt.ply from an already-"
                         "extracted scene dir (main .venv: needs open3d, "
                         "not h5py)")
    ap.add_argument("--upscale", type=int, default=0,
                    help="integer image upscale for the PIPELINE copies "
                         "(SAM3/TRELLIS need more pixels than the 320x180 "
                         "release: task-0000 pilot got 0 SAM3 proposals raw)."
                         " 0 = auto: smallest factor putting the long side "
                         ">= 1280, capped at 4. GT depth and the init cloud "
                         "stay at native resolution.")
    args = ap.parse_args()

    if args.fuse_only:
        fuse_gt_mesh(Path(args.root) / "data" / args.scene_name)
        return

    # heavy env-specific imports only for the extraction path: the
    # --fuse-only branch above must run in the open3d venv, which has no h5py
    import cv2  # noqa: F401  (used by jpeg_bytes/imwrite below)
    import h5py

    task_dir = Path(args.flows_root) / args.task
    eps = sorted(task_dir.glob("episode_*.hdf5"))
    if not eps:
        raise SystemExit(f"[behavior_extract] no episode_*.hdf5 in {task_dir}")
    eps = eps[:args.episodes]
    cams = [f"camera_{c.strip()}" for c in args.cameras.split(",") if c.strip()]

    frames = []      # dicts: name, jpg, depth_mm, w2c
    K_ref = None
    gt_meshes = {}   # full key -> {pts, rgb, T_first, lo, hi (drift track)}
    n_skipped_moving = 0

    for ep_path in eps:
        ep_id = ep_path.stem.replace("episode_", "ep")
        with h5py.File(ep_path, "r") as f:
            clip_keys = sorted(f.keys(), key=lambda s: int(s.split(":")[0]))
            if not clip_keys:
                raise SystemExit(f"[behavior_extract] no clip groups in "
                                 f"{ep_path}; found: {sorted(f.keys())}")
            for ck in clip_keys:
                clip = f[ck]
                if args.static_only and bool(
                        clip.attrs.get("any_object_moving", False)):
                    n_skipped_moving += 1
                    continue
                W2R = np.asarray(need(clip, "world_to_robot", ck),
                                 dtype=np.float64)
                R2W = np.linalg.inv(W2R)
                for cam in cams:
                    cg = need(clip, cam, f"{ep_path.name}:{ck}")
                    where = f"{ep_path.name}:{ck}/{cam}"
                    K = np.asarray(need(cg, "intrinsic", where),
                                   dtype=np.float64)
                    if K_ref is None:
                        K_ref = K
                    elif not np.allclose(K, K_ref, atol=1e-3):
                        raise SystemExit(
                            f"[behavior_extract] intrinsics mismatch at "
                            f"{where}: {K.flatten()[[0, 4, 2, 5]]} vs "
                            f"{K_ref.flatten()[[0, 4, 2, 5]]}; the scene "
                            f"contract has ONE camera - drop --cameras "
                            f"entries that differ (head vs left/right).")
                    # verified: extrinsic = w2c (OpenCV axes) in the clip's
                    # robot frame -> global w2c = extrinsic @ world_to_robot
                    E = np.asarray(need(cg, "extrinsic", where),
                                   dtype=np.float64)
                    frames.append({
                        "name": f"{ep_id}_c{int(ck.split(':')[0]):06d}"
                                f"_{cam.replace('camera_', '')}.jpg",
                        "jpg": jpeg_bytes(need(cg, "initial_rgb", where)),
                        "depth_mm": np.asarray(
                            need(cg, "initial_depth", where)),
                        "w2c": E @ W2R,
                    })
                # GT meshes: local points once, per-clip frame-0 global pose
                # for the drift check (first listed camera carries them all)
                where0 = f"{ep_path.name}:{ck}/{cams[0]}"
                cg0 = clip[cams[0]]
                traj = need(cg0, "scene_mesh_trajectories", where0)
                lpts = need(cg0, "local_scene_points", where0)
                lrgb = cg0.get("local_scene_colors")
                for mk in traj:
                    Tg = R2W @ pose7_to_mat(np.asarray(traj[mk])[0])
                    rec = gt_meshes.get(mk)
                    if rec is None:
                        p = np.asarray(lpts[mk], dtype=np.float32) \
                            if mk in lpts else np.zeros((0, 3), np.float32)
                        c = (np.asarray(lrgb[mk], dtype=np.uint8)
                             if lrgb is not None and mk in lrgb
                             else np.full((len(p), 3), 128, np.uint8))
                        gt_meshes[mk] = {"pts": p, "rgb": c, "T_first": Tg,
                                         "lo": Tg[:3, 3].copy(),
                                         "hi": Tg[:3, 3].copy()}
                    else:
                        rec["lo"] = np.minimum(rec["lo"], Tg[:3, 3])
                        rec["hi"] = np.maximum(rec["hi"], Tg[:3, 3])

    if not frames:
        raise SystemExit(f"[behavior_extract] 0 frames collected from "
                         f"{len(eps)} episodes (skipped {n_skipped_moving} "
                         f"moving clips) - relax --static-only?")
    if len(frames) > args.max_frames:
        keep = np.linspace(0, len(frames) - 1, args.max_frames).astype(int)
        frames = [frames[i] for i in sorted(set(keep.tolist()))]

    h, w = frames[0]["depth_mm"].shape
    u = args.upscale or min(4, max(1, -(-1280 // max(w, h))))
    K_up = K_ref.copy()
    K_up[:2] *= u
    scene_dir = Path(args.root) / "data" / args.scene_name
    img_dir = scene_dir / "dslr" / "resized_undistorted_images"
    gt_depth_dir = scene_dir / "gt" / "depth"
    for d in (img_dir, scene_dir / "dslr" / "nerfstudio",
              scene_dir / "dslr" / "colmap", gt_depth_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- images (raw JPEG bytes, no re-encode) + GT depth + colmap poses --
    lines = ["# emulated colmap images.txt written by behavior_extract",
             f"# task={args.task} episodes={[e.name for e in eps]}"]
    all_pts, all_rgb = [], []
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    for i, fr in enumerate(frames):
        bgr_native = cv2.imdecode(np.frombuffer(fr["jpg"], np.uint8),
                                  cv2.IMREAD_COLOR)
        if u > 1:
            big = cv2.resize(bgr_native, (w * u, h * u),
                             interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(img_dir / fr["name"]), big,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            (img_dir / fr["name"]).write_bytes(fr["jpg"])
        cv2.imwrite(str(gt_depth_dir / (Path(fr["name"]).stem + ".png")),
                    fr["depth_mm"])
        q = rot_to_quat_wxyz_np(fr["w2c"][:3, :3])
        t = fr["w2c"][:3, 3]
        lines.append(f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                     f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {fr['name']}")
        # points line must be NON-empty: load_colmap_w2c drops blank lines
        # then takes every other line, so a blank here would desync it
        lines.append("0.0 0.0 -1")
        # backproject GT depth -> init cloud
        d = fr["depth_mm"].astype(np.float64) / 1000.0
        m = (d > DEPTH_MIN_M) & (d < DEPTH_MAX_M)
        z = d[m]
        x = (uu[m] - K_ref[0, 2]) / K_ref[0, 0] * z
        y = (vv[m] - K_ref[1, 2]) / K_ref[1, 1] * z
        c2w = np.linalg.inv(fr["w2c"])
        all_pts.append((np.stack([x, y, z], 1) @ c2w[:3, :3].T
                        + c2w[:3, 3]).astype(np.float32))
        all_rgb.append(bgr_native[..., ::-1][m])
    (scene_dir / "dslr" / "colmap" / "images.txt").write_text(
        "\n".join(lines) + "\n")

    meta = {"fl_x": K_up[0, 0], "fl_y": K_up[1, 1],
            "cx": K_up[0, 2], "cy": K_up[1, 2], "w": w * u, "h": h * u,
            "camera_model": "PINHOLE",
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}
    save_json(scene_dir / "dslr" / "nerfstudio"
              / "transforms_undistorted.json", meta)

    pts = np.concatenate(all_pts)
    rgb = np.concatenate(all_rgb)
    if len(pts) > INIT_POINTS_CAP:
        sel = np.random.default_rng(0).choice(len(pts), INIT_POINTS_CAP,
                                              replace=False)
        pts, rgb = pts[sel], rgb[sel]
    write_ply_xyzrgb(scene_dir / "init_points.ply", pts, rgb)

    # ---- GT object manifest for the generation-gap eval -------------------
    manifest = {
        "frame": "global BEHAVIOR world (metric, z-up, floor z=0); "
                 "T_first maps mesh-local points to it",
        "task": args.task, "episodes": [e.name for e in eps],
        "cameras": cams, "static_only": bool(args.static_only),
        "n_frames": len(frames), "n_clips_skipped_moving": n_skipped_moving,
        "intrinsics": meta, "meshes": {},
    }
    rng = np.random.default_rng(0)
    for mk, rec in sorted(gt_meshes.items()):
        short = mesh_short(mk)
        p, c = rec["pts"], rec["rgb"]
        if len(p) > GT_MESH_PTS_CAP:
            sel = rng.choice(len(p), GT_MESH_PTS_CAP, replace=False)
            p, c = p[sel], c[sel]
        write_ply_xyzrgb(scene_dir / "gt" / f"{short}.ply", p, c)
        drift = float(np.linalg.norm(rec["hi"] - rec["lo"]))
        q = rot_to_quat_wxyz_np(rec["T_first"][:3, :3])
        manifest["meshes"][short] = {
            "key": mk, "n_points": int(len(rec["pts"])),
            "n_points_saved": int(len(p)), "ply": f"gt/{short}.ply",
            "T_first": rec["T_first"],
            "pose_first": {"pos": rec["T_first"][:3, 3],
                           "quat_wxyz": q},
            "max_pos_drift_m": drift,
            "static": drift < STATIC_DRIFT_M,
        }
    save_json(scene_dir / "gt" / "gt_objects.json", manifest)

    # Native intrinsics sidecar for --fuse-only (transforms carries the
    # UPSCALED K; the GT depth pngs stay native)
    save_json(scene_dir / "gt" / "intrinsics_native.json",
              {"K": K_ref.tolist(), "w": w, "h": h})
    # ORIGINAL mesh: TSDF-fuse the release's GT depth (the simulator's own
    # geometry). open3d lives in the main .venv, not in the h5py env this
    # extraction usually runs under - fall back to the --fuse-only pass.
    try:
        fuse_gt_mesh(scene_dir)
    except ImportError:
        print("[behavior_extract] open3d not in this env - run "
              "`run agents.recon.behavior_extract --fuse-only "
              "--scene-name ... --root ...` in the main venv")

    print(f"[behavior_extract] {scene_dir}: {len(frames)} frames "
          f"({n_skipped_moving} moving clips skipped), "
          f"init_points={len(pts)}, gt_meshes={len(gt_meshes)}, "
          f"K=({K_ref[0, 0]:.2f},{K_ref[1, 1]:.2f},{K_ref[0, 2]:.1f},"
          f"{K_ref[1, 2]:.1f}) {w}x{h} -> pipeline images x{u} "
          f"= {w * u}x{h * u}")


if __name__ == "__main__":
    main()
