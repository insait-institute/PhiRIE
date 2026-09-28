"""oracle/capture_generator.py -- Task 14 capability #1: controlled, phone-like
capture generation for the oracle_causal protocol.

Two subcommands:

  extract   Pull posed RGB-D clips for one BEHAVIOR task directly from the
            PointWorld-BEHAVIOR WebDataset shards into the same emulated
            ScanNet++-style scene-dir contract that
            `agents/recon/behavior_extract.py` produces, so the existing
            recon pipeline tail (gsplat_train -> AUTO stages, invoked via
            run/run_behavior_recon.sh) runs against it unmodified.

            WHY NOT behavior_extract.py's own HDF5 path: its source
            (`${SIMANY_ROOT}/data/pointworld_behavior_restored/behavior/
            flows/`) no longer exists on disk (verified 2026-08-16: `ls`
            returns nothing; only `${SIMANY_ROOT}/data/behavior/wds/`
            (WebDataset shards) and the one already-extracted
            `data/recon_scenes/data/behavior_task0020/` scene dir survive
            from before the restored HDF5 tree was reclaimed).

            CROSS-CLIP ANCHORING GOTCHA (found empirically 2026-08-16, not
            in any memory note): behavior_extract.py's HDF5 path reads a
            per-clip `world_to_robot` 4x4 that anchors that clip into a
            scene-global frame shared across every clip/episode of the
            task. WDS does NOT carry that field -- it carries `base_pose`
            (T,7), the robot-base trajectory *within* the clip, and by the
            release's own convention `base_pose[0]` is identity for EVERY
            clip (verified: printing it for 6+ clips across 2 tasks always
            gives ~0 translation / ~identity quat). That means
            `inverse(pose7_to_mat(base_pose[0]))` is always ~identity too --
            it carries NO cross-clip anchoring information, unlike what an
            earlier version of this module assumed. Concretely: the SAME
            static mesh (e.g. a floor) comes out at wildly different
            positions in different clips of the same episode when frames
            are merged naively using only `extrinsic`/`base_pose` (verified:
            [2.09, 2.14] vs [-2.22, 2.50] for `floors_nbxnpk_0` in two clips
            of the same task-0016 episode).

            FIX: `_register_clips_to_reference` recovers the missing anchor
            the same way it would have to be recovered from any noisy real
            capture -- by rigidly aligning (Kabsch/SVD, RANSAC-lite outlier
            rejection) each clip's LOCAL scene-mesh landmark positions
            (`scene_mesh_trajectories[key][0]`) against a reference clip's,
            using whichever mesh keys the two clips have in common. Task
            objects that genuinely moved between episodes are automatically
            down-weighted by the one-shot outlier-rejection refit (their
            large residual excludes them from the landmark set), so the
            registration lands on the fixtures, not the manipulanda.
            Clips sharing fewer than 3 landmarks with the reference are
            dropped rather than guessed at.

  degrade   THE new capability plan/14 actually asks for: apply controlled,
            metadata-logged degradation (view-count subsampling, 6DoF pose
            noise, Gaussian blur, depth noise, synthetic occlusion patches)
            to an already-extracted CLEAN scene dir (either the one
            `extract` just wrote, or the pre-existing behavior_task0020),
            producing a new scene dir the same pipeline can be pointed at
            unchanged. GT (`gt/gt_objects.json` + per-object `gt/<mesh>.ply`)
            is copied through byte-identical -- degradation acts only on
            the simulated phone CAPTURE, never on the ground truth (that
            separation is what oracle/gt_export.py's access-check verifies).

Usage:
  .venv/bin/python -m oracle.capture_generator extract \
      --task task-0011 --scene-name behavior_task0011 \
      --root data/recon_scenes --episodes 4 \
      --tasks-config configs/oracle/tasks.yaml

  .venv/bin/python -m oracle.capture_generator degrade \
      --scene-name behavior_task0011 --level mild \
      --root data/recon_scenes --out-scene-name behavior_task0011_mild \
      --tasks-config configs/oracle/tasks.yaml
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from agents.core.common import save_json
from agents.recon.behavior_extract import (
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    GT_MESH_PTS_CAP,
    INIT_POINTS_CAP,
    MAX_FRAMES,
    STATIC_DRIFT_M,
    fuse_gt_mesh,
    mesh_short,
    pose7_to_mat,
    rot_to_quat_wxyz_np,
    write_ply_xyzrgb,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WDS_ROOT = str(ROOT / "data" / "behavior" / "wds")
DEFAULT_CAMERAS = "left,right"
MIN_LANDMARKS_FOR_REGISTRATION = 3
REGISTRATION_INLIER_TOL_M = 0.05


# ---------------------------------------------------------------------------
# WDS shard indexing (the HDF5 source `behavior_extract.py` expects is gone;
# see module docstring). `find_task_shards` uses the tasks.yaml hints first
# (fast) and falls back to a bounded directory scan (slow, ~1.6s/shard) so
# this still works for a task id nobody has indexed yet.
# ---------------------------------------------------------------------------

def _load_hints(tasks_config: str | None) -> dict:
    if not tasks_config or not Path(tasks_config).exists():
        return {}
    cfg = yaml.safe_load(Path(tasks_config).read_text()) or {}
    return cfg.get("wds_shard_hints", {})


def find_task_shards(task_id: str, wds_root: str, tasks_config: str | None,
                      splits=("train", "test"), max_scan: int = 80) -> list[Path]:
    wds_root = Path(wds_root)
    hints = _load_hints(tasks_config)
    found = []
    hint = hints.get(task_id)
    if hint:
        p = wds_root / hint
        if p.exists() and _shard_has_task(p, task_id):
            found.append(p)
    if found:
        return found
    print(f"[capture_generator] no valid shard hint for {task_id}; scanning "
          f"up to {max_scan} shards under {wds_root} (slow path)")
    scanned = 0
    for split in splits:
        for shard in sorted((wds_root / split).glob("*.tar")):
            if scanned >= max_scan:
                break
            scanned += 1
            if _shard_has_task(shard, task_id):
                found.append(shard)
                if len(found) >= 3:
                    return found
    if not found:
        raise SystemExit(f"[capture_generator] task {task_id} not found in "
                         f"first {max_scan} shards scanned under {wds_root}; "
                         f"raise --max-shard-scan or add a wds_shard_hints "
                         f"entry once you locate it by hand.")
    return found


def _shard_has_task(shard: Path, task_id: str) -> bool:
    try:
        with tarfile.open(shard) as t:
            for n in t.getnames():
                if n.startswith(task_id + "_"):
                    return True
    except Exception as e:
        print(f"[capture_generator] warn: failed to open {shard}: {e}")
    return False


def _clip_ids_for_task(shard: Path, task_id: str) -> list[str]:
    with tarfile.open(shard) as t:
        names = t.getnames()
    ids = sorted({n.split(".")[0] for n in names if n.startswith(task_id + "_")})
    return ids


def _load_npy(tf: tarfile.TarFile, clip_id: str, field: str):
    return np.load(io.BytesIO(tf.extractfile(f"{clip_id}.{field}.npy").read()))


def _load_pyd(tf: tarfile.TarFile, clip_id: str, field: str):
    return pickle.loads(tf.extractfile(f"{clip_id}.{field}.pyd").read())


def _load_jpg_bytes(tf: tarfile.TarFile, clip_id: str, field: str) -> bytes:
    return tf.extractfile(f"{clip_id}.{field}.jpg").read()


# ---------------------------------------------------------------------------
# Cross-clip registration (Kabsch/SVD on shared scene-mesh landmarks; see
# module docstring "CROSS-CLIP ANCHORING GOTCHA")
# ---------------------------------------------------------------------------

def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid R,t minimizing sum ||R@P_i + t - Q_i||^2. P,Q: (N,3)."""
    cp, cq = P.mean(0), Q.mean(0)
    Pc, Qc = P - cp, Q - cq
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cq - R @ cp
    return R, t


def _register_clips_to_reference(clips: list[dict]) -> list[dict]:
    """clips: list of {clip_id, frames:[...], traj: {key: (T,7) array}}.

    Returns the subset of clips with a 4x4 `D` (clip-local -> shared global
    frame) attached, dropping any clip that shares too few landmark meshes
    with the chosen reference. The reference clip (most landmark keys) gets
    D = identity.
    """
    if not clips:
        return clips
    ref = max(clips, key=lambda c: len(c["traj"]))
    ref_pos = {k: np.asarray(v)[0][:3] for k, v in ref["traj"].items()}
    ref["D"] = np.eye(4)
    kept = [ref]
    n_dropped = 0
    for c in clips:
        if c is ref:
            continue
        c_pos = {k: np.asarray(v)[0][:3] for k, v in c["traj"].items()}
        shared = sorted(set(c_pos) & set(ref_pos))
        if len(shared) < MIN_LANDMARKS_FOR_REGISTRATION:
            n_dropped += 1
            continue
        P = np.stack([c_pos[k] for k in shared])
        Q = np.stack([ref_pos[k] for k in shared])
        R, t = _kabsch(P, Q)
        resid = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        inliers = resid < max(REGISTRATION_INLIER_TOL_M, 2 * np.median(resid))
        if inliers.sum() >= MIN_LANDMARKS_FOR_REGISTRATION:
            R, t = _kabsch(P[inliers], Q[inliers])
        D = np.eye(4)
        D[:3, :3], D[:3, 3] = R, t
        c["D"] = D
        c["n_landmarks"] = len(shared)
        c["n_landmark_inliers"] = int(inliers.sum())
        kept.append(c)
    if n_dropped:
        print(f"[capture_generator] registration: dropped {n_dropped}/"
              f"{len(clips)} clips sharing <{MIN_LANDMARKS_FOR_REGISTRATION} "
              f"landmark meshes with the reference clip {ref['clip_id']}")
    return kept


# ---------------------------------------------------------------------------
# extract: WDS clips -> emulated scene dir (same contract as behavior_extract)
# ---------------------------------------------------------------------------

def extract_task(task_id: str, scene_name: str, root: str, episodes: int = 4,
                  static_only: bool = True, cameras: str = DEFAULT_CAMERAS,
                  max_frames: int = MAX_FRAMES, wds_root: str = DEFAULT_WDS_ROOT,
                  tasks_config: str | None = None, upscale: int = 0) -> Path:
    cams = [c.strip() for c in cameras.split(",") if c.strip()]
    shards = find_task_shards(task_id, wds_root, tasks_config)
    print(f"[capture_generator] {task_id}: using shard(s) {[str(s) for s in shards]}")

    # ---- pass 1: collect raw per-clip data (no cross-clip frame yet) ------
    raw_clips = []   # dicts: clip_id, cam_frames:[{cam, K, E, depth_mm, jpg}],
                      # traj:{key:(T,7)}, lpts, lrgb
    K_ref = None
    n_skipped_moving = 0
    n_episodes_used = 0

    for shard in shards:
        if n_episodes_used >= episodes:
            break
        with tarfile.open(shard) as tf:
            clip_ids = _clip_ids_for_task(shard, task_id)
            eps = {}
            for cid in clip_ids:
                ep = cid.split("_episode_")[1].split("-")[0]
                eps.setdefault(ep, []).append(cid)
            for ep, clip_list in sorted(eps.items()):
                if n_episodes_used >= episodes:
                    break
                n_episodes_used += 1
                for cid in sorted(clip_list):
                    attrs = _load_pyd(tf, cid, "clip_attributes")
                    if static_only and bool(attrs.get("any_object_moving", False)):
                        n_skipped_moving += 1
                        continue
                    cam_frames = []
                    for cam in cams:
                        pref = f"camera_{cam}"
                        K = _load_npy(tf, cid, f"{pref}_intrinsic")
                        if K_ref is None:
                            K_ref = K
                        elif not np.allclose(K, K_ref, atol=1e-3):
                            raise SystemExit(
                                f"[capture_generator] intrinsics mismatch at "
                                f"{cid}/{cam}: {K.flatten()[[0, 4, 2, 5]]} vs "
                                f"{K_ref.flatten()[[0, 4, 2, 5]]}; drop the "
                                f"differing --cameras entry (head vs left/right).")
                        E = _load_npy(tf, cid, f"{pref}_extrinsic")
                        depth_mm = _load_npy(tf, cid, f"{pref}_initial_depth")
                        jpg = _load_jpg_bytes(tf, cid, f"{pref}_initial_rgb")
                        cam_frames.append({"cam": cam, "E": E, "depth_mm": depth_mm, "jpg": jpg})
                    pref0 = f"camera_{cams[0]}"
                    traj = _load_pyd(tf, cid, f"{pref0}_scene_mesh_trajectories")
                    try:
                        lpts = _load_pyd(tf, cid, f"{pref0}_local_scene_points")
                    except KeyError:
                        lpts = {}
                    try:
                        lrgb = _load_pyd(tf, cid, f"{pref0}_local_scene_colors")
                    except KeyError:
                        lrgb = {}
                    raw_clips.append({"clip_id": cid, "ep": ep, "cam_frames": cam_frames,
                                       "traj": traj, "lpts": lpts, "lrgb": lrgb})

    if not raw_clips:
        raise SystemExit(f"[capture_generator] 0 clips collected for "
                         f"{task_id} ({n_skipped_moving} moving clips "
                         f"skipped) - relax static_only or raise episodes?")

    # ---- pass 2: register clips into one shared frame, then merge --------
    registered = _register_clips_to_reference(raw_clips)
    frames = []
    gt_meshes = {}
    for c in registered:
        D = c["D"]
        Dinv = np.linalg.inv(D)
        for i, cf in enumerate(c["cam_frames"]):
            frames.append({
                "name": f"{c['ep']}_{c['clip_id'].split('-')[-1].replace(':', '_')}"
                        f"_{cf['cam']}.jpg",
                "jpg": cf["jpg"], "depth_mm": cf["depth_mm"], "w2c": cf["E"] @ Dinv,
            })
        for mk, tr in c["traj"].items():
            Tg = D @ pose7_to_mat(np.asarray(tr)[0])
            rec = gt_meshes.get(mk)
            if rec is None:
                p = np.asarray(c["lpts"][mk], dtype=np.float32) if mk in c["lpts"] \
                    else np.zeros((0, 3), np.float32)
                cc = (np.asarray(c["lrgb"][mk], dtype=np.uint8)
                      if mk in c["lrgb"] else np.full((len(p), 3), 128, np.uint8))
                gt_meshes[mk] = {"pts": p, "rgb": cc, "T_first": Tg,
                                  "lo": Tg[:3, 3].copy(), "hi": Tg[:3, 3].copy()}
            else:
                rec["lo"] = np.minimum(rec["lo"], Tg[:3, 3])
                rec["hi"] = np.maximum(rec["hi"], Tg[:3, 3])

    if len(frames) > max_frames:
        keep = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in sorted(set(keep.tolist()))]

    h, w = frames[0]["depth_mm"].shape
    u = upscale or min(4, max(1, -(-1280 // max(w, h))))
    K_up = K_ref.copy()
    K_up[:2] *= u
    scene_dir = Path(root) / "data" / scene_name
    img_dir = scene_dir / "dslr" / "resized_undistorted_images"
    gt_depth_dir = scene_dir / "gt" / "depth"
    for d in (img_dir, scene_dir / "dslr" / "nerfstudio",
              scene_dir / "dslr" / "colmap", gt_depth_dir):
        d.mkdir(parents=True, exist_ok=True)

    lines = ["# emulated colmap images.txt written by oracle.capture_generator",
             f"# task={task_id} source=WDS episodes_used={n_episodes_used} "
             f"clips_registered={len(registered)}/{len(raw_clips)}"]
    all_pts, all_rgb = [], []
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    for i, fr in enumerate(frames):
        bgr_native = cv2.imdecode(np.frombuffer(fr["jpg"], np.uint8), cv2.IMREAD_COLOR)
        if u > 1:
            big = cv2.resize(bgr_native, (w * u, h * u), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(img_dir / fr["name"]), big, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            (img_dir / fr["name"]).write_bytes(fr["jpg"])
        cv2.imwrite(str(gt_depth_dir / (Path(fr["name"]).stem + ".png")), fr["depth_mm"])
        q = rot_to_quat_wxyz_np(fr["w2c"][:3, :3])
        t = fr["w2c"][:3, 3]
        lines.append(f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                     f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {fr['name']}")
        lines.append("0.0 0.0 -1")
        d = fr["depth_mm"].astype(np.float64) / 1000.0
        m = (d > DEPTH_MIN_M) & (d < DEPTH_MAX_M)
        z = d[m]
        x = (uu[m] - K_ref[0, 2]) / K_ref[0, 0] * z
        y = (vv[m] - K_ref[1, 2]) / K_ref[1, 1] * z
        c2w = np.linalg.inv(fr["w2c"])
        all_pts.append((np.stack([x, y, z], 1) @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32))
        all_rgb.append(bgr_native[..., ::-1][m])
    (scene_dir / "dslr" / "colmap" / "images.txt").write_text("\n".join(lines) + "\n")

    meta = {"fl_x": K_up[0, 0], "fl_y": K_up[1, 1], "cx": K_up[0, 2], "cy": K_up[1, 2],
            "w": w * u, "h": h * u, "camera_model": "PINHOLE",
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0}
    save_json(scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json", meta)

    pts = np.concatenate(all_pts)
    rgb = np.concatenate(all_rgb)
    if len(pts) > INIT_POINTS_CAP:
        sel = np.random.default_rng(0).choice(len(pts), INIT_POINTS_CAP, replace=False)
        pts, rgb = pts[sel], rgb[sel]
    write_ply_xyzrgb(scene_dir / "init_points.ply", pts, rgb)

    manifest = {
        "frame": "shared frame recovered by Kabsch-registering clip-local "
                 "scene-mesh landmarks to a reference clip (see module "
                 "docstring: WDS has no cross-clip anchor, unlike HDF5's "
                 "world_to_robot); T_first maps mesh-local points to it",
        "task": task_id, "source": "wds", "shards": [str(s) for s in shards],
        "episodes_used": n_episodes_used,
        "clips_registered": len(registered), "clips_total": len(raw_clips),
        "cameras": cams, "static_only": bool(static_only), "n_frames": len(frames),
        "n_clips_skipped_moving": n_skipped_moving, "intrinsics": meta,
        "meshes": {},
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
            "key": mk, "n_points": int(len(rec["pts"])), "n_points_saved": int(len(p)),
            "ply": f"gt/{short}.ply", "T_first": rec["T_first"],
            "pose_first": {"pos": rec["T_first"][:3, 3], "quat_wxyz": q},
            "max_pos_drift_m": drift, "static": drift < STATIC_DRIFT_M,
        }
    save_json(scene_dir / "gt" / "gt_objects.json", manifest)
    save_json(scene_dir / "gt" / "intrinsics_native.json", {"K": K_ref.tolist(), "w": w, "h": h})

    try:
        fuse_gt_mesh(scene_dir)
    except Exception as e:
        print(f"[capture_generator] GT-depth TSDF fuse failed/skipped: {e}")

    print(f"[capture_generator] {scene_dir}: {len(frames)} frames from "
          f"{n_episodes_used} episodes, {len(registered)}/{len(raw_clips)} "
          f"clips registered ({n_skipped_moving} moving clips skipped), "
          f"init_points={len(pts)}, gt_meshes={len(gt_meshes)}")
    return scene_dir


# ---------------------------------------------------------------------------
# degrade: clean scene dir -> controlled-degradation scene dir
# ---------------------------------------------------------------------------

DEFAULT_LEVELS = {
    "clean":  dict(view_keep_frac=1.00, pose_noise_pos_m=0.000, pose_noise_rot_deg=0.0,
                   blur_sigma_px=0.0, depth_noise_rel_std=0.000, occlusion_frac=0.00),
    "mild":   dict(view_keep_frac=0.70, pose_noise_pos_m=0.010, pose_noise_rot_deg=1.0,
                   blur_sigma_px=1.0, depth_noise_rel_std=0.010, occlusion_frac=0.05),
    "severe": dict(view_keep_frac=0.40, pose_noise_pos_m=0.030, pose_noise_rot_deg=4.0,
                   blur_sigma_px=3.0, depth_noise_rel_std=0.030, occlusion_frac=0.20),
}


def _load_levels(tasks_config: str | None) -> dict:
    if not tasks_config or not Path(tasks_config).exists():
        return DEFAULT_LEVELS
    cfg = yaml.safe_load(Path(tasks_config).read_text()) or {}
    return cfg.get("degradation_levels", DEFAULT_LEVELS)


def _small_rotation_matrix(rng: np.random.Generator, angle_deg_std: float) -> np.ndarray:
    """Random small-angle rotation: axis uniform on sphere, angle ~ N(0, std)."""
    if angle_deg_std <= 0:
        return np.eye(3)
    axis = rng.normal(size=3)
    axis /= (np.linalg.norm(axis) + 1e-12)
    angle = np.deg2rad(rng.normal(0.0, angle_deg_std))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def apply_degradation(scene_dir: str, out_dir: str, level: str, seed: int = 0,
                       tasks_config: str | None = None) -> Path:
    """Read a clean emulated scene dir, write a degraded copy at out_dir with
    the SAME contract (so run_behavior_recon.sh's `done_skip "$SD/dslr/colmap
    /images.txt"` treats it as already-extracted and skips straight past
    behavior_extract). GT (`gt/gt_objects.json`, `gt/<mesh>.ply`) is copied
    byte-identical; `gt/depth` + `gt/mesh_gt.ply` are CAPTURE-derived (a real
    phone with a depth sensor would have them) so they ARE regenerated from
    the degraded depth, consistent with degrading the capture, not the truth.
    """
    levels = _load_levels(tasks_config)
    if level not in levels:
        raise SystemExit(f"[capture_generator] unknown level '{level}'; have {list(levels)}")
    cfg = levels[level]
    rng = np.random.default_rng(seed)

    src = Path(scene_dir)
    dst = Path(out_dir)
    if dst.exists():
        shutil.rmtree(dst)
    for d in ("dslr/resized_undistorted_images", "dslr/nerfstudio", "dslr/colmap",
              "gt/depth"):
        (dst / d).mkdir(parents=True, exist_ok=True)

    from agents.core.common import load_colmap_w2c
    w2c_by_name = load_colmap_w2c(src / "dslr" / "colmap" / "images.txt")
    names = sorted(w2c_by_name.keys())
    n_keep = max(3, round(len(names) * cfg["view_keep_frac"]))
    keep_idx = sorted(rng.choice(len(names), size=min(n_keep, len(names)), replace=False))
    keep_names = [names[i] for i in keep_idx]

    meta = json.loads((src / "dslr" / "nerfstudio" / "transforms_undistorted.json").read_text())
    shutil.copy(src / "dslr" / "nerfstudio" / "transforms_undistorted.json",
                dst / "dslr" / "nerfstudio" / "transforms_undistorted.json")
    # GT depth pngs stay at NATIVE resolution (extract_task upscales only the
    # pipeline RGB copies for SAM3/TRELLIS -- see behavior_extract.py's
    # `--upscale` docstring), so backprojection needs the NATIVE K, not the
    # upscaled one `meta` carries. RGB/depth are therefore two different
    # pixel grids here; occlusion boxes are drawn in native space and scaled
    # up only when stamped onto the (larger) RGB frame.
    native = json.loads((src / "gt" / "intrinsics_native.json").read_text())
    K = np.asarray(native["K"], dtype=np.float64)
    nw, nh = int(native["w"]), int(native["h"])
    uu, vv = np.meshgrid(np.arange(nw) + 0.5, np.arange(nh) + 0.5)

    lines = ["# degraded (oracle.capture_generator) colmap images.txt",
             f"# source={src} level={level} seed={seed} params={cfg}"]
    per_frame_log = []
    all_pts, all_rgb = [], []
    for i, name in enumerate(keep_names):
        img_src = src / "dslr" / "resized_undistorted_images" / name
        depth_src = src / "gt" / "depth" / (Path(name).stem + ".png")
        img = cv2.imread(str(img_src))
        depth_mm = cv2.imread(str(depth_src), cv2.IMREAD_UNCHANGED)
        if img is None or depth_mm is None:
            continue
        ih, iw = img.shape[:2]
        sx, sy = iw / nw, ih / nh  # RGB upscale factor vs the native depth grid

        blur_sigma = cfg["blur_sigma_px"]
        if blur_sigma > 0:
            k = max(3, int(2 * round(3 * blur_sigma) + 1))
            img = cv2.GaussianBlur(img, (k, k), blur_sigma)

        depth_m = depth_mm.astype(np.float64) / 1000.0
        rel_std = cfg["depth_noise_rel_std"]
        if rel_std > 0:
            noise = rng.normal(0.0, rel_std * np.maximum(depth_m, 0.3), size=depth_m.shape)
            depth_m = np.clip(depth_m + noise, 0, None)

        occ_boxes = []
        occ_frac = cfg["occlusion_frac"]
        if occ_frac > 0:
            n_patches = max(1, round(occ_frac * 6))
            for _ in range(n_patches):
                pw = int(rng.uniform(0.08, 0.25) * nw)
                ph = int(rng.uniform(0.08, 0.25) * nh)
                x0 = int(rng.uniform(0, max(1, nw - pw)))
                y0 = int(rng.uniform(0, max(1, nh - ph)))
                depth_m[y0:y0 + ph, x0:x0 + pw] = 0.0
                ix0, iy0 = int(x0 * sx), int(y0 * sy)
                ipw, iph = max(1, int(pw * sx)), max(1, int(ph * sy))
                img[iy0:iy0 + iph, ix0:ix0 + ipw] = \
                    (img[iy0:iy0 + iph, ix0:ix0 + ipw] * 0.15).astype(np.uint8)
                occ_boxes.append([x0, y0, pw, ph])

        depth_mm_out = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

        w2c = w2c_by_name[name].copy()
        pos_noise = rng.normal(0.0, cfg["pose_noise_pos_m"], size=3)
        rot_noise = _small_rotation_matrix(rng, cfg["pose_noise_rot_deg"])
        c2w = np.linalg.inv(w2c)
        c2w_noisy = c2w.copy()
        c2w_noisy[:3, :3] = c2w[:3, :3] @ rot_noise
        c2w_noisy[:3, 3] = c2w[:3, 3] + pos_noise
        w2c_noisy = np.linalg.inv(c2w_noisy)

        out_name = name
        cv2.imwrite(str(dst / "dslr" / "resized_undistorted_images" / out_name), img)
        cv2.imwrite(str(dst / "gt" / "depth" / (Path(out_name).stem + ".png")), depth_mm_out)
        q = rot_to_quat_wxyz_np(w2c_noisy[:3, :3])
        t = w2c_noisy[:3, 3]
        lines.append(f"{i + 1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                     f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {out_name}")
        lines.append("0.0 0.0 -1")

        m = (depth_m > DEPTH_MIN_M) & (depth_m < DEPTH_MAX_M)
        z = depth_m[m]
        x = (uu[m] - K[0, 2]) / K[0, 0] * z
        y = (vv[m] - K[1, 2]) / K[1, 1] * z
        c2w_n = np.linalg.inv(w2c_noisy)
        all_pts.append((np.stack([x, y, z], 1) @ c2w_n[:3, :3].T + c2w_n[:3, 3]).astype(np.float32))
        img_native = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA) \
            if (ih, iw) != (nh, nw) else img
        all_rgb.append(img_native[..., ::-1][m])

        per_frame_log.append({
            "name": out_name,
            "pos_noise_m": pos_noise.tolist(),
            "rot_noise_deg_applied_std": cfg["pose_noise_rot_deg"],
            "occlusion_boxes_xywh": occ_boxes,
        })

    (dst / "dslr" / "colmap" / "images.txt").write_text("\n".join(lines) + "\n")

    if all_pts:
        pts = np.concatenate(all_pts)
        rgb = np.concatenate(all_rgb)
        if len(pts) > INIT_POINTS_CAP:
            sel = rng.choice(len(pts), INIT_POINTS_CAP, replace=False)
            pts, rgb = pts[sel], rgb[sel]
        write_ply_xyzrgb(dst / "init_points.ply", pts, rgb)

    for item in (src / "gt").glob("*"):
        if item.name in ("depth",):
            continue
        if item.is_dir():
            shutil.copytree(item, dst / "gt" / item.name, dirs_exist_ok=True)
        else:
            shutil.copy(item, dst / "gt" / item.name)

    try:
        fuse_gt_mesh(dst)
    except Exception as e:
        print(f"[capture_generator] degraded GT-depth TSDF fuse failed/skipped: {e}")

    save_json(dst / "degradation_manifest.json", {
        "source_scene_dir": str(src), "level": level, "params": cfg, "seed": int(seed),
        "n_frames_source": len(names), "n_frames_kept": len(keep_names),
        "kept_frame_names": keep_names, "per_frame": per_frame_log,
    })
    print(f"[capture_generator] {dst}: level={level} kept {len(keep_names)}/{len(names)} "
          f"views, params={cfg}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract")
    ex.add_argument("--task", required=True)
    ex.add_argument("--scene-name", required=True)
    ex.add_argument("--root", required=True)
    ex.add_argument("--episodes", type=int, default=4)
    ex.add_argument("--static-only", action="store_true", default=True)
    ex.add_argument("--cameras", default=DEFAULT_CAMERAS)
    ex.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ex.add_argument("--wds-root", default=DEFAULT_WDS_ROOT)
    ex.add_argument("--tasks-config", default=None)
    ex.add_argument("--upscale", type=int, default=0)

    dg = sub.add_parser("degrade")
    dg.add_argument("--scene-name", required=True)
    dg.add_argument("--root", required=True)
    dg.add_argument("--level", required=True)
    dg.add_argument("--out-scene-name", required=True)
    dg.add_argument("--seed", type=int, default=0)
    dg.add_argument("--tasks-config", default=None)

    args = ap.parse_args()
    if args.cmd == "extract":
        extract_task(args.task, args.scene_name, args.root, args.episodes,
                     args.static_only, args.cameras, args.max_frames,
                     args.wds_root, args.tasks_config, args.upscale)
    elif args.cmd == "degrade":
        scene_dir = Path(args.root) / "data" / args.scene_name
        out_dir = Path(args.root) / "data" / args.out_scene_name
        apply_degradation(scene_dir, out_dir, args.level, args.seed, args.tasks_config)


if __name__ == "__main__":
    main()
