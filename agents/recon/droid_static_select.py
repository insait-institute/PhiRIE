"""DROID static-frame selection (viewpoint-diverse, not just index-uniform)
+ gripper/manipulandum masking for splat training.

Plan Task 16 step 2 ("improve static-frame selection and robot/manipulated-
object masking for splat training"). This factors the "which steps count as
static" half of agents/recon/droid_extract.py's select_steps() out into a
reusable function, and adds two things droid_extract.py's v1 selection does
NOT do:

1. VIEWPOINT-DIVERSE capping. droid_extract.py, once the joint-velocity/
   gripper ladder produces a keep-set, caps it to --max-frames by
   `np.linspace` over the KEPT INDEX ORDER (i.e. uniform in time). Time-
   uniform is not the same as VIEW-uniform: DROID episodes idle with the
   gripper open for long stretches (the ladder's rungs are wide, e.g.
   jv<0.3 keeps 106/401 steps on the pilot episode, mostly one continuous
   "parked" stretch), so a time-uniform cap over-samples one viewpoint
   region and starves the rest - bad for gsplat, which needs baseline
   spread, not frame-rate. `select_diverse(...)` below does the ladder pass
   identically, then greedy farthest-point sampling (FPS) on camera CENTERS
   (from the FK trajectory - already metric, no need to wait for COLMAP) to
   pick the final --max-frames subset, maximizing pairwise coverage instead
   of temporal uniformity.

2. Gripper region masking. The wrist camera is RIGIDLY mounted to the
   end-effector (droid_extract.py's docstring: inv(T_ee) @ T_cam constant to
   <1e-12 m over 402 steps on the verification episode) - the gripper
   fingers therefore sit in an (approximately) FIXED image-space region
   across the WHOLE episode, independent of scene content, camera pose, or
   task. `gripper_fixed_mask()` returns that region once per camera
   resolution; no per-frame computation needed. This does NOT cover the
   manipulated object, which moves through the frame with the arm - that
   residual is the same "dynamic-scene caveat" run_droid_recon.sh already
   documents (v1 accepts manipulandum ghosting). `dynamic_pixel_mask()`
   below is a coarse, best-effort SECOND signal for that harder case: image-
   space frame-to-frame absolute difference (no pose/geometry needed, so it
   runs on raw extracted frames before COLMAP), flagging pixels that change
   a lot between temporally-close kept frames. It is deliberately cheap and
   approximate - a real per-frame object mask needs either the reprojected
   FK/CAD gripper geometry (not available in this env) or SAM3 (already
   used downstream in auto_segment/factory_refine_masks against the fused
   splat mesh, which is the more reliable place to remove transient
   geometry - this module's masks are a fast, pre-COLMAP-only prior, not a
   replacement for that).

CPU, numpy + opencv + PIL only - runs in ANY of the three SimAny envs, and
also under ${SIMANY_H5_ENV} (same env droid_extract.py
uses).

Usage:
    python -m agents.recon.droid_static_select \
        --traj <scene>/gt/droid_traj.json --frames-dir <scene>/wrist_frames \
        --out-dir <scene>/static_select [--max-frames 240]
Writes:
    <out-dir>/selection_report.json  (kept indices, FPS order, diagnostics)
    <out-dir>/gripper_mask.png       (single reusable fixed mask, 255=valid)
    <out-dir>/masks/mask_<idx>.png   (gripper_mask AND-combined with the
                                      per-frame dynamic mask, one per kept
                                      frame; 255=likely-static/valid,
                                      0=likely gripper or transient object)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Same ladder as agents/recon/droid_extract.py FILTER_LADDER - kept in sync
# manually (no shared import to avoid coupling the h5py-only module to this
# one, which is meant to also run standalone on already-extracted frames).
MIN_KEPT = 80
FILTER_LADDER = ((0.15, 0.01), (0.30, 0.01), (0.50, 0.01), (np.inf, np.inf))

# Gripper region, expressed as fractions of (W, H): a wide-ish wedge at the
# bottom-center of the frame. Tuned by eye against the pilot episode's
# extracted wrist frames (IPRL Thu_Aug_24_21:29:53_2023) - the two gripper
# fingers occupy roughly the bottom 30% of the frame, centered, widening
# toward the bottom edge. This is intentionally generous (better to mask a
# little background than leave gripper pixels in the splat's photometric
# loss); re-tune per-rig if a lab's mount geometry visibly differs.
GRIPPER_BOTTOM_FRAC = 0.32
GRIPPER_TOP_HALF_WIDTH_FRAC = 0.22   # half-width at the wedge's top edge
GRIPPER_BOTTOM_HALF_WIDTH_FRAC = 0.40  # half-width at the frame's bottom edge


def select_diverse(cam6, jv, grip, n_valid, max_frames, stride=1):
    """Ladder-filter then viewpoint-diverse cap (replaces droid_extract.py's
    linspace-in-index-order cap). Returns (kept_indices sorted ascending,
    diagnostics dict). `cam6` is the (T,6) FK camera pose array (only the
    xyz columns are used, already metric - no SfM needed for this).
    """
    jv_max_abs = np.abs(jv).max(1)
    dgrip_abs = np.abs(np.gradient(grip))
    keep = None
    filt = None
    for jv_max, grip_eps in FILTER_LADDER:
        cand = np.where((jv_max_abs[:n_valid] < jv_max) &
                        (dgrip_abs[:n_valid] < grip_eps))[0]
        if len(cand) >= MIN_KEPT or not np.isfinite(jv_max):
            keep, filt = cand, {
                "jv_max": float(jv_max) if np.isfinite(jv_max) else None,
                "grip_eps": float(grip_eps) if np.isfinite(grip_eps)
                else None, "n_kept": int(len(cand)), "n_total": int(n_valid)}
            break
    keep = keep[::max(1, stride)]

    method = "all (below max_frames, no capping needed)"
    if len(keep) > max_frames:
        centers = np.asarray(cam6, np.float64)[keep, :3]
        order = _farthest_point_order(centers)
        keep = np.sort(keep[order[:max_frames]])
        method = "farthest-point sampling on FK camera centers"
    filt["cap_method"] = method
    filt["n_after_cap"] = int(len(keep))
    return keep, filt


def _farthest_point_order(points):
    """Greedy farthest-point sampling order over an (N,3) point set: index 0
    seeds, then repeatedly add the point maximizing its min-distance to the
    already-picked set. O(N^2) - fine for N <= ~1200 (this dataset's longest
    episode)."""
    n = len(points)
    if n <= 1:
        return np.arange(n)
    picked = [0]
    d = np.linalg.norm(points - points[0], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm(points - points[nxt], axis=1))
        d[picked] = -1  # never re-pick
    return np.array(picked)


def gripper_fixed_mask(w, h):
    """Fixed image-space mask (H,W) uint8, 255=keep / 0=likely-gripper, for
    ONE camera resolution. See module docstring for why a fixed region is a
    valid approximation for a rigidly wrist-mounted camera."""
    mask = np.full((h, w), 255, np.uint8)
    top_y = int(h * (1.0 - GRIPPER_BOTTOM_FRAC))
    cx = w / 2.0
    poly = np.array([
        [cx - GRIPPER_TOP_HALF_WIDTH_FRAC * w, top_y],
        [cx + GRIPPER_TOP_HALF_WIDTH_FRAC * w, top_y],
        [cx + GRIPPER_BOTTOM_HALF_WIDTH_FRAC * w, h],
        [cx - GRIPPER_BOTTOM_HALF_WIDTH_FRAC * w, h],
    ], np.int32)
    cv2.fillPoly(mask, [poly], 0)
    return mask


def dynamic_pixel_mask(img, prev_img, next_img, thresh=25, blur=5):
    """Coarse per-frame transient mask (H,W) uint8, 255=static/keep,
    0=changed a lot vs. its temporal neighbors. Pure image-space absolute
    difference (no pose/warp - the camera also moves, so this only catches
    LARGE, LOCAL appearance changes like the manipulandum sliding through a
    mostly-similar view over a couple of frames, not the whole moving
    background). Best-effort prior, not a segmentation - see module
    docstring."""
    def d(a, b):
        ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
        gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
        if ga.shape != gb.shape:
            gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
        return cv2.absdiff(ga, gb)
    diffs = [d(img, o) for o in (prev_img, next_img) if o is not None]
    if not diffs:
        return np.full(img.shape[:2], 255, np.uint8)
    changed = np.minimum(*diffs) if len(diffs) == 2 else diffs[0]
    mask = np.where(changed > thresh, 0, 255).astype(np.uint8)
    # light morphological close so single-pixel noise doesn't fragment the
    # mask into a stipple that gsplat's per-pixel loss can't use sensibly
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", required=True, type=Path,
                    help="gt/droid_traj.json written by droid_extract.py")
    ap.add_argument("--frames-dir", required=True, type=Path,
                    help="wrist_frames/ dir (already-extracted frame_*.jpg)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dynamic-thresh", type=int, default=25)
    ap.add_argument("--no-dynamic-mask", action="store_true",
                    help="skip the frame-diff pass, gripper mask only")
    args = ap.parse_args()

    traj = json.loads(args.traj.read_text())
    full = traj["full"]
    cam6_c2w = np.asarray(full["c2w_base"], np.float64)  # (T,4,4)
    if cam6_c2w.ndim == 3:
        centers = cam6_c2w[:, :3, 3]
    else:  # defensive: some earlier/alternate writer might store (T,6)
        centers = cam6_c2w[:, :3]
    jv = np.abs(np.asarray(full["max_abs_joint_velocity"], np.float64))
    jv = jv[:, None] if jv.ndim == 1 else jv  # select_diverse expects (T,K)
    grip = np.asarray(full["gripper_position"], np.float64)
    n_valid = min(len(centers), len(grip))

    keep, filt = select_diverse(centers, jv, grip, n_valid,
                                args.max_frames, args.stride)
    print(f"[droid_static_select] {args.traj.parent.parent.name}: "
          f"{filt['n_kept']}/{filt['n_total']} pass the ladder "
          f"(jv<{filt['jv_max']} |dgrip|<{filt['grip_eps']}), "
          f"{filt['n_after_cap']}/{args.max_frames} after {filt['cap_method']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = {int(p.stem.split("_")[1]): p
                  for p in args.frames_dir.glob("frame_*.jpg")}
    missing = [int(i) for i in keep if int(i) not in frame_paths]
    if missing:
        print(f"[droid_static_select] WARNING: {len(missing)} selected "
              f"indices have no extracted frame in {args.frames_dir} "
              f"(extract with a --max-frames >= this selection's cap or "
              f"re-run droid_extract with matching --stride): {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}")
    kept_present = [int(i) for i in keep if int(i) in frame_paths]

    sample = cv2.imread(str(next(iter(frame_paths.values())))) \
        if frame_paths else None
    if sample is None:
        raise SystemExit(f"[droid_static_select] no frames found in "
                         f"{args.frames_dir}")
    h, w = sample.shape[:2]
    gmask = gripper_fixed_mask(w, h)
    cv2.imwrite(str(args.out_dir / "gripper_mask.png"), gmask)

    masks_dir = args.out_dir / "masks"
    masks_dir.mkdir(exist_ok=True)
    n_dynamic_flagged = 0
    for pos, idx in enumerate(kept_present):
        combined = gmask
        if not args.no_dynamic_mask:
            img = cv2.imread(str(frame_paths[idx]))
            prev_idx = kept_present[pos - 1] if pos > 0 else None
            next_idx = kept_present[pos + 1] if pos + 1 < len(kept_present) else None
            prev_img = cv2.imread(str(frame_paths[prev_idx])) if prev_idx is not None else None
            next_img = cv2.imread(str(frame_paths[next_idx])) if next_idx is not None else None
            dmask = dynamic_pixel_mask(img, prev_img, next_img,
                                       thresh=args.dynamic_thresh)
            combined = cv2.bitwise_and(gmask, dmask)
            if (dmask == 0).mean() > 0.005:
                n_dynamic_flagged += 1
        cv2.imwrite(str(masks_dir / f"mask_{idx:06d}.png"), combined)

    report = {
        "episode": traj.get("episode"), "camera": traj.get("camera"),
        "n_total_steps": int(n_valid), "filter": filt,
        "kept_indices": [int(i) for i in keep],
        "kept_with_frame_on_disk": kept_present,
        "missing_frame_indices": missing,
        "frame_wh": [int(w), int(h)],
        "gripper_mask_kept_px_frac": float((gmask == 255).mean()),
        "n_frames_with_dynamic_content_flagged": n_dynamic_flagged,
        "note": ("gripper_mask.png is one fixed region reused for every "
                "frame (rigid wrist mount, see module docstring); "
                "masks/mask_<idx>.png additionally ANDs in a per-frame "
                "frame-diff dynamic mask unless --no-dynamic-mask. Neither "
                "is a substitute for SAM3-based instance masking "
                "downstream (factory_refine_masks) - these are pre-COLMAP, "
                "geometry-free priors only."),
    }
    (args.out_dir / "selection_report.json").write_text(
        json.dumps(report, indent=1))
    print(f"[droid_static_select] wrote {args.out_dir}/selection_report.json"
          f", gripper_mask.png, masks/*.png ({len(kept_present)} frames, "
          f"{n_dynamic_flagged} flagged with transient content)")


if __name__ == "__main__":
    main()
