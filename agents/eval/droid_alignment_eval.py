"""Held-out validation of the DROID FK<->SfM alignment (plan Task 16 step 3:
"Validate FK/camera calibration and Umeyama alignment on held-out trajectory
points").

agents/recon/align_to_traj.py fits its similarity transform (scale s,
rotation R, translation t) on the SAME frames it then reports residuals for
- a fair characterization of how well the transform explains its own fit
points, but not a test of whether the fit generalizes (with ~100 points and
7 DOF, overfitting is a real possibility, especially for episodes whose kept
"static" frames cluster in a small region of the workspace - exactly the
IPRL/RAIL pilot episodes below, which is also where align_to_traj.py's own
rotation cross-check flagged trouble).

This script:
  1. Deterministically splits the frames that overlap the FK trajectory into
     TRAIN (every Nth frame kept out) and HELD-OUT (the Nth-frame set),
     using the SAME offset already resolved by align_to_traj.py's
     recon_base.npz (frame_offset) so both scripts agree on correspondence.
  2. Re-fits Umeyama on TRAIN only (reusing align_to_traj.umeyama/
     rot_angle_deg - no reimplementation, no drift).
  3. Applies that train-only fit to HELD-OUT and reports center/rotation/
     reprojection residuals there - the actual generalization number.
  4. Reports a coarse "dynamic-scene contamination" proxy: the fraction of
     held-out frames whose reprojection error is a >2x-median outlier
     (large local disagreement between the FK-projected and SfM-projected
     point cloud is what a moving manipulandum/arm baked into the static
     splat's geometry looks like - see run_droid_recon.sh's "dynamic-scene
     caveat" and agents/recon/droid_static_select.py's masking).
  5. Applies a documented pass/fail gate and writes a report that a fleet
     run can aggregate (agents/eval/aggregate_results.py-style).

CPU, numpy-only, main .venv (same requirements as align_to_traj.py). Usage:
    python -m agents.eval.droid_alignment_eval \\
        --recon <scene>/recon/recon.npz --traj <scene>/gt/droid_traj.json \\
        --align-report <scene>/recon/align_report.json \\
        --out <scene>/recon/held_out_eval.json [--held-out-every 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agents.recon.align_to_traj import rot_angle_deg, umeyama

MAX_OFFSET = 2
N_REPROJ_HELDOUT = None  # None = evaluate every held-out frame (small sets)

# Gates - deliberately looser than align_to_traj.py's abort threshold
# (10 cm on the IN-SAMPLE fit): held-out error is expected to be somewhat
# worse than in-sample, and this is a diagnostic report, not an abort switch
# (a failing episode here still gets a splat and a scene, just flagged as
# replay evidence with a quantified, questionable alignment - never silently
# dropped, per docs/ICRA_RESEARCH_CONTRACT.md's no-fabrication rule).
GATE_CENTER_RMS_M = 0.15
GATE_ROTATION_DEG = 20.0


def project(pts, M, K, W, H):
    Xc = pts @ M[:3, :3].T + M[:3, 3]
    uv = Xc[:, :2] / np.clip(Xc[:, 2:3], 1e-6, None)
    uv = uv @ K[:2, :2].T + K[:2, 2]
    vis = (Xc[:, 2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & \
        (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, vis


def residual_metrics(rec, fk_all, idx, rows, s, R, t):
    """Canonical translation, rotation and reprojection residual implementation."""
    c2w = np.linalg.inv(rec["w2c"].astype(np.float64))
    cen_sfm = c2w[:, :3, 3]
    K = rec["K"].astype(np.float64)
    W, H = map(int, rec["frame_wh"])
    cen_fk = np.asarray([fk_all[j][:3, 3] for j in idx[rows]])
    cen_pred = s * cen_sfm[rows] @ R.T + t
    d = cen_fk - cen_pred
    center_rms = float(np.sqrt((d ** 2).sum(1).mean()))
    center_med = float(np.median(np.linalg.norm(d, axis=1)))
    c2w_pred = c2w[rows].copy()
    c2w_pred[:, :3, 3] = cen_pred
    c2w_pred[:, :3, :3] = R @ c2w[rows, :3, :3]
    rot_res = [rot_angle_deg(fk_all[j][:3, :3], c2w_pred[i, :3, :3])
              for i, j in zip(range(len(rows)), idx[rows])]
    pts = rec["points"].astype(np.float64) * s @ R.T + t
    reproj = []
    w2c_pred = np.linalg.inv(c2w_pred)
    for i, j in zip(range(len(rows)), idx[rows]):
        uv_a, vis_a = project(pts, w2c_pred[i], K, W, H)
        uv_b, vis_b = project(pts, np.linalg.inv(fk_all[j]), K, W, H)
        vis = vis_a & vis_b
        if vis.sum() < 50:
            continue
        reproj.append(float(np.median(
            np.linalg.norm(uv_a[vis] - uv_b[vis], axis=1))))
    return {
        "n_frames": int(len(rows)), "center_rms_m": center_rms,
        "center_median_m": center_med,
        "rotation_residual_deg": {"median": float(np.median(rot_res)),
                                  "max": float(np.max(rot_res))},
        "reproj_median_px": (float(np.median(reproj)) if reproj
                            else None),
        "reproj_per_frame_px": reproj,
    }


def evaluate_sealed_fit(fit_directory, reference_path, destination=None):
    """Evaluate the constructor's frozen transform, with no correspondence fit."""
    from agents.recon.align_to_traj import validate_train_fit
    from agents.recon.droid_extract import input_identity, verify_input, validate_frame_plan
    from agents.recon.colmap_poses import write_new_json
    fit_directory, reference_path = map(Path, (fit_directory, reference_path))
    destination = Path(destination) if destination is not None else None
    if destination is not None and destination.exists():
        raise FileExistsError('immutable alignment evaluation already exists')
    fit = validate_train_fit(fit_directory)  # MUST precede reference values.
    ref = json.loads(reference_path.read_text())
    if (set(ref) != {'schema_version', 'role', 'plan', 'fit', 'fk_by_index'}
            or ref['schema_version'] != 1 or ref['role'] != 'held_out_only'
            or ref['fit'] != input_identity(fit_directory / 'fit.json') or ref['plan'] != fit['plan']):
        raise ValueError('reference/constructor binding differs')
    verify_input(ref['fit']); verify_input(ref['plan'])
    plan = validate_frame_plan(json.loads(Path(ref['plan']['path']).read_text()))
    if set(ref['fk_by_index']) != {str(i) for i in plan['split']['held_out_fk_indices']}:
        raise ValueError('reference roster differs or overlaps TRAIN hypotheses')
    fk = {int(i): np.asarray(v, np.float64) for i, v in ref['fk_by_index'].items()}
    if any(v.shape != (4, 4) or not np.isfinite(v).all() for v in fk.values()):
        raise ValueError('invalid reference FK pose')
    rec = dict(np.load(fit['recon']['path'], allow_pickle=False))
    ids = [int(Path(str(n)).stem.split('_')[1]) for n in rec['names']]
    held = plan['split']['held_out_video_indices']
    rows = np.asarray([i for i, idx in enumerate(ids) if idx in held], int)
    solution = fit['solution']; off = solution['frame_offset']
    metrics = (residual_metrics(rec, fk, np.asarray(ids) + off, rows,
               solution['scale'], np.asarray(solution['R']), np.asarray(solution['t']))
               if len(rows) >= 5 else None)
    passed = (metrics['center_rms_m'] < GATE_CENTER_RMS_M and
              metrics['rotation_residual_deg']['median'] < GATE_ROTATION_DEG) if metrics else None
    result = {'schema_version': 1, 'scope': 'droid_independent_kinematic_alignment',
        'fit': input_identity(fit_directory / 'fit.json'), 'reference': input_identity(reference_path),
        'plan': fit['plan'], 'frame_offset': off,
        'planned_reference_frames': len(held), 'evaluated_reference_frames': len(rows),
        'missing_reference_frames': sorted(set(held) - set(ids)),
        'held_out_metrics': metrics,
        'gate': {'center_rms_m': GATE_CENTER_RMS_M, 'rotation_deg': GATE_ROTATION_DEG,
                 'passed': passed}, 'construction_reopened': False,
        'note': 'Held-out robot kinematics only; RGB may enter SfM. No image-fidelity claim.'}
    if destination is not None:
        write_new_json(destination, result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sealed-fit', type=Path)
    ap.add_argument("--recon", type=Path,
                    help="recon.npz (pre-alignment; agents/recon/"
                         "colmap_poses.py or models/vggt_scene.py output)")
    ap.add_argument("--traj", required=True, type=Path)
    ap.add_argument("--align-report", type=Path, default=None,
                    help="align_to_traj.py's align_report.json, for the "
                         "in-sample-vs-held-out comparison in the output")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--held-out-every", type=int, default=5,
                    help="every Nth frame (by fit order) goes to held-out; "
                         "5 -> 80/20 train/held-out split")
    ap.add_argument("--max-offset", type=int, default=MAX_OFFSET)
    args = ap.parse_args()

    if args.sealed_fit:
        evaluate_sealed_fit(args.sealed_fit, args.traj, args.out)
        return
    if args.recon is None:
        ap.error('--recon is required for the legacy diagnostic')

    rec = dict(np.load(args.recon, allow_pickle=False))
    traj = json.loads(args.traj.read_text())
    fk_all = np.asarray(traj["full"]["c2w_base"], np.float64)
    T_steps = len(fk_all)

    names = [str(n) for n in rec["names"]]
    mp4_idx = np.array([int(Path(n).stem.split("_")[1]) for n in names])
    w2c = rec["w2c"].astype(np.float64)
    c2w = np.linalg.inv(w2c)
    cen_sfm = c2w[:, :3, 3]
    K = rec["K"].astype(np.float64)
    W, H = int(rec["frame_wh"][0]), int(rec["frame_wh"][1])

    # ---- resolve the same constant offset align_to_traj.py would (in-
    # sample, over ALL frames) - this is just correspondence resolution, not
    # part of what we're validating, so it is allowed to see everything.
    best = None
    for off in range(-args.max_offset, args.max_offset + 1):
        idx = mp4_idx + off
        ok = (idx >= 0) & (idx < T_steps)
        if ok.sum() < max(10, 0.9 * len(idx)):
            continue
        s, R, t = umeyama(cen_sfm[ok], fk_all[idx[ok], :3, 3])
        res = fk_all[idx[ok], :3, 3] - (s * cen_sfm[ok] @ R.T + t)
        rms = float(np.sqrt((res ** 2).sum(1).mean()))
        if best is None or rms < best["rms"]:
            best = {"off": off, "ok": ok, "rms": rms}
    if best is None:
        raise SystemExit("[droid_alignment_eval] no frame overlap with FK")
    off, ok = best["off"], best["ok"]
    idx = mp4_idx + off
    ok_rows = np.where(ok)[0]  # row indices into rec/cen_sfm with FK overlap

    # ---- deterministic split (order = frame_offset-corrected temporal
    # order, i.e. video order - a real generalization test needs the held-
    # out points to be interpolated/extrapolated from train, not identical
    # to it, so this deliberately does NOT shuffle: held-out frames are
    # temporally interleaved but distinct viewpoints from train)
    n = len(ok_rows)
    if n < 20:
        raise SystemExit(f"[droid_alignment_eval] only {n} FK-overlapping "
                         f"frames - too few for a train/held-out split")
    held_mask = (np.arange(n) % args.held_out_every == 0)
    train_rows, held_rows = ok_rows[~held_mask], ok_rows[held_mask]
    if len(held_rows) < 5 or len(train_rows) < 10:
        raise SystemExit(f"[droid_alignment_eval] split too small: "
                         f"{len(train_rows)} train / {len(held_rows)} "
                         f"held-out (need >=10 / >=5)")

    # ---- fit on TRAIN only ---------------------------------------------
    s, R, t = umeyama(cen_sfm[train_rows], fk_all[idx[train_rows], :3, 3])


    train_metrics = residual_metrics(rec, fk_all, idx, train_rows, s, R, t)
    held_metrics = residual_metrics(rec, fk_all, idx, held_rows, s, R, t)

    # ---- dynamic-scene contamination proxy on held-out reprojection -----
    reproj_vals = np.array([r for r in held_metrics["reproj_per_frame_px"]
                            if r is not None])
    contam_frac = None
    if len(reproj_vals) >= 3 and np.median(reproj_vals) > 0:
        contam_frac = float((reproj_vals >
                             2 * np.median(reproj_vals)).mean())

    passed = (held_metrics["center_rms_m"] < GATE_CENTER_RMS_M and
              held_metrics["rotation_residual_deg"]["median"] <
              GATE_ROTATION_DEG)

    in_sample = None
    if args.align_report and args.align_report.exists():
        in_sample = json.loads(args.align_report.read_text())

    report = {
        "episode": traj.get("episode"), "frame_offset": int(off),
        "split": {"train_n": len(train_rows), "held_out_n": len(held_rows),
                  "held_out_every": args.held_out_every},
        "train_fit_scale": s,
        "train_metrics_in_sample": train_metrics,
        "held_out_metrics": held_metrics,
        "in_sample_align_to_traj_report": in_sample,
        "dynamic_scene_contamination_frac_2x_median_reproj_outliers":
            contam_frac,
        "gate": {"center_rms_m": GATE_CENTER_RMS_M,
                 "rotation_deg": GATE_ROTATION_DEG, "passed": bool(passed)},
        "note": ("held_out_metrics is the generalization number (fit on "
                "train_n frames, evaluated on held_out_n withheld frames); "
                "train_metrics_in_sample is the same fit re-evaluated on "
                "its OWN training frames, included only to show the train/"
                "held-out gap. in_sample_align_to_traj_report (if provided) "
                "is align_to_traj.py's own all-frames fit for comparison - "
                "expect it to sit between the two since it uses more data."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"[droid_alignment_eval] {traj.get('episode')}: held-out "
          f"({len(held_rows)} frames) center RMS "
          f"{held_metrics['center_rms_m']*100:.2f} cm, rotation median "
          f"{held_metrics['rotation_residual_deg']['median']:.1f} deg, "
          f"contamination {contam_frac} -> "
          f"{'PASS' if passed else 'FAIL'} (gates: <{GATE_CENTER_RMS_M*100:.0f}"
          f" cm, <{GATE_ROTATION_DEG:.0f} deg)")
    print(f"[droid_alignment_eval] wrote {args.out}")


if __name__ == "__main__":
    main()
