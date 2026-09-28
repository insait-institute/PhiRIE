"""Umeyama-align a COLMAP/VGGT reconstruction to the DROID FK trajectory.

Takes recon.npz (agents/recon/colmap_poses.py or models/vggt_scene.py; poses
in an arbitrary SfM frame at arbitrary scale) and gt/droid_traj.json
(agents/recon/droid_extract.py; FK wrist-camera c2w in the ROBOT BASE frame,
metric) and solves the similarity transform mapping SfM camera centers onto
the FK centers. Output is the same recon.npz contract, already METRIC and in
the ROBOT BASE frame - metricize.py must NOT run afterwards (its DA3 scale
and floor re-orientation would double-scale and break the exact base-frame
correspondence the robot layer relies on; base z=0 is the robot mount plane,
typically the TABLE, which plays the floor's role downstream).

WHY Umeyama on centers: COLMAP's BA-consistent local poses win splat PSNR
(25.52 vs 22.52 dB, see commits 589ab4a..d318e50),
but its frame/scale are arbitrary; FK gives metric truth per frame. Centers
constrain the similarity fully; rotations then serve as a free cross-check
of the camera-axis convention (reported, not fitted).

Also resolves the constant MP4-vs-h5 frame offset (the MP4 has T-1 frames
for T steps) by re-fitting at offsets in [-max-offset, +max-offset] and
keeping the lowest RMS. Aborts if the best RMS exceeds --max-rms-m: a large
residual means broken frame correspondence or a bent reconstruction, and
everything downstream would inherit silently wrong metric scale.

CPU, numpy-only, main .venv (any env works). Usage:
    python -m agents.recon.align_to_traj --recon recon.npz \
        --traj <scene>/gt/droid_traj.json --out recon_base.npz
Writes align_report.json (residuals, offset, rotation/reprojection checks)
next to --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAX_OFFSET = 2      # constant mp4-vs-h5 shift search range (|off| frames)
MAX_RMS_M = 0.10    # abort above this center residual (frame mismatch)
N_REPROJ_FRAMES = 5  # frames for the FK-pose reprojection cross-check


def umeyama(src, dst):
    """Similarity src -> dst (least squares): returns s, R, t with
    dst ~= s * R @ src + t. Standard Umeyama 1991."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def rot_angle_deg(Ra, Rb):
    """Geodesic angle between two rotation matrices, degrees."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def train_solution(rec, train, plan):
    """Prospective fit: every offset/crosscheck sees TRAIN FK only."""
    from agents.recon.droid_extract import validate_frame_plan
    from agents.eval.droid_alignment_eval import residual_metrics
    validate_frame_plan(plan)
    if (set(train) != {'schema_version', 'role', 'plan', 'fk_by_index'}
            or train['schema_version'] != 1 or train['role'] != 'train_only'):
        raise ValueError('constructor requires sparse TRAIN-only evidence')
    split = plan['split']
    if set(train['fk_by_index']) != {str(i) for i in split['train_fk_indices']}:
        raise ValueError('TRAIN FK roster differs or contains reference evidence')
    fk = {int(i): np.asarray(v, np.float64) for i, v in train['fk_by_index'].items()}
    if any(v.shape != (4, 4) or not np.isfinite(v).all() for v in fk.values()):
        raise ValueError('invalid TRAIN camera poses')
    names = [str(n) for n in rec['names']]
    ids = [int(Path(n).stem.split('_')[1]) for n in names]
    if (len(ids) != len(set(ids)) or any(n != f'frame_{i:06d}.jpg' for n, i in zip(names, ids))
            or not set(ids).issubset(split['rgb_indices'])):
        raise ValueError('reconstruction frame roster differs')
    rows = np.asarray([i for i, idx in enumerate(ids) if idx in split['train_video_indices']], int)
    # All offset hypotheses retain100% of registered TRAIN rows because the
    # metadata margins were fixed first (legacy offset-overlap gate is90%).
    # Missing SfM poses remain reported; do not introduce a new coverage gate.
    if len(rows) < 10:
        raise ValueError('insufficient preregistered TRAIN pose coverage')
    if rec['w2c'].shape != (len(ids), 4, 4) or not np.isfinite(rec['w2c']).all():
        raise ValueError('invalid reconstruction camera poses')
    centers = np.linalg.inv(rec['w2c'].astype(np.float64))[:, :3, 3]
    if np.linalg.matrix_rank(centers[rows] - centers[rows].mean(0)) < 2:
        raise ValueError('degenerate TRAIN camera-center geometry')
    best = None
    for off in split['offsets']:
        dst = np.asarray([fk[ids[i] + off][:3, 3] for i in rows])
        s, R, t = umeyama(centers[rows], dst)
        residual = dst - (s * centers[rows] @ R.T + t)
        rms = float(np.sqrt((residual ** 2).sum(1).mean()))
        if not np.isfinite([s, rms]).all() or s <= 0:
            raise ValueError('degenerate TRAIN similarity')
        if best is None or rms < best['rms']:
            best = {'offset': off, 'scale': s, 'rotation': R.tolist(),
                    'translation': t.tolist(), 'rms': rms}
    s, R, t = best['scale'], np.asarray(best['rotation']), np.asarray(best['translation'])
    idx = np.asarray(ids) + best['offset']
    # A sparse mapping is sufficient: residual_metrics only indexes rows.
    metrics = residual_metrics(rec, fk, idx, rows, s, R, t)
    if best['rms'] > MAX_RMS_M:
        raise ValueError('TRAIN center RMS exceeds unchanged0.10m constructor gate')
    return {'frame_offset': best['offset'], 'scale': s, 'R': R.tolist(), 't': t.tolist(),
            'train_video_indices_used': [ids[i] for i in rows],
            'planned_train_frames': len(split['train_video_indices']),
            'missing_train_video_indices': sorted(set(split['train_video_indices']) - set(ids)),
            'train_metrics': metrics, 'max_rms_m': MAX_RMS_M}


def apply_train_solution(rec, fit):
    """Apply the existing scale-then-rigid reconstruction transform."""
    s, R, t = fit['scale'], np.asarray(fit['R']), np.asarray(fit['t'])
    transform = np.eye(4); transform[:3, :3] = R; transform[:3, 3] = t
    w2c = rec['w2c'].astype(np.float64).copy(); w2c[:, :3, 3] *= s
    return dict(rec, w2c=w2c @ np.linalg.inv(transform),
        points=(rec['points'].astype(np.float64) * s @ R.T + t).astype(np.float32),
        depth=(rec['depth'].astype(np.float32) * s).astype(np.float16),
        metric_scale=np.float64(s), T_align=transform, frame_offset=np.int64(fit['frame_offset']))


def align_train_only(recon_path, train_path, plan_path, destination):
    import sys
    from agents.recon.droid_extract import input_identity, verify_input, validate_frame_plan
    from agents.recon.colmap_poses import write_new_json
    from robo.manifest.hash import canonical_hash, git_snapshot
    recon_path, train_path, plan_path, destination = map(Path, (recon_path, train_path, plan_path, destination))
    if destination.exists():
        raise FileExistsError('immutable TRAIN fit already exists')
    plan = validate_frame_plan(json.loads(plan_path.read_text()))
    train = json.loads(train_path.read_text())
    if train['plan'] != input_identity(plan_path):
        raise ValueError('TRAIN plan identity mismatch')
    verify_input(train['plan'])
    rec = dict(np.load(recon_path, allow_pickle=False))
    fit = train_solution(rec, train, plan)
    aligned = apply_train_solution(rec, fit)
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / 'recon_base.npz', **aligned)
    report = {'schema_version': 1, 'scope': 'droid_train_only_frozen_fit',
              'source': git_snapshot(Path(__file__).resolve().parents[2]),
              'runtime': {'invocation': sys.executable, 'numpy_version': np.__version__,
                          'python': input_identity(Path(sys.executable).resolve())},
              'plan': input_identity(plan_path), 'train': input_identity(train_path),
              'recon': input_identity(recon_path), 'solution': fit,
              'aligned': input_identity(destination / 'recon_base.npz')}
    write_new_json(destination / 'fit.json', report)
    write_new_json(destination / 'seal.json', {'fit_sha256': input_identity(destination / 'fit.json')['sha256'],
                                            'fit_digest': canonical_hash(report)})
    return report


def validate_train_fit(directory):
    from agents.recon.droid_extract import input_identity, verify_input, validate_frame_plan
    from robo.manifest.hash import canonical_hash, git_snapshot
    directory = Path(directory)
    report = json.loads((directory / 'fit.json').read_text())
    seal = json.loads((directory / 'seal.json').read_text())
    if (report.get('scope') != 'droid_train_only_frozen_fit' or
            seal != {'fit_sha256': input_identity(directory / 'fit.json')['sha256'],
                     'fit_digest': canonical_hash(report)}):
        raise ValueError('missing or altered TRAIN fit seal')
    code = git_snapshot(Path(__file__).resolve().parents[2])
    if report['source']['commit'] != code['commit'] or report['source']['dirty'] or code['dirty']:
        raise ValueError('TRAIN fit requires original clean source')
    if report['runtime']['numpy_version'] != np.__version__:
        raise ValueError('TRAIN replay requires original NumPy runtime')
    verify_input(report['runtime']['python'])
    if Path(report['aligned']['path']) != directory.absolute() / 'recon_base.npz':
        raise ValueError('aligned artifact path changed')
    for key in ('plan', 'train', 'recon', 'aligned'):
        verify_input(report[key])
    plan = validate_frame_plan(json.loads(Path(report['plan']['path']).read_text()))
    train = json.loads(Path(report['train']['path']).read_text())
    if train['plan'] != report['plan']:
        raise ValueError('TRAIN plan identity mismatch')
    rec = dict(np.load(report['recon']['path'], allow_pickle=False))
    if train_solution(rec, train, plan) != report['solution']:
        raise ValueError('TRAIN fit replay differs')
    aligned = dict(np.load(report['aligned']['path'], allow_pickle=False))
    expected = apply_train_solution(rec, report['solution'])
    if set(aligned) != set(expected) or any(not np.array_equal(aligned[k], expected[k]) for k in expected):
        raise ValueError('aligned reconstruction differs from frozen TRAIN transform')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recon", required=True, type=Path)
    ap.add_argument("--traj", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-offset", type=int, default=MAX_OFFSET)
    ap.add_argument("--max-rms-m", type=float, default=MAX_RMS_M)
    ap.add_argument('--frame-plan', type=Path)
    args = ap.parse_args()

    if args.frame_plan:
        if args.max_offset != MAX_OFFSET or args.max_rms_m != MAX_RMS_M:
            ap.error('prospective alignment thresholds are frozen')
        align_train_only(args.recon, args.traj, args.frame_plan, args.out)
        return

    rec = dict(np.load(args.recon, allow_pickle=False))
    traj = json.loads(args.traj.read_text())
    fk_all = np.asarray(traj["full"]["c2w_base"], np.float64)  # (T,4,4)
    T_steps = len(fk_all)

    names = [str(n) for n in rec["names"]]
    # frame_<mp4idx>.jpg carries the source MP4 index == provisional h5 index
    mp4_idx = np.array([int(Path(n).stem.split("_")[1]) for n in names])
    w2c = rec["w2c"].astype(np.float64)
    c2w = np.linalg.inv(w2c)
    cen_sfm = c2w[:, :3, 3]

    # ---- constant-offset search + Umeyama fit ------------------------------
    best = None
    for off in range(-args.max_offset, args.max_offset + 1):
        idx = mp4_idx + off
        ok = (idx >= 0) & (idx < T_steps)
        if ok.sum() < max(10, 0.9 * len(idx)):
            continue
        cen_fk = fk_all[idx[ok], :3, 3]
        s, R, t = umeyama(cen_sfm[ok], cen_fk)
        res = cen_fk - (s * cen_sfm[ok] @ R.T + t)
        rms = float(np.sqrt((res ** 2).sum(1).mean()))
        print(f"[align] offset {off:+d}: {ok.sum()} frames, "
              f"scale {s:.5f}, center RMS {rms * 100:.2f} cm")
        if best is None or rms < best["rms"]:
            best = {"off": off, "ok": ok, "s": s, "R": R, "t": t, "rms": rms,
                    "res": res}
    if best is None:
        raise SystemExit("[align] no frame overlap with the FK trajectory")
    off, s, R, t = best["off"], best["s"], best["R"], best["t"]
    med = float(np.median(np.linalg.norm(best["res"], axis=1)))
    print(f"[align] BEST offset {off:+d}: scale {s:.5f}, center RMS "
          f"{best['rms'] * 100:.2f} cm, median {med * 100:.2f} cm")
    if best["rms"] > args.max_rms_m:
        raise SystemExit(
            f"[align] ABORT: center RMS {best['rms'] * 100:.1f} cm > "
            f"{args.max_rms_m * 100:.0f} cm - frame correspondence or the "
            f"reconstruction itself is broken; do not train on this")

    # ---- apply: scale the SfM world, then the rigid base-frame transform ---
    T_rig = np.eye(4)
    T_rig[:3, :3], T_rig[:3, 3] = R, t
    w2c_new = w2c.copy()
    w2c_new[:, :3, 3] *= s                      # camera centers scale with s
    w2c_new = w2c_new @ np.linalg.inv(T_rig)
    pts = rec["points"].astype(np.float64) * s @ R.T + t
    depth = rec["depth"].astype(np.float32) * s

    # ---- cross-checks (reported, never fitted) ------------------------------
    # 1) rotation residual vs FK: small angles => the FK camera rotation is
    #    OpenCV-convention c2w, same as COLMAP's - convention verified.
    idx = mp4_idx + off
    ok = best["ok"]
    c2w_new = np.linalg.inv(w2c_new)
    rot_res = [rot_angle_deg(fk_all[j, :3, :3], c2w_new[i, :3, :3])
               for i, j in zip(np.where(ok)[0], idx[ok])]
    rot_med = float(np.median(rot_res))
    print(f"[align] rotation residual vs FK: median {rot_med:.2f} deg, "
          f"max {max(rot_res):.2f} deg "
          f"({'OpenCV c2w convention CONFIRMED' if rot_med < 10 else 'LARGE - convention mismatch, inspect!'})")

    # 2) reprojection: project the aligned sparse cloud with the FK pose +
    #    COLMAP K and measure pixel displacement vs the COLMAP-pose
    #    projection - the empirical extrinsics-convention test on image
    #    evidence, not just pose algebra.
    K = rec["K"].astype(np.float64)
    W, H = int(rec["frame_wh"][0]), int(rec["frame_wh"][1])

    def project(M):
        Xc = pts @ M[:3, :3].T + M[:3, 3]
        uv = Xc[:, :2] / np.clip(Xc[:, 2:3], 1e-6, None)
        uv = uv @ K[:2, :2].T + K[:2, 2]
        vis = (Xc[:, 2] > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & \
            (uv[:, 1] >= 0) & (uv[:, 1] < H)
        return uv, vis

    reproj_px = []
    ok_rows = np.where(ok)[0]
    for r in np.linspace(0, len(ok_rows) - 1, N_REPROJ_FRAMES).astype(int):
        i = ok_rows[r]
        uv_col, vis = project(w2c_new[i])
        uv_fk, _ = project(np.linalg.inv(fk_all[idx[i]]))
        if vis.sum() < 50:
            continue
        reproj_px.append(float(np.median(
            np.linalg.norm(uv_col[vis] - uv_fk[vis], axis=1))))
    reproj_med = float(np.median(reproj_px)) if reproj_px else float("nan")
    print(f"[align] FK-vs-COLMAP reprojection of the aligned cloud: median "
          f"{reproj_med:.1f} px over {len(reproj_px)} frames ({W}x{H})")

    # 3) where does the support plane sit? base z=0 is the robot mount plane
    #    (usually the table). Report the dominant z mode; NO transform is
    #    applied - the frame must stay exactly the robot base frame.
    z = pts[:, 2]
    zc = z[(z > -1.0) & (z < 1.5)]
    hist, edges = np.histogram(zc, bins=125, range=(-1.0, 1.5))
    z_mode = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)
    cam_z = c2w_new[:, 2, 3]
    print(f"[align] aligned cloud: dominant z mode {z_mode:+.3f} m "
          f"(mount plane is z=0), cameras z {cam_z.min():.2f}.."
          f"{cam_z.max():.2f} m")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, w2c=w2c_new, K=rec["K"], K_depth=rec["K_depth"],
        names=rec["names"], depth=depth.astype(np.float16),
        points=pts.astype(np.float32), points_rgb=rec["points_rgb"],
        frame_wh=rec["frame_wh"], metric_scale=np.float64(s), T_align=T_rig,
        frame_offset=np.int64(off))
    report = {
        "n_frames": int(ok.sum()), "frame_offset": int(off),
        "scale": s, "center_rms_m": best["rms"], "center_median_m": med,
        "rotation_residual_deg": {"median": rot_med,
                                  "max": float(max(rot_res))},
        "reproj_median_px": reproj_med,
        "cloud_z_mode_m": z_mode,
        "convention": ("FK camera pose treated as OpenCV c2w in the robot "
                       "base frame; see rotation/reprojection residuals"),
    }
    rp = args.out.with_name("align_report.json")
    rp.write_text(json.dumps(report, indent=1))
    print(f"[align] wrote {args.out} and {rp}")


if __name__ == "__main__":
    main()
