"""Synthetic partial-view stress suite for the s5 Sim(3) registration optimizer
(the "released with the code" suite referenced in the paper's registration
section).

Protocol, per base shape x 3 random draws: sample a complete canonical surface
cloud (the "asset"), apply a random GT Sim(3) (scale in [0.7, 1.4], random
yaw, tilt < 8 deg, translation in a 1 m box), keep only points visible from a
random camera direction (normal dot-product hemisphere cull, elevation
10-50 deg like a handheld view), then register the complete asset to the
partial cloud with the PRODUCTION align_object (same call as factory_align:
20k asset samples, ~6k target points). Two conditions over the SAME 18
geometric cases: "clean" (partiality only) and "corrupt" (+2 mm Gaussian
noise, +5% depth-bleed outliers smeared 2-12 cm along the view ray).

Reported: scale / translation / yaw error vs GT (yaw modulo each shape's
symmetry; translation at the object centroid so it stays well-defined under
exact symmetries). Each miss is classified by re-scoring the production
objective at the GT pose: 180-degree flips on asymmetric shapes
(data-unobservable), SEARCH-MISS (GT scores better -> optimizer local
minimum), or AMBIG (recovered pose scores better -> the partial view itself
does not identify the GT).

Usage: test_align_synthetic.py [--seed N]   (CPU, ~8 min)
"""
import sys
import time

import numpy as np
import trimesh

from agents.core import common as C
from agents.assets.s5_align import align_object, apply_T, make_T, rz, sym_score

N_ASSET = 20_000    # complete asset samples (matches s5/factory_align)
N_VIEW = 40_000     # dense samples for the visibility cull
N_TGT = 6_000       # partial-cloud size (matches factory ALIGN_TGT_SAMPLES)
DRAWS = 3
CONDITIONS = [("clean", 0.0, 0.0), ("corrupt", 0.002, 0.05)]  # (noise m, outlier frac)
SCALE_TOL_PCT = 5.0
TRANS_TOL_MM = 5.0
YAW_TOL_DEG = 10.0


def _mug():  # cylinder + handle: no yaw symmetry, 180-flip distinguishable
    body = trimesh.creation.cylinder(radius=0.12, height=0.28)
    handle = trimesh.creation.box(extents=(0.14, 0.04, 0.16))
    handle.apply_translation((0.18, 0, 0))
    return trimesh.util.concatenate([body, handle])


def _lshape():  # two boxes in an L: no yaw symmetry
    a = trimesh.creation.box(extents=(0.40, 0.15, 0.12))
    b = trimesh.creation.box(extents=(0.15, 0.30, 0.12))
    b.apply_translation((-0.125, 0.215, 0))
    return trimesh.util.concatenate([a, b])


SHAPES = [  # (name, builder, yaw symmetry period deg; 0 = continuous)
    ("box", lambda: trimesh.creation.box(extents=(0.42, 0.26, 0.18)), 180),
    ("cylinder", lambda: trimesh.creation.cylinder(radius=0.13, height=0.36), 0),
    ("cone", lambda: trimesh.creation.cone(radius=0.16, height=0.34), 0),
    ("capsule", lambda: trimesh.creation.capsule(radius=0.09, height=0.28), 0),
    ("mug", _mug, 360),
    ("lshape", _lshape, 360),
]


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def yaw_of(R):
    return np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))


def random_sim3(rng):
    s = rng.uniform(0.7, 1.4)
    yaw = rng.uniform(0, 2 * np.pi)
    tilt = np.deg2rad(rng.uniform(0, 8.0))
    ax = rng.uniform(0, 2 * np.pi)
    axis = np.array([np.cos(ax), np.sin(ax), 0.0])
    R = trimesh.transformations.rotation_matrix(tilt, axis)[:3, :3] @ rz(yaw)
    return make_T(s, R, rng.uniform(-0.5, 0.5, 3))


def partial_view(mesh, T_gt, rng, rng_noise, noise_m, outlier_frac):
    """World-frame partial observation: hemisphere cull (+noise +depth bleed).
    Geometry draws come from `rng` so both conditions see identical cases."""
    pts, fi = trimesh.sample.sample_surface(mesh, N_VIEW,
                                            seed=int(rng.integers(2 ** 31)))
    s, R, t = C.decompose_similarity(T_gt)
    w = np.asarray(pts, dtype=np.float64) * s @ R.T + t
    nrm = mesh.face_normals[fi] @ R.T
    az, el = rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(10, 50))
    view = np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az), np.sin(el)])  # object -> camera
    w = w[nrm @ view > 0.15]
    w = w[:: max(len(w) // N_TGT, 1)][:N_TGT]
    if noise_m > 0:
        w = w + rng_noise.normal(0, noise_m, w.shape)
    n_out = int(len(w) * outlier_frac)
    if n_out:
        i = rng_noise.choice(len(w), n_out, replace=False)
        w[i] -= view * rng_noise.uniform(0.02, 0.12, (n_out, 1))  # past-surface bleed
    return w


def run_condition(cond, noise_m, outlier_frac, seed):
    rng = np.random.default_rng(seed)          # geometry: same for all conditions
    rng_noise = np.random.default_rng(seed + 1)
    print(f"\n=== condition: {cond} (noise {noise_m * 1000:.0f}mm, "
          f"outliers {outlier_frac:.0%}) ===")
    print(f"{'shape':9s} {'draw':>4s} {'s_gt':>6s} {'s_err%':>7s} {'t_err_mm':>9s} "
          f"{'yaw_err':>8s} {'miss':>12s} {'chamf_mm':>9s} {'sec':>5s}")

    rows = []
    for name, builder, sym in SHAPES:
        mesh = builder()
        mesh.apply_translation(-mesh.centroid)
        for draw in range(DRAWS):
            asset, _ = trimesh.sample.sample_surface(
                mesh, N_ASSET, seed=int(rng.integers(2 ** 31)))
            asset = np.asarray(asset, dtype=np.float64)
            T_gt = random_sim3(rng)
            tgt = partial_view(mesh, T_gt, rng, rng_noise, noise_m, outlier_frac)

            t0 = time.time()
            T, chamfer, _, _ = align_object(asset, tgt)
            dt = time.time() - t0

            s_gt, R_gt, _ = C.decompose_similarity(T_gt)
            s_rec, R_rec, _ = C.decompose_similarity(T)
            s_err = abs(s_rec / s_gt - 1) * 100
            ctr = asset.mean(axis=0)[None]
            t_err = float(np.linalg.norm(
                apply_T(T, ctr) - apply_T(T_gt, ctr))) * 1000
            yaw_raw = abs(wrap180(yaw_of(R_rec) - yaw_of(R_gt)))
            if sym == 0:
                yaw_err = None            # continuous symmetry: yaw undefined
            elif sym == 180:
                yaw_err = min(yaw_raw, 180 - yaw_raw)
            else:
                yaw_err = yaw_raw

            miss = (s_err >= SCALE_TOL_PCT or t_err >= TRANS_TOL_MM
                    or (yaw_err is not None and yaw_err >= YAW_TOL_DEG))
            kind = ""
            if miss:  # classify by re-scoring the production objective at GT
                if sym == 360 and yaw_raw >= 150:
                    kind = "FLIP"         # 180-flip on the asymmetric shape
                elif sym_score(apply_T(T_gt, asset), tgt) < \
                        sym_score(apply_T(T, asset), tgt):
                    kind = "SEARCH-MISS"  # GT scores better: local minimum
                else:
                    kind = "AMBIG"        # partial view itself prefers wrong
            rows.append(dict(name=name, s_err=s_err, t_err=t_err,
                             yaw_err=yaw_err, kind=kind))
            ystr = "-" if yaw_err is None else f"{yaw_err:.1f}"
            print(f"{name:9s} {draw:4d} {s_gt:6.3f} {s_err:7.2f} {t_err:9.1f} "
                  f"{ystr:>8s} {kind:>12s} {chamfer * 1000:9.1f} {dt:5.1f}")

    s_errs = np.array([r["s_err"] for r in rows])
    t_errs = np.array([r["t_err"] for r in rows])
    y_errs = np.array([r["yaw_err"] for r in rows if r["yaw_err"] is not None])
    within = sum(r["s_err"] < SCALE_TOL_PCT and r["t_err"] < TRANS_TOL_MM
                 for r in rows)
    fails = [r for r in rows if r["kind"]]
    n_kind = {k: sum(r["kind"] == k for r in fails)
              for k in ("FLIP", "SEARCH-MISS", "AMBIG")}

    print(f"\n[{cond}] summary over {len(rows)} cases (seed={seed})")
    print(f"  scale err %:  median {np.median(s_errs):.2f}  max {s_errs.max():.2f}")
    print(f"  trans err mm: median {np.median(t_errs):.2f}  max {t_errs.max():.2f}")
    print(f"  yaw err deg (mod symmetry, {len(y_errs)} defined): "
          f"median {np.median(y_errs):.2f}  max {y_errs.max():.2f}")
    print(f"  within {SCALE_TOL_PCT:.0f}% scale AND {TRANS_TOL_MM:.0f}mm trans: "
          f"{within}/{len(rows)}")
    print(f"  misses (any of {SCALE_TOL_PCT:.0f}%/{TRANS_TOL_MM:.0f}mm/"
          f"{YAW_TOL_DEG:.0f}deg): {len(fails)} = {n_kind['FLIP']} symmetry flips"
          f" + {n_kind['SEARCH-MISS']} search misses + {n_kind['AMBIG']} ambiguous")
    for r in fails:
        ystr = "-" if r["yaw_err"] is None else f"{r['yaw_err']:.1f}deg"
        print(f"    [{r['kind']}] {r['name']}: s_err {r['s_err']:.2f}% "
              f"t_err {r['t_err']:.1f}mm yaw_err {ystr}")


def main():
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    print(f"[test_align] seed={seed} shapes={len(SHAPES)} draws={DRAWS} "
          f"conditions={[c[0] for c in CONDITIONS]}")
    for cond, noise_m, outlier_frac in CONDITIONS:
        run_condition(cond, noise_m, outlier_frac, seed)


if __name__ == "__main__":
    main()
