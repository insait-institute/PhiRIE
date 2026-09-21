"""Stage 5: register each canonical TRELLIS asset to the scene (paper's
"Pose Matching": 3-DoF translation + 3-DoF rotation + isotropic scale).

Upright prior (both frames are z-up). Per yaw candidate the scale is
estimated on the target's longest axis (most complete under occlusion, least
relatively noise-inflated); candidates scored by CLIPPED-MEAN one-way chamfer
(partial target -> complete mesh; robust to outliers yet sensitive to the
asymmetric features that disambiguate 180-degree flips). ICP refinement
(partial->complete, then inverted), upright re-snap, then two alternating
rounds of 1-D scale polish + translation-only refinement.
Validated on synthetic partial views: scale err <3%, yaw err <5deg.
FoundationPose refinement from the paper is not available here; this is the
module swap. Also evaluates F1@2/4cm against the GT instance mesh.

GT-FREE FALLBACK: when GT instance files aren't available (no
`scans/segments.json` + `scans/segments_anno.json` for this scene -- the
`recon_scenes`/BEHAVIOR layout never has them) or an explicit `--no-gt` /
SIMANY_NO_GT=1 override is set, the F1-against-GT computation is skipped
entirely -- `C.load_gt_instances()` is never called -- and each object's
`eval` field is reported as an explicit `not_applicable` entry (same
value/status/reason shape as `baselines/raw_reconstruction.py`'s
NA_METRICS/`_na()`), never a fabricated number. The Sim(3) pose-alignment
part above (`align_object`) does not depend on GT at all and still runs
and reports real chamfer/scale/size-sanity numbers in this mode.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from agents.core import common as C

N_MESH_SAMPLES = 20_000
YAW_STEP_DEG = 10
ICP_DIST = 0.03
TILT_SNAP_DEG = 15.0
SIGNED_SOURCE_UP_TIE_EPS_M = 1e-9

_OPEN3D_CPU_PYBIND = None

# Proper rotations which map each possible signed source-frame up axis onto
# world +z.  The order is part of the contract: legacy +z wins numerical ties.
SIGNED_SOURCE_UP_HYPOTHESES = (
    ("+z", np.eye(3, dtype=np.float64)),
    ("-z", np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                    dtype=np.float64)),
    ("+x", np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]],
                    dtype=np.float64)),
    ("-x", np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
                    dtype=np.float64)),
    ("+y", np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                    dtype=np.float64)),
    ("-y", np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                    dtype=np.float64)),
)

# Same honesty contract as baselines/raw_reconstruction.py's NA_METRICS/_na():
# an unsupported cell is reported not_applicable with a real reason, never a
# fabricated number.
GT_NOT_APPLICABLE_REASON = (
    "no ground-truth instance annotations are available for this scene "
    "(scans/segments.json + scans/segments_anno.json not both found), or "
    "--no-gt/SIMANY_NO_GT=1 was set; C.load_gt_instances() was never called "
    "-- the Sim(3) pose alignment above still ran and is reported normally, "
    "only the GT-comparison F1 degrades")


def gt_instances_available() -> bool:
    """Cheap existence check for load_gt_instances()'s two hard file
    dependencies, WITHOUT ever reading/parsing either of them. Duplicated
    (not imported) from agents/discover/s0_select_frame.py -- same pattern
    already used elsewhere in this repo for small env-detection helpers
    that must stay in sync manually rather than coupling two otherwise-
    independent stage modules (see droid_static_select.py's FILTER_LADDER
    docstring for the precedent)."""
    return C.SEGMENTS_JSON.exists() and C.SEGMENTS_ANNO_JSON.exists()


def no_gt_requested(no_gt_flag: bool = False) -> bool:
    """--no-gt CLI flag OR SIMANY_NO_GT=1 (C.env() also honours the legacy
    SIMF_NO_GT prefix, same as every other SIMANY_* setting)."""
    return bool(no_gt_flag) or C.env("NO_GT") == "1"


def na_eval() -> dict:
    return {"value": None, "status": "not_applicable",
            "reason": GT_NOT_APPLICABLE_REASON}


def rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_T(s, R, t):
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def apply_T(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def clipped_mean(d):
    return float(np.minimum(d, ICP_DIST).mean())


def sym_score(src, tgt, tgt_tree=None, n_src=4000):
    """Symmetric clipped chamfer. The mesh->target direction saturates at
    ICP_DIST, so unobserved back sides cost a bounded constant while an
    INFLATED mesh (most samples far from any observation) is heavily
    penalized -- one-way chamfer alone lets the scale explode on real,
    depth-bleed-contaminated clouds."""
    d_tm, _ = cKDTree(src).query(tgt, k=1)
    if tgt_tree is None:
        tgt_tree = cKDTree(tgt)
    sub = src[:: max(len(src) // n_src, 1)]
    d_mt, _ = tgt_tree.query(sub, k=1)
    return clipped_mean(d_tm) + clipped_mean(d_mt)


def init_translation(src, tgt):
    """xy: centroid match; z: 2nd-percentile match (visible bottom)."""
    t = np.zeros(3)
    t[:2] = tgt[:, :2].mean(axis=0) - src[:, :2].mean(axis=0)
    t[2] = np.percentile(tgt[:, 2], 2) - np.percentile(src[:, 2], 2)
    return t


def translation_rounds(T, mesh_pts, tgt, n=3):
    src = apply_T(T, mesh_pts)
    tree = cKDTree(src)
    for _ in range(n):
        d, j = tree.query(tgt, k=1)
        keep = d < ICP_DIST
        if keep.sum() < 50:
            break
        T[:3, 3] += (tgt[keep] - src[j[keep]]).mean(axis=0)
        src = apply_T(T, mesh_pts)
        tree = cKDTree(src)
    return T


def open3d_registration_backend():
    """Load only Open3D's CPU pybind registration backend.

    Importing the public :mod:`open3d` package eagerly imports its optional
    visualization stack (Dash, Flask, Pydantic, and their transitive web
    dependencies).  E3 only needs point clouds and point-to-point ICP, so
    pulling that unrelated stack into the fail-closed controller environment
    both expands the execution contract and can make registration fail before
    ICP starts.  Loading the installed CPU extension directly preserves the
    exact Open3D registration implementation without importing Open3D's
    optional Python visualization/web stack.
    """
    global _OPEN3D_CPU_PYBIND
    if _OPEN3D_CPU_PYBIND is not None:
        return _OPEN3D_CPU_PYBIND

    module_name = "open3d.cpu.pybind"
    existing = sys.modules.get(module_name)
    # Open3D's public import may already have loaded the CUDA distribution of
    # the classic (host-memory) geometry API. Both pybind libraries register
    # the same C++ types, so loading the CPU library again raises duplicate
    # type registration errors. Reuse that existing classic ICP API; a fresh
    # controller process still loads only the CPU extension below.
    if existing is None:
        existing = sys.modules.get("open3d.cuda.pybind")
    if existing is not None:
        _OPEN3D_CPU_PYBIND = existing
        return existing

    package_spec = importlib.util.find_spec("open3d")
    if package_spec is None or package_spec.origin is None:
        raise ModuleNotFoundError("Open3D is not installed")
    package_root = Path(package_spec.origin).resolve().parent
    candidates = sorted((package_root / "cpu").glob("pybind*.so"))
    if len(candidates) != 1:
        raise ImportError(
            "expected exactly one Open3D CPU pybind extension, found "
            f"{len(candidates)} below {package_root / 'cpu'}"
        )
    extension_spec = importlib.util.spec_from_file_location(
        module_name, candidates[0]
    )
    if extension_spec is None or extension_spec.loader is None:
        raise ImportError(f"cannot load Open3D CPU extension: {candidates[0]}")
    backend = importlib.util.module_from_spec(extension_spec)
    sys.modules[module_name] = backend
    try:
        extension_spec.loader.exec_module(backend)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    for required in ("geometry", "utility", "pipelines"):
        if not hasattr(backend, required):
            sys.modules.pop(module_name, None)
            raise ImportError(
                f"Open3D CPU extension lacks required namespace: {required}"
            )
    _OPEN3D_CPU_PYBIND = backend
    return backend


def _align_object(mesh_pts, tgt, source_up_hypotheses):
    lo_t, hi_t = np.percentile(tgt, 2, axis=0), np.percentile(tgt, 98, axis=0)
    ext_t = hi_t - lo_t
    k = int(np.argmax(ext_t))  # most-complete / least-noise-inflated axis
    center = mesh_pts.mean(axis=0)
    mc = mesh_pts - center
    tgt_tree = cKDTree(tgt)

    best = None
    for source_up, base_R in source_up_hypotheses:
        hypothesis_best = None
        for yaw in np.deg2rad(np.arange(0, 360, YAW_STEP_DEG)):
            R = rz(yaw) @ base_R
            rot = mc @ R.T
            lo_r, hi_r = (np.percentile(rot, 2, axis=0),
                          np.percentile(rot, 98, axis=0))
            s_yaw = float(ext_t[k] / max(hi_r[k] - lo_r[k], 1e-6))
            # wide multipliers: depth-bleed smear can inflate ext_t by 2-3x,
            # the symmetric score picks the right one
            for smul in (0.5, 0.65, 0.8, 0.9, 1.0, 1.1, 1.25):
                s0 = s_yaw * smul
                src = rot * s0
                t = init_translation(src, tgt)
                score = sym_score(src + t, tgt, tgt_tree)
                # Keep the legacy strict comparison within each hypothesis.
                if hypothesis_best is None or score < hypothesis_best[0]:
                    T = make_T(s0, R, t - s0 * R @ center)
                    hypothesis_best = (score, T)
        # Fixed ordering plus a one-nanometre score margin makes the selected
        # signed axis deterministic and keeps legacy +z on numerical ties.
        if best is None or hypothesis_best[0] < best[0] - SIGNED_SOURCE_UP_TIE_EPS_M:
            best = (*hypothesis_best, source_up, base_R)
    score0, T0, source_up, base_R = best
    s_init, _, _ = C.decompose_similarity(T0)

    # ICP: move the PARTIAL cloud onto the complete mesh, then invert
    o3d = open3d_registration_backend()
    src_pts = apply_T(T0, mesh_pts)
    pc_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tgt))
    pc_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_pts))
    reg = o3d.pipelines.registration.registration_icp(
        pc_src, pc_dst, ICP_DIST, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    T = np.linalg.inv(reg.transformation) @ T0

    # Upright re-snap relative to the selected source-up basis.  Measuring
    # R[2,2] directly is only valid for the legacy +z hypothesis; for the
    # other five bases it would misreport a correct 90/180-degree rotation as
    # ICP tilt and then snap to the wrong frame.
    s, R, t = C.decompose_similarity(T)
    residual_R = R @ base_R.T
    tilt = np.rad2deg(np.arccos(np.clip(residual_R[2, 2], -1, 1)))
    if tilt < TILT_SNAP_DEG:
        yaw = np.arctan2(residual_R[1, 0], residual_R[0, 0])
        T = make_T(s, rz(yaw) @ base_R, t)

    # alternate 1-D scale polish (about object center) + translation rounds;
    # candidates additionally clamped to [0.6, 1.4] x the extent-based init
    for _ in range(2):
        s, R, t = C.decompose_similarity(T)
        obj_ctr = apply_T(T, mesh_pts).mean(axis=0)
        cand = s * np.linspace(0.85, 1.15, 13)
        cand = cand[(cand > 0.6 * s_init) & (cand < 1.4 * s_init)]
        if len(cand) == 0:
            cand = np.array([np.clip(s, 0.6 * s_init, 1.4 * s_init)])
        scores, ts = [], []
        for sc in cand:
            t2 = obj_ctr - (sc / s) * (obj_ctr - t)
            scores.append(sym_score(mesh_pts * sc @ R.T + t2, tgt, tgt_tree))
            ts.append(t2)
        i = int(np.argmin(scores))
        T = make_T(cand[i], R, ts[i])
        T = translation_rounds(T, mesh_pts, tgt)

    d, _ = cKDTree(apply_T(T, mesh_pts)).query(tgt, k=1)
    return T, float(np.median(d)), score0, tilt, source_up


def align_object(mesh_pts, tgt):
    """Legacy z-up alignment API; intentionally returns the original tuple."""
    return _align_object(mesh_pts, tgt, SIGNED_SOURCE_UP_HYPOTHESES[:1])[:4]


def align_object_with_signed_source_up(mesh_pts, tgt):
    """Align while testing all six signed source-up axes.

    Returns the legacy four alignment values followed by the selected axis
    name.  Candidate scoring, ICP, scale polish, and their thresholds are the
    same as :func:`align_object`.
    """
    return _align_object(mesh_pts, tgt, SIGNED_SOURCE_UP_HYPOTHESES)


def align_object_with_alternative_source_up(mesh_pts, tgt):
    """E3 retry alignment over the five non-legacy source-up hypotheses.

    The initial proposal has already consumed ``+z``.  Keeping that hypothesis
    out of the retry search guarantees the retry is a distinct, predeclared
    intervention while leaving :func:`align_object_with_signed_source_up`
    unchanged for E2 and other callers.
    """
    return _align_object(mesh_pts, tgt, SIGNED_SOURCE_UP_HYPOTHESES[1:])


def f1_eval(gen_pts, gt_pts, taus=(0.02, 0.04)):
    tg = cKDTree(gt_pts)
    tp = cKDTree(gen_pts)
    d_gen, _ = tg.query(gen_pts, k=1)   # gen -> gt (precision)
    d_gt, _ = tp.query(gt_pts, k=1)     # gt -> gen (recall)
    out = {}
    for tau in taus:
        p = float((d_gen < tau).mean())
        r = float((d_gt < tau).mean())
        out[f"f1@{int(tau * 1000)}mm"] = {
            "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}
    out["chamfer_mean_m"] = float(d_gen.mean() + d_gt.mean()) / 2
    return out


_MESH_CACHE = {}


def scene_mesh_arrays(*, mesh_sha256=None):
    # Static exporters may supply the digest they have just authenticated;
    # other callers hash the actual mesh before sharing a process-local cache.
    from agents.orchestrator.artifact import sha256_file
    mesh_path = Path(getattr(C, "PIPELINE_MESH_PLY", C.MESH_PLY)).resolve()
    digest = sha256_file(mesh_path) if mesh_sha256 is None else mesh_sha256
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("scene mesh cache requires a content SHA256")
    cache_key = (str(C.SCENE_ID), str(mesh_path), digest)
    if _MESH_CACHE.get("source_identity") != cache_key:
        _MESH_CACHE.clear()
    if "verts" not in _MESH_CACHE:
        from plyfile import PlyData
        # must be the mesh instance vert_idx indexes into (PIPELINE_MESH_PLY ==
        # MESH_PLY unless SIMANY_MESH_SRC overrides; matches make_raycast_scene)
        ply = PlyData.read(str(getattr(C, "PIPELINE_MESH_PLY", C.MESH_PLY)))
        v = ply["vertex"]
        _MESH_CACHE["verts"] = np.stack(
            [np.asarray(v[a], dtype=np.float64) for a in "xyz"], axis=1)
        _MESH_CACHE["faces"] = np.vstack(ply["face"]["vertex_indices"])
        _MESH_CACHE["source_identity"] = cache_key
    return _MESH_CACHE["verts"], _MESH_CACHE["faces"]


def gt_submesh_points(gt, n=10_000):
    import trimesh
    verts, faces = scene_mesh_arrays()
    inset = np.zeros(len(verts), bool)
    inset[gt["vert_idx"]] = True
    fmask = inset[faces].all(axis=1)
    if fmask.sum() < 4:
        return verts[gt["vert_idx"]]
    sub = trimesh.Trimesh(verts, faces[fmask], process=False)
    pts, _ = trimesh.sample.sample_surface(sub, n)
    return np.asarray(pts)


def main(argv=None):
    import trimesh

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gt", action="store_true",
                     help="force GT-free mode (skip F1-vs-GT, report "
                          "not_applicable) even when GT instance files ARE "
                          "present; SIMANY_NO_GT=1 has the same effect")
    args = ap.parse_args(argv)

    objects = json.loads((C.OUT / "objects" / "objects.json").read_text())
    use_gt = gt_instances_available() and not no_gt_requested(args.no_gt)
    gts = {g["object_id"]: g for g in C.load_gt_instances()} if use_gt else None

    results = []
    for meta in objects:
        odir = C.OUT / "objects" / f"obj_{meta['index']:02d}"
        mesh = trimesh.load(odir / "trellis_mesh.ply", process=False)
        mesh_pts, _ = trimesh.sample.sample_surface(mesh, N_MESH_SAMPLES)
        mesh_pts = np.asarray(mesh_pts, dtype=np.float64)
        o3d = open3d_registration_backend()
        tgt = np.asarray(o3d.io.read_point_cloud(str(odir / "points.ply")).points)

        T, chamfer, chamfer_init, tilt = align_object(mesh_pts, tgt)
        s, _, _ = C.decompose_similarity(T)
        rec = {"index": meta["index"], "label": meta["label"], "scale": s,
               "T": T, "chamfer_med_m": chamfer, "chamfer_init_m": chamfer_init,
               "icp_tilt_deg": tilt}

        # size-sanity gate: final world size must roughly match the lifted
        # observation (degenerate TRELLIS assets - e.g. flat cards from blurry
        # crops - otherwise end up meters wide; the paper handles these with
        # human-in-the-loop GUI intervention)
        world_dims = s * (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
        obs_max = float(np.max(meta["extent"]))
        ratio = float(np.max(world_dims)) / max(obs_max, 1e-6)
        rec["world_dims"] = world_dims
        rec["size_ratio_vs_obs"] = ratio
        if np.max(world_dims) > 0.9 or ratio > 2.0 or ratio < 0.4:
            rec["rejected"] = (f"size sanity: world max {np.max(world_dims):.2f}m, "
                               f"{ratio:.1f}x the observed extent")
            print(f"[s5] obj_{meta['index']:02d} REJECTED ({rec['rejected']})")

        if use_gt:
            if meta.get("gt_object_id") is not None:
                gt_pts = gt_submesh_points(gts[meta["gt_object_id"]])
                rec["eval"] = f1_eval(apply_T(T, mesh_pts), gt_pts)
                e = rec["eval"]
                print(f"[s5] {odir.name} {meta['label']}: s={s:.3f} "
                      f"chamfer={chamfer * 1000:.1f}mm "
                      f"F1@20mm={e['f1@20mm']['f1']:.3f} @40mm={e['f1@40mm']['f1']:.3f}")
            else:
                print(f"[s5] {odir.name} {meta['label']}: s={s:.3f} "
                      f"chamfer={chamfer * 1000:.1f}mm (no GT match)")
        else:
            rec["eval"] = na_eval()
            print(f"[s5] {odir.name} {meta['label']}: s={s:.3f} "
                  f"chamfer={chamfer * 1000:.1f}mm (GT comparison not_applicable)")
        C.save_json(odir / "aligned.json", rec)
        results.append(rec)

    C.save_json(C.OUT / "objects" / "aligned_all.json", results)


if __name__ == "__main__":
    main()
