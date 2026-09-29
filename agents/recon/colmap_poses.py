"""COLMAP pose backend for the video pipeline (pycolmap, no system binary).

Gold-standard SfM when quality matters more than wall time: SIFT + sequential
matching (frames come from one continuous video) + incremental mapping with a
single shared PINHOLE camera. Emits the same recon.npz contract as
agents/models/vggt_scene.py, except `depth` is SPARSE (zeros away from triangulated
track projections) - agents/recon/metricize.py handles that (its valid mask
keys on depth > 0) and densifies the init cloud from metric monodepth.

Fails loudly if fewer than MIN_REGISTERED_FRAC of the frames register, so the
launcher can fall back to the feed-forward backend instead of training a
splat on a broken reconstruction.

CPU-heavy (pycolmap wheels ship CPU SIFT), ~5-15 min for 160-240 frames.
Main .venv. Usage:
    python -m agents.recon.colmap_poses --images-dir D --out recon.npz \
        [--workdir W]
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

MIN_REGISTERED_FRAC = 0.8
SEQ_OVERLAP = 15          # consecutive-frame matching window; quadratic
                          # overlap adds power-of-two long-range pairs, which
                          # closes loops when the camera revisits a viewpoint
MAX_SPARSE_POINTS = 2_000_000


def cpu_thread_options(num_threads):
    """Bound native COLMAP pools; affinity/OMP alone do not size SIFT pools."""
    import pycolmap
    if type(num_threads) is not int or num_threads < 1:
        raise ValueError('COLMAP num_threads must be a positive integer')
    extraction = pycolmap.FeatureExtractionOptions(num_threads=num_threads)
    matching = pycolmap.FeatureMatchingOptions(num_threads=num_threads)
    pipeline = pycolmap.IncrementalPipelineOptions(num_threads=num_threads)
    pipeline.mapper.num_threads = num_threads
    if extraction.use_gpu or matching.use_gpu or pipeline.ba_use_gpu:
        raise ValueError('explicit CPU thread mode requires CPU-only COLMAP defaults')
    return extraction, matching, pipeline


def run_sfm(images_dir: Path, workdir: Path, *, num_threads=None):
    import pycolmap
    db = workdir / "database.db"
    sparse = workdir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    extraction_kwargs, matching_kwargs, mapping_kwargs = {}, {}, {}
    if num_threads is not None:
        import os
        extraction, matching, pipeline = cpu_thread_options(num_threads)
        affinity = sorted(os.sched_getaffinity(0))
        if len(affinity) < num_threads:
            raise ValueError('COLMAP thread request exceeds assigned CPU affinity')
        extraction_kwargs['extraction_options'] = extraction
        matching_kwargs['matching_options'] = matching
        mapping_kwargs['options'] = pipeline
        write_new_json(workdir / 'thread_options.json', {
            'schema_version': 1, 'pycolmap_version': pycolmap.__version__,
            'requested_threads': num_threads, 'cpu_affinity': affinity,
            'effective_threads': {'extraction': extraction.num_threads, 'matching': matching.num_threads,
                                  'pipeline': pipeline.num_threads, 'mapper': pipeline.mapper.num_threads},
            'geometry_and_matching_thresholds': 'unchanged pycolmap defaults',
            'gpu_enabled': bool(extraction.use_gpu or matching.use_gpu or pipeline.ba_use_gpu)})
    pycolmap.extract_features(
        db, images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=pycolmap.ImageReaderOptions(camera_model="PINHOLE"), **extraction_kwargs)
    pycolmap.match_sequential(
        db,
        pairing_options=pycolmap.SequentialPairingOptions(
            overlap=SEQ_OVERLAP, quadratic_overlap=True), **matching_kwargs)
    recs = pycolmap.incremental_mapping(db, images_dir, sparse, **mapping_kwargs)
    if not recs:
        raise SystemExit("[colmap] mapping produced no reconstruction")
    return max(recs.values(), key=lambda r: r.num_reg_images())


def file_identity(path):
    path = Path(path)
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'input changed during hashing: {path}')
    return {'sha256': h.hexdigest(), 'size_bytes': after.st_size}


def write_new_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workdir", type=Path, default=None,
                    help="keep COLMAP artifacts here (default: temp dir)")
    ap.add_argument('--num-threads', type=int, default=None,
                    help='explicit native extraction/matching/mapping pool size; no threshold changes')
    args = ap.parse_args()

    frames = sorted(p.name for p in args.images_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"[colmap] no *.jpg in {args.images_dir}")

    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="colmap_"))
    workdir.mkdir(parents=True, exist_ok=True)
    rec = (run_sfm(args.images_dir, workdir, num_threads=args.num_threads)
           if args.num_threads is not None else run_sfm(args.images_dir, workdir))

    reg = {im.name: im for im in rec.images.values()}
    frac = len(reg) / len(frames)
    print(f"[colmap] registered {len(reg)}/{len(frames)} frames "
          f"({frac:.0%}), {rec.num_points3D()} sparse points")
    if frac < MIN_REGISTERED_FRAC:
        raise SystemExit(
            f"[colmap] only {frac:.0%} of frames registered "
            f"(need {MIN_REGISTERED_FRAC:.0%}) - fall back to the "
            f"feed-forward backend")

    cam = rec.cameras[next(iter(rec.images.values())).camera_id]
    K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
    W, H = int(cam.width), int(cam.height)

    names = [n for n in frames if n in reg]
    w2c = np.zeros((len(names), 4, 4), np.float64)
    depth = np.zeros((len(names), H, W), np.float16)
    for i, n in enumerate(names):
        im = reg[n]
        m = np.asarray(im.cam_from_world().matrix())   # 3x4 OpenCV w2c
        w2c[i, :3, :] = m
        w2c[i, 3, 3] = 1.0
        # sparse depth at triangulated track projections (for metric scaling)
        for p2d in im.points2D:
            if not p2d.has_point3D():
                continue
            X = rec.points3D[p2d.point3D_id].xyz
            z = float(m[2, :3] @ X + m[2, 3])
            u, v = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
            if 0 <= v < H and 0 <= u < W and z > 0:
                depth[i, v, u] = z

    ids = sorted(rec.points3D)
    if len(ids) > MAX_SPARSE_POINTS:
        ids = [ids[j] for j in
               np.random.RandomState(0).choice(len(ids), MAX_SPARSE_POINTS,
                                               replace=False)]
    pts = np.array([rec.points3D[j].xyz for j in ids], np.float32)
    rgb = np.array([rec.points3D[j].color for j in ids], np.uint8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, w2c=w2c, K=K, K_depth=K, names=np.array(names),
        depth=depth, points=pts, points_rgb=rgb,
        frame_wh=np.array([W, H]), backend="colmap")
    print(f"[colmap] wrote {args.out} ({len(names)} frames, {len(pts)} pts)")
    if args.workdir is None:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
