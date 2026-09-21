"""COLMAP pose backend for the video pipeline (pycolmap, no system binary).

Gold-standard SfM when quality matters more than wall time: SIFT + sequential
matching (frames come from one continuous video) + incremental mapping with a
single shared PINHOLE camera. Emits the same recon.npz contract as
models/vggt_scene.py, except `depth` is SPARSE (zeros away from triangulated
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
import time
from pathlib import Path

import numpy as np

from robo.manifest.hash import canonical_hash

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


def pose_headers(path):
    """Stream pose headers, discarding inherited POINTS2D observations entirely."""
    from agents.core.common import quat_to_rot_wxyz
    poses = {}
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith('#') or not line.strip():
                continue
            fields = line.split()
            if len(fields) != 10:
                raise ValueError('invalid COLMAP pose header')
            name = fields[9]
            if name in poses or Path(name).name != name or name in {'.', '..'}:
                raise ValueError('duplicate or unsafe camera name')
            matrix = np.eye(4)
            matrix[:3, :3] = quat_to_rot_wxyz(np.array(fields[1:5], dtype=float))
            matrix[:3, 3] = np.array(fields[5:8], dtype=float)
            if not np.isfinite(matrix).all():
                raise ValueError('nonfinite camera pose')
            poses[name] = matrix
            if next(stream, None) is None:
                raise ValueError('missing COLMAP observation line')
    return poses


def uniform_initialization_names(names, limit):
    """Even integer positions including endpoints; no outcomes or random draws."""
    if type(limit) is not int or limit < 2 or len(names) < limit:
        raise ValueError('insufficient official TRAIN frames for initialization budget')
    if names != sorted(set(names)):
        raise ValueError('initialization requires unique sorted TRAIN frame names')
    return [names[index*(len(names)-1)//(limit-1)] for index in range(limit)]


def initialization_names(manifest):
    names = [row['name'] for row in manifest['frames']]
    selection = manifest.get('initialization_selection')
    if selection is None:
        return names
    if (set(selection) != {'strategy', 'max_frames', 'frame_names'}
            or selection['strategy'] != 'uniform_integer_endpoints_v1'):
        raise ValueError('unknown initialization selection protocol')
    selected = uniform_initialization_names(names, selection['max_frames'])
    if selected != selection['frame_names']:
        raise ValueError('initialization frame selection changed')
    return selected


def prepare_training_scene(source_scene, destination, *, max_train_frames=48, initialization_max_frames=None):
    """Copy only official TRAIN RGB and pose values; never open points3D."""
    from agents.core.common import load_intrinsics
    from agents.discover.training_views import select_training_views
    from PIL import Image
    source_scene, destination = Path(source_scene), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    dslr = source_scene / 'dslr'
    sources = {name: {'path': str((dslr / name).resolve()), **file_identity(dslr / name)}
               for name in ('colmap/images.txt', 'nerfstudio/transforms_undistorted.json',
                            'train_test_lists.json')}
    selected, boundary = select_training_views(sorted(pose_headers(dslr / 'colmap/images.txt').items()),
        dslr / 'train_test_lists.json', 1, max_train_frames)
    if (max_train_frames is not None and len(selected) != max_train_frames) or len(selected) < 2:
        raise ValueError('insufficient official training frames for predeclared limit')
    K, width, height, _ = load_intrinsics(dslr / 'nerfstudio/transforms_undistorted.json')
    calibration = {'K': K.tolist(), 'width': width, 'height': height}
    if not np.isfinite(K).all() or min(width, height, K[0, 0], K[1, 1]) <= 0:
        raise ValueError('invalid calibration')
    images = destination / 'dslr/resized_undistorted_images'
    images.mkdir(parents=True)
    frames = []
    for name, matrix in selected:
        source = dslr / 'resized_undistorted_images' / name
        identity = file_identity(source)
        shutil.copyfile(source, images / name)
        if file_identity(images / name) != identity:
            raise ValueError('RGB changed during staging')
        with Image.open(images / name) as image:
            if image.size != (width, height):
                raise ValueError('RGB dimensions differ from frozen calibration')
        frames.append({'name': name, 'w2c': matrix.tolist(), 'image': identity,
                       'source_rgb': str(source.resolve())})
    for name, identity in sources.items():
        if file_identity(dslr / name) != {k: identity[k] for k in ('sha256', 'size_bytes')}:
            raise ValueError('source metadata changed during staging')
    manifest = {'schema_version': 1, 'kind': 'official_train_rgb_poses_only',
        'paper_ready': False, 'boundary': boundary, 'source_metadata': sources,
        'calibration': calibration, 'frames': frames,
        'pose_values_sha256': canonical_hash({r['name']: r['w2c'] for r in frames}),
        'inherited_points_or_tracks_used': False}
    if initialization_max_frames is not None:
        names = [row['name'] for row in frames]
        manifest['initialization_selection'] = {
            'strategy': 'uniform_integer_endpoints_v1', 'max_frames': initialization_max_frames,
            'frame_names': uniform_initialization_names(names, initialization_max_frames)}
    write_new_json(destination / 'training_inputs.json', manifest)
    return manifest


def validate_training_scene(scene, expected_manifest_sha256=None):
    scene = Path(scene)
    path = scene / 'training_inputs.json'
    if expected_manifest_sha256 is not None and file_identity(path)['sha256'] != expected_manifest_sha256:
        raise ValueError('training input manifest changed')
    manifest = json.loads(path.read_text())
    if manifest.get('kind') != 'official_train_rgb_poses_only' or manifest.get('inherited_points_or_tracks_used') is not False:
        raise ValueError('training provenance schema differs')
    boundary = manifest['boundary']
    split = Path(boundary['split']['path'])
    if file_identity(split)['sha256'] != boundary['split']['sha256']:
        raise ValueError('official split changed')
    from robo.eval.fidelity_replacements import _load_split
    train, _ = _load_split(split)
    names = [row['name'] for row in manifest['frames']]
    if names != sorted(train)[:boundary['max_train_frames']] or names != boundary['selected_frames']:
        raise ValueError('staged cameras differ from official train selection')
    images = scene / 'dslr/resized_undistorted_images'
    if sorted(p.name for p in images.iterdir()) != names:
        raise ValueError('extra or missing staged RGB')
    for row in manifest['frames']:
        path = images / row['name']
        if path.is_symlink() or file_identity(path) != row['image']:
            raise ValueError('staged RGB bytes changed')
    initialization_names(manifest)
    if canonical_hash({r['name']: r['w2c'] for r in manifest['frames']}) != manifest['pose_values_sha256']:
        raise ValueError('pose values changed')
    return manifest


def triangulate_training_scene(scene, destination, *, seed=42, num_threads=4):
    """Fresh CPU SIFT/matches + fixed-pose triangulation in the capture frame."""
    import pycolmap
    scene, destination = Path(scene), Path(destination)
    manifest_hash = file_identity(scene / 'training_inputs.json')['sha256']
    manifest = validate_training_scene(scene, manifest_hash)
    destination.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    pycolmap.set_random_seed(seed)
    names = initialization_names(manifest)
    images = scene / 'dslr/resized_undistorted_images'
    K = np.asarray(manifest['calibration']['K'])
    db_path = destination / 'database.db'
    extraction = pycolmap.FeatureExtractionOptions(num_threads=num_threads, use_gpu=False)
    matching = pycolmap.FeatureMatchingOptions(num_threads=num_threads, use_gpu=False)
    verification = pycolmap.TwoViewGeometryOptions()
    verification.ransac.random_seed = seed
    pycolmap.extract_features(db_path, images, image_names=names,
        camera_mode=pycolmap.CameraMode.SINGLE, device=pycolmap.Device.cpu,
        reader_options=pycolmap.ImageReaderOptions(camera_model='PINHOLE',
            camera_params=','.join(map(str, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))),
        extraction_options=extraction)
    pycolmap.match_exhaustive(db_path, matching_options=matching,
        verification_options=verification, device=pycolmap.Device.cpu)
    # Construct an empty reconstruction in memory. No inherited model is loaded.
    rec = pycolmap.Reconstruction()
    db = pycolmap.Database.open(db_path)
    try:
        cameras = db.read_all_cameras()
        for camera in cameras:
            rec.add_camera_with_trivial_rig(camera)
        db_images = db.read_all_images()
    finally:
        db.close()
    if sorted(i.name for i in db_images) != names:
        raise ValueError('fresh feature database image roster differs')
    matrices = {r['name']: np.asarray(r['w2c']) for r in manifest['frames']}
    for image in db_images:
        rec.add_image_with_trivial_frame(pycolmap.Image(name=image.name,
            camera_id=image.camera_id, image_id=image.image_id),
            pycolmap.Rigid3d(matrices[image.name][:3]))
    if rec.num_points3D() != 0:
        raise ValueError('initial model must be empty')
    options = pycolmap.IncrementalPipelineOptions(num_threads=num_threads,
        random_seed=seed, fix_existing_frames=True, ba_refine_focal_length=False,
        ba_refine_principal_point=False, ba_refine_extra_params=False,
        ba_refine_sensor_from_rig=False)
    options.mapper.random_seed = seed
    options.mapper.fix_existing_frames = True
    options.triangulation.random_seed = seed
    model_dir = destination / 'sparse'
    model_dir.mkdir()
    rec = pycolmap.triangulate_points(rec, db_path, images, model_dir,
        clear_points=True, options=options, refine_intrinsics=False)
    validate_training_scene(scene, manifest_hash)
    if rec.num_points3D() < 4 or sorted(i.name for i in rec.images.values()) != names:
        raise ValueError('fresh triangulation has insufficient points or changed cameras')
    for im in rec.images.values():
        if not np.allclose(im.cam_from_world().matrix(), matrices[im.name][:3], rtol=0, atol=1e-10):
            raise ValueError('triangulation changed the frozen pose/metric frame')
    for camera in rec.cameras.values():
        if not np.allclose(camera.calibration_matrix(), K, rtol=0, atol=1e-10):
            raise ValueError('triangulation changed frozen intrinsics')
    rec.extract_colors_for_all_images(images)
    tracks = []
    for pid in sorted(rec.points3D):
        point = rec.points3D[pid]
        observations = [{'frame': rec.images[e.image_id].name, 'point2D_idx': e.point2D_idx,
                         'xy': rec.images[e.image_id].points2D[e.point2D_idx].xy.tolist()}
                        for e in point.track.elements]
        if len(observations) < 2 or any(o['frame'] not in names for o in observations) or not np.isfinite(point.xyz).all():
            raise ValueError('invalid train-only point track')
        tracks.append({'point3D_id': pid, 'xyz': point.xyz.tolist(),
                       'rgb': point.color.tolist(), 'observations': observations})
    write_new_json(destination / 'tracks.json', tracks)
    rec.export_PLY(destination / 'init_points.ply')
    validate_training_scene(scene, manifest_hash)
    report = {'schema_version': 1, 'kind': 'fresh_train_only_triangulation',
        'paper_ready': False, 'source_input_sha256': manifest_hash,
        'seed': seed, 'pycolmap_version': pycolmap.__version__,
        'initial_point_count': 0, 'clear_points': True, 'refine_intrinsics': False,
        'frame_names': names, 'n_points': rec.num_points3D(),
        'pose_values_sha256': manifest['pose_values_sha256'],
        'tracks': file_identity(destination / 'tracks.json'),
        'init_ply': file_identity(destination / 'init_points.ply'),
        'database': file_identity(db_path), 'wall_s': time.monotonic() - start,
        'options': {'num_threads': num_threads, 'sift_max_num_features': extraction.sift.max_num_features,
                    'matching': 'exhaustive_cpu', 'fix_existing_frames': True}}
    if 'initialization_selection' in manifest:
        report['initialization_selection'] = manifest['initialization_selection']
    write_new_json(destination / 'init_manifest.json', report)
    return report


def validate_training_initialization(scene, init_ply, init_manifest):
    """Training gate: reject unproven/inherited PLY and changed track lineage."""
    init_ply, init_manifest = Path(init_ply), Path(init_manifest)
    report = json.loads(init_manifest.read_text())
    if report.get('kind') != 'fresh_train_only_triangulation' or report.get('initial_point_count') != 0 or report.get('clear_points') is not True or report.get('refine_intrinsics') is not False:
        raise ValueError('initialization is not fresh fixed-pose training triangulation')
    manifest = validate_training_scene(scene, report['source_input_sha256'])
    if init_ply.is_symlink() or file_identity(init_ply) != report['init_ply']:
        raise ValueError('initialization PLY identity differs')
    tracks_path = init_manifest.parent / 'tracks.json'
    if file_identity(tracks_path) != report['tracks']:
        raise ValueError('initialization track bytes differ')
    names = initialization_names(manifest)
    if report.get('initialization_selection') != manifest.get('initialization_selection'):
        raise ValueError('initialization subset declaration differs from frozen training input')
    if report['frame_names'] != names or report['pose_values_sha256'] != manifest['pose_values_sha256']:
        raise ValueError('initialization camera lineage differs')
    tracks = json.loads(tracks_path.read_text())
    if len(tracks) != report['n_points'] or len(tracks) < 4:
        raise ValueError('initialization point count differs')
    for point in tracks:
        if len(point['observations']) < 2 or any(o['frame'] not in names for o in point['observations']):
            raise ValueError('initialization track escapes official training inputs')
    return manifest, report


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
