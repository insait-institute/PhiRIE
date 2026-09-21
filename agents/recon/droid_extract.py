"""DROID raw episode -> wrist frames + FK camera trajectory for recon.

Extracts wrist-camera frames from recordings/MP4/<serial>.mp4 (full-res LEFT
rgb) and dumps the forward-kinematics camera trajectory + joints + gripper +
metadata so agents/recon/align_to_traj.py can Umeyama-align COLMAP poses into
the metric ROBOT BASE frame. Writes into the recon-scene contract dir:

  <root>/data/<scene>/<camera>_frames/frame_<mp4idx:06d>.jpg
  <root>/data/<scene>/gt/droid_traj.json

Pose convention (VERIFIED empirically on IPRL Thu_Aug_24_21:29:53_2023):
  /observation/camera_extrinsics/<serial>_left is (T,6) [x,y,z, rx,ry,rz]
  with EXTRINSIC-xyz euler angles (R = Rz(rz) @ Ry(ry) @ Rx(rx)) and is the
  CAMERA POSE IN THE ROBOT BASE FRAME, i.e. c2w with world = robot base
  (metric, z-up, z=0 at the mount plane - usually the TABLE, not the floor).
  Evidence: with this convention inv(T_ee(t)) @ T_cam(t) is constant to
  <1e-12 m over all 402 steps (rigid wrist mount), ext-camera extrinsics have
  exactly zero variance, and the wrist-camera path length (3.034 m) matches
  the cartesian_position path length (3.086 m). All alternative conventions
  (intrinsic euler, w2c direction) leave >= 1 cm / 40 deg of spread.

Frame filter (dynamic-scene caveat: the arm sweeps through wrist frames; v1
accepts ghosting of the manipulandum): prefer steps where the gripper is not
actuating (|d gripper_position| < GRIP_EPS) AND joint speeds are low
(max |joint_velocity| below a ladder of thresholds), relaxing the ladder
until >= MIN_KEPT frames survive, else keep everything. Thresholds chosen
from the two pilot episodes (jv<0.3 & gv<0.01 keeps 106/402 and 113/260).

MP4-vs-h5 off-by-one: the MP4 carries T-1 frames for T trajectory steps;
mp4 frame k is provisionally mapped to h5 step k (h5_index == mp4_index) and
align_to_traj.py resolves the true constant offset by residual search.

Env: h5py is in NONE of the three SimAny envs - run under the artifixer venv
(h5py + numpy; ffmpeg/ffprobe are system binaries), from the repo root:

  /group/worldcept/artifixer/.venv/bin/python -m agents.recon.droid_extract \
      --episode IPRL/success/2023-08-24/Thu_Aug_24_21:29:53_2023 \
      --scene-name droid_iprl_0824 \
      --root /group/worldcept/PhiRIE/code/SimAny/data/recon_scenes
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from agents.core.common import save_json

RAW_ROOT = Path("/group/worldcept/PhiRIE/data/droid/raw")
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"

MAX_FRAMES = 240        # same budget as agents/recon/frames.py (gsplat cost)
MAX_LONG_SIDE = 1752    # keep downstream pixel gates in their validated band
JPEG_Q = 2              # ~95% quality; frames feed COLMAP and gsplat

MIN_KEPT = 80           # a splat needs this much viewpoint coverage
# (max|joint_velocity| rad/s, |d gripper_position| per step) ladder, relaxed
# until MIN_KEPT frames survive; the last rung keeps everything. Gripper
# range is 0..~0.9, so 0.01 means "not actuating".
FILTER_LADDER = ((0.15, 0.01), (0.30, 0.01), (0.50, 0.01),
                 (np.inf, np.inf))


def euler_xyz_to_R(e):
    """Extrinsic-xyz euler -> rotation matrix: R = Rz(rz) @ Ry(ry) @ Rx(rx).

    Matches scipy Rotation.from_euler('xyz', e) to 1e-15 (verified); written
    out so the h5py env needs no scipy."""
    e = np.asarray(e, dtype=np.float64)
    a, b, c = e[..., 0], e[..., 1], e[..., 2]
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    R = np.empty(e.shape[:-1] + (3, 3))
    R[..., 0, 0] = cc * cb
    R[..., 0, 1] = cc * sb * sa - sc * ca
    R[..., 0, 2] = cc * sb * ca + sc * sa
    R[..., 1, 0] = sc * cb
    R[..., 1, 1] = sc * sb * sa + cc * ca
    R[..., 1, 2] = sc * sb * ca - cc * sa
    R[..., 2, 0] = -sb
    R[..., 2, 1] = cb * sa
    R[..., 2, 2] = cb * ca
    return R


def pose6_to_c2w(v):
    """(...,6) [x y z rx ry rz] -> (...,4,4) camera pose in robot base."""
    v = np.asarray(v, dtype=np.float64)
    T = np.zeros(v.shape[:-1] + (4, 4))
    T[..., :3, :3] = euler_xyz_to_R(v[..., 3:])
    T[..., :3, 3] = v[..., :3]
    T[..., 3, 3] = 1.0
    return T


def count_mp4_frames(video: Path) -> int:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
         str(video)], check=True, capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def select_steps(jv_max_abs, dgrip_abs, n_valid):
    """Filter ladder -> (kept h5 indices, filter description dict). The last
    rung is (inf, inf), so this always returns something."""
    for jv_max, grip_eps in FILTER_LADDER:
        keep = np.where((jv_max_abs[:n_valid] < jv_max) &
                        (dgrip_abs[:n_valid] < grip_eps))[0]
        if len(keep) >= MIN_KEPT or not np.isfinite(jv_max):
            return keep, {
                "jv_max": float(jv_max) if np.isfinite(jv_max) else None,
                "grip_eps": float(grip_eps) if np.isfinite(grip_eps)
                else None,
                "n_kept": int(len(keep)), "n_total": int(n_valid)}


def extract_frames(video: Path, mp4_indices, out_dir: Path):
    """Decode every frame once (no `-vf select`), then keep only the wanted
    indices by renaming; delete the rest.

    ROOT CAUSE (found empirically, see docs/DROID_PROTOCOL.md): ffmpeg's
    `-vf select='eq(n,i0)+eq(n,i1)+...'` pattern used previously blows past a
    fixed-size internal stack in libavutil's expression evaluator
    (av_expr_parse, eval.c) once the OR-chain has roughly >100 terms. Above
    that boundary ffmpeg fails DURING FILTER GRAPH INIT (before any frame is
    processed) with "[AVFilterGraph] Error initializing filters" /
    "Error opening output files: Cannot allocate memory" - a real ffmpeg
    return code (AVERROR(ENOMEM) from the expression parser's internal
    allocator), not an OS/cgroup/`--mem` limit: reproduced identically with
    `ulimit -a` fully unlimited and 64-100G of job memory untouched.
    Confirmed by bisection on this exact video: 100 eq() terms parse fine,
    106 fail every time (exit 244) regardless of --mem.
    select_steps()'s relaxation ladder can legitimately keep any subset of
    up to MAX_FRAMES indices with no guaranteed run-length structure (the
    final rung keeps 100% of frames, and the max-frames cap subsamples via
    linspace, which shatters runs into scattered singletons) - i.e. the
    >100-term case is not an edge case, it is the common case for longer
    episodes. Range-compressing the eq() chain into fewer `between(n,a,b)`
    terms would still fail on scattered selections, so this decodes the
    whole (short: <=1140 frames in the local raw set) episode once instead,
    which sidesteps the expression parser altogether.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    long_cap = (f"scale=w='if(gte(iw\\,ih)\\,min(iw\\,{MAX_LONG_SIDE})\\,-2)':"
                f"h='if(gte(iw\\,ih)\\,-2\\,min(ih\\,{MAX_LONG_SIDE}))'")
    tmp_dir = out_dir / "_all_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_pat = tmp_dir / "all_%06d.jpg"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", long_cap, "-fps_mode", "vfr",
         "-q:v", str(JPEG_Q), "-start_number", "0", str(tmp_pat)],
        check=True)
    tmp = sorted(tmp_dir.glob("all_*.jpg"))
    n_needed = max(int(i) for i in mp4_indices) + 1
    if len(tmp) < n_needed:
        raise SystemExit(f"[droid_extract] ffmpeg decoded {len(tmp)} frames, "
                         f"need index up to {n_needed - 1}")
    names = []
    for idx in mp4_indices:
        src = tmp_dir / f"all_{int(idx):06d}.jpg"
        if not src.exists():
            raise SystemExit(f"[droid_extract] decoded frame {idx} missing "
                             f"from {tmp_dir}")
        nm = f"frame_{int(idx):06d}.jpg"
        src.rename(out_dir / nm)
        names.append(nm)
    for p in tmp_dir.glob("all_*.jpg"):
        p.unlink()  # drop unwanted decoded frames (already excluded above)
    tmp_dir.rmdir()
    return names


def input_identity(path):
    """Small raw inputs are byte-hashed; symlink aliases are not evidence."""
    from agents.recon.colmap_poses import file_identity
    path = Path(path).absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError(f'symlink input: {path}')
    return {'path': str(path), **file_identity(path)}


def verify_input(ref):
    if input_identity(ref['path']) != ref:
        raise ValueError('input identity changed: ' + ref['path'])
    return Path(ref['path'])


def prospective_environment(code_root):
    """Isolate runtime probes, extraction, fitting and evaluation identically."""
    import os
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith(('PYTHON', 'SIMANY_', 'SIMF_', 'LD_', 'OMP_', 'OPENBLAS_', 'MKL_'))}
    environment.update(PYTHONPATH=str(Path(code_root).resolve()), PYTHONNOUSERSITE='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', OMP_NUM_THREADS='8',
        OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', SIMANY_AUTO='1',
        SIMANY_MESH_SRC='derived', SIMANY_NO_GT='1', SIMANY_ROOT=str(Path(code_root).resolve()))
    return environment


def runtime_manifest(packages):
    """Targeted package source/RECORD/native closure, not entire OS closure."""
    import importlib
    import importlib.metadata
    import site
    import sys
    if site.ENABLE_USER_SITE:
        raise ValueError('prospective runtime requires disabled user site')
    result = {'invocation': sys.executable, 'python': input_identity(Path(sys.executable).resolve()),
              'python_version': sys.version, 'user_site_enabled': False, 'packages': {},
              'scope': 'Python binary plus package source, RECORD and native library bytes'}
    for entry in packages:
        package, distribution_name = (entry, entry) if isinstance(entry, str) else entry
        distribution = importlib.metadata.distribution(distribution_name)
        module = importlib.import_module(package)
        files = sorted(distribution.files or [], key=str)
        record = [p for p in files if str(p).endswith('.dist-info/RECORD')]
        if len(record) != 1:
            raise ValueError('installed package requires exactly one RECORD: ' + package)
        selected = [p for p in files if str(p).endswith('.py') or '.so' in p.name]
        result['packages'][package] = {'version': distribution.version,
            'entrypoint': input_identity(module.__file__),
            'record': input_identity(distribution.locate_file(record[0])),
            'files': [input_identity(distribution.locate_file(p).resolve()) for p in selected]}
    return result


def prospective_split(video_frames, trajectory_steps):
    """Metadata-only chronological 80/20 split, before reading FK values.

    Four excluded video indices separate the TRAIN/reference windows. Thus
    their FK index unions remain disjoint for EVERY offset hypothesis [-2,2].
    All RGB selected here may enter SfM; only robot-kinematic evidence is held
    out. No velocity/gripper filtering or post-registration split selection.
    """
    if (type(video_frames) is not int or type(trajectory_steps) is not int
            or not 0 < video_frames <= trajectory_steps + 2):
        raise ValueError('invalid video/trajectory metadata')
    valid = list(range(2, min(video_frames, trajectory_steps - 2)))
    if len(valid) > 240:
        valid = [valid[i * (len(valid) - 1) // 239] for i in range(240)]
    boundary = 2 + (min(video_frames, trajectory_steps - 2) - 2) * 4 // 5
    train = [i for i in valid if i <= boundary - 3]
    held = [i for i in valid if i >= boundary + 2]
    union = lambda ids: sorted({i + off for i in ids for off in range(-2, 3)})
    if len(train) < 10 or len(held) < 5:
        raise ValueError('metadata split needs at least10 TRAIN and5 reference frames')
    return {'rule': 'chronological80_20_guard4_uniform240_v1',
            'video_frames': video_frames, 'trajectory_steps': trajectory_steps,
            'max_frames': 240, 'offsets': [-2, -1, 0, 1, 2],
            'boundary': boundary, 'rgb_indices': sorted(train + held),
            'train_video_indices': train, 'held_out_video_indices': held,
            'train_fk_indices': union(train), 'held_out_fk_indices': union(held),
            'excluded_guard_indices': list(range(boundary - 2, boundary + 2))}


def validate_frame_plan(plan, *, verify_bytes=True):
    if (plan.get('schema_version') != 1 or plan.get('scope') != 'droid_train_only_alignment'
            or plan.get('camera') != 'wrist'):
        raise ValueError('unsupported prospective frame plan')
    split = plan['split']
    expected = prospective_split(split['video_frames'], split['trajectory_steps'])
    if split != expected or set(split['train_fk_indices']) & set(split['held_out_fk_indices']):
        raise ValueError('split/offset hypothesis leakage or changed metadata rule')
    if set(plan['inputs']) != {'metadata', 'video', 'trajectory'}:
        raise ValueError('raw input roster changed')
    episode = Path(plan['episode'])
    if (not episode.is_absolute() or Path(plan['inputs']['metadata']['path']).parent != episode
            or not Path(plan['inputs']['metadata']['path']).name.startswith('metadata_')
            or Path(plan['inputs']['trajectory']['path']) != episode / 'trajectory.h5'
            or Path(plan['inputs']['video']['path']) != episode / 'recordings/MP4' / (plan['serial'] + '.mp4')):
        raise ValueError('raw input path/episode binding differs')
    if verify_bytes:
        for ref in plan['inputs'].values():
            verify_input(ref)
    return plan


def prepare_frame_plan(episode):
    """Read only header metadata and H5 shapes, never robot pose values."""
    import h5py
    episode = Path(episode).absolute()
    metadata = sorted(episode.glob('metadata_*.json'))
    if len(metadata) != 1:
        raise ValueError('exactly one episode metadata file required')
    meta = json.loads(metadata[0].read_text())
    serial = str(meta['wrist_cam_serial'])
    trajectory = episode / 'trajectory.h5'
    video = episode / 'recordings/MP4' / (serial + '.mp4')
    refs = {k: input_identity(p) for k, p in
            [('metadata', metadata[0]), ('video', video), ('trajectory', trajectory)]}
    with h5py.File(trajectory, 'r') as stream:
        shape = stream['observation/camera_extrinsics/' + serial + '_left'].shape
        if len(shape) != 2 or shape[1] != 6:
            raise ValueError('camera trajectory must have shape(T,6)')
    plan = {'schema_version': 1, 'scope': 'droid_train_only_alignment',
            'episode': str(episode), 'camera': 'wrist', 'serial': serial,
            'inputs': refs, 'split': prospective_split(count_mp4_frames(video), int(shape[0]))}
    return validate_frame_plan(plan)


def read_planned_fk(plan, role):
    """Index H5 before materializing values; never load the full trajectory."""
    import h5py
    validate_frame_plan(plan)
    if role not in {'train', 'held_out'}:
        raise ValueError('unsupported evidence role')
    indices = plan['split'][role + '_fk_indices']
    with h5py.File(plan['inputs']['trajectory']['path'], 'r') as stream:
        data = stream['observation/camera_extrinsics/' + plan['serial'] + '_left']
        if list(data.shape) != [plan['split']['trajectory_steps'], 6]:
            raise ValueError('trajectory metadata changed')
        values = np.asarray(data[indices], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite planned FK evidence')
    return {str(i): pose.tolist() for i, pose in zip(indices, pose6_to_c2w(values))}


def extract_prospective(plan_path, destination):
    from agents.recon.colmap_poses import write_new_json
    plan_path, destination = Path(plan_path), Path(destination)
    plan = validate_frame_plan(json.loads(plan_path.read_text()))
    destination.mkdir(parents=True, exist_ok=False)
    names = extract_frames(Path(plan['inputs']['video']['path']),
                           plan['split']['rgb_indices'], destination / 'wrist_frames')
    train = {'schema_version': 1, 'role': 'train_only', 'plan': input_identity(plan_path),
             'fk_by_index': read_planned_fk(plan, 'train')}
    write_new_json(destination / 'train_trajectory.json', train)
    write_new_json(destination / 'extraction_manifest.json', {
        'schema_version': 1, 'plan': input_identity(plan_path),
        'train_trajectory': input_identity(destination / 'train_trajectory.json'),
        'frames': [input_identity(destination / 'wrist_frames' / name) for name in names]})


def export_reference(plan_path, fit_directory, destination, *, validator_python=None):
    """Validate frozen constructor first; only then touch held-out FK values."""
    from agents.recon.align_to_traj import validate_train_fit
    from agents.recon.colmap_poses import write_new_json
    if validator_python is None:
        fit = validate_train_fit(fit_directory)
    else:
        # H5 reading and fitting use separate existing environments. Replay in
        # the original fitting runtime; never relax equality across NumPy builds.
        declared = json.loads((Path(fit_directory) / 'fit.json').read_text())
        if str(validator_python) != declared['runtime']['invocation']:
            raise ValueError('fit validator interpreter differs')
        verify_input(declared['runtime']['python'])
        command = [str(validator_python), '-c',
                   'import json,sys;from agents.recon.align_to_traj import validate_train_fit;print("E7_FIT_JSON="+json.dumps(validate_train_fit(sys.argv[1])))',
                   str(fit_directory)]
        code = Path(__file__).resolve().parents[2]
        stdout = subprocess.check_output(command, text=True, cwd=code, env=prospective_environment(code))
        payloads = [s[len('E7_FIT_JSON='):] for s in stdout.splitlines() if s.startswith('E7_FIT_JSON=')]
        if len(payloads) != 1:
            raise ValueError('fit validator did not return one authenticated payload')
        fit = json.loads(payloads[0])
    plan_path = Path(plan_path)
    if fit['plan'] != input_identity(plan_path):
        raise ValueError('fit belongs to a different reference plan')
    if Path(destination).exists():
        raise FileExistsError('reference destination already exists')
    plan = validate_frame_plan(json.loads(plan_path.read_text()))
    reference = {'schema_version': 1, 'role': 'held_out_only',
                 'plan': input_identity(plan_path),
                 'fit': input_identity(Path(fit_directory) / 'fit.json'),
                 'fk_by_index': read_planned_fk(plan, 'held_out')}
    write_new_json(destination, reference)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare-plan', type=Path)
    ap.add_argument('--frame-plan', type=Path)
    ap.add_argument('--reference-fit', type=Path)
    ap.add_argument('--fit-validator-python')
    ap.add_argument('--prospective-out', type=Path)
    ap.add_argument("--episode",
                    help="path under raw/ (lab/success/date/ts) or absolute")
    ap.add_argument("--scene-name")
    ap.add_argument("--root", type=Path)
    ap.add_argument("--camera", default="wrist",
                    choices=["wrist", "ext1", "ext2"],
                    help="wrist (default): moving camera, the only one SfM "
                         "can reconstruct from alone")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth kept frame before the max-frames cap")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    args = ap.parse_args()

    if args.prepare_plan:
        from agents.recon.colmap_poses import write_new_json
        if not args.episode:
            ap.error('--prepare-plan requires --episode')
        write_new_json(args.prepare_plan, prepare_frame_plan(args.episode))
        return
    if args.frame_plan:
        if not args.prospective_out:
            ap.error('--frame-plan requires --prospective-out')
        if args.reference_fit:
            export_reference(args.frame_plan, args.reference_fit, args.prospective_out,
                             validator_python=args.fit_validator_python)
        else:
            extract_prospective(args.frame_plan, args.prospective_out)
        return
    if not all((args.episode, args.scene_name, args.root)):
        ap.error('legacy extraction requires --episode, --scene-name and --root')

    import h5py

    ep_dir = Path(args.episode)
    if not ep_dir.is_absolute():
        ep_dir = args.raw_root / args.episode
    metas = sorted(ep_dir.glob("metadata_*.json"))
    if not metas:
        raise SystemExit(f"[droid_extract] no metadata_*.json in {ep_dir}")
    meta = json.loads(metas[0].read_text())
    serial = str(meta[f"{args.camera}_cam_serial"])
    video = ep_dir / "recordings" / "MP4" / f"{serial}.mp4"
    if not video.exists():
        raise SystemExit(f"[droid_extract] missing {video} - fetch with "
                         f"run/gcs_fetch.py or run/fetch_droid_raw.py")

    with h5py.File(ep_dir / "trajectory.h5", "r") as f:
        cam6 = np.asarray(
            f[f"/observation/camera_extrinsics/{serial}_left"])
        joints = np.asarray(f["/action/joint_position"])
        jv = np.asarray(f["/action/joint_velocity"])
        grip = np.asarray(f["/observation/robot_state/gripper_position"])
        ext_static = {}
        for cam in ("ext1", "ext2"):
            s = str(meta.get(f"{cam}_cam_serial", ""))
            key = f"/observation/camera_extrinsics/{s}_left"
            if s and key in f:
                v = np.asarray(f[key])
                ext_static[cam] = {"serial": s,
                                   "c2w_base": pose6_to_c2w(v[0]).tolist(),
                                   "pos_std_m": v[:, :3].std(0).tolist()}

    T = len(cam6)
    n_mp4 = count_mp4_frames(video)
    if not (0 < n_mp4 <= T + 2):
        raise SystemExit(f"[droid_extract] MP4 has {n_mp4} frames for {T} "
                         f"trajectory steps - correspondence assumption "
                         f"broken, inspect the episode")
    n_valid = min(T, n_mp4)   # provisional h5_index == mp4_index

    keep, filt = select_steps(np.abs(jv).max(1), np.abs(np.gradient(grip)),
                              n_valid)
    keep = keep[::max(1, args.stride)]
    if len(keep) > args.max_frames:
        sub = np.unique(np.linspace(0, len(keep) - 1,
                                    args.max_frames).astype(int))
        keep = keep[sub]
    print(f"[droid_extract] {ep_dir.name}: T={T}, mp4={n_mp4}, filter "
          f"jv<{filt['jv_max']} |dgrip|<{filt['grip_eps']} kept "
          f"{filt['n_kept']}/{filt['n_total']}, stride {args.stride} + cap "
          f"{args.max_frames} -> {len(keep)} frames")

    scene_dir = args.root / "data" / args.scene_name
    frames_dir = scene_dir / f"{args.camera}_frames"
    names = extract_frames(video, [int(i) for i in keep], frames_dir)

    c2w_all = pose6_to_c2w(cam6)
    traj = {
        "convention": (
            "c2w_base: 4x4 CAMERA POSE IN THE ROBOT BASE FRAME (c2w, world="
            "robot base; metric, z-up, z=0 at the mount plane - typically "
            "the TABLE). Source: /observation/camera_extrinsics/<serial>_"
            "left (T,6) [x,y,z,rx,ry,rz], EXTRINSIC-xyz euler "
            "(R=Rz@Ry@Rx). Verified: inv(T_ee) @ T_cam constant <1e-12 m; "
            "ext cams exactly static; wrist path len == cartesian path len."),
        "frame_offset_note": (
            f"MP4 has {n_mp4} frames for {T} h5 steps; h5_index below is the "
            "PROVISIONAL mapping h5_index==mp4_index; align_to_traj.py "
            "resolves the true constant offset by Umeyama residual search."),
        "episode": str(ep_dir),
        "camera": args.camera, "serial": serial,
        "metadata": {k: meta.get(k) for k in
                     ("uuid", "lab", "scene_id", "success", "current_task",
                      "trajectory_length", "robot_serial", "wrist_cam_serial",
                      "ext1_cam_serial", "ext2_cam_serial")},
        "filter": {**filt, "stride": args.stride,
                   "ladder": [[v if np.isfinite(v) else None for v in rung]
                              for rung in FILTER_LADDER],
                   "min_kept": MIN_KEPT},
        "ext_cams": ext_static,
        "frames": [{"name": nm, "mp4_index": int(i), "h5_index": int(i),
                    "c2w_base": c2w_all[i].tolist(),
                    "joint_position": joints[i].tolist(),
                    "gripper_position": float(grip[i])}
                   for nm, i in zip(names, keep)],
        "full": {"c2w_base": c2w_all.tolist(),
                 "joint_position": joints.tolist(),
                 "gripper_position": grip.tolist(),
                 "max_abs_joint_velocity": np.abs(jv).max(1).tolist()},
    }
    save_json(scene_dir / "gt" / "droid_traj.json", traj)
    print(f"[droid_extract] {len(names)} frames -> {frames_dir}")
    print(f"[droid_extract] trajectory -> {scene_dir / 'gt/droid_traj.json'}")


if __name__ == "__main__":
    main()
