"""Emit an emulated ScanNet++-style scene dir from a metric reconstruction.

Writes exactly the three inputs agents/core/common.py reads (verified by the
--selftest round-trip through common.load_colmap_w2c / load_intrinsics):
    <root>/data/<scene>/dslr/resized_undistorted_images/*.jpg
    <root>/data/<scene>/dslr/nerfstudio/transforms_undistorted.json
    <root>/data/<scene>/dslr/colmap/images.txt   (+ cameras.txt)
plus <root>/data/<scene>/init_points.ply (xyz+rgb) for gsplat_train.

The data/ level exists because common.py composes
SCENE_DIR = $SIMANY_SCANNETPP_ROOT/data/$SIMANY_SCENE, so pointing the
pipeline at a reconstructed scene is just:
    SIMANY_SCANNETPP_ROOT=<root> SIMANY_SCENE=<scene>
    SIMANY_SPLATS_ROOT=<root>/splats

CPU, any env. Usage:
    python -m agents.recon.make_scene_dir --recon recon_metric.npz \
        --frames-dir D --scene <name> \
        --root /group/worldcept/PhiRIE/code/SimAny/data/recon_scenes
    python -m agents.recon.make_scene_dir --selftest
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from agents.core import common as C

DEFAULT_ROOT = Path("/group/worldcept/PhiRIE/code/SimAny/data/recon_scenes")
# common.load_colmap_w2c drops blank lines and then reads every OTHER line,
# so the POINTS2D line after each image line MUST be non-empty - a single
# dummy observation keeps the alternation intact.
DUMMY_POINTS_LINE = "0.0 0.0 -1"


def rot_to_quat_wxyz_np(R):
    """Pure-numpy rotation -> wxyz quaternion (Shepperd); common's version
    pulls in the utils3d shim, which the selftest must not depend on."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(R[i, i] - R[j, j] - R[k, k] + 1.0, 1e-12)) * 2
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def write_scene_dir(rec, frames_dir, scene, root):
    scene_dir = Path(root) / "data" / scene
    img_dir = scene_dir / "dslr" / "resized_undistorted_images"
    ns_dir = scene_dir / "dslr" / "nerfstudio"
    cm_dir = scene_dir / "dslr" / "colmap"
    for d in (img_dir, ns_dir, cm_dir):
        d.mkdir(parents=True, exist_ok=True)

    names = [str(n) for n in rec["names"]]
    w2c = np.asarray(rec["w2c"], dtype=np.float64)
    K = np.asarray(rec["K"], dtype=np.float64)

    for nm in names:
        src = Path(frames_dir) / nm
        if not src.exists():
            raise SystemExit(f"[scene_dir] frame missing: {src}")
        shutil.copyfile(src, img_dir / nm)

    from PIL import Image
    w, h = Image.open(img_dir / names[0]).size
    if "frame_wh" in rec:
        assert (int(rec["frame_wh"][0]), int(rec["frame_wh"][1])) == (w, h), \
            "frames on disk do not match the resolution K was computed for"

    # nerfstudio json: common reads only the scalars; the frames list (c2w in
    # nerfstudio's OpenGL convention) is included so standard viewers work
    gl_flip = np.diag([1.0, -1.0, -1.0, 1.0])
    frames = [{"file_path": nm,
               "transform_matrix": (np.linalg.inv(m) @ gl_flip).tolist()}
              for nm, m in zip(names, w2c)]
    meta = {"camera_model": "PINHOLE",
            "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "w": int(w), "h": int(h), "has_mask": False,
            "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
            "frames": frames}
    (ns_dir / "transforms_undistorted.json").write_text(
        json.dumps(meta, indent=1))

    g = lambda v: f"{v:.17g}"  # full float64 precision through the text file
    lines = ["# COLMAP-style image list, written by agents.recon."
             "make_scene_dir",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
             "#   POINTS2D[] as (X, Y, POINT3D_ID)  (dummy, poses only)"]
    for i, (nm, m) in enumerate(zip(names, w2c), start=1):
        q = rot_to_quat_wxyz_np(m[:3, :3])
        t = m[:3, 3]
        lines.append(" ".join([str(i), *map(g, q), *map(g, t), "1", nm]))
        lines.append(DUMMY_POINTS_LINE)
    (cm_dir / "images.txt").write_text("\n".join(lines) + "\n")
    (cm_dir / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {w} {h} {g(K[0, 0])} {g(K[1, 1])} "
        f"{g(K[0, 2])} {g(K[1, 2])}\n")

    from plyfile import PlyData, PlyElement
    pts = np.asarray(rec["points"], dtype=np.float32)
    rgb = np.asarray(rec["points_rgb"], dtype=np.uint8)
    arr = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                    ("red", "u1"), ("green", "u1"),
                                    ("blue", "u1")])
    arr["x"], arr["y"], arr["z"] = pts.T
    arr["red"], arr["green"], arr["blue"] = rgb.T
    PlyData([PlyElement.describe(arr, "vertex")]).write(
        str(scene_dir / "init_points.ply"))

    print(f"[scene_dir] {scene}: {len(names)} frames, {len(pts)} init "
          f"points -> {scene_dir}")
    return scene_dir


def selftest():
    """Pure-numpy round-trip: fake 3-frame recon -> scene dir -> common
    parsers -> exact pose/K recovery. CPU, no models."""
    import tempfile
    from PIL import Image

    tmp = Path(tempfile.mkdtemp(prefix="recon_selftest_"))
    frames_dir = tmp / "frames"
    frames_dir.mkdir()
    rng = np.random.RandomState(7)
    names, w2c = [], []
    for i in range(3):
        nm = f"frame_{i:06d}.jpg"
        Image.fromarray(rng.randint(0, 255, (6, 8, 3), np.uint8)).save(
            frames_dir / nm)
        q = rng.randn(4)
        M = np.eye(4)
        M[:3, :3] = C.quat_to_rot_wxyz(q)
        M[:3, 3] = rng.randn(3) * 2.0
        names.append(nm)
        w2c.append(M)
    w2c = np.stack(w2c)
    K = np.array([[431.25, 0, 4.0], [0, 430.75, 3.0], [0, 0, 1.0]])
    rec = {"names": np.array(names), "w2c": w2c, "K": K,
           "points": rng.randn(10, 3).astype(np.float32),
           "points_rgb": rng.randint(0, 255, (10, 3), np.uint8),
           "frame_wh": np.array([8, 6])}
    np.savez(tmp / "recon_metric.npz", **rec)

    scene_dir = write_scene_dir(dict(np.load(tmp / "recon_metric.npz")),
                                frames_dir, "selftest", tmp / "root")

    got = C.load_colmap_w2c(scene_dir / "dslr" / "colmap" / "images.txt")
    assert sorted(got) == sorted(names), (sorted(got), names)
    for nm, M in zip(names, w2c):
        assert np.allclose(got[nm], M, atol=1e-9), (nm, got[nm] - M)
    K2, w, h, _ = C.load_intrinsics(
        scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json")
    assert np.allclose(K2, K, atol=1e-12) and (w, h) == (8, 6)
    from plyfile import PlyData
    v = PlyData.read(str(scene_dir / "init_points.ply"))["vertex"]
    assert len(v) == 10 and {"x", "red"} <= {p.name for p in v.properties}
    print(f"[scene_dir] selftest OK (tmp={tmp})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recon", type=Path)
    ap.add_argument("--frames-dir", type=Path)
    ap.add_argument("--scene", type=str)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.recon and args.frames_dir and args.scene):
        ap.error("--recon, --frames-dir and --scene are required "
                 "(or use --selftest)")
    rec = dict(np.load(args.recon, allow_pickle=False))
    write_scene_dir(rec, args.frames_dir, args.scene, args.root)


if __name__ == "__main__":
    main()
