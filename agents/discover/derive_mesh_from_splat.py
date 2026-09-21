"""Ablation: derive a scan-mesh substitute from the TRAINED SPLAT ONLY
(no ScanNet++ dataset mesh) via rendered-depth TSDF fusion.

Validated on c50d2d1d42: splat-rendered depth (gsplat RGB+ED) agrees with
the dataset mesh's raycast depth to within ~2mm median, ratio 1.0000 (no
scale bug) - see outputs/depth_check2.log.

Two subcommands because gsplat and open3d don't reliably coexist in one
env on this cluster (gsplat: mini-viewer env; open3d/TSDF: .venv):
  render  (mini-viewer env, GPU) -> per-view color/depth/alpha/pose npz
  fuse    (.venv env, GPU for splat load but TSDF itself is CPU) -> mesh.ply

Usage:
  SIMANY_SCENE=x SIMANY_OUT=... mini-viewer/python derive_mesh_from_splat.py render
  SIMANY_SCENE=x SIMANY_OUT=... .venv/python      derive_mesh_from_splat.py fuse
"""
import argparse

import numpy as np

from agents.core import common as C

STRIDE = 3          # every Nth training DSLR frame (denser: finer voxels
                    # need more multi-view agreement to stay solid, not holey)
SCALE = 0.5         # render resolution factor (876x584 at ScanNet++ DSLR res)
ALPHA_MIN = 0.6      # exclude low-confidence/unreconstructed pixels
# 2cm/6cm (first attempt) smoothed small manipulable objects (mice ~2-5cm
# tall, headphones ~8cm) into the support surface they sit on - the
# truncation band was taller than the object. 2DGS's own TSDF-extraction
# convention (voxel 4mm / trunc 2cm) is the literature-standard finer
# setting for exactly this reason; we use 5mm voxel / 2cm trunc (trunc
# matches that convention, voxel slightly coarser).
VOXEL = 0.005
SDF_TRUNC = 0.02


def reconstruction_cameras(args):
    """Optional explicit producer scene; public callers must provide this path."""
    directory = getattr(args, "scene_dir", None)
    if directory is None:
        K, W, H, meta = C.load_intrinsics()
        poses = C.load_colmap_w2c()
    else:
        from pathlib import Path
        directory = Path(directory)
        K, W, H, meta = C.load_intrinsics(directory / "dslr/nerfstudio/transforms_undistorted.json")
        poses = C.load_colmap_w2c(directory / "dslr/colmap/images.txt")
    return K, W, H, meta, sorted(poses.items())


def seg_render(args):
    stride = getattr(args, "frame_stride", STRIDE)
    if stride < 1: raise ValueError("render frame stride must be positive")
    views_dir = C.OUT / "mesh_derive"
    strict_split = getattr(args, "train_split", None)
    if getattr(args, "max_train_frames", None) is not None and not strict_split:
        raise ValueError("training frame limit requires an official split")
    if strict_split and views_dir.exists():
        raise FileExistsError("training-only render refuses an existing depth directory")
    views_dir.mkdir(parents=True, exist_ok=True)
    # clear stale views: a prior denser-STRIDE run leaves higher-index files
    # that fuse's glob would silently integrate at incompatible settings
    for f in views_dir.glob("view_*.npz"):
        f.unlink()
    K, W, H, _, w2c_all = reconstruction_cameras(args)
    if strict_split:
        from agents.discover.training_views import select_training_views, write_training_manifest
        w2c_all, manifest = select_training_views(w2c_all, strict_split, stride, getattr(args, "max_train_frames", None))
        write_training_manifest(views_dir / "training_views.json", manifest)
    else:
        w2c_all = w2c_all[::stride]
    splat = getattr(args, "splat_ply", None)
    gs = C.load_gaussians(splat) if splat else C.load_gaussians()

    for i, (fname, w2c) in enumerate(w2c_all):
        rgb, depth, alpha = C.render_view(gs, w2c, K, W, H,
                                          render_mode="RGB+ED", scale=SCALE)
        np.savez(views_dir / f"view_{i:04d}.npz",
                 rgb=(np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                 depth=depth.astype(np.float32),
                 alpha=alpha.astype(np.float32),
                 w2c=w2c, fname=fname)
        if (i + 1) % 20 == 0:
            print(f"[dm] rendered {i + 1}/{len(w2c_all)}", flush=True)
    print(f"[dm] {len(w2c_all)} views -> {views_dir}")


def new_tsdf_volume(voxel_length, sdf_trunc):
    """Shared CPU RGB-D fusion primitive, also used by native observed control."""
    import open3d as o3d
    if not 0 < voxel_length < sdf_trunc:
        raise ValueError('TSDF requires positive voxel length below truncation')
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


def integrate_rgbd(volume, rgb, depth, intrinsic, world_to_camera):
    import open3d as o3d
    rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,depth_trunc=8.0,convert_rgb_to_intensity=False)
    volume.integrate(rgbd,intrinsic,np.asarray(world_to_camera,dtype=np.float64))


def seg_fuse(args):
    import open3d as o3d

    views_dir = C.OUT / "mesh_derive"
    files = sorted(views_dir.glob("view_*.npz"))
    if not files:
        raise SystemExit(f"[dm] no rendered views in {views_dir} - run "
                         "'render' first (mini-viewer env)")
    manifest_path = views_dir / "training_views.json"
    strict_split = getattr(args, "train_split", None)
    if strict_split and not manifest_path.is_file():
        raise ValueError("training-only fusion requires a render training-view manifest")
    if manifest_path.exists():
        import json
        from agents.discover.training_views import validate_fusion_frames
        if (C.OUT / "derived_mesh.ply").exists():
            raise FileExistsError("training-only fusion refuses an existing derived mesh")
        def frame_name(path):
            with np.load(path, allow_pickle=False) as data:
                return str(data["fname"].item())
        manifest = json.loads(manifest_path.read_text())
        if strict_split:
            from pathlib import Path
            if str(Path(strict_split).resolve()) != manifest["split"]["path"]:
                raise ValueError("fusion and rendering declare different official splits")
        validate_fusion_frames(files, manifest, frame_name)

    K, W, H, _, _ = reconstruction_cameras(args)
    Ks = np.asarray(K, dtype=np.float64).copy()
    Ks[:2] *= SCALE
    Ws, Hs = int(round(W * SCALE)), int(round(H * SCALE))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        Ws, Hs, Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2])

    volume = new_tsdf_volume(VOXEL, SDF_TRUNC)

    for i, f in enumerate(files):
        d = np.load(f)
        rgb, depth, alpha = d["rgb"], d["depth"], d["alpha"]
        depth = depth.copy()
        depth[alpha < ALPHA_MIN] = 0.0  # invalid -> excluded by Open3D
        depth[depth < 0.05] = 0.0
        integrate_rgbd(volume, rgb, depth, intrinsic, d["w2c"])
        if (i + 1) % 20 == 0:
            print(f"[dm] fused {i + 1}/{len(files)}", flush=True)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    n_tri = len(mesh.triangles)
    # aggressive decimation at fine (5mm) voxels would smooth away the same
    # small-object bumps the fine voxel size exists to preserve - cap much
    # higher than the coarse-voxel setting used, only to bound raycast cost
    CAP = 3_000_000
    if n_tri > CAP:
        mesh = mesh.simplify_quadric_decimation(CAP)
    out_path = C.OUT / "derived_mesh.ply"
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    print(f"[dm] {len(files)} views fused -> {out_path} "
          f"({len(mesh.vertices)} verts, {len(mesh.triangles)} tri, "
          f"was {n_tri} before simplify)")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    render_parser = sp.add_parser("render")
    render_parser.add_argument("--frame-stride", type=int, default=STRIDE)
    render_parser.add_argument("--scene-dir", type=str)
    render_parser.add_argument("--splat-ply", type=str)
    render_parser.add_argument("--max-train-frames", type=int)
    render_parser.add_argument("--train-split", help="official train_test_lists.json; filter before stride")
    fuse_parser = sp.add_parser("fuse")
    fuse_parser.add_argument("--scene-dir", type=str)
    fuse_parser.add_argument("--train-split", help="require the matching training-only render manifest")
    args = ap.parse_args()
    {"render": seg_render, "fuse": seg_fuse}[args.cmd](args)


if __name__ == "__main__":
    main()
