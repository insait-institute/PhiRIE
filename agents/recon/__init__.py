"""Scene reconstruction from casual video / posed frames.

Turns an unposed phone video into an emulated ScanNet++-style scene dir
(images + PINHOLE intrinsics + COLMAP w2c poses) plus a trained 3DGS splat,
so the whole GT-free SimAny pipeline can run on scenes we captured ourselves:

    frames.py       video -> uniformly subsampled jpg frames        (any env)
    vggt_scene.py   frames -> VGGT poses/depth/point cloud (npz)    (run)
    metricize.py    npz -> metric scale + z-up + floor at z=0       (run)
    make_scene_dir.py  npz -> emulated scene dir + init_points.ply  (any env)
    gsplat_train.py    scene dir -> Inria 3DGS ply + train report   (run_gs)

Emulated scenes live under data/recon_scenes/data/<scene>/ with splats at
data/recon_scenes/splats/<scene>.ply; point the pipeline at them via
SIMANY_SCANNETPP_ROOT=data/recon_scenes (common.py composes
$SCANNETPP_ROOT/data/$SCENE) and SIMANY_SPLATS_ROOT=data/recon_scenes/splats.
"""
