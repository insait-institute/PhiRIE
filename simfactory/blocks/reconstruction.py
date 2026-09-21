"""Reconstruction adapters over the stable backend modules."""
from __future__ import annotations

from simfactory.registry import register
from simfactory.runner import sh

# video/frames -> posed metric scene dir + trained splat. For scannetpp
# sources the splat already exists (SceneSplat), so 'scenesplat' is a no-op
# passthrough; known-pose sources extract from their native releases.


@register("reconstruction", "colmap",
          help="pycolmap SfM poses (best splat PSNR: 25.52 dB pilot) + "
               "DA3 metric scale + gsplat training")
def recon_colmap(ctx, entry):
    _recon_from_frames(ctx, entry, backend="colmap")


@register("reconstruction", "vggt_omega",
          help="VGGT-Omega feed-forward (5.0cm med pose err, all frames; "
               "22.52 dB pilot splat)")
def recon_omega(ctx, entry):
    _recon_from_frames(ctx, entry, backend="omega")


@register("reconstruction", "vggt",
          help="original VGGT feed-forward (12.3cm med; 20.98 dB pilot)")
def recon_vggt(ctx, entry):
    _recon_from_frames(ctx, entry, backend="vggt")


def _recon_from_frames(ctx, entry, backend):
    """Mirrors run/run_video2sim.sh stage-for-stage."""
    video = ctx.config["scene"].get("video")
    if not video:
        raise SystemExit("[simfactory] reconstruction from video needs "
                         "scene.video in the config")
    recon = ctx.out / "recon"
    frames = recon / "frames"
    sh(ctx, "agents.recon.frames", "--video", video, "--out-dir", frames,
       "--max-frames", ctx.opt("max_frames", 240))
    if backend == "colmap":
        sh(ctx, "agents.recon.colmap_poses", "--images-dir", frames,
           "--out", recon / "recon.npz", "--workdir", recon / "colmap")
    else:
        sh(ctx, "models.vggt_scene", "--images-dir", frames,
           "--out", recon / "recon.npz", "--backend", backend)
    sh(ctx, "agents.recon.metricize", "--recon", recon / "recon.npz",
       "--images-dir", frames, "--out", recon / "recon_metric.npz")
    sh(ctx, "agents.recon.make_scene_dir", "--recon",
       recon / "recon_metric.npz", "--frames-dir", frames,
       "--scene", ctx.scene, "--root", ctx.scannetpp_root.parent
       if ctx.scannetpp_root.name == "data" else ctx.scannetpp_root)
    sh(ctx, "agents.recon.gsplat_train", "--scene-dir", ctx.scene_dir,
       "--init-ply", ctx.scene_dir / "init_points.ply",
       "--out", ctx.splats_root / f"{ctx.scene}.ply",
       "--iters", ctx.opt("gs_iters", 30000), env_kind="gsplat")


@register("reconstruction", "scenesplat",
          help="prebuilt SceneSplat 3DGS (ScanNet++ scenes) - no work")
def recon_scenesplat(ctx, entry):
    ply = ctx.splats_root / f"{ctx.scene}.ply"
    print(f"[simfactory] using prebuilt splat {ply} "
          f"({'exists' if ply.exists() else 'MISSING'})")


@register("reconstruction", "known_pose_behavior",
          help="PointWorld-BEHAVIOR posed RGB-D extraction + gsplat "
               "(GT sim poses; original GT-depth mesh)")
def recon_behavior(ctx, entry):
    sh(ctx, "agents.recon.behavior_extract",
       "--task", ctx.config["scene"].get("task", ctx.scene),
       "--episodes", ctx.opt("episodes", 4), "--static-only",
       "--scene-name", ctx.scene, "--root", ctx.scannetpp_root,
       env_kind="h5")
    sh(ctx, "agents.recon.behavior_extract", "--fuse-only",
       "--task", ctx.config["scene"].get("task", ctx.scene),
       "--scene-name", ctx.scene, "--root", ctx.scannetpp_root)
    sh(ctx, "agents.recon.gsplat_train", "--scene-dir", ctx.scene_dir,
       "--init-ply", ctx.scene_dir / "init_points.ply",
       "--out", ctx.splats_root / f"{ctx.scene}.ply", env_kind="gsplat")


@register("reconstruction", "known_pose_droid",
          help="raw DROID episode: COLMAP wrist poses Umeyama-aligned to "
               "FK camera trajectory (metric, robot base frame)")
def recon_droid(ctx, entry):
    episode = ctx.config["scene"].get("episode")
    if not episode:
        raise SystemExit("[simfactory] droid recon needs scene.episode")
    sh(ctx, "agents.recon.droid_extract", "--episode", episode,
       "--scene-name", ctx.scene, "--root", ctx.scannetpp_root,
       env_kind="h5")
