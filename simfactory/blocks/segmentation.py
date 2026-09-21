"""Segmentation adapters over the stable backend modules."""
from __future__ import annotations

from simfactory.registry import register
from simfactory.runner import sh

@register("segmentation", "sam3",
          help="SAM3 multi-frame proposals + mesh raycast + >=2-frame merge "
               "(agents.discover.auto_segment)")
def seg_sam3(ctx, entry):
    args = ["--scene-dir", ctx.scene_dir, "--out-dir", ctx.out]
    mesh = entry.get("mesh_path") or ctx.opt("mesh_path")
    if mesh:
        args += ["--mesh-path", mesh]
    if ctx.opt("frame_stride"):
        args += ["--frame-stride", ctx.opt("frame_stride")]
    sh(ctx, "agents.discover.auto_segment", *args, env_kind="sam3",
       extra_env={"SIMANY_AUTO": "1"})
    _prepare_and_refine(ctx)


@register("segmentation", "chorus",
          help="Chorus (CVPR'26) open-vocab 3DGS encoding -> instances "
               "(models.chorus_segment)")
def seg_chorus(ctx, entry):
    sh(ctx, "models.chorus_segment", "--scene-dir", ctx.scene_dir,
       "--splat", ctx.splats_root / f"{ctx.scene}.ply",
       "--out-dir", ctx.out,
       env_kind=entry.get("env", "venv"),
       extra_env={"SIMANY_AUTO": "1", "SIMANY_INSTANCES_SRC": "chorus"})
    _prepare_and_refine(ctx, extra={"SIMANY_INSTANCES_SRC": "chorus"})


@register("segmentation", "gt",
          help="ScanNet++ GT instance annotations (benchmark mode)")
def seg_gt(ctx, entry):
    _prepare_and_refine(ctx, auto=False)


def _prepare_and_refine(ctx, auto=True, extra=None):
    env = {"SIMANY_AUTO": "1"} if auto else {}
    env.update(extra or {})
    sh(ctx, "agents.discover.factory_prepare", extra_env=env)
    sh(ctx, "agents.discover.factory_refine_masks",
       "--images-dir", ctx.scene_dir / "dslr" / "resized_undistorted_images",
       "--out-dir", ctx.out, env_kind="sam3", extra_env=env)
