"""Inpainting adapters over the stable backend modules."""
from __future__ import annotations

from robo.simfactory.registry import register
from robo.simfactory.runner import sh


@register("inpainting", "qwen",
          help="Qwen-Image-Edit-2509 removal+fill (best on structured "
               "surfaces; sam3 env)")
def inpaint_qwen(ctx, entry):
    _inpaint(ctx, qwen="force")


@register("inpainting", "lama",
          help="LaMa fallback fill (CPU-friendly)")
def inpaint_lama(ctx, entry):
    _inpaint(ctx, qwen="0")


def _inpaint(ctx, qwen):
    env = {"SIMANY_QWEN": qwen}
    sh(ctx, "agents.edit.inpaint_prepare", extra_env=env)
    sh(ctx, "agents.edit.inpaint_masks", "--images-dir",
       ctx.scene_dir / "dslr" / "resized_undistorted_images",
       "--out-dir", ctx.out, env_kind="sam3", extra_env=env)
    if qwen == "force":
        sh(ctx, "agents.edit.inpaint_qwen", env_kind="sam3", extra_env=env)
    sh(ctx, "agents.edit.inpaint_fill", env_kind="gsplat", extra_env=env)
