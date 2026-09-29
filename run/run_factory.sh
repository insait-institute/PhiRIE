#!/bin/bash
# GT-driven asset factory (benchmark mode): one scene, all whitelist objects.
# This is the "GT-driven" reference mode.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export SIMANY_OUT=${SIMANY_OUT:-$ROOT/outputs/${SIMANY_SCENE}_factory}

stage_timed "factory_prepare (GT enumerate + best-view crops)"
done_skip "$SIMANY_OUT/objects/objects.json" || run agents.discover.factory_prepare

stage_timed "factory_refine_masks (SAM3 evidence for transparent objects)"
# marker guard: rewriting rgba.png would needlessly invalidate the s4 cache
done_skip "$SIMANY_OUT/objects/.masks_refined" || {
  run_sam3 agents.discover.factory_refine_masks --images-dir "$IMG" --out-dir "$SIMANY_OUT" &&
  touch "$SIMANY_OUT/objects/.masks_refined"; }

stage_timed "s4 TRELLIS image-to-3D"
run agents.models.s4_trellis

stage_timed "factory_align (register to GT submesh + tiers)"
run agents.assets.factory_align

stage_timed "s6 CoACD + physics annotation + URDF"
run agents.assets.s6_physics

stage_timed "factory_report (drop test + yield)"
run agents.eval.factory_report

stage_timed "factory_eval_render (PSNR/SSIM/LPIPS)"
run_gs agents.eval.factory_eval_render

stage_timed "DONE -> $SIMANY_OUT"
