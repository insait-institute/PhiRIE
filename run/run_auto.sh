#!/bin/bash
# Fully automatic mode ("Automatic (no GT)"):
# posed images + mesh + gaussians -> sim-ready scene. NO semantic annotations.
# Outputs MuJoCo MJCF + Isaac URDF manifest.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export SIMANY_OUT=${SIMANY_OUT:-$ROOT/outputs/${SIMANY_SCENE}_auto}
export SIMANY_AUTO=1

stage_timed "auto_segment (SAM3 multi-frame discovery, no GT)"
done_skip "$SIMANY_OUT/auto_instances.npz" || \
  run_sam3 agents.discover.auto_segment --scene-dir "$SD" --out-dir "$SIMANY_OUT"

stage_timed "factory_prepare (auto instances + best-view crops)"
done_skip "$SIMANY_OUT/objects/objects.json" || run agents.discover.factory_prepare

stage_timed "factory_refine_masks (SAM3 evidence)"
done_skip "$SIMANY_OUT/objects/.masks_refined" || {
  run_sam3 agents.discover.factory_refine_masks --images-dir "$IMG" --out-dir "$SIMANY_OUT" &&
  touch "$SIMANY_OUT/objects/.masks_refined"; }

stage_timed "s4 TRELLIS image-to-3D"
run agents.models.s4_trellis

stage_timed "factory_align (register to own extraction)"
run agents.assets.factory_align

stage_timed "s6 CoACD + physics annotation + URDF"
run agents.assets.s6_physics

stage_timed "factory_report (drop test + yield)"
run agents.eval.factory_report

stage_timed "export MJCF + Isaac manifest + MuJoCo settle test"
run robo.sim.export_mjcf --test

stage_timed "DONE -> $SIMANY_OUT"
