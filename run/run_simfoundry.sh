#!/bin/bash
# SimFoundry-reproduction baseline: the paper-faithful reimplementation of
# the prior SimFoundry system (arXiv:2606.28276) this project began from —
# one representative frame, monocular metric depth, no GT. Reported as the
# paper's ablation row D. Run on a GPU node. (Formerly run_scene.sh.)
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OUT=$ROOT/outputs/$SIMANY_SCENE

stage "s0 select frame"
done_skip "$OUT/frame/rep_frame.json" || run agents.discover.s0_select_frame

FRAME=$($VENV -c "import json;print(json.load(open('$OUT/frame/rep_frame.json'))['frame'])")
stage "s1 SAM3 segmentation on $FRAME"
done_skip "$OUT/masks/masks.npz" || run_sam3 models.s1_segment \
  --image "$OUT/frame/$FRAME" --out-dir "$OUT/masks" \
  --prompts bottle mug "computer mouse" keyboard headphones telephone box

stage "s2 DA3 metric depth"
done_skip "$OUT/depth/depth.npz" || run models.s2_depth

stage "s3 lift objects"
done_skip "$OUT/objects/objects.json" || run agents.discover.s3_lift

stage "s4 TRELLIS image-to-3D"
run models.s4_trellis

stage "s5 pose alignment + F1 eval"
run agents.assets.s5_align

stage "s6 CoACD + physics annotation + URDF"
run agents.assets.s6_physics

stage "s7 PyBullet settle + dynamics"
run robo.sim.s7_sim

stage "s8 gsplat renders"
run_gs agents.render.s8_render

stage "DONE -> $OUT"
