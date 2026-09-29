#!/bin/bash
# Gaussian-native object removal + background completion for a factory scene.
# Produces clean_background.ply (drop-in Inria ply). Run on a GPU machine.
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export SIMANY_OUT=${SIMANY_OUT:-$ROOT/outputs/${SIMANY_SCENE}_factory}

# LaMa fallback weights (prefetched; GPU machines may lack egress)
BL=$(ls ~/.cache/torch/hub/checkpoints/*lama*.pt 2>/dev/null | head -1)
[ -n "$BL" ] && export LAMA_MODEL="$BL"

stage "inpaint_prepare (removal sets + support planes + view lists)"
run agents.edit.inpaint_prepare

stage "inpaint_masks (SAM3 union masks)"
run_sam3 agents.edit.inpaint_masks --images-dir "$IMG" --out-dir "$SIMANY_OUT"

stage "inpaint_qwen (erase objects in views)"
# QWEN_PY defaults to the main venv, whose torch 2.4.1 cannot run
# Qwen-Image-Edit (needs >=2.5). Set QWEN_PY=$SAM3PY on a node with the VRAM.
run_qwen agents.edit.inpaint_qwen

stage "inpaint_fill (carve + plane fill + gsplat refine)"
run_gs agents.edit.inpaint_fill

stage "DONE -> $SIMANY_OUT/inpaint"
