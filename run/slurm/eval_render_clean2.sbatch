#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --job-name=simany_evalrender
#SBATCH --output=/group/worldcept/PhiRIE/code/SimAny/outputs/eval_render_clean2_%j.log
# Refresh render_metrics_v2 for the two scenes whose clean_background.ply
# postdates their last eval run (adds clean_bg / ours_twin_clean variants).
set -e
SIMANY_ROOT=/group/worldcept/PhiRIE/code/SimAny
source "$SIMANY_ROOT/run/env.sh"
for S in 7b6477cb95 825d228aec; do
  export SIMANY_SCENE=$S SIMANY_OUT=$SIMANY_ROOT/outputs/${S}_factory
  echo "=== eval_render $S ==="
  run_gs agents.eval.factory_eval_render
done
echo "EVAL_RENDER_CLEAN2_DONE"
