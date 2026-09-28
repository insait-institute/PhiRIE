#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=simany_verify2
#SBATCH --output=outputs/fleet/verify_rerun.log

# Re-run the pilot-scene removal-verification battery after the held-view
# region fix (verify_removal_check.py whole-frame bug) + FlashSplat with the
# corrected reference set (fill_gaussians.npz removal_idx incl. mask vote).
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
export SIMANY_SCENE=c50d2d1d42
export SIMANY_OUT=$SIMANY_ROOT/outputs/c50d2d1d42_factory
echo "=== verify_removal_render (held regions + renders) ==="
run_gs agents.eval.verify_removal_render

echo "=== verify_removal_check (SAM3 residual re-detection, localized) ==="
run_sam3 agents.eval.verify_removal_check --out-dir "$SIMANY_OUT"

echo "=== baseline_flashsplat (corrected reference set) ==="
run_gs agents.baselines.flashsplat

echo "DONE"
