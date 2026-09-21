#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=simany_mc_s12
#SBATCH --output=/group/worldcept/PhiRIE/code/SimAny/outputs/fleet/mc_s12.log

# Matched-frame-density MaskClustering baseline (stride 12 = same 28 frames
# auto_segment sees), fixing the ~6x density confound in the Table III run.
set -e
SIMANY_ROOT=/group/worldcept/PhiRIE/code/SimAny
source "$SIMANY_ROOT/run/env.sh"
MCPY=$SIMANY_ROOT/.venv-mc/bin/python

echo "=== [1/3] SAM3 masks (28 frames) ==="
run_sam3 agents.baselines.mc_masks --seq c50d2d1d42_s12

echo "=== [2/3] MaskClustering ==="
# MaskClustering resolves its configs relative to its own tree
(cd $SIMANY_ROOT/third_party/MaskClustering &&
 $MCPY main.py --config scannetpp_dslr --seq_name_list c50d2d1d42_s12)

echo "=== [3/3] convert + eval ==="
$MCPY -m agents.baselines.maskclustering convert \
  --seq c50d2d1d42_s12 --out-dir $SIMANY_ROOT/outputs/c50d2d1d42_mc_s12
SIMANY_SCENE=c50d2d1d42 run agents.eval.eval_instances \
  --npz $SIMANY_ROOT/outputs/c50d2d1d42_mc_s12/auto_instances.npz
echo "DONE"
