#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=03:50:00
#SBATCH --job-name=simany_behavior
# Usage: sbatch --export=ALL,TASK=task-0000 --output=... behavior_recon.sbatch
# Single job, single node, start to finish - same reasoning as
# ablation_rowC_scene.sbatch (derived_mesh.ply across separate jobs once hit
# stale cross-node NFS reads).
set -e
SIMANY_ROOT=/group/worldcept/PhiRIE/code/SimAny
export TASK=${TASK:-task-0000}
export EPISODES=${EPISODES:-3}
export STATIC_ONLY=${STATIC_ONLY:-1}

bash "$SIMANY_ROOT/run/run_behavior_recon.sh"

echo "BEHAVIOR_RECON_DONE $TASK"
