#!/bin/bash
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --exclude=msp3-[0-7]
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=100G
#SBATCH --time=03:50:00
#SBATCH --job-name=simany_oracle
#SBATCH --array=0-10
# Task 14 oracle-causal benchmark: DEEP-TIER matrix (real SAM3+TRELLIS+CoACD+
# gsplat pipeline, via the existing run/run_behavior_recon.sh, unmodified).
#
# This is deliberately separate from the FAST/proxy tier
# (oracle/evaluate_reconstruction.py's DBSCAN-clustering scorer), which
# already produced real numbers for all 6 scenes x 3 levels without a GPU.
# This sbatch fills in the "real reconstruction" numbers behind the same
# JSONL schema, once each array task finishes (hours, not blocked on here).
#
# Usage: sbatch run/slurm/oracle_matrix.sbatch
# Check progress:  squeue -u $USER -n simany_oracle
#
# Each array index is (scene_id, task_id, degradation_level). `clean` means
# the scene dir written by `oracle/capture_generator.py extract` (or, for
# behavior_task0020, the original HDF5-era extraction); `severe` means the
# `<scene>_severe` dir written by `oracle/capture_generator.py degrade`
# (already generated interactively for every scene in this matrix -- see
# docs/ORACLE_PROTOCOL.md). `mild` is intentionally skipped here to bound
# GPU cost; it already has fast-tier numbers.
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
cd "$SIMANY_ROOT"

# ~/.cache is a symlink to hala's NODE-LOCAL /scratch -- this job lands on
# H200/GCP nodes where that path doesn't exist, so any HF/torch-hub-cached
# checkpoint (SAM3, TRELLIS-image-large, dinov2) FileNotFoundErrors instead
# of falling back online (HF_HUB_OFFLINE=1 is set repo-wide). Point both
# caches at the shared-storage-hosted shared copies instead.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}

# index -> "scene_id task_id level"
TRIPLES=(
  "behavior_task0011 task-0011 clean"
  "behavior_task0011 task-0011 severe"
  "behavior_task0023 task-0023 clean"
  "behavior_task0023 task-0023 severe"
  "behavior_task0027 task-0027 clean"
  "behavior_task0027 task-0027 severe"
  "behavior_task0045 task-0045 clean"
  "behavior_task0045 task-0045 severe"
  "behavior_task0002 task-0002 clean"
  "behavior_task0002 task-0002 severe"
  "behavior_task0020 task-0020 severe"
)
read -r SCENE_ID TASK_ID LEVEL <<< "${TRIPLES[$SLURM_ARRAY_TASK_ID]}"
SCENE_NAME=$SCENE_ID
[ "$LEVEL" != "clean" ] && SCENE_NAME="${SCENE_ID}_${LEVEL}"

echo "[oracle_matrix] array=$SLURM_ARRAY_TASK_ID scene_id=$SCENE_ID task_id=$TASK_ID level=$LEVEL scene_name=$SCENE_NAME"

# The degraded scene dirs already exist on shared storage (written
# interactively by oracle/capture_generator.py degrade); this is just a
# safety net if this array task runs before that -- regenerates from the
# clean extraction deterministically (seed=0, same as the interactive run).
if [ "$LEVEL" != "clean" ] && [ ! -f "data/recon_scenes/data/${SCENE_NAME}/dslr/colmap/images.txt" ]; then
  echo "[oracle_matrix] $SCENE_NAME missing -- regenerating via capture_generator degrade"
  .venv/bin/python -m oracle.capture_generator degrade \
    --scene-name "$SCENE_ID" --root data/recon_scenes --level "$LEVEL" \
    --out-scene-name "$SCENE_NAME" --seed 0 --tasks-config configs/oracle/tasks.yaml
fi

# RESUME=1 is required for run/env.sh's done_skip() to actually skip a
# stage whose output already exists (it short-circuits to false otherwise,
# regardless of file presence) -- without it, run_behavior_recon.sh would
# always retry `behavior_extract` (HDF5 -> gone, see docs/ORACLE_PROTOCOL.md
# section 0) even though this scene dir was already fully populated by
# oracle/capture_generator.py.
TASK=$TASK_ID SCENE_NAME=$SCENE_NAME EPISODES=1 STATIC_ONLY=1 RESUME=1 \
  bash run/run_behavior_recon.sh

echo "[oracle_matrix] recon done for $SCENE_NAME; merging deep-tier scores"
.venv/bin/python -m oracle.evaluate_reconstruction run-matrix \
  --tasks-config configs/oracle/tasks.yaml --root data/recon_scenes \
  --vault-root outputs/oracle_gt_vault --levels "$LEVEL" --deep \
  --out "outputs/oracle_eval/results_deep_${SCENE_ID}_${LEVEL}.jsonl"

echo "ORACLE_MATRIX_DONE $SCENE_ID $LEVEL"
