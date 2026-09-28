#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-7
#SBATCH --job-name=simany_render_v2
#SBATCH --output=outputs/fleet/render_v2_%a.log

# Re-run appearance eval with the OFFICIAL ScanNet++ dslr train/test split
# (render_metrics_v2.json; legacy linspace numbers in render_metrics.json kept
# for provenance). Covers BOTH the GT-driven (*_factory) and automatic (*_auto)
# fleets - the latter had no render metrics at all before this.
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
mapfile -t SCENES < /data/ScanNetpp/splits/nvs_sem_val.txt
STRIDE=${SLURM_ARRAY_TASK_COUNT:-8}

for ((i=SLURM_ARRAY_TASK_ID; i<${#SCENES[@]}; i+=STRIDE)); do
  SCENE=${SCENES[$i]}
  for MODE in factory auto; do
    OUT=$SIMANY_ROOT/outputs/${SCENE}_${MODE}
    [ -d "$OUT/objects" ] || { echo "SKIP $SCENE $MODE (no objects dir)"; continue; }
    [ -e "$OUT/render_metrics_v2.json" ] && { echo "SKIP $SCENE $MODE (v2 done)"; continue; }
    echo "=== [$(date +%H:%M:%S)] render_v2 $SCENE ($MODE) ==="
    SIMANY_SCENE=$SCENE SIMANY_OUT=$OUT run_gs agents.eval.factory_eval_render \
      || echo "FAIL $SCENE $MODE (continuing)"
  done
done
echo "ARRAY_TASK ${SLURM_ARRAY_TASK_ID} DONE"
