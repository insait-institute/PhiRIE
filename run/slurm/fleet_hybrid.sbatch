#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --job-name=simany_hybrid
#SBATCH --output=outputs/fleet_hybrid_%a.log
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
mapfile -t SCENES < /data/ScanNetpp/splits/nvs_sem_val.txt
STRIDE=4
for ((i=SLURM_ARRAY_TASK_ID; i<${#SCENES[@]}; i+=STRIDE)); do
  s=${SCENES[$i]}
  out=$SIMANY_ROOT/outputs/${s}_factory
  [ -f "$out/objects/objects.json" ] || { echo "[skip] $s no objects"; continue; }
  [ -f "$out/objects/hybrid_all.json" ] && { echo "[skip] $s done"; continue; }
  echo "=== [$(date +%H:%M:%S)] hybrid $s ==="
  SIMANY_SCENE=$s SIMANY_OUT=$out timeout 40m \
    run agents.assets.factory_hybrid || echo "[FAIL] $s ($?)"
done
echo "SHARD_DONE"
