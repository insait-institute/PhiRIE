#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=64G
#SBATCH --time=03:50:00
#SBATCH --array=0-7
#SBATCH --job-name=simany_autofleet
#SBATCH --output=outputs/fleet/auto_slurm_%a.log

mkdir -p outputs/fleet
mapfile -t SCENES < /data/ScanNetpp/splits/nvs_sem_val.txt
STRIDE=${SLURM_ARRAY_TASK_COUNT:-8}

for ((i=SLURM_ARRAY_TASK_ID; i<${#SCENES[@]}; i+=STRIDE)); do
  s=${SCENES[$i]}
  out=outputs/${s}_auto
  if [ -f "$out/report.json" ] && [ -f "$out/sim_export/isaac_manifest.json" ] \
     && [ -f "$out/sim_export/mujoco_settle.json" ]; then
    echo "SKIP $s (done)"
    continue
  fi
  echo "=== SCENE $s ($(date +%H:%M:%S)) ==="
  RESUME=1 SIMANY_SCENE=$s timeout 3600 bash run/run_auto.sh \
    > outputs/fleet/${s}_auto.log 2>&1 && echo "OK $s" || echo "FAILED $s ($?)"
done
echo "SHARD $SLURM_ARRAY_TASK_ID DONE"
