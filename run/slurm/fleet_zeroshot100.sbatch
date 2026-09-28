#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=64G
#SBATCH --time=03:50:00
#SBATCH --array=0-7
#SBATCH --job-name=simany_zs100
#SBATCH --output=outputs/fleet/zs100_slurm_%a.log
# Zero-shot 100-scene fleet: fully automatic mode over the manipulation-rich
# scene set picked by agents/eval/select_manip_scenes.py (disjoint from the
# 50-scene benchmark split). ~17 min/scene on one A6000; RESUME=1 makes
# re-submission cheap. debug QoS runs at most 4 array shards concurrently.
cd ${SIMANY_ROOT:-$PWD}
mkdir -p outputs/fleet
mapfile -t SCENES < data/zeroshot_100_scenes.txt
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
    > outputs/fleet/${s}_zs100.log 2>&1 && echo "OK $s" || echo "FAILED $s ($?)"
done
echo "SHARD $SLURM_ARRAY_TASK_ID DONE"
