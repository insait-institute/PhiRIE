#!/bin/bash
#SBATCH --job-name=studio
#SBATCH --output=/group/worldcept/code/PhysicalView/outputs/logs/studio_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:50:00
# GPU type is chosen at submit time (see below); defaults to hala's A6000 on the debug partition:
#   sbatch run/studio.sbatch                                   # a6000 / debug
#   sbatch --partition=batch --gres=gpu:h200:1 --exclude=msp3-[0-7] run/studio.sbatch
#   sbatch --partition=batch --gres=gpu:a100-80g:1 run/studio.sbatch
#   sbatch --partition=batch --gres=gpu:rtx6000:1 -w gcp-eu1-rtx6000-vz3w run/studio.sbatch
#   PORT=8081 sbatch ... run/studio.sbatch                     # custom port
# Then:  tail -f outputs/logs/studio_<jobid>.log   (prints the ssh tunnel line)
#SBATCH --partition=debug
#SBATCH --gres=gpu:a6000:1
set -euo pipefail
ROOT=$(cd "$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*Command=\([^ ]*\).*/\1/p' | head -1)")/.." 2>/dev/null && pwd || echo /group/worldcept/code/PhysicalView)
cd "$ROOT"
mkdir -p outputs/logs
ls /data/ScanNetpp >/dev/null 2>&1 || true            # trigger the autofs mount
source run/env.sh
PORT=${PORT:-8080}
NODE=$(hostname)
echo "=================================================================="
echo " PhysicalView  job=$SLURM_JOB_ID  node=$NODE  port=$PORT"
echo " GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
echo " tunnel from your laptop:   ssh -L ${PORT}:${NODE}:${PORT} <login-host>"
echo " then open:                 http://localhost:${PORT}"
echo " stop:                      scancel $SLURM_JOB_ID"
echo "=================================================================="
exec "$STUDIO_PY" -m physicalview.app --port "$PORT" --host 0.0.0.0 ${STUDIO_ARGS:-}
