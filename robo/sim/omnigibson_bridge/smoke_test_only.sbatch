#!/bin/bash
#SBATCH --job-name=simany_og_smoke
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodelist=hala
#SBATCH --gpus=a6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=outputs/logs/og_smoke_%j.log

unset PYTHONPATH PYTHONHOME
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin

set -x
CONDA_BASE=${CONDA_ROOT:-$HOME/miniconda3}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior1k
nvidia-smi --query-gpu=name,driver_version --format=csv

set +e
OMNIGIBSON_HEADLESS=1 python ${SIMANY_ROOT:-$PWD}/robo/sim/omnigibson_bridge/smoke_test.py
SMOKE_RC=$?
set -e
echo "=== SMOKE TEST EXIT CODE: $SMOKE_RC ==="
exit "$SMOKE_RC"
