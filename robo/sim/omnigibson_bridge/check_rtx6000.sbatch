#!/bin/bash
#SBATCH --job-name=simany_rtx6000_check
#SBATCH --partition=non-reserved
#SBATCH --qos=normal
#SBATCH --gpus=rtx6000:1
#SBATCH --time=02:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=outputs/logs/og_rtx6000_check_%j.log

unset PYTHONPATH PYTHONHOME
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin

set -x
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
which nvcc
nvcc -V 2>&1 || echo "no nvcc"

CONDA_BASE=${CONDA_ROOT:-$HOME/miniconda3}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior1k

cd ${SIMANY_ROOT:-$PWD}/robo/sim/omnigibson_bridge
OMNIGIBSON_HEADLESS=1 python smoke_test.py
echo "=== SMOKE EXIT: $? ==="
