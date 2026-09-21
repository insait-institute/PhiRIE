#!/bin/bash
#SBATCH --job-name=simany_render_smoke
#SBATCH --partition=non-reserved
#SBATCH --qos=normal
#SBATCH --gpus=rtx6000:1
#SBATCH --time=02:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=/group/worldcept/PhiRIE/code/SimAny/outputs/logs/og_render_smoke_%j.log

unset PYTHONPATH PYTHONHOME
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin
export HOME=/home/runyi_yang

set -x
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

CONDA_BASE=/group/streetsplat/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior1k

cd /group/worldcept/PhiRIE/code/SimAny/robo/sim/omnigibson_bridge
OMNIGIBSON_HEADLESS=1 python render_smoke_test.py --out /group/worldcept/PhiRIE/code/SimAny/outputs/omnigibson_export/render_smoke_frame.png
echo "=== RENDER SMOKE EXIT CODE: $? ==="
