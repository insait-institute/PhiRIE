#!/bin/bash
#SBATCH --job-name=simany_og_convert
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodelist=hala
#SBATCH --gpus=a6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=/group/worldcept/PhiRIE/code/SimAny/outputs/logs/og_convert_%j.log

unset PYTHONPATH PYTHONHOME
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin
export HOME=/home/runyi_yang

set -x
CONDA_BASE=/group/streetsplat/miniconda3
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior1k

cd /group/worldcept/PhiRIE/code/SimAny/robo/sim/omnigibson_bridge
OMNIGIBSON_HEADLESS=1 python convert_assets.py
echo "=== CONVERT EXIT CODE: $? ==="
