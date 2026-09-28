#!/bin/bash
#SBATCH --job-name=simany_og_task
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodelist=hala
#SBATCH --gpus=a6000:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=outputs/logs/og_task_%j.log

unset PYTHONPATH PYTHONHOME
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin

set -x
CONDA_BASE=${CONDA_ROOT:-$HOME/miniconda3}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior1k

cd ${SIMANY_ROOT:-$PWD}/robo/sim/omnigibson_bridge
OMNIGIBSON_HEADLESS=1 python import_and_run.py --task "$1" --arm "$2"
echo "=== TASK EXIT CODE: $? ==="
