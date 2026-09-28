#!/bin/bash
#SBATCH --job-name=simany_robot_render
#SBATCH --partition=non-reserved
#SBATCH --qos=normal
#SBATCH --gpus=rtx6000:1
#SBATCH --time=02:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=outputs/logs/og_robot_render_%j.log

unset PYTHONPATH PYTHONHOME
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin

set -x
CONDA_BASE=${CONDA_ROOT:-$HOME/miniconda3}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior1k

cd ${SIMANY_ROOT:-$PWD}/robo/sim/omnigibson_bridge
OMNIGIBSON_HEADLESS=1 python robot_pick_place_demo.py --render --out-dir ${SIMANY_ROOT:-$PWD}/outputs/omnigibson_export/robot_demo_frames
echo "=== ROBOT RENDER DEMO EXIT CODE: $? ==="
