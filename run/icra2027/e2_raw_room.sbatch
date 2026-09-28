#!/bin/bash
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --nodelist=hala
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
set -euo pipefail
: "${E2_RAW_CODE:?}" "${E2_RAW_FREEZE:?}"
cd "$E2_RAW_CODE"
export PYTHONPATH="$PWD"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
unset PYTHONHOME
hostname
scontrol show job "$SLURM_JOB_ID"
nvidia-smi --query-gpu=uuid,name,memory.total,memory.used --format=csv,noheader
args=(--config "${E2_RAW_CONFIG:-configs/experiments/icra2027/e2_raw_room.yaml}"
      --freeze-root "$E2_RAW_FREEZE" --phase "${E2_RAW_PHASE:-run}")
if [[ -n "${E2_RAW_SCENE:-}" ]]; then args+=(--scene-id "$E2_RAW_SCENE"); fi
${SIMANY_ROOT:-$PWD}/.venv/bin/python -m run.icra2027.e2_raw_room "${args[@]}"
