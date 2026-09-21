#!/bin/bash
#SBATCH --account=runyi_yang
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
set -euo pipefail
: "${E3_RVG_CODE:?}" "${E3_RVG_FREEZE:?}"
if [[ -n "${SLURM_ARRAY_JOB_ID:-}${SLURM_ARRAY_TASK_ID:-}${SBATCH_ARRAY_INX:-}" ]]; then
  echo "RVG accepts independent jobs only" >&2
  exit 2
fi
E3_RVG_CONFIG=${E3_RVG_CONFIG:-configs/experiments/icra2027/e3_fresh_rvg.yaml}
E3_RVG_PHASE=${E3_RVG_PHASE:-run}
case "$E3_RVG_PHASE" in plan|run) ;; *) echo "RVG phase must be plan or run" >&2; exit 2 ;; esac
cd "$E3_RVG_CODE"
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
if [[ "$E3_RVG_PHASE" == run ]]; then
  nvidia-smi --query-gpu=uuid,name,memory.total,memory.used --format=csv,noheader
fi
/group/worldcept/PhiRIE/code/SimAny/.venv/bin/python -m run.icra2027.e3_rvg_generation_pilot \
 --config "$E3_RVG_CONFIG" \
 --freeze-root "$E3_RVG_FREEZE" --phase "$E3_RVG_PHASE"
