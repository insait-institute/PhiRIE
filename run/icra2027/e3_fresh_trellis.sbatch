#!/bin/bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
set -euo pipefail
: "${E3_TRELLIS_CODE:?}" "${E3_TRELLIS_FREEZE:?}"
cd "$E3_TRELLIS_CODE"
export PYTHONPATH="$PWD"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec ${SIMANY_ROOT:-$PWD}/.venv/bin/python -m run.icra2027.e3_trellis_generation_pilot \
 --config configs/experiments/icra2027/e3_fresh_trellis_pilot.yaml \
 --freeze-root "$E3_TRELLIS_FREEZE" --phase run
