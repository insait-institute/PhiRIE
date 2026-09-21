#!/bin/bash
#SBATCH --account=runyi_yang
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
set -euo pipefail
: "${E3_TRELLIS2_CODE:?}" "${E3_TRELLIS2_FREEZE:?}" "${E3_TRELLIS2_CONFIG:?}"
cd "$E3_TRELLIS2_CODE"
export PYTHONPATH="$PWD" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR="$E3_TRELLIS2_FREEZE/dispatch/tmp-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
# GPU allocation is explicit at submission. CPU audit/probe requests no GPU.
exec /group/worldcept/PhiRIE/code/SimAny/.venv/bin/python -m run.icra2027.e3_trellis_generation_pilot \
  --config "$E3_TRELLIS2_CONFIG" --freeze-root "$E3_TRELLIS2_FREEZE" \
  --phase "${E3_TRELLIS2_PHASE:-run}"
