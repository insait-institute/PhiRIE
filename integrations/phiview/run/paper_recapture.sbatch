#!/usr/bin/env bash
#SBATCH --job-name=phiview-paper-recapture
#SBATCH --partition=non-reserved
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --exclude=msp3-[0-7]
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/group/worldcept/code/PhysicalView/outputs/paper-capture-20260913/recapture-%A_%a.log
set -euo pipefail
cd /group/worldcept/code/PhysicalView
source run/env.sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec "$STUDIO_PY" -m physicalview.paper_recapture \
  --root "$PHYSICALVIEW_ROOT/outputs/paper-capture-20260913" --index "${SLURM_ARRAY_TASK_ID:?}"
