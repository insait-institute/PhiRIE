#!/usr/bin/env bash
#SBATCH --job-name=phiview-next-dataset
#SBATCH --partition=non-reserved
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=msp3-[0-7]
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/group/worldcept/code/PhysicalView/outputs/paper-capture-20260913/foreign-%A_%a.log
set -euo pipefail
cd /group/worldcept/code/PhysicalView
source run/env.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$PHYSICALVIEW_ROOT/outputs/paper-capture-20260913/tmp-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
if [ "${SLURM_ARRAY_TASK_ID:-0}" = 0 ]; then
  case "${DATASET:?Choose libero or behavior}" in
    libero) previous_dataset=scannetpp ;;
    behavior) previous_dataset=libero ;;
  esac
  "$STUDIO_PY" -m physicalview.paper_require_complete --root "$PHYSICALVIEW_ROOT/outputs/paper-capture-20260913" --dataset "$previous_dataset"
fi
exec "$STUDIO_PY" -m physicalview.paper_foreign --root "$PHYSICALVIEW_ROOT/outputs/paper-capture-20260913" --dataset "${DATASET:?Choose libero or behavior}" --index "${SLURM_ARRAY_TASK_ID:-0}"
