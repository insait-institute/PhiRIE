#!/usr/bin/env bash
#SBATCH --job-name=phiview-paper10
#SBATCH --partition=non-reserved
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=msp3-[0-7]
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/group/worldcept/code/PhysicalView/outputs/paper-capture-20260913/campaign-%A_%a.log
set -euo pipefail
cd /group/worldcept/code/PhysicalView
source run/env.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$PHYSICALVIEW_ROOT/outputs/paper-capture-20260913/tmp-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec "$STUDIO_PY" -m physicalview.paper_campaign --root "$PHYSICALVIEW_ROOT/outputs/paper-capture-20260913" --index "$SLURM_ARRAY_TASK_ID"
