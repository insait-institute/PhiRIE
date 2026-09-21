#!/usr/bin/env bash
#SBATCH --job-name=phiview-paper
#SBATCH --partition=non-reserved
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=msp3-[0-7]
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/group/worldcept/code/PhysicalView/outputs/paper-capture-20260913/job-%j.log
set -euo pipefail
cd /group/worldcept/code/PhysicalView
source run/env.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TMPDIR="$PHYSICALVIEW_ROOT/outputs/paper-capture-20260913/tmp-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec "$STUDIO_PY" -m physicalview.paper_capture "$@"
