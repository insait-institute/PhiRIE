#!/usr/bin/env bash
#SBATCH --job-name=phiview-demo
#SBATCH --partition=batch
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=msp3-[0-7]
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/group/worldcept/code/PhysicalView/outputs/phiview-dev/job-%j.log
set -euo pipefail
cd "${PHIVIEW_ROOT:-/group/worldcept/code/PhysicalView}"
source run/env.sh
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TMPDIR="$PHYSICALVIEW_ROOT/outputs/phiview-dev/tmp-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
PHIVIEW_OUT="$PHYSICALVIEW_ROOT/outputs/phiview-h200-$SLURM_JOB_ID"
echo "Tunnel: ssh -N -L 8095:$(hostname):8095 runyi_yang@hala.slurm.insait.ai"
exec "$STUDIO_PY" -m physicalview.phiview --demo --selftest --out "$PHIVIEW_OUT" --port "${PORT:-8095}"
