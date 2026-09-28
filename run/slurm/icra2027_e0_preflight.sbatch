#!/usr/bin/env bash
#SBATCH --job-name=icra-e0-contract
#SBATCH --no-requeue
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/icra2027/slurm/e0-%j.out
#SBATCH --error=outputs/icra2027/slurm/e0-%j.err

# E0 is deliberately CPU-only. No GPU/GRES is requested: this verification
# must not occupy an accelerator needed by the quantitative experiments.
set -euo pipefail

ROOT=${SIMANY_ROOT:-$PWD}
cd "$ROOT"
export TMPDIR="$ROOT/.tmp/icra2027"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

bash run/icra2027/preflight.sh --smoke
