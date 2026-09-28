#!/usr/bin/env bash
# Exactly one synthetic service smoke, without environment or rollout imports.
#SBATCH --job-name=e4-policy-service
#SBATCH --partition=batch
#SBATCH --nodelist=hala
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=00:25:00
#SBATCH --signal=B:TERM@60
set -euo pipefail
if [[ $# != 4 ]]; then
  echo 'usage: policy_service_smoke.sbatch ROOT EXPECTED_COMMIT CONTRACT_DIR OUTPUT_DIR' >&2
  exit 2
fi
ROOT=$1
SOURCE_COMMIT=$2
CONTRACT_DIR=$3
OUTPUT_DIR=$4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:${OPENPI_ROOT:?set OPENPI_ROOT to the openpi checkout}/worktrees/e4-policy-server/packages/openpi-client/src"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$ROOT"
exec ${OPENPI_ROOT:?set OPENPI_ROOT to the openpi checkout}/.venv/bin/python \
  run/icra2027/policy_service_smoke.py \
  --config configs/experiments/icra2027/policy_service_smoke.yaml \
  --expected-code-commit "$SOURCE_COMMIT" \
  --contract "$CONTRACT_DIR" --out "$OUTPUT_DIR"
