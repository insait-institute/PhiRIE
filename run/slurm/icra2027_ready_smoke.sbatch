#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/icra2027/slurm/%x-%j.out
#SBATCH --error=outputs/icra2027/slurm/%x-%j.err

# E1 moved to icra2027_e1_construction.sbatch once its strict, evidence-backed
# inventory contract replaced the old placeholder CSV. This launcher now owns
# only the dependency-light E4 synthetic contract smoke and requests no GRES.
set -euo pipefail

ROOT=${SIMANY_ROOT:-$PWD}
: "${ICRA_SMOKE:?submit with ICRA_SMOKE=e4}"
: "${SLURM_JOB_ID:?this launcher must run under Slurm}"

cd "$ROOT"
export TMPDIR="$ROOT/.tmp/icra2027/slurm-$SLURM_JOB_ID"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$ROOT/outputs/icra2027/slurm"
source run/env.sh

case "$ICRA_SMOKE" in
  e4)
    run robo.eval.harness_smoke \
      --config configs/experiments/harness_smoke.yaml \
      --out "outputs/icra2027/icra2027-contract-v1/harness/smoke-$SLURM_JOB_ID" \
      --resets 3
    ;;
  *)
    echo "unsupported ICRA_SMOKE=$ICRA_SMOKE (expected e4; E1 has a dedicated launcher)" >&2
    exit 2
    ;;
esac
