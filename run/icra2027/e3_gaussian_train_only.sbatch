#!/bin/bash
#SBATCH --account=runyi_yang
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
: "${E3_GAUSSIAN_CODE:?}" "${E3_GAUSSIAN_FREEZE:?}" "${E3_GAUSSIAN_PHASE:?}"
cd "$E3_GAUSSIAN_CODE"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=42
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
config=${E3_GAUSSIAN_CONFIG:-configs/experiments/icra2027/e3_gaussian_train_only.yaml}
args=(--config "$config" --freeze-root "$E3_GAUSSIAN_FREEZE")
if [[ -n "${E3_GAUSSIAN_SCENE:-}" ]]; then
  args+=(--scene-id "$E3_GAUSSIAN_SCENE")
fi
case "$E3_GAUSSIAN_PHASE" in
  initialize) phases=(prepare triangulate) ;;
  smoke-and-full) phases=(train-smoke train-full) ;;
  *) phases=("$E3_GAUSSIAN_PHASE") ;;
esac
for phase in "${phases[@]}"; do
  /group/worldcept/PhiRIE/code/SimAny/.venv/bin/python -m run.icra2027.e3_gaussian_train_only "${args[@]}" --phase "$phase"
done
