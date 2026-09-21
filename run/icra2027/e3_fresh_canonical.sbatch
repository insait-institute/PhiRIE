#!/bin/bash
#SBATCH --account=runyi_yang
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
: "${E3_CANONICAL_CODE:?}" "${E3_CANONICAL_FREEZE:?}" "${E3_CANONICAL_PHASE:?}" "${E3_CANONICAL_CONFIG:?}"
[[ -z "${SLURM_ARRAY_JOB_ID:-}" ]] || { echo "ordinary jobs required" >&2; exit 2; }
cd "$E3_CANONICAL_CODE"
export SIMANY_EVIDENCE_ROOT=/group/worldcept/PhiRIE/code/SimAny
export PYTHONPATH="$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export SIMANY_ROOT="$PWD" SIMANY_NO_GT=1 SIMANY_AUTO=1 SIMANY_MESH_SRC=derived
scene_args=()
if [[ "$E3_CANONICAL_PHASE" == observe || "$E3_CANONICAL_PHASE" == control ]]; then
  : "${E3_CANONICAL_SCENE:?}"
  export SIMANY_SCENE="$E3_CANONICAL_SCENE"
  scene_args=(--scene-id "$E3_CANONICAL_SCENE")
else
  unset SIMANY_SCENE
fi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
phase_python=/group/worldcept/PhiRIE/code/SimAny/.venv/bin/python
if [[ "$E3_CANONICAL_PHASE" == observe ]]; then
  phase_python=/group/worldcept/code/affordancept/.envs/mini-viewer/bin/python
  export PYTHONPATH="$PWD:/group/worldcept/PhiRIE/code/SimAny/.cache/icra2027/e3-gaussian-runtime-py310-pydantic-2.13.4"
  export TORCH_EXTENSIONS_DIR=/group/worldcept/PhiRIE/code/SimAny/.cache/icra2027/e3-gaussian-torch-extensions-py310-pt241-cu124
fi
exec "$phase_python" -m run.icra2027.e3_fresh_canonical_phase \
  --config "$E3_CANONICAL_CONFIG" \
  --freeze-root "$E3_CANONICAL_FREEZE" --phase "$E3_CANONICAL_PHASE" "${scene_args[@]}"
