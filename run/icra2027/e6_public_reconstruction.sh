#!/usr/bin/env bash
# Submit independently with --export=ALL --no-requeue, one >=48GB GPU, 30min.
set -euo pipefail
: "${E6_CODE_ROOT:?exact immutable source checkout required}"
: "${E6_CONFIG:?sealed generated execution config required}"
: "${E6_STAGE:?new canonical stage required}"
: "${E6_CONDITION:?explicit clean/mild/severe condition required}"
cd "$E6_CODE_ROOT"
export PYTHONPATH="$E6_CODE_ROOT"
export SIMANY_AUTO=1 SIMANY_MESH_SRC=derived
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR=/group/worldcept/PhiRIE/code/SimAny/.tmp/icra2027
export XDG_CACHE_HOME=/group/worldcept/PhiRIE/code/SimAny/.cache/icra2027/e6-runtime
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /group/worldcept/PhiRIE/code/SimAny/.venv/bin/python -m run.icra2027.e6_public_reconstruction \
  --config "$E6_CONFIG" --stage-root "$E6_STAGE" --condition "$E6_CONDITION"
