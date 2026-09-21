#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-$MUJOCO_GL}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
: "${SR_NATIVE_PY:=$ROOT/.venv-native/bin/python}"
exec "$SR_NATIVE_PY" -m robo.roundtrip.reference "$@"
