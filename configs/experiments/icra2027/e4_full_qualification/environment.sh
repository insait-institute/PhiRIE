#!/usr/bin/env bash
# Frozen CPU qualification/camera environment; execute from this exact checkout.
set -euo pipefail
unset PYTHONHOME SIMANY_SCENE SIMANY_OUT SIMANY_SCENE_DIR
source ${SIMANY_ROOT:-$PWD}/outputs/test-headless-setup/env.sh
export TMPDIR=${SIMANY_ROOT:-$PWD}/.tmp/icra2027
export XDG_CACHE_HOME=${SIMANY_ROOT:-$PWD}/.cache/icra2027
export SIMANY_EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}
export PYTHONPATH=.:${OPENPI_ROOT:?set OPENPI_ROOT to the openpi checkout}/worktrees/e4-policy-server/packages/openpi-client/src
export LP_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
