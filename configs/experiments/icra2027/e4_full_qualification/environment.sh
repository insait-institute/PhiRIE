#!/usr/bin/env bash
# Frozen CPU qualification/camera environment; execute from this exact checkout.
set -euo pipefail
unset PYTHONHOME SIMANY_SCENE SIMANY_OUT SIMANY_SCENE_DIR
source /group/worldcept/code/SimAny/outputs/test-headless-setup/env.sh
export TMPDIR=/group/worldcept/code/SimAny/.tmp/icra2027
export XDG_CACHE_HOME=/group/worldcept/code/SimAny/.cache/icra2027
export SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny
export PYTHONPATH=.:/group/worldcept/code/openpi-wt/e4-policy-server/packages/openpi-client/src
export LP_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
