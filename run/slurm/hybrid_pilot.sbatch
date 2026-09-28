#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=simany_hybrid
#SBATCH --output=outputs/run_hybrid_pilot.log

# Hybrid V_mesh pilot: TRELLIS vs ReconViaGen per object, chamfer-selected.
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
export SIMANY_OUT=${SIMANY_OUT:-$SIMANY_ROOT/outputs/${SIMANY_SCENE}_factory}

run agents.assets.factory_hybrid "$@"
rc=$?
echo "EXIT_CODE=$rc"
exit $rc
