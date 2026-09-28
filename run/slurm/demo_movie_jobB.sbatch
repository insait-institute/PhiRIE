#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --job-name=simany_demoB
#SBATCH --output=outputs/demo_movie/jobB.log
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
OUTD=$SIMANY_ROOT/outputs/demo_movie
echo "=== rescan d755 (floater cull) ==="
SIMANY_SCENE=d755b3d9d8 SIMANY_OUT=$SIMANY_ROOT/outputs/d755b3d9d8_factory \
  run_gs interface.demo_movie scan --out "$OUTD/scan_d755b3d9d8.mp4"

export SIMANY_SCENE=c50d2d1d42 SIMANY_OUT=$SIMANY_ROOT/outputs/c50d2d1d42_factory
export SIMANY_DEMO_EXCLUDE=5,6
echo "=== discover c50 v2 (hybrid assets, corner boxes restored) ==="
run_gs interface.demo_movie discover --out "$OUTD/discover_c50.mp4"
echo "=== interact c50 v2 (big moves, action camera) ==="
run interface.demo_movie interact-sim --out "$OUTD/interact_c50.json"
run_gs interface.demo_movie interact-render --poses "$OUTD/interact_c50.json" \
  --out "$OUTD/interact_c50.mp4"
echo "JOBB_DONE"
