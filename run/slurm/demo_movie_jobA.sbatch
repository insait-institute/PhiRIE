#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=simany_demoA
#SBATCH --output=outputs/demo_movie/jobA.log
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
OUTD=$SIMANY_ROOT/outputs/demo_movie
mkdir -p "$OUTD"
for S in 578511c8a9 c50d2d1d42 d755b3d9d8; do
  echo "=== scan $S ==="
  SIMANY_SCENE=$S SIMANY_OUT=$SIMANY_ROOT/outputs/${S}_factory \
    run_gs interface.demo_movie scan --out "$OUTD/scan_$S.mp4"
done

export SIMANY_SCENE=c50d2d1d42 SIMANY_OUT=$SIMANY_ROOT/outputs/c50d2d1d42_factory
echo "=== discover c50 ==="
run_gs interface.demo_movie discover --out "$OUTD/discover_c50.mp4"
echo "=== interact c50 ==="
run interface.demo_movie interact-sim --out "$OUTD/interact_c50.json"
run_gs interface.demo_movie interact-render --poses "$OUTD/interact_c50.json" \
  --out "$OUTD/interact_c50.mp4"
echo "JOBA_DONE"
