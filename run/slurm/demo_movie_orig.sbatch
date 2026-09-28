#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --job-name=simany_demoOrig
#SBATCH --output=outputs/demo_movie/jobOrig.log
# Demo v4: objects keep their ORIGINAL scene gaussians (DEMO_OBJ_SRC=orig);
# generated assets still drive the physics. Reuses the v3 interact pose jsons.
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
OUTD=$SIMANY_ROOT/outputs/demo_movie
export SIMANY_DEMO_SCALE=1.0 SIMANY_DEMO_OBJ_SRC=orig

export SIMANY_SCENE=c50d2d1d42 SIMANY_OUT=$SIMANY_ROOT/outputs/c50d2d1d42_factory
export SIMANY_DEMO_EXCLUDE=5,6
echo "=== discover c50 (orig) ==="
run_gs interface.demo_movie discover --out "$OUTD/discoverhq_c50_orig.mp4"
echo "=== interact c50 (orig, reuse v3 poses) ==="
run_gs interface.demo_movie interact-render --poses "$OUTD/interacthq_c50.json" \
  --out "$OUTD/interacthq_c50_orig.mp4"

export SIMANY_SCENE=45b0dac5e3 SIMANY_OUT=$SIMANY_ROOT/outputs/45b0dac5e3_factory
unset SIMANY_DEMO_EXCLUDE
echo "=== discover 45b0 (orig) ==="
run_gs interface.demo_movie discover --out "$OUTD/discoverhq_45b0_orig.mp4"
echo "=== interact 45b0 (orig, reuse v3 poses) ==="
run_gs interface.demo_movie interact-render --poses "$OUTD/interacthq_45b0.json" \
  --out "$OUTD/interacthq_45b0_orig.mp4"
echo "JOB_ORIG_DONE"
