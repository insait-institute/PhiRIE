#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --job-name=simany_demoOrig2
#SBATCH --output=outputs/demo_movie/jobOrig2.log
# Re-render ONLY the interact segments: origobj membership now follows the
# poses json (kept-set drift stranded the throw actor as a static paste-back).
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
OUTD=$SIMANY_ROOT/outputs/demo_movie
export SIMANY_DEMO_SCALE=1.0 SIMANY_DEMO_OBJ_SRC=orig

export SIMANY_SCENE=c50d2d1d42 SIMANY_OUT=$SIMANY_ROOT/outputs/c50d2d1d42_factory
export SIMANY_DEMO_EXCLUDE=5,6
run_gs interface.demo_movie interact-render --poses "$OUTD/interacthq_c50.json" \
  --out "$OUTD/interacthq_c50_orig.mp4"

export SIMANY_SCENE=45b0dac5e3 SIMANY_OUT=$SIMANY_ROOT/outputs/45b0dac5e3_factory
unset SIMANY_DEMO_EXCLUDE
run_gs interface.demo_movie interact-render --poses "$OUTD/interacthq_45b0.json" \
  --out "$OUTD/interacthq_45b0_orig.mp4"
echo "JOB_ORIG2_DONE"
