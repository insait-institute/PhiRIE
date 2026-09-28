#!/bin/bash
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=simany_replay
#SBATCH --output=outputs/demo_sessions/replay_%j.log
# Render a viewer-recorded take photoreal (objects = original scene splats).
#   sbatch --export=ALL,TAKE=outputs/demo_sessions/<set>/take_X.json run/slurm/replay_take.sbatch
# Optional: SCALE=1.0 (default, full 1752x1168), SMOOTH=0.6, CAPTION="..."
set -e
SIMANY_ROOT=${SIMANY_ROOT:-$PWD}
source "$SIMANY_ROOT/run/env.sh"
[ -n "$TAKE" ] || { echo "set TAKE=<take json>"; exit 1; }
SET_DIR=$(basename "$(dirname "$TAKE")")           # e.g. c50d2d1d42_factory
export SIMANY_SCENE=${SET_DIR%%_*}
export SIMANY_OUT=$SIMANY_ROOT/outputs/$SET_DIR
export SIMANY_DEMO_SCALE=${SCALE:-1.0}
OUT_MP4=${OUT_MP4:-${TAKE%.json}.mp4}
run_gs interface.demo_movie replay --take "$TAKE" --out "$OUT_MP4" \
  --smooth "${SMOOTH:-0.6}" ${CAPTION:+--caption "$CAPTION"}
echo "REPLAY_DONE $OUT_MP4"
