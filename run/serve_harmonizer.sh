#!/usr/bin/env bash
set -euo pipefail

: "${HARMONIZER_SOURCE:?Set HARMONIZER_SOURCE to NVIDIA/harmonizer/src}"
: "${HARMONIZER_CHECKPOINT:?Set HARMONIZER_CHECKPOINT to the released checkpoint}"
SOCKET=${HARMONIZER_SOCKET:-/tmp/simany_harmonizer.sock}
RESOLUTION=${HARMONIZER_RESOLUTION:-1024}
TIMESTEP=${HARMONIZER_TIMESTEP:-250}

exec python -m tools.harmonizer.server \
  --socket "$SOCKET" \
  --backend nvidia \
  --source-dir "$HARMONIZER_SOURCE" \
  --model-path "$HARMONIZER_CHECKPOINT" \
  --resolution "$RESOLUTION" \
  --timestep "$TIMESTEP"
