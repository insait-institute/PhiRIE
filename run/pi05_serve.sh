#!/bin/bash
# Serve pi05_droid_jointpos (openpi websocket policy server, port 8000).
# Run on a GPU machine with >=100 GB host RAM (an A6000-class GPU is plenty,
# ~8 GB for inference, but jax stages the 12 GB checkpoint through host RAM
# during restore, so a low memory limit OOM-kills the restore):
#   bash run/pi05_serve.sh
# The checkpoint is pre-downloaded; --policy.dir uses the local path so no
# GCS access happens at serve time.
set -e
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/checkpoints/openpi_cache}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
cd "${OPENPI_ROOT:?set OPENPI_ROOT to the openpi checkout (github.com/xuningy/openpi fork)}"
# NB: tyro parses global args (--port) BEFORE the policy:checkpoint subcommand
# SIMANY_PI05_CKPT selects the checkpoint SUBPATH under openpi-assets-simeval
# (default = the original zero-shot policy; e.g. set to
# droid_pi05_jointpos_with_web_and_sim/80000 for the sim-co-trained variant).
# All variants share the pi05_droid_jointpos architecture/config.
CKPT_NAME=${SIMANY_PI05_CKPT:-pi05_droid_jointpos}
SHARED_CKPT=${OPENPI_DATA_HOME:-${SIMANY_ROOT:-$PWD}/checkpoints/openpi_cache}/openpi-assets-simeval/$CKPT_NAME
CKPT_DIR=$SHARED_CKPT
# Optional: SIMANY_PI05_LOCAL_CACHE=<fast local dir> rsyncs the checkpoint
# there once (idempotent, .copy_complete marker) and serves from the copy.
# Useful when the checkpoint lives on a network filesystem: orbax/zarr
# restore does many small scattered reads, which can stall on some network
# mounts even though a plain copy of the same 12 GB completes in seconds.
if [ -n "${SIMANY_PI05_LOCAL_CACHE:-}" ] && [ -d "$SHARED_CKPT" ]; then
  LOCAL_CKPT=$SIMANY_PI05_LOCAL_CACHE/${CKPT_NAME//\//_}
  if [ ! -f "$LOCAL_CKPT/.copy_complete" ]; then
    mkdir -p "$LOCAL_CKPT" \
      && rsync -a "$SHARED_CKPT/" "$LOCAL_CKPT/" \
      && touch "$LOCAL_CKPT/.copy_complete" || true
  fi
  if [ -f "$LOCAL_CKPT/.copy_complete" ]; then
    CKPT_DIR=$LOCAL_CKPT
  fi
fi
# The with_web_and_sim checkpoints use an unpadded 8-dim action head and need
# the pi05_droid_jointpos_sim config (added to our openpi fork 2026-08-01).
CONFIG=${SIMANY_PI05_CONFIG:-pi05_droid_jointpos}
echo "[pi05_serve] checkpoint: $CKPT_DIR  config: $CONFIG"
exec uv run --no-sync scripts/serve_policy.py "$@" policy:checkpoint \
  --policy.config=$CONFIG \
  --policy.dir=$CKPT_DIR
