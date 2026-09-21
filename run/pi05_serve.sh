#!/bin/bash
# Serve pi05_droid_jointpos (openpi websocket policy server, port 8000).
# Run on a GPU node (A6000 is plenty: ~8 GB for inference):
#   srun -p debug --gres=gpu:a6000:1 --mem=100G --time=240 bash run/pi05_serve.sh
# (--mem matters: jax stages the 12 GB checkpoint through host RAM, and the
#  cgroup default of 2G OOM-kills the restore.)
# The checkpoint is pre-downloaded; --policy.dir uses the local path so no
# GCS access happens at serve time.
set -e
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/group/worldcept/PhiRIE/checkpoints/openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
cd /group/worldcept/PhiRIE/code/openpi
# NB: tyro parses global args (--port) BEFORE the policy:checkpoint subcommand
# Prefer a local-scratch copy of the checkpoint: orbax/zarr restore does many
# small scattered reads, and that access pattern got stuck for 1h+ (zero
# progress, no error) against the CephFS-hosted copy twice in a row on
# 2026-07-26/27, even though a plain rsync of the same 12GB path completed
# in 8.7s moments later - so it wasn't a general filesystem outage, just
# something about the checkpoint-restore read pattern specifically.
# SIMANY_PI05_CKPT selects the checkpoint SUBPATH under openpi-assets-simeval
# (default = the original zero-shot policy; e.g. set to
# droid_pi05_jointpos_with_web_and_sim/80000 for the sim-co-trained variant).
# All variants share the pi05_droid_jointpos architecture/config.
CKPT_NAME=${SIMANY_PI05_CKPT:-pi05_droid_jointpos}
SHARED_CKPT=/group/worldcept/PhiRIE/checkpoints/openpi_cache/openpi-assets-simeval/$CKPT_NAME
LOCAL_CKPT=/scratch/runyi_yang/openpi_cache_local/${CKPT_NAME//\//_}
if [ ! -f "$LOCAL_CKPT/.copy_complete" ] && [ -d "$SHARED_CKPT" ]; then
  # make the node-local copy ourselves (idempotent; ~9 s for 12 GB)
  mkdir -p "$LOCAL_CKPT" \
    && rsync -a "$SHARED_CKPT/" "$LOCAL_CKPT/" \
    && touch "$LOCAL_CKPT/.copy_complete" || true
fi
if [ -f "$LOCAL_CKPT/.copy_complete" ]; then
  CKPT_DIR=$LOCAL_CKPT
else
  CKPT_DIR=$SHARED_CKPT
fi
# The with_web_and_sim checkpoints use an unpadded 8-dim action head and need
# the pi05_droid_jointpos_sim config (added to our openpi fork 2026-08-01).
CONFIG=${SIMANY_PI05_CONFIG:-pi05_droid_jointpos}
echo "[pi05_serve] checkpoint: $CKPT_DIR  config: $CONFIG"
exec uv run --no-sync scripts/serve_policy.py "$@" policy:checkpoint \
  --policy.config=$CONFIG \
  --policy.dir=$CKPT_DIR
