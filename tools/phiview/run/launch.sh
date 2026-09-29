#!/bin/bash
# Launch SimAny Studio on a GPU node and print the SSH tunnel command.
#
#   bash run/launch.sh [--gpu a6000|h200|a100|rtx6000] [--port 8080] [--time 03:50:00]
#                             [--mem 64G] [--local] [--share] [-- extra app args]
#
#   --local   run on the current node (must already have a GPU / be inside an allocation)
#   default   srun an interactive allocation on the requested GPU type and start the app
#             there; the app prints "listening on http://0.0.0.0:<port>"; tunnel with
#             ssh -L <port>:<node>:<port> <login-host>   then open http://localhost:<port>
#
# GPU types map to Slurm partitions/gres per configs/default.yaml (mirrored here so
# the launcher needs no python). H200 jobs exclude msp3-* nodes (no /group mount).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=a6000; PORT=8080; TIME=03:50:00; MEM=64G; LOCAL=0; SHARE=""; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --time) TIME=$2; shift 2;;
    --mem) MEM=$2; shift 2;;
    --local) LOCAL=1; shift;;
    --share) SHARE="--share"; shift;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
case "$GPU" in
  a6000)   PART=debug; GRES=gpu:a6000:1; XTRA="";;
  h200)    PART=batch; GRES=gpu:h200:1; XTRA="--exclude=msp3-[0-7]";;
  a100)    PART=batch; GRES=gpu:a100-80g:1; XTRA="";;
  rtx6000) PART=batch; GRES=gpu:rtx6000:1; XTRA="";;
  *) echo "unknown --gpu $GPU (a6000|h200|a100|rtx6000)" >&2; exit 2;;
esac

APP_CMD="cd '$ROOT' && source run/env.sh && echo \"[studio] node: \$(hostname)  tunnel: ssh -L $PORT:\$(hostname):$PORT \$USER@\$(hostname -f | sed 's/^[^.]*\.//;s/^/login./')\" && exec \"\$STUDIO_PY\" -m physicalview.app --port $PORT $SHARE ${EXTRA[*]:-}"

if [ "$LOCAL" = 1 ]; then
  bash -lc "$APP_CMD"
else
  echo "[studio] requesting $GPU ($PART, $GRES) for $TIME ..." >&2
  exec srun --partition="$PART" --gres="$GRES" $XTRA --mem="$MEM" --cpus-per-task=8 \
       --time="$TIME" --job-name="studio-$GPU" --pty bash -lc "$APP_CMD"
fi
