#!/usr/bin/env bash
# One sealed native DEV instance and static capture per ordinary GPU job.
set -euo pipefail
slot_id=${1:?usage: acquire_instance.sh SLOT_ID NEW_INSTANCE_OUT}
instance_out=${2:?new immutable instance output required}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python_native=${SIMANY_ROOT:-$PWD}/worktrees/sr0-native/.venv-native/bin/python
source run/roundtrip/native_env.sh
exec "$python_native" -X faulthandler -m robo.roundtrip.matrix --phase acquire \
 --config configs/experiments/sim_recon_sim/scale_up/dev.yaml --slot-id "$slot_id" --out "$instance_out" \
 --width 1280 --height 720 "${@:3}"
