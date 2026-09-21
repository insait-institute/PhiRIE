#!/usr/bin/env bash
# One original pilot canonical instance per ordinary job. No scheduler submission.
set -euo pipefail
seed=${1:?usage: capture_pilot_seed.sh SEED NEW_STAGE_ROOT}
stage=${2:?new immutable stage root required}
case "$seed" in 0|1|2|3|4) ;; *) exit 2;; esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python_native=/group/worldcept/PhiRIE/code/SimAny-wt/sr0-native/.venv-native/bin/python
pilot=/group/worldcept/PhiRIE/code/SimAny/outputs/icra2027/20260906-4ee6462-v2/sim_recon_sim/reference/pilot
identity=/group/worldcept/PhiRIE/code/SimAny/outputs/icra2027/20260906-67ee364-v1/sim_recon_sim/reference/identity/identity_report.json
source run/roundtrip/native_env.sh
capture_id=$("$python_native" - "$pilot/canonical_seed$seed/canonical_state.json" <<'PY'
import sys,hashlib
print('c-'+hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:16])
PY
)
exec "$python_native" -X faulthandler -m robo.roundtrip.capture_native \
 --config configs/experiments/sim_recon_sim/reference.yaml --identity-report "$identity" \
 --canonical-reference "$pilot/canonical_seed$seed" --reference-seed "$seed" --capture-id "$capture_id" \
 --out "$stage/capture_seed$seed" --public-out "$stage/public_seed$seed" --vault "$stage/vault_seed$seed" \
 --width 1280 --height 720
