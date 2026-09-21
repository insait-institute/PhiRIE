#!/usr/bin/env bash
# Compatibility entry point only. The current final plan owns all CLI semantics.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ -z "${SIMANY_CAMPAIGN_PY:-}" && -n "${SIMANY_FINAL_PY:-}" ]]; then
  export SIMANY_CAMPAIGN_PY="$SIMANY_FINAL_PY"
fi
printf '%s\n' 'Using run/campaign/finalize.sh; read plan/icra2027/14_final_experiments/RUNBOOK.md. Older ZIP flags are not translated.' >&2
exec bash "$ROOT/run/campaign/finalize.sh" "$@"
