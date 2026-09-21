#!/usr/bin/env bash
# Launch one existing canonical phase. Submit ordinary jobs externally only
# after the exact config/source E0 and smoke gates pass.
set -euo pipefail
: "${E4_COMPACT_CODE:?explicit frozen checkout required}"
: "${E4_COMPACT_FREEZE:?explicit immutable freeze required}"
: "${E4_COMPACT_PHASE:?explicit canonical phase required}"
[[ "$E4_COMPACT_CODE" = /* && "$E4_COMPACT_FREEZE" = /* ]] || exit 2
[[ -z "${SLURM_ARRAY_JOB_ID:-}" ]] || { echo "job arrays are forbidden" >&2; exit 2; }
case "$E4_COMPACT_PHASE" in
  preflight|inventory) ;;
  observe|control)
    case "${E4_COMPACT_SCENE:-}" in 27dd4da69e|40aec5fffa) ;; *) exit 2 ;; esac
    export E3_CANONICAL_SCENE="$E4_COMPACT_SCENE"
    ;;
  *) exit 2 ;;
esac
export E3_CANONICAL_CODE="$E4_COMPACT_CODE"
export E3_CANONICAL_FREEZE="$E4_COMPACT_FREEZE"
export E3_CANONICAL_PHASE="$E4_COMPACT_PHASE"
export E3_CANONICAL_CONFIG="$E4_COMPACT_CODE/configs/experiments/icra2027/e4_compact_canonical/agentic_fresh_execution.yaml"
exec bash "$E4_COMPACT_CODE/run/icra2027/e3_fresh_canonical.sbatch"
