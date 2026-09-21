#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${SIMANY_PY:-"$ROOT/.venv/bin/python"}

if [[ $# -ne 1 ]]; then
  echo "usage: bash run/icra2027/e6_fast_smoke.sh <fresh-repo-local-run-dir>" >&2
  exit 2
fi

RUN_OUT=$1
if [[ -e "$RUN_OUT" || -L "$RUN_OUT" ]]; then
  echo "refusing to overwrite E6 smoke run: $RUN_OUT" >&2
  exit 1
fi

cd "$ROOT"
"$PY" -m robo.eval.build_task_support_dataset \
  --smoke \
  --out "$RUN_OUT/features"
"$PY" -m robo.eval.audit_loso \
  --input "$RUN_OUT/features/task_local_features_and_labels.csv" \
  --out "$RUN_OUT/predictions" \
  --l2 1.0
"$PY" -m robo.eval.audit_metrics \
  --input "$RUN_OUT/predictions/heldout_predictions.csv" \
  --out "$RUN_OUT/table"

test -s "$RUN_OUT/table/audit_table.csv"
test -s "$RUN_OUT/table/risk_coverage.csv"
echo "E6 synthetic fast smoke complete: $RUN_OUT"
