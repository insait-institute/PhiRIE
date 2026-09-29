#!/bin/bash
# Import-smoke harness: one interpreter process per module (stage modules
# mutate sys.path, so a shared process would cross-contaminate).
#
#   bash run/smoke_imports.sh > /tmp/smoke_pre.txt    # before a refactor
#   bash run/smoke_imports.sh > /tmp/smoke_post.txt   # after
#   diff /tmp/smoke_pre.txt /tmp/smoke_post.txt       # criterion: same set
#
# Known environmental failures (not in .venv, expected): viser (interface
# viewers), omnigibson (bridge smoke test), one argparse-guarded script.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${SIMANY_PY:-$ROOT/.venv/bin/python}
cd "$ROOT"
export PYTHONPATH=$ROOT

try_one() {
  local m=$1 err
  err=$(timeout 120 "$PY" -c "import importlib,sys; importlib.import_module(sys.argv[1])" "$m" 2>&1)
  if [ $? -eq 0 ]; then
    echo "$m OK"
  else
    echo "$m FAIL $(echo "$err" | grep -E "Error|error|Traceback|timeout" | tail -1 | head -c 200)"
  fi
}
export -f try_one
export PY

find agents robo interface capture -name "*.py" -not -path "*__pycache__*" \
  | sed 's|/|.|g; s|\.py$||; s|\.__init__$||' | sort -u \
  | xargs -P 6 -I{} bash -c 'try_one {}' | sort
