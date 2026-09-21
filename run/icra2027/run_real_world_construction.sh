#!/usr/bin/env bash
# CPU-only historical batch adapter. It cannot launch an unfrozen GPU build.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 FREEZE_ID OUTPUT_ROOT (requires matching E0 contract at OUTPUT_ROOT/contract)" >&2
  exit 2
fi
cd "$ROOT"
PYTHON_BIN="${SIMANY_PYTHON:-$ROOT/.venv/bin/python}"
exec "$PYTHON_BIN" -m robo.eval.real_world_records \
  --config configs/experiments/icra2027/real_world.yaml \
  --repo-root "$ROOT" --freeze-id "$1" --out "$2/real_world"
