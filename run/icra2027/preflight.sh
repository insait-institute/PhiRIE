#!/usr/bin/env bash
# CPU-only ICRA 2027 contract preflight.  All temporary and durable products
# stay beneath the SimAny checkout; no system temporary directory is used.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${SIMANY_PY:-$ROOT/.venv/bin/python}
MODE=${1:-}

if [[ "$MODE" != "--smoke" ]]; then
  echo "usage: bash run/icra2027/preflight.sh --smoke" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "python interpreter is not executable: $PY" >&2
  exit 2
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="preflight-smoke-${STAMP}-$$"
RUN_ROOT="$ROOT/outputs/icra2027/$RUN_ID"
export TMPDIR="$ROOT/.tmp/icra2027"
BASE_TEMP="$TMPDIR/pytest-e0-${STAMP}-$$"
LOG="$RUN_ROOT/preflight.log"

mkdir -p "$RUN_ROOT" "$TMPDIR"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec > >(tee -a "$LOG") 2>&1

echo "ICRA 2027 E0 preflight"
echo "run_root=$RUN_ROOT"
echo "tmpdir=$TMPDIR"
echo "python=$PY"

echo "[1/5] canonical imports"
"$PY" - <<'PY'
import importlib

modules = (
    "robo.manifest.hash",
    "robo.manifest.io",
    "robo.manifest.schema",
    "robo.eval.freeze",
    "robo.eval.harness_spec",
    "robo.eval.harness_validation",
    "robo.eval.harness_smoke",
    "robo.eval.agentic_ablation",
    "agents.orchestrator.controller",
    "agents.orchestrator.runtime",
)
for name in modules:
    importlib.import_module(name)
    print(f"{name} OK")
PY

echo "[2/5] manifest, E0, and E3 exact-commit tests"
"$PY" -m pytest -q \
  tests/test_manifest_roundtrip.py \
  tests/test_harness_spec.py \
  tests/test_harness_validation.py \
  tests/test_freeze.py \
  tests/test_agentic_ablation.py \
  tests/test_agentic_configs.py \
  tests/test_agentic_orchestrator.py \
  tests/test_agentic_runtime.py \
  tests/test_e3_slurm_contract.py \
  --basetemp="$BASE_TEMP"

echo "[3/5] validate the frozen ICRA harness declaration"
"$PY" - <<'PY'
from robo.eval.harness_spec import load_harness_spec

spec = load_harness_spec("configs/experiments/icra2027/harness.yaml")
print(f"treatments={len(spec.treatments)} comparisons={len(spec.comparisons)} OK")
PY

echo "[4/5] complete synthetic treatment/reset matrix"
"$PY" -m robo.eval.harness_smoke \
  --config configs/experiments/icra2027/harness.yaml \
  --out "$RUN_ROOT/harness" \
  --resets 3

echo "[5/5] dry-run and publish an isolated smoke freeze"
"$PY" -m robo.eval.freeze \
  --config configs/experiments/icra2027/freeze.yaml \
  --out "$RUN_ROOT/dry-run-contract" \
  --allow-dirty-for-smoke \
  --dry-run | tee "$RUN_ROOT/freeze_dry_run.json"
echo "pre-publish imports/tests/harness/dry-run gates: PASS"
"$PY" -m robo.eval.freeze \
  --config configs/experiments/icra2027/freeze.yaml \
  --out "$RUN_ROOT/contract" \
  --allow-dirty-for-smoke \
  --preflight-log "$LOG"

echo "PREFLIGHT=PASS"
echo "output=$RUN_ROOT"
