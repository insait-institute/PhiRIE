#!/usr/bin/env bash
#SBATCH --job-name=icra-e1-audit
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/icra2027/slurm/e1-%j.out
#SBATCH --error=outputs/icra2027/slurm/e1-%j.err

# Read-only E1 legacy-evidence inventory. This job never launches a constructor
# and never requests a GPU. It publishes either a two-scene/two-regime smoke or
# the complete 50-scene/five-regime preliminary audit below a fresh freeze ID.
set -euo pipefail

ROOT=${SIMANY_ROOT:-$PWD}
PYTHON="$ROOT/.venv/bin/python"
ROSTER="$ROOT/configs/experiments/icra2027/construction_regimes.yaml"

: "${SLURM_JOB_ID:?this launcher must run under Slurm}"
: "${E1_MODE:?submit with E1_MODE=smoke or E1_MODE=full-audit}"
: "${E1_FREEZE_ID:?submit with a unique E1_FREEZE_ID}"
: "${E1_CODE_COMMIT:?submit with the exact 40-character code commit}"
: "${E1_CONTRACT_MANIFEST:?submit with the repository-local E0 freeze_manifest.json}"

[[ "$E1_FREEZE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "E1_FREEZE_ID contains unsafe path characters: $E1_FREEZE_ID" >&2
  exit 2
}
[[ "$E1_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "E1_CODE_COMMIT must be an exact lowercase 40-character Git SHA" >&2
  exit 2
}

cd "$ROOT"
[[ -x "$PYTHON" ]] || { echo "missing E1 Python: $PYTHON" >&2; exit 2; }
[[ -s "$ROSTER" ]] || { echo "missing E1 roster: $ROSTER" >&2; exit 2; }
case "$E1_CONTRACT_MANIFEST" in
  "$ROOT"/*) ;;
  *) echo "E1_CONTRACT_MANIFEST must be below $ROOT" >&2; exit 2 ;;
esac
[[ -s "$E1_CONTRACT_MANIFEST" && ! -L "$E1_CONTRACT_MANIFEST" ]] || {
  echo "missing, empty, or symlinked E0 contract: $E1_CONTRACT_MANIFEST" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "$E1_CODE_COMMIT" ]] || {
  echo "code commit drift: expected $E1_CODE_COMMIT, found $(git rev-parse HEAD)" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo "refusing E1 inventory from a dirty worktree" >&2
  git status --short >&2
  exit 2
}

export TMPDIR="$ROOT/.tmp/icra2027/e1-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$ROOT/.cache/icra2027/e1/xdg"
export MPLCONFIGDIR="$ROOT/.cache/icra2027/e1/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" \
  "$ROOT/outputs/icra2027/slurm"

case "$E1_MODE" in
  smoke)
    EFFECTIVE_FREEZE_ID="${E1_FREEZE_ID}-smoke-${SLURM_JOB_ID}"
    INVENTORY_ARGS=(
      --smoke
      --scene-id 38d58a7a31
      --scene-id c4c04e6d6c
      --regime gt_segments_scan_mesh
      --regime auto_discovery_scan_mesh
    )
    METRICS_ARGS=(--smoke --allow-preliminary)
    ;;
  full-audit)
    EFFECTIVE_FREEZE_ID="$E1_FREEZE_ID"
    INVENTORY_ARGS=()
    METRICS_ARGS=(--allow-preliminary)
    ;;
  *)
    echo "unsupported E1_MODE=$E1_MODE (expected smoke or full-audit)" >&2
    exit 2
    ;;
esac

CONSTRUCTION_OUT="$ROOT/outputs/icra2027/$EFFECTIVE_FREEZE_ID/construction"
[[ ! -e "$CONSTRUCTION_OUT" && ! -L "$CONSTRUCTION_OUT" ]] || {
  echo "refusing to overwrite E1 output: $CONSTRUCTION_OUT" >&2
  exit 2
}

"$PYTHON" -m robo.eval.construction_inventory \
  --outputs-root "$ROOT/outputs" \
  --out "$CONSTRUCTION_OUT" \
  --freeze-id "$EFFECTIVE_FREEZE_ID" \
  --expected-scenes 50 \
  --scene-list "$ROSTER" \
  --contract-manifest "$E1_CONTRACT_MANIFEST" \
  "${INVENTORY_ARGS[@]}"

"$PYTHON" -m robo.eval.construction_metrics \
  --input "$CONSTRUCTION_OUT/scene_records.csv" \
  --out "$CONSTRUCTION_OUT/table" \
  "${METRICS_ARGS[@]}"

"$PYTHON" - "$E1_MODE" "$CONSTRUCTION_OUT" "$E1_CODE_COMMIT" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

mode, output_text, code_commit = sys.argv[1:]
output = Path(output_text)
with (output / "scene_records.csv").open(newline="", encoding="utf-8") as handle:
    records = list(csv.DictReader(handle))
summary = json.loads((output / "missing_artifacts.json").read_text(encoding="utf-8"))
table = json.loads((output / "table/construction_table.json").read_text(encoding="utf-8"))

assert summary["paper_ready"] is False
assert table["paper_ready"] is False
assert table["preliminary_override"] is True
assert not any(row["record_valid"].lower() == "true" for row in records)

if mode == "smoke":
    assert len(records) == 4
    assert len(table["rows"]) == 2
    assert summary["population"]["selected_scene_ids"] == ["38d58a7a31", "c4c04e6d6c"]
    totals = [(row["instances"], row["accepted_instances"]) for row in table["rows"]]
    assert totals == [(4, 3), (3, 2)]
else:
    assert len(records) == 250
    assert len(table["rows"]) == 5
    totals = [(row["instances"], row["accepted_instances"]) for row in table["rows"]]
    assert totals == [(789, 457), (1082, 671), (678, 380), (934, 559), (406, 67)]
    assert [row["planned_scenes"] for row in table["rows"]] == [50] * 5
    assert [row["f1_20"] for row in table["rows"][2:]] == [None, None, None]

manifest = {
    "schema_version": 1,
    "task": "E1",
    "mode": mode,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
    "node": os.environ.get("SLURMD_NODENAME"),
    "code_commit": code_commit,
    "freeze_id": summary["freeze_id"],
    "paper_ready": False,
    "records": len(records),
}
path = output / "run_manifest.json"
with path.open("x", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

(
  cd "$CONSTRUCTION_OUT"
  find . -type f ! -name artifact_hashes.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum > artifact_hashes.sha256
)

echo "E1_AUDIT=PASS"
echo "mode=$E1_MODE"
echo "freeze_id=$EFFECTIVE_FREEZE_ID"
echo "output=$CONSTRUCTION_OUT"
echo "paper_ready=false"
