#!/usr/bin/env bash
# Submit exactly one ordinary A100 camera/workspace/scorer gate job.
set -euo pipefail

CODE_ROOT=/group/worldcept/PhiRIE/code/SimAny-wt/e4-paired-pilot
EVIDENCE_ROOT=/group/worldcept/PhiRIE/code/SimAny
PYTHON=/group/worldcept/PhiRIE/code/SimAny/.venv/bin/python
LAUNCHER="$CODE_ROOT/run/slurm/icra2027_e4_camera_scorer_gpu.sbatch"
MENAGERIE_ROOT="$CODE_ROOT/third_party/mujoco_menagerie"
OPENPI_ROOT=/group/worldcept/PhiRIE/code/openpi-wt/e4-policy-server
OPENPI_CLIENT_SRC="$OPENPI_ROOT/packages/openpi-client/src"
CPU_FREEZE_ID=icra2027-contract-v1-e4-0dda134578b2-region-cpu-prep1-20260904T135257Z
CPU_PRODUCER_COMMIT=0dda134578b25cd12d1791193bb437c6d310776f
CPU_GATE_SHA256=b4bb242ace4ce641051e58df003d126a1383ae0ddd003e3fe4dce3e0493ef355
MENAGERIE_SOURCE_COMMIT=71f066ad0be9cd271f7ed58c030243ef157af9f4
ACCOUNT=runyi_yang
SUBMIT_USER=$(id -un)
GPU_NODE=gcp-eu1-a100-80g-qrfh
GPU_GRES=a100-80g:1
if [[ -v SBATCH_BIN || -v SACCTMGR_BIN ]]; then
  [[ "${E4_TEST_TOOL_OVERRIDES:-}" == 1 ]] || {
    echo "SBATCH_BIN/SACCTMGR_BIN overrides are test-only" >&2
    exit 2
  }
fi
SBATCH_BIN=${SBATCH_BIN:-sbatch}
SACCTMGR_BIN=${SACCTMGR_BIN:-sacctmgr}

: "${E4_CAMERA_GATE_ID:?set a fresh E4_CAMERA_GATE_ID}"
[[ "$E4_CAMERA_GATE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_CAMERA_GATE_ID: $E4_CAMERA_GATE_ID" >&2; exit 2;
}
[[ "$E4_CAMERA_GATE_ID" != "$CPU_FREEZE_ID" ]] || {
  echo "camera gate must use a fresh sibling ID" >&2; exit 2;
}
[[ "$SUBMIT_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe submit user: $SUBMIT_USER" >&2; exit 2;
}
for array_variable in \
  SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP \
  SBATCH_ARRAY_INX SBATCH_ARRAY
do
  if [[ -v "$array_variable" ]]; then
    echo "E4 camera submitter refuses array context: $array_variable is set" >&2
    exit 2
  fi
done

command -v "$SBATCH_BIN" >/dev/null || { echo "missing sbatch" >&2; exit 2; }
command -v "$SACCTMGR_BIN" >/dev/null || { echo "missing sacctmgr" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
[[ -s "$LAUNCHER" && ! -L "$LAUNCHER" ]] || {
  echo "missing, empty, or symlinked launcher: $LAUNCHER" >&2; exit 2;
}
[[ -d "$MENAGERIE_ROOT" && ! -L "$MENAGERIE_ROOT" ]] || {
  echo "missing local Menagerie closure" >&2; exit 2;
}
cpu_gate="$EVIDENCE_ROOT/outputs/icra2027/$CPU_FREEZE_ID/preflight/gate.json"
[[ -s "$cpu_gate" && ! -L "$cpu_gate" ]] || { echo "missing CPU gate" >&2; exit 2; }
[[ "$(sha256sum "$cpu_gate" | awk '{print $1}')" == "$CPU_GATE_SHA256" ]] || {
  echo "CPU gate SHA differs" >&2; exit 2;
}

CODE_COMMIT=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "invalid code commit" >&2; exit 2; }
[[ "$(git -C "$CODE_ROOT" rev-parse --show-toplevel)" == "$CODE_ROOT" ]] || {
  echo "submitter is not using the isolated worktree" >&2; exit 2;
}
if [[ -n "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "refusing submission from a dirty worktree" >&2
  git -C "$CODE_ROOT" status --short >&2
  exit 2
fi

# Validate the ignored worktree-local runtime copy by its full 94-file closure.
SIMANY_EVIDENCE_ROOT="$EVIDENCE_ROOT" \
  PYTHONPATH="$OPENPI_CLIENT_SRC:$CODE_ROOT" "$PYTHON" - \
  "$MENAGERIE_ROOT" "$MENAGERIE_SOURCE_COMMIT" <<'PY'
import sys
from pathlib import Path
from robo.eval.e4_camera_scorer_gate import _menagerie_snapshot, _openpi_snapshot
snapshot = _menagerie_snapshot(Path(sys.argv[1]), sys.argv[2])
if snapshot["file_count"] != 94:
    raise SystemExit("Menagerie closure count differs")
_openpi_snapshot()
PY

assoc_rows=$(
  "$SACCTMGR_BIN" -n -P show assoc \
    where user="$SUBMIT_USER" account="$ACCOUNT" \
    format=Account,User,Partition,QOS
)
if ! awk -F '|' -v account="$ACCOUNT" -v user="$SUBMIT_USER" '
  $1 == account && $2 == user {
    n = split($4, qos, ",")
    for (i = 1; i <= n; i++) if (qos[i] == "normal") found = 1
  }
  END { exit !found }
' <<<"$assoc_rows"; then
  echo "account/user association does not explicitly permit qos=normal" >&2
  echo "$assoc_rows" >&2
  exit 2
fi

EXPERIMENT_ROOT="$EVIDENCE_ROOT/outputs/icra2027/$E4_CAMERA_GATE_ID"
QUALIFIER_PREFLIGHT_ROOT="$EVIDENCE_ROOT/outputs/icra2027/${E4_CAMERA_GATE_ID}-qualifier-preflight"
SUBMISSION_DIR="$EVIDENCE_ROOT/outputs/icra2027/submissions/${E4_CAMERA_GATE_ID}-camera-scorer"
LOG_DIR="$SUBMISSION_DIR/logs"
RECEIPT_DIR="$SUBMISSION_DIR/sbatch_receipts"
LEDGER="$SUBMISSION_DIR/jobs.tsv"
[[ ! -e "$EXPERIMENT_ROOT" && ! -L "$EXPERIMENT_ROOT" ]] || {
  echo "refusing to reuse camera gate output: $EXPERIMENT_ROOT" >&2; exit 2;
}
[[ ! -e "$QUALIFIER_PREFLIGHT_ROOT" && ! -L "$QUALIFIER_PREFLIGHT_ROOT" ]] || {
  echo "refusing to reuse qualifier preflight output: $QUALIFIER_PREFLIGHT_ROOT" >&2; exit 2;
}
[[ ! -e "$SUBMISSION_DIR" && ! -L "$SUBMISSION_DIR" ]] || {
  echo "refusing to reuse submission directory: $SUBMISSION_DIR" >&2; exit 2;
}

# Spend CPU before GPU.  This path constructs the exact pi0.5 rig and reuses
# DroidSimEnv.reset's 900-step settle, but it never constructs a Renderer.
# A failed qualifier contract remains as sealed diagnostics and stops here.
if ! env -u MUJOCO_GL -u PYOPENGL_PLATFORM \
  SIMANY_EVIDENCE_ROOT="$EVIDENCE_ROOT" \
  SIMANY_ROOT="$CODE_ROOT" \
  PYTHONPATH="$OPENPI_CLIENT_SRC:$CODE_ROOT" \
  PYTHONNOUSERSITE=1 \
  "$PYTHON" -m robo.eval.e4_camera_scorer_gate \
    --qualifier-preflight-only \
    --gate-id "$E4_CAMERA_GATE_ID" \
    --cpu-freeze-id "$CPU_FREEZE_ID" \
    --cpu-producer-commit "$CPU_PRODUCER_COMMIT" \
    --expected-cpu-gate-sha256 "$CPU_GATE_SHA256" \
    --expected-code-commit "$CODE_COMMIT" \
    --menagerie-root "$MENAGERIE_ROOT" \
    --expected-menagerie-commit "$MENAGERIE_SOURCE_COMMIT"
then
  echo "CPU qualifier preflight failed; refusing to call sbatch" >&2
  exit 4
fi

QUALIFIER_GATE="$QUALIFIER_PREFLIGHT_ROOT/gate.json"
QUALIFIER_SEAL="$QUALIFIER_PREFLIGHT_ROOT/seal.json"
"$PYTHON" - "$QUALIFIER_GATE" "$QUALIFIER_SEAL" "$CODE_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

gate_path, seal_path = (Path(value) for value in sys.argv[1:3])
expected_commit = sys.argv[3]
for path in (gate_path, seal_path):
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise SystemExit(f"invalid qualifier preflight artifact: {path}")
gate_bytes = gate_path.read_bytes()
gate = json.loads(gate_bytes)
seal = json.loads(seal_path.read_text(encoding="utf-8"))
identity = seal.get("members", {}).get("gate.json", {})
if identity != {
    "sha256": hashlib.sha256(gate_bytes).hexdigest(),
    "size_bytes": len(gate_bytes),
}:
    raise SystemExit("qualifier preflight gate is not sealed")
if not (
    gate.get("manifest_kind") == "e4_cpu_qualifier_preflight_gate"
    and gate.get("camera_validator_code", {}).get("commit") == expected_commit
    and gate.get("cell_coverage", {}).get("cells") == 40
    and gate.get("preflight_cells_passed") == 40
    and gate.get("preflight_cells_failed") == 0
    and gate.get("qualifier_margin_cells_passed") == 40
    and gate.get("qualifier_margin_cells_failed") == 0
    and gate.get("settle_contract_cells_passed") == 40
    and gate.get("settle_contract_cells_failed") == 0
    and gate.get("reset_contract_cells_passed") == 40
    and gate.get("reset_contract_cells_failed") == 0
    and gate.get("camera_job_submission_allowed") is True
    and gate.get("real_policy_infra_smoke_allowed") is False
    and gate.get("large_rollout_launch_allowed") is False
    and gate.get("paper_ready") is False
    and gate.get("renderer_constructed") is False
    and gate.get("gpu_work_executed") is False
):
    raise SystemExit("qualifier preflight is not a 40/40 camera-only authorization")
PY
QUALIFIER_GATE_SHA256=$(sha256sum "$QUALIFIER_GATE" | awk '{print $1}')

mkdir -p "$EVIDENCE_ROOT/outputs/icra2027/submissions"
mkdir "$SUBMISSION_DIR" "$LOG_DIR" "$RECEIPT_DIR"

receipt="$RECEIPT_DIR/01-camera-workspace-scorer.stdout"
export_spec="E4_CAMERA_GATE_ID=$E4_CAMERA_GATE_ID,E4_CAMERA_CODE_COMMIT=$CODE_COMMIT,E4_SUBMIT_USER=$SUBMIT_USER,E4_QUALIFIER_PREFLIGHT_GATE_SHA256=$QUALIFIER_GATE_SHA256"
sbatch_rc=0
umask 077
"$SBATCH_BIN" --parsable --no-requeue \
  --account="$ACCOUNT" --job-name=icra-e4-camera-gate \
  --partition=batch --qos=normal --nodelist="$GPU_NODE" \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --gpus="$GPU_GRES" \
  --output="$LOG_DIR/camera-gate-%j.out" \
  --error="$LOG_DIR/camera-gate-%j.err" \
  --export="$export_spec" \
  "$LAUNCHER" > "$receipt" || sbatch_rc=$?
sync -f "$receipt"
chmod 0444 "$receipt"
sync -f "$RECEIPT_DIR"
if [[ $sbatch_rc -ne 0 ]]; then
  echo "sbatch failed rc=$sbatch_rc; receipt=$receipt" >&2
  exit "$sbatch_rc"
fi
mapfile -t receipt_lines < "$receipt"
[[ ${#receipt_lines[@]} -eq 1 && "${receipt_lines[0]}" =~ ^([0-9]+)(\;[A-Za-z0-9._-]+)?$ ]] || {
  echo "sbatch did not return one ordinary numeric job ID" >&2; exit 2;
}
job_id=${BASH_REMATCH[1]}
receipt_sha=$(sha256sum "$receipt" | awk '{print $1}')
ledger_tmp="$SUBMISSION_DIR/.jobs.tsv.$$"
submitted_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
(
  umask 077
  printf 'sequence\tsubmitted_utc\tstage\tjob_id\tdependency\tprofile\tcpus\tmem\ttime\tpartition\tqos\taccount\tsubmit_user\tnodelist\tgres\tcode_commit\tcpu_freeze_id\tcpu_gate_sha256\tmenagerie_source_commit\tqualifier_preflight_gate\tqualifier_preflight_gate_sha256\tlauncher\tsbatch_receipt\tsbatch_receipt_sha256\n'
  printf '1\t%s\tcamera-workspace-scorer\t%s\t-\tgcp-a100\t4\t32G\t01:00:00\tbatch\tnormal\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$submitted_utc" "$job_id" "$ACCOUNT" "$SUBMIT_USER" "$GPU_NODE" "$GPU_GRES" \
    "$CODE_COMMIT" "$CPU_FREEZE_ID" "$CPU_GATE_SHA256" "$MENAGERIE_SOURCE_COMMIT" \
    "${QUALIFIER_GATE#"$EVIDENCE_ROOT"/}" "$QUALIFIER_GATE_SHA256" \
    "${LAUNCHER#"$CODE_ROOT"/}" "${receipt#"$SUBMISSION_DIR"/}" "$receipt_sha"
) > "$ledger_tmp"
sync -f "$ledger_tmp"
mv -T -- "$ledger_tmp" "$LEDGER"
chmod 0444 "$LEDGER"
sync -f "$SUBMISSION_DIR"
ledger_sha=$(sha256sum "$LEDGER" | awk '{print $1}')
echo "submitted 1 ordinary GPU job (no array)"
echo "job_id=$job_id"
echo "ledger=$LEDGER"
echo "ledger_sha256=$ledger_sha"
