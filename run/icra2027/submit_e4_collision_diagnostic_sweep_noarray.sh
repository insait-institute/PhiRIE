#!/usr/bin/env bash
# Enqueue the exact 65-job E4 collision diagnostic DAG as ordinary CPU jobs:
# 16 common builds, 32 policy evaluations, 16 afterany comparisons, 1 aggregate.
set -euo pipefail

CODE_ROOT=${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot
EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}
PYTHON=${SIMANY_ROOT:-$PWD}/.venv/bin/python
LAUNCHER="$CODE_ROOT/run/slurm/icra2027_e4_collision_diagnostic_cpu.sbatch"
SWEEP_SCRIPT="$CODE_ROOT/run/icra2027/e4_collision_diagnostic_sweep.py"
EXPECTED_E3_ROOT=outputs/icra2027/icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic
ACCOUNT=${SLURM_ACCOUNT:-phirie}
CPU_NODELIST='sof1-h200-[0-7]'
SETFACL=/usr/bin/setfacl
GETFACL=/usr/bin/getfacl
SUBMIT_USER=$(id -un)

if [[ -v SBATCH_BIN || -v SACCTMGR_BIN ]]; then
  [[ "${E4_TEST_TOOL_OVERRIDES:-}" == 1 ]] || {
    echo "SBATCH_BIN/SACCTMGR_BIN overrides are test-only" >&2
    exit 2
  }
fi
SBATCH_BIN=${SBATCH_BIN:-sbatch}
SACCTMGR_BIN=${SACCTMGR_BIN:-sacctmgr}

: "${E4_REPAIR_SWEEP_ID:?set a fresh E4_REPAIR_SWEEP_ID before submitting}"
[[ "$E4_REPAIR_SWEEP_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe sweep ID: $E4_REPAIR_SWEEP_ID" >&2
  exit 2
}
[[ "$SUBMIT_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe submit user" >&2
  exit 2
}

for array_variable in \
  SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP \
  SBATCH_ARRAY_INX SBATCH_ARRAY
do
  if [[ -v "$array_variable" ]]; then
    echo "diagnostic submitter refuses Slurm array context: $array_variable is set" >&2
    exit 2
  fi
done

for executable in "$SBATCH_BIN" "$SACCTMGR_BIN"; do
  command -v "$executable" >/dev/null || {
    echo "missing executable: $executable" >&2
    exit 2
  }
done
[[ -x "$PYTHON" && -x "$SETFACL" && -x "$GETFACL" ]] || {
  echo "missing Python or ACL tools" >&2
  exit 2
}
for source in "$LAUNCHER" "$SWEEP_SCRIPT"; do
  [[ -s "$source" && ! -L "$source" ]] || {
    echo "missing, empty, or symlinked diagnostic source: $source" >&2
    exit 2
  }
done
[[ -d "$EVIDENCE_ROOT" && ! -L "$EVIDENCE_ROOT" && "$CODE_ROOT" != "$EVIDENCE_ROOT" ]] || {
  echo "invalid dual-root binding" >&2
  exit 2
}
[[ -d "$EVIDENCE_ROOT/$EXPECTED_E3_ROOT" && ! -L "$EVIDENCE_ROOT/$EXPECTED_E3_ROOT" ]] || {
  echo "missing sealed E3 root" >&2
  exit 2
}

CODE_COMMIT=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$CODE_COMMIT" =~ ^[0-9a-f]{40}$ \
  && "$(git -C "$CODE_ROOT" rev-parse --show-toplevel)" == "$CODE_ROOT" ]] || {
  echo "invalid isolated code snapshot" >&2
  exit 2
}
if [[ -n "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "refusing submission from a dirty code worktree" >&2
  git -C "$CODE_ROOT" status --short >&2
  exit 2
fi

PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$CODE_ROOT" \
  SIMANY_EVIDENCE_ROOT="$EVIDENCE_ROOT" \
  "$PYTHON" - "$SWEEP_SCRIPT" "$CODE_ROOT" "$EVIDENCE_ROOT" \
  "$EXPECTED_E3_ROOT" <<'PY'
import importlib.util
import sys
from pathlib import Path

from robo.sim import export_mjcf

script, code_root, evidence_root, e3_root = sys.argv[1:]
code_root, evidence_root, e3_root = map(Path, (code_root, evidence_root, e3_root))
spec = importlib.util.spec_from_file_location("sealed_e4_collision_diagnostic_sweep", script)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load diagnostic-sweep script")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if module.CODE_ROOT != code_root or module.EXPECTED_EVIDENCE_ROOT != evidence_root:
    raise SystemExit("diagnostic-sweep dual-root constants differ")
if module.EXPECTED_E3_ROOT != e3_root:
    raise SystemExit("diagnostic-sweep E3 root differs")
if len(module.VARIANT_IDS) != 16 or tuple(module.POLICIES) != ("A0", "A4"):
    raise SystemExit("diagnostic-sweep frozen axes differ")
for name in (
    "load_room_diagnostic_spec",
    "build_common_room_static_package",
    "load_common_room_static_package",
):
    if not callable(getattr(export_mjcf, name, None)):
        raise SystemExit(f"exporter lacks diagnostic API: {name}")
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

OUTPUTS_ROOT="$EVIDENCE_ROOT/outputs/icra2027"
SWEEP_ROOT="$OUTPUTS_ROOT/$E4_REPAIR_SWEEP_ID"
SUBMISSIONS_ROOT="$OUTPUTS_ROOT/submissions"
SUBMISSION_DIR="$SUBMISSIONS_ROOT/${E4_REPAIR_SWEEP_ID}-collision-diagnostic-sweep"
LOG_DIR="$SUBMISSION_DIR/logs"
RECEIPT_DIR="$SUBMISSION_DIR/sbatch_receipts"
RUNTIME_DIR="$SUBMISSION_DIR/runtime"
for fresh_path in "$SWEEP_ROOT" "$SUBMISSION_DIR"; do
  [[ ! -e "$fresh_path" && ! -L "$fresh_path" ]] || {
    echo "refusing to reuse diagnostic path: $fresh_path" >&2
    exit 2
  }
done

umask 077
mkdir -p "$SUBMISSIONS_ROOT"
[[ -d "$SUBMISSIONS_ROOT" && ! -L "$SUBMISSIONS_ROOT" ]] || {
  echo "missing or symlinked submissions root" >&2
  exit 2
}

PRIVATE_DIRS=()
make_private_directory() {
  local directory=$1
  mkdir "$directory"
  "$SETFACL" -k -- "$directory"
  chmod 00700 "$directory"
  [[ -d "$directory" && ! -L "$directory" \
    && "$(stat -c '%a' -- "$directory")" == 700 \
    && "$("$GETFACL" -cp -- "$directory")" != *"default:"* ]] || {
    echo "directory is not private ACL-free mode 700: $directory" >&2
    exit 2
  }
  PRIVATE_DIRS+=("$directory")
}

make_private_directory "$SUBMISSION_DIR"
make_private_directory "$LOG_DIR"
make_private_directory "$RECEIPT_DIR"
make_private_directory "$RUNTIME_DIR"
make_private_directory "$SWEEP_ROOT"
make_private_directory "$SWEEP_ROOT/variants"

VARIANTS=(
  3db-hraw-pfail
  3db-hraw-pdemote
  3db-hclip-pfail
  3db-hclip-pdemote
  d755-r20-zextent-hraw
  d755-r20-zextent-hclip
  d755-r20-zmean-hraw
  d755-r20-zmean-hclip
  d755-r25-zextent-hraw
  d755-r25-zextent-hclip
  d755-r25-zmean-hraw
  d755-r25-zmean-hclip
  d755-r30-zextent-hraw
  d755-r30-zextent-hclip
  d755-r30-zmean-hraw
  d755-r30-zmean-hclip
)
for variant in "${VARIANTS[@]}"; do
  variant_root="$SWEEP_ROOT/variants/$variant"
  make_private_directory "$variant_root"
  make_private_directory "$variant_root/construction_variants"
  make_private_directory "$variant_root/construction_variants/A0"
  make_private_directory "$variant_root/construction_variants/A4"
  make_private_directory "$variant_root/policy_evals"
done
for directory in "${PRIVATE_DIRS[@]}"; do sync -f "$directory"; done
sync -f "$SUBMISSIONS_ROOT"

SEQUENCE=0
LEDGER="$SUBMISSION_DIR/jobs.tsv"
LEDGER_TEMP=""
on_exit() {
  local rc=$?
  if [[ -n "$LEDGER_TEMP" && -f "$LEDGER_TEMP" && ! -L "$LEDGER_TEMP" ]]; then
    local partial="$SUBMISSION_DIR/jobs.tsv.incomplete.${SEQUENCE}.$$"
    mv -T -- "$LEDGER_TEMP" "$partial"
    chmod 0444 "$partial"
    sync -f "$partial"
  fi
  if [[ -f "$LEDGER" && ! -L "$LEDGER" ]]; then
    chmod 0444 "$LEDGER"
    sync -f "$LEDGER"
    sync -f "$SUBMISSION_DIR"
  fi
  if [[ $rc -ne 0 ]]; then
    echo "diagnostic submission stopped after $SEQUENCE recorded ordinary jobs" >&2
    echo "partial ledger: $LEDGER" >&2
    echo "receipts: $RECEIPT_DIR" >&2
  fi
}
trap on_exit EXIT

LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.initialize.$$"
(
  umask 077
  printf '%s\n' \
    $'sequence\tsubmitted_utc\tstage\tvariant_id\tscene_id\tpolicy_id\tjob_id\tdependency\tkill_on_invalid_dep\tprofile\tcpus\tmem\ttime\tpartition\tqos\taccount\tsubmit_user\tnodelist\tgres\tcode_commit\te3_root\tsweep_id\tlauncher\targuments\tsbatch_stdout_receipt\tsbatch_stdout_sha256\tsbatch_stderr_receipt\tsbatch_stderr_sha256' \
    > "$LEDGER_TEMP"
)
sync -f "$LEDGER_TEMP"
mv -T -- "$LEDGER_TEMP" "$LEDGER"
sync -f "$SUBMISSION_DIR"
LEDGER_TEMP=""

validate_job_id() {
  local raw=$1
  [[ "$raw" =~ ^([0-9]+)(;[A-Za-z0-9._-]+)?$ ]] || {
    echo "invalid sbatch --parsable receipt: $raw" >&2
    return 2
  }
  SUBMITTED_JOB_ID=${BASH_REMATCH[1]}
}

record_job() {
  local stage=$1 variant=$2 scene=$3 policy=$4 job_id=$5 dependency=$6
  local cpus=$7 mem=$8 time_limit=$9 launcher=${10} arguments=${11}
  local stdout_receipt=${12} stdout_sha=${13} stderr_receipt=${14} stderr_sha=${15}
  local kill_on_invalid_dep=no
  [[ "$dependency" == - ]] || kill_on_invalid_dep=yes
  local submitted_utc
  submitted_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  SEQUENCE=$((SEQUENCE + 1))
  LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.${SEQUENCE}.$$"
  cp -- "$LEDGER" "$LEDGER_TEMP"
  {
    printf '%s\t' \
      "$SEQUENCE" "$submitted_utc" "$stage" "$variant" "$scene" "$policy" \
      "$job_id" "$dependency" "$kill_on_invalid_dep" sof1-cpu \
      "$cpus" "$mem" "$time_limit" batch normal "$ACCOUNT" "$SUBMIT_USER" \
      "$CPU_NODELIST" none "$CODE_COMMIT" "$EXPECTED_E3_ROOT" \
      "$E4_REPAIR_SWEEP_ID" "${launcher#"$CODE_ROOT"/}" "$arguments" \
      "$stdout_receipt" "$stdout_sha" "$stderr_receipt"
    printf '%s\n' "$stderr_sha"
  } >> "$LEDGER_TEMP"
  sync -f "$LEDGER_TEMP"
  mv -T -- "$LEDGER_TEMP" "$LEDGER"
  sync -f "$SUBMISSION_DIR"
  LEDGER_TEMP=""
}

submit_job() {
  local stage=$1 variant=$2 scene=$3 policy=$4 dependency=$5
  local cpus=$6 mem=$7 time_limit=$8 job_name=$9
  shift 9
  local -a launcher_args=("$@") dependency_args=()
  if [[ "$dependency" != - ]]; then
    [[ "$dependency" =~ ^after(ok|any):[0-9]+(:[0-9]+)*(,after(ok|any):[0-9]+(:[0-9]+)*)*$ ]] || {
      echo "unsafe dependency: $dependency" >&2
      return 2
    }
    dependency_args=(--dependency="$dependency" --kill-on-invalid-dep=yes)
  fi
  local next_sequence=$((SEQUENCE + 1))
  local receipt_stem stdout_receipt stderr_receipt receipt
  printf -v receipt_stem '%02d-%s-%s-%s-%s' \
    "$next_sequence" "$stage" "$variant" "$scene" "$policy"
  stdout_receipt="$RECEIPT_DIR/${receipt_stem}.stdout"
  stderr_receipt="$RECEIPT_DIR/${receipt_stem}.stderr"
  for receipt in "$stdout_receipt" "$stderr_receipt"; do
    [[ ! -e "$receipt" && ! -L "$receipt" ]] || {
      echo "refusing to overwrite receipt: $receipt" >&2
      return 2
    }
  done
  local export_spec
  export_spec="E4_REPAIR_SWEEP_ID=$E4_REPAIR_SWEEP_ID,E4_CODE_COMMIT=$CODE_COMMIT,E4_E3_ROOT=$EXPECTED_E3_ROOT,E4_SUBMIT_USER=$SUBMIT_USER"
  local sbatch_rc=0
  "$SBATCH_BIN" --parsable --no-requeue \
    --account="$ACCOUNT" --job-name="$job_name" \
    --partition=batch --qos=normal --nodelist="$CPU_NODELIST" \
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem" --time="$time_limit" \
    --output="$LOG_DIR/${stage}-${variant}-${policy}-%j.out" \
    --error="$LOG_DIR/${stage}-${variant}-${policy}-%j.err" \
    --export="$export_spec" "${dependency_args[@]}" \
    "$LAUNCHER" "${launcher_args[@]}" \
    > "$stdout_receipt" 2> "$stderr_receipt" || sbatch_rc=$?
  sync -f "$stdout_receipt"
  sync -f "$stderr_receipt"
  chmod 0444 "$stdout_receipt" "$stderr_receipt"
  sync -f "$RECEIPT_DIR"
  if [[ $sbatch_rc -ne 0 ]]; then
    echo "sbatch failed for sequence $next_sequence (rc=$sbatch_rc)" >&2
    return "$sbatch_rc"
  fi
  local -a receipt_lines=()
  mapfile -t receipt_lines < "$stdout_receipt"
  [[ ${#receipt_lines[@]} -eq 1 ]] || {
    echo "sbatch receipt must contain exactly one line" >&2
    return 2
  }
  validate_job_id "${receipt_lines[0]}"
  local stdout_sha stderr_sha rendered_args
  stdout_sha=$(sha256sum "$stdout_receipt" | awk '{print $1}')
  stderr_sha=$(sha256sum "$stderr_receipt" | awk '{print $1}')
  printf -v rendered_args '%q ' "${launcher_args[@]}"
  record_job "$stage" "$variant" "$scene" "$policy" "$SUBMITTED_JOB_ID" \
    "$dependency" "$cpus" "$mem" "$time_limit" "$LAUNCHER" \
    "${rendered_args% }" "${stdout_receipt#"$SUBMISSION_DIR"/}" "$stdout_sha" \
    "${stderr_receipt#"$SUBMISSION_DIR"/}" "$stderr_sha"
}

declare -A BUILD_JOBS EVAL_A0_JOBS EVAL_A4_JOBS COMPARE_JOBS

# 1-16: each job serially materializes fresh A0/A4 factories and creates one
# sealed static package. No failed repair output is used as an input.
for variant in "${VARIANTS[@]}"; do
  if [[ "$variant" == 3db-* ]]; then scene=3db0a1c8f3; else scene=d755b3d9d8; fi
  submit_job build-common "$variant" "$scene" paired - \
    16 96G 04:00:00 icra-e4-diag-build build-common "$variant"
  BUILD_JOBS["$variant"]=$SUBMITTED_JOB_ID
done

# 17-48: both policy evaluations consume exactly their variant's successful
# common package. They never depend on another variant or policy evaluation.
for variant in "${VARIANTS[@]}"; do
  if [[ "$variant" == 3db-* ]]; then scene=3db0a1c8f3; else scene=d755b3d9d8; fi
  dependency="afterok:${BUILD_JOBS[$variant]}"
  submit_job eval-policy "$variant" "$scene" A0 "$dependency" \
    4 64G 01:00:00 icra-e4-diag-eval eval-policy "$variant" A0
  EVAL_A0_JOBS["$variant"]=$SUBMITTED_JOB_ID
  submit_job eval-policy "$variant" "$scene" A4 "$dependency" \
    4 64G 01:00:00 icra-e4-diag-eval eval-policy "$variant" A4
  EVAL_A4_JOBS["$variant"]=$SUBMITTED_JOB_ID
done

# 49-64: comparisons are afterany so policy failures become sealed evidence.
for variant in "${VARIANTS[@]}"; do
  if [[ "$variant" == 3db-* ]]; then scene=3db0a1c8f3; else scene=d755b3d9d8; fi
  dependency="afterany:${EVAL_A0_JOBS[$variant]}:${EVAL_A4_JOBS[$variant]}"
  submit_job compare-pair "$variant" "$scene" paired "$dependency" \
    2 16G 00:30:00 icra-e4-diag-compare compare-pair "$variant"
  COMPARE_JOBS["$variant"]=$SUBMITTED_JOB_ID
done

# 65: the diagnostics-only aggregate always runs after every comparison and
# is hard-coded by the runner to keep GPU/large-rollout/paper claims false.
compare_ids=()
for variant in "${VARIANTS[@]}"; do compare_ids+=("${COMPARE_JOBS[$variant]}"); done
aggregate_dependency=$(IFS=:; printf 'afterany:%s' "${compare_ids[*]}")
submit_job aggregate - - - "$aggregate_dependency" \
  2 16G 00:30:00 icra-e4-diag-aggregate aggregate
AGGREGATE_JOB=$SUBMITTED_JOB_ID

[[ $SEQUENCE -eq 65 ]] || {
  echo "internal error: expected 65 jobs, recorded $SEQUENCE" >&2
  exit 2
}
chmod 0444 "$LEDGER"
sync -f "$LEDGER"
sync -f "$SUBMISSION_DIR"
ledger_sha256=$(sha256sum "$LEDGER" | awk '{print $1}')
echo "submitted 65 ordinary CPU diagnostic jobs (no Slurm arrays, no GPU)"
echo "common_build_jobs=16"
echo "policy_eval_jobs=32"
echo "pair_compare_jobs=16"
echo "aggregate_job=$AGGREGATE_JOB"
echo "ledger=$LEDGER"
echo "ledger_sha256=$ledger_sha256"
