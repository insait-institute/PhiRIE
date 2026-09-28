#!/usr/bin/env bash
# Enqueue the complete E4 candidate-screen DAG as 21 ordinary CPU jobs.
#
#   10 materialize (five scenes x A0/A4)
#      -> 5 paired prepare-scene
#      -> 5 qualify-scene
#      -> 1 afterany aggregate
#
# The aggregate is diagnostic and deliberately runs after any qualifier
# outcome.  This script never requests a GPU and never submits a Slurm array.
set -euo pipefail

CODE_ROOT=${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot
EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}
PYTHON=${SIMANY_ROOT:-$PWD}/.venv/bin/python
LAUNCHER="$CODE_ROOT/run/slurm/icra2027_e4_candidate_cpu.sbatch"
MENAGERIE_ROOT="$CODE_ROOT/third_party/mujoco_menagerie"
EXPECTED_E3_ROOT=outputs/icra2027/icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic
EXPECTED_MENAGERIE_COMMIT=71f066ad0be9cd271f7ed58c030243ef157af9f4
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

: "${E4_SCREEN_ID:?set a fresh E4_SCREEN_ID before submitting}"
[[ "$E4_SCREEN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_SCREEN_ID: $E4_SCREEN_ID" >&2
  exit 2
}
[[ "$SUBMIT_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe submit user: $SUBMIT_USER" >&2
  exit 2
}

for array_variable in \
  SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP \
  SBATCH_ARRAY_INX SBATCH_ARRAY
do
  if [[ -v "$array_variable" ]]; then
    echo "E4 candidate submitter refuses Slurm array context: $array_variable is set" >&2
    exit 2
  fi
done

command -v "$SBATCH_BIN" >/dev/null || {
  echo "sbatch executable not found: $SBATCH_BIN" >&2
  exit 2
}
command -v "$SACCTMGR_BIN" >/dev/null || {
  echo "sacctmgr executable not found: $SACCTMGR_BIN" >&2
  exit 2
}
[[ -x "$SETFACL" && -x "$GETFACL" ]] || {
  echo "setfacl/getfacl are required to seal private runtime directories" >&2
  exit 2
}
[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
[[ -s "$LAUNCHER" && ! -L "$LAUNCHER" ]] || {
  echo "missing, empty, or symlinked candidate launcher: $LAUNCHER" >&2
  exit 2
}
[[ -d "$EVIDENCE_ROOT" && ! -L "$EVIDENCE_ROOT" ]] || {
  echo "missing or symlinked evidence root: $EVIDENCE_ROOT" >&2
  exit 2
}
[[ "$CODE_ROOT" != "$EVIDENCE_ROOT" ]] || {
  echo "code and evidence roots must be distinct" >&2
  exit 2
}
[[ -d "$EVIDENCE_ROOT/$EXPECTED_E3_ROOT" \
  && ! -L "$EVIDENCE_ROOT/$EXPECTED_E3_ROOT" ]] || {
  echo "missing or symlinked sealed E3 root: $EVIDENCE_ROOT/$EXPECTED_E3_ROOT" >&2
  exit 2
}
[[ -d "$MENAGERIE_ROOT" && ! -L "$MENAGERIE_ROOT" ]] || {
  echo "missing or symlinked worktree-local Menagerie closure: $MENAGERIE_ROOT" >&2
  exit 2
}

CODE_COMMIT=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "git did not return a full lowercase commit" >&2
  exit 2
}
[[ "$(git -C "$CODE_ROOT" rev-parse --show-toplevel)" == "$CODE_ROOT" ]] || {
  echo "submitter is not using the isolated E4 worktree" >&2
  exit 2
}
if [[ -n "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "refusing candidate submission from a dirty code worktree" >&2
  git -C "$CODE_ROOT" status --short >&2
  exit 2
fi

# Import through the exact isolated worktree before any scheduler mutation.
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$CODE_ROOT" \
  SIMANY_EVIDENCE_ROOT="$EVIDENCE_ROOT" \
  "$PYTHON" - "$CODE_ROOT" "$EVIDENCE_ROOT" "$EXPECTED_E3_ROOT" \
  "$MENAGERIE_ROOT" "$EXPECTED_MENAGERIE_COMMIT" <<'PY'
import sys
from pathlib import Path

import robo.eval.e4_candidate_screen as candidate

code_root = Path(sys.argv[1])
evidence_root = Path(sys.argv[2])
expected_e3_root = Path(sys.argv[3])
menagerie_root = Path(sys.argv[4])
menagerie_commit = sys.argv[5]
observed = Path(candidate.__file__).resolve(strict=True)
if code_root not in observed.parents:
    raise SystemExit(f"wrong candidate-screen import origin: {observed}")
if candidate.CODE_ROOT != code_root:
    raise SystemExit("candidate-screen CODE_ROOT differs from isolated worktree")
if candidate.EXPECTED_EVIDENCE_ROOT != evidence_root:
    raise SystemExit("candidate-screen evidence root differs")
if candidate.EXPECTED_E3_ROOT != expected_e3_root:
    raise SystemExit("candidate-screen E3 root differs")
if candidate.EXPECTED_MENAGERIE_ROOT != menagerie_root:
    raise SystemExit("candidate-screen Menagerie root differs")
if candidate.EXPECTED_MENAGERIE_COMMIT != menagerie_commit:
    raise SystemExit("candidate-screen Menagerie commit differs")
if tuple(candidate.SCENE_IDS) != (
    "3db0a1c8f3", "27dd4da69e", "d755b3d9d8", "acd95847c5", "40aec5fffa",
):
    raise SystemExit("candidate-screen scene roster differs")
PY

# Refuse scheduler-side QoS rewriting before enqueueing any job.
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

EXPERIMENT_ROOT="$EVIDENCE_ROOT/outputs/icra2027/$E4_SCREEN_ID"
SUBMISSIONS_ROOT="$EVIDENCE_ROOT/outputs/icra2027/submissions"
SUBMISSION_DIR="$SUBMISSIONS_ROOT/${E4_SCREEN_ID}-candidate-screen"
LOG_DIR="$SUBMISSION_DIR/logs"
RECEIPT_DIR="$SUBMISSION_DIR/sbatch_receipts"
RUNTIME_DIR="$SUBMISSION_DIR/runtime"
LEDGER="$SUBMISSION_DIR/jobs.tsv"
[[ ! -e "$EXPERIMENT_ROOT" && ! -L "$EXPERIMENT_ROOT" ]] || {
  echo "refusing to reuse candidate-screen output: $EXPERIMENT_ROOT" >&2
  exit 2
}
[[ ! -e "$SUBMISSION_DIR" && ! -L "$SUBMISSION_DIR" ]] || {
  echo "refusing to reuse candidate submission directory: $SUBMISSION_DIR" >&2
  exit 2
}
umask 077
mkdir -p "$SUBMISSIONS_ROOT"
[[ -d "$SUBMISSIONS_ROOT" && ! -L "$SUBMISSIONS_ROOT" ]] || {
  echo "missing or symlinked submissions root: $SUBMISSIONS_ROOT" >&2
  exit 2
}
mkdir "$SUBMISSION_DIR"
mkdir "$LOG_DIR" "$RECEIPT_DIR" "$RUNTIME_DIR"
for durable_directory in "$SUBMISSION_DIR" "$LOG_DIR" "$RECEIPT_DIR" "$RUNTIME_DIR"; do
  [[ -d "$durable_directory" && ! -L "$durable_directory" ]] || {
    echo "submission directory is missing or symlinked: $durable_directory" >&2
    exit 2
  }
  # Shared filesystems may propagate both setgid and a permissive default ACL.
  # Clear the latter before chmod so children cannot regain group access.
  "$SETFACL" -k -- "$durable_directory"
  chmod 00700 "$durable_directory"
  durable_mode=$(stat -c '%a' -- "$durable_directory")
  durable_acl=$("$GETFACL" -cp -- "$durable_directory")
  [[ -d "$durable_directory" && ! -L "$durable_directory" \
    && "$durable_mode" == 700 \
    && "$durable_acl" != *"default:"* ]] || {
    echo "submission directory is not private ACL-free mode 700: $durable_directory" >&2
    exit 2
  }
done
# Materialize, prepare, and qualify workers publish distinct immutable leaves,
# but their atomic writers intentionally use strict non-idempotent mkdir for
# missing parents. Create every shared parent once before the first sbatch so
# concurrently starting workers cannot race on the common experiment hierarchy.
EXPERIMENT_PARENT_DIRS=(
  "$EXPERIMENT_ROOT"
  "$EXPERIMENT_ROOT/construction_variants"
  "$EXPERIMENT_ROOT/construction_variants/A0"
  "$EXPERIMENT_ROOT/construction_variants/A4"
  "$EXPERIMENT_ROOT/candidate_suites"
  "$EXPERIMENT_ROOT/task_freezes"
  "$EXPERIMENT_ROOT/scene_prepares"
  "$EXPERIMENT_ROOT/scene_qualifiers"
)
for experiment_directory in "${EXPERIMENT_PARENT_DIRS[@]}"; do
  mkdir "$experiment_directory"
  "$SETFACL" -k -- "$experiment_directory"
  chmod 00700 "$experiment_directory"
  experiment_mode=$(stat -c '%a' -- "$experiment_directory")
  experiment_acl=$("$GETFACL" -cp -- "$experiment_directory")
  [[ -d "$experiment_directory" && ! -L "$experiment_directory" \
    && "$experiment_mode" == 700 \
    && "$experiment_acl" != *"default:"* ]] || {
    echo "experiment parent is not private ACL-free mode 700: $experiment_directory" >&2
    exit 2
  }
  sync -f "$experiment_directory"
done
# A job may start as soon as sbatch returns.  Persist the worker's runtime
# parent before the first scheduler mutation so its job-specific plain mkdir
# is both fresh-only and safe under concurrent starts.
sync -f "$RUNTIME_DIR"
sync -f "$SUBMISSION_DIR"
sync -f "$SUBMISSIONS_ROOT"

SEQUENCE=0
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
    echo "candidate submission stopped after $SEQUENCE recorded ordinary jobs" >&2
    echo "partial ledger: $LEDGER" >&2
    echo "receipts: $RECEIPT_DIR" >&2
  fi
}
trap on_exit EXIT

LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.initialize.$$"
(
  umask 077
  printf '%s\n' \
    $'sequence\tsubmitted_utc\tstage\tscene_id\tpolicy_id\tjob_id\tdependency\tkill_on_invalid_dep\tprofile\tcpus\tmem\ttime\tpartition\tqos\taccount\tsubmit_user\tnodelist\tgres\tcode_commit\te3_root\tlauncher\targuments\tsbatch_stdout_receipt\tsbatch_stdout_sha256\tsbatch_stderr_receipt\tsbatch_stderr_sha256' \
    > "$LEDGER_TEMP"
)
sync -f "$LEDGER_TEMP"
mv -T -- "$LEDGER_TEMP" "$LEDGER"
sync -f "$SUBMISSION_DIR"
LEDGER_TEMP=""

record_job() {
  local stage=$1 scene_id=$2 policy_id=$3 job_id=$4 dependency=$5
  local cpus=$6 mem=$7 time_limit=$8 arguments=$9
  local stdout_receipt=${10} stdout_sha=${11} stderr_receipt=${12} stderr_sha=${13}
  local kill_on_invalid_dep=no
  [[ "$dependency" == - ]] || kill_on_invalid_dep=yes
  local submitted_utc
  submitted_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  SEQUENCE=$((SEQUENCE + 1))
  LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.${SEQUENCE}.$$"
  cp -- "$LEDGER" "$LEDGER_TEMP"
  {
    printf '%s\t' \
      "$SEQUENCE" "$submitted_utc" "$stage" "$scene_id" "$policy_id" \
      "$job_id" "$dependency" "$kill_on_invalid_dep" sof1-cpu "$cpus" \
      "$mem" "$time_limit" batch normal "$ACCOUNT" "$SUBMIT_USER" \
      "$CPU_NODELIST" none "$CODE_COMMIT" "$EXPECTED_E3_ROOT" \
      "${LAUNCHER#"$CODE_ROOT"/}" "$arguments" "$stdout_receipt" \
      "$stdout_sha" "$stderr_receipt"
    printf '%s\n' "$stderr_sha"
  } >> "$LEDGER_TEMP"
  sync -f "$LEDGER_TEMP"
  mv -T -- "$LEDGER_TEMP" "$LEDGER"
  sync -f "$SUBMISSION_DIR"
  LEDGER_TEMP=""
}

validate_job_id() {
  local raw=$1
  [[ "$raw" =~ ^([0-9]+)(\;[A-Za-z0-9._-]+)?$ ]] || {
    echo "sbatch did not return one ordinary numeric job ID: $raw" >&2
    return 2
  }
  local candidate=${BASH_REMATCH[1]}
  if awk -F '\t' -v id="$candidate" \
    'NR > 1 && $6 == id {found=1} END {exit !found}' "$LEDGER"; then
    echo "duplicate Slurm job ID returned: $candidate" >&2
    return 2
  fi
  SUBMITTED_JOB_ID=$candidate
}

submit_job() {
  local stage=$1 scene_id=$2 policy_id=$3 dependency=$4
  local cpus=$5 mem=$6 time_limit=$7
  shift 7
  local -a launcher_args=("$@")
  local -a dependency_args=()
  if [[ "$dependency" != - ]]; then
    [[ "$dependency" =~ ^after(ok|any):[0-9]+(:[0-9]+)*$ ]] || {
      echo "unsafe dependency: $dependency" >&2
      return 2
    }
    dependency_args=(--dependency="$dependency" --kill-on-invalid-dep=yes)
  fi

  local next_sequence=$((SEQUENCE + 1))
  local log_tag="${stage}-${scene_id}-${policy_id}"
  local receipt_stem stdout_receipt stderr_receipt receipt
  printf -v receipt_stem '%02d-%s-%s-%s' \
    "$next_sequence" "$stage" "$scene_id" "$policy_id"
  stdout_receipt="$RECEIPT_DIR/${receipt_stem}.stdout"
  stderr_receipt="$RECEIPT_DIR/${receipt_stem}.stderr"
  for receipt in "$stdout_receipt" "$stderr_receipt"; do
    [[ ! -e "$receipt" && ! -L "$receipt" ]] || {
      echo "refusing to overwrite sbatch receipt: $receipt" >&2
      return 2
    }
  done

  # Export only four validated bindings. Slurm supplies its own SLURM_*
  # variables; caller Python, CUDA, and SIMANY state is never inherited.
  local export_spec="E4_SCREEN_ID=$E4_SCREEN_ID,E4_CODE_COMMIT=$CODE_COMMIT,E4_E3_ROOT=$EXPECTED_E3_ROOT,E4_SUBMIT_USER=$SUBMIT_USER"
  local sbatch_rc=0
  umask 077
  "$SBATCH_BIN" --parsable --no-requeue \
    --account="$ACCOUNT" \
    --job-name="icra-e4-candidate-$stage" \
    --partition=batch --qos=normal --nodelist="$CPU_NODELIST" \
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem" --time="$time_limit" \
    --output="$LOG_DIR/${log_tag}-%j.out" \
    --error="$LOG_DIR/${log_tag}-%j.err" \
    --export="$export_spec" \
    "${dependency_args[@]}" \
    "$LAUNCHER" "${launcher_args[@]}" \
    > "$stdout_receipt" 2> "$stderr_receipt" || sbatch_rc=$?
  sync -f "$stdout_receipt"
  sync -f "$stderr_receipt"
  chmod 0444 "$stdout_receipt" "$stderr_receipt"
  sync -f "$RECEIPT_DIR"
  if [[ $sbatch_rc -ne 0 ]]; then
    echo "sbatch failed for sequence $next_sequence (rc=$sbatch_rc)" >&2
    echo "stdout receipt: $stdout_receipt" >&2
    echo "stderr receipt: $stderr_receipt" >&2
    return "$sbatch_rc"
  fi

  local -a receipt_lines=()
  mapfile -t receipt_lines < "$stdout_receipt"
  [[ ${#receipt_lines[@]} -eq 1 ]] || {
    echo "sbatch stdout receipt must contain exactly one line: $stdout_receipt" >&2
    return 2
  }
  validate_job_id "${receipt_lines[0]}"
  local stdout_sha stderr_sha rendered_args
  stdout_sha=$(sha256sum "$stdout_receipt" | awk '{print $1}')
  stderr_sha=$(sha256sum "$stderr_receipt" | awk '{print $1}')
  printf -v rendered_args '%q ' "${launcher_args[@]}"
  record_job "$stage" "$scene_id" "$policy_id" "$SUBMITTED_JOB_ID" \
    "$dependency" "$cpus" "$mem" "$time_limit" "${rendered_args% }" \
    "${stdout_receipt#"$SUBMISSION_DIR"/}" "$stdout_sha" \
    "${stderr_receipt#"$SUBMISSION_DIR"/}" "$stderr_sha"
}

SCENES=(3db0a1c8f3 27dd4da69e d755b3d9d8 acd95847c5 40aec5fffa)
MATERIALIZE_A0_JOBS=()
MATERIALIZE_A4_JOBS=()
PREPARE_JOBS=()
QUALIFY_JOBS=()

# First enqueue all ten independent construction jobs.
for scene_id in "${SCENES[@]}"; do
  submit_job materialize "$scene_id" A0 - 8 64G 01:00:00 \
    materialize "$scene_id" A0
  MATERIALIZE_A0_JOBS+=("$SUBMITTED_JOB_ID")
  submit_job materialize "$scene_id" A4 - 8 64G 01:00:00 \
    materialize "$scene_id" A4
  MATERIALIZE_A4_JOBS+=("$SUBMITTED_JOB_ID")
done

# One paired preparation per scene waits for exactly its A0/A4 constructions.
for index in "${!SCENES[@]}"; do
  scene_id=${SCENES[$index]}
  dependency="afterok:${MATERIALIZE_A0_JOBS[$index]}:${MATERIALIZE_A4_JOBS[$index]}"
  submit_job prepare "$scene_id" paired "$dependency" 16 96G 04:00:00 \
    prepare-scene "$scene_id"
  PREPARE_JOBS+=("$SUBMITTED_JOB_ID")
done

# Each reset/physics qualifier waits only for its own prepared scene.
for index in "${!SCENES[@]}"; do
  scene_id=${SCENES[$index]}
  dependency="afterok:${PREPARE_JOBS[$index]}"
  submit_job qualify "$scene_id" paired "$dependency" 4 32G 01:00:00 \
    qualify-scene "$scene_id"
  QUALIFY_JOBS+=("$SUBMITTED_JOB_ID")
done

# The diagnostic aggregate must run even when one or more candidates fail.
qualifier_ids=$(IFS=:; printf '%s' "${QUALIFY_JOBS[*]}")
submit_job aggregate - - "afterany:$qualifier_ids" 4 32G 00:30:00 aggregate
AGGREGATE_JOB=$SUBMITTED_JOB_ID

[[ $SEQUENCE -eq 21 ]] || {
  echo "internal error: expected 21 jobs, recorded $SEQUENCE" >&2
  exit 2
}
chmod 0444 "$LEDGER"
sync -f "$LEDGER"
sync -f "$SUBMISSION_DIR"
ledger_sha256=$(sha256sum "$LEDGER" | awk '{print $1}')
echo "submitted 21 ordinary CPU jobs at once (no Slurm arrays, no GPU)"
echo "aggregate_job=$AGGREGATE_JOB"
echo "aggregate_dependency=afterany:$qualifier_ids"
echo "ledger=$LEDGER"
echo "ledger_sha256=$ledger_sha256"
