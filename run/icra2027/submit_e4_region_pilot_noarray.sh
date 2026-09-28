#!/usr/bin/env bash
# Submit seven ordinary CPU jobs at once; this script never submits a GPU job.
#
# DAG:
#   4 x materialize (A0/A4 x b0a08200c9/825d228aec)
#       -> 2 x per-scene full-room export + MuJoCo test + task generation
#       -> 1 x A4-planned max_tasks=2 freeze + harness resolution dry-run
#
# The final report intentionally leaves gpu_launch_allowed=false because this
# CPU DAG does not perform camera/oracle validation or a real-policy smoke.
set -euo pipefail

CODE_ROOT=${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot
EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}
LAUNCHER="$CODE_ROOT/run/slurm/icra2027_e4_region_cpu.sbatch"
if [[ -v SBATCH_BIN || -v SACCTMGR_BIN ]]; then
  [[ "${E4_TEST_TOOL_OVERRIDES:-}" == 1 ]] || {
    echo "SBATCH_BIN/SACCTMGR_BIN overrides are test-only" >&2
    exit 2
  }
fi
SBATCH_BIN=${SBATCH_BIN:-sbatch}
SACCTMGR_BIN=${SACCTMGR_BIN:-sacctmgr}
ACCOUNT=${SLURM_ACCOUNT:-phirie}
SUBMIT_USER=$(id -un)
if [[ -v E4_ACCOUNT && "$E4_ACCOUNT" != "$ACCOUNT" ]]; then
  echo "E4_ACCOUNT cannot override canonical account=$ACCOUNT" >&2
  exit 2
fi
if [[ -v E4_SUBMIT_USER && "$E4_SUBMIT_USER" != "$SUBMIT_USER" ]]; then
  echo "E4_SUBMIT_USER must equal the actual submit user=$SUBMIT_USER" >&2
  exit 2
fi
CPU_NODELIST=${E4_CPU_NODELIST:-sof1-h200-[0-7]}
EXPECTED_E3_ROOT=outputs/icra2027/icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic
if [[ -v E4_E3_ROOT && "$E4_E3_ROOT" != "$EXPECTED_E3_ROOT" ]]; then
  echo "E4_E3_ROOT cannot override the frozen two-room pilot source" >&2
  exit 2
fi
E4_E3_ROOT=$EXPECTED_E3_ROOT

: "${E4_FREEZE_ID:?set a fresh E4_FREEZE_ID before submitting}"
[[ "$E4_FREEZE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_FREEZE_ID: $E4_FREEZE_ID" >&2
  exit 2
}
[[ "$ACCOUNT" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_ACCOUNT: $ACCOUNT" >&2
  exit 2
}
[[ "$SUBMIT_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_SUBMIT_USER: $SUBMIT_USER" >&2
  exit 2
}
[[ "$CPU_NODELIST" =~ ^sof1-h200-\[[0-7,-]+\]$ \
  || "$CPU_NODELIST" =~ ^sof1-h200-[0-7]$ ]] || {
  echo "E4_CPU_NODELIST must select only sof1-h200-[0-7]: $CPU_NODELIST" >&2
  exit 2
}
for array_variable in \
  SLURM_ARRAY_JOB_ID \
  SLURM_ARRAY_TASK_ID \
  SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN \
  SLURM_ARRAY_TASK_MAX \
  SLURM_ARRAY_TASK_STEP \
  SBATCH_ARRAY_INX \
  SBATCH_ARRAY
do
  if [[ -v "$array_variable" ]]; then
    echo "E4 submitter refuses Slurm array context: $array_variable is set" >&2
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
[[ -s "$LAUNCHER" && ! -L "$LAUNCHER" ]] || {
  echo "missing, empty, or symlinked E4 CPU launcher: $LAUNCHER" >&2
  exit 2
}
[[ -d "$EVIDENCE_ROOT/$E4_E3_ROOT" && ! -L "$EVIDENCE_ROOT/$E4_E3_ROOT" ]] || {
  echo "missing or symlinked E3 root: $EVIDENCE_ROOT/$E4_E3_ROOT" >&2
  exit 2
}

CODE_COMMIT=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "git did not return an exact 40-character commit" >&2
  exit 2
}
[[ "$(git -C "$CODE_ROOT" rev-parse --show-toplevel)" == "$CODE_ROOT" ]] || {
  echo "submitter is not using the isolated E4 code worktree" >&2
  exit 2
}
if [[ -n "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "refusing E4 submission from a dirty code worktree" >&2
  git -C "$CODE_ROOT" status --short >&2
  exit 2
fi

# Fail before sbatch if normal is not an actual account association.  This
# prevents a scheduler-side silent QoS rewrite from changing provenance.
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

EXPERIMENT_ROOT="$EVIDENCE_ROOT/outputs/icra2027/$E4_FREEZE_ID"
SUBMISSION_DIR="$EVIDENCE_ROOT/outputs/icra2027/submissions/${E4_FREEZE_ID}-cpu-prereq"
LEDGER="$SUBMISSION_DIR/jobs.tsv"
LOG_DIR="$SUBMISSION_DIR/logs"
RECEIPT_DIR="$SUBMISSION_DIR/sbatch_receipts"
[[ ! -e "$EXPERIMENT_ROOT" && ! -L "$EXPERIMENT_ROOT" ]] || {
  echo "refusing to reuse E4 experiment root: $EXPERIMENT_ROOT" >&2
  exit 2
}
[[ ! -e "$SUBMISSION_DIR" && ! -L "$SUBMISSION_DIR" ]] || {
  echo "refusing to reuse E4 submission directory: $SUBMISSION_DIR" >&2
  exit 2
}
mkdir -p "$EVIDENCE_ROOT/outputs/icra2027/submissions"
mkdir "$SUBMISSION_DIR"
mkdir "$LOG_DIR"
mkdir "$RECEIPT_DIR"

SEQUENCE=0
LEDGER_TEMP=""
on_exit() {
  local rc=$?
  if [[ -n "$LEDGER_TEMP" && -f "$LEDGER_TEMP" && ! -L "$LEDGER_TEMP" ]]; then
    rm -f -- "$LEDGER_TEMP"
  fi
  if [[ $rc -ne 0 ]]; then
    echo "E4 submission stopped after $SEQUENCE recorded ordinary jobs; ledger: $LEDGER" >&2
  fi
}
trap on_exit EXIT

LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.initialize.$$"
(
  umask 077
  printf 'sequence\tsubmitted_utc\tstage\tscene_id\tpolicy_id\tjob_id\tdependency\tkill_on_invalid_dep\tprofile\tcpus\tmem\ttime\tpartition\tqos\taccount\tsubmit_user\tnodelist\tgres\tcode_commit\te3_root\tlauncher\targuments\tsbatch_receipt\tsbatch_receipt_sha256\n' \
    > "$LEDGER_TEMP"
)
sync -f "$LEDGER_TEMP"
mv -T -- "$LEDGER_TEMP" "$LEDGER"
sync -f "$SUBMISSION_DIR"
LEDGER_TEMP=""

record_job() {
  local stage=$1 scene_id=$2 policy_id=$3 job_id=$4 dependency=$5
  local cpus=$6 mem=$7 time_limit=$8 arguments=$9 receipt=${10} receipt_sha=${11}
  local kill_on_invalid_dep=no
  [[ "$dependency" == - ]] || kill_on_invalid_dep=yes
  local submitted_utc
  submitted_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  SEQUENCE=$((SEQUENCE + 1))
  LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.${SEQUENCE}.$$"
  cp -- "$LEDGER" "$LEDGER_TEMP"
  printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tsof1-cpu\t%s\t%s\t%s\tbatch\tnormal\t%s\t%s\t%s\tnone\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$SEQUENCE" "$submitted_utc" "$stage" "$scene_id" "$policy_id" \
    "$job_id" "$dependency" "$kill_on_invalid_dep" "$cpus" "$mem" "$time_limit" \
    "$ACCOUNT" "$SUBMIT_USER" "$CPU_NODELIST" "$CODE_COMMIT" "$E4_E3_ROOT" \
    "${LAUNCHER#"$CODE_ROOT"/}" "$arguments" "$receipt" "$receipt_sha" \
    >> "$LEDGER_TEMP"
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
  if awk -F '\t' -v id="$candidate" 'NR > 1 && $6 == id {found=1} END {exit !found}' "$LEDGER"; then
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
  local -a dependency_arg=()
  if [[ "$dependency" != - ]]; then
    [[ "$dependency" =~ ^afterok:[0-9]+(:[0-9]+)*$ ]] || {
      echo "unsafe dependency: $dependency" >&2
      return 2
    }
    dependency_arg=(--dependency="$dependency" --kill-on-invalid-dep=yes)
  fi
  local log_tag="${stage}-${scene_id}-${policy_id}"
  # This DAG requests no GRES/GPU, so RestrictedCoresPerGPU is inapplicable;
  # the CPU and memory contracts below are the complete resource request.
  # Export only the four validated bindings. Slurm adds its own SLURM_*
  # variables; caller Python/CUDA/SIMANY state is intentionally not inherited.
  local export_spec="E4_FREEZE_ID=$E4_FREEZE_ID,E4_CODE_COMMIT=$CODE_COMMIT,E4_E3_ROOT=$E4_E3_ROOT,E4_SUBMIT_USER=$SUBMIT_USER"
  local next_sequence=$((SEQUENCE + 1))
  local receipt_name receipt receipt_sha
  printf -v receipt_name '%02d-%s-%s-%s.stdout' \
    "$next_sequence" "$stage" "$scene_id" "$policy_id"
  receipt="$RECEIPT_DIR/$receipt_name"
  [[ ! -e "$receipt" && ! -L "$receipt" ]] || {
    echo "refusing to overwrite sbatch receipt: $receipt" >&2
    return 2
  }
  local sbatch_rc=0
  umask 077
  "$SBATCH_BIN" --parsable --no-requeue \
    --account="$ACCOUNT" \
    --job-name="icra-e4-$stage" \
    --partition=batch --qos=normal --nodelist="$CPU_NODELIST" \
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem" --time="$time_limit" \
    --output="$LOG_DIR/${log_tag}-%j.out" \
    --error="$LOG_DIR/${log_tag}-%j.err" \
    --export="$export_spec" \
    "${dependency_arg[@]}" \
    "$LAUNCHER" "${launcher_args[@]}" > "$receipt" || sbatch_rc=$?
  sync -f "$receipt"
  chmod 0444 "$receipt"
  sync -f "$RECEIPT_DIR"
  if [[ $sbatch_rc -ne 0 ]]; then
    echo "sbatch failed for sequence $next_sequence (rc=$sbatch_rc); receipt: $receipt" >&2
    return "$sbatch_rc"
  fi
  local -a receipt_lines=()
  mapfile -t receipt_lines < "$receipt"
  [[ ${#receipt_lines[@]} -eq 1 ]] || {
    echo "sbatch receipt must contain exactly one line: $receipt" >&2
    return 2
  }
  local raw=${receipt_lines[0]}
  validate_job_id "$raw"
  receipt_sha=$(sha256sum "$receipt" | awk '{print $1}')
  local rendered_args
  printf -v rendered_args '%q ' "${launcher_args[@]}"
  record_job "$stage" "$scene_id" "$policy_id" "$SUBMITTED_JOB_ID" \
    "$dependency" "$cpus" "$mem" "$time_limit" "${rendered_args% }" \
    "${receipt#"$SUBMISSION_DIR"/}" "$receipt_sha"
}

# Four independent materializations.
submit_job materialize b0a08200c9 A0 - 8 64G 01:00:00 materialize b0a08200c9 A0
b0_a0=$SUBMITTED_JOB_ID
submit_job materialize b0a08200c9 A4 - 8 64G 01:00:00 materialize b0a08200c9 A4
b0_a4=$SUBMITTED_JOB_ID
submit_job materialize 825d228aec A0 - 8 64G 01:00:00 materialize 825d228aec A0
s825_a0=$SUBMITTED_JOB_ID
submit_job materialize 825d228aec A4 - 8 64G 01:00:00 materialize 825d228aec A4
s825_a4=$SUBMITTED_JOB_ID

# Each full-room preparation waits for exactly its two construction arms.
submit_job prepare b0a08200c9 paired "afterok:$b0_a0:$b0_a4" \
  16 96G 02:00:00 prepare-scene b0a08200c9
b0_prepare=$SUBMITTED_JOB_ID
submit_job prepare 825d228aec paired "afterok:$s825_a0:$s825_a4" \
  16 96G 02:00:00 prepare-scene 825d228aec
s825_prepare=$SUBMITTED_JOB_ID

# Final task freeze and harness dry-run waits for both scenes.  There is no
# downstream GPU submission in this script.
submit_job final - paired "afterok:$b0_prepare:$s825_prepare" \
  4 32G 00:30:00 finalize
final_job=$SUBMITTED_JOB_ID

chmod 0444 "$LEDGER"
ledger_sha256=$(sha256sum "$LEDGER" | awk '{print $1}')
echo "submitted 7 ordinary CPU jobs (no arrays, no GPU)"
echo "final_job=$final_job"
echo "ledger=$LEDGER"
echo "ledger_sha256=$ledger_sha256"
