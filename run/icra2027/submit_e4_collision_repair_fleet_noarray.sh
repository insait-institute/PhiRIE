#!/usr/bin/env bash
# Enqueue 50 ordinary CPU jobs: a 29-job collision-repair gate followed by
# the existing 21-job five-scene candidate screen. Every downstream job has
# a direct afterok dependency on the unique repair aggregate.
set -euo pipefail

CODE_ROOT=${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot
EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}
PYTHON=${SIMANY_ROOT:-$PWD}/.venv/bin/python
REPAIR_LAUNCHER="$CODE_ROOT/run/slurm/icra2027_e4_collision_repair_cpu.sbatch"
CANDIDATE_LAUNCHER="$CODE_ROOT/run/slurm/icra2027_e4_candidate_cpu.sbatch"
PILOT_SCRIPT="$CODE_ROOT/run/icra2027/e4_collision_repair_pilot.py"
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

: "${E4_REPAIR_ID:?set a fresh E4_REPAIR_ID before submitting}"
: "${E4_SCREEN_ID:?set a fresh E4_SCREEN_ID before submitting}"
for run_id in "$E4_REPAIR_ID" "$E4_SCREEN_ID"; do
  [[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "unsafe run ID: $run_id" >&2
    exit 2
  }
done
[[ "$E4_REPAIR_ID" != "$E4_SCREEN_ID" \
  && "${E4_REPAIR_ID}-room" != "$E4_SCREEN_ID" \
  && "${E4_REPAIR_ID}-shim" != "$E4_SCREEN_ID" ]] || {
  echo "repair, mode, and candidate IDs must be distinct" >&2
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
    echo "repair fleet refuses Slurm array context: $array_variable is set" >&2
    exit 2
  fi
done

for executable in "$SBATCH_BIN" "$SACCTMGR_BIN"; do
  command -v "$executable" >/dev/null || { echo "missing executable: $executable" >&2; exit 2; }
done
[[ -x "$PYTHON" && -x "$SETFACL" && -x "$GETFACL" ]] || {
  echo "missing Python or ACL tools" >&2
  exit 2
}
for source in "$REPAIR_LAUNCHER" "$CANDIDATE_LAUNCHER" "$PILOT_SCRIPT"; do
  [[ -s "$source" && ! -L "$source" ]] || {
    echo "missing, empty, or symlinked fleet source: $source" >&2
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
[[ -d "$MENAGERIE_ROOT" && ! -L "$MENAGERIE_ROOT" ]] || {
  echo "missing worktree-local Menagerie closure" >&2
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
  "$PYTHON" - "$PILOT_SCRIPT" "$CODE_ROOT" "$EVIDENCE_ROOT" \
  "$EXPECTED_E3_ROOT" "$MENAGERIE_ROOT" "$EXPECTED_MENAGERIE_COMMIT" <<'PY'
import importlib.util
import sys
from pathlib import Path

import robo.eval.e4_candidate_screen as candidate
import robo.sim.export_mjcf as exporter

script, code_root, evidence_root, e3_root, menagerie, menagerie_commit = sys.argv[1:]
code_root, evidence_root, e3_root, menagerie = map(
    Path, (code_root, evidence_root, e3_root, menagerie)
)
spec = importlib.util.spec_from_file_location("sealed_e4_collision_repair_pilot", script)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load repair pilot")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
if pilot.CODE_ROOT != code_root or pilot.EXPECTED_EVIDENCE_ROOT != evidence_root:
    raise SystemExit("repair-pilot dual-root constants differ")
if tuple(pilot.PILOT_SCENES) != ("3db0a1c8f3", "d755b3d9d8"):
    raise SystemExit("repair-pilot scene roster differs")
if tuple(candidate.SCENE_IDS) != (
    "3db0a1c8f3", "27dd4da69e", "d755b3d9d8", "acd95847c5", "40aec5fffa"
):
    raise SystemExit("candidate-screen scene roster differs")
if candidate.CODE_ROOT != code_root or candidate.EXPECTED_EVIDENCE_ROOT != evidence_root:
    raise SystemExit("candidate-screen dual-root constants differ")
if candidate.EXPECTED_E3_ROOT != e3_root:
    raise SystemExit("candidate E3 root differs")
if candidate.EXPECTED_MENAGERIE_ROOT != menagerie:
    raise SystemExit("candidate Menagerie root differs")
if candidate.EXPECTED_MENAGERIE_COMMIT != menagerie_commit:
    raise SystemExit("candidate Menagerie commit differs")
if not callable(getattr(exporter, "load_common_carve_inputs", None)):
    raise SystemExit("exporter lacks paired common-carve API")
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
REPAIR_ROOT="$OUTPUTS_ROOT/$E4_REPAIR_ID"
REPAIR_ROOM_ROOT="$OUTPUTS_ROOT/${E4_REPAIR_ID}-room"
REPAIR_SHIM_ROOT="$OUTPUTS_ROOT/${E4_REPAIR_ID}-shim"
CANDIDATE_ROOT="$OUTPUTS_ROOT/$E4_SCREEN_ID"
SUBMISSIONS_ROOT="$OUTPUTS_ROOT/submissions"
SUBMISSION_DIR="$SUBMISSIONS_ROOT/${E4_REPAIR_ID}-collision-repair-fleet"
CANDIDATE_SUPPORT_DIR="$SUBMISSIONS_ROOT/${E4_SCREEN_ID}-candidate-screen"
LOG_DIR="$SUBMISSION_DIR/logs"
RECEIPT_DIR="$SUBMISSION_DIR/sbatch_receipts"
RUNTIME_DIR="$SUBMISSION_DIR/runtime"
LEDGER="$SUBMISSION_DIR/jobs.tsv"
for fresh_path in \
  "$REPAIR_ROOT" "$REPAIR_ROOM_ROOT" "$REPAIR_SHIM_ROOT" \
  "$CANDIDATE_ROOT" "$SUBMISSION_DIR" "$CANDIDATE_SUPPORT_DIR"
do
  [[ ! -e "$fresh_path" && ! -L "$fresh_path" ]] || {
    echo "refusing to reuse fleet path: $fresh_path" >&2
    exit 2
  }
done

umask 077
mkdir -p "$SUBMISSIONS_ROOT"
[[ -d "$SUBMISSIONS_ROOT" && ! -L "$SUBMISSIONS_ROOT" ]] || {
  echo "missing or symlinked submissions root" >&2
  exit 2
}

# Create all shared parents before the first scheduler mutation. Strict atomic
# writers may then create only their unique leaf directories.
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
make_private_directory "$CANDIDATE_SUPPORT_DIR"
make_private_directory "$CANDIDATE_SUPPORT_DIR/logs"
make_private_directory "$CANDIDATE_SUPPORT_DIR/sbatch_receipts"
make_private_directory "$CANDIDATE_SUPPORT_DIR/runtime"

make_private_directory "$REPAIR_ROOT"
for group in export_receipts export_validations; do
  make_private_directory "$REPAIR_ROOT/$group"
  for mode in room shim; do
    make_private_directory "$REPAIR_ROOT/$group/$mode"
    for policy in A0 A4; do
      make_private_directory "$REPAIR_ROOT/$group/$mode/$policy"
    done
  done
done
make_private_directory "$REPAIR_ROOT/mode_comparisons"
make_private_directory "$REPAIR_ROOT/mode_comparisons/A0"
make_private_directory "$REPAIR_ROOT/mode_comparisons/A4"

for mode_root in "$REPAIR_ROOM_ROOT" "$REPAIR_SHIM_ROOT"; do
  make_private_directory "$mode_root"
  make_private_directory "$mode_root/construction_variants"
  make_private_directory "$mode_root/construction_variants/A0"
  make_private_directory "$mode_root/construction_variants/A4"
done

make_private_directory "$CANDIDATE_ROOT"
for relative in \
  construction_variants \
  construction_variants/A0 \
  construction_variants/A4 \
  candidate_suites task_freezes scene_prepares scene_qualifiers
do
  make_private_directory "$CANDIDATE_ROOT/$relative"
done
for directory in "${PRIVATE_DIRS[@]}"; do sync -f "$directory"; done
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
    echo "repair fleet stopped after $SEQUENCE recorded ordinary jobs" >&2
    echo "partial ledger: $LEDGER" >&2
    echo "receipts: $RECEIPT_DIR" >&2
  fi
}
trap on_exit EXIT

LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.initialize.$$"
(
  umask 077
  printf '%s\n' \
    $'sequence\tsubmitted_utc\tpipeline\tstage\tcollision_mode\tscene_id\tpolicy_id\tjob_id\tdependency\tkill_on_invalid_dep\tprofile\tcpus\tmem\ttime\tpartition\tqos\taccount\tsubmit_user\tnodelist\tgres\tcode_commit\te3_root\trepair_id\tscreen_id\tlauncher\targuments\tsbatch_stdout_receipt\tsbatch_stdout_sha256\tsbatch_stderr_receipt\tsbatch_stderr_sha256' \
    > "$LEDGER_TEMP"
)
sync -f "$LEDGER_TEMP"
mv -T -- "$LEDGER_TEMP" "$LEDGER"
sync -f "$SUBMISSION_DIR"
LEDGER_TEMP=""

validate_job_id() {
  local raw=$1
  [[ "$raw" =~ ^([0-9]+)(\;[A-Za-z0-9._-]+)?$ ]] || {
    echo "sbatch did not return one ordinary numeric job ID: $raw" >&2
    return 2
  }
  local candidate=${BASH_REMATCH[1]}
  if awk -F '\t' -v id="$candidate" \
    'NR > 1 && $8 == id {found=1} END {exit !found}' "$LEDGER"; then
    echo "duplicate Slurm job ID returned: $candidate" >&2
    return 2
  fi
  SUBMITTED_JOB_ID=$candidate
}

record_job() {
  local pipeline=$1 stage=$2 mode=$3 scene=$4 policy=$5 job_id=$6 dependency=$7
  local cpus=$8 mem=$9 time_limit=${10} launcher=${11} arguments=${12}
  local stdout_receipt=${13} stdout_sha=${14} stderr_receipt=${15} stderr_sha=${16}
  local kill_on_invalid_dep=no
  [[ "$dependency" == - ]] || kill_on_invalid_dep=yes
  local submitted_utc
  submitted_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  SEQUENCE=$((SEQUENCE + 1))
  LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.${SEQUENCE}.$$"
  cp -- "$LEDGER" "$LEDGER_TEMP"
  {
    printf '%s\t' \
      "$SEQUENCE" "$submitted_utc" "$pipeline" "$stage" "$mode" "$scene" \
      "$policy" "$job_id" "$dependency" "$kill_on_invalid_dep" sof1-cpu \
      "$cpus" "$mem" "$time_limit" batch normal "$ACCOUNT" "$SUBMIT_USER" \
      "$CPU_NODELIST" none "$CODE_COMMIT" "$EXPECTED_E3_ROOT" "$E4_REPAIR_ID" \
      "$E4_SCREEN_ID" "${launcher#"$CODE_ROOT"/}" "$arguments" \
      "$stdout_receipt" "$stdout_sha" "$stderr_receipt"
    printf '%s\n' "$stderr_sha"
  } >> "$LEDGER_TEMP"
  sync -f "$LEDGER_TEMP"
  mv -T -- "$LEDGER_TEMP" "$LEDGER"
  sync -f "$SUBMISSION_DIR"
  LEDGER_TEMP=""
}

submit_job() {
  local pipeline=$1 stage=$2 mode=$3 scene=$4 policy=$5 dependency=$6
  local cpus=$7 mem=$8 time_limit=$9 job_name=${10} launcher=${11}
  shift 11
  local -a launcher_args=("$@") dependency_args=()
  if [[ "$dependency" != - ]]; then
    [[ "$dependency" =~ ^after(ok|any):[0-9]+(:[0-9]+)*(,after(ok|any):[0-9]+(:[0-9]+)*)*$ ]] || {
      echo "unsafe dependency: $dependency" >&2
      return 2
    }
    dependency_args=(--dependency="$dependency" --kill-on-invalid-dep=yes)
  fi
  local next_sequence=$((SEQUENCE + 1))
  local tag="${pipeline}-${stage}-${mode}-${scene}-${policy}"
  local receipt_stem stdout_receipt stderr_receipt receipt
  printf -v receipt_stem '%02d-%s-%s-%s-%s-%s' \
    "$next_sequence" "$pipeline" "$stage" "$mode" "$scene" "$policy"
  stdout_receipt="$RECEIPT_DIR/${receipt_stem}.stdout"
  stderr_receipt="$RECEIPT_DIR/${receipt_stem}.stderr"
  for receipt in "$stdout_receipt" "$stderr_receipt"; do
    [[ ! -e "$receipt" && ! -L "$receipt" ]] || {
      echo "refusing to overwrite receipt: $receipt" >&2
      return 2
    }
  done
  local export_spec
  if [[ "$pipeline" == repair ]]; then
    export_spec="E4_REPAIR_ID=$E4_REPAIR_ID,E4_CODE_COMMIT=$CODE_COMMIT,E4_E3_ROOT=$EXPECTED_E3_ROOT,E4_SUBMIT_USER=$SUBMIT_USER"
  else
    export_spec="E4_SCREEN_ID=$E4_SCREEN_ID,E4_CODE_COMMIT=$CODE_COMMIT,E4_E3_ROOT=$EXPECTED_E3_ROOT,E4_SUBMIT_USER=$SUBMIT_USER"
  fi
  local sbatch_rc=0
  "$SBATCH_BIN" --parsable --no-requeue \
    --account="$ACCOUNT" --job-name="$job_name" \
    --partition=batch --qos=normal --nodelist="$CPU_NODELIST" \
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem="$mem" --time="$time_limit" \
    --output="$LOG_DIR/${tag}-%j.out" --error="$LOG_DIR/${tag}-%j.err" \
    --export="$export_spec" "${dependency_args[@]}" \
    "$launcher" "${launcher_args[@]}" \
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
  record_job "$pipeline" "$stage" "$mode" "$scene" "$policy" \
    "$SUBMITTED_JOB_ID" "$dependency" "$cpus" "$mem" "$time_limit" \
    "$launcher" "${rendered_args% }" \
    "${stdout_receipt#"$SUBMISSION_DIR"/}" "$stdout_sha" \
    "${stderr_receipt#"$SUBMISSION_DIR"/}" "$stderr_sha"
}

PILOT_SCENES=(3db0a1c8f3 d755b3d9d8)
MODES=(room shim)
POLICIES=(A0 A4)
declare -A REPAIR_MATERIALIZE_JOBS REPAIR_EXPORT_JOBS REPAIR_VALIDATE_JOBS
declare -A REPAIR_COMPARE_JOBS

# 1-8: independent mode-specific materializations. They are duplicated only
# because exporter output is in-place and both room/shim trees must survive.
for scene in "${PILOT_SCENES[@]}"; do
  for mode in "${MODES[@]}"; do
    for policy in "${POLICIES[@]}"; do
      submit_job repair materialize "$mode" "$scene" "$policy" - \
        8 64G 01:00:00 icra-e4-repair-materialize "$REPAIR_LAUNCHER" \
        materialize "$mode" "$scene" "$policy"
      REPAIR_MATERIALIZE_JOBS["$scene/$mode/$policy"]=$SUBMITTED_JOB_ID
    done
  done
done

# 9-16: each export waits for both policies of its scene/mode. This is needed
# even for one output factory because room mode builds a paired-policy union.
for scene in "${PILOT_SCENES[@]}"; do
  for mode in "${MODES[@]}"; do
    paired_materializations="afterok:${REPAIR_MATERIALIZE_JOBS[$scene/$mode/A0]}:${REPAIR_MATERIALIZE_JOBS[$scene/$mode/A4]}"
    for policy in "${POLICIES[@]}"; do
      submit_job repair export "$mode" "$scene" "$policy" "$paired_materializations" \
        16 96G 04:00:00 icra-e4-repair-export "$REPAIR_LAUNCHER" \
        export "$mode" "$scene" "$policy"
      REPAIR_EXPORT_JOBS["$scene/$mode/$policy"]=$SUBMITTED_JOB_ID
    done
  done
done

# 17-24: validators run afterany so failed/partial exports still yield a sealed
# invalid diagnostic when possible.
for scene in "${PILOT_SCENES[@]}"; do
  for mode in "${MODES[@]}"; do
    for policy in "${POLICIES[@]}"; do
      dependency="afterany:${REPAIR_EXPORT_JOBS[$scene/$mode/$policy]}"
      submit_job repair validate "$mode" "$scene" "$policy" "$dependency" \
        4 32G 01:00:00 icra-e4-repair-validate "$REPAIR_LAUNCHER" \
        validate "$mode" "$scene" "$policy"
      REPAIR_VALIDATE_JOBS["$scene/$mode/$policy"]=$SUBMITTED_JOB_ID
    done
  done
done

# 25-28: paired mode comparisons are diagnostic. Only room hard checks and
# room stable slots contribute to the final release gate; shim cannot unlock it.
for scene in "${PILOT_SCENES[@]}"; do
  for policy in "${POLICIES[@]}"; do
    dependency="afterany:${REPAIR_VALIDATE_JOBS[$scene/room/$policy]}:${REPAIR_VALIDATE_JOBS[$scene/shim/$policy]}"
    submit_job repair compare room-vs-shim "$scene" "$policy" "$dependency" \
      2 16G 00:30:00 icra-e4-repair-compare "$REPAIR_LAUNCHER" \
      compare "$scene" "$policy"
    REPAIR_COMPARE_JOBS["$scene/$policy"]=$SUBMITTED_JOB_ID
  done
done

# 29: unique fail-closed repair release gate.
repair_compare_ids=$(printf '%s\n' "${REPAIR_COMPARE_JOBS[@]}" | sort -n | paste -sd: -)
submit_job repair aggregate - - - "afterany:$repair_compare_ids" \
  2 16G 00:30:00 icra-e4-repair-aggregate "$REPAIR_LAUNCHER" aggregate
REPAIR_AGGREGATE_JOB=$SUBMITTED_JOB_ID

# 30-50: the existing five-scene screen, all directly bound to the unique
# repair aggregate afterok in addition to their own stage dependencies.
CANDIDATE_SCENES=(3db0a1c8f3 27dd4da69e d755b3d9d8 acd95847c5 40aec5fffa)
CANDIDATE_A0_JOBS=()
CANDIDATE_A4_JOBS=()
CANDIDATE_PREPARE_JOBS=()
CANDIDATE_QUALIFY_JOBS=()
for scene in "${CANDIDATE_SCENES[@]}"; do
  dependency="afterok:$REPAIR_AGGREGATE_JOB"
  submit_job candidate materialize - "$scene" A0 "$dependency" \
    8 64G 01:00:00 icra-e4-candidate-materialize "$CANDIDATE_LAUNCHER" \
    materialize "$scene" A0
  CANDIDATE_A0_JOBS+=("$SUBMITTED_JOB_ID")
  submit_job candidate materialize - "$scene" A4 "$dependency" \
    8 64G 01:00:00 icra-e4-candidate-materialize "$CANDIDATE_LAUNCHER" \
    materialize "$scene" A4
  CANDIDATE_A4_JOBS+=("$SUBMITTED_JOB_ID")
done
for index in "${!CANDIDATE_SCENES[@]}"; do
  scene=${CANDIDATE_SCENES[$index]}
  dependency="afterok:$REPAIR_AGGREGATE_JOB:${CANDIDATE_A0_JOBS[$index]}:${CANDIDATE_A4_JOBS[$index]}"
  submit_job candidate prepare - "$scene" paired "$dependency" \
    16 96G 04:00:00 icra-e4-candidate-prepare "$CANDIDATE_LAUNCHER" \
    prepare-scene "$scene"
  CANDIDATE_PREPARE_JOBS+=("$SUBMITTED_JOB_ID")
done
for index in "${!CANDIDATE_SCENES[@]}"; do
  scene=${CANDIDATE_SCENES[$index]}
  dependency="afterok:$REPAIR_AGGREGATE_JOB:${CANDIDATE_PREPARE_JOBS[$index]}"
  submit_job candidate qualify - "$scene" paired "$dependency" \
    4 32G 01:00:00 icra-e4-candidate-qualify "$CANDIDATE_LAUNCHER" \
    qualify-scene "$scene"
  CANDIDATE_QUALIFY_JOBS+=("$SUBMITTED_JOB_ID")
done
qualifier_ids=$(IFS=:; printf '%s' "${CANDIDATE_QUALIFY_JOBS[*]}")
dependency="afterok:$REPAIR_AGGREGATE_JOB,afterany:$qualifier_ids"
submit_job candidate aggregate - - - "$dependency" \
  4 32G 00:30:00 icra-e4-candidate-aggregate "$CANDIDATE_LAUNCHER" aggregate
CANDIDATE_AGGREGATE_JOB=$SUBMITTED_JOB_ID

[[ $SEQUENCE -eq 50 ]] || {
  echo "internal error: expected 50 jobs, recorded $SEQUENCE" >&2
  exit 2
}
chmod 0444 "$LEDGER"
sync -f "$LEDGER"
sync -f "$SUBMISSION_DIR"
ledger_sha256=$(sha256sum "$LEDGER" | awk '{print $1}')
echo "submitted 50 ordinary CPU jobs at once (no Slurm arrays, no GPU)"
echo "repair_aggregate_job=$REPAIR_AGGREGATE_JOB"
echo "candidate_aggregate_job=$CANDIDATE_AGGREGATE_JOB"
echo "downstream_gate=afterok:$REPAIR_AGGREGATE_JOB"
echo "ledger=$LEDGER"
echo "ledger_sha256=$ledger_sha256"
