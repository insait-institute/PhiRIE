#!/usr/bin/env bash
# Submit the exact 33-job, four-scene robust-floor/support CPU extension.
set -euo pipefail

CODE_ROOT=/group/worldcept/PhiRIE/code/SimAny-wt/e4-paired-pilot
EVIDENCE_ROOT=/group/worldcept/PhiRIE/code/SimAny
PYTHON=/group/worldcept/PhiRIE/code/SimAny/.venv/bin/python
LAUNCHER="$CODE_ROOT/run/slurm/icra2027_e4_robust_floor_support_extension_cpu.sbatch"
SWEEP_SCRIPT="$CODE_ROOT/run/icra2027/e4_robust_floor_support_extension.py"
EXPECTED_E3_ROOT=outputs/icra2027/icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic
EXTERNAL_SWEEP_ID=icra2027-contract-v1-e4-8c7b796b3f25-robust-floor-support-20260904T192849Z
EXTERNAL_AGGREGATE_MANIFEST_SHA256=7d12d1cf7c212c97da0f97596970c0da983f450a89da7abe17035a98b4a7d5a6
EXTERNAL_COMPARISON_MANIFEST_SHA256=99b979ded06854426f2cab4942086401ef6a27f7b0c499ca4f80a1f341c0bc19
ACCOUNT=runyi_yang
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

: "${E4_ROBUST_EXTENSION_ID:?set a fresh E4_ROBUST_EXTENSION_ID before submitting}"
[[ "$E4_ROBUST_EXTENSION_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe extension ID: $E4_ROBUST_EXTENSION_ID" >&2; exit 2;
}
[[ "$E4_ROBUST_EXTENSION_ID" != "$EXTERNAL_SWEEP_ID" ]] || {
  echo "extension ID must differ from its external source" >&2; exit 2;
}
[[ "$SUBMIT_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe submit user" >&2; exit 2;
}
for variable in SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP \
  SBATCH_ARRAY_INX SBATCH_ARRAY; do
  [[ ! -v "$variable" ]] || {
    echo "robust extension submitter refuses Slurm array context: $variable" >&2
    exit 2
  }
done
for executable in "$SBATCH_BIN" "$SACCTMGR_BIN"; do
  command -v "$executable" >/dev/null || {
    echo "missing executable: $executable" >&2; exit 2;
  }
done
[[ -x "$PYTHON" && -x "$SETFACL" && -x "$GETFACL" ]] || {
  echo "missing Python or ACL tools" >&2; exit 2;
}
for source in "$LAUNCHER" "$SWEEP_SCRIPT"; do
  [[ -s "$source" && ! -L "$source" ]] || {
    echo "missing, empty, or symlinked source: $source" >&2; exit 2;
  }
done
[[ -d "$EVIDENCE_ROOT" && ! -L "$EVIDENCE_ROOT" \
  && "$CODE_ROOT" != "$EVIDENCE_ROOT" ]] || {
  echo "invalid dual-root binding" >&2; exit 2;
}
[[ -d "$EVIDENCE_ROOT/$EXPECTED_E3_ROOT" \
  && ! -L "$EVIDENCE_ROOT/$EXPECTED_E3_ROOT" ]] || {
  echo "missing sealed E3 root" >&2; exit 2;
}
EXTERNAL_ROOT="$EVIDENCE_ROOT/outputs/icra2027/$EXTERNAL_SWEEP_ID"
for external in \
  "$EXTERNAL_ROOT/aggregate" \
  "$EXTERNAL_ROOT/variants/d755-fs/comparison" \
  "$EXTERNAL_ROOT/variants/d755-fs/common_static"; do
  [[ -d "$external" && ! -L "$external" ]] || {
    echo "missing or symlinked external winner evidence: $external" >&2; exit 2;
  }
done

CODE_COMMIT=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$CODE_COMMIT" =~ ^[0-9a-f]{40}$ \
  && "$(git -C "$CODE_ROOT" rev-parse --show-toplevel)" == "$CODE_ROOT" ]] || {
  echo "invalid isolated code snapshot" >&2; exit 2;
}
if [[ -n "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "refusing submission from a dirty code worktree" >&2
  git -C "$CODE_ROOT" status --short >&2
  exit 2
fi

# The runner is the single frozen matrix source. This preflight also performs
# the full sealed d755 winner-registry validation before the first sbatch call.
matrix_output=$(
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$CODE_ROOT" \
    SIMANY_EVIDENCE_ROOT="$EVIDENCE_ROOT" \
    "$PYTHON" - "$SWEEP_SCRIPT" "$CODE_ROOT" "$EVIDENCE_ROOT" \
    "$EXPECTED_E3_ROOT" "$EXTERNAL_SWEEP_ID" \
    "$EXTERNAL_AGGREGATE_MANIFEST_SHA256" \
    "$EXTERNAL_COMPARISON_MANIFEST_SHA256" <<'PY'
import importlib.util
import sys
from pathlib import Path

(script, code_root, evidence_root, e3_root, external_sweep,
 aggregate_sha, comparison_sha) = sys.argv[1:]
code_root, evidence_root, e3_root = map(Path, (code_root, evidence_root, e3_root))
spec = importlib.util.spec_from_file_location("sealed_e4_robust_extension", script)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load robust extension runner")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if module.CODE_ROOT != code_root or module.EXPECTED_EVIDENCE_ROOT != evidence_root:
    raise SystemExit("robust extension dual-root constants differ")
if module.EXPECTED_E3_ROOT != e3_root or tuple(module.POLICIES) != ("A0", "A4"):
    raise SystemExit("robust extension E3/policy binding differs")
if len(module.SCENE_VARIANTS) != 4 or len(module.VARIANT_IDS) != 8:
    raise SystemExit("robust extension matrix cardinality differs")
if tuple(module.SCENE_VARIANTS) != (
    "27dd4da69e", "acd95847c5", "1ada7a0617", "25f3b7a318"
):
    raise SystemExit("robust extension scene roster differs")
if {"3db0a1c8f3", "d755b3d9d8", "40aec5fffa"} & set(module.SCENE_VARIANTS):
    raise SystemExit("robust extension includes a forbidden prior scene")
registry = module._validated_external_winners(evidence_root)
if registry["source_sweep_id"] != external_sweep:
    raise SystemExit("external winner sweep differs")
winner = registry["winners"]["d755b3d9d8"]
if (winner["aggregate_bundle_manifest_sha256"] != aggregate_sha
        or winner["comparison_bundle_manifest_sha256"] != comparison_sha):
    raise SystemExit("external winner manifest pin differs")
flattened = tuple(v for pair in module.SCENE_VARIANTS.values() for v in pair)
if flattened != tuple(module.VARIANT_IDS) or set(module.VARIANTS) != set(flattened):
    raise SystemExit("robust extension variant ordering differs")
for scene_id, (floor_id, support_id) in module.SCENE_VARIANTS.items():
    floor, support = module.VARIANTS[floor_id], module.VARIANTS[support_id]
    if (floor["scene_id"] != scene_id or support["scene_id"] != scene_id
            or floor["room_surface_policy"] != "robust_floor_only"
            or support["room_surface_policy"] != "robust_floor_plus_support"):
        raise SystemExit("robust extension scene/variant binding differs")
    print(f"{scene_id}\t{floor_id}\t{support_id}")
PY
)
mapfile -t MATRIX_ROWS <<<"$matrix_output"
[[ ${#MATRIX_ROWS[@]} -eq 4 ]] || {
  echo "robust extension preflight did not return four matrix rows" >&2; exit 2;
}
declare -a SCENES VARIANTS
declare -A VARIANT_SCENE
for row in "${MATRIX_ROWS[@]}"; do
  IFS=$'\t' read -r scene floor_variant support_variant extra <<<"$row"
  [[ -z "${extra:-}" && "$scene" =~ ^[0-9a-f]{10}$ \
    && "$floor_variant" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
    && "$support_variant" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
    && "$floor_variant" != "$support_variant" \
    && ! -v "VARIANT_SCENE[$floor_variant]" \
    && ! -v "VARIANT_SCENE[$support_variant]" ]] || {
    echo "unsafe or duplicate robust extension matrix row" >&2; exit 2;
  }
  SCENES+=("$scene")
  VARIANTS+=("$floor_variant" "$support_variant")
  VARIANT_SCENE[$floor_variant]=$scene
  VARIANT_SCENE[$support_variant]=$scene
done
[[ ${#VARIANTS[@]} -eq 8 ]] || { echo "expected eight variants" >&2; exit 2; }

assoc_rows=$("$SACCTMGR_BIN" -n -P show assoc where user="$SUBMIT_USER" \
  account="$ACCOUNT" format=Account,User,Partition,QOS)
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
SWEEP_ROOT="$OUTPUTS_ROOT/$E4_ROBUST_EXTENSION_ID"
SUBMISSIONS_ROOT="$OUTPUTS_ROOT/submissions"
SUBMISSION_DIR="$SUBMISSIONS_ROOT/${E4_ROBUST_EXTENSION_ID}-robust-floor-support-extension"
LOG_DIR="$SUBMISSION_DIR/logs"
RECEIPT_DIR="$SUBMISSION_DIR/sbatch_receipts"
RUNTIME_DIR="$SUBMISSION_DIR/runtime"
for path in "$SWEEP_ROOT" "$SUBMISSION_DIR"; do
  [[ ! -e "$path" && ! -L "$path" ]] || {
    echo "refusing to reuse robust extension path: $path" >&2; exit 2;
  }
done

umask 077
mkdir -p "$SUBMISSIONS_ROOT"
[[ -d "$SUBMISSIONS_ROOT" && ! -L "$SUBMISSIONS_ROOT" ]] || {
  echo "missing or symlinked submissions root" >&2; exit 2;
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
    echo "directory is not private ACL-free mode 700: $directory" >&2; exit 2;
  }
  PRIVATE_DIRS+=("$directory")
}
make_private_directory "$SUBMISSION_DIR"
make_private_directory "$LOG_DIR"
make_private_directory "$RECEIPT_DIR"
make_private_directory "$RUNTIME_DIR"
make_private_directory "$SWEEP_ROOT"
make_private_directory "$SWEEP_ROOT/variants"
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
declare -A SEEN_JOB_IDS
on_exit() {
  local rc=$?
  if [[ -n "$LEDGER_TEMP" && -f "$LEDGER_TEMP" && ! -L "$LEDGER_TEMP" ]]; then
    local partial="$SUBMISSION_DIR/jobs.tsv.incomplete.${SEQUENCE}.$$"
    mv -T -- "$LEDGER_TEMP" "$partial"
    chmod 0444 "$partial"; sync -f "$partial"
  fi
  if [[ -f "$LEDGER" && ! -L "$LEDGER" ]]; then
    chmod 0444 "$LEDGER"; sync -f "$LEDGER"; sync -f "$SUBMISSION_DIR"
  fi
  if [[ $rc -ne 0 ]]; then
    echo "robust extension submission stopped after $SEQUENCE jobs" >&2
    echo "receipts: $RECEIPT_DIR" >&2
  fi
}
trap on_exit EXIT

LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.initialize.$$"
printf '%s\n' $'sequence\tsubmitted_utc\tstage\tvariant_id\tscene_id\tpolicy_id\tjob_id\tdependency\tkill_on_invalid_dep\tprofile\tcpus\tmem\ttime\tpartition\tqos\taccount\tsubmit_user\tnodelist\tgres\tcode_commit\te3_root\texternal_sweep_id\texternal_aggregate_manifest_sha256\texternal_comparison_manifest_sha256\tsweep_id\tlauncher\targuments\tsbatch_stdout_receipt\tsbatch_stdout_sha256\tsbatch_stderr_receipt\tsbatch_stderr_sha256' > "$LEDGER_TEMP"
sync -f "$LEDGER_TEMP"; mv -T -- "$LEDGER_TEMP" "$LEDGER"; sync -f "$SUBMISSION_DIR"
LEDGER_TEMP=""

record_job() {
  local stage=$1 variant=$2 scene=$3 policy=$4 job_id=$5 dependency=$6
  local cpus=$7 mem=$8 time_limit=$9 launcher=${10} arguments=${11}
  local stdout_receipt=${12} stdout_sha=${13} stderr_receipt=${14} stderr_sha=${15}
  local kill_on_invalid_dep=no submitted_utc
  [[ "$dependency" == - ]] || kill_on_invalid_dep=yes
  submitted_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  SEQUENCE=$((SEQUENCE + 1))
  LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.${SEQUENCE}.$$"
  cp -- "$LEDGER" "$LEDGER_TEMP"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$SEQUENCE" "$submitted_utc" "$stage" "$variant" "$scene" "$policy" \
    "$job_id" "$dependency" "$kill_on_invalid_dep" sof1-cpu "$cpus" "$mem" \
    "$time_limit" batch normal "$ACCOUNT" "$SUBMIT_USER" "$CPU_NODELIST" none \
    "$CODE_COMMIT" "$EXPECTED_E3_ROOT" "$EXTERNAL_SWEEP_ID" \
    "$EXTERNAL_AGGREGATE_MANIFEST_SHA256" \
    "$EXTERNAL_COMPARISON_MANIFEST_SHA256" "$E4_ROBUST_EXTENSION_ID" \
    "${launcher#"$CODE_ROOT"/}" "$arguments" "$stdout_receipt" "$stdout_sha" \
    "$stderr_receipt" "$stderr_sha" >> "$LEDGER_TEMP"
  sync -f "$LEDGER_TEMP"; mv -T -- "$LEDGER_TEMP" "$LEDGER"; sync -f "$SUBMISSION_DIR"
  LEDGER_TEMP=""
}

submit_job() {
  local stage=$1 variant=$2 scene=$3 policy=$4 dependency=$5
  local cpus=$6 mem=$7 time_limit=$8 job_name=$9
  shift 9
  local -a launcher_args=("$@") dependency_args=() receipt_lines=()
  if [[ "$dependency" != - ]]; then
    [[ "$dependency" =~ ^after(ok|any):[0-9]+(:[0-9]+)*$ ]] || {
      echo "unsafe dependency: $dependency" >&2; return 2;
    }
    dependency_args=(--dependency="$dependency" --kill-on-invalid-dep=yes)
  fi
  local next_sequence=$((SEQUENCE + 1)) stem stdout_receipt stderr_receipt
  printf -v stem '%02d-%s-%s-%s-%s' \
    "$next_sequence" "$stage" "$variant" "$scene" "$policy"
  stdout_receipt="$RECEIPT_DIR/${stem}.stdout"
  stderr_receipt="$RECEIPT_DIR/${stem}.stderr"
  [[ ! -e "$stdout_receipt" && ! -L "$stdout_receipt" \
    && ! -e "$stderr_receipt" && ! -L "$stderr_receipt" ]] || {
    echo "refusing to overwrite sbatch receipt" >&2; return 2;
  }
  local export_spec sbatch_rc=0
  export_spec="E4_ROBUST_EXTENSION_ID=$E4_ROBUST_EXTENSION_ID,E4_CODE_COMMIT=$CODE_COMMIT,E4_E3_ROOT=$EXPECTED_E3_ROOT,E4_SUBMIT_USER=$SUBMIT_USER"
  "$SBATCH_BIN" --parsable --no-requeue --account="$ACCOUNT" --job-name="$job_name" \
    --partition=batch --qos=normal --nodelist="$CPU_NODELIST" --nodes=1 --ntasks=1 \
    --cpus-per-task="$cpus" --mem="$mem" --time="$time_limit" \
    --output="$LOG_DIR/${stage}-${variant}-${policy}-%j.out" \
    --error="$LOG_DIR/${stage}-${variant}-${policy}-%j.err" \
    --export="$export_spec" "${dependency_args[@]}" "$LAUNCHER" "${launcher_args[@]}" \
    > "$stdout_receipt" 2> "$stderr_receipt" || sbatch_rc=$?
  sync -f "$stdout_receipt"; sync -f "$stderr_receipt"
  chmod 0444 "$stdout_receipt" "$stderr_receipt"; sync -f "$RECEIPT_DIR"
  [[ $sbatch_rc -eq 0 ]] || {
    echo "sbatch failed for sequence $next_sequence" >&2; return "$sbatch_rc";
  }
  mapfile -t receipt_lines < "$stdout_receipt"
  [[ ${#receipt_lines[@]} -eq 1 \
    && "${receipt_lines[0]}" =~ ^([0-9]+)(\;[A-Za-z0-9._-]+)?$ ]] || {
    echo "invalid sbatch --parsable receipt" >&2; return 2;
  }
  SUBMITTED_JOB_ID=${BASH_REMATCH[1]}
  [[ ! -v "SEEN_JOB_IDS[$SUBMITTED_JOB_ID]" ]] || {
    echo "duplicate sbatch job ID: $SUBMITTED_JOB_ID" >&2; return 2;
  }
  SEEN_JOB_IDS[$SUBMITTED_JOB_ID]=1
  local stdout_sha stderr_sha rendered_args
  stdout_sha=$(sha256sum "$stdout_receipt" | awk '{print $1}')
  stderr_sha=$(sha256sum "$stderr_receipt" | awk '{print $1}')
  printf -v rendered_args '%q ' "${launcher_args[@]}"
  record_job "$stage" "$variant" "$scene" "$policy" "$SUBMITTED_JOB_ID" \
    "$dependency" "$cpus" "$mem" "$time_limit" "$LAUNCHER" "${rendered_args% }" \
    "${stdout_receipt#"$SUBMISSION_DIR"/}" "$stdout_sha" \
    "${stderr_receipt#"$SUBMISSION_DIR"/}" "$stderr_sha"
}

declare -A BUILD_JOBS EVAL_A0_JOBS EVAL_A4_JOBS COMPARE_JOBS
for variant in "${VARIANTS[@]}"; do
  scene=${VARIANT_SCENE[$variant]}
  submit_job build-common "$variant" "$scene" paired - 16 96G 04:00:00 \
    icra-e4-robustx-build build-common "$variant"
  BUILD_JOBS[$variant]=$SUBMITTED_JOB_ID
done
for variant in "${VARIANTS[@]}"; do
  scene=${VARIANT_SCENE[$variant]}
  dependency="afterok:${BUILD_JOBS[$variant]}"
  submit_job eval-policy "$variant" "$scene" A0 "$dependency" 4 64G 01:00:00 \
    icra-e4-robustx-eval eval-policy "$variant" A0
  EVAL_A0_JOBS[$variant]=$SUBMITTED_JOB_ID
  submit_job eval-policy "$variant" "$scene" A4 "$dependency" 4 64G 01:00:00 \
    icra-e4-robustx-eval eval-policy "$variant" A4
  EVAL_A4_JOBS[$variant]=$SUBMITTED_JOB_ID
done
for variant in "${VARIANTS[@]}"; do
  scene=${VARIANT_SCENE[$variant]}
  dependency="afterany:${EVAL_A0_JOBS[$variant]}:${EVAL_A4_JOBS[$variant]}"
  submit_job compare-pair "$variant" "$scene" paired "$dependency" \
    2 16G 00:30:00 icra-e4-robustx-compare compare-pair "$variant"
  COMPARE_JOBS[$variant]=$SUBMITTED_JOB_ID
done
compare_ids=()
for variant in "${VARIANTS[@]}"; do compare_ids+=("${COMPARE_JOBS[$variant]}"); done
aggregate_dependency=$(IFS=:; printf 'afterany:%s' "${compare_ids[*]}")
submit_job aggregate - - - "$aggregate_dependency" 2 16G 00:30:00 \
  icra-e4-robustx-aggregate aggregate
AGGREGATE_JOB=$SUBMITTED_JOB_ID

[[ $SEQUENCE -eq 33 ]] || {
  echo "internal error: expected 33 jobs, recorded $SEQUENCE" >&2; exit 2;
}
chmod 0444 "$LEDGER"; sync -f "$LEDGER"; sync -f "$SUBMISSION_DIR"
ledger_sha256=$(sha256sum "$LEDGER" | awk '{print $1}')
echo "submitted 33 ordinary CPU robust extension jobs (no Slurm arrays, no GPU)"
echo "common_build_jobs=8"
echo "policy_eval_jobs=16"
echo "pair_compare_jobs=8"
echo "aggregate_job=$AGGREGATE_JOB"
echo "external_winner=d755b3d9d8/d755-fs"
echo "ledger=$LEDGER"
echo "ledger_sha256=$ledger_sha256"
