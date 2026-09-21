#!/usr/bin/env bash
# Submit the complete GT-isolated E3 DAG as ordinary Slurm jobs:
# E0 -> inventory -> synthetic GPU/MVPY smoke -> one complete real RGB+ED
# pilot observe -> pilot control -> pilot evaluate -> 49 remaining GPU
# observes, with every remaining scene continuing through its own CPU control
# -> CPU evaluate chain -> aggregate. The pilot supplies one job in each of
# the 50-scene phase collections, so the DAG remains 154 ordinary jobs.
#
# Required:
#   E3_FREEZE_ID=icra2027-contract-v1-e3-<tag> \
#     bash run/icra2027/submit_e3_noarray.sh
# Select Hala explicitly with E3_GPU_PROFILE=hala-a6000. The default remains
# gcp-a100 for compatibility with the original frozen submission contract.
#
# Every accepted job ID is immediately persisted by atomic replacement of a
# repository-local ledger. A partial submission is therefore recoverable, but
# this script intentionally refuses to append to or replay an existing ledger.
set -euo pipefail

ROOT=/group/worldcept/PhiRIE/code/SimAny
PYTHON="$ROOT/.venv/bin/python"
ROSTER="$ROOT/configs/experiments/icra2027/construction_regimes.yaml"
JOBS="$ROOT/configs/experiments/icra2027/agentic_jobs.yaml"
POLICIES="$ROOT/configs/experiments/icra2027/agentic_policies.yaml"
E0_LAUNCHER="$ROOT/run/slurm/icra2027_e0_preflight.sbatch"
E3_LAUNCHER="$ROOT/run/slurm/icra2027_e3_agentic.sbatch"
SBATCH_BIN=${SBATCH_BIN:-sbatch}
ACCOUNT=${E3_ACCOUNT:-runyi_yang}
CPU_NODELIST=${E3_CPU_NODELIST:-sof1-h200-[0-7]}
GPU_PROFILE=${E3_GPU_PROFILE:-gcp-a100}
GPU_NODE=""
GPU_GRES=""
GPU_SUMMARY=""
declare -a GPU_RESOURCE_ARGS
case "$GPU_PROFILE" in
  gcp-a100)
    GPU_NODE=gcp-eu1-a100-80g-qrfh
    GPU_GRES=a100-80g:1
    GPU_SUMMARY="GCP A100-SXM4-80GB"
    GPU_RESOURCE_ARGS=(--nodelist="$GPU_NODE" --gpus="$GPU_GRES")
    ;;
  hala-a6000)
    GPU_NODE=hala
    GPU_GRES=gpu:a6000:1
    GPU_SUMMARY="Hala NVIDIA RTX A6000"
    GPU_RESOURCE_ARGS=(--nodelist="$GPU_NODE" --gres="$GPU_GRES")
    ;;
  *)
    echo "unsupported E3_GPU_PROFILE=$GPU_PROFILE (expected gcp-a100 or hala-a6000)" >&2
    exit 2
    ;;
esac
PILOT_SCENE=5748ce6f01

: "${E3_FREEZE_ID:?set a fresh E3_FREEZE_ID before submitting}"
[[ "$E3_FREEZE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "E3_FREEZE_ID contains unsafe path characters: $E3_FREEZE_ID" >&2
  exit 2
}
[[ "$ACCOUNT" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "E3_ACCOUNT contains unsafe characters: $ACCOUNT" >&2
  exit 2
}
[[ "$CPU_NODELIST" =~ ^sof1-h200-\[[0-7,-]+\]$ \
  || "$CPU_NODELIST" =~ ^sof1-h200-[0-7]$ ]] || {
  echo "E3_CPU_NODELIST must select only sof1-h200-[0-7] hosts: $CPU_NODELIST" >&2
  exit 2
}

for ARRAY_VARIABLE in \
  SLURM_ARRAY_JOB_ID \
  SLURM_ARRAY_TASK_ID \
  SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN \
  SLURM_ARRAY_TASK_MAX \
  SLURM_ARRAY_TASK_STEP \
  SBATCH_ARRAY_INX \
  SBATCH_ARRAY
do
  if [[ -v "$ARRAY_VARIABLE" ]]; then
    echo "E3 submitter refuses Slurm array context: $ARRAY_VARIABLE is set" >&2
    exit 2
  fi
done

cd "$ROOT"
[[ -x "$PYTHON" ]] || { echo "missing E3 Python: $PYTHON" >&2; exit 2; }
command -v "$SBATCH_BIN" >/dev/null || {
  echo "sbatch executable not found: $SBATCH_BIN" >&2
  exit 2
}
for INPUT in "$ROSTER" "$JOBS" "$POLICIES" "$E0_LAUNCHER" "$E3_LAUNCHER"; do
  [[ -s "$INPUT" && ! -L "$INPUT" ]] || {
    echo "missing, empty, or symlinked submission input: $INPUT" >&2
    exit 2
  }
done

CODE_COMMIT=$(git rev-parse HEAD)
[[ "$CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "git did not return an exact 40-character commit" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo "refusing E3 submission from a dirty worktree" >&2
  git status --short >&2
  exit 2
}

mapfile -t SCENES < <(
  "$PYTHON" - "$ROSTER" "$JOBS" "$PILOT_SCENE" <<'PY'
import re
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = yaml.safe_load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    jobs = yaml.safe_load(handle)
pilot_scene = sys.argv[3]
population = payload.get("population", {})
scenes = [str(value).strip() for value in population.get("scene_ids", [])]
if population.get("planned_scenes") != 50:
    raise SystemExit("E3 roster must declare planned_scenes=50")
if len(scenes) != 50 or len(set(scenes)) != 50:
    raise SystemExit("E3 roster must contain exactly 50 unique scenes")
if not all(re.fullmatch(r"[0-9a-f]{10}", value) for value in scenes):
    raise SystemExit("E3 roster contains an invalid scene ID")
if pilot_scene not in scenes:
    raise SystemExit("E3 RGB+ED pilot is absent from the frozen scene roster")
snapshots = {
    str(row.get("scene_id")): row
    for row in jobs.get("population", {}).get("scene_snapshots", [])
}
pilot = snapshots.get(pilot_scene)
if not isinstance(pilot, dict) or pilot.get("accepted_jobs") != 1:
    raise SystemExit("E3 RGB+ED pilot must have exactly one frozen object")
for scene in scenes:
    print(scene)
PY
)
[[ ${#SCENES[@]} -eq 50 ]] || {
  echo "E3 roster parser did not return exactly 50 scenes" >&2
  exit 2
}

SUBMISSION_DIR="$ROOT/outputs/icra2027/submissions/$E3_FREEZE_ID"
LEDGER="$SUBMISSION_DIR/jobs.tsv"
LOG_DIR="$ROOT/outputs/icra2027/slurm"
EXPERIMENT_ROOT="$ROOT/outputs/icra2027/$E3_FREEZE_ID"
[[ ! -e "$EXPERIMENT_ROOT" && ! -L "$EXPERIMENT_ROOT" ]] || {
  echo "refusing to reuse E3 experiment root: $EXPERIMENT_ROOT" >&2
  exit 2
}
[[ ! -e "$SUBMISSION_DIR" && ! -L "$SUBMISSION_DIR" ]] || {
  echo "refusing to reuse E3 submission directory: $SUBMISSION_DIR" >&2
  exit 2
}
mkdir -p "$ROOT/outputs/icra2027/submissions" "$LOG_DIR"
mkdir "$SUBMISSION_DIR"

SEQUENCE=0
LEDGER_TEMP=""
on_exit() {
  local rc=$?
  if [[ -n "$LEDGER_TEMP" && -f "$LEDGER_TEMP" && ! -L "$LEDGER_TEMP" ]]; then
    rm -f -- "$LEDGER_TEMP"
  fi
  if [[ $rc -ne 0 ]]; then
    echo "E3 submission stopped after $SEQUENCE recorded jobs; partial ledger: $LEDGER" >&2
  fi
}
trap on_exit EXIT

LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.initialize.$$"
(
  umask 077
  printf 'sequence\tsubmitted_utc\tstage\tphase\tscene_id\tjob_id\tdependency\tprofile\tcpus\tmem\ttime\tpartition\tqos\tnodelist\tgres\tlauncher\n' \
    > "$LEDGER_TEMP"
)
mv -T -- "$LEDGER_TEMP" "$LEDGER"
LEDGER_TEMP=""

record_job() {
  local stage="$1"
  local phase="$2"
  local scene_id="$3"
  local job_id="$4"
  local dependency="$5"
  local profile="$6"
  local cpus="$7"
  local time_limit="$8"
  local nodelist="$9"
  local gres="${10}"
  local launcher="${11}"
  local submitted_utc
  submitted_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  SEQUENCE=$((SEQUENCE + 1))
  LEDGER_TEMP="$SUBMISSION_DIR/.jobs.tsv.${SEQUENCE}.$$"
  cp -- "$LEDGER" "$LEDGER_TEMP"
  printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t64G\t%s\tbatch\tnormal\t%s\t%s\t%s\n' \
    "$SEQUENCE" "$submitted_utc" "$stage" "$phase" "$scene_id" \
    "$job_id" "$dependency" "$profile" "$cpus" "$time_limit" "$nodelist" \
    "$gres" "$launcher" >> "$LEDGER_TEMP"
  mv -T -- "$LEDGER_TEMP" "$LEDGER"
  LEDGER_TEMP=""
}

validate_job_id() {
  local raw="$1"
  local job_id
  [[ "$raw" =~ ^([0-9]+)(\;[A-Za-z0-9._-]+)?$ ]] || {
    echo "sbatch did not return one ordinary numeric job ID: $raw" >&2
    return 2
  }
  job_id="${BASH_REMATCH[1]}"
  if awk -F '\t' -v id="$job_id" 'NR > 1 && $6 == id {found=1} END {exit !found}' "$LEDGER"; then
    echo "duplicate Slurm job ID returned during E3 submission: $job_id" >&2
    return 2
  fi
  SUBMITTED_JOB_ID="$job_id"
}

submit_e0() {
  local raw
  raw=$("$SBATCH_BIN" --parsable --no-requeue \
    --account="$ACCOUNT" \
    --job-name=icra-e3-e0 \
    --partition=batch --qos=normal --nodelist="$CPU_NODELIST" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --time=00:30:00 \
    --output="$LOG_DIR/e3-e0-%j.out" \
    --error="$LOG_DIR/e3-e0-%j.err" \
    "$E0_LAUNCHER")
  validate_job_id "$raw"
  record_job e0 e0 - "$SUBMITTED_JOB_ID" - sof1-cpu 8 00:30:00 \
    "$CPU_NODELIST" none "${E0_LAUNCHER#"$ROOT"/}"
}

submit_e3() {
  local stage="$1"
  local phase="$2"
  local scene_id="$3"
  local cpus="$4"
  local time_limit="$5"
  local dependency="$6"
  local require_nonempty_proposals="${7:-0}"
  local job_name="icra-e3-$stage"
  local log_tag="e3-$stage"
  local profile
  local nodelist
  local gres
  local export_spec
  local raw
  local -a resource_args
  if [[ "$phase" == smoke || "$phase" == observe ]]; then
    profile="$GPU_PROFILE"
    nodelist="$GPU_NODE"
    gres="$GPU_GRES"
    resource_args=("${GPU_RESOURCE_ARGS[@]}")
  else
    profile=sof1-cpu
    nodelist="$CPU_NODELIST"
    gres=none
    resource_args=(--nodelist="$CPU_NODELIST")
  fi
  export_spec="ALL,E3_PHASE=$phase,E3_FREEZE_ID=$E3_FREEZE_ID,E3_CODE_COMMIT=$CODE_COMMIT,E3_E0_JOB_ID=$E0_JOB,E3_E0_LOG=$E0_LOG"
  if [[ "$phase" == smoke || "$phase" == observe ]]; then
    export_spec+=",E3_GPU_PROFILE=$GPU_PROFILE"
  fi
  if [[ "$phase" == observe || "$phase" == control || "$phase" == evaluate ]]; then
    export_spec+=",E3_SCENE_ID=$scene_id"
  fi
  case "$require_nonempty_proposals" in
    0) ;;
    1)
      [[ "$stage" == "control-pilot-$PILOT_SCENE" \
        && "$phase" == control \
        && "$scene_id" == "$PILOT_SCENE" ]] || {
        echo "internal error: nonempty-proposal postcondition is pilot-control only" >&2
        return 2
      }
      export_spec+=",E3_REQUIRE_NONEMPTY_PROPOSALS=1"
      ;;
    *)
      echo "internal error: invalid nonempty-proposal postcondition value" >&2
      return 2
      ;;
  esac
  raw=$("$SBATCH_BIN" --parsable --no-requeue \
    --account="$ACCOUNT" \
    --job-name="$job_name" \
    --partition=batch --qos=normal "${resource_args[@]}" \
    --nodes=1 --ntasks=1 --cpus-per-task="$cpus" --mem=64G --time="$time_limit" \
    --dependency="afterok:$dependency" --kill-on-invalid-dep=yes \
    --output="$LOG_DIR/$log_tag-%j.out" \
    --error="$LOG_DIR/$log_tag-%j.err" \
    --export="$export_spec" \
    "$E3_LAUNCHER")
  validate_job_id "$raw"
  record_job "$stage" "$phase" "$scene_id" "$SUBMITTED_JOB_ID" \
    "afterok:$dependency" "$profile" "$cpus" "$time_limit" "$nodelist" \
    "$gres" "${E3_LAUNCHER#"$ROOT"/}"
}

# Avoid leaking stale phase/scene/contract values through --export=ALL.
unset E3_PHASE E3_SCENE_ID E3_CODE_COMMIT E3_CONTRACT_MANIFEST \
  E3_E0_JOB_ID E3_E0_LOG E3_GPU_PROFILE E3_REQUIRE_NONEMPTY_PROPOSALS

SUBMITTED_JOB_ID=""
submit_e0
E0_JOB="$SUBMITTED_JOB_ID"
E0_LOG="$LOG_DIR/e3-e0-${E0_JOB}.out"

submit_e3 inventory inventory - 8 01:00:00 "$E0_JOB"
INVENTORY_JOB="$SUBMITTED_JOB_ID"

submit_e3 smoke smoke - 8 00:30:00 "$INVENTORY_JOB"
SMOKE_JOB="$SUBMITTED_JOB_ID"

submit_e3 "observe-pilot-$PILOT_SCENE" observe "$PILOT_SCENE" 8 02:00:00 "$SMOKE_JOB"
PILOT_OBSERVE_JOB="$SUBMITTED_JOB_ID"

submit_e3 "control-pilot-$PILOT_SCENE" control "$PILOT_SCENE" 8 04:00:00 \
  "$PILOT_OBSERVE_JOB" 1
PILOT_CONTROL_JOB="$SUBMITTED_JOB_ID"

submit_e3 "evaluate-pilot-$PILOT_SCENE" evaluate "$PILOT_SCENE" 8 02:00:00 "$PILOT_CONTROL_JOB"
PILOT_EVALUATE_JOB="$SUBMITTED_JOB_ID"

OBSERVE_JOBS=("$PILOT_OBSERVE_JOB")
CONTROL_JOBS=("$PILOT_CONTROL_JOB")
EVALUATE_JOBS=("$PILOT_EVALUATE_JOB")
for SCENE_ID in "${SCENES[@]}"; do
  if [[ "$SCENE_ID" == "$PILOT_SCENE" ]]; then
    continue
  fi

  submit_e3 "observe-$SCENE_ID" observe "$SCENE_ID" 8 02:00:00 "$PILOT_EVALUATE_JOB"
  OBSERVE_JOB="$SUBMITTED_JOB_ID"
  OBSERVE_JOBS+=("$OBSERVE_JOB")

  submit_e3 "control-$SCENE_ID" control "$SCENE_ID" 8 04:00:00 "$OBSERVE_JOB"
  CONTROL_JOB="$SUBMITTED_JOB_ID"
  CONTROL_JOBS+=("$CONTROL_JOB")

  submit_e3 "evaluate-$SCENE_ID" evaluate "$SCENE_ID" 8 02:00:00 "$CONTROL_JOB"
  EVALUATE_JOBS+=("$SUBMITTED_JOB_ID")
done
[[ ${#OBSERVE_JOBS[@]} -eq 50 \
  && ${#CONTROL_JOBS[@]} -eq 50 \
  && ${#EVALUATE_JOBS[@]} -eq 50 ]] || {
  echo "internal error: expected 50 ordinary jobs for each scene phase" >&2
  exit 2
}
EVALUATE_DEPENDENCY=$(IFS=:; printf '%s' "${EVALUATE_JOBS[*]}")

submit_e3 aggregate aggregate - 16 01:00:00 "$EVALUATE_DEPENDENCY"
AGGREGATE_JOB="$SUBMITTED_JOB_ID"

[[ $SEQUENCE -eq 154 ]] || {
  echo "internal error: expected 154 recorded ordinary jobs, found $SEQUENCE" >&2
  exit 2
}

OBSERVE_DEPENDENCY=$(IFS=:; printf '%s' "${OBSERVE_JOBS[*]}")
CONTROL_DEPENDENCY=$(IFS=:; printf '%s' "${CONTROL_JOBS[*]}")
printf 'E3 submission complete: 154 ordinary jobs (51 %s GPU, 103 CPU), no arrays\n' \
  "$GPU_PROFILE"
printf 'gpu_hardware=%s gpu_node=%s gpu_gres=%s\n' \
  "$GPU_SUMMARY" "$GPU_NODE" "$GPU_GRES"
printf 'freeze_id=%s\n' "$E3_FREEZE_ID"
printf 'code_commit=%s\n' "$CODE_COMMIT"
printf 'e0=%s inventory=%s smoke=%s aggregate=%s\n' \
  "$E0_JOB" "$INVENTORY_JOB" "$SMOKE_JOB" "$AGGREGATE_JOB"
printf 'rgb_ed_pilot_scene=%s observe=%s control=%s evaluate=%s\n' \
  "$PILOT_SCENE" "$PILOT_OBSERVE_JOB" "$PILOT_CONTROL_JOB" \
  "$PILOT_EVALUATE_JOB"
printf 'observe_jobs=%s\n' "$OBSERVE_DEPENDENCY"
printf 'control_jobs=%s\n' "$CONTROL_DEPENDENCY"
printf 'evaluate_jobs=%s\n' "$EVALUATE_DEPENDENCY"
printf 'ledger=%s\n' "$LEDGER"
printf 'monitor: bash run/icra2027/monitor_e3_noarray.sh --ledger %q --watch\n' \
  "$LEDGER"
