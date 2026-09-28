#!/usr/bin/env bash
# One ordinary CPU worker for the sealed E4 collision-repair fleet.
set -euo pipefail
umask 077

export PATH=/usr/local/bin:/usr/bin:/bin
CODE_ROOT=${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot
EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}
PYTHON=${SIMANY_ROOT:-$PWD}/.venv/bin/python
PILOT_SCRIPT="$CODE_ROOT/run/icra2027/e4_collision_repair_pilot.py"
EXPECTED_E3_ROOT=outputs/icra2027/icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic
ACCOUNT=${SLURM_ACCOUNT:-phirie}
SETFACL=/usr/bin/setfacl
GETFACL=/usr/bin/getfacl

: "${E4_REPAIR_ID:?E4_REPAIR_ID is required}"
: "${E4_CODE_COMMIT:?E4_CODE_COMMIT is required}"
: "${E4_E3_ROOT:?E4_E3_ROOT is required}"
: "${E4_SUBMIT_USER:?E4_SUBMIT_USER is required}"
[[ "$E4_REPAIR_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_REPAIR_ID: $E4_REPAIR_ID" >&2
  exit 2
}
[[ "$E4_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "E4_CODE_COMMIT must be a full lowercase SHA" >&2
  exit 2
}
[[ "$E4_E3_ROOT" == "$EXPECTED_E3_ROOT" ]] || {
  echo "E4_E3_ROOT differs from the sealed E3 source" >&2
  exit 2
}
[[ "$E4_SUBMIT_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe submit user: $E4_SUBMIT_USER" >&2
  exit 2
}
[[ "$(id -un)" == "$E4_SUBMIT_USER" ]] || {
  echo "worker user differs from submit user" >&2
  exit 2
}

for array_variable in \
  SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP \
  SBATCH_ARRAY_INX SBATCH_ARRAY
do
  if [[ -v "$array_variable" ]]; then
    echo "repair worker refuses Slurm array context: $array_variable is set" >&2
    exit 2
  fi
done

[[ -x "$PYTHON" && -f "$PILOT_SCRIPT" && ! -L "$PILOT_SCRIPT" ]] || {
  echo "missing Python or repair-pilot script" >&2
  exit 2
}
[[ -d "$EVIDENCE_ROOT" && ! -L "$EVIDENCE_ROOT" && "$CODE_ROOT" != "$EVIDENCE_ROOT" ]] || {
  echo "invalid dual-root binding" >&2
  exit 2
}
[[ -x "$SETFACL" && -x "$GETFACL" ]] || {
  echo "setfacl/getfacl are required" >&2
  exit 2
}
[[ -d "$EVIDENCE_ROOT/$E4_E3_ROOT" && ! -L "$EVIDENCE_ROOT/$E4_E3_ROOT" ]] || {
  echo "missing sealed E3 root" >&2
  exit 2
}

observed_root=$(git -C "$CODE_ROOT" rev-parse --show-toplevel)
observed_commit=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$observed_root" == "$CODE_ROOT" && "$observed_commit" == "$E4_CODE_COMMIT" ]] || {
  echo "repair worker code root/commit drift" >&2
  exit 2
}
if [[ -n "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "repair worker refuses a dirty code worktree" >&2
  git -C "$CODE_ROOT" status --short >&2
  exit 2
fi

PILOT_SCENES=(3db0a1c8f3 d755b3d9d8)
is_pilot_scene() {
  local candidate=$1 scene
  for scene in "${PILOT_SCENES[@]}"; do
    [[ "$candidate" == "$scene" ]] && return 0
  done
  return 1
}

phase=${1:-}
case "$phase" in
  materialize)
    [[ $# -eq 4 ]] || { echo "usage: $0 materialize room|shim SCENE A0|A4" >&2; exit 2; }
    mode=$2 scene_id=$3 policy_id=$4
    [[ "$mode" == room || "$mode" == shim ]] || { echo "invalid mode" >&2; exit 2; }
    is_pilot_scene "$scene_id" || { echo "invalid pilot scene" >&2; exit 2; }
    [[ "$policy_id" == A0 || "$policy_id" == A4 ]] || { echo "invalid policy" >&2; exit 2; }
    expected_cpus=8 expected_mem_mb=65536 expected_mem_label=64G
    expected_time=01:00:00 expected_job_name=icra-e4-repair-materialize
    ;;
  export)
    [[ $# -eq 4 ]] || { echo "usage: $0 export room|shim SCENE A0|A4" >&2; exit 2; }
    mode=$2 scene_id=$3 policy_id=$4
    [[ "$mode" == room || "$mode" == shim ]] || { echo "invalid mode" >&2; exit 2; }
    is_pilot_scene "$scene_id" || { echo "invalid pilot scene" >&2; exit 2; }
    [[ "$policy_id" == A0 || "$policy_id" == A4 ]] || { echo "invalid policy" >&2; exit 2; }
    expected_cpus=16 expected_mem_mb=98304 expected_mem_label=96G
    expected_time=04:00:00 expected_job_name=icra-e4-repair-export
    ;;
  validate)
    [[ $# -eq 4 ]] || { echo "usage: $0 validate room|shim SCENE A0|A4" >&2; exit 2; }
    mode=$2 scene_id=$3 policy_id=$4
    [[ "$mode" == room || "$mode" == shim ]] || { echo "invalid mode" >&2; exit 2; }
    is_pilot_scene "$scene_id" || { echo "invalid pilot scene" >&2; exit 2; }
    [[ "$policy_id" == A0 || "$policy_id" == A4 ]] || { echo "invalid policy" >&2; exit 2; }
    expected_cpus=4 expected_mem_mb=32768 expected_mem_label=32G
    expected_time=01:00:00 expected_job_name=icra-e4-repair-validate
    ;;
  compare)
    [[ $# -eq 3 ]] || { echo "usage: $0 compare SCENE A0|A4" >&2; exit 2; }
    scene_id=$2 policy_id=$3
    is_pilot_scene "$scene_id" || { echo "invalid pilot scene" >&2; exit 2; }
    [[ "$policy_id" == A0 || "$policy_id" == A4 ]] || { echo "invalid policy" >&2; exit 2; }
    expected_cpus=2 expected_mem_mb=16384 expected_mem_label=16G
    expected_time=00:30:00 expected_job_name=icra-e4-repair-compare
    ;;
  aggregate)
    [[ $# -eq 1 ]] || { echo "usage: $0 aggregate" >&2; exit 2; }
    expected_cpus=2 expected_mem_mb=16384 expected_mem_label=16G
    expected_time=00:30:00 expected_job_name=icra-e4-repair-aggregate
    ;;
  *) echo "unknown repair phase: ${phase:-<empty>}" >&2; exit 2 ;;
esac

: "${SLURM_JOB_ID:?repair worker must run inside Slurm}"
[[ "${SLURM_JOB_ACCOUNT:-}" == "$ACCOUNT" \
  && "${SLURM_JOB_PARTITION:-}" == batch \
  && "${SLURM_JOB_QOS:-}" == normal ]] || {
  echo "repair allocation account/partition/QoS differs" >&2
  exit 2
}
[[ "${SLURMD_NODENAME:-}" =~ ^sof1-h200-[0-7]$ ]] || {
  echo "repair job is outside sof1-h200-[0-7]" >&2
  exit 2
}
[[ "${SLURM_NNODES:-}" == 1 && "${SLURM_NTASKS:-}" == 1 \
  && "${SLURM_CPUS_PER_TASK:-}" == "$expected_cpus" ]] || {
  echo "repair CPU allocation differs" >&2
  exit 2
}
case "${SLURM_MEM_PER_NODE:-}" in
  "$expected_mem_mb"|"$expected_mem_label") ;;
  *) echo "repair memory allocation differs" >&2; exit 2 ;;
esac
for gpu_variable in \
  CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES \
  SLURM_JOB_GPUS SLURM_STEP_GPUS SLURM_GPUS SLURM_GPUS_ON_NODE
do
  if [[ -n "${!gpu_variable-}" && "${!gpu_variable}" != NoDevFiles ]]; then
    echo "repair allocation is not CPU-only: $gpu_variable=${!gpu_variable}" >&2
    exit 2
  fi
done

job_record=$(scontrol show job "$SLURM_JOB_ID" --oneliner)
[[ "$job_record" == *" JobName=$expected_job_name "* \
  && "$job_record" == *" Partition=batch "* \
  && "$job_record" == *" QOS=normal "* \
  && "$job_record" == *" Account=$ACCOUNT "* \
  && "$job_record" == *" UserId=$E4_SUBMIT_USER("* \
  && "$job_record" == *" Requeue=0 "* ]] || {
  echo "repair scontrol identity/provenance differs: $job_record" >&2
  exit 2
}
[[ "$job_record" == *" TimeLimit=$expected_time "* \
  && "$job_record" == *" NumNodes=1 "* \
  && "$job_record" == *" NumCPUs=$expected_cpus "* \
  && "$job_record" == *" NumTasks=1 "* \
  && "$job_record" == *" CPUs/Task=$expected_cpus "* ]] || {
  echo "repair scontrol resource contract differs: $job_record" >&2
  exit 2
}
case "$job_record" in
  *" MinMemoryNode=$expected_mem_label "*|*" MinMemoryNode=${expected_mem_mb}M "*|*" MinMemoryNode=$expected_mem_mb "*) ;;
  *) echo "repair scontrol memory contract differs: $job_record" >&2; exit 2 ;;
esac
[[ "$job_record" == *" Command=$CODE_ROOT/run/slurm/icra2027_e4_collision_repair_cpu.sbatch "* ]] || {
  echo "repair launcher command differs" >&2
  exit 2
}
job_record_lower=${job_record,,}
if [[ "$job_record_lower" == *"gres/gpu"* \
  || "$job_record_lower" == *"trespernode=gres"* \
  || "$job_record_lower" == *" gres="* ]]; then
  echo "repair job unexpectedly owns GRES resources" >&2
  exit 2
fi

unset PYTHONPATH PYTHONOPTIMIZE PYTHONUSERBASE RESUME AUTO FULL
unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
while IFS= read -r inherited_variable; do unset "$inherited_variable"; done \
  < <(compgen -A variable SIMANY_ || true)
while IFS= read -r inherited_variable; do unset "$inherited_variable"; done \
  < <(compgen -A variable SIMF_ || true)
export PYTHONPATH="$CODE_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export SIMANY_EVIDENCE_ROOT="$EVIDENCE_ROOT"
export OMP_NUM_THREADS="$expected_cpus"

submission_dir="$EVIDENCE_ROOT/outputs/icra2027/submissions/${E4_REPAIR_ID}-collision-repair-fleet"
runtime_parent="$submission_dir/runtime"
[[ -d "$runtime_parent" && ! -L "$runtime_parent" \
  && "$(stat -c '%a' -- "$runtime_parent")" == 700 \
  && "$("$GETFACL" -cp -- "$runtime_parent")" != *"default:"* ]] || {
  echo "missing private repair runtime parent" >&2
  exit 2
}
runtime_root="$runtime_parent/$SLURM_JOB_ID"
mkdir "$runtime_root"
"$SETFACL" -k -- "$runtime_root"
chmod 00700 "$runtime_root"
mkdir "$runtime_root/tmp" "$runtime_root/xdg-cache"
for runtime_directory in "$runtime_root" "$runtime_root/tmp" "$runtime_root/xdg-cache"; do
  "$SETFACL" -k -- "$runtime_directory"
  chmod 00700 "$runtime_directory"
  [[ -d "$runtime_directory" && ! -L "$runtime_directory" \
    && "$(stat -c '%a' -- "$runtime_directory")" == 700 \
    && "$("$GETFACL" -cp -- "$runtime_directory")" != *"default:"* ]] || {
    echo "repair runtime directory is not private" >&2
    exit 2
  }
done
export TMPDIR="$runtime_root/tmp"
export XDG_CACHE_HOME="$runtime_root/xdg-cache"
cd "$CODE_ROOT"

"$PYTHON" - "$PILOT_SCRIPT" "$CODE_ROOT" "$EVIDENCE_ROOT" <<'PY'
import importlib.util
import sys
from pathlib import Path

script, code_root, evidence_root = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("sealed_e4_collision_repair_pilot", script)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load repair-pilot script")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if module.CODE_ROOT != code_root or module.EXPECTED_EVIDENCE_ROOT != evidence_root:
    raise SystemExit("repair-pilot script dual-root constants differ")
if tuple(module.PILOT_SCENES) != ("3db0a1c8f3", "d755b3d9d8"):
    raise SystemExit("repair-pilot scene roster differs")
if tuple(module.POLICIES) != ("A0", "A4") or tuple(module.MODES) != ("room", "shim"):
    raise SystemExit("repair-pilot axis roster differs")
PY

common_args=(--repair-id "$E4_REPAIR_ID" --expected-code-commit "$E4_CODE_COMMIT")
case "$phase" in
  materialize)
    exec "$PYTHON" -m robo.eval.e4_candidate_screen materialize \
      --screen-id "${E4_REPAIR_ID}-${mode}" \
      --scene-id "$scene_id" --policy-id "$policy_id" \
      --e3-root "$E4_E3_ROOT" --expected-code-commit "$E4_CODE_COMMIT"
    ;;
  export)
    exec "$PYTHON" "$PILOT_SCRIPT" export "${common_args[@]}" \
      --collision-mode "$mode" --scene-id "$scene_id" --policy-id "$policy_id"
    ;;
  validate)
    exec "$PYTHON" "$PILOT_SCRIPT" validate-export "${common_args[@]}" \
      --collision-mode "$mode" --scene-id "$scene_id" --policy-id "$policy_id"
    ;;
  compare)
    exec "$PYTHON" "$PILOT_SCRIPT" compare "${common_args[@]}" \
      --scene-id "$scene_id" --policy-id "$policy_id"
    ;;
  aggregate)
    exec "$PYTHON" "$PILOT_SCRIPT" aggregate "${common_args[@]}"
    ;;
esac
