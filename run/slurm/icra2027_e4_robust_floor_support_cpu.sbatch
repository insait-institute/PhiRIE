#!/usr/bin/env bash
# One ordinary CPU worker for the sealed 17-job robust-floor/support DAG.
set -euo pipefail
umask 077

export PATH=/usr/local/bin:/usr/bin:/bin
CODE_ROOT=/group/worldcept/PhiRIE/code/SimAny-wt/e4-paired-pilot
EVIDENCE_ROOT=/group/worldcept/PhiRIE/code/SimAny
PYTHON=/group/worldcept/PhiRIE/code/SimAny/.venv/bin/python
SWEEP_SCRIPT="$CODE_ROOT/run/icra2027/e4_robust_floor_support_diagnostic.py"
EXPECTED_E3_ROOT=outputs/icra2027/icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic
ACCOUNT=runyi_yang
SETFACL=/usr/bin/setfacl
GETFACL=/usr/bin/getfacl

: "${E4_ROBUST_SWEEP_ID:?E4_ROBUST_SWEEP_ID is required}"
: "${E4_CODE_COMMIT:?E4_CODE_COMMIT is required}"
: "${E4_E3_ROOT:?E4_E3_ROOT is required}"
: "${E4_SUBMIT_USER:?E4_SUBMIT_USER is required}"
[[ "$E4_ROBUST_SWEEP_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_ROBUST_SWEEP_ID" >&2; exit 2;
}
[[ "$E4_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "E4_CODE_COMMIT must be a full lowercase SHA" >&2; exit 2;
}
[[ "$E4_E3_ROOT" == "$EXPECTED_E3_ROOT" ]] || {
  echo "E4_E3_ROOT differs from the sealed E3 source" >&2; exit 2;
}
[[ "$E4_SUBMIT_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
  && "$(id -un)" == "$E4_SUBMIT_USER" ]] || {
  echo "worker user differs from submit user" >&2; exit 2;
}
for variable in SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP \
  SBATCH_ARRAY_INX SBATCH_ARRAY; do
  [[ ! -v "$variable" ]] || {
    echo "robust diagnostic worker refuses Slurm array context: $variable" >&2
    exit 2
  }
done
[[ -x "$PYTHON" && -f "$SWEEP_SCRIPT" && ! -L "$SWEEP_SCRIPT" ]] || {
  echo "missing Python or robust diagnostic runner" >&2; exit 2;
}
[[ -d "$EVIDENCE_ROOT" && ! -L "$EVIDENCE_ROOT" \
  && "$CODE_ROOT" != "$EVIDENCE_ROOT" ]] || {
  echo "invalid dual-root binding" >&2; exit 2;
}
[[ -x "$SETFACL" && -x "$GETFACL" \
  && -d "$EVIDENCE_ROOT/$E4_E3_ROOT" \
  && ! -L "$EVIDENCE_ROOT/$E4_E3_ROOT" ]] || {
  echo "missing ACL tools or sealed E3 root" >&2; exit 2;
}

observed_root=$(git -C "$CODE_ROOT" rev-parse --show-toplevel)
observed_commit=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$observed_root" == "$CODE_ROOT" && "$observed_commit" == "$E4_CODE_COMMIT" ]] || {
  echo "robust diagnostic worker code root/commit drift" >&2; exit 2;
}
if [[ -n "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "robust diagnostic worker refuses a dirty code worktree" >&2
  git -C "$CODE_ROOT" status --short >&2
  exit 2
fi

VARIANTS=(3db-f 3db-fs d755-f d755-fs)
is_variant() {
  local candidate=$1 variant
  for variant in "${VARIANTS[@]}"; do [[ "$candidate" == "$variant" ]] && return 0; done
  return 1
}
phase=${1:-}
case "$phase" in
  build-common)
    [[ $# -eq 2 ]] || { echo "usage: $0 build-common VARIANT" >&2; exit 2; }
    variant_id=$2; is_variant "$variant_id" || { echo "invalid variant" >&2; exit 2; }
    expected_cpus=16 expected_mem_mb=98304 expected_mem_label=96G
    expected_time=04:00:00 expected_job_name=icra-e4-robust-build
    ;;
  eval-policy)
    [[ $# -eq 3 ]] || { echo "usage: $0 eval-policy VARIANT A0|A4" >&2; exit 2; }
    variant_id=$2 policy_id=$3
    is_variant "$variant_id" || { echo "invalid variant" >&2; exit 2; }
    [[ "$policy_id" == A0 || "$policy_id" == A4 ]] || { echo "invalid policy" >&2; exit 2; }
    expected_cpus=4 expected_mem_mb=65536 expected_mem_label=64G
    expected_time=01:00:00 expected_job_name=icra-e4-robust-eval
    ;;
  compare-pair)
    [[ $# -eq 2 ]] || { echo "usage: $0 compare-pair VARIANT" >&2; exit 2; }
    variant_id=$2; is_variant "$variant_id" || { echo "invalid variant" >&2; exit 2; }
    expected_cpus=2 expected_mem_mb=16384 expected_mem_label=16G
    expected_time=00:30:00 expected_job_name=icra-e4-robust-compare
    ;;
  aggregate)
    [[ $# -eq 1 ]] || { echo "usage: $0 aggregate" >&2; exit 2; }
    expected_cpus=2 expected_mem_mb=16384 expected_mem_label=16G
    expected_time=00:30:00 expected_job_name=icra-e4-robust-aggregate
    ;;
  *) echo "unknown robust diagnostic phase: ${phase:-<empty>}" >&2; exit 2 ;;
esac

: "${SLURM_JOB_ID:?worker must run inside Slurm}"
[[ "${SLURM_JOB_ACCOUNT:-}" == "$ACCOUNT" \
  && "${SLURM_JOB_PARTITION:-}" == batch \
  && "${SLURM_JOB_QOS:-}" == normal \
  && "${SLURMD_NODENAME:-}" =~ ^sof1-h200-[0-7]$ \
  && "${SLURM_NNODES:-}" == 1 && "${SLURM_NTASKS:-}" == 1 \
  && "${SLURM_CPUS_PER_TASK:-}" == "$expected_cpus" ]] || {
  echo "robust diagnostic allocation provenance/resources differ" >&2; exit 2;
}
case "${SLURM_MEM_PER_NODE:-}" in
  "$expected_mem_mb"|"$expected_mem_label") ;;
  *) echo "robust diagnostic memory allocation differs" >&2; exit 2 ;;
esac
for variable in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES \
  SLURM_JOB_GPUS SLURM_STEP_GPUS SLURM_GPUS SLURM_GPUS_ON_NODE; do
  if [[ -n "${!variable-}" && "${!variable}" != NoDevFiles ]]; then
    echo "robust diagnostic allocation is not CPU-only: $variable=${!variable}" >&2
    exit 2
  fi
done

job_record=$(scontrol show job "$SLURM_JOB_ID" --oneliner)
[[ "$job_record" == *" JobName=$expected_job_name "* \
  && "$job_record" == *" Partition=batch "* \
  && "$job_record" == *" QOS=normal "* \
  && "$job_record" == *" Account=$ACCOUNT "* \
  && "$job_record" == *" UserId=$E4_SUBMIT_USER("* \
  && "$job_record" == *" Requeue=0 "* \
  && "$job_record" == *" TimeLimit=$expected_time "* \
  && "$job_record" == *" NumNodes=1 "* \
  && "$job_record" == *" NumCPUs=$expected_cpus "* \
  && "$job_record" == *" NumTasks=1 "* \
  && "$job_record" == *" CPUs/Task=$expected_cpus "* ]] || {
  echo "robust diagnostic scontrol identity/resource contract differs: $job_record" >&2
  exit 2
}
case "$job_record" in
  *" MinMemoryNode=$expected_mem_label "*|*" MinMemoryNode=${expected_mem_mb}M "*|*" MinMemoryNode=$expected_mem_mb "*) ;;
  *) echo "robust diagnostic scontrol memory contract differs" >&2; exit 2 ;;
esac
[[ "$job_record" == *" Command=$CODE_ROOT/run/slurm/icra2027_e4_robust_floor_support_cpu.sbatch "* ]] || {
  echo "robust diagnostic launcher command differs" >&2; exit 2;
}
job_record_lower=${job_record,,}
if [[ "$job_record_lower" == *"gres/gpu"* \
  || "$job_record_lower" == *"trespernode=gres"* \
  || "$job_record_lower" == *" gres="* ]]; then
  echo "robust diagnostic job unexpectedly owns GRES resources" >&2; exit 2
fi

unset PYTHONPATH PYTHONOPTIMIZE PYTHONUSERBASE RESUME AUTO FULL
unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
while IFS= read -r variable; do unset "$variable"; done < <(compgen -A variable SIMANY_ || true)
while IFS= read -r variable; do unset "$variable"; done < <(compgen -A variable SIMF_ || true)
export PYTHONPATH="$CODE_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export SIMANY_EVIDENCE_ROOT="$EVIDENCE_ROOT"
export OMP_NUM_THREADS="$expected_cpus"

submission_dir="$EVIDENCE_ROOT/outputs/icra2027/submissions/${E4_ROBUST_SWEEP_ID}-robust-floor-support-diagnostic"
runtime_parent="$submission_dir/runtime"
[[ -d "$runtime_parent" && ! -L "$runtime_parent" \
  && "$(stat -c '%a' -- "$runtime_parent")" == 700 \
  && "$("$GETFACL" -cp -- "$runtime_parent")" != *"default:"* ]] || {
  echo "missing private robust diagnostic runtime parent" >&2; exit 2;
}
runtime_root="$runtime_parent/$SLURM_JOB_ID"
mkdir "$runtime_root"
"$SETFACL" -k -- "$runtime_root"; chmod 00700 "$runtime_root"
mkdir "$runtime_root/tmp" "$runtime_root/xdg-cache"
for directory in "$runtime_root" "$runtime_root/tmp" "$runtime_root/xdg-cache"; do
  "$SETFACL" -k -- "$directory"; chmod 00700 "$directory"
  [[ -d "$directory" && ! -L "$directory" \
    && "$(stat -c '%a' -- "$directory")" == 700 \
    && "$("$GETFACL" -cp -- "$directory")" != *"default:"* ]] || {
    echo "runtime directory is not private" >&2; exit 2;
  }
done
export TMPDIR="$runtime_root/tmp"
export XDG_CACHE_HOME="$runtime_root/xdg-cache"
cd "$CODE_ROOT"

"$PYTHON" - "$SWEEP_SCRIPT" "$CODE_ROOT" "$EVIDENCE_ROOT" <<'PY'
import importlib.util
import sys
from pathlib import Path
script, code_root, evidence_root = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("sealed_e4_robust_floor_support", script)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load robust diagnostic runner")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if module.CODE_ROOT != code_root or module.EXPECTED_EVIDENCE_ROOT != evidence_root:
    raise SystemExit("robust diagnostic dual-root constants differ")
if tuple(module.VARIANT_IDS) != ("3db-f", "3db-fs", "d755-f", "d755-fs"):
    raise SystemExit("robust diagnostic frozen variants differ")
PY

common_args=(--sweep-id "$E4_ROBUST_SWEEP_ID" --expected-code-commit "$E4_CODE_COMMIT")
case "$phase" in
  build-common)
    exec "$PYTHON" "$SWEEP_SCRIPT" build-common "${common_args[@]}" \
      --variant-id "$variant_id" --e3-root "$E4_E3_ROOT" ;;
  eval-policy)
    exec "$PYTHON" "$SWEEP_SCRIPT" eval-policy "${common_args[@]}" \
      --variant-id "$variant_id" --policy-id "$policy_id" ;;
  compare-pair)
    exec "$PYTHON" "$SWEEP_SCRIPT" compare-pair "${common_args[@]}" \
      --variant-id "$variant_id" ;;
  aggregate)
    exec "$PYTHON" "$SWEEP_SCRIPT" aggregate "${common_args[@]}" ;;
esac
