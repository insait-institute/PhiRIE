#!/usr/bin/env bash
# One ordinary CPU worker for the sealed two-room E4 prerequisite DAG.
# Resources and dependencies are supplied by submit_e4_region_pilot_noarray.sh.
set -euo pipefail

export PATH=/usr/local/bin:/usr/bin:/bin
CODE_ROOT=${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot
EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}
PYTHON=${SIMANY_ROOT:-$PWD}/.venv/bin/python
SCANNETPP_ROOT=/data/ScanNetpp
SPLATS_ROOT=/data/ScanNetppv2_gsplat/splats
ACCOUNT=${SLURM_ACCOUNT:-phirie}
EXPECTED_E3_ROOT=outputs/icra2027/icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic

: "${E4_FREEZE_ID:?E4_FREEZE_ID is required}"
: "${E4_CODE_COMMIT:?E4_CODE_COMMIT is required}"
: "${E4_E3_ROOT:?E4_E3_ROOT is required}"
: "${E4_SUBMIT_USER:?E4_SUBMIT_USER is required}"

[[ "$E4_FREEZE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_FREEZE_ID: $E4_FREEZE_ID" >&2
  exit 2
}
[[ "$E4_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "E4_CODE_COMMIT must be a full lowercase SHA" >&2
  exit 2
}
[[ "$E4_SUBMIT_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_SUBMIT_USER: $E4_SUBMIT_USER" >&2
  exit 2
}
[[ "$(id -un)" == "$E4_SUBMIT_USER" ]] || {
  echo "worker user differs from actual submit user=$E4_SUBMIT_USER" >&2
  exit 2
}
[[ "$E4_E3_ROOT" == "$EXPECTED_E3_ROOT" ]] || {
  echo "E4_E3_ROOT differs from the frozen pilot source: $E4_E3_ROOT" >&2
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
    echo "E4 CPU launcher refuses Slurm array context: $array_variable is set" >&2
    exit 2
  fi
done

[[ -d "$CODE_ROOT/.git" || -f "$CODE_ROOT/.git" ]] || {
  echo "missing isolated E4 Git worktree: $CODE_ROOT" >&2
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
[[ -x "$PYTHON" ]] || { echo "missing Python environment: $PYTHON" >&2; exit 2; }
[[ -d "$SCANNETPP_ROOT" && ! -L "$SCANNETPP_ROOT" ]] || {
  echo "missing or symlinked ScanNet++ root: $SCANNETPP_ROOT" >&2
  exit 2
}
[[ -d "$SPLATS_ROOT" && ! -L "$SPLATS_ROOT" ]] || {
  echo "missing or symlinked splat root: $SPLATS_ROOT" >&2
  exit 2
}

observed_root=$(git -C "$CODE_ROOT" rev-parse --show-toplevel)
observed_commit=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$observed_root" == "$CODE_ROOT" ]] || {
  echo "E4 launcher resolved a different code root: $observed_root" >&2
  exit 2
}
[[ "$observed_commit" == "$E4_CODE_COMMIT" ]] || {
  echo "E4 code commit drift: expected=$E4_CODE_COMMIT observed=$observed_commit" >&2
  exit 2
}
if [[ -n "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "E4 CPU launcher refuses a dirty code worktree" >&2
  git -C "$CODE_ROOT" status --short >&2
  exit 2
fi
[[ -d "$EVIDENCE_ROOT/$E4_E3_ROOT" && ! -L "$EVIDENCE_ROOT/$E4_E3_ROOT" ]] || {
  echo "missing or symlinked sealed E3 root: $EVIDENCE_ROOT/$E4_E3_ROOT" >&2
  exit 2
}

# Parse the phase before checking the allocation so every branch has one exact
# CPU/memory contract.  No manual/non-Slurm execution is accepted here.
phase=${1:-}
case "$phase" in
  materialize)
    expected_cpus=8
    expected_mem_mb=65536
    expected_mem_label=64G
    expected_time=01:00:00
    ;;
  prepare-scene)
    expected_cpus=16
    expected_mem_mb=98304
    expected_mem_label=96G
    expected_time=02:00:00
    ;;
  finalize)
    expected_cpus=4
    expected_mem_mb=32768
    expected_mem_label=32G
    expected_time=00:30:00
    ;;
  *)
    echo "unknown E4 CPU phase: ${phase:-<empty>}" >&2
    exit 2
    ;;
esac

: "${SLURM_JOB_ID:?E4 CPU launcher must run inside a Slurm allocation}"
[[ "${SLURM_JOB_ACCOUNT:-}" == "$ACCOUNT" ]] || {
  echo "E4 CPU job account differs: ${SLURM_JOB_ACCOUNT:-<unset>}" >&2
  exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == batch ]] || {
  echo "E4 CPU job partition differs: ${SLURM_JOB_PARTITION:-<unset>}" >&2
  exit 2
}
[[ "${SLURM_JOB_QOS:-}" == normal ]] || {
  echo "E4 CPU job QoS differs: ${SLURM_JOB_QOS:-<unset>}" >&2
  exit 2
}
[[ "${SLURMD_NODENAME:-}" =~ ^sof1-h200-[0-7]$ ]] || {
  echo "E4 CPU job is not running on one sof1 host: ${SLURMD_NODENAME:-<unset>}" >&2
  exit 2
}
[[ "${SLURM_NNODES:-}" == 1 && "${SLURM_NTASKS:-}" == 1 ]] || {
  echo "E4 CPU job requires exactly one node and one task" >&2
  exit 2
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "$expected_cpus" ]] || {
  echo "E4 $phase requires cpus-per-task=$expected_cpus, observed=${SLURM_CPUS_PER_TASK:-<unset>}" >&2
  exit 2
}
case "${SLURM_MEM_PER_NODE:-}" in
  "$expected_mem_mb"|"$expected_mem_label") ;;
  *)
    echo "E4 $phase requires mem=$expected_mem_label, observed=${SLURM_MEM_PER_NODE:-<unset>}" >&2
    exit 2
    ;;
esac
for gpu_variable in \
  CUDA_VISIBLE_DEVICES \
  NVIDIA_VISIBLE_DEVICES \
  ROCR_VISIBLE_DEVICES \
  SLURM_JOB_GPUS \
  SLURM_STEP_GPUS \
  SLURM_GPUS \
  SLURM_GPUS_ON_NODE
do
  # Slurm uses NoDevFiles on CPU allocations on this cluster.  It denotes no
  # visible device and is cleared below with the other scheduler variables.
  if [[ -n "${!gpu_variable-}" && "${!gpu_variable}" != NoDevFiles ]]; then
    echo "E4 prerequisite allocation is not CPU-only: $gpu_variable=${!gpu_variable}" >&2
    exit 2
  fi
done
job_record=$(scontrol show job "$SLURM_JOB_ID" --oneliner)
[[ "$job_record" == *" Partition=batch "* \
  && "$job_record" == *" QOS=normal "* \
  && "$job_record" == *" Account=$ACCOUNT "* \
  && "$job_record" == *" UserId=$E4_SUBMIT_USER("* \
  && "$job_record" == *" Requeue=0 "* ]] || {
  echo "scontrol partition/QoS/account/user/requeue provenance differs: $job_record" >&2
  exit 2
}
[[ "$job_record" == *" TimeLimit=$expected_time "* \
  && "$job_record" == *" NumNodes=1 "* \
  && "$job_record" == *" NumCPUs=$expected_cpus "* \
  && "$job_record" == *" NumTasks=1 "* \
  && "$job_record" == *" CPUs/Task=$expected_cpus "* ]] || {
  echo "scontrol time/CPU task contract differs for $phase: $job_record" >&2
  exit 2
}
case "$job_record" in
  *" MinMemoryNode=$expected_mem_label "*|*" MinMemoryNode=${expected_mem_mb}M "*|*" MinMemoryNode=$expected_mem_mb "*) ;;
  *)
    echo "scontrol memory contract differs for $phase: $job_record" >&2
    exit 2
    ;;
esac
[[ "$job_record" == *" Command=$CODE_ROOT/run/slurm/icra2027_e4_region_cpu.sbatch "* ]] || {
  echo "scontrol launcher command provenance differs: $job_record" >&2
  exit 2
}
job_record_lower=${job_record,,}
if [[ "$job_record_lower" == *"gres/"* \
  || "$job_record_lower" == *" trespernode=gres"* \
  || "$job_record_lower" == *" gres="* ]]; then
  echo "E4 prerequisite unexpectedly owns GRES resources: $job_record" >&2
  exit 2
fi

# Drop caller/login-shell state before imports.  Dataset and artifact roots are
# then reconstructed from fixed canonical values only.
unset PYTHONPATH PYTHONOPTIMIZE PYTHONUSERBASE RESUME AUTO FULL
unset CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
while IFS= read -r inherited_variable; do
  unset "$inherited_variable"
done < <(compgen -A variable SIMANY_ || true)
while IFS= read -r inherited_variable; do
  unset "$inherited_variable"
done < <(compgen -A variable SIMF_ || true)
export PYTHONPATH="$CODE_ROOT"
export SIMANY_EVIDENCE_ROOT="$EVIDENCE_ROOT"
export SIMANY_ROOT="$CODE_ROOT"
export SIMANY_SCANNETPP_ROOT="$SCANNETPP_ROOT"
export SIMANY_SPLATS_ROOT="$SPLATS_ROOT"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
runtime_root="$EVIDENCE_ROOT/outputs/icra2027/$E4_FREEZE_ID/runtime/${SLURM_JOB_ID:-manual}"
mkdir -p "$runtime_root/tmp" "$runtime_root/xdg-cache"
export TMPDIR="$runtime_root/tmp"
export XDG_CACHE_HOME="$runtime_root/xdg-cache"
cd "$CODE_ROOT"

# Prove the module resolver uses this exact clean worktree, not the main
# checkout, the submission directory, or an inherited PYTHONPATH entry.
"$PYTHON" - <<'PY'
from pathlib import Path
import robo.eval.e3_factory_materializer as materializer
import robo.eval.e4_region_pilot as pilot

expected = Path("${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot")
for module in (materializer, pilot):
    observed = Path(module.__file__).resolve(strict=True)
    if expected not in observed.parents:
        raise SystemExit(f"wrong E4 import origin: {module.__name__} -> {observed}")
if materializer.CODE_ROOT != expected or pilot.CODE_ROOT != expected:
    raise SystemExit("E4 module CODE_ROOT differs from isolated worktree")
if materializer.REPOSITORY_ROOT != Path("${SIMANY_ROOT:-$PWD}"):
    raise SystemExit("materializer evidence root differs")
PY

case "$phase" in
  materialize)
    [[ $# -eq 3 ]] || { echo "usage: $0 materialize SCENE A0|A4" >&2; exit 2; }
    scene_id=$2
    policy_id=$3
    [[ "$scene_id" == b0a08200c9 || "$scene_id" == 825d228aec ]] || {
      echo "scene is outside the frozen E4 pilot: $scene_id" >&2
      exit 2
    }
    [[ "$policy_id" == A0 || "$policy_id" == A4 ]] || {
      echo "policy must be A0 or A4" >&2
      exit 2
    }
    out="outputs/icra2027/$E4_FREEZE_ID/construction_variants/$policy_id/${scene_id}_factory"
    exec "$PYTHON" -m robo.eval.e3_factory_materializer \
      --e3-root "$E4_E3_ROOT" \
      --scene-id "$scene_id" \
      --policy-id "$policy_id" \
      --out "$out"
    ;;
  prepare-scene)
    [[ $# -eq 2 ]] || { echo "usage: $0 prepare-scene SCENE" >&2; exit 2; }
    scene_id=$2
    [[ "$scene_id" == b0a08200c9 || "$scene_id" == 825d228aec ]] || {
      echo "scene is outside the frozen E4 pilot: $scene_id" >&2
      exit 2
    }
    exec "$PYTHON" -m robo.eval.e4_region_pilot prepare-scene \
      --freeze-id "$E4_FREEZE_ID" \
      --scene-id "$scene_id" \
      --expected-code-commit "$E4_CODE_COMMIT"
    ;;
  finalize)
    [[ $# -eq 1 ]] || { echo "usage: $0 finalize" >&2; exit 2; }
    exec "$PYTHON" -m robo.eval.e4_region_pilot finalize \
      --freeze-id "$E4_FREEZE_ID" \
      --expected-code-commit "$E4_CODE_COMMIT"
    ;;
esac
