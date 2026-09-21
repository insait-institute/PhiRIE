#!/usr/bin/env bash
# One ordinary CPU worker for the sealed E4 candidate-screen fleet.
# Exact resources and dependencies are supplied by the no-array submitter.
set -euo pipefail

export PATH=/usr/local/bin:/usr/bin:/bin
CODE_ROOT=/group/worldcept/PhiRIE/code/SimAny-wt/e4-paired-pilot
EVIDENCE_ROOT=/group/worldcept/PhiRIE/code/SimAny
PYTHON=/group/worldcept/PhiRIE/code/SimAny/.venv/bin/python
SCANNETPP_ROOT=/data/ScanNetpp
SPLATS_ROOT=/data/ScanNetppv2_gsplat/splats
MENAGERIE_ROOT="$CODE_ROOT/third_party/mujoco_menagerie"
EXPECTED_E3_ROOT=outputs/icra2027/icra2027-contract-v1-e3-48fa807844ef-prelim-full-hala-r2/agentic
EXPECTED_MENAGERIE_COMMIT=71f066ad0be9cd271f7ed58c030243ef157af9f4
ACCOUNT=runyi_yang
SETFACL=/usr/bin/setfacl
GETFACL=/usr/bin/getfacl

: "${E4_SCREEN_ID:?E4_SCREEN_ID is required}"
: "${E4_CODE_COMMIT:?E4_CODE_COMMIT is required}"
: "${E4_E3_ROOT:?E4_E3_ROOT is required}"
: "${E4_SUBMIT_USER:?E4_SUBMIT_USER is required}"
[[ "$E4_SCREEN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_SCREEN_ID: $E4_SCREEN_ID" >&2
  exit 2
}
[[ "$E4_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "E4_CODE_COMMIT must be a full lowercase SHA" >&2
  exit 2
}
[[ "$E4_E3_ROOT" == "$EXPECTED_E3_ROOT" ]] || {
  echo "E4_E3_ROOT differs from the sealed E3 source: $E4_E3_ROOT" >&2
  exit 2
}
[[ "$E4_SUBMIT_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe submit user: $E4_SUBMIT_USER" >&2
  exit 2
}
[[ "$(id -un)" == "$E4_SUBMIT_USER" ]] || {
  echo "worker user differs from submit user=$E4_SUBMIT_USER" >&2
  exit 2
}

for array_variable in \
  SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN SLURM_ARRAY_TASK_MAX SLURM_ARRAY_TASK_STEP \
  SBATCH_ARRAY_INX SBATCH_ARRAY
do
  if [[ -v "$array_variable" ]]; then
    echo "E4 candidate worker refuses Slurm array context: $array_variable is set" >&2
    exit 2
  fi
done

[[ -d "$CODE_ROOT/.git" || -f "$CODE_ROOT/.git" ]] || {
  echo "missing isolated E4 worktree: $CODE_ROOT" >&2
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
[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
[[ -x "$SETFACL" && -x "$GETFACL" ]] || {
  echo "setfacl/getfacl are required to seal private runtime directories" >&2
  exit 2
}
[[ -d "$SCANNETPP_ROOT" && ! -L "$SCANNETPP_ROOT" ]] || {
  echo "missing or symlinked ScanNet++ root: $SCANNETPP_ROOT" >&2
  exit 2
}
[[ -d "$SPLATS_ROOT" && ! -L "$SPLATS_ROOT" ]] || {
  echo "missing or symlinked splat root: $SPLATS_ROOT" >&2
  exit 2
}
[[ -d "$MENAGERIE_ROOT" && ! -L "$MENAGERIE_ROOT" ]] || {
  echo "missing or symlinked Menagerie closure: $MENAGERIE_ROOT" >&2
  exit 2
}
[[ -d "$EVIDENCE_ROOT/$E4_E3_ROOT" && ! -L "$EVIDENCE_ROOT/$E4_E3_ROOT" ]] || {
  echo "missing or symlinked sealed E3 root: $EVIDENCE_ROOT/$E4_E3_ROOT" >&2
  exit 2
}

observed_root=$(git -C "$CODE_ROOT" rev-parse --show-toplevel)
observed_commit=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$observed_root" == "$CODE_ROOT" ]] || {
  echo "candidate launcher resolved a different code root: $observed_root" >&2
  exit 2
}
[[ "$observed_commit" == "$E4_CODE_COMMIT" ]] || {
  echo "candidate code commit drift: expected=$E4_CODE_COMMIT observed=$observed_commit" >&2
  exit 2
}
if [[ -n "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "candidate launcher refuses a dirty code worktree" >&2
  git -C "$CODE_ROOT" status --short >&2
  exit 2
fi

SCENES=(3db0a1c8f3 27dd4da69e d755b3d9d8 acd95847c5 40aec5fffa)
is_candidate_scene() {
  local candidate=$1 scene
  for scene in "${SCENES[@]}"; do
    [[ "$candidate" == "$scene" ]] && return 0
  done
  return 1
}

# Bind every phase to one exact argv and resource tuple before inspecting the
# allocation.  Aggregate intentionally takes neither a scene nor a policy.
phase=${1:-}
case "$phase" in
  materialize)
    [[ $# -eq 3 ]] || {
      echo "usage: $0 materialize SCENE A0|A4" >&2
      exit 2
    }
    scene_id=$2
    policy_id=$3
    is_candidate_scene "$scene_id" || {
      echo "scene is outside the frozen candidate roster: $scene_id" >&2
      exit 2
    }
    [[ "$policy_id" == A0 || "$policy_id" == A4 ]] || {
      echo "materialize policy must be A0 or A4" >&2
      exit 2
    }
    expected_cpus=8
    expected_mem_mb=65536
    expected_mem_label=64G
    expected_time=01:00:00
    expected_job_name=icra-e4-candidate-materialize
    ;;
  prepare-scene)
    [[ $# -eq 2 ]] || { echo "usage: $0 prepare-scene SCENE" >&2; exit 2; }
    scene_id=$2
    is_candidate_scene "$scene_id" || {
      echo "scene is outside the frozen candidate roster: $scene_id" >&2
      exit 2
    }
    expected_cpus=16
    expected_mem_mb=98304
    expected_mem_label=96G
    expected_time=04:00:00
    expected_job_name=icra-e4-candidate-prepare
    ;;
  qualify-scene)
    [[ $# -eq 2 ]] || { echo "usage: $0 qualify-scene SCENE" >&2; exit 2; }
    scene_id=$2
    is_candidate_scene "$scene_id" || {
      echo "scene is outside the frozen candidate roster: $scene_id" >&2
      exit 2
    }
    expected_cpus=4
    expected_mem_mb=32768
    expected_mem_label=32G
    expected_time=01:00:00
    expected_job_name=icra-e4-candidate-qualify
    ;;
  aggregate)
    [[ $# -eq 1 ]] || { echo "usage: $0 aggregate" >&2; exit 2; }
    expected_cpus=4
    expected_mem_mb=32768
    expected_mem_label=32G
    expected_time=00:30:00
    expected_job_name=icra-e4-candidate-aggregate
    ;;
  *)
    echo "unknown candidate-screen phase: ${phase:-<empty>}" >&2
    exit 2
    ;;
esac

: "${SLURM_JOB_ID:?candidate worker must run inside Slurm}"
[[ "${SLURM_JOB_ACCOUNT:-}" == "$ACCOUNT" ]] || {
  echo "wrong account: ${SLURM_JOB_ACCOUNT:-<unset>}" >&2
  exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == batch ]] || {
  echo "wrong partition: ${SLURM_JOB_PARTITION:-<unset>}" >&2
  exit 2
}
[[ "${SLURM_JOB_QOS:-}" == normal ]] || {
  echo "wrong QoS: ${SLURM_JOB_QOS:-<unset>}" >&2
  exit 2
}
[[ "${SLURMD_NODENAME:-}" =~ ^sof1-h200-[0-7]$ ]] || {
  echo "candidate job is not on sof1-h200-[0-7]: ${SLURMD_NODENAME:-<unset>}" >&2
  exit 2
}
[[ "${SLURM_NNODES:-}" == 1 && "${SLURM_NTASKS:-}" == 1 ]] || {
  echo "candidate job requires exactly one node and one task" >&2
  exit 2
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "$expected_cpus" ]] || {
  echo "$phase requires cpus-per-task=$expected_cpus, observed=${SLURM_CPUS_PER_TASK:-<unset>}" >&2
  exit 2
}
case "${SLURM_MEM_PER_NODE:-}" in
  "$expected_mem_mb"|"$expected_mem_label") ;;
  *)
    echo "$phase requires mem=$expected_mem_label, observed=${SLURM_MEM_PER_NODE:-<unset>}" >&2
    exit 2
    ;;
esac

for gpu_variable in \
  CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES \
  SLURM_JOB_GPUS SLURM_STEP_GPUS SLURM_GPUS SLURM_GPUS_ON_NODE
do
  if [[ -n "${!gpu_variable-}" && "${!gpu_variable}" != NoDevFiles ]]; then
    echo "candidate allocation is not CPU-only: $gpu_variable=${!gpu_variable}" >&2
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
  echo "scontrol identity/provenance contract differs: $job_record" >&2
  exit 2
}
[[ "$job_record" == *" TimeLimit=$expected_time "* \
  && "$job_record" == *" NumNodes=1 "* \
  && "$job_record" == *" NumCPUs=$expected_cpus "* \
  && "$job_record" == *" NumTasks=1 "* \
  && "$job_record" == *" CPUs/Task=$expected_cpus "* ]] || {
  echo "scontrol time/CPU contract differs for $phase: $job_record" >&2
  exit 2
}
case "$job_record" in
  *" MinMemoryNode=$expected_mem_label "*|*" MinMemoryNode=${expected_mem_mb}M "*|*" MinMemoryNode=$expected_mem_mb "*) ;;
  *)
    echo "scontrol memory contract differs for $phase: $job_record" >&2
    exit 2
    ;;
esac
[[ "$job_record" == *" Command=$CODE_ROOT/run/slurm/icra2027_e4_candidate_cpu.sbatch "* ]] || {
  echo "scontrol launcher command differs: $job_record" >&2
  exit 2
}
job_record_lower=${job_record,,}
if [[ "$job_record_lower" == *"gres/gpu"* \
  || "$job_record_lower" == *"trespernode=gres"* \
  || "$job_record_lower" == *" gres="* ]]; then
  echo "candidate job unexpectedly owns GRES resources: $job_record" >&2
  exit 2
fi

# Explicitly rebuild Python/data state after validating the allocation.
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
export SIMANY_ROOT="$CODE_ROOT"
export SIMANY_SCANNETPP_ROOT="$SCANNETPP_ROOT"
export SIMANY_SPLATS_ROOT="$SPLATS_ROOT"
export OMP_NUM_THREADS="$expected_cpus"

submission_dir="$EVIDENCE_ROOT/outputs/icra2027/submissions/${E4_SCREEN_ID}-candidate-screen"
runtime_parent="$submission_dir/runtime"
[[ -d "$submission_dir" && ! -L "$submission_dir" \
  && -d "$runtime_parent" && ! -L "$runtime_parent" \
  && "$(stat -c '%a' -- "$runtime_parent")" == 700 \
  && "$("$GETFACL" -cp -- "$runtime_parent")" != *"default:"* ]] || {
  echo "missing durable submission directory: $submission_dir" >&2
  exit 2
}
runtime_root="$runtime_parent/$SLURM_JOB_ID"
mkdir "$runtime_root"
"$SETFACL" -k -- "$runtime_root"
chmod 00700 "$runtime_root"
[[ -d "$runtime_root" && ! -L "$runtime_root" \
  && "$(stat -c '%a' -- "$runtime_root")" == 700 \
  && "$("$GETFACL" -cp -- "$runtime_root")" != *"default:"* ]] || {
  echo "job runtime directory is not private ACL-free mode 700: $runtime_root" >&2
  exit 2
}
mkdir "$runtime_root/tmp" "$runtime_root/xdg-cache"
for runtime_directory in "$runtime_root/tmp" "$runtime_root/xdg-cache"; do
  "$SETFACL" -k -- "$runtime_directory"
  chmod 00700 "$runtime_directory"
  [[ -d "$runtime_directory" && ! -L "$runtime_directory" \
    && "$(stat -c '%a' -- "$runtime_directory")" == 700 \
    && "$("$GETFACL" -cp -- "$runtime_directory")" != *"default:"* ]] || {
    echo "runtime child is not private ACL-free mode 700: $runtime_directory" >&2
    exit 2
  }
done
export TMPDIR="$runtime_root/tmp"
export XDG_CACHE_HOME="$runtime_root/xdg-cache"
cd "$CODE_ROOT"

# Prove the executed module comes from this exact clean worktree and carries
# the same dual-root and frozen-roster constants as the scheduler contract.
"$PYTHON" - "$CODE_ROOT" "$EVIDENCE_ROOT" "$E4_E3_ROOT" \
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

case "$phase" in
  materialize)
    exec "$PYTHON" -m robo.eval.e4_candidate_screen materialize \
      --screen-id "$E4_SCREEN_ID" \
      --scene-id "$scene_id" \
      --policy-id "$policy_id" \
      --e3-root "$E4_E3_ROOT" \
      --expected-code-commit "$E4_CODE_COMMIT"
    ;;
  prepare-scene)
    exec "$PYTHON" -m robo.eval.e4_candidate_screen prepare-scene \
      --screen-id "$E4_SCREEN_ID" \
      --scene-id "$scene_id" \
      --expected-code-commit "$E4_CODE_COMMIT"
    ;;
  qualify-scene)
    exec "$PYTHON" -m robo.eval.e4_candidate_screen qualify-scene \
      --screen-id "$E4_SCREEN_ID" \
      --scene-id "$scene_id" \
      --expected-code-commit "$E4_CODE_COMMIT" \
      --menagerie-root "$MENAGERIE_ROOT" \
      --expected-menagerie-commit "$EXPECTED_MENAGERIE_COMMIT"
    ;;
  aggregate)
    exec "$PYTHON" -m robo.eval.e4_candidate_screen aggregate \
      --screen-id "$E4_SCREEN_ID" \
      --expected-code-commit "$E4_CODE_COMMIT"
    ;;
esac
