#!/usr/bin/env bash
#SBATCH --job-name=icra-e3-agentic
#SBATCH --no-requeue
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=/group/worldcept/PhiRIE/code/SimAny/outputs/icra2027/slurm/e3-%j.out
#SBATCH --error=/group/worldcept/PhiRIE/code/SimAny/outputs/icra2027/slurm/e3-%j.err

# Resource-split E3 driver. Inventory, control, evaluation, and aggregation
# are CPU-only on sof1. Observation (RGB+ED Scene-Gaussian rendering) uses one
# explicitly selected and validated GCP A100-80G or Hala RTX A6000. The
# synthetic three-object controller smoke runs in that same GPU/MVPY profile
# as an interpreter/CUDA gate; the submitter then
# gates the 49-scene render fleet on one real one-object RGB+ED pilot. Each
# scene/phase is an independent ordinary Slurm job; arrays are a hard error.
set -euo pipefail

ROOT=/group/worldcept/PhiRIE/code/SimAny
PYTHON="$ROOT/.venv/bin/python"
MVPY=/group/worldcept/code/affordancept/.envs/mini-viewer/bin/python
JOBS="$ROOT/configs/experiments/icra2027/agentic_jobs.yaml"
POLICIES="$ROOT/configs/experiments/icra2027/agentic_policies.yaml"
ROSTER="$ROOT/configs/experiments/icra2027/construction_regimes.yaml"
PILOT_SCENE=5748ce6f01

: "${SLURM_JOB_ID:?this launcher must run under Slurm}"
: "${SLURMD_NODENAME:?missing Slurm execution node}"
: "${SLURM_JOB_PARTITION:?missing Slurm partition}"
: "${SLURM_JOB_QOS:?missing Slurm QoS}"
: "${E3_PHASE:?submit with E3_PHASE=inventory, smoke, observe, control, evaluate, or aggregate}"
: "${E3_FREEZE_ID:?submit with a unique E3_FREEZE_ID}"
: "${E3_CODE_COMMIT:?submit with the exact 40-character code commit}"

for ARRAY_VARIABLE in \
  SLURM_ARRAY_JOB_ID \
  SLURM_ARRAY_TASK_ID \
  SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN \
  SLURM_ARRAY_TASK_MAX \
  SLURM_ARRAY_TASK_STEP
do
  if [[ -v "$ARRAY_VARIABLE" ]]; then
    echo "E3 must not run with Slurm array context: $ARRAY_VARIABLE is set" >&2
    exit 2
  fi
done

[[ "$E3_FREEZE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "E3_FREEZE_ID contains unsafe path characters: $E3_FREEZE_ID" >&2
  exit 2
}
[[ "$E3_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "E3_CODE_COMMIT must be an exact lowercase 40-character Git SHA" >&2
  exit 2
}

case "$E3_PHASE" in
  inventory|control|evaluate)
    EXPECTED_CPUS=8
    RESOURCE_PROFILE=sof1-cpu
    ;;
  smoke|observe)
    EXPECTED_CPUS=8
    : "${E3_GPU_PROFILE:?GPU phase=$E3_PHASE requires E3_GPU_PROFILE=gcp-a100 or hala-a6000}"
    case "$E3_GPU_PROFILE" in
      gcp-a100|hala-a6000) RESOURCE_PROFILE="$E3_GPU_PROFILE" ;;
      *)
        echo "unsupported E3_GPU_PROFILE=$E3_GPU_PROFILE (expected gcp-a100 or hala-a6000)" >&2
        exit 2
        ;;
    esac
    ;;
  aggregate)
    EXPECTED_CPUS=16
    RESOURCE_PROFILE=sof1-cpu
    ;;
  *)
    echo "unsupported E3_PHASE=$E3_PHASE (expected inventory, smoke, observe, control, evaluate, or aggregate)" >&2
    exit 2
    ;;
esac
if [[ -v E3_REQUIRE_NONEMPTY_PROPOSALS ]]; then
  [[ "$E3_PHASE" == control ]] || {
    echo "E3_REQUIRE_NONEMPTY_PROPOSALS is valid only for phase=control" >&2
    exit 2
  }
  [[ "$E3_REQUIRE_NONEMPTY_PROPOSALS" == 1 ]] || {
    echo "E3_REQUIRE_NONEMPTY_PROPOSALS must equal 1 when set" >&2
    exit 2
  }
  [[ "${E3_SCENE_ID:-}" == "$PILOT_SCENE" ]] || {
    echo "E3_REQUIRE_NONEMPTY_PROPOSALS is valid only for pilot scene=$PILOT_SCENE" >&2
    exit 2
  }
fi
if [[ "$RESOURCE_PROFILE" == sof1-cpu && -v E3_GPU_PROFILE ]]; then
  echo "E3 CPU phase=$E3_PHASE rejects leaked E3_GPU_PROFILE" >&2
  exit 2
fi

[[ "$SLURM_JOB_PARTITION" == batch ]] || {
  echo "E3 requires partition=batch; found $SLURM_JOB_PARTITION" >&2
  exit 2
}
[[ "$SLURM_JOB_QOS" == normal ]] || {
  echo "E3 requires qos=normal; found $SLURM_JOB_QOS" >&2
  exit 2
}
[[ "${SLURM_CPUS_PER_TASK:-}" == "$EXPECTED_CPUS" ]] || {
  echo "E3 phase=$E3_PHASE requires cpus-per-task=$EXPECTED_CPUS; found ${SLURM_CPUS_PER_TASK:-unset}" >&2
  exit 2
}
[[ "${SLURM_MEM_PER_NODE:-}" == 65536 ]] || {
  echo "E3 requires mem=64G (65536 MiB); found ${SLURM_MEM_PER_NODE:-unset}" >&2
  exit 2
}

if [[ "$RESOURCE_PROFILE" == sof1-cpu ]]; then
  [[ "$SLURMD_NODENAME" =~ ^sof1-h200-[0-7]$ ]] || {
    echo "E3 CPU profile requires a sof1-h200 node; found $SLURMD_NODENAME" >&2
    exit 2
  }
  for GPU_VARIABLE in SLURM_JOB_GPUS SLURM_STEP_GPUS SLURM_GPUS; do
    if [[ -n "${!GPU_VARIABLE:-}" ]]; then
      echo "E3 CPU profile must not request a GPU: $GPU_VARIABLE=${!GPU_VARIABLE}" >&2
      exit 2
    fi
  done
  case "${CUDA_VISIBLE_DEVICES:-}" in
    ""|NoDevFiles) ;;
    *)
      echo "E3 CPU profile must not expose CUDA devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
      exit 2
      ;;
  esac
elif [[ "$RESOURCE_PROFILE" == gcp-a100 ]]; then
  [[ "$SLURMD_NODENAME" == gcp-eu1-a100-80g-qrfh ]] || {
    echo "E3 A100 profile requires node=gcp-eu1-a100-80g-qrfh; found $SLURMD_NODENAME" >&2
    exit 2
  }
elif [[ "$RESOURCE_PROFILE" == hala-a6000 ]]; then
  [[ "$SLURMD_NODENAME" == hala ]] || {
    echo "E3 A6000 profile requires node=hala; found $SLURMD_NODENAME" >&2
    exit 2
  }
fi
if [[ "$RESOURCE_PROFILE" == gcp-a100 || "$RESOURCE_PROFILE" == hala-a6000 ]]; then
  [[ -n "${SLURM_JOB_GPUS:-}" ]] || {
    echo "E3 GPU profile=$RESOURCE_PROFILE requires one requested Slurm GPU" >&2
    exit 2
  }
  VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-}"
  [[ -n "$VISIBLE_GPUS" && "$VISIBLE_GPUS" != NoDevFiles && "$VISIBLE_GPUS" != *,* ]] || {
    echo "E3 GPU profile=$RESOURCE_PROFILE requires exactly one visible GPU; CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS" >&2
    exit 2
  }
fi

cd "$ROOT"
[[ -x "$PYTHON" ]] || { echo "missing E3 Python: $PYTHON" >&2; exit 2; }
if [[ "$RESOURCE_PROFILE" == gcp-a100 || "$RESOURCE_PROFILE" == hala-a6000 ]]; then
  [[ -x "$MVPY" ]] || { echo "missing E3 gsplat Python: $MVPY" >&2; exit 2; }
fi
for INPUT in "$JOBS" "$POLICIES" "$ROSTER"; do
  [[ -s "$INPUT" && ! -L "$INPUT" ]] || {
    echo "missing, empty, or symlinked E3 input: $INPUT" >&2
    exit 2
  }
done
[[ "$(git rev-parse HEAD)" == "$E3_CODE_COMMIT" ]] || {
  echo "code commit drift: expected $E3_CODE_COMMIT, found $(git rev-parse HEAD)" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo "refusing E3 execution from a dirty worktree" >&2
  git status --short >&2
  exit 2
}

mapfile -t FROZEN_SCENES < <(
  "$PYTHON" - "$ROSTER" <<'PY'
import re
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
population = config.get("population", {})
scenes = [str(value).strip() for value in population.get("scene_ids", [])]
if population.get("planned_scenes") != 50:
    raise SystemExit("E3 roster must declare planned_scenes=50")
if len(scenes) != 50 or len(set(scenes)) != 50:
    raise SystemExit("E3 roster must contain exactly 50 unique scenes")
if not all(re.fullmatch(r"[0-9a-f]{10}", value) for value in scenes):
    raise SystemExit("E3 roster contains an invalid scene ID")
for scene in scenes:
    print(scene)
PY
)
[[ ${#FROZEN_SCENES[@]} -eq 50 ]] || {
  echo "E3 roster parser did not return exactly 50 scenes" >&2
  exit 2
}

if [[ "$E3_PHASE" == observe || "$E3_PHASE" == control || "$E3_PHASE" == evaluate ]]; then
  : "${E3_SCENE_ID:?phase=$E3_PHASE requires one exact E3_SCENE_ID}"
  [[ "$E3_SCENE_ID" =~ ^[0-9a-f]{10}$ ]] || {
    echo "E3_SCENE_ID must be a lowercase 10-character scene ID: $E3_SCENE_ID" >&2
    exit 2
  }
  SCENE_FOUND=false
  for EXPECTED_SCENE_ID in "${FROZEN_SCENES[@]}"; do
    if [[ "$E3_SCENE_ID" == "$EXPECTED_SCENE_ID" ]]; then
      SCENE_FOUND=true
      break
    fi
  done
  [[ "$SCENE_FOUND" == true ]] || {
    echo "E3_SCENE_ID=$E3_SCENE_ID is not in the exact frozen 50-scene roster" >&2
    exit 2
  }
  TASK_TAG="$E3_PHASE-$E3_SCENE_ID"
elif [[ -v E3_SCENE_ID ]]; then
  echo "E3_SCENE_ID is valid only for observe, control, or evaluate" >&2
  exit 2
else
  TASK_TAG="$E3_PHASE"
fi

checked_repo_file() {
  local path="$1"
  local label="$2"
  case "$path" in
    "$ROOT"/*) ;;
    *) echo "$label must be an absolute path below $ROOT: $path" >&2; return 2 ;;
  esac
  local relative="${path#"$ROOT"/}"
  local current="$ROOT"
  local part
  IFS=/ read -r -a parts <<< "$relative"
  for part in "${parts[@]}"; do
    [[ -n "$part" && "$part" != . && "$part" != .. ]] || {
      echo "$label contains an unsafe path component: $path" >&2
      return 2
    }
    current="$current/$part"
    [[ ! -L "$current" ]] || {
      echo "$label uses a symlink component: $current" >&2
      return 2
    }
  done
  [[ -s "$path" && -f "$path" ]] || {
    echo "$label is missing, empty, or not a regular file: $path" >&2
    return 2
  }
}

# A caller may pass an already-resolved contract. For the one-command DAG,
# the contract path is only known after E0 starts; dependent jobs resolve the
# unique run_root line from that E0 job's repository-local log at runtime.
if [[ -n "${E3_CONTRACT_MANIFEST:-}" ]]; then
  CONTRACT="$E3_CONTRACT_MANIFEST"
else
  : "${E3_E0_JOB_ID:?set E3_CONTRACT_MANIFEST or the ordinary E3_E0_JOB_ID}"
  [[ "$E3_E0_JOB_ID" =~ ^[0-9]+$ ]] || {
    echo "E3_E0_JOB_ID must be one ordinary numeric Slurm job ID" >&2
    exit 2
  }
  E0_LOG="${E3_E0_LOG:-$ROOT/outputs/icra2027/slurm/e3-e0-${E3_E0_JOB_ID}.out}"
  checked_repo_file "$E0_LOG" "E0 log"
  mapfile -t E0_RUN_ROOTS < <(sed -n 's/^run_root=//p' "$E0_LOG")
  [[ ${#E0_RUN_ROOTS[@]} -eq 1 ]] || {
    echo "E0 log must contain exactly one run_root= line: $E0_LOG" >&2
    exit 2
  }
  case "${E0_RUN_ROOTS[0]}" in
    "$ROOT"/outputs/icra2027/preflight-smoke-*) ;;
    *)
      echo "E0 run_root is outside the expected repository-local preflight namespace: ${E0_RUN_ROOTS[0]}" >&2
      exit 2
      ;;
  esac
  CONTRACT="${E0_RUN_ROOTS[0]}/contract/freeze_manifest.json"
fi
checked_repo_file "$CONTRACT" "E0 contract manifest"

"$PYTHON" - "$CONTRACT" "$E3_CODE_COMMIT" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    contract = json.load(handle)
expected = sys.argv[2]
code = contract.get("code", {})
observed = code.get("commit")
if code.get("dirty") is not False:
    raise SystemExit("E0 contract records a dirty source tree")
if not isinstance(observed, str) or not re.fullmatch(r"[0-9a-f]{40}", observed):
    raise SystemExit("E0 contract does not contain an exact code commit")
if observed != expected:
    raise SystemExit(f"E0/E3 code commit mismatch: expected {expected}, found {observed}")
if contract.get("mode") not in {"smoke", "paper"}:
    raise SystemExit("E0 contract mode must be smoke or paper")
PY

E3_ROOT="$ROOT/outputs/icra2027/$E3_FREEZE_ID/agentic"
RUNTIME="$ROOT/outputs/icra2027/$E3_FREEZE_ID/runtime/e3/$TASK_TAG/$SLURM_JOB_ID"
export TMPDIR="$RUNTIME/tmp"
export XDG_CACHE_HOME="$RUNTIME/xdg"
export XDG_CONFIG_HOME="$RUNTIME/xdg-config"
export XDG_DATA_HOME="$RUNTIME/xdg-data"
export XDG_STATE_HOME="$RUNTIME/xdg-state"
export HF_HOME="$RUNTIME/huggingface"
export TORCH_HOME="$RUNTIME/torch"
export MPLCONFIGDIR="$RUNTIME/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$RUNTIME/torchinductor"
export NUMBA_CACHE_DIR="$RUNTIME/numba"
export JOBLIB_TEMP_FOLDER="$RUNTIME/joblib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset PYTHONPATH PYTHONOPTIMIZE
# Defense in depth for every pre-decision process. Some legacy alignment
# modules expose optional GT helpers even though E3 calls only their GT-free
# primitives; force those helpers closed. Evaluation opens GT only after the
# immutable controller shard has been sealed, so it must not inherit this flag.
case "$E3_PHASE" in
  smoke|observe|control) export SIMANY_NO_GT=1 ;;
  *) unset SIMANY_NO_GT SIMF_NO_GT ;;
esac
RUNTIME_DIRS=(
  "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME"
  "$XDG_STATE_HOME" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"
  "$TORCHINDUCTOR_CACHE_DIR" "$NUMBA_CACHE_DIR" "$JOBLIB_TEMP_FOLDER"
  "$ROOT/outputs/icra2027/slurm"
)
unset CC CXX CUDAHOSTCXX NVCC_APPEND_FLAGS
if [[ "$RESOURCE_PROFILE" == gcp-a100 || "$RESOURCE_PROFILE" == hala-a6000 ]]; then
  case "$RESOURCE_PROFILE" in
    gcp-a100)
      GPU_ARCH=8.0
      GPU_ARCH_TAG=sm80
      export E3_EXPECTED_GPU_NAME="NVIDIA A100-SXM4-80GB"
      ;;
    hala-a6000)
      GPU_ARCH=8.6
      GPU_ARCH_TAG=sm86
      export E3_EXPECTED_GPU_NAME="NVIDIA RTX A6000"
      ;;
  esac
  export E3_EXPECTED_GPU_CAPABILITY="$GPU_ARCH"
  export TORCH_EXTENSIONS_DIR="$RUNTIME/$RESOURCE_PROFILE/torch-extensions-$GPU_ARCH_TAG"
  export CUDA_CACHE_PATH="$RUNTIME/$RESOURCE_PROFILE/cuda-$GPU_ARCH_TAG"
  export TRITON_CACHE_DIR="$RUNTIME/$RESOURCE_PROFILE/triton-$GPU_ARCH_TAG"
  export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
  export CUMM_CUDA_ARCH_LIST="$GPU_ARCH"
  for GPU_CXX in /usr/bin/g++-12 /usr/bin/g++-13; do
    if [[ -x "$GPU_CXX" ]]; then
      export CXX="$GPU_CXX"
      export CC="${GPU_CXX/g++/gcc}"
      export CUDAHOSTCXX="$GPU_CXX"
      break
    fi
  done
  export NVCC_APPEND_FLAGS=--allow-unsupported-compiler
  RUNTIME_DIRS+=("$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR")
  EXEC_PYTHON="$MVPY"
else
  unset TORCH_EXTENSIONS_DIR CUDA_CACHE_PATH TRITON_CACHE_DIR
  unset TORCH_CUDA_ARCH_LIST CUMM_CUDA_ARCH_LIST
  unset E3_EXPECTED_GPU_NAME E3_EXPECTED_GPU_CAPABILITY
  EXEC_PYTHON="$PYTHON"
fi
mkdir -p "${RUNTIME_DIRS[@]}"
[[ "$($EXEC_PYTHON -c 'print(__debug__)')" == True ]] || {
  echo "E3 launcher requires Python assertions enabled" >&2
  exit 2
}

if [[ "$E3_PHASE" == control ]]; then
  "$PYTHON" - "$ROOT/.venv" <<'PY'
import importlib.metadata
import sys
from pathlib import Path

from agents.assets.s5_align import open3d_registration_backend

venv = Path(sys.argv[1]).resolve()
backend = open3d_registration_backend()
backend_path = Path(backend.__file__).resolve()
if not backend_path.is_relative_to(venv):
    raise SystemExit(
        f"E3 Open3D registration backend is outside the repository venv: {backend_path}"
    )
if not hasattr(backend.pipelines.registration, "registration_icp"):
    raise SystemExit("E3 Open3D CPU backend lacks registration_icp")
for forbidden in ("open3d.visualization", "dash", "flask", "pydantic"):
    if forbidden in sys.modules:
        raise SystemExit(f"E3 registration preflight imported forbidden optional module: {forbidden}")
print(f"open3d_version={importlib.metadata.version('open3d')}")
print(f"open3d_registration_backend={backend_path}")
PY
fi

if [[ "$RESOURCE_PROFILE" == gcp-a100 || "$RESOURCE_PROFILE" == hala-a6000 ]]; then
  "$EXEC_PYTHON" - <<'PY'
import os

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to the E3 GPU phase")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"E3 GPU phase requires one visible CUDA device; found {torch.cuda.device_count()}"
    )
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
capability = f"{major}.{minor}"
expected_name = os.environ["E3_EXPECTED_GPU_NAME"]
expected_capability = os.environ["E3_EXPECTED_GPU_CAPABILITY"]
if name != expected_name:
    raise SystemExit(f"E3 GPU name mismatch: {name}")
if capability != expected_capability:
    raise SystemExit(f"E3 GPU compute capability mismatch: {capability}")
print(f"gpu_name={name}")
print(f"gpu_compute_capability={capability}")
PY
fi

COMMAND=(
  "$EXEC_PYTHON" -m robo.eval.agentic_ablation
  --jobs "$JOBS"
  --policies "$POLICIES"
  --contract-manifest "$CONTRACT"
  --freeze-id "$E3_FREEZE_ID"
  --out "$E3_ROOT"
)
case "$E3_PHASE" in
  inventory) COMMAND+=(--inventory) ;;
  smoke) COMMAND+=(--smoke) ;;
  observe) COMMAND+=(--observe --scene-id "$E3_SCENE_ID") ;;
  control) COMMAND+=(--control --scene-id "$E3_SCENE_ID") ;;
  evaluate) COMMAND+=(--evaluate --scene-id "$E3_SCENE_ID") ;;
  aggregate) COMMAND+=(--aggregate) ;;
esac

"${COMMAND[@]}"

# The synthetic smoke verifies the interpreter/controller plumbing, but it
# cannot prove that the real registration backend produced a candidate.  The
# pilot control must therefore verify its just-sealed shard before Slurm may
# release the pilot evaluation and the remaining 49 observations.
if [[ -v E3_REQUIRE_NONEMPTY_PROPOSALS ]]; then
  "$PYTHON" - "$E3_ROOT" "$E3_SCENE_ID" <<'PY'
import sys
from pathlib import Path

from robo.eval.agentic_ablation import _load_control_scene

out_dir = Path(sys.argv[1])
scene_id = sys.argv[2]
_, shard, seal = _load_control_scene(out_dir, scene_id)
members = seal.get("members")
if not isinstance(members, dict) or set(members) != {
    "controller_shard.json",
    "job_ledger.jsonl",
}:
    raise SystemExit("E3 pilot control seal has invalid member coverage")
job_count = shard.get("job_count")
proposals = shard.get("proposals")
if type(job_count) is not int or job_count < 1:
    raise SystemExit(
        f"E3 pilot controller shard has invalid or empty job_count: {job_count!r}"
    )
if not isinstance(proposals, list) or not proposals:
    raise SystemExit("E3 pilot controller shard contains no real proposals")
print(f"pilot_controller_job_count={job_count}")
print(f"pilot_controller_real_proposals={len(proposals)}")
PY
fi

echo "E3_PHASE_COMPLETE=$E3_PHASE"
echo "freeze_id=$E3_FREEZE_ID"
echo "task=$TASK_TAG"
echo "contract_manifest=$CONTRACT"
echo "output=$E3_ROOT"
