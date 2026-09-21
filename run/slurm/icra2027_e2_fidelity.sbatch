#!/usr/bin/env bash
#SBATCH --job-name=icra-e2-fidelity
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/group/worldcept/PhiRIE/code/SimAny/outputs/icra2027/slurm/e2-%j.out
#SBATCH --error=/group/worldcept/PhiRIE/code/SimAny/outputs/icra2027/slurm/e2-%j.err

# E2 has three resource-distinct phases and therefore intentionally declares no
# partition, GPU, CPU, memory, or time options here. Supply the exact options
# documented in plan/icra2027/02_fidelity_metrics/README.md: one independent
# GPU render-room job per selected scene, a dependent CPU inventory, then one
# dependent GPU metrics job. GPU jobs must select an exact validated hardware
# profile explicitly. This keeps inventory from reserving an idle GPU while
# avoiding prohibitively slow CPU LPIPS over the full image population.
set -euo pipefail

ROOT=/group/worldcept/PhiRIE/code/SimAny
PYTHON="$ROOT/.venv/bin/python"
CONFIG="$ROOT/configs/experiments/icra2027/fidelity_manifest.json"
ROSTER="$ROOT/configs/experiments/icra2027/construction_regimes.yaml"

: "${SLURM_JOB_ID:?this launcher must run under Slurm}"
: "${E2_PHASE:?submit with E2_PHASE=render-room, inventory, or metrics}"
: "${E2_MODE:?submit with E2_MODE=smoke or E2_MODE=full}"
: "${E2_FREEZE_ID:?submit with a unique E2_FREEZE_ID}"
: "${E2_CODE_COMMIT:?submit with the exact 40-character code commit}"
: "${E2_CONTRACT_MANIFEST:?submit with a fresh repository-local E0 freeze_manifest.json}"

[[ "$E2_FREEZE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "E2_FREEZE_ID contains unsafe path characters: $E2_FREEZE_ID" >&2
  exit 2
}
[[ "$E2_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "E2_CODE_COMMIT must be an exact lowercase 40-character Git SHA" >&2
  exit 2
}
case "$E2_MODE" in
  smoke|full) ;;
  *) echo "unsupported E2_MODE=$E2_MODE (expected smoke or full)" >&2; exit 2 ;;
esac
case "$E2_PHASE" in
  render-room|inventory|metrics) ;;
  *) echo "unsupported E2_PHASE=$E2_PHASE (expected render-room, inventory, or metrics)" >&2; exit 2 ;;
esac

# GPU phases have no implicit cluster default.  Each profile binds the job to
# one exact node/partition/QoS/device/compute-capability tuple and to
# architecture-specific JIT caches.  Inventory remains strictly CPU-only and
# therefore rejects, rather than ignores, a GPU profile.
if [[ "$E2_PHASE" == render-room || "$E2_PHASE" == metrics ]]; then
  : "${E2_GPU_PROFILE:?GPU phases require E2_GPU_PROFILE=hala-a6000 or gcp-a100}"
  case "$E2_GPU_PROFILE" in
    hala-a6000)
      E2_EXPECTED_NODE=hala
      E2_EXPECTED_PARTITION=debug
      E2_EXPECTED_QOS=debug
      E2_EXPECTED_GPU_NAME="NVIDIA RTX A6000"
      E2_EXPECTED_GPU_CC=8.6
      E2_TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"
      E2_TORCH_EXTENSIONS_DIR="$ROOT/.cache/icra2027/e2/torch-extensions-a6000"
      E2_CUDA_CACHE_PATH="$ROOT/.cache/icra2027/e2/cuda"
      ;;
    gcp-a100)
      E2_EXPECTED_NODE=gcp-eu1-a100-80g-qrfh
      E2_EXPECTED_PARTITION=batch
      E2_EXPECTED_QOS=normal
      E2_EXPECTED_GPU_NAME="NVIDIA A100-SXM4-80GB"
      E2_EXPECTED_GPU_CC=8.0
      E2_TORCH_CUDA_ARCH_LIST=8.0
      E2_TORCH_EXTENSIONS_DIR="$ROOT/.cache/icra2027/e2/gcp-a100/torch-extensions-sm80"
      E2_CUDA_CACHE_PATH="$ROOT/.cache/icra2027/e2/gcp-a100/cuda-sm80"
      ;;
    *)
      echo "unsupported E2_GPU_PROFILE=$E2_GPU_PROFILE (expected hala-a6000 or gcp-a100)" >&2
      exit 2
      ;;
  esac
  export E2_GPU_PROFILE E2_EXPECTED_GPU_NAME E2_EXPECTED_GPU_CC
else
  if [[ -v E2_GPU_PROFILE ]]; then
    echo "inventory is CPU-only and must not set E2_GPU_PROFILE" >&2
    exit 2
  fi
fi

# E2 uses only ordinary Slurm jobs. Fail closed if any array metadata is
# present, including an explicitly exported empty value, so a job-array launch
# cannot accidentally masquerade as an individual scene job.
for ARRAY_VARIABLE in \
  SLURM_ARRAY_JOB_ID \
  SLURM_ARRAY_TASK_ID \
  SLURM_ARRAY_TASK_COUNT \
  SLURM_ARRAY_TASK_MIN \
  SLURM_ARRAY_TASK_MAX \
  SLURM_ARRAY_TASK_STEP
do
  if [[ -v "$ARRAY_VARIABLE" ]]; then
    echo "E2 must not run with Slurm array context: $ARRAY_VARIABLE is set" >&2
    exit 2
  fi
done

if [[ "$E2_PHASE" == render-room ]]; then
  : "${E2_SCENE_ID:?render-room requires one exact E2_SCENE_ID per individual job}"
  [[ "$E2_SCENE_ID" =~ ^[0-9a-f]{10}$ ]] || {
    echo "E2_SCENE_ID must be a lowercase 10-character scene ID: $E2_SCENE_ID" >&2
    exit 2
  }
  TASK_TAG="scene-$E2_SCENE_ID"
else
  TASK_TAG="$E2_PHASE"
fi

cd "$ROOT"
[[ -x "$PYTHON" ]] || { echo "missing E2 Python: $PYTHON" >&2; exit 2; }
[[ -s "$CONFIG" && ! -L "$CONFIG" ]] || {
  echo "missing, empty, or symlinked E2 config: $CONFIG" >&2
  exit 2
}
[[ -s "$ROSTER" && ! -L "$ROSTER" ]] || {
  echo "missing, empty, or symlinked E1/E2 roster: $ROSTER" >&2
  exit 2
}
case "$E2_CONTRACT_MANIFEST" in
  "$ROOT"/*) ;;
  *) echo "E2_CONTRACT_MANIFEST must be an absolute path below $ROOT" >&2; exit 2 ;;
esac
[[ -s "$E2_CONTRACT_MANIFEST" && ! -L "$E2_CONTRACT_MANIFEST" ]] || {
  echo "missing, empty, or symlinked E0 contract: $E2_CONTRACT_MANIFEST" >&2
  exit 2
}
[[ "$(git rev-parse HEAD)" == "$E2_CODE_COMMIT" ]] || {
  echo "code commit drift: expected $E2_CODE_COMMIT, found $(git rev-parse HEAD)" >&2
  exit 2
}
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo "refusing E2 execution from a dirty worktree" >&2
  git status --short >&2
  exit 2
}

# Validate the fixed 50-scene E2 population against E1 and bind this launch to
# an E0 contract that actually froze the current E2 config.  An older E0 smoke
# that contains the former empty placeholder manifest is not acceptable.
"$PYTHON" - "$ROOT" "$CONFIG" "$ROSTER" "$E2_CONTRACT_MANIFEST" "$E2_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1]).resolve()
config_path, roster_path, contract_path = map(Path, sys.argv[2:5])
code_commit = sys.argv[5]

def checked_repo_file(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{label} is outside the SimAny repository: {path}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SystemExit(f"{label} uses a symlink component: {current}")
    if not lexical.is_file():
        raise SystemExit(f"{label} is not a regular file: {lexical}")
    return lexical

config_path = checked_repo_file(config_path, "E2 config")
roster_path = checked_repo_file(roster_path, "E1/E2 roster")
contract_path = checked_repo_file(contract_path, "E0 contract")
config = json.loads(config_path.read_text(encoding="utf-8"))
roster = yaml.safe_load(roster_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))

population = config.get("population", {})
scene_ids = population.get("scene_ids")
smoke_ids = population.get("smoke_scene_ids")
if config.get("schema_version") != 2:
    raise SystemExit("E2 config schema_version must be 2")
if population.get("planned_scenes") != 50 or not isinstance(scene_ids, list):
    raise SystemExit("E2 config must declare planned_scenes=50 and scene_ids")
if len(scene_ids) != 50 or len(set(scene_ids)) != 50:
    raise SystemExit("E2 config scene_ids must contain 50 unique scenes")
if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{10}", value)
           for value in scene_ids):
    raise SystemExit("E2 config contains an invalid scene ID")
if not isinstance(smoke_ids, list) or len(smoke_ids) != 2 \
        or len(set(smoke_ids)) != 2 or not set(smoke_ids) <= set(scene_ids):
    raise SystemExit("E2 smoke_scene_ids must select two unique full-population scenes")
roster_population = roster.get("population", {})
raw_roster_ids = roster_population.get("scene_ids")
if not isinstance(raw_roster_ids, list):
    raise SystemExit("E1 roster population.scene_ids must be a list")
roster_scene_ids = [str(value).strip() for value in raw_roster_ids]
if roster_population.get("dataset") != "scannetpp_v2" \
        or roster_population.get("split") != "nvs_sem_val" \
        or roster_population.get("planned_scenes") != 50:
    raise SystemExit("E1 roster population metadata disagrees with E2")
if len(roster_scene_ids) != 50 or len(set(roster_scene_ids)) != 50 \
        or not all(re.fullmatch(r"[0-9a-f]{10}", value)
                   for value in roster_scene_ids):
    raise SystemExit("E1 roster must contain exactly 50 unique valid scene IDs")
if scene_ids != roster_scene_ids:
    raise SystemExit("E1 and E2 pinned scene rosters disagree")

expected_room_sources = [
    "input_scene_gaussian",
    "factorized_gt_discovery",
    "factorized_auto_discovery",
    "harmonizer_option_c",
]
observed_room_sources = [
    spec.get("source_id") for spec in config.get("room_methods", {}).values()
]
if observed_room_sources != expected_room_sources:
    raise SystemExit("E2 room method order/source IDs drifted")
minimums = config.get("coverage", {}).get("minimum_room_views_by_method", {})
if [minimums.get(source) for source in expected_room_sources] != [400, 400, 400, 0]:
    raise SystemExit("E2 full room coverage contract must be 400/400/400/0")

snapshot = contract.get("code", {})
contract_commit = snapshot.get("commit")
if snapshot.get("dirty") is not False or not isinstance(contract_commit, str) \
        or not re.fullmatch(r"[0-9a-f]{40}", contract_commit) \
        or contract_commit != code_commit:
    raise SystemExit("E0 contract is dirty or was not frozen at the E2 code commit")
declared_contract_sha = str(contract.get("contract_sha256", ""))
observed_contract_sha = hashlib.sha256(json.dumps(
    {
        key: value for key, value in contract.items()
        if key not in {"created_utc", "environment", "contract_sha256"}
    },
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")).hexdigest()
if declared_contract_sha != observed_contract_sha:
    raise SystemExit("E0 contract_sha256 does not match the contract content")
records = [item for item in contract.get("configs", [])
           if item.get("field") == "fidelity_manifest"]
if len(records) != 1:
    raise SystemExit("E0 contract must contain exactly one fidelity_manifest record")
canonical = json.dumps(
    config, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode("utf-8")
config_sha = hashlib.sha256(canonical).hexdigest()
if records[0].get("sha256") != config_sha:
    raise SystemExit(
        "E0 contract did not freeze the current E2 config; run a fresh E0 preflight"
    )
PY

FIDELITY_ROOT="$ROOT/outputs/icra2027/$E2_FREEZE_ID/fidelity"
ROOM_RUNS="$FIDELITY_ROOT/room_runs"
REPLACEMENTS="$FIDELITY_ROOT/replacements"
LPIPS_TORCH_HOME="$ROOT/.cache/icra2027/e2-render-compat/torch"
LPIPS_WEIGHTS="$LPIPS_TORCH_HOME/hub/checkpoints/alexnet-owt-7be5be79.pth"
LPIPS_WEIGHTS_SIZE=244408911
LPIPS_WEIGHTS_SHA256=7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02
export TMPDIR="$ROOT/.tmp/icra2027/e2-${SLURM_JOB_ID}-${TASK_TAG}"
export XDG_CACHE_HOME="$ROOT/.cache/icra2027/e2/xdg"
export HF_HOME="$ROOT/.cache/icra2027/e2/huggingface"
export TORCH_HOME="$LPIPS_TORCH_HOME"
export MPLCONFIGDIR="$ROOT/.cache/icra2027/e2/matplotlib"
if [[ "$E2_PHASE" == render-room || "$E2_PHASE" == metrics ]]; then
  export TORCH_EXTENSIONS_DIR="$E2_TORCH_EXTENSIONS_DIR"
  export CUDA_CACHE_PATH="$E2_CUDA_CACHE_PATH"
  export TORCH_CUDA_ARCH_LIST="$E2_TORCH_CUDA_ARCH_LIST"
else
  unset TORCH_EXTENSIONS_DIR CUDA_CACHE_PATH TORCH_CUDA_ARCH_LIST
fi
export E2_LPIPS_WEIGHTS_PATH="${LPIPS_WEIGHTS#$ROOT/}"
export E2_LPIPS_WEIGHTS_SIZE="$LPIPS_WEIGHTS_SIZE"
export E2_LPIPS_WEIGHTS_SHA256="$LPIPS_WEIGHTS_SHA256"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONOPTIMIZE
RUNTIME_CACHE_DIRS=("$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR")
if [[ "$E2_PHASE" == render-room || "$E2_PHASE" == metrics ]]; then
  RUNTIME_CACHE_DIRS+=("$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH")
fi
mkdir -p "${RUNTIME_CACHE_DIRS[@]}"
[[ "$($PYTHON -c 'print(__debug__)')" == True ]] || {
  echo "E2 launcher requires Python assertions enabled" >&2
  exit 2
}

# Both GPU phases use the already proven repository-local AlexNet cache.  Do
# not copy or download it: a missing, symlinked, resized, or modified file is a
# provenance failure.  Export the identity so the metrics subprocess can bind
# it into the final run manifest.
if [[ "$E2_PHASE" == render-room || "$E2_PHASE" == metrics ]]; then
  [[ -d "$TORCH_HOME" && ! -L "$TORCH_HOME" ]] || {
    echo "missing or symlinked proven TORCH_HOME: $TORCH_HOME" >&2
    exit 2
  }
  [[ -f "$LPIPS_WEIGHTS" && ! -L "$LPIPS_WEIGHTS" ]] || {
    echo "missing or symlinked AlexNet LPIPS weights: $LPIPS_WEIGHTS" >&2
    exit 2
  }
  [[ "$(stat -c %s "$LPIPS_WEIGHTS")" == "$LPIPS_WEIGHTS_SIZE" ]] || {
    echo "AlexNet LPIPS weight size drift: $LPIPS_WEIGHTS" >&2
    exit 2
  }
  [[ "$(sha256sum "$LPIPS_WEIGHTS" | cut -d' ' -f1)" == "$LPIPS_WEIGHTS_SHA256" ]] || {
    echo "AlexNet LPIPS weight SHA-256 drift: $LPIPS_WEIGHTS" >&2
    exit 2
  }
fi

validate_gpu_runtime() {
  local phase="$1"
  [[ "${SLURMD_NODENAME:-}" == "$E2_EXPECTED_NODE" ]] || {
    echo "$phase profile=$E2_GPU_PROFILE requires node=$E2_EXPECTED_NODE; found ${SLURMD_NODENAME:-unset}" >&2
    exit 2
  }
  [[ "${SLURM_JOB_PARTITION:-}" == "$E2_EXPECTED_PARTITION" ]] || {
    echo "$phase profile=$E2_GPU_PROFILE requires partition=$E2_EXPECTED_PARTITION; found ${SLURM_JOB_PARTITION:-unset}" >&2
    exit 2
  }
  [[ "${SLURM_JOB_QOS:-}" == "$E2_EXPECTED_QOS" ]] || {
    echo "$phase profile=$E2_GPU_PROFILE requires qos=$E2_EXPECTED_QOS; found ${SLURM_JOB_QOS:-unset}" >&2
    exit 2
  }
  local visible_gpus="${CUDA_VISIBLE_DEVICES:-}"
  [[ -n "$visible_gpus" && "$visible_gpus" != NoDevFiles && "$visible_gpus" != *,* ]] || {
    echo "$phase requires exactly one visible GPU; CUDA_VISIBLE_DEVICES=$visible_gpus" >&2
    exit 2
  }
  "$PYTHON" - "$E2_EXPECTED_GPU_NAME" "$E2_EXPECTED_GPU_CC" <<'PY'
import sys

import torch

expected_name, expected_cc = sys.argv[1:]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to the E2 GPU phase")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"E2 GPU phase requires one visible CUDA device; found {torch.cuda.device_count()}"
    )
observed_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
observed_cc = f"{major}.{minor}"
if observed_name != expected_name:
    raise SystemExit(
        f"GPU name mismatch: expected {expected_name!r}, found {observed_name!r}"
    )
if observed_cc != expected_cc:
    raise SystemExit(
        f"GPU compute capability mismatch: expected {expected_cc}, found {observed_cc}"
    )
print(f"gpu_profile={__import__('os').environ['E2_GPU_PROFILE']}")
print(f"gpu_name={observed_name}")
print(f"gpu_compute_capability={observed_cc}")
PY
}

mapfile -t FULL_SCENES < <(
  "$PYTHON" - "$CONFIG" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
for scene_id in config["population"]["scene_ids"]:
    print(scene_id)
PY
)
mapfile -t SMOKE_SCENES < <(
  "$PYTHON" - "$CONFIG" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
for scene_id in config["population"]["smoke_scene_ids"]:
    print(scene_id)
PY
)
[[ ${#FULL_SCENES[@]} -eq 50 ]] || {
  echo "E2 config must pin exactly 50 full scenes; found ${#FULL_SCENES[@]}" >&2
  exit 2
}
[[ ${#SMOKE_SCENES[@]} -eq 2 ]] || {
  echo "E2 config must pin exactly two smoke scenes; found ${#SMOKE_SCENES[@]}" >&2
  exit 2
}
if [[ "$E2_MODE" == full ]]; then
  SCENES=("${FULL_SCENES[@]}")
else
  SCENES=("${SMOKE_SCENES[@]}")
fi

postcheck_room_scene() {
  "$PYTHON" - "$1" "$2" "$E2_FREEZE_ID" "$E2_CODE_COMMIT" "$ROOT" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path

scene_root = Path(sys.argv[1])
expected_scene, expected_freeze, expected_commit = sys.argv[2:5]
repo_root = Path(sys.argv[5])
manifest_path = scene_root / "manifest.json"
if manifest_path.is_symlink() or not manifest_path.is_file():
    raise SystemExit(f"missing or symlinked room manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
methods = [
    "input_scene_gaussian",
    "factorized_gt_discovery",
    "factorized_auto_discovery",
]
assert manifest["schema_version"] == 1
assert manifest["dataset_id"] == "scannetpp_v2"
assert manifest["split_id"] == "nvs_sem_val:official-test"
assert manifest["scene_id"] == expected_scene
assert manifest["freeze_id"] == expected_freeze
assert manifest["evaluation_unit"] == "held_out_view"
assert manifest["n_eval_frames"] == 8
assert len(manifest["eval_frames"]) == len(set(manifest["eval_frames"])) == 8
assert manifest["methods"] == methods
assert manifest["git"]["commit"] == expected_commit
assert manifest["git"]["dirty"] is False
assert manifest["paper_ready"] is False

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def check_file(path, record, label):
    assert not path.is_symlink() and path.is_file(), (label, path)
    assert record["size_bytes"] == path.stat().st_size, (label, path, "size")
    assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), (label, "sha")
    assert sha256(path) == record["sha256"], (label, path, "drift")

train = manifest["official_train_frames"]
optimization = manifest["optimization_input_frames"]
assert isinstance(train, list) and train and len(train) == len(set(train))
assert isinstance(optimization, list) and optimization and len(optimization) == len(set(optimization))
assert all(Path(name).name == name and name not in {".", ".."} for name in train)
assert all(Path(name).name == name and name not in {".", ".."} for name in optimization)
assert not (set(manifest["eval_frames"]) & set(optimization))

dataset_scene = Path("/data/ScanNetpp/data") / expected_scene
expected_upstream = (
    (Path("/data/ScanNetppv2_gsplat/splats") / f"{expected_scene}.ply",
     manifest["source_scene_gaussian"], "scene Gaussian"),
    (dataset_scene / "dslr/train_test_lists.json",
     manifest["split_artifact"], "official split"),
    (dataset_scene / "dslr/nerfstudio/transforms_undistorted.json",
     manifest["camera_artifacts"]["intrinsics"], "intrinsics"),
    (dataset_scene / "dslr/colmap/images.txt",
     manifest["camera_artifacts"]["poses"], "camera poses"),
)
for expected_path, record, label in expected_upstream:
    assert Path(record["path"]) == expected_path, (label, record["path"])
    check_file(expected_path, record, label)
for record in manifest["source_images"]:
    frame = record["frame"]
    assert frame in manifest["eval_frames"] and Path(frame).name == frame
    expected_path = dataset_scene / "dslr/resized_undistorted_images" / frame
    assert Path(record["path"]) == expected_path
    check_file(expected_path, record, "source image")

for source_id in ("factorized_gt_discovery", "factorized_auto_discovery"):
    source = manifest["construction_sources"][source_id]
    source_frames = source["optimization_input_frames"]
    assert set(source_frames) <= set(optimization)
    assert not (set(source_frames) & set(manifest["eval_frames"]))
    for record in source["artifacts"]:
        artifact_path = Path(os.path.abspath(repo_root / record["path"]))
        try:
            artifact_path.relative_to(repo_root)
        except ValueError:
            raise AssertionError((source_id, "artifact outside repository"))
        check_file(artifact_path, record, f"{source_id} artifact")
    replacement = source.get("replacement_bundle")
    if replacement is not None:
        replacement_manifest = Path(os.path.abspath(
            repo_root / replacement["manifest_path"]
        ))
        replacement_manifest.relative_to(repo_root)
        assert not replacement_manifest.is_symlink() and replacement_manifest.is_file()
        assert sha256(replacement_manifest) == replacement["manifest_sha256"]
        bundle_root = replacement_manifest.parent
        new_frames = set()
        old_frames = set()
        for target in replacement["targets"]:
            assert target["optimization_input_frames"] == [target["new_train_frame"]]
            new_frames.add(target["new_train_frame"])
            old_frames.add(target["old_test_frame"])
            for record in target["generated_artifacts"]:
                check_file(bundle_root / record["path"], record, "replacement artifact")
        assert new_frames <= set(source_frames)
        assert not (old_frames & set(source_frames))

expected_names = sorted(f"{Path(name).stem}.png" for name in manifest["eval_frames"])
assert len(expected_names) == len(set(expected_names)) == 8
expected_paths = set()
for directory in ["gt", *methods]:
    path = scene_root / directory
    assert path.is_dir() and not path.is_symlink()
    names = sorted(p.name for p in path.iterdir() if p.is_file() and not p.is_symlink())
    assert names == expected_names, (directory, names, expected_names)
    assert not any(p.is_dir() or p.is_symlink() for p in path.iterdir())
    expected_paths.update(f"{directory}/{name}" for name in names)

artifacts = manifest["output_artifacts"]
assert len(artifacts) == 32
assert {item["relative_path"] for item in artifacts} == expected_paths
for item in artifacts:
    assert re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
    path = scene_root / item["relative_path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == item["sha256"], path
PY
}

case "$E2_PHASE" in
  render-room)
    SCENE_ID="$E2_SCENE_ID"
    SCENE_SELECTED=false
    for EXPECTED_SCENE_ID in "${SCENES[@]}"; do
      if [[ "$SCENE_ID" == "$EXPECTED_SCENE_ID" ]]; then
        SCENE_SELECTED=true
        break
      fi
    done
    [[ "$SCENE_SELECTED" == true ]] || {
      echo "E2_SCENE_ID=$SCENE_ID is not in the exact $E2_MODE scene roster" >&2
      exit 2
    }
    validate_gpu_runtime render-room
    SCENE_OUT="$ROOM_RUNS/$SCENE_ID"
    if [[ -e "$SCENE_OUT" || -L "$SCENE_OUT" ]]; then
      [[ -d "$SCENE_OUT" && ! -L "$SCENE_OUT" ]] || {
        echo "existing E2 room output is not a regular directory: $SCENE_OUT" >&2
        exit 2
      }
      # Resume without mutation: a repeated individual job verifies and reuses
      # an already complete scene bundle. Corrupt or drifted bundles fail
      # closed in postcheck_room_scene and are never overwritten implicitly.
      postcheck_room_scene "$SCENE_OUT" "$SCENE_ID"
      echo "E2_RENDER_ROOM=REUSED"
      echo "mode=$E2_MODE"
      echo "scene_id=$SCENE_ID"
      echo "output=$SCENE_OUT"
      exit 0
    fi
    mkdir -p "$ROOM_RUNS"
    export SIMANY_ROOT="$ROOT"
    export SIMANY_SCANNETPP_ROOT=/data/ScanNetpp
    export SIMANY_SPLATS_ROOT=/data/ScanNetppv2_gsplat/splats
    unset SIMF_ROOT SIMF_SCANNETPP_ROOT SIMF_SPLATS_ROOT
    # Exact already-validated read-only CUDA environment. Generated caches and
    # every output remain pinned under SimAny by the exports above.
    export SIMANY_GSPLAT_PY=/group/worldcept/code/affordancept/.envs/mini-viewer/bin/python
    source "$ROOT/run/env.sh"
    [[ "$ROOT" == /group/worldcept/PhiRIE/code/SimAny ]] || {
      echo "run/env.sh changed the pinned SimAny root: $ROOT" >&2
      exit 2
    }
    [[ -x "$MVPY" ]] || { echo "gsplat Python is not executable: $MVPY" >&2; exit 2; }
    run_gs agents.eval.fidelity_room_export \
      --scene-id "$SCENE_ID" \
      --factory-source "$ROOT/outputs/${SCENE_ID}_factory" \
      --auto-source "$ROOT/outputs/${SCENE_ID}_auto" \
      --out "$SCENE_OUT" \
      --freeze-id "$E2_FREEZE_ID" \
      --config "$CONFIG" \
      --replacement-root "$REPLACEMENTS"
    postcheck_room_scene "$SCENE_OUT" "$SCENE_ID"
    echo "E2_RENDER_ROOM=PASS"
    echo "mode=$E2_MODE"
    echo "scene_id=$SCENE_ID"
    echo "output=$SCENE_OUT"
    echo "gpu_profile=$E2_GPU_PROFILE"
    echo "gpu_name=$E2_EXPECTED_GPU_NAME"
    echo "gpu_compute_capability=$E2_EXPECTED_GPU_CC"
    echo "torch_home=$TORCH_HOME"
    echo "lpips_weights_sha256=$E2_LPIPS_WEIGHTS_SHA256"
    ;;

  inventory)
    case "${CUDA_VISIBLE_DEVICES:-}" in
      ""|NoDevFiles) ;;
      *) echo "inventory must be CPU-only; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2; exit 2 ;;
    esac
    : "${E2_REVIEWED_OBJECT_AGGREGATE:=$ROOT/outputs/review_recompute_hybrid.json}"
    case "$E2_REVIEWED_OBJECT_AGGREGATE" in
      "$ROOT"/*) ;;
      *) echo "E2_REVIEWED_OBJECT_AGGREGATE must stay below $ROOT" >&2; exit 2 ;;
    esac
    [[ -s "$E2_REVIEWED_OBJECT_AGGREGATE" && ! -L "$E2_REVIEWED_OBJECT_AGGREGATE" ]] || {
      echo "missing, empty, or symlinked reviewed object aggregate: $E2_REVIEWED_OBJECT_AGGREGATE" >&2
      exit 2
    }
    [[ -d "$ROOM_RUNS" && ! -L "$ROOM_RUNS" ]] || {
      echo "missing or symlinked E2 room-runs root: $ROOM_RUNS" >&2
      exit 2
    }
    [[ ! -e "$FIDELITY_ROOT/inventory" && ! -L "$FIDELITY_ROOT/inventory" ]] || {
      echo "refusing to overwrite E2 inventory: $FIDELITY_ROOT/inventory" >&2
      exit 2
    }
    "$PYTHON" - "$ROOM_RUNS" "${SCENES[@]}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = set(sys.argv[2:])
observed = {path.name for path in root.iterdir() if path.is_dir() and not path.is_symlink()}
assert observed == expected, {"missing": sorted(expected - observed), "unexpected": sorted(observed - expected)}
PY
    for SCENE_ID in "${SCENES[@]}"; do
      postcheck_room_scene "$ROOM_RUNS/$SCENE_ID" "$SCENE_ID"
    done

    INVENTORY_ARGS=()
    if [[ "$E2_MODE" == smoke ]]; then
      INVENTORY_ARGS=(
        --smoke
        --scene-id "${SMOKE_SCENES[0]}"
        --scene-id "${SMOKE_SCENES[1]}"
      )
    fi
    "$PYTHON" -m robo.eval.fidelity_inventory \
      --config "$CONFIG" \
      --scene-list "$ROSTER" \
      --outputs-root "$ROOT/outputs" \
      --room-runs-root "$ROOM_RUNS" \
      --reviewed-object-aggregate "$E2_REVIEWED_OBJECT_AGGREGATE" \
      --freeze-id "$E2_FREEZE_ID" \
      --out "$FIDELITY_ROOT/inventory" \
      --contract-manifest "$E2_CONTRACT_MANIFEST" \
      "${INVENTORY_ARGS[@]}"

    "$PYTHON" - "$E2_MODE" "$FIDELITY_ROOT" "$E2_FREEZE_ID" "$E2_CODE_COMMIT" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

mode, root_text, freeze_id, code_commit = sys.argv[1:]
root = Path(root_text)
validation = json.loads((root / "inventory/validation_report.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "inventory/manifests/fidelity_manifest.json").read_text(encoding="utf-8"))

selected_scenes = 50 if mode == "full" else 2
expected_views = 400 if mode == "full" else 16
assert validation["mode"] == mode
assert validation["population"]["planned_scene_count"] == 50
assert validation["population"]["selected_scene_count"] == selected_scenes
assert validation["population"]["full_population_preserved"] is (mode == "full")
assert validation["valid"] is True
assert re.fullmatch(r"[0-9a-f]{64}", validation["manifest_sha256"])
assert validation["inventory_git"]["commit"] == code_commit
assert validation["inventory_git"]["dirty"] is False
assert validation["room_exports"]["valid_scene_count"] == selected_scenes
assert validation["room_exports"]["invalid_scene_count"] == 0
assert manifest["freeze_id"] == freeze_id
assert manifest["paper_ready"] is False
for source_id in (
    "input_scene_gaussian",
    "factorized_gt_discovery",
    "factorized_auto_discovery",
):
    assert validation["coverage"]["observed_room_views_by_method"][source_id] == expected_views
assert validation["coverage"]["observed_room_views_by_method"]["harmonizer_option_c"] == 0
if mode == "full":
    assert validation["room_anchor_reproduction"]["all_reproduced"] is True

run_manifest = {
    "schema_version": 1,
    "task": "E2",
    "phase": "inventory",
    "mode": mode,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
    "node": os.environ.get("SLURMD_NODENAME"),
    "code_commit": code_commit,
    "freeze_id": freeze_id,
    "paper_ready": False,
    "scene_count": selected_scenes,
    "views_per_non_harmonized_method": expected_views,
}
with (root / "inventory_run_manifest.json").open("x", encoding="utf-8") as handle:
    json.dump(run_manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
    echo "E2_INVENTORY=PASS"
    echo "mode=$E2_MODE"
    echo "freeze_id=$E2_FREEZE_ID"
    echo "output=$FIDELITY_ROOT"
    echo "paper_ready=false"
    ;;

  metrics)
    validate_gpu_runtime metrics
    [[ -s "$FIDELITY_ROOT/inventory/validation_report.json" ]] || {
      echo "missing E2 inventory validation report" >&2
      exit 2
    }
    [[ -s "$FIDELITY_ROOT/inventory/manifests/fidelity_manifest.json" ]] || {
      echo "missing E2 canonical fidelity manifest" >&2
      exit 2
    }
    [[ ! -e "$FIDELITY_ROOT/table" && ! -L "$FIDELITY_ROOT/table" ]] || {
      echo "refusing to overwrite E2 table: $FIDELITY_ROOT/table" >&2
      exit 2
    }
    [[ ! -e "$FIDELITY_ROOT/run_manifest.json" && ! -L "$FIDELITY_ROOT/run_manifest.json" ]] || {
      echo "refusing to overwrite E2 final run manifest" >&2
      exit 2
    }
    [[ ! -e "$FIDELITY_ROOT/artifact_hashes.sha256" && ! -L "$FIDELITY_ROOT/artifact_hashes.sha256" ]] || {
      echo "refusing to overwrite E2 artifact hash list" >&2
      exit 2
    }
    "$PYTHON" - "$FIDELITY_ROOT/inventory/validation_report.json" \
      "$FIDELITY_ROOT/inventory/manifests/fidelity_manifest.json" \
      "$E2_FREEZE_ID" "$E2_CODE_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

validation_path, manifest_path = map(Path, sys.argv[1:3])
freeze_id, code_commit = sys.argv[3:]
validation = json.loads(validation_path.read_text(encoding="utf-8"))
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert validation["valid"] is True
assert validation["freeze_id"] == freeze_id
assert validation["inventory_git"]["commit"] == code_commit
assert validation["inventory_git"]["dirty"] is False
assert validation["manifest_sha256"] == manifest_sha
PY
    "$PYTHON" -m robo.eval.fidelity_metrics \
      --manifest "$FIDELITY_ROOT/inventory/manifests/fidelity_manifest.json" \
      --out "$FIDELITY_ROOT/table" \
      --lpips-device cuda

    "$PYTHON" - "$E2_MODE" "$FIDELITY_ROOT" "$E2_FREEZE_ID" "$E2_CODE_COMMIT" <<'PY'
import hashlib
import json
import os
import re
import sys
from pathlib import Path

mode, root_text, freeze_id, code_commit = sys.argv[1:]
root = Path(root_text)
validation = json.loads((root / "inventory/validation_report.json").read_text(encoding="utf-8"))
table_path = root / "table/fidelity_table.json"
csv_path = root / "table/fidelity_table.csv"
table = json.loads(table_path.read_text(encoding="utf-8"))
selected_scenes = 50 if mode == "full" else 2
expected_views = 400 if mode == "full" else 16
assert validation["valid"] is True
assert validation["freeze_id"] == freeze_id
assert validation["inventory_git"]["commit"] == code_commit
assert table["schema_version"] == 2
assert table["freeze_id"] == freeze_id
assert table["paper_ready"] is False
assert table["validation"]["valid"] is True
assert table["validation"]["paper_ready"] is False
assert table["provenance"]["code_commit"] == code_commit
assert table["provenance"]["freeze_id"] == freeze_id
assert re.fullmatch(r"[0-9a-f]{64}", table["manifest_sha256"])
manifest_path = root / "inventory/manifests/fidelity_manifest.json"
assert Path(table["manifest_path"]).resolve() == manifest_path.resolve()
assert table["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert table["provenance"]["manifest_sha256"] == table["manifest_sha256"]
assert table["lpips_backend_error"] is None
rows = table["rows"]
assert len(rows) == 8
assert table["rows_count"] == len(rows)
assert csv_path.is_file() and not csv_path.is_symlink()
by_source = {row["source_id"]: row for row in rows}
for source_id in (
    "input_scene_gaussian",
    "factorized_gt_discovery",
    "factorized_auto_discovery",
):
    row = by_source[source_id]
    assert row["unit"] == "room"
    assert row["n_images"] == expected_views
    assert row["metric_samples"]["psnr"] == expected_views
    assert row["metric_samples"]["ssim"] == expected_views
    assert row["metric_samples"]["lpips"] == expected_views

run_manifest = {
    "schema_version": 1,
    "task": "E2",
    "phase": "metrics",
    "mode": mode,
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
    "node": os.environ.get("SLURMD_NODENAME"),
    "code_commit": code_commit,
    "freeze_id": freeze_id,
    "paper_ready": False,
    "scene_count": selected_scenes,
    "views_per_non_harmonized_method": expected_views,
    "lpips_device": "cuda",
    "gpu": {
        "profile": os.environ["E2_GPU_PROFILE"],
        "name": os.environ["E2_EXPECTED_GPU_NAME"],
        "compute_capability": os.environ["E2_EXPECTED_GPU_CC"],
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "qos": os.environ["SLURM_JOB_QOS"],
    },
    "lpips_weights": {
        "path": os.environ["E2_LPIPS_WEIGHTS_PATH"],
        "size_bytes": int(os.environ["E2_LPIPS_WEIGHTS_SIZE"]),
        "sha256": os.environ["E2_LPIPS_WEIGHTS_SHA256"],
    },
}
with (root / "run_manifest.json").open("x", encoding="utf-8") as handle:
    json.dump(run_manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

    (
      cd "$FIDELITY_ROOT"
      { find inventory table room_runs -type f -print0; \
        if [[ -d replacements ]]; then find replacements -type f -print0; fi; \
        printf '%s\0' inventory_run_manifest.json run_manifest.json; } \
        | sort -z \
        | xargs -0 sha256sum > artifact_hashes.sha256
    )
    echo "E2_METRICS=PASS"
    echo "mode=$E2_MODE"
    echo "freeze_id=$E2_FREEZE_ID"
    echo "output=$FIDELITY_ROOT"
    echo "gpu_profile=$E2_GPU_PROFILE"
    echo "gpu_name=$E2_EXPECTED_GPU_NAME"
    echo "gpu_compute_capability=$E2_EXPECTED_GPU_CC"
    echo "lpips_weights_sha256=$E2_LPIPS_WEIGHTS_SHA256"
    echo "paper_ready=false"
    ;;
esac
