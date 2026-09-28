#!/usr/bin/env bash
# One ordinary A100 job for the sealed E4 camera/workspace/scorer gate.
# Exact resources are supplied by submit_e4_camera_scorer_noarray.sh.
set -euo pipefail

export PATH=/usr/local/bin:/usr/bin:/bin
CODE_ROOT=${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot
EVIDENCE_ROOT=${SIMANY_ROOT:-$PWD}
PYTHON=${SIMANY_ROOT:-$PWD}/.venv/bin/python
MENAGERIE_ROOT="$CODE_ROOT/third_party/mujoco_menagerie"
OPENPI_ROOT=${OPENPI_ROOT:?set OPENPI_ROOT to the openpi checkout}/worktrees/e4-policy-server
OPENPI_CLIENT_SRC="$OPENPI_ROOT/packages/openpi-client/src"
OPENPI_COMMIT=2f51088169d2e2b480ce54faa737be9b0279eed4
CPU_FREEZE_ID=icra2027-contract-v1-e4-0dda134578b2-region-cpu-prep1-20260904T135257Z
CPU_PRODUCER_COMMIT=0dda134578b25cd12d1791193bb437c6d310776f
CPU_GATE_SHA256=b4bb242ace4ce641051e58df003d126a1383ae0ddd003e3fe4dce3e0493ef355
MENAGERIE_SOURCE_COMMIT=71f066ad0be9cd271f7ed58c030243ef157af9f4
ACCOUNT=${SLURM_ACCOUNT:-phirie}
GPU_NODE=gcp-eu1-a100-80g-qrfh
EXPECTED_TIME=01:00:00
EXPECTED_MEM_MB=32768

: "${E4_CAMERA_GATE_ID:?E4_CAMERA_GATE_ID is required}"
: "${E4_CAMERA_CODE_COMMIT:?E4_CAMERA_CODE_COMMIT is required}"
: "${E4_SUBMIT_USER:?E4_SUBMIT_USER is required}"
: "${E4_QUALIFIER_PREFLIGHT_GATE_SHA256:?qualifier preflight gate SHA is required}"
[[ "$E4_CAMERA_GATE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_CAMERA_GATE_ID: $E4_CAMERA_GATE_ID" >&2
  exit 2
}
[[ "$E4_CAMERA_CODE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "E4_CAMERA_CODE_COMMIT must be a full lowercase SHA" >&2
  exit 2
}
[[ "$E4_QUALIFIER_PREFLIGHT_GATE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
  echo "qualifier preflight gate SHA must be a lowercase SHA-256" >&2
  exit 2
}
[[ "$E4_SUBMIT_USER" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "unsafe E4_SUBMIT_USER: $E4_SUBMIT_USER" >&2
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
    echo "E4 camera gate refuses Slurm array context: $array_variable is set" >&2
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
[[ -d "$MENAGERIE_ROOT" && ! -L "$MENAGERIE_ROOT" ]] || {
  echo "missing or symlinked local Menagerie copy: $MENAGERIE_ROOT" >&2
  exit 2
}
[[ -d "$OPENPI_CLIENT_SRC/openpi_client" && ! -L "$OPENPI_ROOT" ]] || {
  echo "missing clean OpenPI client source: $OPENPI_CLIENT_SRC" >&2
  exit 2
}
[[ "$(git -C "$OPENPI_ROOT" rev-parse --show-toplevel)" == "$OPENPI_ROOT" \
  && "$(git -C "$OPENPI_ROOT" rev-parse HEAD)" == "$OPENPI_COMMIT" \
  && -z "$(git -C "$OPENPI_ROOT" status --porcelain --untracked-files=normal)" ]] || {
  echo "OpenPI resize source is not at its clean pinned commit" >&2
  exit 2
}
cpu_gate="$EVIDENCE_ROOT/outputs/icra2027/$CPU_FREEZE_ID/preflight/gate.json"
[[ -s "$cpu_gate" && ! -L "$cpu_gate" ]] || {
  echo "missing sealed CPU gate: $cpu_gate" >&2
  exit 2
}
[[ "$(sha256sum "$cpu_gate" | awk '{print $1}')" == "$CPU_GATE_SHA256" ]] || {
  echo "CPU gate SHA differs from external anchor" >&2
  exit 2
}
qualifier_gate="$EVIDENCE_ROOT/outputs/icra2027/${E4_CAMERA_GATE_ID}-qualifier-preflight/gate.json"
qualifier_seal="$EVIDENCE_ROOT/outputs/icra2027/${E4_CAMERA_GATE_ID}-qualifier-preflight/seal.json"
[[ -s "$qualifier_gate" && ! -L "$qualifier_gate" \
  && -s "$qualifier_seal" && ! -L "$qualifier_seal" ]] || {
  echo "missing sealed CPU qualifier preflight" >&2
  exit 2
}
[[ "$(sha256sum "$qualifier_gate" | awk '{print $1}')" == \
  "$E4_QUALIFIER_PREFLIGHT_GATE_SHA256" ]] || {
  echo "CPU qualifier preflight SHA differs from submission anchor" >&2
  exit 2
}

observed_root=$(git -C "$CODE_ROOT" rev-parse --show-toplevel)
observed_commit=$(git -C "$CODE_ROOT" rev-parse HEAD)
[[ "$observed_root" == "$CODE_ROOT" ]] || {
  echo "launcher resolved a different code root: $observed_root" >&2
  exit 2
}
[[ "$observed_commit" == "$E4_CAMERA_CODE_COMMIT" ]] || {
  echo "camera validator commit drift: expected=$E4_CAMERA_CODE_COMMIT observed=$observed_commit" >&2
  exit 2
}
if [[ -n "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "camera launcher refuses a dirty code worktree" >&2
  git -C "$CODE_ROOT" status --short >&2
  exit 2
fi

: "${SLURM_JOB_ID:?camera gate must run under Slurm}"
[[ "${SLURM_JOB_ACCOUNT:-}" == "$ACCOUNT" ]] || {
  echo "wrong account: ${SLURM_JOB_ACCOUNT:-unset}" >&2; exit 2;
}
[[ "${SLURM_JOB_PARTITION:-}" == batch ]] || {
  echo "wrong partition: ${SLURM_JOB_PARTITION:-unset}" >&2; exit 2;
}
[[ "${SLURM_JOB_QOS:-}" == normal ]] || {
  echo "wrong QoS: ${SLURM_JOB_QOS:-unset}" >&2; exit 2;
}
[[ "${SLURMD_NODENAME:-}" == "$GPU_NODE" ]] || {
  echo "wrong GPU node: ${SLURMD_NODENAME:-unset}" >&2; exit 2;
}
[[ "${SLURM_NNODES:-}" == 1 && "${SLURM_NTASKS:-}" == 1 ]] || {
  echo "camera gate requires exactly one node and task" >&2; exit 2;
}
[[ "${SLURM_CPUS_PER_TASK:-}" == 4 ]] || {
  echo "qrfh camera gate requires exactly 4 CPUs per GPU" >&2; exit 2;
}
case "${SLURM_MEM_PER_NODE:-}" in
  "$EXPECTED_MEM_MB"|32G) ;;
  *) echo "camera gate requires mem=32G" >&2; exit 2 ;;
esac
visible=${CUDA_VISIBLE_DEVICES:-}
[[ -n "$visible" && "$visible" != NoDevFiles && "$visible" != *,* ]] || {
  echo "camera gate requires exactly one visible GPU: $visible" >&2
  exit 2
}
[[ -n "${SLURM_JOB_GPUS:-}" && "${SLURM_JOB_GPUS}" != *,* ]] || {
  echo "camera gate requires one Slurm GPU" >&2
  exit 2
}

job_record=$(scontrol show job "$SLURM_JOB_ID" --oneliner)
[[ "$job_record" == *" Partition=batch "* \
  && "$job_record" == *" QOS=normal "* \
  && "$job_record" == *" Account=$ACCOUNT "* \
  && "$job_record" == *" UserId=$E4_SUBMIT_USER("* \
  && "$job_record" == *" Requeue=0 "* \
  && "$job_record" == *" TimeLimit=$EXPECTED_TIME "* \
  && "$job_record" == *" NumNodes=1 "* \
  && "$job_record" == *" NumCPUs=4 "* \
  && "$job_record" == *" NumTasks=1 "* \
  && "$job_record" == *" CPUs/Task=4 "* ]] || {
  echo "scontrol allocation contract differs: $job_record" >&2
  exit 2
}
case "$job_record" in
  *" MinMemoryNode=32G "*|*" MinMemoryNode=${EXPECTED_MEM_MB}M "*|*" MinMemoryNode=$EXPECTED_MEM_MB "*) ;;
  *) echo "scontrol memory contract differs: $job_record" >&2; exit 2 ;;
esac
[[ "$job_record" == *" Command=$CODE_ROOT/run/slurm/icra2027_e4_camera_scorer_gpu.sbatch "* ]] || {
  echo "scontrol launcher command differs: $job_record" >&2
  exit 2
}
job_record_lower=${job_record,,}
[[ "$job_record_lower" == *"gres/gpu"* && "$job_record_lower" == *"a100-80g:1"* ]] || {
  echo "scontrol GPU GRES contract differs: $job_record" >&2
  exit 2
}

# Drop caller state, retain only scheduler-provided GPU visibility, and build
# the renderer environment from fixed values. Runtime scratch belongs to the
# durable submission directory, never the still-fresh experiment root.
unset PYTHONPATH PYTHONOPTIMIZE PYTHONUSERBASE RESUME AUTO FULL
while IFS= read -r inherited_variable; do unset "$inherited_variable"; done \
  < <(compgen -A variable SIMANY_ || true)
while IFS= read -r inherited_variable; do unset "$inherited_variable"; done \
  < <(compgen -A variable SIMF_ || true)
export PYTHONPATH="$OPENPI_CLIENT_SRC:$CODE_ROOT"
export PYTHONNOUSERSITE=1
export SIMANY_EVIDENCE_ROOT="$EVIDENCE_ROOT"
export SIMANY_ROOT="$CODE_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4
"$PYTHON" - "$qualifier_gate" "$qualifier_seal" "$E4_CAMERA_CODE_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

gate_path, seal_path = (Path(value) for value in sys.argv[1:3])
expected_commit = sys.argv[3]
for path in (gate_path, seal_path):
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise SystemExit(f"invalid qualifier preflight artifact: {path}")
gate_bytes = gate_path.read_bytes()
gate = json.loads(gate_bytes)
seal = json.loads(seal_path.read_text(encoding="utf-8"))
if seal.get("members", {}).get("gate.json") != {
    "sha256": hashlib.sha256(gate_bytes).hexdigest(),
    "size_bytes": len(gate_bytes),
}:
    raise SystemExit("qualifier preflight gate is not sealed")
if not (
    gate.get("manifest_kind") == "e4_cpu_qualifier_preflight_gate"
    and gate.get("camera_validator_code", {}).get("commit") == expected_commit
    and gate.get("cell_coverage", {}).get("cells") == 40
    and gate.get("preflight_cells_passed") == 40
    and gate.get("preflight_cells_failed") == 0
    and gate.get("qualifier_margin_cells_passed") == 40
    and gate.get("qualifier_margin_cells_failed") == 0
    and gate.get("settle_contract_cells_passed") == 40
    and gate.get("settle_contract_cells_failed") == 0
    and gate.get("reset_contract_cells_passed") == 40
    and gate.get("reset_contract_cells_failed") == 0
    and gate.get("camera_job_submission_allowed") is True
    and gate.get("real_policy_infra_smoke_allowed") is False
    and gate.get("large_rollout_launch_allowed") is False
    and gate.get("paper_ready") is False
    and gate.get("renderer_constructed") is False
    and gate.get("gpu_work_executed") is False
):
    raise SystemExit("qualifier preflight is not a 40/40 camera-only authorization")
PY
submission_dir="$EVIDENCE_ROOT/outputs/icra2027/submissions/${E4_CAMERA_GATE_ID}-camera-scorer"
[[ -d "$submission_dir" && ! -L "$submission_dir" ]] || {
  echo "missing durable submission directory: $submission_dir" >&2
  exit 2
}
runtime_root="$submission_dir/runtime/$SLURM_JOB_ID"
mkdir "$runtime_root"
mkdir "$runtime_root/tmp" "$runtime_root/xdg-cache"
export TMPDIR="$runtime_root/tmp"
export XDG_CACHE_HOME="$runtime_root/xdg-cache"
cd "$CODE_ROOT"

"$PYTHON" - <<'PY'
from pathlib import Path
import robo.eval.e4_camera_scorer_gate as gate

expected = Path("${SIMANY_ROOT:-$PWD}/worktrees/e4-paired-pilot")
observed = Path(gate.__file__).resolve(strict=True)
if expected not in observed.parents or gate.CODE_ROOT != expected:
    raise SystemExit(f"wrong camera gate import origin: {observed}")
if gate.EXPECTED_EVIDENCE_ROOT != Path("${SIMANY_ROOT:-$PWD}"):
    raise SystemExit("camera gate evidence root differs")
PY

exec "$PYTHON" -m robo.eval.e4_camera_scorer_gate \
  --gate-id "$E4_CAMERA_GATE_ID" \
  --cpu-freeze-id "$CPU_FREEZE_ID" \
  --cpu-producer-commit "$CPU_PRODUCER_COMMIT" \
  --expected-cpu-gate-sha256 "$CPU_GATE_SHA256" \
  --expected-code-commit "$E4_CAMERA_CODE_COMMIT" \
  --menagerie-root "$MENAGERIE_ROOT" \
  --expected-menagerie-commit "$MENAGERIE_SOURCE_COMMIT"
