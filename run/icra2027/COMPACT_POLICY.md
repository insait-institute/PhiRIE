# Compact canonical policy stages

Run from the same clean source commit and worktree used for automatic CPU
qualification and camera/scorer diagnosis. The preparer consumes those existing
validators and the fixed protocol; it never substitutes a task or produces a
second rollout ledger. Actual camera outputs are required before preparation.

Each changed runtime configuration gets a separately reserved stage freeze ID:
qualification/camera, scripted smoke, and real policy use distinct IDs while
retaining the same code commit. Generated configs remain immutable output
artifacts; no tracked source/config edit is needed between stages.

Prepare scripted smoke (replace the uppercase arguments with exact frozen paths,
SHA256 values and newly reserved IDs):

```bash
export SIMANY_EVIDENCE_ROOT=${SIMANY_ROOT}
source ${SIMANY_ROOT}/outputs/test-headless-setup/env.sh
PY=${SIMANY_ROOT}/.venv/bin/python
$PY -m run.icra2027.e4_compact_policy prepare --mode scripted \
  --camera-config-path CAMERA_CONFIG --camera-config-sha256 CAMERA_CONFIG_SHA256 \
  --camera-output CAMERA_OUTPUT --expected-code-commit SOURCE_COMMIT \
  --freeze-id SCRIPTED_FREEZE_ID \
  --out ${SIMANY_ROOT}/outputs/icra2027/SCRIPTED_FREEZE_ID/harness/compact_scripted_smoke \
  --openpi-root ${OPENPI_ROOT}/worktrees/e4-policy-server \
  --openpi-commit 2f51088169d2e2b480ce54faa737be9b0279eed4
```

Preparation authenticates the complete CPU/camera chain, full task suites,
registry checkpoint, control/camera sources, and exactly 20 canonical ID/seed
reset definitions for 40 paired cells. `reset_states.json` contains no poses or
measured telemetry. The existing harness consumes that bank while retaining
full `planning_tasks` paths. Both policy modes retain a 32 s frozen horizon.

`harness_config.yaml`, `reset_states.json`, `preparation_receipt.json`, its seal,
and an existing-schema `stage_freeze.yaml` are written atomically. Before any
execution, complete exact-source E0 and use the existing freeze producer:

```bash
bash run/icra2027/preflight.sh --smoke
$PY -m robo.eval.freeze --config STAGE_OUT/stage_freeze.yaml \
  --out STAGE_FREEZE_ROOT/contract --preflight-log EXACT_SOURCE_PREFLIGHT_LOG
$PY -m run.icra2027.e4_compact_policy run --mode scripted \
  --out STAGE_OUT --expected-code-commit SOURCE_COMMIT --contract STAGE_FREEZE_ROOT/contract
```

E0 must bind the exact stage config, reset bank, camera config/gate, preparation
receipt, executing Python, policy checkpoint, control source and runtime roots.
Execution refuses missing, changed, wrong-source or wrong-stage contracts.
Menagerie runtime uses the clean Git source only after its entire Panda/Robotiq
file closure matches the camera's immutable runtime copy. OpenPI must match the
camera preprocessing source. The existing scoped `/scratch/USER/icra2027/` policy
checkpoint cache convention is declared only; preparation creates no files there.

Scripted execution uses `robo.eval.harness_runner.run_matrix`, preserving every
CPU/camera rejection as a typed canonical failure before policy creation. Its
`scripted_smoke_gate.json` requires all 40 ledger rows, canonical source/paired
validation, and complete runtime/scorer traces for every eligible cell. Task
success is **not** required; all-zero results remain visible. Crashes/timeouts or
missing traces fail the runtime gate. This stage is engineering smoke only.

Prepare the real stage with a different reserved ID, `--mode real`, output name
`compact_policy`, plus `--scripted-output SCRIPTED_OUT` and
`--scripted-gate-sha256 SCRIPTED_GATE_SHA256` when any cell is eligible. The real
stage revalidates the entire scripted ledger and exactly paired geometry,
camera, reset, instruction and control declarations. Freeze this generated
config with E0, then execute `run --mode real`. The canonical harness verifies
live server identity and performs real-observation policy warmup before actions.
This compact engineering pilot does not enable a full manipulation claim.

If every cell is rejected, real preparation needs no scripted gate and
`run --mode prebuild-only` writes all 40 truthful non-invocation records through
the canonical harness without connecting to a server. It refuses any eligible
cell. No synthetic server identity, reset state, success or policy observation
is created by this preparer. Final paper/full-experiment gates remain external.
