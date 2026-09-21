# Full frozen TRAIN-only background erasure

Owner: E2 execution. Branch: `agent/icra-e2-full-erasure`.

This stage extends the fixed 50-scene / 1,871-job / 399-accepted-object room
appearance population using the existing `agents.edit.inpaint_qwen` strict
LaMa CPU producer. Its name is historical; this execution explicitly selects
`lama_cpu` and never falls back to Qwen, raw RGB, or an unmasked image.

The input protocol will be published only after the complete mask integrity
receipt authenticates all 40 applicable mask bundles, including the original
first structurally fillable pilot. Ten original TRAIN structural blockers remain
in the 400 planned TEST-view denominator and do not invoke erasure. Four
zero-accepted scenes have explicit CPU no-op records; their unchanged backgrounds
are not successful object construction. The six pilot erasures are reused with
their original source, config, E0, and seal identities.

Each remaining scene uses a separate ordinary CPU job, four CPUs and 32 GiB,
with the pinned original LaMa checkpoint and Python/package bytes. Every source
mask and input image is authenticated by the existing producer; pixels outside
the erasure mask must remain byte-exact. All planned object/view rows survive,
including typed enhancer failures. Failed outputs are preserved and cannot be
passed to the fill producer as successful erasures.

Reproduction after publication:

```bash
PYTHONNOUSERSITE=1 SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny \
PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/group/worldcept/code/SimAny/.venv/bin/python -m agents.edit.inpaint_qwen \
  --public-context configs/experiments/icra2027/e2_full_erasure/SCENE.yaml \
  --contract-manifest /group/worldcept/code/SimAny/outputs/icra2027/FREEZE/contract/freeze_manifest.json
```

Use the absolute pinned Python invocation; invoking a worktree symlink changes
runtime-origin evidence and is rejected. No official TEST image is read and no
threshold, camera, source mesh, controller decision, or metric is changed.
The final source commit, E0, command, scheduler receipt, and validation gate are
recorded in this stage's immutable freeze. No full background or fidelity claim
is enabled by erasure completion alone.
