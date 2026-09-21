# Full public factorization: predeclared extension

The frozen protocol in `configs/experiments/icra2027/e2_full_factorized/protocol.yaml`
binds the complete canonical TRAIN construction roster (50 scenes, 1,871 planned
jobs), the original serialized 400-camera TEST bank, and E4's immutable public
factory source. A4's selected assets are reused without generation or alignment
changes. The pilot is `09c1414f1b`, the first lexicographic scene in that original
roster, selected before new background outcomes. All 112 planned object slots
remain in this pilot. This is a new full-cohort input protocol; it does not
replace the earlier compact-pair negative diagnostic.

This first source admission executes only the existing CPU removal-preparation
producer. Its context and full protocol are E0 inputs. The wrapper rejects a
non-pilot scene. It adds no rendering, segmentation, inpainting, or metric math.

```bash
SIMANY_AUTO=1 SIMANY_NO_GT=1 SIMANY_MESH_SRC=derived \
SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny PYTHONPATH=. \
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python -m run.icra2027.e2_full_factorized_protocol \
  --protocol configs/experiments/icra2027/e2_full_factorized/protocol.yaml \
  --config configs/experiments/icra2027/e2_full_factorized/preparation_pilot.yaml \
  --contract /group/worldcept/code/SimAny/outputs/icra2027/20260906-6dcb0e7-v3/contract/freeze_manifest.json \
  --phase prepare
```

The first output is
`20260906-6dcb0e7-v3/fidelity/removal/09c1414f1b/inpaint/`.
It is not a clean background or a fidelity result. Existing preparation preserves
NO_PLANE, NO_VIEW, rejected/abstained slots, and typed failures. Its source factory
is read-only. Subsequent SAM3, LaMa CPU, and fixed 1,500-step Gaussian-fill stages
each require a new context/source/E0 after the previous stage seal. Existing
immutable model/runtime assets are reused. No threshold is selected from TEST
quality. Each stage's real pilot and integrity gate precede full execution.

The 9 E4 scenes with no task queries have no factory; they remain in E2 and need
separate public materialization in a new E2 freeze using the same sealed E3
source. No E4 output is mutated. The other 41 A4 factories are reused as they seal.
No task eligibility criterion selects the E2 population.

Independent rendering will extend the existing `e2_public_factorized` consumer
only after construction seals. All 400 planned composite views remain accounted
for. Missing/failed backgrounds retain eight missing views per scene, with no
raw-background substitution. Raw images reuse the original clean source and
exact camera/renderer/runtime validation. Canonical `fidelity_metrics` evaluates
all common available raw/composite views; original planned/available counts are
reported separately. Quality remains conditional on successful background
availability. Zero-accepted unchanged-background cases retain zero object
coverage and cannot establish successful factorization. No automatic scientific
claim is enabled by a valid artifact seal.

Tests: `pytest -q tests/test_e2_full_factorized_protocol.py
 tests/test_public_inpaint_prepare.py tests/test_e3_factory_materializer.py`.
