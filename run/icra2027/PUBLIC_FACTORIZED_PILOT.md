# Compact public factorized held-out consumer

The fixed compact pair is `27dd4da69e` (7 planned automatic object jobs) and
`40aec5fffa` (10). This engineering pilot preserves all 17 object jobs and all
16 original official TEST views per room method. It is scoped to the already
sealed compact construction; it neither opens full-cohort E3 evaluation data
nor feeds TEST evidence back into construction. It does not replace Table II's
full 50-room population or promote a claim.

`validate_public_fill` authenticates the archived fill/preparation/mask/erasure
and factory chain. `_load_composite(public_fill=...)` loads the sealed cleaned
background and each accepted native proposal Gaussian, applies its canonical
alignment once, and retains its automatic identity. There is no raw-room
substitution when accepted-object removal is incomplete. Zero accepted objects
are an explicitly counted `NO_REMOVAL` source copy.

`e2_raw_room.export_raw` supports an explicit preloaded composite. The existing
camera matrices, full resolution, renderer, SH handling, RGB clipping and PNG
quantization are unchanged. The original `cbba8ff` producer validates reuse in
its own unchanged checkout. Its source/config/E0/runtime/camera/image hashes
remain provenance of reused raw images; they are never attributed to the new
consumer source. Raw images are copied byte-exactly into the new freeze because
the canonical metric producer requires paths within its evidence repository.
The original 400-view aggregate cannot provide pair-specific LPIPS statistics;
only the pair's canonical metrics are recomputed.

The new config binds `scene_ids`, `planned_objects`, `fill_seals`, the original
`raw_source` (source checkout/commit, config and E0 identities, interpreter),
`metric_source` (unchanged canonical checkout/commit/module identity), and
`execution` (original render/metric runtime identities, interpreters, immutable
renderer dependency overlay and LPIPS checkpoint). Seed42, bootstrap2000/42,
full resolution and eight TEST views are fixed before metrics are inspected.
Create a fresh E0-bound config from the read-only source validation receipt,
commit it, merge to main, then run exact E0 before jobs.

Commands, each with the same new config and contract:

```bash
python -m run.icra2027.e2_public_factorized --config CONFIG --contract E0 --phase context
RENDER_PYTHON -m run.icra2027.e2_public_factorized --config CONFIG --contract E0 --phase render --scene 27dd4da69e
CPU_PYTHON -m run.icra2027.e2_public_factorized --config CONFIG --contract E0 --phase render --scene 40aec5fffa
CPU_PYTHON -m run.icra2027.e2_public_factorized --config CONFIG --contract E0 --phase metrics
```

The first scene uses one compatible GPU; the known NO_PLANE scene only seals a
typed CPU result. Both are ordinary jobs, without arrays. The final metric driver
uses one compatible GPU for the existing LPIPS evaluator. Set the pinned runtime
via `e2_raw_room.environment(execution, stage)` and canonical evidence root;
all jobs use literal `--export=ALL`, `--no-requeue`, and recorded resource receipts.
No stage overwrites an existing attempt. Failed partial render artifacts remain
sealed with `FAILED`; they do not become metric views.

Outputs are `fidelity/public_factorized/scenes/SCENE/{result,seal}.json`, reused
raw RGB, composite RGB where valid, and `metrics/{coverage.json,fidelity_manifest.json,
table/fidelity_table.json,table/fidelity_table.csv,receipt.json,seal.json}`.
Coverage is explicit before conditional PSNR/SSIM/LPIPS. The second scene's eight
missing composite views have no numerical quality value. No full-E3 GT access,
policy changes, new metric implementation, or automatic paper readiness is allowed.

Focused tests:

```bash
python -m pytest -q tests/test_e2_public_factorized.py tests/test_public_factorized_export.py tests/test_public_inpaint_fill.py tests/test_e2_raw_room.py tests/test_fidelity_room_export.py
```

## Metrics-only recovery

The first metric job833264 stopped before evaluating an image because the
runtime probe helper did not exist in the archived metric checkout. Preserve
that failed attempt and all original scene outputs. A new `render_source` config
binds the original4462 producer/config/E0 and both exact scene seals. The new
consumer calls that original producer's `scene_result` validator in its own
unchanged checkout. Every treatment, fill, camera, runtime and metric source
field must agree; only the consumer freeze changes. Nested reuse and rendering
from a metrics-only config are rejected. Original image paths/hashes remain
explicit provenance; no rendering or generation is repeated.

The runtime helper executes from the frozen consumer checkout, which contains
it. The actual metric module still executes from the separately pinned archive.
Their metric implementation bytes must match. A regression test uses an archive
that has the metric file and lacks the runtime helper entirely.

## Predeclared common-view diagnostic

The follow-up `e2_common_view_analysis/protocol.yaml` was committed before the
first metric values were inspected. It fixes all eight common TEST cameras of
27dd4da69e, both original scenes and all 17 planned objects, and the first TEST
camera for a qualitative triptych. The opt-in `paired_protocol` consumer checks
that immutable protocol against the original sealed scene results before passing
raw and composite images on exactly the same eight cameras to the unchanged
canonical fidelity evaluator. Coverage still reports 16 planned views per method
and the eight unavailable composite views of 40aec5fffa. `analysis_views` records
the conditional quality denominator separately. This is a one-scene descriptive
diagnostic; the evaluator's scene-bootstrap output does not establish population
uncertainty or a headline improvement. No images are rendered again.

Reproducible command after its own source/E0 publication:

```bash
python -m run.icra2027.e2_public_factorized --phase metrics \
  --config configs/experiments/icra2027/e2_common_view_analysis/evaluation.yaml \
  --contract /group/worldcept/code/SimAny/outputs/icra2027/20260906-0281856-v1/contract/freeze_manifest.json
```
