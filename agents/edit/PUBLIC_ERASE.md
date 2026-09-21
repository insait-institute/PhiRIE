# Strict automatic TRAIN-only erasure

The opt-in path in `agents.edit.inpaint_qwen` declares **LaMa CPU** as its only
backend. It never tries Qwen, changes backend after failure, or labels an
unchanged input as an enhanced image. The legacy launcher remains available
for historical diagnostics and is not the strict experiment command.

The predecessor is the immutable output of `inpaint_masks` and its public
validator. The new erasure freeze must bind a YAML context as an E0 input:

```yaml
schema_version: 1
scope: automatic_train_only_erasure
freeze_id: <new canonical freeze ID>
scene_id: <original scene ID>
mask_seal: {path: <prior mask seal.json>, bytes: <actual>, sha256: <actual>}
backend: lama_cpu
checkpoint: {path: <pinned big-lama.pt>, bytes: <actual>, sha256: <actual>}
runtime: <public_runtime() output for the exact execution interpreter>
algorithm:
  crop_pad: 0.35
  crop_max: 1024
  blur_radius: 2
  outside_mask: byte_exact
  backend: lama_cpu
  seed: 0
```

Run from the clean, committed worktree with `SIMANY_EVIDENCE_ROOT` pointing at
the canonical artifact checkout and user-site imports disabled:

```bash
PYTHONNOUSERSITE=1 python -m agents.edit.inpaint_qwen --runtime
PYTHONNOUSERSITE=1 python -m agents.edit.inpaint_qwen \
  --public-context <context.yaml> --contract-manifest <new E0 manifest>
```

The runtime probe emits one `E2_ERASURE_RUNTIME_JSON=` line. It records the
interpreter, installed versions, RECORD files, Python sources and native
library bytes for Torch, NumPy, SimpleLaMa and Pillow using the existing
package-closure helper. This is a targeted runtime closure, not an assertion
that every operating-system dependency has been snapshotted.

The original crop size, padding, mask resize and blurred compositing recipe
are preserved. The strict path explicitly restores all pixels outside the
binary removal mask after compositing: legacy blurred alpha also changed its
exterior boundary band. This correction is declared in the treatment config
before real evaluation. No blending or crop parameter is tuned on TEST views.

Outputs are atomic and exclusive under
`<new freeze>/fidelity/erasure/<scene>/`: `erasure.json`, `seal.json`, and
`views/obj_<id>/inpainted_<view>.png`. They retain all planned object states and
every predecessor view. Empty projected masks and objects without a support
plane remain explicit non-invocations. Per-view model exceptions are recorded
as `ENHANCER_FAILED`, keep their denominator, and produce no substitute image.
A failed result makes the CLI exit nonzero. Unexpected loading or publication
errors retain a separate failure receipt and any partial directory; an
exclusive attempt claim prevents silently retrying the same freeze.

The old factory, preparation, mask and RGB artifacts are read-only. The
predecessor validator authenticates its exact TRAIN boundary and source
provenance; the new process forbids TEST-image and network reads after that
validation. This stage only creates 2D erasure targets. It does not create a
clean Gaussian background or establish an appearance-quality improvement.

Focused smoke (synthetic inputs and negative provenance/failure tests):

```bash
python -m pytest -q tests/test_public_inpaint_erase.py
```

The real LaMa TorchScript runtime was separately exercised on a synthetic
64-by-64 image with a binary mask. Its receipt is an engineering smoke only;
no real scene or independent evaluation view entered that check.

`validate_public_erasure(directory)` is the read-only handoff for the existing
fill producer. It authenticates the complete mask/context/E0 ancestry, exact
file roster and count semantics, rejects failed erasures even if resealed,
and checks byte-exact exterior pixels against original TRAIN RGB. It returns
`directory`, `seal_identity`, `context`, `context_identity`,
`contract_identity`, `result`, and the original authenticated `mask_bundle`.
It never reloads a model or reopens a frozen construction decision.
