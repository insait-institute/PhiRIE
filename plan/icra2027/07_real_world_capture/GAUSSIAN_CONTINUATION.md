# DROID Gaussian continuation from sealed TRAIN fits

This separate engineering stage retains the original ten attempted workspaces.
Eligibility is a valid sealed TRAIN robot-alignment fit plus its original public
RGB. Reference availability, held-out accuracy, and CPU overall outcome are not
eligibility inputs. The recorded rule accurately states that reference artifacts
already existed when archived, but their values had not been inspected. It is
not described as registration before artifact generation.

The constructor never reads `result.json`, `held_out_reference.json`, or
`alignment_evaluation.json`. It authenticates the original CPU config/E0 and
source, runs the original TRAIN validator on its original CPU host, and checks
fit-linked extraction/image identities. A missing fit is recorded as NOT_RUN
only after its original CPU attempt is terminal. A valid sealed fit may
continue immediately while unrelated CPU attempts are running. No task is substituted.

The existing `make_scene_dir.write_scene_dir` writes an exclusively owned new
scene. The existing `agents.recon.gsplat_train` then uses the unchanged DROID
30,000-iteration recipe, seed zero and every-tenth-view internal RGB diagnostic.
All registered RGB may have participated in SfM; that diagnostic is not an
independent image-fidelity measurement. No derived mesh, object construction,
full simulator build, or physical success is implied.

## Frozen config contract

`execution.json` has scope `droid_sealed_train_fit_gaussian_continuation`,
`paper_ready: false`, `iters: 30000`, `smoke_iters: 16`, `seed: 0`,
`holdout_every: 10`, `full_build: NOT_RUN`,
`independent_image_fidelity: false`, and a new canonical `freeze_id`.
It includes:

- `cpu_source`: original clean `code_root`, `commit`, `stage_root`, Python
  invocation, `validation_host`, and `{path,sha256,size_bytes}` config/E0 identities;
- all ten original `workspace_ids` in order and the first `pilot_workspace`;
- `cpu_submissions`: all ten original accepted submission identities;
- the exact `continuation_rule` identity and primary `evidence_root`;
- pinned `training_python`, `training_dependency_root`,
  `training_dependency_tree_sha256` (canonical hash of `tree(root)`),
  `torch_extensions_root`, and `training_runtime_identity`.

E0 must bind the complete config as resource `e7_gaussian_config`; exact clean
current source and E0 are checked for every stage. The existing mini-viewer
Python 3.10/torch2.4.1+cu124/gsplat1.5.3 runtime uses the existing scoped pydantic
overlay, with user-site imports disabled. No environment installation is needed.

## Reproducible execution

Use the final published source/CWD and exact config. First run the complete CPU
preparation population on the original numerical replay host:

```bash
python -m run.icra2027.e7_droid_gaussian --config CONFIG --stage-root FREEZE \
  --workspace ORIGINAL_ID --phase prepare
```

Every original slot gets an immutable `gaussian/ID/prepare_receipt.json`.
Successful receipts authenticate original TRAIN evidence, public RGB and the
new scene inventory. Missing fits have no invented model or metrics.

After preparation, run only the fixed first-IPRL native 16-iteration smoke on
one authorized Hala/gcp*/sof1* GPU, then its full 30,000-iteration pilot:

```bash
python -m run.icra2027.e7_droid_gaussian --config CONFIG --stage-root FREEZE \
  --workspace FIRST_IPRL_ID --phase train-smoke
python -m run.icra2027.e7_droid_gaussian --config CONFIG --stage-root FREEZE \
  --workspace FIRST_IPRL_ID --phase train
```

After genuine full-recipe pilot integrity passes, submit the other eligible
original slots with `--phase train`. The wrapper checks that pilot receipt and
artifact hashes, not its diagnostic quality. Use individual jobs, no arrays.
GPU reports retain model count, actual latency and peak CUDA allocation from
the existing trainer. Each phase has an exclusive lock and receipt; failures,
logs and partial outputs cannot be overwritten. Changed source/config or a
failed-phase rerun requires a new freeze.

## Verification

```bash
PYTHONPATH=. python -m pytest -q tests/test_e7_droid_gaussian.py \
  tests/test_droid_prospective_alignment.py tests/test_colmap_cpu_threads.py
```

Tests cover real canonical scene export, missing-fit NOT_RUN, no overwrite,
source/roster/rule tampering, forbidden evaluation inputs, same-host numerical
validation, package-environment isolation, and training-report population and
recipe integrity. These are engineering gates; full E7 construction remains
unfinished until its separate complete build pipeline exists.
