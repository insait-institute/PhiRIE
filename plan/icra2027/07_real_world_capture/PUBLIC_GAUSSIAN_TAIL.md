# E7 public Gaussian-to-object continuation

Owner: E7. Scope: the original ten prospective DROID capture slots, including
CLVR. This stage adds no proposal generator, simulator, policy, evaluation
reference, or physical-success claim. It preserves unavailable TRAIN fits and
failed GS training as unavailable construction inputs. Alignment reference
accuracy never selects workspaces. The first original IPRL slot is the fixed
pilot; remaining execution waits for its complete integrity receipt.

`run.icra2027.e7_droid_public_tail` invokes the existing producers:

1. CPU `--phase plan`: authenticate the original clean GS source/E0/config and
   original-source validator, retaining its actual paths and hashes.
2. GPU `--phase render`: explicit sealed Gaussian and public scene cameras,
   using `derive_mesh_from_splat`.
3. CPU `--phase fuse`: the same existing TSDF producer.
4. GPU `--phase discover`: existing SAM3 `auto_segment`.
5. CPU `--phase prepare`: existing `factory_prepare` object contracts.
6. CPU `--phase finalize`, then `--phase validate`: authenticate every phase,
   exact render/crop camera populations and the canonical instance aggregator.

Each command takes `--config <execution.json> --stage-root <new freeze>
--workspace <original ID>`. Submit ordinary jobs with no arrays. GPU phases
require exactly one allocated GPU; CPU phases reject a visible allocated GPU.
A phase owns an exclusive invocation lock; no failed phase is silently retried
or overwritten. Its successor authenticates every prior artifact. A changed
source or configuration requires a new freeze.

The fixed legacy DROID recipe remains render/discovery stride 3, render scale
0.5, alpha 0.6, TSDF voxel 5 mm/truncation 20 mm, SAM3 score 0.45, two-frame
support, 25 voxels, merge IoU 0.25, pixel stride 4, maximum extent 0.9 m and the
base manipulable-object vocabulary. Crop pixel gates use the original
metadata-only image-height scaling. Existing preparation rejection rules and
object IDs remain unchanged; the existing `summarize_instances` retains every
discovered instance before reporting prepared subsets.

Required configuration fields: `scope`, `freeze_id`, `workspace_ids`,
`pilot_workspace`, exact `protocol`, `gaussian_source` (original code root,
commit, Python, stage root, config and contract identities), `cpu_python`,
`cpu_runtime`, `discovery_runtime`, `torch_extensions_root`, `evidence_root`,
and false `paper_ready`, `generate_assets`, `full_build`. E0 resource IDs are
`e7_tail_config`, `e7_tail_runtime`, `e7_tail_cpu_runtime`, `sam3_checkpoint`.
The scoped CPU runtime hashes Python, package RECORD/source/native bytes;
existing SAM3/renderer runtime and checkpoint validation is reused. No model
installation or external API is required.

The constructor never opens CPU result/reference evaluation files or historical
unprovenanced reconstructions. Public RGB may have participated in Gaussian
initialization, so no independent appearance metric is produced. Existing AUTO
`gt_points.ply` filenames denote derived observation surfaces here; they are
not evaluation ground truth.

Validation command:

```bash
PYTHONPATH=. /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q \
  tests/test_e7_droid_public_tail.py tests/test_e7_droid_gaussian.py \
  tests/test_droid_prospective_alignment.py
```

Implementation smoke: 112 tests pass, including source/roster/receipt tamper,
forbidden reference reads, no-overwrite, CPU/GPU separation, predecessor
failure, missing-input nulls, real NPZ/JSON camera population replay and existing
object-denominator aggregation. An original-source read-only bridge validated
the genuine first-IPRL 30k Gaussian artifact; the isolated seven-package CPU
runtime probe passed. Real tail pilot/full execution remains NOT_RUN until a
new exact source/config/E0 is reviewed and sealed.
