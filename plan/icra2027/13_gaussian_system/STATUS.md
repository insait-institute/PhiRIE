# Implementation and execution status

New systems campaign, 2026-09-09. Base reviewed source: PhiRoom 4015403980ef3ade3ef986997449b3add2769f2f. Latest observed previous experiment release: results/simulation-submission-20260908 at 8e7ae22b2a0531a9666503ce80972374b76fb8a9.

## Implemented now
- Dataset capability catalogue, true-source inventory checks, family/split leakage checks, deterministic quotas and source-only HTML gallery.
- Finite component task expansion, immutable dependency receipts, model-specific Python environments, ordinary Slurm submit/queue cap/collection, independent same-model batch reuse.
- Official COLMAP/3DGS RGB-video launcher, with metric calibration explicitly not invented.
- Actual TRELLIS/TRELLIS.2/ReconViaGen-v0.2 model API wrappers and canonical proposal export.
- Telea, SDXL masked diffusion, Qwen Edit2511 and explicitly approved Gemini image-edit adapters, raw output retention and exact unmasked compositing.
- Actual Chorus encoding, matching local text encoding and persistent-ID spatial/feature grouping.
- State/motion/depth/ID validated temporal correspondence and bounded protected residuals; official Harmonizer sequence integration; trainable StateResidual16 training AND inference.
- Sparse regularized Gaussian SH-DC color-cache solver, fixed nonappearance attributes, actual responsibility input contract.
- Official SimFoundry reconstruction command adapter, PolaRiS native-bundle validator and common-native asset contract conversion.
- Ten agent task READMEs, full datasets/experiments/runbook/prompt and source documentation.

Local CPU validation: 43 tests passed (3.93s on the last recorded full test invocation), including numerical geometry/motion, learned checkpoint inference and actual sparse color solver, file/queue preparation, masks, split isolation and failure guards. Python compile/import and CLI checks are also required on the final patch. These are NOT native-model scientific results. Tests use lightweight real arrays/meshes and controlled mock boundaries, not project checkpoint inference.

## Not executed here
No dataset was downloaded here. No external image was uploaded and no API money was spent. No pretrained TRELLIS/Chorus/Qwen/Harmonizer checkpoint was run. No COLMAP reconstruction, native benchmark scene import, policy rollout or Slurm experiment was launched in this environment. Model environments/data/native assets must be resolved on the user's compute machines.

Full native PolaRiS/OmniGibson task-policy glue, metric calibration on an actual RGB video, renderer compositing-responsibility export, model-specific real smoke and expanded full-room admissibility still need their documented integration gates. The command interface does not magically implement missing native tasks. RVG-v0.5 and extra inpainters are separately admitted optional adapters, not implemented aliases. A baseline lacking input/access is unmeasured/not applicable, not a zero-score claimed victory.

Append actual source/model hashes, commands, jobs, eligible/planned/measured counts and failures to each task STATUS.md. Do not mark an experiment complete from code existence or unit tests. Preserve prior results and do not change the paper's measured claims before data exists.

## Source publication and revalidation

2026-09-09: the source package is being published as normal UTF-8 repository files on `agent/gaussian-system-publish-20260909`, based on the unchanged main revision above. The earlier ZIP was a delivery fallback. Git history records the actual publication and main promotion; the directory in this repository is the authoritative source.

Revalidated the delivered source locally: **43 tests passed in 6.30s**, with no failures or skips. `compileall`, `bash -n run/campaign/run.sh`, and `python -m robo.campaign --help` also passed. This revalidation did not download models, submit cluster work or run scientific experiments. The supplied CI workflow repeats CPU checks without model credentials or experiment dispatch.
