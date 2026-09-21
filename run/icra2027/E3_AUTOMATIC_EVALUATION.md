# Evaluation-only automatic-job matching protocol

Status: implementation smoke only; no real GT read, pilot, full evaluation, or
claim promotion is authorized by this document. Matching/evaluation require a
new clean published source commit, new E0 contract, and a new immutable freeze.
Old construction freezes are read-only. `paper_ready` remains false.

## Frozen protocol

The pilot is **38d58a7a31, all 15 planned jobs**, selected before matching.
The full population is the exact ordered 50-scene roster from the hash-bound
`configs/experiments/icra2027/construction_regimes.yaml`, **1871 jobs**, including
70 preparation failures. The full manifest is exported only after all A0–A4
controller ledgers are sealed; the pilot requires all five rows for all 15 jobs.
Do not select scenes or adjust thresholds after inspecting evaluation results.

Each automatic identity is matched once using its sealed TRAIN-only discovery
geometry (`derived_mesh.ply`, `auto_instances.npz`), independently of generated
proposals and A0–A4 choices. Reuse `agents.eval.eval_vs_gt.best_match`, retaining
its structural exclusion, bounding-box pruning, mutual nearest-neighbor minimum
coverage, 2 cm strict-distance tolerance, 4000-point stride convention, and
MATCH_THR=0.25. Multiple discoveries matching one GT instance are retained and
reported in `duplicate_gt_matches`; no new assignment algorithm is introduced.

GT is exactly `/data/ScanNetpp/data/<scene>/scans/mesh_aligned_0.05.ply` with
`segments.json` and `segments_anno.json`. Pin all three files' SHA256/size/path.
Use the existing GT instance sampler, 20000 samples, fixed seed 42 (vertices
when the legacy sampler's <4-face rule applies). Coordinates are ScanNet++ world,
metres. No GT registration or alignment fitting is performed. Construction
`gt_points.ply` and TRAIN-derived surfaces are never evaluation references.
Existing frozen registration transforms are applied only to predicted assets.
PSNR/SSIM/LPIPS remain a separate held-out image evaluation.

## Config schema and evidence closure

A matching YAML contains `freeze_id`, `mode: pilot|full`, `paper_ready: false`,
`dataset_root: /data/ScanNetpp`, `planned_scenes`, `planned_jobs`, and `protocol`
exactly equal to `agents.eval.automatic_matching_manifest.PROTOCOL`.
`population_roster` is `{path: <absolute current roster>, size_bytes, sha256}`.
`scenes` is ordered and every entry contains:

- `scene_id` (quote numeric-looking IDs), `planned_jobs`, `construction_root`
  (absolute sealed E3 `agentic` root), `control_seal` (absolute identity).
- `discovery`: the existing per-scene generation config's `source_pilot`,
  `source_summary_sha256`, `source_output_hashes_sha256`,
  `source_gaussian_training_provenance: FRESH_OFFICIAL_TRAIN_ONLY`,
  `source_discovery_hashes`, `source_discovery_config`,
  `source_discovery_commit`, and `source_discovery_contract`. These are existing
  frozen source identities, never newly minted hashes of edited artifacts.
- `gt_inputs`: keys exactly the three original scan filenames, each with an
  absolute `path`, `size_bytes`, and `sha256`.

Commit the concrete config before E0; include the matching YAML, population
roster, copied construction jobs YAML and unchanged policy YAML as E0 content
resources. The matching exporter verifies every construction seal and complete
job/policy denominator **before hashing or reading GT**. The exported manifest
contains GT identities, construction seals, source discovery proofs, assignment
scores, explicit null unmatched references, duplicate diagnostics, GT samples,
source/config/E0 identities, and a complete no-overwrite content seal. The
controller cannot open paths under `evaluation_matching`.

## Reproducible commands

Run from the new frozen checkout. `SIMANY_EVIDENCE_ROOT` may point to the shared
primary repository as supported by the existing evidence-root contract. Define
`MATCHING_CONFIG`, `EVAL_CONTRACT`, `EVAL_FREEZE`, `CONSTRUCTION_FREEZE`, `JOBS`,
`POLICIES`, and `SCENE` to exact committed/absolute identities, not globs.

```bash
python -m agents.eval.eval_vs_gt --matching-config "$MATCHING_CONFIG" \
  --contract-manifest "$EVAL_CONTRACT" \
  --out "$SIMANY_EVIDENCE_ROOT/outputs/icra2027/$EVAL_FREEZE/evaluation_matching"
```

Record the returned manifest SHA256 as `MATCHING_SHA256` in the launch receipt.
Evaluate each scene with the existing canonical evaluator:

```bash
python -m robo.eval.agentic_ablation --evaluate --jobs "$JOBS" \
  --policies "$POLICIES" --contract-manifest "$EVAL_CONTRACT" \
  --freeze-id "$CONSTRUCTION_FREEZE" --scene-id "$SCENE" \
  --out "$SIMANY_EVIDENCE_ROOT/outputs/icra2027/$CONSTRUCTION_FREEZE/agentic" \
  --evaluation-manifest "$SIMANY_EVIDENCE_ROOT/outputs/icra2027/$EVAL_FREEZE/evaluation_matching/evaluation_references.json" \
  --evaluation-manifest-sha256 "$MATCHING_SHA256"
```

Evaluation writes only `$EVAL_FREEZE/agentic/evaluation/<scene>`. After all
required scenes pass seal validation, aggregate into that same new freeze:

```bash
python -m robo.eval.agentic_ablation --aggregate --jobs "$JOBS" \
  --policies "$POLICIES" --contract-manifest "$EVAL_CONTRACT" \
  --freeze-id "$CONSTRUCTION_FREEZE" \
  --out "$SIMANY_EVIDENCE_ROOT/outputs/icra2027/$CONSTRUCTION_FREEZE/agentic" \
  --evaluation-root "$SIMANY_EVIDENCE_ROOT/outputs/icra2027/$EVAL_FREEZE/agentic"
```

If construction scenes have separate roots, export can bind each separately;
run per-scene evaluation against that scene's exact root. The existing canonical
aggregate requires one complete inventory root; consolidation must authenticate
and preserve all original controller seals before a full aggregate is possible.
Do not silently concatenate incompatible freezes.

## Denominators and pilot gate

`planned_jobs` and coverage include all jobs; unmatched GT produces null geometry.
`geometry_evaluated_jobs` counts accepted, matched, actually evaluated assets.
`physical_tested_jobs` and `physical_stable_jobs` use accepted selected assets'
construction physics evidence regardless of matching. Report both denominators
in JSON/CSV and the accepted coverage first. The residual sweep likewise counts
accepted unmatched jobs in coverage and physics, with conditional GT quality.

Pilot gate: complete 15×5 sealed controller product; no GT access before seals;
all 15 matching rows preserved; matcher settings/source/config/E0 identical to
predeclared protocol; correct frame and independent surface provenance; all
matched surfaces finite; all proposal evaluation failures explicit (no dropped
jobs); duplicate/unmatched diagnostics inspected without tuning; full canonical
JSON/CSV shape produced. Sparse/no matches are a scientific limitation, not a
reason to tune the test matcher. Full starts only after protocol review and
pilot validation, via individual jobs (no Slurm arrays).

Smoke:
```bash
python -m pytest -q tests/test_automatic_evaluation_matching.py \
  tests/test_agentic_ablation.py --basetemp outputs/evaluation-matching-smoke/pytest
```
