# E1 — Automatic Room-Scale Construction

**Priority:** P0  
**Paper output:** Table I  
**Depends on:** E0

## Research question

Under which input regimes can SimAnyRoom automatically produce accepted simulation-ready object assets at room scale?

This task freezes the existing 50-scene construction evidence. It must not change the constructor to improve numbers after evaluation begins.

## Read first

- `run/run_factory.sh`
- `run/run_auto.sh`
- `agents/eval/aggregate_results.py`
- `agents/eval/eval_vs_gt.py`
- `agents/eval/factory_report.py`
- `robo/eval/construction_metrics.py`
- `configs/experiments/construction_records.template.csv`
- `docs/CONTRIBUTIONS.md`

## Fixed regimes

Use exactly these rows and names:

1. `GT segments + scan mesh`
2. `Auto discovery + scan mesh`
3. `GT segments + splat-fused mesh`
4. `Auto discovery + splat-fused mesh`
5. `Single RGB + metric depth`

Do not add another row unless the paper table is revised first.

## Evaluation population

- Primary benchmark: all 50 ScanNet++ v2 `nvs_sem_val` scenes.
- A missing or failed scene remains visible. For the splat-fused regime, record the failed scene explicitly rather than changing the denominator silently.
- Automatic discovery quality is evaluated only through a documented matching process against held-out GT. Own-extraction scoring may be archived as a diagnostic, but it is not the headline F1.
- Build yield is computed over all input instances for that regime.
- Stability is computed over all bodies for which the declared drop test was attempted. Missing tests are not passes.

## Required outputs

Create one row per scene and regime:

```text
outputs/icra2027/<freeze_id>/construction/scene_records.csv
```

Required columns:

```text
freeze_id
regime
scene_id
scene_status
input_instances
accepted_instances
f1_20
f1_weight
stable_instances
tested_instances
runtime_minutes
build_commit
build_manifest_path
failure_reason
source_artifact_hash
record_valid
validity_reasons
geometry_reference_status
```

The final four columns are fail-closed provenance fields. Legacy imports with
missing build identity, leaked geometry, or incoherent snapshots must set
`record_valid=false`; the default aggregator rejects them. Use
`--allow-preliminary` only to render an explicitly watermarked audit table.

Then aggregate with the existing producer:

```bash
python -m robo.eval.construction_metrics \
  --input outputs/icra2027/<freeze_id>/construction/scene_records.csv \
  --out outputs/icra2027/<freeze_id>/construction/table
```

Expected products:

```text
construction_table.json
construction_table.csv
construction_table.tex
```

## Implementation steps

### 1. Inventory existing builds

Write or extend a scanner that walks the current output roots and reports, per scene/regime:

- discovery output present;
- candidate generation complete;
- registration result present;
- accepted/rejected status present;
- physics/drop result present;
- runtime source present;
- GT matching result present.

The scanner must produce a missing-artifact report before any expensive rerun.

### 2. Re-run only missing or invalid jobs

Use the existing launchers and preserve old outputs. New runs go under the current freeze root or a uniquely versioned build root. Never overwrite the evidence used by a previous result.

### 3. Normalize denominators

For every regime, explicitly record:

- total scenes planned;
- scenes successfully reconstructed;
- instances presented to the asset stage;
- accepted assets;
- instances with independent GT geometry;
- bodies with a valid drop-test result.

The table may show one compact denominator, but the JSON must contain all of them.

### 4. Verify known headline values

The aggregate should reproduce the already audited values before any new method change:

- GT segments + scan mesh: 789 instances / 50 scenes, 58.0% yield, F1@20 0.708, 77.2% stability, 11.6 min/scene.
- Auto discovery + scan mesh: 1,082 / 50, 62.0%, 0.582, 75.4%, 17.0 min/scene.
- GT segments + splat-fused mesh: 678 / 49, 56.0%, 0.693.
- Auto discovery + splat-fused mesh: 930 / 49, 59.0%, 0.553.
- Single RGB + metric depth: 389 / 40, 17.2%, 0.309.

A mismatch is not patched in the CSV. Trace it to its source and document the reason.

## Tests

- aggregation is invariant to row order;
- duplicate `(regime, scene_id)` rows fail;
- a failed scene is preserved in the scene count;
- yield uses summed counts, not an unweighted mean of per-scene yields;
- F1 weighting is explicit and tested;
- missing stability produces null, never zero or one by default;
- all five required regimes appear exactly once in the aggregate.

## Fast smoke

Use two evidence-bearing scenes and the first two regimes. All temporary and
output paths remain inside this repository:

```bash
FREEZE_ID=e1-local-smoke
OUT=outputs/icra2027/${FREEZE_ID}/construction

python -m robo.eval.construction_inventory \
  --outputs-root outputs \
  --out "$OUT" \
  --freeze-id "$FREEZE_ID" \
  --expected-scenes 50 \
  --scene-list configs/experiments/icra2027/construction_regimes.yaml \
  --contract-manifest outputs/icra2027/<e0-run>/contract/freeze_manifest.json \
  --smoke \
  --scene-id 38d58a7a31 \
  --scene-id c4c04e6d6c \
  --regime gt_segments_scan_mesh \
  --regime auto_discovery_scan_mesh

python -m robo.eval.construction_metrics \
  --input "$OUT/scene_records.csv" \
  --out "$OUT/table" \
  --smoke \
  --allow-preliminary
```

The legacy smoke is expected to be `paper_ready=false`; it validates the
inventory and denominator contract, not the publication claim. For Slurm, use
`run/slurm/icra2027_e1_construction.sbatch` with `E1_MODE=smoke`.

## Acceptance criteria

- [ ] One scene-level CSV covers every planned scene/regime, including failures.
- [ ] The five-row aggregate is generated by code.
- [ ] Existing headline values are reproduced or discrepancies are explained with source artifacts.
- [ ] No manual table edits are required.
- [ ] Every row points to a build manifest and commit.
- [ ] Table I can be regenerated from a clean checkout plus the frozen data root.

## Handoff

Create `STATUS.md` with the missing-artifact inventory, rerun list, final aggregate paths, hashes, and any denominator differences from the current paper.
