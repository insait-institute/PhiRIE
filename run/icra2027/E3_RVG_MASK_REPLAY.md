# Exact frozen CPU view replay for RVG

This is an opt-in compatibility repair requiring a **new source/config/E0/freeze**.
The original `3ff95f4` producer, old successful proposals, CPU-selected masks,
view receipts, failure logs and failed pool rows remain unchanged. No tolerance,
view ranking, visibility threshold, generator, checkpoint, seed or crop changes.
No recovery generation has been submitted by this patch.

Read-only diagnostics found the original CPU plans ran on `sof1-h200-5` and the
failed replay workers on `hala`:

| Scene | CPU plan | GPU process | First mismatch | Differing mask pixels |
|---|---|---|---|---|
| 3f15a9266d | 830094, completed | 830518, failed before generation | object19, view3, DSC07323.JPG | 2 / 459828 |
| 40aec5fffa | 830095, completed | 830532, failed before generation | object6, view0, DSC09644.JPG | 1 / 1949976 |

In both cases the original complete source/receipt hashes authenticated, frame
selection and crop bounds matched, and two further Hala replays of the first
failing object were byte-identical. This supports host-dependent numerical
raycasting replay incompatibility; exact low-level CPU instruction attribution
was not measured. Diagnostics stopped at the first differing object; the table
is not a count of all potentially differing masks. No true evaluation GT or
model inference was accessed. Source diagnostic reports and script are retained
under `/group/worldcept/code/SimAny-wt/e3-rvg-mask-replay/outputs/rvg-mask-diagnosis`.

## New run configuration

Copy the original scene's RVG config, preserving source input identities,
models, seed and runtime. Allocate a new freeze and update only output/E0 scope
plus this explicit mode:

```yaml
frozen_view_replay:
  mode: exact_frozen_cpu_views_v1
  producer_commit: 3ff95f4ec7dee0a46d8259c8fabbd16b1704bfcf
  source_config: {path: ABSOLUTE_ORIGINAL_CONFIG, bytes: ORIGINAL_SIZE, sha256: ORIGINAL_SHA}
  source_contract: {path: ABSOLUTE_ORIGINAL_E0, bytes: ORIGINAL_SIZE, sha256: ORIGINAL_SHA}
  view_manifest: {path: ABSOLUTE_ORIGINAL_VIEW_MANIFEST, bytes: ORIGINAL_SIZE, sha256: ORIGINAL_SHA}
  views_receipt: {path: ABSOLUTE_ORIGINAL_VIEWS_RECEIPT, bytes: ORIGINAL_SIZE, sha256: ORIGINAL_SHA}
```

Hashes come from original authenticated bytes. The new config itself must be
committed, published, and an E0 resource before CPU plan/GPU execution. The
source receipt must originate from CPU collection, not another replay chain.
The reader authenticates source config/E0, TRAIN-only discovery, original
receipt, every mask and RGB identity; checks the new model/input recipe is
unchanged; materializes the same masks; and checks exact RGBA pixel hashes.
Its new view manifest changes only local mask paths. Generation continues
through `run_rvg(..., precollected_views=...)` and validates actual saved PNGs.
Any mismatch fails closed. Existing configs retain strict recomputation.

## Terminal failure audits

`e3_rvg_terminal_audit.py` handles failed pools with **zero per-object producer
records and zero generated artifacts**. Every planned job remains unavailable;
`generation_performed` in the old pool is only a process-launch flag. The adapter
uses the original `execution_claim.json`, `runtime.json`, `view_manifest.json`,
and `views_receipt.json`. It never invents `execution_status.json`. Original
source/config/E0, pool/JSONL/job denominator and every view-mask hash are checked
before the terminal audit can be consumed by canonical inventory. Mixed or
partially generated pools are rejected and require the full artifact audit.

```bash
SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny python -m run.icra2027.e3_rvg_terminal_audit \
  --producer-config /group/worldcept/code/SimAny-wt/e3-rvg-cohort-launch/configs/experiments/icra2027/rvg_cohort/SCENE.yaml \
  --producer-freeze /group/worldcept/code/SimAny/outputs/icra2027/20260905-d1e8e21-v1 \
  --out /group/worldcept/code/SimAny/outputs/icra2027/20260905-d1e8e21-v1/terminal_audit/UNUSED_BATCH/SCENE.json
```

Use each exact sidecar path/hash with the existing canonical terminal-pool
adapter. Terminal audits are diagnostic provenance, not successful proposals or
paper metrics. Never overwrite an audit or a failed pool.

## Exact recovery cohort and gates (not submitted)

The complete repair population is **118 planned jobs**, all already prepared and
all having at least two frozen CPU views: 10 in `40aec5fffa`, 108 in `3f15a9266d`.
Both old pools have zero per-object producer records. Do not recover any other
scene or re-run any successful object. Config-bound anchor recipes are committed
in `configs/experiments/icra2027/rvg_mask_replay_sources/<scene>.yaml`.

Before creating configs, reserve a new canonical ID with
`robo.eval.freeze.reserve_freeze_id` against current main. After review, create
both configs together with this recipe from the clean source checkout:

```python
import os
from pathlib import Path
import yaml
new_freeze = os.environ['E3_RECOVERY_FREEZE_ID']  # already reserved, never reused
root = Path.cwd()
old = Path('/group/worldcept/code/SimAny-wt/e3-rvg-cohort-launch')
output = root / 'configs/experiments/icra2027/rvg_mask_recovery'
output.mkdir(exist_ok=False)
resources = []
for scene in ('40aec5fffa', '3f15a9266d'):
    config = yaml.safe_load((old / f'configs/experiments/icra2027/rvg_cohort/{scene}.yaml').read_text())
    recipe = yaml.safe_load((root / f'configs/experiments/icra2027/rvg_mask_replay_sources/{scene}.yaml').read_text())
    config['freeze_id'] = new_freeze
    config.update(recipe)
    path = output / f'{scene}.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    resources.append(dict(id=config['contract_resource_id'], path=str(path.relative_to(root)), kind='experiment_config', required=True))
freeze = yaml.safe_load((root / 'configs/experiments/icra2027/rvg_cohort_freeze.yaml').read_text())
freeze.update(freeze_id=new_freeze, input_roots=resources)
(output / 'freeze.yaml').write_text(yaml.safe_dump(freeze, sort_keys=False))
```

Commit and publish those configs before any plan/run. On the exact clean source:

```bash
export SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python
"$SIMANY_PY" -m pytest -q tests/test_e3_fresh_rvg.py tests/test_e3_terminal_pools.py \
  --basetemp outputs/rvg-mask-smoke/pytest
bash run/icra2027/preflight.sh --smoke
"$SIMANY_PY" -m robo.eval.freeze \
  --config configs/experiments/icra2027/rvg_mask_recovery/freeze.yaml \
  --out "/group/worldcept/code/SimAny/outputs/icra2027/$E3_RECOVERY_FREEZE_ID/contract"
```

Run CPU `--phase plan` for both declared scene configs through the existing
`e3_fresh_rvg.sbatch` launcher with `E3_RVG_PHASE=plan`. Require new view receipts
and exact equality of every RGB/crop/mask content hash to the corresponding
original receipt. The read-only CPU diagnostics above are **not** a GPU pilot.

The fixed GPU pilot is the complete 10-job scene `40aec5fffa`, selected now for
its smaller population before any recovery outcome. Submit it as one independent
job through the same launcher, `E3_RVG_PHASE=run`, new freeze/config/source,
on an allowed available GPU. Require exit 0, all 10 rows retained, source/config/
E0 and actual consumed PNG hashes verified, typed failures retained, and a finite
nonempty audit of every artifact marked available. Do not promote based merely
on process completion. Check no silent raw-RGB or single-image fallback.

Only after that gate passes may the independent 108-job `3f15a9266d` recovery
run proceed, with no parameter/input changes. No arrays. Preserve both original
failed pools and point later canonical inventory explicitly to the new pools,
with original provenance retained. These are replacement proposals for missing
units in the declared pool, not a new agentic treatment or genuine retry count.
