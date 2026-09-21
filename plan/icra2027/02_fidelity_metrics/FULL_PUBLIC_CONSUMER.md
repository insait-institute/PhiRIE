# Full public-factorized held-out consumer

Owner: `agent/icra-e2-full-factorized-consumer`.
The existing `run.icra2027.e2_public_factorized` consumer now admits an opt-in
`full_public_factorized_official_test_evaluation` scope. Its compact pilot and
single-scene paired-diagnostic schemas remain unchanged. This source contains
no experiment config or measurement and does not authorize a GPU run without
a new clean source, frozen config, and exact E0 contract.

## Frozen inputs

Keep the existing config fields (`schema_version: 1`, `freeze_id`, `seed: 42`,
`planned_views_per_scene: 8`, `bootstrap_samples: 2000`, `bootstrap_seed: 42`,
`paper_ready: false`, `full_e3_gt_access: false`, `raw_source`, `metric_source`,
`execution`, and `fill_seals`). Add:

```yaml
scope: full_public_factorized_official_test_evaluation
scene_ids: # exact ordered 50-scene original protocol roster
planned_objects: # exact original per-scene population, total 1871
full_protocol: {path: ..., bytes: ..., sha256: ...}
preparation_inventory:
  code_root: /group/worldcept/code/SimAny-wt/e2-preparation-remainder
  code_commit: 59c823d4f117c5203753f504a2dc7efb36c93456
  config: {path: ..., bytes: ..., sha256: ...}
  contract: {path: ..., bytes: ..., sha256: ...}
  seal: {path: ..., bytes: ..., sha256: ...}
```

The completed TRAIN inventory is anchored to seal
`8442489b9a7c078344b2d55069d415a4b1c3f672c0e54571772347b17d4d99e0`
and retains all 50 scenes / 1871 planned object jobs / 399 accepted objects.
Its original E0/config and per-scene preparation sources are authenticated.
The original protocol separately authenticates the full constructor's exact
object slots and all 400 serialized TEST camera selections. The inventory
was selected from TRAIN construction evidence before mask/fill outcomes.

`fill_seals` must contain exactly the 36 structurally fillable scenes plus
four zero-accepted scenes. Those entries must be actual successful original
fill outputs with the same preparation seal/context/E0 chain. Zero-accepted
scenes require the existing producer's byte-exact `NO_REMOVAL` output and
remain explicitly marked as unchanged backgrounds with zero object coverage.

The ten structurally blocked scenes have no `fill_seals` entries. Each is
revalidated through the existing preparation consumer; its original TRAIN
boundary, factory identity, slots, accepted count, and structural reasons must
agree. All eight TEST views remain missing for the composite. Neither a raw
background nor a fabricated fill receipt is substituted.

The raw comparison reuses only source
`cbba8ff63bfec9002a740826b55259106373a7af`, config SHA-256
`c881d9ea03e77bee62022bedd099d1f21db4298dadd30bc2286cb112b9eac57f`,
freeze `20260905-3ff95f4-v2`. Its original source/config/E0/receipt validators
and PNG hashes execute unchanged, followed by exact original-camera-bank
comparison. There is no raw rerender path.

## Execution and outputs

Use the existing CLI with a new frozen config and E0:

```bash
python -m run.icra2027.e2_public_factorized --config CONFIG --contract E0 --phase context
python -m run.icra2027.e2_public_factorized --config CONFIG --contract E0 --phase render --scene SCENE
python -m run.icra2027.e2_public_factorized --config CONFIG --contract E0 --phase metrics
```

Submit each render as an ordinary individual job, never an array. TRAIN-blocked
scenes need only CPU reuse/copy/accounting; actual composite renders use the
pinned renderer runtime. All original scene bundles must exist before metrics.
The output remains `outputs/icra2027/FREEZE/fidelity/public_factorized/` with
sealed `scenes/SCENE` outputs and canonical `metrics/fidelity_manifest.json`,
`metrics/coverage.json`, and `metrics/table/fidelity_table.json`.

The full quality support is predeclared as **all common available raw/composite
view IDs**. Coverage first retains 400 planned views for each method and all
1871 planned objects, including the 80 TRAIN-blocked composite views. Raw
availability stays 400 even when conditional metric support is smaller.
Existing `_load_composite`, `raw.export_raw`, and `robo.eval.fidelity_metrics`
perform rendering and metric computation unchanged. The full manifest does
not adopt the old single-scene diagnostic's inference wording or enable any
headline fidelity-gain claim. E9 must independently consume the exact new
freeze and determine its paper gate.

## Explicit failure boundary

This version admits authenticated TRAIN blockers and complete/NO_REMOVAL fill
products. An absent fill, invalid provenance, or malformed/resealed output
fails closed. A genuine later mask/erasure/fill failure needs a separately
reviewed narrow consumer of the actual producer's typed terminal receipt,
config, source and E0, before publishing the final evaluation config. It must
retain the scene's eight missing views and cannot turn provenance errors into
scientific missing-data rows. It does not justify another generation attempt
or a raw-composite fallback. No such model-failure receipt is fabricated here.

## CPU smoke

```bash
mkdir -p outputs
.venv/bin/python -m pytest -q tests/test_e2_full_factorized_consumer.py \
  tests/test_e2_public_factorized.py tests/test_e2_common_view_analysis.py \
  tests/test_public_factorized_export.py --basetemp=outputs/consumer-tests-final
```

The fixtures cover exact full coverage and common support; original-camera
reuse; source/config/E0 drift; missing and resealed inventory members; changed
object populations; changed preparation/fill chains; relabeled TRAIN blockers;
zero-accepted semantics; and different paired reference bytes. They use no
real TEST pixels, model inference, or experiment results.
