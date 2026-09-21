# Fresh Gaussian raw-room cohort prerequisite

The adapter in `e2_raw_room.py` uses all 50 declared E1/E2 scenes and each
scene's completed 9ef4ab1 TRAIN-only Gaussian. The smoke scenes are fixed to
38d58a7a31 and 5748ce6f01. Eight official TEST views per scene are selected by
`fidelity_room_export.select_official_frames` before viewing their results.
Full TRAIN source/initialization receipts, camera metadata, Gaussian bytes and
RGB-content disjointness are validated. The existing renderer and metric
formulas are unchanged. Missing units stay in aggregate coverage.json and
prevent a complete-table claim.

Configs reserve smoke20260905-b3b85c6-v1 and full20260905-b3b85c6-v2 together.
They must execute from the same clean source, with separate exact-source E0
contracts. Full GPU execution requires published-main source and a validated
16-view two-scene smoke aggregate; changing the protocol invalidates that gate.
There is no threshold on metric quality and no view/scene selection by result.

```bash
# Use the matching *_freeze.yaml with robo.eval.freeze first.
python -m run.icra2027.e2_raw_room --config configs/experiments/icra2027/e2_raw_cohort_smoke.yaml --freeze-root /group/worldcept/code/SimAny/outputs/icra2027/20260905-b3b85c6-v1 --phase plan --scene-id 38d58a7a31
# Repeat CPU plan for the other fixed smoke scene.
# Existing e2_raw_room.sbatch accepts E2_RAW_CODE, E2_RAW_FREEZE,
# E2_RAW_CONFIG, E2_RAW_SCENE and E2_RAW_PHASE=run or aggregate.
# Freeze full plans for all50 only with e2_raw_cohort_full.yaml.
```

Use one allocated GPU for rendering/metrics. Smoke units compute their exact
canonical output format, then the aggregate computes all16 per-view records.
Full units render only; one aggregate evaluates all400 views with one LPIPS
model load. Full aggregation never averages scene-level tables. The seven
other room/object method rows remain null. `paper_ready=false` throughout:
this completes only the raw Gaussian row prerequisite, not Table II or E2.

Rendered/reference PNGs and canonical metric products remain in this source
checkout's outputs/icra2027/<freeze>/raw_room to satisfy the evaluator's
repository-local contract. Shared-root receipts bind those paths and hashes.
No phase overwrites existing claims, outputs or failed staging directories.

Smoke and schema tests (including 50-scene/400-view canonical synthetic
aggregation, source drift, byte leakage, negative gates and worker routing):

```bash
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl TMPDIR="$PWD/.t" /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q tests/test_e2_raw_cohort.py tests/test_e2_raw_room.py tests/test_e2_fresh_readiness.py tests/test_e3_discovery_cohort.py --basetemp=.t/c
```

## Exact camera-bank recovery after cross-runtime replay failure

The 24 unexecuted units in `20260905-b3b85c6-v2` failed because replay
recomputed quaternion rotations under another NumPy/BLAS runtime. A new
config may bind `camera_bank` to a byte identity returned by:

```bash
python -m run.icra2027.e2_raw_room \
  --config /group/worldcept/code/SimAny-wt/e2-raw-cohort/configs/experiments/icra2027/e2_raw_cohort_full.yaml \
  --freeze-root /group/worldcept/code/SimAny/outputs/icra2027/20260905-b3b85c6-v2 \
  --phase camera-bank --camera-bank-out <new-bank.json>
```

This CPU-only command authenticates all 50 original plan receipts against the
original config and E0 contract, then binds their exact serialized camera
plans. It refuses overwrite. Put the returned path/bytes/SHA256 identity in
both new smoke/full configs before their new exact-source E0 freezes. Neither
an old freeze nor a config may be edited in place.

At execution, the config-bound bank authenticates the original source,
contract, config, receipt, and plan bytes. The current Gaussian, training
inputs, calibration, selected frame order, and RGB identities must still
match. The renderer receives the original serialized matrices verbatim.
COLMAP headers still establish registered frame membership; their recomputed
rotation values do not replace the bank. No numerical tolerance is relaxed.
New plans record both `camera_bank` and `original_camera_plan` identities.

Recovery sequence: commit this repair; create new smoke/full configs and
freezes; run fresh E0; run the fixed two-scene/16-view smoke and aggregate;
then submit ordinary independent GPU jobs only for missing units once reuse
is explicitly accounted for. Do not resume with the old source/config freeze.
The 26 completed bundles remain untouched. This camera-bank adapter imports
**camera inputs only**: it does not relabel those bundles or make the current
single-source aggregate accept them. Cross-source artifact reuse must retain
original producer/config/freeze identities and authenticate identical render
implementation, Gaussian, camera, runtime, treatment and image hashes before
new-freeze aggregation; that reuse adapter is a separate pending step.

CPU validation at this repair: 91 focused tests passed, including camera
byte drift, source/config/receipt binding, malformed transforms, image and
calibration drift, and no-overwrite negatives. A real 50-scene/400-view bank
replayed exactly in both the NumPy 1.26 driver and NumPy 2.2 renderer
interpreters with zero model or metric calls. Evidence is under
`outputs/icra2027/audits/e2-camera-bank-20260905/` in the repair checkout.

### Declared reuse into a new evaluation freeze

`--phase reuse-bank --camera-bank <bank.json> --reuse-bank-out <new.json>`
now seals the full ordered recovery population. The existing full config and
freeze-root arguments identify the original producer. Completed units must
have valid execution/render receipts and all 16 image hashes; unexecuted
units must have no claim, partial bundle or render receipt. No failed unit is
removed. The bank pins original producer/config identities and exact relevant
source bytes (`common.py`, canonical metrics, shared input validation and the
unchanged `export_raw` function). Those source fingerprints and the complete
render/runtime/treatment config must match before reuse is allowed.

The prepared configs are `e2_raw_recovery_{smoke,full}.yaml` and corresponding
`_freeze.yaml` files. Canonical allocator reservations are:

- smoke `20260905-3ff95f4-v1`: two fresh scene renders plus16-view metrics;
- full `20260905-3ff95f4-v2`: 26 byte-verified imports plus24 fresh renders,
  then the existing400-view canonical aggregate.

Both configs use the same camera/reuse banks and must run at the same clean
published source. After each full scene's normal CPU `--phase plan`, execute
`--phase reuse --scene-id <id>` for a declared `VERIFIED_REUSE` unit, or the
ordinary GPU `--phase run` for a declared `RENDER` unit after smoke passes.
`run` refuses to regenerate a scene declared for reuse. No arrays are needed.
`recovery_plan.json` under the CPU audit directory lists both ordered rosters.

Reuse copies the exact16 PNG bytes into the new repository-local output and
keeps `original_manifest.json`. Its canonical bundle retains the original
plan and producer commit. `reuse_receipt.json` names the importer separately
and states zero render/metric calls; no render receipt is fabricated. During
aggregation, `freeze_id` means the new evaluation freeze, while each imported
record's `render_producer` retains original commit, freeze, config, plan and
execution/render receipt identities. The canonical evaluator receives all400
per-view records and still computes every number itself. Corrupted source or
copied pixels, mismatched producer/config/runtime/camera/recipe and missing
units fail closed. No older metric summary is imported.

The earlier camera-only limitation is superseded by this explicit adapter.
The26 original bundles themselves remain unchanged. Full scientific E2/TableII
readiness remains false because seven method rows still lack evidence.
