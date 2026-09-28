# RVG complete discovery cohort continuation

Owner: e3_rvg_cohort. Branch: `agent/icra-e3-rvg-cohort-launch`.
This extends the canonical RVG producer and config builder. No new generation,
metric, or rollout producer is introduced. All initial proposals are shared
across A1–A4; each scene keeps its complete discovery-job denominator.

Source discovery: `20260905-76c15d5-v1`, exact source
`86613e37d7bdfcef5ad10eb5cf289051055cdec3`, complete ordered 50-scene roster.
The predeclared pilot is the complete `38d58a7a31` scene (15 prepared/planned
jobs). It is not a successful-object subset.

## Config and contract

Reserve a fresh canonical freeze before writing configs. For each scene, run
this existing builder (substitute the reserved ID, scene ID and destination):

```bash
python -m run.icra2027.e3_fresh_rvg_config \
  --source-freeze-root ${SIMANY_ROOT}/outputs/icra2027/20260905-76c15d5-v1 \
  --source-config ${SIMANY_ROOT}/worktrees/e3-discovery-hala/configs/experiments/icra2027/e3_discovery_hala.yaml \
  --source-commit 86613e37d7bdfcef5ad10eb5cf289051055cdec3 \
  --recipe-config configs/experiments/icra2027/e3_fresh_rvg.yaml \
  --scene-id 38d58a7a31 --freeze-id "$E3_RVG_ID" \
  --output-config "configs/experiments/icra2027/rvg_cohort/38d58a7a31.yaml"
```

Each config binds its five discovery manifests, original source/config/E0,
scene, complete jobs, shared model hashes, seed42, and fresh TRAIN-only
Gaussian proof. Config resources use `e3_rvg_config_<scene_id>`; list all 50 in
the shared E0 config before freezing the source. Run E0 from the exact clean
published source before planning. Output-scene changes require a new config
and freeze. Do not use an old pilot runtime/pool as fresh generated evidence.

## Independent CPU and GPU jobs

The same launcher accepts `E3_RVG_CONFIG` and `E3_RVG_PHASE=plan|run`.
Arrays are rejected before any work. `plan` uses no GPU and performs import,
checkpoint/source, copied geometry, and exact RGB/mask view checks. `run`
revalidates these receipts and invokes the existing persistent RVG model once
per scene. All files are written under
`<freeze>/rvg_initial/<scene>/`, including `view_manifest.json`,
`views_receipt.json`, `proposal_pool.json`, and `proposal_records.jsonl`.

For example (ordinary jobs; root coordinates actual idle-node selection):

```bash
sbatch --parsable --partition=batch --qos=normal --constraint=zone-sof1 \
  --gres=none --cpus-per-task=4 --mem=64G --time=02:00:00 \
  --export=ALL,E3_RVG_CODE="$PWD",E3_RVG_FREEZE="$E3_RVG_FREEZE",E3_RVG_CONFIG="$E3_RVG_CONFIG",E3_RVG_PHASE=plan \
  run/icra2027/e3_fresh_rvg.sbatch

sbatch --parsable --partition=debug --qos=debug --nodelist=hala \
  --gres=gpu:a6000:1 --cpus-per-task=4 --mem=64G --time=02:00:00 \
  --dependency=afterok:"$RVG_PLAN_JOB" \
  --export=ALL,E3_RVG_CODE="$PWD",E3_RVG_FREEZE="$E3_RVG_FREEZE",E3_RVG_CONFIG="$E3_RVG_CONFIG",E3_RVG_PHASE=run \
  run/icra2027/e3_fresh_rvg.sbatch
```

Complete the real pilot and validate its exact-source proposal pool before
releasing the other 49 generation jobs. Reuse its completed scene in the full
population. Successful artifact production is not acceptance/fidelity.

The authenticated source's TRAIN limit now drives collection: full-cohort
inputs use all official TRAIN frames; the legacy pilot retains 48. The
existing view selector still produces at most 12 views per object, with the
same minimum two-view requirement. No TEST view is available to construction.
This is a necessary population-boundary extension and is recorded in a new
source/freeze; it is not a threshold change. CPU and worker both recompute
selection for independent receipt validation.

## Verification and limits

Smoke command:

```bash
python -m pytest -q tests/test_e3_rvg_cohort.py tests/test_e3_fresh_rvg.py \
  tests/test_e3_rvg_generation_pilot.py tests/test_e3_generation_cohort.py \
  --basetemp=.t/rvg-cohort
bash -n run/icra2027/e3_fresh_rvg.sbatch
```

Tests include changed TRAIN population, out-of-roster/path-scoped scene IDs,
wrong proposal freeze identity, copied-source drift, model cache separation,
and array rejection. A CPU-only builder audit for real scene `38d58a7a31`
passed with 15/15 jobs prepared; it does not replace a real generation pilot.
The shared per-tool model cache is populated by the first plan, then validates
bound model hashes and file metadata without 49 redundant weight reads.

Runtime estimate is provisional: historical A6000 RVG two-object producer
wall time was 130.03 seconds (freeze `20260905-4d0787c-v1`); direct scaling
would be about 16 minutes for 15 pilot inputs or 32.5 GPU-hours for 1,801
prepared inputs. This is only a planning estimate: full-TRAIN CPU collection,
object complexity, unavailable views, model load amortization and failures
change it. Use the real pilot timing before allocating the full generation
budget. Hardware-dependent runtime and per-object failures remain recorded.

Full: NOT_RUN until root's E0, real pilot, and receipt gates pass. No Slurm job
was submitted by this agent. Claim gate: NOT_RUN. Independent held-out metrics,
registration/physics, A0–A4, and paper promotion remain downstream tasks.
