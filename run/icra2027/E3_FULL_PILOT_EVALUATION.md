# Independent evaluation gate for the predeclared full-cohort pilot

The full construction source declares its pilot using only complete TRAIN
object counts before controller outcomes. After that scene's controller seals,
this wrapper prepares the existing evaluation-only matching configuration.
It authenticates the full construction E0/50-scene inventory and pilot roster
before reading or hashing any GT file. Original construction jobs/policies are
copied byte-for-byte. The matcher, surface sampler and geometry metrics are
unchanged; unmatched jobs keep null geometry.

Reserve a new evaluation freeze, then run from this clean evaluation worktree:

```bash
SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny PYTHONPATH=. \
/group/worldcept/code/SimAny/.venv/bin/python -m run.icra2027.e3_pilot_evaluation prepare \
  --execution-config /absolute/frozen/construction/execution.yaml \
  --freeze-id NEW_EVALUATION_ID \
  --destination configs/experiments/icra2027/e3_full_pilot_evaluation
```

Commit the generated configurations, merge/publish, run exact-source E0 with
`SIMANY_EVIDENCE_ROOT` unset for synthetic tests, and create the experiment E0
from the generated `freeze.yaml` with the production evidence root restored.
On an ordinary CPU job on Hala/gcp*/sof1*, execute:

```bash
SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny PYTHONPATH=. \
/group/worldcept/code/SimAny/.venv/bin/python -m run.icra2027.e3_pilot_evaluation run \
  --config configs/experiments/icra2027/e3_full_pilot_evaluation/matching.yaml \
  --contract /group/worldcept/code/SimAny/outputs/icra2027/NEW_EVALUATION_ID/contract/freeze_manifest.json
```

The existing producers write the matching manifest and the canonical per-scene
`agentic/evaluation/<scene>/eval_shard.json` plus seal into the new evaluation
freeze. The wrapper reports integrity PASS only for the complete pilot job and
five-policy product. It never requires acceptance or positive fidelity. Resume
verifies completed source-bound products and executes only missing phases.
It does not aggregate a partially evaluated 50-scene cohort or write new metrics.
All 50 evaluated scenes are required for the final canonical JSON/CSV aggregate.

## Complete cohort after the pilot integrity gate

The same preparer accepts `--mode full`. This requires all 50 controller seals,
1871 planned jobs, the original roster, the authenticated construction execution
and inventory, and unchanged TRAIN discovery provenance before any GT read.
It copies the same jobs/policies bytes and publishes an independent evaluation
configuration. Use a new reserved freeze and a separate clean source checkout;
never modify the running construction or the completed pilot evaluation source.

```bash
python -m run.icra2027.e3_pilot_evaluation prepare \
  --mode full --execution-config /absolute/frozen/construction/execution.yaml \
  --freeze-id NEW_FULL_EVALUATION_ID \
  --destination configs/experiments/icra2027/e3_full_evaluation
```

Commit the configurations, pass exact-source E0, and publish the evaluation
contract as above. Submit one ordinary CPU matching job using the existing
producer:

```bash
python -m agents.eval.eval_vs_gt \
  --matching-config configs/experiments/icra2027/e3_full_evaluation/matching.yaml \
  --contract-manifest /absolute/full-evaluation/contract/freeze_manifest.json \
  --out /absolute/full-evaluation/evaluation_matching
```

After its sealed reference manifest exists, submit ordinary per-scene CPU jobs
(no arrays), passing that manifest's exact SHA256. Each job invokes:

```bash
python -m robo.eval.agentic_ablation --evaluate \
  --jobs configs/experiments/icra2027/e3_full_evaluation/construction_jobs.yaml \
  --policies configs/experiments/icra2027/e3_full_evaluation/construction_policies.yaml \
  --contract-manifest /absolute/full-evaluation/contract/freeze_manifest.json \
  --freeze-id CONSTRUCTION_FREEZE_ID --scene-id SCENE_ID \
  --out /absolute/construction/agentic \
  --evaluation-manifest /absolute/full-evaluation/evaluation_matching/evaluation_references.json \
  --evaluation-manifest-sha256 EXACT_REFERENCE_SHA256
```

Reuse only a completed shard authenticated to these exact source/config/reference
identities. The old pilot belongs to a different freeze/reference configuration;
retain it as a gate, rather than relabeling it as a new full-cohort shard. This
CPU evaluation does not repeat generation or controller decisions.

Only after all 50 canonical evaluation shards pass, invoke the existing
`robo.eval.agentic_ablation --aggregate` with the same jobs, policies, contract,
construction freeze and construction output arguments, plus
`--evaluation-root /absolute/full-evaluation/agentic` and
`--uncertainty-config configs/experiments/icra2027/e3_full_evaluation/matching.yaml`.
The matching configuration, including the fixed uncertainty protocol, must be
listed in the evaluation E0 contract. The producer rejects
missing scenes or a row count different from 9355. The wrapper's `run` command
is intentionally pilot-only so it cannot accidentally publish a partial full
cohort. No new matching rules, surface samplers, metrics or table formats are
introduced by this preparation support.

The supplementary `agentic_paired_uncertainty.json` and `.csv` contain all 16
predeclared combinations of A1−A0, A2−A1, A3−A2, and A4−A0 with build coverage,
F1@20, CD in centimeters, and stable fraction. The existing fidelity bootstrap
resamples whole scenes 2000 times with seed 0 and reports pointwise descriptive
95% percentile intervals for object-pair-weighted differences. Coverage includes
all planned jobs; rejected, abstained, and failed jobs contribute zero acceptance.
F1 and CD use the same common accepted, matched, finite-geometry job pairs.
Stability uses common accepted construction probes and is selection-conditioned
construction evidence, not independent physical validation. Every row records
planned and eligible counts, exclusions, supported scenes, and a paired-job-ID
hash. Fewer than two supported scenes yields null intervals and NOT_ESTIMABLE.
There is no seed-level replication, multiplicity correction, new headline
metric, or automatic claim promotion. Main aggregate values remain unchanged.

After complete aggregation, run the source-bound completion audit:

```bash
python -m run.icra2027.e3_pilot_evaluation audit-full \
  --config configs/experiments/icra2027/e3_full_evaluation/matching.yaml \
  --contract /absolute/full-evaluation/contract/freeze_manifest.json
```

This checks all construction seals before GT, the complete 50/1871/9355
population, every evaluation/control/reference identity, null geometry for
unmatched jobs, and new signed-axis retry artifacts. It authenticates measured
metric shards; it does not implement or claim a second geometric estimator.
It then invokes the existing aggregate producer in the separate
`integrity_replay_agentic/` directory of the same evaluation freeze. No generator,
registration, physics probe, matching threshold or geometric metric is rerun.
All generated aggregate members, runtime accounting, selected-asset bytes and
empty selected directories, and both sealed uncertainty products must match the
original publication under the same predeclared protocol. The replay and
completion receipt refuse overwrite; a failed partial replay is retained for
diagnosis rather than silently reused.

The resulting `independent_evaluation_completion_audit.json` has scope
`complete_cohort_independent_evaluation_integrity`. Its PASS certifies integrity,
not an improvement: `headline_eligible=false`, `paper_ready=false`, and
`claim_gate=NOT_RUN` remain explicit. E9 separately evaluates scientific claims.
Runtime remains the existing attributed algorithm-phase cost, including failed
generation attempts, rather than capture-to-simulator or fleet elapsed time.

Smoke, including complete synthetic roster and tampered/partial negative cases:

```bash
python -m pytest -q tests/test_e3_full_completion.py \
  tests/test_e3_pilot_evaluation.py tests/test_automatic_evaluation_matching.py
```
