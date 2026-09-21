# Submission without physical robot experiments: launch instructions

**Active next experiment track, 2026-09-07.** Paper: `RunyiYang/SimAnyRoom`. Code: `RunyiYang/PhiRoom`. The submission combines real-scan construction with simulated robot evaluation. Physical robot trials, sim-to-real transfer, BEHAVIOR expansion, a new learned verifier, and Harmonizer are NOT blockers or promised results for this revision.

The original 48-instance L0 TEST cohort, 24-instance L1 phase, 50-room ScanNet++ study, failures and ledgers remain immutable. B3 is the measured core; B4/BM stay as contextual verification/repair analyses. Source baseline before this implementation: `d42f87351a0d7e5d985f9fba67c549cc4cb7a3b9`. The code in this directory is implemented, not a request to write another plan.

## Three independent jobs

| Track | Implemented entry | New GPU policy inference? | Output |
|---|---|---|---|
| F1 mechanism controls | `robo.roundtrip.mechanism_followup` | Yes, same-process REF/B0/B1/B2/B3 | canonical paired ledger and component contrasts |
| F2 scorer sensitivity | `robo.roundtrip.scorer_sensitivity` | No | read-only final-state diagnostic records |
| F3 target-region appearance | `robo.roundtrip.object_fidelity` | No | reference masks, masked PSNR, crop SSIM/LPIPS |

F2/F3 still need the existing native simulator/rendering environment to restore/render source scenes. They are not synthetic substitutes and do not modify any official success. On a validated CPU/Mesa setup they can avoid inference GPUs; otherwise run inside a normal GPU allocation. No heavy work on login nodes.

## 0. Checkout and validate

Use a clean detached worktree at the merged code revision. Reuse the existing native/GS/policy environments; do not upgrade the pinned native benchmark, policy or checkpoint. The original cluster interpreter was `/group/worldcept/code/SimAny-wt/sr0-native/.venv-native/bin/python`; verify it still exists.

```bash
git fetch origin
git worktree add --detach ../PhiRoom-submission-followup origin/main
cd ../PhiRoom-submission-followup
export SIMANY_NATIVE_PY=/group/worldcept/code/SimAny-wt/sr0-native/.venv-native/bin/python
bash run/roundtrip/submission_followup.sh test
bash run/roundtrip/submission_followup.sh mechanism --help
bash run/roundtrip/submission_followup.sh scorer --help
bash run/roundtrip/submission_followup.sh fidelity --help
```

CPU CI runs pure geometry, new schema/roster/native-engine preparation, fake-scene integration, actual PSNR/SSIM and existing table-regression tests. It does NOT run the gated native assets, learned policy, native segmentation or checkpoint LPIPS on real scenes. Pass one real source-matched scene in the allocated runtime before releasing the remaining jobs.

## 1. Resolve existing inputs, do not regenerate them

The previous publication root was:
`/group/worldcept/code/SimAny/outputs/icra2027/20260907-3b5e853-v1/sim_recon_sim/scale_up/paper_tables/`.

Read its `release_manifest.json`, `table_provenance.json`, prior N1/N2 dispatch receipts, and `plan/icra2027/11_sim_recon_sim/09_scale_up/EXECUTION_STATUS.md` to resolve actual source paths. A publication directory is not automatically a capture or a build directory.

Required F1 inputs:

- `source_planned_units`: exact original primary L0 TEST plan, all 48 instances/480 reference resets.
- `source_bindings`: canonical acquisition JSONL with `canonical_instance_id`, `capture_manifest_sha256`, capture and bundle locations.
- `source_build_bindings`: exact current B0/B3 native binding JSONL(s), including original discovery failures. Do not feed every historical dispatch snapshot; conflicting bindings are rejected.
- `candidate_pool_roots`: immutable target `candidate_pool.json` files. Existing outcomes already include `B1_FIXED_PRIORITY` and `B2_EVIDENCE`; no new TRELLIS/RVG generation is required.
- `engine_admission`: existing successful `per_canonical_engine_v1` admission for the unchanged policy stack.
- `worker_source`: the clean detached checkout that the ordinary jobs will execute.

Copy `configs/experiments/sim_recon_sim/mechanism_followup.template.yaml` to a resolved config outside tracked source, replace all paths, set permitted Slurm arguments and `execution_ready: true`. If a pool is missing, the tool only propagates a hash-bound common TRAIN discovery failure. Any other missing source is a setup error, not a fabricated build failure.

## 2. F1: prepare and launch B1/B2 mechanism controls

```bash
export NEW_ROOT=/group/worldcept/code/SimAny/outputs/icra2027/NEW_IMMUTABLE_STAGE/simulation_submission
export CONFIG=/absolute/path/mechanism_followup.resolved.yaml

bash run/roundtrip/submission_followup.sh mechanism prepare \
  --config "$CONFIG" --out "$NEW_ROOT/mechanism"

# Inspect the planned command and resource request. This submits NOTHING.
bash run/roundtrip/submission_followup.sh mechanism launch \
  --bundle "$NEW_ROOT/mechanism" --max-jobs 1

# Submit the first canonical-instance block after checking paths/resources.
bash run/roundtrip/submission_followup.sh mechanism launch \
  --bundle "$NEW_ROOT/mechanism" --max-jobs 1 --submit
```

Every instance job runs a fresh REF/B0/B1/B2/B3 block on ONE policy process with the original reset/horizon/preprocessing/RNG contract. There are 48 ordinary jobs and 2,400 planned units for the complete original roster. Some units may be nonexecuted construction failures. Old REF outcomes are NEVER reused across the new engine process. All preexisting candidate artifacts are reused after hash validation.

Check `jobs/<instance>/submission.json`, the native endpoint log, worker terminal shards and source identities. If the first block completes its execution contract (regardless of whether the method succeeds), submit the next bounded batch:

```bash
bash run/roundtrip/submission_followup.sh mechanism launch \
  --bundle "$NEW_ROOT/mechanism" --max-jobs 4 --submit
```

`--max-jobs` limits submissions in that invocation, not total running account jobs. Inspect `squeue` and quotas before each batch. Existing submission intents are never blindly resubmitted. An ambiguous failed `sbatch` response requires scheduler inspection, not deleting the intent. If a policy engine dies, do not resume its remaining arms against an old REF under a new process; start a separately recorded fresh paired block.

### Collect and generate tables

```bash
bash run/roundtrip/submission_followup.sh mechanism collect \
  --bundle "$NEW_ROOT/mechanism" --out "$NEW_ROOT/collection-001"

"$SIMANY_NATIVE_PY" -m robo.eval.paper_pipeline \
  --config "$NEW_ROOT/collection-001/paper_pipeline.json" \
  --out "$NEW_ROOT/tables-001"
```

The existing canonical collector and table pipeline produce coverage, native success and paired B1−B0, B2−B1, B3−B2, B3−B0 contrasts. Use another collection/output directory for later snapshots. Do not add `--paper-root` yet: the old native publication prose must not overwrite the newly scoped manuscript. F1 is a **mechanism follow-up on an already observed TEST roster**, not an independent unseen confirmatory test. Do not tune the constructor on these outcomes or pool the follow-up statistically with the old five-arm release.

## 3. F2: read-only scorer sensitivity

See [02_scorer_sensitivity.md](02_scorer_sensitivity.md). Start on an explicit DEV roster, then the unchanged full original L0 plan/ledger:

```bash
bash run/roundtrip/submission_followup.sh scorer \
  --planned "$ORIGINAL_PRIMARY_PLAN" --ledger "$ORIGINAL_PRIMARY_LEDGER" \
  --out "$NEW_ROOT/scorer"
```

This restores original assets and final qpos/qvel, calls `mj_forward` for kinematics/contacts, and checks that native success exactly matches the logged result. It never calls a policy or integrates a new physics trajectory. Original success and counts remain unchanged. The output reports origin/visual-centroid retreat disagreement and declared-goal-region vertex tests; these are supplementary geometric sensitivity checks, not a new success definition or cavity ground truth. Missing state, XML mismatch and unsupported goals stay unavailable. This first implementation covers L0; do not feed L1 fixture replacements into it.

## 4. F3: independent target-region appearance

See [03_object_fidelity.md](03_object_fidelity.md). Use the original acquisition bindings and existing held-out appearance render roots:

```bash
bash run/roundtrip/submission_followup.sh fidelity masks \
  --bindings "$ORIGINAL_BINDINGS" --out "$NEW_ROOT/reference-masks"

bash run/roundtrip/submission_followup.sh fidelity metrics \
  --masks "$NEW_ROOT/reference-masks/mask_manifest.json" \
  --renders "$ORIGINAL_NATIVE_RENDER_ROOT" --out "$NEW_ROOT/object-metrics" --device cuda
```

Masks use native evaluator segmentation after a byte-exact reference RGB check, never SAM predictions. Crop coordinates are reference-mask bbox plus fixed padding. Metrics reuse the existing PSNR/SSIM/LPIPS implementations. Missing target visibility and missing method renders remain in coverage. Explicit `--skip-lpips` is a partial diagnostic only, not a finished LPIPS result. Mask extraction supports `--shard-index i --shard-count n`; run ordinary independent jobs and pass ALL mask manifests with repeated `--masks` flags to the metrics command.

## 5. Paper handoff

The paper is now scoped to real-scan construction and simulated interaction, with B3 core and B4/BM/L1 limitations retained. Deliver F1 tables and contrast JSON, F2 sensitivity records, F3 ROI metric CSV/JSON, source hashes and a short claim decision. Integrate only measured rows through a source-bound generator. No hand-typed performance values, no physical-trial placeholders, no stronger preservation claim from a nonsignificant difference.

Authoritative detailed task notes: [01_mechanism_controls.md](01_mechanism_controls.md), [02_scorer_sensitivity.md](02_scorer_sensitivity.md), [03_object_fidelity.md](03_object_fidelity.md). Lead-agent prompt: [AGENT_PROMPT.md](AGENT_PROMPT.md).
