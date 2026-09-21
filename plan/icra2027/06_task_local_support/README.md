# E6 — Task-Local Support and Selective Use

**Priority:** P1  
**Paper output:** task-support table and risk-coverage curves  
**Depends on:** E0, frozen construction/fidelity evidence, hidden-GT scenes

## Research question

Can construction-time evidence identify when a particular manipulation query is unsupported, and does task-local aggregation outperform scene-global scores?

This is not a formal proof of simulator correctness and not a real-world success predictor. It is a held-out classification/selective-use experiment over predeclared task requirements.

## Read first

- `robo/certification/task_graph.py`
- `robo/certification/grounding.py`
- `robo/certification/features/visual.py`
- `robo/certification/features/geometry.py`
- `robo/certification/features/support_contact.py`
- `robo/certification/features/robot_control.py`
- `robo/certification/features/dynamics_probes.py`
- `robo/certification/model.py`
- `robo/certification/calibrate.py`
- `robo/eval/audit_labels.py`
- `robo/eval/audit_loso.py`
- `robo/eval/audit_metrics.py`
- `plan/11_TASK_INTERACTION_GRAPH.md`
- `plan/12_TASK_CONDITIONED_CERTIFICATE.md`
- `plan/14_ORACLE_BEHAVIOR_BENCHMARK.md`

## Experimental unit

One row is one `(scene, task query, degradation/build condition)` tuple.

Minimum target:

```text
6 hidden-GT scenes
× 4 task families or templates
× 3 conditions (clean, mild, severe)
= at least 72 query rows
```

Increase the set if any leave-one-scene-out fold lacks both valid and invalid labels.

## Hidden-GT boundary

Construction and feature extraction must not read:

- GT object identity mapping;
- GT object pose/scale;
- GT support graph;
- GT collision geometry;
- validity labels;
- evaluation thresholds.

Vault the references in a separate path or process. Produce build/features first, hash them, then join labels in a separate evaluation command.

## Task graph

For each query, ground:

```text
robot
policy cameras
manipulated object
target/receptacle
support surfaces
obstacles intersecting the robot/object swept workspace
one-hop visibility relations
```

Unrelated room objects remain in simulation but do not contribute evidence to the task-local row.

Every graph must be serialized to JSON with node/edge provenance and unresolved references. Missing grounding is not silently dropped; it becomes missing evidence and may make the query invalid.

## Validity labels

Define labels in a frozen YAML before evaluating predictions. A query is invalid when any required condition fails, including:

- required manipulated object or target not recovered;
- translation, rotation, or scale error beyond a predeclared task tolerance;
- incorrect support relation;
- penetration/collision discrepancy beyond tolerance;
- required target or workspace not visible in the policy camera;
- robot-to-scene alignment beyond tolerance;
- build failure for any required node.

Store both the binary label and all reason codes. Do not derive labels from policy success, because policy competence and construction validity are different variables.

## Fixed comparison rows

Use the same low-capacity estimator and split wherever possible. Change only the evidence support:

1. `Global PSNR only`
2. `Global geometry F1 only`
3. `Scene-global evidence`
4. `Manipulated-object evidence`
5. `Task graph, visual + geometry`
6. `Full task-interaction graph`

The full graph adds support/contact, visibility, and robot-frame evidence.

Current `robo.eval.audit_loso` groups must be updated to match these exact rows. In particular, add an explicit `global_f1` group and align display names with the paper. Do not keep a stale Global-LPIPS row in code while the paper reports Global-F1.

## Required feature CSV

Produce:

```text
outputs/icra2027/<freeze_id>/audit/task_local_features_and_labels.csv
```

Required identity/label columns:

```text
freeze_id
scene_id
task_id
task_family
condition_id
invalid_label
invalid_reasons
graph_path
build_manifest_path
```

Feature prefixes must remain stable:

```text
global_psnr_*
global_f1_*
scene_*
visual_*
geometry_*
support_*
robot_*
```

Missing features are empty/NaN plus explicit missingness indicators. Never impute them as a pass before the fold-specific transform.

## Required commands

### Build graphs and features

Create one canonical driver, for example:

```bash
python -m robo.eval.build_task_support_dataset \
  --config configs/experiments/icra2027/audit.yaml \
  --out outputs/icra2027/<freeze_id>/audit/features
```

It must write features without labels first, then invoke a separate label join using the vaulted reference.

### Leave-one-scene-out predictions

```bash
python -m robo.eval.audit_loso \
  --input outputs/icra2027/<freeze_id>/audit/task_local_features_and_labels.csv \
  --out outputs/icra2027/<freeze_id>/audit/predictions \
  --l2 1.0
```

### Standard metrics/table

```bash
python -m robo.eval.audit_metrics \
  --input outputs/icra2027/<freeze_id>/audit/predictions/heldout_predictions.csv \
  --out outputs/icra2027/<freeze_id>/audit/table
```

## Model contract

Use a deliberately low-capacity model. The current L2-regularized logistic regression is appropriate.

- Standardization, median imputation, and missingness indicators are fit inside each training fold.
- Hyperparameters are frozen globally or tuned only within nested training folds.
- Do not use a tree ensemble or neural model as the headline result on this sample size.
- Archive coefficients per fold for inspection.

## Metrics

Use:

- AUROC;
- AUPRC;
- Brier score;
- Risk@80 coverage;
- Risk@60 coverage;
- full risk-coverage curve in supplement.

Risk at coverage `c` is the invalid-query fraction among the `c%` lowest predicted-risk queries. Failed builds remain invalid rows in the denominator.

## Required outputs

```text
outputs/icra2027/<freeze_id>/audit/
  config_resolved.yaml
  graphs/<scene>/<task>/<condition>.json
  features_unlabeled.csv
  labels.csv
  task_local_features_and_labels.csv
  predictions/heldout_predictions.csv
  predictions/fold_models.json
  table/audit_table.json
  table/audit_table.csv
  table/audit_table.tex
  risk_coverage.csv
  risk_coverage.pdf
  counterexamples.csv
```

Counterexamples are selected algorithmically:

- globally strong but locally invalid;
- globally weak but locally valid;
- correct abstention due to missing evidence;
- false accept;
- false abstention.

## Tests

- controller/feature process cannot access vaulted paths;
- graph serialization is deterministic;
- unresolved task nodes remain explicit;
- label join is one-to-one;
- every LOSO test scene is absent from its training fold;
- transform statistics use training rows only;
- missingness is preserved;
- `global_f1` row exists and stale row names fail schema validation;
- risk-at-coverage matches a hand-calculated example;
- failed builds remain in labels and predictions;
- every fold contains both classes or the run fails with a diagnostic.

## Fast smoke

Use 2 synthetic scenes, 3 queries each, and deterministic toy features. Verify the six signal rows, fold isolation, AUROC/Brier, and risk-coverage outputs in under 1 minute.

## Claim gate

The local-support claim passes only if, on held-out scenes:

- full task-interaction evidence improves selective risk over global PSNR/F1 and scene-global evidence;
- risk generally decreases as coverage is reduced;
- the result is not explained by one scene or one degradation level;
- failure rows remain in the denominator.

High AUROC with non-improving risk-coverage does not pass the selective-use claim.

## Acceptance criteria

The sealed cohort contains all 72 queries and six declared folds, but every
query is invalid under missing-evidence requirements. The canonical join refuses
LOSO estimation; the remaining performance criteria are unmet, not completed
with surrogate predictions. `LOCAL_SUPPORT_GATE=FAIL`; see validity freeze
`20260906-047bd5d-v1` and paper `18def27`.

- [ ] At least 72 query rows and 6 held-out scene folds.
- [ ] Both valid and invalid examples occur in every test analysis and training fold.
- [ ] The six paper rows are generated from the same split and compatible estimator.
- [x] No GT leakage into construction/features.
- [ ] Risk-coverage and reason-code breakdowns are archived.
- [x] Paper wording is narrowed if the claim gate fails.

## Handoff

`STATUS.md` must report row counts by scene/task/condition/label, fold health, feature groups, metric table, risk-coverage result, dominant failure reasons, and `LOCAL_SUPPORT_GATE=PASS|FAIL`.