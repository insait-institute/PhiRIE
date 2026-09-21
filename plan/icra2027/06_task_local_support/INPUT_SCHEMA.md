# E6 real-input stages

The canonical producer is `robo.eval.build_task_support_dataset`. Its old
`--smoke` mode remains explicitly synthetic. The new stages consume public
construction evidence and separate evaluation measurements. They do not
reconstruct a room, annotate task queries, unlock vaults, or replace the
existing feature extractors, validity producer, LOSO estimator, or metrics.

## Declare the population before observing labels

Resolve `audit.yaml` to a new freeze ID. Supply `public_roots`, a hashed
`public_manifest`, and four public task IDs for each of the six declared
scenes. The same task IDs appear for clean/mild/severe. The full tier requires
exactly 72 unique planned keys; explicit failed/rejected/abstained builds are
retained. Do not build the roster by enumerating successfully recovered
objects. Pilot subsets must also be predeclared.

Pin the SHA256 of `audit_validity_protocol.yaml` in the resolved config.
The existing `audit_labels.ValidityThresholds` defaults are pinned there:
5 cm translation, 15 degrees rotation, 15 percent scale error, and 1 cm
penetration. These values were selected from existing source before labels
were inspected. The feature process receives only the protocol hash and
never opens the protocol. Labels use the existing `audit_labels.label_record`.

The public manifest is JSON or YAML:

```yaml
freeze_id: <resolved freeze>
rows:
  - freeze_id: <same freeze>
    scene_id: <predeclared scene>
    task_id: <public task ID>
    condition_id: clean
    task_family: object_to_receptacle
    build_manifest: {path: /absolute/public/bundle.json, sha256: <SHA256>}
```

Each hashed construction bundle contains those four identity fields plus:

- `source_kind: real` (`synthetic_fixture` is accepted only in fixture tier).
- `evidence_source: construction_observation` and an explicit terminal
  `build_status`: `complete`, `failed`, `rejected`, or `abstained`.
- For complete builds, `scene` and `task` dictionaries in the existing
  `robo.certification.task_graph` schemas. Supply robot geometry, cameras,
  task grounding parameters, public role labels, and object AABBs explicitly.
  Scene/task IDs must match the planned row. These dictionaries contain no
  private object identity mapping. Missing scene/task data cannot masquerade
  as a declared failed build.
- Optional `audit` in the existing construction build-audit schema. Evidence
  must be computed against construction observations, never evaluation
  references. Absent evidence remains empty plus explicit missingness flags.

The driver checks hashes, root boundaries (including symlinks), identity,
source-kind consistency, and known forbidden fields. `gt/`, `ground_truth/`
and any vault directory cannot be feature inputs even when nested under an
approved root. Declaration is not independent proof of GT-free provenance:
the upstream producer/input hashes still require review before paper use.

## Build and seal construction-only features

```bash
python -m robo.eval.build_task_support_dataset --stage features \
  --config /absolute/resolved-audit.yaml \
  --out outputs/icra2027/<freeze>/audit/features
```

This publishes `features_unlabeled.csv`, graph JSON files,
`config_resolved.json`, and `feature_seal.json` atomically without overwrite.
The seal hashes every member, all construction inputs, the driver, and the
predeclared label protocol. Nodes/edges point to the construction manifest
hash. Missing grounding remains explicit. Scene evidence aggregates all
construction objects, `object_*` aggregates manipulated-object hypotheses,
and `visual_*`/`geometry_*` use the graph role hypotheses. Full graph adds
`support_*` and `robot_*`. Scene-level held-out rendering, alignment residual,
and scale-confidence values are excluded from local evidence. Existing
geometric reach/support proxies retain their producer semantics; they are
not a new IK solver or physical validity oracle.

## Join evaluation measurements in a separate process

Only after the feature seal exists may an authorized evaluation process
produce a JSON/YAML measurement file with `freeze_id`,
`feature_seal_sha256`, and `rows`. Every row contains the exact four-field
join key and `measurement_scope: all_required_task_invariants`.

Required measurements are `object_recovered`, `translation_cm`,
`rotation_deg`, `scale_error_pct`, `penetration_cm`, `support_correct`,
`collision_valid`, `required_visible`, and `robot_alignment_correct`.
For multiple required nodes, booleans must be conjunctions and errors must
be worst-case errors across those nodes; recovering only the manipulated
object cannot make a missing receptacle pass. Alignment boolean requires
the independently declared capture/alignment protocol. Missing evidence is
invalid; this driver does not invent another alignment tolerance.

Optional global baseline columns `global_psnr_db` and `global_f1_score`
enter only here, after construction features are sealed. They are
evaluation-only predictor baselines and never inputs to graph construction.
Policy success, reward, and supplied validity labels are not accepted as
construction-validity evidence.

```bash
python -m robo.eval.build_task_support_dataset --stage join \
  --features-dir outputs/icra2027/<freeze>/audit/features \
  --label-protocol configs/experiments/icra2027/audit_validity_protocol.yaml \
  --measurements /authorized/evaluation/measurements.json \
  --measurements-sha256 <SHA256> \
  --out outputs/icra2027/<freeze>/audit/joined
```

No permission changes occur. The driver verifies the feature members,
unchanged construction sources, protocol, producer, measurement hash,
freeze, and exact one-to-one key coverage before publication. The output
contains separate labels, the joined CSV, and `label_join_manifest.json`.
Both classes must occur in every training and test fold; otherwise the
command fails with a diagnostic **after preserving all joined rows**. It
does not prune scenes to repair class balance. Final evaluation uses the
existing `audit_loso` and `audit_metrics` commands on that joined CSV.

The driver always reports `paper_ready=false`. Actual full outputs,
fold-model archives, plots/counterexamples, and the scientific claim gate
remain downstream work requiring admissible real inputs. Unit fixtures
and the old six-row smoke are not a real-data pilot.
