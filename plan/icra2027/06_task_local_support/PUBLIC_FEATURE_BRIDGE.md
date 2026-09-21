# Public grounding to canonical E6 features

The adapter `robo.certification.public_feature_bridge` consumes only the sealed
public grounding publication produced by exact source f13248f. It authenticates
that source and invokes its unchanged output validator before adapting a unit.
The original source gate, query hash, selected object identity and observed
region-overlap score remain attached to each construction role binding.

`construction_role_bindings` is distinct from benchmark `rubric.role_refs`,
which remain forbidden in public feature inputs. The graph admits exactly the
observed selected object and does not rerank it with language/category priors.
Unresolved or ambiguous public roles remain unresolved. Support candidates do
not establish contact, and visible region patches do not establish solid
receptacle volume. Separate region IDs and query identities remain in the
public-role evidence. Unknown robot frames, policy cameras and physics remain
null or missing. Scope is public RGB geometry and roles, not a simulatable twin.

The bridge prepares the existing scene/task/audit bundle schema and a canonical
feature configuration. `robo.eval.build_task_support_dataset.generate_features`
performs the feature extraction and seals all rows before evaluation labels may
be opened. It reauthenticates public bindings and compares the entire adapted
bundle with the deterministic original-gate adaptation; edits to robot, camera,
physics, query, role, object or patch volume fail. Original-validation results
are cached only within one invocation and each referenced config/gate identity
is still checked on access. No mask, registration, stability or robot telemetry
is invented from discovery confidence or a visible AABB.

Pilot preparation requires the original scene0020 clean/mild/severe sequence
(12 queries). Full preparation requires all six scenes and three conditions,
exactly four queries per condition (72 rows), including failed constructors.
No query is replaced because its roles or source camera are missing.

```bash
python -m robo.certification.public_feature_bridge --config BRIDGE_CONFIG --out PREPARED_BUNDLES
python -m robo.eval.build_task_support_dataset --stage features --config PREPARED_BUNDLES/features_config.json --out SEALED_FEATURES
```

These are CPU commands. `SIMANY_EVIDENCE_ROOT` can select the existing validated
shared evidence root when executing from a frozen worktree; no-overwrite and
symlink/path validation remain active. A separate freeze/E0 binds source,
configuration and public input references. The feature stage consumes only the
predeclared label-protocol hash string; it does not read labels or the vault.

Authentication batches only repeated context validation: one subprocess runs
the unchanged f13248f `validate` once and `_validate_output` for every requested
unit in the exact fixed pilot/full sequence. Returned gates must match every
expected gate byte-derived value in order. The invocation-local cache keys
include producer identity, configuration identity and the full gate reference.
Mixed sources/configurations, reordered/omitted/extra units and stale gate
identities fail. A later preparation or feature invocation validates anew.
The canonical graph and feature extractors are unchanged by this batching.
