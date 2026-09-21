# Automatic construction handoff (engineering scope)

The existing `robo.eval.e3_factory_materializer` accepts
`--automatic-scene-contract <descriptor.json>` for schema-2 automatic E3 inputs.
It consumes sealed input inventory, observation, control, and selected-artifact
closures. Evaluation aggregation is not a prerequisite for this engineering
export. `paper_ready` remains false; geometry evaluation and final qualification
are separate gates.

The descriptor is an exact object with these fields:

```json
{
  "schema_version": 1,
  "scene_id": "09c1414f1b",
  "discovery_directory": "outputs/icra2027/<source-freeze>/auto_discovery_pilot",
  "discovery_hashes": {
    "pilot_summary.json": "<sha256>",
    "input_manifest.json": "<sha256>",
    "output_hashes.json": "<sha256>",
    "postrun_audit.json": "<sha256>"
  }
}
```

For `FRESH_OFFICIAL_TRAIN_ONLY`, `discovery_hashes` must additionally include
`all_jobs_manifest.json`; the existing canonical `source_jobs` validator checks
its completion seal, complete segmentation population, stage roster, and
training-frame crop identities. Gaussian provenance must agree across input,
summary, and E3 audit. Four-member `UNKNOWN` sources remain engineering only.

Hashes must match the automatic E3 inventory audit. The descriptor binds the
actual derived mesh, automatic segmentation, prepared-object metadata, and
original Gaussian/camera identities. It cannot silently replace a Gaussian
whose training-view provenance is unknown. The latter status is retained.

Stable exported IDs are `obj_<automatic_instance_id>`. Prepared crop/generator
indices remain a separate explicit namespace. The old discovery metadata key
`gt_object_id` is checked only as its producer's automatic-ID alias and is never
copied into the new factory or dispatched to a GT lookup. The materializer calls
the existing automatic loader with explicit mesh and segmentation paths.

All planned jobs receive metadata and selected-decision records. Accepted jobs
reuse canonical registration, physics, collision, probe, raw-tool, and URDF
validation. A0 rejection and A4 abstention produce no fabricated dynamic assets.
Automatic acceptance is `construction_eligible`; no evaluation quality tier is
copied or inferred. Existing size, mass, stability, task, and camera predicates
remain unchanged.

For an exact clean source and a new reserved output root:

```bash
python -m robo.eval.e3_factory_materializer \
  --e3-root outputs/icra2027/<e3-freeze>/agentic \
  --scene-id 09c1414f1b --policy-id A0 \
  --automatic-scene-contract <descriptor.json> \
  --out outputs/icra2027/<new-freeze>/harness/materialized/A0
```

Repeat the same command with A4 and a distinct destination. The canonical
`robo.sim.export_mjcf` must use `SIMANY_AUTO=1`, `SIMANY_MESH_SRC=derived`, and
`SIMANY_OUT=<matching materialized factory>`. Missing automatic mode or a scan
mesh is rejected before instance loading. Paired static carving uses the union
of actually accepted A0/A4 replacement hulls and retains the complete planned
metadata. Rejected/abstained source geometry stays static. Missing geometry in an
incomplete automatic reconstruction is not synthesized and remains a coverage
limitation to report during qualification. Mesh caches are keyed by scene ID, resolved
path, and authenticated content digest, preventing GT/derived and scene reuse.

Focused smoke (including negative tests):

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q \
  tests/test_e4_automatic_materializer.py tests/test_e3_factory_materializer.py \
  tests/test_e4_collision_repair.py tests/test_e4_robust_floor_support.py \
  tests/test_e4_candidate_screen.py --basetemp=outputs/automatic-materializer-tests
```

Synthetic smoke is not a two-room pilot or a manipulation result. No policy or
GPU run is authorized by the existence of this export alone; the existing
workspace, collision, reset, camera, scorer, and source-contract gates still
apply to the complete declared candidate population.
