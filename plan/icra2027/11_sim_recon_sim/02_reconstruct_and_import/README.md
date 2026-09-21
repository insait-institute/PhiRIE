# SR2 — Reconstruct once, import into the native engine

**Owner:** reconstruction/import agent. **Priority:** P0. **Depends on:** SR1 public capture and SR0 native adapter. Read ../PROTOCOL.md. The first goal is one replaced object, not every room or generator.

## Reuse

Read `agents/recon/gsplat_train.py`, `agents/discover/auto_segment.py`, `agents/discover/factory_prepare.py`, `agents/assets/factory_hybrid.py`, `agents/assets/s5_align.py`, `agents/assets/s6_physics.py`, `agents/orchestrator/`, `robo/sim/export_mjcf.py`, and `robo/eval/fidelity_metrics.py`. Read the existing automatic-inventory/schema code before adding another object representation.

Initial proposal pool: the existing TRELLIS and ReconViaGen tools. TRELLIS.2 is not a dependency and must not be injected into a frozen initial pool. No broad generation-model sweep.

## Reconstruction stages

1. Build source Gaussian and a metric reconstruction from TRAIN sensor inputs. RGB-D may initialize from TRAIN depth/TSDF; posed RGB-only must not. Record source GS training separately from asset-stage runtime.
2. Discover instances with current automatic masks and cross-view merging. Tune resolution-sensitive pixel gates only on development views; save them in config. Do not retain old 1752x1168 pixel thresholds blindly at a different resolution.
3. Create one persistent record per discovered object: observed surface, candidate meshes/appearance, metric Sim(3), collision, physical prior, removal set and provenance. Shared object identity is required from generation to renderer and simulator.
4. Generate and cache the initial proposal pool once per object; share it among B0-B4 wherever the policy permits. Run selection/registration via existing orchestrator, without evaluation GT.
5. Produce two assets: visual mesh/material or object GS, and collision representation. Track canonical frame, applied scale and link/COM transforms explicitly.
6. Produce clean background using automatic removal/completion. Never use a hidden reference render with the object deleted as the method's clean background. Such a render is an oracle diagnostic only.

## Native importer

Add `robo/roundtrip/importers/robocasa.py` and a thin `robo.roundtrip.build` entry point. Do not replace the native robot, controller or task implementation with the DROID wrapper. The importer modifies environment assets, then reconnects native body/fixture handles through the evaluator's post-build correspondence.

Import invariants:

- meters, right-handed frame and explicit quaternion convention;
- apply scale once only; test translated/off-center meshes;
- transform inertia/COM consistently, validate positive mass and inertia;
- remove original visual AND collision geoms for replaced entities;
- no hidden proxy supports, fixed welds, gravity overrides or original colliders;
- do not fill an open container with a single convex hull; preserve concavity using validated multi-part decomposition;
- use safe minimum collision thickness declared globally, and log every geometry inflation;
- reconstruct room supports and obstacles for full-room scope;
- record retained native entities and original fixture metadata access.

The native success evaluator must operate on reconstructed bodies/geometry, not stale original fixtures. For fixture bounds/sites used by the native rubric, derive replacements from reconstructed geometry when possible. If an indispensable original cavity annotation remains necessary, label an `oracle_task_metadata` diagnostic and do not call that row fully reconstructed. Merely copying old fixture AABBs into a new object is not an acceptable hidden shortcut.

## Incremental replacement ladder

On DEVELOPMENT only, run:

1. target-only, oracle room context;
2. target + open receptacle;
3. target, receptacle and supporting workspace;
4. full declared room.

Every stage uses the same reference task and records replacement scope. Do not silently change goals, destination, camera or robot to make the new asset work. A failed full-room stage does not invalidate a useful task-workspace result, but the title/table claim must name that scope.

Maintain `entity_inventory.json`: every original entity is `known_robot`, `retained_oracle_diagnostic`, `replaced`, `reconstructed_static`, `unavailable`, or `outside_declared_scope`. Full-room mode rejects residual native environment colliders inside scope. Missing manipulated objects remain construction failures.

## Scorer integrity fixtures

Verify native goal results on manually constructed unit-test scenes: clearly outside, inside open container, intersecting wall, hovering over support, on support after release, missing target. These test fixtures may be synthetic and are not reported as reconstruction results. A sealed reference object can initialize an evaluator control; it never enters the construction pipeline.

## Proposed command

```bash
python -m robo.roundtrip.build --config configs/experiments/sim_recon_sim/build.yaml --capture "$CAPTURE_ID" --method B3 --scope target_only --out "$OUT/reconstruction"
python -m robo.roundtrip.native_import --config configs/experiments/sim_recon_sim/import.yaml --build "$BUILD_MANIFEST" --out "$OUT/native_import"
```

Implement these CLIs first and preserve legacy launchers. They invoke existing stages rather than duplicating generation/metrics.

## Outputs and tests

Outputs: `build_manifest.json`, discovery/candidate/selected records, `entity_inventory.json`, room/object meshes/collisions, GS assets, clean background, native environment artifact, `role_binding_report.json`, `import_receipt.json`, stage runtime records, and failure inventory.

Tests: asymmetric frame/COM fixture, no double scaling, no original collider left behind, concave container insertion, missing texture/collision failure, native scorer rebind, two differently sized fixtures, pose estimate not overwritten by GT, replay after save/restore, RGB-only depth access rejection.

Smoke: one known synthetic mesh import without generation, then one real sensor-reconstructed object. Pilot: two development task builds before full scope. Handoff a compiled native environment plus exact object record to SR3/SR4/SR5, even if its measured task fails. Preserve failure evidence; do not spend another full sweep before diagnosing support/pose/collision.
