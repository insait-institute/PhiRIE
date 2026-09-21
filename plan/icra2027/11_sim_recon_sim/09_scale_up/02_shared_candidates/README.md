# N2 — Shared candidate construction, B0/B3 and observed-surface control

**Priority P0. Owner: asset-construction agent.** Depend on N1 public captures, not completed policy runs. Read the common matrix, parent SR2/SR3 tasks and the actual `robo/roundtrip/build.py`.

The current bridge deliberately supports one DEV target, B0 TRELLIS, ideal RGB-D and a narrow instruction pattern. It records no room Gaussian. Extend it with a versioned multi-object contract and reuse existing `agents/orchestrator/{controller,evidence,policies,runtime,job_graph}.py`, SAM3 wrappers, TRELLIS/RVG wrappers, `agents/assets/s6_physics.py` and canonical fidelity metrics. Do not implement another selector.

## TODO

- [ ] Preserve the original B0 DEV path and tests. Add method/scope/instance fields rather than deleting all current validation.
- [ ] Ground target/destination/support roles from public task language and image/depth evidence. Replace the single regular-expression assumption with a validated task-role adapter. No native asset IDs, hidden segmentation or GT geometry enter candidate selection.
- [ ] Freeze one discovery roster per capture. Log unresolved roles, duplicate detections and merged/missed objects. A missing object is a method outcome, not a reason to replace the test instance.
- [ ] Generate the same initial TRELLIS/RVG proposal pool once. B0 uses TRELLIS; B1 fixed priority and B2 evidence selection are cheap diagnostic selections; B3 reuses existing bounded registration retry. Distinguish an actual new generation call from an alternative registration action.
- [ ] Keep registration observations separate from held-out surfaces. Write raw selection evidence, transforms, reasons, candidate hashes, parent IDs and algorithm-stage timing. Never use held-out CD/F1 to choose candidates.
- [ ] Materialize each selected candidate with actual production collision parts, not the isolated hull probe. Record visual mesh, native CoACD meshes, mass/contact/inertia priors, canonical frame and scale applications. B3 vs B0 comparisons use the same material/export convention.
- [ ] Expose selected-artifact handles to N3/N4. Do not pass a failed isolated-hull flag as proof that the actual native CoACD asset cannot execute. Record both representations and actual-context tests separately.
- [ ] Implement an observed-surface/TSDF control from the same TRAIN RGB-D. Use the same importer and physical-prior policy. Any watertight repair/completion must be disclosed and charged; it is not magically complete geometry. Start with geometry and replay on DEV plus a fixed TEST subset, not another full policy factorial.
- [ ] Reuse `robo.eval.fidelity_metrics`: world-aligned estimated meshes, independent native-reference surfaces accessible only after build freeze, standard CD/F1 and held-out image metrics. Unmatched objects stay null; retain denominators. Add exact asset->export->reference binding before reusing metrics.

## Required products

Per canonical instance and method: `discovery.json`, `candidate_pool.json`, `selection_ledger.jsonl`, `build_manifest.json`, selected mesh/Gaussian/collision paths as applicable, `runtime.jsonl`, failure record where needed. Shared candidates have one physical storage location and explicit logical per-method compute attribution. Cache identity includes model weights, generator seed, capture bytes, crop/mask, registration and collision settings.

N4 receives immutable estimated placement and shape. It may not recenter in a way that changes world geometry or use GT pose to simplify import. If appearance remains uniform-color, label it honestly; do not imply this is a GS experiment.

## Tests and real pilot

Synthetic fixtures cover one missing tool, both proposals poor, retry-created transform, deterministic ordering, missing target and duplicate candidate IDs. Verify B1 cannot access evidence for scheduling and that B0/B3 reuse the exact original pool. Simulate a tool crash and preserve its planned object row.

Real DEV pilot: produce B0 and B3 for at least two different object shapes in separate canonical instances. Render native collision/visual overlays and hand the actual imported artifacts to N5. Pass construction integrity with measured negative outputs allowed. No requirement to tune until B3 wins.

## Acceptance

- [ ] Shared initial proposals exist for every planned eligible build, with explicit failures otherwise.
- [ ] B0/B3 assets and inexpensive B1/B2 geometry diagnostics derive from code, not a manually assembled result table.
- [ ] Actual CoACD and physical priors are bound; isolated vs native-context evidence is not conflated.
- [ ] Independent geometry/appearance evaluation and standard definitions are retained.
- [ ] Runtime reports scope/exclusions, failed calls and reused work transparently.
- [ ] N3/N4 can consume selected artifacts without rerunning generation or interpreting hidden paths.

Handoff `STATUS.md`: build/candidate counts, hash-bound outputs, paired reference support, missing cells, commands, stage commits, measured runtime and known generalization limits.
