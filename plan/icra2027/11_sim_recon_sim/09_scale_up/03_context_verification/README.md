# N3 — Task-Conditioned System Verification and bounded repair

**Priority P0. Owner: verification/controller agent.** Read N2/N4 contracts and parent `03_system_verification/README.md`. The central test is whether deployment-matched verification improves interaction, not whether a self-selected subset passes its own probe.

## What changes from the isolated probe

The first native milestone used a single convex hull for an isolated Bullet probe but five CoACD parts for actual native manipulation. It is not a same-representation test. Measure the ACTUAL exported parts, estimated room placement, native MuJoCo contact parameters, support and destination. Never teleport the object to a standard plane and call that room verification.

## TODO: verify actual context

- [ ] Build the task dependency graph from public instruction, reconstructed roles and the known benchmark robot/cameras. Cover both robot-to-object approach and object-to-destination transport. A geometric corridor is explicitly a proxy, not a motion-planning guarantee.
- [ ] Check finite units/poses/scales and role grounding. Distinguish an unavailable required object from an unused unrelated object.
- [ ] Measure object/support closest points, overlap, penetration, actual initial pose, contact list and native settle displacement/rotation. Include the reconstructed destination's interior access. Do not infer an open cavity from its category label.
- [ ] Record collision part hashes and engine/contact parameters for every measurement. Nominal thresholds depend on task scale and are set only on DEV. Do not transfer prior standard-plane pass/fail directly.
- [ ] Known robot/camera setup is supplied, not inferred from scene capture. Record real IK/collision tests separately from geometric reach/visibility proxies.
- [ ] Tag verification context: L0 uses retained oracle room support; L1 is partial reconstructed context; L2 uses reconstructed relevant workspace. A L0 repair result does not establish reference-free room repair.

## Bounded action bank

Start from B3. B4 can execute at most TWO additional tool actions per affected object/task dependency:

1. alternative observation-supported registration/scale initialization;
2. bounded support-consistent placement correction using observed/reconstructed support;
3. collision regeneration with frozen multi-part/thickness parameters;
4. reselection of an already generated candidate using actual-context evidence.

These are candidate actions, not mandatory calls. Freeze their preconditions, parameter ranges, ordering logic and maximum calls on DEV. No hidden native pose/shape/mass/contact parameter is an optimization target. Diagnostic oracle interventions live in a separate privileged run and never become B4 evidence.

Each repair produces new artifact identity, before/after geometry or pose, evidence, reason, elapsed cost and invalidated dependencies. Keep visual mesh, collision, GS transform, background removal/completion, task handles and contact lists consistent. After validation probes, restore the declared initial state. Do not preserve a favorable post-settle state as an undeclared reset repair.

Never weld/freeze a movable object, remove a task obstacle, plug/unplug a container using GT, loosen a TEST threshold, or call repeated rescoring a new repair. Cosmetic smoothing only counts as a tool action if it genuinely changes a declared artifact and is fully costed.

## Controls required for the conclusion

- B3: current evidence selection/retry, logging room-context diagnostics without blanket new conservative rejection.
- V1: same checks and final acceptance as B4, no repair; optional separate policy arm for direct reject-vs-repair attribution.
- B4: condition-dependent bounded action choice.
- BM: same extra action bank, maximum calls, input pool, final selection and acceptance. Execute a predeclared fixed order without failure-specific scheduling. Skip only formally inapplicable actions with logged preconditions. Report realized CPU/GPU time and calls; maximum-budget matching is not exact compute matching.

Initially repair and freeze each canonical build once BEFORE TEST policy outcomes. Do not adapt the build after inspecting reference trajectories, native test success or which reset failed. A future per-reset online-repair method is a separate protocol/budget.

## Do not recreate the all-invalid experiment

Mandatory schema/load/nonfinite failure ends execution with a typed failure. A finite drifting asset is eligible to produce a negative baseline measurement. Only an arm whose specified policy abstains is withheld, and its lost planned coverage remains. Verification accuracy and native-task success are separate. Missing optional confidence evidence is not an excuse to globally discard all methods' tasks.

## Outputs

Extend the existing orchestrator ledger. Add or complete `robo/roundtrip/system_verification.py` only if not already present; expose native adapter hooks rather than a second engine. Output `verification_before.json`, `repair_ledger.jsonl`, `verification_after.json`, `build_manifest.json`, `context_privileges.json`, `action_costs.csv`, failure reasons and synchronized before/after diagnostic frames.

## Tests and pilot

Synthetic fixtures: floating target, overlapping support, wrong metric scale, sealed convex container, wrong task-role binding, a valid scene requiring no action, and an irreparable scene. Test bounds, maximum calls, no GT access, no fake welding, dependency refresh, restoration after probe, V1 vs B4 differences and BM fixed order. Native DEV pilot must expose actual contacts and at least one genuine action; do not invent a successful repair if none occurs.

Implementation passes when decisions and measurements are correct. The scientific gate passes only from N5 native outcomes at stated coverage and cost, not B4's own pass fraction. Hand N5 immutable B3/B4/BM assets, optionally V1. Report all failure transitions and write `STATUS.md` with `IMPLEMENTATION_READY` and `OUTCOME_CLAIM` as distinct fields.
