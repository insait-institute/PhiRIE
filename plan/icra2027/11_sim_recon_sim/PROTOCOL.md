# Shared experimental protocol

Read with README.md. All statements here are prospective design constraints. Nothing authorizes changing the previously published cohorts or labeling simulated outcomes as physical trials.

## 1. Information boundary

Three roles must be separate processes or principals:

- **Reference collector/evaluator:** may read native meshes, materials, collision, object poses, dynamics and task instance bindings. Saves the canonical state, captures permitted sensor observations and seals hidden references.
- **Constructor/verifier:** reads only the public capture bundle, known robot configuration and task instruction. Has no simulator-native asset registry, GT mesh/state directory, instance labels, segmentation IDs, native mass/friction, reference outcomes or hidden goal geometry.
- **Evaluation harness:** joins frozen reconstruction outputs to native task roles, instantiates paired conditions, and measures results. It never sends hidden evaluator measurements back to the constructor during a TEST build.

Public inputs are posed RGB, K, timestamps and declared metric camera transforms; add depth only in an explicitly labeled RGB-D condition. Known robot URDF, base placement, controller and policy cameras are provided configuration, NOT quantities the room constructor must rediscover. A public task instruction is allowed. Instance masks remain automatically inferred in the main condition. Oracle segmentation, oracle pose, oracle background, oracle parameters and target-only replacement are separately labeled diagnostics.

Use a real read-only allowlist mount or separate user/container. A directory named `gt/` is not isolation if the constructor can read it. Strip original asset/model IDs from exported filenames and metadata. Assert no native mesh file or asset retrieval API is opened by constructor workers. Restrict network asset lookup during construction. The evaluator's role correspondence is measured setup, not a construction feature.

## 2. Native simulator invariant

RoboCasa reference and reconstruction both run the same MuJoCo/robosuite build. BEHAVIOR reference and reconstruction both run the same OmniGibson/PhysX build. Freeze engine version, solver, timestep, substeps, robot assets, controller gains, action representation, gripper mode, camera model, observation preprocessing, checkpoint/normalization, action chunking, instruction, horizon, termination and success rubric.

Same engine is necessary but insufficient: an importer can silently rescale, recenter, change inertia, close a cavity, alter contact filtering or leave original colliders active. Two identity controls are required:

- **U0 wrapper identity:** untouched reference through the new adapter produces equivalent initial observations/state and identical native predicate results.
- **U1 import identity:** original permitted reference assets pass through the importer without reconstruction. Compare frames, masses/inertia, collisions, settling, actions and outcomes. This is a privileged engineering control. Run it in the evaluator, never in constructor workers.

Export/import translation or orientation corrections made solely to convert file conventions are implementation details and must pass a known-transform fixture. Pose/scale corrections inferred from observations are method actions and must be logged. GT-driven corrections are oracle ablations.

## 3. Task scope and eligibility

Begin with rigid object-to-surface/region and object-to-open-container tasks. Use exact native task IDs discovered from the pinned release. A generic name in this plan is NOT an official ID. Preserve the full native goal when reporting native-task success. A shortened BDDL subgoal is a custom subtask, labeled as such.

Exclude articulation, liquids, cloth, heat, cooking and state-conversion requirements BEFORE outcome collection. Freeze scene/task IDs and exclusion reasons by capability, not reconstructed quality. Use native reference setup to validate nonterminal reset, camera visibility and legal robot state. Do not impose another discovered-size whitelist if the native task already defines the manipulandum. Do not remove a TEST task after SimAnyRoom fails to find/export its target.

Native reference incompetence and reconstruction failure are different. Development requires at least one successful learned-policy episode and one reproducible diagnostic action trajectory to establish the interface; do not promise a specific success rate. If all reference development episodes fail, debug policy version/robot/action/normalization/horizon before reconstructing a fleet. TEST reference failures remain included. Report planned counts and native difficulty, not just a conditional success gap.

## 4. Capture, canonical state and reset mapping

Capture one canonical, settled, STATIC state per task instance. Log any immobilization and release it before native evaluation. Robot removed/parked for scanning is allowed only if declared equally for every reconstruction; task objects and supports must not change across capture frames. Do not pool multiple demonstration episodes with different object placements into one static GS.

Prefer 120 TRAIN views, 20 spatially held-out DEV views and 40 disjoint TEST views per build at 1280x720, subject to capability validation BEFORE freezing. Source camera bounds/trajectory use public workspace bounds, not per-method errors. Different camera K values are supported explicitly. No TEST view may initialize/optimize GS, choose crops, infer masks, fit registration, tune verification, or train Harmonizer.

For a fixed canonical object transform T_ref_i^0 and its reconstructed estimate T_rec_i^0, draw predeclared small world-frame perturbations D_i^k. Instantiate T_ref_i^k = D_i^k T_ref_i^0 and T_rec_i^k = D_i^k T_rec_i^0. This preserves estimation error while pairing the intervention. Apply identical robot reset instructions. Never silently replace T_rec_i^k by T_ref_i^k. Save both commanded and realized states after the same settle procedure. Fresh instability is a method outcome, not permission for per-reset hidden pose repair.

Native task evaluator role binding may use a sealed one-to-one mapping AFTER reconstruction. Main reporting includes detection/role-binding failures. A method missing a role is unavailable, not repaired by an oracle label. A second `POSE_ORACLE` condition may align poses for attribution, but is never mixed into end-to-end results.

## 5. Replacement scope

Freeze one of `target_only`, `target_and_receptacle`, `task_workspace`, `full_room` for every row. Start incrementally in development. Report the largest scope actually implemented; workspace success does not imply full-room success.

The original room in a target-only diagnostic is explicitly oracle context. Full-room replacement removes ALL original nonrobot environment visuals/colliders in the declared room. Known lights, camera rigs and task semantics can remain as declared infrastructure. Preserve an inventory of retained native entities with justifications. Objects that abstain cannot secretly remain as original dynamic assets. Nonreconstructed content may be static reconstructed background only, with inventory and capability limits reported. Never silently delete a relevant obstacle to improve success.

## 6. Predeclared treatment matrix

| ID | Reconstruction | Room verification/repair | Physics | Observation | Interpretation |
|---|---|---|---|---|---|
| REF_NATIVE | None | None | Native original | Native original | Reference, not a competitor |
| FIXED_NATIVE | B0 fixed TRELLIS | Logging only | Reconstructed | Native textured render | Baseline system |
| AGENT_NATIVE | B3 evidence + registration retry | Logging only | Reconstructed | Native textured render | Construction feedback |
| ROOM_REPAIR_NATIVE | B4 = B3 + bounded room-context repair | Enabled | Reconstructed | Native textured render | Proposed full physical build |
| ROOM_REPAIR_GS | Same B4 build | Same | Same B4 physics | GS composite | RGB-only intervention |
| REF_GS_DIAGNOSTIC | B4 visual assets | Not applied to physics | Oracle original | GS driven by reference state | Privileged appearance diagnosis |
| ROOM_REPAIR_HC | Same B4 build | Same | Same B4 physics | GS + Harmonizer Option C | Optional RGB-only intervention |

For construction effects compare FIXED_NATIVE/AGENT_NATIVE/ROOM_REPAIR_NATIVE with identical rendering settings, task/robot config and resets. Their changed asset bundle includes visible geometry/materials, so CLOSED-LOOP differences are not automatically pure physics effects. Fixed-action replay supplies a perception-independent physical diagnostic.

For observation effects compare ROOM_REPAIR_NATIVE/GS/HC with identical compiled physics. For REF_NATIVE vs full reconstruction, explicitly declare a JOINT end-to-end system contrast rather than fraudulently passing it as one-axis. Version the typed validator to support this labeled joint contrast; do not disable existing drift validation.

## 7. Policy and action contract

Use a RoboCasa-compatible frozen checkpoint/config/stats. Do not reuse a DROID policy merely because its network name is pi0.5. Keep the native robot and all proprioceptive channels. Bypass low-dimensional object state/GT embeddings unless the selected baseline explicitly requires them, in which case label it state-based. No policy training on reconstruction TEST scenes. A second checkpoint is a separate block.

Bind policy RNG to scene/task/reset/chunk and separate warmup streams. Exogenous randomness is paired; sampled actions and contact states may diverge after differing observations. Matching means the same reset INTERVENTION and known contract, not identical reconstructed qpos after settling. Physics differences caused by assets are outcomes.

Fixed-action replay executes the identical recorded robot control sequence through both native controllers. Never teleport objects, replay reference object poses, force grasps, or apply different safety clamps. State-based controllers and semantic primitives are engineering diagnostics, not visual-policy results. Assisted/sticky grasping, if required by a checkpoint, must be identical and explicitly reported; do not change grasping mode to improve the reconstructed arm.

## 8. Counting, freezing and failure handling

A planned unit is (platform, scene instance, task instance, policy checkpoint, build method, reset, treatment, execution kind). Unique IDs include all axes. Shard workers write separate ledgers; one canonical merger rejects duplicates and validates expected units. Successful resume reuses only hash-identical terminal units. Source change creates a new stage; previous failures stay in their original stage.

Record `completed_success`, `completed_task_failure`, `construction_unavailable`, `verification_rejected`, `role_binding_failure`, `initialization_failure`, `policy_timeout`, `environment_crash`, `enhancer_failure`, `blocked_external`, and `not_scheduled`. Stop reasons are explicit; do not call a failed prerequisite an executed policy trial.

Report rollout coverage and conditional success, plus end-to-end successes/planned tasks where unavailable constructed environments count as inability to serve that planned task. Unmeasured external/not-scheduled cells are not silently zeros: table-level scope remains incomplete until resolved or the planned claim is narrowed. Publish evaluable counts beside gaps/RMSE. Do not use only common successful builds as the population-wide result.

All method tuning occurs on development scenes. Bug fixes after TEST inspection require a new version and explicit reuse/retest policy. No post-hoc threshold relaxation, favorable-reset replacement or repeated significance hunting. Negative results remain visible.
