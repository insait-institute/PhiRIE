# SR7 — BEHAVIOR room-scale extension in native OmniGibson

**Owner:** OmniGibson adapter agent. **Priority:** P1 after RoboCasa vertical slice, hardware/access check immediately. Read ../PROTOCOL.md and ../SOURCES.md. This track is simulation-reference evaluation, NOT the complete BEHAVIOR challenge and NOT physical real-world validation.

## Version, hardware and data gate

Pin OmniGibson, Isaac Sim/PhysX, BDDL, dataset assets, robot and task-instance versions as one compatible stack. Official challenge baseline docs inspected 2026-09-06 recommend BEHAVIOR v3.9.1 rather than v3.9.0; confirm actual pinned capabilities before installing. Do not mix latest generated API pages with an older local release.

Use a documented compatible RTX-capable rendering machine for native scene capture and simulation. Do not assume a large H200 allocation meets Isaac renderer requirements. Run the native compatibility check/headless sample on the intended hardware. GS/generation may run separately on compute GPUs.

Download only required scene/task assets and one policy's dependencies. Full challenge demonstrations are unnecessary for native capture. Native BEHAVIOR assets are encrypted/licensed; do not decrypt for redistribution or publish original meshes. Use permitted in-process reference evaluation; if mesh export is not allowed or unavailable, keep geometry metrics unmeasured until a legitimate reference path exists. Public release defaults to configs/IDs/code/metrics and permitted videos, not raw assets.

## Task roster

Discover pre-sampled native task instances using the installed dataset. Freeze `(scene_model, activity_name, activity_definition_id, activity_instance_id, robot, grasp_mode)` and use `online_object_sampling=False` for reproducibility. These are reference setup fields, not public constructor inputs.

Start with 4 room instances x 2 rigid tasks x 10 reset interventions. Pick tasks involving object-to-surface and open-container placement with no articulation, liquid, cloth, heat, cutting or irreversible transition requirement. Retain the COMPLETE native goal. A shortened activity is a custom task, not official task success. If fewer suitable native tasks exist, report actual scope and do not invent task IDs.

## Import implementation

Add `robo/roundtrip/adapters/omnigibson.py` and `robo/roundtrip/importers/omnigibson.py` behind the SAME shared adapter/ledger contract. Use the pinned release's supported custom-USD conversion/import interface. The public docs advertise custom USD support, but concrete classes and state abilities vary by version; inspect the installed source and test before assuming a generic USDObject instantiation works.

USD requirements: meters/up-axis, baked scale once, visual/collision separation, positive mass/inertia, correct fixed/dynamic status, collision groups, convex decomposition that preserves receptacle openings, material binding, named links and object-state abilities. Record all original environment bodies removed or retained. Restart or reload physics when required for changed scale/collision properties.

Native BDDL task roles must bind to reconstructed objects after the build is frozen. For `Inside`, `OnTop`, `Touching`, etc., derive geometric metadata from reconstructed assets and check native predicates on positive/negative fixtures. Renaming a USD body to an original name does not restore the original object's abilities or cavity. Missing abilities are `task_binding_unsupported`, never silently skipped goals. Any indispensable reference semantic metadata copied into an evaluation adapter is declared, and oracle geometric metadata is a separate diagnostic condition.

## Same-engine controls

Run U0 native wrapper identity, then U1 unchanged-asset conversion/import in the privileged evaluator. Keep original robot/controller/action space/cameras/task evaluator and grasping mode. Physical, assisted and sticky grasping are different conditions. Preserve a checkpoint's native mode for comparability; physical-grasp diagnosis can be separate. No original collision or hidden attachment may remain behind a replaced object.

Do NOT use the existing `run/run_behavior_recon.sh` output MJCF as the reconstructed main arm against OmniGibson. That script is a useful input-contract prototype, not a same-engine reference adapter. Its GT-depth initialization and multi-episode mixing also require SR1's corrections.

## Policies and completion tiers

The inspected official baseline page lists pi0.5 and GR00T N1.7 training/evaluation support, but provided fine-tuned checkpoints are explicitly listed for `turning_on_radio`. Do not assume they solve the proposed rigid task set or silently substitute a Fetch policy for R1Pro.

Tier A: native captures, GS/asset reconstruction, custom USD import, verified task semantics and fixed-action replay. This is valuable even without a general visuomotor checkpoint.

Tier B: reference and reconstructed closed-loop on a compatible released checkpoint for the actual robot/tasks. Baseline policy MUST run natively first. If absent, either record NOT_RUN or fine-tune on a separately frozen native TRAIN split and report that training. Do not start a multi-day policy-training project before the import pilot works.

Use native semantic action primitives only as `state_based_diagnostic`; never call them VLA success. BEHAVIOR closed-loop tables remain separate from RoboCasa because robots, tasks and engines differ.

## Proposed commands and outputs

Use the same `robo.roundtrip.reference/capture/build/native_import/replay/policy` frontends with `backend: omnigibson` in a new platform config. These are proposed interfaces to implement, not existing functionality. Export environment lock, task roster, capture manifests, USD import report, predicate tests, fixed-action traces, coverage, native-goal success/partial completion where available, and videos.

## Acceptance

- [ ] Native engine/robot/task/policy versions verified on actual hardware.
- [ ] One unchanged native instance can be saved/restored and identity-imported.
- [ ] One reconstructed object replaces both original visual and collision geometry.
- [ ] Open-container/OnTop predicates use reconstructed geometry and pass synthetic truth fixtures.
- [ ] Same actions run through both native engines without pose playback.
- [ ] Full planned roster, build/import/binding failures and restricted references are documented.
- [ ] Any learned-policy result has a truly compatible frozen checkpoint.

Handoff no fabricated native task success, no cross-engine attribution, and no claim that replacing only the workspace reconstructs the whole room.
