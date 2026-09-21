# SR0 — Native reference, policy compatibility and identity controls

**Owner:** native-environment agent. **Priority:** P0. Read ../PROTOCOL.md and ../SOURCES.md. Deliver a working reference before constructing a fleet. This README specifies new implementation, not evidence of completed native integration.

## Reuse and inspect

Read `robo/eval/harness_runner.py`, `harness_spec.py`, `harness_validation.py`, `episode_log.py`, `robo/policy/`, and `robo/manifest/`. The current harness types are limited to existing SimAny scene/collision/observation variants and its environment construction is DROID-specific. A RoboCasa config cannot simply be passed through unchanged. Reuse provenance/reset/ledger logic, but put native environment/controller/scorer behavior behind an adapter.

## Implementation scope

Add a small adapter package, with names adjusted only if the repository already has an equivalent:

```text
robo/roundtrip/__init__.py
robo/roundtrip/adapters/base.py
robo/roundtrip/adapters/robocasa.py
robo/roundtrip/reference.py
robo/roundtrip/spec.py
configs/experiments/sim_recon_sim/reference.yaml
run/sim_recon_sim/reference.sh
tests/test_roundtrip_adapter.py
```

Required adapter interface: `load_instance`, `reset_from_spec`, `step_native_action`, `get_policy_observation`, `render_capture`, `native_success`, `native_stage_state`, `get_named_body_state`, `export_reference_for_evaluator`, `close`. Policy cameras and action semantics are discovered from the compatible native policy runner, not hardcoded to DROID keys or 224x224. Put an integration shim behind the canonical harness; do not fork a second independent success/coverage aggregator.

The reference/evaluator methods are privileged and must not be available inside a constructor worker.

## Environment and data setup

1. Inventory installed RoboCasa/robosuite/MuJoCo versions, GPU rendering support and checkpoint directories. Record exact checkout commits and asset release hash. Follow the official installation docs in SOURCES.md in an isolated environment, not the generation environment.
2. Discover exact rigid native task IDs, scene layout/style seeds and object instances. Record required fixtures/goals and controller/robot. Do not invent task names from the plan's generic task families.
3. Choose a publicly released compatible policy, initially the official RoboCasa pi0.5 path if it runs cleanly. Pin model bytes, training config, normalization statistics, action horizon, observation order and inference code. The official horizon changed in RoboCasa 1.0.1; use one coherent version and its native per-task horizon everywhere.
4. Download only assets/checkpoint and optional demonstrations required for the pilot. Training from scratch and downloading every demonstration are not prerequisites. If pi0.5 access fails, use another genuinely compatible official checkpoint and name it accurately.
5. Confirm policy input channels: RGB/proprioception only for visual-policy claims. Store action and observation schema before inference.

## First development run

Load one native task; prove initial success is false, reset repeats and grasp mode is fixed. Run the policy in the untouched environment. Preserve every pilot outcome. At least one genuine native success is needed before claiming the policy stack is operational. A recorded demonstration/action sequence can validate dynamics, but cannot replace this policy check. If no success occurs, inspect controller/action scaling, cameras, normalization and task horizon. Do not diagnose reconstruction before the original interface works.

Prepare two development layouts with two rigid tasks each, five resets each. TEST roster is separately frozen after interface debugging, based on native task capability rather than reconstructed or TEST-policy success. Every selected task must preserve its full native goal. An open-cabinet subtask is custom if the original task requires opening the cabinet.

## Identity gates

- U0: untouched native environment through adapter. Match initial images/state and native predicate answers. Replay a short fixed action sequence twice. Record tolerances measured on reference repeatability rather than assuming cross-machine bit equality.
- U1: lossless import/re-export of reference assets in the evaluator. Check mesh scale/orientation, collision placement, mass/inertia, contact filters and material. Native robot/controller/camera and task functions remain the same. Compare a fixed-action trace and the original trace; the adapter must not cause a large task change by itself.
- Unit fixtures: asymmetric mesh with offset COM; two cameras with different K; object behind robot and robot behind object; open box cavity. Wrong quaternion order and 100x scale must fail.

Reference repeatability checks are engineering controls, not a row claiming perfect reconstruction. If exact USD/MJCF lossless import is impossible, quantify the conversion baseline and label it; do not attribute conversion error to SimAnyRoom.

## Proposed commands, implement and test before use

```bash
python -m robo.roundtrip.reference --config configs/experiments/sim_recon_sim/reference.yaml --phase inventory --out "$OUT/reference/inventory"
python -m robo.roundtrip.reference --config configs/experiments/sim_recon_sim/reference.yaml --phase identity --out "$OUT/reference/identity"
python -m robo.roundtrip.reference --config configs/experiments/sim_recon_sim/reference.yaml --phase pilot --out "$OUT/reference/pilot"
```

`$OUT` is a new stage's sim_recon_sim output root. Each phase refuses overwrite. `--help` and synthetic schema tests run without initializing graphics.

## Deliverables and acceptance

`environment_lock.json`, `policy_lock.json`, `available_tasks.json`, `dev_roster.json`, proposed `test_roster.json`, `identity_report.json`, reference reset bank, native videos/actions/states, and `STATUS.md`.

- [ ] One native learned-policy episode and one fixed-action trace are valid.
- [ ] U0 passes and U1 conversion effect is quantified.
- [ ] Exact native success function and body/fixture bindings are documented.
- [ ] Camera/controller/horizon versions are frozen.
- [ ] Task IDs and permissions are real, not placeholders.
- [ ] No expensive reconstruction sweep started before these checks.

If blocked, record the concrete dependency and continue CPU adapter/tests. Do not redesign the paper or reimplement its entire manifest stack.
