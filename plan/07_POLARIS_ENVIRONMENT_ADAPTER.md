# Task 07 — PolaRiS Import, Reconstruction, and Fair Environment Adapter

**Priority:** P0  
**Suggested owner:** Isaac Lab / PolaRiS engineer  
**Depends on:** Tasks 00–01, 04, 06  
**Blocks:** Tasks 08–10, 18

## Objective
Create two environments per selected PolaRiS task: the official manually composed environment and a SimAny reconstruction, while sharing the exact robot, camera, control, reset, language, horizon, and rubric contract.

## Existing/reference code
- PolaRiS public repository and environment definitions.
- `robo/sim/export_omnigibson.py` and planned USD exporter.
- `run/run_video2sim.sh` for reconstruction.

## Outputs
- `robo/polaris/import_official.py`
- `robo/polaris/export_simany.py`
- `robo/polaris/task_adapter.py`
- `robo/polaris/validate_pair.py`
- `configs/polaris/tasks/*.yaml`

## Implementation steps
1. Vendor/pin the exact PolaRiS commit and asset release; record licenses.
2. Extract robot USD, cameras, task language, initial conditions, rubric, and controller settings into the common manifest.
3. Obtain the corresponding capture inputs when public; otherwise document and reproduce an allowed capture route without reading hidden evaluation values.
4. Export SimAny objects/background/collision into a separate scene layer while referencing the same robot/task layers.
5. Validate camera images, robot zero pose, action units, gripper convention, control rate, and rubric events.
6. Generate a machine-readable diff proving only the scene-construction method differs.

## Tests
- Official environment reproduces its public smoke-test score/video.
- SimAny and official environments share identical frozen-field hashes.
- A scripted action produces matching robot joint trajectories in both scenes before contact.
- Rubric unit tests fire on hand-constructed success/failure states.

## Acceptance criteria
- [ ] At least four tasks instantiate in both environments.
- [ ] Pair validator reports no undeclared differences.
- [ ] Manual human minutes for official construction are logged using a predefined protocol.

## Paper artifact unlocked
Primary PolaRiS comparison rows and manual-time claim.
