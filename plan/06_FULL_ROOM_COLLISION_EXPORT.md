# Task 06 — Full Task-Workspace and Room Collision Export

**Priority:** P0  
**Suggested owner:** simulation / geometry engineer  
**Depends on:** Tasks 03–05  
**Blocks:** Tasks 07–09, 14, 17–18

## Objective
Replace legacy per-object support shims with collision geometry that preserves robot-room, object-room, and cross-object contacts over the task workspace in MuJoCo and Isaac/PolaRiS exports.

## Existing code
- `robo/sim/export_mjcf.py`: current support-shim limitation.
- `robo/sim/s7_sim.py`: PyBullet carved-scan collision reference.
- `agents/assets/s6_physics.py`: CoACD object collisions.
- `docs/ROBOT.md`: known mismatch.

## Outputs
- `robo/sim/room_collision.py`
- updates to `export_mjcf.py`
- `robo/sim/export_usd.py`
- `tests/test_room_collision.py`

## Implementation steps
1. Extract floor, table/support surfaces, walls, and swept-workspace obstacles from the reconstructed mesh.
2. Produce decimated static collision with bounded Hausdorff/occupancy error; use primitives for major support planes only when validated.
3. Define collision groups consistently across robot, room, manipulanda, and receptacles.
4. Preserve one common world/robot-base frame across visual and collision assets.
5. Add penetration and ray/point collision coverage metrics against the source mesh.
6. Benchmark compile time, physics speed, memory, settle drift, and contact trajectories.
7. Retain support shims only as an explicit ablation.

## Tests
- Robot sweep collides with table and nearby obstacles.
- Object pushed off its original support location continues to contact the actual room rather than falling off a private slab.
- Cross-object collision is active.
- MuJoCo and Isaac agree on a fixed set of drop/push probes within tolerance.

## Acceptance criteria
- [ ] Main policy runs use no private support shims.
- [ ] Collision coverage and penetration are reported per scene.
- [ ] 1000-step deterministic smoke tests pass for three phone/reconstructed scenes.

## Paper artifact unlocked
“Full-room collision” contribution, support-shim ablation, and contact-valid policy evaluation.
