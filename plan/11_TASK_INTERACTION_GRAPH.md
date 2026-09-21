# Task 11 — Task-Conditioned Interaction Graph

**Priority:** P1  
**Suggested owner:** scene understanding / robotics researcher  
**Depends on:** Tasks 01, 05–08  
**Blocks:** Tasks 12–13

## Objective
Ground each task into the smallest scene subgraph that can causally affect its outcome: robot, policy cameras, manipulated object, target/receptacle, supports, tools, and swept-workspace obstacles.

## Outputs
- `robo/certification/task_graph.py`
- `robo/certification/grounding.py`
- `configs/task_graph/*.yaml`
- `tests/test_task_graph.py`

## Implementation steps
1. Parse task/rubric metadata and language into candidate roles.
2. Resolve roles to scene instance IDs using labels, geometry, and rubric references; permit explicit benchmark mappings but no evaluation-result leakage.
3. Compute support/contact edges and robot swept-workspace intersections.
4. Select camera-visible and collision-relevant objects; retain uncertainty/multiple hypotheses.
5. Serialize graph nodes, transforms, evidence, and confidence.
6. Provide a visualizer overlaying graph roles in real and simulated policy views.

## Tests
- Hand-labeled graphs for at least ten tasks.
- Distractor object of the same category is not selected solely by label.
- Distant room errors are excluded unless they intersect camera or swept workspace.
- Missing/ambiguous targets propagate uncertainty rather than selecting arbitrarily.

## Acceptance criteria
- [ ] Role accuracy and ambiguity rate are reported.
- [ ] Graph is deterministic under a frozen scene/task manifest.
- [ ] Certificate features can be computed only on graph-local evidence.

## Paper artifact unlocked
The task-locality insight and global-vs-task-local verification ablation.
