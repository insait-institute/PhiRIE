# Task 14 — Controlled Oracle Reconstruction Benchmark

**Priority:** P0/P1  
**Suggested owner:** OmniGibson / benchmark engineer  
**Depends on:** Tasks 01, 03, 06, 09–10  
**Blocks:** controlled error-attribution result

## Objective
Render phone-like captures from interactive ground-truth scenes, hide all simulator state from the reconstruction pipeline, reconstruct with SimAny/baselines, and compare static and closed-loop outcomes to the oracle.

## Existing code
- `run/run_behavior_recon.sh`
- `agents/recon/behavior_extract.py`
- OmniGibson/BEHAVIOR bridge under `robo/sim/`

## Outputs
- `oracle/capture_generator.py`
- `oracle/gt_export.py`
- `oracle/evaluate_reconstruction.py`
- `configs/oracle/tasks.yaml`
- `run/slurm/oracle_matrix.sbatch`

## Protocol
Choose diverse pick/place, receptacle, support, obstacle, transparent/thin, and clutter cases. Generate RGB or RGB-D capture trajectories with controlled view count, pose noise, blur, depth noise, and occlusion. SimAny receives only the declared capture product.

## Metrics
Object recall, pose/scale, surface F1, collision IoU, support/contact graph F1, penetration, probe response, predicate agreement, staged rollout agreement, policy ranking agreement, trajectory/contact divergence.

## Tests
- Oracle row scores exactly one under identity export.
- GT files are inaccessible to reconstruction processes.
- Controlled perturbation magnitude is recoverable from metadata.
- Frozen policy is identical between oracle and reconstructed scenes.

## Acceptance criteria
- [ ] At least ten scenes/tasks and multiple degradation levels run end to end.
- [ ] Results expose at least one geometry-better/behavior-worse counterexample.
- [ ] Error attribution uses paired seeds and reports coverage.

## Paper artifact unlocked
Oracle reconstruction table and causal link between construction errors and robot behavior.
