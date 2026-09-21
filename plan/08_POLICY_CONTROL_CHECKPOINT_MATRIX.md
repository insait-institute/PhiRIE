# Task 08 — Frozen Policy, Control, and Checkpoint Matrix

**Priority:** P0  
**Suggested owner:** policy inference engineer  
**Depends on:** Tasks 00–01, 07  
**Blocks:** Tasks 09–10, 18

## Objective
Run a non-degenerate matrix of public policies whose checkpoint bytes correspond to the real-world results being used, with identical preprocessing and control in every simulator variant.

## Candidate matrix
Start with PolaRiS/DROID-compatible π0.5, π0-FAST, π0, π0-100k, and PaliGemma-binning checkpoints. Base real-data policies and sim-co-trained policies must be separate analyses.

## Existing code
- `robo/eval/pi05_eval.py`
- `run/pi05_serve.sh`
- current openpi fork and policy clients.

## Outputs
- `robo/policy/registry.py`
- `robo/policy/control_contract.py`
- adapters under `robo/policy/clients/`
- `configs/policies/*.yaml`
- `tests/test_control_contract.py`

## Implementation steps
1. Pin checkpoint URI, resolved local hash, training config, action semantics, image preprocessing, language template, chunking, and temporal aggregation.
2. Normalize all clients to one environment-facing action schema without changing model behavior.
3. Verify absolute vs delta joints, gripper range/direction, control rate, and action clipping against source implementations.
4. Warm up models before timing and log server/client versions.
5. Add deterministic observation/action trace replay.
6. Match checkpoint identities to published real scores or label unmatched checkpoints exploratory.

## Tests
- Golden observation produces a stable action trace within numerical tolerance.
- Same open-loop trace yields identical pre-contact joint paths in official and SimAny environments.
- Intentionally swapped absolute/delta convention is detected.
- Hash mismatch prevents a run from entering the main matrix.

## Acceptance criteria
- [ ] At least three policies have verified real-result correspondence.
- [ ] No single policy is constant across all tasks.
- [ ] Every checkpoint and preprocessing asset is content-hashed.

## Paper artifact unlocked
Multi-policy paired matrix, leave-one-policy-out analysis, and ranking metrics.
