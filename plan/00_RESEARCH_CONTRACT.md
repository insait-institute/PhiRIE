# Task 00 — Freeze the Research Contract and Go/No-Go Gates

**Priority:** P0  
**Suggested owner:** senior research lead  
**Depends on:** none  
**Blocks:** every experiment task

## Objective
Create one versioned experiment specification that fixes the paper thesis, evidence tracks, baselines, metric definitions, validity criteria, and fallback claims before expensive rollouts begin.

## Why this matters
Without a frozen contract, each agent will optimize a different story and pooled correlations will be vulnerable to post-hoc task, policy, or seed selection. This task protects the central claim: a phone scan becomes a predictive robot simulator rather than merely a visually plausible scene.

## Read first
- `docs/EXPERIMENT_DESIGN.md`: current audited results and missing evidence.
- `docs/CONTRIBUTIONS.md`: claims already supported by ScanNet++.
- `docs/ROBOT.md`: current robot-layer limitations and variance.
- `run/run_video2sim.sh`, `run/run_behavior_recon.sh`, `run/run_droid_recon.sh`: existing evidence paths.

## Outputs
- `configs/experiments/icra_contract_v1.yaml`
- `docs/ICRA_RESEARCH_CONTRACT.md`
- `configs/experiments/frozen_fields.yaml`

The YAML must enumerate protocols (`scannet_scale`, `oracle_reconstruction`, `polaris_paired`, `droid_paired`, `phone_rooms`), admissible baselines, primary/secondary metrics, task/policy lists, reset distributions, episode budgets, confidence intervals, and go/no-go thresholds.

## Implementation steps
1. Write one-sentence primary and fallback theses.
2. Define which dataset answers which question; prohibit cross-protocol numerical comparisons.
3. Freeze baseline eligibility: official release, partial reproduction, or literature-only.
4. Define G1–G6 gates and the exact action when a gate fails.
5. Add a schema validator and CI test that rejects unknown fields or mutable defaults.
6. Add a change log requiring rationale and author for any post-freeze modification.

## Tests
- `python -m agents.eval.validate_contract configs/experiments/icra_contract_v1.yaml`
- Deliberately alter a frozen camera field and verify validation fails.
- Verify every planned table column maps to a declared metric.

## Acceptance criteria
- [ ] Every experiment has one protocol, independent variable, frozen fields, and primary metric.
- [ ] Every headline claim has a quantitative go/no-go threshold.
- [ ] No result can be silently excluded after execution.
- [ ] Contract is reviewed by at least one person not implementing the runner.

## Stop conditions
Stop if the real-result checkpoint identities for PolaRiS cannot be matched to public bytes; the paired protocol must then be narrowed before any policy matrix is run.

## Paper artifact unlocked
The Experimental Design section and the credibility of all main tables.
