# F1 — Candidate diversity, selection and retry in native manipulation

Owner: native-experiment agent. Code: `robo.roundtrip.mechanism_followup`; integration extends existing `spec`, `local_policy_instance`, `native_scale_tables`. No second policy runner or metric implementation.

## Hypotheses and fixed comparisons

B1−B0 measures adding the original second proposal with fixed priority. B2−B1 measures evidence-conditioned choice on the same initial candidate pool. B3−B2 measures the existing registration retry. B3−B0 remains the complete bundle contrast. Additional runtime is not assumed equal; original stage costs and actual native inference costs must be reported separately.

Methods are exactly `REF_NATIVE`, `B0_FIXED_NATIVE`, `B1_FIXED_PRIORITY`, `B2_EVIDENCE`, `B3_AGENT_NATIVE`. B1/B2 select stored initial proposal IDs, not regenerated or TEST-optimized candidates. Initial pool identity, capture identity, original selected mesh/transform/physics/collision hashes are verified. Near-threshold nonconvex collision is handled by the unchanged native import predicate, with the original asset and failure retained, never repaired in place.

## Execution

Follow parent README. One ordinary job per canonical instance, ten resets and five methods per instance on a fresh per-canonical policy engine. No old engine REF reuse. Five-arm planned count is 2,400; actual episode count excludes method build failures. Do not drop the reference failures or difficult instances. First launch one block as an integration pilot, without modifying any method based on observed TEST success. Then release bounded ordinary jobs under quota.

`prepare` creates the new plan, method configs, fresh worker roots, exact commands and a sealed input manifest. `launch` defaults to dry-run, rejects changed source/input files and records submission intents before `sbatch`. `collect` uses the existing canonical collector and prepares an existing `paper_pipeline` configuration.

## Required checks

- Native package/checkpoint, all method-observation inputs, robot/controller, horizons and resets unchanged.
- Every B1/B2 candidate exists in the original initial pool and matches the canonical capture.
- Actual native engine preflight passes; B1/B2 are admitted ONLY under the new versioned follow-up protocol.
- 48 original instances remain in the roster; no selection from favorable episode outcomes.
- One terminal record per planned unit, with unavailable engineering outcomes not silently scored as failures.
- Cluster-level paired intervals and discordant outcomes accompany component differences.

## Interpretation

This is an additional analysis of the already inspected cohort. Label it as a mechanism follow-up. Do not state that this new design was predeclared before the original TEST. It can clarify the frozen mechanism, but cannot support tuning on the same scenes. A new independent cohort requires another prospective acquisition/roster protocol, not just a renamed ID.

Deliver commands, input/output hashes, job IDs, actual planned/executed/completed counts and generated contrasts. A null or negative component difference is a valid result. Do not wait for real robots or Harmonizer.
