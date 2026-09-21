"""Formal release/status contract for scene-construction baselines (Task 15).

Distinct from `agents/baselines/` (the working baseline *implementations*
already in this repo: MaskClustering, FlashSplat, and the SimFoundry-
reproduction launcher `run/run_simfoundry.sh`). That package is not moved or
renamed by this task.

This package is the audit layer plan/15_CONSTRUCTION_BASELINES.md asks for:

- `release_status.yaml` -- one row per named baseline recording what is
  really runnable (`implemented`), runnable-with-caveats (`partial`),
  citable-but-not-runnable (`literature_only`), or simply not found
  (`unavailable`). See `tests/test_baseline_release_status.py`.
- `raw_reconstruction.py` -- the one genuinely new runnable baseline this
  task adds: the fused-reconstruction, no-object-factorization ablation.
- `simfoundry_repro.py` -- makes the existing ScanNet++-only SimFoundry-
  reproduction ablation (`run/run_simfoundry.sh`, ablation row D) callable
  against the `recon_scenes` layout the oracle/DROID predictive tracks use.
  Still glue over the same shared pipeline stages, still `partial` status;
  only the reachable scene layout changed (added 2026-08-30).
- `simrecon_adapter.py` / `replicate_adapter.py` -- inert stubs. Both
  baselines have no public runnable release as of this audit, so both
  modules raise `NotImplementedError` on every call; nothing here can
  silently fabricate a number for an unreleased method.

See `docs/BASELINE_REPRODUCTION.md` for exact reproduction commands and
`plan/15_CONSTRUCTION_BASELINES.md` for the task contract this satisfies.
"""
