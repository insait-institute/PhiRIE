# Changelog

## Unreleased — open-source cleanup

- Remove all Slurm launchers (`run/slurm`, `run/icra2027`, `run/campaign`,
  `run/finalize`, `run/roundtrip`, `run/sim_recon_sim`, `run/demo`,
  `run/harness`) and the paper-campaign and experiment layer: `robo/campaign`,
  `robo/roundtrip`, `robo/certification`, `robo/polaris`, the paired evaluation
  harness in `robo/eval`, `agents/orchestrator`, the paper-only evaluation
  scripts (result aggregation, paper tables, coverage, bootstrap and power
  analysis), `tools/demo`, the `experiments`, `oracle` and `baselines`
  top-level directories, `configs/experiments`, `configs/demo`,
  `configs/droid`, `configs/oracle`, `configs/polaris`, `configs/task_graph`
  and the campaign-related documentation pages.
- Remove `tests/` and `checkpoints/` from the tree (the weights notes moved to
  the README); the CI workflow now runs ruff, `tools/release/verify.py`,
  `run/phiroom.sh modules` and `uv build`.
- Merge `models/` into `agents/models/`, `simfactory/` into `robo/simfactory/`,
  `integrations/` into `tools/` (`tools/phiview`, `tools/harmonizer`) and
  `pipelines/` into `phiroom/pipelines/`; the `experiments` module is renamed
  `construction` (actions `factory`, `construct-auto`, `construct-gt`) and the
  `paper` module is removed, leaving 15 module contracts.
- Add the MIT `LICENSE`; PhiView is bundled under `tools/phiview/` with
  `SOURCE_PROVENANCE.json` instead of a Git submodule, so anonymous clones need
  no extra step.
- Remove internal material from the tree: experiment output snapshots
  (`outputs/`), agent task plans (`plan/`), coding-agent memory notes, CI
  receipts, paper sources and planning notes, website delivery reports, and
  internal campaign workflows.
- Replace cluster-specific absolute paths and user names with environment
  variables (`SIMANY_ROOT`, `SIMANY_*_PY`, `OPENPI_DATA_HOME`, `HF_HOME`) and
  portable defaults.
- Move the README images and recordings to `media/`; remove the project-page
  source (`website/`) and its deployment workflow, which are maintained separately.
- Remove `AUTHORS.md`, `AUTHORS.json`, `RELEASE_PLAN.md` and `RELEASE_SOURCE.json`;
  rename `requirements-harness.txt` to `requirements.txt`.

## 2.0.1 — 2026-09-21

PhiRIE naming and release metadata; main-branch consolidation with recoverable
historical refs; published generator comparison; consolidated workspace path fixes;
locked CPU regression dependencies; optional OpenPI imports; isolated test scratch
and portable socket tests; read-only CI reporting and repository ignore cleanup.

See [the release report](docs/releases/v2.0.1.md) for validation and limitations.

## 2.0.0 — 2026-09-14

- Release the existing PhiRoom/SimAny main-branch system under version 2.0.0.
- Add PhiView as a Git submodule at `tools/phiview`, pinned to PhysicalView
  commit `77d5999a7d659b55f6f214685923ccf51e6cfbd7` (v0.2.0).
- Document recursive checkout, PhiView's four locked uv profiles, compatible backend
  setup, and a root-level `run/phiview.sh` command.
- Preserve the existing construction, simulator, evaluation and final-experiment APIs.

This software release does not certify experimental completion or paper-quality results.
See [release details](docs/releases/v2.0.0.md).

### Modular feature interfaces

- Added 16 feature contracts and separate pipeline calls through `phiroom`.
- Split SimFactory adapters by feature while preserving existing backend entry points.
- Added locked uv control environment, composable recipes, operational receipts,
  contract tests, feature branches and contribution documentation.
