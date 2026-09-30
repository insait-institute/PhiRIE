# Changelog

## 2.0.0 — 2026-09-30

First public release of the ϕ-RIE pipeline code: scene reconstruction, object
discovery, asset generation, registration, background completion, physics,
simulator export, the bundled PhiView viewer and pi0.5 robot evaluation.
Compared with the internal 2.0.x snapshots:

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

## Development history (internal snapshots before the public release)

- 2026-09-21: PhiRIE naming and release metadata; main-branch consolidation;
  consolidated workspace path fixes; locked CPU control environment.
- 2026-09-14: modular pipeline with 16 feature contracts and the `phiroom`
  control CLI; PhiView pinned as a separate viewer project; SimFactory adapters
  split by feature; locked uv control environment and composable recipes.
- Earlier milestones: 0.0.0 reconstruction prototype, 1.0.0 object construction
  and simulation.
