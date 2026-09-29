# Changelog

## Unreleased — open-source cleanup

- Add the MIT `LICENSE` and switch the PhiView submodule URL to HTTPS so
  anonymous recursive clones work.
- Remove internal material from the tree: experiment output snapshots
  (`outputs/`), agent task plans (`plan/`), coding-agent memory notes, CI
  receipts, paper sources and planning notes, website delivery reports, and
  internal campaign workflows.
- Replace cluster-specific absolute paths, user names and Slurm accounts with
  environment variables (`SIMANY_ROOT`, `SIMANY_*_PY`, `OPENPI_DATA_HOME`,
  `HF_HOME`, `SLURM_ACCOUNT`) and portable defaults.
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
- Add PhiView as a Git submodule at `integrations/phiview`, pinned to PhysicalView
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
