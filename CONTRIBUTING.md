# Contributing to PhiRIE

Start with [module ownership and interfaces](docs/MODULES.md). Each feature has a
contract in `phiroom/modules/`, a pipeline guide in `pipelines/<feature>/`, and an
explicit implementation owner. Share orchestration contracts in `phiroom/core/`;
keep models and optional GPU imports inside the owning backend runtime.

1. Keep `main` as the only long-lived branch. Use a detached worktree for local
   preparation, or a fork for a pull request. Historical branch tips are preserved
   in the v2.0.1 history bundle rather than retained as development branches.
2. Change one feature and its contract together. Keep existing backend import paths
   compatible; document changes to native flags and artifact formats.
3. Inspect the planned commands and run the relevant CPU tests. GPU behavior changes
   additionally need a bounded real backend check with source, runtime, data and output
   provenance; a fixture or successful command is not a benchmark result.
4. Open a PR with the trigger, resulting behavior, validation commands and limitations.
   Merge the exact reviewed head after checking that `main` has not changed. Preserve
   feature history; never rewrite another contributor's branch or dirty worktree.

```bash
uv sync --project envs/control --locked
uv run --project envs/control --locked pytest -q tests/test_module_interfaces.py
bash run/phiroom.sh pipeline run pipelines/preflight.json
uv build

# Optional full CPU regression dependencies:
uv sync --project envs/control --locked --group regression
MUJOCO_GL=disable envs/control/.venv/bin/python -m pytest -q tests --basetemp=.tmp/pytest-regression
```

Add new modules by declaring actions, runtimes, inputs, outputs and tool dependencies
in one contract JSON and adding its pipeline guide. Unknown action names fail before
execution. Do not place shell command strings or credentials in a recipe: arguments
are arrays, and credentials remain in the process environment.

PhiView is bundled at an immutable upstream revision. `SOURCE_PROVENANCE.json`
records every upstream file hash. Update the upstream source and manifest together
when taking a new revision; retain its license and record the upstream commit.
See [PhiView setup](docs/PHIVIEW.md).

For releases, run `tools/release/verify.py`, build the Python distributions, and create
a full source archive that includes the bundled PhiView source. Tag the exact merged main
commit. Publish the manifest, validation receipt and checksums with the archives.
Unavailable hosted CI must be reported explicitly with local evidence; it is never a
passing check. Research claims remain governed by `FINAL_EXPERIMENTS.md`.
