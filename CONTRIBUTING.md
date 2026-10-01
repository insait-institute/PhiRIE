# Contributing to PhiRIE

Start with [module ownership and interfaces](docs/MODULES.md). Each feature has a
contract in `phiroom/modules/<feature>.json`, a guide in
`phiroom/pipelines/<feature>/README.md`, and an explicit implementation owner.
Shared orchestration code lives in `phiroom/core/`; models and optional GPU
imports stay inside the owning backend module.

1. Keep `main` as the only long-lived branch. Prepare changes on a branch of your
   fork and open a pull request.
2. Change one feature and its contract together. Keep existing backend import
   paths compatible; document changes to native flags and artifact formats.
3. Inspect the planned commands (`bash run/phiroom.sh plan <module> <action>`)
   and run the checks below. GPU behavior changes additionally need a real backend
   run with source, runtime, data and output provenance; a successful command is
   not a benchmark result.
4. Describe in the PR what triggers the change, the resulting behavior, the
   validation commands and the limitations. Merge the exact reviewed head after
   checking that `main` has not changed.

```bash
uv sync --project envs/control --locked
bash run/phiroom.sh modules
uv run --project envs/control --locked ruff check phiroom robo/simfactory tools/release
uv run --project envs/control --locked python tools/release/verify.py
uv build
```

These are the same checks the `module-contracts` GitHub workflow runs. Add a new
module by declaring its actions, runtimes, inputs, outputs and tool dependencies
in one contract JSON and adding its guide under `phiroom/pipelines/<feature>/`.
Unknown action names fail before execution. Do not place shell command strings
or credentials in a recipe: arguments are arrays, and credentials remain in the
process environment.

PhiView is developed upstream at
[RunyiYang/PhysicalView](https://github.com/RunyiYang/PhysicalView). Send viewer
changes there; a new revision is then re-bundled under `tools/phiview/` together
with a regenerated `SOURCE_PROVENANCE.json`, which records the upstream commit and
every file hash, and its license. `tools/release/verify.py` checks the bundle. See
[PhiView setup](docs/PHIVIEW.md).

For releases, run `tools/release/verify.py`, build the Python distributions, and
create the full source archive with `tools/release/archive.py`. Tag the exact
merged `main` commit and publish the checksums with the archives.

## Project page

The page at https://insait-institute.github.io/PhiRIE/ is built from the
`website` branch of this repository (Vite project: `index.html`, `src/`,
`public/`), so `main` carries only the released code. Every push to `website`
runs the `project-page` workflow, which builds the page and publishes it with
GitHub Pages; the workflow can also be dispatched manually.
