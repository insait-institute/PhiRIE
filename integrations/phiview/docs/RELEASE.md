# Release 0.2.0

## Installation and support boundary

Python 3.11/3.12 CPU development uses the root `uv.lock`. GPU profiles use Linux x86_64,
Python 3.11 and separate lockfiles in `envs/`. All four profiles were installed with
`uv sync --locked` on 2026-09-14. The generation profile needed a retry with
`UV_HTTP_TIMEOUT=180` for a large CUDA wheel; the retry completed successfully.

The studio profile was tested with PyTorch 2.9.1+cu128, gsplat 1.5.3, MuJoCo 3.13.0 and
CUDA toolkit 12.8.1. Inference/generation installation establishes Python dependency
resolution, not end-to-end model operation. Download/access weights separately; TRELLIS
native extensions and upstream SAM3D/OpenPI server/Isaac Sim environments have additional
requirements described in [envs/README.md](../envs/README.md).

## Recorded checks

- Portable CPU suite and backend-enabled suite: see the committed validation summary
  and test receipts in [releases/0.2.0](releases/0.2.0).
- H200 job **896677**, `sof1-h200-3`: all seven critical self-test groups passed. A fresh
  gsplat extension compiled for sm_90; the real scene loaded **1,499,998 SH-3 Gaussians**
  and rendered 640×360. MuJoCo EGL, OpenPI client and backend imports passed.
- Explicit GPU tests: **17 passed, 1 skipped** (`tests/test_render.py tests/test_phiview.py`).
- [gpu-896677.json](releases/0.2.0/gpu-896677.json) and its log retain exact versions,
  dimensions, timings and paths. This is a renderer/environment check, not a new browser
  validation, model-quality evaluation or robot success benchmark.
- Package checks verify the wheel includes browser/config/block resources and the source
  distribution includes all four lockfiles. An isolated wheel install exercises the CLI.

The GPU run used the integrated tree at `6b555894cbcb7c7127ebb17af8b3012a911dda34` before
release-only CLI/docs/packaging fixes. Renderer and simulation code were unchanged after
that run; their hashes are recorded in the release validation summary.

## Integration safety

The original checkout and its untracked work were preserved. Integration used an isolated
worktree and a pre-merge Git bundle. All three contribution branches merged without
conflicts; each original head is retained in the branch audit. Ten block branches and the
shared runtime/release branches remain available. SimAny/PhiRoom is a separately pinned
dependency and was not merged into this repository's history.

GitHub CI is configured to run CPU tests, check all lockfiles, build both distributions
and test an isolated wheel install. Backend/GPU-only skips are explicit. Preserve the
original branch histories with a merge commit after validating the integration head.

For this release, GitHub run `34845525682` stopped before any steps ran. Its annotation
states: **"The job was not started because an Actions budget is preventing further use."**
Both Python 3.11 and 3.12 suites and the package/install checks were therefore run locally.
The user-authorized integration proceeds on those recorded checks. GitHub CI is blocked,
not passing; account budget settings were not changed. Restore Actions capacity to run
the hosted checks and tag automation. The annotation is retained with the release evidence.

## Creating a release

1. Update version in the package and all environment projects, regenerate affected locks,
   update CHANGELOG and record validation. Keep source data/model weights out of archives.
2. Run `uv run pytest -q`, `uv build` and `uv run python tools/validation/check_release.py`.
   Perform the relevant backend/GPU/browser checks for the actual change.
3. Merge the reviewed PR into main after CI. Fetch and verify all intended branch heads
   are ancestors of main. Do not force-push or delete outstanding work.
4. Tag that main commit (for this version: `git tag -a v0.2.0 -m 'PhysicalView 0.2.0'`)
   and push the tag. The Draft release workflow reruns CI and verifies version/main ancestry.
5. The workflow creates a **draft** GitHub release with wheel, source distribution and
   SHA256SUMS. Review artifacts before publishing the draft. PyPI publication is not configured.

Release automation does not change repository visibility, distribute dataset/model assets,
or certify that every demo screenshot has passed publication review.

When hosted Actions cannot start, prepare the same checked wheel/source distributions
locally from the clean tagged main checkout. `uv run python tools/release/draft.py --tag
v0.2.0` prints the draft and asset hashes. Add `--create` to create/upload the draft using
`GH_TOKEN`, `GITHUB_TOKEN` or the configured GitHub HTTPS credential helper. It refuses
to overwrite a published release or an asset with a different/unverifiable digest.
