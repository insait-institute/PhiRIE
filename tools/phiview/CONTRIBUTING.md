# Contributing

Start from current `main` in a clean worktree. Each block has a documented input/output
contract in `pipelines/<block>/README.md` and a machine-readable manifest in
`physicalview/resources/pipelines/`. Keep both current when changing a public call.

```bash
git fetch origin
git worktree add -b feature/viewer-my-change ../PhysicalView-viewer origin/main
cd ../PhysicalView-viewer
uv sync --locked
uv run physicalview blocks
uv run pytest -q
```

Use `feature/<block>-<topic>` for bounded changes. Retained `block/<block>` branches mark
integrated contributions; they are not independent deployment versions. Shared config and
environments belong to runtime; packaging and CI belong to release. All blocks run from main.

1. Describe the concrete behavior and the relevant block in a pull request.
2. Preserve experiment outputs, existing worktrees and scientific provenance. Put generated
   files in `outputs/`; never commit data, model caches, credentials or environment folders.
3. Use `physicalview plan BLOCK` to review pipeline changes. Avoid hidden fallback models.
   Preserve errors, commands, input identities and artifact checks in job receipts.
4. Run tests appropriate to the change. CPU CI skips backend/GPU-only tests. Rendering
   changes need an allocated GPU check; browser changes need an actual browser check.
   A CPU fixture cannot validate inference or image quality.
5. Regenerate and install affected lockfiles with `uv sync --locked` after dependency changes.
   Record native build/toolkit requirements separately.
6. Merge with a merge commit after review/checks to retain branch provenance. Fetch before
   merging and require the reviewed head SHA. Never force-push main or delete unmerged work.

```bash
uv run pytest -q
uv build
uv run python tools/validation/check_release.py
# With the backend configured:
SIMANY_ROOT=/path/to/PhiRoom uv run pytest -q
# On an allocated GPU with studio, EGL and CUDA configured:
SIMANY_ROOT=/path/to/PhiRoom MUJOCO_GL=egl envs/studio/.venv/bin/python -m pytest --run-gpu -q tests/test_render.py tests/test_phiview.py
```

Screenshots retain unmodified PNGs and sidecars. Caption GT assistance, source resolution,
model choices and scripted control honestly. Publication approval is a separate review
state and must be cleared when frames are replaced.
