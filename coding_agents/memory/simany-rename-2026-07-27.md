---
name: simany-rename-2026-07-27
description: "SimFoundry -> SimAny repo rename (2026-07-27): new paths, package imports, SIMANY_* env vars, and what it silently broke"
metadata: 
  node_type: memory
  type: project
  originSessionId: 557e99ab-bcd8-44e5-8f1b-22d3105d0258
  modified: 2026-08-01T00:13:20.177Z
---

On 2026-07-27 ~15:02-15:11 UTC the repo was renamed **SimFoundry -> SimAny**
(github.com/RunyiYang/SimAny) and simultaneously refactored from flat scripts
into a real package. A compat symlink `/group/worldcept/code/SimFoundry ->
SimAny` exists, so old absolute paths to the repo ROOT still resolve — but
paths that reach *into* `simfoundry/` do NOT.

**New layout** (`/group/worldcept/code/SimAny/`):
- `simfoundry/` -> `simany/`, split into subpackages: `simany/robot/`
  (pi05_env, pi05_eval, pi05_render, pi05_tasks, pi05_rig), `simany/core/`
  (common.py), `simany/sim/` (export_mjcf, export_omnigibson,
  omnigibson_bridge/), plus assets/ baselines/ discover/ edit/ eval/ render/
  single_image/ tests/ viz/.
- `*.sbatch` -> `scripts/slurm/`, shell scripts -> `scripts/`
  (e.g. `scripts/pi05_serve.sh`).

**Three things it silently broke** (all hit while fixing the pi0.5 job):
1. Flat imports became package imports (`from simany.robot import pi05_env`),
   but **`simany` is NOT pip-installed into `.venv`** (pyproject deps are
   deliberately empty — three incompatible torch envs). So you must either
   `export PYTHONPATH=/group/worldcept/code/SimAny` or run `python -m
   simany.robot.pi05_eval`. Plain `python simany/robot/pi05_eval.py` fails.
2. Env vars `SIMF_*` -> `SIMANY_*` (SIMANY_SCENE/SIMANY_OUT/SIMANY_AUTO/
   SIMANY_MESH_SRC). `simany/core/common.py` still honours the `SIMF_` prefix
   "for one release" with a stderr deprecation warning — do not rely on it.
3. ~~A sed pass left `cd $ROOT/simfoundry` in the sbatch files.~~
   **FIXED 2026-08-01**: all 13 `scripts/slurm/*.sbatch` now `source
   $SIMANY_ROOT/scripts/env.sh` (which cds to the repo root) and invoke stages
   via the `run` / `run_sam3` / `run_gs` / `run_qwen` helpers. The last two
   stragglers were `hybrid_pilot.sbatch` (no env.sh at all) and
   `mc_s12_rerun.sbatch` (cd'd into third_party/MaskClustering, which broke
   the following `-m simany.*` call; now a subshell). All pass `bash -n`.

**Verification harness** (reuse it before trusting any future refactor):
`smoke_imports.sh` imports every module and diffs against a pre-refactor
baseline. Post-refactor: 56 pass / 4 fail, where all 4 are environmental
(`viser` x2 = same as baseline, `omnigibson` x1, one argparse-guarded script).
It caught two real regressions the rename introduced: a `C.env` that had to be
`_C.env` in discover/auto_segment.py, and behavior1k_coverage.py's repo-root
depth shifting from parents[1] to parents[2].

**Do not mistake this for data loss**: an audit agent reported "every *.log
under simfoundry/ was deleted at 15:04" — that was the rename in progress, not
destruction. Everything survived under the new names.

**Why the rename**: arXiv:2606.28276 is *itself titled* "SimFoundry"
(NVIDIA/Stanford/GaTech/UT/Toronto) — the very system this repo began as a
reproduction of and cites as prior work. In the paper and code, "SimFoundry"
now means ONLY that prior system; `\cite{simfoundry}` and its four prose
mentions in root.tex must never be renamed. `\name` in root.tex is SimAny
(was PhiRoom). The one surviving mention in the package is a docstring in
`simany/discover/s0_select_frame.py`, now carrying the arXiv id inline.

Docs written 2026-08-01: `docs/CONTRIBUTIONS.md`, `docs/BASELINES.md`
(rebuilt), `docs/PAPER_REVISIONS.md`, `docs/ENVIRONMENTS.md`. Report artifact:
https://claude.ai/code/artifact/65cae5fc-86a0-4fb4-abe7-4a5cf0285cdc

Related: [[simfoundry-repro-status]], [[pi05-droid-sim-eval]],
[[simany-baseline-positioning]].
