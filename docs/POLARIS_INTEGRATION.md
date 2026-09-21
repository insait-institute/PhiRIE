# PolaRiS integration (Task 07/08) — status as of 2026-08-16

**TL;DR: Isaac Sim's dependency install and CPU-only headless launch both
work on this cluster. Camera-enabled headless launch — which every real
PolaRiS environment needs — segfaults, due to a known upstream bug (NVIDIA
driver 595.x branch vs. Isaac Sim 5.1.0's RTX renderer). This is a
cluster-wide infrastructure issue, not a config/code problem, and it is not
fixable from inside this repo. `docs/ICRA_RESEARCH_CONTRACT.md` Decision 1
has been updated to BLOCKED; `mujoco_paired` (`robo/envs`/`robo/rigs`/
`robo/tasks`/`robo/eval`) is the permanent path for Tasks 07-08, not a
placeholder. What follows is the full repro, root cause, what was salvaged
(static task-metadata extraction that needed no live Isaac Sim), and the
exact resume plan for a future session once the cluster driver situation
changes.**

Related: `plan/07_POLARIS_ENVIRONMENT_ADAPTER.md`,
`plan/08_POLICY_CONTROL_CHECKPOINT_MATRIX.md`, `docs/ROBOT.md`,
`configs/experiments/frozen_fields.yaml`, `robo/manifest/schema.py`.

## Environment

- Repo: `third_party/PolaRiS`, untracked vendored clone of
  `github.com/arhanjain/PolaRiS` (arXiv:2512.16881), commit `129abc4`
  ("Merge pull request #22 from linbo328/fix/readme"), cloned 2026-08-14.
- Cluster: SLURM, `hala` (8x A6000, debug/batch/rendering partitions),
  `msp3-*`/`sof1-h200-*` (8x H200 each), `gcp-eu1-a100-80g-*` (8x A100-80G
  each). Isaac Sim was only exercised on `hala` (A6000) and briefly on one
  H200 node for the driver-version check below.
- `uv 0.11.31`, Python 3.11.15, `third_party/PolaRiS/.venv` (~16 GB after
  full sync).

## Stage 1 — `uv sync` (DONE)

The feasibility probe's suggested fix was half right. Two real fixes were
needed, applied to `third_party/PolaRiS/pyproject.toml`:

```toml
[tool.uv.extra-build-dependencies]
flatdict = ["setuptools<81"]
```

Bare `"setuptools"` (the probe's original suggestion) is **not** enough:
`setuptools>=81` dropped bundled `pkg_resources`, which `flatdict==4.0.1`'s
legacy `setup.py` imports at build time. `uv`'s own error message actually
suggests `pkg_resources` as the extra-build-dependency name, but there is no
installable standalone `pkg_resources` PyPI package for this to resolve to
in a build-isolation venv — pinning `setuptools<81` (which still bundles
`pkg_resources`) is the actual fix.

Second, unrelated to the papercut: this login environment leaks
`TORCH_CUDA_ARCH_LIST=9.0+PTX`, `TCNN_CUDA_ARCHITECTURES=90`, and an active
`nvidia-cuda-12.4.1` + `gcc-11.5.0` module pair (all H200-oriented — the
same class of gotcha `docs/ROBOT.md` already documents for gsplat). Two of
PolaRiS's dependencies (`diff-surfel-rasterization`, `simple-knn`) are
locally-pathed, `editable=true` CUDA extensions that either build at `uv
sync` time or JIT-compile on first import, and both need a correct,
consistent compiler/CUDA/arch setup:

```bash
eval "$(module bash-hook)"
eval "$(module bash-reset)"
module load nvidia-cuda-13.2.0 gcc-13.3.0
export TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"   # 8.6 = A6000 (sm_86); keep 9.0+PTX for H200 nodes too
unset TCNN_CUDA_ARCHITECTURES
```

`nvidia-cuda-12.4.1`'s `nvcc` refuses to compile against the system default
`gcc` (Debian's `gcc 14.2.0` — "unsupported GNU version! gcc versions later
than 13 are not supported!"); `nvidia-cuda-13.2.0` + `gcc-13.3.0` works.
Without the explicit `TORCH_CUDA_ARCH_LIST` override, the leaked `9.0+PTX`
means the compiled kernels target H100/H200 only (`compute_90`/`sm_90`),
which is silently the wrong architecture on an A6000 (`sm_86`) job.

With both fixes, from a GPU node:

```bash
cd third_party/PolaRiS
uv sync   # resolves 237 packages; builds flatdict, isaacsim, polaris; ~1 min once wheels are cached
```

completes cleanly (`isaacsim==5.1.0.0`, `isaaclab==2.3.0`, `torch==2.13.0+cu130`
all install). Verified imports on an A6000 (`srun -p debug --gpus=a6000:1
--mem=100G --time=3:00:00`) with the module/env setup above:

```
torch 2.13.0+cu130 cuda avail: True device: NVIDIA RTX A6000
diff_surfel_rasterization OK   (JIT-compiled first time, ~1 min)
simple_knn OK                  (JIT-compiled first time, ~1 min)
openpi_client OK
```

`PolaRiS-Hub` (the environment/asset dataset) also downloads fine, no GPU
needed:

```bash
cd third_party/PolaRiS
uvx hf download owhan/PolaRiS-Hub --repo-type=dataset --local-dir ./PolaRiS-Hub
# 1.7 GB: 6 task scenes (pan_clean, block_stack_kitchen, move_latte_cup,
# organize_tools, food_bussing, tape_into_container) + shared nvidia_droid
# robot/segmentation assets.
```

## Stage 2 — headless Isaac Sim launch (BLOCKED for the part that matters)

Isaac Sim's own EULA prompt needed handling first — non-interactive jobs
hang on `Do you accept the EULA? (Yes/No):` and then exit 1 on EOF. Fix:

```bash
export OMNI_KIT_ACCEPT_EULA=Y
```

(mechanism: `isaacsim/kit/kit_app.py` checks this env var, or a persisted
`isaacsim/kit/EULA_ACCEPTED` marker file, before prompting.)

With the EULA and CUDA/gcc/arch setup from Stage 1, adapting the README's
minimal example (`AppLauncher(headless=True, enable_cameras=True)` ->
`gym.make("DROID-FoodBussing", ...)`):

- **`enable_cameras=False`: SUCCEEDS.** `AppLauncher` returns in ~7.2s, no
  crash. (Isolated with a small ablation script,
  `.tmp/polaris_logs/stage2_ablate_cameras.py` — kept out of the repo
  proper since it's a throwaway diagnostic, not a deliverable; re-derive it
  from this doc if needed.)
- **`enable_cameras=True`: SEGFAULTS**, inside `AppLauncher()` itself —
  never reaches the caller's code, let alone `gym.make`. Crash backtrace
  (via Kit's own crash reporter,
  `third_party/PolaRiS/.venv/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_*.log`):

  ```
  librtx.scenedb.plugin.so!...vector<tuple<...>>::_M_realloc_insert(...)
    -> librtx.scenedb.plugin.so!carbOnPluginStartup
    -> libcarb.scenerenderer-rtx.plugin.so
    -> libomni.hydra.rtx.plugin.so
    -> libomni.usd.so!omni::usd::UsdManager::createHydraEngine
    -> libomni.usd.so!omni::usd::UsdContext::newStage()
  ```

  i.e. it crashes while creating the RTX Hydra render engine for the
  default stage, which only happens when the camera-capable renderer is
  requested.

### Root cause (confirmed, not a guess)

This is a **known, already-reported upstream bug**, not a cluster
misconfiguration on our end:

- [isaac-sim/IsaacSim discussion #648](https://github.com/isaac-sim/IsaacSim/discussions/648)
  and [issue #651](https://github.com/isaac-sim/IsaacSim/issues/651) —
  identical crash signature (`librtx.scenedb.plugin.so`,
  `carbOnPluginStartup`), reported across RTX 4070/4090/5070Ti/5080/5090 on
  both Windows and Linux.
- [NVIDIA developer forum thread](https://forums.developer.nvidia.com/t/isaac-sim-5-1-0-crashes-on-startup-with-rtx-4090-on-ubuntu-24-04-4-segfaulting-in-librtx-scenedb-plugin-so-after-iommu-was-disabled/371957) —
  same crash; IOMMU was a red herring (user disabled it, crash persisted).
- **Driver 595.x (R590 branch) is broken; Isaac Sim 5.1.0's validated/
  supported driver is 580.65.06.** An isaac-sim maintainer confirms this
  directly in #648 and says documentation improvements are coming in Isaac
  Sim 6.0 GA, with no fix for 5.1.0 on the 595.x branch.
- **No software workaround exists.** One user tried
  `--/rtx/verifyDriverVersion/enabled=false` — that flag only suppresses a
  *warning*, not the actual crash, and did not help. No alternate render
  settings, env vars, or Kit config flags are documented as fixing it.

### This is cluster-wide, not hala-specific

Checked whether a different GPU type on this cluster might carry a
compatible driver:

```
hala      (A6000): driver 595.91.07
msp3/sof1 (H200):   driver 595.71.05   <- the exact version cited as broken upstream
```

Both are on the 595.x/R590 branch. This looks like a uniform cluster-wide
driver deployment, so A100 nodes (not directly checked — the probe job
timed out waiting for a free A100 slot rather than confirming a different
driver) are very likely the same. **The fix is a driver downgrade to
580.65.06, which is a sysadmin action outside this session's authority and
would affect every other user's CUDA workloads on these shared nodes.** No
driver change was attempted or requested.

Vulkan/device enumeration itself was double-checked and is **not** the
problem — `vulkaninfo` correctly enumerates exactly one real GPU (A6000,
UUID matching `nvidia-smi` exactly) plus the harmless `llvmpipe` software
fallback; there's no multi-GPU index mismatch between CUDA and Vulkan
device numbering.

### Why this specifically blocks PolaRiS (not just "cameras are nice to have")

Checked `src/polaris/environments/manager_based_rl_splat_environment.py`:
PolaRiS's actual visual observation pipeline composites its own splat
render (`SplatRenderer`, built on the same `diff_surfel_rasterization`
kernel that already works fine standalone on this A6000) with a
**segmentation mask read from Isaac Sim's RTX `Camera` sensor**
(`base_cam.data.output["semantic_segmentation"]`) to know which pixels are
robot vs. background. So `enable_cameras=True` is load-bearing for every
real PolaRiS task, not an optional extra — there is no way to get a
PolaRiS environment to `reset()`/`step()` usefully without it.

## Stage 3/4 — not attempted

Per direction from the coordinating session: since Stage 2's camera path is
blocked, running `scripts/eval.py` (Stage 3, needs camera observations for
the policy) or writing `export_simany.py`/`task_adapter.py`/
`validate_pair.py` (Stage 4, needs a live official environment to compare
against) would either fail immediately or have to fabricate results.
Neither was attempted. This matches the contract's own stop-condition
philosophy ("narrow the claim rather than invent evidence").

## What WAS built: static task-metadata extraction (no Isaac Sim needed)

Reading `third_party/PolaRiS/src/polaris/environments/{__init__,droid_cfg,
robot_cfg}.py` (plain Python source) and
`third_party/PolaRiS/PolaRiS-Hub/<task>/{scene.usda,initial_conditions.json}`
(plain USD-ASCII text + JSON — no `pxr`/USD library needed, and none is
importable standalone from this venv anyway; `pxr` only becomes importable
after Kit's app bootstrap wires up its `sys.path`, i.e. after the exact
`AppLauncher` call that crashes) does **not** require Isaac Sim to launch at
all. This is real, checked-in-source information — not a guess — but it has
**not** been cross-validated against a live running official environment
(Task 07 step 5, "Validate camera images, robot zero pose, action units,
gripper convention, control rate, and rubric events" — exactly the part
that's blocked). Every extracted value should be read as "what the source
declares," not "what was observed running."

- `robo/polaris/__init__.py` — package docstring stating the above status.
- `robo/polaris/import_official.py` — extracts, per task: USD scene path,
  language instruction + count of initial-condition sets (100 per task) +
  one example initial pose set, camera prims declared directly in the
  task's `scene.usda` (regex-extracted position/orientation/focal
  length/aperture/derived FOV — verified correct against the nested-prim
  USD structure, see the module's own comments), the shared robot/action/
  control static facts (transcribed by hand from `robot_cfg.py`/
  `droid_cfg.py` with file:line citations — not dynamically imported,
  since those classes are `@configclass`es that assume a live IsaacLab
  process), and the per-task rubric criteria + dependency graph
  (transcribed from the `gym.register(...)` calls in `environments/
  __init__.py`).
- `configs/polaris/tasks/{food_bussing,block_stack_kitchen,pan_clean,
  move_latte_cup,organize_tools,tape_into_container}.yaml` — one manifest
  per registered PolaRiS task, generated by the script above.

Run it yourself:

```bash
.venv/bin/python -m robo.polaris.import_official --all --out-dir configs/polaris/tasks
# or a single task:
.venv/bin/python -m robo.polaris.import_official --task food_bussing --out configs/polaris/tasks/food_bussing.yaml
```

(Use the **main SimAny `.venv`**, not `third_party/PolaRiS/.venv` — the
script only needs `pyyaml`, which the PolaRiS venv's pip-less/uv-managed
environment doesn't have installed standalone, and the main venv already
has it plus this is where `robo.*` lives on `PYTHONPATH`.)

### Genuine frozen-field matches found (useful even without a live run)

Comparing the extracted official values against
`configs/experiments/frozen_fields.yaml` by hand:

| field | official PolaRiS (source) | SimAny frozen | match? |
|---|---|---|---|
| reset joint pose | `[0,-pi/5,0,-4pi/5,0,3pi/5,0]` (`robot_cfg.py`) | `[0,-pi/5,0,-4pi/5,0,3pi/5,0]` | **identical** |
| action convention | absolute joint position (`droid_cfg.py` `use_default_offset=False`) | `absolute_joint_position` | **identical** |
| action_dim | 8 (7 arm + 1 gripper) | 8 | **identical** |
| gripper binarize threshold | 0.5 (`BinaryJointPositionZeroToOneAction`) | 0.5 | **identical** |
| control rate | 15 Hz (`decimation=8`, `sim.dt=1/120`) | 15 Hz | **identical** (different substep counts: PhysX 8×1/120 vs. MuJoCo 40×1/600, same product) |
| horizon | 30 s (`episode_length_s`) | 32 s | **differs** — SimAny's 32s comes from a documented MuJoCo-suite-default gotcha in `docs/ROBOT.md`, not from PolaRiS; not currently reconciled |
| rubric structure | variable-length (3-7) dependency-graph criteria, progress = fraction ever-satisfied | fixed 4-stage {grasp,lift,hover,place}, 0.25/stage | **structurally different** — no declared mapping exists; see `note_vs_simany` in the generated YAMLs |

This is a real, useful finding independent of the render blocker: the robot/
action/control contract lines up almost exactly between PolaRiS and SimAny's
own frozen fields (both ultimately derive from the same DROID convention),
so `task_adapter.py` (when it can eventually be built) has a narrower job
than it might have — mostly reconciling the rubric and the 30s/32s horizon,
not re-deriving the whole control contract from scratch.

### What was deliberately NOT built

- `robo/polaris/export_simany.py`, `robo/polaris/task_adapter.py`,
  `robo/polaris/validate_pair.py` — all three either need a working live
  official environment to render/compare against (crashes today) or would
  have to fabricate what "matches" means without ever running anything.
  Writing them now would make Task 07 look further along than it is.
- Any manifest instantiated against `robo/manifest/schema.py`
  (`SceneBuildManifest`/`RolloutManifest`) — that schema's
  `observation_preprocessing.mode` is `Literal["raster","composite"]`
  (the MuJoCo track's two modes) and its hash fields
  (`controller_config_hash`, `camera_config_hash`, etc.) are meant to be
  real content hashes of a build that ran, not placeholders. Force-fitting
  the static PolaRiS extract into that schema would mean either fabricating
  hashes or lying about `observation_preprocessing.mode`. The extracted
  YAMLs under `configs/polaris/tasks/` are intentionally a separate,
  purpose-built format instead.

## Resume plan (if a future session picks this up)

1. **Check whether the driver situation has changed first, before
   re-attempting anything else**: `nvidia-smi --query-gpu=driver_version
   --format=csv` on `hala`. If it's `580.65.06` (or another
   Isaac-Sim-5.x-validated version, check
   [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim) release
   notes for the current validated version), re-run the Stage 2
   `enable_cameras=True` test — the exact repro is: apply the Stage 1
   `pyproject.toml`/module/env setup above, `export
   OMNI_KIT_ACCEPT_EULA=Y`, then adapt the README's minimal example with
   `args_cli.enable_cameras = True`. If it launches and `gym.make` +
   `env.reset()` + a few random `env.step()`s succeed, Stage 2 is
   unblocked — proceed to Stage 3 (serve pi0.5 via PolaRiS's vendored
   `third_party/openpi`, run `scripts/eval.py`) and then Stage 4 (the real
   adapter: `export_simany.py`, `task_adapter.py`, `validate_pair.py`,
   using `configs/polaris/tasks/*.yaml` from this pass as the starting
   point for the official-side manifest).
2. If the driver is unchanged, this remains blocked; do not re-run the
   same diagnostic loop — the root cause is confirmed and external.
   Worth a periodic cheap check (just the `nvidia-smi` query above) rather
   than a full re-attempt.
3. If PolaRiS itself ships a new Isaac Sim version pin (6.0 GA, per the
   maintainer's own timeline in discussion #648) that's validated against
   whatever driver this cluster ends up on, that's the other unblock path
   — re-check `third_party/PolaRiS/pyproject.toml`'s `isaaclab[all,
   isaacsim]==` pin against upstream before re-attempting Stage 1.
4. The six `configs/polaris/tasks/*.yaml` files and
   `robo/polaris/import_official.py` do not need to be redone — they're
   static and don't depend on the driver. If PolaRiS's vendored commit is
   updated, re-run `import_official.py --all` to refresh them (it will
   need updating too if task registration, camera prims, or rubric
   criteria change upstream — check `git -C third_party/PolaRiS log` for
   what moved).
5. `third_party/PolaRiS/.venv` (~16 GB, fully synced) and
   `third_party/PolaRiS/PolaRiS-Hub` (~1.7 GB) are both already on disk and
   don't need re-downloading unless the pin changes.

## Exact commands for anyone re-verifying this report

```bash
# module/env setup (needed for anything that touches CUDA extensions or Isaac Sim)
eval "$(module bash-hook)"
eval "$(module bash-reset)"
module load nvidia-cuda-13.2.0 gcc-13.3.0
export TORCH_CUDA_ARCH_LIST="8.6;9.0+PTX"
unset TCNN_CUDA_ARCHITECTURES
export OMNI_KIT_ACCEPT_EULA=Y

# reproduce the crash (expect a segfault within a few seconds)
cd third_party/PolaRiS
srun --partition=debug --gpus=a6000:1 --mem=100G --time=0:10:00 .venv/bin/python -c "
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); a, _ = p.parse_known_args()
a.enable_cameras = True; a.headless = True
AppLauncher(a)
"

# confirm the no-camera path still works (expect success in ~7s)
srun --partition=debug --gpus=a6000:1 --mem=100G --time=0:10:00 .venv/bin/python -c "
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); a, _ = p.parse_known_args()
a.enable_cameras = False; a.headless = True
AppLauncher(a)
print('OK')
"

# check driver version on any node
srun --partition=<partition> --gpus=<type>:1 --mem=4G --time=0:03:00 \
  nvidia-smi --query-gpu=driver_version,name --format=csv
```
