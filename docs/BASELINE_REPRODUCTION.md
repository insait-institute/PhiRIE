# Baseline reproduction record

This is the reproduction ledger plan/15_CONSTRUCTION_BASELINES.md step 1 and
3 ask for: for every construction baseline, exactly how it was run, what
official commit/weights/license it is pinned to, and every piece of glue
code added on top of the official release. The machine-readable summary
lives in [`baselines/release_status.yaml`](../baselines/release_status.yaml)
(`status: implemented / partial / literature_only / unavailable`); this
document is the narrative version, plus commands. Cross-reference:
[`docs/BASELINES.md`](BASELINES.md) (the wider literature survey this table
was audited against) and [`agents/baselines/README.md`](../agents/baselines/README.md)
(the pre-existing, already-working adapters this task did not move).

Two Python packages exist and are intentionally kept separate:

| package | what it is |
|---|---|
| `agents/baselines/` | the working baseline *implementations* that predate this task: MaskClustering, FlashSplat, the SimFoundry-reproduction launcher. Not touched by this task. |
| `baselines/` | this task's audit layer: `release_status.yaml`, the one new runnable ablation (`raw_reconstruction.py`), and inert stubs for baselines with no public release (`simrecon_adapter.py`, `replicate_adapter.py`). |

Everything below was verified by inspection on 2026-08-16 (git log/remote in
each `third_party/*` checkout, LICENSE files, and `docs/BASELINES.md`'s own
audit trail) -- not asserted from memory.

---

## 1. MaskClustering -- implemented

**Paper:** CVPR 2024, PKU-EPIC, arXiv:2401.07745.
**Pinned commit:** `eb2d41c2267cde21966f4340c37b8cf94b05c23c` (2024-04-25),
`third_party/MaskClustering`, remote `github.com/PKU-EPIC/MaskClustering`.
**License:** none declared in-repo (no LICENSE file in the vendored
checkout) -- treated as all-rights-reserved research code; do not
redistribute weights/outputs without contacting the authors.
**Weights:** `Mask2Former_hornet_3x_576d0b.pth` (Cropformer checkpoint,
HuggingFace) -- see blocker below; not actually used in the run we scored.

### What official code runs unmodified
`third_party/MaskClustering/main.py` (their mask-graph clustering) and their
`dataset/scannetpp.py` dataset loader, run against inputs built in their
exact expected layout.

### Glue added (audited, all in `agents/baselines/`)
- `agents/baselines/maskclustering.py` — `prepare` subcommand builds
  `third_party/MaskClustering/data/scannetpp/data/<seq>/` in their layout:
  ray-casts our scan mesh to synthesize `iphone/render_depth/*.png` (their
  own ScanNetPPDataset already expects *rendered*, not sensor, depth — so
  ScanNet++ DSLR, which has no native depth sensor stream, is a legitimate
  fit, not a workaround), writes a single-camera `colmap/cameras.txt` +
  `colmap/images.txt` from our own intrinsics/poses, and voxel-downsamples
  our scan mesh to their expected `pcld_0.25/<seq>.pth` scene point cloud.
  `convert` subcommand maps their class-agnostic scene-point-cloud masks
  back to mesh-vertex index sets (`auto_instances.npz`, this project's own
  contract, consumed by `agents/core/common.load_auto_instances`).
- **Documented blocker, not hidden:** their published 2D segmenter
  (Cropformer: detectron2 + Entity CropFormer + an MSDeformAttn CUDA op via
  `ops/make.sh` + mmcv) needs a from-source CUDA build this cluster cannot
  produce. `agents/baselines/mc_masks.py` substitutes our own SAM3
  segmenter for that one step — chosen deliberately because it also
  isolates the comparison to the 3D aggregation method (MaskClustering's
  mask-graph clustering vs. our multi-view mask lifting), rather than
  conflating two different 2D segmenters into the result.

### Command actually run
```
# CPU (login node OK): build the scene in MaskClustering's expected layout
.venv/bin/python -m agents.baselines.maskclustering prepare \
  --scene-dir /data/ScanNetpp/data/c50d2d1d42 --stride 12

# GPU (sam3 env): our SAM3 masks standing in for Cropformer
srun --partition=a6000 --gres=gpu:1 --time=1:00:00 \
  <sam3-env>/bin/python -m agents.baselines.mc_masks --seq c50d2d1d42

# GPU (.venv-mc env): their official clustering, unmodified
srun --partition=a6000 --gres=gpu:1 --time=1:00:00 \
  .venv-mc/bin/python main.py --config scannetpp_dslr --seq_name_list c50d2d1d42

# CPU: convert to our auto_instances.npz contract
.venv/bin/python -m agents.baselines.maskclustering convert \
  --seq c50d2d1d42 --out-dir outputs/c50d2d1d42_mc
```
(`agents/baselines/maskclustering.py srun --seq <seq>` prints the exact
commands above with paths filled in.)

### Result
F1@0.25 **0.605** vs our 0.571; F1@0.5 **0.419** vs our 0.286, same SAM3
masks, matched 28-frame budget (`docs/BASELINES.md` section 1). Caveat
carried over: single-pilot-scene comparison.

---

## 2. FlashSplat -- implemented (reimplementation, not the official binary)

**Paper:** ECCV 2024, arXiv:2409.08270, `github.com/florinshen/FlashSplat`.
**Pinned commit:** `3e3b14786333bf0163ba1b8541e86a3765112d7` (2024-09-13),
`third_party/FlashSplat` (reference only — see below).
**License:** Gaussian-Splatting License (Inria / MPII) — research and
evaluation use only, no commercial use (`third_party/FlashSplat/LICENSE.md`).

### Why the official rasterizer is not used
Their pipeline depends on a custom CUDA fork
(`submodules/flashsplat-rasterization`, which accumulates
`atomicAdd(&used_count[obj_id * P + gid], alpha * T)` inside the forward
kernel). There is no prebuilt wheel for it anywhere (PyPI has nothing for
`flashsplat-rasterization` or `diff-gaussian-rasterization`), and a source
build needs a toolchain this cluster does not have (system gcc 14, no
nvcc/g++ ≤ 13 on the GPU nodes). This is a documented blocker, not a
silent substitution.

### Glue added (all of `agents/baselines/flashsplat.py` — self-contained)
A from-scratch reimplementation of their exact published algorithm on
`gsplat` (already in this repo's stack):
- `flashsplat_assign` is a verbatim port of their `multi_instance_opt`
  binary-label solver (`objremoval.py`), same slackness/background-bias
  semantics (negative dilates the removed set, their object-removal script
  hardcodes -0.4; positive shrinks it, their segmentation script uses
  0.4–0.8).
- `view_counts` reproduces their `used_count[2, N]` accumulation — NOT via
  their custom kernel, but via an autograd-gradient identity: rendering is
  linear in per-gaussian color, so backward of a dummy two-channel
  masked-sum loss through `gsplat`'s standard rasterizer gives exactly
  their per-gaussian alpha·T sums inside/outside the mask. Proven
  equivalent to a float64 CPU reference renderer (Inria/gsplat compositing
  rules, explicit EWA projection) to `1e-9` max error — run
  `.venv/bin/python -m agents.baselines.flashsplat --selftest` (CPU-only,
  no scene data, no GPU).

### Command actually run
```
srun --partition=debug --gpus=a6000:1 --mem=48G --time=00:30:00 \
  bash -c '\
    SIMANY_SCENE=c50d2d1d42 \
    SIMANY_OUT=outputs/c50d2d1d42_factory \
    .venv/bin/python -m agents.baselines.flashsplat --slackness -0.4'
```

### Result
Union removal-set IoU **0.649** vs our final removal set (recall 0.87),
same trained splat + same per-object masks (`docs/BASELINES.md` section 1).
Any future comparison against a *different* FlashSplat rasterizer build
(e.g. if the CUDA op becomes buildable here) is not directly comparable to
this row without re-running the `--selftest` equivalence check against that
build too.

---

## 3. SimFoundry-reproduction -- partial (no official code exists)

**Paper:** arXiv:2606.28276 (Jun 2026), NVIDIA/Stanford/GaTech/UT/Toronto.
**Official code:** none (`docs/BASELINES.md` section 2.1: "No code
released."). This row therefore has no external commit to pin — it is a
paper-faithful reimplementation of their *described* method, composed
entirely from this project's own pipeline stages, run degenerately.

### What it is
`run/run_simfoundry.sh` — no dedicated code file, by design
(`agents/baselines/README.md`): it chains
`agents.discover.s0_select_frame` → `models.s1_segment` →
`models.s2_depth` → `agents.discover.s3_lift` → `models.s4_trellis` →
`agents.assets.s5_align` → `agents.assets.s6_physics` → `robo.sim.s7_sim` →
`agents.render.s8_render` restricted to one representative frame,
monocular metric depth, single-view TRELLIS, and no GT — matching the
paper's own described ablation row D (one frame, no multi-view generation,
no hybrid slot).

### Command actually run
```
srun --partition=a6000 --gres=gpu:1 --time=2:00:00 \
  SIMANY_SCENE=c50d2d1d42 bash run/run_simfoundry.sh
```

### The one number that must never be conflated
Reported F1 for this reproduction is **@20 mm on ScanNet++ instance
submeshes**. SimFoundry's own paper reports F1 **@10 mm on 12 staged YCB
tabletop scenes with FoundationPose quasi-GT**. Different threshold,
different scenes, different reference — `docs/BASELINES.md` section 0(2)
flags this as a previously-made and now-corrected error; this document
repeats the warning so it does not regress. Never place the two F1 numbers
in the same table cell.

---

## 4. raw_reconstruction -- implemented (new in this task, our own ablation)

**What it is:** the "raw fused reconstruction / no object factorization"
row plan/15_CONSTRUCTION_BASELINES.md asks for. Not a third-party release —
it is the counterfactual for every "per-object factorization" claim the
main pipeline makes: take the upstream capture + reconstruction outputs
(COLMAP-posed frames → trained 3DGS splat, plus either the ScanNet++
dataset scan mesh or a splat-derived TSDF mesh from
`agents/discover/derive_mesh_from_splat.py`) and use them exactly as-is.
Zero per-object stages run: no discovery, no removal, no registration, no
collision decomposition, no physics, no articulation.

**File:** [`baselines/raw_reconstruction.py`](../baselines/raw_reconstruction.py).

### Command actually run (verified against real scene dirs, 2026-08-16)
```
# Full ScanNet++ layout (mesh + splat + GT instances all present)
.venv/bin/python -m baselines.raw_reconstruction \
  --scene-dir /data/ScanNetpp/data/c50d2d1d42 \
  --splats-root /data/ScanNetppv2_gsplat/splats \
  --out-dir outputs/c50d2d1d42_baselines/raw_reconstruction

# This repo's own recon_scenes layout (splat present, no dataset mesh --
# mesh-dependent fields correctly fall back to explicit N/A)
.venv/bin/python -m baselines.raw_reconstruction \
  --scene-dir data/recon_scenes/data/pilot_a29cccc784 \
  --out-dir outputs/pilot_a29cccc784_baselines/raw_reconstruction

# Optional GPU-only train-view render self-check (needs CUDA + gsplat;
# NOT the official held-out DSLR-test-split protocol -- see the script's
# _render_selfcheck docstring)
.venv/bin/python -m baselines.raw_reconstruction \
  --scene-dir /data/ScanNetpp/data/c50d2d1d42 \
  --splats-root /data/ScanNetppv2_gsplat/splats --render-eval
```
On a CPU-only node (no CUDA), `--render-eval` was verified to fail loudly
and get logged in the report's `failures` list with `status: error` (not
silently converted to zero or omitted) — the run's other metrics still
complete and `build_success` stays `true`.

### Result shape
All 7 oracle-table metric cells (`instance_discovery_f1`, `removal_iou`,
`registration_error_m`, `stability_ratio`, `penetration_count`,
`policy_success_rate`, and the render metric when not explicitly requested)
are reported `not_applicable` with a specific, per-metric reason — never a
bare `0`. Descriptive stats (vertex/triangle/gaussian counts, watertightness,
convex-hull volume, GT-instance context) are real numbers computed from the
actual files on disk, with a `sha256_head` for provenance.

---

## 5. Unreleased / not-found baselines -- documented as N/A, not zero

Per plan/15_CONSTRUCTION_BASELINES.md's acceptance criterion ("Failure to
run is documented with logs and does not become an unfair zero. Mark
unsupported cells N/A, never zero."), the following have **no numeric row**
and **no adapter code that could fabricate one** — see
`baselines/release_status.yaml` for the full audit trail per baseline, and
`tests/test_baseline_release_status.py` for the automated check that these
stay inert.

| baseline | status | why |
|---|---|---|
| **SimRecon** | `literature_only` | arXiv:2603.02133 (CVPR 2026). `docs/BASELINES.md` explicitly lists it Tier 2, "cite and contrast, do not run: ... SimRecon (until code lands)". No public code found. Stub: `baselines/simrecon_adapter.py` (raises `NotImplementedError` on every call). |
| **ReplicateAnyScene** | `unavailable` | Name appears only in `plan/15_CONSTRUCTION_BASELINES.md`; no matching entry anywhere in `docs/BASELINES.md`'s full survey or `agents/baselines/README.md`. Not guessed at. Stub: `baselines/replicate_adapter.py` (raises `NotImplementedError` on every call). |
| **RoboSnap** | `literature_only` | arXiv:2607.06699 (Jul 2026). `docs/BASELINES.md` section 3.G describes it favorably but lists it Tier 2 ("cite and contrast, do not run") — no public code found. `plan/15`'s own instruction is conditional ("only if an official runnable release supports the frozen protocol"). No adapter file exists for it. |
| **PolaRiS-manual-construction** | `partial` | See the discrepancy note below — this one needed real investigation, not a quick "no code" verdict. |

### PolaRiS: a discrepancy worth flagging explicitly

`docs/ICRA_RESEARCH_CONTRACT.md`'s "Decision 1" (frozen 2026-08-16) states
that a PolaRiS/Isaac Lab integration "does not exist anywhere in this
codebase — verified by repo-wide grep, 2026-08-16" and deliberately scopes
the paper to a hand-built MuJoCo reference instead of a real PolaRiS
comparison.

That statement no longer matches the repository as of this same date:
**`third_party/PolaRiS` exists**, cloned 2026-08-14 (commit
`129abc45f680559a7da62b9280fdb127b02b7275`, remote
`github.com/arhanjain/PolaRiS`, MIT-licensed). It appears a parallel task
(plan Task 07, `robo/polaris/*`) vendored the official repository after — or
concurrently with — the research contract's freeze.

**What this means for this table specifically:** even though the official
code is now present, `robo/polaris/*` (the adapter plan Task 07 would add)
does not exist yet, so nothing in this repo actually runs PolaRiS end to
end under our frozen contract. There is therefore still no honest numeric
row to report here today — `baselines/release_status.yaml` marks this
`partial` (not `implemented`, because nothing runs it; not
`literature_only`, because runnable code genuinely is present) and no
adapter file is written under `baselines/` for it. **This discrepancy is
outside this task's ownership to resolve** (it touches
`docs/ICRA_RESEARCH_CONTRACT.md` and `plan/07_POLARIS_ENVIRONMENT_ADAPTER.md`,
neither owned by this task) and should be raised with whoever maintains
those documents / Task 07.
