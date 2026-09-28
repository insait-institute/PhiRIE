# DROID reconstruction protocol

Plan Task 16 (`plan/16_DROID_PAIRED_TRACK.md`), contract track `droid_replay`
(`docs/ICRA_RESEARCH_CONTRACT.md` §2): does the pipeline reconstruct real,
uncontrolled DROID robot workspaces and align them to the real FK
trajectory? **Replay/coverage evidence only** - no autonomous policy
success claim is made or implied by anything in this document. DROID's own
`success` label describes the ORIGINAL human/scripted-policy episode, not
anything this pipeline runs; it is carried through metadata for
provenance and is never treated as a label for a policy evaluated here.

## 1. Data

Raw DROID release at `${SIMANY_ROOT}/data/droid/raw/<lab>/success/<date>/
<timestamp>/` (`trajectory.h5` + `recordings/MP4/<serial>.mp4` per camera +
`metadata_*.json`), fetched by `run/fetch_droid_raw.py` (which walks episode
paths referenced in the local `droid_100` RLDS shards and re-fetches the
matching raw-release files via `run/gcs_fetch.py`). 47 episodes across 9
labs are present locally as of 2026-08-16 (`configs/droid/frozen_episodes.yaml`
has the full audited pool summary + a frozen stratified selection).

**Known pool gap**: every locally-fetched episode has `success=true` -
`run/fetch_droid_raw.py`'s episode-path regex is anchored on the literal
`success/` path component because that's what the seed RLDS shards
reference; no failure episodes are fetched yet. Success/failure
stratification (plan step 1) cannot be done until that changes.

## 2. Pipeline

`run/run_droid_recon.sh`: raw episode -> wrist-camera frames + FK trajectory
(`agents/recon/droid_extract.py`) -> COLMAP poses (`agents/recon/
colmap_poses.py`, VGGT fallback) -> Umeyama-align to FK, metric ROBOT BASE
frame (`agents/recon/align_to_traj.py`) -> emulated ScanNet++ scene dir
(`agents/recon/make_scene_dir.py`) -> 3DGS splat -> the same GT-free AUTO
tail as `run/run_video2sim.sh` (derive mesh, SAM3 auto-segment, TRELLIS,
CoACD/physics, MJCF export, `robo.tasks.pi05_tasks`).

```
EPISODE=<lab>/success/<date>/<ts> run/run_droid_recon.sh          # foreground
sbatch --export=ALL,EPISODE=<lab>/success/<date>/<ts>,RESUME=1 \
    run/slurm/droid_recon.sbatch                                  # cluster
```

`run/slurm/droid_recon.sbatch` MUST be used (or `bash run/run_droid_recon.sh`
directly) - see §3.1, this is load-bearing, not a style preference.
`debug` QoS caps concurrent GPU jobs at 4/user (`docs/ENVIRONMENTS.md`); a
5th `sbatch` queues with reason `AssocGrpGRES` until one finishes, which is
expected, not a hang. For longer/unattended batches use `--partition=batch
--qos=normal` targeting `sof1-h200-*`, adding `--exclude=msp3-[0-7]` if
requesting h200s - `msp3-*` nodes do not mount `${PHIRIE_WORKSPACE}`
(`docs/ENVIRONMENTS.md`).

## 3. Two pilot failures, root-caused and fixed (2026-08-16)

### 3.1 `outputs/droid_recon_pilot_703918.log`: `slurm_script: 4: source: not found`

**Root cause**: launched via something equivalent to `sbatch --wrap="source
run/env.sh; ..."`. `sbatch --wrap` runs the generated batch script under
`/bin/sh`, which on this cluster is `dash` - dash has no `source` builtin
(POSIX only requires `.`), so it died at the first `source` line before
`run_droid_recon.sh` (which itself does `source "$SIMANY_ROOT/run/env.sh"`,
`run/run_droid_recon.sh:34`, and uses bash-only syntax throughout) ever ran.

**Fix**: `run/slurm/droid_recon.sbatch` (new), `#!/bin/bash` shebang,
following the same convention as every other launcher in `run/slurm/`
(`behavior_recon.sbatch`, `replay_take.sbatch`, etc: bash shebang + SBATCH
directives + `bash run/run_*.sh`). Never launch this pipeline via
`--wrap=...` or a bare `sh` invocation.

### 3.2 `outputs/droid_recon_pilot_704172.log`: `Cannot allocate memory` in ffmpeg frame extraction

**Root cause** (`agents/recon/droid_extract.py`, `extract_frames`, then at
line ~124-146 before this fix): frame selection built one `-vf select`
filter with an OR-chain of `eq(n,i)` terms, one per kept frame index
(`sel = "+".join(f"eq(n\\,{i})" for i in mp4_indices)`). ffmpeg's expression
evaluator (`libavutil/eval.c`, used by the `select` filter) has a **fixed
internal parser stack limit around 100 terms**. Above it, filter-graph
construction fails during INIT (before decoding a single frame) with
`[AVFilterGraph] Error initializing filters` / `Error opening output files:
Cannot allocate memory` - a real ffmpeg `AVERROR(ENOMEM)` from the
expression parser's own allocator, unrelated to process/job memory.

Verified empirically on `hala` (`srun --partition=debug --mem=64G`, i.e. the
exact job class that failed): `ulimit -a` fully unlimited (`RLIMIT_AS`,
`RLIMIT_DATA` both unlimited, `RLIMIT_STACK` default 8 MB), 100 `eq(n,i)`
terms parse and run fine, 106 fail with the identical error and exit code
244 every time, independent of `--mem` (tested 64G and 100G). This matches
the pilot log exactly: `filter jv<0.3 |dgrip|<0.01 kept 106/401` -> 106
terms -> just over the boundary.

This is not an unlucky edge case: `select_steps()`'s relaxation ladder can
legitimately keep up to `--max-frames` (default 240) frames with no
run-length structure at all (its last rung keeps 100% of frames, and the
existing cap subsampled the kept set via `np.linspace` in INDEX order,
which shatters any contiguous runs into scattered singletons) - i.e. any
episode long/dynamic enough to need real filtering was always going to hit
this once its keep-count crossed ~100.

**Fix**: `agents/recon/droid_extract.py`, `extract_frames()` rewritten to
decode the WHOLE episode once with no `select` filter at all (`-vf` is just
the existing long-side `scale` filter, unaffected - the bug is specific to
`select`'s OR-chain, confirmed by bisection: `frames.py`'s own single-term
`select='not(mod(n,step))'` for uniform sampling was never at risk), then
keep only the wanted frames by renaming and delete the rest. Robust to any
distribution of kept indices at any count, at the cost of decoding
frames that get discarded - cheap here: the longest episode in the local
pool is 1140 frames (`configs/droid/frozen_episodes.yaml` pool summary),
a few hundred MB of throwaway JPEGs for a few extra seconds of ffmpeg time.
Range-compressing the OR-chain into fewer `between(n,a,b)` terms was
considered and rejected: it degrades gracefully on the pilot's specific
keep-set (15 runs) but not in general, since the post-filter `--max-frames`
cap can scatter an arbitrarily long contiguous run into up to 240 isolated
singletons - the same failure mode, just harder to trigger.

Verified fix: re-ran `droid_extract` standalone on the exact failing episode
(IPRL `Thu_Aug_24_21:29:53_2023`) - now succeeds, `kept 106/401` (previously
crashed the pipeline here), 106 frames written.

## 4. First end-to-end completions (2026-08-16)

5 episodes launched via `run/slurm/droid_recon.sbatch`
(`configs/droid/frozen_episodes.yaml` `priority_batch_v1`), spanning 5 labs.
**All 5 ran to full completion** (`DONE`, including the GT-free AUTO tail:
derived-mesh TSDF fusion, SAM3 auto-segmentation, TRELLIS image-to-3D,
CoACD+physics/URDF, MJCF export + MuJoCo settle test, `pi05_tasks` sanity
check) within this pass - job IDs 706918 (IPRL), 706930 (RAIL), 706928
(AUTOLab), 706929 (TRI), 706931 (RPL). All 5 got past `align_to_traj.py`
and produced a real `align_report.json`, and all 5 additionally have a
held-out validation report from `agents/eval/droid_alignment_eval.py`
(80/20 train/held-out split, fit re-estimated on train only):

| episode | lab | frames | center RMS (in-sample / held-out) | rotation median (in-sample / held-out) | held-out gate |
|---|---|---|---|---|---|
| `Thu_Aug_24_21:29:53_2023` | IPRL | 100 | 7.82 / 7.66 cm | 54.4 / 50.1 deg | **FAIL** (rotation) |
| `Tue_Oct_24_10:23:40_2023` | RAIL | 100 | 5.25 / 5.49 cm | 169.3 / 168.9 deg | **FAIL** (rotation) |
| `Fri_Jul_14_16:55:45_2023` | AUTOLab | 92 | 1.43 / 1.53 cm | 3.7 / 3.7 deg | PASS |
| `Thu_Sep_21_17:26:41_2023` | TRI | 103 | 1.65 / 1.56 cm | 11.6 / 11.6 deg | PASS |
| `Mon_Jun__5_16:11:49_2023` | RPL | 86 | 0.19 / 0.23 cm | 1.3 / 1.3 deg | PASS |

Held-out numbers track the in-sample numbers closely everywhere (no
overfitting signature) - the rotation problem on IPRL/RAIL is a property of
the reconstruction itself, not an artifact of evaluating on the same points
that were fit, which the held-out split rules out directly. Every episode
also produced a non-empty splat + derived mesh + MJCF scene; `pi05_tasks`
soft-failed on all 5 with "no tabletop cluster with graspable objects"
(expected for short wrist-camera-only episodes - noted in §6, not a
regression, and no policy-success number is claimed either way per
`docs/ICRA_RESEARCH_CONTRACT.md`).

**Finding, reported honestly rather than smoothed over**: translation/scale
alignment is consistently good (all 5 episodes: center RMS < 8 cm on ~90-100
frames); rotation is NOT consistently good. `align_to_traj.py` fits its
similarity transform on camera CENTERS only (rotation is applied globally
but never itself part of the fit objective - "reported, not fitted" per its
own docstring) and 2/5 episodes cleanly confirm the OpenCV c2w convention
(<4 deg) while 2/5 disagree by 54-169 degrees and 1/5 sits at a borderline
11.6 degrees. Because the clean and dirty cases run through IDENTICAL code,
this is evidence of a per-episode RECONSTRUCTION-QUALITY effect, not a
convention bug in `align_to_traj.py`/`droid_extract.py`: the IPRL COLMAP run
logged multiple `Ceres`/`Eigen` "Linear solver failure... Unable to perform
dense Cholesky factorization" warnings during bundle adjustment, consistent
with a weakly-constrained SfM problem on a "static-frame-filtered" set that,
by construction, has LOW baseline motion. A rotation fit fully determined by
center positions alone is exactly the kind of constraint that degrades
first when the underlying point trajectory doesn't carry much rotational
information. This is the open item for plan step 3 ("validate ... Umeyama
alignment on held-out points") - see `agents/eval/droid_alignment_eval.py`
below, and is NOT masked in any reported number: the held-out eval reports
both episodes as gate FAILures rather than hiding the rotation residual
inside an aggregate that happens to look fine on translation alone.

## 5. New tooling this pass

- **`configs/droid/frozen_episodes.yaml`**: audited pool summary (47
  episodes/9 labs/success-only gap/camera-quality split/task families) +
  stratification criteria + a frozen 20-episode selection across all 9 labs
  and both length tiers + the `priority_batch_v1` subset actually launched.
  Frozen BEFORE reconstruction quality was known, per
  `docs/ICRA_RESEARCH_CONTRACT.md`'s no-post-hoc-filtering rule.
- **`agents/recon/droid_static_select.py`**: standalone, reusable static-
  frame selection + masking, improving on `droid_extract.py`'s v1 in two
  ways: (a) viewpoint-diverse capping via farthest-point sampling on FK
  camera centers instead of `np.linspace`-in-time (avoids over-sampling one
  "parked" stretch and starving the rest - directly relevant to §4's
  rotation-fit weakness, since more viewpoint spread should help constrain
  rotation too); (b) a fixed gripper-region image mask (valid because the
  wrist camera is rigidly mounted - `droid_extract.py`'s own verified
  evidence) ANDed with a coarse per-frame frame-difference "transient
  content" mask for the freely-moving manipulandum. **Not yet wired into
  `run/run_droid_recon.sh`**'s default call path (deliberately, to avoid
  changing methodology under the batch jobs launched this same pass) -
  next step is an A/B rerun of one already-flagged episode (IPRL or RAIL)
  with FPS-capped frames feeding COLMAP, to test the rotation-weakness
  hypothesis above directly.
- **`agents/eval/droid_alignment_eval.py`**: held-out validation wrapper.
  Re-fits Umeyama on an 80% train split of the FK-overlapping frames (same
  offset resolution, same `umeyama`/`rot_angle_deg` helpers - imported, not
  reimplemented) and reports center/rotation/reprojection residual on the
  held-out 20%, plus a coarse dynamic-scene-contamination proxy (fraction of
  held-out frames whose reprojection error is a >2x-median outlier) and a
  documented pass/fail gate (<15 cm center RMS, <20 deg rotation median).
  Run on the two completed pilot episodes: IPRL and RAIL both **FAIL** the
  gate on held-out rotation (50.1 deg / 168.9 deg) while translation
  generalizes fine (7.66 cm / 5.49 cm) - consistent with §4, and reported
  rather than hidden.
- **This document.**

## 5b. Second batch, launched same pass (2026-08-16, in flight at hand-off)

`priority_batch_v2` (`configs/droid/frozen_episodes.yaml`): the 4 labs batch
v1 didn't touch (CLVR, IRIS, PennPAL, REAL) + a second IPRL episode, to
reach all 9 labs / 10 total workspaces queued across both batches. Job IDs
707064 (CLVR), 707065 (IRIS), 707066 (PennPAL), 707067 (REAL), 707068
(IPRL #2).

707064 (CLVR) already surfaced a genuine, correctly-handled failure mode:
COLMAP evidently under-registered this episode's frames and
`run_droid_recon.sh`'s VGGT-Omega fallback fired (`[omega] 88 frames ...`),
but the resulting reconstruction's camera centers still disagreed with the
FK trajectory by 16.2 cm RMS - over `align_to_traj.py`'s `MAX_RMS_M=0.10`
abort gate, so it exited loudly (`[align] ABORT: center RMS 16.2 cm > 10 cm
- frame correspondence or the reconstruction itself is broken; do not train
on this`) instead of silently continuing on a broken alignment. `set -e`
then stopped that job's script there, as designed - this is the
acceptance-criteria-required "loud fail" behavior working correctly, not a
bug, and is left as-is rather than "fixed" by loosening the gate. The other
3 (707065-707067) were still in COLMAP registration at hand-off; 707068
started as soon as 707064's slot freed (`debug` QoS caps 4 concurrent GPU
jobs/user - see `docs/ENVIRONMENTS.md`).

## 6. Toward the 3-lab/10-workspace acceptance bar

Reconstructed this pass: 5 episodes, 5 labs (IPRL, RAIL, AUTOLab, TRI, RPL),
**all 5 to full completion** with a real `align_report.json` +
`held_out_eval.json` each (3/5 pass the held-out gate, 2/5 correctly flagged
as rotation failures - §4). That clears the 3-lab minimum on lab COUNT;
workspace count (5) is short of the plan's 10. Remaining work, in priority
order:
1. Walk `configs/droid/frozen_episodes.yaml`'s remaining 15 frozen entries
   (all 9 labs already represented) through the same `sbatch` command to
   reach 10 workspaces - mechanical, no code changes needed, same recipe
   that produced the 5/5 completion rate this pass.
2. Investigate the rotation-fit weakness (§4) properly: either (a) confirm
   it correlates with COLMAP registration/BA quality (weak-baseline static
   subsets) by comparing IPRL/RAIL's COLMAP BA logs against the 3 clean
   episodes', or (b) try `droid_static_select.py`'s FPS-diverse frame set
   as an intervention on IPRL/RAIL specifically and re-run
   `droid_alignment_eval.py` to see if the held-out rotation gate then
   passes.
3. Re-fetch `PennPAL/success/2023-10-18/Thu_Oct_19_00:13:07_2023` (wrist MP4
   missing locally) or drop it from the frozen pool.
4. Fetch at least a few failure-labeled episodes if/when a source manifest
   for them is identified, to close the success/failure stratification gap
   noted in §1 and `configs/droid/frozen_episodes.yaml`.
5. Wire `agents/recon/droid_static_select.py` into `run/run_droid_recon.sh`
   as the default frame-selection step once (2) above decides whether its
   FPS-diverse capping is worth the switch, and score gsplat PSNR/SSIM with
   vs. without its gripper+dynamic masking feeding `gsplat_train`.

No autonomous policy success number is claimed anywhere in this document or
the artifacts it describes - `pi05_tasks` runs as a soft-fail sanity check
only (`run/run_droid_recon.sh`'s tail) and all 5 completed episodes
reported `no tabletop cluster with graspable objects` (too few
auto-discovered objects on a wrist-camera-only reconstruction of a short
manipulation episode - expected, not a regression; DROID's wrist camera
alone gives thin single-view coverage of anything the arm doesn't sweep
past).
