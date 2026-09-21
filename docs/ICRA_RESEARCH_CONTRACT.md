# ICRA Research Contract v1 — "Scan Any Room into a Predictive Robot Simulator"

Frozen 2026-08-16. This document, plus `configs/experiments/icra_contract_v1.yaml`
and `configs/experiments/frozen_fields.yaml`, is the single source of truth for
what this paper claims, what evidence backs each claim, and what happens if a
gate fails. It supersedes ambition stated in `plan/*.md` where the two
disagree — the task files describe an idealized program; this contract states
what is actually being executed and why, with three scope decisions logged
below. Any post-freeze change requires a dated entry in §6 with rationale.

## 0. Why this version narrows the plan

The 22-task plan in `plan/` was written assuming resources this project does
not currently have readily available: a working PolaRiS/Isaac Lab
integration (the source is vendored at `third_party/PolaRiS` but untracked
and unintegrated — see the correction in Decision 1 below), physical robot
hardware (`docs/CONTRIBUTIONS.md`: "we have no robot", audited 2026-08-10,
unchanged), and multiple freshly-captured phone rooms (one real pilot clip
exists: `videos/pilot_a29cccc784.mp4`). Rather than block on acquiring those,
three scope decisions are made now, matching the plan's own stop-condition
philosophy ("narrow the claim rather than invent evidence"):

**Decision 1 — PolaRiS BLOCKED on a cluster-wide driver incompatibility;
MuJoCo fallback is now the confirmed, permanent path for this paper, not a
placeholder.** Full history, most recent finding first:

CORRECTION (2026-08-16, integration attempt, Stage 2): the dependency-install
probe's "GO" verdict checked CUDA *compute* compatibility only. Isaac Sim
also needs Vulkan/RTX *render* compatibility, a separate axis, and it fails
hard: with `enable_cameras=True` (required by every real PolaRiS task — they
are all vision-based), `AppLauncher()` segfaults inside
`librtx.scenedb.plugin.so` while creating the RTX Hydra render engine, before
ever reaching `gym.make`. Root-caused to a KNOWN, upstream-confirmed bug:
NVIDIA driver branch 595.x (this cluster ships 595.91.07 on A6000 and
595.71.05 on H200 — confirmed the SAME broken branch on both node types, so
this is a uniform fleet-wide deployment, not a hala quirk) is incompatible
with Isaac Sim 5.1.0's RTX renderer, whose validated driver is 580.65.06
(github.com/isaac-sim/IsaacSim discussions #648/#651, corroborated by
multiple independent RTX 4070/4090/5070Ti/5080/5090 reports on NVIDIA's dev
forum). No application-level workaround exists (`--/rtx/verifyDriverVersion/
enabled=false` only silences the warning, confirmed by another user, does not
fix the crash). The only fix is a cluster-wide driver downgrade to 580.65.06
— a sysadmin action affecting every other user's CUDA workloads on shared
nodes, outside this project's authority and not attempted or requested.

Everything upstream of this was real, working progress, not wasted effort:
`uv sync` was fully fixed (two real papercuts beyond the original probe: the
`flatdict` build-isolation pin needed `"setuptools<81"` specifically, since
setuptools>=81 dropped bundled `pkg_resources` which flatdict's legacy
`setup.py` imports; and the login shell leaks an H200-oriented
`TORCH_CUDA_ARCH_LIST=9.0+PTX` + old CUDA 12.4.1 module pairing that silently
breaks the custom CUDA extension builds for A6000/sm_86 unless overridden —
the exact same class of gotcha `docs/ROBOT.md` already documents for gsplat).
With those fixed, Isaac Sim launches headless fine WITHOUT cameras (7.2s
boot), and torch/diff-surfel-rasterization/simple-knn/openpi-client all
import and run on this cluster's A6000. So the CUDA-13/driver story from the
original probe was correct as far as it went — the render stack is the part
that fails, invisibly to a dependency-resolution-only probe.

**Verdict: BLOCKED, not merely deferred.** The MuJoCo + pi0.5 fallback
(`robo/envs`/`robo/rigs`/`robo/tasks`/`robo/eval`, `mujoco_paired` protocol,
built and smoke-verified 2026-08-16 on a real reconstructed scene — see the
`mujoco_paired` row below) is the numbers that ship for the "official vs.
reconstructed" comparison (plan Tasks 07-08), full stop, unless a cluster
driver downgrade happens for reasons outside this paper's scope. Remaining
PolaRiS work is bounded to what doesn't need the crashing renderer: static
USD/JSON metadata extraction (robot/camera/rubric/initial-condition manifest
fields) via plain `pxr.Usd`, honestly labeled as partial and non-executable
progress, documented in `docs/POLARIS_INTEGRATION.md` with full repro steps
so a future session can resume in ~10 minutes if the driver is ever fixed.

**One genuinely useful result survived the block**: static extraction of all
6 PolaRiS task definitions (`robo/polaris/import_official.py`,
`configs/polaris/tasks/*.yaml`, no Isaac Sim launch needed) found that
PolaRiS's robot reset pose, action convention (absolute joint + 0.5-threshold
binarized gripper), action_dim (8), and control rate (15 Hz) are IDENTICAL to
this project's own `configs/experiments/frozen_fields.yaml` — independent
confirmation that both conventions trace to the same DROID standard, not a
coincidence. Real declared differences: horizon (30s official vs our 32s) and
rubric structure (PolaRiS variable-length dependency-graph criteria vs. our
fixed 4-stage 0.25/stage) — both left as honest, documented differences, not
reconciled by fiat.

**Decision 2 — No real-robot claim.** Task 18 is dropped as a blocking
requirement. The headline predictive-validation evidence becomes the
**oracle track**: BEHAVIOR-1K/OmniGibson interactive scenes where ground
truth is known and hidden from the reconstruction process, so
reconstruction-vs-oracle policy agreement is a real causal measurement, not a
correlation of convenience. If real robot access becomes available later,
Task 18 slots in as a strengthening addition, not a required one.

**Decision 3 — Phone-room claim narrowed to available footage.** The paper
reports whatever real phone/video clips exist at freeze time (currently
n=1, `pilot_a29cccc784`) as deployment-timing/success evidence, explicitly
labeled as n=1 (or n=k), not generalized. Task 17's ≥3-room target remains
open; if more footage arrives before freeze it is added without re-opening
other gates.

## 1. Primary and fallback thesis

**Primary thesis:** An automatically constructed digital twin from a casual
video/phone scan preserves enough of a manipulation scene's causal structure
that a frozen policy's behavior in the twin predicts its behavior in the
source environment, measured (a) causally where ground truth is known and
hidden from construction (oracle track), and (b) as coverage/replay evidence
where it is not (DROID track), and the twin's own certificate predicts where
this correlation will hold.

**Fallback thesis** (if oracle correlation/ranking gates in §4 fail): An
automatically certified digital twin correctly abstains on constructions it
cannot validate, and abstention/repair improves the accepted subset's
agreement with oracle ground truth even when raw unfiltered correlation is
weak. This narrows "predictive simulator" to "self-aware simulator" — still a
real, falsifiable claim, and the one to fall back to rather than filtering
seeds/tasks to inflate correlation.

## 2. Evidence tracks (protocols) — no cross-protocol numeric comparison

| id | name | what it answers | ground truth | status |
|---|---|---|---|---|
| `scannet_scale` | construction at scale | does automatic factorization hold up over 50 scenes? | ScanNet++ laser scan + human instance labels | DONE — existing construction paper (`SimAnyRoom/body.tex`), reused as related evidence, not re-litigated here |
| `oracle_causal` | **headline**: does construction error causally predict policy behavior change? | full OmniGibson/BEHAVIOR-1K sim state, hidden from reconstruction | 6 scenes/4 families built, fast-tier scored (18 combos), deep-tier (real SAM3+TRELLIS+CoACD+gsplat) running; the earlier-cited \"7/7 BDDL\" number is NOT attributable to this track (mis-pathed glob, corrected 2026-08-16) |
| `mujoco_paired` | does construction method (holding robot/policy/camera/control fixed) change outcome vs. a hand-built reference scene? | none (paired comparison, not GT) | runner built + smoke-verified on a real scene (frozen-field diff={} between conditions); not yet run at matrix scale |
| `droid_replay` | does the pipeline reconstruct real, uncontrolled robot workspaces and align to real trajectories? | DROID FK trajectory (replay agreement only, no autonomous success labels) | crash fixed 2026-08-16; 5/9 labs reconstructed end-to-end (2/5 show large rotation residual, traced to weak SfM/BA, not a convention bug); 2nd batch of 4 labs in flight |
| `phone_deploy` | capture-to-sim time/success on genuinely casual footage | none — descriptive | n=1 real clip |

A single number from one track may never be quoted as if it were another
track's evidence (e.g., oracle causal agreement is not a "real-robot" claim).

## 3. Baselines admissible for the predictive paper

Reused from the construction paper, unchanged: MaskClustering, FlashSplat,
SimFoundry-reproduction (`agents/baselines/`, ablation row D). For the new
tracks: **raw reconstruction / no factorization** and **legacy support-shim
MJCF export** are admissible in both `oracle_causal` and `mujoco_paired`.
**`simfoundry_repro`** (`baselines/simfoundry_repro.py`, added 2026-08-30) is
admissible in `mujoco_paired` ONLY, not `oracle_causal` — correction, same
day: `s0_select_frame.py` and `s5_align.py` both hard-depend on
`C.load_gt_instances()`, which only resolves ScanNet++-style
`segments.json`/`segments_anno.json`. On BEHAVIOR scenes this either crashes
outright or, if a same-shaped GT file ever existed there, would leak hidden
oracle GT into the baseline's own frame selection — a real contract
violation for a track whose entire point is that GT stays hidden from
reconstruction. `mujoco_paired`'s scenes (c50d2d1d42 and the other
composite-eval-ready ScanNet++ scenes) are exactly the layout those stages
already support safely, so no code changed there, only where it's admissible.
SimFoundry the *paper* (as opposed to this in-repo reproduction), WANDA,
Lumera remain literature-only citations per `docs/BASELINES.md` — never
re-benchmarked against our closed-loop numbers, since their released
artifacts do not support running our policy/task suite.

## 4. Primary/secondary metrics and go/no-go thresholds

Primary (oracle_causal track, per policy-task pair, aggregated with
hierarchical bootstrap over scenes then seeds — see `agents/eval/bootstrap.py`):
- staged-progress MAE between reconstructed and oracle rollout (0.25 per
  stage: grasp/lift/hover/place) — **go/no-go: 95% CI upper bound < 0.35**
  (roughly one stage of disagreement) or the primary thesis is narrowed to
  the fallback in §1.
- Spearman rank agreement across task-difficulty ordering, reported alongside
  MAE — **go/no-go: rho > 0 with CI excluding 0**, else report descriptively.

Secondary: pairwise policy-preference accuracy, MMRV, calibration/Brier of
the certificate's predicted error, risk-coverage AUC (certificate vs. global
PSNR/F1/drop-stability baselines).

Minimum non-degenerate pairs before any correlation is reported: **12**
policy-task pairs (2 policies × ≥6 task instances, oracle track). Below this,
report descriptive per-pair numbers only — this is the concrete instantiation
of Task 10's stop condition.

## 5. Frozen fields (see `configs/experiments/frozen_fields.yaml` for the
machine-readable list)

Camera intrinsics/extrinsics per task family, control rate (15 Hz) and
action convention (absolute joint + binarized gripper), robot reset pose,
horizon (32 s per `docs/ROBOT.md`'s documented gotcha), rubric definition
(staged 0.25/stage), episode seed list, task language template. These are
frozen once `configs/experiments/example_manifest.yaml` (Task 01) is
generated for a track and may not change without a new track version.

## 5.5 Certificate mechanism status (Task 12, built 2026-08-16)

`robo/certification/{features,model,calibrate,report}.py` is complete and
tested (24 passing tests). On a synthetic fixture designed to match the
plan's acceptance criterion (a task-local defect drives real error while
global PSNR/F1/drop-stability stay independent of it), the certificate
clearly beats all three global baselines under both leave-one-scene-out and
leave-one-task-family-out CV (risk-coverage AUC ~0.17 vs ~0.30-0.40, no
leakage). This demonstrates the mechanism is *capable* of the claim, not that
it holds on this paper's real data yet: **zero real (features x
ground-truth-error) joined pairs exist today** — the deep-tier oracle build
those pairs depend on (`configs/experiments/icra_contract_v1.yaml`'s
`oracle_causal` protocol, job 708437) was still running at last check, and
only 1/6 scenes has any real build_audit-computable evidence at all, which
doesn't cleanly join to ground truth yet either. Per the contract's own stop
condition (`min_nondegenerate_pairs: 12`, currently 0 real ones): report this
honestly as "mechanism validated on synthetic fixtures, real evaluation
pending deep-tier completion" — do not force a real-data number before the
oracle build actually finishes.

## 6. Post-freeze change log

- 2026-08-16: initial freeze, Decisions 1-3 above (no PolaRiS, no real robot,
  phone claim narrowed to n=1). Author: agent session, approved by user
  directive to proceed in full-auto mode after the three scope questions were
  answered by direction rather than by explicit selection.
- 2026-08-16 (same day, post-freeze correction #1): Decision 1's claim that
  PolaRiS "does not exist anywhere in this codebase" was factually wrong — a
  parallel Task 15 agent found `third_party/PolaRiS`, a real untracked clone
  dated 2026-08-14. Rewrote Decision 1 to "deferred pending a feasibility
  probe" rather than "dropped because absent." Does not change the near-term
  execution path (MuJoCo fallback still used until a probe runs) but changes
  the reason and reopens PolaRiS as a strictly-better upgrade if the probe
  succeeds. Author: agent session, triggered by cross-checking a subagent's
  report against the filesystem rather than trusting it directly.
- 2026-08-16 (same day, post-freeze correction #2): the `oracle_causal` row's
  cited "7/7 BDDL" evidence for `behavior_task-0020` was found by the Task 14
  agent to be mis-attributed — `export_omnigibson.py` globs `*_factory`
  directories, which that scene's output dir is not named as, so the number
  is a general ablation result, not oracle-track evidence. Removed the
  citation; oracle_causal status now points to the real 6-scene build-out
  (fast-tier scored, deep-tier in flight) instead. Author: agent session.
- 2026-08-16 (same day, post-freeze correction #3): `droid_replay` status
  updated from `blocked_on_pipeline_fix` to `in_progress` — the Task 16 agent
  root-caused and fixed both crash causes (non-bash sbatch shebang; an
  ffmpeg `select` filter term-count limit, not a memory limit despite the
  "Cannot allocate memory" message) and reconstructed 5/9 labs end-to-end.
- 2026-08-16 (same day, post-freeze correction #4): Decision 1 updated to GO
  after the PolaRiS feasibility probe (driver/CUDA match on both A6000 and
  H200, 233/237 packages install cleanly via uv sync); a full integration
  attempt is now running. Separately, G3's premise was wrong: the Task 08
  policy-registry agent found `droid_pi05_jointpos_with_web_and_sim` has
  never been run through closed-loop eval despite being downloaded/wired —
  it is `exploratory`, not `verified`. G3 text corrected; a real verification
  run is dispatched as a strengthening step.
- 2026-08-16 (same day, post-freeze correction #6): Decision 1's "GO"
  downgraded to "BLOCKED" — the integration attempt reached Stage 2 (headless
  Isaac Sim launch) and hit a real, upstream-confirmed driver incompatibility
  (cluster ships NVIDIA 595.x, Isaac Sim 5.1.0's RTX renderer needs
  580.65.06; segfaults with cameras enabled, which every real PolaRiS task
  needs). Confirmed fleet-wide (same broken driver branch on both A6000 and
  H200 node types), not a single-node fluke. No software workaround exists;
  the only fix is a sysadmin-level driver downgrade, correctly not attempted
  by the agent since it would affect every other user's CUDA workloads on
  shared infrastructure. MuJoCo fallback is now the confirmed, non-provisional
  path for Tasks 07-08, not a placeholder pending PolaRiS.
- 2026-08-16 (same day, post-freeze correction #5): the verification run
  dispatched in correction #4 completed (SLURM job 707333, debug partition
  on hala, 21 min: serve `droid_pi05_jointpos_with_web_and_sim/80000` with
  `SIMANY_PI05_CONFIG=pi05_droid_jointpos_sim` via `run/pi05_serve.sh`, then
  `robo/eval/pi05_eval.py` raster obs against scene `c50d2d1d42`, 30
  episodes = 3 tasks x 10, `--time-limit 32`). Real result: 5/30 nonzero
  (16.7%), 2 grasp+lift (score 0.50), 3 grasp-only (score 0.25), 0/30 full
  "place" success, mean staged score 0.058
  (`outputs/pi05_runs/verify0816_sim_raster/results.json`) — one JAX-jit
  warmup websocket disconnect (2 reconnects) on first inference, the
  documented docs/ROBOT.md gotcha, recovered normally and is not counted as
  a failure. This is non-degenerate, rubric-consistent staged behavior at
  the same order of magnitude as `pi05_droid_jointpos`'s own same-scene
  raster baseline (2/15 nonzero, 2 lifts), so
  `droid_pi05_jointpos_with_web_and_sim` is promoted from `exploratory` to
  `verified` in `configs/policies/pi05_droid_jointpos_sim.yaml`, and G3
  below is updated to match. The result was not cherry-picked: this was the first and only
  run submitted for this checkpoint, on a scene chosen in advance (the one
  scene in docs/ROBOT.md's 6-scene inpainted set where the *other*,
  zero-shot checkpoint is already documented to score nonzero), and the
  real numbers are reported as obtained regardless of which way they came
  out. Author: agent session (Task 08 follow-up).
- 2026-08-16 (same day, post-freeze correction #7): certificate mechanism
  (Task 12) built and tested; beats global baselines on a synthetic fixture
  designed to match the acceptance criterion, but has zero real joined pairs
  yet since the deep-tier oracle build it depends on is still running.
  Documented in the new section 5.5 rather than claiming a real-data result
  prematurely.
- 2026-08-30 (post-freeze correction #8): added `simfoundry_repro` to
  `oracle_causal`/`mujoco_paired`'s admissible_baselines after building
  `baselines/simfoundry_repro.py`, which makes the existing ScanNet++-only
  SimFoundry-reproduction ablation (row D) callable against the
  `recon_scenes` layout the new tracks use. No change to the underlying
  method or its `partial` release status -- only to what scenes it can run
  against.
- 2026-08-30 (post-freeze correction #9): walked back correction #8's
  addition of `simfoundry_repro` to `oracle_causal`'s admissible_baselines
  after checking the actual stage code before launching -- `s0_select_frame`
  and `s5_align` both hard-depend on ScanNet++-only GT-instance files,
  making them unsafe (crash, or worse, a hidden-GT leak) on BEHAVIOR scenes.
  Kept admissible in `mujoco_paired` only, where the scene layout genuinely
  matches. Also fixed a real bug in simfoundry_repro.py's run_pipeline()
  found in the same pass: every stage was routed through the main venv
  interpreter, but s1 (SAM3) needs its own torch-2.10 env and s1's --image/
  --out-dir/--prompts args were never resolved from s0's actual chosen
  frame -- both would have failed immediately on a real cluster run.
- 2026-08-31 (post-freeze correction #10): the first real 3-condition
  mujoco_paired run (job 791521) reported simfoundry_repro coverage loss
  as a raw-index mismatch, but the actual fix (robo/eval/task_regrounding.py)
  found something worse underneath: across 5 real scenes / 13 tasks, 4 had
  the SAME raw object index resolving to a DIFFERENT physical object in
  the simfoundry_repro build (same id, wrong label) -- the old code would
  have silently scored the policy against the wrong target with no crash
  at all, not merely lost coverage. Label-based re-grounding (falls back
  through exact label -> category synonym -> geometry tie-break, reusing
  robo/certification/grounding.py's confidence machinery) now catches and
  refuses these instead of guessing. Real coverage after the fix: 5/13
  resolved, 8/13 honestly unresolved (7 of those are true absences --
  SimFoundry-repro's single-representative-frame discovery just did not
  find that many objects, consistent with the construction paper's own
  ablation-row-D finding that single-image zero-shot yield collapses to
  17.2%). The 791521 run's numbers in the oracle_causal/mujoco_paired
  table above are superseded, not merely underpowered -- rerun scheduled.
- 2026-08-31 (post-freeze correction #11): the first post-regrounding real
  benchmark attempt (job 792111) hit two more issues, both fixed same day.
  (a) `configs/experiments/simfoundry_comparison_real.yaml` used relative
  `factory_dir`/`tasks_json` paths, which resolve wrong under
  `run/slurm/paired_matrix.sbatch`'s `cd $SIMANY_ROOT/robo` -- fixed by
  using absolute paths (matches the identical CONFIG/OUT-path lesson from
  correction earlier the same day). (b)
  `robo/policy/control_contract.validate_env_action()` (Task 08, built
  2026-08-16) rejected ~25 otherwise-normal episodes across ALL THREE
  conditions with `EnvActionShapeError` on tiny negative gripper values
  (e.g. -0.0025) -- a real model-output range, not a client bug, given the
  environment binarizes the gripper channel at a 0.5 threshold downstream
  regardless. The `+/-1e-6` tolerance was simply too strict for ordinary
  inference noise. Fixed to clamp into range (raising only for NaN/Inf or a
  channel far outside any plausible range). This bug predates today but
  was invisible until the registry-client-construction bug (correction #11
  earlier) was fixed enough for the real Pi05PolicyClient path to actually
  run instead of silently falling back to the older unvalidated
  pi05_eval.py path. Rerun scheduled (job 792157).
- 2026-08-31 (post-freeze correction #12): correction #11's own gripper
  clamp fix had a bug that made ALL 78/78 episodes of the next attempt
  (job 792562) crash with `ValueError: assignment destination is
  read-only` -- `np.asarray(action, dtype=float)` returns a VIEW (not a
  copy) when the input is already float64, and the in-place gripper-clamp
  write then fails on a read-only input (a real case for
  robo.policy.clients.pi05_client's chunk slices). Fixed by using
  `np.array` (always copies) instead of `np.asarray`. Also discovered
  while chasing this: job 792562 was the FIRST real run to actually go
  through the new `Pi05PolicyClient` registry path end-to-end (every
  earlier "successful" run, including 791521, silently fell back to the
  older `pi05_eval.py` path because of the `home`/`open_loop_horizon`
  TypeError fixed earlier the same day) -- so today's registry fixes have
  not yet been validated by a genuinely clean real-policy run. Rerun
  scheduled (job 792587).
- 2026-08-31 (post-freeze correction #13, clean run achieved): job 792587
  completed with ZERO infrastructure crashes -- corrections #9 through #12
  (task regrounding, config path resolution, gripper-range clamp, and the
  clamp's own read-only-array bug) are all confirmed fixed by a real
  end-to-end run, not just unit tests. Real numbers exist for
  mujoco_paired (see the protocol table above) for the first time. They
  remain explicitly NOT citable as a paper finding (n=26 per condition,
  below the power analysis's own floor) -- what changed is that the
  measurement pipeline itself is now trustworthy; a larger-n rerun is the
  next step before any comparative claim.
- 2026-08-31 (post-freeze correction #14, MATERIAL, awaiting a scope
  decision): SimFoundry (arXiv:2606.28276) now has an official code
  release -- github.com/NVlabs/SimFoundry, created 2026-07-15 (after
  docs/BASELINES.md's original audit), Apache-2.0, real and substantial
  (13-stage reconstruction pipeline, augmentation, OmniGibson application,
  scene editors, 281 stars, actively pushed as of 2026-08-27). This was
  previously documented as "no code released," driving the `partial`
  status and our own from-description reimplementation
  (`baselines/simfoundry_repro.py`). Environment check: disk is fine
  (3.4TB free vs their ~250GB need); `facebook/sam3` gated access is
  already confirmed (this repo depends on it directly); their other two
  gated models (`facebook/dinov3-vitl16-pretrain-lvd1689m`,
  `briaai/RMBG-2.0`) are unconfirmed for this account; their VLM stages
  need a GCP project with Vertex AI or a Gemini API key, and this
  account's gcloud currently has no active project set. NOT YET DECIDED:
  whether to attempt integrating the real official pipeline (a
  multi-hour-to-multi-day effort of similar scale to the PolaRiS attempt,
  which itself hit a real blocker) as a strictly stronger `implemented`
  baseline, replacing or supplementing `simfoundry_repro.py`. Their own
  README states the sim-to-real policy training/data-generation code
  behind their headline policy-comparison numbers is explicitly NOT
  released, so even full integration would not unlock everything their
  paper claims.

## 7. Gates (redefined from `plan/README.md` G1–G6 given Decisions 1–3)

- **G1**: one phone/video scan reaches a full-collision MuJoCo simulator
  (Task 06) and produces a deterministic smoke-test rollout.
- **G2** (redefined, no PolaRiS): one hand-built reference task variant runs
  in MuJoCo with an identical robot/camera/control/rubric contract to the
  SimAny reconstruction of the same task family.
- **G3** (redefined, no real robot; corrected 2026-08-16, updated same day
  once the dispatched verification run finished — see change log): both real
  pi0.5 checkpoints now have verified checkpoint-to-behavior correspondence.
  `pi05_droid_jointpos`'s 0.28 pooled nonzero rate matches docs/ROBOT.md's
  cited 28% RoboLab-120 baseline on the same checkpoint bytes.
  `droid_pi05_jointpos_with_web_and_sim` was downloaded and wired 2026-08-04
  but sat unrun until a real closed-loop robo.eval.pi05_eval run (SLURM job
  707333, 2026-08-16, raster obs, scene c50d2d1d42, 30 episodes across 3
  tasks) scored 5/30 nonzero (16.7%), 2 grasp+lift episodes, mean staged
  score 0.058, 0/30 full success — non-degenerate and the same order of
  magnitude as `pi05_droid_jointpos`'s own same-scene raster baseline (2/15
  nonzero, 2 lifts), so Task 08's registry now labels it `verified` too
  (`configs/policies/pi05_droid_jointpos_sim.yaml`). G3 is satisfied at
  2 verified + 1 scripted policy across ≥6 oracle task instances. Neither
  pi0.5 checkpoint achieves full task success in this suite yet (0/30 place
  here, and low absolute rates throughout docs/ROBOT.md) — "verified" is a
  checkpoint-to-behavior correspondence claim, not a competence claim.
- **G4**: the task certificate improves risk-coverage on held-out oracle
  scenes vs. every global single-metric baseline, or the predictive claim is
  narrowed to the fallback thesis.
- **G5** (redefined): oracle track covers ≥6 scenes across ≥2 task families
  with ≥1 controlled degradation level each; phone track reports whatever
  n it has (≥1) honestly, never rounded up.
- **G6**: every paper number is generated from committed manifests
  (Task 01) and passes the submission audit (Task 21).

## Stop condition inherited from Task 00

If fewer than 12 non-degenerate policy-task pairs exist after the oracle
track is built out (§4), report descriptive results only and narrow the
predictive claim in the paper text — do not increase episode count by
resampling only favorable scenes, and do not drop pairs post-hoc to raise
correlation.
