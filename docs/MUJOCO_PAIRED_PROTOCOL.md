# MuJoCo paired protocol (`mujoco_paired`)

Task 09 (`plan/09_PAIRED_ROLLOUT_RUNNER.md`), scoped by
`docs/ICRA_RESEARCH_CONTRACT.md` Decision 1 and gate **G2**. This document
is the design record for the one genuinely open question in that task: with
PolaRiS/Isaac Lab deferred, what does an "official/reference" condition even
mean, built entirely inside the existing MuJoCo + pi0.5 stack? It also
records the resulting runner's invariants, output layout, and outcome
taxonomy.

## 1. What `mujoco_paired` answers

Per `docs/ICRA_RESEARCH_CONTRACT.md` section 2's evidence-track table:

> does construction method (holding robot/policy/camera/control fixed)
> change outcome vs. a hand-built reference scene? ground truth: **none**
> (paired comparison, not GT).

This is explicitly **not** the oracle track. There is no ground truth here,
and no claim that either condition is "correct." The only claim
`mujoco_paired` can support is: *holding every frozen field byte-identical,
does a policy behave differently in the SimAny reconstruction than it does
in a hand-built stand-in for the same task family?* A difference is
evidence that construction quality matters to closed-loop outcomes; the
absence of a difference is evidence it might not, for that policy/task. It
is a paired comparison, the same statistical shape as an A/B test — never a
number to be read as "vs. real world" or "vs. ground truth."

Gate **G2** (redefined in the contract, no PolaRiS): *"one hand-built
reference task variant runs in MuJoCo with an identical robot/camera/
control/rubric contract to the SimAny reconstruction of the same task
family."* This document's job is to say precisely what "hand-built
reference task variant" means when built with the tools actually available
today (`robo/envs`, `robo/rigs`, `robo/tasks`, `robo/eval`, `robo/sim`) —
not with Isaac Lab / PolaRiS scenes, which don't exist in a runnable state
yet in this repo.

## 2. The design decision: what is a "reference" scene?

### 2.1 Constraints from the contract and the existing stack

- **Everything control/camera/robot/rubric-related must be byte-identical**
  between conditions (`configs/experiments/frozen_fields.yaml`,
  `docs/ICRA_RESEARCH_CONTRACT.md` section 5). That rules out building the
  reference variant as a separate task suite with its own robot placement,
  camera pose, or rubric — it must reuse the SAME
  `sim_export/pi05_tasks.json` the SimAny reconstruction already produced
  for that scene (`robo/tasks/pi05_tasks.py`'s output), not a hand-authored
  parallel one.
- **Task 09 step 1's invariant ("generate reset states once, never
  independently resample by method") extends past RNG seeds to geometry.**
  If the reference scene placed its objects at even slightly different
  initial poses than the reconstruction, a policy divergence could be
  explained by "the reset states weren't actually paired," which would
  contaminate every result from this track before analysis even starts.
  So object initial pose must come from the SAME measurement the
  reconstruction itself used — not a hand-picked stand-in pose.
- **No PolaRiS/Isaac Lab oracle scene exists to import** (Decision 1: a
  feasibility probe just returned GO and integration is a separate,
  in-flight track — see `robo/polaris` once it lands). Building a "real"
  independently-modeled reference room by hand for each scene (measuring a
  real desk with a tape measure, e.g.) is out of scope for one task and
  would not obviously be more "official" than the alternative below — it
  would just be a second, unvalidated reconstruction.

### 2.2 The decision

**A reference scene is the SAME scene manifest (robot pose, table, camera,
task list, rubric) with every reconstructed asset — object meshes AND the
carved room background (`robo/sim/room_collision.py`, `--collision-mode
room`) — replaced by the plainest primitive that preserves the same
physical footprint:**

- Each object becomes **one box**, sized to that object's own measured
  `world_dims` (`objects/*/aligned.json`), at the exact same world
  position/orientation the reconstruction's own `T` matrix places it at,
  with the same mass/friction (`objects/*/physics.json`) — i.e. the SAME
  physics inputs the SimAny pipeline already computed, just rendered as a
  box instead of a CoACD-decomposed scanned mesh.
- The scanned room background (floor + walls + supports,
  `robo/sim/room_collision.py`'s carved output) is replaced by **a flat
  floor plane**. The table itself needs no separate reference treatment:
  `robo/rigs/pi05_rig.py::build_scene_model` already adds a single solid
  box `tabletop` geom from the suite's shared `table_box` field in BOTH
  conditions — it was never part of the reconstructed asset to begin with.
- Everything else (robot base pose, ext/wrist camera, control contract,
  rubric, task language, `exclude_objects`) is not duplicated by this
  module at all — `robo.eval.paired_runner` passes the SAME suite fields to
  the SAME `robo.rigs.pi05_rig.build_scene_model` /
  `robo.envs.pi05_env.DroidSimEnv` code path for both conditions. The only
  difference between the two calls is which `scene.xml` gets loaded.

This is implemented in `robo/eval/reference_scene.py::build_reference_scene_xml`.

### 2.3 Why this and not the alternatives considered

- **A literal "official" room (independently modeled by hand, not derived
  from the scan at all)** would be the most defensible "reference" in
  principle, but building even one by hand (let alone one per scene/task
  family) is its own multi-day modeling task, is not obviously any more
  "ground truth" than a primitive stand-in (a hand-modeled room is still
  someone's guess at the room, not a measurement), and does not fit this
  task's scope. It is the natural upgrade path once/if a real oracle
  (PolaRiS/Isaac Lab, or a laser-scanned CAD room) becomes available — it
  would slot in as a *replacement* for `reference_scene.py`'s output
  without touching anything else in `paired_runner.py`, since the pairing
  contract (identical robot/camera/control/rubric, only scene-construction
  differs) is exactly the same either way.
- **Re-using the reconstruction's own object poses but its own meshes too,
  varying only the room background** was considered and rejected: object
  mesh fidelity (CoACD collision quality, TRELLIS shape accuracy) is
  exactly the kind of construction-quality question this track exists to
  probe. Freezing objects and varying only the room would silently narrow
  the claim to "does room reconstruction matter" while still labeling it
  "construction method," which overstates what was actually tested.
- **A world-axis-aligned bounding box** (using `objects.json`'s scan-frame
  AABB directly, skipping the rotation decomposition) was considered
  because it needs less code, but for objects that sit at an angle to the
  world axes it can be substantially larger than the object's own oriented
  footprint, silently changing the physical scene (more table area
  occluded, different graspability) rather than only its visual identity.
  Decomposing `aligned.json`'s `T` into `(pos, quat)` and sizing the box to
  `world_dims` in the object's OWN frame keeps the footprint faithful.
- **Keeping the reconstruction's exact collision hulls but stripping
  texture/visual fidelity only** was rejected because it isn't actually
  simpler to build (the CoACD parts are the expensive, error-prone part of
  construction this track is meant to probe) and wouldn't read as
  "hand-built" by any reasonable definition — it would just be the SimAny
  condition with the textures turned off.

### 2.4 What this buys, and what it doesn't

**What it buys:** every DECLARED scene-construction fact (asset identity:
scanned mesh vs. primitive; collision representation: CoACD parts + carved
room vs. box + floor) differs between conditions, and *only* that differs —
robot pose, camera, control contract, rubric, task language, and every
object's initial world pose are identical by construction, not by
convention. This is directly checkable:
`robo.manifest.io.diff_manifests(reference_manifest, simany_manifest,
frozen_only=True)` (using `robo/manifest/schema.py`'s
`FROZEN_ROLLOUT_FIELDS`) returns `{}` for any matched pair produced by this
runner — verified against a real scene in
`docs/MUJOCO_PAIRED_PROTOCOL.md`'s validation run below, and asserted
implicitly by `tests/test_paired_runner.py::test_paired_reset_poses_match_across_conditions`.

**What it doesn't buy:** this is not a claim that the primitive-box scene is
"what a human would build by hand for this room," and it is not a
ground-truth environment — no success/failure number from this track may be
quoted as an oracle-track number (`docs/ICRA_RESEARCH_CONTRACT.md` section
2: "no cross-protocol numeric comparison"). It answers exactly one
question: given the identical task contract, does replacing the
reconstructed asset with the plainest same-footprint primitive change a
frozen policy's behavior? A large gap is evidence reconstruction fidelity
matters to closed-loop success on this task family; a small gap is evidence
it might not — for that policy, that task, that scene.

## 3. Runner design (`robo/eval/paired_runner.py`)

### 3.1 Reset states, generated once (Task 09 step 1)

`robo.eval.episode_log.ResetState` + `derive_reset_seed(base_seed, task_id,
ep)` — a pure hash of the identifying triple, not a running
`np.random.RandomState` threaded across a whole suite (what
`robo/eval/pi05_eval.py` does today). The whole reset-state list is computed
once per output directory and persisted to `<out_dir>/reset_states.json`;
every subsequent invocation against the same `--out` reuses that file
verbatim (`--episodes`/seeds on a resumed run are ignored, loudly). Both
conditions read the same list; the jitter draw for episode N of task T
under seed S is identical whichever condition runs it, and independent of
which condition happens to run first.

### 3.2 Coverage: build failures are rows, not omissions (step 2)

`robo.eval.paired_runner.build_env` never lets an exception escape
un-classified — every failure becomes `robo.eval.episode_log.
BuildFailureError`. `run_matrix` logs one `EpisodeRecord` with
`outcome=build_failure` for **every** reset_state that would have used a
failed (scene, condition) build, before moving on. `matrix_summary.json`'s
`coverage_by_outcome` therefore always sums to the number of planned
episodes across both conditions — a scene that fails to build entirely
still occupies its denominator slots.

### 3.3 Frozen synchronization (step 3)

`horizon_s`/`rate_hz`/`physics_dt`/`substeps_per_tick`/action convention all
come from `robo.policy.control_contract.FROZEN_CONTROL_CONTRACT`
(`configs/experiments/frozen_fields.yaml`), the same source
`robo/eval/pi05_eval.py` and `robo/envs/pi05_env.py` already use. The
policy client itself (`robo.policy.registry.PolicyRegistry` when present,
else `robo/eval/pi05_eval.py`'s own `ScriptedPolicy`/`ServerPolicy` classes
reused unmodified) is constructed once per scene and shared across both
conditions — literally the same Python object runs both rollouts of a
matched pair, so "identical policy" is a code-identity guarantee, not a
configuration-matching one.

### 3.4 What gets recorded per episode (step 4)

Beyond the final staged-progress summary `robo/eval/pi05_eval.py`'s
`results.json` already writes, each episode gets a gzip-compressed
`timeseries.json.gz` (`<out_dir>/episodes/<episode_id>/`) with one row per
control tick: joint position, gripper position, the raw action, **every**
free body's world pose, the live contact-pair list (body-name pairs, from
`MjData.contact`), and a snapshot of the scorer's per-stage booleans. A
compressed `video.mp4` (same exterior+wrist concatenation
`robo/eval/pi05_eval.py::run_episode` already builds) is written per
episode when `video: true`. A `robo.manifest.schema.RolloutManifest`
(`robo/manifest/io.py::write_manifest`) is written per episode under
`<out_dir>/manifests/<episode_id>/manifest.json`.

### 3.5 Resume at episode granularity (step 5)

`robo.eval.episode_log.EpisodeLedger` is an append-only JSONL keyed by
`episode_id = f"{condition}__{reset_state_id}"`. `run_matrix` loads it at
startup; any `episode_id` already present is skipped (not re-executed, not
re-appended). A restart after a mid-matrix crash therefore continues at the
next incomplete episode, using the identical persisted reset-state list —
never duplicating or reordering already-logged IDs (see
`tests/test_paired_runner.py::test_crash_and_restart_does_not_duplicate_or_reorder_episode_ids`).

### 3.6 Degenerate-run diagnostic (step 6)

`robo.eval.episode_log.check_degenerate` runs after each scene's episodes
and loudly warns (never discards data) if every scored episode came back
exactly `0.0` or exactly the max staged score — the historical failure mode
this codebase already lived through once (`docs/ROBOT.md`: "Baseline (pre
geometry fix): all zero — 32 episodes ... zero successes").

### 3.7 Five distinct outcome categories (step 7)

`robo.eval.episode_log.Outcome`: `success`, `task_failure`,
`build_failure`, `policy_timeout`, `safety_termination`, `env_crash`. Every
exception raised anywhere in an episode's lifecycle funnels through
`classify_exception`, so "failure" is never a single collapsed bucket in
the ledger.

## 4. Output layout

```
<out_dir>/
  reset_states.json          # the once-generated plan (Task 09 step 1)
  episode_ledger.jsonl       # append-only outcome ledger (resume + coverage)
  matrix_summary.json        # coverage_by_outcome + episode_ids_by_outcome
  scenes/
    <scene>__reference_scene.xml   # generated reference variant
    <scene>__<condition>_rig.xml   # full compiled rig dump, per condition
  episodes/<condition>__<reset_state_id>/
    timeseries.json.gz
    video.mp4                # if video: true
  manifests/<condition>__<reset_state_id>/
    manifest.json             # RolloutManifest, robo/manifest/schema.py
```

## 5. Validation

`tests/test_paired_runner.py` (19 tests, run via `.venv/bin/python -m
pytest -q tests/test_paired_runner.py`; needs a real GPU allocation on this
cluster for `mujoco.Renderer` — `srun -p debug --gres=gpu:a6000:1 -t 20:00
.venv/bin/python -m pytest -q tests/test_paired_runner.py`) covers: paired
reset-pose equality across differently-shaped scenes, deterministic replay
of one manifest, coverage accounting for a deliberately-broken build,
crash/restart episode-ID stability, resume-without-rerun, and independent
classification of all five non-success outcome categories.

The fast-validation command (`python -m robo.eval.paired_runner --config
configs/experiments/paired_smoke.yaml --episodes 2`) runs against the real
`outputs/c50d2d1d42_factory` scene (16 real reconstructed objects) and
produces real per-episode manifests, timeseries, and videos for both
conditions — confirmed by manually diffing a `reference`/`simany` manifest
pair for the same `(task_id, seed, ep)`: `robo.manifest.io.diff_manifests(
..., frozen_only=True)` returns `{}` (no frozen-field drift), while the
all-field diff shows exactly `scene_manifest_hash`/`scene_id`/artifact-path
fields differing — the acceptance criterion in
`plan/09_PAIRED_ROLLOUT_RUNNER.md` ("official and SimAny runs differ only
in declared scene-construction fields"), demonstrated on a real build, not
only a synthetic fixture.
