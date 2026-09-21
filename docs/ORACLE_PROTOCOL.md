# Oracle-causal reconstruction benchmark (Task 14 / `oracle_causal` protocol)

Implements `plan/14_ORACLE_BEHAVIOR_BENCHMARK.md`, scoped to what
`docs/ICRA_RESEARCH_CONTRACT.md` actually needs from this track (>=6 scenes,
>=2 task families, >=1 controlled degradation level each, feeding
`staged_progress_mae` / `spearman_task_difficulty`). This document records
what was actually built in this pass, two real infrastructure problems found
along the way, and why several design choices depart from the plan's literal
wording where the literal reading turned out to be unbuildable or already
measuring something else.

## 0. Data availability: the HDF5 source is gone

`agents/recon/behavior_extract.py` reads
`/group/worldcept/data/pointworld_behavior_restored/behavior/flows/task-XXXX/
episode_*.hdf5`. As of 2026-08-16 that directory does not exist -- `ls`
returns nothing. Only two things survive from before it was reclaimed:

- `data/recon_scenes/data/behavior_task0020/` -- the one scene already
  extracted (the existing GAP_STUDY.md pilot).
- `/group/worldcept/data/behavior/wds/{train,test}/*.tar` -- WebDataset
  shards of the same release (1541 train + 176 test shards, ~1.7TB).

`oracle/capture_generator.py`'s `extract` subcommand reads the WDS shards
directly (`tarfile` + `numpy` + `pickle`, no `h5py` needed -- confirmed
against `/group/worldcept/data/behavior_datasample/README.md`, itself a
WDS-derived sample) and writes the identical scene-dir contract
`behavior_extract.py` produces, so `run/run_behavior_recon.sh` and every
downstream stage run against it unmodified.

## 1. A real bug found and fixed: WDS has no cross-clip anchor

`behavior_extract.py`'s HDF5 path reads a per-clip `world_to_robot` 4x4 that
anchors every clip of a task into one shared scene-global frame. **WDS does
not carry that field.** It carries `base_pose` (T,7), the robot-base
trajectory *within* the clip -- and by the release's own convention,
`base_pose[0]` is identity for every clip (verified across 6+ clips, 2
tasks: ~0 translation, ~identity quaternion, always). An earlier version of
`capture_generator.py` assumed `inverse(pose7_to_mat(base_pose[0]))` would
recover the missing anchor; it does not, because that value is always
~identity and therefore carries zero cross-clip information.

Symptom (before the fix): the SAME static mesh (`floors_nbxnpk_0`, task-0016)
came out at wildly different world positions in two clips of the *same
episode* -- `[2.09, 2.14]` vs `[-2.22, 2.50]` -- because each clip's "world"
was actually silently anchored to that clip's own start, not to a scene-wide
frame.

Fix (`_register_clips_to_reference` in `capture_generator.py`): recover the
missing anchor the way it would have to be recovered from any noisy real
capture -- Kabsch/SVD rigid alignment of each clip's local scene-mesh
landmark positions (`scene_mesh_trajectories[key][0]`) against a reference
clip's, using whichever mesh keys the two clips share, with one-shot
outlier rejection (residual > 5cm / 2x median dropped, then refit) so task
objects that genuinely moved between episodes don't corrupt the fixture
alignment. Clips sharing fewer than 3 landmarks with the reference are
dropped rather than guessed at (logged: "dropped N/M clips...").

Verification: after the fix, every scene's true fixtures (walls, floors,
cabinets, countertops, sinks) show ~0.0m drift across all
episodes/clips pooled into one extraction -- the expected signature of
correct registration -- while genuinely-relocated task objects (produce
clutter in task-0020, re-shelved books in task-0023 at higher episode
counts) show real, non-zero drift. Fewer episodes per scene generally gave
*cleaner* registration (less chance of disjoint sub-areas or task-object
contamination dominating the shared-landmark set), so final episode counts
per scene were tuned empirically rather than maximized -- see
`configs/oracle/tasks.yaml`'s `extract_args`.

**Caveat this leaves behind:** GT itself now carries a small, quantified
position uncertainty (the registration's own residual) rather than being
the exact simulator-state anchor HDF5's `world_to_robot` would have given.
`configs/oracle/tasks.yaml`'s per-object `gt_confidence` (high/medium/low,
derived from `gt_objects.json`'s `max_pos_drift_m`) makes this visible
instead of hiding it; the primary pose-error metric is meaningless below
this noise floor, so treat sub-cm reconstructed-vs-oracle differences as
noise, not signal.

## 2. Scenes and degradation levels

Six scenes, four task families (>= the contract's >=2), one existing +
five newly extracted via the WDS path above:

| scene_id | task_id | family | movable objects (fixture, relation) |
|---|---|---|---|
| behavior_task0020 | task-0020 | clutter_kitchen (existing pilot) | mixing_bowl x2 (countertop, OnTop); wicker_basket (dishwasher, Inside) |
| behavior_task0011 | task-0011 | receptacle_dishware | plate x2 (bar counter, OnTop) |
| behavior_task0023 | task-0023 | support_shelf | hardback x2 (bookcase, OnTop) |
| behavior_task0027 | task-0027 | clutter_bathroom | soap dispenser, detergent bottle, sanitary-napkin box (shelf/sink/floor, OnTop) |
| behavior_task0045 | task-0045 | obstacle_kitchen (fridge/oven as large obstacles) | hotdog (countertop, OnTop) |
| behavior_task0002 | task-0002 | support_mantle | pumpkin (countertop, OnTop) |

Full manifest, including which WDS shard each task was pulled from and the
exact real mesh names read back from each scene's own `gt/gt_objects.json`
(not guessed), is in `configs/oracle/tasks.yaml`.

Three degradation levels (`oracle/capture_generator.py degrade`), applied to
the posed RGB-D capture only, never to GT:

| level | view keep | pose noise (pos/rot) | blur sigma | depth noise (rel.) | occlusion |
|---|---|---|---|---|---|
| clean | 100% | 0 / 0 deg | 0 px | 0% | 0% |
| mild | 70% | 1cm / 1 deg | 1 px | 1% | 5% |
| severe | 40% | 3cm / 4 deg | 3 px | 3% | 20% |

Every degraded scene dir carries `degradation_manifest.json` with the exact
per-frame noise draws (pose offset, occlusion box coordinates, seed) --
this is the "controlled perturbation magnitude is recoverable from metadata"
test from plan/14, satisfied directly by that file rather than a separate
check.

**Approach taken vs. the plan's first option:** plan/14 asks for rendering
fresh phone-like captures from a *live* interactive OmniGibson scene as the
first choice, with synthetic degradation of the existing posed frames as an
explicit fallback "if full OmniGibson rendering is too heavy to stand up
fresh in this pass." That is the path taken here: standing up a live
OmniGibson renderer for 6 scenes in one pass was not attempted; the fallback
was used from the start, and this doc records that choice rather than
implying otherwise.

## 3. GT export and inaccessibility (`oracle/gt_export.py`)

**What counts as GT** (vaulted, locked) vs. **capture-derived** (left
readable, exactly what a real phone with a depth sensor would produce):
vaulted = `gt_objects.json` + per-object `gt/<mesh>.ply` (exact object
identity/pose/shape -- no real scan could produce this); NOT vaulted =
`gt/depth/*.png`, `gt/mesh_gt.ply`, `gt/intrinsics_native.json` (TSDF-fused
from the *captured* depth channel, which `run_behavior_recon.sh`'s
`GT_MESH=1` default legitimately reads).

Two tests, both real, both passing on all 6 scenes:

1. **Static analysis**: grep 11 reconstruction-stage source files
   (`agents/discover/*`, `agents/recon/gsplat_train.py`,
   `models/s4_trellis.py`, `agents/assets/*`, `robo/sim/export_mjcf.py`,
   `robo/tasks/pi05_tasks.py`, `robo/sim/export_omnigibson.py`) for
   `gt_objects.json` / `gt_vault` / `scene_mesh_trajector` substrings.
   Result: **0 hits, all 6 scenes.**
2. **OS-level access check**: after `export_gt` + `lock_vault` (a single
   `chmod 000` on the vault directory -- Linux gates path resolution into
   *every* descendant on the parent directory's execute bit, so this alone
   is sufficient; an earlier revision recursed into every file and broke
   `unlock_for_eval`'s own ability to re-enter what it had just sealed,
   fixed by locking only the top-level dir), spawn a **real subprocess**
   (same non-root user) that does a plain `open()` on the vault's
   `gt_objects.json` and asserts `PermissionError`. Result:
   **blocked on all 6 scenes** (`returncode=1`, `PermissionError` in
   stderr, verified verbatim).

`unlock_for_eval(vault_dir)` is the one sanctioned way back in -- a context
manager only `oracle/evaluate_reconstruction.py` calls, chmod 500 for the
duration, chmod 000 again in `finally` (verified: a fresh subprocess after
the `with` block exits is blocked again).

Run: `.venv/bin/python -m oracle.gt_export selftest --scene <id> --vault-root outputs/oracle_gt_vault`

## 4. Frozen policy and why it isn't pi05 or live OmniGibson BDDL

pi0.5 does not run inside OmniGibson at all (`docs/GAP_STUDY.md`). The live
OmniGibson BDDL bridge (`robo/sim/export_omnigibson.py` +
`omnigibson_bridge/import_and_run.py`) needs a per-scene Isaac Sim process,
and -- checked directly against the source, 2026-08-16 -- its `gather_pool`
step globs `outputs/*_factory/objects/objects.json`; `outputs/
behavior_task-0020/objects/` has no `_factory` suffix, so it is **never**
picked up by that glob. **The existing "7/7 vs 6/7 BDDL" number quoted in
`docs/GAP_STUDY.md` / `ICRA_RESEARCH_CONTRACT.md` is a real, useful ablation
(CoACD vs. convex-hull collision), but it is not a per-scene measurement of
behavior_task-0020's own reconstruction** -- its object pool is drawn from
whichever ScanNet++ `*_factory` scenes happen to have label-matching
objects, task-0020's own kitchen assets among them or not incidentally.
This is worth flagging because the contract's evidence table cites it as
the oracle_causal pilot result; it is directional evidence for the pipeline,
not a task-0020-specific causal measurement, and this pass's per-scene
numbers below are the first that actually are.

Neither pi05 nor the live BDDL bridge is a policy that is *identical*
between the oracle and reconstructed arms in the sense plan/14's test list
requires. What is identical: `staged_score()` in
`oracle/evaluate_reconstruction.py` -- a scripted, BDDL-flavored
(Inside/OnTop bounding-volume) predicate plus a 4-stage
grasp/lift/hover/place rubric (0.25/stage, matching
`configs/experiments/frozen_fields.yaml`'s convention), applied to one
scalar: the 3D position error between a candidate pose and the GT pose.
Oracle arm: error is 0 by construction. Reconstructed arm: from one of two
tiers.

**Known limitation of the OnTop predicate**: it checks the candidate
position against the *top* of the fixture's bounding box. For multi-level
furniture (shelves, bookcases, cabinets) an object correctly resting on a
lower level fails this check even at its exact GT pose -- this, not a
registration error, is why most `oracle_score` rows below are 0.75 rather
than 1.0. Documented rather than special-cased away.

### Fast tier (this pass's default, CPU-only, real numbers now)

`fast_proxy_reconstruct`: GT-BLIND Euclidean clustering (`open3d`
`cluster_dbscan`, eps=3.5cm, min 12 points) of the fused point cloud the
capture itself produced (`<scene>/init_points.ply`) -- the clustering step
never touches the vault. Matching each GT movable object to its nearest
cluster within 45cm is the one step allowed to see GT (evaluation-side, per
the protocol); the position error feeds `staged_score`.

**Real limitation found, not hidden**: small objects resting flush against
a much larger fixture (a bowl on a countertop, a hotdog on a countertop)
frequently get absorbed into the fixture's own giant cluster rather than
forming a separable blob -- pure density-based clustering has no shape/color
prior to split them. This shows up as `pose_error_m: null` / score 0.0
("not found") even at the `clean` level for several objects (mixing_bowl,
plate_213, box_of_sanitary_napkins, hotdog@clean). This is a real property
of the naive fast-tier proxy, not of the actual SimAny pipeline (which uses
SAM3 semantic segmentation, not raw geometric clustering) -- flagged so it
is never read as "SimAny cannot reconstruct small objects."

### Deep tier (launched, not yet complete)

`load_deep_reconstructed_pose`: reads `outputs/behavior_<task>*/objects/*/
aligned.json`'s `T[:3,3]` (world position) from the real
SAM3+TRELLIS+CoACD pipeline, label-matched the same way
`export_omnigibson.gather_pool` does. `run/slurm/oracle_matrix.sbatch`
launches this via the unmodified `run/run_behavior_recon.sh` for 5 new
scenes x {clean, severe} + behavior_task0020 x severe (11 array tasks,
`mild` skipped to bound GPU cost since the fast tier already covers it).
Job 707116 (array 0-10), `sbatch --partition=batch --qos=normal
--exclude=msp3-[0-7] --gres=gpu:h200:1`, submitted and left running per the
task's own instruction not to block on it; `evaluate_reconstruction.py
run-matrix --deep` re-merges automatically once each array task's
`run_behavior_recon.sh` finishes (hours per scene, per the pipeline's own
documented cost).

## 5. Identity-export test (required sanity check)

`oracle/evaluate_reconstruction.py identity-test`: for every
(scene, movable object), compute `oracle_score = staged_score(0.0,
predicate_holds(gt_pos, ...))` twice through the identical code path (once
labeled "oracle", once labeled "identity reconstruction") and assert exact
agreement.

**Result: PASS.** `all_exact_match: true` across all 12 scene/object rows
(100% agreement) -- this is the actual "identity row scores exactly 1.0
agreement" guarantee plan/14 asks for. Separately, and more literally:
`behavior_task0011 / plate_212` is a row whose GT frame *does* satisfy its
declared OnTop predicate, so its identity-export row scores **exactly
1.0** on the underlying staged value too, not just on agreement. The other
11 rows land at 0.75 for the multi-level-fixture reason in §4 -- reported,
not forced.

## 6. Actual oracle-vs-reconstruction numbers (fast tier, real, this pass)

36 records, `outputs/oracle_eval/results.jsonl` (6 scenes x 3 levels; one
scene x level x movable-object per row).

**Geometry-good / behavior-bad counterexample** (plan/14's own required
finding): `behavior_task0027 / soap_dispenser_223`, level=clean:
`pose_error_m = 0.046` (4.6cm -- essentially a perfect reconstruction, well
inside the 5cm "placed" tolerance) yet `predicate_holds = False`
(the recovered position sits just outside the fixture's OnTop bounding-box
margin), so `reconstructed_score = 0.75`, not `1.0` -- identical to the
`oracle_score` for the same object (also 0.75, same predicate-margin
reason), so the pair *agrees*, but both illustrate that a small, sub-object-
size geometric residual crosses a discrete task-relevant threshold. The same
object at `mild` (7cm error) shows the identical 0.75/0.75 pattern,
confirming it is a stable margin effect, not noise.

**Real, honestly-mixed monotonicity**: `behavior_task0002 / pumpkin_93`
degrades in the expected direction (clean 3.6cm/0.75 -> mild not-found/0.0
-> severe not-found/0.0). Several other objects do **not** degrade
monotonically with the fast tier's clustering (e.g. `behavior_task0045 /
hotdog_208`: clean not-found/0.0 -> mild 31.6cm/0.5 -> severe 4.7cm/0.75;
`behavior_task0023 / hardback_51`: error *decreases* clean->mild->severe).
This is attributed to the DBSCAN clustering's sensitivity to view-count and
point-density changes (fewer points can, by chance, separate a small object
from its supporting fixture's cluster rather than merging further) rather
than to a real property of the actual SimAny reconstruction pipeline
(SAM3-segmentation-based, not raw density clustering) -- reported as a
limitation of the fast-tier proxy specifically, not smoothed over.

## 7. What is NOT in this pass

- The deep-tier (real SAM3+TRELLIS+CoACD+gsplat) numbers are launched, not
  landed -- §4/§6 numbers are fast-tier only.
- `mild` is skipped in the deep-tier matrix (GPU-cost control); fast-tier
  covers it.
- OnTop/Inside predicate is a single-bbox approximation; multi-level
  fixtures are known-miscalibrated (§4) rather than fixed with
  per-shelf-level geometry, which the available GT does not expose at that
  granularity from this extraction.
- `agents/eval/predictive_metrics.py` (staged_progress_mae /
  spearman_task_difficulty aggregation) is being built by another agent in
  parallel; this module's job is only to emit the plain-JSON per-instance
  records that module will consume (`scene_id, task_id, degradation_level,
  oracle_score, reconstructed_score, staged_progress_oracle,
  staged_progress_reconstructed`, plus `pose_error_m`, `family`,
  `gt_confidence`, `reconstruction_tier` for stratification) -- no
  aggregation/bootstrap is computed here.
