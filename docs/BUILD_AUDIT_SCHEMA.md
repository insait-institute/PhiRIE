# Build Audit Schema (Task 05)

`agents/eval/build_audit.py` consolidates the outputs of the existing,
disparate QA components into **one** versioned JSON per build
(`build_audit.json`, plus a human-readable `build_audit_report.md`). It does
not re-run any of that component code and does not require ScanNet++ ground
truth: every check reads whatever JSON evidence already sits under the
build directory and degrades honestly to `not_applicable` when that
evidence is absent.

```
python -m agents.eval.build_audit outputs/<scene>_factory
python -m agents.eval.build_audit outputs/<scene>_factory --out /tmp/a.json --report-out /tmp/a.md
```

Programmatic use: `agents.eval.build_audit.run_audit(build_dir) -> dict`.

## The one invariant

Every check result is exactly one of:

| status | meaning |
|---|---|
| `pass` | evidence exists and is within the accepted band |
| `warning` | evidence exists and is in a degraded-but-tolerable band |
| `fail` | evidence exists and is outside the accepted band |
| `not_applicable` | the evidence needed to compute this check does not exist (file missing, field missing, upstream component not yet built) |

**Missing evidence is always `not_applicable`, never `pass`.** Every
`check_*()` function in `build_audit.py` returns `not_applicable` as soon as
a required input is absent, *before* it ever touches a numeric threshold.
`not_applicable` is treated as the *weakest* signal in every aggregation
(see below) -- it must never be promoted to `pass`, and it must never hide a
real `fail`/`warning` sitting next to it.

## Aggregation: worst-of, never averaged

Severity order: `not_applicable`(0) < `pass`(1) < `warning`(2) < `fail`(3).

- **Per-object -> scene-level**, for each of the 12 object-scoped checks: the
  scene-level status is the *worst* status among all objects that had
  evidence for that check (`_aggregate_over_objects`). The result records
  `counts` (how many objects landed in each status) and `worst_object`, but
  never a mean/score field -- there is nothing to average, by construction.
  One failing object among ten passing ones still reports `fail` at the
  scene level.
- **Scene-level -> overall**, `overall_status` is the single worst status
  found across *all* 22 checks (`_overall_status`). A build that is mostly
  `not_applicable` (e.g. a fresh phone capture with no physics run yet) but
  has real passing evidence elsewhere reads as `pass`, not `not_applicable`
  -- `not_applicable` never outranks a genuine `pass`. Any real `fail`
  anywhere outranks everything.

## Top-level JSON shape

```jsonc
{
  "schema_version": "1.0.0",
  "generated_at": "2026-08-16T12:00:00+00:00",   // null when called as a library without `now=`
  "build_dir": "outputs/c50d2d1d42_factory",
  "scene_id": "c50d2d1d42_factory",              // build_dir basename, a label only
  "overall_status": "pass|warning|fail|not_applicable",
  "checks": { "<check_name>": { "status": ..., "reason": ..., ...evidence }, ... },  // 22 keys, scene-level
  "objects": [
    {
      "object_id": "obj_00",
      "label": "mug",
      "gt_object_id": 1,
      "tier": "A",
      "rejected": null,
      "object_status": "pass|warning|fail|not_applicable",   // worst-of this object's 12 checks
      "checks": { "<check_name>": { "status": ..., "reason": ..., ... }, ... },  // 12 object-scoped keys
      "evidence": { "aligned_json": "objects/obj_00/aligned.json", ... }          // relative FILE PATHS, not inlined data
    }
  ],
  "repair_queue": [ { "object_id": "obj_00", "stage": "scale_sanity", "reason": "..." } ],
  "sources_present": { "report.json": true, "eval_vs_gt.json": false, ... }
}
```

Per instruction 4 in `plan/05_SCENE_QUALITY_AUDIT.md`, per-object and
scene-level entries carry small numeric evidence and *paths* to the diagnostic
renders/JSON that produced them (`evidence.*`, `checks.*.source`), never the
large upstream blobs (`verify_report.json`'s `detail`, `render_metrics_v2.json`'s
per-frame `frames` array, etc.) inlined into the audit.

## Versioning

`SCHEMA_VERSION` is a plain `MAJOR.MINOR.PATCH` string, independent of the
pipeline/scene versioning. Bump MAJOR on a breaking key rename/removal,
MINOR on adding a new check or field, PATCH on a threshold or wording tweak.
Current: `1.0.0` (initial).

## The 22 checks

Object-scoped checks are aggregated per-object first, then worst-of'd to a
scene-level entry under the same key in `checks`. Scene-only checks have no
per-object counterpart.

### 1. Pose/reconstruction coverage and held-out rendering

| check | scope | source | not_applicable when |
|---|---|---|---|
| `reconstruction_coverage` | scene | `objects/objects.json` + each `objects/obj_XX/aligned.json`'s `rejected` field | `objects.json` lists zero discovered objects |
| `held_out_rendering` | scene | `render_metrics_v2.json` (`agents/eval/factory_eval_render.py`), `mean.twin`/`mean.bg` | file missing, or `mean.twin.psnr` missing |

`reconstruction_coverage` = fraction of discovered objects that reached an
*accepted* registration (`aligned.json` exists and `rejected` is falsy).
Thresholds (new; no repo precedent for this specific ratio):
pass `>= 0.85`, warning `>= 0.50`, fail `< 0.50`.

`held_out_rendering` reads the composited-twin PSNR against the held-out
DSLR test split and the background-splat-alone PSNR as a reference. The
absolute floor for the **background** splat, `PSNR >= 26 dB`, is not a new
number -- it is the existing zero-shot-100 scene-inclusion bar documented in
`docs/GAP_STUDY.md` ("a trained splat must exist with PSNR >= 26 dB"),
reused here (`BG_PSNR_INCLUSION_DB`) as the twin's warning floor too, since a
build whose composited twin can't clear the bar its own background splat
needed to clear in the first place is clearly regressing. The rest --
absolute fail floor `20 dB`, and the bg-vs-twin delta bands (warn `>4dB`,
fail `>8dB`) -- are new, task-authored thresholds: no prior number in the
repo compares twin-vs-bg PSNR delta, so a pragmatic floor plus a regression
budget was chosen rather than reusing an unrelated number.

### 2. Duplicate/missing instance indicators

| check | scope | source | not_applicable when |
|---|---|---|---|
| `duplicate_instances` | scene | `eval_vs_gt.json` (`agents/eval/eval_vs_gt.py`), or a GT-free heuristic over `objects.json`+`aligned.json` | no accepted objects with pose data AND no `eval_vs_gt.json` |
| `missing_instances` | scene | `eval_vs_gt.json`'s `n_gt_missed`/`n_gt_whitelist` | `eval_vs_gt.json` missing, or `n_gt_whitelist == 0` |

`duplicate_instances` prefers `eval_vs_gt.json` (GT-based, authoritative): if
the same `matched_gt_id` is claimed by more than one tier-A/B object, that is
a hard `fail`. When no GT evaluation exists for the build (the normal case
for a phone/video capture -- the acceptance criterion is "runs without
ScanNet++ GT"), it falls back to a GT-free heuristic: same label + world
positions within `DUPLICATE_HEURISTIC_DIST_M` (new threshold, 5cm, chosen as
a plausible object-footprint radius) among accepted objects. Because this is
a heuristic, not ground truth, it is capped at `warning` and can never reach
`fail` on its own.

`missing_instances` has no GT-free fallback (there is no way to know what
was missed without a ground-truth instance count) and is honestly
`not_applicable` for every build without `eval_vs_gt.json`. Thresholds:
`n_gt_missed == 0` -> pass; missed fraction `<= 0.30` -> warning (new,
task-authored); else fail.

### 3. Registration residual and candidate disagreement

| check | scope | source | not_applicable when |
|---|---|---|---|
| `registration_residual` | object | `objects/obj_XX/aligned.json`'s `chamfer_med_m`, `icp_tilt_deg` | `aligned.json` missing or missing `chamfer_med_m` |
| `candidate_disagreement` | object | `objects/obj_XX/hybrid.json`'s `sym_chamfer_{trellis,rvg,sam3d}_m` (`agents/assets/factory_hybrid.py`) | `hybrid.json` missing (single-candidate run), or fewer than 2 candidates scored |

`registration_residual` bands the ICP chamfer residual in mm: pass `<=20mm`,
warning `<=40mm`, fail `>40mm`. These are *not* new numbers -- they mirror
`TIER_A_F1_20`/`TIER_B_F1_40` (0.40 / 0.20 F1 at the 20mm/40mm radii) already
frozen in `agents/assets/factory_align.py`, reused here for scale consistency
even though chamfer-at-a-distance and F1-at-a-radius are different
statistics: the repo already treats 20mm/40mm as the meaningful precision
bands for this pipeline, and inventing a third, unrelated mm scale here
would only add confusion.

`candidate_disagreement` only has evidence when
`SIMANY_HYBRID_CANDIDATES` was used for that object (multi-candidate
registration contest). Spread = max-min of the scored candidates' symmetric
chamfer, in mm. Thresholds are new (`CANDIDATE_DISAGREEMENT_WARN_MM=10`,
`..._FAIL_MM=30`), loosely anchored to the same 20/40mm registration bands
above for consistency, since large spread between candidates is exactly the
same kind of "how far off is the geometry" signal, just measured between two
predictions instead of prediction-vs-observation.

This category also carries a related, independently-computed check that the
plan's "registration residual" bullet subsumes but that deserves its own
identity (see the acceptance criterion that ghost/scale/etc. fixtures must
each trip a *distinct* check, not a shared blob):

| check | scope | source | not_applicable when |
|---|---|---|---|
| `scale_sanity` | object | `aligned.json`'s `size_ratio_vs_obs` | field missing |

`scale_sanity` is deliberately computed **independently of** the upstream
`rejected` string (which may or may not mention size) and independently of
`registration_residual`'s chamfer value: it only looks at
`size_ratio_vs_obs` against `SIZE_RATIO_RANGE = (0.4, 2.5)`, which is not a
new number -- it is `agents/assets/factory_align.py`'s `SIZE_RATIO_RANGE`
constant, reused verbatim. Warning band: within 10% of either edge of that
range (`SCALE_SANITY_INNER_MARGIN_FRAC`, new, task-authored -- a round,
conservative margin so borderline-legal scales get flagged before they
become borderline-illegal on the next run).

### 4. Visual-to-collision surface distance

| check | scope | source | not_applicable when |
|---|---|---|---|
| `visual_collision_surface_distance` | object | `objects/obj_XX/{trellis_mesh,mesh_sim}.ply` vs `objects/obj_XX/collision/part_*.obj` (CoACD output, `agents/assets/s6_physics.py`) | either mesh side is missing, or `trimesh`/`scipy` are unavailable, or the computation raises |

No existing component computes this distance; `s6_physics.py`'s CoACD stage
produces the collision parts but never checks them against the visual mesh.
This is the one check that does real (if lightweight) geometry: it samples
the visual surface, samples the union of collision parts, and takes the 95th
percentile of nearest-neighbour distance from visual to collision (mm).
Thresholds are new and task-authored: pass `<=5mm`, warning `<=15mm`, fail
`>15mm` -- a visual surface should sit close to its own collision proxy; 5mm
is roughly print/scan-noise scale, 15mm is where a robot's contact model and
what it looks like meaningfully diverge. Not one of the 5 mandatory
synthetic fixtures (no mesh geometry is shipped in the fixtures), so this
check exercises only its `not_applicable` path in tests; it is written to
run for real once meshes/collision parts exist in a live build.

### 5. Removal alpha/depth residual and residual re-detection

| check | scope | source | not_applicable when |
|---|---|---|---|
| `removal_alpha_coverage` | object | `inpaint/verify/verify_report.json`'s `summary[].alpha_cov`, or `inpaint/verify/verify_render.json`'s per-view `alpha_cov` (mean) if stage 2 was never run | neither file has an `alpha_cov` for the object |
| `removal_depth_residual` | object | same files, `depth_plane_mm` / `depth_vs_plane_med_mm` | neither file has a depth value |
| `removal_redetection` | object | `verify_report.json`'s `det_after`/`removed_ok` (`agents/eval/verify_removal_check.py`) | `verify_report.json` missing or has no `det_after` (SAM3 stage 2 not run) |

`agents/eval/verify_removal_render.py` (stage 1) writes alpha coverage and
depth-vs-support-plane residuals; `agents/eval/verify_removal_check.py`
(stage 2) adds the SAM3 residual re-detection score and merges stage 1's
metrics into `verify_report.json`. The audit prefers the stage-2 file
(has everything); if only stage 1 ran, it averages stage 1's per-view fields
itself so alpha/depth checks still work with only `verify_render.json`
present -- but `removal_redetection` needs `det_after`, which only stage 2
produces, so it is honestly `not_applicable` until stage 2 runs.

Thresholds: `removal_alpha_coverage` pass `>=0.98`, warning `>=0.90`, fail
`<0.90` (new, task-authored -- coverage should be near-total, since the
metric *is* "did the hole get filled"). `removal_depth_residual` pass
`<=15mm`, warning `<=40mm`, fail `>40mm` (new, anchored to the same
15/40mm order of magnitude used for `visual_collision_surface_distance` and
`registration_residual` respectively, for a consistent mm vocabulary across
the audit). `removal_redetection`'s fail boundary, `0.5`, is **not** new --
it is `verify_removal_check.py`'s own `removed_ok = max(s_after) < 0.5`
cutoff, reused verbatim; the warning band at `0.3` is new (task-authored),
giving visibility into borderline cases that the existing binary flag would
otherwise report as a clean success right up to the 0.5 cliff.

### 6. Penetration, support overlap, settle drift, drop response

| check | scope | source | not_applicable when |
|---|---|---|---|
| `penetration` | object | drop-test record's `sunk` field | no drop-test record has a `sunk` field |
| `drop_response` | object | drop-test record's `stable` field | no drop-test record has a `stable` field |
| `settle_drift` | object | drop-test record's `drift_m` field | no drop-test record has a `drift_m` field |
| `support_overlap` | object | `inpaint/obj_XX/plane.json`'s `trim_ok` (`agents/edit/inpaint_prepare.py`) | `plane.json` missing, or missing `trim_ok` |

"Drop-test record" = `drop_v2.json`'s per-object entry
(`robo/sim/redrop.py`, link-frame-corrected) when present, else
`report.json`'s embedded `drop_test` (`agents/eval/factory_report.py`);
`drop_v2.json` supersedes `report.json` when both exist, per `redrop.py`'s
own docstring policy.

These three read *different fields of the same upstream record* on purpose:
`sunk` (interpenetrated the support plane and kept falling) and `stable`
(the overall drop-test verdict, `drift < 3cm and not sunk`) are logically
coupled by the source formula -- a sunk object is always also unstable -- but
they are still surfaced as two distinctly-named, distinctly-reasoned checks
rather than collapsed into one, because a repair agent (Task 13) needs to
know *which* problem it is looking at. `settle_drift` is the one check in
this group that is **not** coupled to the other two: it grades the raw drift
magnitude on its own scale, so an object that sinks with near-zero drift
(the mandatory `penetration` fixture) shows `settle_drift: pass` right next
to `penetration: fail` -- proof the two are independently computed, not
aliases of each other.

Thresholds: `penetration`/`drop_response` reuse `agents/eval/factory_report.py`'s
own cutoffs verbatim (`sunk` at `z < -0.05m`, `stable` at `drift < 0.03m`).
`settle_drift`'s own pass/warning boundary, `10mm`, is new (task-authored);
its warning/fail boundary reuses the existing `0.03m` stable cutoff so the
`fail` band lines up exactly with the point where the upstream `stable`
flag would also flip.

`support_overlap` reads `plane.json`, the support-plane fit computed at the
object's *original* scene location (during removal-verification prep, not
the isolated per-object drop test). `trim_ok: true` means the ring-around-
the-footprint plane fit converged cleanly; `false` means it fell back to a
contaminated one-shot fit -- a real, distinct "the object's actual support
surface cannot be confidently modeled" signal, hence `fail`. Absence of
`plane.json` is intentionally `not_applicable`, not `fail`:
`inpaint_prepare.py` also skips writing this file for legitimately
floor-standing/shelved objects with no ring to fit (see its "no support
ring ... floor-standing or shelved object; skip plane fill" log line), and
the audit cannot distinguish that benign case from removal verification
never having run at all -- so it declines to guess rather than risk a false
`fail`.

### 7. Room-collision coverage and disconnected floating components

| check | scope | source (EXPECTED, not yet produced) | status today |
|---|---|---|---|
| `room_collision_coverage` | scene | `room_collision_report.json`.`coverage_fraction` | `not_applicable` |
| `room_disconnected_components` | scene | `room_collision_report.json`.`disconnected_components` | `not_applicable` |
| `room_penetration_max` | scene | `room_collision_report.json`.`penetration_max_mm` | `not_applicable` |
| `room_settle_drift` | scene | `room_collision_report.json`.`settle_drift_mm` | `not_applicable` |

**Pending Task 06** (`robo/sim/room_collision.py`, built concurrently with
this task). `build_audit.py` never imports that module; it only reads
`<build_dir>/room_collision_report.json` with `.get()` against the expected
field names given in the Task 05 course-correction brief. Until that file
exists, all four checks report `not_applicable` with a reason string naming
Task 06 explicitly, so a reader of the audit knows this is a *missing
upstream component*, not a build defect. Thresholds are pre-registered so no
further threshold-design work is needed once Task 06 lands: `coverage_fraction`
pass `>=0.85`/fail `<0.50` (same bands as `reconstruction_coverage`, for a
consistent "fraction covered" vocabulary); `disconnected_components` pass
`0`/warning `1-2`/fail `>=3`; `penetration_max_mm` pass `<=2mm`/warning
`<=10mm`/fail `>10mm`; `settle_drift_mm` pass `<=5mm`/warning `<=20mm`/fail
`>20mm` (tighter than the per-object `settle_drift` bands above, since this
is multi-object *room-scale* rigid settling, where accumulated drift across
several contacting bodies should be smaller per-body than one object
dropped in isolation). All four are new, task-authored, forward-declared
thresholds -- there is no repo precedent for room-scale collision QA yet.

### 8. Metric scale and robot-alignment status

| check | scope | source (EXPECTED, not yet produced) | status today |
|---|---|---|---|
| `metric_scale` | scene | `alignment_report.json`.`metric_scale_status` | `not_applicable` |
| `robot_alignment` | scene | `alignment_report.json`.`robot_alignment_status` | `not_applicable` |

**Pending Task 04** (`agents/recon/robot_align.py` /
`agents/recon/alignment_report.py`, per `plan/04_METRIC_SCALE_ROBOT_ALIGNMENT.md`
-- neither file exists in the repo yet). Expected `alignment_report.json`
shape, per that plan's "Outputs" section ("Report scale factor and CI,
rotation/translation residuals, reprojection residual, floor-normal error,
table-height error, and held-out marker/trajectory error"):

```jsonc
{
  "scale_factor": 1.0, "scale_ci": [0.98, 1.02],
  "rotation_residual_deg": 0.3, "translation_residual_mm": 4.0,
  "reprojection_residual_px": 1.2,
  "floor_normal_error_deg": 0.5, "table_height_error_mm": 3.0,
  "held_out_error_m": 0.01,
  "metric_scale_status": "pass|warning|fail",      // Task 04's own verdict
  "robot_alignment_status": "pass|warning|fail"     // Task 04's own verdict
}
```

Rather than re-deriving pass/warning/fail bands for seven heterogeneous
physical quantities it did not design, `build_audit.py` reads Task 04's
*own* self-classification (`metric_scale_status`/`robot_alignment_status`)
directly when present and valid, and reports `not_applicable` (never a
guess) when the file or those exact fields are absent. This is the
correct, honest degrade for a check whose owning component has not been
designed yet -- inventing thresholds for numbers no other code in the repo
has ever produced would be indistinguishable from a magic number with no
rationale, which the task instructions explicitly rule out.

## `repair_queue`

One entry per **`fail`** occurrence (never `warning`/`not_applicable` --
Task 13 should not be queued work for a check that had no evidence, and
`warning` is "keep an eye on it", not "go fix it"):

```json
{"object_id": "obj_03", "stage": "penetration", "reason": "object sank below the support plane during settle (z < -0.05m)"}
```

`object_id` is the `obj_XX` id for object-scoped failures. For scene-only
failures (e.g. `duplicate_instances`) that name specific implicated objects,
one entry is queued per implicated object with `object_id` set accordingly;
scene-only failures with no specific object (e.g. `held_out_rendering`) use
`object_id: "scene"`. `stage` is the check's key name (stable API for Task
13 to switch on). `repair_queue` never modifies any pipeline output in
place -- it is purely descriptive.

## Test fixtures

`tests/data/builds/audit_fixture/<scenario>/` (regenerated by
`gen_fixtures.py`, not checked in as a generator -- the JSON trees
themselves are what's committed):

| scenario | trips |
|---|---|
| `happy_path` | everything computable passes; the two pending-upstream categories are `not_applicable` |
| `ghost_object` | `ghost_object` only |
| `penetration` | `penetration` + `drop_response` (coupled by the real upstream formula), NOT `settle_drift` |
| `missing_support` | `support_overlap` only |
| `bad_scale` | `scale_sanity` only |
| `transparent_residue` | `removal_alpha_coverage` + `removal_depth_residual` + `removal_redetection` (all three, independently reasoned) |
| `mixed_scene` | 3 clean objects + 1 penetrating one, for the "not averaged away" test |
| `no_evidence` | zero objects, zero other files -- every one of the 22 checks must read `not_applicable` |
