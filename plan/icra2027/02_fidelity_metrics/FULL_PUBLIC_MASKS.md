# Full-cohort SAM3 removal masks

The sealed full TRAIN preparation inventory retains all 50 scenes, 1,871 planned
objects and 399 accepted assets. The unchanged structural rule admits 36 scenes
with required support/visibility inputs and identifies ten blocked scenes. Four
zero-accepted scenes are explicit no-ops, never successful object construction.
The original first-fillable pilot 0d2ee665be already has six authenticated masks.

Freeze 20260906-18984b8-v1 therefore declares 35 new positive GPU jobs, four CPU
no-op producers, one immutable pilot reuse and ten blocked scene records. It
retains 400 planned TEST views; the blocked scenes retain 80 missing views.
No TEST pixels or model quality selected these cases. SAM3 source, checkpoint,
runtime, seed zero, IoU threshold 0.15, dilation 9, confidence 0.4 and resolution
1008 are identical to the passing real pilot. Mask quality does not replace any
planned scene or object. Later erasure/fill still require their own pilot gates.

Each positive scene invokes the existing `agents.edit.inpaint_masks` with its
`configs/experiments/icra2027/e2_full_masks/<scene>.yaml` public context and the
freeze's `contract/freeze_manifest.json`. Use the pinned SAM3 Python/source,
4 CPUs/32GB and one compatible GPU in an ordinary 30-minute job, no arrays.
The four zero-accepted scenes run the same producer on ordinary CPU jobs and
must report zero model loads and NO_APPLICABLE_VIEWS. Their output is an explicit
empty handoff for a later unchanged-background condition, not a fabricated
object recovery. All old preparation and pilot bundles remain immutable.

The expected per-scene outputs are `fidelity/removal_masks/<scene>` with exact
planned mask rows, typed failures and no-overwrite claims. Full publication
requires all 50 protocol rows to retain their declared terminal meaning.
