# Full-input Gaussian background fill pilot

The frozen originalTRAIN prefix chose0d2ee665be before model or TEST quality.
SAM3 and LaMa completed all6 planned views with exact exterior preservation.
This new freeze20260906-860f3fc-v2 retains29 planned objects and2accepted assets and invokes
the existing strict fill producer for exactly1500 iterations. Grid, carving,
primary-view supervision, seed0 and all thresholds remain unchanged.
The mini-viewer runtime and immutable Pydantic overlay are identical to the
completed compact fill producer. No raw-background or missing-erasure fallback
is permitted. Diagnostic hole PSNR is TRAIN erasure-target fit only.

Run agents.edit.inpaint_fill with --public-context
configs/experiments/icra2027/e2_full_model_fill/pilot.yaml and
--contract-manifest /group/worldcept/code/SimAny/outputs/icra2027/20260906-860f3fc-v2/contract/freeze_manifest.json.
Use the exact mini-viewer interpreter, source PYTHONPATH plus the pinned overlay,
and SIMANY_FILL_DEPENDENCY_OVERLAY from the config. One ordinary GPU job uses
4CPUs/32GB/30min. Expected fidelity/background_fill/0d2ee665be/clean_background.ply.
Only authenticated complete fill output can release full background model jobs.
