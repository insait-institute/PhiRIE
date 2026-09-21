# First full-cohort scene: SAM3 stage

This stage preserves the first lexicographic full-cohort pilot `09c1414f1b`.
Preparation job `834231` completed in 50 seconds on `sof1-h200-5` using the
existing producer at `776d37c045fe63647e10b246d0c4ea512ab90a67`.
The authenticated result contains 112 planned objects, nine accepted objects,
27 selected TRAIN views, one accepted object without a support plane, and no
object without a selected view. The sealed preparation and independent consumer
check are bound in the new context and protocol.

The SAM3 stage uses the unchanged checkpoint, runtime, confidence, mask union,
IoU threshold, and dilation from the prior compact pilot. All 27 selected views
remain planned; the three views belonging to the object without a support plane
cannot become fillable-background success. The scene is not replaced because
this implies a future complete-background blocker. Mask engineering integrity
and complete background availability are separate gates.

```bash
SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny SIMANY_AUTO=1 SIMANY_NO_GT=1 \
SIMANY_MESH_SRC=derived SIMANY_SCENE=09c1414f1b PYTHONPATH=. \
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/group/worldcept/code/SimAny/.envs/sam3d-objects/bin/python -m agents.edit.inpaint_masks \
  --public-context configs/experiments/icra2027/e2_full_factorized_masks/pilot.yaml \
  --contract-manifest /group/worldcept/code/SimAny/outputs/icra2027/20260906-02e9d33-v1/contract/freeze_manifest.json
```

Output: `20260906-02e9d33-v1/fidelity/removal_masks/09c1414f1b/`.
No TEST RGB, metric output, object generation, or background optimization enters
this stage. The subsequent LaMa and fill contexts require new freezes after
predecessor seals. This commit does not admit a full SAM3 cohort run.
