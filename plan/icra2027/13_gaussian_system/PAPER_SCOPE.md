# Locked paper direction and claim boundaries

The submission remains a robotics systems paper: observation-derived, editable,
semantically addressable Gaussian simulation and controlled robot evaluation.
No physical-robot experiment, sim-to-real performance or dynamics-identification
claim is required. Keep real-video/real-scan evidence, not only synthetic scenes.

## Intended claims, each contingent on experiment evidence

C1. RGB-video-to-interactive-scene construction. Raw RGB video means cameras/depth
are estimated, not supplied by the simulator. Intrinsic calibration and metric
scale source (learned estimate / known robot / marker) are disclosed. Known poses
are a separate posed-RGB regime. Ideal RGB-D remains a diagnostic, never relabeled.

C2. Gaussian-native identity connects semantics, editing and simulation. Observed
Gaussians remain appearance sources; mesh completion primarily fills hidden shape
and supports physics. Chorus's model and prior semantic contribution are credited.
Our part is persistent ID transfer through discovery, crop selection, source
removal, object motion, physics handles and observation protection. Open-vocabulary
localization is NOT open-ended manipulation competence. Report pretraining overlap.

C3. State-conditioned observation harmonization. Use current reconstructed depth,
visible IDs, rigid motions and contact/robot protection to constrain appearance
correction. Pixels protected by exact copy have a local invariant; the whole image
is NOT formally certified. Fixed Gaussian geometry or small RGB residual does not
prove that generated shadows/texture cannot mislead a policy. Test this explicitly.

C4. System evaluation against official SimFoundry/PolaRiS on declared native tasks.
An adapted common-importer row is labeled adapted; a native-faithful row preserves
method-specific representation and effort. A high success rate alone is not high
physical fidelity. Keep reference success, action replay, geometry and coverage.

## Do not claim

- first video-to-sim, first GS-based simulation, first temporal harmonizer;
- a new Chorus foundation model or ownership of NVIDIA's enhancement method;
- that replacing one target reconstructs the whole room;
- that PolaRiS has no GS, or SimFoundry has no official code;
- that proxy baseline scripts represent either official system;
- that all datasets have compatible robot policies, interaction labels or asset rights;
- that a narrow bounding-region probe is ground-truth cavity containment;
- that stronger abstention is better unless service success/coverage support it;
- that our method wins before those baselines have actually run.

## Historical evidence retained

The 20260908 results branch reports B3 128/480 versus B0 72/480, paired gain
11.67pp [4.58,20.21], but B1/B2 each have 20 unmeasured units. This is an observed
cohort follow-up, not independent replication. F3: 20/48 common instances, 34 views;
B1 best on target-region appearance. Preserve F2 unavailable diagnostics and the
reference's own region-check disagreements. These facts motivate the new campaign;
no old result is replaced, no test-threshold tuning is justified by them.

## Submission structure

Main paper: systems figure; inputs/representation; construction/semantics;
state-conditioned observation; official-system evaluation; generator/inpainting/
semantic/observation ablations; limitations. Keep B3/B4/BM failures in the relevant
analysis. Do not fill empty new tables with old numbers from another input regime.
New optional modules enter the abstract only after genuine metric and policy gates.
Use System Verification in paper-facing text. Internal existing audit filenames
need no broad rename. Do not change the user's title automatically.
