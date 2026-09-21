# F1 — final official-system construction comparison

Fixed budget:24 instances x10 resets x4 arms=960 logical units.
Shared input:RGB TRAIN video and explicitly declared public marker calibration.
Estimated placement is preserved. Oracle room/destination context remains for
L0. Common native renderer/import assumptions are identical and fully disclosed.

Arms: native reference, official SimFoundry-adapted, official PolaRiS-adapted,
fixed DEV-selected SimAnyRoom. Resolve official source revisions and complete
actual construction. SF stage7 mesh is not a finished scene; validate all needed
pose/physical stages. PolaRiS6/100 on its DROID/Isaac task is not this comparison.

PolaRiS extra scans/assisted composition and each method's human minutes must be
recorded. Compare under a declared acquisition budget; when extra inputs differ,
label method-native input and report them. Do not claim equal-input superiority
where input differs. Unsupported acquisition is not a zero-success baseline.

First DEV controls: original scene through wrapper, original asset through the
common importer, native role/origin checks. Do not weld objects, delete obstacles,
change grasp mode or simplify the success rubric to improve our score.

All constructors finish before reading hidden geometry/labels. Resolve action,
robot, camera, policy/checkpoint, horizon, reset and rubric once. Reuse same
instance asset across resets. Cost includes failed construction and actual manual
work. Report official-native results separately from adapted/common-importer.

Output: canonical ledgers, per-task native success, success/planned,
executed/planned, success/executed, paired scene/instance/reset intervals,
reference success, independent geometry and object-region fidelity. No new
simulator score. Higher success due to a missing obstacle is not higher fidelity.

Acceptance: entire registered roster measured or explicitly unmeasured, valid
fresh-reference pairs, official-method provenance and no hidden adaptation.
No guarantee or stopping rule requires SimAnyRoom to win.
