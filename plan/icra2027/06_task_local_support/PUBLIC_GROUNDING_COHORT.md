# Full public grounding cohort publication

`run.icra2027.e6_grounding_cohort` only admits and publishes references to the
unchanged e7c5265 grounding producer. It does not create a second segmentation,
association, metric, graph, label, or rollout implementation.

The declared roster remains six scenes × three conditions, four exact public
query hashes per condition: 18 conditions and 72 queries. The four predeclared
reuse units are scene0023 clean (the smallest clean RGB smoke) and scene0020
clean/mild/severe (the fixed pilot). Reuse requires successful original-source
validation; the original files, configuration and manifests retain their source
and freeze identities. The exact constructor configuration, RGB source roster,
SAM3 and rendering runtime/checkpoint references, protocol, roles, queries,
scene/condition and actual original object identities remain authoritative.
The only config field excluded from the measurement-recipe comparison is
`freeze_id`, which changes publication location without changing measurement.
Unknown config fields fail closed.

A new pinned-e7 measurement stage executes only the ten remaining valid RGB
constructors' grounding chains. The four original RGB constructor rejections
produce their sixteen unresolved query rows through the same e7 CPU-only
failure closure; they invoke no models. A separate source-bound publication
stage owns eighteen unit manifests and links to their authentic original
artifacts. Its E0 records the new publication implementation, while the
measurement stage E0 records e7. Neither source identity is relabeled.

Both stages must be reserved, configured and E0-frozen before full execution.
Original smoke and all three pilot integrity gates must pass before full
admission. Production runs retain separate ordinary GPU rendering, CPU TSDF
fusion, GPU discovery and CPU association jobs. No Slurm arrays are used.

Commands (all paths refer to immutable generated configs):

```bash
python -m run.icra2027.e6_grounding_cohort --config COHORT/execution.json --stage-root COHORT --validate
python -m run.icra2027.e6_grounding_cohort --config COHORT/execution.json --stage-root COHORT --publish-available
python -m run.icra2027.e6_grounding_cohort --config COHORT/execution.json --stage-root COHORT --snapshot COHORT/audit/cohort_final
```

`--publish-available` follows the complete fixed roster, validates already
published units, and creates only missing completed units. Existing partial
publications are not overwritten. Snapshots retain NOT_RUN and INCOMPLETE
conditions, all failure rows, and the explicit 72-query denominator. Each
published unit is independently revalidated through the original e7 CLI.

A PASS here means reconstruction/grounding artifact integrity only. Unresolved
manipulated objects, observed regions without physical volume, missing robot
frames, and unknown physics remain unchanged in the copied original gate.
Feature rows remain zero; no task-local prediction, LOSO, calibration, physics,
or policy-success claim is enabled by this producer.

## Original-host numerical reproducibility

The pilot clean/severe finalizers ran on sof1-h200-3, while the first independent
Hala replay differed in six region AABB scalar values by at most 2.22e-16 m.
Discrete roles, object IDs and decisions were unchanged. This cross-host failure
is preserved; it is not declared a pass. Exact original e7 validators subsequently
passed on each original host, without any tolerance or metric changes.

Cohort publication is explicitly admitted on Hala. The three pilot units use
E0-bound native audit receipts, with the exact audit script SHA, original source,
CLI, host/allocation, stdout bytes and full original output-tree hashes checked.
The original construction input closure is still revalidated through e7's public
`original_unit` API. Only host-dependent association arithmetic is taken from
its authenticated original-host replay. The original smoke and fresh units
retain direct original-source validation on Hala; new finalizers must use Hala.
No numerical tolerance or original source/output modification is introduced.

## Source-relative query identities

The cohort roster is validated with the existing exact-7c source loader from
`e6_public_reconstruction_full.producer`, using the measurement-producer reference
in the authenticated original construction configuration. Query identity records
retain their original checkout paths. Running the wrapper-relative roster
validator would incorrectly reject byte-identical public query files in a new
checkout; the failed first full-admission preflight is retained. No query source
is rebound to the publication checkout and changed source or roster bytes fail.
