# Full automatic+splat E1 construction/drop closure

This stage preserves the original 50 scenes, 1,871 object jobs and A4/FULL
constructor decisions. It extends the existing `e1_current_drop` publisher;
`factory_report.drop_test` and `construction_metrics` retain measurement
ownership. No model, task planner, rollout, or evaluation-GT producer is called.
The source-bound two-scene pilot is an admission gate. The full stage repeats
its 11 cheap drops under the new full-stage source/config; it does not relabel
prior pilot measurements as new-source outputs.

The prospective source map has four paths, chosen from immutable construction
artifacts before the new measurements:

- **Original export:** authenticate original source553's materialization and
  complete full-room export through its existing validators. Rehash the frozen
  factory/export anchors and every URDF/mesh before measuring.
- **No accepted bodies:** preserve all original jobs, zero verified bodies,
  zero attempted drops and an explicit empty construction record. This does
  not claim that an empty room collision package was exported.
- **Observed paired-room rejection:** re-execute only the original source's
  read-only validation of the rejected A0 shared-room package. Preserve the
  exact protected-carve intrusion and all original logs/identities. A4 export
  remains blocked; zero bodies are verified, and no physical drop is invoked.
- **Missing export:** use the canonical materializer for A0/A4, copying existing
  selected E3 assets without model generation. Existing E2 A4 materializations
  provide an additional byte-equality check on the output-member map. Their
  frozen directories and original source manifests remain unchanged. Produce
  only A4's required room export, using the existing paired-static exporter;
  then run its standard validation and the unchanged canonical drop.

The current metadata audit finds 38 original exports, four zero-accept scenes,
six missing exports and two observed paired-room rejections. These counts are
an execution plan, not verified full-run yield. The first missing-export scene,
`1ada7a0617`, passed the read-only canonical-product/E2-member equality smoke
for all 25 jobs/eight accepted objects, with no models or physics invoked.

The acceptance scope is explicit: **A4 bodies verified only after complete
paired A0/A4 static full-room export validation**. It is not isolated-body URDF
availability and is not controller acceptance. A paired static rejection stays
in the construction denominator even when an individual body's isolated drop
could be measured. Prior E3/E4 physical negatives are neither overwritten nor
pooled with this new historical 5 mm canonical drop protocol.

Every scene gets a sealed record, including missing/export-rejected/no-body
terminals. Unexpected validation/resource failures retain null unmeasured
numerators and physical values. Per-body drop errors remain attempted rows with
null telemetry and null incomplete scene stability. No outcome-based scene,
object, collision threshold, or retry substitution is permitted.

A new allocator-reserved freeze binds the final clean source, generated full
configuration, original source/E0, original pilot/E0, runtime content anchors,
and the fixed SOF6-compatible CPU environment. Use ordinary CPU jobs with four
native math threads and the existing AVX512-disable mask; do not reserve GPUs.
Original-export/no-body jobs are separate from the heavier missing-export jobs.
No full submissions precede source review, focused negative tests and exact E0.

```bash
python -m run.icra2027.e1_full_drop --config <stage>/execution.json \
  --stage <stage> --expected-commit <full-source-sha> --scene <frozen-scene-id>
python -m run.icra2027.e1_full_drop --config <stage>/execution.json \
  --stage <stage> --expected-commit <full-source-sha> --aggregate
```

Aggregation requires all 50 terminal seals and produces the canonical 250
scene/regime records. The four other input regimes have explicit NOT_RUN source
manifests, null counts/quality/runtime, and retain all 50 planned scenes each.
The existing `construction_metrics` produces JSON/CSV/LaTeX with a preliminary
watermark; this is not a completed five-input result.

Independent F1 and total constructor runtime remain null. Only after the full
construction/drop seal may a separate read-only join inspect source1b3's existing
independent per-object geometry results. Reuse then requires exact object,
proposal, evaluated mesh, transform and metric-implementation identities; an
exported URDF visual mesh is not automatically the evaluated E3 geometry.

Fresh exports explicitly pass the frozen numerical subprocess environment:
`NPY_DISABLE_CPU_FEATURES` uses the previously authenticated SOF6 mask, all four
native thread caps equal 4, and `MUJOCO_GL`/`PYOPENGL_PLATFORM` equal `osmesa`. The exact `LD_LIBRARY_PATH` is recorded with these
fields in the immutable execution config and compared to the executing process
before admission. Both static-package and A4-export subprocesses retain their
actual command, working directory and environment in the sealed export receipt.
The exporter accepts this map only through an explicit allowlisted argument;
existing callers retain their original restricted environment.

Full aggregation requires exactly the three canonical unit seal members at
`construction_drop/<scene>/{drop_report,build_manifest,construction_record_aggregate}.json`.
An omitted member, another path containing identical bytes, a symbolic alias,
or another scene's resealed member cannot authenticate a consumed record.
