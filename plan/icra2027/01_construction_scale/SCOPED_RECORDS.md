# Prospective scoped E1 records

`construction_metrics` retains the original schema and behavior unless every
input record supplies `measurement_scope`, `controller_accepted_instances` and
`runtime_components_seconds`. New records must use the exact versioned
`MEASUREMENT_SCOPE`, shared by every compared input regime. Mixed legacy/scoped
records or different constructor, acceptance, physical, collision, geometry or
runtime protocol IDs fail validation. Source manifests bind the extra fields.

For scoped records, `accepted_instances` means verified simulator-export bodies;
controller acceptance has its own count and coverage. Unknown input or export
counts remain null. A partial export count is shown only as an explicitly
observed count with its scene count; complete yield remains null until all
scene denominators and export numerators are known. Independent F1 support must
be a subset of the verified exported bodies. A completed independent evaluation
with zero matches uses `geometry_reference_status=independent_gt_evaluation_no_matches`,
null F1 and zero F1 weight. This measured empty reference support does not invalidate
verified export yield or measured drop/runtime. It is distinct from a reference
evaluation that was not run; the latter cannot claim measured zero matches. No E3/E4 probe is relabeled as the
new canonical factory-report drop measurement.

Runtime exposes six required component durations. A scene total must equal the
sum of complete components; an unknown component forbids a total. The aggregate
does not average only the completed scenes into a complete runtime. Runtime is
an attributed sum of stage durations, not the elapsed latency of a parallel
fleet. Missing drop coverage, export counts or runtime prevent paper readiness.

JSON reports schema 3 for this opt-in scope. CSV serializes scope/component
mappings as JSON strings. The LaTeX caption identifies the new measurement
semantics, while legacy output remains unchanged. Scoped build manifests may
use explicit `SIMANY_EVIDENCE_ROOT` only when the existing validator authenticates
it as a sibling checkout in the same Git repository; legacy path handling is
unchanged.

`configs/experiments/icra2027/e1_scoped_protocol.json` prospectively binds common
A4/FULL policy and the historical `factory_report.drop_test` implementation for
new measurements on current exported collision assets. It records all 50 scenes,
all five regimes, and the fixed first-two-scene CPU pilot. Its execution flag is
false until source, exact config, resource and negative-test review completes.
The four other input producers/reference contracts remain explicit prerequisites;
no GPU generation is launched by this schema extension.
