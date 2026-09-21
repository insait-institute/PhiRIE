# Automatic export reuse

The compact recovery reuses the original sealed static packages and measured A0
exports. It does not recompute the A0 settle measurement. Original XML bytes,
absolute compiler mesh directory, source mesh references, settle values and
producer identity remain unchanged. The new materializations are independently
validated against canonical E3 outputs. Their complete immutable manifest core
must equal the original core; only creation time, destination and the explicit
materializer provenance record differ.

`e4_compact_qualification --prepare-config ... --export-reuse-root <old-freeze>`
runs the original clean-source validator with `SIMANY_SCENE` fixed separately
for each declared scene. This corrects the historical process's scene context
without editing that source. The generated configuration records the old E0,
config, Python, source commit/root, all materialized input and object hashes,
complete shared-static and A0 generated trees, and original validation results.
Seal this generated config through the normal new-stage E0 before execution.

Each new candidate stage publishes an explicit `materialized/export_reuse`
sealed declaration before copying. A0 `sim` and `sim_export` are copied byte for
byte through atomic directory publication. A completed destination is verified,
never overwritten. New A4 generation uses the original sealed static package
and its exact original paired carve manifests, with the current A4 factory as
active output. Both arms retain identical static package identity. All source
and output identities are checked again during ordinary candidate, task and
camera validation; no central factory-source exemption is introduced.

The export report records `static_producer_commit`, `export_producer_commit`
and `measured_settle_reused`. The original runtime and negative drift results
remain attached to their original source artifacts. New-wave copying and
validation are execution overhead, not new measured construction success or
newly incurred original generation cost. No runtime headline is derived here.

For the interrupted `20260905-4d4e53b-v1` stage, both static packages and both A0
exports are complete. Both A4 exports and all CPU reset/camera qualification
remain missing. Reuse does not change the fixed task population or thresholds.
The existing canonical footprint validation is still required: it is an
integrity replay, not a replacement exported settle observation.
