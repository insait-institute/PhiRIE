# Prospective CPU drop adapter

`run.icra2027.e1_current_drop` admits only the original first two lexicographic
scenes, `09c1414f1b` and `0d2ee665be`, under the new scoped automatic+splat regime.
It retains all 50 planned scene IDs in each pilot receipt. A successful two-scene
pilot is not the five-input ladder or a completed 50-scene result.

The adapter reads the original E4 source `553db0575241a6858921b6d648d24c29392161fd`
and E0 `b963de90d9e277a9aa8c48ff2776240a5d3c675ee72411abc371f8d19022aea7`.
It invokes that clean checkout's existing materialization and generic full-room
export validators. The latter does not require a manipulation query, so the
nine no-query scenes remain eligible for an eventual E1 export extension.
Authenticated materialization members bind every URDF and collision mesh, with
before/after measurement rehashing. Original E4 settle values remain attributed
export diagnostics and never become the new E1 drop values.

Measurement directly calls `agents.eval.factory_report.drop_test` once per
exported body. It preserves the historical 5 mm base origin, default orientation,
2 s velocity-zeroed settle, 2 s free dynamics and canonical link-frame drift
criterion. No corrected-AABB placement or new threshold is introduced. It
measures isolated current collision assets, not their posed full-room behavior.
A thrown per-body error remains an attempted body with null telemetry; incomplete
scene stability stays null, and attempted/completed coverage is separately saved.

A source-bound new stage must be reserved with the canonical allocator. Call
`prepare_config(freeze_id=..., out=...)` only from the final clean producer, then
bind its exact config as E0 resource `e1_drop_config` and each returned `runtime`
entry under its declared resource ID. Runtime anchors include the actual Python,
PyBullet extension, NumPy entrypoint/norm code/core binary, and historical drop
and link-pose source files. They are explicit measurement anchors, not a claim
of hashing the entire operating system. Run exact E0 and obtain internal source
and config review before any real-data pilot execution.

```bash
SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny \
  /group/worldcept/code/SimAny/.venv/bin/python -m run.icra2027.e1_current_drop \
  --config <new-stage-config> --stage <new-stage> \
  --expected-commit <exact-clean-producer-sha> --scene 09c1414f1b
```

Use separate ordinary CPU jobs, no GPU reservation and no arrays. Pin all math
threads to four and preserve the declared CPU environment in each launcher.
Keep TMPDIR/XDG_CACHE_HOME under the primary stage. The second job uses
`0d2ee665be`; no outcome-based replacement is allowed. On an original-export
validation error, preserve stdout/stderr and `failure.json`; do not run physics.
Existing outputs cannot be overwritten.

`drop_report.json`, its build manifest and seal retain exact source/body
identities. Existing `construction_metrics.aggregate` produces the watermarked
pilot aggregate. Geometry reference status is explicitly `NOT_RUN` and F1 stays
null until a separate independent-match join is authenticated. This does not
change an exported scene's success status. Only the newly measured drop-runtime
component is filled; the overall constructor runtime stays null. No paper-ready
or comparable five-row result is produced by this adapter.
