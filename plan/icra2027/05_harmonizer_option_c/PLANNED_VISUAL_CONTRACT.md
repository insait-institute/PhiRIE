# Planned visual evaluation contract

Schema 2 extends the existing `robo.eval.harmony_visual_metrics` producer. It
preserves the legacy evaluator and adds mandatory planned-frame accounting to
new five-condition experiments. The ICRA manifest is an explicitly empty,
non-executable template until real aligned sequences are available.

The condition keys are exactly `raw_composite`, `color_match`,
`harmonizer_non_temporal`, `harmonizer_temporal`, and
`harmonizer_robot_restore_c`. Each condition contains `render_dir`, `gt_dir`, and
`frame_records`. Every planned frame has one terminal record in every condition,
even if enhancement failed or was not run. Failed records require
`failure_reason`, have no `render` reference, and cannot leave a file in the
condition's render directory. Extra/missing filenames fail before LPIPS loads.

Each `planned_frames` entry contains a safe relative `filename`, `run_id`,
`scene_id`, `task_id`, `reset_id`, `camera_id`, and integer `frame_index`.
The tuple and filename must be unique. `raw` and `gt` are absolute
`{path, sha256}` file references. Option C additionally requires a hashed
`robot_core_mask`. Successful condition records contain `{filename, status:
"success", render: {path, sha256}, end_to_end_ms}`. The other statuses are
`enhancer_failure`, `missing`, and `not_run`.

`end_to_end_ms` is one measured sample per attempted camera frame, including
enhancer failures. Unattempted frames have null latency. It must include the
whole observation operation on the actual hardware; combining nested inference
and round-trip durations is forbidden. The evaluator uses this single field in
schema 2, retaining the old latency parser only for legacy input. The source
sequence producer still must authenticate timestamps, model/checkpoint, runtime,
stream isolation, and construction/evaluation view separation under its E0.

Successful render/GT dimensions must match the original raw frame. Preservation
and explicit warped-temporal directory pairs must have matching filenames and
dimensions, on a subset of processed frames. Their existing metric sample
counts remain explicit; missing masks/warps do not establish preservation.
Option C's core equality is recomputed from pixels, not an asserted metadata
boolean. An empty core is rejected, and even a one-byte mismatch is retained as
`robot_core_exact=false`. Such a result cannot pass D0's preservation gate.

JSON/CSV and generated LaTeX place processed/planned coverage and enhancer
failure counts before conditional quality. New output directories refuse
overwrite and publish atomically using the existing shared producer utility;
the exact input manifest is archived alongside the table. An injected write
failure cannot leave a partial published table. Declared appearance masks must
cover exactly the processed frames and be aligned/nonempty, so a missing mask
cannot silently turn an object metric into a full-frame metric. Existing
image/IoU/LPIPS metrics are called without reimplementation.

Reproduction from a clean source checkout with actual frozen input:

```bash
python -m robo.eval.harmony_visual_metrics \
  --manifest configs/experiments/icra2027/harmony_visual_manifest.json \
  --out outputs/icra2027/NEW_FREEZE/harmony/visual_table --lpips-device cuda
```

The current template intentionally fails with an empty roster. Synthetic unit
tests prove contract and negative paths only. No new real model, camera stream,
full five-condition pilot, measured latency, or visual-quality result is claimed.
Authenticated Cosmos metadata was last rechecked at 2026-09-06 05:33 UTC:
`GatedRepoError`, HTTP 403, `missing_authorized_gated_model_access`; credentials
were not logged and no download or fallback was attempted.
