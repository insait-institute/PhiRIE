# Prospective DROID CPU alignment stage

Owner: E7 (`agent/icra-e7-train-only`). This is a new engineering stage using
the original ten attempts across nine labs, including CLVR. It does not reuse
historical reconstruction or alignment outputs. Two additional phone captures
remain unavailable. No physical robot success is measured.

The only input roster is the existing `real_world.yaml` plus its authenticated
`configs/droid/frozen_episodes.yaml`. A preparation command hashes the ten raw
wrist videos, metadata and H5 files and creates all frame plans before reading
any robot pose values. Plans use a chronological 80/20 split, a four-index guard
and a uniform cap of 240 RGB frames. The complete FK index unions for every
offset in [-2, 2] are disjoint. The cap is applied before excluding the guard;
it is a maximum, not a promise that 240 frames survive.

SfM may consume every planned RGB image. This is independent **robot-kinematic
alignment** evaluation, not held-out image reconstruction quality. Extraction
reads only TRAIN FK indices; velocity/gripper selection and all-FK calibration
crosschecks are disabled in this explicit mode. Offset selection, Umeyama fit,
rotation and reprojection crosschecks all use TRAIN. The existing 0.10 m
constructor RMS gate, 0.15 m/20 degree evaluation gates, COLMAP 80% registration
gate and minimum 10 TRAIN/5 reference poses are retained. No backend fallback
is permitted. Missing registered reference frames remain in the planned frame
denominator and unmeasured alignment stays null.

The reference exporter validates the frozen fit and aligned reconstruction
before accessing reference FK. It replays numerical validation in the original
fitting Python/NumPy environment, even though H5 reading uses the separate
existing h5py environment. The evaluator applies this sealed transform and
offset; it never chooses a new offset or refits on reference observations.
Both paths use the same extracted canonical residual implementation.

The execution config binds interpreter bytes, installed package source files,
RECORD files, native libraries, ffmpeg, ffprobe and taskset. Runtime subprocesses
disable user-site loading, set the exact code PYTHONPATH, strip conflicting
Python/SimAny/loader/thread environment variables, and use eight assigned CPU
cores. This is targeted package provenance, not a complete operating-system
image attestation. E0 binds the complete execution config and thus all input,
frame-plan and runtime hashes.

The prospective path also passes `--num-threads 8` to COLMAP, setting the native
extractor, matcher, pipeline and nested mapper pools explicitly. Affinity and
OMP alone do not size native SIFT pools. An initial CPU pilot (`832746`, source
`2958e02`, freeze `20260906-490639d-v1`) exhausted its 32 GB allocation while
those native defaults remained `-1`. Its extracted images, partial database,
terminal failure and scheduler evidence are retained. The resource-only fix
keeps every feature/matching/geometric setting and threshold unchanged and is
used only in a new freeze. Fresh extraction is inexpensive; no historical
image outputs are relabeled as new-source evidence.

## Commands and gates

Use a new canonical ID from `reserve_freeze_id`; never type a reused ID.
Run preparation from the exact code checkout. `PREP` and `STAGE` below must be
new directories under approved repository outputs. Preparation performs only
metadata extraction and hashing, not SfM or a reference-value read.

```bash
python -m robo.eval.real_world_records \
  --config configs/experiments/icra2027/real_world.yaml \
  --prospective-prepare --tier pilot_then_full \
  --freeze-id "$FREEZE_ID" --out "$PREP" --repo-root "$PWD"
```

Bind `$PREP/execution.json` as E0's `real_world_config`, commit any tracked
configuration, obtain clean-source focused/schema and exact-E0 PASS, and merge
the source before execution. The first original IPRL workspace is the fixed CPU
pilot. This single config covers the later nine units without repeating the
pilot. Each ordinary Slurm job runs:

```bash
SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python \
SIMANY_EVIDENCE_ROOT=/group/worldcept/code/SimAny \
bash run/run_droid_recon.sh \
  --prospective-config "$PREP/execution.json" \
  --stage-root "$STAGE" --workspace "$WORKSPACE_ID"
```

Suggested initial resources: eight CPU cores, 32 GB, 30 minutes; no GPU, no
array, no requeue. The existing COLMAP producer estimates 5–15 minutes per
160–240-frame scene; this is an estimate, not a result. Pilot memory/time will
determine later requests. Root reviews the genuine pilot before dispatching
the other nine. The code also refuses later units until the first pilot has
authenticated completed producers and at least five evaluated reference poses.
A negative held-out accuracy gate does **not** prevent the remaining units or
select a replacement pilot.

Each workspace has an exclusive output directory, exact commands, logs, timing,
input identities, typed failure and terminal `result.json`. Existing attempts
are not overwritten or rerun. A crashed unsealed attempt remains incomplete and
requires an explicitly declared recovery; no implicit resume guesses. Aggregate
into a new destination with the existing producer:

```bash
python -m robo.eval.real_world_records \
  --config "$PREP/execution.json" --prospective-summary \
  --stage-root "$STAGE" --freeze-id "$FREEZE_ID" \
  --out "$STAGE/real_world/cpu_summary" --repo-root "$PWD"
```

All ten planned rows remain visible, including unattempted rows and failures.
The sole `real_world_metrics` aggregator retains its units and denominators.
`runtime_minutes` stays null because CPU-stage cost is not end-to-end
construction runtime; measured CPU cost is separately attributed.
`reconstruction_success=false`, `full_build_success=false`, accepted objects
null and Gaussian/full-build stage statuses NOT_RUN are intentional: CPU SfM
and alignment alone do not satisfy E7's trained-Gaussian reconstruction contract.
Fresh 30,000-iteration Gaussian training and the automatic construction tail
remain deferred until pilot validity and resource review.

## Verification

```bash
python -m pytest -q tests/test_colmap_cpu_threads.py tests/test_droid_prospective_alignment.py \
  tests/test_real_world_records.py tests/test_real_world_metrics.py
SIMANY_PY=/group/worldcept/code/SimAny/.venv/bin/python \
  bash run/icra2027/preflight.sh --smoke
```

Tests cover disjoint offset unions, sparse H5 access, actual NumPy geometry,
reference perturbation independence, missing-fit ordering, original-runtime
replay, source/config/roster drift, no-overwrite, tamper after rehash, missing
reference coverage, unchanged legacy aggregation and the first-IPRL gate.
Real-data CPU pilot/full-stage results and all scientific claims remain NOT_RUN
until new E0-bound execution receipts exist.
