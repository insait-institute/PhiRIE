# TRELLIS.2 integration and fixed pilot

Owner: E3 / root. Branch: `agent/icra-e3-trellis2`.

The user requested TRELLIS.2 as another generator. The original A0–A4
TRELLIS/ReconViaGen experiment and every existing freeze remain immutable.
This extension has its own source/config freeze and `trellis2` tool/proposal
identity. Its current scope is mesh generation followed by the same canonical
construction-time registration and physical probe. It is not a completed
photoreal twin or an additional result in the existing A0–A4 table.

## Fixed design

- Use all 15 already declared automatic jobs in scene `38d58a7a31` and their
  exact TRAIN RGBA bytes from discovery freeze `20260905-76c15d5-v1`.
- TRELLIS.2 upstream `75fbf0183001ed9876c8dbb35de6b68552ee08bd`, official 4B
  checkpoint revision `af44b45f2e35a493886929c6d786e563ec68364d`, DINOv3
  `ea8dc2863c51be0a264bab82070e3e8836b02d51`; local bytes are hashed in config.
- Seed 42; `512` pipeline; upstream sampler settings; 40,000 simulation
  triangles. Resolution and thresholds are fixed before inspecting outputs.
- Use the supplied mask with official alpha/bounding-box preprocessing.
  Save the actual RGB conditioning image and hash. The unused background
  removal model is explicitly disabled; there is no automatic remasking.
- Preserve native mesh and PBR voxel attributes. Official TRELLIS.2 does not
  provide Gaussian output. Do not borrow a v1 Gaussian or relabel another
  generator's artifact. `native_gaussian=false`, `full_twin_ready=false`.
- Load one model instance per scene. All planned jobs remain in the records;
  initialization failures, per-object failures, invalid artifacts and missing
  inputs remain visible. No in-place rerun or implicit cache fallback.
- Reuse the sealed TRAIN depth observations from canonical pilot
  `20260905-33bd974-v1`. Call `agents.orchestrator.runtime.align_and_probe`
  unchanged, with the original initial-registration settings. No evaluation
  surfaces, evaluation images or GT matching enter the probe.
- Runtime is separately recorded for model process/object generation and
  registration/physics. Do not claim end-to-end capture-to-simulation latency.

## Existing producer and reproducible commands

The existing `run.icra2027.e3_trellis_generation_pilot` supports an opt-in
`generator: trellis2`. Its default remains TRELLIS v1. The pipeline stages are:

```
frozen discovery crops -> CPU plan -> TRELLIS.2 GPU generation
 -> CPU artifact audit -> canonical TRAIN registration/physics probe
```

After the dedicated environment is installed, allocate a fresh ID with
`robo.eval.freeze.reserve_freeze_id`, then write explicit configuration:

```bash
python -m run.icra2027.e3_trellis2_config \
  --freeze-id <new-id> --out configs/experiments/icra2027/trellis2_pilot
# Commit configurations; run exact-source E0 preflight; create E0 contract.
python -m robo.eval.freeze \
  --config configs/experiments/icra2027/trellis2_pilot/freeze.yaml \
  --out outputs/icra2027/<new-id>/contract
python -m run.icra2027.e3_trellis_generation_pilot \
  --config configs/experiments/icra2027/trellis2_pilot/pilot.yaml \
  --freeze-root outputs/icra2027/<new-id> --phase plan
```

Use the same command with `--phase run` on one allocated allowed GPU, then
`--phase audit` and `--phase mesh-probe` on CPU. The launcher is
`run/icra2027/e3_trellis2.sbatch`, using explicit `E3_TRELLIS2_CODE`,
`E3_TRELLIS2_FREEZE`, `E3_TRELLIS2_CONFIG`, `E3_TRELLIS2_PHASE` values.
Submit independent jobs, never arrays. GPU environment/kernel verification
must pass before the real pilot. A full study requires a new explicit
complete roster, smoke/pilot gates and source/config freeze.

Outputs: `trellis2_initial/38d58a7a31/{input_manifest.json,
proposal_pool.json,proposal_records.jsonl,postrun_audit.json,mesh_probe/}`.
These are engineering evidence, not final paper table values. Final metrics
must use existing evaluation producers and `robo.eval.paper_pipeline`.

## Sources

- https://github.com/microsoft/TRELLIS.2/tree/75fbf0183001ed9876c8dbb35de6b68552ee08bd
- https://huggingface.co/microsoft/TRELLIS.2-4B/tree/af44b45f2e35a493886929c6d786e563ec68364d
