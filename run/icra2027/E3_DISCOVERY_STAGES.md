# Separate resources for the existing automatic discovery producer

The complete TRAIN/full-vocabulary first-scene job 825335 exceeded its 64 GiB
host-memory allocation during fusion, after rendering succeeded. The failed
20260905-ea7cc7f-v2 freeze is preserved. The new staged adapter does not change
rendering, TSDF integration, segmentation, preparation, frame selection or thresholds.

Each scene uses the existing config and launcher, in this order:

```bash
python -m run.icra2027.e3_auto_discovery_pilot --config configs/experiments/icra2027/e3_discovery_stages.yaml --freeze-root /group/worldcept/code/SimAny/outputs/icra2027/20260905-4ff0d0e-v1 --scene-id 09c1414f1b --phase plan
# Submit run/icra2027/e3_fresh_discovery.sbatch with explicit E3_DISCOVERY_CODE,
# E3_DISCOVERY_FREEZE, E3_DISCOVERY_CONFIG and E3_DISCOVERY_SCENE exports.
# E3_DISCOVERY_PHASE=run-render: one GPU, 32 GiB host memory, 15 minutes.
# E3_DISCOVERY_PHASE=run-fuse: zero GPUs, 256 GiB host memory, two hours.
# E3_DISCOVERY_PHASE=run-objects: one GPU, 64 GiB host memory, two hours.
# Use afterok dependencies. Restrict each allocation to hala/gcp*/sof1*.
```

The CPU fusion process never probes CUDA. Each phase has an exclusive claim,
runtime receipt and a seal binding exact source, config, input plan, predecessor
seals and output bytes. A failed phase cannot be replayed in place or promoted.
The final summary preserves all five original actions and every discovered
instance. This is a construction prerequisite; paper_ready remains false.

The older render is not imported into this freeze: its exact source/config
hashes differ. Its 16-second execution and files remain available for inspection.
The existing Gaussian source (9ef4ab1) remains bound by its original E0 and
training receipts; it is not regenerated.

Smoke (including mutation, missing dependency, failure and overwrite checks):

```bash
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl TMPDIR="$PWD/.t" /group/worldcept/code/SimAny/.venv/bin/python -m pytest -q tests/test_e3_discovery_stages.py tests/test_e3_discovery_cohort.py tests/test_e3_auto_discovery_pilot.py tests/test_training_views.py --basetemp=.t/s
```

The remaining 49 discovery units wait for the real first-scene staged pilot.
