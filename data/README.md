# data/

Documented mount point for datasets. Contents are gitignored (only this
README is tracked). The pipeline reads the datasets from the locations below
by default; set the variables to point at your own copies:

| dataset | default location | override |
|---|---|---|
| ScanNet++ v2 (frames, meshes, splits) | `/data/ScanNetpp` | `SIMANY_SCANNETPP_ROOT` |
| Per-scene 3D Gaussian splats | `/data/ScanNetppv2_gsplat/splats` | `SIMANY_SPLATS_ROOT` |

Defaults and overrides are defined in [`../run/env.sh`](../run/env.sh) and
[`../agents/core/common.py`](../agents/core/common.py); the pipeline launchers
(`run_auto.sh`, `run_factory.sh`, `run_inpaint.sh`, `run_simfoundry.sh`)
source that file, so no stage hardcodes a dataset path. To use a local copy,
export the variables before running:

```sh
export SIMANY_SCANNETPP_ROOT=/path/to/ScanNetpp
export SIMANY_SPLATS_ROOT=/path/to/splats
```

Two subdirectories are created here by launchers that build their own scene
inputs: `recon_scenes/` (ScanNet++-style scene directories plus `splats/`
written by `run/run_video2sim.sh`, `run/run_droid_recon.sh` and
`run/run_behavior_recon.sh`) and `droid/` (DROID downloads from
`run/fetch_droid_raw.py`; `SIMANY_DROID_RLDS_DIR`, `SIMANY_DROID_RAW_ROOT`).

See [`../docs/DATA_AND_WEIGHTS.md`](../docs/DATA_AND_WEIGHTS.md) for the
expected directory layout and details.
