# third_party/

Vendored external checkouts. Everything in this directory is gitignored
except this README. [`../run/setup_env.sh`](../run/setup_env.sh) re-clones
`TRELLIS/` if it is missing; the rest are documented here so they can be
restored by hand. Override variables are read by the modules named in the
table; see [docs/DATA_AND_WEIGHTS.md](../docs/DATA_AND_WEIGHTS.md) for details.

| checkout | what it is | used by |
|---|---|---|
| `TRELLIS/` | Microsoft image-to-3D generative model (`SIMANY_TRELLIS_DIR`) | [`agents/models/s4_trellis.py`](../agents/models/s4_trellis.py) (per-object asset generation) |
| `ReconViaGen/` | multi-view image-to-3D (VGGT-conditioned TRELLIS), with its vendored VGGT (`SIMANY_RVG_DIR`) | [`agents/models/s4_reconviagen.py`](../agents/models/s4_reconviagen.py), [`agents/models/vggt_scene.py`](../agents/models/vggt_scene.py), [`agents/assets/factory_hybrid.py`](../agents/assets/factory_hybrid.py) |
| `vggt-omega/` | VGGT-Omega source (weights in `checkpoints/vggt-omega/`) | [`agents/models/vggt_scene.py`](../agents/models/vggt_scene.py) (`--backend omega`) |
| `sam-3d-objects/` | SAM 3D Objects source and checkpoint config (`checkpoints/hf/pipeline.yaml`; own environment, `SIMANY_SAM3D_PY`) | [`agents/models/s4_sam3d.py`](../agents/models/s4_sam3d.py) |
| `MaskClustering/` | CVPR 2024 class-agnostic 3D instance discovery | [`agents/baselines/maskclustering.py`](../agents/baselines/maskclustering.py) (discovery baseline; own `.venv-mc` env) |
| `FlashSplat/` | ECCV 2024 2D-mask-to-3DGS segmentation | reference clone only — [`agents/baselines/flashsplat.py`](../agents/baselines/flashsplat.py) reimplements it on gsplat and does not import this checkout |
| `BEHAVIOR-1K/` | OmniGibson simulator + BDDL toolchain | [`robo/sim/export_omnigibson.py`](../robo/sim/export_omnigibson.py), [`robo/sim/omnigibson_bridge/`](../robo/sim/omnigibson_bridge/) |
| `behavior1k_datasets/` | OmniGibson asset datasets, incl. the `phiroom_custom` dataset of converted twins | [`robo/sim/omnigibson_bridge/convert_assets.py`](../robo/sim/omnigibson_bridge/convert_assets.py) |
| `mujoco_menagerie/` | DeepMind robot model zoo (Franka Panda, Robotiq 2F-85; `SIMANY_MUJOCO_MENAGERIE_ROOT`) | [`robo/rigs/pi05_rig.py`](../robo/rigs/pi05_rig.py) |

`BEHAVIOR-1K/` is a manual clone; running
[`robo/sim/omnigibson_bridge/install_omnigibson.sh`](../robo/sim/omnigibson_bridge/install_omnigibson.sh)
from it installs the `behavior1k` conda env and downloads
`behavior1k_datasets/`. Each checkout keeps its own upstream license.
