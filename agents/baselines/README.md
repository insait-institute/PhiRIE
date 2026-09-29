# Baselines

Three baselines are maintained against SimAny, all run on our own inputs so
the comparisons are controlled (same scenes, same masks where applicable).

| Baseline | What it is | Where |
|---|---|---|
| **SimFoundry reproduction** | paper-faithful reimplementation of the prior SimFoundry system (arXiv:2606.28276) this project began from: one representative frame, monocular metric depth, single-view TRELLIS — the paper's ablation row D | launcher [`run/run_simfoundry.sh`](../../run/run_simfoundry.sh), composed from the shared stages (`agents.discover.s0_select_frame` → `agents.models.s1_segment` → `agents.models.s2_depth` → `agents.discover.s3_lift` → `agents.models.s4_trellis` → `agents.assets.s5_align` → …) |
| **MaskClustering** | CVPR 2024 multi-view instance discovery, run on the same SAM3 masks and matched frame density | [`maskclustering.py`](maskclustering.py) (adapter + eval), [`mc_masks.py`](mc_masks.py) (mask export); checkout in `third_party/MaskClustering` |
| **FlashSplat** | ECCV 2024 2D-mask-to-3DGS assignment, reimplemented exactly on gsplat for the removal comparison | [`flashsplat.py`](flashsplat.py) (self-contained; `third_party/FlashSplat` is reference only) |

The SimFoundry reproduction has no code of its own by design: it is the
degenerate configuration of the full pipeline (no GT, no multi-view
generation, no hybrid slot), which is what makes ablation rows A→D a
controlled ladder rather than a cross-codebase comparison.
